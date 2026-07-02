import argparse
import zarr
import time
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
# from joblib import Parallel, delayed

import torch
import tifffile


def find_max_batch_size(model, input_size=256, n_channels=1, start=4, max_limit=512):

    batch_size = start
    best = start

    device = next(model.parameters()).device

    while batch_size <= max_limit:
        try:
            with torch.inference_mode():
                test_batch = torch.zeros(
                    (batch_size, n_channels, input_size, input_size),
                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    device=device
                )
                _ = model(test_batch)

            best = batch_size
            batch_size *= 2

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                break
            else:
                raise e

    del test_batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best

def predict_block(model, block, num_classes=2, n_channels=1, batch_size=8, axes=[0,1,2]):
    """
    block: (D, H, W) float32 tensor, values in [0, 1]

    For n_channels=1: each slice is fed as a single-channel image.
    For n_channels=3 (2.5D): each slice i is stacked with its neighbours
        [i-1, i, i+1], with reflect padding at boundaries.
    """

    input_size = block.shape[0]
    device = next(model.parameters()).device

    block_prediction = np.zeros((input_size, input_size, input_size, num_classes), dtype=np.float32)

    for axis in axes:

        with torch.inference_mode():

            block_t = torch.moveaxis(block, axis, 0)  # (N, *, *)
            N = block_t.shape[0]

            for i in range(0, N, batch_size):

                sl = block_t[i:i+batch_size]  # (B, H, W)

                if n_channels == 1:
                    batch = sl.unsqueeze(1)  # (B, 1, H, W)
                else:
                    # 2.5D: stack [i-1, i, i+1] as channels with reflect padding
                    indices = torch.arange(i, min(i + batch_size, N))
                    prev_idx = torch.clamp(indices - 1, 0, N - 1)
                    next_idx = torch.clamp(indices + 1, 0, N - 1)
                    batch = torch.stack([
                        block_t[prev_idx],
                        block_t[indices],
                        block_t[next_idx],
                    ], dim=1)  # (B, 3, H, W)

                batch = batch.to(device)
                if device.type == "cuda":
                    batch = batch.half()

                batch_prediction = model(batch)
                batch_prediction = batch_prediction.permute(0, 2, 3, 1).float().cpu().numpy()

                if axis == 0:
                    block_prediction[i:i+batch_size, :, :, :] += batch_prediction
                elif axis == 1:
                    block_prediction[:, i:i+batch_size, :, :] += batch_prediction.transpose(1, 0, 2, 3)
                elif axis == 2:
                    block_prediction[:, :, i:i+batch_size, :] += batch_prediction.transpose(1, 2, 0, 3)

    block_prediction /= len(axes)

    return block_prediction

_DEFAULT_CHUNK = 64
_DEFAULT_SHARD = 512


class _TifVolume:
    """Thin wrapper around a tifffile memmap so predict_volume can treat it like a zarr array."""

    def __init__(self, path: Path):
        try:
            self._arr = tifffile.memmap(str(path))
        except Exception:
            # Fall back to full load if the tif layout isn't memmap-compatible
            print(f"  Note: {path.name} cannot be memory-mapped, loading into RAM...")
            self._arr = tifffile.imread(str(path))
        if self._arr.ndim != 3:
            raise ValueError(f"Expected a 3-D TIF stack, got shape {self._arr.shape}")
        self.chunks = (_DEFAULT_CHUNK, _DEFAULT_CHUNK, _DEFAULT_CHUNK)
        self.shards = (_DEFAULT_SHARD, _DEFAULT_SHARD, _DEFAULT_SHARD)

    @property
    def shape(self):
        return self._arr.shape

    def __getitem__(self, idx):
        return self._arr[idx]


class _TifSeriesVolume:
    """Assembles a sorted directory of 2-D TIF slices into a virtual (Z, Y, X) volume."""

    def __init__(self, directory: Path):
        self._paths = sorted(directory.glob('*.tif')) + sorted(directory.glob('*.tiff'))
        # deduplicate while preserving sort order (a file won't match both, but just in case)
        seen = set()
        self._paths = [p for p in self._paths if not (p in seen or seen.add(p))]
        if not self._paths:
            raise ValueError(f"No .tif/.tiff files found in {directory}")
        first = tifffile.imread(str(self._paths[0]))
        if first.ndim != 2:
            raise ValueError(f"Expected 2-D TIF slices, got shape {first.shape} in {self._paths[0].name}")
        self._shape = (len(self._paths), first.shape[0], first.shape[1])
        self.chunks = (_DEFAULT_CHUNK, _DEFAULT_CHUNK, _DEFAULT_CHUNK)
        self.shards = (_DEFAULT_SHARD, _DEFAULT_SHARD, _DEFAULT_SHARD)
        print(f"  TIF series: {len(self._paths)} slices → volume {self._shape}")

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        z_sl, y_sl, x_sl = idx
        z_range = range(*z_sl.indices(self._shape[0]))
        planes = [tifffile.imread(str(self._paths[z]))[y_sl, x_sl] for z in z_range]
        return np.stack(planes, axis=0) if planes else np.empty((0,) + self._shape[1:], dtype=np.uint8)


def _open_volume(path):
    """Open a zarr, tif series dir, single-file tif, or remote WebKnossos volume."""
    url = str(path)
    if url.startswith(("http://", "https://")):
        return open_webknossos_zarr(url)
    path = Path(path)
    if path.is_dir() and not str(path).endswith('.zarr'):
        return _TifSeriesVolume(path)
    suffix = path.suffix.lower()
    if suffix in ('.tif', '.tiff'):
        return _TifVolume(path)
    return zarr.open(str(path), mode='r')['0']


def setup_model(model_path, input_size=256, n_channels=1, batch_size=None):

    torch.set_float32_matmul_precision('medium')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path.is_file():
        model = torch.load(model_path, weights_only=False)['model'].to(device)
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model.eval()

    if device.type == "cuda":
        model = model.half()

    if batch_size is None:
        batch_size = find_max_batch_size(model, input_size=input_size, n_channels=n_channels, start=4, max_limit=input_size)
        print(f'Found optimal inference batch size of {batch_size}.')
    else:
        print(f'Using batch size of {batch_size}.')

    return model, batch_size


def _write_tiff_from_zarr(prediction_file, out_dir, stem, channels):
    channel_map = {'foreground': 0, 'boundary': 1}
    arr = zarr.open(str(prediction_file), 'r')['0']  # (Z, Y, X, 2) uint8
    for name in channels:
        out_path = out_dir / f'{stem}_{name}.tif'
        ch = channel_map[name]
        with tifffile.TiffWriter(str(out_path), bigtiff=True) as tw:
            for z in range(arr.shape[0]):
                tw.write(arr[z, :, :, ch])
        print(f'  Saved {out_path.name}')


def predict_volume(zarr_file, prediction_file, temp_folder, model, window, input_size=256, n_channels=1, num_classes=2, batch_size=None, overlap=0.25, axes=[0,1,2], save_tiff=None):

    if batch_size is None:
        raise ValueError("batch_size must be provided")
    if model is None:
        raise ValueError("model must be provided")
    if window is None:
        window = gaussian_3d(input_size, sigma=0.125).astype('float32')

    start_time = time.time()

    volume = _open_volume(zarr_file)
    chunk_size = volume.chunks[0]
    shard_size = volume.shards[0]
    input_volume_shape = np.array(volume.shape)
    output_volume_shape = np.append(input_volume_shape, num_classes)

    prediction_file.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(prediction_file), mode='w')
    final_predictions = root.create_array(name='0',
                                          shape=output_volume_shape.astype(int).tolist(),
                                          chunks=(chunk_size, chunk_size, chunk_size, num_classes),
                                          shards=(shard_size, shard_size, shard_size, num_classes),
                                          dtype='uint8',
                                          overwrite=True)

    temp_folder.mkdir(parents=True, exist_ok=True)
    for p in (temp_folder / 'pred.zarr', temp_folder / 'weight.zarr'):
        if p.is_dir():
            shutil.rmtree(p)

    try:
        pred_root = zarr.open(str(temp_folder / 'pred.zarr'), mode='w')
        pred = pred_root.create_array(name='0',
                                      shape=output_volume_shape.astype(int).tolist(),
                                      chunks=(chunk_size, chunk_size, chunk_size, num_classes),
                                      shards=(shard_size, shard_size, shard_size, num_classes),
                                      dtype='float32',
                                      overwrite=True)

        weight_root = zarr.open(str(temp_folder / 'weight.zarr'), mode='w')
        weight = weight_root.create_array(name='0',
                                          shape=input_volume_shape.astype(int).tolist(),
                                          chunks=(chunk_size, chunk_size, chunk_size),
                                          shards=(shard_size, shard_size, shard_size),
                                          dtype='float32',
                                          overwrite=True)

        block_coords, padded_block_coords, local_block_coords = get_block_coordinates(input_volume_shape, input_size=input_size, overlap=overlap)
        num_blocks = len(padded_block_coords)

        print(f'\nSegmenting {zarr_file.name}...')
        for i in tqdm(range(num_blocks)):

            padded_block = torch.tensor(get_padded_block(volume, *padded_block_coords[i]).astype('float32') / 255.0)

            predicted_block = predict_block(model, padded_block, num_classes=num_classes, n_channels=n_channels, batch_size=batch_size, axes=axes)

            i0, j0, k0, i1, j1, k1 = block_coords[i]
            l_i0, l_j0, l_k0, l_i1, l_j1, l_k1 = local_block_coords[i]

            pred[i0:i1, j0:j1, k0:k1] += predicted_block[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1, :] * window[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1, None]
            weight[i0:i1, j0:j1, k0:k1] += window[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1]

        del volume

        print('Postprocessing and generating multiscale pyramid...')

        shard_coordinates = get_shard_coordinates(input_volume_shape, shard_size=shard_size)
        def normalize_shard(final_predictions, pred, weight, coords, eps=1e-3):
            i0, j0, k0, i1, j1, k1 = coords
            final_predictions[i0:i1, j0:j1, k0:k1] = (255 * pred[i0:i1, j0:j1, k0:k1] / np.maximum(weight[i0:i1, j0:j1, k0:k1], eps)[...,None]).astype('uint8')
        for coords in shard_coordinates:
            normalize_shard(final_predictions, pred, weight, coords)

        del pred, weight, final_predictions

    finally:
        if temp_folder.exists():
            shutil.rmtree(temp_folder)

    if save_tiff:
        _write_tiff_from_zarr(prediction_file, prediction_file.parent, zarr_file.stem, save_tiff)

    time_elapsed = time.time() - start_time
    print(f'Completed volume {zarr_file.name} {tuple(input_volume_shape.astype(int).tolist())} in {time_elapsed}.')


def predict_all_volumes(zarr_files, project_path, input_size=256, n_channels=1, num_classes=2, batch_size=None, overlap=0.25, axes=[0,1,2], save_tiff=None):

    project_path = Path(project_path)

    model_path = project_path / 'model.ckpt'
    model, batch_size = setup_model(model_path, input_size=input_size, n_channels=n_channels, batch_size=batch_size)

    window = gaussian_3d(input_size, sigma=0.125).astype('float32')

    predictions_dir = project_path / 'predictions'
    temp_dir = project_path / 'temp'

    for zarr_file in zarr_files:
        zarr_file = Path(zarr_file)

        predict_volume(
            zarr_file=zarr_file,
            prediction_file=predictions_dir / zarr_file.name,
            temp_folder=temp_dir,
            model=model,
            window=window,
            input_size=input_size,
            n_channels=n_channels,
            num_classes=num_classes,
            batch_size=batch_size,
            overlap=overlap,
            axes=axes,
            save_tiff=save_tiff,
        )

    print('\nAll volumes segmented.\n')


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def reflect_index(idx, size):
    if size == 1:
        return np.zeros_like(idx)
    period = 2 * size - 2
    idx = np.abs(idx) % period
    return np.where(idx < size, idx, period - idx)

def get_padded_block(volume, i0, j0, k0, i1, j1, k1):
    volume_shape = volume.shape

    pad_before = [max(0, -i0), max(0, -j0), max(0, -k0)]
    pad_after  = [max(0, i1 - volume_shape[0]), max(0, j1 - volume_shape[1]), max(0, k1 - volume_shape[2])]

    c_i0, c_i1 = max(i0, 0), min(i1, volume_shape[0])
    c_j0, c_j1 = max(j0, 0), min(j1, volume_shape[1])
    c_k0, c_k1 = max(k0, 0), min(k1, volume_shape[2])

    block = volume[c_i0:c_i1, c_j0:c_j1, c_k0:c_k1]

    padding = ((pad_before[0], pad_after[0]),
               (pad_before[1], pad_after[1]),
               (pad_before[2], pad_after[2]))

    return np.pad(block, pad_width=padding, mode='reflect')

def get_shard_coordinates(volume_shape, shard_size=128):
    starts = [np.arange(0, s, shard_size) for s in volume_shape]
    chunk_coordinates = np.stack(np.meshgrid(*starts, indexing='ij'), -1).reshape(-1, 3)
    chunk_coordinates = np.concatenate([chunk_coordinates, np.minimum(chunk_coordinates + shard_size, volume_shape)], axis=1)
    return chunk_coordinates

def gaussian_3d(input_size, sigma=0.125, eps=1e-3):
    sigma *= input_size
    coords = np.arange(input_size, dtype=np.float32) - (input_size - 1) / 2.0
    g = np.exp(-(coords**2) / (2 * sigma**2)).astype(np.float32)
    g /= g.max()
    gaussian = g[:, None, None] * g[None, :, None] * g[None, None, :]
    gaussian /= gaussian.max()
    gaussian = np.clip(gaussian, max(gaussian.min(), eps), 1.0)
    return gaussian

def hanning_3d(input_size, eps=1e-3):
    h = np.hanning(input_size)
    hanning = h[:, None, None] * h[None, :, None] * h[None, None, :]
    hanning /= hanning.max()
    hanning = np.clip(hanning, max(hanning.min(), eps), 1.0)
    return hanning.astype('float32')

def get_block_coordinates(volume_shape, input_size=256, overlap=0.25):

    blocks_per_axis = np.ceil((volume_shape - overlap * input_size) / (input_size - overlap * input_size)).astype(int)
    padded_volume_shape = np.round(blocks_per_axis * input_size - (blocks_per_axis - 1) * input_size * overlap).astype(int)

    padding_shift = (padded_volume_shape - volume_shape) // 2
    padding_shift = np.array(list(padding_shift) + list(padding_shift))

    block_coords = []
    padded_block_coords = []
    local_block_coords = []

    for i in range(blocks_per_axis[0]):
        p_i0 = i * input_size * (1 - overlap)
        p_i1 = p_i0 + input_size

        for j in range(blocks_per_axis[1]):
            p_j0 = j * input_size * (1 - overlap)
            p_j1 = p_j0 + input_size

            for k in range(blocks_per_axis[2]):
                p_k0 = k * input_size * (1 - overlap)
                p_k1 = p_k0 + input_size

                coords = np.array([p_i0, p_j0, p_k0, p_i1, p_j1, p_k1]) - padding_shift
                coords = coords.astype(int)
                padded_block_coords.append(coords)

                i0, j0, k0, i1, j1, k1 = coords
                i0_c, i1_c = max(0, i0), min(volume_shape[0], i1)
                j0_c, j1_c = max(0, j0), min(volume_shape[1], j1)
                k0_c, k1_c = max(0, k0), min(volume_shape[2], k1)
                block_coords.append([i0_c, j0_c, k0_c, i1_c, j1_c, k1_c])

                l_i0, l_i1 = i0_c - i0, i1_c - i0
                l_j0, l_j1 = j0_c - j0, j1_c - j0
                l_k0, l_k1 = k0_c - k0, k1_c - k0
                local_block_coords.append([l_i0, l_j0, l_k0, l_i1, l_j1, l_k1])

    return np.array(block_coords), np.array(padded_block_coords), np.array(local_block_coords)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Run 2D UNet inference on 3D volumes (.tif or .zarr).'
    )
    p.add_argument('--input',      required=True, nargs='+', type=Path,
                   help='One or more .tif/.tiff/.zarr volume paths.')
    p.add_argument('--project',    required=True, type=Path,
                   help='Project directory containing model.ckpt; predictions saved under <project>/predictions/.')
    p.add_argument('--input_size', type=int,   default=256,  # must match training --input_size
                   help='Sliding-window block size in voxels (default: 256).')
    p.add_argument('--n_channels', type=int,   default=1,
                   help='1 = single-slice, 3 = 2.5D triplet — must match training (default: 1).')
    p.add_argument('--overlap',    type=float, default=0.25,
                   help='Fractional overlap between adjacent blocks (default: 0.25).')
    p.add_argument('--axes',       type=int,   nargs='+', default=[0, 1, 2],
                   help='Axes to average predictions over (default: 0 1 2).')
    p.add_argument('--batch_size', type=int,   default=None,
                   help='Inference batch size; auto-detected from GPU memory if omitted.')
    p.add_argument('--num_classes', type=int,  default=2,
                   help='Number of output classes (default: 2).')
    p.add_argument('--save_tiff', nargs='*', metavar='CHANNEL',
                   help='Write per-channel BigTIFF stacks. '
                        'No args = both channels; '
                        'or specify: foreground boundary')
    args = p.parse_args()

    if args.save_tiff is not None:
        valid = {'foreground', 'boundary'}
        bad = set(args.save_tiff) - valid
        if bad:
            p.error(f'--save_tiff: unknown channel(s) {bad}. Choose from: {valid}')
        if not args.save_tiff:
            args.save_tiff = ['foreground', 'boundary']

    # Expand any glob-like paths the shell didn't expand
    inputs = []
    for path in args.input:
        if path.exists():
            inputs.append(path)
        else:
            matches = sorted(path.parent.glob(path.name))
            if not matches:
                p.error(f"No files matched: {path}")
            inputs.extend(matches)

    predict_all_volumes(
        zarr_files  = inputs,
        project_path= args.project,
        input_size  = args.input_size,
        n_channels  = args.n_channels,
        num_classes = args.num_classes,
        batch_size  = args.batch_size,
        overlap     = args.overlap,
        axes        = args.axes,
        save_tiff   = args.save_tiff,
    )


if __name__ == '__main__':
    main()
