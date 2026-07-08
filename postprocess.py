"""
postprocess.py
--------------
Convert raw probability predictions into instance labels.

Strategy:
  1. Threshold foreground channel  →  binary foreground mask
  2. Threshold boundary channel    →  binary boundary mask
  3. Seeds = foreground AND NOT boundary, connected-component labelled
  4. Watershed from seeds over the foreground probability landscape
  5. Remove objects smaller than --min_size voxels

Output: zarr store  shape (Z, Y, X)  dtype uint32  — each unique non-zero
        integer is one instance.

Usage:
    python postprocess.py \\
        --input  /scratch/asvetlove/project_2d/predictions/volume.tif \\
        --output /scratch/asvetlove/project_2d/instances/volume.tif \\
        --fg_threshold 0.5 \\
        --bd_threshold 0.4 \\
        --min_size 500
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr
import cc3d
from skimage.morphology import remove_small_objects

try:
    from cucim.skimage.segmentation import watershed
    _WATERSHED_BACKEND = 'cucim (GPU)'
except ImportError:
    from skimage.segmentation import watershed
    _WATERSHED_BACKEND = 'skimage (CPU)'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_dtype(instances: np.ndarray) -> np.dtype:
    """
    Pick the smallest integer dtype that holds the instance labels.
    uint16 is plenty for < 65 536 instances (the normal case); fall back to
    uint32 with a warning if a volume ever exceeds that (prevents overflow).
    """
    max_label = int(instances.max()) if instances.size else 0
    if max_label <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    print(f"  Note: {max_label:,} instances exceed uint16 range — using uint32.")
    return np.dtype(np.uint32)


def upload_to_webknossos(
    instances:    np.ndarray,      # (Z, Y, X) integer instance labels
    raw_path:     Path,            # original image volume (tif dir / .tif / .zarr)
    sample_name:  str,
    url:          str,
    token:        str,
    voxel_size:   tuple = (650, 650, 650),
    norm_min:     float = 0.0,
    norm_max:     float = 1.0,
    folder:       str   = 'IMATREC',
    temp_root:    Path  = None,
) -> str:
    """
    Upload a new WebKnossos dataset 'IMATREC_<sample_name>' with two layers:
        - color        : the raw image, normalised to uint8 with the fixed window
        - segmentation : the instance labels (uint16, or uint32 if they overflow)

    Both layers share the whole-volume bounding box at origin (0, 0, 0).
    The dataset is staged locally (under temp_root / $TMPDIR) then uploaded and
    moved into the `folder` on the server.
    """
    import webknossos as wk
    from predict import _open_volume

    # --- Raw image → uint8 with the same fixed window used in training ---
    # Normalise in-place to avoid extra full-size float32 copies of a large volume.
    print(f"Loading raw volume for color layer: {raw_path}")
    raw_vol = _open_volume(raw_path)
    raw = np.asarray(raw_vol[:, :, :]).astype(np.float32, copy=False)  # (Z, Y, X)
    np.clip(raw, norm_min, norm_max, out=raw)
    raw -= norm_min
    raw /= (norm_max - norm_min)
    raw *= 255.0
    raw = raw.astype(np.uint8)

    if raw.shape != instances.shape:
        raise ValueError(
            f"Raw shape {raw.shape} != instances shape {instances.shape}; "
            f"the raw volume must be the same one that was segmented."
        )

    seg_dtype = _label_dtype(instances)

    # WebKnossos stores data in (X, Y, Z) order; our arrays are (Z, Y, X).
    raw_xyz = np.ascontiguousarray(raw.transpose(2, 1, 0))                    # (X, Y, Z) uint8
    seg_xyz = np.ascontiguousarray(instances.transpose(2, 1, 0)).astype(seg_dtype)
    size    = raw_xyz.shape
    bbox    = wk.BoundingBox((0, 0, 0), size)

    scratch    = temp_root or os.environ.get('TMPDIR') or os.environ.get('SLURM_TMPDIR')
    tmp_parent = Path(scratch) if scratch else Path('.')
    tmp_parent.mkdir(parents=True, exist_ok=True)

    dataset_name = f"IMATREC_{sample_name}"

    with tempfile.TemporaryDirectory(dir=str(tmp_parent)) as tmp_dir:
        print(f"Staging local WebKnossos dataset '{dataset_name}' under {tmp_dir} ...")
        local_ds = wk.Dataset(Path(tmp_dir) / 'wk_upload',
                              voxel_size=voxel_size, name=dataset_name)

        color = local_ds.add_layer(
            layer_name        = 'raw',
            category          = 'color',
            dtype_per_channel = np.uint8,
            num_channels      = 1,
            data_format       = wk.DataFormat.Zarr,
        )
        color.bounding_box = bbox
        color.add_mag('1', compress=True).write(raw_xyz, allow_resize=True)
        print("  color layer written")

        seg = local_ds.add_layer(
            layer_name         = 'instances',
            category           = 'segmentation',
            dtype_per_channel  = seg_dtype,
            num_channels       = 1,
            data_format        = wk.DataFormat.Zarr,
            largest_segment_id = int(instances.max()) if instances.size else 0,
        )
        seg.bounding_box = bbox
        seg.add_mag('1', compress=True).write(seg_xyz, allow_resize=True)
        print(f"  segmentation layer written ({seg_dtype})")

        print(f"Uploading '{dataset_name}' to {url} ...")
        with wk.webknossos_context(token=token, url=url):
            remote = local_ds.upload()
            try:
                remote.folder = wk.RemoteFolder.get_by_path(folder)
                print(f"  moved to folder '{folder}'")
            except Exception as e:
                print(f"  Warning: could not move to folder '{folder}' ({e}); left in root.")

    print(f"Uploaded: {remote.url}")
    return remote.url


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def postprocess(
    input_path:    Path,
    output_path:   Path,
    fg_threshold:  float = 0.5,
    bd_threshold:  float = 0.4,
    min_size:      int   = 500,
    upload:        dict  = None,   # kwargs for upload_to_webknossos, or None to skip
):
    t0 = time.time()

    # --- Load predictions ---
    # Keep the probability channels as uint8 (0-255) rather than converting to
    # float32. Thresholds and the watershed landscape work just as well in uint8,
    # at 1/4 the RAM — critical for large volumes (float32 here OOM-kills a
    # ~9e9-voxel volume even at 200 GB).
    pred_store = zarr.open(str(input_path), mode='r')['0']
    print(f"Prediction shape : {pred_store.shape}  dtype={pred_store.dtype}")
    fg_thr = int(round(fg_threshold * 255))
    bd_thr = int(round(bd_threshold * 255))

    print("Loading foreground channel...")
    fg = pred_store[:, :, :, 0]   # (Z, Y, X) uint8
    print("Loading boundary channel...")
    bd = pred_store[:, :, :, 1]   # (Z, Y, X) uint8

    # --- Threshold (compared in uint8 space) ---
    foreground = fg > fg_thr   # bool (Z, Y, X)
    boundary   = bd > bd_thr   # bool (Z, Y, X)
    del bd

    print(f"Foreground voxels : {foreground.sum():,}  ({100*foreground.mean():.1f}%)")

    # --- Seeds: foreground regions not touching boundaries ---
    seeds_binary = foreground & ~boundary
    del boundary
    print(f"Labelling seeds (cc3d)...")
    seeds = cc3d.connected_components(seeds_binary, out_dtype=np.uint32)
    n_seeds = int(seeds.max())
    del seeds_binary
    print(f"Seeds found       : {n_seeds:,}")

    # --- Watershed ---
    # Landscape: invert foreground probability (uint8) so high-confidence voxels
    # fill first — equivalent to -fg_prob but 1/4 the memory.
    print(f"Running watershed ({_WATERSHED_BACKEND})...")
    landscape = 255 - fg
    del fg
    instances = watershed(landscape, markers=seeds, mask=foreground, compactness=0)
    del landscape, foreground, seeds

    # --- Remove small objects ---
    print(f"Removing objects < {min_size} voxels...")
    instances = remove_small_objects(instances.astype(bool) if min_size == 0 else instances,
                                     min_size=min_size)
    n_instances = len(np.unique(instances)) - 1  # subtract background
    print(f"Instances kept    : {n_instances:,}")

    # --- Save ---
    seg_dtype = _label_dtype(instances)
    output_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(output_path), mode='w')
    root.create_array(
        name    = '0',
        data    = instances.astype(seg_dtype),
        chunks  = (64, 64, 64),
        shards  = (512, 512, 512),
        overwrite = True,
    )

    # --- Optional: upload raw + instances to WebKnossos ---
    if upload is not None:
        upload_to_webknossos(instances, **upload)

    del instances

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  →  {output_path}")
    print(f"Instances: {n_instances:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Instance segmentation from foreground + boundary predictions.'
    )
    p.add_argument('--input',        required=True, type=Path,
                   help='Prediction zarr (shape Z,Y,X,2  uint8).')
    p.add_argument('--output',       required=True, type=Path,
                   help='Output zarr path for instance labels (uint16).')
    p.add_argument('--fg_threshold', type=float, default=0.5,
                   help='Foreground probability threshold (default: 0.5).')
    p.add_argument('--bd_threshold', type=float, default=0.4,
                   help='Boundary probability threshold (default: 0.4).')
    p.add_argument('--min_size',     type=int,   default=500,
                   help='Minimum object size in voxels (default: 500 = ~137 µm³ = ~6.4 µm sphere at 650 nm voxels).')

    # --- Optional WebKnossos upload ---
    try:
        from config import DEFAULT_CONFIG as _cfg
        _norm_min, _norm_max = _cfg.norm_min, _cfg.norm_max
    except Exception:
        _norm_min, _norm_max = 0.0, 1.0

    p.add_argument('--upload_wk',    action='store_true',
                   help='Upload raw + instances to WebKnossos after postprocessing.')
    p.add_argument('--raw',          type=Path, default=None,
                   help='Raw image volume (tif dir / .tif / .zarr); required with --upload_wk.')
    p.add_argument('--sample_name',  default=None,
                   help='Sample name; dataset is named IMATREC_<sample_name> (default: --raw stem).')
    p.add_argument('--wk_url',       default='https://webknossos.embl-hamburg.de',
                   help='WebKnossos server URL.')
    p.add_argument('--wk_token',     default=os.environ.get('WK_TOKEN'),
                   help='WebKnossos API token (default: WK_TOKEN env var).')
    p.add_argument('--wk_folder',    default='IMATREC',
                   help='Target folder on the WebKnossos server (default: IMATREC).')
    p.add_argument('--wk_voxel_size', type=float, nargs=3, default=[650.0, 650.0, 650.0],
                   metavar=('X', 'Y', 'Z'), help='Voxel size in nm (default: 650 650 650).')
    p.add_argument('--norm_min',     type=float, default=_norm_min,
                   help='Lower intensity bound for raw→uint8 (default: config value).')
    p.add_argument('--norm_max',     type=float, default=_norm_max,
                   help='Upper intensity bound for raw→uint8 (default: config value).')
    args = p.parse_args()

    if not args.input.exists():
        p.error(f"Input not found: {args.input}")

    upload = None
    if args.upload_wk:
        if not args.raw:
            p.error("--upload_wk requires --raw (the original image volume).")
        if not args.raw.exists():
            p.error(f"--raw not found: {args.raw}")
        if not args.wk_token:
            p.error("--upload_wk requires --wk_token or the WK_TOKEN environment variable.")
        sample_name = args.sample_name or args.raw.stem or args.output.stem
        upload = dict(
            raw_path    = args.raw,
            sample_name = sample_name,
            url         = args.wk_url,
            token       = args.wk_token,
            voxel_size  = tuple(args.wk_voxel_size),
            norm_min    = args.norm_min,
            norm_max    = args.norm_max,
            folder      = args.wk_folder,
        )

    postprocess(
        input_path   = args.input,
        output_path  = args.output,
        fg_threshold = args.fg_threshold,
        bd_threshold = args.bd_threshold,
        min_size     = args.min_size,
        upload       = upload,
    )


if __name__ == '__main__':
    main()
