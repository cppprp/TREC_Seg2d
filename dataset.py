"""
dataset.py
----------
PyTorch Datasets that serve 2D training samples: randomly-tilted slices from
annotated 3D patches (TiltedSliceDataset) and random crops from fully-annotated
2D images (FlatSliceDataset).

Expected folder layout (same as your existing 3D training setup):

    patches_dir/
        volume_001/
            image.npy   or   image.zarr/
            mask.npy    or   mask.zarr/
        volume_002/
            ...

Usage:
    dataset = TiltedSliceDataset(patches_dir='path/to/patches')
    loader  = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    for images, targets, weights in loader:
        ...  # images: (B, C, H, W),  targets: (B, 2, H, W),  weights: (B, 2, H, W)
"""

import numpy as np
import torch
import zarr
import tifffile
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

from extract_slices import extract_random_slice, mask_transform_2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_volume(path: Path) -> np.ndarray:
    """Load a volume from .npy, .zarr directory, or .tif/.tiff."""
    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix in ('.tif', '.tiff'):
        return tifffile.imread(str(path))
    elif path.is_dir():
        return np.array(zarr.open(str(path), mode='r')['0'])
    else:
        raise ValueError(f"Unknown volume format: {path}  (expected .npy, .tif, or .zarr/)")


def _find_patch_pairs(patches_dir: Path) -> list[tuple[Path, Path]]:
    """
    Scan patches_dir for (image, mask) pairs.
    Accepts both .npy and .zarr layouts.
    """
    pairs = []
    for vol_dir in sorted(patches_dir.iterdir()):
        if not vol_dir.is_dir():
            continue

        # Try .npy first, then .zarr, then .tif/.tiff
        image_path = mask_path = None
        for ext in ['.npy', '.zarr', '.tif', '.tiff']:
            img_candidate  = vol_dir / f'image{ext}'
            mask_candidate = vol_dir / f'mask{ext}'
            if img_candidate.exists() and mask_candidate.exists():
                image_path = img_candidate
                mask_path  = mask_candidate
                break

        if image_path is None:
            print(f"  Warning: no image/mask pair found in {vol_dir}, skipping.")
            continue

        pairs.append((image_path, mask_path))

    return pairs


def _compute_sample_weight(mask_slice: np.ndarray) -> torch.Tensor:
    """
    Per-pixel loss weight map — (2, H, W) float32.
    Upweights foreground and boundary pixels to counter background dominance.
    Returns a weight tensor matching the 2-channel target shape.
    """
    foreground = (mask_slice > 0).astype(np.float32)
    n_fg = foreground.sum()
    n_bg = foreground.size - n_fg

    # Weight ratio: background gets 1.0, foreground gets bg/fg (capped at 10)
    if n_fg > 0:
        w_fg = min(n_bg / (n_fg + 1e-6), 10.0)
    else:
        w_fg = 1.0

    weight_map = np.where(foreground > 0, w_fg, 1.0).astype(np.float32)

    # Same weight map for both channels (foreground + boundary)
    return torch.tensor(np.stack([weight_map, weight_map], axis=0))  # (2, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TiltedSliceDataset(Dataset):
    """
    Online random slice extraction from 3D annotated patches.

    Args:
        patches_dir     : root folder containing per-volume subdirectories
        slices_per_patch: how many virtual samples each patch contributes per epoch
        output_size     : H and W of extracted slices in pixels
        n_channels      : 1 = single 2D slice,  3 = 2.5D triplet (central ± spacing)
        channel_spacing : voxel spacing between 2.5D channels
        max_tilt_deg    : maximum tilt from XY plane (90 = fully random)
        augment         : apply random flips during training
        preload         : load all patches into RAM at init (faster if RAM allows)
        fg_min_frac     : minimum fraction of foreground pixels to keep a slice
                          (set 0.0 to keep all slices including background-only)
        norm_min        : fixed intensity lower bound (clipped and scaled to 0)
        norm_max        : fixed intensity upper bound (clipped and scaled to 1)
    """

    def __init__(
        self,
        patches_dir:      str | Path,
        slices_per_patch: int   = 20,
        output_size:      int   = 256,
        n_channels:       int   = 1,
        channel_spacing:  float = 1.0,
        max_tilt_deg:     float = 90.0,
        augment:          bool  = True,
        preload:          bool  = True,
        fg_min_frac:      float = 0.0,
        norm_min:         float = 0.0,
        norm_max:         float = 1.0,
    ):
        self.patches_dir      = Path(patches_dir)
        self.slices_per_patch = slices_per_patch
        self.output_size      = output_size
        self.n_channels       = n_channels
        self.channel_spacing  = channel_spacing
        self.max_tilt_deg     = max_tilt_deg
        self.augment          = augment
        self.fg_min_frac      = fg_min_frac
        self.norm_min         = norm_min
        self.norm_max         = norm_max

        # Discover patch pairs
        pairs = _find_patch_pairs(self.patches_dir)
        if len(pairs) == 0:
            raise FileNotFoundError(f"No image/mask pairs found under {self.patches_dir}")
        print(f"Found {len(pairs)} patch pairs in {self.patches_dir}")

        # Load volumes
        if preload:
            print("Preloading patches into RAM...")
            self.patches = []
            for img_path, mask_path in pairs:
                img  = _load_volume(img_path).astype(np.float32)
                mask = _load_volume(mask_path).astype(np.uint8)
                self.patches.append((img, mask))
                print(f"  {img_path.parent.name}  image{img.shape}  mask{mask.shape}")
        else:
            self.patch_paths = pairs
            self.patches     = None  # lazy load in __getitem__

        # Augmentation transforms (applied to image + mask together)
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]) if augment else None

    def __len__(self) -> int:
        n = len(self.patches) if self.patches is not None else len(self.patch_paths)
        return n * self.slices_per_patch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image   : (C, H, W) float32, normalised to [0, 1]
            target  : (2, H, W) float32  — channel 0 = foreground, channel 1 = boundary
            weight  : (2, H, W) float32  — per-pixel loss weights
        """
        patch_idx = idx % len(self.patches if self.patches is not None else self.patch_paths)

        # Load patch (from RAM or disk)
        if self.patches is not None:
            image_patch, mask_patch = self.patches[patch_idx]
        else:
            img_path, mask_path = self.patch_paths[patch_idx]
            image_patch = _load_volume(img_path).astype(np.float32)
            mask_patch  = _load_volume(mask_path).astype(np.uint8)

        # Extract a valid tilted slice, retry until success
        result = None
        for _ in range(50):  # hard cap to avoid infinite loop
            result = extract_random_slice(
                image_patch, mask_patch,
                output_size=self.output_size,
                n_channels=self.n_channels,
                channel_spacing=self.channel_spacing,
                max_tilt_deg=self.max_tilt_deg,
            )
            if result is None:
                continue
            img_slice, mask_slice = result
            # Optionally enforce minimum foreground fraction
            if self.fg_min_frac > 0 and (mask_slice > 0).mean() < self.fg_min_frac:
                result = None
                continue
            break

        if result is not None:
            img_slice, mask_slice = result
        else:
            # Fallback: return a zero slice (very rare, only if patch is tiny)
            img_slice  = np.zeros((self.n_channels, self.output_size, self.output_size), np.float32)
            mask_slice = np.zeros((self.output_size, self.output_size), np.uint8)

        # Normalise image to [0, 1] using fixed window
        img_slice = np.clip(img_slice, self.norm_min, self.norm_max)
        img_slice = (img_slice - self.norm_min) / (self.norm_max - self.norm_min)

        # Build target + weight
        target = mask_transform_2d(mask_slice)           # (2, H, W) float32 tensor
        weight = _compute_sample_weight(mask_slice)      # (2, H, W) float32 tensor

        image_t = torch.tensor(img_slice, dtype=torch.float32)  # (C, H, W)

        # Augmentation (flips only — safe for both image and mask)
        if self.transforms is not None:
            image_t = tv_tensors.Image(image_t)
            target  = tv_tensors.Mask(target)
            weight  = tv_tensors.Mask(weight)
            image_t, target, weight = self.transforms(image_t, target, weight)
            image_t = image_t.as_subclass(torch.Tensor)
            target  = target.as_subclass(torch.Tensor)
            weight  = weight.as_subclass(torch.Tensor)

        return image_t, target, weight


# ---------------------------------------------------------------------------
# 2D flat-image dataset
# ---------------------------------------------------------------------------

class FlatSliceDataset(Dataset):
    """
    Random crops from fully-annotated 2D images.

    Expected layout:
        patches_2d_dir/
            image_001/
                image.tif   (H, W) — or .npy
                mask.tif    (H, W) uint8 instance labels
            image_002/
                ...

    Always returns (1, H, W) images regardless of n_channels — 2.5D has no
    meaning for 2D sources.  Must be used with n_channels=1 in train.py.
    """

    def __init__(
        self,
        patches_dir:      str | Path,
        patches_per_image: int  = 30,
        output_size:      int   = 256,
        augment:          bool  = True,
        preload:          bool  = True,
        norm_min:         float = 0.0,
        norm_max:         float = 1.0,
    ):
        self.patches_dir       = Path(patches_dir)
        self.patches_per_image = patches_per_image
        self.output_size       = output_size
        self.augment           = augment
        self.norm_min          = norm_min
        self.norm_max          = norm_max

        pairs = _find_patch_pairs(self.patches_dir)
        if len(pairs) == 0:
            raise FileNotFoundError(f"No image/mask pairs found under {self.patches_dir}")
        print(f"Found {len(pairs)} 2D image pairs in {self.patches_dir}")

        if preload:
            print("Preloading 2D images into RAM...")
            self.patches = []
            for img_path, mask_path in pairs:
                img  = _load_volume(img_path).astype(np.float32)
                mask = _load_volume(mask_path).astype(np.uint8)
                if img.ndim != 2 or mask.ndim != 2:
                    raise ValueError(
                        f"FlatSliceDataset expects 2-D arrays; "
                        f"got image {img.shape} mask {mask.shape} in {img_path.parent}"
                    )
                self.patches.append((img, mask))
                print(f"  {img_path.parent.name}  image{img.shape}  mask{mask.shape}")
        else:
            self.patch_paths = pairs
            self.patches     = None

        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]) if augment else None

    def __len__(self) -> int:
        n = len(self.patches) if self.patches is not None else len(self.patch_paths)
        return n * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_pairs = len(self.patches) if self.patches is not None else len(self.patch_paths)
        patch_idx = idx % n_pairs

        if self.patches is not None:
            image, mask = self.patches[patch_idx]
        else:
            img_path, mask_path = self.patch_paths[patch_idx]
            image = _load_volume(img_path).astype(np.float32)
            mask  = _load_volume(mask_path).astype(np.uint8)

        H, W = image.shape
        S = self.output_size

        # Random crop (reflect-pad if image is smaller than crop size)
        if H < S or W < S:
            pad_h = max(0, S - H)
            pad_w = max(0, S - W)
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask  = np.pad(mask,  ((0, pad_h), (0, pad_w)), mode='reflect')
            H, W  = image.shape

        y0 = np.random.randint(0, H - S + 1)
        x0 = np.random.randint(0, W - S + 1)
        img_crop  = image[y0:y0+S, x0:x0+S]
        mask_crop = mask[y0:y0+S, x0:x0+S]

        # Normalise to [0, 1] using fixed window
        img_crop = np.clip(img_crop, self.norm_min, self.norm_max)
        img_crop = (img_crop - self.norm_min) / (self.norm_max - self.norm_min)

        target = mask_transform_2d(mask_crop)       # (2, H, W) float32
        weight = _compute_sample_weight(mask_crop)  # (2, H, W) float32
        image_t = torch.tensor(img_crop[None], dtype=torch.float32)  # (1, H, W)

        if self.transforms is not None:
            image_t = tv_tensors.Image(image_t)
            target  = tv_tensors.Mask(target)
            weight  = tv_tensors.Mask(weight)
            image_t, target, weight = self.transforms(image_t, target, weight)
            image_t = image_t.as_subclass(torch.Tensor)
            target  = target.as_subclass(torch.Tensor)
            weight  = weight.as_subclass(torch.Tensor)

        return image_t, target, weight


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    patches_dir = sys.argv[1] if len(sys.argv) > 1 else './patches'
    ds = TiltedSliceDataset(patches_dir, slices_per_patch=5, output_size=256)
    print(f"Dataset length: {len(ds)}")
    img, tgt, wgt = ds[0]
    print(f"  image : {img.shape}  {img.dtype}  [{img.min():.2f}, {img.max():.2f}]")
    print(f"  target: {tgt.shape}  {tgt.dtype}  fg={tgt[0].mean():.3f}  bd={tgt[1].mean():.3f}")
    print(f"  weight: {wgt.shape}  {wgt.dtype}  [{wgt.min():.2f}, {wgt.max():.2f}]")