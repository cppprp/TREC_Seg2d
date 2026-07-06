"""
Stage 1 — Visual sanity check for tilted slice extraction.

Run from your project root:
    python check_slices.py --patch path/to/patch.zarr --mask path/to/mask.zarr --n 20

Or pass numpy arrays directly if you have them saved as .npy:
    python check_slices.py --patch patch.npy --mask mask.npy --n 20

Output: ./slice_check/  folder with side-by-side image+mask PNGs.
What to look for:
  - Plankton cross-sections at various angles (circles, ellipses, odd shapes)
  - Mask outlines aligning with visible organisms
  - No systematic blank/black strips (would indicate coord bug)
  - Roughly 50%+ of slices containing at least one plankton
"""

import argparse
import numpy as np
import zarr
import tifffile
from pathlib import Path
from PIL import Image, ImageDraw
from extract_slices import extract_random_slice, mask_transform_2d


def load_volume(path):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix in ('.tif', '.tiff'):
        return tifffile.imread(str(path))
    elif path.is_dir():  # zarr
        return np.array(zarr.open(str(path), mode='r')['0'])
    else:
        raise ValueError(f"Unknown format: {path}")


def save_check_image(img_slice, mask_slice, out_path, channel=0):
    """
    Save a side-by-side PNG: raw image | foreground overlay | boundary overlay
    img_slice : (C, H, W) float32  (we show channel 0 = central slice)
    mask_slice: (H, W) uint8 instance labels
    """
    H, W = mask_slice.shape

    # Normalise image channel to uint8
    img = img_slice[channel]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_u8 = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_u8 = np.zeros_like(img, dtype=np.uint8)

    # Compute foreground + boundary from instance mask
    fg_bg = mask_transform_2d(mask_slice)  # (2, H, W) tensor
    foreground = fg_bg[0].numpy().astype(np.uint8) * 255
    boundary   = fg_bg[1].numpy().astype(np.uint8) * 255

    # Build 3-panel image: raw | foreground | boundary
    panel_w = W
    canvas = Image.new('RGB', (panel_w * 3 + 4, H), color=(40, 40, 40))

    # Panel 1: raw image
    raw_rgb = Image.fromarray(img_u8).convert('RGB')
    canvas.paste(raw_rgb, (0, 0))

    # Panel 2: foreground overlay (green)
    fg_rgb = Image.fromarray(img_u8).convert('RGB')
    fg_mask_img = Image.fromarray(foreground)
    green_layer = Image.new('RGB', fg_rgb.size, (0, 200, 80))
    fg_rgb = Image.composite(green_layer, fg_rgb,
                             Image.fromarray((foreground * 0.5).astype(np.uint8)))
    canvas.paste(fg_rgb, (panel_w + 2, 0))

    # Panel 3: boundary overlay (red)
    bd_rgb = Image.fromarray(img_u8).convert('RGB')
    red_layer = Image.new('RGB', bd_rgb.size, (220, 50, 50))
    bd_rgb = Image.composite(red_layer, bd_rgb,
                             Image.fromarray((boundary * 0.6).astype(np.uint8)))
    canvas.paste(bd_rgb, (panel_w * 2 + 4, 0))

    # Labels
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2),            "image",      fill=(255,255,100))
    draw.text((panel_w + 6, 2),  "foreground", fill=(255,255,100))
    draw.text((panel_w*2 + 6, 2),"boundary",   fill=(255,255,100))

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch',      required=True,  help='Path to image patch (.npy or .zarr dir)')
    parser.add_argument('--mask',       required=True,  help='Path to mask patch  (.npy or .zarr dir)')
    parser.add_argument('--n',          type=int, default=20,  help='Number of slices to extract')
    parser.add_argument('--size',       type=int, default=256, help='Output slice size in pixels')
    parser.add_argument('--channels',   type=int, default=1,   help='1=single slice, 3=2.5D triplet')
    parser.add_argument('--max_tilt',   type=float, default=90.0, help='Max tilt in degrees (90=fully random)')
    parser.add_argument('--out',        default='./slice_check', help='Output folder')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading volumes...")
    image_patch = load_volume(args.patch).astype(np.float32)
    mask_patch  = load_volume(args.mask).astype(np.uint8)

    print(f"Image shape: {image_patch.shape}  dtype: {image_patch.dtype}")
    print(f"Mask  shape: {mask_patch.shape}   dtype: {mask_patch.dtype}")
    print(f"Unique mask labels: {np.unique(mask_patch)}")
    print(f"\nExtracting {args.n} slices...")

    n_saved   = 0
    n_attempts = 0
    n_with_fg  = 0

    while n_saved < args.n and n_attempts < args.n * 20:
        n_attempts += 1
        result = extract_random_slice(
            image_patch, mask_patch,
            output_size=args.size,
            n_channels=args.channels,
            max_tilt_deg=args.max_tilt,
        )
        if result is None:
            continue

        img_slice, mask_slice = result
        has_fg = mask_slice.max() > 0
        if has_fg:
            n_with_fg += 1

        fname = out_dir / f"slice_{n_saved:03d}{'_fg' if has_fg else ''}.png"
        save_check_image(img_slice, mask_slice, fname)
        n_saved += 1

    print(f"\nSaved {n_saved} slices to {out_dir}/")
    print(f"  {n_with_fg}/{n_saved} contain at least one plankton ({100*n_with_fg//max(n_saved,1)}%)")
    print(f"  {n_attempts} attempts needed (efficiency: {100*n_saved//max(n_attempts,1)}%)")
    print(f"\nOpen {out_dir}/ and check:")
    print("  - Plankton visible as ellipses/blobs at various orientations")
    print("  - Green foreground overlay matches organisms")
    print("  - Red boundary overlay traces edges correctly")


if __name__ == '__main__':
    main()