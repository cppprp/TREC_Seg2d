# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tifffile",
#   "imagecodecs",
#   "zarr>=3",
#   "numpy",
# ]
# ///
"""
convert_tif_to_zarr.py
----------------------
Convert annotated TIF patches to zarr format for training.

Walks an ML_training_data directory tree:

    ML_training_data/
        ATH_10to40_20240702_AM_01_epo_01_P1/
            patch_0000.tif
            patch_0000.tif.labels.tif   ← variant 1
            patch_0003.tif
            patch_0003.labels.tif       ← variant 2
            ...
        ATH_10to40_20240702_AM_01_epo_01_P2/
            ...

Produces paired zarr stores in a flat output directory:

    output_dir/
        ATH_10to40_20240702_AM_01_epo_01_P1_patch_0000/
            image.zarr/
            mask.zarr/
        ...

The output layout is directly consumable by dataset.py / TiltedSliceDataset.

Usage:
    uv run convert_tif_to_zarr.py --input /path/to/ML_training_data --output ./patches
    uv run convert_tif_to_zarr.py --input /path/to/ML_training_data --output ./patches --overwrite
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

CHUNKS = (64, 64, 64)


def find_label(image_path: Path) -> Path | None:
    """Return the label file for image_path, or None if not found.

    Checks two naming conventions:
      1. patch_0000.tif  →  patch_0000.tif.labels.tif
      2. patch_0000.tif  →  patch_0000.labels.tif
    """
    candidate1 = image_path.parent / (image_path.name + '.labels.tif')
    if candidate1.exists():
        return candidate1

    candidate2 = image_path.parent / (image_path.stem + '.labels.tif')
    if candidate2.exists():
        return candidate2

    return None


def write_zarr(array: np.ndarray, store_path: Path) -> None:
    z = zarr.open_group(str(store_path), mode='w')
    z.create_array('0', data=array, chunks=CHUNKS)


def convert(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_found = 0
    n_converted = 0
    n_skipped_no_label = 0
    n_skipped_exists = 0

    location_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not location_dirs:
        print(f"No subdirectories found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    for loc_dir in location_dirs:
        tif_files = sorted(loc_dir.glob('*.tif'))
        for tif_path in tif_files:
            # Skip label files themselves
            if '.labels.' in tif_path.name:
                continue

            label_path = find_label(tif_path)
            if label_path is None:
                n_skipped_no_label += 1
                continue

            n_found += 1
            out_subdir = output_dir / f"{loc_dir.name}_{tif_path.stem}"
            image_store = out_subdir / 'image.zarr'
            mask_store  = out_subdir / 'mask.zarr'

            if image_store.exists() and mask_store.exists() and not overwrite:
                n_skipped_exists += 1
                print(f"  [skip]  {out_subdir.name}  (already exists)")
                continue

            print(f"  [convert]  {loc_dir.name}/{tif_path.name} + {label_path.name}")

            image = tifffile.imread(str(tif_path)).astype(np.float32)
            mask  = tifffile.imread(str(label_path)).astype(np.uint8)

            if image.shape != mask.shape:
                print(f"    SKIPPING: shape mismatch — image{image.shape} mask{mask.shape}", file=sys.stderr)
                n_found -= 1
                continue

            out_subdir.mkdir(parents=True, exist_ok=True)
            write_zarr(image, image_store)
            write_zarr(mask,  mask_store)
            n_converted += 1

    print()
    print("── Summary ──────────────────────────────────────────────")
    print(f"  Labelled pairs found  : {n_found}")
    print(f"  Converted             : {n_converted}")
    print(f"  Skipped (no label)    : {n_skipped_no_label}")
    print(f"  Skipped (exists)      : {n_skipped_exists}")
    print(f"  Output directory      : {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TIF training patches to zarr format.")
    parser.add_argument('--input',     required=True, type=Path,
                        help="Root ML_training_data directory containing location subdirs.")
    parser.add_argument('--output',    required=True, type=Path,
                        help="Output directory for zarr patch pairs.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing zarr stores (default: skip).")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")

    convert(args.input, args.output, args.overwrite)


if __name__ == '__main__':
    main()
