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
# Core
# ---------------------------------------------------------------------------

def postprocess(
    input_path:    Path,
    output_path:   Path,
    fg_threshold:  float = 0.5,
    bd_threshold:  float = 0.4,
    min_size:      int   = 500,
):
    t0 = time.time()

    # --- Load predictions ---
    pred_store = zarr.open(str(input_path), mode='r')['0']
    print(f"Prediction shape : {pred_store.shape}  dtype={pred_store.dtype}")
    print("Loading foreground channel...")
    fg_prob = pred_store[:, :, :, 0].astype(np.float32) / 255.0   # (Z, Y, X)
    print("Loading boundary channel...")
    bd_prob = pred_store[:, :, :, 1].astype(np.float32) / 255.0   # (Z, Y, X)

    # --- Threshold ---
    foreground = fg_prob > fg_threshold   # bool (Z, Y, X)
    boundary   = bd_prob > bd_threshold   # bool (Z, Y, X)
    del bd_prob

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
    # Landscape: invert foreground probability so high-confidence voxels fill first
    print(f"Running watershed ({_WATERSHED_BACKEND})...")
    instances = watershed(-fg_prob, markers=seeds, mask=foreground, compactness=0)
    del fg_prob, foreground, seeds

    # --- Remove small objects ---
    print(f"Removing objects < {min_size} voxels...")
    instances = remove_small_objects(instances.astype(bool) if min_size == 0 else instances,
                                     min_size=min_size)
    n_instances = len(np.unique(instances)) - 1  # subtract background
    print(f"Instances kept    : {n_instances:,}")

    # --- Save ---
    output_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(output_path), mode='w')
    root.create_array(
        name    = '0',
        data    = instances.astype(np.uint32),
        chunks  = (64, 64, 64),
        shards  = (512, 512, 512),
        overwrite = True,
    )
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
                   help='Output zarr path for instance labels (uint32).')
    p.add_argument('--fg_threshold', type=float, default=0.5,
                   help='Foreground probability threshold (default: 0.5).')
    p.add_argument('--bd_threshold', type=float, default=0.4,
                   help='Boundary probability threshold (default: 0.4).')
    p.add_argument('--min_size',     type=int,   default=500,
                   help='Minimum object size in voxels (default: 500 = ~137 µm³ = ~6.4 µm sphere at 650 nm voxels).')
    args = p.parse_args()

    if not args.input.exists():
        p.error(f"Input not found: {args.input}")

    postprocess(
        input_path   = args.input,
        output_path  = args.output,
        fg_threshold = args.fg_threshold,
        bd_threshold = args.bd_threshold,
        min_size     = args.min_size,
    )


if __name__ == '__main__':
    main()
