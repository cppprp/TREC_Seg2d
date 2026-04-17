"""
extract_slices.py
-----------------
Core tilted-slice extraction logic.
Reuses Camera directly from interactive_unet — no reinventing.

Imported by: check_slices.py, dataset_2d.py, train_2d.py
"""

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from skimage.morphology import disk, binary_dilation
from skimage.segmentation import find_boundaries

from interactive_unet.volume.camera import Camera


# ---------------------------------------------------------------------------
# Mask transform — identical logic to your 3D version, just disk() not ball()
# ---------------------------------------------------------------------------

def mask_transform_2d(mask_2d: np.ndarray) -> torch.Tensor:
    """
    Convert a 2D instance-labelled mask → (2, H, W) float32 tensor.
        Channel 0: foreground  (any labelled pixel)
        Channel 1: boundary    (edges between instances, dilated by disk(1))

    Args:
        mask_2d: (H, W) uint8/int, each plankton has a unique integer ID, 0 = background

    Returns:
        torch.Tensor of shape (2, H, W), dtype float32, values in {0.0, 1.0}
    """
    foreground = (mask_2d > 0).astype(np.float32)

    boundaries = np.zeros_like(foreground, dtype=bool)
    for label_id in np.unique(mask_2d):
        if label_id == 0:
            continue
        label_mask = (mask_2d == label_id)
        label_boundaries = find_boundaries(label_mask, mode='thick')
        label_boundaries = binary_dilation(label_boundaries, disk(1))
        boundaries = np.logical_or(boundaries, label_boundaries)

    return torch.stack([
        torch.tensor(foreground),
        torch.tensor(boundaries.astype(np.float32))
    ])  # (2, H, W)


# ---------------------------------------------------------------------------
# Single tilted slice extractor
# ---------------------------------------------------------------------------

def extract_random_slice(
    image_patch: np.ndarray,
    mask_patch:  np.ndarray,
    output_size: int   = 256,
    n_channels:  int   = 1,       # 1 = single slice, 3 = 2.5D triplet
    channel_spacing: float = 1.0, # voxels between channels for 2.5D
    max_tilt_deg: float = 90.0,   # 90 = fully random, <90 = restrict tilt from XY plane
    min_valid_frac: float = 0.5,  # discard slice if <50% pixels land inside the patch
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract one randomly-oriented 2D slice (or 2.5D triplet) from a 3D patch.

    Returns:
        (img_slice, mask_slice) or None if the slice fell mostly outside the patch.
        img_slice  : (n_channels, H, W) float32  — NOT normalised yet
        mask_slice : (H, W) uint8 instance labels — ready for mask_transform_2d()

    The caller should retry if None is returned.
    """
    D, H, W = image_patch.shape

    # --- Camera centred on the patch, random orientation ---
    cam = Camera()

    if max_tilt_deg >= 89.9:
        # Fully random orientation — use Camera's built-in method
        cam.randomize()
    else:
        # Start axis-aligned (u = depth, v = rows, w = cols)
        # then tilt by at most max_tilt_deg away from the XY plane.
        # Analogy: tilt a coin lying flat on a table by at most N degrees.
        cam.reset(np.array([D, H, W], dtype=np.float32))
        tilt_rad = np.deg2rad(max_tilt_deg)
        angle    = np.random.uniform(0, tilt_rad)
        phi      = np.random.uniform(0, 2 * np.pi)
        # Random axis lying in the v-w (in-plane) direction
        axis = float(np.cos(phi)) * cam.v + float(np.sin(phi)) * cam.w
        axis = axis / np.linalg.norm(axis)
        R = Rotation.from_rotvec(axis.astype(np.float64) * angle)
        cam.u = R.apply(cam.u.astype(np.float64)).astype(np.float32)
        cam.v = R.apply(cam.v.astype(np.float64)).astype(np.float32)
        cam.w = R.apply(cam.w.astype(np.float64)).astype(np.float32)
        cam._orthonormalize()

    # Always pin the origin to the patch centre after randomize()/reset(),
    # which may set their own origin.
    cam.origin = np.array([D / 2.0, H / 2.0, W / 2.0], dtype=np.float32)

    # --- 2D sampling grid centred on origin ---
    half = output_size / 2.0
    ys = np.linspace(-half, half, output_size, dtype=np.float32)  # v direction
    xs = np.linspace(-half, half, output_size, dtype=np.float32)  # w direction
    gy, gx = np.meshgrid(ys, xs, indexing='ij')  # both (S, S)

    # Depth offsets per channel
    # Single: [0.0]   2.5D triplet: [-spacing, 0, +spacing]
    if n_channels == 1:
        depths = [0.0]
    else:
        half_ch = (n_channels - 1) / 2.0
        depths  = [(i - half_ch) * channel_spacing for i in range(n_channels)]

    # --- Validity check on the central slice ---
    # world_coords(d, y, x) = origin + d*u + y*v + x*w
    coords_center = (
        cam.origin[:, None, None]
        + gy[None] * cam.v[:, None, None]
        + gx[None] * cam.w[:, None, None]
    )  # (3, S, S)

    valid = (
        (coords_center[0] >= 0) & (coords_center[0] < D) &
        (coords_center[1] >= 0) & (coords_center[1] < H) &
        (coords_center[2] >= 0) & (coords_center[2] < W)
    )
    if valid.mean() < min_valid_frac:
        return None

    # --- Sample image at each depth offset ---
    img_channels = []
    for d in depths:
        coords = (
            cam.origin[:, None, None]
            + float(d) * cam.u[:, None, None]
            + gy[None]  * cam.v[:, None, None]
            + gx[None]  * cam.w[:, None, None]
        )  # (3, S, S)

        img_ch = map_coordinates(
            image_patch.astype(np.float32),
            coords, order=1, mode='constant', cval=0.0
        )  # (S, S)
        img_channels.append(img_ch)

    img_slice = np.stack(img_channels, axis=0)  # (C, S, S)

    # --- Sample mask at central slice, nearest-neighbour ---
    mask_slice = map_coordinates(
        mask_patch.astype(np.float32),
        coords_center, order=0, mode='constant', cval=0.0
    ).astype(np.uint8)  # (S, S)

    return img_slice, mask_slice


# ---------------------------------------------------------------------------
# Batch generator — multiple slices from one patch, with auto-retry
# ---------------------------------------------------------------------------

def generate_slices_from_patch(
    image_patch:  np.ndarray,
    mask_patch:   np.ndarray,
    n_slices:     int   = 20,
    output_size:  int   = 256,
    n_channels:   int   = 1,
    channel_spacing: float = 1.0,
    max_tilt_deg: float = 90.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate `n_slices` valid random slices from a single 3D patch.
    Returns list of (img_slice, mask_slice) tuples.
    """
    results   = []
    attempts  = 0
    max_attempts = n_slices * 15

    while len(results) < n_slices and attempts < max_attempts:
        attempts += 1
        out = extract_random_slice(
            image_patch, mask_patch,
            output_size=output_size,
            n_channels=n_channels,
            channel_spacing=channel_spacing,
            max_tilt_deg=max_tilt_deg,
        )
        if out is not None:
            results.append(out)

    return results