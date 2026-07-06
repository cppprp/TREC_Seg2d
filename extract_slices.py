"""
extract_slices.py
-----------------
Core tilted-slice extraction logic.
Camera is inlined below (originally from interactive_unet.volume.camera).

Imported by: check_slices.py, dataset.py
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from skimage.morphology import disk, dilation
from skimage.segmentation import find_boundaries


# ---------------------------------------------------------------------------
# Camera — inlined from interactive_unet.volume.camera
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """
    A dataclass that stores parameters of a camera.
    """
    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))

    u: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    w: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))

    zoom: float = 1.0

    def __post_init__(self):
        self._update_orientation_vectors(self.u)

    def copy(self):
        cls = self.__class__
        cam = cls.__new__(cls)
        cam.origin = self.origin.copy()
        cam.u = self.u.copy()
        cam.v = self.v.copy()
        cam.w = self.w.copy()
        cam.zoom = self.zoom
        return cam

    def to_dict(self):
        return {
            "origin": self.origin.tolist(),
            "u": self.u.tolist(),
            "v": self.v.tolist(),
            "w": self.w.tolist(),
            "zoom": float(self.zoom),
        }

    @classmethod
    def from_dict(cls, d):
        cam = cls()
        cam.origin = np.array(d["origin"], dtype=np.float32)
        cam.u = np.array(d["u"], dtype=np.float32)
        cam.v = np.array(d["v"], dtype=np.float32)
        cam.w = np.array(d["w"], dtype=np.float32)
        cam.zoom = float(d["zoom"])
        return cam

    def reset(self, volume_shape):
        self.origin = volume_shape / 2
        self.u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.w = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.zoom = 1.0

    @property
    def uvw(self):
        return self.u.astype(np.float32), self.v.astype(np.float32), self.w.astype(np.float32)

    def plane_coords(self, p):
        r = np.asarray(p, np.float32) - self.origin.astype(np.float32)
        return float(r @ self.u), float(r @ self.v), float(r @ self.w)

    def world_coords(self, d, y, x):
        return self.origin + float(d) * self.u + float(y) * self.v + float(x) * self.w

    def slice_axes(self, axis=0):
        basis = self.uvw
        n = basis[axis]
        a0, a1 = [basis[i] for i in range(3) if i != axis]
        return n, a0, a1

    def _normalize(self, v):
        return v / np.linalg.norm(v)

    def _orthonormalize(self):
        u = self._normalize(self.u)
        v = self.v - (u @ self.v) * u
        v = self._normalize(v)
        w = np.cross(u, v)
        w = self._normalize(w)
        self.u, self.v, self.w = u, v, w

    def _update_orientation_vectors(self, rotation_vector):
        rotation_vector = self._normalize(rotation_vector)
        rot, _ = Rotation.align_vectors(
            [rotation_vector.astype(np.float64)],
            [[1.0, 0.0, 0.0]]
        )
        M = rot.as_matrix().astype(np.float32)
        self.u = M @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.v = M @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.w = M @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._orthonormalize()

    def _generate_uniformly_random_unit_vector(self, ndim=3):
        u = np.random.normal(size=ndim)
        while np.linalg.norm(u) < 0.0001:
            u = np.random.normal(size=ndim)
        return self._normalize(u)

    def randomize(self):
        rotation_vector = self._generate_uniformly_random_unit_vector().astype(np.float32)
        self._update_orientation_vectors(rotation_vector)

    def _translate(self, t=(0, 0, 0)):
        t = np.array(t, dtype=np.float32) * float(self.zoom)
        delta_world = t[0] * self.u + t[1] * self.v + t[2] * self.w
        self.origin = self.origin + delta_world

    def pan(self, dx, dy):
        self._translate((0, dy, dx))

    def scroll(self, dz):
        self._translate((dz, 0, 0))

    def rotate(self, dx, dy, sensitivity=0.002, tol=1e-8):
        dx, dy = float(dx), float(dy)
        drag = -dy * self.v + dx * self.w
        drag_norm = float(np.linalg.norm(drag))
        if drag_norm < tol:
            return
        axis = np.cross(self.u, drag)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < tol:
            return
        axis /= axis_norm
        theta = drag_norm * float(sensitivity)
        R = Rotation.from_rotvec(axis.astype(np.float64) * theta)
        self.u = R.apply(self.u.astype(np.float64)).astype(np.float32)
        self.v = R.apply(self.v.astype(np.float64)).astype(np.float32)
        self.w = R.apply(self.w.astype(np.float64)).astype(np.float32)
        self._orthonormalize()

    def rotate_axis(self, axis, angle):
        rot_axis = {"u": self.u, "v": self.v, "w": self.w}[axis]
        rot_axis = self._normalize(rot_axis)
        R = Rotation.from_rotvec(rot_axis * float(angle))
        self.u = R.apply(self.u.astype(np.float64)).astype(np.float32)
        self.v = R.apply(self.v.astype(np.float64)).astype(np.float32)
        self.w = R.apply(self.w.astype(np.float64)).astype(np.float32)
        self._orthonormalize()

    def zoom_by(self, zoom_factor):
        self.zoom *= zoom_factor


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
        label_boundaries = dilation(label_boundaries, disk(1))
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