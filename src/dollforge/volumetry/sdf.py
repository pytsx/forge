from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes


def signed_distance_field(occupancy: np.ndarray, voxel_size: float) -> np.ndarray:
    occupancy = np.asarray(occupancy, dtype=bool)
    if occupancy.ndim != 3:
        raise ValueError("Occupancy must be a 3D grid")
    spacing = np.asarray(voxel_size, dtype=np.float64)
    if np.any(spacing <= 0):
        raise ValueError("voxel_size must be positive")
    if not occupancy.any():
        raise ValueError("Cannot build SDF from an empty occupancy grid")

    sampling = tuple(float(value) for value in np.broadcast_to(spacing, 3))
    inside = distance_transform_edt(occupancy, sampling=sampling)
    outside = distance_transform_edt(~occupancy, sampling=sampling)
    # The grid axes are x/y/z throughout Forge; scipy arrays are indexed in
    # that same order because the occupancy field is created explicitly.
    return (outside - inside).astype(np.float32)


def extract_zero_surface(
    sdf: np.ndarray,
    origin_xyz: tuple[float, float, float],
    voxel_size: float | tuple[float, float, float],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    field = np.asarray(sdf, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError("SDF must be a 3D grid")
    if field.min() >= 0 or field.max() <= 0:
        raise ValueError("SDF must contain both inside and outside samples")

    spacing = tuple(float(value) for value in np.broadcast_to(voxel_size, 3))
    vertices, faces, _normals, _values = marching_cubes(
        field,
        level=0.0,
        spacing=spacing,
        allow_degenerate=False,
    )
    vertices += np.asarray(origin_xyz, dtype=np.float64)
    return vertices.tolist(), faces.astype(np.int64).tolist()
