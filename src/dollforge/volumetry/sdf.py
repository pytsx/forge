from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes


def signed_distance_field(occupancy: np.ndarray, voxel_size: float) -> np.ndarray:
    occupancy = np.asarray(occupancy, dtype=bool)
    if occupancy.ndim != 3:
        raise ValueError("Occupancy must be a 3D grid")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if not occupancy.any():
        raise ValueError("Cannot build SDF from an empty occupancy grid")

    inside = distance_transform_edt(occupancy)
    outside = distance_transform_edt(~occupancy)
    return ((outside - inside) * voxel_size).astype(np.float32)


def extract_zero_surface(
    sdf: np.ndarray,
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    field = np.asarray(sdf, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError("SDF must be a 3D grid")
    if field.min() >= 0 or field.max() <= 0:
        raise ValueError("SDF must contain both inside and outside samples")

    vertices, faces, _normals, _values = marching_cubes(
        field,
        level=0.0,
        spacing=(voxel_size, voxel_size, voxel_size),
        allow_degenerate=False,
    )
    vertices += np.asarray(origin_xyz, dtype=np.float64)
    return vertices.tolist(), faces.astype(np.int64).tolist()
