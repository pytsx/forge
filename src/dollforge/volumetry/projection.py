from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dollforge.domain.models import CameraEstimate


@dataclass(frozen=True)
class Ray:
    origin_xyz: tuple[float, float, float]
    direction_xyz: tuple[float, float, float]


def _matrix(camera: CameraEstimate) -> np.ndarray:
    if not camera.world_from_view:
        raise ValueError("Calibrated camera requires world_from_view")
    matrix = np.asarray(camera.world_from_view, dtype=np.float64).reshape(4, 4)
    return matrix


def project_world_to_pixel(
    point_xyz: tuple[float, float, float] | np.ndarray,
    camera: CameraEstimate,
) -> tuple[float, float]:
    if not camera.calibrated or camera.principal_point_px is None:
        raise ValueError("Camera is not calibrated")
    scale = camera.world_units_per_pixel
    if scale is None or scale <= 0:
        raise ValueError("Camera scale is unavailable")

    world_from_view = _matrix(camera)
    rotation = world_from_view[:3, :3]
    translation = world_from_view[:3, 3]
    point = np.asarray(point_xyz, dtype=np.float64)
    view = rotation.T @ (point - translation)
    cx, cy = camera.principal_point_px
    return float(cx + view[0] / scale), float(cy - view[1] / scale)


def project_world_points(
    points_xyz: np.ndarray,
    camera: CameraEstimate,
) -> np.ndarray:
    if not camera.calibrated or camera.principal_point_px is None:
        raise ValueError("Camera is not calibrated")
    scale = camera.world_units_per_pixel
    if scale is None or scale <= 0:
        raise ValueError("Camera scale is unavailable")

    points = np.asarray(points_xyz, dtype=np.float64)
    world_from_view = _matrix(camera)
    rotation = world_from_view[:3, :3]
    translation = world_from_view[:3, 3]
    view = (points - translation) @ rotation
    cx, cy = camera.principal_point_px
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = cx + view[:, 0] / scale
    pixels[:, 1] = cy - view[:, 1] / scale
    return pixels


def backproject_pixel_ray(
    pixel_xy: tuple[float, float],
    camera: CameraEstimate,
) -> Ray:
    if not camera.calibrated or camera.principal_point_px is None:
        raise ValueError("Camera is not calibrated")
    scale = camera.world_units_per_pixel
    if scale is None or scale <= 0:
        raise ValueError("Camera scale is unavailable")

    world_from_view = _matrix(camera)
    rotation = world_from_view[:3, :3]
    translation = world_from_view[:3, 3]
    cx, cy = camera.principal_point_px
    u, v = pixel_xy
    point_view = np.asarray([(u - cx) * scale, -(v - cy) * scale, 0.0])
    origin = translation + rotation @ point_view
    direction = rotation[:, 2]
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    return Ray(
        origin_xyz=tuple(float(value) for value in origin),
        direction_xyz=tuple(float(value) for value in direction),
    )
