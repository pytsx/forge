from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from dollforge.contracts import ReprojectionMetric
from dollforge.domain.models import CameraEstimate
from dollforge.volumetry.projection import project_world_points


def silhouette_iou(generated: np.ndarray, reference: np.ndarray) -> float:
    generated = np.asarray(generated, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    union = np.logical_or(generated, reference).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(generated, reference).sum() / union)


def boundary_rmse(generated: np.ndarray, reference: np.ndarray) -> float | None:
    generated = np.asarray(generated, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    gen_boundary = generated ^ binary_erosion(generated)
    ref_boundary = reference ^ binary_erosion(reference)
    if not gen_boundary.any() or not ref_boundary.any():
        return None

    distance_to_ref = distance_transform_edt(~ref_boundary)
    distance_to_gen = distance_transform_edt(~gen_boundary)
    errors = np.concatenate([
        distance_to_ref[gen_boundary],
        distance_to_gen[ref_boundary],
    ])
    return float(np.sqrt(np.mean(errors * errors))) if len(errors) else None


def reproject_occupancy(
    occupied_points_xyz: np.ndarray,
    camera: CameraEstimate,
    image_shape: tuple[int, int],
    voxel_size: float,
) -> np.ndarray:
    height, width = image_shape
    generated = np.zeros((height, width), dtype=bool)
    if len(occupied_points_xyz) == 0:
        return generated

    pixels = project_world_points(occupied_points_xyz, camera)
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    valid = (u >= 0) & (v >= 0) & (u < width) & (v < height)
    generated[v[valid], u[valid]] = True

    scale = camera.world_units_per_pixel or voxel_size
    radius = max(0, int(np.ceil(voxel_size / max(scale, 1e-12) * .35)))
    if radius:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        generated = binary_dilation(generated, structure=structure)
    return generated


def evaluate_reprojection(
    occupied_points_xyz: np.ndarray,
    camera: CameraEstimate,
    reference_mask: np.ndarray,
    voxel_size: float,
) -> ReprojectionMetric:
    reference = np.asarray(reference_mask, dtype=bool)
    generated = reproject_occupancy(
        occupied_points_xyz,
        camera,
        reference.shape,
        voxel_size,
    )
    iou = silhouette_iou(generated, reference)
    reference_area = max(1, int(reference.sum()))
    area_error = abs(int(generated.sum()) - int(reference.sum())) / reference_area
    return ReprojectionMetric(
        view_id=camera.view_id,
        view_label=str(camera.label),
        silhouette_iou=float(np.clip(iou, 0.0, 1.0)),
        area_error_ratio=float(area_error),
        boundary_rmse_px=boundary_rmse(generated, reference),
    )
