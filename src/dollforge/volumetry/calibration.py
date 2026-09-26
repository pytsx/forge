from __future__ import annotations

from statistics import median
from uuid import UUID

import numpy as np

from dollforge.domain.models import (
    CameraEstimate,
    ImageView,
    PartObservation,
    Provenance,
    ReviewState,
    ScaleEstimate,
    ViewLabel,
)


def object_bounds_by_view(
    observations: list[PartObservation],
) -> dict[UUID, tuple[float, float, float, float]]:
    grouped: dict[UUID, list[tuple[float, float, float, float]]] = {}
    for observation in observations:
        if observation.review_state == ReviewState.REJECTED:
            continue
        grouped.setdefault(observation.view_id, []).append(observation.bbox_xyxy)
    return {
        view_id: (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        for view_id, boxes in grouped.items()
        if boxes
    }


def _axes(label: ViewLabel) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if label == ViewLabel.FRONT:
        return (
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([0.0, -1.0, 0.0]),
            0.0,
        )
    if label == ViewLabel.BACK:
        return (
            np.asarray([-1.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([0.0, 1.0, 0.0]),
            180.0,
        )
    if label == ViewLabel.LEFT:
        return (
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([1.0, 0.0, 0.0]),
            -90.0,
        )
    if label == ViewLabel.RIGHT:
        return (
            np.asarray([0.0, -1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([-1.0, 0.0, 0.0]),
            90.0,
        )
    raise ValueError(f"Unsupported calibrated view: {label}")


def calibrate_views(
    scale: ScaleEstimate,
    observations: list[PartObservation],
    views: list[ImageView],
) -> list[CameraEstimate]:
    bounds = object_bounds_by_view(observations)
    heights = [
        box[3] - box[1]
        for view_id, box in bounds.items()
        if view_id in {view.view_id for view in views}
    ]
    if not heights:
        return []

    median_height = float(median(heights))
    output: list[CameraEstimate] = []
    for view in views:
        if view.label not in (
            ViewLabel.FRONT,
            ViewLabel.BACK,
            ViewLabel.LEFT,
            ViewLabel.RIGHT,
        ):
            continue
        box = bounds.get(view.view_id)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        pixel_height = max(1.0, y1 - y0)
        units_per_pixel = scale.canonical_height / pixel_height
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        axis_x, axis_y, axis_z, yaw = _axes(view.label)
        rotation = np.stack([axis_x, axis_y, axis_z], axis=1)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        # The canonical object center is the view's principal point.
        matrix[:3, 3] = [0.0, 0.0, 0.0]

        height_agreement = min(pixel_height, median_height) / max(pixel_height, median_height)
        confidence = min(
            .98,
            .45 + .35 * scale.confidence + .20 * height_agreement,
        )
        output.append(CameraEstimate(
            view_id=view.view_id,
            label=view.label,
            projection="orthographic_calibrated",
            yaw_deg=yaw,
            calibrated=True,
            world_units_per_pixel=float(units_per_pixel),
            mm_per_pixel=float(units_per_pixel) if scale.unit == "mm" else None,
            principal_point_px=(float(cx), float(cy)),
            world_from_view=matrix.flatten().tolist(),
            confidence=float(confidence),
            provenance=Provenance(
                type="derived_geometry",
                source="orthographic_height_calibration_v1",
                evidence=[view.normalized_artifact_id],
                note=(
                    "Escala independente por vista: altura canônica dividida pela altura observada "
                    "em pixels; origem canônica no centro da silhueta."
                ),
            ),
        ))
    return output
