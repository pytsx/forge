from uuid import uuid4

import numpy as np

from dollforge.domain.models import (
    ImageQA,
    ImageView,
    PartObservation,
    Provenance,
    ReviewState,
    ScaleEstimate,
)
from dollforge.volumetry.calibration import calibrate_views
from dollforge.volumetry.projection import backproject_pixel_ray, project_world_to_pixel


def provenance(kind="observed"):
    return Provenance(type=kind, source="test")


def make_view(project_id, label, width, height):
    return ImageView(
        project_id=project_id,
        label=label,
        original_artifact_id=uuid4(),
        normalized_artifact_id=uuid4(),
        qa=ImageQA(width=width, height=height, sharpness=100),
        provenance=provenance(),
    )


def make_observation(view, box):
    return PartObservation(
        view_id=view.view_id,
        part_class="torso",
        side="center",
        confidence=.95,
        bbox_xyxy=box,
        mask_artifact_id=uuid4(),
        provenance=provenance("human_approved"),
        review_state=ReviewState.APPROVED,
    )


def test_per_view_scale_is_invariant_to_image_resolution():
    project_id = uuid4()
    low = make_view(project_id, "front", 800, 800)
    high = make_view(project_id, "back", 1600, 1600)
    observations = [
        make_observation(low, (200, 0, 600, 800)),
        make_observation(high, (400, 0, 1200, 1600)),
    ]
    scale = ScaleEstimate(
        mode="absolute",
        canonical_height=100,
        unit="mm",
        confidence=.95,
        provenance=provenance("human_edited"),
    )

    cameras = calibrate_views(scale, observations, [low, high])
    by_label = {str(camera.label): camera for camera in cameras}

    assert np.isclose(by_label["front"].mm_per_pixel, .125)
    assert np.isclose(by_label["back"].mm_per_pixel, .0625)

    low_width_mm = 400 * by_label["front"].mm_per_pixel
    high_width_mm = 800 * by_label["back"].mm_per_pixel
    assert abs(low_width_mm - high_width_mm) / low_width_mm < .01


def test_opposite_views_share_canonical_axes_with_mirror_convention():
    project_id = uuid4()
    views = [
        make_view(project_id, "front", 400, 400),
        make_view(project_id, "back", 400, 400),
        make_view(project_id, "left", 400, 400),
        make_view(project_id, "right", 400, 400),
    ]
    observations = [make_observation(view, (100, 50, 300, 350)) for view in views]
    scale = ScaleEstimate(
        mode="absolute",
        canonical_height=120,
        unit="mm",
        confidence=.95,
        provenance=provenance("human_edited"),
    )
    cameras = calibrate_views(scale, observations, views)
    by_label = {str(camera.label): camera for camera in cameras}

    front = project_world_to_pixel((10.0, 0.0, 5.0), by_label["front"])
    back = project_world_to_pixel((10.0, 0.0, 5.0), by_label["back"])
    left = project_world_to_pixel((0.0, 10.0, 5.0), by_label["left"])
    right = project_world_to_pixel((0.0, 10.0, 5.0), by_label["right"])

    cx = by_label["front"].principal_point_px[0]
    assert front[0] > cx
    assert back[0] < cx
    assert left[0] > by_label["left"].principal_point_px[0]
    assert right[0] < by_label["right"].principal_point_px[0]

    ray = backproject_pixel_ray(front, by_label["front"])
    assert np.isclose(ray.origin_xyz[0], 10.0, atol=1e-6)
    assert np.isclose(ray.origin_xyz[2], 5.0, atol=1e-6)
