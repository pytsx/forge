from io import BytesIO
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from dollforge.contracts import VolumetryRequest
from dollforge.domain.models import (
    DollGraph,
    ImageQA,
    ImageView,
    PartInstance,
    PartObservation,
    Provenance,
    ReviewState,
    ScaleEstimate,
)
from dollforge.perception.models import (
    EvidenceKind,
    PartGeometry,
    PartPerception,
    PerceptionGraph,
    ViewGeometry,
)
from dollforge.volumetry.calibration import calibrate_views
from dollforge.volumetry.visual_hull import CalibratedVisualHullSDF


def provenance(kind="derived_geometry"):
    return Provenance(type=kind, source="test")


def png_mask(box, size=(200, 200)):
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).rectangle(box, fill=255)
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def make_view(project_id, label):
    return ImageView(
        project_id=project_id,
        label=label,
        original_artifact_id=uuid4(),
        normalized_artifact_id=uuid4(),
        qa=ImageQA(width=200, height=200, sharpness=100),
        provenance=provenance("observed"),
    )


def build_case(front_width_px=80, evidence_kind=EvidenceKind.OBSERVED_MULTIVIEW):
    project_id = uuid4()
    views = [
        make_view(project_id, "front"),
        make_view(project_id, "back"),
        make_view(project_id, "left"),
        make_view(project_id, "right"),
    ]
    boxes = {
        "front": (100 - front_width_px // 2, 30, 100 + front_width_px // 2, 150),
        "back": (60, 30, 140, 150),
        "left": (70, 30, 130, 150),
        "right": (70, 30, 130, 150),
    }
    observations = []
    masks = {}
    for view in views:
        observation = PartObservation(
            view_id=view.view_id,
            part_class="torso",
            side="center",
            confidence=.98,
            bbox_xyxy=boxes[str(view.label)],
            mask_artifact_id=uuid4(),
            provenance=provenance("human_approved"),
            review_state=ReviewState.APPROVED,
        )
        observations.append(observation)
        masks[observation.observation_id] = png_mask(boxes[str(view.label)])

    part = PartInstance(
        part_instance_id=uuid4(),
        part_class="torso",
        side="center",
        observation_ids=[observation.observation_id for observation in observations],
        confidence=.98,
        provenance=provenance(),
    )
    scale = ScaleEstimate(
        mode="absolute",
        canonical_height=60,
        unit="mm",
        confidence=.98,
        provenance=provenance("human_edited"),
    )
    graph = DollGraph(
        project_id=project_id,
        root_part_id=part.part_instance_id,
        parts=[part],
        joints=[],
        scale=scale,
    )
    cameras = calibrate_views(scale, observations, views)

    view_geometries = []
    for observation, view in zip(observations, views):
        width = observation.bbox_xyxy[2] - observation.bbox_xyxy[0]
        view_geometries.append(ViewGeometry(
            observation_id=observation.observation_id,
            view_id=view.view_id,
            view_label=view.label,
            center_xy_norm=(.5, .5),
            width_norm=width / 120,
            height_norm=1.0,
            area_ratio=.25,
            fill_ratio=1.0,
            boundary_strength=1.0,
            width_profile=[1.0] * 16,
            provenance=provenance("observed"),
        ))
    perceived = PartPerception(
        part_instance_id=part.part_instance_id,
        part_class="torso",
        side="center",
        region="upper_body",
        observations=view_geometries,
        geometry=PartGeometry(
            height_norm=1.0,
            frontal_width_norm=40 / 60,
            depth_norm=30 / 60,
            depth_to_width=.75,
            shape_family="rounded_tapered_body",
            completeness=1.0,
            evidence_kind=evidence_kind,
            provenance=provenance(),
        ),
        confidence=.98,
        provenance=provenance(),
    )
    perception = PerceptionGraph(
        project_id=project_id,
        symmetry="unknown",
        major_regions=["upper_body"],
        parts=[perceived],
        relations=[],
        interfaces=[],
        provenance=provenance(),
    )
    request = VolumetryRequest(
        project_id=project_id,
        graph=graph,
        perception=perception,
        observations=observations,
        views=views,
        cameras=cameras,
        mask_png_by_observation=masks,
        resolution=64,
    )
    return request


def field_occupancy(payload):
    with np.load(BytesIO(payload)) as data:
        return data["occupancy"].astype(bool)


def test_calibrated_visual_hull_recovers_synthetic_cuboid_dimensions():
    result, fields = CalibratedVisualHullSDF().build_with_fields(build_case())

    assert len(result.volumes) == 1
    volume = result.volumes[0]
    assert len(fields) == 1
    voxel = volume.field.voxel_size_mm
    assert voxel is not None

    expected = np.asarray([40.0, 30.0, 60.0])
    actual = np.asarray(volume.extents_xyz)
    assert np.all(np.abs(actual - expected) <= voxel * 1.5)
    assert volume.mean_reprojection_iou >= .95
    assert all(metric.silhouette_iou >= .93 for metric in volume.reprojection_metrics)
    assert volume.field.grid_shape == (64, 64, 64)
    assert volume.concavity_support is False


def test_one_confirmed_view_can_carve_geometry_seen_by_other_views():
    full_result, full_fields = CalibratedVisualHullSDF().build_with_fields(build_case(80))
    narrow_result, narrow_fields = CalibratedVisualHullSDF().build_with_fields(build_case(60))

    full = field_occupancy(full_fields[0].payload)
    narrow = field_occupancy(narrow_fields[0].payload)

    assert narrow.sum() < full.sum()
    assert narrow_result.volumes[0].extents_xyz[0] < full_result.volumes[0].extents_xyz[0]


def test_prior_label_never_changes_visual_carving():
    observed_result, observed_fields = CalibratedVisualHullSDF().build_with_fields(
        build_case(evidence_kind=EvidenceKind.OBSERVED_MULTIVIEW)
    )
    prior_result, prior_fields = CalibratedVisualHullSDF().build_with_fields(
        build_case(evidence_kind=EvidenceKind.PRIOR)
    )

    observed = field_occupancy(observed_fields[0].payload)
    prior = field_occupancy(prior_fields[0].payload)

    assert np.array_equal(observed, prior)
    assert np.allclose(
        observed_result.volumes[0].extents_xyz,
        prior_result.volumes[0].extents_xyz,
    )
