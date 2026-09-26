from io import BytesIO
from uuid import uuid4

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from dollforge.adapters.volumetric import SilhouetteVolumeReconstructor
from dollforge.contracts import VolumetryRequest
from dollforge.domain.models import (
    DollGraph,
    ImageQA,
    ImageView,
    PartInstance,
    PartObservation,
    Provenance,
    ScaleEstimate,
)
from dollforge.perception.models import (
    EvidenceKind,
    PartGeometry,
    PartPerception,
    PerceptionGraph,
    ViewGeometry,
)
from dollforge.volumetry.silhouette import SilhouetteVisualHull


def provenance(kind="derived_geometry", source="test"):
    return Provenance(type=kind, source=source)


def png(image):
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def view(project_id, label):
    return ImageView(
        project_id=project_id,
        label=label,
        original_artifact_id=uuid4(),
        normalized_artifact_id=uuid4(),
        qa=ImageQA(width=200, height=240, sharpness=100),
        provenance=provenance("observed"),
    )


def observation(item_view, box):
    return PartObservation(
        view_id=item_view.view_id,
        part_class="head",
        side="center",
        confidence=.92,
        bbox_xyxy=box,
        mask_artifact_id=uuid4(),
        provenance=provenance("observed"),
    )


def ellipse_mask(box):
    image = Image.new("L", (200, 240), 0)
    ImageDraw.Draw(image).ellipse(box, fill=255)
    return png(image)


def test_visual_hull_uses_front_and_side_profiles_to_make_real_volume():
    project_id = uuid4()
    front = view(project_id, "front")
    left = view(project_id, "left")
    front_obs = observation(front, (40, 20, 160, 140))
    left_obs = observation(left, (60, 20, 140, 140))
    part = PartInstance(
        part_instance_id=uuid4(),
        part_class="head",
        side="center",
        observation_ids=[front_obs.observation_id, left_obs.observation_id],
        confidence=.9,
        provenance=provenance(),
    )
    graph = DollGraph(
        project_id=project_id,
        root_part_id=part.part_instance_id,
        parts=[part],
        joints=[],
        scale=ScaleEstimate(
            mode="absolute",
            canonical_height=120,
            unit="mm",
            confidence=.9,
            provenance=provenance(),
        ),
    )
    dummy_profile = [1.0] * 16
    perceived = PartPerception(
        part_instance_id=part.part_instance_id,
        part_class="head",
        side="center",
        region="head",
        observations=[
            ViewGeometry(
                observation_id=front_obs.observation_id,
                view_id=front.view_id,
                view_label="front",
                center_xy_norm=(.5, .33),
                width_norm=.6,
                height_norm=.5,
                area_ratio=.2,
                fill_ratio=.78,
                boundary_strength=.8,
                width_profile=dummy_profile,
                provenance=provenance("observed"),
            ),
            ViewGeometry(
                observation_id=left_obs.observation_id,
                view_id=left.view_id,
                view_label="left",
                center_xy_norm=(.5, .33),
                width_norm=.4,
                height_norm=.5,
                area_ratio=.14,
                fill_ratio=.78,
                boundary_strength=.8,
                width_profile=dummy_profile,
                provenance=provenance("observed"),
            ),
        ],
        geometry=PartGeometry(
            height_norm=.5,
            frontal_width_norm=.6,
            depth_norm=.4,
            depth_to_width=2 / 3,
            shape_family="rounded_spheroid",
            completeness=.5,
            evidence_kind=EvidenceKind.OBSERVED_MULTIVIEW,
            provenance=provenance(),
        ),
        confidence=.9,
        provenance=provenance(),
    )
    perception = PerceptionGraph(
        project_id=project_id,
        symmetry="unknown",
        major_regions=["head"],
        parts=[perceived],
        relations=[],
        interfaces=[],
        provenance=provenance(),
    )

    result = SilhouetteVisualHull().build(VolumetryRequest(
        project_id=project_id,
        graph=graph,
        perception=perception,
        observations=[front_obs, left_obs],
        views=[front, left],
        mask_png_by_observation={
            front_obs.observation_id: ellipse_mask(front_obs.bbox_xyxy),
            left_obs.observation_id: ellipse_mask(left_obs.bbox_xyxy),
        },
        resolution=32,
    ))

    assert len(result.volumes) == 1
    volume = result.volumes[0]
    assert volume.concavity_support is False
    assert volume.extents_xyz[0] > volume.extents_xyz[1]
    assert len(volume.slices) == 32
    assert volume.slices[len(volume.slices) // 2].half_width_norm > volume.slices[0].half_width_norm
    mesh = trimesh.Trimesh(np.asarray(volume.vertices), np.asarray(volume.faces), process=True)
    assert mesh.is_watertight
    assert mesh.volume > 0

    reconstructed = SilhouetteVolumeReconstructor().reconstruct(
        graph, [front_obs, left_obs], [front, left], result
    )
    assert len(reconstructed) == 1
    assert reconstructed[0].provenance.source == "silhouette_volume_mesh_v1"


def test_pipeline_defaults_to_volumetry_and_volume_mesh():
    from dollforge.domain.models import PipelineConfig, Stage

    config = PipelineConfig()
    assert config.volumetry_adapter == "calibrated_visual_hull_sdf_v2"
    assert config.reconstruction_adapter == "silhouette_volume_mesh_v1"
    assert config.volumetry_resolution == 64
    assert Stage.VOLUMETRY == "S09V"
