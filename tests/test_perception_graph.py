from io import BytesIO
from uuid import uuid4

from PIL import Image, ImageDraw

from dollforge.contracts import PerceptionRequest
from dollforge.domain.models import (
    DollGraph,
    ImageQA,
    ImageView,
    JointSpec,
    PartInstance,
    PartObservation,
    Provenance,
    ReviewState,
    ScaleEstimate,
)
from dollforge.perception.graph import StructuredPerceptionBuilder
from dollforge.perception.models import EvidenceKind, RelationType


def png(image):
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def provenance(kind="observed", source="test"):
    return Provenance(type=kind, source=source)


def make_view(project_id, label):
    return ImageView(
        project_id=project_id,
        label=label,
        original_artifact_id=uuid4(),
        normalized_artifact_id=uuid4(),
        qa=ImageQA(width=200, height=300, sharpness=100),
        provenance=provenance(),
    )


def make_observation(view, part_class, box, state=ReviewState.NEEDS_REVIEW):
    return PartObservation(
        view_id=view.view_id,
        part_class=part_class,
        side="center",
        confidence=.9,
        bbox_xyxy=box,
        mask_artifact_id=uuid4(),
        provenance=provenance("human_edited" if state == ReviewState.CORRECTED else "observed"),
        review_state=state,
    )


def mask_for(box):
    mask = Image.new("L", (200, 300), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(box, fill=255)
    return png(mask)


def scene_image():
    image = Image.new("RGB", (200, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((50, 15, 150, 105), fill=(70, 70, 70))
    draw.rounded_rectangle((65, 105, 135, 205), radius=20, fill=(95, 95, 95))
    draw.ellipse((72, 195, 128, 245), fill=(115, 115, 115))
    return png(image)


def test_perception_graph_separates_observed_geometry_from_prior_interfaces():
    project_id = uuid4()
    front = make_view(project_id, "front")
    left = make_view(project_id, "left")

    head_front = make_observation(front, "head", (50, 15, 150, 105), ReviewState.CORRECTED)
    head_left = make_observation(left, "head", (62, 15, 138, 105), ReviewState.CORRECTED)
    torso_front = make_observation(front, "torso", (65, 105, 135, 205))
    torso_left = make_observation(left, "torso", (72, 105, 128, 205))
    pelvis_front = make_observation(front, "pelvis", (72, 195, 128, 245))
    pelvis_left = make_observation(left, "pelvis", (78, 195, 122, 245))
    observations = [
        head_front,
        head_left,
        torso_front,
        torso_left,
        pelvis_front,
        pelvis_left,
    ]

    head = PartInstance(
        part_instance_id=uuid4(),
        part_class="head",
        side="center",
        observation_ids=[head_front.observation_id, head_left.observation_id],
        confidence=.9,
        provenance=provenance("derived_geometry"),
    )
    torso = PartInstance(
        part_instance_id=uuid4(),
        part_class="torso",
        side="center",
        observation_ids=[torso_front.observation_id, torso_left.observation_id],
        confidence=.9,
        provenance=provenance("derived_geometry"),
    )
    pelvis = PartInstance(
        part_instance_id=uuid4(),
        part_class="pelvis",
        side="center",
        observation_ids=[pelvis_front.observation_id, pelvis_left.observation_id],
        confidence=.9,
        provenance=provenance("derived_geometry"),
    )
    graph = DollGraph(
        project_id=project_id,
        root_part_id=pelvis.part_instance_id,
        parts=[head, torso, pelvis],
        joints=[
            JointSpec(
                joint_id=uuid4(),
                parent_part_id=pelvis.part_instance_id,
                child_part_id=torso.part_instance_id,
                joint_type="fixed",
                confidence=.25,
                provenance=provenance("rule_based", "anatomy_v1"),
            ),
            JointSpec(
                joint_id=uuid4(),
                parent_part_id=torso.part_instance_id,
                child_part_id=head.part_instance_id,
                joint_type="ball_socket",
                confidence=.25,
                provenance=provenance("rule_based", "anatomy_v1"),
            ),
        ],
        scale=ScaleEstimate(
            mode="relative",
            canonical_height=1,
            unit="relative",
            confidence=.25,
            provenance=provenance("rule_based"),
        ),
    )

    image = scene_image()
    masks = {
        head_front.observation_id: mask_for(head_front.bbox_xyxy),
        head_left.observation_id: mask_for(head_left.bbox_xyxy),
        torso_front.observation_id: mask_for(torso_front.bbox_xyxy),
        torso_left.observation_id: mask_for(torso_left.bbox_xyxy),
        pelvis_front.observation_id: mask_for(pelvis_front.bbox_xyxy),
        pelvis_left.observation_id: mask_for(pelvis_left.bbox_xyxy),
    }

    result = StructuredPerceptionBuilder().describe(PerceptionRequest(
        project_id=project_id,
        graph_artifact_id=uuid4(),
        graph=graph,
        observations=observations,
        views=[front, left],
        image_png_by_view={front.view_id: image, left.view_id: image},
        mask_png_by_observation=masks,
        style_family="cute_collectible",
    ))

    assert result.object_type == "stylized_modular_doll"
    assert result.style_family == "cute_collectible"
    assert result.major_regions == ["head", "upper_body", "lower_body"]

    perceived_head = next(part for part in result.parts if part.part_class == "head")
    assert perceived_head.geometry.frontal_width_norm is not None
    assert perceived_head.geometry.depth_norm is not None
    assert perceived_head.geometry.depth_to_width is not None
    assert perceived_head.geometry.evidence_kind == EvidenceKind.HUMAN_CONFIRMED
    assert perceived_head.parent_part_id == torso.part_instance_id

    assert any(
        relation.subject_part_id == head.part_instance_id
        and relation.predicate == RelationType.ABOVE
        and relation.object_part_id == torso.part_instance_id
        and relation.evidence_kind == EvidenceKind.OBSERVED_MULTIVIEW
        for relation in result.relations
    )
    neck = next(
        interface for interface in result.interfaces
        if interface.part_b_id == head.part_instance_id
    )
    assert neck.candidate_joint_type == "ball_socket"
    assert neck.evidence_kind == EvidenceKind.PRIOR


def test_pipeline_config_enables_structured_perception_by_default():
    from dollforge.domain.models import PipelineConfig, Stage

    config = PipelineConfig()
    assert config.perception_adapter == "structured_perception_v1"
    assert Stage.PERCEPTION == "S09"
