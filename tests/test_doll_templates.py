from uuid import uuid4

import numpy as np

from dollforge.adapters.templates import DollTemplateReconstructor
from dollforge.domain.models import (
    DollGraph,
    ImageQA,
    ImageView,
    PartInstance,
    PartObservation,
    Provenance,
    ScaleEstimate,
)


def provenance():
    return Provenance(type="rule_based", source="test")


def test_doll_templates_create_distinct_semantic_shapes():
    project_id = uuid4()
    view = ImageView(
        project_id=project_id,
        label="front",
        original_artifact_id=uuid4(),
        normalized_artifact_id=uuid4(),
        qa=ImageQA(width=200, height=300, sharpness=100),
        provenance=provenance(),
    )

    classes = [
        ("head", "center", (45, 10, 155, 95)),
        ("torso", "center", (60, 100, 140, 180)),
        ("footwear", "left", (35, 230, 95, 280)),
    ]
    observations = []
    parts = []
    for part_class, side, box in classes:
        observation = PartObservation(
            view_id=view.view_id,
            **{"class": part_class},
            side=side,
            confidence=.8,
            bbox_xyxy=box,
            mask_artifact_id=uuid4(),
            provenance=provenance(),
        )
        observations.append(observation)
        parts.append(PartInstance(
            part_instance_id=uuid4(),
            **{"class": part_class},
            side=side,
            observation_ids=[observation.observation_id],
            confidence=.8,
            provenance=provenance(),
        ))

    # Keep graph structurally valid with pelvis as root for this geometry-only test.
    pelvis = PartInstance(
        part_instance_id=uuid4(),
        **{"class": "pelvis"},
        side="center",
        observation_ids=[uuid4()],
        confidence=.8,
        provenance=provenance(),
    )
    graph = DollGraph.model_construct(
        project_id=project_id,
        root_part_id=pelvis.part_instance_id,
        parts=parts,
        joints=[],
        scale=ScaleEstimate(
            mode="absolute",
            canonical_height=100,
            unit="mm",
            confidence=.9,
            provenance=provenance(),
        ),
    )

    meshes = DollTemplateReconstructor().reconstruct(graph, observations, [view])
    assert len(meshes) == 3

    extents = {}
    for item in meshes:
        vertices = np.asarray(item.vertices)
        extents[item.name.split("__")[0]] = vertices.max(axis=0) - vertices.min(axis=0)

    assert extents["head"][1] > extents["head"][0] * .70
    assert extents["torso"][2] > extents["torso"][1]
    assert extents["footwear"][1] > extents["footwear"][2]
