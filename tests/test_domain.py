from uuid import uuid4

import pytest
from pydantic import ValidationError

from dollforge.domain.models import (
    DollGraph,
    JointSpec,
    PartInstance,
    Provenance,
    ScaleEstimate,
)


def provenance():
    return Provenance(type="rule_based", source="test")


def part(part_class, side="center"):
    return PartInstance(
        part_instance_id=uuid4(),
        **{"class": part_class},
        side=side,
        observation_ids=[uuid4()],
        confidence=0.9,
        provenance=provenance(),
    )


def test_doll_graph_accepts_connected_tree():
    pelvis = part("pelvis")
    torso = part("torso")
    head = part("head")
    graph = DollGraph(
        project_id=uuid4(),
        root_part_id=pelvis.part_instance_id,
        parts=[pelvis, torso, head],
        joints=[
            JointSpec(
                joint_id=uuid4(),
                parent_part_id=pelvis.part_instance_id,
                child_part_id=torso.part_instance_id,
                joint_type="fixed",
                confidence=0.8,
                provenance=provenance(),
            ),
            JointSpec(
                joint_id=uuid4(),
                parent_part_id=torso.part_instance_id,
                child_part_id=head.part_instance_id,
                joint_type="ball_socket",
                confidence=0.8,
                provenance=provenance(),
            ),
        ],
        scale=ScaleEstimate(
            mode="relative",
            canonical_height=1.0,
            unit="relative",
            confidence=0.5,
            provenance=provenance(),
        ),
    )
    assert len(graph.parts) == 3


def test_doll_graph_rejects_cycle():
    pelvis = part("pelvis")
    torso = part("torso")
    with pytest.raises(ValidationError):
        DollGraph(
            project_id=uuid4(),
            root_part_id=pelvis.part_instance_id,
            parts=[pelvis, torso],
            joints=[
                JointSpec(
                    joint_id=uuid4(),
                    parent_part_id=pelvis.part_instance_id,
                    child_part_id=torso.part_instance_id,
                    joint_type="fixed",
                    confidence=0.8,
                    provenance=provenance(),
                ),
                JointSpec(
                    joint_id=uuid4(),
                    parent_part_id=torso.part_instance_id,
                    child_part_id=pelvis.part_instance_id,
                    joint_type="fixed",
                    confidence=0.8,
                    provenance=provenance(),
                ),
            ],
            scale=ScaleEstimate(
                mode="relative",
                canonical_height=1.0,
                unit="relative",
                confidence=0.5,
                provenance=provenance(),
            ),
        )


def test_absolute_scale_requires_millimeters():
    with pytest.raises(ValidationError):
        ScaleEstimate(
            mode="absolute",
            canonical_height=150,
            unit="relative",
            confidence=0.9,
            provenance=provenance(),
        )
