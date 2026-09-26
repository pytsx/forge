from __future__ import annotations

from uuid import uuid5

from dollforge.domain.models import DollGraph, PartClass, Provenance, ProvenanceType, Side
from dollforge.perception.models import (
    EvidenceKind,
    InterfaceHypothesis,
    PartPerception,
    RelationType,
    SpatialRelation,
)


def _mean_y(part: PartPerception) -> float:
    return sum(item.center_xy_norm[1] for item in part.observations) / len(part.observations)


def build_relations(graph: DollGraph, parts: list[PartPerception]) -> list[SpatialRelation]:
    by_id = {part.part_instance_id: part for part in parts}
    relations: list[SpatialRelation] = []

    for joint in graph.joints:
        if joint.parent_part_id not in by_id or joint.child_part_id not in by_id:
            continue
        predicate = (
            RelationType.ATTACHED_TO if str(joint.joint_type) == "fixed"
            else RelationType.CONNECTED_TO
        )
        basis = (
            EvidenceKind.PRIOR
            if joint.provenance.type == ProvenanceType.RULE
            else EvidenceKind.INFERRED
        )
        relations.append(SpatialRelation(
            subject_part_id=joint.child_part_id,
            predicate=predicate,
            object_part_id=joint.parent_part_id,
            confidence=joint.confidence,
            evidence_kind=basis,
            provenance=Provenance(
                type=joint.provenance.type,
                source="perception_relation_from_joint_v1",
                evidence=joint.provenance.evidence,
                note="Relação estrutural herdada do DollGraph; não prova a geometria do encaixe.",
            ),
        ))

    seen_pairs: set[frozenset] = set()
    for part in parts:
        partner_id = next(
            (
                candidate.part_instance_id
                for candidate in parts
                if candidate.part_class == part.part_class
                and candidate.side == (Side.RIGHT if part.side == Side.LEFT else Side.LEFT)
            ),
            None,
        ) if part.side in (Side.LEFT, Side.RIGHT) else None
        if partner_id is None:
            continue
        pair = frozenset({part.part_instance_id, partner_id})
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        relations.append(SpatialRelation(
            subject_part_id=part.part_instance_id,
            predicate=RelationType.SYMMETRIC_TO,
            object_part_id=partner_id,
            confidence=min(part.confidence, by_id[partner_id].confidence),
            evidence_kind=EvidenceKind.INFERRED,
            provenance=Provenance(
                type="derived_geometry",
                source="bilateral_part_pair_v1",
                evidence=[],
                note="Simetria inferida por classe, lado e consistência multi-view.",
            ),
        ))

    def first(part_class: PartClass, side: Side | None = None) -> PartPerception | None:
        return next(
            (
                part for part in parts
                if part.part_class == part_class and (side is None or part.side == side)
            ),
            None,
        )

    canonical_pairs = [
        (first(PartClass.HEAD), first(PartClass.TORSO)),
        (first(PartClass.TORSO), first(PartClass.PELVIS)),
    ]
    for side in (Side.LEFT, Side.RIGHT):
        leg = first(PartClass.LEG, side) or first(PartClass.SHIN, side) or first(PartClass.THIGH, side)
        footwear = first(PartClass.FOOTWEAR, side) or first(PartClass.FOOT, side)
        canonical_pairs.append((leg, footwear))

    for upper, lower in canonical_pairs:
        if upper is None or lower is None or _mean_y(upper) >= _mean_y(lower):
            continue
        evidence = sorted({
            artifact_id
            for item in (*upper.observations, *lower.observations)
            for artifact_id in item.provenance.evidence
        }, key=str)
        relations.append(SpatialRelation(
            subject_part_id=upper.part_instance_id,
            predicate=RelationType.ABOVE,
            object_part_id=lower.part_instance_id,
            confidence=min(upper.confidence, lower.confidence),
            evidence_kind=EvidenceKind.OBSERVED_MULTIVIEW,
            provenance=Provenance(
                type="derived_geometry",
                source="relative_position_v1",
                evidence=evidence,
                note="Relação vertical calculada dos centros das máscaras revisáveis.",
            ),
        ))
    return relations


def build_interfaces(graph: DollGraph, parts: list[PartPerception]) -> list[InterfaceHypothesis]:
    by_id = {part.part_instance_id: part for part in parts}
    output: list[InterfaceHypothesis] = []
    for joint in graph.joints:
        parent = by_id.get(joint.parent_part_id)
        child = by_id.get(joint.child_part_id)
        if parent is None or child is None:
            continue
        parent_views = {item.view_id for item in parent.observations}
        child_views = {item.view_id for item in child.observations}
        visible_in = sorted(parent_views & child_views, key=str)
        prior_based = joint.provenance.type == ProvenanceType.RULE
        output.append(InterfaceHypothesis(
            interface_id=uuid5(joint.joint_id, "perception_interface"),
            part_a_id=joint.parent_part_id,
            part_b_id=joint.child_part_id,
            role="contact_surface" if str(joint.joint_type) == "fixed" else "joint_candidate",
            candidate_joint_type=str(joint.joint_type),
            visible_in=visible_in,
            confidence=joint.confidence,
            evidence_kind=EvidenceKind.PRIOR if prior_based else EvidenceKind.INFERRED,
            provenance=Provenance(
                type="rule_based" if prior_based else "model_inferred",
                source="interface_hypothesis_v1",
                evidence=joint.provenance.evidence,
                note=(
                    "Hipótese de interface para orientar revisão futura. A presença do joint "
                    "não significa que a forma mecânica tenha sido observada na imagem."
                ),
            ),
        ))
    return output
