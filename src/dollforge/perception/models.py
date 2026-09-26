from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from dollforge.domain.models import DTO, PartClass, Provenance, Score, Side, ViewLabel


class EvidenceKind(StrEnum):
    OBSERVED = "observed"
    OBSERVED_MULTIVIEW = "observed_multiview"
    INFERRED = "inferred"
    PRIOR = "prior"
    HUMAN_CONFIRMED = "human_confirmed"


class RelationType(StrEnum):
    ABOVE = "above"
    BELOW = "below"
    CONNECTED_TO = "connected_to"
    ATTACHED_TO = "attached_to"
    SYMMETRIC_TO = "symmetric_to"


class ViewGeometry(DTO):
    observation_id: UUID
    view_id: UUID
    view_label: ViewLabel
    center_xy_norm: tuple[float, float]
    width_norm: float = Field(ge=0)
    height_norm: float = Field(ge=0)
    area_ratio: float = Field(ge=0, le=1)
    fill_ratio: float = Field(ge=0, le=1)
    boundary_strength: Score
    width_profile: list[Score] = Field(min_length=8, max_length=64)
    evidence_kind: Literal[EvidenceKind.OBSERVED] = EvidenceKind.OBSERVED
    provenance: Provenance


class PartGeometry(DTO):
    height_norm: float = Field(ge=0)
    frontal_width_norm: float | None = Field(default=None, ge=0)
    depth_norm: float | None = Field(default=None, ge=0)
    depth_to_width: float | None = Field(default=None, ge=0)
    shape_family: str
    completeness: Score
    evidence_kind: EvidenceKind
    provenance: Provenance


class PartPerception(DTO):
    part_instance_id: UUID
    part_class: PartClass = Field(alias="class")
    side: Side
    region: Literal["head", "upper_body", "lower_body", "surface_feature", "accessory"]
    parent_part_id: UUID | None = None
    observations: list[ViewGeometry] = Field(min_length=1)
    geometry: PartGeometry
    confidence: Score
    provenance: Provenance


class SpatialRelation(DTO):
    subject_part_id: UUID
    predicate: RelationType
    object_part_id: UUID
    confidence: Score
    evidence_kind: EvidenceKind
    provenance: Provenance


class InterfaceHypothesis(DTO):
    interface_id: UUID
    part_a_id: UUID
    part_b_id: UUID
    role: Literal["joint_candidate", "contact_surface", "assembly_interface"]
    candidate_joint_type: str | None = None
    visible_in: list[UUID] = Field(default_factory=list)
    confidence: Score
    evidence_kind: EvidenceKind
    provenance: Provenance


class PerceptionGraph(DTO):
    schema_version: Literal["1.0.0"] = "1.0.0"
    project_id: UUID
    object_type: Literal["stylized_modular_doll"] = "stylized_modular_doll"
    construction: Literal["multi_part"] = "multi_part"
    style_family: str | None = None
    symmetry: Literal["bilateral", "approximate_bilateral", "unknown"]
    major_regions: list[str]
    parts: list[PartPerception]
    relations: list[SpatialRelation]
    interfaces: list[InterfaceHypothesis]
    warnings: list[str] = Field(default_factory=list)
    provenance: Provenance
