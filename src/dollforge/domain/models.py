from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Score = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[float, Field(gt=0)]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DTO(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)


class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    BILATERAL = "bilateral"
    UNKNOWN = "unknown"


class PartClass(StrEnum):
    HEAD = "head"
    TORSO = "torso"
    PELVIS = "pelvis"
    ARM = "arm"
    LEG = "leg"
    FOOTWEAR = "footwear"
    HAND = "hand"
    FOOT = "foot"
    UPPER_ARM = "upper_arm"
    FOREARM = "forearm"
    THIGH = "thigh"
    SHIN = "shin"
    HAIR = "hair"
    FACE = "face"
    TOP = "top"
    BOTTOM = "bottom"
    ACCESSORY = "accessory"


class ViewLabel(StrEnum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    THREE_QUARTER_FRONT = "three_quarter_front"
    THREE_QUARTER_BACK = "three_quarter_back"
    DETAIL = "detail"
    UNKNOWN = "unknown"


class ReviewState(StrEnum):
    UNREVIEWED = "unreviewed"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CORRECTED = "corrected"


class ProvenanceType(StrEnum):
    OBSERVED = "observed"
    GEOMETRY = "derived_geometry"
    REFERENCE = "retrieved_reference"
    RULE = "rule_based"
    MODEL = "model_inferred"
    EDITED = "human_edited"
    APPROVED = "human_approved"


class Provenance(DTO):
    type: ProvenanceType
    source: str = Field(min_length=1)
    evidence: list[UUID] = Field(default_factory=list)
    note: str = ""


class Stage(StrEnum):
    INTAKE = "S00"
    QA = "S01"
    CAMERA = "S03"
    SEGMENTATION = "S05"
    MATCHING = "S06"
    SCALE = "S07"
    GRAPH = "S08"
    RECONSTRUCTION = "S10"
    BLENDER = "S15"
    VALIDATION = "S16"
    REVIEW = "S17"
    KNOWLEDGE = "S18"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REVIEW = "waiting_for_review"


class CreateProject(DTO):
    name: str = Field(min_length=1, max_length=120)
    style_family: str | None = None
    known_height_mm: Positive | None = None
    manufacturing_profile_id: str | None = None


class DollProject(CreateProject):
    schema_version: Literal["1.0.0"] = "1.0.0"
    project_id: UUID = Field(default_factory=uuid4)
    unit: Literal["mm"] = "mm"
    coordinate_system: Literal["RH_Z_UP"] = "RH_Z_UP"
    views: list[UUID] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ImageQA(DTO):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sharpness: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class ImageView(DTO):
    view_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    label: ViewLabel
    original_artifact_id: UUID
    normalized_artifact_id: UUID
    qa: ImageQA
    provenance: Provenance


class CameraEstimate(DTO):
    view_id: UUID
    label: ViewLabel
    projection: Literal["orthographic_assumption"] = "orthographic_assumption"
    yaw_deg: float
    calibrated: Literal[False] = False
    confidence: Score = 0.3
    provenance: Provenance


class PartObservation(DTO):
    observation_id: UUID = Field(default_factory=uuid4)
    view_id: UUID
    part_class: PartClass = Field(alias="class")
    side: Side
    confidence: Score
    bbox_xyxy: tuple[float, float, float, float]
    mask_artifact_id: UUID
    provenance: Provenance
    review_state: ReviewState = ReviewState.NEEDS_REVIEW
    alternatives: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_box(self):
        x0, y0, x1, y1 = self.bbox_xyxy
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("bbox must have positive area and nonnegative coordinates")
        return self


class PartInstance(DTO):
    part_instance_id: UUID
    part_class: PartClass = Field(alias="class")
    side: Side
    observation_ids: list[UUID] = Field(min_length=1)
    confidence: Score
    provenance: Provenance
    symmetry_partner_id: UUID | None = None
    review_state: ReviewState = ReviewState.NEEDS_REVIEW
    canonical_transform: list[float] = Field(
        default_factory=lambda: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        min_length=16, max_length=16,
    )


class JointType(StrEnum):
    FIXED = "fixed"
    PEG_SOCKET = "peg_socket"
    BALL_SOCKET = "ball_socket"
    HINGE = "hinge"
    SWIVEL = "swivel"
    DOUBLE_JOINT = "double_joint"
    SNAP_FIT = "snap_fit"
    FRICTION_FIT = "friction_fit"
    MAGNETIC = "magnetic"
    CUSTOM = "custom"


class JointSpec(DTO):
    joint_id: UUID
    parent_part_id: UUID
    child_part_id: UUID
    joint_type: JointType
    axis_xyz: tuple[float, float, float] | None = None
    range_deg: tuple[float, float] | None = None
    confidence: Score
    provenance: Provenance
    hidden: bool = True
    review_state: ReviewState = ReviewState.NEEDS_REVIEW
    alternatives: list[JointType] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_joint(self):
        if self.parent_part_id == self.child_part_id:
            raise ValueError("A joint must connect different parts")
        if self.range_deg and self.range_deg[0] > self.range_deg[1]:
            raise ValueError("Joint range is inverted")
        if self.axis_xyz and sum(v * v for v in self.axis_xyz) < 1e-10:
            raise ValueError("Joint axis cannot be zero")
        return self


class ScaleEstimate(DTO):
    mode: Literal["relative", "absolute"]
    canonical_height: Positive
    unit: Literal["relative", "mm"]
    confidence: Score
    provenance: Provenance

    @model_validator(mode="after")
    def unit_matches(self):
        if (self.mode == "absolute") != (self.unit == "mm"):
            raise ValueError("Absolute scale requires mm; relative scale must stay relative")
        return self


class DollGraph(DTO):
    schema_version: Literal["1.1.0"] = "1.1.0"
    project_id: UUID
    root_part_id: UUID
    parts: list[PartInstance]
    joints: list[JointSpec]
    scale: ScaleEstimate

    @model_validator(mode="after")
    def valid_graph(self):
        ids = {p.part_instance_id for p in self.parts}
        if len(ids) != len(self.parts) or self.root_part_id not in ids:
            raise ValueError("Graph requires unique parts and an existing root")
        observations = [o for p in self.parts for o in p.observation_ids]
        if len(observations) != len(set(observations)):
            raise ValueError("An observation cannot belong to two instances")
        parents: dict[UUID, UUID] = {}
        for j in self.joints:
            if j.parent_part_id not in ids or j.child_part_id not in ids:
                raise ValueError("Dangling joint")
            if j.child_part_id in parents:
                raise ValueError("Multiple parents")
            parents[j.child_part_id] = j.parent_part_id
        if self.root_part_id in parents:
            raise ValueError("Root cannot have a parent")
        for item in ids:
            seen: set[UUID] = set()
            while item in parents:
                if item in seen:
                    raise ValueError("Graph cycle")
                seen.add(item)
                item = parents[item]
            if item != self.root_part_id:
                raise ValueError("Graph must connect to root")
        for p in self.parts:
            if p.symmetry_partner_id and p.symmetry_partner_id not in ids:
                raise ValueError("Dangling symmetry partner")
        return self


class Lineage(DTO):
    input_artifact_ids: list[UUID] = Field(default_factory=list)
    code_version: str
    model_version: str
    parameters_hash: str
    seed: int
    environment_fingerprint: str


class Artifact(DTO):
    schema_version: Literal["1.1.0"] = "1.1.0"
    artifact_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    run_id: UUID
    stage: Stage
    version: int = Field(ge=1)
    kind: str
    media_type: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)
    lineage: Lineage
    provenance: Provenance
    created_at: datetime = Field(default_factory=utcnow)


class PipelineConfig(DTO):
    seed: int = 42
    segmentation_adapter: Literal["silhouette_rules_v1"] = "silhouette_rules_v1"
    matching_adapter: Literal["semantic_side_matching_v1", "multisignal_v1"] = "multisignal_v1"
    reconstruction_adapter: Literal["ellipsoid_multiview_v1"] = "ellipsoid_multiview_v1"
    foreground_threshold: int = Field(default=32, ge=1, le=254)
    review_threshold: Score = 0.85
    build_blender: bool = True
    cache: bool = True


class StageResult(DTO):
    node_id: str
    stage: Stage
    status: JobStatus
    input_artifact_ids: list[UUID]
    output_artifact_id: UUID | None = None
    cache_key: str
    cached: bool = False
    invalidated: bool = False
    error: str | None = None


class RunManifest(DTO):
    schema_version: Literal["1.1.0"] = "1.1.0"
    run_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    project_snapshot: DollProject
    views: list[ImageView]
    config: PipelineConfig
    status: JobStatus = JobStatus.QUEUED
    stages: list[StageResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    replay_of: UUID | None = None
    error: str | None = None


class ReviewAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    RELABEL = "relabel"
    REMASK = "remask"
    REMATCH = "rematch"
    FINAL = "final_evaluation"


class ReviewRequest(DTO):
    artifact_id: UUID
    action: ReviewAction
    target_id: UUID | None = None
    reviewer: str = Field(min_length=1, max_length=120)
    reason_code: str = Field(min_length=1, max_length=120)
    dimension: str = "segmentation_accuracy"
    score: Score | None = None
    comment: str = Field(default="", max_length=4000)
    part_class: PartClass | None = None
    side: Side | None = None
    polygon: list[tuple[Score, Score]] | None = Field(default=None, min_length=3, max_length=5000)
    mask_png_base64: str | None = Field(default=None, max_length=16_000_000)
    assignments: dict[UUID, UUID] | None = None
    scores: dict[str, Score] = Field(default_factory=dict)


class FeedbackEvent(DTO):
    event_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    run_id: UUID
    stage: Stage
    target: UUID
    action: ReviewAction
    reviewer: str
    reason_code: str
    dimension: str
    score: Score | None
    scores: dict[str, Score]
    comment: str
    before_artifact_id: UUID
    after_artifact_id: UUID | None
    provenance: Provenance
    created_at: datetime = Field(default_factory=utcnow)
