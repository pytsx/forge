from typing import Protocol
from uuid import UUID

from pydantic import Field

from dollforge.domain.models import (
    DTO,
    CameraEstimate,
    DollGraph,
    ImageView,
    PartInstance,
    PartObservation,
    Provenance,
    Score,
)
from dollforge.perception.models import PerceptionGraph


class MaskProposal(DTO):
    part_class: str
    side: str
    bbox_xyxy: tuple[int, int, int, int]
    confidence: Score
    provenance: Provenance
    mask_png: bytes


class SegmentationRequest(DTO):
    view: ImageView
    image_png: bytes
    threshold: int
    seed: int


class SegmentationResult(DTO):
    observations: list[PartObservation]
    warnings: list[str] = Field(default_factory=list)


class MatchingRequest(DTO):
    project_id: UUID
    observations: list[PartObservation]
    views: list[ImageView]
    image_png_by_view: dict[UUID, bytes]
    mask_png_by_observation: dict[UUID, bytes]


class MatchingResult(DTO):
    parts: list[PartInstance]
    warnings: list[str] = Field(default_factory=list)


class CameraResult(DTO):
    cameras: list[CameraEstimate]


class PerceptionRequest(DTO):
    project_id: UUID
    graph_artifact_id: UUID
    graph: DollGraph
    observations: list[PartObservation]
    views: list[ImageView]
    image_png_by_view: dict[UUID, bytes]
    mask_png_by_observation: dict[UUID, bytes]
    style_family: str | None = None


class VolumeSlice(DTO):
    z_norm: Score
    center_x_norm: float
    half_width_norm: float = Field(ge=0)
    center_y_norm: float
    half_depth_norm: float = Field(ge=0)
    confidence: Score


class VolumeCandidate(DTO):
    part_instance_id: UUID
    name: str
    source_views: list[str]
    extents_xyz: tuple[float, float, float]
    center_xyz: tuple[float, float, float]
    slices: list[VolumeSlice] = Field(min_length=4)
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    confidence: Score
    concavity_support: bool = False
    provenance: Provenance


class VolumetryRequest(DTO):
    project_id: UUID
    graph: DollGraph
    perception: PerceptionGraph
    observations: list[PartObservation]
    views: list[ImageView]
    mask_png_by_observation: dict[UUID, bytes]
    resolution: int = Field(ge=16, le=64)


class VolumetryResult(DTO):
    volumes: list[VolumeCandidate]
    unit: str
    method: str
    warnings: list[str] = Field(default_factory=list)


class MeshCandidate(DTO):
    part_instance_id: UUID
    name: str
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    confidence: Score
    provenance: Provenance
    transform: list[float] = Field(min_length=16, max_length=16)


class MeshRecord(DTO):
    part_instance_id: UUID
    name: str
    mesh_artifact_id: UUID
    preview_artifact_id: UUID
    confidence: Score
    provenance: Provenance
    transform: list[float]


class ReconstructionResult(DTO):
    meshes: list[MeshRecord]
    unit: str
    warnings: list[str]


class Check(DTO):
    code: str
    status: str
    part_instance_id: UUID | None = None
    measurement: float | None = None
    message: str


class ManufacturingReport(DTO):
    status: str
    manufacturable: bool
    checks: list[Check]


class BlenderResult(DTO):
    blend_artifact_id: UUID
    object_count: int
    blender_version: str
    unit: str


class SegmentationAdapter(Protocol):
    model_id: str
    model_version: str

    def predict(self, request: SegmentationRequest) -> list[MaskProposal]: ...


class MatchingAdapter(Protocol):
    model_id: str
    model_version: str

    def match(self, request: MatchingRequest) -> MatchingResult: ...


class PerceptionAdapter(Protocol):
    model_id: str
    model_version: str

    def describe(self, request: PerceptionRequest) -> PerceptionGraph: ...


class VolumetryAdapter(Protocol):
    model_id: str
    model_version: str

    def build(self, request: VolumetryRequest) -> VolumetryResult: ...


class ReconstructionAdapter(Protocol):
    model_id: str
    model_version: str

    def reconstruct(self, graph: DollGraph, observations: list[PartObservation],
                    views: list[ImageView],
                    volumetry: VolumetryResult | None = None) -> list[MeshCandidate]: ...


class BlenderAdapter(Protocol):
    model_version: str

    def build(self, meshes: list[MeshCandidate], unit: str) -> tuple[bytes, str]: ...
