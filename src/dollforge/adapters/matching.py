from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import log, sqrt
from uuid import UUID, uuid5

import numpy as np
from PIL import Image

from dollforge.contracts import MatchingRequest, MatchingResult
from dollforge.domain.models import PartInstance, PartObservation, Provenance, Side


@dataclass(frozen=True)
class Descriptor:
    observation: PartObservation
    view_id: UUID
    vector: np.ndarray


def _normalized_box(observation: PartObservation, width: int, height: int) -> tuple[float, ...]:
    x0, y0, x1, y1 = observation.bbox_xyxy
    w = max((x1 - x0) / width, 1e-6)
    h = max((y1 - y0) / height, 1e-6)
    cx = ((x0 + x1) / 2) / width
    cy = ((y0 + y1) / 2) / height
    return cx, cy, w, h


def _masked_appearance(image_png: bytes, mask_png: bytes) -> tuple[float, ...]:
    image = np.asarray(Image.open(BytesIO(image_png)).convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(BytesIO(mask_png)).convert("L")) > 127
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask and image dimensions differ")
    pixels = image[mask]
    if len(pixels) == 0:
        return (0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    fill = float(mask.mean())
    return (*mean.tolist(), *std.tolist(), fill)


def compatible(a: PartObservation, b: PartObservation) -> bool:
    if a.part_class != b.part_class:
        return False
    if a.view_id == b.view_id:
        return False
    if a.side == b.side:
        return True
    return Side.UNKNOWN in (a.side, b.side) or Side.BILATERAL in (a.side, b.side)


class MultiSignalMatcher:
    """Deterministic cross-view matcher using geometry, mask shape and appearance.

    This is intentionally lightweight. The descriptor boundary is designed so a learned
    embedding can replace the hand-crafted appearance features later without changing
    the matching contract or review workflow.
    """

    model_id = "multisignal_v1"
    model_version = "1.0.0"

    def __init__(self, max_distance: float = 0.42):
        self.max_distance = max_distance

    def describe(self, request: MatchingRequest, observation: PartObservation) -> Descriptor:
        view = next(v for v in request.views if v.view_id == observation.view_id)
        box = _normalized_box(observation, view.qa.width, view.qa.height)
        appearance = _masked_appearance(
            request.image_png_by_view[observation.view_id],
            request.mask_png_by_observation[observation.observation_id],
        )
        cx, cy, width, height = box
        mean_r, mean_g, mean_b, std_r, std_g, std_b, fill = appearance
        aspect = width / height
        vector = np.asarray(
            [
                cx,
                cy,
                log(width),
                log(height),
                log(max(aspect, 1e-6)),
                mean_r,
                mean_g,
                mean_b,
                std_r,
                std_g,
                std_b,
                fill,
            ],
            dtype=np.float32,
        )
        return Descriptor(observation=observation, view_id=observation.view_id, vector=vector)

    @staticmethod
    def distance(a: Descriptor, b: Descriptor) -> float:
        av, bv = a.vector, b.vector
        position = abs(float(av[1] - bv[1]))  # vertical location is stable across orthogonal views
        size = min(1.0, (abs(float(av[2] - bv[2])) + abs(float(av[3] - bv[3]))) / 2)
        aspect = min(1.0, abs(float(av[4] - bv[4])) / 2)
        color = min(1.0, float(np.linalg.norm(av[5:8] - bv[5:8])) / sqrt(3))
        texture = min(1.0, float(np.linalg.norm(av[8:11] - bv[8:11])) / sqrt(3))
        fill = min(1.0, abs(float(av[11] - bv[11])) * 2)
        return (
            0.30 * position
            + 0.20 * size
            + 0.10 * aspect
            + 0.20 * color
            + 0.10 * texture
            + 0.10 * fill
        )

    def match(self, request: MatchingRequest) -> MatchingResult:
        observations = [
            observation
            for observation in request.observations
            if observation.review_state != "rejected"
        ]
        descriptors = [self.describe(request, observation) for observation in observations]
        descriptors.sort(
            key=lambda d: (
                str(d.observation.part_class),
                str(d.observation.side),
                str(d.view_id),
                str(d.observation.observation_id),
            )
        )

        groups: list[list[Descriptor]] = []
        for descriptor in descriptors:
            candidates: list[tuple[float, int]] = []
            for index, group in enumerate(groups):
                if any(not compatible(descriptor.observation, member.observation) for member in group):
                    continue
                distance = float(np.mean([self.distance(descriptor, member) for member in group]))
                candidates.append((distance, index))
            if candidates:
                distance, index = min(candidates)
                if distance <= self.max_distance:
                    groups[index].append(descriptor)
                    continue
            groups.append([descriptor])

        parts: list[PartInstance] = []
        for group in groups:
            items = [item.observation for item in group]
            semantic_side = next((item.side for item in items if item.side != Side.UNKNOWN), items[0].side)
            pair_distances = [
                self.distance(group[i], group[j])
                for i in range(len(group))
                for j in range(i + 1, len(group))
            ]
            agreement = 1.0 - min(1.0, float(np.mean(pair_distances))) if pair_distances else 0.5
            confidence = max(
                0.0,
                min(1.0, min(item.confidence for item in items) * (0.6 + 0.4 * agreement)),
            )
            identity = "|".join(sorted(str(item.observation_id) for item in items))
            parts.append(
                PartInstance(
                    part_instance_id=uuid5(
                        request.project_id,
                        f"{self.model_id}:{items[0].part_class}:{semantic_side}:{identity}",
                    ),
                    part_class=items[0].part_class,
                    side=semantic_side,
                    observation_ids=[item.observation_id for item in items],
                    confidence=confidence,
                    provenance=Provenance(
                        type="derived_geometry",
                        source=self.model_id,
                        evidence=[item.mask_artifact_id for item in items],
                        note=(
                            "Match usa classe/lado como gates e combina posição vertical, "
                            "escala, proporção, máscara e aparência."
                        ),
                    ),
                )
            )

        for part in parts:
            if part.side in (Side.LEFT, Side.RIGHT):
                partner_side = Side.RIGHT if part.side == Side.LEFT else Side.LEFT
                partner = next(
                    (
                        candidate
                        for candidate in parts
                        if candidate.part_class == part.part_class and candidate.side == partner_side
                    ),
                    None,
                )
                part.symmetry_partner_id = partner.part_instance_id if partner else None

        return MatchingResult(
            parts=parts,
            warnings=[
                "Matching multi-sinal ainda não usa câmera calibrada nem embedding aprendido; "
                "ambiguidade deve ser revisada pelo humano."
            ],
        )
