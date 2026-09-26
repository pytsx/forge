from __future__ import annotations

from io import BytesIO
from statistics import median
from uuid import UUID

import numpy as np
from PIL import Image

from dollforge.domain.models import ImageView, PartClass, PartObservation, Provenance, ReviewState
from dollforge.perception.models import EvidenceKind, PartGeometry, ViewGeometry
from dollforge.vision.edges import boundary_adherence, sobel_edge_map


def observation_bounds_by_view(
    observations: list[PartObservation],
) -> dict[UUID, tuple[float, float, float, float]]:
    grouped: dict[UUID, list[tuple[float, float, float, float]]] = {}
    for observation in observations:
        if observation.review_state == ReviewState.REJECTED:
            continue
        grouped.setdefault(observation.view_id, []).append(observation.bbox_xyxy)
    return {
        view_id: (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        for view_id, boxes in grouped.items()
        if boxes
    }


def _width_profile(mask: np.ndarray, box: tuple[int, int, int, int], samples: int = 16) -> list[float]:
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    values: list[float] = []
    for index in range(samples):
        y = min(y1 - 1, y0 + int((index + 0.5) / samples * height))
        xs = np.where(mask[y, x0:x1])[0]
        values.append(0.0 if len(xs) == 0 else float((xs.max() - xs.min() + 1) / width))
    if len(values) >= 5:
        smooth = np.convolve(np.asarray(values, dtype=np.float32), np.ones(5) / 5, mode="same")
        values = smooth.tolist()
    return [float(np.clip(value, 0.0, 1.0)) for value in values]


def view_geometry(
    observation: PartObservation,
    view: ImageView,
    image_png: bytes,
    mask_png: bytes,
    object_bounds: tuple[float, float, float, float],
) -> ViewGeometry:
    image = np.asarray(Image.open(BytesIO(image_png)).convert("RGB"))
    mask = np.asarray(Image.open(BytesIO(mask_png)).convert("L")) > 127
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask and image dimensions differ")

    x0, y0, x1, y1 = (int(round(value)) for value in observation.bbox_xyxy)
    x0 = max(0, min(mask.shape[1] - 1, x0))
    y0 = max(0, min(mask.shape[0] - 1, y0))
    x1 = max(x0 + 1, min(mask.shape[1], x1))
    y1 = max(y0 + 1, min(mask.shape[0], y1))

    ox0, oy0, ox1, oy1 = object_bounds
    object_width = max(1.0, ox1 - ox0)
    object_height = max(1.0, oy1 - oy0)
    box_width = max(1.0, x1 - x0)
    box_height = max(1.0, y1 - y0)

    area = float(mask[y0:y1, x0:x1].sum())
    box_area = box_width * box_height
    object_area = object_width * object_height
    edges = sobel_edge_map(image)
    strength = boundary_adherence(mask, edges)

    center_x = (((x0 + x1) / 2) - ox0) / object_width
    center_y = (((y0 + y1) / 2) - oy0) / object_height

    return ViewGeometry(
        observation_id=observation.observation_id,
        view_id=view.view_id,
        view_label=view.label,
        center_xy_norm=(float(center_x), float(center_y)),
        width_norm=float(box_width / object_width),
        height_norm=float(box_height / object_height),
        area_ratio=float(np.clip(area / object_area, 0.0, 1.0)),
        fill_ratio=float(np.clip(area / box_area, 0.0, 1.0)),
        boundary_strength=float(np.clip(strength, 0.0, 1.0)),
        width_profile=_width_profile(mask, (x0, y0, x1, y1)),
        provenance=Provenance(
            type="observed",
            source="mask_geometry_v1",
            evidence=[observation.mask_artifact_id, view.normalized_artifact_id],
            note="Medidas extraídas diretamente da máscara e da imagem desta vista.",
        ),
    )


def shape_family(part_class: PartClass) -> str:
    mapping = {
        PartClass.HEAD: "rounded_spheroid",
        PartClass.TORSO: "rounded_tapered_body",
        PartClass.PELVIS: "rounded_pelvis",
        PartClass.ARM: "rounded_limb",
        PartClass.UPPER_ARM: "rounded_limb",
        PartClass.FOREARM: "rounded_limb",
        PartClass.LEG: "rounded_limb",
        PartClass.THIGH: "rounded_limb",
        PartClass.SHIN: "rounded_limb",
        PartClass.HAND: "rounded_hand",
        PartClass.FOOT: "rounded_foot",
        PartClass.FOOTWEAR: "rounded_footwear",
        PartClass.HAIR: "surface_feature",
        PartClass.FACE: "surface_feature",
        PartClass.TOP: "garment_shell",
        PartClass.BOTTOM: "garment_shell",
        PartClass.ACCESSORY: "accessory",
    }
    return mapping.get(part_class, "unknown")


def summarize_geometry(
    part_class: PartClass,
    observations: list[PartObservation],
    geometries: list[ViewGeometry],
) -> PartGeometry:
    front = [
        geometry.width_norm
        for geometry in geometries
        if geometry.view_label in ("front", "back")
    ]
    sides = [
        geometry.width_norm
        for geometry in geometries
        if geometry.view_label in ("left", "right")
    ]
    heights = [geometry.height_norm for geometry in geometries]
    frontal_width = float(median(front)) if front else None
    depth = float(median(sides)) if sides else None
    ratio = depth / frontal_width if depth is not None and frontal_width and frontal_width > 1e-8 else None
    unique_views = {geometry.view_label for geometry in geometries}
    completeness = min(1.0, len(unique_views) / 4)

    reviewed = {
        observation.review_state
        for observation in observations
    }
    if reviewed and reviewed <= {ReviewState.APPROVED, ReviewState.CORRECTED}:
        evidence_kind = EvidenceKind.HUMAN_CONFIRMED
        provenance_type = "human_approved"
        source = "reviewed_multiview_geometry_v1"
    elif front and sides:
        evidence_kind = EvidenceKind.OBSERVED_MULTIVIEW
        provenance_type = "derived_geometry"
        source = "multiview_geometry_v1"
    else:
        evidence_kind = EvidenceKind.OBSERVED
        provenance_type = "derived_geometry"
        source = "single_axis_geometry_v1"

    evidence = sorted({
        artifact_id
        for geometry in geometries
        for artifact_id in geometry.provenance.evidence
    }, key=str)
    return PartGeometry(
        height_norm=float(median(heights)) if heights else 0.0,
        frontal_width_norm=frontal_width,
        depth_norm=depth,
        depth_to_width=float(ratio) if ratio is not None else None,
        shape_family=shape_family(part_class),
        completeness=float(completeness),
        evidence_kind=evidence_kind,
        provenance=Provenance(
            type=provenance_type,
            source=source,
            evidence=evidence,
            note=(
                "Profundidade é estimada pela projeção lateral; não é ainda um mapa de profundidade "
                "nem uma superfície 3D."
            ),
        ),
    )
