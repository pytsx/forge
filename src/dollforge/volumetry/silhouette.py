from __future__ import annotations

from io import BytesIO
from statistics import median
from uuid import UUID

import numpy as np
import trimesh
from PIL import Image

from dollforge.contracts import VolumeCandidate, VolumeSlice, VolumetryRequest, VolumetryResult
from dollforge.domain.models import PartClass, PartObservation, Provenance, ReviewState, ViewLabel


def _object_bounds(
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


def _mask(mask_png: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(mask_png)).convert("L")) > 127


def _sample_profile(
    mask: np.ndarray,
    box: tuple[float, float, float, float],
    samples: int,
    mirror_horizontal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)

    centers = np.zeros(samples, dtype=np.float64)
    radii = np.zeros(samples, dtype=np.float64)
    for index in range(samples):
        fraction = (index + .5) / samples
        y = min(y1 - 1, y0 + int(fraction * box_height))
        xs = np.where(mask[y, x0:x1])[0]
        if len(xs) == 0:
            # Thin antialiased tips may disappear on one exact scanline. Search locally.
            for delta in (1, -1, 2, -2, 3, -3):
                yy = y + delta
                if yy < y0 or yy >= y1:
                    continue
                xs = np.where(mask[yy, x0:x1])[0]
                if len(xs):
                    break
        if len(xs) == 0:
            continue

        left = float(xs.min() / box_width)
        right = float((xs.max() + 1) / box_width)
        if mirror_horizontal:
            left, right = 1.0 - right, 1.0 - left
        centers[index] = (left + right) / 2 - .5
        radii[index] = max(0.0, (right - left) / 2)

    return _smooth_profile(centers, radii)


def _smooth_profile(
    centers: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.where(radii > 1e-5)[0]
    if len(valid) == 0:
        return centers, radii

    x = np.arange(len(radii))
    filled_radii = np.interp(x, valid, radii[valid])
    filled_centers = np.interp(x, valid, centers[valid])

    first, last = int(valid[0]), int(valid[-1])
    if first > 0:
        for index in range(first):
            filled_radii[index] *= (index + 1) / (first + 1)
            filled_centers[index] = filled_centers[first]
    if last < len(radii) - 1:
        span = len(radii) - last
        for index in range(last + 1, len(radii)):
            filled_radii[index] *= (len(radii) - index) / span
            filled_centers[index] = filled_centers[last]

    kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    padded_radii = np.pad(filled_radii, 2, mode="edge")
    padded_centers = np.pad(filled_centers, 2, mode="edge")
    smooth_radii = np.convolve(padded_radii, kernel, mode="valid")
    smooth_centers = np.convolve(padded_centers, kernel, mode="valid")
    return smooth_centers, smooth_radii


def _combine_profiles(
    profiles: list[tuple[np.ndarray, np.ndarray]],
    samples: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not profiles:
        return None
    centers = np.stack([item[0] for item in profiles])
    radii = np.stack([item[1] for item in profiles])
    output_center = np.zeros(samples, dtype=np.float64)
    output_radius = np.zeros(samples, dtype=np.float64)
    for index in range(samples):
        positive = radii[:, index] > 1e-5
        if not np.any(positive):
            continue
        output_center[index] = float(np.median(centers[positive, index]))
        # Median keeps opposing-view disagreement visible instead of letting one mask dominate.
        output_radius[index] = float(np.median(radii[positive, index]))
    return _smooth_profile(output_center, output_radius)


def _prior_depth_ratio(part_class: PartClass) -> float:
    return {
        PartClass.HEAD: .84,
        PartClass.TORSO: .58,
        PartClass.PELVIS: .72,
        PartClass.ARM: .82,
        PartClass.UPPER_ARM: .82,
        PartClass.FOREARM: .82,
        PartClass.HAND: .86,
        PartClass.LEG: .88,
        PartClass.THIGH: .88,
        PartClass.SHIN: .88,
        PartClass.FOOT: 1.15,
        PartClass.FOOTWEAR: 1.22,
    }.get(part_class, .75)


def _cross_section_exponent(part_class: PartClass) -> float:
    if part_class == PartClass.TORSO:
        return 2.55
    if part_class == PartClass.PELVIS:
        return 2.8
    if part_class in (PartClass.FOOT, PartClass.FOOTWEAR):
        return 3.1
    if part_class == PartClass.HEAD:
        return 2.25
    return 2.15


def _center_world(
    observations: list[PartObservation],
    view_labels: dict[UUID, ViewLabel],
    bounds: dict[UUID, tuple[float, float, float, float]],
    total_height: float,
) -> tuple[float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for observation in observations:
        object_box = bounds.get(observation.view_id)
        if object_box is None:
            continue
        bx0, by0, bx1, by1 = object_box
        factor = total_height / max(1.0, by1 - by0)
        x0, y0, x1, y1 = observation.bbox_xyxy
        horizontal = ((x0 + x1 - bx0 - bx1) / 2) * factor
        vertical = ((by0 + by1 - y0 - y1) / 2) * factor
        label = view_labels[observation.view_id]
        if label in (ViewLabel.FRONT, ViewLabel.BACK):
            xs.append(-horizontal if label == ViewLabel.BACK else horizontal)
        elif label in (ViewLabel.LEFT, ViewLabel.RIGHT):
            ys.append(-horizontal if label == ViewLabel.RIGHT else horizontal)
        zs.append(vertical)
    return (
        float(median(xs)) if xs else 0.0,
        float(median(ys)) if ys else 0.0,
        float(median(zs)) if zs else 0.0,
    )


def _surface_mesh(
    extents: np.ndarray,
    center: tuple[float, float, float],
    x_profile: tuple[np.ndarray, np.ndarray],
    y_profile: tuple[np.ndarray, np.ndarray],
    part_class: PartClass,
    samples: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    x_centers, x_radii = x_profile
    y_centers, y_radii = y_profile
    segments = max(24, samples)
    exponent = _cross_section_exponent(part_class)
    power = 2.0 / exponent

    vertices: list[list[float]] = []
    for index in range(samples):
        z = (.5 - (index + .5) / samples) * extents[2] + center[2]
        cx = center[0] + x_centers[index] * extents[0]
        cy = center[1] + y_centers[index] * extents[1]
        rx = max(x_radii[index] * extents[0], extents[0] * .008)
        ry = max(y_radii[index] * extents[1], extents[1] * .008)
        for segment in range(segments):
            theta = 2 * np.pi * segment / segments
            cosine, sine = np.cos(theta), np.sin(theta)
            x = cx + np.sign(cosine) * abs(cosine) ** power * rx
            y = cy + np.sign(sine) * abs(sine) ** power * ry
            vertices.append([float(x), float(y), float(z)])

    faces: list[list[int]] = []
    for index in range(samples - 1):
        for segment in range(segments):
            a = index * segments + segment
            b = index * segments + (segment + 1) % segments
            c = (index + 1) * segments + (segment + 1) % segments
            d = (index + 1) * segments + segment
            faces.extend([[a, d, b], [b, d, c]])

    top = len(vertices)
    bottom = top + 1
    vertices.append([center[0], center[1], center[2] + extents[2] / 2])
    vertices.append([center[0], center[1], center[2] - extents[2] / 2])
    last_ring = (samples - 1) * segments
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append([top, segment, nxt])
        faces.append([bottom, last_ring + nxt, last_ring + segment])

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )
    if mesh.volume < 0:
        mesh.invert()
    return mesh.vertices.tolist(), mesh.faces.tolist()


class SilhouetteVisualHull:
    """Fuse orthographic silhouettes into a smooth cross-sectional visual hull.

    This is deliberately a geometric baseline: it captures changing width, depth and
    centerline from multi-view masks, but does not claim to recover invisible concavities.
    """

    model_id = "silhouette_visual_hull_v1"
    model_version = "1.0.0"

    def build(self, request: VolumetryRequest) -> VolumetryResult:
        view_labels = {view.view_id: view.label for view in request.views}
        observations = {
            observation.observation_id: observation
            for observation in request.observations
            if observation.review_state != ReviewState.REJECTED
        }
        bounds = _object_bounds(request.observations)
        perceived = {part.part_instance_id: part for part in request.perception.parts}
        total_height = request.graph.scale.canonical_height
        volumes: list[VolumeCandidate] = []

        for instance in request.graph.parts:
            perception = perceived.get(instance.part_instance_id)
            if perception is None:
                continue
            source = [
                observations[observation_id]
                for observation_id in instance.observation_ids
                if observation_id in observations
                and observation_id in request.mask_png_by_observation
            ]
            if not source:
                continue

            frontal_profiles: list[tuple[np.ndarray, np.ndarray]] = []
            side_profiles: list[tuple[np.ndarray, np.ndarray]] = []
            source_views: list[str] = []
            for observation in source:
                label = view_labels.get(observation.view_id, ViewLabel.UNKNOWN)
                source_views.append(str(label))
                profile = _sample_profile(
                    _mask(request.mask_png_by_observation[observation.observation_id]),
                    observation.bbox_xyxy,
                    request.resolution,
                    mirror_horizontal=label in (ViewLabel.BACK, ViewLabel.RIGHT),
                )
                if label in (ViewLabel.FRONT, ViewLabel.BACK):
                    frontal_profiles.append(profile)
                elif label in (ViewLabel.LEFT, ViewLabel.RIGHT):
                    side_profiles.append(profile)

            x_profile = _combine_profiles(frontal_profiles, request.resolution)
            y_profile = _combine_profiles(side_profiles, request.resolution)
            geometry = perception.geometry
            part_height = max(
                geometry.height_norm * total_height,
                total_height * .01,
            )
            width = (
                geometry.frontal_width_norm * total_height
                if geometry.frontal_width_norm is not None
                else part_height * .65
            )
            depth = (
                geometry.depth_norm * total_height
                if geometry.depth_norm is not None
                else width * _prior_depth_ratio(instance.part_class)
            )

            if x_profile is None and y_profile is not None:
                y_centers, y_radii = y_profile
                ratio = max(width / max(depth, 1e-8), .2)
                x_profile = (np.zeros_like(y_centers), np.clip(y_radii * ratio, 0, .5))
            if y_profile is None and x_profile is not None:
                x_centers, x_radii = x_profile
                ratio = max(depth / max(width, 1e-8), .2)
                y_profile = (np.zeros_like(x_centers), np.clip(x_radii * ratio, 0, .5))
            if x_profile is None or y_profile is None:
                continue

            extents = np.maximum(
                np.asarray([width, depth, part_height], dtype=np.float64),
                total_height * .008,
            )
            center = _center_world(source, view_labels, bounds, total_height)
            vertices, faces = _surface_mesh(
                extents,
                center,
                x_profile,
                y_profile,
                instance.part_class,
                request.resolution,
            )

            x_centers, x_radii = x_profile
            y_centers, y_radii = y_profile
            axis_completeness = (bool(frontal_profiles) + bool(side_profiles)) / 2
            confidence = min(
                .92,
                perception.confidence * (.70 + .30 * axis_completeness),
            )
            slices = [
                VolumeSlice(
                    z_norm=float((index + .5) / request.resolution),
                    center_x_norm=float(x_centers[index]),
                    half_width_norm=float(np.clip(x_radii[index], 0, .5)),
                    center_y_norm=float(y_centers[index]),
                    half_depth_norm=float(np.clip(y_radii[index], 0, .5)),
                    confidence=float(confidence),
                )
                for index in range(request.resolution)
            ]
            evidence = sorted({
                observation.mask_artifact_id for observation in source
            }, key=str)
            volumes.append(VolumeCandidate(
                part_instance_id=instance.part_instance_id,
                name=f"{instance.part_class}__{instance.side}__{str(instance.part_instance_id)[:8]}",
                source_views=sorted(set(source_views)),
                extents_xyz=tuple(float(value) for value in extents),
                center_xyz=center,
                slices=slices,
                vertices=vertices,
                faces=faces,
                confidence=float(confidence),
                concavity_support=False,
                provenance=Provenance(
                    type="derived_geometry",
                    source=self.model_id,
                    evidence=evidence,
                    note=(
                        "Visual hull por seções fundindo silhuetas ortográficas. Preserva variação "
                        "de largura/profundidade e eixo da peça; concavidades internas exigem depth/normals."
                    ),
                ),
            ))

        return VolumetryResult(
            volumes=volumes,
            unit=request.graph.scale.unit,
            method=self.model_id,
            warnings=[
                "Volume guiado por silhuetas multi-view; não é mais uma primitiva geométrica fixa.",
                "Concavidades que não alteram a silhueta ainda não são recuperáveis nesta versão.",
                "A próxima evolução deve fundir depth maps e surface normals ao mesmo contrato.",
            ],
        )
