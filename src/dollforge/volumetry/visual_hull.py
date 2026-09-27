from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from statistics import median
from uuid import UUID

import numpy as np
import trimesh
from PIL import Image

from dollforge.contracts import (
    VolumeCandidate,
    VolumeFieldDescriptor,
    VolumeSlice,
    VolumetryRequest,
    VolumetryResult,
)
from dollforge.domain.models import PartClass, PartObservation, ReviewState
from dollforge.volumetry.metrics import evaluate_reprojection
from dollforge.volumetry.projection import backproject_pixel_ray, project_world_points
from dollforge.volumetry.sdf import extract_zero_surface, signed_distance_field


@dataclass
class VolumeFieldPayload:
    part_instance_id: UUID
    payload: bytes


def _mask(mask_png: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(mask_png)).convert("L")) > 127


def _depth_ratio(part_class: PartClass) -> float:
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


def _combine_interval(intervals: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not intervals:
        return None
    lows = [item[0] for item in intervals]
    highs = [item[1] for item in intervals]
    return float(median(lows)), float(median(highs))


def _part_bounds(
    instance,
    observations: list[PartObservation],
    cameras,
    perception,
    canonical_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    camera_map = {camera.view_id: camera for camera in cameras if camera.calibrated}
    x_intervals: list[tuple[float, float]] = []
    y_intervals: list[tuple[float, float]] = []
    z_intervals: list[tuple[float, float]] = []

    for observation in observations:
        camera = camera_map.get(observation.view_id)
        if camera is None:
            continue
        x0, y0, x1, y1 = observation.bbox_xyxy
        center_px = ((x0 + x1) / 2, (y0 + y1) / 2)
        center = np.asarray(backproject_pixel_ray(center_px, camera).origin_xyz, dtype=np.float64)
        matrix = np.asarray(camera.world_from_view, dtype=np.float64).reshape(4, 4)
        horizontal_axis = int(np.argmax(np.abs(matrix[:3, 0])))
        half_horizontal = (x1 - x0) * (camera.world_units_per_pixel or 1.0) / 2
        half_vertical = (y1 - y0) * (camera.world_units_per_pixel or 1.0) / 2

        if horizontal_axis == 0:
            x_intervals.append((center[0] - half_horizontal, center[0] + half_horizontal))
        elif horizontal_axis == 1:
            y_intervals.append((center[1] - half_horizontal, center[1] + half_horizontal))
        z_intervals.append((center[2] - half_vertical, center[2] + half_vertical))

    x_interval = _combine_interval(x_intervals)
    y_interval = _combine_interval(y_intervals)
    z_interval = _combine_interval(z_intervals)

    geometry = perception.geometry
    fallback_height = max(geometry.height_norm * canonical_height, canonical_height * .01)
    fallback_width = (
        geometry.frontal_width_norm * canonical_height
        if geometry.frontal_width_norm is not None
        else fallback_height * .65
    )
    fallback_depth = (
        geometry.depth_norm * canonical_height
        if geometry.depth_norm is not None
        else fallback_width * _depth_ratio(instance.part_class)
    )

    center_x = (sum(x_interval) / 2) if x_interval else 0.0
    center_y = (sum(y_interval) / 2) if y_interval else 0.0
    center_z = (sum(z_interval) / 2) if z_interval else 0.0
    width = (x_interval[1] - x_interval[0]) if x_interval else fallback_width
    depth = (y_interval[1] - y_interval[0]) if y_interval else fallback_depth
    height = (z_interval[1] - z_interval[0]) if z_interval else fallback_height

    center = np.asarray([center_x, center_y, center_z], dtype=np.float64)
    extents = np.maximum(
        np.asarray([width, depth, height], dtype=np.float64),
        canonical_height * .008,
    )
    return center, extents


def _evidence_weight(observation: PartObservation) -> tuple[float, bool]:
    if observation.review_state in (ReviewState.APPROVED, ReviewState.CORRECTED):
        return 1.0, True
    if str(observation.provenance.type) in ("human_edited", "human_approved"):
        return 1.0, True
    return max(.50, float(observation.confidence)), False


def _grid(
    center: np.ndarray,
    extents: np.ndarray,
    resolution: int,
    anisotropic: bool = False,
) -> tuple[np.ndarray, tuple[float, float, float], float | tuple[float, float, float]]:
    """Create an anisotropic grid that keeps detail in thin parts.

    The old cubic grid gave an arm the same number of samples as its longest
    axis.  Most samples were therefore wasted in empty depth/height and the
    SDF could only produce a blocky cross-section.  We keep a common nominal
    voxel scale, but allocate at least ``resolution // 2`` samples on every
    axis.  Marching cubes and the SDF both receive the real per-axis spacing.
    """
    padded = np.maximum(np.asarray(extents, dtype=np.float64) * 1.12, 1e-9)
    longest = float(padded.max())
    nominal = longest / resolution
    if anisotropic:
        shape = np.maximum(24, np.ceil(padded / nominal).astype(int))
        shape = np.minimum(shape, max(resolution * 3, 96))
        spacing = padded / shape
    else:
        shape = np.asarray([resolution, resolution, resolution], dtype=int)
        spacing = np.asarray([nominal, nominal, nominal], dtype=np.float64)
    minimum = center - padded / 2 + spacing / 2
    axes = [
        minimum[index] + np.arange(shape[index], dtype=np.float64) * spacing[index]
        for index in range(3)
    ]
    points = np.empty((*shape, 3), dtype=np.float64)
    points[..., 0] = axes[0][:, None, None]
    points[..., 1] = axes[1][None, :, None]
    points[..., 2] = axes[2][None, None, :]
    points = points.reshape(-1, 3)
    return (
        points,
        tuple(float(value) for value in minimum),
        (tuple(float(value) for value in spacing) if anisotropic else float(nominal)),
    )


def _inside_mask(
    points: np.ndarray, camera, mask: np.ndarray, chunk_size: int = 65_536,
) -> np.ndarray:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    height, width = mask.shape
    inside = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        pixels = project_world_points(points[start:start + chunk_size], camera)
        u = np.rint(pixels[:, 0]).astype(np.int64)
        v = np.rint(pixels[:, 1]).astype(np.int64)
        valid = (u >= 0) & (v >= 0) & (u < width) & (v < height)
        block = inside[start:start + chunk_size]
        block[valid] = mask[v[valid], u[valid]]
    return inside


def _volume_slices(
    occupancy: np.ndarray,
    origin: tuple[float, float, float],
    voxel_size: float | tuple[float, float, float],
    center: np.ndarray,
    extents: np.ndarray,
    confidence: float,
) -> list[VolumeSlice]:
    resolution = occupancy.shape[2]
    spacing = np.asarray(np.broadcast_to(voxel_size, 3), dtype=np.float64)
    slices: list[VolumeSlice] = []
    for z_index in range(resolution):
        x_indices, y_indices = np.where(occupancy[:, :, z_index])
        if len(x_indices):
            x_min = origin[0] + x_indices.min() * spacing[0]
            x_max = origin[0] + x_indices.max() * spacing[0]
            y_min = origin[1] + y_indices.min() * spacing[1]
            y_max = origin[1] + y_indices.max() * spacing[1]
            center_x = ((x_min + x_max) / 2 - center[0]) / max(extents[0], 1e-12)
            center_y = ((y_min + y_max) / 2 - center[1]) / max(extents[1], 1e-12)
            half_width = (x_max - x_min + spacing[0]) / 2 / max(extents[0], 1e-12)
            half_depth = (y_max - y_min + spacing[1]) / 2 / max(extents[1], 1e-12)
        else:
            center_x = center_y = half_width = half_depth = 0.0
        slices.append(VolumeSlice(
            z_norm=float((z_index + .5) / resolution),
            center_x_norm=float(center_x),
            half_width_norm=float(max(0.0, half_width)),
            center_y_norm=float(center_y),
            half_depth_norm=float(max(0.0, half_depth)),
            confidence=float(confidence),
        ))
    return slices


def _field_bytes(
    occupancy: np.ndarray,
    sdf: np.ndarray,
    origin: tuple[float, float, float],
    voxel_size: tuple[float, float, float],
) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        occupancy=occupancy.astype(np.uint8),
        sdf=sdf.astype(np.float32),
        origin_xyz=np.asarray(origin, dtype=np.float64),
        voxel_size=np.asarray(voxel_size, dtype=np.float64),
        shape_xyz=np.asarray(occupancy.shape, dtype=np.int32),
    )
    return buffer.getvalue()


class CalibratedVisualHullSDF:
    """Voxel visual hull in a calibrated canonical coordinate system."""

    model_id = "calibrated_visual_hull_sdf_v2"
    model_version = "2.1.0"

    def build_with_fields(
        self,
        request: VolumetryRequest,
    ) -> tuple[VolumetryResult, list[VolumeFieldPayload]]:
        cameras = {
            camera.view_id: camera
            for camera in request.cameras
            if camera.calibrated
        }
        observations = {
            observation.observation_id: observation
            for observation in request.observations
            if observation.review_state != ReviewState.REJECTED
        }
        perceived = {part.part_instance_id: part for part in request.perception.parts}
        volumes: list[VolumeCandidate] = []
        fields: list[VolumeFieldPayload] = []
        warnings: list[str] = []
        resolution = int(request.parameters.get("resolution", request.resolution))
        resolution = max(32, min(128, resolution))
        support_threshold = float(
            request.parameters.get("soft_support_threshold", .80)
        )
        support_threshold = max(.50, min(.98, support_threshold))

        for instance in request.graph.parts:
            perception = perceived.get(instance.part_instance_id)
            if perception is None:
                continue
            source = [
                observations[observation_id]
                for observation_id in instance.observation_ids
                if observation_id in observations
                and observation_id in request.mask_png_by_observation
                and observations[observation_id].view_id in cameras
            ]
            if not source:
                continue

            center, nominal_extents = _part_bounds(
                instance,
                source,
                list(cameras.values()),
                perception,
                request.graph.scale.canonical_height,
            )
            points, origin, voxel_size = _grid(
                center, nominal_extents, resolution, anisotropic=True
            )

            weighted_inside = np.zeros(len(points), dtype=np.float64)
            total_weight = 0.0
            hard_inside = np.ones(len(points), dtype=bool)
            source_views: list[str] = []
            used_cameras = []
            masks_by_view: dict[UUID, np.ndarray] = {}
            hard_by_view: dict[UUID, bool] = {}

            for observation in source:
                camera = cameras[observation.view_id]
                mask = _mask(request.mask_png_by_observation[observation.observation_id])
                weight, hard = _evidence_weight(observation)
                inside = _inside_mask(points, camera, mask)
                weighted_inside += weight * inside.astype(np.float64)
                total_weight += weight
                if hard:
                    hard_inside &= inside
                source_views.append(str(camera.label))
                used_cameras.append(camera)
                masks_by_view[camera.view_id] = mask
                hard_by_view[camera.view_id] = hard

            if total_weight <= 0:
                continue
            support = weighted_inside / total_weight
            occupancy_flat = hard_inside & (support >= support_threshold)
            if not occupancy_flat.any():
                # Automatic masks can disagree. Relax only non-human evidence; human edits remain hard.
                occupancy_flat = hard_inside & (support >= max(.55, support_threshold - .15))
                warnings.append(
                    f"{instance.part_class}:{instance.side} precisou de carving relaxado por conflito "
                    "entre máscaras automáticas."
                )
            if not occupancy_flat.any():
                warnings.append(
                    f"{instance.part_class}:{instance.side} não produziu occupancy consistente."
                )
                continue

            padded_extents = np.maximum(np.asarray(nominal_extents) * 1.12, 1e-9)
            grid_shape = tuple(
                int(value) for value in np.minimum(
                    np.maximum(24, np.ceil(padded_extents / (padded_extents.max() / resolution))),
                    max(resolution * 3, 96),
                )
            ) if isinstance(voxel_size, tuple) else (resolution,) * 3
            occupancy = occupancy_flat.reshape(grid_shape)
            sdf = signed_distance_field(occupancy, voxel_size)
            try:
                vertices, faces = extract_zero_surface(sdf, origin, voxel_size)
            except ValueError:
                warnings.append(
                    f"{instance.part_class}:{instance.side} não possui superfície SDF extraível."
                )
                continue

            mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=True,
            )
            if mesh.volume < 0:
                mesh.invert()
            vertices = mesh.vertices.tolist()
            faces = mesh.faces.tolist()

            occupied_points = points[occupancy_flat]
            spacing = np.asarray(np.broadcast_to(voxel_size, 3), dtype=np.float64)
            minimum = occupied_points.min(axis=0) - spacing / 2
            maximum = occupied_points.max(axis=0) + spacing / 2
            extents = np.maximum(maximum - minimum, spacing)
            volume_center = (minimum + maximum) / 2

            reprojection_metrics = []
            for camera in used_cameras:
                reference = masks_by_view[camera.view_id]
                reprojection_metrics.append(evaluate_reprojection(
                    occupied_points,
                    camera,
                    reference,
                    float(np.mean(spacing)),
                    hard_constraint=hard_by_view.get(camera.view_id, False),
                ))
            mean_iou = (
                float(np.mean([metric.silhouette_iou for metric in reprojection_metrics]))
                if reprojection_metrics else 0.0
            )
            camera_confidence = (
                float(np.mean([camera.confidence for camera in used_cameras]))
                if used_cameras else 0.0
            )
            coverage = min(1.0, len({camera.label for camera in used_cameras}) / 4)
            confidence = (
                .30 * perception.confidence
                + .20 * camera_confidence
                + .20 * coverage
                + .30 * mean_iou
            )
            confidence = float(np.clip(confidence, 0.0, 1.0))

            descriptor = VolumeFieldDescriptor(
                grid_shape=tuple(int(value) for value in occupancy.shape),
                # The public descriptor remains scalar for compatibility with
                # manufacturing limits; the exact anisotropic spacing is
                # persisted in the field artifact.
                voxel_size_world=float(np.mean(spacing)),
                voxel_size_mm=(float(np.mean(spacing))
                               if request.graph.scale.unit == "mm" else None),
                unit=request.graph.scale.unit,
                origin_xyz=origin,
            )
            field_payload = _field_bytes(occupancy, sdf, origin, voxel_size)
            fields.append(VolumeFieldPayload(
                part_instance_id=instance.part_instance_id,
                payload=field_payload,
            ))

            volumes.append(VolumeCandidate(
                part_instance_id=instance.part_instance_id,
                name=f"{instance.part_class}__{instance.side}__{str(instance.part_instance_id)[:8]}",
                source_views=sorted(set(source_views)),
                extents_xyz=tuple(float(value) for value in extents),
                center_xyz=tuple(float(value) for value in volume_center),
                slices=_volume_slices(
                    occupancy,
                    origin,
                    tuple(float(value) for value in spacing),
                    volume_center,
                    extents,
                    confidence,
                ),
                vertices=vertices,
                faces=faces,
                confidence=confidence,
                reprojection_metrics=reprojection_metrics,
                mean_reprojection_iou=float(np.clip(mean_iou, 0.0, 1.0)),
                field=descriptor,
                concavity_support=False,
                provenance={
                    "type": "derived_geometry",
                    "source": self.model_id,
                    "evidence": sorted(
                        {observation.mask_artifact_id for observation in source},
                        key=str,
                    ),
                    "note": (
                        "Occupancy voxelado obtido por reprojeção calibrada das silhuetas visuais. "
                        "Priors e relações inferidas não removem voxels."
                    ),
                },
            ))

        return VolumetryResult(
            volumes=volumes,
            unit=request.graph.scale.unit,
            method=self.model_id,
            warnings=[
                "Visual Hull v2 usa voxels em um sistema canônico calibrado por vista.",
                "SDF é persistido separadamente como artefato binário comprimido.",
                "Concavidades invisíveis na silhueta continuam fora do escopo.",
                *warnings,
            ],
        ), fields

    def build(self, request: VolumetryRequest) -> VolumetryResult:
        result, _fields = self.build_with_fields(request)
        return result
