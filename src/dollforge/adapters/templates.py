from __future__ import annotations

from math import pi

import numpy as np
import trimesh

from dollforge.contracts import MeshCandidate, VolumetryResult
from dollforge.domain.models import (
    DollGraph,
    ImageView,
    PartClass,
    PartObservation,
    Provenance,
    Side,
)


def _superellipsoid(extents: np.ndarray, exponent_xy: float = 2.0,
                    exponent_z: float = 2.0, rings: int = 20, segments: int = 32) -> trimesh.Trimesh:
    vertices = []
    faces = []
    for i in range(rings + 1):
        phi = -pi / 2 + pi * i / rings
        cp, sp = np.cos(phi), np.sin(phi)
        z_factor = np.sign(sp) * abs(sp) ** (2.0 / exponent_z)
        radial = abs(cp) ** (2.0 / exponent_z)
        for j in range(segments):
            theta = 2 * pi * j / segments
            ct, st = np.cos(theta), np.sin(theta)
            x = np.sign(ct) * abs(ct) ** (2.0 / exponent_xy) * radial
            y = np.sign(st) * abs(st) ** (2.0 / exponent_xy) * radial
            vertices.append([x * extents[0] / 2, y * extents[1] / 2, z_factor * extents[2] / 2])
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            faces.extend([[a, b, c], [a, c, d]])
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)


def _tapered_body(extents: np.ndarray, top_scale: float = .72,
                  bottom_scale: float = 1.0, rings: int = 18, segments: int = 32) -> trimesh.Trimesh:
    vertices = []
    faces = []
    for i in range(rings + 1):
        t = i / rings
        z = (t - .5) * extents[2]
        easing = .5 - .5 * np.cos(pi * t)
        radius_scale = top_scale * (1 - easing) + bottom_scale * easing
        round_cap = max(.88, 1 - .12 * abs(2 * t - 1) ** 4)
        for j in range(segments):
            theta = 2 * pi * j / segments
            vertices.append([
                np.cos(theta) * extents[0] / 2 * radius_scale * round_cap,
                np.sin(theta) * extents[1] / 2 * radius_scale * round_cap,
                z,
            ])
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            faces.extend([[a, b, c], [a, c, d]])
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    return mesh


def _shape_for(part_class: PartClass, extents: np.ndarray) -> trimesh.Trimesh:
    if part_class == PartClass.HEAD:
        return _superellipsoid(extents, exponent_xy=2.35, exponent_z=2.15)
    if part_class == PartClass.TORSO:
        return _tapered_body(extents, top_scale=.72, bottom_scale=1.0)
    if part_class == PartClass.PELVIS:
        return _superellipsoid(extents * np.array([1.0, .96, .72]), 3.4, 2.8)
    if part_class in (PartClass.ARM, PartClass.UPPER_ARM, PartClass.FOREARM):
        return _superellipsoid(extents * np.array([.80, .82, 1.0]), 2.25, 2.05)
    if part_class in (PartClass.LEG, PartClass.THIGH, PartClass.SHIN):
        return _superellipsoid(extents * np.array([.88, .92, 1.0]), 2.5, 2.15)
    if part_class in (PartClass.FOOT, PartClass.FOOTWEAR):
        return _superellipsoid(extents * np.array([1.08, 1.18, .72]), 3.0, 2.8)
    if part_class == PartClass.HAND:
        return _superellipsoid(extents * np.array([1.04, .92, .82]), 2.4, 2.3)
    return _superellipsoid(extents, 2.4, 2.4)


class DollTemplateReconstructor:
    """Shape-aware multi-view baseline for dolls.

    Still procedural, but each semantic class gets its own design prior instead of
    every part becoming the same ellipsoid.
    """

    model_id = "doll_templates_multiview_v2"
    model_version = "2.0.0"

    def reconstruct(self, graph: DollGraph, observations: list[PartObservation],
                    views: list[ImageView],
                    volumetry: VolumetryResult | None = None) -> list[MeshCandidate]:
        by_id = {o.observation_id: o for o in observations}
        view_map = {v.view_id: v for v in views}
        bounds = {}
        for view in views:
            boxes = [o.bbox_xyxy for o in observations if o.view_id == view.view_id]
            if boxes:
                bounds[view.view_id] = (
                    min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes),
                )

        height = graph.scale.canonical_height
        output: list[MeshCandidate] = []
        for part in graph.parts:
            obs = [by_id[o] for o in part.observation_ids if o in by_id]
            if not obs:
                continue
            frontal = [o for o in obs if view_map[o.view_id].label in ("front", "back")]
            profile = [o for o in obs if view_map[o.view_id].label in ("left", "right")]
            reference = (frontal or obs)[0]
            x0, y0, x1, y1 = reference.bbox_xyxy
            bx0, by0, bx1, by1 = bounds[reference.view_id]
            factor = height / max(1, by1 - by0)

            width = (x1 - x0) * factor
            part_height = (y1 - y0) * factor
            depth = width * .72
            if profile:
                depths = [
                    (o.bbox_xyxy[2] - o.bbox_xyxy[0]) * height /
                    max(1, bounds[o.view_id][3] - bounds[o.view_id][1])
                    for o in profile
                ]
                depth = float(np.median(depths))

            extents = np.maximum(
                np.asarray([width, depth, part_height], dtype=float),
                height * .012,
            )

            # Doll-specific sanity constraints prevent implausibly thin pieces.
            if part.part_class == PartClass.HEAD:
                extents[1] = max(extents[1], extents[0] * .78)
            elif part.part_class == PartClass.TORSO:
                extents[1] = max(extents[1], extents[0] * .46)
            elif part.part_class in (PartClass.ARM, PartClass.UPPER_ARM, PartClass.FOREARM):
                extents[:2] = np.maximum(extents[:2], part_height * .22)
            elif part.part_class in (PartClass.LEG, PartClass.THIGH, PartClass.SHIN):
                extents[:2] = np.maximum(extents[:2], part_height * .28)

            center_x = ((x0 + x1 - bx0 - bx1) / 2) * factor
            if view_map[reference.view_id].label == "back":
                center_x *= -1
            if part.side == Side.CENTER:
                center_x = 0
            center_z = (by1 - (y0 + y1) / 2) * factor - height * .43

            mesh = _shape_for(part.part_class, extents)
            mesh.apply_translation([center_x, 0, center_z])
            transform = np.eye(4)
            transform[:3, 3] = [center_x, 0, center_z]

            confidence = min(.72, max(.42, float(np.mean([o.confidence for o in obs])) + .10))
            output.append(MeshCandidate(
                part_instance_id=part.part_instance_id,
                name=f"{part.part_class}__{part.side}__{str(part.part_instance_id)[:8]}",
                vertices=mesh.vertices.tolist(),
                faces=mesh.faces.tolist(),
                confidence=confidence,
                transform=transform.flatten().tolist(),
                provenance=Provenance(
                    type="derived_geometry",
                    source=self.model_id,
                    evidence=[o.mask_artifact_id for o in obs],
                    note="Template semântico de boneco ajustado às medidas multi-view.",
                ),
            ))
        return output
