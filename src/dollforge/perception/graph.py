from __future__ import annotations

from dollforge.contracts import PerceptionRequest
from dollforge.domain.models import PartClass, Provenance, ReviewState, Side
from dollforge.perception.geometry import (
    observation_bounds_by_view,
    summarize_geometry,
    view_geometry,
)
from dollforge.perception.models import (
    EvidenceKind,
    PartPerception,
    PerceptionGraph,
)
from dollforge.perception.relations import build_interfaces, build_relations


def _region(part_class: PartClass) -> str:
    if part_class == PartClass.HEAD:
        return "head"
    if part_class in (
        PartClass.TORSO,
        PartClass.ARM,
        PartClass.UPPER_ARM,
        PartClass.FOREARM,
        PartClass.HAND,
        PartClass.TOP,
    ):
        return "upper_body"
    if part_class in (
        PartClass.PELVIS,
        PartClass.LEG,
        PartClass.THIGH,
        PartClass.SHIN,
        PartClass.FOOT,
        PartClass.FOOTWEAR,
        PartClass.BOTTOM,
    ):
        return "lower_body"
    if part_class in (PartClass.FACE, PartClass.HAIR):
        return "surface_feature"
    return "accessory"


def _symmetry(parts: list[PartPerception]) -> str:
    lateral = [part for part in parts if part.side in (Side.LEFT, Side.RIGHT)]
    if not lateral:
        return "unknown"
    matched = 0
    for part in lateral:
        opposite = Side.RIGHT if part.side == Side.LEFT else Side.LEFT
        if any(
            candidate.part_class == part.part_class and candidate.side == opposite
            for candidate in lateral
        ):
            matched += 1
    ratio = matched / len(lateral)
    if ratio >= .8:
        return "bilateral"
    if ratio >= .35:
        return "approximate_bilateral"
    return "unknown"


class StructuredPerceptionBuilder:
    """Build an explicit, reviewable description before volumetry.

    The builder does not generate 3D geometry. It turns masks, cross-view identity
    and the DollGraph into structured evidence that later stages can consume.
    """

    model_id = "structured_perception_v1"
    model_version = "1.0.0"

    def describe(self, request: PerceptionRequest) -> PerceptionGraph:
        bounds = observation_bounds_by_view(request.observations)
        views = {view.view_id: view for view in request.views}
        observations = {
            observation.observation_id: observation
            for observation in request.observations
            if observation.review_state != ReviewState.REJECTED
        }
        parents = {
            joint.child_part_id: joint.parent_part_id
            for joint in request.graph.joints
        }

        parts: list[PartPerception] = []
        for instance in request.graph.parts:
            source_observations = [
                observations[observation_id]
                for observation_id in instance.observation_ids
                if observation_id in observations
            ]
            geometries = []
            for observation in source_observations:
                view = views.get(observation.view_id)
                object_bounds = bounds.get(observation.view_id)
                image_png = request.image_png_by_view.get(observation.view_id)
                mask_png = request.mask_png_by_observation.get(observation.observation_id)
                if view is None or object_bounds is None or image_png is None or mask_png is None:
                    continue
                geometries.append(view_geometry(
                    observation=observation,
                    view=view,
                    image_png=image_png,
                    mask_png=mask_png,
                    object_bounds=object_bounds,
                ))
            if not geometries:
                continue

            geometry = summarize_geometry(instance.part_class, source_observations, geometries)
            human_reviewed = bool(source_observations) and all(
                observation.review_state in (ReviewState.APPROVED, ReviewState.CORRECTED)
                for observation in source_observations
            )
            boundary_quality = sum(item.boundary_strength for item in geometries) / len(geometries)
            confidence = min(
                1.0,
                instance.confidence
                * (.65 + .20 * geometry.completeness + .15 * boundary_quality),
            )
            provenance_type = "human_approved" if human_reviewed else "derived_geometry"
            evidence = sorted({
                artifact_id
                for item in geometries
                for artifact_id in item.provenance.evidence
            }, key=str)
            parts.append(PartPerception(
                part_instance_id=instance.part_instance_id,
                part_class=instance.part_class,
                side=instance.side,
                region=_region(instance.part_class),
                parent_part_id=parents.get(instance.part_instance_id),
                observations=geometries,
                geometry=geometry,
                confidence=float(confidence),
                provenance=Provenance(
                    type=provenance_type,
                    source=self.model_id,
                    evidence=evidence,
                    note=(
                        "Descrição da peça derivada de máscaras e correspondência multi-view. "
                        "Forma oculta não é tratada como observação direta."
                    ),
                ),
            ))

        relations = build_relations(request.graph, parts)
        interfaces = build_interfaces(request.graph, parts)
        ordered_regions = [
            region for region in ("head", "upper_body", "lower_body", "surface_feature", "accessory")
            if any(part.region == region for part in parts)
        ]
        evidence = [request.graph_artifact_id]
        evidence.extend(sorted({
            artifact_id
            for part in parts
            for artifact_id in part.provenance.evidence
        }, key=str))

        warnings = [
            "Perception Graph descreve evidências 2D/multi-view; não é uma reconstrução 3D.",
            "depth_norm vem da projeção lateral e será substituído/refinado pela etapa de volumetria.",
            "interfaces mecânicas são hipóteses até receberem evidência visual ou revisão humana.",
        ]
        if any(part.geometry.evidence_kind == EvidenceKind.OBSERVED for part in parts):
            warnings.append(
                "Algumas peças não possuem evidência frontal+lateral suficiente para geometria multi-view."
            )

        return PerceptionGraph(
            project_id=request.project_id,
            style_family=request.style_family,
            symmetry=_symmetry(parts),
            major_regions=ordered_regions,
            parts=parts,
            relations=relations,
            interfaces=interfaces,
            warnings=warnings,
            provenance=Provenance(
                type="derived_geometry",
                source=self.model_id,
                evidence=evidence,
                note=(
                    "Camada descritiva entre percepção visual e volumetria. Mantém separado "
                    "o que foi observado, inferido por múltiplas vistas e sugerido por priors."
                ),
            ),
        )
