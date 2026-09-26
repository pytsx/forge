from __future__ import annotations

import platform
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid5

import numpy as np
import structlog
import trimesh
from pydantic import BaseModel

from dollforge.contracts import (
    BlenderAdapter,
    BlenderResult,
    CameraResult,
    Check,
    ManufacturingReport,
    MatchingAdapter,
    MatchingRequest,
    MatchingResult,
    MeshCandidate,
    MeshRecord,
    PerceptionAdapter,
    PerceptionRequest,
    ReconstructionAdapter,
    ReconstructionResult,
    SegmentationAdapter,
    SegmentationRequest,
    SegmentationResult,
    VolumetryAdapter,
    VolumetryRequest,
    VolumetryResult,
)
from dollforge.domain.models import (
    Artifact,
    CameraEstimate,
    DollGraph,
    JobStatus,
    JointSpec,
    Lineage,
    PartInstance,
    PartObservation,
    Provenance,
    RunManifest,
    ScaleEstimate,
    Stage,
    StageResult,
    utcnow,
)
from dollforge.errors import DomainError, InvalidInput, QualityLimitExceeded
from dollforge.perception.models import PerceptionGraph
from dollforge.quality.loop import run_quality_loop
from dollforge.quality.models import LimitTrace, TrainingSignal
from dollforge.quality.policies import (
    matching_limit,
    segmentation_limit,
    tune_matching,
    tune_segmentation,
    tune_volumetry,
    volumetry_limit,
)
from dollforge.storage import Store, canonical, digest
from dollforge.volumetry.calibration import calibrate_views

log = structlog.get_logger()


def source_fingerprint() -> str:
    root = Path(__file__).parent
    return digest(b"".join(p.relative_to(root).as_posix().encode() + p.read_bytes()
                           for p in sorted(root.rglob("*.py"))))


def lineage(inputs: list[UUID], version: str = "1.0.0", seed: int = 42,
            params=None) -> Lineage:
    return Lineage(input_artifact_ids=inputs, code_version=source_fingerprint(),
                   model_version=version, seed=seed, parameters_hash=digest(params or {}),
                   environment_fingerprint=f"python-{platform.python_version()}-{platform.system()}")


class Engine:
    def __init__(self, store: Store, segmenter: SegmentationAdapter,
                 reconstructor: ReconstructionAdapter, blender: BlenderAdapter,
                 segmenters: dict[str, SegmentationAdapter] | None = None,
                 matcher: MatchingAdapter | None = None,
                 perception: PerceptionAdapter | None = None,
                 volumetry: VolumetryAdapter | None = None,
                 volumetries: dict[str, VolumetryAdapter] | None = None,
                 reconstructors: dict[str, ReconstructionAdapter] | None = None):
        self.store = store
        self.segmenter = segmenter
        self.segmenters = {segmenter.model_id: segmenter, **(segmenters or {})}
        self.reconstructor = reconstructor
        self.reconstructors = {reconstructor.model_id: reconstructor, **(reconstructors or {})}
        self.blender = blender
        self.matcher = matcher
        self.perception = perception
        self.volumetry = volumetry
        self.volumetries = (
            {volumetry.model_id: volumetry, **(volumetries or {})}
            if volumetry is not None else dict(volumetries or {})
        )

    def record_quality_trace(
        self,
        run: RunManifest,
        stage: Stage,
        scope: str,
        inputs: list[UUID],
        trace: LimitTrace,
    ) -> Artifact:
        trace_artifact = self.put(
            run,
            stage,
            f"quality_limit:{scope}",
            trace,
            inputs,
            provenance=Provenance(
                type="derived_geometry",
                source="imperative_quality_limit_v1",
                evidence=inputs,
                note=(
                    "Prova quantitativa executada após geração. A etapa só pode avançar "
                    "quando o limite configurado é atingido."
                ),
            ),
            version="1.0.0",
        )
        node_id = f"{stage}:{scope}"
        node = next((item for item in reversed(run.stages) if item.node_id == node_id), None)
        if node is not None:
            node.quality_status = trace.status
            node.quality_attempts = len(trace.attempts)
            node.quality_score = trace.best_score
            node.quality_trace_artifact_id = trace_artifact.artifact_id

        if trace.status == "retrain_candidate":
            best = next(
                attempt for attempt in trace.attempts
                if attempt.attempt == trace.best_attempt
            )
            failed = [
                metric.code
                for metric in best.evaluation.metrics
                if not metric.passed
            ]
            if run.config.retrain_on_limit_exhaustion:
                signal = TrainingSignal(
                    stage=stage,
                    scope=scope,
                    specialist=trace.specialist,
                    input_artifact_ids=inputs,
                    trace_artifact_id=trace_artifact.artifact_id,
                    best_attempt=trace.best_attempt,
                    best_score=trace.best_score,
                    best_parameters=best.parameters,
                    failed_metrics=failed,
                )
                training_artifact = self.put(
                    run,
                    Stage.KNOWLEDGE,
                    f"retrain_candidate:{stage}:{scope}",
                    signal,
                    [trace_artifact.artifact_id, *inputs],
                    provenance=Provenance(
                        type="model_inferred",
                        source="quality_limit_retrain_trigger_v1",
                        evidence=[trace_artifact.artifact_id],
                        note=(
                            "Falha persistente após autoajuste. Este caso é candidato a "
                            "retreino offline; pesos não são alterados durante o run."
                        ),
                    ),
                )
                if node is not None:
                    node.training_signal_artifact_id = training_artifact.artifact_id
            if run.config.quality_fail_closed:
                raise QualityLimitExceeded(
                    f"{stage}:{scope} não atingiu o limite após "
                    f"{len(trace.attempts)} tentativas. Caso marcado para revisão/retreino."
                )
        return trace_artifact

    def segmentation_adapter(self, adapter_id: str) -> SegmentationAdapter:
        adapter = self.segmenters.get(adapter_id)
        if adapter is None:
            available = ", ".join(sorted(self.segmenters))
            raise InvalidInput(
                f"Segmentador '{adapter_id}' não configurado. Disponíveis: {available}"
            )
        return adapter

    def perception_adapter(self, adapter_id: str) -> PerceptionAdapter:
        if self.perception is None or self.perception.model_id != adapter_id:
            available = self.perception.model_id if self.perception is not None else "none"
            raise InvalidInput(
                f"Perception adapter '{adapter_id}' não configurado. Disponível: {available}"
            )
        return self.perception

    def volumetry_adapter(self, adapter_id: str) -> VolumetryAdapter:
        adapter = self.volumetries.get(adapter_id)
        if adapter is None:
            available = ", ".join(sorted(self.volumetries)) or "none"
            raise InvalidInput(
                f"Volumetry adapter '{adapter_id}' não configurado. Disponíveis: {available}"
            )
        return adapter

    def reconstruction_adapter(self, adapter_id: str) -> ReconstructionAdapter:
        adapter = self.reconstructors.get(adapter_id)
        if adapter is None:
            available = ", ".join(sorted(self.reconstructors))
            raise InvalidInput(
                f"Reconstrutor '{adapter_id}' não configurado. Disponíveis: {available}"
            )
        return adapter

    def put(self, run: RunManifest, stage: Stage, kind: str, payload,
            inputs: list[UUID], media_type="application/json", provenance=None,
            version="1.0.0") -> Artifact:
        return self.store.put(
            payload if isinstance(payload, bytes) else canonical(payload),
            project_id=run.project_id, run_id=run.run_id, stage=stage, kind=kind,
            lineage=lineage(inputs, version, run.config.seed, run.config.model_dump()),
            provenance=provenance or Provenance(type="rule_based", source=kind),
            media_type=media_type,
        )

    def node(self, run: RunManifest, stage: Stage, scope: str, inputs: list[UUID],
             execute: Callable[[], BaseModel], version="1.0.0",
             replay: RunManifest | None = None) -> Artifact:
        node_id = f"{stage}:{scope}"
        key = digest({"project": str(run.project_id), "stage": node_id,
                      "inputs": [{"id": str(i), "sha": self.store.metadata(i).sha256} for i in inputs],
                      "config": run.config.model_dump(exclude={"cache"}),
                      "height": run.project_snapshot.known_height_mm,
                      "code": source_fingerprint(), "model": version})
        result = StageResult(node_id=node_id, stage=stage, status=JobStatus.RUNNING,
                             input_artifact_ids=inputs, cache_key=key)
        run.stages.append(result)
        self.store.save("run", run.run_id, run)
        quality_gated = (
            run.config.quality_loop_enabled
            and stage in (Stage.SEGMENTATION, Stage.MATCHING, Stage.VOLUMETRY)
        )
        cached = self.store.cached(key) if run.config.cache and not quality_gated else None
        # Quality-gated stages rerun their proof even during replay; deterministic inputs
        # should reproduce the same accepted attempt while preserving an explicit trace.
        if (
            replay
            and not quality_gated
            and stage in (Stage.SEGMENTATION, Stage.MATCHING, Stage.GRAPH)
        ):
            old = next((s for s in replay.stages if s.node_id == node_id and not s.invalidated), None)
            if old and old.output_artifact_id:
                cached = self.store.metadata(old.output_artifact_id)
        try:
            if cached:
                self.store.read(cached.artifact_id)
                artifact = cached
                result.cached = True
            else:
                value = execute()
                artifact = self.put(run, stage, node_id, value, inputs, version=version)
                self.store.set_cache(key, artifact.artifact_id)
            result.output_artifact_id = artifact.artifact_id
            result.status = JobStatus.SUCCEEDED
            log.info("stage_succeeded", run_id=str(run.run_id), stage=stage,
                     node_id=node_id, cached=result.cached)
            return artifact
        except Exception as exc:
            result.status = (
                JobStatus.REVIEW if isinstance(exc, QualityLimitExceeded)
                else JobStatus.FAILED
            )
            result.error = str(exc) if isinstance(exc, DomainError) else type(exc).__name__
            log.exception("stage_failed", run_id=str(run.run_id), stage=stage)
            raise
        finally:
            self.store.save("run", run.run_id, run)

    def execute(self, run: RunManifest, replay: RunManifest | None = None) -> RunManifest:
        run.status = JobStatus.RUNNING
        self.store.save("run", run.run_id, run)
        try:
            labels = {v.label for v in run.views}
            if not {"front", "back", "left", "right"} <= labels:
                raise InvalidInput("Envie as quatro vistas: front, back, left e right.")
            raw = [v.normalized_artifact_id for v in run.views]
            camera = self.node(run, Stage.CAMERA, "all", raw, lambda: CameraResult(cameras=[
                CameraEstimate(view_id=v.view_id, label=v.label,
                    yaw_deg={"front": 0, "right": 90, "back": 180, "left": -90}.get(v.label, 0),
                    provenance=Provenance(type="rule_based", source="view_label_yaw_v1",
                        evidence=[v.normalized_artifact_id], note="Câmera não calibrada."))
                for v in run.views]), replay=replay)
            segments = []
            observations = []
            segmenter = self.segmentation_adapter(run.config.segmentation_adapter)
            for view in run.views:
                def segment(view=view):
                    image_png = self.store.read(view.normalized_artifact_id)
                    initial_parameters = {
                        "foreground_threshold": run.config.foreground_threshold,
                        "edge_threshold": .62,
                        "color_tolerance": .46,
                        "box_threshold": .28,
                        "text_threshold": .22,
                    }

                    def generate_segmentation(parameters):
                        request = SegmentationRequest(
                            view=view,
                            image_png=image_png,
                            threshold=int(parameters.get(
                                "foreground_threshold",
                                run.config.foreground_threshold,
                            )),
                            seed=run.config.seed + int(
                                parameters.get("attempt_seed_offset", 0)
                            ),
                            parameters=parameters,
                        )
                        return segmenter.predict(request)

                    if run.config.quality_loop_enabled:
                        proposals, trace = run_quality_loop(
                            stage=Stage.SEGMENTATION,
                            scope=str(view.view_id),
                            specialist=segmenter.model_id,
                            initial_parameters=initial_parameters,
                            max_attempts=run.config.quality_max_attempts,
                            generate=generate_segmentation,
                            evaluate=lambda proposals: segmentation_limit(
                                scope=str(view.view_id),
                                view_label=view.label,
                                image_png=image_png,
                                proposals=proposals,
                                boundary_threshold=run.config.segmentation_boundary_limit,
                                confidence_threshold=run.config.segmentation_confidence_limit,
                                coverage_threshold=run.config.segmentation_coverage_limit,
                            ),
                            tune=tune_segmentation,
                        )
                        self.record_quality_trace(
                            run,
                            Stage.SEGMENTATION,
                            str(view.view_id),
                            [view.normalized_artifact_id],
                            trace,
                        )
                    else:
                        proposals = generate_segmentation(initial_parameters)

                    if not proposals:
                        raise InvalidInput(
                            f"Silhueta não encontrada em {view.label}. Use fundo uniforme."
                        )
                    items = []
                    for proposal in proposals:
                        mask = self.put(run, Stage.SEGMENTATION, f"mask:{view.view_id}",
                            proposal.mask_png, [view.normalized_artifact_id], "image/png",
                            proposal.provenance, segmenter.model_version)
                        items.append(PartObservation(
                            observation_id=uuid5(view.view_id, f"{proposal.part_class}:{proposal.side}"),
                            view_id=view.view_id, part_class=proposal.part_class, side=proposal.side,
                            confidence=proposal.confidence, bbox_xyxy=proposal.bbox_xyxy,
                            mask_artifact_id=mask.artifact_id, provenance=proposal.provenance,
                            alternatives=["relabel", "remask"],
                        ))
                    warnings = {
                        "silhouette_rules_v1":
                            "Propostas proporcionais; revise todas as vistas.",
                        "contour_rules_v2":
                            "Máscaras seguem contorno e priors de bonecos; revise limites entre peças.",
                        "grounded_sam2_v1":
                            "Segmentação Grounding DINO + SAM 2; revise baixa confiança e oclusões.",
                    }
                    warning = warnings.get(
                        segmenter.model_id,
                        "Segmentação automática; revise limites e identidade das peças.",
                    )
                    return SegmentationResult(observations=items, warnings=[warning])
                artifact = self.node(run, Stage.SEGMENTATION, str(view.view_id),
                    [view.normalized_artifact_id], segment, segmenter.model_version, replay)
                segments.append(artifact.artifact_id)
                observations.extend(SegmentationResult.model_validate(
                    self.store.json(artifact.artifact_id)).observations)
            matching_inputs = [
                *segments,
                *[view.normalized_artifact_id for view in run.views],
                *[observation.mask_artifact_id for observation in observations],
            ]

            def match_views():
                initial_parameters = {"max_distance": .42}

                def generate_matching(parameters):
                    if run.config.matching_adapter == "semantic_side_matching_v1":
                        return self.match(run, observations)
                    if self.matcher is None or self.matcher.model_id != run.config.matching_adapter:
                        raise InvalidInput(
                            f"Matcher '{run.config.matching_adapter}' não está configurado."
                        )
                    return self.matcher.match(MatchingRequest(
                        project_id=run.project_id,
                        observations=observations,
                        views=run.views,
                        image_png_by_view={
                            view.view_id: self.store.read(view.normalized_artifact_id)
                            for view in run.views
                        },
                        mask_png_by_observation={
                            observation.observation_id: self.store.read(
                                observation.mask_artifact_id
                            )
                            for observation in observations
                        },
                        parameters=parameters,
                    ))

                if not run.config.quality_loop_enabled:
                    return generate_matching(initial_parameters)

                specialist = (
                    run.config.matching_adapter
                    if run.config.matching_adapter == "semantic_side_matching_v1"
                    else self.matcher.model_id
                )
                value, trace = run_quality_loop(
                    stage=Stage.MATCHING,
                    scope="all",
                    specialist=specialist,
                    initial_parameters=initial_parameters,
                    max_attempts=(
                        1 if run.config.matching_adapter == "semantic_side_matching_v1"
                        else run.config.quality_max_attempts
                    ),
                    generate=generate_matching,
                    evaluate=lambda result: matching_limit(
                        result,
                        confidence_threshold=run.config.matching_confidence_limit,
                        coverage_threshold=run.config.matching_coverage_limit,
                    ),
                    tune=tune_matching,
                )
                self.record_quality_trace(
                    run,
                    Stage.MATCHING,
                    "all",
                    matching_inputs,
                    trace,
                )
                return value

            matching_version = (
                self.matcher.model_version
                if self.matcher and run.config.matching_adapter == self.matcher.model_id
                else "1.0.0"
            )
            matches = self.node(
                run, Stage.MATCHING, "all", matching_inputs, match_views, matching_version,
                replay=replay,
            )
            matched = MatchingResult.model_validate(self.store.json(matches.artifact_id))
            scale = self.node(run, Stage.SCALE, "all", [camera.artifact_id], lambda: ScaleEstimate(
                mode="absolute" if run.project_snapshot.known_height_mm else "relative",
                canonical_height=run.project_snapshot.known_height_mm or 1.0,
                unit="mm" if run.project_snapshot.known_height_mm else "relative",
                confidence=.8 if run.project_snapshot.known_height_mm else .25,
                provenance=Provenance(type="human_edited" if run.project_snapshot.known_height_mm
                    else "rule_based", source="project_height" if run.project_snapshot.known_height_mm
                    else "normalized_height_1", note="Altura fornecida" if
                    run.project_snapshot.known_height_mm else "Sem escala física conhecida")), replay=replay)
            scale_value = ScaleEstimate.model_validate(self.store.json(scale.artifact_id))
            calibration = self.node(
                run,
                Stage.CALIBRATION,
                "all",
                [scale.artifact_id, *segments],
                lambda: CameraResult(cameras=calibrate_views(
                    scale_value,
                    observations,
                    run.views,
                )),
                version="1.0.0",
                replay=replay,
            )
            calibrated_cameras = CameraResult.model_validate(
                self.store.json(calibration.artifact_id)
            )
            graph_art = self.node(run, Stage.GRAPH, "all", [matches.artifact_id, scale.artifact_id],
                lambda: self.graph(run, matched, scale_value), replay=replay)
            graph = DollGraph.model_validate(self.store.json(graph_art.artifact_id))

            perception_adapter = self.perception_adapter(run.config.perception_adapter)
            perception_inputs = [
                graph_art.artifact_id,
                matches.artifact_id,
                *segments,
                *[view.normalized_artifact_id for view in run.views],
                *[observation.mask_artifact_id for observation in observations],
            ]

            def describe_scene():
                return perception_adapter.describe(PerceptionRequest(
                    project_id=run.project_id,
                    graph_artifact_id=graph_art.artifact_id,
                    graph=graph,
                    observations=observations,
                    views=run.views,
                    image_png_by_view={
                        view.view_id: self.store.read(view.normalized_artifact_id)
                        for view in run.views
                    },
                    mask_png_by_observation={
                        observation.observation_id: self.store.read(observation.mask_artifact_id)
                        for observation in observations
                    },
                    style_family=run.project_snapshot.style_family,
                ))

            perception_art = self.node(
                run,
                Stage.PERCEPTION,
                "all",
                perception_inputs,
                describe_scene,
                perception_adapter.model_version,
                replay=replay,
            )
            perception_value = PerceptionGraph.model_validate(
                self.store.json(perception_art.artifact_id)
            )

            volumetry_adapter = self.volumetry_adapter(run.config.volumetry_adapter)
            volumetry_inputs = [
                perception_art.artifact_id,
                graph_art.artifact_id,
                calibration.artifact_id,
                *[observation.mask_artifact_id for observation in observations],
            ]

            def build_volume():
                masks = {
                    observation.observation_id: self.store.read(observation.mask_artifact_id)
                    for observation in observations
                }
                initial_parameters = {
                    "resolution": run.config.volumetry_resolution,
                    "soft_support_threshold": .80,
                }

                def generate_volume(parameters):
                    request = VolumetryRequest(
                        project_id=run.project_id,
                        graph=graph,
                        perception=perception_value,
                        observations=observations,
                        views=run.views,
                        cameras=calibrated_cameras.cameras,
                        mask_png_by_observation=masks,
                        resolution=int(parameters.get(
                            "resolution",
                            run.config.volumetry_resolution,
                        )),
                        parameters=parameters,
                    )
                    if hasattr(volumetry_adapter, "build_with_fields"):
                        return volumetry_adapter.build_with_fields(request)
                    return volumetry_adapter.build(request), []

                if run.config.quality_loop_enabled:
                    candidate, trace = run_quality_loop(
                        stage=Stage.VOLUMETRY,
                        scope="all",
                        specialist=volumetry_adapter.model_id,
                        initial_parameters=initial_parameters,
                        max_attempts=run.config.quality_max_attempts,
                        generate=generate_volume,
                        evaluate=lambda candidate: volumetry_limit(
                            candidate[0],
                            iou_threshold=run.config.volumetry_iou_limit,
                            outside_area_threshold=run.config.volumetry_outside_area_limit,
                            overshoot_px_threshold=run.config.volumetry_overshoot_px_limit,
                        ),
                        tune=tune_volumetry,
                    )
                    self.record_quality_trace(
                        run,
                        Stage.VOLUMETRY,
                        "all",
                        volumetry_inputs,
                        trace,
                    )
                else:
                    candidate = generate_volume(initial_parameters)

                result, fields = candidate
                by_part = {volume.part_instance_id: volume for volume in result.volumes}
                for field in fields:
                    volume = by_part.get(field.part_instance_id)
                    if volume is None or volume.field is None:
                        continue
                    artifact = self.put(
                        run,
                        Stage.VOLUMETRY,
                        f"volume_field:{field.part_instance_id}.npz",
                        field.payload,
                        volumetry_inputs,
                        media_type="application/x-npz",
                        provenance=volume.provenance,
                        version=volumetry_adapter.model_version,
                    )
                    volume.field.artifact_id = artifact.artifact_id
                return result

            volumetry_art = self.node(
                run,
                Stage.VOLUMETRY,
                "all",
                volumetry_inputs,
                build_volume,
                volumetry_adapter.model_version,
                replay=replay,
            )
            volumetry_value = VolumetryResult.model_validate(
                self.store.json(volumetry_art.artifact_id)
            )
            reconstructor = self.reconstruction_adapter(run.config.reconstruction_adapter)

            def reconstruct():
                candidates = reconstructor.reconstruct(
                    graph, observations, run.views, volumetry_value
                )
                records = []
                for candidate in candidates:
                    mesh = trimesh.Trimesh(candidate.vertices, candidate.faces, process=False)
                    parent = [graph_art.artifact_id, volumetry_art.artifact_id]
                    stl = self.put(run, Stage.RECONSTRUCTION, f"mesh:{candidate.part_instance_id}",
                        mesh.export(file_type="stl"), parent, "model/stl", candidate.provenance,
                        reconstructor.model_version)
                    preview = self.put(run, Stage.RECONSTRUCTION,
                        f"preview:{candidate.part_instance_id}", candidate, parent,
                        provenance=candidate.provenance, version=reconstructor.model_version)
                    records.append(MeshRecord(part_instance_id=candidate.part_instance_id,
                        name=candidate.name, mesh_artifact_id=stl.artifact_id,
                        preview_artifact_id=preview.artifact_id, confidence=candidate.confidence,
                        provenance=candidate.provenance, transform=candidate.transform))
                warnings = {
                    "silhouette_volume_mesh_v1":
                        "Superfície guiada por volume multi-view; relevos internos ainda exigem depth/normals.",
                    "doll_templates_multiview_v2":
                        "Templates semânticos de bonecos; forma melhorada, ainda sem encaixes mecânicos.",
                    "ellipsoid_multiview_v1":
                        "Geometria baseline; não reproduz detalhes nem encaixes.",
                }
                warning = warnings.get(
                    reconstructor.model_id,
                    "Reconstrução automática; revise fidelidade e continuidade.",
                )
                return ReconstructionResult(meshes=records, unit=graph.scale.unit,
                    warnings=[warning])
            reconstruction = self.node(
                run,
                Stage.RECONSTRUCTION,
                "all",
                [graph_art.artifact_id, volumetry_art.artifact_id],
                reconstruct,
                reconstructor.model_version,
                replay,
            )
            reconstructed = ReconstructionResult.model_validate(self.store.json(reconstruction.artifact_id))
            candidates = [MeshCandidate.model_validate(self.store.json(m.preview_artifact_id))
                          for m in reconstructed.meshes]
            self.node(run, Stage.VALIDATION, "all", [reconstruction.artifact_id],
                      lambda: self.validate(
                          candidates,
                          graph.scale.unit,
                          volumetry_value,
                          calibrated_cameras.cameras,
                      ), replay=replay)
            if run.config.build_blender:
                def build():
                    data, version = self.blender.build(candidates, graph.scale.unit)
                    blend = self.put(run, Stage.BLENDER, "assembly.blend", data,
                        [reconstruction.artifact_id], "application/x-blender", version=version)
                    return BlenderResult(blend_artifact_id=blend.artifact_id,
                        object_count=len(candidates), blender_version=version, unit=graph.scale.unit)
                self.node(run, Stage.BLENDER, "all", [reconstruction.artifact_id], build,
                          self.blender.model_version, replay)
            run.status = JobStatus.REVIEW
        except QualityLimitExceeded as exc:
            run.status = JobStatus.REVIEW
            run.error = str(exc)
            log.warning("run_quality_limit_exhausted", run_id=str(run.run_id), error=str(exc))
        except Exception as exc:
            run.status = JobStatus.FAILED
            run.error = str(exc) if isinstance(exc, DomainError) else f"Falha interna: {type(exc).__name__}"
            log.exception("run_failed", run_id=str(run.run_id))
        run.completed_at = utcnow()
        self.store.save("run", run.run_id, run)
        self.put(run, Stage.REVIEW, "run_manifest", run,
                 [s.output_artifact_id for s in run.stages if s.output_artifact_id])
        return run

    def match(self, run: RunManifest, observations: list[PartObservation]) -> MatchingResult:
        groups = {}
        for observation in observations:
            if observation.review_state == "rejected":
                continue
            key = (observation.part_class, observation.side)
            if observation.side == "unknown":
                key = (*key, str(observation.observation_id))
            groups.setdefault(key, []).append(observation)
        parts = [PartInstance(
            part_instance_id=uuid5(run.project_id, ":".join(key)),
            part_class=items[0].part_class, side=items[0].side,
            observation_ids=[o.observation_id for o in items],
            confidence=min(o.confidence for o in items),
            provenance=Provenance(type="rule_based", source="semantic_side_matching_v1",
                                 evidence=[o.mask_artifact_id for o in items]),
        ) for key, items in groups.items()]
        for p in parts:
            if p.side in ("left", "right"):
                partner = next((q for q in parts if q.part_class == p.part_class and
                                q.side == ("right" if p.side == "left" else "left")), None)
                p.symmetry_partner_id = partner.part_instance_id if partner else None
        return MatchingResult(parts=parts,
            warnings=["Correspondência semântica; oclusões e identidades exigem revisão."])

    def graph(self, run: RunManifest, matches: MatchingResult, scale: ScaleEstimate) -> DollGraph:
        root = next((p for p in matches.parts if p.part_class == "pelvis"), None)
        torso = next((p for p in matches.parts if p.part_class == "torso"), None)
        if not root or not torso:
            raise InvalidInput("DollGraph precisa de pelvis e torso. Corrija os rótulos.")
        joints = []
        for part in matches.parts:
            if part.part_instance_id == root.part_instance_id:
                continue
            parent = torso if part.part_class in ("head", "arm", "upper_arm") else root
            if part.part_class == "footwear":
                parent = next((p for p in matches.parts if p.part_class == "leg" and p.side == part.side), root)
            joints.append(JointSpec(joint_id=uuid5(part.part_instance_id, "parent_joint"),
                parent_part_id=parent.part_instance_id, child_part_id=part.part_instance_id,
                joint_type="ball_socket" if part.part_class in ("head", "arm", "leg") else "fixed",
                confidence=.25, provenance=Provenance(type="rule_based", source="anatomy_v1",
                    note="Conexão candidata oculta; nenhuma geometria mecânica validada."),
                alternatives=["hinge", "peg_socket"]))
        return DollGraph(project_id=run.project_id, root_part_id=root.part_instance_id,
                         parts=matches.parts, joints=joints, scale=scale)

    def validate(
        self,
        candidates: list[MeshCandidate],
        unit: str,
        volumetry: VolumetryResult | None = None,
        cameras: list[CameraEstimate] | None = None,
    ) -> ManufacturingReport:
        checks = []
        for candidate in candidates:
            mesh = trimesh.Trimesh(candidate.vertices, candidate.faces, process=False)
            for name, passed in [("manifold", mesh.is_watertight),
                                  ("normals", mesh.is_winding_consistent and mesh.volume > 0),
                                  ("nonzero_faces", bool(np.all(mesh.area_faces > 1e-12)))]:
                checks.append(Check(code=name, status="pass" if passed else "fail",
                                    part_instance_id=candidate.part_instance_id, message=name))
        checks.append(Check(code="physical_scale", status="pass" if unit == "mm" else "warn",
                            message="Escala física fornecida" if unit == "mm" else "Escala relativa"))
        if cameras is not None:
            calibrated = [camera for camera in cameras if camera.calibrated]
            checks.append(Check(
                code="scale_calibration",
                status="pass" if len(calibrated) >= 4 else "warn",
                measurement=float(len(calibrated)),
                message=f"{len(calibrated)}/4 vistas calibradas no sistema canônico.",
            ))
        if volumetry is not None:
            for volume in volumetry.volumes:
                for metric in volume.reprojection_metrics:
                    checks.append(Check(
                        code=f"reprojection_{metric.view_label}",
                        status="pass" if metric.silhouette_iou >= .90 else "warn",
                        part_instance_id=volume.part_instance_id,
                        measurement=metric.silhouette_iou,
                        message=(
                            f"Fidelidade de silhueta {metric.view_label}: "
                            f"{metric.silhouette_iou:.1%}"
                        ),
                    ))
                if volume.reprojection_metrics:
                    checks.append(Check(
                        code="multiview_consistency",
                        status="pass" if volume.mean_reprojection_iou >= .90 else "warn",
                        part_instance_id=volume.part_instance_id,
                        measurement=volume.mean_reprojection_iou,
                        message=f"Mean reprojection IoU: {volume.mean_reprojection_iou:.1%}",
                    ))
        for name in ("wall_thickness", "clearance", "collisions", "joint_range", "separability"):
            checks.append(Check(code=name, status="not_evaluated",
                                message="Requer validação mecânica especializada."))
        return ManufacturingReport(status="needs_review", manufacturable=False, checks=checks)
