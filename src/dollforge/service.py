from __future__ import annotations

import base64
import binascii
from io import BytesIO
from uuid import UUID, uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from dollforge.adapters.baseline import png
from dollforge.adapters.contour import transfer_human_mask
from dollforge.contracts import MatchingResult, SegmentationResult
from dollforge.domain.models import (
    CreateProject, DollGraph, DollProject, FeedbackEvent, ImageQA, ImageView, JobStatus,
    PartInstance, PipelineConfig, Provenance, ReviewAction, ReviewRequest, ReviewState,
    RunManifest, Stage, ViewLabel,
)
from dollforge.errors import Conflict, InvalidInput
from dollforge.orchestration import Engine, lineage
from dollforge.storage import Store, canonical

MAX_UPLOAD = 20 * 1024 * 1024
FINAL_DIMENSIONS = {"segmentation_accuracy", "cross_view_consistency", "shape_fidelity",
    "style_fidelity", "joint_correctness", "connector_correctness", "assembly_quality",
    "printability", "editability"}


def decode_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD:
        raise InvalidInput("Imagem excede o limite de 20 MB.")
    try:
        image = Image.open(BytesIO(data))
        if image.format not in ("PNG", "JPEG", "WEBP"):
            raise InvalidInput("Use PNG, JPEG ou WebP.")
        if image.width * image.height > 24_000_000:
            raise InvalidInput("Imagem excede 24 megapixels.")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidInput("Arquivo de imagem inválido.") from exc


class Service:
    def __init__(self, store: Store, engine: Engine):
        self.store = store
        self.engine = engine

    def create_project(self, request: CreateProject) -> DollProject:
        project = DollProject(**request.model_dump())
        self.store.save("project", project.project_id, project)
        return project

    def add_view(self, project_id: UUID, label: ViewLabel, data: bytes) -> ImageView:
        with self.store.lock:
            project = self.store.get("project", project_id, DollProject)
            image = decode_image(data)
            gray = np.asarray(image.convert("L").resize((256, 256)), dtype=float)
            sharpness = float(np.var(gray[1:] - gray[:-1]) + np.var(gray[:, 1:] - gray[:, :-1]))
            warnings = []
            if min(image.size) < 256:
                warnings.append("low_resolution")
            if sharpness < 15:
                warnings.append("possible_blur")
            warnings.append("occlusion_and_crop_require_review")
            run_id = uuid4()
            provenance = Provenance(type="observed", source="local_upload")
            raw = self.store.put(data, project_id=project_id, run_id=run_id, stage=Stage.INTAKE,
                kind="raw_image", lineage=lineage([]), provenance=provenance,
                media_type={"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[
                    Image.open(BytesIO(data)).format])
            normalized = self.store.put(png(image), project_id=project_id, run_id=run_id,
                stage=Stage.INTAKE, kind="normalized_image", lineage=lineage([raw.artifact_id]),
                provenance=Provenance(type="derived_geometry", source="exif_orientation_rgba",
                                      evidence=[raw.artifact_id]), media_type="image/png")
            view = ImageView(project_id=project_id, label=label, original_artifact_id=raw.artifact_id,
                normalized_artifact_id=normalized.artifact_id,
                qa=ImageQA(width=image.width, height=image.height, sharpness=sharpness, warnings=warnings),
                provenance=provenance)
            # Replacing a view only changes the project pointer; old views and blobs remain immutable.
            project.views = [identifier for identifier in project.views
                             if self.store.get("view", identifier, ImageView).label != label]
            project.views.append(view.view_id)
            self.store.save("view", view.view_id, view)
            self.store.save("project", project_id, project)
            return view

    def new_run(self, project_id: UUID, config: PipelineConfig,
                replay_id: UUID | None = None) -> RunManifest:
        with self.store.lock:
            if replay_id:
                previous = self.store.get("run", replay_id, RunManifest)
                if previous.project_id != project_id:
                    raise Conflict("Run pertence a outro projeto.")
                if any(s.invalidated for s in previous.stages):
                    raise Conflict("Run possui etapas obsoletas. Reprocesse antes de reproduzir.")
                project, views, config = previous.project_snapshot, previous.views, previous.config
            else:
                project = self.store.get("project", project_id, DollProject)
                views = [self.store.get("view", identifier, ImageView) for identifier in project.views]
            if not {"front", "back", "left", "right"} <= {v.label for v in views}:
                raise InvalidInput("Envie frente, costas, esquerda e direita antes de executar.")
            run = RunManifest(project_id=project_id, project_snapshot=project, views=views,
                              config=config, replay_of=replay_id)
            self.store.save("run", run.run_id, run)
            return run

    def execute(self, run_id: UUID) -> RunManifest:
        with self.store.lock:
            run = self.store.get("run", run_id, RunManifest)
            previous = self.store.get("run", run.replay_of, RunManifest) if run.replay_of else None
            return self.engine.execute(run, previous)

    @staticmethod
    def _is_human_locked(observation) -> bool:
        return observation.review_state in (ReviewState.APPROVED, ReviewState.CORRECTED) or (
            observation.provenance.type in ("human_edited", "human_approved")
        )

    def _propagate_segmentation_feedback(self, run: RunManifest, source_observation,
                                         source_mask_artifact_id: UUID,
                                         reviewer: str) -> list[UUID]:
        """Recalculate untouched views from one human-corrected mask.

        This is project-local online adaptation, not weight training. Human-reviewed
        targets are never overwritten. Each generated mask remains reviewable and
        records the corrected source mask as evidence.
        """
        source_mask = self.store.read(source_mask_artifact_id)
        changed_previous_artifacts: list[UUID] = []

        for node in run.stages:
            if node.stage != Stage.SEGMENTATION or node.invalidated or not node.output_artifact_id:
                continue
            payload = SegmentationResult.model_validate(self.store.json(node.output_artifact_id))
            changed = False
            new_inputs = [node.output_artifact_id, source_mask_artifact_id]

            for observation in payload.observations:
                if observation.observation_id == source_observation.observation_id:
                    continue
                if observation.part_class != source_observation.part_class:
                    continue
                if observation.side != source_observation.side:
                    continue
                if self._is_human_locked(observation):
                    continue

                view = next(v for v in run.views if v.view_id == observation.view_id)
                transferred = transfer_human_mask(
                    source_mask_png=source_mask,
                    target_image_png=self.store.read(view.normalized_artifact_id),
                    target_mask_png=self.store.read(observation.mask_artifact_id),
                    target=observation,
                    threshold=run.config.foreground_threshold,
                )
                if not transferred:
                    continue

                mask_png, box = transferred
                provenance = Provenance(
                    type="model_inferred",
                    source="human_guided_mask_transfer_v1",
                    evidence=[source_mask_artifact_id, observation.mask_artifact_id],
                    note=(
                        "Máscara recalculada a partir de correção humana em outra vista; "
                        "permanece pendente de revisão."
                    ),
                )
                mask_art = self.engine.put(
                    run,
                    Stage.SEGMENTATION,
                    f"mask:{view.view_id}",
                    mask_png,
                    [source_mask_artifact_id, observation.mask_artifact_id],
                    "image/png",
                    provenance,
                    version="1.0.0",
                )
                observation.mask_artifact_id = mask_art.artifact_id
                observation.bbox_xyxy = box
                observation.confidence = min(.92, max(observation.confidence, .66))
                observation.review_state = ReviewState.NEEDS_REVIEW
                observation.provenance = provenance
                new_inputs.append(mask_art.artifact_id)
                changed = True

            if changed:
                previous = node.output_artifact_id
                revised = self.engine.put(
                    run,
                    Stage.SEGMENTATION,
                    self.store.metadata(previous).kind,
                    payload,
                    new_inputs,
                    provenance=Provenance(
                        type="model_inferred",
                        source="human_guided_mask_transfer_v1",
                        evidence=[source_mask_artifact_id],
                        note=f"Adaptação online do projeto após correção por {reviewer}.",
                    ),
                    version="1.0.0",
                )
                node.output_artifact_id = revised.artifact_id
                self.store.set_cache(node.cache_key, revised.artifact_id)
                changed_previous_artifacts.append(previous)

        return changed_previous_artifacts

    def review(self, run_id: UUID, request: ReviewRequest) -> FeedbackEvent:
        with self.store.lock:
            run = self.store.get("run", run_id, RunManifest)
            if run.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                raise Conflict("Aguarde a execução terminar para revisar.")
            stage = next((s for s in run.stages if s.output_artifact_id == request.artifact_id), None)
            if not stage or stage.invalidated:
                raise Conflict("A revisão deve apontar para a versão atual de uma etapa válida.")
            artifact = self.store.metadata(request.artifact_id)
            payload = self.store.json(request.artifact_id)
            provenance = Provenance(type="human_approved" if request.action == ReviewAction.APPROVE
                else "human_edited", source=request.reviewer, evidence=[request.artifact_id],
                note=request.reason_code)
            inputs = [request.artifact_id]
            if request.action == ReviewAction.FINAL:
                if any(s.invalidated for s in run.stages):
                    raise Conflict("Reprocesse as etapas obsoletas antes da avaliação final.")
                if not FINAL_DIMENSIONS <= request.scores.keys():
                    raise InvalidInput("A avaliação final exige as nove dimensões de qualidade.")
                revised = None
            elif stage.stage == Stage.SEGMENTATION:
                output = SegmentationResult.model_validate(payload)
                targets = [o for o in output.observations if request.target_id is None or
                           o.observation_id == request.target_id]
                if not targets:
                    raise InvalidInput("Observação não encontrada.")
                if request.action in (ReviewAction.REMASK, ReviewAction.RELABEL) and len(targets) != 1:
                    raise InvalidInput("Selecione exatamente uma observação.")
                for observation in targets:
                    if request.action == ReviewAction.REMASK:
                        view = next(v for v in run.views if v.view_id == observation.view_id)
                        if request.mask_png_base64:
                            try:
                                mask = decode_image(base64.b64decode(request.mask_png_base64, validate=True)).convert("L")
                            except (binascii.Error, ValueError) as exc:
                                raise InvalidInput("Máscara base64 inválida.") from exc
                            if mask.size != (view.qa.width, view.qa.height):
                                raise InvalidInput("Máscara deve ter as dimensões da imagem normalizada.")
                            mask = mask.point(lambda p: 255 if p >= 128 else 0)
                        elif request.polygon:
                            mask = Image.new("L", (view.qa.width, view.qa.height))
                            ImageDraw.Draw(mask).polygon([(round(x * (mask.width - 1)),
                                round(y * (mask.height - 1))) for x, y in request.polygon], fill=255)
                        else:
                            raise InvalidInput("Informe um polígono ou máscara PNG.")
                        box = mask.getbbox()
                        if not box or box[2] - box[0] < 2 or box[3] - box[1] < 2:
                            raise InvalidInput("Máscara vazia ou degenerada.")
                        mask_art = self.engine.put(run, Stage.SEGMENTATION,
                            f"mask:{view.view_id}", png(mask),
                            [observation.mask_artifact_id], "image/png", provenance)
                        inputs.append(mask_art.artifact_id)
                        observation.mask_artifact_id = mask_art.artifact_id
                        observation.bbox_xyxy = box
                    elif request.action == ReviewAction.RELABEL:
                        if request.part_class is None and request.side is None:
                            raise InvalidInput("Informe classe ou lado.")
                        observation.part_class = request.part_class or observation.part_class
                        observation.side = request.side or observation.side
                    elif request.action not in (ReviewAction.APPROVE, ReviewAction.REJECT):
                        raise InvalidInput("Ação incompatível com segmentação.")
                    observation.review_state = self.review_state(request.action)
                    observation.provenance = provenance
                revised = output
            elif stage.stage == Stage.MATCHING:
                output = MatchingResult.model_validate(payload)
                if request.action == ReviewAction.REMATCH:
                    if not request.assignments:
                        raise InvalidInput("Informe os pares observation_id → part_instance_id.")
                    observations = {o: p for p in output.parts for o in p.observation_ids}
                    if not request.assignments.keys() <= observations.keys():
                        raise InvalidInput("Observação desconhecida no rematch.")
                    ids = {p.part_instance_id: p for p in output.parts}
                    updated = {p.part_instance_id: [] for p in output.parts}
                    for oid, origin in observations.items():
                        target = request.assignments.get(oid, origin.part_instance_id)
                        if target not in ids:
                            ids[target] = PartInstance(**{**origin.model_dump(),
                                "part_instance_id": target, "observation_ids": [oid],
                                "symmetry_partner_id": None})
                        updated.setdefault(target, []).append(oid)
                    output.parts = []
                    for pid, oids in updated.items():
                        if oids:
                            part = ids[pid]
                            part.observation_ids = oids
                            part.symmetry_partner_id = None
                            part.provenance = provenance
                            part.review_state = ReviewState.CORRECTED
                            output.parts.append(part)
                elif request.action in (ReviewAction.APPROVE, ReviewAction.REJECT):
                    selected = [p for p in output.parts if request.target_id is None or
                                p.part_instance_id == request.target_id]
                    if not selected:
                        raise InvalidInput("Peça não encontrada.")
                    for part in selected:
                        part.review_state = self.review_state(request.action)
                        part.provenance = provenance
                else:
                    raise InvalidInput("Ação incompatível com correspondência.")
                revised = output
            elif stage.stage == Stage.GRAPH:
                if request.action not in (ReviewAction.APPROVE, ReviewAction.REJECT):
                    raise InvalidInput("Grafo permite aprovação ou rejeição nesta versão.")
                output = DollGraph.model_validate(payload)
                for part in output.parts:
                    part.review_state = self.review_state(request.action)
                    part.provenance = provenance
                # Hidden joint decisions remain needs_review even when approving graph structure.
                revised = output
            else:
                raise InvalidInput("Esta etapa aceita apenas avaliação final multidimensional.")
            after = None
            propagated_dirty: list[UUID] = []
            if revised is not None:
                after = self.engine.put(run, stage.stage, artifact.kind, revised, inputs,
                                        provenance=provenance)
                stage.output_artifact_id = after.artifact_id
                self.store.set_cache(stage.cache_key, after.artifact_id)

                if stage.stage == Stage.SEGMENTATION and request.action == ReviewAction.REMASK:
                    source = next(
                        o for o in revised.observations
                        if o.observation_id == request.target_id
                    )
                    propagated_dirty = self._propagate_segmentation_feedback(
                        run, source, source.mask_artifact_id, request.reviewer
                    )

                dirty = {request.artifact_id, *propagated_dirty}
                for node in run.stages:
                    if node is not stage and dirty.intersection(node.input_artifact_ids):
                        node.invalidated = True
                        if node.output_artifact_id:
                            dirty.add(node.output_artifact_id)
                run.status = JobStatus.REVIEW
                self.store.save("run", run.run_id, run)
            event = FeedbackEvent(project_id=run.project_id, run_id=run_id, stage=stage.stage,
                target=request.target_id or request.artifact_id, action=request.action,
                reviewer=request.reviewer, reason_code=request.reason_code, dimension=request.dimension,
                score=request.score, scores=request.scores, comment=request.comment,
                before_artifact_id=request.artifact_id,
                after_artifact_id=after.artifact_id if after else None, provenance=provenance)
            self.store.save("feedback", event.event_id, event)
            self.engine.put(run, Stage.REVIEW, "feedback", event,
                [request.artifact_id] + ([after.artifact_id] if after else []), provenance=provenance)
            return event

    @staticmethod
    def review_state(action: ReviewAction) -> ReviewState:
        return {ReviewAction.APPROVE: ReviewState.APPROVED,
                ReviewAction.REJECT: ReviewState.REJECTED}.get(action, ReviewState.CORRECTED)

    def promote(self, run_id: UUID, reviewer: str) -> dict:
        run = self.store.get("run", run_id, RunManifest)
        if run.status != JobStatus.REVIEW or any(s.invalidated for s in run.stages):
            raise Conflict("Execute e revise um run atual antes de promover referências.")
        for node in run.stages:
            if node.stage not in (Stage.SEGMENTATION, Stage.MATCHING, Stage.GRAPH):
                continue
            data = self.store.json(node.output_artifact_id)
            items = data.get("observations", data.get("parts", []))
            if any(i["review_state"] != "approved" for i in items):
                raise Conflict("Todas as observações, correspondências e peças exigem aprovação explícita.")
        final = [f for f in self.store.list("feedback", FeedbackEvent)
                 if f.run_id == run_id and f.action == ReviewAction.FINAL]
        if not final:
            raise Conflict("Avaliação final multidimensional obrigatória.")
        # Gold here is perception-only; mechanical candidates are never silently promoted.
        artifact = self.engine.put(run, Stage.KNOWLEDGE, "gold_perception_reference",
            {"run_id": str(run_id), "reviewer": reviewer, "scope": "perception_only",
             "training_authorized": False, "ontology_version": "1.0.0"},
            [n.output_artifact_id for n in run.stages if n.stage in
             (Stage.SEGMENTATION, Stage.MATCHING)],
            provenance=Provenance(type="human_approved", source=reviewer))
        return artifact.model_dump(mode="json")
