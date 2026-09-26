from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dollforge.bootstrap import create_service
from dollforge.domain.models import (
    DTO,
    CreateProject,
    DollProject,
    FeedbackEvent,
    ImageView,
    JobStatus,
    PipelineConfig,
    ReviewRequest,
    RunManifest,
    ViewLabel,
)
from dollforge.errors import Conflict, DomainError, InvalidInput, NotFound
from dollforge.service import MAX_UPLOAD, Service


class PromoteRequest(DTO):
    reviewer: str = Field(min_length=1, max_length=120)


def create_app(service: Service | None = None) -> FastAPI:
    service = service or create_service()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dollforge")

    @asynccontextmanager
    async def lifespan(app):
        # Preserve interrupted run history and make interruption explicit on restart.
        for run in service.store.list("run", RunManifest):
            if run.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                run.status = JobStatus.FAILED
                run.error = "Execução interrompida pelo encerramento do serviço. Execute novamente."
                service.store.save("run", run.run_id, run)
        yield
        executor.shutdown(wait=True)

    app = FastAPI(title="DollForge", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_origin(request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            if urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Origem não autorizada."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(DomainError)
    async def domain_error(request, exc):
        status = 404 if isinstance(exc, NotFound) else 409 if isinstance(exc, Conflict) else 422
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "local_only": True,
                "blender": service.engine.blender.model_version,
                "segmentation": service.engine.segmenter.model_id,
                "perception": service.engine.perception.model_id if service.engine.perception else "unavailable",
                "reconstruction": service.engine.reconstructor.model_id}

    @app.get("/api/projects")
    def projects() -> list[DollProject]:
        return service.store.list("project", DollProject)

    @app.post("/api/projects", status_code=201)
    def create_project(request: CreateProject) -> DollProject:
        return service.create_project(request)

    @app.get("/api/projects/{project_id}")
    def project(project_id: UUID) -> DollProject:
        return service.store.get("project", project_id, DollProject)

    @app.get("/api/projects/{project_id}/views")
    def views(project_id: UUID) -> list[ImageView]:
        project = service.store.get("project", project_id, DollProject)
        return [service.store.get("view", v, ImageView) for v in project.views]

    @app.post("/api/projects/{project_id}/views", status_code=201)
    async def add_view(project_id: UUID, label: ViewLabel = Form(), file: UploadFile = File()) -> ImageView:
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise InvalidInput("Imagem excede 20 MB.")
        return service.add_view(project_id, label, data)

    @app.get("/api/projects/{project_id}/runs")
    def runs(project_id: UUID) -> list[RunManifest]:
        service.store.get("project", project_id, DollProject)
        return [r for r in service.store.list("run", RunManifest) if r.project_id == project_id]

    @app.post("/api/projects/{project_id}/runs", status_code=202)
    def start_run(project_id: UUID, config: PipelineConfig) -> RunManifest:
        run = service.new_run(project_id, config)
        executor.submit(service.execute, run.run_id)
        return run

    @app.get("/api/runs/{run_id}")
    def run(run_id: UUID) -> RunManifest:
        return service.store.get("run", run_id, RunManifest)

    @app.post("/api/runs/{run_id}/replay", status_code=202)
    def replay(run_id: UUID) -> RunManifest:
        previous = service.store.get("run", run_id, RunManifest)
        run = service.new_run(previous.project_id, previous.config, run_id)
        executor.submit(service.execute, run.run_id)
        return run

    @app.post("/api/runs/{run_id}/reviews", status_code=201)
    def review(run_id: UUID, request: ReviewRequest) -> FeedbackEvent:
        return service.review(run_id, request)

    @app.get("/api/projects/{project_id}/feedback")
    def feedback(project_id: UUID) -> list[FeedbackEvent]:
        return [f for f in service.store.list("feedback", FeedbackEvent) if f.project_id == project_id]

    @app.post("/api/runs/{run_id}/promote")
    def promote(run_id: UUID, request: PromoteRequest):
        return service.promote(run_id, request.reviewer)

    @app.get("/api/artifacts/{artifact_id}")
    def artifact_metadata(artifact_id: UUID):
        return service.store.metadata(artifact_id)

    @app.get("/api/artifacts/{artifact_id}/content")
    def artifact_content(artifact_id: UUID):
        metadata = service.store.metadata(artifact_id)
        return Response(service.store.read(artifact_id), media_type=metadata.media_type)

    @app.get("/api/runs/{run_id}/download")
    def download(run_id: UUID):
        run = service.store.get("run", run_id, RunManifest)
        if any(s.invalidated for s in run.stages):
            raise Conflict("Reprocesse as alterações antes de exportar este run.")
        identifiers = set()
        def collect(identifier):
            if identifier in identifiers:
                return
            identifiers.add(identifier)
            meta = service.store.metadata(identifier)
            for parent in meta.lineage.input_artifact_ids:
                collect(parent)
            if meta.media_type == "application/json":
                def walk(value):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if key.endswith("artifact_id") and isinstance(item, str):
                                collect(UUID(item))
                            else:
                                walk(item)
                    elif isinstance(value, list):
                        for item in value:
                            walk(item)
                walk(service.store.json(identifier))
        for stage in run.stages:
            if stage.output_artifact_id:
                collect(stage.output_artifact_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", run.model_dump_json(indent=2))
            for identifier in sorted(identifiers, key=str):
                meta = service.store.metadata(identifier)
                extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp",
                    "application/x-blender": "blend", "model/stl": "stl"}.get(meta.media_type, "json")
                archive.writestr(f"artifacts/{identifier}.{extension}", service.store.read(identifier))
                archive.writestr(f"metadata/{identifier}.json", meta.model_dump_json(indent=2))
            archive.writestr("README.txt", "DollForge: baseline supervisionado. Não validado para fabricação.\n"
                "manifest.json registra inputs, revisões e parâmetros. Artefatos preservam lineage e hashes.")
        return Response(buffer.getvalue(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="dollforge-{str(run_id)[:8]}.zip"'})

    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html")

    return app
