import csv
import io
import zipfile
from types import SimpleNamespace

from fastapi.testclient import TestClient

from dollforge.api import create_app
from dollforge.contracts import Check, ManufacturingReport
from dollforge.domain.models import (
    CreateProject,
    JobStatus,
    PipelineConfig,
    Provenance,
    RunManifest,
    Stage,
    StageResult,
)
from dollforge.orchestration import lineage
from dollforge.service import Service
from dollforge.storage import Store


def test_export_includes_inspection_and_blocks_in_progress_runs(tmp_path):
    store = Store(tmp_path)
    service = Service(store, SimpleNamespace())
    project = service.create_project(CreateProject(name="export inspection"))
    run = RunManifest(project_id=project.project_id, project_snapshot=project, views=[],
                      status=JobStatus.REVIEW, config=PipelineConfig())
    report = ManufacturingReport(status="blocked", manufacturable=False,
                                 geometry_status="blocked", failed_checks=1,
                                 checks=[Check(code="physical_scale", status="fail",
                                               message="Escala relativa")])
    artifact = store.put(report.model_dump_json().encode(), project_id=project.project_id,
                         run_id=run.run_id, stage=Stage.VALIDATION, kind="inspection",
                         lineage=lineage([]), provenance=Provenance(type="rule_based", source="test"))
    run.stages.append(StageResult(node_id="S16:all", stage=Stage.VALIDATION,
                                  status=JobStatus.SUCCEEDED, input_artifact_ids=[],
                                  output_artifact_id=artifact.artifact_id, cache_key="test"))
    store.save("run", run.run_id, run)
    with TestClient(create_app(service)) as client:
        response = client.get(f"/api/runs/{run.run_id}/download")
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert ManufacturingReport.model_validate_json(archive.read("inspection.json")) == report
            rows = list(csv.DictReader(io.StringIO(archive.read("inspection.csv").decode("utf-8-sig"))))
            assert rows[0]["status"] == "fail"
            assert rows[0]["message"] == "Escala relativa"
        run.status = JobStatus.RUNNING
        store.save("run", run.run_id, run)
        assert client.get(f"/api/runs/{run.run_id}/download").status_code == 409
        # Exercise forward references and nested limits in the real API schema.
        assert client.get("/openapi.json").status_code == 200
