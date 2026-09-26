from uuid import uuid4

from pydantic import BaseModel

from dollforge.domain.models import (
    DollProject,
    PipelineConfig,
    Provenance,
    RunManifest,
    Stage,
)
from dollforge.orchestration import Engine, lineage
from dollforge.storage import Store


class Payload(BaseModel):
    value: int


class UnusedAdapter:
    model_id = "unused"
    model_version = "unused"


def test_node_cache_reuses_artifact_for_same_inputs_and_configuration(tmp_path):
    store = Store(tmp_path)
    project = DollProject(name="cache-test")
    run_id = uuid4()
    source = store.put(
        b"input",
        project_id=project.project_id,
        run_id=run_id,
        stage=Stage.INTAKE,
        kind="input",
        lineage=lineage([]),
        provenance=Provenance(type="observed", source="pytest"),
    )
    engine = Engine(store, UnusedAdapter(), UnusedAdapter(), UnusedAdapter())
    calls = {"count": 0}

    def execute():
        calls["count"] += 1
        return Payload(value=7)

    first_run = RunManifest(
        run_id=run_id,
        project_id=project.project_id,
        project_snapshot=project,
        views=[],
        config=PipelineConfig(build_blender=False),
    )
    first = engine.node(first_run, Stage.QA, "cache", [source.artifact_id], execute)

    second_run = RunManifest(
        project_id=project.project_id,
        project_snapshot=project,
        views=[],
        config=PipelineConfig(build_blender=False),
    )
    second = engine.node(second_run, Stage.QA, "cache", [source.artifact_id], execute)

    assert calls["count"] == 1
    assert first.artifact_id == second.artifact_id
    assert second_run.stages[-1].cached is True
