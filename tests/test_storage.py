from uuid import uuid4

from dollforge.domain.models import Lineage, Provenance, Stage
from dollforge.storage import Store


def lineage(inputs=None):
    return Lineage(
        input_artifact_ids=inputs or [],
        code_version="test-code",
        model_version="test-model",
        parameters_hash="params",
        seed=42,
        environment_fingerprint="pytest",
    )


def provenance():
    return Provenance(type="rule_based", source="pytest")


def test_content_addressed_blobs_are_reused_but_metadata_is_versioned(tmp_path):
    store = Store(tmp_path)
    project_id = uuid4()
    run_id = uuid4()

    first = store.put(
        b"same payload",
        project_id=project_id,
        run_id=run_id,
        stage=Stage.INTAKE,
        kind="fixture",
        lineage=lineage(),
        provenance=provenance(),
        media_type="application/octet-stream",
    )
    second = store.put(
        b"same payload",
        project_id=project_id,
        run_id=run_id,
        stage=Stage.INTAKE,
        kind="fixture",
        lineage=lineage(),
        provenance=provenance(),
        media_type="application/octet-stream",
    )

    assert first.sha256 == second.sha256
    assert first.artifact_id != second.artifact_id
    assert first.version == 1
    assert second.version == 2
    assert store.read(first.artifact_id) == b"same payload"
    assert store.read(second.artifact_id) == b"same payload"


def test_artifact_lineage_requires_existing_parent_from_same_project(tmp_path):
    store = Store(tmp_path)
    project_id = uuid4()
    other_project_id = uuid4()
    run_id = uuid4()

    parent = store.put(
        b"parent",
        project_id=project_id,
        run_id=run_id,
        stage=Stage.INTAKE,
        kind="parent",
        lineage=lineage(),
        provenance=provenance(),
    )

    child = store.put(
        b"child",
        project_id=project_id,
        run_id=run_id,
        stage=Stage.QA,
        kind="child",
        lineage=lineage([parent.artifact_id]),
        provenance=provenance(),
    )
    assert child.lineage.input_artifact_ids == [parent.artifact_id]

    from dollforge.errors import Conflict

    try:
        store.put(
            b"invalid",
            project_id=other_project_id,
            run_id=run_id,
            stage=Stage.QA,
            kind="invalid",
            lineage=lineage([parent.artifact_id]),
            provenance=provenance(),
        )
    except Conflict:
        pass
    else:
        raise AssertionError("cross-project lineage must be rejected")
