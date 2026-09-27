from io import BytesIO

from PIL import Image

from dollforge.domain.models import CreateProject
from dollforge.service import Service, decode_image
from dollforge.storage import Store


def image_bytes(size=(512, 512)):
    image = Image.new("RGB", size, "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_add_view_preserves_original_and_replaces_project_pointer(tmp_path):
    service = Service(Store(tmp_path), engine=None)
    project = service.create_project(CreateProject(name="reviewable doll"))

    first = service.add_view(project.project_id, "front", image_bytes())
    second = service.add_view(project.project_id, "front", image_bytes((640, 640)))

    current = service.store.get("project", project.project_id, type(project))
    assert current.views == [second.view_id]
    assert service.store.read(first.original_artifact_id)
    assert service.store.read(second.original_artifact_id)
    assert first.view_id != second.view_id


def test_decode_image_normalizes_to_rgba():
    image = decode_image(image_bytes())
    assert image.mode == "RGBA"
    assert image.size == (512, 512)


def test_pipeline_does_not_block_progress_reads_or_execute_a_run_twice(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import pytest

    from dollforge.domain.models import JobStatus, PipelineConfig, RunManifest
    from dollforge.errors import Conflict

    started, release = Event(), Event()

    class SlowEngine:
        def execute(self, run, previous):
            started.set()
            assert release.wait(timeout=5)
            run.status = JobStatus.REVIEW
            store.save("run", run.run_id, run)
            return run

    store = Store(tmp_path)
    service = Service(store, SlowEngine())
    project = service.create_project(CreateProject(name="responsive"))
    run = RunManifest(project_id=project.project_id, project_snapshot=project, views=[],
                      config=PipelineConfig())
    store.save("run", run.run_id, run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(service.execute, run.run_id)
        try:
            assert started.wait(timeout=2)
            reading = pool.submit(store.get, "run", run.run_id, RunManifest)
            assert reading.result(timeout=1).status == JobStatus.RUNNING
        finally:
            release.set()
        assert running.result(timeout=2).status == JobStatus.REVIEW
    with pytest.raises(Conflict):
        service.execute(run.run_id)
