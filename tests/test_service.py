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
