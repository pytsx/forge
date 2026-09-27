from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from dollforge.contracts import ManufacturingReport, MeshCandidate
from dollforge.domain.models import PrintProfile, Provenance
from dollforge.manufacturing import inspect_meshes
from dollforge.orchestration import Engine
from dollforge.storage import Store


def candidate(mesh=None):
    mesh = trimesh.creation.box(extents=(10, 20, 30)) if mesh is None else mesh
    return MeshCandidate(part_instance_id=uuid4(), name="test_part",
                         vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist(),
                         transform=np.eye(4).ravel().tolist(), confidence=.9,
                         provenance=Provenance(type="derived_geometry", source="test"))


def statuses(checks):
    return {c.code: c.status for c in checks}


def test_box_measurements_are_physical_and_inspection_does_not_mutate_geometry():
    part = candidate()
    original = part.model_dump()
    checks, parts = inspect_meshes([part], "mm", PrintProfile(build_volume_mm=(10, 20, 30)),
                                   {part.part_instance_id})
    assert all(c.status == "pass" for c in checks)
    assert parts[0].extents_mm == (10, 20, 30)
    assert parts[0].volume_mm3 == pytest.approx(6000)
    assert parts[0].surface_area_mm2 == pytest.approx(2200)
    assert parts[0].shell_count == 1
    assert part.model_dump() == original


@pytest.mark.parametrize("damage,code", [
    ("open", "manifold"), ("inverted", "normals"),
    ("duplicate", "duplicate_faces"), ("degenerate", "nonzero_faces"),
    ("bad_index", "mesh_data"), ("negative_index", "mesh_data"), ("empty", "mesh_data"),
])
def test_invalid_geometry_fails_without_repair(damage, code):
    part = candidate()
    if damage == "open":
        part.faces.pop()
    elif damage == "inverted":
        part.faces = [tuple(reversed(face)) for face in part.faces]
    elif damage == "duplicate":
        part.faces.append(part.faces[0])
    elif damage == "degenerate":
        part.faces.append((0, 0, 0))
    elif damage == "bad_index":
        part.faces[0] = (0, 1, len(part.vertices))
    elif damage == "negative_index":
        part.faces[0] = (-1, 0, 1)
    else:
        part.faces = []
    checks, _ = inspect_meshes([part], "mm", PrintProfile())
    assert statuses(checks)[code] == "fail"


def test_inverted_shell_cannot_hide_behind_positive_total_volume():
    large = trimesh.creation.box(extents=(10, 10, 10))
    small = trimesh.creation.box()
    small.apply_translation((20, 0, 0))
    small.invert()
    combined = trimesh.util.concatenate([large, small])
    assert combined.volume > 0
    checks, parts = inspect_meshes([candidate(combined)], "mm", PrintProfile())
    assert statuses(checks)["normals"] == "fail"
    assert statuses(checks)["connected_shells"] == "fail"
    assert parts[0].volume_mm3 is None


def test_empty_missing_and_duplicate_parts_cannot_pass():
    part = candidate()
    checks, _ = inspect_meshes([], "mm", PrintProfile(), {part.part_instance_id})
    assert statuses(checks)["mesh_presence"] == "fail"
    assert statuses(checks)["missing_part"] == "fail"
    checks, _ = inspect_meshes([part, part], "mm", PrintProfile())
    assert statuses(checks)["unique_parts"] == "fail"


def test_relative_scale_has_no_physical_measurements_and_oversize_fails():
    part = candidate()
    checks, parts = inspect_meshes([part], "relative", PrintProfile())
    assert statuses(checks)["physical_scale"] == "fail"
    assert parts[0].extents_mm is None
    assert parts[0].volume_mm3 is None
    checks, _ = inspect_meshes([part], "mm", PrintProfile(build_volume_mm=(9, 20, 30)))
    assert statuses(checks)["build_volume"] == "fail"


def test_complexity_budget_stops_expensive_analysis():
    checks, parts = inspect_meshes([candidate()], "mm", PrintProfile(max_faces_per_part=4))
    assert statuses(checks)["mesh_budget"] == "fail"
    assert parts[0].shell_count is None


@pytest.mark.parametrize("limits", [
    {"max_voxel_size_mm": 0}, {"max_voxel_size_mm": float("nan")},
    {"build_volume_mm": (200, -1, 200)}, {"build_volume_mm": (200, 200)},
])
def test_profile_rejects_invalid_limits(limits):
    with pytest.raises(ValidationError):
        PrintProfile(**limits)


def test_report_distinguishes_geometry_from_manufacturing_release(tmp_path):
    unused = SimpleNamespace(model_id="unused")
    engine = Engine(Store(tmp_path), unused, unused, unused)
    part = candidate()
    report = engine.validate([part], "mm")
    assert report.geometry_status == "passed"
    assert report.status == "needs_review"
    assert not report.manufacturable
    assert report.pending_checks > 0
    assert ManufacturingReport.model_validate_json(report.model_dump_json()) == report
    report = engine.validate([part], "mm", profile=PrintProfile(max_voxel_size_mm=.2))
    assert report.geometry_status == "blocked"
    assert statuses(report.checks)["voxel_resolution"] == "fail"


@pytest.mark.parametrize("voxel,expected", [(.1, "pass"), (.2, "pass"), (.21, "fail")])
def test_voxel_limit_uses_evidence_for_each_part(tmp_path, voxel, expected):
    unused = SimpleNamespace(model_id="unused")
    engine = Engine(Store(tmp_path), unused, unused, unused)
    part = candidate()
    volume = SimpleNamespace(part_instance_id=part.part_instance_id,
                             field=SimpleNamespace(voxel_size_mm=voxel),
                             reprojection_metrics=[])
    report = engine.validate([part], "mm", volumetry=SimpleNamespace(volumes=[volume]),
                             profile=PrintProfile(max_voxel_size_mm=.2))
    assert statuses(report.checks)["voxel_resolution"] == expected
