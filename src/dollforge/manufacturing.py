"""Non-destructive mesh inspection for additive prototyping.

Vertices are already baked into canonical world coordinates by reconstructors.
No repair, welding or transform is applied during inspection.
"""
from collections import Counter
from uuid import UUID

import numpy as np
import trimesh

from dollforge.contracts import Check, MeshCandidate, PartInspection
from dollforge.domain.models import PrintProfile


def inspect_meshes(
    candidates: list[MeshCandidate],
    unit: str,
    profile: PrintProfile,
    expected_part_ids: set[UUID] | None = None,
) -> tuple[list[Check], list[PartInspection]]:
    checks: list[Check] = []
    inspections: list[PartInspection] = []

    def check(code, passed, message, identifier=None, measurement=None):
        checks.append(Check(code=code, status="pass" if passed else "fail",
                            message=message, part_instance_id=identifier,
                            measurement=measurement))

    counts = Counter(c.part_instance_id for c in candidates)
    check("mesh_presence", bool(candidates), "A reconstrução deve conter peças.")
    check("unique_parts", all(count == 1 for count in counts.values()),
          "Cada identidade deve produzir exatamente uma malha.")
    if expected_part_ids is not None:
        missing = expected_part_ids - counts.keys()
        unexpected = counts.keys() - expected_part_ids
        check("part_coverage", not missing and not unexpected,
              f"Peças ausentes: {len(missing)}; inesperadas: {len(unexpected)}.")
        for identifier in sorted(missing, key=str):
            check("missing_part", False, "Peça identificada sem malha reconstruída.", identifier)
    check("physical_scale", unit == "mm",
          "Escala em milímetros informada; confirmar por medição física."
          if unit == "mm" else "Informe a altura real para obter escala em milímetros.")

    for candidate in candidates:
        identifier = candidate.part_instance_id
        inspection = PartInspection(part_instance_id=identifier, name=candidate.name,
                                    vertex_count=len(candidate.vertices),
                                    face_count=len(candidate.faces))
        inspections.append(inspection)
        vertices = np.asarray(candidate.vertices, dtype=np.float64)
        faces = np.asarray(candidate.faces, dtype=np.int64)
        valid = (vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 3
                 and faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
                 and np.isfinite(vertices).all()
                 and faces.min() >= 0 and faces.max() < len(vertices))
        check("mesh_data", bool(valid), "Vértices finitos e índices de faces válidos.", identifier)
        if not valid:
            continue
        within_budget = len(faces) <= profile.max_faces_per_part
        check("mesh_budget", within_budget,
              f"{len(faces)} faces; limite por peça: {profile.max_faces_per_part}.",
              identifier, float(len(faces)))
        if not within_budget:
            # Bound expensive topology work; an unchecked mesh cannot pass the gate.
            continue

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        unique = len(np.unique(np.sort(faces, axis=1), axis=0))
        check("duplicate_faces", unique == len(faces),
              f"Faces duplicadas: {len(faces) - unique}.", identifier, float(len(faces) - unique))
        check("nonzero_faces", bool(np.all(mesh.area_faces > 1e-12)),
              "Triângulos devem ter área maior que 1e-12 unidades².", identifier)
        check("manifold", bool(mesh.is_watertight),
              "Cada aresta deve pertencer a duas faces; não verifica autointerseções.", identifier)
        components = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1)
        inspection.shell_count = len(components)
        check("connected_shells", not profile.require_single_shell or len(components) == 1,
              f"{len(components)} componentes conectados por arestas.",
              identifier, float(len(components)))
        # A positive aggregate volume must not hide an inverted disconnected shell.
        outward = bool(mesh.is_winding_consistent and mesh.is_watertight)
        if outward:
            outward = all(trimesh.Trimesh(vertices, faces[component], process=False).volume > 0
                          for component in components)
        check("normals", outward, "Cada componente deve ter orientação externa e volume positivo.",
              identifier)
        if unit == "mm":
            # Only referenced vertices contribute to printed dimensions.
            used = vertices[np.unique(faces)]
            extents = np.ptp(used, axis=0)
            inspection.extents_mm = tuple(float(x) for x in extents)
            inspection.surface_area_mm2 = float(mesh.area)
            if outward and unique == len(faces):
                inspection.volume_mm3 = float(mesh.volume)
            if profile.build_volume_mm is not None:
                check("build_volume", bool(np.all(extents <= profile.build_volume_mm)),
                      "Dimensões XYZ na orientação atual: "
                      + " × ".join(f"{x:.3f}" for x in extents)
                      + " mm. Reorientação e suportes não avaliados.", identifier)
    if profile.build_volume_mm is None:
        checks.append(Check(code="build_volume", status="not_evaluated",
                            message="Informe o volume útil XYZ da impressora em milímetros."))
    return checks, inspections
