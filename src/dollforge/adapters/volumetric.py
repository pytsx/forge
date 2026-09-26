from __future__ import annotations

import numpy as np
import trimesh

from dollforge.contracts import MeshCandidate, VolumetryResult
from dollforge.domain.models import DollGraph, ImageView, PartObservation, Provenance


class SilhouetteVolumeReconstructor:
    """Promote the volumetric hull surface into the reconstruction stage."""

    model_id = "silhouette_volume_mesh_v1"
    model_version = "1.0.0"

    def reconstruct(
        self,
        graph: DollGraph,
        observations: list[PartObservation],
        views: list[ImageView],
        volumetry: VolumetryResult | None = None,
    ) -> list[MeshCandidate]:
        if volumetry is None:
            return []

        output: list[MeshCandidate] = []
        part_ids = {part.part_instance_id for part in graph.parts}
        for volume in volumetry.volumes:
            if volume.part_instance_id not in part_ids:
                continue
            mesh = trimesh.Trimesh(
                vertices=np.asarray(volume.vertices, dtype=np.float64),
                faces=np.asarray(volume.faces, dtype=np.int64),
                process=True,
            )
            if mesh.volume < 0:
                mesh.invert()
            transform = np.eye(4)
            output.append(MeshCandidate(
                part_instance_id=volume.part_instance_id,
                name=volume.name,
                vertices=mesh.vertices.tolist(),
                faces=mesh.faces.tolist(),
                confidence=volume.confidence,
                transform=transform.flatten().tolist(),
                provenance=Provenance(
                    type="derived_geometry",
                    source=self.model_id,
                    evidence=volume.provenance.evidence,
                    note=(
                        "Superfície reconstruída do visual hull multi-view. A forma varia por seção "
                        "e segue as silhuetas; relevos internos ainda dependem de depth/normals."
                    ),
                ),
            ))
        return output
