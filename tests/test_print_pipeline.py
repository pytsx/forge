from io import BytesIO

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from dollforge.bootstrap import create_service
from dollforge.contracts import ManufacturingReport, MaskProposal, MeshCandidate
from dollforge.domain.models import CreateProject, JobStatus, PipelineConfig, PrintProfile, Stage


@pytest.mark.parametrize("block", [False, True])
def test_real_pipeline_persists_inspection_and_gates_blender(tmp_path, block):
    service = create_service(tmp_path)
    provenance = {"type": "observed", "source": "synthetic_pipeline_test"}

    def png(image):
        buffer = BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    class Segmenter:
        model_id = "contour_rules_v2"
        model_version = "test"

        def predict(self, request):
            proposals = []
            for name, box in (("torso", (40, 10, 88, 60)), ("pelvis", (40, 60, 88, 110))):
                mask = Image.new("L", (128, 128))
                ImageDraw.Draw(mask).rectangle(box, fill=255)
                proposals.append(MaskProposal(part_class=name, side="center", bbox_xyxy=box,
                                               confidence=.99, provenance=provenance,
                                               mask_png=png(mask)))
            return proposals

    class Reconstructor:
        model_id = "silhouette_volume_mesh_v1"
        model_version = "test"

        def reconstruct(self, graph, observations, views, volumetry):
            mesh = trimesh.creation.box(extents=(10, 20, 30))
            return [MeshCandidate(part_instance_id=p.part_instance_id, name=str(p.part_class),
                                  vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist(),
                                  transform=np.eye(4).ravel().tolist(), confidence=.99,
                                  provenance=provenance) for p in graph.parts]

    class Blender:
        model_version = "test"
        calls = 0

        def build(self, candidates, unit):
            self.calls += 1
            return b"test only", "test"

    service.engine.segmenters[Segmenter.model_id] = Segmenter()
    service.engine.reconstructors[Reconstructor.model_id] = Reconstructor()
    blender = Blender()
    service.engine.blender = blender
    project = service.create_project(CreateProject(name="synthetic print pipeline", known_height_mm=100))
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).rectangle((40, 10, 88, 110), fill="black")
    for label in ("front", "back", "left", "right"):
        service.add_view(project.project_id, label, png(image))
    # Synthetic two-part case isolates S16; full segmentation quality is tested separately.
    config = PipelineConfig(build_blender=True, matching_adapter="semantic_side_matching_v1",
                            volumetry_resolution=32, segmentation_boundary_limit=0,
                            segmentation_coverage_limit=0, volumetry_iou_limit=.5,
                            volumetry_outside_area_limit=.2, volumetry_overshoot_px_limit=10,
                            print_profile=PrintProfile(max_voxel_size_mm=.0001 if block else 10,
                                                       build_volume_mm=(200, 200, 200)))
    run = service.execute(service.new_run(project.project_id, config).run_id)
    assert run.status == JobStatus.REVIEW, run.error
    stage = next(s for s in run.stages if s.stage == Stage.VALIDATION)
    report = ManufacturingReport.model_validate(service.store.json(stage.output_artifact_id))
    assert report.geometry_status == ("blocked" if block else "passed")
    assert report.print_profile == config.print_profile
    assert len(report.parts) == 2
    assert not report.manufacturable
    assert blender.calls == (0 if block else 1)
    assert bool(run.error) == block
    assert {service.store.metadata(i).stage for i in stage.input_artifact_ids} == {
        Stage.RECONSTRUCTION, Stage.VOLUMETRY, Stage.GRAPH, Stage.CALIBRATION,
    }
