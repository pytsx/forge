from dollforge.adapters.grounded_sam2 import normalize_phrase
from dollforge.domain.models import PipelineConfig


def test_pipeline_accepts_grounded_sam2_adapter():
    config = PipelineConfig(segmentation_adapter="grounded_sam2_v1")
    assert config.segmentation_adapter == "grounded_sam2_v1"


def test_semantic_phrase_aliases_are_canonical():
    assert normalize_phrase("upper_arm_left") == "left upper arm"
    assert normalize_phrase("Hand-Right") == "right hand"
