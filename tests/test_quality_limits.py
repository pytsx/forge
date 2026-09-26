from uuid import uuid4

from dollforge.contracts import (
    ReprojectionMetric,
    VolumeCandidate,
    VolumeSlice,
    VolumetryResult,
)
from dollforge.domain.models import PipelineConfig, Provenance, Stage
from dollforge.quality.loop import run_quality_loop
from dollforge.quality.models import LimitEvaluation, LimitMetric
from dollforge.quality.policies import tune_volumetry, volumetry_limit


def evaluation(value: float, threshold: float = 2.0) -> LimitEvaluation:
    passed = value >= threshold
    metric = LimitMetric(
        code="synthetic_quality",
        value=value,
        threshold=threshold,
        direction="gte",
        passed=passed,
        hard=True,
    )
    return LimitEvaluation(
        stage=Stage.SEGMENTATION,
        scope="test",
        score=min(1.0, value / threshold),
        passed=passed,
        metrics=[metric],
        hard_failures=[] if passed else [metric.code],
    )


def test_quality_loop_uses_failed_result_to_tune_and_retry():
    value, trace = run_quality_loop(
        stage=Stage.SEGMENTATION,
        scope="test",
        specialist="synthetic_specialist",
        initial_parameters={"quality": 0},
        max_attempts=3,
        generate=lambda params: float(params["quality"]),
        evaluate=evaluation,
        tune=lambda params, result, attempt: {
            **params,
            "quality": int(params["quality"]) + 1,
            "seen_score": result.score,
            "attempt": attempt,
        },
    )

    assert value == 2
    assert trace.status == "passed"
    assert trace.accepted_attempt == 3
    assert [item.parameters["quality"] for item in trace.attempts] == [0, 1, 2]
    assert trace.attempts[1].parameters["seen_score"] == .0


def test_exhausted_loop_becomes_retraining_candidate_instead_of_silent_acceptance():
    value, trace = run_quality_loop(
        stage=Stage.SEGMENTATION,
        scope="test",
        specialist="synthetic_specialist",
        initial_parameters={"quality": 0},
        max_attempts=2,
        generate=lambda params: float(params["quality"]),
        evaluate=lambda value: evaluation(value, threshold=10.0),
        tune=lambda params, result, attempt: {
            **params,
            "quality": int(params["quality"]) + 1,
        },
    )

    assert value == 1
    assert trace.status == "retrain_candidate"
    assert trace.accepted_attempt is None
    assert len(trace.attempts) == 2


def volume_result(metric: ReprojectionMetric) -> VolumetryResult:
    slice_value = VolumeSlice(
        z_norm=.5,
        center_x_norm=0,
        half_width_norm=.4,
        center_y_norm=0,
        half_depth_norm=.3,
        confidence=.9,
    )
    volume = VolumeCandidate(
        part_instance_id=uuid4(),
        name="head",
        source_views=["front"],
        extents_xyz=(40, 35, 45),
        center_xyz=(0, 0, 0),
        slices=[slice_value, slice_value, slice_value, slice_value],
        vertices=[],
        faces=[],
        confidence=.9,
        reprojection_metrics=[metric],
        mean_reprojection_iou=metric.silhouette_iou,
        provenance=Provenance(type="derived_geometry", source="test"),
    )
    return VolumetryResult(
        volumes=[volume],
        unit="mm",
        method="test",
    )


def test_confirmed_boundary_overshoot_is_an_imperative_failure_and_changes_parameters():
    metric = ReprojectionMetric(
        view_id=uuid4(),
        view_label="front",
        silhouette_iou=.98,
        area_error_ratio=.03,
        outside_area_ratio=.031,
        max_overshoot_px=3.2,
        max_overshoot_mm=.8,
        boundary_rmse_px=.5,
        hard_constraint=True,
        hard_boundary_compliant=False,
    )
    result = volumetry_limit(
        volume_result(metric),
        iou_threshold=.95,
        outside_area_threshold=.02,
        overshoot_px_threshold=2.0,
    )

    assert not result.passed
    assert "outside_silhouette_error" in result.hard_failures
    assert "max_boundary_overshoot_px" in result.hard_failures

    tuned = tune_volumetry(
        {"resolution": 64, "soft_support_threshold": .80},
        result,
        attempt=1,
    )
    assert tuned["resolution"] == 80
    assert tuned["soft_support_threshold"] > .80


def test_quality_limit_is_not_disableable_in_normal_pipeline_config():
    config = PipelineConfig()
    assert config.quality_loop_enabled is True
    assert config.quality_fail_closed is True
    assert config.retrain_on_limit_exhaustion is True
    assert config.quality_max_attempts == 3
