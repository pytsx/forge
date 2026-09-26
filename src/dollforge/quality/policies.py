from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from dollforge.contracts import (
    MatchingResult,
    MaskProposal,
    ReprojectionMetric,
    VolumetryResult,
)
from dollforge.domain.models import PartInstance, Side, Stage, ViewLabel
from dollforge.quality.models import LimitEvaluation, LimitMetric
from dollforge.vision.edges import boundary_adherence, sobel_edge_map


def _metric(
    code: str,
    value: float,
    threshold: float,
    direction: str,
    *,
    hard: bool = False,
    message: str = "",
) -> LimitMetric:
    passed = value >= threshold if direction == "gte" else value <= threshold
    return LimitMetric(
        code=code,
        value=float(value),
        threshold=float(threshold),
        direction=direction,
        passed=bool(passed),
        hard=hard,
        message=message,
    )


def _score(metrics: list[LimitMetric]) -> float:
    if not metrics:
        return 0.0
    normalized = []
    for metric in metrics:
        if metric.direction == "gte":
            value = metric.value / max(metric.threshold, 1e-8)
        else:
            value = metric.threshold / max(metric.value, metric.threshold, 1e-8)
        normalized.append(float(np.clip(value, 0.0, 1.0)))
    return float(np.mean(normalized))


def segmentation_limit(
    *,
    scope: str,
    view_label: ViewLabel,
    image_png: bytes,
    proposals: list[MaskProposal],
    boundary_threshold: float,
    confidence_threshold: float,
    coverage_threshold: float,
) -> LimitEvaluation:
    image = np.asarray(Image.open(BytesIO(image_png)).convert("RGB"))
    edges = sobel_edge_map(image)
    adherences: list[float] = []
    confidences: list[float] = []
    for proposal in proposals:
        mask = np.asarray(Image.open(BytesIO(proposal.mask_png)).convert("L")) > 127
        if mask.shape != image.shape[:2] or not mask.any():
            adherences.append(0.0)
        else:
            adherences.append(boundary_adherence(mask, edges))
        confidences.append(float(proposal.confidence))

    expected = 7 if view_label in (ViewLabel.FRONT, ViewLabel.BACK) else 5
    coverage = min(1.0, len(proposals) / expected)
    metrics = [
        _metric(
            "boundary_adherence",
            float(np.mean(adherences)) if adherences else 0.0,
            boundary_threshold,
            "gte",
            hard=True,
            message="A máscara precisa acompanhar bordas reais da imagem.",
        ),
        _metric(
            "proposal_confidence",
            float(np.mean(confidences)) if confidences else 0.0,
            confidence_threshold,
            "gte",
        ),
        _metric(
            "semantic_coverage",
            coverage,
            coverage_threshold,
            "gte",
        ),
    ]
    failures = [metric.code for metric in metrics if metric.hard and not metric.passed]
    return LimitEvaluation(
        stage=Stage.SEGMENTATION,
        scope=scope,
        score=_score(metrics),
        passed=all(metric.passed for metric in metrics),
        metrics=metrics,
        hard_failures=failures,
    )


def tune_segmentation(
    parameters: dict,
    evaluation: LimitEvaluation,
    attempt: int,
) -> dict:
    tuned = dict(parameters)
    failed = {metric.code for metric in evaluation.metrics if not metric.passed}
    if "boundary_adherence" in failed:
        tuned["edge_threshold"] = min(.88, float(tuned.get("edge_threshold", .62)) + .06)
        tuned["color_tolerance"] = max(.18, float(tuned.get("color_tolerance", .46)) - .05)
    if "semantic_coverage" in failed:
        tuned["foreground_threshold"] = max(
            8, int(tuned.get("foreground_threshold", 32)) - 4
        )
        tuned["box_threshold"] = max(.12, float(tuned.get("box_threshold", .28)) - .03)
        tuned["text_threshold"] = max(.10, float(tuned.get("text_threshold", .22)) - .02)
    if "proposal_confidence" in failed and "semantic_coverage" not in failed:
        tuned["box_threshold"] = min(.45, float(tuned.get("box_threshold", .28)) + .02)
    tuned["attempt_seed_offset"] = attempt
    return tuned


def _expected_views(part: PartInstance) -> int:
    if part.side == Side.CENTER:
        return 4
    if part.side in (Side.LEFT, Side.RIGHT):
        return 3
    return 2


def matching_limit(
    result: MatchingResult,
    *,
    confidence_threshold: float,
    coverage_threshold: float,
) -> LimitEvaluation:
    if result.parts:
        confidence = float(np.mean([part.confidence for part in result.parts]))
        coverage = float(np.mean([
            min(1.0, len(part.observation_ids) / _expected_views(part))
            for part in result.parts
        ]))
    else:
        confidence = coverage = 0.0
    metrics = [
        _metric("matching_confidence", confidence, confidence_threshold, "gte"),
        _metric("cross_view_coverage", coverage, coverage_threshold, "gte", hard=True),
    ]
    return LimitEvaluation(
        stage=Stage.MATCHING,
        scope="all",
        score=_score(metrics),
        passed=all(metric.passed for metric in metrics),
        metrics=metrics,
        hard_failures=[
            metric.code for metric in metrics if metric.hard and not metric.passed
        ],
    )


def tune_matching(
    parameters: dict,
    evaluation: LimitEvaluation,
    attempt: int,
) -> dict:
    tuned = dict(parameters)
    failed = {metric.code for metric in evaluation.metrics if not metric.passed}
    distance = float(tuned.get("max_distance", .42))
    if "cross_view_coverage" in failed:
        distance = min(.62, distance + .05)
    elif "matching_confidence" in failed:
        distance = max(.28, distance - .03)
    tuned["max_distance"] = distance
    tuned["attempt_seed_offset"] = attempt
    return tuned


def _reprojection_metrics(result: VolumetryResult) -> list[ReprojectionMetric]:
    return [
        metric
        for volume in result.volumes
        for metric in volume.reprojection_metrics
    ]


def volumetry_limit(
    result: VolumetryResult,
    *,
    iou_threshold: float,
    outside_area_threshold: float,
    overshoot_px_threshold: float,
) -> LimitEvaluation:
    reprojection = _reprojection_metrics(result)
    if reprojection:
        mean_iou = float(np.mean([metric.silhouette_iou for metric in reprojection]))
        max_outside = float(max(metric.outside_area_ratio for metric in reprojection))
        max_overshoot = float(max(metric.max_overshoot_px for metric in reprojection))
    else:
        mean_iou = 0.0
        max_outside = 1.0
        max_overshoot = float("inf")

    metrics = [
        _metric("mean_reprojection_iou", mean_iou, iou_threshold, "gte", hard=True),
        _metric(
            "outside_silhouette_error",
            max_outside,
            outside_area_threshold,
            "lte",
            hard=True,
            message="Nenhuma geração pode ultrapassar a borda além da tolerância.",
        ),
        _metric(
            "max_boundary_overshoot_px",
            max_overshoot,
            overshoot_px_threshold,
            "lte",
            hard=True,
        ),
    ]
    return LimitEvaluation(
        stage=Stage.VOLUMETRY,
        scope="all",
        score=_score(metrics),
        passed=all(metric.passed for metric in metrics),
        metrics=metrics,
        hard_failures=[
            metric.code for metric in metrics if metric.hard and not metric.passed
        ],
    )


def tune_volumetry(
    parameters: dict,
    evaluation: LimitEvaluation,
    attempt: int,
) -> dict:
    tuned = dict(parameters)
    failed = {metric.code for metric in evaluation.metrics if not metric.passed}
    resolution = int(tuned.get("resolution", 64))
    support = float(tuned.get("soft_support_threshold", .80))

    if "outside_silhouette_error" in failed or "max_boundary_overshoot_px" in failed:
        support = min(.95, support + .05)
        resolution = min(128, resolution + 16)
    elif "mean_reprojection_iou" in failed:
        support = max(.65, support - .05)
        resolution = min(128, resolution + 16)

    tuned["resolution"] = resolution
    tuned["soft_support_threshold"] = support
    tuned["attempt_seed_offset"] = attempt
    return tuned
