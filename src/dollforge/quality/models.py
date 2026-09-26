from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, JsonValue

from dollforge.domain.models import DTO, Score, Stage


class LimitMetric(DTO):
    code: str = Field(min_length=1)
    value: float
    threshold: float
    direction: Literal["gte", "lte"]
    passed: bool
    hard: bool = False
    message: str = ""


class LimitEvaluation(DTO):
    stage: Stage
    scope: str
    score: Score
    passed: bool
    metrics: list[LimitMetric]
    hard_failures: list[str] = Field(default_factory=list)


class LimitAttempt(DTO):
    attempt: int = Field(ge=1)
    parameters: dict[str, JsonValue]
    evaluation: LimitEvaluation
    output_artifact_id: UUID | None = None


class LimitTrace(DTO):
    stage: Stage
    scope: str
    specialist: str
    max_attempts: int = Field(ge=1)
    attempts: list[LimitAttempt]
    accepted_attempt: int | None = None
    status: Literal["passed", "retrain_candidate"]
    best_attempt: int
    best_score: Score


class TrainingSignal(DTO):
    stage: Stage
    scope: str
    specialist: str
    reason: Literal["quality_limit_exhausted"] = "quality_limit_exhausted"
    input_artifact_ids: list[UUID]
    trace_artifact_id: UUID
    best_attempt: int
    best_score: Score
    best_parameters: dict[str, JsonValue]
    failed_metrics: list[str]
    training_authorized: Literal[False] = False
