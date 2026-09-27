from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import TypeVar

from dollforge.quality.models import LimitAttempt, LimitEvaluation, LimitTrace

T = TypeVar("T")


def run_quality_loop(
    *,
    stage,
    scope: str,
    specialist: str,
    initial_parameters: dict,
    max_attempts: int,
    generate: Callable[[dict], T],
    evaluate: Callable[[T], LimitEvaluation],
    tune: Callable[[dict, LimitEvaluation, int], dict],
    on_attempt: Callable[[LimitAttempt], None] | None = None,
) -> tuple[T, LimitTrace]:
    parameters = deepcopy(initial_parameters)
    attempts: list[LimitAttempt] = []
    best_value: T | None = None
    best_score = -1.0
    best_attempt = 1

    for attempt in range(1, max_attempts + 1):
        value = generate(parameters)
        evaluation = evaluate(value)
        attempts.append(LimitAttempt(
            attempt=attempt,
            parameters=deepcopy(parameters),
            evaluation=evaluation,
        ))
        if on_attempt is not None:
            on_attempt(attempts[-1])
        if evaluation.score > best_score:
            best_score = evaluation.score
            best_value = value
            best_attempt = attempt
        if evaluation.passed:
            return value, LimitTrace(
                stage=stage,
                scope=scope,
                specialist=specialist,
                max_attempts=max_attempts,
                attempts=attempts,
                accepted_attempt=attempt,
                status="passed",
                best_attempt=attempt,
                best_score=evaluation.score,
            )
        if attempt < max_attempts:
            parameters = tune(parameters, evaluation, attempt)

    assert best_value is not None
    return best_value, LimitTrace(
        stage=stage,
        scope=scope,
        specialist=specialist,
        max_attempts=max_attempts,
        attempts=attempts,
        accepted_attempt=None,
        status="retrain_candidate",
        best_attempt=best_attempt,
        best_score=max(0.0, best_score),
    )
