# SPDX-License-Identifier: Apache-2.0
"""Per-language cohorts (SPEC.md section 15).

Scenario results are grouped by ``(transcript.language, transcript.accent)``; cohorts are never
blended. ``cohort_key = "{language}/{accent or 'default'}"``.

- ``gap_vs_best``: the best cohort pass rate minus this cohort's; ``gap_warning`` when it is
  strictly greater than 0.15, both sides rounded to 6 decimals (so a float artefact such as
  ``1.0 - 0.85 = 0.15000000000000002`` is not a warning).
- ``mean_stt_confidence``: mean of the non-None ``stt_confidence`` values over every turn in the
  cohort, None when there are none; ``stt_warning`` when the best cohort mean minus this cohort's is
  strictly greater than 0.10 (same rounding).
- ``metric_means``: per metric ID, the mean score of the results that are neither skipped nor
  informational.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from impulse_evals.models import CohortResult, ScenarioResult

GAP_WARNING_THRESHOLD = 0.15
STT_WARNING_THRESHOLD = 0.10
ROUND_DECIMALS = 6


def cohort_key(language: str, accent: Optional[str]) -> str:
    """``"{language}/{accent or 'default'}"``."""
    return f"{language}/{accent or 'default'}"


def _strictly_greater(value: float, threshold: float) -> bool:
    return round(value, ROUND_DECIMALS) > round(threshold, ROUND_DECIMALS)


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _metric_means(results: list[ScenarioResult]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for result in results:
        for metric in result.metric_results:
            if metric.skipped or metric.informational:
                continue
            scores.setdefault(metric.metric_id, []).append(metric.score)
    return {metric_id: sum(values) / len(values) for metric_id, values in scores.items()}


def _mean_stt_confidence(results: list[ScenarioResult]) -> Optional[float]:
    return _mean(
        [
            turn.stt_confidence
            for result in results
            for turn in result.transcript.turns
            if turn.stt_confidence is not None
        ]
    )


def build_cohorts(scenario_results: Iterable[ScenarioResult]) -> list[CohortResult]:
    """One ``CohortResult`` per ``(language, accent)``, in first-seen order."""
    groups: dict[tuple[str, Optional[str]], list[ScenarioResult]] = {}
    for result in scenario_results:
        key = (result.transcript.language, result.transcript.accent)
        groups.setdefault(key, []).append(result)
    if not groups:
        return []

    stats = [_GroupStats(language, accent, results) for (language, accent), results in groups.items()]
    best_pass_rate = max(group.pass_rate for group in stats)
    confidences = [group.mean_stt_confidence for group in stats if group.mean_stt_confidence is not None]
    # "Best" for STT confidence is the highest cohort mean (cohorts without confidences take no part).
    best_confidence = max(confidences) if confidences else None

    cohorts: list[CohortResult] = []
    for group in stats:
        gap = best_pass_rate - group.pass_rate
        mean_confidence = group.mean_stt_confidence
        stt_warning = (
            best_confidence is not None
            and mean_confidence is not None
            and _strictly_greater(best_confidence - mean_confidence, STT_WARNING_THRESHOLD)
        )
        cohorts.append(
            CohortResult(
                cohort_key=cohort_key(group.language, group.accent),
                language=group.language,
                accent=group.accent,
                total=group.total,
                passed=group.passed,
                failed=group.total - group.passed,
                pass_rate=group.pass_rate,
                metric_means=_metric_means(group.results),
                mean_stt_confidence=mean_confidence,
                gap_vs_best=gap,
                gap_warning=_strictly_greater(gap, GAP_WARNING_THRESHOLD),
                stt_warning=stt_warning,
            )
        )
    return cohorts


class _GroupStats:
    """Counts of one ``(language, accent)`` group."""

    def __init__(self, language: str, accent: Optional[str], results: list[ScenarioResult]) -> None:
        self.language = language
        self.accent = accent
        self.results = results
        self.total = len(results)
        self.passed = sum(1 for result in results if result.passed)
        self.pass_rate = self.passed / self.total
        self.mean_stt_confidence = _mean_stt_confidence(results)
