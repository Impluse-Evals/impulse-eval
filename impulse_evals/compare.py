# SPDX-License-Identifier: AGPL-3.0-only
"""``impulse compare``: diff two stored runs (SPEC.md sections 7 and 15).

``compare_runs(run_a, run_b) -> CompareResult`` (run A is the baseline, run B the new run):

- Scenarios are matched by ``scenario_name``. A scenario in both runs gets a ``ScenarioDiff``; only
  the ``pass->fail`` ones are kept in ``regressions`` and the ``fail->pass`` ones in
  ``improvements``, while ``pass->pass`` / ``fail->fail`` are counted in ``unchanged_pass`` /
  ``unchanged_fail``. Scenarios in only one run are listed by name in ``added_scenarios`` (B only)
  or ``removed_scenarios`` (A only); no ``ScenarioDiff`` is built for them.
- ``metric_deltas``: ``score_b - score_a`` per metric ID present, non-skipped, and non-informational
  in both runs' results of the scenario (skipped and informational results are never in a delta).
- ``assertion_changes``: ``"{rule_id}: {result_a} -> {result_b}"`` for every assertion whose result
  changed (``missing`` for a rule the other run did not evaluate).
- ``notes``: ``"{id} newly failing"`` / ``"{id} newly passing"`` when a metric's label moves into or
  out of ``poor`` (from or to any other state: good, warning, skipped, or absent in that run);
  ``"{id} improved {d:.2f}"`` / ``"{id} worsened {d:.2f}"`` (``d`` = the size of the delta) when a
  ``metric_deltas`` entry is at least 0.1 in size. A metric informational in either run gets no note.
  A scenario that newly errors (``error`` None in A, set in B) gets the last note
  ``"scenario error: {error}"``.
- ``metric_summary_deltas``: ``score_b - score_a`` per metric ID in both runs' ``metric_summary``.
- ``critical_regressions``: names of scenarios in both runs where an assertion ``rule_id`` with
  severity ``critical`` is ``fail`` in B and not ``fail`` in A (first entry per rule ID), or whose ``error`` is None in A and
  set in B (a scenario that newly errors, including the critical no-verdict error of SPEC.md section
  13; the same condition makes ``impulse run --fail-on critical`` exit 1). An error in both runs, or
  only in A, is not a critical regression; added scenarios never count. Each name is listed once.
- ``check_regressions`` (FIXES-ROUND-2-SPEC.md A3 and A3-rev): the ``ScenarioDiff`` of every
  ``fail->fail`` scenario with at least one newly failing check, computed from the results (never
  from the note text): a metric counted and poor in B, not poor in A, and informational in neither
  run (the condition of the ``"{id} newly failing"`` note); an assertion whose result is ``fail`` in B
  and not ``fail`` in A (``missing`` in A included); or a scenario that newly errors in B (its diff
  ends with the ``"scenario error: ..."`` note, so the critical line always has a row explaining it).
  A scenario that errored in A (no results) makes its B failures newly failing.
- ``high_regressions``: names of scenarios in both runs (any status change) where an assertion with
  severity ``high`` is ``fail`` in B and not ``fail`` in A, or ``UNIV-025`` is poor (counted: not
  skipped, not informational) in B and not poor in A. Each name is listed once.

Order: scenarios in both runs and removed scenarios follow run A; added scenarios follow run B;
metrics and assertions follow run A, then run B. A name or ID repeated inside one run counts once
(its first entry).

``dry_run_notes(run_a, run_b) -> list[str]``: one ``DRY_RUN_NOTE`` sentence per dry run (run A
first). A run is a dry run when any of its metric or assertion results is skipped with the reason
exactly ``"dry run"`` (the runner's dry-run skip reason). ``CompareResult`` carries no such marker:
the CLI prints each sentence after the table (prefixed "Note: ") and the compare HTML shows it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, Optional

from impulse_evals.models import (
    CompareResult,
    MetricResult,
    RunResult,
    ScenarioDiff,
    ScenarioResult,
    ScoreResult,
)

__all__ = [
    "DRY_RUN_NOTE",
    "MISSING_RESULT",
    "NOTE_DELTA_THRESHOLD",
    "SCENARIO_ERROR_NOTE",
    "compare_runs",
    "diff_scenario",
    "dry_run_notes",
    "is_dry_run",
]

# A delta of at least this size gets an "improved" / "worsened" note (SPEC.md section 15).
NOTE_DELTA_THRESHOLD = 0.1
# Shown in ``assertion_changes`` for a rule that only one of the two runs evaluated.
MISSING_RESULT = "missing"
# The skip reason of every judge metric and assertion in a dry run (runner.DRY_RUN_REASON; repeated
# here so that comparing stored runs never imports the runner).
DRY_RUN_REASON = "dry run"
DRY_RUN_NOTE = "run {run_id} is a dry run: judge metrics and assertions were skipped, so its results are not verdicts."
# Prefix of the note of a scenario that newly errors in B: "scenario error: {error}".
SCENARIO_ERROR_NOTE = "scenario error: "
# Suffix of the metric note "{id} newly failing" (``_newly_poor_metrics`` is the same condition).
NEWLY_FAILING_SUFFIX = " newly failing"
# UNIV-025 (expected tool calls) newly poor is a high regression (FIXES-ROUND-2-SPEC.md A3); so is UNIV-019
# (harmful or misleading content) since fixes round 4 (O3). The same metrics count as severity high for
# ``impulse run --fail-on`` (``cli.fail_on_reasons``).
EXPECTED_TOOL_CALLS_METRIC = "UNIV-025"
HIGH_METRICS = ("UNIV-019", "UNIV-025")
# Deltas are compared after rounding, so 0.7 - 0.6 (0.0999...) still counts as 0.1.
_ROUND_DIGITS = 9

StatusChange = Literal["pass->fail", "fail->pass", "pass->pass", "fail->fail"]


def _status(passed: bool) -> str:
    return "pass" if passed else "fail"


def _first_by(items: Iterable, key) -> dict:
    """``{key(item): item}`` keeping the first item per key, in order."""
    out: dict = {}
    for item in items:
        out.setdefault(key(item), item)
    return out


def _counted(metric: MetricResult) -> bool:
    """Skipped and informational results are excluded from every delta (SPEC.md section 7)."""
    return not metric.skipped and not metric.informational


def _is_poor(metric: Optional[MetricResult]) -> bool:
    """Label ``poor`` on a counted result; absent, skipped, good, and warning results are not poor."""
    return metric is not None and _counted(metric) and metric.label == "poor"


def _metric_deltas(a: ScenarioResult, b: ScenarioResult) -> tuple[dict[str, float], list[str]]:
    """``(metric_deltas, notes)`` for one scenario present in both runs.

    Label notes cover every metric of either run: "newly failing" when the label moves into ``poor``
    from any other state (good, warning, skipped, or absent in A), "newly passing" when it moves out
    of ``poor`` (to good, warning, skipped, or absent in B); a metric informational in either run gets
    no note. Delta notes come only with a ``metric_deltas`` entry. Per metric: label note, then delta.
    """
    metrics_a = _first_by(a.metric_results, lambda m: m.metric_id)
    metrics_b = _first_by(b.metric_results, lambda m: m.metric_id)
    deltas: dict[str, float] = {}
    notes: list[str] = []
    for metric_id in list(metrics_a) + [metric_id for metric_id in metrics_b if metric_id not in metrics_a]:
        metric_a, metric_b = metrics_a.get(metric_id), metrics_b.get(metric_id)
        if any(metric is not None and metric.informational for metric in (metric_a, metric_b)):
            continue
        poor_a, poor_b = _is_poor(metric_a), _is_poor(metric_b)
        if poor_b and not poor_a:
            notes.append(f"{metric_id}{NEWLY_FAILING_SUFFIX}")
        elif poor_a and not poor_b:
            notes.append(f"{metric_id} newly passing")
        if metric_a is None or metric_b is None or not _counted(metric_a) or not _counted(metric_b):
            continue
        delta = metric_b.score - metric_a.score
        deltas[metric_id] = delta
        size = round(abs(delta), _ROUND_DIGITS)
        if size >= NOTE_DELTA_THRESHOLD:
            notes.append(f"{metric_id} {'improved' if delta > 0 else 'worsened'} {size:.2f}")
    return deltas, notes


def _assertion_changes(a: ScenarioResult, b: ScenarioResult) -> list[str]:
    results_a = _first_by(a.assertion_results, lambda s: s.rule_id)
    results_b = _first_by(b.assertion_results, lambda s: s.rule_id)
    changes: list[str] = []
    for rule_id in list(results_a) + [rule_id for rule_id in results_b if rule_id not in results_a]:
        score_a: Optional[ScoreResult] = results_a.get(rule_id)
        score_b: Optional[ScoreResult] = results_b.get(rule_id)
        result_a = score_a.result if score_a is not None else MISSING_RESULT
        result_b = score_b.result if score_b is not None else MISSING_RESULT
        if result_a != result_b:
            changes.append(f"{rule_id}: {result_a} -> {result_b}")
    return changes


def _newly_errors(a: ScenarioResult, b: ScenarioResult) -> bool:
    """The scenario has no ``error`` in A and one in B (e.g. the critical no-verdict error, SPEC.md
    section 13)."""
    return a.error is None and b.error is not None


def _newly_failing_assertions(a: ScenarioResult, b: ScenarioResult) -> list[ScoreResult]:
    """Assertions of B whose result is ``fail`` while the same ``rule_id`` is not ``fail`` in A (first
    entry per rule ID on both sides, as everywhere in compare; ``missing`` in A included)."""
    first_a = _first_by(a.assertion_results, lambda s: s.rule_id)
    newly_failing: list[ScoreResult] = []
    for rule_id, score in _first_by(b.assertion_results, lambda s: s.rule_id).items():
        score_a = first_a.get(rule_id)
        if score.result == "fail" and (score_a is None or score_a.result != "fail"):
            newly_failing.append(score)
    return newly_failing


def _newly_poor_metrics(a: ScenarioResult, b: ScenarioResult) -> list[str]:
    """IDs of metrics that are poor (counted) in B and not poor in A, informational in neither run
    (first entry per metric ID on both sides): the condition of the "{id} newly failing" note."""
    metrics_a = _first_by(a.metric_results, lambda m: m.metric_id)
    newly_poor: list[str] = []
    for metric_id, metric_b in _first_by(b.metric_results, lambda m: m.metric_id).items():
        metric_a = metrics_a.get(metric_id)
        if metric_b.informational or (metric_a is not None and metric_a.informational):
            continue
        if _is_poor(metric_b) and not _is_poor(metric_a):
            newly_poor.append(metric_id)
    return newly_poor


def _is_critical_regression(a: ScenarioResult, b: ScenarioResult) -> bool:
    """An assertion ``rule_id`` with severity critical is ``fail`` in B and not ``fail`` in A (first
    entry per rule ID on both sides), or the scenario newly errors in B."""
    if _newly_errors(a, b):
        return True
    return any(score.severity == "critical" for score in _newly_failing_assertions(a, b))


def _is_check_regression(diff: ScenarioDiff, a: ScenarioResult, b: ScenarioResult) -> bool:
    """A ``fail->fail`` scenario with a metric newly poor, an assertion newly ``fail``, or a new
    scenario error (FIXES-ROUND-2-SPEC.md A3-rev: computed from the results, not from the notes)."""
    if diff.status_change != "fail->fail":
        return False
    return _newly_errors(a, b) or bool(_newly_poor_metrics(a, b)) or bool(_newly_failing_assertions(a, b))


def _is_high_regression(a: ScenarioResult, b: ScenarioResult) -> bool:
    """A ``high`` assertion is ``fail`` in B and not in A, or a ``HIGH_METRICS`` metric (UNIV-019,
    UNIV-025) is newly poor (counted) in B."""
    if any(score.severity == "high" for score in _newly_failing_assertions(a, b)):
        return True
    metrics_a = _first_by(a.metric_results, lambda m: m.metric_id)
    metrics_b = _first_by(b.metric_results, lambda m: m.metric_id)
    return any(_is_poor(metrics_b.get(metric_id)) and not _is_poor(metrics_a.get(metric_id)) for metric_id in HIGH_METRICS)


def diff_scenario(a: ScenarioResult, b: ScenarioResult) -> ScenarioDiff:
    """The ``ScenarioDiff`` of one scenario present in both runs (``a`` from run A, ``b`` from run B)."""
    deltas, notes = _metric_deltas(a, b)
    if _newly_errors(a, b):
        notes.append(f"{SCENARIO_ERROR_NOTE}{b.error}")
    status_change: StatusChange = f"{_status(a.passed)}->{_status(b.passed)}"  # type: ignore[assignment]
    return ScenarioDiff(
        scenario_name=b.scenario_name,
        status_change=status_change,
        metric_deltas=deltas,
        assertion_changes=_assertion_changes(a, b),
        notes=notes,
    )


def _metric_summary_deltas(summary_a: dict[str, MetricResult], summary_b: dict[str, MetricResult]) -> dict[str, float]:
    return {
        metric_id: summary_b[metric_id].score - metric_a.score
        for metric_id, metric_a in summary_a.items()
        if metric_id in summary_b
    }


def _by_name(results: Sequence[ScenarioResult]) -> dict[str, ScenarioResult]:
    return _first_by(results, lambda result: result.scenario_name)


def is_dry_run(run: RunResult) -> bool:
    """True when any metric or assertion result of ``run`` is skipped with the reason exactly "dry run"."""
    return any(
        any(metric.skipped and metric.reason == DRY_RUN_REASON for metric in result.metric_results)
        or any(score.result == "skip" and score.reason == DRY_RUN_REASON for score in result.assertion_results)
        for result in run.scenario_results
    )


def dry_run_notes(run_a: RunResult, run_b: RunResult) -> list[str]:
    """One ``DRY_RUN_NOTE`` sentence per dry run of the two (run A first; without "Note: ")."""
    return [DRY_RUN_NOTE.format(run_id=run.run_id) for run in (run_a, run_b) if is_dry_run(run)]


def compare_runs(run_a: RunResult, run_b: RunResult) -> CompareResult:
    """Diff run B against run A (SPEC.md section 15). Neither run is modified."""
    scenarios_a = _by_name(run_a.scenario_results)
    scenarios_b = _by_name(run_b.scenario_results)

    regressions: list[ScenarioDiff] = []
    improvements: list[ScenarioDiff] = []
    unchanged_pass = unchanged_fail = 0
    critical_regressions: list[str] = []
    check_regressions: list[ScenarioDiff] = []
    high_regressions: list[str] = []
    for name, result_a in scenarios_a.items():
        result_b = scenarios_b.get(name)
        if result_b is None:
            continue
        diff = diff_scenario(result_a, result_b)
        if diff.status_change == "pass->fail":
            regressions.append(diff)
        elif diff.status_change == "fail->pass":
            improvements.append(diff)
        elif diff.status_change == "pass->pass":
            unchanged_pass += 1
        else:
            unchanged_fail += 1
        if _is_critical_regression(result_a, result_b):
            critical_regressions.append(name)
        if _is_check_regression(diff, result_a, result_b):
            check_regressions.append(diff)
        if _is_high_regression(result_a, result_b):
            high_regressions.append(name)

    return CompareResult(
        run_id_a=run_a.run_id,
        run_id_b=run_b.run_id,
        timestamp_a=run_a.timestamp,
        timestamp_b=run_b.timestamp,
        overall_pass_rate_delta=run_b.overall_pass_rate - run_a.overall_pass_rate,
        regressions=regressions,
        improvements=improvements,
        unchanged_pass=unchanged_pass,
        unchanged_fail=unchanged_fail,
        added_scenarios=[name for name in scenarios_b if name not in scenarios_a],
        removed_scenarios=[name for name in scenarios_a if name not in scenarios_b],
        metric_summary_deltas=_metric_summary_deltas(run_a.metric_summary, run_b.metric_summary),
        critical_regressions=critical_regressions,
        check_regressions=check_regressions,
        high_regressions=high_regressions,
    )
