# SPDX-License-Identifier: Apache-2.0
"""Phase 6: per-language cohorts (SPEC.md section 15; cohort.build_cohorts).

Cohorts group scenario results by ``(transcript.language, transcript.accent)``; they are never
blended. Boundaries are strict after rounding both sides to 6 decimals: a gap of exactly 0.15 gives
no ``gap_warning``, 0.16 does; the same strict rule applies to ``stt_warning`` at 0.10.
"""

import pytest


def _build(scenario_results):
    from impulse_evals.cohort import build_cohorts

    return {cohort.cohort_key: cohort for cohort in build_cohorts(scenario_results)}


def _results(run_builder, language, accent, passed, failed, confidences=None, prefix=None):
    """``passed`` passing and ``failed`` failing results; ``confidences`` sets every turn's
    ``stt_confidence`` (a list is cycled over the turns of each result)."""
    prefix = prefix or f"{language}_{accent or 'default'}"
    out = []
    for index in range(passed + failed):
        values = confidences if isinstance(confidences, list) else [confidences]
        turns = [
            run_builder.turn("user", "Hello, I need an appointment.", 0, stt_confidence=values[0 % len(values)]),
            run_builder.turn("agent", "Sure, which day?", 900, stt_confidence=values[1 % len(values)]),
            run_builder.turn("user", "Tuesday.", 3000, stt_confidence=values[2 % len(values)]),
        ]
        name = f"{prefix}_{index}"
        transcript = run_builder.transcript(name, turns, language=language, accent=accent)
        score = 1.0 if index < passed else 0.0
        out.append(run_builder.scenario_result(name, transcript=transcript, metrics=[("UNIV-001", score)]))
    return out


def test_cohort_key_format_counts_and_fields(run_builder):
    results = _results(run_builder, "en-IN", None, 3, 1) + _results(run_builder, "hi-IN", "north-indian", 2, 2)
    cohorts = _build(results)
    assert set(cohorts) == {"en-IN/default", "hi-IN/north-indian"}
    english = cohorts["en-IN/default"]
    assert (english.language, english.accent) == ("en-IN", None)
    assert (english.total, english.passed, english.failed) == (4, 3, 1)
    assert english.pass_rate == pytest.approx(0.75)
    assert english.gap_vs_best == pytest.approx(0.0)
    hindi = cohorts["hi-IN/north-indian"]
    assert (hindi.language, hindi.accent) == ("hi-IN", "north-indian")
    assert (hindi.total, hindi.passed, hindi.failed) == (4, 2, 2)
    assert hindi.pass_rate == pytest.approx(0.5)
    assert hindi.gap_vs_best == pytest.approx(0.25)
    assert hindi.gap_warning is True
    assert english.gap_warning is False


def test_cohorts_are_never_blended(run_builder):
    results = (
        _results(run_builder, "hi-IN", "north-indian", 1, 0)
        + _results(run_builder, "hi-IN", "south-indian", 0, 1)
        + _results(run_builder, "hi-IN", None, 1, 1)
    )
    cohorts = _build(results)
    assert set(cohorts) == {"hi-IN/north-indian", "hi-IN/south-indian", "hi-IN/default"}
    assert cohorts["hi-IN/north-indian"].total == 1
    assert cohorts["hi-IN/south-indian"].total == 1
    assert cohorts["hi-IN/default"].total == 2


def test_gap_of_exactly_015_gives_no_warning(run_builder):
    # 1.0 - 0.85 is 0.15000000000000002 in floating point; rounded to 6 decimals it is 0.15.
    results = _results(run_builder, "en-IN", None, 20, 0) + _results(run_builder, "hi-IN", None, 17, 3)
    cohorts = _build(results)
    hindi = cohorts["hi-IN/default"]
    assert hindi.pass_rate == pytest.approx(0.85)
    assert hindi.gap_vs_best == pytest.approx(0.15)
    assert hindi.gap_warning is False


def test_gap_of_016_gives_a_warning(run_builder):
    results = _results(run_builder, "en-IN", None, 25, 0) + _results(run_builder, "hi-IN", None, 21, 4)
    cohorts = _build(results)
    hindi = cohorts["hi-IN/default"]
    assert hindi.pass_rate == pytest.approx(0.84)
    assert hindi.gap_vs_best == pytest.approx(0.16)
    assert hindi.gap_warning is True
    assert cohorts["en-IN/default"].gap_warning is False


def test_stt_gap_of_exactly_010_gives_no_warning(run_builder):
    # 0.8 - 0.7 is 0.10000000000000009 in floating point; rounded to 6 decimals it is 0.1.
    results = _results(run_builder, "en-IN", None, 10, 0, confidences=0.8) + _results(
        run_builder, "hi-IN", None, 9, 1, confidences=0.7
    )
    cohorts = _build(results)
    assert cohorts["en-IN/default"].mean_stt_confidence == pytest.approx(0.8)
    assert cohorts["hi-IN/default"].mean_stt_confidence == pytest.approx(0.7)
    assert cohorts["hi-IN/default"].stt_warning is False
    assert cohorts["en-IN/default"].stt_warning is False


def test_stt_gap_of_011_gives_a_warning(run_builder):
    results = _results(run_builder, "en-IN", None, 10, 0, confidences=0.81) + _results(
        run_builder, "hi-IN", None, 9, 1, confidences=0.7
    )
    cohorts = _build(results)
    assert cohorts["hi-IN/default"].stt_warning is True
    assert cohorts["en-IN/default"].stt_warning is False


def test_mean_stt_confidence_ignores_missing_values(run_builder):
    results = _results(run_builder, "en-IN", None, 2, 0, confidences=[0.9, None, 0.7])
    cohort = _build(results)["en-IN/default"]
    assert cohort.mean_stt_confidence == pytest.approx(0.8)


def test_mean_stt_confidence_none_without_confidences(run_builder):
    results = _results(run_builder, "en-IN", None, 2, 0, confidences=0.95) + _results(
        run_builder, "hi-IN", None, 1, 1, confidences=None
    )
    cohorts = _build(results)
    hindi = cohorts["hi-IN/default"]
    assert hindi.mean_stt_confidence is None
    assert hindi.stt_warning is False


def test_metric_means_exclude_skipped_and_informational(run_builder):
    first = run_builder.scenario_result(
        "first",
        metrics=[
            ("UNIV-001", 1.0),
            ("UNIV-003", 0.0, {"skipped": True}),
            ("UNIV-013", 1.0),
            ("VOICE-011", 0.0, {"informational": True, "label": "poor"}),
        ],
    )
    second = run_builder.scenario_result(
        "second",
        metrics=[
            ("UNIV-001", 0.0),
            ("UNIV-003", 0.0, {"skipped": True}),
            ("UNIV-013", 0.0, {"skipped": True}),
            ("VOICE-011", 1.0, {"informational": True}),
        ],
    )
    cohort = _build([first, second])["en-IN/default"]
    assert cohort.metric_means["UNIV-001"] == pytest.approx(0.5)
    assert cohort.metric_means["UNIV-013"] == pytest.approx(1.0)
    assert "UNIV-003" not in cohort.metric_means
    assert "VOICE-011" not in cohort.metric_means
    assert (cohort.total, cohort.passed, cohort.failed) == (2, 1, 1)
