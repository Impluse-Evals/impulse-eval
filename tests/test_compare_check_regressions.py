# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 2, A3 (DEFECT-4): ``impulse compare`` reports fail -> fail check regressions and can
gate on high (FIXES-ROUND-2-SPEC.md A3; SPEC.md sections 7, 15, 16, 17).

* ``CompareResult.check_regressions: list[ScenarioDiff] = []``: fail -> fail scenarios (in both runs)
  with at least one newly failing check: a metric note ``"{id} newly failing"``, or an assertion whose
  result is ``fail`` in B and not ``fail`` in A.
* ``CompareResult.high_regressions: list[str] = []``: names of scenarios in both runs (any status
  change) where a ``high`` assertion is ``fail`` in B and not ``fail`` in A, or UNIV-025 is poor
  (counted: not skipped, not informational) in B and not poor in A; each name once, run A order.
* ``impulse compare RUN_A RUN_B [--fail-on critical|high]`` (default ``critical``; config.toml is not
  consulted): ``high`` also exits 1 when ``high_regressions`` is non-empty.
* Terminal: after the improvements block, ``Newly failing checks (fail -> fail): {n}`` and one line
  per diff (regression layout); after the critical line, ``High regressions: {names}`` when non-empty.
* Compare HTML: a "Newly failing checks (fail -> fail)" table after the improvements table and a
  banner "High regressions: {names}" when non-empty.

Campaign data (trimmed copies, transcripts emptied; ``tests/data/fixes_round2/compare/``): SupportBot
text runs 20261023T193004Z-a9be47 (A) and 20261023T193530Z-784ac4 (B). Every scenario is fail ->
fail or pass -> pass, no assertion is evaluated, and UNIV-025 goes good -> poor in
``refund_outside_policy`` and ``damaged_item_ticket``. ``order_status`` is a check regression too
(UNIV-001 and UNIV-009 go good -> poor), which the spec does not name but its definition covers.

Ordering choice (simplest reading, the order every other compare list uses): ``check_regressions``
follows run A, like ``regressions``.
"""

import html as html_lib
import json
import re
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data" / "fixes_round2" / "compare"
CAMPAIGN_A = "20261023T193004Z-a9be47"
CAMPAIGN_B = "20261023T193530Z-784ac4"
CAMPAIGN_CHECK_REGRESSIONS = ["order_status", "refund_outside_policy", "damaged_item_ticket"]
CAMPAIGN_HIGH_REGRESSIONS = ["refund_outside_policy", "damaged_item_ticket"]

CONSENT = "dpdp.consent_before_pii"     # critical (conftest DPDP_RULE_META)
PURPOSE = "dpdp.purpose_disclosure"     # high
WITHDRAWAL = "dpdp.withdrawal_path"     # medium

RUN_A = "20261011T120000Z-a3a3a3"
RUN_B = "20261012T120000Z-b3b3b3"
EXTERNAL = re.compile(r"""(?:\b(?:src|href)\s*=\s*["']?\s*|url\(\s*["']?\s*)(?:https?:|//)""", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _compare(run_a, run_b):
    from impulse_evals.compare import compare_runs

    return compare_runs(run_a, run_b)


def _campaign_run(run_id):
    from impulse_evals.models import RunResult

    return RunResult.model_validate_json((DATA / f"supportbot_text_{run_id}.json").read_text(encoding="utf-8"))


def _campaign():
    return _campaign_run(CAMPAIGN_A), _campaign_run(CAMPAIGN_B)


def _names(diffs):
    return [diff.scenario_name for diff in diffs]


def _diffs(diffs):
    return {diff.scenario_name: diff for diff in diffs}


def _page_text(page):
    """The HTML page as text: tags removed (no space added, so ``<code>a</code>, <code>b</code>`` reads
    "a, b"), entities unescaped, whitespace collapsed."""
    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", "", page)).split())


def _block(text, start, end):
    """The part of ``text`` from ``start`` up to the next ``end`` after it."""
    begin = text.index(start)
    return text[begin : text.index(end, begin)]


@pytest.fixture
def campaign_runs(impulse_project, run_builder):
    """The two campaign runs stored under ``.impulse/runs/<run_id>/run.json`` of a temporary project."""
    run_a, run_b = _campaign()
    run_builder.write_run(run_a, impulse_project.runs_dir)
    run_builder.write_run(run_b, impulse_project.runs_dir)
    return run_a, run_b


def _outputs(project, run_id_a, run_id_b):
    stem = f"compare_{run_id_a}_vs_{run_id_b}"
    return project.runs_dir / f"{stem}.json", project.runs_dir / f"{stem}.html"


# =============================================================================================
# Campaign data (SupportBot text, A = 193004Z-a9be47, B = 193530Z-784ac4)
# =============================================================================================


def test_campaign_fixture_matches_the_defect_report():
    # Regression guard on the fixture itself (passes before the fix too): the shape DEFECT-4 describes.
    run_a, run_b = _campaign()
    by_a = {s.scenario_name: s for s in run_a.scenario_results}
    by_b = {s.scenario_name: s for s in run_b.scenario_results}
    assert list(by_a) == list(by_b)
    assert [s.passed for s in run_a.scenario_results] == [s.passed for s in run_b.scenario_results]
    for name in CAMPAIGN_HIGH_REGRESSIONS:
        univ025_a = {m.metric_id: m for m in by_a[name].metric_results}["UNIV-025"]
        univ025_b = {m.metric_id: m for m in by_b[name].metric_results}["UNIV-025"]
        assert (univ025_a.label, univ025_b.label) == ("good", "poor")
        assert not univ025_b.skipped and not univ025_b.informational
    assert all(not s.assertion_results for s in run_a.scenario_results + run_b.scenario_results)
    assert all(s.error is None for s in run_a.scenario_results + run_b.scenario_results)


def test_campaign_compare_lists_fail_to_fail_check_regressions():
    result = _compare(*_campaign())
    assert result.regressions == [] and result.improvements == []
    assert result.critical_regressions == []
    assert _names(result.check_regressions) == CAMPAIGN_CHECK_REGRESSIONS
    diffs = _diffs(result.check_regressions)
    for name in ("refund_outside_policy", "damaged_item_ticket"):
        assert diffs[name].status_change == "fail->fail"
        assert "UNIV-025 newly failing" in diffs[name].notes
    assert diffs["damaged_item_ticket"].notes == ["UNIV-025 newly failing", "UNIV-025 worsened 1.00"]
    assert diffs["order_status"].notes[0] == "UNIV-001 newly failing"
    assert "UNIV-009 newly failing" in diffs["order_status"].notes
    # angry_wants_human only gets better (UNIV-001 and UNIV-023 newly passing): not a check regression.
    assert "angry_wants_human" not in _names(result.check_regressions)


def test_campaign_compare_lists_univ025_high_regressions():
    result = _compare(*_campaign())
    assert result.high_regressions == CAMPAIGN_HIGH_REGRESSIONS


def test_campaign_compare_counts_are_unchanged():
    # Regression guard (passes before the fix too): fail -> fail scenarios stay in unchanged_fail.
    result = _compare(*_campaign())
    assert (result.unchanged_pass, result.unchanged_fail) == (2, 5)


def test_campaign_compare_swapped_has_no_high_regression():
    # B -> A: UNIV-025 goes poor -> good; angry_wants_human gets UNIV-001 and UNIV-023 newly failing.
    result = _compare(_campaign_run(CAMPAIGN_B), _campaign_run(CAMPAIGN_A))
    assert result.high_regressions == []
    assert _names(result.check_regressions) == ["angry_wants_human"]


def test_compare_result_new_fields_round_trip():
    from impulse_evals.models import CompareResult

    result = _compare(*_campaign())
    assert CompareResult.model_validate_json(result.model_dump_json()) == result


def test_older_compare_json_without_the_new_fields_still_loads():
    from impulse_evals.models import CompareResult

    data = json.loads(_compare(*_campaign()).model_dump_json())
    data.pop("check_regressions")
    data.pop("high_regressions")
    old = CompareResult.model_validate(data)
    assert old.check_regressions == []
    assert old.high_regressions == []


# ---------------------------------------------------------------------------------------------
# CLI on the campaign runs
# ---------------------------------------------------------------------------------------------


def test_cli_campaign_default_exits_0(impulse_project, campaign_runs, cli):
    result = cli.invoke("compare", CAMPAIGN_A, CAMPAIGN_B)
    assert result.exit_code == 0, cli.text(result)
    assert "No critical regressions." in cli.text(result)


def test_cli_campaign_fail_on_critical_exits_0(impulse_project, campaign_runs, cli):
    result = cli.invoke("compare", CAMPAIGN_A, CAMPAIGN_B, "--fail-on", "critical")
    assert result.exit_code == 0, cli.text(result)


def test_cli_campaign_fail_on_high_exits_1(impulse_project, campaign_runs, cli):
    result = cli.invoke("compare", CAMPAIGN_A, CAMPAIGN_B, "--fail-on", "high")
    assert result.exit_code == 1, cli.text(result)


def test_cli_compare_does_not_read_fail_on_from_config_toml(impulse_project, campaign_runs, cli):
    impulse_project.write_toml('[defaults]\nfail_on = "high"\n')
    result = cli.invoke("compare", CAMPAIGN_A, CAMPAIGN_B)
    assert result.exit_code == 0, cli.text(result)


def test_cli_campaign_terminal_blocks(impulse_project, campaign_runs, cli):
    result = cli.invoke("compare", CAMPAIGN_A, CAMPAIGN_B)
    text = cli.text(result)
    title = "Newly failing checks (fail -> fail): 3"
    assert title in text, text
    assert text.index("Improvements (fail -> pass): 0") < text.index(title) < text.index("Unchanged:")
    block = _block(text, title, "Unchanged:")
    assert "damaged_item_ticket UNIV-025 newly failing; UNIV-025 worsened 1.00" in block
    assert "refund_outside_policy" in block and "order_status" in block
    assert "UNIV-025 newly failing" in _block(block, "refund_outside_policy", "damaged_item_ticket")
    assert "CRITICAL" not in block   # no critical regression in these runs
    high_line = "High regressions: refund_outside_policy, damaged_item_ticket"
    assert high_line in text
    assert text.index("No critical regressions.") < text.index(high_line)


def test_cli_campaign_compare_json_has_the_new_fields(impulse_project, campaign_runs, cli):
    from impulse_evals.models import CompareResult

    run_a, run_b = campaign_runs
    cli.invoke("compare", CAMPAIGN_A, CAMPAIGN_B)
    json_path, _ = _outputs(impulse_project, CAMPAIGN_A, CAMPAIGN_B)
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert [diff["scenario_name"] for diff in raw["check_regressions"]] == CAMPAIGN_CHECK_REGRESSIONS
    assert raw["high_regressions"] == CAMPAIGN_HIGH_REGRESSIONS
    assert CompareResult.model_validate(raw) == _compare(run_a, run_b)


def test_cli_campaign_compare_html_table_and_banner(impulse_project, campaign_runs, cli):
    cli.invoke("compare", CAMPAIGN_A, CAMPAIGN_B)
    _, html_path = _outputs(impulse_project, CAMPAIGN_A, CAMPAIGN_B)
    page = html_path.read_text(encoding="utf-8")
    text = _page_text(page)
    heading = "Newly failing checks (fail -> fail)"
    assert heading in text
    assert text.index("Improvements (fail -> pass)") < text.index(heading)
    table = text[text.index(heading):]
    for name in CAMPAIGN_CHECK_REGRESSIONS:
        assert name in table, name
    assert "fail->fail" in table
    assert "UNIV-025 newly failing" in table
    for column in ("scenario_name", "status_change", "assertion_changes", "metric_deltas", "notes"):
        assert column in table, column
    assert "High regressions: refund_outside_policy, damaged_item_ticket" in text
    assert "<script" not in page.lower()
    assert not EXTERNAL.search(page)


# =============================================================================================
# Hand-built edge cases
# =============================================================================================


def _edge_run_a(b):
    results = [
        # fail -> fail; a high assertion goes pass -> fail inside it
        b.scenario_result("ff_high_assertion", assertions=[(PURPOSE, "pass")], metrics=[("UNIV-001", 0.0)]),
        # fail -> fail; a medium assertion goes skip -> fail (check regression, not high)
        b.scenario_result("ff_medium_assertion", assertions=[(WITHDRAWAL, "skip")], metrics=[("UNIV-001", 0.0)]),
        # fail -> fail; the critical rule goes skip -> fail (critical regression, not high)
        b.scenario_result("ff_critical_assertion", assertions=[(CONSENT, "skip")], metrics=[("UNIV-001", 0.0)]),
        # fail -> fail; nothing newly fails (an assertion fail -> fail, a metric poor -> poor)
        b.scenario_result("ff_steady", assertions=[(PURPOSE, "fail")], metrics=[("UNIV-001", 0.0)]),
        # fail -> fail; only improvements inside (UNIV-013 newly passing)
        b.scenario_result("ff_better", metrics=[("UNIV-001", 0.0), ("UNIV-013", 0.0)]),
        # fail -> fail; UNIV-025 absent in A, poor in B
        b.scenario_result("ff_univ025_new", metrics=[("UNIV-001", 0.0)]),
        # fail -> fail; UNIV-025 poor in both runs (not newly poor)
        b.scenario_result("ff_univ025_still", metrics=[("UNIV-025", 0.0)]),
        # pass -> pass; UNIV-025 skipped in B
        b.scenario_result("pp_univ025_skipped", metrics=[("UNIV-025", 1.0)]),
        # pass -> pass; UNIV-025 informational and poor in B
        b.scenario_result("pp_univ025_informational", metrics=[("UNIV-025", 1.0)]),
        # pass -> fail; high assertion newly fails AND UNIV-025 newly poor (listed once in high_regressions)
        b.scenario_result("pf_high_twice", assertions=[(PURPOSE, "pass")], metrics=[("UNIV-025", 1.0)]),
        # pass -> fail on a metric only (a regression, never a check regression)
        b.scenario_result("pf_metric", metrics=[("UNIV-013", 1.0)]),
    ]
    return b.run(results, run_id=RUN_A, timestamp="2026-10-11T12:00:00Z")


def _edge_run_b(b):
    results = [
        b.scenario_result("ff_high_assertion", assertions=[(PURPOSE, "fail")], metrics=[("UNIV-001", 0.0)]),
        b.scenario_result("ff_medium_assertion", assertions=[(WITHDRAWAL, "fail")], metrics=[("UNIV-001", 0.0)]),
        b.scenario_result("ff_critical_assertion", assertions=[(CONSENT, "fail")], metrics=[("UNIV-001", 0.0)]),
        b.scenario_result("ff_steady", assertions=[(PURPOSE, "fail")], metrics=[("UNIV-001", 0.0)]),
        b.scenario_result("ff_better", metrics=[("UNIV-001", 0.0), ("UNIV-013", 1.0)]),
        b.scenario_result("ff_univ025_new", metrics=[("UNIV-001", 0.0), ("UNIV-025", 0.0)]),
        b.scenario_result("ff_univ025_still", metrics=[("UNIV-025", 0.0)]),
        b.scenario_result("pp_univ025_skipped", metrics=[("UNIV-025", 0.0, {"skipped": True})]),
        b.scenario_result("pp_univ025_informational", metrics=[("UNIV-025", 0.0, {"informational": True})]),
        b.scenario_result("pf_high_twice", assertions=[(PURPOSE, "fail")], metrics=[("UNIV-025", 0.0)]),
        b.scenario_result("pf_metric", metrics=[("UNIV-013", 0.0)]),
        # only in B: a failed high rule and a poor UNIV-025 never count
        b.scenario_result("b_only", assertions=[(PURPOSE, "fail")], metrics=[("UNIV-025", 0.0)]),
    ]
    return b.run(results, run_id=RUN_B, timestamp="2026-10-12T12:00:00Z")


EDGE_CHECK_REGRESSIONS = ["ff_high_assertion", "ff_medium_assertion", "ff_critical_assertion", "ff_univ025_new"]
EDGE_HIGH_REGRESSIONS = ["ff_high_assertion", "ff_univ025_new", "pf_high_twice"]


def test_edge_fixture_runs_are_consistent(run_builder):
    run_a, run_b = _edge_run_a(run_builder), _edge_run_b(run_builder)
    passed_a = {s.scenario_name: s.passed for s in run_a.scenario_results}
    passed_b = {s.scenario_name: s.passed for s in run_b.scenario_results}
    for name in passed_a:
        expected = {"ff": (False, False), "pp": (True, True), "pf": (True, False)}[name[:2]]
        assert (passed_a[name], passed_b[name]) == expected, name


def test_check_regressions_are_fail_to_fail_with_a_newly_failing_check(run_builder):
    result = _compare(_edge_run_a(run_builder), _edge_run_b(run_builder))
    assert _names(result.check_regressions) == EDGE_CHECK_REGRESSIONS
    assert all(diff.status_change == "fail->fail" for diff in result.check_regressions)
    diffs = _diffs(result.check_regressions)
    assert diffs["ff_high_assertion"].assertion_changes == [f"{PURPOSE}: pass -> fail"]
    assert diffs["ff_medium_assertion"].assertion_changes == [f"{WITHDRAWAL}: skip -> fail"]
    assert "UNIV-025 newly failing" in diffs["ff_univ025_new"].notes


def test_check_regressions_exclude_steady_and_improving_fail_to_fail(run_builder):
    result = _compare(_edge_run_a(run_builder), _edge_run_b(run_builder))
    names = _names(result.check_regressions)
    for name in ("ff_steady", "ff_better", "ff_univ025_still"):
        assert name not in names, name
    # pass -> fail scenarios stay in regressions only; B-only scenarios are never diffed.
    assert "pf_metric" not in names and "pf_high_twice" not in names and "b_only" not in names
    assert {"pf_metric", "pf_high_twice"} <= set(_names(result.regressions))


def test_high_regressions(run_builder):
    result = _compare(_edge_run_a(run_builder), _edge_run_b(run_builder))
    assert result.high_regressions == EDGE_HIGH_REGRESSIONS


def test_high_regressions_ignore_critical_medium_skipped_informational_and_b_only(run_builder):
    result = _compare(_edge_run_a(run_builder), _edge_run_b(run_builder))
    for name in (
        "ff_critical_assertion",     # critical, not high: a critical regression
        "ff_medium_assertion",       # medium
        "ff_univ025_still",          # UNIV-025 poor in A too
        "pp_univ025_skipped",        # skipped UNIV-025 is not counted
        "pp_univ025_informational",  # informational UNIV-025 is not counted
        "b_only",                    # only in run B
        "ff_steady",                 # high rule fail -> fail
    ):
        assert name not in result.high_regressions, name
    assert result.critical_regressions == ["ff_critical_assertion"]


def test_existing_compare_lists_are_unchanged_by_the_new_fields(run_builder):
    # Regression guard (passes before the fix too).
    result = _compare(_edge_run_a(run_builder), _edge_run_b(run_builder))
    assert _names(result.regressions) == ["pf_high_twice", "pf_metric"]
    assert result.improvements == []
    assert result.added_scenarios == ["b_only"]
    assert result.unchanged_fail == 7 and result.unchanged_pass == 2


def test_swapped_edge_runs_have_no_check_or_high_regressions(run_builder):
    result = _compare(_edge_run_b(run_builder), _edge_run_a(run_builder))
    # B -> A: the high and medium rules go fail -> pass / fail -> skip; UNIV-025 goes poor -> good or
    # absent; UNIV-013 in ff_better goes good -> poor (a check regression, not high).
    assert _names(result.check_regressions) == ["ff_better"]
    assert result.high_regressions == []


# ---------------------------------------------------------------------------------------------
# CLI gates on hand-built runs
# ---------------------------------------------------------------------------------------------

GATE_A = "20261013T120000Z-c3c3c3"
GATE_B = "20261014T120000Z-d3d3d3"


def _gate_runs(b, kind):
    """Runs where B differs from A by one change: ``high`` (a high rule newly fails), ``critical`` (the
    critical rule newly fails), ``medium`` (a medium rule newly fails in a fail -> fail scenario)."""
    rule = {"high": PURPOSE, "critical": CONSENT, "medium": WITHDRAWAL}[kind]
    base_metrics = [("UNIV-001", 0.0)] if kind == "medium" else [("UNIV-001", 1.0)]
    run_a = b.run(
        [b.scenario_result("gate", assertions=[(rule, "pass")], metrics=base_metrics)],
        run_id=GATE_A, timestamp="2026-10-13T12:00:00Z",
    )
    run_b = b.run(
        [b.scenario_result("gate", assertions=[(rule, "fail")], metrics=base_metrics)],
        run_id=GATE_B, timestamp="2026-10-14T12:00:00Z",
    )
    return run_a, run_b


@pytest.mark.parametrize(
    "kind, fail_on, exit_code",
    [
        ("high", None, 0),
        ("high", "critical", 0),
        ("high", "high", 1),
        ("critical", None, 1),
        ("critical", "critical", 1),
        ("critical", "high", 1),
        ("medium", "critical", 0),
        ("medium", "high", 0),
    ],
)
def test_cli_compare_fail_on_gates(impulse_project, run_builder, cli, kind, fail_on, exit_code):
    run_a, run_b = _gate_runs(run_builder, kind)
    run_builder.write_run(run_a, impulse_project.runs_dir)
    run_builder.write_run(run_b, impulse_project.runs_dir)
    args = ["compare", GATE_A, GATE_B] + (["--fail-on", fail_on] if fail_on else [])
    result = cli.invoke(*args)
    assert result.exit_code == exit_code, cli.text(result)


def test_cli_compare_without_high_regressions_prints_no_high_line_or_banner(impulse_project, run_builder, cli):
    run_a, run_b = _gate_runs(run_builder, "medium")
    run_builder.write_run(run_a, impulse_project.runs_dir)
    run_builder.write_run(run_b, impulse_project.runs_dir)
    result = cli.invoke("compare", GATE_A, GATE_B, "--fail-on", "high")
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert "Newly failing checks (fail -> fail): 1" in text
    assert f"gate {WITHDRAWAL}: pass -> fail" in _block(text, "Newly failing checks (fail -> fail): 1", "Unchanged:")
    assert "High regressions" not in text
    _, html_path = _outputs(impulse_project, GATE_A, GATE_B)
    page_text = _page_text(html_path.read_text(encoding="utf-8"))
    assert "High regressions:" not in page_text   # no banner (a section heading is not ruled out)
    assert "Newly failing checks (fail -> fail)" in page_text


def test_cli_check_regression_line_carries_the_critical_flag(impulse_project, run_builder, cli):
    b = run_builder
    run_a = b.run(
        [b.scenario_result("crit_ff", assertions=[(CONSENT, "skip")], metrics=[("UNIV-001", 0.0)])],
        run_id=GATE_A, timestamp="2026-10-13T12:00:00Z",
    )
    run_b = b.run(
        [b.scenario_result("crit_ff", assertions=[(CONSENT, "fail")], metrics=[("UNIV-001", 0.0)])],
        run_id=GATE_B, timestamp="2026-10-14T12:00:00Z",
    )
    b.write_run(run_a, impulse_project.runs_dir)
    b.write_run(run_b, impulse_project.runs_dir)
    result = cli.invoke("compare", GATE_A, GATE_B)
    text = cli.text(result)
    assert result.exit_code == 1, text
    block = _block(text, "Newly failing checks (fail -> fail): 1", "Unchanged:")
    assert f"CRITICAL crit_ff {CONSENT}: skip -> fail" in block


# =============================================================================================
# A3-rev (after Stage D review R2): computed from the results, first entry on both sides,
# a fail -> fail scenario that newly errors is a check regression
# =============================================================================================

REV_A = "20261015T120000Z-e3e3e3"
REV_B = "20261016T120000Z-f3f3f3"
X_RULE = "rbi_kyc.x"     # not in DPDP_RULE_META: high unless a severity is given


def _plugin_metric(b, metric_id, label):
    """A plugin metric result whose ID has a space (``MetricResult.metric_id`` is a plain ``str``)."""
    return b.models().MetricResult(
        metric_id=metric_id, name=metric_id, category="universal",
        score={"good": 1.0, "warning": 0.5, "poor": 0.0}[label], label=label, reason=f"{metric_id}: {label}",
    )


def _rev_compare(b, scenarios_a, scenarios_b):
    run_a = b.run(scenarios_a, run_id=REV_A, timestamp="2026-10-15T12:00:00Z")
    run_b = b.run(scenarios_b, run_id=REV_B, timestamp="2026-10-16T12:00:00Z")
    return run_a, run_b, _compare(run_a, run_b)


def test_rev_plugin_metric_id_with_a_space_newly_poor_is_a_check_regression(run_builder):
    # R2-L6: computed from the results, not by parsing the note text.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("brand", metrics=[_plugin_metric(b, "Brand voice", "good"), ("UNIV-001", 0.0)])],
        [b.scenario_result("brand", metrics=[_plugin_metric(b, "Brand voice", "poor"), ("UNIV-001", 0.0)])],
    )
    assert result.unchanged_fail == 1
    assert _names(result.check_regressions) == ["brand"]
    diff = result.check_regressions[0]
    assert diff.status_change == "fail->fail"
    assert "Brand voice newly failing" in diff.notes
    assert result.high_regressions == [] and result.critical_regressions == []


def test_rev_plugin_metric_id_with_a_space_still_poor_is_not_a_check_regression(run_builder):
    # Regression guard (passes before the fix too).
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("brand", metrics=[_plugin_metric(b, "Brand voice", "poor")])],
        [b.scenario_result("brand", metrics=[_plugin_metric(b, "Brand voice", "poor")])],
    )
    assert result.check_regressions == []


def test_rev_assertion_first_entry_in_a_is_used(run_builder):
    # R2-L7 probe dup_rule_in_A: A [x pass, x fail], B [x fail]. The first entry of A is pass, so x is
    # newly failing (the diff shows "x: pass -> fail"): a check regression and, x being high, a high
    # regression.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("dup_a", assertions=[(X_RULE, "pass"), (X_RULE, "fail")], metrics=[("UNIV-001", 0.0)])],
        [b.scenario_result("dup_a", assertions=[(X_RULE, "fail")], metrics=[("UNIV-001", 0.0)])],
    )
    assert _names(result.check_regressions) == ["dup_a"]
    assert result.check_regressions[0].assertion_changes == [f"{X_RULE}: pass -> fail"]
    assert result.high_regressions == ["dup_a"]


def test_rev_assertion_first_entry_in_a_fail_is_not_newly_failing(run_builder):
    # Regression guard (passes before the fix too): A [x fail, x pass], B [x fail]: first entry fail
    # on both sides, nothing newly fails.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("dup_a", assertions=[(X_RULE, "fail"), (X_RULE, "pass")], metrics=[("UNIV-001", 0.0)])],
        [b.scenario_result("dup_a", assertions=[(X_RULE, "fail")], metrics=[("UNIV-001", 0.0)])],
    )
    assert result.check_regressions == [] and result.high_regressions == []


def test_rev_assertion_first_entry_in_b_is_used(run_builder):
    # Regression guard (passes before the fix too): A [x pass], B [x pass, x fail]: the first entry
    # of B is pass, so nothing newly fails.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("dup_b", assertions=[(X_RULE, "pass")], metrics=[("UNIV-001", 0.0)])],
        [b.scenario_result("dup_b", assertions=[(X_RULE, "pass"), (X_RULE, "fail")], metrics=[("UNIV-001", 0.0)])],
    )
    assert result.check_regressions == [] and result.high_regressions == []


def test_rev_critical_regression_uses_the_first_entry_in_b(run_builder):
    # R2-L7 probe dup_rule_in_B: A [y pass], B [y pass, y fail critical]. The first entry of B is
    # pass: no assertion change in the diff, so no critical regression either.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("dup_b", assertions=[("rbi_kyc.y", "pass", {"severity": "critical"})],
                           metrics=[("UNIV-001", 0.0)])],
        [b.scenario_result("dup_b", assertions=[("rbi_kyc.y", "pass", {"severity": "critical"}),
                                                ("rbi_kyc.y", "fail", {"severity": "critical"})],
                           metrics=[("UNIV-001", 0.0)])],
    )
    assert result.critical_regressions == []
    assert result.check_regressions == []


def test_rev_critical_regression_uses_the_first_entry_in_a(run_builder):
    # A [c pass, c fail], B [c fail] (critical): first entry pass -> fail, a critical regression with
    # its row in check_regressions (fail -> fail).
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("dup_a", assertions=[(CONSENT, "pass"), (CONSENT, "fail")])],
        [b.scenario_result("dup_a", assertions=[(CONSENT, "fail")])],
    )
    assert result.critical_regressions == ["dup_a"]
    assert _names(result.check_regressions) == ["dup_a"]
    assert result.check_regressions[0].assertion_changes == [f"{CONSENT}: pass -> fail"]


def test_rev_metric_first_entry_on_both_sides(run_builder):
    # Regression guard (passes before the fix too): first entry per metric ID on both sides.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [
            # A [good, poor], B [poor]: first entry good -> poor, newly failing
            b.scenario_result("metric_dup_a", metrics=[("UNIV-013", 1.0), ("UNIV-013", 0.0), ("UNIV-001", 0.0)]),
            # A [good], B [good, poor]: first entry of B good, not newly failing
            b.scenario_result("metric_dup_b", metrics=[("UNIV-013", 1.0), ("UNIV-001", 0.0)]),
        ],
        [
            b.scenario_result("metric_dup_a", metrics=[("UNIV-013", 0.0), ("UNIV-001", 0.0)]),
            b.scenario_result("metric_dup_b", metrics=[("UNIV-013", 1.0), ("UNIV-013", 0.0), ("UNIV-001", 0.0)]),
        ],
    )
    assert _names(result.check_regressions) == ["metric_dup_a"]


def test_rev_univ025_first_entry_on_both_sides_for_high(run_builder):
    # Regression guard (passes before the fix too): UNIV-025 A [good, poor] -> B [poor] is newly poor;
    # A [good] -> B [good, poor] is not.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [
            b.scenario_result("u_dup_a", metrics=[("UNIV-025", 1.0), ("UNIV-025", 0.0)]),
            b.scenario_result("u_dup_b", metrics=[("UNIV-025", 1.0), ("UNIV-001", 0.0)]),
        ],
        [
            b.scenario_result("u_dup_a", metrics=[("UNIV-025", 0.0)]),
            b.scenario_result("u_dup_b", metrics=[("UNIV-025", 1.0), ("UNIV-025", 0.0), ("UNIV-001", 0.0)]),
        ],
    )
    assert result.high_regressions == ["u_dup_a"]


def test_rev_metric_informational_in_either_run_is_not_a_check_regression(run_builder):
    # Regression guard (passes before the fix too): "not informational in either".
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [
            b.scenario_result("info_a", metrics=[("UNIV-013", 1.0, {"informational": True}), ("UNIV-001", 0.0)]),
            b.scenario_result("info_b", metrics=[("UNIV-013", 1.0), ("UNIV-001", 0.0)]),
        ],
        [
            b.scenario_result("info_a", metrics=[("UNIV-013", 0.0), ("UNIV-001", 0.0)]),
            b.scenario_result("info_b", metrics=[("UNIV-013", 0.0, {"informational": True}), ("UNIV-001", 0.0)]),
        ],
    )
    assert result.check_regressions == []


def test_rev_fail_to_fail_scenario_that_newly_errors_is_a_check_regression(run_builder):
    # R2-L8 (a): the critical line always has a row explaining it.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("err_ff", metrics=[("UNIV-001", 0.0)])],
        [b.scenario_result("err_ff", metrics=[("UNIV-001", 0.0)], error="agent crashed")],
    )
    assert result.critical_regressions == ["err_ff"]
    assert _names(result.check_regressions) == ["err_ff"]
    diff = result.check_regressions[0]
    assert diff.status_change == "fail->fail"
    assert diff.notes[-1] == "scenario error: agent crashed"


def test_rev_error_in_both_runs_is_not_a_check_regression(run_builder):
    # Regression guard (passes before the fix too): only a NEW error counts.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("err_both", metrics=[("UNIV-001", 0.0)], error="agent crashed")],
        [b.scenario_result("err_both", metrics=[("UNIV-001", 0.0)], error="agent crashed again")],
    )
    assert result.check_regressions == [] and result.critical_regressions == []


def test_rev_scenario_errored_in_a_makes_b_failures_newly_failing(run_builder):
    # Regression guard (passes before the fix too; A3-rev keeps it as specified): A errored before
    # evaluation (no results), B fails a high rule: a check regression and a high regression, not a
    # critical regression.
    b = run_builder
    _, _, result = _rev_compare(
        b,
        [b.scenario_result("err_in_a", error="LiveKit agent did not finish its turn within 30 s.")],
        [b.scenario_result("err_in_a", assertions=[(X_RULE, "fail")], metrics=[("UNIV-001", 0.0)])],
    )
    assert _names(result.check_regressions) == ["err_in_a"]
    assert result.high_regressions == ["err_in_a"]
    assert result.critical_regressions == []


def test_cli_rev_newly_errored_fail_to_fail_row_carries_the_critical_flag(impulse_project, run_builder, cli):
    b = run_builder
    run_a = b.run([b.scenario_result("err_ff", metrics=[("UNIV-001", 0.0)])],
                  run_id=GATE_A, timestamp="2026-10-13T12:00:00Z")
    run_b = b.run([b.scenario_result("err_ff", metrics=[("UNIV-001", 0.0)], error="agent crashed")],
                  run_id=GATE_B, timestamp="2026-10-14T12:00:00Z")
    b.write_run(run_a, impulse_project.runs_dir)
    b.write_run(run_b, impulse_project.runs_dir)
    result = cli.invoke("compare", GATE_A, GATE_B)
    text = cli.text(result)
    assert result.exit_code == 1, text
    block = _block(text, "Newly failing checks (fail -> fail): 1", "Unchanged:")
    assert "CRITICAL err_ff scenario error: agent crashed" in block
    assert "Critical regressions: err_ff" in text
