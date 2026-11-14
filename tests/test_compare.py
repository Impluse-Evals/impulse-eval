# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 8: ``impulse compare RUN_A RUN_B`` and ``compare.compare_runs`` (SPEC.md sections 7, 15, 16, 17).

Two stored ``run.json`` fixtures are built in ``tmp_path`` from SPEC.md section 7 models (JSON files
carry no SPDX header). Run A -> run B:

* ``s_regress``   pass -> fail; its critical ``dpdp.consent_before_pii`` goes pass -> fail
* ``s_high``      pass -> fail on a high rule only (a regression, not a critical regression)
* ``s_improve``   fail -> pass
* ``s_same_pass`` pass -> pass (counted in ``unchanged_pass``; no ``ScenarioDiff`` list holds it)
* ``s_same_fail`` fail -> fail; its critical rule goes skip -> fail (a critical regression)
* ``s_removed``   only in A; ``s_added`` only in B (with a failed critical rule: never a critical regression)
"""

import re

import pytest

CONSENT = "dpdp.consent_before_pii"
PURPOSE = "dpdp.purpose_disclosure"
RUN_A = "20261001T120000Z-aaaaaa"
RUN_B = "20261002T120000Z-bbbbbb"
EXTERNAL = re.compile(r"""(?:\b(?:src|href)\s*=\s*["']?\s*|url\(\s*["']?\s*)(?:https?:|//)""", re.IGNORECASE)


def _compare(run_a, run_b):
    from impulse_evals.compare import compare_runs

    return compare_runs(run_a, run_b)


def _run_a(b):
    results = [
        b.scenario_result("s_regress", assertions=[(CONSENT, "pass")], metrics=[("UNIV-001", 1.0), ("UNIV-013", 1.0)]),
        b.scenario_result("s_high", assertions=[(PURPOSE, "pass")], metrics=[("UNIV-013", 1.0)]),
        b.scenario_result("s_improve", assertions=[(PURPOSE, "fail")], metrics=[("UNIV-013", 0.0)]),
        b.scenario_result(
            "s_same_pass",
            metrics=[("UNIV-001", 1.0), ("UNIV-003", 0.0, {"skipped": True}), ("VOICE-029", 1.0)],
        ),
        b.scenario_result("s_same_fail", assertions=[(CONSENT, "skip")], metrics=[("UNIV-001", 0.0)]),
        b.scenario_result("s_removed", metrics=[("UNIV-001", 1.0)]),
    ]
    summary = {
        "UNIV-001": b.metric("UNIV-001", 0.6),
        "UNIV-013": b.metric("UNIV-013", 0.5),
        "VOICE-029": b.metric("VOICE-029", 1.0),
    }
    return b.run(results, run_id=RUN_A, timestamp="2026-10-01T12:00:00Z", metric_summary=summary)


def _run_b(b):
    results = [
        b.scenario_result("s_regress", assertions=[(CONSENT, "fail")], metrics=[("UNIV-001", 0.0), ("UNIV-013", 1.0)]),
        b.scenario_result("s_high", assertions=[(PURPOSE, "fail")], metrics=[("UNIV-013", 1.0)]),
        b.scenario_result("s_improve", assertions=[(PURPOSE, "pass")], metrics=[("UNIV-013", 1.0)]),
        b.scenario_result(
            "s_same_pass",
            metrics=[("UNIV-001", 1.0), ("UNIV-003", 1.0), ("VOICE-029", 0.5)],
        ),
        b.scenario_result("s_same_fail", assertions=[(CONSENT, "fail")], metrics=[("UNIV-001", 0.0)]),
        b.scenario_result("s_added", assertions=[(CONSENT, "fail")], metrics=[("UNIV-001", 1.0)]),
    ]
    summary = {
        "UNIV-001": b.metric("UNIV-001", 0.4),
        "UNIV-013": b.metric("UNIV-013", 1.0),
        "UNIV-025": b.metric("UNIV-025", 1.0),
    }
    return b.run(results, run_id=RUN_B, timestamp="2026-10-02T12:00:00Z", metric_summary=summary)


def _diffs(diffs):
    return {diff.scenario_name: diff for diff in diffs}


# ============================================================================================
# compare_runs
# ============================================================================================


def test_fixture_runs_are_consistent(run_builder):
    run_a, run_b = _run_a(run_builder), _run_b(run_builder)
    assert [s.passed for s in run_a.scenario_results] == [True, True, False, True, False, True]
    assert [s.passed for s in run_b.scenario_results] == [False, False, True, True, False, False]


def test_compare_result_header_fields(run_builder):
    run_a, run_b = _run_a(run_builder), _run_b(run_builder)
    result = _compare(run_a, run_b)
    assert (result.run_id_a, result.run_id_b) == (RUN_A, RUN_B)
    assert (result.timestamp_a, result.timestamp_b) == ("2026-10-01T12:00:00Z", "2026-10-02T12:00:00Z")
    assert result.overall_pass_rate_delta == pytest.approx(run_b.overall_pass_rate - run_a.overall_pass_rate)


def test_regressions_and_improvements_detected(run_builder):
    result = _compare(_run_a(run_builder), _run_b(run_builder))
    regressions = _diffs(result.regressions)
    assert set(regressions) == {"s_regress", "s_high"}
    assert all(diff.status_change == "pass->fail" for diff in result.regressions)
    improvements = _diffs(result.improvements)
    assert set(improvements) == {"s_improve"}
    assert improvements["s_improve"].status_change == "fail->pass"
    assert result.unchanged_pass == 1
    assert result.unchanged_fail == 1


def test_added_and_removed_scenarios_by_name_only(run_builder):
    result = _compare(_run_a(run_builder), _run_b(run_builder))
    assert result.added_scenarios == ["s_added"]
    assert result.removed_scenarios == ["s_removed"]
    named = {diff.scenario_name for diff in result.regressions + result.improvements}
    assert "s_added" not in named and "s_removed" not in named


def test_critical_regressions(run_builder):
    result = _compare(_run_a(run_builder), _run_b(run_builder))
    # s_high fails only a high rule; s_added is new; s_same_fail went skip -> fail on a critical rule.
    assert sorted(result.critical_regressions) == ["s_regress", "s_same_fail"]


def test_metric_deltas_skip_skipped_metrics(run_builder):
    result = _compare(_run_a(run_builder), _run_b(run_builder))
    regress = _diffs(result.regressions)["s_regress"]
    assert regress.metric_deltas == pytest.approx({"UNIV-001": -1.0, "UNIV-013": 0.0})
    improve = _diffs(result.improvements)["s_improve"]
    assert improve.metric_deltas == pytest.approx({"UNIV-013": 1.0})


def test_assertion_changes_format(run_builder):
    result = _compare(_run_a(run_builder), _run_b(run_builder))
    assert _diffs(result.regressions)["s_regress"].assertion_changes == [f"{CONSENT}: pass -> fail"]
    assert _diffs(result.regressions)["s_high"].assertion_changes == [f"{PURPOSE}: pass -> fail"]
    assert _diffs(result.improvements)["s_improve"].assertion_changes == [f"{PURPOSE}: fail -> pass"]


def test_notes_for_label_changes_and_large_deltas(run_builder):
    result = _compare(_run_a(run_builder), _run_b(run_builder))
    regress_notes = _diffs(result.regressions)["s_regress"].notes
    assert "UNIV-001 newly failing" in regress_notes
    assert any(re.fullmatch(r"UNIV-001 worsened -?1\.00", note) for note in regress_notes)
    assert not any(note.startswith("UNIV-013") for note in regress_notes)  # delta 0.0, label unchanged
    improve_notes = _diffs(result.improvements)["s_improve"].notes
    assert "UNIV-013 newly passing" in improve_notes
    assert "UNIV-013 improved 1.00" in improve_notes


def test_metric_summary_deltas_only_for_metrics_in_both_runs(run_builder):
    result = _compare(_run_a(run_builder), _run_b(run_builder))
    assert result.metric_summary_deltas == pytest.approx({"UNIV-001": -0.2, "UNIV-013": 0.5})


def test_compare_result_round_trips(run_builder):
    from impulse_evals.models import CompareResult

    result = _compare(_run_a(run_builder), _run_b(run_builder))
    assert isinstance(result, CompareResult)
    assert CompareResult.model_validate_json(result.model_dump_json()) == result


def test_no_critical_regressions_when_runs_are_swapped(run_builder):
    result = _compare(_run_b(run_builder), _run_a(run_builder))
    assert result.critical_regressions == []
    assert {diff.scenario_name for diff in result.regressions} == {"s_improve"}


# ============================================================================================
# impulse compare (CLI)
# ============================================================================================


@pytest.fixture
def stored_runs(impulse_project, run_builder):
    run_a, run_b = _run_a(run_builder), _run_b(run_builder)
    path_a = run_builder.write_run(run_a, impulse_project.runs_dir)
    path_b = run_builder.write_run(run_b, impulse_project.runs_dir)
    return run_a, run_b, path_a, path_b


def _outputs(project):
    return project.runs_dir / f"compare_{RUN_A}_vs_{RUN_B}.json", project.runs_dir / f"compare_{RUN_A}_vs_{RUN_B}.html"


def test_cli_compare_by_run_id_exits_1_on_critical_regression(impulse_project, stored_runs, cli):
    from impulse_evals.models import CompareResult

    run_a, run_b, _, _ = stored_runs
    result = cli.invoke("compare", RUN_A, RUN_B)
    assert result.exit_code == 1, cli.text(result)
    json_path, html_path = _outputs(impulse_project)
    stored = CompareResult.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert stored == _compare(run_a, run_b)
    html = html_path.read_text(encoding="utf-8")
    for name in ("s_regress", "s_high", "s_improve", "s_same_fail"):
        assert name in html
    assert "<svg" in html
    assert not EXTERNAL.search(html)
    text = cli.text(result)
    assert "s_regress" in text
    assert "s_improve" in text


def test_cli_compare_exits_0_without_critical_regressions(impulse_project, stored_runs, cli):
    result = cli.invoke("compare", RUN_B, RUN_A)
    assert result.exit_code == 0, cli.text(result)
    assert (impulse_project.runs_dir / f"compare_{RUN_B}_vs_{RUN_A}.json").is_file()
    assert (impulse_project.runs_dir / f"compare_{RUN_B}_vs_{RUN_A}.html").is_file()


@pytest.mark.parametrize("form", ["run_json", "run_folder"])
def test_cli_compare_accepts_paths(impulse_project, stored_runs, cli, form):
    _, _, path_a, path_b = stored_runs
    args = (path_a, path_b) if form == "run_json" else (path_a.parent, path_b.parent)
    result = cli.invoke("compare", *args)
    assert result.exit_code == 1, cli.text(result)
    json_path, html_path = _outputs(impulse_project)
    assert json_path.is_file()
    assert html_path.is_file()


@pytest.mark.parametrize("reporter, json_expected, html_expected", [("json", True, False), ("html", False, True)])
def test_cli_compare_reporter_flag(impulse_project, stored_runs, cli, reporter, json_expected, html_expected):
    result = cli.invoke("compare", RUN_A, RUN_B, "--reporter", reporter)
    assert result.exit_code == 1, cli.text(result)
    json_path, html_path = _outputs(impulse_project)
    assert json_path.is_file() is json_expected
    assert html_path.is_file() is html_expected


def test_cli_compare_missing_run_exits_2(impulse_project, stored_runs, cli):
    assert cli.invoke("compare", RUN_A, "20261003T120000Z-cccccc").exit_code == 2
    assert cli.invoke("compare", "no_such_run", RUN_B).exit_code == 2


def test_cli_compare_leaves_run_files_unchanged(impulse_project, stored_runs, cli):
    _, _, path_a, path_b = stored_runs
    before = (path_a.read_bytes(), path_b.read_bytes())
    cli.invoke("compare", RUN_A, RUN_B)
    assert (path_a.read_bytes(), path_b.read_bytes()) == before


# ============================================================================================
# Audit batch 2 (AUDIT.md finding 9): a run is a dry run when any of its metric or assertion
# results is skipped with the reason exactly "dry run". For each such run `impulse compare` prints
# "Note: run {run_id} is a dry run: judge metrics and assertions were skipped, so its results are
# not verdicts." after the table (run A first), and the compare HTML shows the same sentence
# (without "Note: ") above its tables. CompareResult is unchanged.
# ============================================================================================

DRY_NOTE = "run {run_id} is a dry run: judge metrics and assertions were skipped, so its results are not verdicts."
REAL_RUN = "20261001T120000Z-cccccc"
REAL_RUN_2 = "20261002T120000Z-dddddd"


def _dry_config(project):
    project.write_agent()  # copy of examples/sample_agent.py
    return project.write_config(
        [project.scenario("clinic_booking", assertions=[CONSENT, PURPOSE], metrics=["UNIV-001", "UNIV-013"], max_turns=4)],
        judge={"primary_model": "YOUR_PRIMARY_MODEL", "fast_model": "YOUR_FAST_MODEL"},
        simulator={"model": "YOUR_SIMULATOR_MODEL", "persona_style": "realistic", "mode": "cascade"},
    )


def _cli_dry_run(project, cli, config):
    before = set(project.run_dirs())
    result = cli.invoke("run", config, "--dry-run", "--reporter", "json")
    assert result.exit_code == 0, cli.text(result)
    (run_dir,) = set(project.run_dirs()) - before
    return project.read_run(run_dir)


def _real_failed_run(b, run_id=REAL_RUN, timestamp="2026-10-01T12:00:00Z"):
    result = b.scenario_result(
        "clinic_booking",
        assertions=[(CONSENT, "fail"), (PURPOSE, "pass")],
        metrics=[("UNIV-001", 0.0), ("UNIV-013", 1.0, {"skipped": True, "reason": "No tone to judge."})],
    )
    return b.run([result], run_id=run_id, timestamp=timestamp)


def _compare_html(project, run_id_a, run_id_b):
    return (project.runs_dir / f"compare_{run_id_a}_vs_{run_id_b}.html").read_text(encoding="utf-8")


_NOTE_RE = re.compile(r"Note: run \S+ is a dry run: [^.]*\.")


def _note_lines(text):
    """Every "Note: run ... is a dry run: ..." sentence of the normalised console text, in order (Rich
    may wrap a long line, so the notes are found in the whitespace-collapsed text)."""
    return _NOTE_RE.findall(text)


def test_cli_compare_real_vs_dry_run_prints_the_dry_run_note(impulse_project, mock_llm, cli, run_builder):
    from impulse_evals.models import CompareResult

    dry = _cli_dry_run(impulse_project, cli, _dry_config(impulse_project))
    real = _real_failed_run(run_builder)
    run_builder.write_run(real, impulse_project.runs_dir)
    result = cli.invoke("compare", REAL_RUN, dry.run_id)
    text = cli.text(result)
    note = "Note: " + DRY_NOTE.format(run_id=dry.run_id)
    assert note in text, text
    assert text.count("is a dry run") == 1, text   # run A is a real run: no note for it
    assert text.index(note) > text.index("Unchanged:"), "the note follows the table"
    assert _note_lines(cli.text(result)) == [note]
    html = _compare_html(impulse_project, REAL_RUN, dry.run_id)
    sentence = DRY_NOTE.format(run_id=dry.run_id)
    assert sentence in html
    assert "Note: run" not in html
    assert html.index(sentence) < html.index("<table"), "the sentence is shown above the tables"
    # Regression guard: CompareResult and the exit code are unchanged by the note.
    stored = CompareResult.model_validate_json(
        (impulse_project.runs_dir / f"compare_{REAL_RUN}_vs_{dry.run_id}.json").read_text(encoding="utf-8")
    )
    assert stored == _compare(real, dry)
    assert result.exit_code == 0


def test_cli_compare_two_dry_runs_notes_both_run_a_first(impulse_project, mock_llm, cli):
    config = _dry_config(impulse_project)
    dry_a = _cli_dry_run(impulse_project, cli, config)
    dry_b = _cli_dry_run(impulse_project, cli, config)
    assert dry_a.run_id != dry_b.run_id
    result = cli.invoke("compare", dry_a.run_id, dry_b.run_id)
    assert result.exit_code == 0, cli.text(result)
    notes = ["Note: " + DRY_NOTE.format(run_id=run.run_id) for run in (dry_a, dry_b)]
    assert _note_lines(cli.text(result)) == notes
    html = _compare_html(impulse_project, dry_a.run_id, dry_b.run_id)
    first, second = (html.index(note[len("Note: "):]) for note in notes)
    assert first < second < html.index("<table")


@pytest.mark.parametrize("where", ["metric", "assertion"])
def test_cli_compare_detects_a_dry_run_from_one_dry_run_skip_reason(impulse_project, cli, run_builder, where):
    # A hand-built run B without the run warning: one result skipped with the reason "dry run" is enough.
    b = run_builder
    real = _real_failed_run(b)
    if where == "metric":
        scenario = b.scenario_result(
            "clinic_booking",
            assertions=[(CONSENT, "pass"), (PURPOSE, "pass")],
            metrics=[("UNIV-001", 1.0), ("UNIV-013", 0.0, {"skipped": True, "reason": "dry run"})],
        )
    else:
        scenario = b.scenario_result(
            "clinic_booking",
            assertions=[(CONSENT, "skip", {"reason": "dry run"}), (PURPOSE, "pass")],
            metrics=[("UNIV-001", 1.0)],
        )
    marked = b.run([scenario], run_id=REAL_RUN_2, timestamp="2026-10-02T12:00:00Z")
    assert marked.warnings == []
    for run in (real, marked):
        b.write_run(run, impulse_project.runs_dir)
    result = cli.invoke("compare", REAL_RUN, REAL_RUN_2)
    assert _note_lines(cli.text(result)) == ["Note: " + DRY_NOTE.format(run_id=REAL_RUN_2)], cli.text(result)
    assert DRY_NOTE.format(run_id=REAL_RUN_2) in _compare_html(impulse_project, REAL_RUN, REAL_RUN_2)


def test_cli_compare_of_real_runs_prints_no_dry_run_note(impulse_project, cli, run_builder):
    # Regression guard (passes before the fix too): skip reasons other than exactly "dry run" never
    # mark a run, and neither stored fixture run is a dry run.
    b = run_builder
    real = _real_failed_run(b)
    other = b.run(
        [
            b.scenario_result(
                "clinic_booking",
                assertions=[(CONSENT, "skip", {"reason": "No personal data was requested."}), (PURPOSE, "pass")],
                metrics=[("UNIV-001", 1.0), ("UNIV-013", 0.0, {"skipped": True, "reason": "dry run skipped by hand"})],
            )
        ],
        run_id=REAL_RUN_2,
        timestamp="2026-10-02T12:00:00Z",
    )
    for run in (real, other):
        b.write_run(run, impulse_project.runs_dir)
    result = cli.invoke("compare", REAL_RUN, REAL_RUN_2)
    assert "is a dry run" not in cli.text(result)
    assert "is a dry run" not in _compare_html(impulse_project, REAL_RUN, REAL_RUN_2)


def test_cli_compare_stored_fixtures_print_no_dry_run_note(impulse_project, stored_runs, cli):
    # Regression guard (passes before the fix too).
    result = cli.invoke("compare", RUN_A, RUN_B)
    assert "is a dry run" not in cli.text(result)
    _, html_path = _outputs(impulse_project)
    assert "is a dry run" not in html_path.read_text(encoding="utf-8")


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 41; maintainer decision #41): a scenario present in both runs
# whose error is None in A and set in B is a critical regression (including the critical
# no-verdict error of SPEC.md section 13), so `impulse compare` exits 1; its ScenarioDiff.notes
# gains "scenario error: {error}". An error in both runs, or only in A, is not a critical regression.
# ============================================================================================

ERR_RUN_A = "20261003T120000Z-eeeeee"
ERR_RUN_B = "20261004T120000Z-ffffff"
NO_VERDICT_ERROR = f"critical check {CONSENT} could not be evaluated: judge returned no verdict"
CRASH_ERROR = "agent crashed while answering: RuntimeError('boom')"
OLD_ERROR = "Pipecat bot did not finish its turn within 30 s."


def _error_run_a(b):
    results = [
        b.scenario_result("s_noverdict", assertions=[(CONSENT, "pass")], metrics=[("UNIV-001", 1.0)]),
        b.scenario_result("s_crash", assertions=[(CONSENT, "pass")], metrics=[("UNIV-001", 1.0)]),
        b.scenario_result("s_was_failing", assertions=[(PURPOSE, "fail")], metrics=[("UNIV-001", 0.0)]),
        b.scenario_result("s_both_error", error=OLD_ERROR),
        b.scenario_result("s_fixed", error=OLD_ERROR),
        b.scenario_result("s_steady", assertions=[(CONSENT, "pass")], metrics=[("UNIV-001", 1.0)]),
    ]
    return b.run(results, run_id=ERR_RUN_A, timestamp="2026-10-03T12:00:00Z")


def _error_run_b(b):
    results = [
        # SPEC.md section 13: the critical rule is recorded as skip with no verdict, the other results
        # are kept, the scenario error is set.
        b.scenario_result(
            "s_noverdict",
            assertions=[(CONSENT, "skip", {"no_verdict": True, "reason": "Judge returned invalid JSON"})],
            metrics=[("UNIV-001", 1.0)],
            error=NO_VERDICT_ERROR,
        ),
        # An error before evaluation: empty result lists.
        b.scenario_result("s_crash", error=CRASH_ERROR),
        b.scenario_result("s_was_failing", error=CRASH_ERROR),
        b.scenario_result("s_both_error", error=CRASH_ERROR),
        b.scenario_result("s_fixed", assertions=[(CONSENT, "pass")], metrics=[("UNIV-001", 1.0)]),
        b.scenario_result("s_steady", assertions=[(CONSENT, "pass")], metrics=[("UNIV-001", 1.0)]),
        b.scenario_result("s_new_error", error=CRASH_ERROR),   # only in B: never a critical regression
    ]
    return b.run(results, run_id=ERR_RUN_B, timestamp="2026-10-04T12:00:00Z")


def test_error_fixture_runs_are_consistent(run_builder):
    run_a, run_b = _error_run_a(run_builder), _error_run_b(run_builder)
    assert [s.passed for s in run_a.scenario_results] == [True, True, False, False, False, True]
    assert [s.passed for s in run_b.scenario_results] == [False, False, False, False, True, True, False]
    assert run_b.scenario_results[0].assertion_results[0].no_verdict is True


def test_scenario_that_newly_errors_is_a_critical_regression(run_builder):
    result = _compare(_error_run_a(run_builder), _error_run_b(run_builder))
    assert sorted(result.critical_regressions) == ["s_crash", "s_noverdict", "s_was_failing"]


def test_newly_erroring_scenario_diff_notes_the_error(run_builder):
    result = _compare(_error_run_a(run_builder), _error_run_b(run_builder))
    regressions = _diffs(result.regressions)
    assert set(regressions) == {"s_noverdict", "s_crash"}
    assert f"scenario error: {NO_VERDICT_ERROR}" in regressions["s_noverdict"].notes
    assert f"scenario error: {CRASH_ERROR}" in regressions["s_crash"].notes


def test_error_in_both_runs_or_only_in_a_is_not_a_critical_regression(run_builder):
    # Regression guard (passes before the fix too).
    result = _compare(_error_run_a(run_builder), _error_run_b(run_builder))
    for name in ("s_both_error", "s_fixed", "s_steady", "s_new_error"):
        assert name not in result.critical_regressions, name
    assert set(_diffs(result.improvements)) == {"s_fixed"}
    assert result.added_scenarios == ["s_new_error"]


def test_swapped_error_runs_flag_only_the_scenario_that_newly_errors(run_builder):
    # Run B -> run A: s_noverdict, s_crash and s_was_failing lose their error, s_both_error keeps
    # one, and only s_fixed (no error in B, an error in A) newly errors.
    result = _compare(_error_run_b(run_builder), _error_run_a(run_builder))
    assert result.critical_regressions == ["s_fixed"]


def test_newly_erroring_scenario_with_a_failed_critical_rule_is_listed_once(run_builder):
    # Regression guard (passes before the fix too, through the failed critical rule): one entry.
    b = run_builder
    run_a = b.run([b.scenario_result("s", assertions=[(CONSENT, "pass")])], run_id=ERR_RUN_A)
    run_b = b.run([b.scenario_result("s", assertions=[(CONSENT, "fail")], error=CRASH_ERROR)], run_id=ERR_RUN_B)
    assert _compare(run_a, run_b).critical_regressions == ["s"]


def test_cli_compare_exits_1_when_a_scenario_newly_errors(impulse_project, run_builder, cli):
    from impulse_evals.models import CompareResult

    b = run_builder
    run_a, run_b = _error_run_a(b), _error_run_b(b)
    b.write_run(run_a, impulse_project.runs_dir)
    b.write_run(run_b, impulse_project.runs_dir)
    result = cli.invoke("compare", ERR_RUN_A, ERR_RUN_B)
    text = cli.text(result)
    assert result.exit_code == 1, text
    assert "No critical regressions." not in text
    assert "Critical regressions:" in text
    critical_line = text.split("Critical regressions:", 1)[1]
    for name in ("s_noverdict", "s_crash", "s_was_failing"):
        assert name in critical_line, name
    assert f"scenario error: {NO_VERDICT_ERROR}" in text
    stored = CompareResult.model_validate_json(
        (impulse_project.runs_dir / f"compare_{ERR_RUN_A}_vs_{ERR_RUN_B}.json").read_text(encoding="utf-8")
    )
    assert sorted(stored.critical_regressions) == ["s_crash", "s_noverdict", "s_was_failing"]
    html = (impulse_project.runs_dir / f"compare_{ERR_RUN_A}_vs_{ERR_RUN_B}.html").read_text(encoding="utf-8")
    assert "No critical regressions." not in html
    assert "could not be evaluated: judge returned no verdict" in html


def test_cli_compare_exits_0_when_no_scenario_newly_errors(impulse_project, run_builder, cli):
    # Regression guard (passes before the fix too): an error in both runs, or fixed in B, exits 0.
    b = run_builder
    run_a = b.run(
        [b.scenario_result("s_both_error", error=OLD_ERROR), b.scenario_result("s_fixed", error=OLD_ERROR)],
        run_id=ERR_RUN_A,
    )
    run_b = b.run(
        [b.scenario_result("s_both_error", error=CRASH_ERROR), b.scenario_result("s_fixed", metrics=[("UNIV-001", 1.0)])],
        run_id=ERR_RUN_B,
    )
    b.write_run(run_a, impulse_project.runs_dir)
    b.write_run(run_b, impulse_project.runs_dir)
    result = cli.invoke("compare", ERR_RUN_A, ERR_RUN_B)
    assert result.exit_code == 0, cli.text(result)
    assert "No critical regressions." in cli.text(result)


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 39): the compare JSON and HTML files are written to a temporary
# file in the same folder and os.replace()d onto the target; on any error the temporary file is
# removed and an existing output is left untouched (the CLI exits 2: "cannot write ...").
# ============================================================================================

PREVIOUS_OUTPUT = b"previous compare output (kept)\n"


def _fail_replace_onto(monkeypatch, target):
    """``os.replace`` raises (permission denied) when its destination is ``target``; otherwise it works."""
    import errno
    import os
    from pathlib import Path

    real_replace = os.replace
    destination = Path(target).resolve()

    def failing_replace(src, dst, *args, **kwargs):
        if Path(dst).resolve() == destination:
            raise OSError(errno.EACCES, "Permission denied (test fault)", str(dst))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)


@pytest.mark.parametrize("reporter, index", [("json", 0), ("html", 1)])
def test_cli_compare_keeps_the_previous_output_when_the_final_replace_fails(
    impulse_project, stored_runs, cli, monkeypatch, reporter, index
):
    target = _outputs(impulse_project)[index]
    target.write_bytes(PREVIOUS_OUTPUT)
    before = sorted(path.name for path in impulse_project.runs_dir.iterdir())
    _fail_replace_onto(monkeypatch, target)
    result = cli.invoke("compare", RUN_A, RUN_B, "--reporter", reporter)
    assert target.read_bytes() == PREVIOUS_OUTPUT, f"{target.name} was rewritten in place"
    assert sorted(path.name for path in impulse_project.runs_dir.iterdir()) == before, "no temporary file is left"
    assert result.exit_code == 2, cli.text(result)
