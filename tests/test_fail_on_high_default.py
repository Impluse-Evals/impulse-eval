# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 4, O3 and L16 (FINAL-TEST.md, maintainer instruction 2026-11-02): ``--fail-on`` gates.

* O3: a poor UNIV-019 (harmful or misleading content: false urgency, credential requests, ...) counts as
  severity ``high`` for ``impulse run --fail-on`` and for ``impulse compare`` (``high_regressions``), like
  a poor UNIV-025. ``cli.HIGH_METRICS == ("UNIV-019", "UNIV-025")``. Reason texts: "{name}: UNIV-019 is
  poor (counts as severity high)"; both poor: "{name}: UNIV-019 and UNIV-025 are poor (count as severity
  high)" (one line per scenario, as before).
* O3: the default ``--fail-on`` of ``impulse run`` is ``high``: no ``.impulse/config.toml``, or one whose
  ``[defaults]`` has no ``fail_on``, gives ``high`` (``TomlDefaults.fail_on`` defaults to "high"; ``impulse
  init`` writes ``fail_on = "high"``). So a bot that asks for a password exits 1 by default in any language.
  ``impulse compare`` keeps its default ``critical``; ``assert_no_critical`` keeps its name and rule.
* L16: at ``--fail-on high`` (and ``any``) a ``high`` assertion with no judge verdict (``result == "skip"``,
  ``no_verdict=True``) counts like a high failure: "{name}: high assertion had no verdict: {rule_ids}".
  Critical no-verdict stays the scenario error (SPEC.md section 13); medium stays a plain skip.

The judge and the simulator are ``mock_llm`` routed by ``llm_router``; CLI through ``CliRunner``. No network.
"""

import pytest

CONSENT = "dpdp.consent_before_pii"
PURPOSE = "dpdp.purpose_disclosure"
WITHDRAWAL = "dpdp.withdrawal_path"
HIGH_METRICS = ("UNIV-019", "UNIV-025")


# ---------------------------------------------------------------------------------------------
# fail_on_reasons on hand-built runs
# ---------------------------------------------------------------------------------------------


def _run(run_builder, **scenarios):
    b = run_builder
    results = [b.scenario_result(name, **fields) for name, fields in scenarios.items()]
    return b.run(results)


def _reasons(run, level):
    from impulse_evals.cli import fail_on_reasons

    return fail_on_reasons(run, level)


def test_high_metrics_constant():
    from impulse_evals import cli

    assert cli.HIGH_METRICS == HIGH_METRICS


def test_poor_univ019_counts_as_high(run_builder):
    b = run_builder
    run = _run(run_builder, urgency={"metrics": [b.metric("UNIV-019", 0.0), b.metric("UNIV-013", 1.0)]})
    assert _reasons(run, "critical") == []
    assert _reasons(run, "high") == ["urgency: UNIV-019 is poor (counts as severity high)"]
    assert _reasons(run, "any") == ["urgency: UNIV-019 is poor (counts as severity high)"]


def test_poor_univ019_and_univ025_give_one_line(run_builder):
    b = run_builder
    run = _run(run_builder, both={"metrics": [b.metric("UNIV-025", 0.0), b.metric("UNIV-019", 0.0)]})
    assert _reasons(run, "high") == ["both: UNIV-019 and UNIV-025 are poor (count as severity high)"]
    only_025 = _run(run_builder, one={"metrics": [b.metric("UNIV-025", 0.0)]})
    assert _reasons(only_025, "high") == ["one: UNIV-025 is poor (counts as severity high)"]


@pytest.mark.parametrize("label", ["warning", "good"])
def test_univ019_not_poor_does_not_count(run_builder, label):
    b = run_builder
    score = 0.5 if label == "warning" else 1.0
    run = _run(run_builder, fine={"metrics": [b.metric("UNIV-019", score)]})
    assert _reasons(run, "high") == []


def test_skipped_or_informational_univ019_does_not_count(run_builder):
    b = run_builder
    skipped = _run(run_builder, s={"metrics": [b.metric("UNIV-019", skipped=True)]})
    informational = _run(run_builder, i={"metrics": [b.metric("UNIV-019", 0.0, informational=True)]})
    assert _reasons(skipped, "high") == [] and _reasons(informational, "high") == []
    assert _reasons(informational, "any") == []   # informational never fails a scenario either


def test_high_assertion_failure_still_comes_first(run_builder):
    b = run_builder
    run = _run(run_builder, mixed={"assertions": [b.assertion(PURPOSE, "fail")], "metrics": [b.metric("UNIV-019", 0.0)]})
    assert _reasons(run, "high") == [f"mixed: high assertion failed: {PURPOSE}"]


def test_high_no_verdict_counts_at_high_and_any(run_builder):
    b = run_builder
    run = _run(
        run_builder,
        novv={"assertions": [b.assertion(PURPOSE, "skip", no_verdict=True, reason="judge returned no verdict")]},
    )
    assert run.scenario_results[0].passed is True   # a skip never fails the scenario (unchanged)
    assert _reasons(run, "critical") == []
    assert _reasons(run, "high") == [f"novv: high assertion had no verdict: {PURPOSE}"]
    assert _reasons(run, "any") == [f"novv: high assertion had no verdict: {PURPOSE}"]


def test_plain_high_skip_and_medium_no_verdict_do_not_count(run_builder):
    b = run_builder
    run = _run(
        run_builder,
        skips={
            "assertions": [
                b.assertion(PURPOSE, "skip", reason="no personal data was requested"),
                b.assertion(WITHDRAWAL, "skip", no_verdict=True, reason="judge returned no verdict"),
            ]
        },
    )
    assert _reasons(run, "high") == []
    assert _reasons(run, "any") == []


def test_high_no_verdict_after_a_high_failure_is_not_a_second_line(run_builder):
    b = run_builder
    run = _run(
        run_builder,
        two={"assertions": [b.assertion(PURPOSE, "fail"), b.assertion("rbi_kyc.no_unauthorized_commitment", "skip", severity="high", no_verdict=True)]},
    )
    assert _reasons(run, "high") == [f"two: high assertion failed: {PURPOSE}"]


# ---------------------------------------------------------------------------------------------
# defaults: config.toml, init, the CLI
# ---------------------------------------------------------------------------------------------


def test_toml_defaults_fail_on_is_high():
    from impulse_evals.config import TomlConfig, TomlDefaults

    assert TomlDefaults().fail_on == "high"
    assert TomlConfig().defaults.fail_on == "high"


def test_toml_without_fail_on_gives_high(impulse_project):
    from impulse_evals.config import load_project_toml

    impulse_project.write_toml('[defaults]\nreporter = "all"\n')
    assert load_project_toml().defaults.fail_on == "high"


def test_init_writes_fail_on_high(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--profile", "free", "--non-interactive", "--language", "en-US")
    assert result.exit_code == 0, cli.text(result)
    text = (impulse_project.root / ".impulse" / "config.toml").read_text(encoding="utf-8")
    assert 'fail_on = "high"' in text
    assert 'fail_on = "critical"' not in text


def _password_bot_config(project, llm_router, **fields):
    """One scenario judged on UNIV-019; the mocked judge fails it (a credential request)."""
    project.write_agent()
    values = {"assertions": [], "metrics": ["UNIV-019"], "max_turns": 1}
    values.update(fields)
    llm_router.on_judge(llm_router.verdict("fail", reason="the agent asked for the user's password"), criterion="UNIV-019")
    return project.write_config([project.scenario("password_request", **values)])


def test_run_without_toml_exits_1_on_a_poor_univ019_by_default(impulse_project, llm_router, cli):
    config = _password_bot_config(impulse_project, llm_router)
    result = cli.invoke("run", config)
    assert result.exit_code == 1, cli.text(result)
    text = cli.text(result)
    assert "--fail-on high: exit 1" in text
    assert "password_request: UNIV-019 is poor (counts as severity high)" in text


def test_run_without_toml_exits_1_on_a_high_assertion_by_default(impulse_project, llm_router, cli):
    impulse_project.write_agent()
    llm_router.on_judge(llm_router.verdict("fail", reason="scripted failure"), criterion=PURPOSE)
    config = impulse_project.write_config(
        [impulse_project.scenario("purpose", assertions=[CONSENT, PURPOSE, WITHDRAWAL], metrics=["UNIV-013"], max_turns=1)]
    )
    assert cli.invoke("run", config).exit_code == 1
    assert cli.invoke("run", config, "--fail-on", "critical").exit_code == 0


def test_toml_fail_on_critical_still_lets_a_poor_univ019_exit_0(impulse_project, llm_router, cli):
    config = _password_bot_config(impulse_project, llm_router)
    impulse_project.write_toml('[defaults]\nfail_on = "critical"\n')
    assert cli.invoke("run", config).exit_code == 0
    assert cli.invoke("run", config, "--fail-on", "high").exit_code == 1


def test_run_help_names_high_as_the_default(cli):
    result = cli.invoke("run", "--help")
    assert result.exit_code == 0
    assert "else high" in cli.text(result).replace("\n", " ")


def test_assert_no_critical_keeps_its_critical_rule(run_builder):
    from impulse_evals.testing import assert_no_critical

    b = run_builder
    high_only = _run(run_builder, h={"metrics": [b.metric("UNIV-019", 0.0)], "assertions": [b.assertion(PURPOSE, "fail")]})
    assert_no_critical(high_only)   # no critical failure, no error: passes
    critical = _run(run_builder, c={"assertions": [b.assertion(CONSENT, "fail")]})
    with pytest.raises(AssertionError):
        assert_no_critical(critical)


# ---------------------------------------------------------------------------------------------
# impulse compare: UNIV-019 newly poor is a high regression
# ---------------------------------------------------------------------------------------------


def test_compare_univ019_newly_poor_is_a_high_regression(run_builder):
    from impulse_evals.compare import compare_runs

    b = run_builder
    run_a = _run(run_builder, urgency={"metrics": [b.metric("UNIV-019", 1.0)]})
    run_b = _run(run_builder, urgency={"metrics": [b.metric("UNIV-019", 0.0)]})
    compare = compare_runs(run_a, run_b)
    assert compare.high_regressions == ["urgency"]
    assert compare.critical_regressions == []
    assert [diff.scenario_name for diff in compare.regressions] == ["urgency"]


def test_compare_univ019_poor_in_both_is_not_a_high_regression(run_builder):
    from impulse_evals.compare import compare_runs

    b = run_builder
    run_a = _run(run_builder, urgency={"metrics": [b.metric("UNIV-019", 0.0)]})
    run_b = _run(run_builder, urgency={"metrics": [b.metric("UNIV-019", 0.0)]})
    assert compare_runs(run_a, run_b).high_regressions == []


def test_compare_command_fail_on_high_exits_1_on_univ019(impulse_project, run_builder, cli):
    import json

    b = run_builder
    run_a = _run(run_builder, urgency={"metrics": [b.metric("UNIV-019", 1.0)]})
    run_b = b.run([b.scenario_result("urgency", metrics=[b.metric("UNIV-019", 0.0)])], run_id="20261004T130000Z-b7e2d1")
    for run in (run_a, run_b):
        folder = impulse_project.root / ".impulse" / "runs" / run.run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.json").write_text(json.dumps(run.model_dump(mode="json")), encoding="utf-8")
    assert cli.invoke("compare", run_a.run_id, run_b.run_id).exit_code == 0            # default critical
    result = cli.invoke("compare", run_a.run_id, run_b.run_id, "--fail-on", "high")
    assert result.exit_code == 1, cli.text(result)
    assert "High regressions: urgency" in cli.text(result)
