# SPDX-License-Identifier: Apache-2.0
"""Phase 10: the pytest API in ``impulse_evals.testing`` (SPEC.md sections 15 and 18.1).

``impulse_check(config_path, scenario_name=None) -> RunResult`` runs a config the way ``impulse run``
does (it is a plain, synchronous function for use inside a user's pytest test) and, like the CLI,
writes ``run.json``, ``scenarios/<name>.json`` and ``report.html`` under ``./.impulse/runs/<run_id>/``
(SPEC.md section 15). ``scenario_name`` runs only that scenario; a name that is not in the config
raises an exception naming it; a config error raises ``ConfigError``. ``impulse_check`` never applies
``--fail-on``: it returns the result, and ``assert_no_critical(run_result)`` is the gate.

``assert_no_critical(run_result)`` raises ``AssertionError`` under the same conditions as the default
``--fail-on critical`` (SPEC.md section 17): any failed ``critical`` assertion, or any scenario
``error`` (which includes a critical check the judge could not evaluate). The message names the
scenario and the rule (or the error). Failed high or medium assertions and poor metrics pass.
"""

from pathlib import Path

import pytest

CONSENT = "dpdp.consent_before_pii"
PURPOSE = "dpdp.purpose_disclosure"
WITHDRAWAL = "dpdp.withdrawal_path"


def _testing():
    from impulse_evals import testing

    return testing


def _two_scenarios(project, **fields):
    project.write_agent()  # copy of examples/sample_agent.py
    return project.write_config(
        [
            project.scenario("first_call", assertions=[CONSENT], metrics=["UNIV-013"], max_turns=1, **fields),
            project.scenario("second_call", assertions=[CONSENT], metrics=["UNIV-013"], max_turns=1, **fields),
        ]
    )


# ============================================================================================
# impulse_check
# ============================================================================================


def test_impulse_check_runs_every_scenario_and_writes_run_files(impulse_project, llm_router):
    from impulse_evals.models import RunResult

    config = _two_scenarios(impulse_project)
    result = _testing().impulse_check(str(config))
    assert isinstance(result, RunResult)
    assert [s.scenario_name for s in result.scenario_results] == ["first_call", "second_call"]
    assert result.total_scenarios == 2
    run_dir = impulse_project.only_run_dir()
    assert run_dir.name == result.run_id
    assert RunResult.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8")) == result
    assert (run_dir / "scenarios" / "first_call.json").is_file()
    assert (run_dir / "scenarios" / "second_call.json").is_file()
    assert (run_dir / "report.html").is_file()


def test_impulse_check_accepts_a_path_object(impulse_project, llm_router):
    config = _two_scenarios(impulse_project)
    result = _testing().impulse_check(Path(config))
    assert result.total_scenarios == 2


def test_impulse_check_runs_only_the_named_scenario(impulse_project, llm_router):
    config = _two_scenarios(impulse_project)
    result = _testing().impulse_check(config, scenario_name="second_call")
    assert [s.scenario_name for s in result.scenario_results] == ["second_call"]
    assert result.total_scenarios == 1
    first_persona = impulse_project.persona("first_call")
    assert all(llm_router.persona_of(call) != first_persona for call in llm_router.calls("simulator"))
    for call in llm_router.calls("judge"):
        _criterion, transcript = llm_router.judge_sections(call)
        assert impulse_project.opening("first_call") not in transcript


def test_impulse_check_unknown_scenario_name_raises(impulse_project, llm_router):
    config = _two_scenarios(impulse_project)
    with pytest.raises(Exception) as excinfo:
        _testing().impulse_check(config, scenario_name="no_such_scenario")
    assert "no_such_scenario" in str(excinfo.value)


def test_impulse_check_config_error_raises_config_error(impulse_project, llm_router):
    from impulse_evals.config import ConfigError

    project = impulse_project
    project.write_agent()
    config = project.write_config([project.scenario("bad_metric", metrics=["UNIV-999"])])
    with pytest.raises(ConfigError):
        _testing().impulse_check(config)


def test_impulse_check_returns_failing_runs_without_raising(impulse_project, llm_router):
    config = _two_scenarios(impulse_project)
    llm_router.on_judge(llm_router.verdict("fail", reason="asked for the phone first"), criterion=CONSENT)
    result = _testing().impulse_check(config)
    assert len(result.critical_failures) == 2
    assert all(not scenario.passed for scenario in result.scenario_results)


def test_impulse_check_then_assert_no_critical(impulse_project, llm_router):
    config = _two_scenarios(impulse_project)
    llm_router.on_judge(llm_router.verdict("fail", reason="asked for the phone first"), criterion=CONSENT)
    result = _testing().impulse_check(config, scenario_name="first_call")
    with pytest.raises(AssertionError) as excinfo:
        _testing().assert_no_critical(result)
    assert CONSENT in str(excinfo.value)
    assert "first_call" in str(excinfo.value)


def test_impulse_check_passing_run_passes_assert_no_critical(impulse_project, llm_router):
    config = _two_scenarios(impulse_project)
    result = _testing().impulse_check(config)
    assert result.critical_failures == []
    assert _testing().assert_no_critical(result) is None


# ============================================================================================
# assert_no_critical on hand-built results
# ============================================================================================


def _run(run_builder, *scenario_results):
    return run_builder.run(list(scenario_results))


def test_assert_no_critical_raises_on_a_critical_failure(run_builder):
    run = _run(
        run_builder,
        run_builder.scenario_result("ok_call", assertions=[(CONSENT, "pass")]),
        run_builder.scenario_result("bad_call", assertions=[(CONSENT, "fail"), (PURPOSE, "pass")]),
    )
    with pytest.raises(AssertionError) as excinfo:
        _testing().assert_no_critical(run)
    assert CONSENT in str(excinfo.value)
    assert "bad_call" in str(excinfo.value)


def test_assert_no_critical_raises_on_a_scenario_error(run_builder):
    run = _run(run_builder, run_builder.scenario_result("broken_call", error="agent crashed while answering"))
    with pytest.raises(AssertionError) as excinfo:
        _testing().assert_no_critical(run)
    assert "broken_call" in str(excinfo.value)
    assert "agent crashed while answering" in str(excinfo.value)


def test_assert_no_critical_raises_on_a_critical_check_without_verdict(run_builder):
    error = f"critical check {CONSENT} could not be evaluated: judge returned no verdict"
    scenario = run_builder.scenario_result(
        "unjudged_call",
        assertions=[(CONSENT, "skip", {"no_verdict": True, "reason": "Judge returned invalid JSON"})],
        error=error,
    )
    with pytest.raises(AssertionError) as excinfo:
        _testing().assert_no_critical(_run(run_builder, scenario))
    assert CONSENT in str(excinfo.value)


@pytest.mark.parametrize(
    "assertions, metrics",
    [
        pytest.param([(CONSENT, "pass"), (PURPOSE, "fail")], [], id="high_failure"),
        pytest.param([(CONSENT, "pass"), (WITHDRAWAL, "fail")], [], id="medium_failure"),
        pytest.param([(CONSENT, "pass")], [("UNIV-013", 0.0), ("UNIV-025", 0.0)], id="poor_metrics"),
        pytest.param([(CONSENT, "skip")], [("UNIV-001", 0.0, {"skipped": True})], id="skips"),
        pytest.param([], [], id="nothing_listed"),
    ],
)
def test_assert_no_critical_passes_without_critical_failures_or_errors(run_builder, assertions, metrics):
    run = _run(run_builder, run_builder.scenario_result("call", assertions=assertions, metrics=metrics))
    assert _testing().assert_no_critical(run) is None


def test_assert_no_critical_passes_an_empty_run(run_builder):
    assert _testing().assert_no_critical(_run(run_builder)) is None
