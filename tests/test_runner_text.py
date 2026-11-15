# SPDX-License-Identifier: Apache-2.0
"""Phase 6: runner.run in text mode (SPEC.md sections 7, 9, 10, 12, 13, 15, 17).

``await runner.run(config, *, dry_run=False, concurrency=1, adapter_factory=None)`` takes the
``ProjectConfig`` from ``load_config``, returns a ``RunResult``, and writes no run files. These tests
check the returned ``RunResult`` and the console only (report HTML: ``test_reporter_fields.py``).
Fake adapters come from the ``fake_adapter`` fixture; LLM calls go to ``mock_llm`` through
``llm_router`` (judge verdicts by criterion, simulator turns by persona).
"""

import re

import pytest

CONSENT = "dpdp.consent_before_pii"
PURPOSE = "dpdp.purpose_disclosure"
WITHDRAWAL = "dpdp.withdrawal_path"
EMPTY_REASON = "Judge returned empty (safety filter or rate limit)"
INVALID_REASON = "Judge returned invalid JSON"
WARNING_PREFIX = "Critical checks not evaluated (judge returned no verdict): "
UNKNOWN_SUMMARY = "Unknown; infer the agent's purpose from its replies."
SAMPLE_TOOL_CALL = {
    "name": "book_slot",
    "arguments": {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"},
    "result": {"success": True},
    "success": True,
}


def _m():
    from impulse_evals import models

    return models


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


def _by_name(result):
    return {scenario.scenario_name: scenario for scenario in result.scenario_results}


def _metrics(scenario_result):
    return {metric.metric_id: metric for metric in scenario_result.metric_results}


def _assertions(scenario_result):
    return {score.rule_id: score for score in scenario_result.assertion_results}


def _user_texts(scenario_result):
    return [turn.text for turn in scenario_result.transcript.turns if turn.speaker == "user"]


def _critical_error(rule_id):
    return f"critical check {rule_id} could not be evaluated: judge returned no verdict"


# ============================================================================================
# End-to-end text run with the sample agent
# ============================================================================================


@pytest.mark.asyncio
async def test_end_to_end_text_run_returns_valid_run_result_and_writes_no_files(impulse_project, llm_router):
    m = _m()
    project = impulse_project
    entrypoint = project.write_agent()  # copy of examples/sample_agent.py
    project.write_config(
        [
            project.scenario(
                "clinic_booking",
                opening_user_message="Hello, I would like an appointment on Tuesday.",
                assertions=[CONSENT],
                metrics=["UNIV-001", "UNIV-013"],
                expected_tool_calls=[{"name": "book_slot", "required_arguments": ["date", "phone"]}],
            ),
            project.scenario("clinic_question", language="hi-IN", accent="north-indian", metrics=["UNIV-001"]),
        ]
    )
    llm_router.script(project.persona("clinic_booking"), "Please book me for Tuesday morning.", "<END_CALL>")
    cfg = project.load()
    before = project.files()

    result = await _run(cfg)

    assert project.files() == before, "runner.run must not write any file in text mode"
    assert not (project.root / ".impulse").exists()
    assert isinstance(result, m.RunResult)
    assert m.RunResult.model_validate_json(result.model_dump_json()) == result
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", result.run_id)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z", result.timestamp)
    assert result.agent_entrypoint == entrypoint
    assert result.agent_type == "chat"
    assert result.agent_domain == "healthcare"
    assert result.judge_model == "ollama/judge-primary"
    assert result.voice_mode is False
    assert result.total_scenarios == 2
    assert [s.scenario_name for s in result.scenario_results] == ["clinic_booking", "clinic_question"]
    for scenario_result in result.scenario_results:
        assert scenario_result.error is None
        assert scenario_result.passed == m.compute_passed(scenario_result)
    assert result.passed == sum(1 for s in result.scenario_results if s.passed)
    assert result.failed == result.total_scenarios - result.passed
    assert result.overall_pass_rate == pytest.approx(result.passed / result.total_scenarios)

    booking = _by_name(result)["clinic_booking"]
    turns = booking.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert turns[0].text == "Hello, I would like an appointment on Tuesday."
    assert turns[1].text == "You said: Hello, I would like an appointment on Tuesday."
    assert turns[2].text == "Please book me for Tuesday morning."
    assert turns[3].tool_calls == [m.ToolCall(**SAMPLE_TOOL_CALL)]
    assert booking.transcript.scenario_name == "clinic_booking"
    assert booking.transcript.language == "en-IN"
    assert booking.transcript.voice_mode is False
    # The config.yaml system prompt wins over sample_agent.SYSTEM_PROMPT.
    assert booking.transcript.agent_system_prompt == project.system_prompt
    metrics = _metrics(booking)
    assert {"UNIV-001", "UNIV-013", "UNIV-025"} <= set(metrics)
    assert metrics["UNIV-025"].label == "good"
    assert _assertions(booking)[CONSENT].result == "pass"

    question = _by_name(result)["clinic_question"]
    assert question.transcript.language == "hi-IN"
    assert question.transcript.accent == "north-indian"
    assert {cohort.cohort_key for cohort in result.cohorts} == {"en-IN/default", "hi-IN/north-indian"}


@pytest.mark.asyncio
async def test_metrics_selected_include_automatic_univ025(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("tool_expectation", metrics=["UNIV-013"], forbidden_tool_calls=["cancel_slot"])]
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    metrics = _metrics(result.scenario_results[0])
    assert set(metrics) == {"UNIV-013", "UNIV-025"}
    assert metrics["UNIV-025"].label == "good"


# ============================================================================================
# Adapter warnings and metadata; config and judge warnings
# ============================================================================================


@pytest.mark.asyncio
async def test_adapter_warnings_appear_once_and_metadata_is_merged(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config([project.scenario("first_call"), project.scenario("second_call")])
    warning = (
        "Pipecat bot reports function calls without names or arguments. Set the RTVI function call "
        "report level to FULL so UNIV-023 can check tool arguments."
    )
    fake_adapter.warnings = [warning]
    pipecat_metrics = [
        {"turn_index": 2, "kind": "ttfb", "processor": "FakeLLMService#0", "model": "fake-llm", "value_ms": 412.0}
    ]
    fake_adapter.metadata = {
        "first_call": {"fake_adapter": "first", "pipecat_metrics": pipecat_metrics},
        "second_call": {"fake_adapter": "second"},
    }
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.warnings.count(warning) == 1
    by_name = _by_name(result)
    first_metadata = by_name["first_call"].transcript.metadata
    assert first_metadata["fake_adapter"] == "first"
    assert first_metadata["pipecat_metrics"] == pipecat_metrics
    assert by_name["second_call"].transcript.metadata["fake_adapter"] == "second"


@pytest.mark.asyncio
async def test_config_warnings_are_copied_into_run_warnings(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("unknown_domain_one", domain="insurance"), project.scenario("unknown_domain_two", domain="insurance")]
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.warnings.count("Unknown domain insurance; using general defaults.") == 1
    assert result.agent_domain == "general"


@pytest.mark.asyncio
async def test_metrics_disabled_runs_no_metrics_and_warns(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [
            project.scenario(
                "metrics_off",
                metrics=["UNIV-001", "UNIV-013"],
                assertions=[CONSENT],
                expected_tool_calls=[{"name": "book_slot"}],
            )
        ],
        metrics={"enabled": False},
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.metric_results == []
    assert [score.rule_id for score in scenario_result.assertion_results] == [CONSENT]
    assert "metrics.enabled is false; no metrics will run, including expected tool call checks." in result.warnings
    assert llm_router.judge_calls_for("UNIV-001") == []


@pytest.mark.asyncio
async def test_agent_model_equal_to_judge_model_warns_once(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("first_call"), project.scenario("second_call")],
        agent={"model": "ollama/judge-primary"},
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    expected = (
        "agent.model is the same as the judge model (ollama/judge-primary); "
        "the judge may be biased toward the agent's own outputs."
    )
    assert result.warnings.count(expected) == 1


@pytest.mark.asyncio
async def test_judge_num_ctx_warning_is_copied_into_run_warnings(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("long_prompt")],
        judge={
            "primary_model": "ollama/judge-primary",
            "fast_model": "ollama/judge-fast",
            "primary_options": {"num_ctx": 64},
            "fast_options": {},
        },
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    found = [w for w in result.warnings if w.startswith("Judge prompt is about ")]
    assert len(found) == 1
    assert "over 80% of num_ctx=64 for ollama/judge-primary; the judge may truncate the transcript." in found[0]


@pytest.mark.asyncio
async def test_hosted_model_key_from_config_toml_is_passed_and_never_shown(
    impulse_project, llm_router, fake_adapter, capsys
):
    # SPEC.md section 8.1: key precedence is environment, then config.toml. No key variable is set
    # here, so the key can only come from .impulse/config.toml in the current directory.
    project = impulse_project
    secret = "sk-impulse-test-SECRET-toml-9931"
    hosted = "openai/impulse-test-model"
    project.write_toml(f'[keys]\nOPENAI_API_KEY = "{secret}"\n')
    project.write_config(
        [project.scenario("hosted_call", max_turns=2, assertions=[CONSENT], metrics=["UNIV-013"])],
        judge={"primary_model": hosted, "fast_model": hosted},
        simulator={"model": hosted, "persona_style": "realistic", "mode": "cascade"},
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].error is None
    calls = llm_router.calls("judge") + llm_router.calls("simulator")
    assert llm_router.calls("judge") and llm_router.calls("simulator")
    assert all(call["model"] == hosted for call in calls)
    assert all(call["kwargs"].get("api_key") == secret for call in calls)
    assert secret not in result.model_dump_json()
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


# ============================================================================================
# Scenario errors before evaluation
# ============================================================================================


@pytest.mark.asyncio
async def test_error_at_start_gives_empty_results_and_failed_scenario(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [
            project.scenario("broken_connection", assertions=[CONSENT], metrics=["UNIV-001"]),
            project.scenario("healthy_call", assertions=[CONSENT], metrics=["UNIV-001"]),
        ]
    )
    fake_adapter.start_errors["broken_connection"] = ConnectionError("fake bot refused the connection")
    result = await _run(project.load(), adapter_factory=fake_adapter)
    by_name = _by_name(result)
    broken = by_name["broken_connection"]
    assert broken.error is not None
    assert "fake bot refused the connection" in broken.error
    assert broken.assertion_results == []
    assert broken.metric_results == []
    assert broken.passed is False
    healthy = by_name["healthy_call"]
    assert healthy.error is None
    assert healthy.passed is True
    assert result.failed == 1
    assert result.passed == 1
    # Only the healthy scenario was judged: one metric and one assertion.
    assert len(llm_router.calls("judge")) == 2


@pytest.mark.asyncio
async def test_error_during_conversation_gives_empty_results(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("turn_timeout", max_turns=3, assertions=[CONSENT], metrics=["UNIV-001", "UNIV-013"])]
    )
    llm_router.script(project.persona("turn_timeout"), "Tuesday, please.", "Morning.")
    fake_adapter.send_errors["turn_timeout"] = (2, RuntimeError("Pipecat bot did not finish its turn within 30 s."))
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert "Pipecat bot did not finish its turn within 30 s." in scenario_result.error
    assert scenario_result.assertion_results == []
    assert scenario_result.metric_results == []
    assert scenario_result.passed is False
    assert llm_router.calls("judge") == []


@pytest.mark.asyncio
async def test_agent_exception_is_a_scenario_error(impulse_project, llm_router):
    project = impulse_project
    project.write_agent(
        """
        def respond(user_text, history):
            raise RuntimeError("agent crashed while answering")
        """
    )
    project.write_config([project.scenario("crashing_agent", assertions=[CONSENT], metrics=["UNIV-001"])])
    result = await _run(project.load())
    scenario_result = result.scenario_results[0]
    assert "agent crashed while answering" in scenario_result.error
    assert scenario_result.assertion_results == []
    assert scenario_result.metric_results == []
    assert scenario_result.passed is False


@pytest.mark.asyncio
async def test_entrypoint_import_failure_is_a_config_error_not_a_scenario_error(impulse_project, llm_router):
    # SPEC.md section 8.2: the entrypoint is imported when the generic adapter starts, and an import
    # failure is a config error (CLI exit 2), so runner.run raises it instead of recording an error.
    from impulse_evals.config import ConfigError

    project = impulse_project
    project.write_config(
        [project.scenario("bad_entrypoint")], agent={"entrypoint": "impulse_test_no_such_module_4f2a:respond"}
    )
    with pytest.raises(ConfigError) as info:
        await _run(project.load())
    assert info.value.field == "agent.entrypoint"


# ============================================================================================
# Critical check with no judge verdict (SPEC.md section 13)
# ============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("content, reason", [(None, EMPTY_REASON), ("not json {at all", INVALID_REASON)])
async def test_critical_no_verdict_sets_error_keeps_results_and_warns(
    impulse_project, llm_router, fake_adapter, content, reason
):
    m = _m()
    project = impulse_project
    project.write_config(
        [project.scenario("clinic_intake", assertions=[CONSENT, PURPOSE, WITHDRAWAL], metrics=["UNIV-001", "UNIV-013"])]
    )
    llm_router.on_judge(content, criterion=CONSENT)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]

    assertions = _assertions(scenario_result)
    assert set(assertions) == {CONSENT, PURPOSE, WITHDRAWAL}
    consent = assertions[CONSENT]
    assert consent.result == "skip"
    assert consent.no_verdict is True
    assert consent.severity == "critical"
    assert consent.reason == reason
    assert assertions[PURPOSE].result == "pass"
    assert assertions[WITHDRAWAL].result == "pass"
    metrics = _metrics(scenario_result)
    assert metrics["UNIV-001"].label == "good"
    assert metrics["UNIV-013"].label == "good"

    assert scenario_result.error == _critical_error(CONSENT)
    assert scenario_result.passed is False
    assert m.compute_passed(scenario_result) is False
    assert result.failed == 1
    assert result.critical_failures == []
    assert result.warnings.count(WARNING_PREFIX + f"clinic_intake:{CONSENT}") == 1


@pytest.mark.asyncio
async def test_no_verdict_warning_lists_every_critical_check(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    names = ["first_intake", "second_intake", "third_intake"]
    project.write_config([project.scenario(name, assertions=[CONSENT], metrics=["UNIV-013"]) for name in names])
    llm_router.on_judge(None, criterion=CONSENT, transcript=project.opening("first_intake"))
    llm_router.on_judge(None, criterion=CONSENT, transcript=project.opening("third_intake"))
    result = await _run(project.load(), adapter_factory=fake_adapter)
    by_name = _by_name(result)
    assert by_name["first_intake"].error == _critical_error(CONSENT)
    assert by_name["second_intake"].error is None
    assert by_name["second_intake"].passed is True
    assert by_name["third_intake"].error == _critical_error(CONSENT)
    listed = [w for w in result.warnings if w.startswith(WARNING_PREFIX)]
    assert listed == [WARNING_PREFIX + f"first_intake:{CONSENT}, third_intake:{CONSENT}"]


@pytest.mark.asyncio
async def test_high_and_medium_no_verdict_stay_plain_skips(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("clinic_intake", assertions=[CONSENT, PURPOSE, WITHDRAWAL], metrics=["UNIV-013"])]
    )
    llm_router.on_judge(None, criterion=PURPOSE)
    llm_router.on_judge(None, criterion=WITHDRAWAL)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assertions = _assertions(scenario_result)
    assert assertions[PURPOSE].result == "skip"
    assert assertions[WITHDRAWAL].result == "skip"
    assert assertions[CONSENT].result == "pass"
    assert scenario_result.error is None
    assert scenario_result.passed is True
    assert not [w for w in result.warnings if w.startswith(WARNING_PREFIX)]


# ============================================================================================
# Conversation flow: outbound opener, opening_user_message, END_CALL, max_turns
# ============================================================================================


@pytest.mark.asyncio
async def test_outbound_sends_none_none_first_and_opener_does_not_count(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("outbound_reminder", direction="outbound", max_turns=2, opening_user_message=None)]
    )
    llm_router.script(
        project.persona("outbound_reminder"), "Yes, this is Test User.", "What is this about?", "Tell me more."
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    sends = fake_adapter.sends("outbound_reminder")
    assert sends[0] == (None, None)
    assert sends[1:] == [("Yes, this is Test User.", None), ("What is this about?", None)]
    turns = result.scenario_results[0].transcript.turns
    assert [turn.speaker for turn in turns] == ["agent", "user", "agent", "user", "agent"]
    assert turns[0].text == fake_adapter.OPENER


@pytest.mark.asyncio
async def test_inbound_does_not_wait_for_an_opener(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config([project.scenario("inbound_call", max_turns=1)])
    await _run(project.load(), adapter_factory=fake_adapter)
    assert fake_adapter.sends("inbound_call") == [(project.opening("inbound_call"), None)]


@pytest.mark.asyncio
async def test_opening_user_message_sent_verbatim_without_llm_call(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    opening = "Hi,  I cleared the first round.  Can I schedule my next interview?"
    project.write_config(
        [project.scenario("interview_booking", opening_user_message=opening, max_turns=5, seed=7, domain="recruitment")]
    )
    llm_router.script(project.persona("interview_booking"), "Next Monday works for me.", "<END_CALL>")
    result = await _run(project.load(), adapter_factory=fake_adapter)
    sends = fake_adapter.sends("interview_booking")
    assert sends == [(opening, None), ("Next Monday works for me.", None)]
    scenario_result = result.scenario_results[0]
    assert _user_texts(scenario_result) == [opening, "Next Monday works for me."]
    simulator_calls = llm_router.calls("simulator")
    assert len(simulator_calls) == 2  # turn 2 and the <END_CALL>; none for the opening message
    prompt = llm_router.simulator_prompt(simulator_calls[0])
    assert f"Your persona: {project.persona('interview_booking')}" in prompt
    assert "Style: realistic" in prompt
    assert "Your conversation seed is 7. Produce the same user turns for the same seed and persona." in prompt
    first_contents = llm_router.contents(simulator_calls[0])
    assert opening in first_contents
    assert any(text.startswith("Fake agent reply 1") for text in first_contents)


@pytest.mark.asyncio
async def test_end_call_anywhere_ends_the_call_and_nothing_more_is_sent(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config([project.scenario("short_call", max_turns=6)])
    llm_router.script(project.persona("short_call"), "Okay.", "Thanks, that is all. <END_CALL>", "never sent")
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert [text for text, _ in fake_adapter.sends("short_call")] == [project.opening("short_call"), "Okay."]
    assert _user_texts(result.scenario_results[0]) == [project.opening("short_call"), "Okay."]


@pytest.mark.asyncio
async def test_max_turns_counts_user_turns(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config([project.scenario("long_call", max_turns=2)])
    llm_router.script(project.persona("long_call"), "second", "third", "fourth")
    result = await _run(project.load(), adapter_factory=fake_adapter)
    turns = result.scenario_results[0].transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert _user_texts(result.scenario_results[0]) == [project.opening("long_call"), "second"]


# ============================================================================================
# Where AgentReply fields land; latency profile
# ============================================================================================


@pytest.mark.asyncio
async def test_heard_text_on_user_turn_interrupted_on_agent_turn(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config([project.scenario("hearing_probe", max_turns=2)])
    llm_router.script(project.persona("hearing_probe"), "Tuesday morning.")
    fake_adapter.replies["hearing_probe"] = [
        {"text": "Sure, which day?", "heard_text": "hello this is the caller for hearing probe", "interrupted": True},
        {"text": "Tuesday at 10 is free.", "heard_text": "tuesday morning"},
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    turns = result.scenario_results[0].transcript.turns
    assert turns[0].speaker == "user"
    assert turns[0].heard_text == "hello this is the caller for hearing probe"
    assert turns[0].interrupted is False
    assert turns[1].speaker == "agent"
    assert turns[1].text == "Sure, which day?"
    assert turns[1].heard_text is None
    assert turns[1].interrupted is True
    assert turns[2].heard_text == "tuesday morning"
    assert turns[3].interrupted is False


@pytest.mark.asyncio
async def test_latency_profile_from_agent_turns_excludes_outbound_opener(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("latency_probe", direction="outbound", max_turns=4, opening_user_message=None)]
    )
    llm_router.script(project.persona("latency_probe"), "one", "two", "three", "four")
    fake_adapter.replies["latency_probe"] = [
        {"text": "Opener", "latency_ms": 5000.0},
        {"text": "r1", "latency_ms": 100.0},
        {"text": "r2", "latency_ms": 200.0},
        {"text": "r3", "latency_ms": 300.0},
        {"text": "r4", "latency_ms": 400.0},
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    profile = result.latency_profile
    assert sorted(profile.all_latencies_ms) == [100.0, 200.0, 300.0, 400.0]
    assert profile.p50_ms == pytest.approx(250.0)
    assert profile.p75_ms == pytest.approx(325.0)
    assert profile.p95_ms == pytest.approx(385.0)
    assert profile.p99_ms == pytest.approx(397.0)
    assert profile.is_real_audio is False
    agent_turns = [turn for turn in result.scenario_results[0].transcript.turns if turn.speaker == "agent"]
    assert [turn.latency_ms for turn in agent_turns[1:]] == [100.0, 200.0, 300.0, 400.0]


# ============================================================================================
# System prompt resolution and the simulator's summary of it (SPEC.md sections 9 and 10)
# ============================================================================================


@pytest.mark.asyncio
async def test_summary_made_once_per_system_prompt(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config([project.scenario("first_call"), project.scenario("second_call")])
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert len(llm_router.calls("other")) == 1
    for call in llm_router.calls("simulator"):
        assert f"The agent's purpose is: {llm_router.other_text}" in llm_router.simulator_prompt(call)
    for scenario_result in result.scenario_results:
        assert scenario_result.transcript.agent_system_prompt == project.system_prompt


@pytest.mark.asyncio
async def test_no_system_prompt_uses_fixed_summary_without_llm_call(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config([project.scenario("no_prompt", max_turns=2)], system_prompt=None)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert llm_router.calls("other") == []
    simulator_calls = llm_router.calls("simulator")
    assert simulator_calls
    assert f"The agent's purpose is: {UNKNOWN_SUMMARY}" in llm_router.simulator_prompt(simulator_calls[0])
    assert result.scenario_results[0].transcript.agent_system_prompt is None


@pytest.mark.asyncio
async def test_adapter_prompt_used_only_without_config_prompt(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    fake_adapter.system_prompt = "Prompt exposed by the fake agent."
    project.write_config([project.scenario("adapter_prompt", max_turns=1)], system_prompt=None)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].transcript.agent_system_prompt == "Prompt exposed by the fake agent."


@pytest.mark.asyncio
async def test_config_prompt_wins_over_adapter_prompt(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    fake_adapter.system_prompt = "Prompt exposed by the fake agent."
    project.write_config([project.scenario("config_prompt", max_turns=1)])
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].transcript.agent_system_prompt == project.system_prompt


# ============================================================================================
# Run summary fields
# ============================================================================================


@pytest.mark.asyncio
async def test_metric_summary_and_critical_failures(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [
            project.scenario("passing_call", max_turns=1, assertions=[CONSENT], metrics=["UNIV-003", "UNIV-013"]),
            project.scenario("failing_call", max_turns=1, assertions=[CONSENT], metrics=["UNIV-003", "UNIV-013"]),
        ]
    )
    failing = project.opening("failing_call")
    llm_router.on_judge(llm_router.verdict("fail", reason="rude tone"), criterion="UNIV-013", transcript=failing)
    llm_router.on_judge(llm_router.verdict("fail", reason="no consent asked"), criterion=CONSENT, transcript=failing)
    result = await _run(project.load(), adapter_factory=fake_adapter)

    assert set(result.metric_summary) == {"UNIV-013"}  # UNIV-003 is skipped everywhere (fewer than 4 turns)
    tone = result.metric_summary["UNIV-013"]
    assert tone.metric_id == "UNIV-013"
    assert tone.name == "Tone"
    assert tone.category == "universal"
    assert tone.score == pytest.approx(0.5)
    assert tone.label == "warning"
    assert tone.reason == "mean over 2 scenarios (0 skipped)"
    assert tone.turn_index is None

    failed_consent = _assertions(_by_name(result)["failing_call"])[CONSENT]
    assert failed_consent.result == "fail"
    assert result.critical_failures == [failed_consent]
    assert result.passed == 1
    assert result.failed == 1
    assert result.overall_pass_rate == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_metric_summary_counts_skipped_results_and_means_raw_values(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    fields = {
        "metrics": ["UNIV-003"],
        "expected_tool_calls": [{"name": "book_slot"}],
        "forbidden_tool_calls": ["cancel_slot"],
    }
    project.write_config(
        [project.scenario("long_call", max_turns=2, **fields), project.scenario("short_call", max_turns=1, **fields)]
    )
    llm_router.script(project.persona("long_call"), "Tuesday at ten, please.")
    fake_adapter.replies["long_call"] = [
        {"text": "Booked.", "tool_calls": [{"name": "book_slot", "arguments": {}, "success": True}]}
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    by_name = _by_name(result)
    assert _metrics(by_name["long_call"])["UNIV-003"].skipped is False  # 4 turns
    assert _metrics(by_name["short_call"])["UNIV-003"].skipped is True  # 2 turns

    retention = result.metric_summary["UNIV-003"]
    assert retention.score == pytest.approx(1.0)
    assert retention.label == "good"
    assert retention.raw_value is None
    assert retention.skipped is False
    assert retention.reason == "mean over 1 scenarios (1 skipped)"

    assert _metrics(by_name["long_call"])["UNIV-025"].raw_value == pytest.approx(1.0)
    assert _metrics(by_name["short_call"])["UNIV-025"].raw_value == pytest.approx(0.5)
    tools = result.metric_summary["UNIV-025"]
    assert (tools.name, tools.category) == ("Expected tool calls", "universal")
    assert tools.score == pytest.approx(0.5)
    assert tools.label == "warning"
    assert tools.raw_value == pytest.approx(0.75)
    assert tools.reason == "mean over 2 scenarios (0 skipped)"


@pytest.mark.asyncio
async def test_agent_domain_mixed_for_several_domains(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("clinic_call", domain="healthcare"), project.scenario("loan_call", domain="fsi")]
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.agent_domain == "mixed"


@pytest.mark.asyncio
async def test_results_keep_config_order_with_concurrency(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    names = ["alpha_call", "bravo_call", "charlie_call", "delta_call"]
    project.write_config([project.scenario(name, max_turns=1) for name in names])
    result = await _run(project.load(), concurrency=3, adapter_factory=fake_adapter)
    assert [s.scenario_name for s in result.scenario_results] == names
    assert all(s.error is None for s in result.scenario_results)


@pytest.mark.asyncio
async def test_empty_run_has_zero_pass_rate_and_empty_latency(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config([])
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.total_scenarios == 0
    assert result.overall_pass_rate == 0.0
    assert result.latency_profile.all_latencies_ms == []
    assert result.latency_profile.p50_ms == 0.0
    assert result.latency_profile.p99_ms == 0.0


# ============================================================================================
# Dry run (SPEC.md section 17): zero LLM calls, scripted simulator, judge checks skipped
# ============================================================================================


@pytest.mark.asyncio
async def test_dry_run_makes_no_llm_calls(impulse_project, mock_llm):
    project = impulse_project
    project.write_agent()  # copy of examples/sample_agent.py
    expectation = [{"name": "book_slot", "required_arguments": ["date", "phone"]}]
    project.write_config(
        [
            project.scenario(
                "dry_inbound", max_turns=4, assertions=[CONSENT, PURPOSE], metrics=["UNIV-001"],
                expected_tool_calls=expectation,
            ),
            project.scenario(
                "dry_outbound", direction="outbound", opening_user_message=None, max_turns=4,
                assertions=[CONSENT], metrics=["UNIV-013"], expected_tool_calls=expectation,
            ),
        ],
        judge={"primary_model": "YOUR_PRIMARY_MODEL", "fast_model": "YOUR_FAST_MODEL"},
        simulator={"model": "YOUR_SIMULATOR_MODEL", "persona_style": "realistic", "mode": "cascade"},
        voice_config={
            "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
            "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
        },
    )
    before = project.files()
    result = await _run(project.load(), dry_run=True)

    assert mock_llm.call_count == 0
    assert project.files() == before
    assert result.voice_mode is False  # dry run forces text mode
    by_name = _by_name(result)
    inbound = by_name["dry_inbound"]
    assert _user_texts(inbound) == [project.opening("dry_inbound"), "I'd like to book"]
    outbound = by_name["dry_outbound"]
    assert outbound.transcript.turns[0].speaker == "agent"
    assert _user_texts(outbound) == ["Hello", "I'd like to book"]
    for scenario_result in result.scenario_results:
        assert scenario_result.error is None
        assert scenario_result.transcript.voice_mode is False
        for score in scenario_result.assertion_results:
            assert (score.result, score.reason) == ("skip", "dry run")
        metrics = _metrics(scenario_result)
        assert metrics["UNIV-025"].skipped is False
        assert metrics["UNIV-025"].label == "good"
        for metric_id, metric in metrics.items():
            if metric_id != "UNIV-025":
                assert (metric.skipped, metric.reason) == (True, "dry run")
    assert {score.rule_id for score in inbound.assertion_results} == {CONSENT, PURPOSE}


# ============================================================================================
# Audit batch 1 (AUDIT.md findings 1 and 6). F1: the agent closes the conversation before its
# first turn (AgentClosed with no agent content so far) is a scenario error, same shape as a
# start() error; AgentClosed after an agent turn stays the SPEC.md section 10 normal end. F6:
# literal credentials written into judge / simulator option leaves (config.yaml and config.toml)
# are redacted from scenario errors, run warnings, and metric reasons, while still being sent.
# ============================================================================================

CLOSED_BEFORE_FIRST_TURN = "the agent closed the conversation before its first turn: "
PIPECAT_CLOSE_1006 = "the Pipecat bot closed the connection (close code 1006)"
JUDGE_BEARER = "LITERAL789"
FAST_BEARER = "LITERALFAST987"
SIMULATOR_KEY = "LITERALSIM555"
TOML_BEARER = "LITERALTOML321"
REDACTED_HEADERS_ERROR = (
    "evaluation of UNIV-001 failed: 401 Unauthorized; request headers: {'Authorization': 'Bearer [REDACTED]'}"
)
REDACTED_HEADERS_WARNING = (
    "clinic: evaluation of UNIV-013 failed: 401 Unauthorized; request headers: {'Authorization': 'Bearer [REDACTED]'}"
)


def _agent_closed(message=PIPECAT_CLOSE_1006, reply=None):
    from impulse_evals.adapters.base import AgentClosed

    return AgentClosed(message, reply=reply)


def _close_events(fake_adapter, scenario_name):
    return [e for e in fake_adapter.events if e["scenario"] == scenario_name and e["event"] == "close"]


def _judge_calls_raise(llm_router, make_message):
    """Every judge call raises ``RuntimeError(make_message(call))`` (an HTTP client error that echoes
    the request); every other call keeps the router's usual answer."""
    previous = llm_router.mock.responder

    def responder(call):
        if llm_router.kind(call) == "judge":
            raise RuntimeError(make_message(call))
        return previous(call)

    llm_router.mock.responder = responder


def _echo_headers(call):
    return f"401 Unauthorized; request headers: {call['kwargs'].get('extra_headers')}"


def _assert_absent(secret, result):
    for warning in result.warnings:
        assert secret not in warning, warning
    for scenario_result in result.scenario_results:
        for metric in scenario_result.metric_results:
            assert secret not in (metric.reason or ""), metric.reason
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
async def test_agent_closed_before_first_agent_turn_is_a_scenario_error_inbound(
    impulse_project, llm_router, fake_adapter
):
    project = impulse_project
    name = "hangs_up_before_speaking"
    project.write_config([project.scenario(name, assertions=[CONSENT], metrics=["UNIV-001"])])
    fake_adapter.send_errors[name] = (1, _agent_closed())   # the first (opening) user message: no reply
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert [turn.speaker for turn in scenario_result.transcript.turns] == ["user"]
    assert scenario_result.error == CLOSED_BEFORE_FIRST_TURN + PIPECAT_CLOSE_1006
    assert scenario_result.assertion_results == []
    assert scenario_result.metric_results == []
    assert scenario_result.passed is False
    assert llm_router.calls("judge") == []
    assert result.failed == 1
    assert _close_events(fake_adapter, name), "the adapter is still closed"


@pytest.mark.asyncio
async def test_agent_closed_on_outbound_opener_is_a_scenario_error(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    name = "outbound_dead_on_join"
    project.write_config([project.scenario(name, direction="outbound", assertions=[CONSENT], metrics=["UNIV-001"])])
    fake_adapter.send_errors[name] = (1, _agent_closed())   # send(None, None): the agent never speaks
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert fake_adapter.sends(name) == [(None, None)]
    assert scenario_result.transcript.turns == []
    assert scenario_result.error is not None
    assert scenario_result.error.startswith(CLOSED_BEFORE_FIRST_TURN)
    assert scenario_result.error == CLOSED_BEFORE_FIRST_TURN + PIPECAT_CLOSE_1006
    assert scenario_result.assertion_results == []
    assert scenario_result.metric_results == []
    assert scenario_result.passed is False
    assert llm_router.calls("judge") == []
    assert _close_events(fake_adapter, name)


@pytest.mark.asyncio
async def test_agent_closed_after_an_agent_turn_is_still_a_normal_end(impulse_project, llm_router, fake_adapter):
    # Regression guard (SPEC.md section 10): goodbye, then the agent hangs up on the second send.
    project = impulse_project
    name = "goodbye_then_hangup"
    project.write_config([project.scenario(name, max_turns=3, assertions=[CONSENT], metrics=["UNIV-001"])])
    llm_router.script(project.persona(name), "Thanks, bye.", "Still there?")
    fake_adapter.send_errors[name] = (2, _agent_closed())
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert [turn.speaker for turn in scenario_result.transcript.turns] == ["user", "agent", "user"]
    assert scenario_result.error is None
    assert scenario_result.passed is True
    assert llm_router.calls("judge")
    assert fake_adapter.sends(name) == [(project.opening(name), None), ("Thanks, bye.", None)]


@pytest.mark.asyncio
async def test_literal_option_credentials_are_redacted_from_errors_and_warnings(
    impulse_project, llm_router, fake_adapter
):
    project = impulse_project
    project.write_config(
        # UNIV-001 is a primary-tier metric, UNIV-013 a fast-tier one: both tiers carry a literal.
        [project.scenario("clinic", max_turns=1, metrics=["UNIV-001", "UNIV-013"])],
        judge={
            "primary_model": "ollama/judge-primary",
            "fast_model": "ollama/judge-fast",
            "primary_options": {"extra_headers": {"Authorization": f"Bearer {JUDGE_BEARER}"}},
            "fast_options": {"extra_headers": {"Authorization": f"Bearer {FAST_BEARER}"}},
        },
        simulator={
            "model": "ollama/simulator",
            "persona_style": "realistic",
            "mode": "cascade",
            "options": {"extra_headers": {"X-Api-Key": SIMULATOR_KEY}},
        },
    )
    _judge_calls_raise(llm_router, _echo_headers)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == REDACTED_HEADERS_ERROR
    assert result.warnings.count(REDACTED_HEADERS_WARNING) == 1
    for secret in (JUDGE_BEARER, FAST_BEARER, SIMULATOR_KEY):
        _assert_absent(secret, result)
    assert scenario_result.passed is False
    # The headers still reach the LLM calls unchanged (redaction is for texts only).
    judge_headers = [call["kwargs"].get("extra_headers") for call in llm_router.calls("judge")]
    assert judge_headers[0] == {"Authorization": f"Bearer {JUDGE_BEARER}"}
    assert {"Authorization": f"Bearer {FAST_BEARER}"} in judge_headers


@pytest.mark.asyncio
async def test_literal_toml_option_credentials_are_redacted(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_toml(
        "[model_options.primary]\n"
        f'extra_params = {{ extra_headers = {{ Authorization = "Bearer {TOML_BEARER}" }} }}\n'
    )
    project.write_config([project.scenario("clinic", max_turns=1, metrics=["UNIV-001"])])
    _judge_calls_raise(llm_router, _echo_headers)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == REDACTED_HEADERS_ERROR
    _assert_absent(TOML_BEARER, result)
    judge_calls = llm_router.calls("judge")
    assert judge_calls[0]["kwargs"].get("extra_headers") == {"Authorization": f"Bearer {TOML_BEARER}"}


@pytest.mark.asyncio
async def test_numeric_option_values_are_never_redacted(impulse_project, llm_router, fake_adapter):
    # Only string leaves are secrets: a numeric option echoed by an error survives verbatim
    # (tests/test_small_judge_warning.py relies on "num_ctx=..." staying readable in warnings).
    project = impulse_project
    project.write_toml("[model_options.primary]\nextra_params = { num_ctx = 4096 }\n")
    project.write_config([project.scenario("clinic", max_turns=1, metrics=["UNIV-001"])])
    _judge_calls_raise(llm_router, lambda call: f"401 Unauthorized; request options: num_ctx={call['kwargs'].get('num_ctx')}")
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == "evaluation of UNIV-001 failed: 401 Unauthorized; request options: num_ctx=4096"
    assert "num_ctx=4096" in result.model_dump_json()
    assert "[REDACTED]" not in result.model_dump_json()


# ============================================================================================
# Audit batch 1 review fix-ups (AUDIT.md finding 6, follow-up): the values of the known non-secret
# LiteLLM option keys (custom_llm_provider, api_version, region_name, aws_region_name,
# vertex_project, vertex_location) are never redacted, so an echoed error stays readable; every
# other string leaf (an unknown header or body field) is still redacted (fail closed).
# ============================================================================================

NON_SECRET_OPTIONS = {
    "custom_llm_provider": "openai",
    "api_version": "2024-02-01",
    "region_name": "us-east-1",
    "aws_region_name": "ap-south-1",
    "vertex_project": "impulse-test-project",
    "vertex_location": "asia-south1",
}
NON_SECRET_API_BASE = "https://api.openai.com/v1"
UNKNOWN_HEADER_SECRET = "CUSTOMVALUE77"
UNKNOWN_BODY_SECRET = "TENANTLITERAL55"
ECHOED_PROVIDER_OPTIONS_ERROR = (
    "evaluation of UNIV-001 failed: model openai/ollama/judge-primary at https://api.openai.com/v1 "
    "(api_version 2024-02-01, region us-east-1, aws_region ap-south-1, vertex impulse-test-project/asia-south1); "
    "request headers: {'X-Custom': '[REDACTED]'}; request body: {'tenant': '[REDACTED]'}"
)


def _echo_provider_options(call):
    kwargs = call["kwargs"]
    return (
        f"model {kwargs.get('custom_llm_provider')}/{call['model']} at {kwargs.get('api_base')} "
        f"(api_version {kwargs.get('api_version')}, region {kwargs.get('region_name')}, "
        f"aws_region {kwargs.get('aws_region_name')}, "
        f"vertex {kwargs.get('vertex_project')}/{kwargs.get('vertex_location')}); "
        f"request headers: {kwargs.get('extra_headers')}; request body: {kwargs.get('extra_body')}"
    )


def _toml_inline(mapping):
    """A TOML inline table for a flat mapping of strings."""
    return "{ " + ", ".join(f'{key} = "{value}"' for key, value in mapping.items()) + " }"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["yaml", "toml"])
async def test_known_non_secret_option_values_are_never_redacted(impulse_project, llm_router, fake_adapter, source):
    project = impulse_project
    headers = {"X-Custom": UNKNOWN_HEADER_SECRET}
    body = {"tenant": UNKNOWN_BODY_SECRET}
    judge = None
    if source == "yaml":
        judge = {
            "primary_model": "ollama/judge-primary",
            "fast_model": "ollama/judge-fast",
            "primary_options": dict(
                NON_SECRET_OPTIONS, api_base=NON_SECRET_API_BASE, extra_headers=headers, extra_body=body
            ),
            "fast_options": {},
        }
    else:   # config.toml: the same keys inside [model_options.primary] extra_params
        extra = ", ".join(f'{key} = "{value}"' for key, value in NON_SECRET_OPTIONS.items())
        project.write_toml(
            "[model_options.primary]\n"
            f'api_base = "{NON_SECRET_API_BASE}"\n'
            f"extra_params = {{ {extra}, extra_headers = {_toml_inline(headers)}, extra_body = {_toml_inline(body)} }}\n"
        )
    project.write_config([project.scenario("clinic", max_turns=1, metrics=["UNIV-001"])], judge=judge)
    _judge_calls_raise(llm_router, _echo_provider_options)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == ECHOED_PROVIDER_OPTIONS_ERROR
    for value in NON_SECRET_OPTIONS.values():
        assert value in result.model_dump_json(), value
    for secret in (UNKNOWN_HEADER_SECRET, UNKNOWN_BODY_SECRET):
        _assert_absent(secret, result)
    # Every option still reaches the judge call unchanged.
    kwargs = llm_router.calls("judge")[0]["kwargs"]
    assert {key: kwargs.get(key) for key in NON_SECRET_OPTIONS} == NON_SECRET_OPTIONS
    assert kwargs.get("extra_headers") == headers
    assert kwargs.get("extra_body") == body


# ============================================================================================
# Audit batch 2 (AUDIT.md finding 9): every dry run carries the run warning DRY_RUN_WARNING once
# (generic, pipecat, and livekit adapters; the pipecat / livekit "validated without connecting"
# warning stays as well), so run.json, the report header, and impulse report show that its results
# are not verdicts. A real run never carries it. RunResult gets no new field.
# ============================================================================================

DRY_RUN_WARNING = "Dry run: judge metrics and assertions were skipped; results are not verdicts."
DRY_RUN_REMOTE_WARNING = (
    "Dry run: scenarios run only with the generic adapter; the {adapter} config was validated without connecting."
)
DRY_RUN_MODELS = {
    "judge": {"primary_model": "YOUR_PRIMARY_MODEL", "fast_model": "YOUR_FAST_MODEL"},
    "simulator": {"model": "YOUR_SIMULATOR_MODEL", "persona_style": "realistic", "mode": "cascade"},
}


@pytest.mark.asyncio
async def test_generic_dry_run_has_the_dry_run_warning_once(impulse_project, mock_llm):
    project = impulse_project
    project.write_agent()  # copy of examples/sample_agent.py
    project.write_config(
        [
            project.scenario("dry_one", max_turns=2, assertions=[CONSENT, PURPOSE], metrics=["UNIV-001"]),
            project.scenario("dry_two", direction="outbound", opening_user_message=None, max_turns=2,
                             assertions=[CONSENT], metrics=["UNIV-013"]),
        ],
        **DRY_RUN_MODELS,
    )
    result = await _run(project.load(), dry_run=True)
    assert mock_llm.call_count == 0
    assert result.total_scenarios == 2
    assert result.warnings.count(DRY_RUN_WARNING) == 1, result.warnings
    m = _m()
    assert m.RunResult.model_validate_json(result.model_dump_json()).warnings.count(DRY_RUN_WARNING) == 1


@pytest.mark.asyncio
async def test_dry_run_with_a_fake_adapter_has_the_dry_run_warning_once(impulse_project, mock_llm, fake_adapter):
    project = impulse_project
    project.write_config([project.scenario("dry_fake", max_turns=2, assertions=[CONSENT])], **DRY_RUN_MODELS)
    fake_adapter.warnings = ["fake adapter warning"]
    result = await _run(project.load(), dry_run=True, adapter_factory=fake_adapter)
    assert mock_llm.call_count == 0
    assert result.warnings.count(DRY_RUN_WARNING) == 1, result.warnings
    assert result.warnings.count("fake adapter warning") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["pipecat", "livekit"])
async def test_remote_adapter_dry_run_has_both_dry_run_warnings_once(impulse_project, mock_llm, adapter):
    project = impulse_project
    agent = {"adapter": adapter}
    if adapter == "pipecat":
        agent["pipecat"] = {"mode": "eval", "url": "ws://localhost:7860"}
    else:
        agent["livekit"] = {"url": "wss://livekit.impulse-test.invalid"}
    project.write_config([project.scenario("remote_bot", metrics=["UNIV-013"])], agent=agent, **DRY_RUN_MODELS)
    result = await _run(project.load(), dry_run=True)
    assert mock_llm.call_count == 0
    assert result.scenario_results == []
    assert result.warnings.count(DRY_RUN_WARNING) == 1, result.warnings
    # Regression guard: the existing "validated without connecting" warning stays.
    assert result.warnings.count(DRY_RUN_REMOTE_WARNING.format(adapter=adapter)) == 1, result.warnings


@pytest.mark.asyncio
async def test_real_run_has_no_dry_run_warning(impulse_project, llm_router, fake_adapter):
    # Regression guard (passes before the fix too): only dry runs are marked.
    project = impulse_project
    project.write_config([project.scenario("real_call", max_turns=2, assertions=[CONSENT], metrics=["UNIV-013"])])
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert llm_router.calls("judge")
    assert DRY_RUN_WARNING not in result.warnings
    assert not [warning for warning in result.warnings if warning.startswith("Dry run")], result.warnings


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 42): the adapter's get_metadata() is redacted before it is merged
# into CallTranscript.metadata, with the same redactor as get_warnings(): every string (keys and
# values, recursing into dicts, lists and tuples) has the known key values replaced by [REDACTED];
# numbers, booleans and None are unchanged.
# ============================================================================================

METADATA_SECRET = "sk-impulse-test-SECRET-metadata-4242"   # ANTHROPIC_API_KEY, a standard key variable
METADATA_LITERAL = "LITERALMETA4242"                        # a literal credential in judge.primary_options
SNOWFLAKE_MODEL = "snowflake/impulse-test-model"
SNOWFLAKE_TOKEN = "impulse-test-SECRET-snowflake-jwt-4242"  # SNOWFLAKE_JWT: known only once resolved for a role


@pytest.mark.asyncio
async def test_adapter_metadata_is_redacted_like_warnings(impulse_project, llm_router, fake_adapter, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", METADATA_SECRET)
    project = impulse_project
    name = "leaky_metadata"
    project.write_config(
        [project.scenario(name, max_turns=1, metrics=["UNIV-013"])],
        judge={
            "primary_model": "ollama/judge-primary",
            "fast_model": "ollama/judge-fast",
            "primary_options": {"extra_headers": {"Authorization": f"Bearer {METADATA_LITERAL}"}},
            "fast_options": {},
        },
    )
    pipecat_metrics = [
        {"turn_index": 2, "kind": "ttfb", "processor": "FakeLLMService#0", "model": "fake-llm", "value_ms": 412.0}
    ]
    fake_adapter.warnings = [f"adapter warning carrying {METADATA_SECRET}"]
    fake_adapter.metadata = {
        name: {
            "adapter_note": f"connected with key {METADATA_SECRET}",
            "nested": {
                "token": METADATA_SECRET,
                "headers": {"Authorization": f"Bearer {METADATA_LITERAL}"},
                "count": 3,
                "ok": True,
                "nothing": None,
            },
            "items": [f"key={METADATA_SECRET}", 7, {"deep": METADATA_LITERAL}],
            "pair": (METADATA_SECRET, 1.5),
            f"echo {METADATA_SECRET}": "a key that holds a secret",
            "pipecat_metrics": pipecat_metrics,
        }
    }
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    assert "adapter warning carrying [REDACTED]" in result.warnings   # control: warnings are redacted
    metadata = scenario_result.transcript.metadata
    assert metadata["adapter_note"] == "connected with key [REDACTED]"
    assert metadata["nested"] == {
        "token": "[REDACTED]",
        "headers": {"Authorization": "Bearer [REDACTED]"},
        "count": 3,
        "ok": True,
        "nothing": None,
    }
    assert metadata["items"] == ["key=[REDACTED]", 7, {"deep": "[REDACTED]"}]
    assert list(metadata["pair"]) == ["[REDACTED]", 1.5]
    assert metadata.get("echo [REDACTED]") == "a key that holds a secret", sorted(metadata)
    assert metadata["pipecat_metrics"] == pipecat_metrics   # nothing without a secret changes
    text = result.model_dump_json()
    assert METADATA_SECRET not in text and METADATA_LITERAL not in text


@pytest.mark.asyncio
async def test_adapter_metadata_redaction_knows_the_keys_resolved_for_the_roles(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    # The same redactor as the warnings: it also knows a key resolved for a role from a variable no
    # config entry names and whose name is not a *_API_KEY name (snowflake reads SNOWFLAKE_JWT).
    monkeypatch.setenv("SNOWFLAKE_JWT", SNOWFLAKE_TOKEN)
    project = impulse_project
    name = "role_key_metadata"
    model = SNOWFLAKE_MODEL
    project.write_config(
        [project.scenario(name, max_turns=1, metrics=["UNIV-013"])],
        judge={"primary_model": model, "fast_model": model, "primary_options": {}, "fast_options": {}},
        simulator={"model": model, "persona_style": "realistic", "mode": "cascade", "options": {}},
    )
    fake_adapter.warnings = [f"bot echoed {SNOWFLAKE_TOKEN}"]
    fake_adapter.metadata = {name: {"bot_echo": f"token {SNOWFLAKE_TOKEN}"}}
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    assert "bot echoed [REDACTED]" in result.warnings   # control: the warning redactor knows it
    assert scenario_result.transcript.metadata["bot_echo"] == "token [REDACTED]"
    assert SNOWFLAKE_TOKEN not in result.model_dump_json()


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 30): redaction also replaces, for each known value, its
# URL-encoded form (urllib.parse.quote(value, safe="")), its base64 text at each of the three byte
# alignments (only the characters that depend on the value's bytes alone: "Basic base64(user:key)"
# keeps the user part), and the provider-masked form (a prefix of 3+ characters, 3 or more "*", a
# suffix of 2+ characters; values of 12+ characters). Each match becomes [REDACTED]; nothing else in
# the text changes.
# ============================================================================================

import base64  # noqa: E402
import urllib.parse  # noqa: E402

F30_KEY = "sk-test-0123456789abcdef"            # OPENAI_API_KEY
F30_SPECIAL_KEY = "sk-test/ab+cd=ef0123456789"  # GEMINI_API_KEY; URL-reserved characters
F30_MASKED = ["sk-test-************cdef", "sk-t***ef"]
F30_QUOTED = urllib.parse.quote(F30_SPECIAL_KEY, safe="")


def _b64(text):
    return base64.b64encode(text.encode()).decode()


def _b64_with_value_redacted(prefix, value):
    """``base64(prefix + value)`` with the characters that depend only on ``value``'s bytes replaced."""
    encoded = _b64(prefix + value)
    head = len(prefix.encode())
    group, offset = head // 3 * 4, head % 3
    start = group + -(-8 * offset // 6)
    end = group + (8 * offset + 8 * len(value.encode())) // 6
    return encoded[:start] + "[REDACTED]" + encoded[end:]


F30_MESSAGE = (
    f"Incorrect API key provided: {F30_MASKED[0]} (also {F30_MASKED[1]}); GET /v1/models?key={F30_QUOTED}; "
    f"Authorization: Basic {_b64('impulse-user:' + F30_KEY)}; Proxy-Authorization: Basic {_b64('user:' + F30_SPECIAL_KEY)}; "
    f"X-Raw: {_b64(F30_KEY)}"
)
F30_EXPECTED = (
    "Incorrect API key provided: [REDACTED] (also [REDACTED]); GET /v1/models?key=[REDACTED]; "
    f"Authorization: Basic {_b64_with_value_redacted('impulse-user:', F30_KEY)}; "
    f"Proxy-Authorization: Basic {_b64_with_value_redacted('user:', F30_SPECIAL_KEY)}; "
    "X-Raw: [REDACTED]"
)


@pytest.mark.asyncio
async def test_masked_and_encoded_key_forms_are_redacted_from_scenario_errors_and_warnings(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", F30_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", F30_SPECIAL_KEY)
    project = impulse_project
    project.write_config([project.scenario("clinic", max_turns=1, metrics=["UNIV-001", "UNIV-013"])])
    _judge_calls_raise(llm_router, lambda call: F30_MESSAGE)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == "evaluation of UNIV-001 failed: " + F30_EXPECTED
    assert "clinic: evaluation of UNIV-013 failed: " + F30_EXPECTED in result.warnings, result.warnings
    text = result.model_dump_json()
    for form in F30_MASKED + [F30_QUOTED, _b64(F30_KEY)]:
        assert form not in text, form


@pytest.mark.asyncio
async def test_short_values_and_unrelated_text_are_left_alone(impulse_project, llm_router, fake_adapter, monkeypatch):
    # Regression guard: masked-looking text that is no masked form of a known value, and a run of
    # stars, are not redacted; the plain value still is.
    monkeypatch.setenv("OPENAI_API_KEY", F30_KEY)
    project = impulse_project
    project.write_config([project.scenario("clinic", max_turns=1, metrics=["UNIV-001"])])
    message = f"request req_xyz****uvw failed; password ******** rejected; key {F30_KEY}"
    _judge_calls_raise(llm_router, lambda call: message)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].error == (
        "evaluation of UNIV-001 failed: request req_xyz****uvw failed; password ******** rejected; key [REDACTED]"
    )


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 31): a ConfigError raised while a scenario runs (here: the generic
# adapter's "cannot import module ..." when the user's agent module fails at import with an error
# echoing a key) leaves runner.run with its message redacted by the run's full redactor (the same
# field), which also knows the keys resolved for the roles from provider-specific variables.
# ============================================================================================

GROQ_MODEL = "groq/impulse-test-model"
GROQ_KEY = "gsk_impulse_test_SECRET_groq_42bb88"
IMPORT_TIME_KEY_ECHO = """
import os

raise RuntimeError("GroqException - AuthenticationError: Invalid API Key: " + os.environ[{name!r}])


def respond(user_text, history):
    return "never reached"
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, key_name, secret",
    [
        pytest.param(GROQ_MODEL, "GROQ_API_KEY", GROQ_KEY, id="groq"),
        pytest.param(SNOWFLAKE_MODEL, "SNOWFLAKE_JWT", SNOWFLAKE_TOKEN, id="snowflake"),
    ],
)
async def test_config_error_from_an_agent_import_is_redacted_by_the_runner(
    impulse_project, llm_router, monkeypatch, model, key_name, secret
):
    from impulse_evals.config import ConfigError

    monkeypatch.setenv(key_name, secret)
    project = impulse_project
    project.write_agent(IMPORT_TIME_KEY_ECHO.format(name=key_name))
    project.write_config(
        [project.scenario("import_echo", max_turns=1, metrics=["UNIV-013"])],
        judge={"primary_model": model, "fast_model": model, "primary_options": {}, "fast_options": {}},
        simulator={"model": model, "persona_style": "realistic", "mode": "cascade", "options": {}},
    )
    with pytest.raises(ConfigError) as info:
        await _run(project.load())
    assert info.value.field == "agent.entrypoint"
    assert info.value.message.startswith("cannot import module "), info.value.message
    assert "Invalid API Key: [REDACTED]" in info.value.message, info.value.message
    assert secret not in info.value.message and secret not in str(info.value)
    assert llm_router.mock.calls == []


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 43): every Impulse module that imports litellm sets
# litellm.suppress_debug_info = True right after the import, so LiteLLM's "Give Feedback / Get
# Help" banner (printed to stdout by its exception mapping on any failed call) never reaches the
# console during a run. The flag is forced back to False first (monkeypatch restores it).
# ============================================================================================

LITELLM_BANNER = "Give Feedback / Get Help"


def _litellm_style_failure(call):
    """What litellm's exception mapping does: the banner on stdout unless suppress_debug_info, then raise."""
    import litellm

    if litellm.suppress_debug_info is False:
        print()
        print(f"\033[1;31m{LITELLM_BANNER}: https://github.com/BerriAI/litellm/issues/new\033[0m")
        print("LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'.")
        print()
    raise ConnectionError("model server is not reachable")


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["judge", "simulator", "other"])
async def test_failed_llm_calls_in_a_run_print_no_litellm_banner(
    impulse_project, llm_router, fake_adapter, monkeypatch, capsys, failing
):
    import litellm

    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    previous = llm_router.mock.responder

    def responder(call):
        if llm_router.kind(call) == failing:
            return _litellm_style_failure(call)
        return previous(call)

    llm_router.mock.responder = responder
    project = impulse_project
    project.write_config([project.scenario("llm_down", max_turns=2, metrics=["UNIV-013"])])
    await _run(project.load(), adapter_factory=fake_adapter)
    assert llm_router.calls(failing), f"no {failing} call was made"
    captured = capsys.readouterr()
    assert LITELLM_BANNER not in captured.out + captured.err
    assert litellm.suppress_debug_info is True


# ============================================================================================
# Audit batch 3 review fix-ups (AUDIT.md finding 42, residue): the adapter's get_metadata() is
# redacted as it will be written. Values that model_dump_json writes as text, but that are no str /
# dict / list / tuple (bytes, set, frozenset, a pydantic model such as ToolCall, also nested), are
# converted to their JSON form first (pydantic_core.to_jsonable_python(metadata, fallback=str) or
# equivalent) and redacted like every other string, so no known key reaches run.json.
# ============================================================================================

import json  # noqa: E402

F42_SECRET = "sk-impulse-test-SECRET-metadata-f42-9090"   # OPENAI_API_KEY, set after the fixture


def _f42_metadata(case):
    """``(metadata key, adapter metadata)`` holding ``F42_SECRET`` in a value of the given kind."""
    from impulse_evals.models import ToolCall

    if case == "bytes":
        return "raw_frame", {"raw_frame": f"bot said: key {F42_SECRET}".encode()}
    if case == "set":
        return "tokens", {"tokens": {F42_SECRET, "not a secret"}}
    if case == "frozenset":
        return "tokens", {"tokens": frozenset({f"Bearer {F42_SECRET}"})}
    if case == "pydantic_model":
        call = ToolCall(name="lookup_order", arguments={"api_key": F42_SECRET}, error=f"denied for {F42_SECRET}")
        return "last_tool_call", {"last_tool_call": call}
    # nested: each kind inside the containers the redactor already walks
    return "nested", {
        "nested": {
            "frames": [b"key " + F42_SECRET.encode(), (frozenset({F42_SECRET}),)],
            "calls": [ToolCall(name="lookup_order", arguments={"token": F42_SECRET})],
            "seen": {F42_SECRET},
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["bytes", "set", "frozenset", "pydantic_model", "nested"])
async def test_adapter_metadata_values_written_as_text_are_redacted(
    impulse_project, llm_router, fake_adapter, monkeypatch, case
):
    monkeypatch.setenv("OPENAI_API_KEY", F42_SECRET)
    project = impulse_project
    name = "typed_metadata"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-013"])])
    key, metadata = _f42_metadata(case)
    fake_adapter.warnings = [f"adapter warning carrying {F42_SECRET}"]
    fake_adapter.metadata = {name: metadata}
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    assert "adapter warning carrying [REDACTED]" in result.warnings   # control: warnings are redacted
    text = result.model_dump_json()
    assert F42_SECRET not in text, f"the {case} metadata value reached run.json unredacted"
    written = json.loads(text)["scenario_results"][0]["transcript"]["metadata"]
    assert key in written, sorted(written)                     # the entry is kept, not dropped
    assert "[REDACTED]" in json.dumps(written[key]), written[key]
    _m().RunResult.model_validate_json(text)                   # still a valid run.json


# ============================================================================================
# Audit batch 3 review fix-ups (AUDIT.md finding 31, residue): the redacted copy of a ConfigError
# that leaves runner.run is raised with no unredacted exception reachable from it: neither its
# __context__ nor its __cause__ (walked recursively) holds the key (`raise ... from None` only hides
# the original, which still holds the key, as __context__). The copy keeps the original traceback
# frames (with_traceback), so the traceback still reaches the adapter's start() in
# adapters/generic_adapter.py, where the agent module was imported.
# ============================================================================================

import traceback  # noqa: E402
from pathlib import Path  # noqa: E402


def _exception_chain(exc):
    """``exc`` and every exception reachable from it through ``__cause__`` / ``__context__``."""
    chain, todo = [], [exc]
    while todo:
        current = todo.pop()
        if current is None or any(current is seen for seen in chain):
            continue
        chain.append(current)
        todo.extend([current.__cause__, current.__context__])
    return chain


def _exception_texts(exc):
    texts = [str(exc), repr(exc)] + [str(arg) for arg in getattr(exc, "args", ())]
    for attribute in ("message", "field"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str):
            texts.append(value)
    return texts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, key_name, secret",
    [
        pytest.param(GROQ_MODEL, "GROQ_API_KEY", GROQ_KEY, id="groq"),
        pytest.param(SNOWFLAKE_MODEL, "SNOWFLAKE_JWT", SNOWFLAKE_TOKEN, id="snowflake"),
    ],
)
async def test_redacted_config_error_keeps_no_unredacted_exception_in_its_chain(
    impulse_project, llm_router, monkeypatch, model, key_name, secret
):
    from impulse_evals.config import ConfigError

    monkeypatch.setenv(key_name, secret)
    project = impulse_project
    project.write_agent(IMPORT_TIME_KEY_ECHO.format(name=key_name))
    project.write_config(
        [project.scenario("import_echo", max_turns=1, metrics=["UNIV-013"])],
        judge={"primary_model": model, "fast_model": model, "primary_options": {}, "fast_options": {}},
        simulator={"model": model, "persona_style": "realistic", "mode": "cascade", "options": {}},
    )
    with pytest.raises(ConfigError) as info:
        await _run(project.load())
    exc = info.value
    assert exc.field == "agent.entrypoint"   # control: the redacted copy itself
    assert "Invalid API Key: [REDACTED]" in exc.message, exc.message
    chain = _exception_chain(exc)
    for item in chain:
        leaked = [text for text in _exception_texts(item) if secret in text]
        assert not leaked, (
            f"{type(item).__name__} reachable from the raised ConfigError (__context__ / __cause__) holds the key: "
            f"{[type(link).__name__ for link in chain]}"
        )
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert secret not in formatted
    frames = traceback.extract_tb(exc.__traceback__)
    assert any(
        frame.name == "start" and Path(frame.filename).as_posix().endswith("impulse_evals/adapters/generic_adapter.py")
        for frame in frames
    ), f"the traceback lost the agent-loading frames: {[frame.name for frame in frames]}"
    assert llm_router.mock.calls == []


# ============================================================================================
# Audit batch 3 review fix-ups (AUDIT.md finding 30, performance): masked-form redaction stays fast
# on a long text with many "***" runs. Here a scenario error of about 100 KB with about 12,500 star
# runs and 20 known 64-character keys (each the value of a *_API_KEY variable) is redacted within
# one run that finishes in well under 3 seconds of real time (about 10 s before the fix); the
# redacted error is exactly the text with the two masked key forms in it replaced by [REDACTED] and
# nothing else changed (the star runs stay).
# ============================================================================================

import random  # noqa: E402
import string  # noqa: E402
import time  # noqa: E402

F30_PERF_RUN_LIMIT_S = 3.0
F30_PERF_FILLER = "a *** b "   # a star run between spaces: no key prefix before it, no key suffix after it


def _f30_perf_keys():
    """20 deterministic 64-character letter-and-digit values."""
    rng = random.Random(3030)
    alphabet = string.ascii_letters + string.digits
    return ["".join(rng.choice(alphabet) for _ in range(64)) for _ in range(20)]


F30_PERF_KEYS = _f30_perf_keys()
F30_PERF_MASKED = [
    F30_PERF_KEYS[0][:6] + "*" * 12 + F30_PERF_KEYS[0][-4:],   # the usual provider mask
    F30_PERF_KEYS[7][:3] + "***" + F30_PERF_KEYS[7][-2:],      # the shortest masked form
]
F30_PERF_MESSAGE = (
    F30_PERF_FILLER * 6250
    + f"Incorrect API key provided: {F30_PERF_MASKED[0]}. "
    + F30_PERF_FILLER * 6250
    + f"Also tried {F30_PERF_MASKED[1]} "
    + F30_PERF_FILLER * 10
    + "end of the error"
)
F30_PERF_EXPECTED = F30_PERF_MESSAGE.replace(F30_PERF_MASKED[0], "[REDACTED]").replace(F30_PERF_MASKED[1], "[REDACTED]")


def test_f30_perf_probe_values_are_what_the_test_assumes():
    # Sanity of the test data itself (passes before the fix too).
    assert len(set(F30_PERF_KEYS)) == 20 and all(len(key) == 64 and key.isalnum() for key in F30_PERF_KEYS)
    assert 95_000 <= len(F30_PERF_MESSAGE) <= 110_000
    assert F30_PERF_MESSAGE.count("***") >= 12_000
    assert all(key not in F30_PERF_MESSAGE for key in F30_PERF_KEYS)
    assert F30_PERF_EXPECTED.count("[REDACTED]") == 2


@pytest.mark.asyncio
async def test_masked_form_redaction_of_a_long_error_with_many_star_runs_is_fast(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    for index, key in enumerate(F30_PERF_KEYS):
        monkeypatch.setenv(f"IMPULSE_TEST_F30_PERF_{index:02d}_API_KEY", key)
    project = impulse_project
    name = "long_error"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-013"])])
    cfg = project.load()
    # Warm-up run with a short error: imports and first-call costs are not timed.
    fake_adapter.send_errors[name] = (1, RuntimeError("short warm-up error"))
    warm = await _run(cfg, adapter_factory=fake_adapter)
    assert warm.scenario_results[0].error == "short warm-up error"

    fake_adapter.send_errors[name] = (1, RuntimeError(F30_PERF_MESSAGE))
    started = time.perf_counter()
    result = await _run(cfg, adapter_factory=fake_adapter)
    elapsed = time.perf_counter() - started
    error = result.scenario_results[0].error
    assert error is not None and error.startswith(F30_PERF_FILLER), (error or "")[:200]
    assert error == F30_PERF_EXPECTED, "the redacted error is not the text with only the masked key forms replaced"
    assert elapsed < F30_PERF_RUN_LIMIT_S, (
        f"runner.run took {elapsed:.1f} s to redact a 100 KB error with {F30_PERF_MESSAGE.count('***')} star runs "
        f"and {len(F30_PERF_KEYS)} known keys (limit {F30_PERF_RUN_LIMIT_S:.0f} s)"
    )


# ============================================================================================
# Audit batch 3 round 2 (AUDIT.md finding 42, residue): adapter metadata bytes that are not UTF-8
# are written as base64 in pydantic's URL-safe alphabet ("-" and "_"), and one such value switches
# the other bytes values of that metadata to base64 too. The redactor also knows the URL-safe base64
# forms of every known value, so a key whose standard base64 has "+" or "/" at every byte alignment
# (it holds "~", "?", ">" and non-ASCII characters) is in no decodable form in run.json: every
# base64-looking run of every string in it, decoded in either alphabet at every character offset,
# holds no part of the key beyond two bytes at each end, and the key part becomes [REDACTED].
# ============================================================================================

F42_B64_SECRET = "sk-impulse-test-SECRET-~~~?>>>-éüß-~?>-4242"   # OPENAI_API_KEY
F42_B64_RUN = re.compile(r"[A-Za-z0-9+/_-]{4,}")


def _b64_key_chars(value, offset, encode):
    """The base64 characters (``encode``'s alphabet) that depend only on ``value``'s bytes when it
    starts ``offset`` bytes into a 3-byte group."""
    raw = value.encode("utf-8")
    text = encode(b"\0" * offset + raw).decode("ascii")
    return text[-(-8 * offset // 6): 8 * (offset + len(raw)) // 6]


def _json_strings(value):
    """Every string (mapping keys included) in a ``json.loads`` result."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)


def _base64_decodings(text):
    """The bytes every base64-looking run of ``text`` decodes to, in the standard and in the URL-safe
    alphabet (each run read with "-" / "_" as "+" / "/" and the other way round, which decodes a run
    of either alphabet or of both), starting at each of the four character offsets."""
    decoded = []
    for match in F42_B64_RUN.finditer(text):
        run = match.group()
        for offset in range(4):
            chunk = run[offset:]
            chunk = chunk[: len(chunk) // 4 * 4]
            if not chunk:
                continue
            decoded.append(base64.b64decode(chunk.replace("-", "+").replace("_", "/")))
            decoded.append(base64.urlsafe_b64decode(chunk.replace("+", "-").replace("/", "_")))
    return decoded


def _key_leaks_in_json(text, key):
    """Where ``key`` is readable in the JSON ``text``: as text, or base64-decodable (any part of the
    key bytes beyond the first and last two bytes)."""
    core = key.encode("utf-8")[2:-2]
    leaks = []
    for string_value in _json_strings(json.loads(text)):
        if key in string_value:
            leaks.append(("text", string_value[:80]))
        if any(core in data for data in _base64_decodings(string_value)):
            leaks.append(("base64", string_value[:80]))
    return leaks


def test_f42_base64_probe_values_are_what_the_test_assumes():
    # Sanity of the test data itself (passes before the fix too): the standard-alphabet forms the
    # redactor knew before differ from the URL-safe text pydantic writes, at every alignment.
    for offset in range(3):
        standard = _b64_key_chars(F42_B64_SECRET, offset, base64.b64encode)
        assert "+" in standard or "/" in standard, offset
        assert standard != _b64_key_chars(F42_B64_SECRET, offset, base64.urlsafe_b64encode)
    frame = b"\x80\x81auth " + F42_B64_SECRET.encode("utf-8") + b"\xff"
    encoded = base64.urlsafe_b64encode(frame).decode("ascii")
    assert _key_leaks_in_json(json.dumps({"raw_frame": encoded}), F42_B64_SECRET), "the decoder finds a plain leak"
    assert not _key_leaks_in_json(json.dumps({"raw_frame": encoded[:12] + "[REDACTED]" + encoded[-4:]}), F42_B64_SECRET)


F42_B64_CASES = {
    "non_utf8_frame": ("raw_frame", {"raw_frame": b"\x80\x81auth " + F42_B64_SECRET.encode("utf-8") + b"\xff"}),
    "utf8_frame_next_to_a_non_utf8_one": (
        "ok_frame",
        {"ok_frame": b"key " + F42_B64_SECRET.encode("utf-8"), "bin_frame": b"\xff\xfe"},
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(F42_B64_CASES))
async def test_adapter_metadata_bytes_written_as_url_safe_base64_hold_no_decodable_key(
    impulse_project, llm_router, fake_adapter, monkeypatch, case
):
    monkeypatch.setenv("OPENAI_API_KEY", F42_B64_SECRET)
    project = impulse_project
    name = "binary_metadata"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-013"])])
    key, metadata = F42_B64_CASES[case]
    fake_adapter.warnings = [f"adapter warning carrying {F42_B64_SECRET}"]
    fake_adapter.metadata = {name: metadata}
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    assert "adapter warning carrying [REDACTED]" in result.warnings   # control: the key is a known value
    text = result.model_dump_json()
    leaks = _key_leaks_in_json(text, F42_B64_SECRET)
    assert not leaks, f"the key is readable in run.json ({case}): {leaks}"
    written = json.loads(text)["scenario_results"][0]["transcript"]["metadata"]
    assert key in written, sorted(written)          # the entry is kept, not dropped
    assert "[REDACTED]" in json.dumps(written[key]), written[key]
    _m().RunResult.model_validate_json(text)        # still a valid run.json


# ============================================================================================
# Audit batch 3 round 2 (AUDIT.md finding 31, dry run): a dry run resolves no key through LiteLLM
# (it never imports LiteLLM), yet its redactor knows the environment values of every variable in
# Impulse's own static key tables (standard names, alias targets, explicit names such as
# SNOWFLAKE_JWT) and of every variable whose name ends in _API_KEY, _SECRET_KEY or _JWT. So a
# generic agent whose respond() raises with a role key read from SNOWFLAKE_JWT or SCW_SECRET_KEY
# gives a dry-run scenario error with [REDACTED] in its place, through runner.run itself.
# ============================================================================================

DRY_RUN_RESPOND_KEY_ECHO = """
import os


def respond(user_text, history):
    raise RuntimeError("backend said: Invalid API Key: " + os.environ[{name!r}])
"""
DRY_RUN_ROLE_KEYS = [
    pytest.param("snowflake/impulse-test-model", "SNOWFLAKE_JWT", "impulse-test-SECRET-snowflake-dry-respond-2121", id="snowflake"),
    pytest.param("scaleway/impulse-test-model", "SCW_SECRET_KEY", "impulse-test-SECRET-scaleway-dry-respond-2323", id="scaleway"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model, key_name, secret", DRY_RUN_ROLE_KEYS)
async def test_dry_run_scenario_error_from_respond_never_holds_a_role_key(
    impulse_project, mock_llm, monkeypatch, model, key_name, secret
):
    monkeypatch.setenv(key_name, secret)
    project = impulse_project
    project.write_agent(DRY_RUN_RESPOND_KEY_ECHO.format(name=key_name))
    project.write_config(
        [project.scenario("dry_respond_echo", max_turns=1, metrics=["UNIV-013"])],
        judge={"primary_model": model, "fast_model": model, "primary_options": {}, "fast_options": {}},
        simulator={"model": model, "persona_style": "realistic", "mode": "cascade", "options": {}},
    )
    result = await _run(project.load(), dry_run=True)
    error = result.scenario_results[0].error
    assert error is not None, "the agent's respond() raised: a scenario error"
    assert secret not in error, error
    assert error == "backend said: Invalid API Key: [REDACTED]", error
    text = result.model_dump_json()
    assert secret not in text
    assert mock_llm.call_count == 0   # a dry run makes no LLM call


# ============================================================================================
# Final check N2 (side effect of the AUDIT.md finding 31 fix): a value found ONLY through a wildcard
# name rule (any *_API_KEY variable; in a dry run also *_SECRET_KEY and *_JWT) is redacted only when
# it is at least 12 characters long, so a non-secret placeholder such as MAPS_API_KEY=book no longer
# hides that word everywhere ("Please [REDACTED] a slot"). A variable named explicitly (the standard
# names, config.toml [keys], an api_key_env, Impulse's static key tables such as SNOWFLAKE_JWT, a
# resolved role key) keeps the 4-character rule.
# ============================================================================================

WILDCARD_PLACEHOLDER_ERROR = "Please book a slot; the placeholder text and the calendar stay as they are"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dry_run, variables",
    [
        pytest.param(
            False,
            {"MAPS_API_KEY": "book", "PLACEHOLDER_API_KEY": "placeholder", "CALENDAR_API_KEY": "calendar"},
            id="run_api_key_wildcard",
        ),
        pytest.param(
            True,
            {"MAPS_API_KEY": "book", "BOOKING_SECRET_KEY": "placeholder", "CALENDAR_JWT": "calendar"},
            id="dry_run_wildcards",
        ),
    ],
)
async def test_short_values_found_only_by_a_wildcard_name_rule_are_not_redacted(
    impulse_project, llm_router, fake_adapter, monkeypatch, dry_run, variables
):
    for name, value in variables.items():   # 4 to 11 characters, in variables nothing names explicitly
        monkeypatch.setenv(name, value)
    project = impulse_project
    name = "placeholder_values"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-013"])])
    fake_adapter.send_errors[name] = (1, RuntimeError(WILDCARD_PLACEHOLDER_ERROR))
    result = await _run(project.load(), dry_run=dry_run, adapter_factory=fake_adapter)
    assert result.scenario_results[0].error == WILDCARD_PLACEHOLDER_ERROR, result.scenario_results[0].error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dry_run, variable",
    [
        pytest.param(False, "IMPULSE_TEST_UNNAMED_API_KEY", id="run_api_key"),
        pytest.param(True, "IMPULSE_TEST_UNNAMED_API_KEY", id="dry_run_api_key"),
        pytest.param(True, "IMPULSE_TEST_UNNAMED_SECRET_KEY", id="dry_run_secret_key"),
        pytest.param(True, "IMPULSE_TEST_UNNAMED_JWT", id="dry_run_jwt"),
    ],
)
async def test_values_of_12_characters_found_by_a_wildcard_name_rule_are_still_redacted(
    impulse_project, llm_router, fake_adapter, monkeypatch, dry_run, variable
):
    # Regression guard (passes before the fix too).
    secret = "q7Lm2Xv9Rt4K"   # exactly 12 characters
    monkeypatch.setenv(variable, secret)
    project = impulse_project
    name = "unnamed_key"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-013"])])
    fake_adapter.send_errors[name] = (1, RuntimeError(f"token {secret} rejected"))
    result = await _run(project.load(), dry_run=dry_run, adapter_factory=fake_adapter)
    assert result.scenario_results[0].error == "token [REDACTED] rejected", result.scenario_results[0].error


SHORT_NAMED_KEY = "k3y9zq"   # 6 characters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dry_run, source",
    [
        pytest.param(False, "api_key_env", id="run_api_key_env"),
        pytest.param(True, "api_key_env", id="dry_run_api_key_env"),
        pytest.param(False, "standard_name", id="run_standard_name"),
        pytest.param(False, "config_toml_keys", id="run_config_toml_keys"),
        pytest.param(True, "static_table_name", id="dry_run_static_table_name"),
    ],
)
async def test_short_keys_named_explicitly_are_still_redacted(
    impulse_project, llm_router, fake_adapter, monkeypatch, dry_run, source
):
    # Regression guard (passes before the fix too): named variables keep the 4-character rule.
    project = impulse_project
    judge = None
    if source == "api_key_env":
        monkeypatch.setenv("IMPULSE_TEST_SHORT_JUDGE_TOKEN", SHORT_NAMED_KEY)
        judge = {
            "primary_model": "ollama/judge-primary",
            "fast_model": "ollama/judge-fast",
            "primary_options": {"api_key_env": "IMPULSE_TEST_SHORT_JUDGE_TOKEN"},
            "fast_options": {},
        }
    elif source == "standard_name":
        monkeypatch.setenv("OPENAI_API_KEY", SHORT_NAMED_KEY)
    elif source == "config_toml_keys":
        project.write_toml(f'[keys]\nTOOLBOX_API_KEY = "{SHORT_NAMED_KEY}"\n')
    else:
        monkeypatch.setenv("SNOWFLAKE_JWT", SHORT_NAMED_KEY)
    name = "short_named_key"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-013"])], judge=judge)
    fake_adapter.send_errors[name] = (1, RuntimeError(f"token {SHORT_NAMED_KEY} rejected"))
    result = await _run(project.load(), dry_run=dry_run, adapter_factory=fake_adapter)
    assert result.scenario_results[0].error == "token [REDACTED] rejected", result.scenario_results[0].error
