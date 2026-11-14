# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 1: data models (SPEC.md section 7) and the scenario pass logic (compute_passed)."""

import pytest

DISCLAIMER = (
    "Technical signal only. Not legal certification. "
    "Consult qualified counsel before relying on this result "
    "for regulatory purposes."
)

EXPECTED_FIELDS = {
    "ToolCall": {"name", "arguments", "result", "success", "error"},
    "Turn": {
        "speaker", "text", "timestamp_ms", "end_ms", "audio_path", "latency_ms", "interrupted",
        "stt_confidence", "heard_text", "intent_confidence", "classified_intent", "tool_calls",
    },
    "CallTranscript": {
        "turns", "duration_ms", "metadata", "scenario_name", "language", "accent", "voice_mode",
        "recording_path", "agent_system_prompt",
    },
    "ExpectedToolCall": {
        "name", "required_arguments", "argument_values", "must_succeed", "min_count", "max_count",
    },
    "AgentTool": {"name", "description"},
    "InjectEvent": {"at_turn", "type", "tool", "error_message", "silence_ms", "note"},
    "Scenario": {
        "name", "language", "accent", "domain", "direction", "max_turns", "persona",
        "opening_user_message", "expected_outcome", "key_terms", "code_switching", "assertions",
        "metrics", "inject_events", "seed", "expected_tool_calls", "forbidden_tool_calls",
        "noise",   # noise simulation (experimental, 0.1.0): superseded field set, see DECISIONS.md
    },
    "ScoreResult": {
        "rule_id", "regulation_ref", "result", "severity", "reason", "turn_index", "confidence",
        "no_verdict", "disclaimer",
    },
    "MetricResult": {
        "metric_id", "name", "category", "score", "raw_value", "label", "reason", "threshold_warn",
        "threshold_fail", "turn_index", "requires_audio", "is_proxy", "skipped", "informational",
    },
    "WordTiming": {"word", "start_ms", "end_ms", "confidence"},
    "FillerWord": {"word", "start_ms"},
    "STTResult": {"text", "confidence", "words", "filler_words"},
    "LatencyProfile": {"p50_ms", "p75_ms", "p95_ms", "p99_ms", "all_latencies_ms", "is_real_audio"},
    "CohortResult": {
        "cohort_key", "language", "accent", "total", "passed", "failed", "pass_rate", "metric_means",
        "mean_stt_confidence", "gap_vs_best", "gap_warning", "stt_warning",
    },
    "ScenarioResult": {
        "scenario_name", "transcript", "assertion_results", "metric_results", "passed", "error",
        "inject_events",
    },
    "RunResult": {
        "run_id", "timestamp", "agent_entrypoint", "agent_type", "agent_domain", "judge_model",
        "voice_mode", "tts_provider", "stt_provider", "total_scenarios", "passed", "failed",
        "overall_pass_rate", "scenario_results", "cohorts", "critical_failures", "metric_summary",
        "latency_profile", "warnings",
    },
    "ScenarioDiff": {"scenario_name", "status_change", "metric_deltas", "assertion_changes", "notes"},
    "CompareResult": {
        "run_id_a", "run_id_b", "timestamp_a", "timestamp_b", "overall_pass_rate_delta",
        "regressions", "improvements", "unchanged_pass", "unchanged_fail", "added_scenarios",
        "removed_scenarios", "metric_summary_deltas", "critical_regressions",
        "check_regressions", "high_regressions",
    },
}

EXPECTED_REQUIRED = {
    "ToolCall": {"name", "arguments"},
    "Turn": {"speaker", "text", "timestamp_ms"},
    "CallTranscript": {"turns", "duration_ms", "metadata", "scenario_name", "language"},
    "ExpectedToolCall": {"name"},
    "AgentTool": {"name"},
    "InjectEvent": {"at_turn", "type"},
    "Scenario": {"name", "language", "persona"},
    "ScoreResult": {"rule_id", "regulation_ref", "result", "severity", "reason"},
    "MetricResult": {"metric_id", "name", "category", "score", "label", "reason"},
    "WordTiming": {"word", "start_ms", "end_ms", "confidence"},
    "FillerWord": {"word", "start_ms"},
    "STTResult": {"text"},
    "LatencyProfile": {"p50_ms", "p75_ms", "p95_ms", "p99_ms", "all_latencies_ms"},
    "CohortResult": {
        "cohort_key", "language", "total", "passed", "failed", "pass_rate", "metric_means",
        "gap_vs_best", "gap_warning",
    },
    "ScenarioResult": {"scenario_name", "transcript", "assertion_results", "metric_results", "passed"},
    "RunResult": {
        "run_id", "timestamp", "agent_entrypoint", "agent_type", "agent_domain", "judge_model",
        "total_scenarios", "passed", "failed", "overall_pass_rate", "scenario_results", "cohorts",
        "critical_failures", "metric_summary", "latency_profile",
    },
    "ScenarioDiff": {"scenario_name", "status_change", "metric_deltas", "assertion_changes", "notes"},
    "CompareResult": EXPECTED_FIELDS["CompareResult"] - {"check_regressions", "high_regressions"},
}


def _m():
    from impulse_evals import models

    return models


def _validation_error():
    import pydantic

    return pydantic.ValidationError


def _turn(speaker="user", text="hello", timestamp_ms=0, **kwargs):
    return _m().Turn(speaker=speaker, text=text, timestamp_ms=timestamp_ms, **kwargs)


def _transcript(turns=None, **kwargs):
    return _m().CallTranscript(
        turns=turns if turns is not None else [_turn("user", "hi", 0), _turn("agent", "hello", 500)],
        duration_ms=1000,
        metadata={},
        scenario_name="clinic_appointment_booking",
        language="en-IN",
        **kwargs,
    )


def _metric(metric_id="UNIV-001", label="good", score=1.0, **kwargs):
    return _m().MetricResult(
        metric_id=metric_id,
        name="Task completion",
        category="universal",
        score=score,
        label=label,
        reason="test reason",
        **kwargs,
    )


def _score(result="pass", severity="critical", **kwargs):
    return _m().ScoreResult(
        rule_id="dpdp.consent_before_pii",
        regulation_ref="DPDP Act 2023 (notice and consent provisions)",
        result=result,
        severity=severity,
        reason="test reason",
        **kwargs,
    )


def _scenario_result(metrics=(), assertions=(), error=None, passed=True, **kwargs):
    return _m().ScenarioResult(
        scenario_name="clinic_appointment_booking",
        transcript=_transcript(),
        assertion_results=list(assertions),
        metric_results=list(metrics),
        passed=passed,
        error=error,
        **kwargs,
    )


def _latency():
    return _m().LatencyProfile(
        p50_ms=100.0, p75_ms=200.0, p95_ms=300.0, p99_ms=400.0, all_latencies_ms=[100.0, 400.0]
    )


def _run_result(**kwargs):
    values = dict(
        run_id="20261004T120000Z-a3f9c2",
        timestamp="2026-10-04T12:00:00Z",
        agent_entrypoint="sample_agent:respond",
        agent_type="voice",
        agent_domain="healthcare",
        judge_model="ollama/judge-primary",
        total_scenarios=1,
        passed=1,
        failed=0,
        overall_pass_rate=1.0,
        scenario_results=[_scenario_result(metrics=[_metric()], assertions=[_score()])],
        cohorts=[],
        critical_failures=[],
        metric_summary={"UNIV-001": _metric()},
        latency_profile=_latency(),
    )
    values.update(kwargs)
    return _m().RunResult(**values)


# --------------------------------------------------------------------------------------------
# Exact field names
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(EXPECTED_FIELDS))
def test_exact_field_names(model_name):
    cls = getattr(_m(), model_name)
    assert set(cls.model_fields) == EXPECTED_FIELDS[model_name]


@pytest.mark.parametrize("model_name", sorted(EXPECTED_REQUIRED))
def test_required_fields(model_name):
    cls = getattr(_m(), model_name)
    required = {name for name, field in cls.model_fields.items() if field.is_required()}
    assert required == EXPECTED_REQUIRED[model_name]


# --------------------------------------------------------------------------------------------
# ToolCall, ExpectedToolCall, AgentTool, InjectEvent
# --------------------------------------------------------------------------------------------


def test_tool_call_fields_and_defaults():
    call = _m().ToolCall(name="book_slot", arguments={"date": "2026-01-01"})
    assert call.name == "book_slot"
    assert call.arguments == {"date": "2026-01-01"}
    assert call.result is None
    assert call.success is None
    assert call.error is None


def test_tool_call_full_round_trip():
    m = _m()
    call = m.ToolCall(
        name="book_slot",
        arguments={"date": "2026-01-01", "phone": "0000000000"},
        result={"success": False, "error": "slot taken"},
        success=False,
        error="slot taken",
    )
    assert m.ToolCall.model_validate_json(call.model_dump_json()) == call


def test_tool_call_requires_arguments():
    with pytest.raises(_validation_error()):
        _m().ToolCall(name="book_slot")


def test_expected_tool_call_defaults():
    expected = _m().ExpectedToolCall(name="book_slot")
    assert expected.required_arguments == []
    assert expected.argument_values == {}
    assert expected.must_succeed is True
    assert expected.min_count == 1
    assert expected.max_count is None


def test_agent_tool_defaults():
    tool = _m().AgentTool(name="book_slot")
    assert tool.name == "book_slot"
    assert tool.description is None


@pytest.mark.parametrize(
    "event_type", ["tool_timeout", "tool_error", "silence", "transfer_fail", "user_change_of_mind"]
)
def test_inject_event_types(event_type):
    event = _m().InjectEvent(at_turn=2, type=event_type)
    assert event.at_turn == 2
    assert event.type == event_type
    assert event.tool is None
    assert event.error_message is None
    assert event.silence_ms is None
    assert event.note is None


def test_inject_event_rejects_unknown_type():
    with pytest.raises(_validation_error()):
        _m().InjectEvent(at_turn=1, type="network_partition")


# --------------------------------------------------------------------------------------------
# Turn, CallTranscript, Scenario
# --------------------------------------------------------------------------------------------


def test_turn_defaults():
    turn = _turn("agent", "Hello", 1200)
    assert turn.end_ms is None
    assert turn.audio_path is None
    assert turn.latency_ms is None
    assert turn.interrupted is False
    assert turn.stt_confidence is None
    assert turn.heard_text is None
    assert turn.intent_confidence is None
    assert turn.classified_intent is None
    assert turn.tool_calls == []


def test_turn_tool_calls_default_not_shared():
    m = _m()
    first = _turn("agent", "a", 0)
    second = _turn("agent", "b", 0)
    first.tool_calls.append(m.ToolCall(name="x", arguments={}))
    assert second.tool_calls == []


def test_turn_speaker_literal():
    _turn("user")
    _turn("agent")
    with pytest.raises(_validation_error()):
        _turn("system")


def test_turn_tool_calls_coerced_from_dicts():
    m = _m()
    turn = _turn("agent", "Booked", 10, tool_calls=[{"name": "book_slot", "arguments": {}}])
    assert isinstance(turn.tool_calls[0], m.ToolCall)


def test_call_transcript_defaults():
    transcript = _transcript()
    assert transcript.accent is None
    assert transcript.voice_mode is False
    assert transcript.recording_path is None
    assert transcript.agent_system_prompt is None


def test_scenario_defaults():
    scenario = _m().Scenario(name="s1", language="en-IN", persona="a patient")
    assert scenario.accent is None
    assert scenario.domain == "general"
    assert scenario.direction == "inbound"
    assert scenario.max_turns == 9
    assert scenario.opening_user_message is None
    assert scenario.expected_outcome is None
    assert scenario.key_terms == []
    assert scenario.code_switching is False
    assert scenario.assertions == []
    assert scenario.metrics == []
    assert scenario.inject_events == []
    assert scenario.seed is None
    assert scenario.expected_tool_calls == []
    assert scenario.forbidden_tool_calls == []


def test_scenario_direction_literal():
    m = _m()
    assert m.Scenario(name="s", language="en-IN", persona="p", direction="outbound").direction == "outbound"
    with pytest.raises(_validation_error()):
        m.Scenario(name="s", language="en-IN", persona="p", direction="sideways")


def test_scenario_tool_expectation_fields_round_trip():
    m = _m()
    scenario = m.Scenario(
        name="clinic_appointment_booking",
        language="en-IN",
        domain="healthcare",
        max_turns=12,
        persona="patient",
        expected_tool_calls=[
            {"name": "book_slot", "required_arguments": ["date", "patient_name", "phone"], "max_count": 1}
        ],
        forbidden_tool_calls=["cancel_slot"],
        inject_events=[{"at_turn": 2, "type": "tool_error", "tool": "book_slot", "error_message": "boom"}],
        seed=42,
    )
    assert isinstance(scenario.expected_tool_calls[0], m.ExpectedToolCall)
    assert isinstance(scenario.inject_events[0], m.InjectEvent)
    assert m.Scenario.model_validate_json(scenario.model_dump_json()) == scenario


# --------------------------------------------------------------------------------------------
# ScoreResult, MetricResult
# --------------------------------------------------------------------------------------------


def test_score_result_defaults():
    score = _score()
    assert score.turn_index is None
    assert score.confidence == 0.0
    assert score.no_verdict is False
    assert score.disclaimer == DISCLAIMER
    assert score.disclaimer.startswith("Technical signal only. Not legal certification.")


def test_score_result_literals():
    for result in ("pass", "fail", "skip"):
        _score(result=result)
    for severity in ("critical", "high", "medium"):
        _score(severity=severity)
    with pytest.raises(_validation_error()):
        _score(result="error")
    with pytest.raises(_validation_error()):
        _score(severity="low")


def test_metric_result_defaults():
    metric = _metric()
    assert metric.raw_value is None
    assert metric.threshold_warn == 0.6
    assert metric.threshold_fail == 0.4
    assert metric.turn_index is None
    assert metric.requires_audio is False
    assert metric.is_proxy is False
    assert metric.skipped is False
    assert metric.informational is False


def test_metric_result_label_accepts_skipped():
    metric = _metric(label="skipped", score=0.0, skipped=True)
    assert metric.label == "skipped"
    for label in ("good", "warning", "poor"):
        _metric(label=label)
    with pytest.raises(_validation_error()):
        _metric(label="excellent")


def test_metric_result_category_literal():
    m = _m()
    for category in ("universal", "voice", "chat"):
        m.MetricResult(metric_id="X", name="x", category=category, score=1.0, label="good", reason="r")
    with pytest.raises(_validation_error()):
        m.MetricResult(metric_id="X", name="x", category="compliance", score=1.0, label="good", reason="r")


# --------------------------------------------------------------------------------------------
# Other result models
# --------------------------------------------------------------------------------------------


def test_stt_and_latency_defaults():
    m = _m()
    stt = m.STTResult(text="hello")
    assert stt.confidence is None
    assert stt.words == []
    assert stt.filler_words == []
    assert _latency().is_real_audio is False


def test_cohort_result_defaults():
    cohort = _m().CohortResult(
        cohort_key="en-IN/default",
        language="en-IN",
        total=2,
        passed=1,
        failed=1,
        pass_rate=0.5,
        metric_means={},
        gap_vs_best=0.0,
        gap_warning=False,
    )
    assert cohort.accent is None
    assert cohort.mean_stt_confidence is None
    assert cohort.stt_warning is False


def test_scenario_result_inject_events_default_and_round_trip():
    m = _m()
    assert _scenario_result().inject_events == []
    result = _scenario_result(inject_events=[m.InjectEvent(at_turn=1, type="silence", silence_ms=800)])
    restored = m.ScenarioResult.model_validate_json(result.model_dump_json())
    assert restored.inject_events == result.inject_events
    assert restored.error is None


def test_run_result_defaults():
    run = _run_result()
    assert run.voice_mode is False
    assert run.tts_provider is None
    assert run.stt_provider is None
    assert run.warnings == []


@pytest.mark.parametrize("agent_type", ["voice", "chat", "multimodal"])
def test_run_result_agent_type_literal_accepts(agent_type):
    assert _run_result(agent_type=agent_type).agent_type == agent_type


def test_run_result_agent_type_literal_rejects_other():
    with pytest.raises(_validation_error()):
        _run_result(agent_type="phone")


def test_run_result_round_trip_with_warnings():
    m = _m()
    run = _run_result(warnings=["Unknown domain sales; using general defaults."])
    restored = m.RunResult.model_validate_json(run.model_dump_json())
    assert restored == run
    assert restored.warnings == ["Unknown domain sales; using general defaults."]


@pytest.mark.parametrize(
    "status", ["pass->fail", "fail->pass", "pass->pass", "fail->fail", "added", "removed"]
)
def test_scenario_diff_status_change_literal(status):
    diff = _m().ScenarioDiff(
        scenario_name="s", status_change=status, metric_deltas={}, assertion_changes=[], notes=[]
    )
    assert diff.status_change == status


def test_scenario_diff_rejects_unknown_status():
    with pytest.raises(_validation_error()):
        _m().ScenarioDiff(
            scenario_name="s", status_change="pass->skip", metric_deltas={}, assertion_changes=[], notes=[]
        )


def test_compare_result_fields_round_trip():
    m = _m()
    diff = m.ScenarioDiff(
        scenario_name="clinic",
        status_change="pass->fail",
        metric_deltas={"UNIV-023": -1.0},
        assertion_changes=["dpdp.consent_before_pii: pass -> fail"],
        notes=["UNIV-023 newly failing"],
    )
    compare = m.CompareResult(
        run_id_a="20261004T120000Z-a3f9c2",
        run_id_b="20261005T120000Z-b4e8d1",
        timestamp_a="2026-10-04T12:00:00Z",
        timestamp_b="2026-10-05T12:00:00Z",
        overall_pass_rate_delta=-0.5,
        regressions=[diff],
        improvements=[],
        unchanged_pass=1,
        unchanged_fail=0,
        added_scenarios=["new_one"],
        removed_scenarios=["old_one"],
        metric_summary_deltas={"UNIV-023": -0.5},
        critical_regressions=["clinic"],
    )
    assert m.CompareResult.model_validate_json(compare.model_dump_json()) == compare


# --------------------------------------------------------------------------------------------
# Pass logic: compute_passed (SPEC.md section 7)
# --------------------------------------------------------------------------------------------


def _compute(result):
    return _m().compute_passed(result)


def test_pass_logic_all_good():
    result = _scenario_result(metrics=[_metric(), _metric("UNIV-013")], assertions=[_score()])
    assert _compute(result) is True


def test_pass_logic_no_results_no_error():
    assert _compute(_scenario_result()) is True


@pytest.mark.parametrize("severity", ["critical", "high", "medium"])
def test_pass_logic_failed_assertion_of_any_severity_fails(severity):
    result = _scenario_result(metrics=[_metric()], assertions=[_score("fail", severity)])
    assert _compute(result) is False


def test_pass_logic_skipped_assertion_does_not_fail():
    assert _compute(_scenario_result(assertions=[_score("skip")])) is True


def test_pass_logic_poor_metric_fails():
    result = _scenario_result(metrics=[_metric(), _metric("UNIV-005", label="poor", score=0.0)])
    assert _compute(result) is False


def test_pass_logic_warning_metric_never_fails():
    assert _compute(_scenario_result(metrics=[_metric(label="warning", score=0.5)])) is True


def test_pass_logic_skipped_metric_excluded():
    result = _scenario_result(metrics=[_metric(label="skipped", score=0.0, skipped=True)])
    assert _compute(result) is True


def test_pass_logic_poor_but_skipped_metric_excluded():
    result = _scenario_result(metrics=[_metric(label="poor", score=0.0, skipped=True)])
    assert _compute(result) is True


def test_pass_logic_poor_informational_metric_excluded():
    result = _scenario_result(
        metrics=[_metric("VOICE-011", label="poor", score=0.0, informational=True)]
    )
    assert _compute(result) is True


def test_pass_logic_error_fails():
    assert _compute(_scenario_result(error="Pipecat bot did not finish its turn within 30 s.")) is False


def test_pass_logic_error_fails_even_with_passing_results():
    result = _scenario_result(
        metrics=[_metric()],
        assertions=[_score("skip", no_verdict=True)],
        error="critical check dpdp.consent_before_pii could not be evaluated: judge returned no verdict",
    )
    assert _compute(result) is False
