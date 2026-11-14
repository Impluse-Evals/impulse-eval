# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 3: UNIV-025 expected tool calls (METRICS.md section 1), its config rules (SPEC.md section 8.2),
the tool expectation models, and the sample agent's dummy tool call (SPEC.md section 8.2)."""

import copy

import pytest
import yaml

SYSTEM_PROMPT = "You are the booking assistant of Sunrise Clinic. Use book_slot to book appointments."
BOOK_ARGS = {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}
SAMPLE_TOOL_CALL = {
    "name": "book_slot",
    "arguments": {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"},
    "result": {"success": True},
    "success": True,
}


def _m():
    from impulse_evals import models

    return models


def _config():
    from impulse_evals import config

    return config


def _ctx(**kwargs):
    from impulse_evals.metrics.base import MetricContext

    return MetricContext(**kwargs)


def _call(name="book_slot", arguments=None, success=True, error=None):
    return _m().ToolCall(
        name=name,
        arguments=copy.deepcopy(BOOK_ARGS) if arguments is None else arguments,
        success=success,
        error=error,
    )


def _transcript(*agent_calls):
    """User/agent pairs; the agent turn k (1-based pair index) sits at turn 2k and carries agent_calls[k-1]."""
    m = _m()
    turns = []
    for index, calls in enumerate(agent_calls):
        turns.append(m.Turn(speaker="user", text=f"user message {index + 1}", timestamp_ms=index * 4000))
        turns.append(
            m.Turn(
                speaker="agent",
                text=f"agent reply {index + 1}",
                timestamp_ms=index * 4000 + 1000,
                tool_calls=list(calls),
            )
        )
    return m.CallTranscript(
        turns=turns,
        duration_ms=len(agent_calls) * 4000,
        metadata={},
        scenario_name="clinic_appointment_booking",
        language="en-IN",
        agent_system_prompt=SYSTEM_PROMPT,
    )


def _scenario(expected=(), forbidden=(), metrics=None):
    m = _m()
    return m.Scenario(
        name="clinic_appointment_booking",
        language="en-IN",
        domain="healthcare",
        max_turns=12,
        persona="patient who wants the earliest slot on Tuesday",
        metrics=list(metrics or []),
        expected_tool_calls=[m.ExpectedToolCall(**e) if isinstance(e, dict) else e for e in expected],
        forbidden_tool_calls=list(forbidden),
    )


async def _univ025(transcript, scenario, fake_judge, mock_llm):
    """Evaluate UNIV-025 and check it made zero judge and zero LLM calls."""
    from impulse_evals.metrics.base import get_metric

    judge = fake_judge()
    result = await get_metric("UNIV-025").evaluate(transcript, scenario, judge, _ctx())
    assert judge.calls == [], "UNIV-025 must never call the judge"
    assert mock_llm.call_count == 0, "UNIV-025 must never call an LLM"
    assert result.metric_id == "UNIV-025"
    assert result.category == "universal"
    assert result.skipped is False
    return result


def _assert_poor(result):
    assert result.score == 0.0
    assert result.label == "poor"


def _assert_good(result):
    assert result.score == 1.0
    assert result.label == "good"


FULL_EXPECTATION = {
    "name": "book_slot",
    "required_arguments": ["date", "patient_name", "phone"],
    "argument_values": {"date": "2026-01-01"},
}


# ============================================================================================
# UNIV-025 pass and fail reasons
# ============================================================================================


@pytest.mark.asyncio
async def test_pass_when_expected_call_has_required_arguments_and_values(fake_judge, mock_llm):
    result = await _univ025(_transcript([_call()]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)
    _assert_good(result)
    assert result.raw_value == pytest.approx(1.0)
    assert result.turn_index is None
    assert "never called" not in result.reason
    assert "missing" not in result.reason


@pytest.mark.asyncio
async def test_fail_never_called(fake_judge, mock_llm):
    result = await _univ025(_transcript([]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == "expected tool book_slot was never called"
    assert result.raw_value == pytest.approx(0.0)
    assert result.turn_index is None


@pytest.mark.asyncio
async def test_fail_missing_required_arguments(fake_judge, mock_llm):
    call = _call(arguments={"date": "2026-01-01", "patient_name": ""})
    result = await _univ025(_transcript([call]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == "tool book_slot is missing required arguments: patient_name, phone"
    assert result.turn_index == 2


@pytest.mark.asyncio
async def test_none_empty_list_and_empty_dict_count_as_missing(fake_judge, mock_llm):
    expectation = {"name": "book_slot", "required_arguments": ["date", "patient_name", "phone"]}
    call = _call(arguments={"date": None, "patient_name": [], "phone": {}})
    result = await _univ025(_transcript([call]), _scenario([expectation]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == "tool book_slot is missing required arguments: date, patient_name, phone"


@pytest.mark.asyncio
async def test_fail_wrong_value_shows_expected_and_actual(fake_judge, mock_llm):
    call = _call(arguments={"date": "2026-01-02", "patient_name": "Test User", "phone": "0000000000"})
    result = await _univ025(_transcript([call]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == 'tool book_slot argument date: expected "2026-01-01", got "2026-01-02"'
    assert result.turn_index == 2


@pytest.mark.asyncio
async def test_wrong_values_use_json_equality_one_reason_per_key(fake_judge, mock_llm):
    expectation = {"name": "book_slot", "argument_values": {"date": "2026-01-01", "slots": 1}}
    call = _call(arguments={"date": "2026-01-02", "slots": "1"})
    result = await _univ025(_transcript([call]), _scenario([expectation]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == (
        'tool book_slot argument date: expected "2026-01-01", got "2026-01-02"; '
        'tool book_slot argument slots: expected 1, got "1"'
    )


@pytest.mark.asyncio
async def test_fail_success_false_with_must_succeed(fake_judge, mock_llm):
    call = _call(success=False, error="slot taken")
    result = await _univ025(_transcript([call]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == "tool book_slot call failed: slot taken"
    assert result.turn_index == 2


@pytest.mark.asyncio
async def test_fail_success_false_without_error_message(fake_judge, mock_llm):
    call = _call(success=False, error=None)
    result = await _univ025(_transcript([call]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == "tool book_slot call failed: no error message"


@pytest.mark.asyncio
async def test_success_false_allowed_when_must_succeed_is_false(fake_judge, mock_llm):
    expectation = dict(FULL_EXPECTATION, must_succeed=False)
    call = _call(success=False, error="slot taken")
    result = await _univ025(_transcript([call]), _scenario([expectation]), fake_judge, mock_llm)
    _assert_good(result)


@pytest.mark.asyncio
async def test_success_none_passes_with_note(fake_judge, mock_llm):
    call = _call(success=None)
    result = await _univ025(_transcript([call]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)
    _assert_good(result)
    assert result.raw_value == pytest.approx(1.0)
    assert result.reason == "tool book_slot: success not captured"


@pytest.mark.asyncio
async def test_fail_more_than_max_count(fake_judge, mock_llm):
    expectation = dict(FULL_EXPECTATION, max_count=1)
    result = await _univ025(
        _transcript([_call()], [_call()]), _scenario([expectation]), fake_judge, mock_llm
    )
    _assert_poor(result)
    assert result.reason == "tool book_slot called 2 times, expected at most 1"


@pytest.mark.asyncio
async def test_fail_fewer_than_min_count(fake_judge, mock_llm):
    expectation = dict(FULL_EXPECTATION, min_count=2)
    result = await _univ025(_transcript([_call()]), _scenario([expectation]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == "tool book_slot called 1 times, expected at least 2"


@pytest.mark.asyncio
async def test_calls_counted_across_all_agent_turns(fake_judge, mock_llm):
    expectation = dict(FULL_EXPECTATION, min_count=2, max_count=2)
    result = await _univ025(
        _transcript([_call()], [], [_call()]), _scenario([expectation]), fake_judge, mock_llm
    )
    _assert_good(result)


@pytest.mark.asyncio
async def test_min_zero_with_max_zero(fake_judge, mock_llm):
    expectation = {"name": "book_slot", "min_count": 0, "max_count": 0}
    result = await _univ025(_transcript([]), _scenario([expectation]), fake_judge, mock_llm)
    _assert_good(result)
    result = await _univ025(_transcript([_call()]), _scenario([expectation]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == "tool book_slot called 1 times, expected at most 0"


@pytest.mark.asyncio
async def test_fail_forbidden_call(fake_judge, mock_llm):
    transcript = _transcript([], [_call("cancel_slot", {"booking_id": "B-17"})])
    result = await _univ025(transcript, _scenario(forbidden=["cancel_slot"]), fake_judge, mock_llm)
    _assert_poor(result)
    assert result.reason == "forbidden tool cancel_slot was called at turn 4"
    assert result.turn_index == 4
    assert result.raw_value == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_forbidden_tool_not_called_passes(fake_judge, mock_llm):
    result = await _univ025(_transcript([_call()]), _scenario(forbidden=["cancel_slot"]), fake_judge, mock_llm)
    _assert_good(result)
    assert result.raw_value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_raw_value_counts_each_expected_entry_and_forbidden_name(fake_judge, mock_llm):
    scenario = _scenario(
        expected=[FULL_EXPECTATION, {"name": "send_sms"}],
        forbidden=["cancel_slot", "refund"],
    )
    transcript = _transcript([_call()], [_call("refund", {"amount": 100})])
    result = await _univ025(transcript, scenario, fake_judge, mock_llm)
    _assert_poor(result)
    assert result.raw_value == pytest.approx(0.5)
    assert result.reason == (
        "expected tool send_sms was never called; forbidden tool refund was called at turn 4"
    )
    assert result.turn_index == 4


@pytest.mark.asyncio
async def test_reasons_taken_from_first_matching_call(fake_judge, mock_llm):
    missing_phone = _call(arguments={"date": "2026-01-01", "patient_name": "Test User"})
    wrong_date = _call(arguments={"date": "2026-01-09", "patient_name": "Test User", "phone": "0000000000"})
    result = await _univ025(
        _transcript([missing_phone], [wrong_date]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm
    )
    _assert_poor(result)
    assert result.reason == "tool book_slot is missing required arguments: phone"
    assert result.turn_index == 2


@pytest.mark.asyncio
async def test_one_satisfying_call_is_enough(fake_judge, mock_llm):
    missing_phone = _call(arguments={"date": "2026-01-01", "patient_name": "Test User"})
    result = await _univ025(
        _transcript([missing_phone], [_call()]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm
    )
    _assert_good(result)


# ============================================================================================
# UNIV-025 runs automatically (select_metrics) and fails the scenario (compute_passed)
# ============================================================================================


def _select(scenario, **ctx_kwargs):
    from impulse_evals.metrics.base import select_metrics

    return select_metrics(scenario, _ctx(**ctx_kwargs))


def test_select_metrics_adds_univ025_for_expected_tool_calls():
    selected = _select(_scenario([FULL_EXPECTATION], metrics=["UNIV-001"]))
    assert "UNIV-025" in selected
    assert "UNIV-001" in selected


def test_select_metrics_adds_univ025_for_forbidden_tool_calls():
    selected = _select(_scenario(forbidden=["cancel_slot"], metrics=["UNIV-013"]))
    assert "UNIV-025" in selected
    assert "UNIV-013" in selected


def test_select_metrics_without_tool_expectations_has_no_univ025():
    selected = _select(_scenario(metrics=["UNIV-001", "UNIV-013"]))
    assert "UNIV-025" not in selected
    assert set(selected) == {"UNIV-001", "UNIV-013"}


def test_select_metrics_lists_univ025_once():
    selected = _select(_scenario([FULL_EXPECTATION], metrics=["UNIV-025", "UNIV-001"]))
    assert selected.count("UNIV-025") == 1


def test_select_metrics_empty_when_metrics_disabled():
    assert _select(_scenario([FULL_EXPECTATION], metrics=["UNIV-001"]), metrics_enabled=False) == []


@pytest.mark.asyncio
async def test_poor_univ025_fails_the_scenario(fake_judge, mock_llm):
    m = _m()
    transcript = _transcript([])
    poor = await _univ025(transcript, _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)
    good = await _univ025(_transcript([_call()]), _scenario([FULL_EXPECTATION]), fake_judge, mock_llm)

    def scenario_result(metric):
        return m.ScenarioResult(
            scenario_name="clinic_appointment_booking",
            transcript=transcript,
            assertion_results=[],
            metric_results=[metric],
            passed=True,
        )

    assert m.compute_passed(scenario_result(poor)) is False
    assert m.compute_passed(scenario_result(good)) is True


# ============================================================================================
# Config rules for tool expectations (SPEC.md section 8.2)
# ============================================================================================


def _config_data(tools=None, scenarios=None):
    base_scenario = {
        "name": "clinic_appointment_booking",
        "language": "en-IN",
        "domain": "healthcare",
        "max_turns": 12,
        "persona": "patient who wants the earliest slot on Tuesday",
    }
    return {
        "agent": {
            "type": "chat",
            "adapter": "generic",
            "entrypoint": "sample_agent:respond",
            "system_prompt": "system_prompt.txt",
            "model": None,
            "tools": tools,
        },
        "judge": {
            "primary_model": "ollama/judge-primary",
            "fast_model": "ollama/judge-fast",
            "primary_options": {},
            "fast_options": {},
        },
        "simulator": {"model": "ollama/simulator", "persona_style": "realistic", "mode": "cascade", "options": {}},
        "metrics": {"enabled": True},
        "scenarios": scenarios if scenarios is not None else [base_scenario],
    }


def _scenario_data(name="clinic_appointment_booking", **extra):
    data = {
        "name": name,
        "language": "en-IN",
        "domain": "healthcare",
        "max_turns": 12,
        "persona": "patient who wants the earliest slot on Tuesday",
    }
    data.update(extra)
    return data


TOOLS = ["book_slot", {"name": "cancel_slot", "description": "Cancel a booking"}]


@pytest.fixture
def load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("IMPULSE_MODEL_PRIMARY", "IMPULSE_MODEL_FAST", "IMPULSE_MODEL_SIMULATOR"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")

    def _load(data):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return _config().load_config(path)

    return _load


def _config_error(load, data):
    config = _config()
    with pytest.raises(config.ConfigError) as info:
        load(data)
    return info.value


def test_config_error_expected_tool_not_in_agent_tools(load):
    data = _config_data(
        tools=TOOLS,
        scenarios=[
            _scenario_data("first", expected_tool_calls=[{"name": "book_slot"}]),
            _scenario_data("second", expected_tool_calls=[{"name": "send_sms"}]),
        ],
    )
    assert _config_error(load, data).field == "scenarios[1].expected_tool_calls[0].name"


def test_config_error_forbidden_tool_not_in_agent_tools(load):
    data = _config_data(tools=TOOLS, scenarios=[_scenario_data(forbidden_tool_calls=["refund"])])
    assert _config_error(load, data).field == "scenarios[0].forbidden_tool_calls[0]"


def test_unknown_tool_names_allowed_when_agent_tools_unset(load):
    data = _config_data(
        tools=None,
        scenarios=[_scenario_data(expected_tool_calls=[{"name": "send_sms"}], forbidden_tool_calls=["refund"])],
    )
    cfg = load(data)
    assert cfg.agent.tools is None
    assert cfg.scenarios[0].expected_tool_calls[0].name == "send_sms"


def test_config_error_max_count_below_min_count(load):
    data = _config_data(
        scenarios=[_scenario_data(expected_tool_calls=[{"name": "book_slot", "min_count": 2, "max_count": 1}])]
    )
    assert _config_error(load, data).field in {
        "scenarios[0].expected_tool_calls[0].max_count",
        "scenarios[0].expected_tool_calls[0].min_count",
    }


def test_config_error_min_count_zero_without_max_count(load):
    data = _config_data(scenarios=[_scenario_data(expected_tool_calls=[{"name": "book_slot", "min_count": 0}])])
    assert _config_error(load, data).field == "scenarios[0].expected_tool_calls[0].min_count"


def test_config_error_negative_min_count(load):
    data = _config_data(
        scenarios=[_scenario_data(expected_tool_calls=[{"name": "book_slot", "min_count": -1, "max_count": 1}])]
    )
    assert _config_error(load, data).field == "scenarios[0].expected_tool_calls[0].min_count"


def test_config_error_tool_both_expected_and_forbidden(load):
    data = _config_data(
        scenarios=[
            _scenario_data(expected_tool_calls=[{"name": "book_slot"}], forbidden_tool_calls=["book_slot"])
        ]
    )
    assert _config_error(load, data).field in {
        "scenarios[0].forbidden_tool_calls[0]",
        "scenarios[0].expected_tool_calls[0].name",
    }


def test_min_count_zero_with_max_count_is_valid(load):
    data = _config_data(
        scenarios=[_scenario_data(expected_tool_calls=[{"name": "book_slot", "min_count": 0, "max_count": 0}])]
    )
    cfg = load(data)
    assert cfg.scenarios[0].expected_tool_calls[0].max_count == 0


# ============================================================================================
# Models round-trip (through config.yaml and JSON)
# ============================================================================================


def test_agent_tools_normalized_to_agent_tool_list(load):
    m = _m()
    cfg = load(_config_data(tools=TOOLS))
    assert cfg.agent.tools == [
        m.AgentTool(name="book_slot"),
        m.AgentTool(name="cancel_slot", description="Cancel a booking"),
    ]
    assert all(isinstance(tool, m.AgentTool) for tool in cfg.agent.tools)


def test_agent_tools_empty_list_and_null(load):
    assert load(_config_data(tools=[])).agent.tools == []
    assert load(_config_data(tools=None)).agent.tools is None


def test_scenario_tool_fields_round_trip_through_config(load):
    m = _m()
    data = _config_data(
        tools=TOOLS,
        scenarios=[
            _scenario_data(
                expected_tool_calls=[
                    {
                        "name": "book_slot",
                        "required_arguments": ["date", "patient_name", "phone"],
                        "argument_values": {"date": "2026-01-01"},
                        "must_succeed": True,
                        "max_count": 1,
                    }
                ],
                forbidden_tool_calls=["cancel_slot"],
            )
        ],
    )
    scenario = load(data).scenarios[0]
    assert isinstance(scenario, m.Scenario)
    expected = scenario.expected_tool_calls[0]
    assert isinstance(expected, m.ExpectedToolCall)
    assert expected.name == "book_slot"
    assert expected.required_arguments == ["date", "patient_name", "phone"]
    assert expected.argument_values == {"date": "2026-01-01"}
    assert expected.must_succeed is True
    assert expected.min_count == 1
    assert expected.max_count == 1
    assert scenario.forbidden_tool_calls == ["cancel_slot"]
    assert m.Scenario.model_validate_json(scenario.model_dump_json()) == scenario


def test_expected_tool_call_and_agent_tool_json_round_trip():
    m = _m()
    expected = m.ExpectedToolCall(
        name="book_slot",
        required_arguments=["date"],
        argument_values={"date": "2026-01-01", "slots": 1},
        must_succeed=False,
        min_count=0,
        max_count=2,
    )
    assert m.ExpectedToolCall.model_validate_json(expected.model_dump_json()) == expected
    tool = m.AgentTool(name="book_slot", description="Book an appointment slot")
    assert m.AgentTool.model_validate_json(tool.model_dump_json()) == tool


# ============================================================================================
# examples/sample_agent.py (SPEC.md section 8.2)
# ============================================================================================


def _tool_calls(reply):
    return reply.get("tool_calls", []) if isinstance(reply, dict) else []


def _text(reply):
    return reply["text"] if isinstance(reply, dict) else reply


def test_sample_agent_exposes_system_prompt(sample_agent):
    assert isinstance(sample_agent.SYSTEM_PROMPT, str)
    assert sample_agent.SYSTEM_PROMPT.strip()


@pytest.mark.parametrize("text", ["I want to book a slot", "BOOK please", "Can you Book.", "book"])
def test_sample_agent_returns_book_slot_call_on_word_book(sample_agent, text):
    reply = sample_agent.respond(text, [])
    assert _tool_calls(reply) == [SAMPLE_TOOL_CALL]
    assert text in _text(reply)


@pytest.mark.parametrize("text", ["booking for two", "my notebook", "I booked yesterday", "hello"])
def test_sample_agent_no_tool_call_without_whole_word_book(sample_agent, text):
    reply = sample_agent.respond(text, [{"role": "assistant", "content": "Hi"}])
    assert _tool_calls(reply) == []
    assert text in _text(reply)


def test_sample_agent_outbound_opener(sample_agent):
    reply = sample_agent.respond("", [])
    assert _text(reply).strip()
    assert _tool_calls(reply) == []


def test_sample_agent_tool_call_is_fresh_each_time(sample_agent):
    first = sample_agent.respond("book", [])
    first["tool_calls"][0]["arguments"]["date"] = "changed"
    second = sample_agent.respond("book", [])
    assert _tool_calls(second) == [SAMPLE_TOOL_CALL]


@pytest.mark.asyncio
async def test_sample_agent_tool_call_satisfies_clinic_expectation(sample_agent, fake_judge, mock_llm):
    m = _m()
    reply = sample_agent.respond("Please book me for Tuesday", [])
    calls = [m.ToolCall(**call) for call in _tool_calls(reply)]
    expectation = {"name": "book_slot", "required_arguments": ["date", "patient_name", "phone"], "max_count": 1}
    result = await _univ025(_transcript(calls), _scenario([expectation]), fake_judge, mock_llm)
    _assert_good(result)
