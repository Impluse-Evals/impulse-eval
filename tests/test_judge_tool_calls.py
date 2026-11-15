# SPDX-License-Identifier: Apache-2.0
"""Tool call lines in transcripts (SPEC.md sections 12 and 15) and UNIV-023 (METRICS.md section 1).

Phase 2 runs only the formatting tests (`-k format`); the whole file passes at phase 3.
"""

import json

import pytest

PRIMARY = "ollama/judge-primary"
FAST = "ollama/judge-fast"
SYSTEM_PROMPT = (
    "You are the booking assistant of Sunrise Clinic. To book an appointment, call the book_slot "
    "tool with date, patient_name and phone. Only confirm a booking after book_slot succeeds."
)
EXPECTED_OUTCOME = "appointment booked and confirmed back to the patient"

UNIV023_CRITERION = (
    "Look at every tool call in the transcript (shown as AGENT TOOL CALL lines). For each one: "
    "(1) Was the right tool called given the user's request and the system prompt's tool descriptions? "
    "(2) Were the arguments correct and complete — not missing required fields, not using wrong values? "
    "(3) Does `success=False` or `success=None` with a visible error suggest the downstream action failed? "
    "(4) Did the agent verbally claim the action succeeded even when the tool call shows it failed or errored? "
    "(5) Did the agent claim to have performed an action (for example booked, cancelled, paid, refunded, "
    "updated, sent) without a matching AGENT TOOL CALL line, or did the user's request, read with the "
    "system prompt and the agent's available tools, require a tool call that the agent never made? "
    "A pass requires all calls to use the right tool with correct arguments, no verbal confirmation of an "
    "action that the tool log shows failed, and no claimed or required action without a matching tool call. "
    'If the agent has no tools and none were needed, return skip with the reason "agent has no tools".'
)

BOOK_ARGS = {"phone": "0000000000", "date": "2026-01-01", "patient_name": "Test User"}
BOOK_ARGS_JSON = '{"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}'


def _m():
    from impulse_evals import models

    return models


def _format(transcript):
    from impulse_evals.transcriber import format_transcript

    return format_transcript(transcript)


def _turn(speaker, text, timestamp_ms, **kwargs):
    return _m().Turn(speaker=speaker, text=text, timestamp_ms=timestamp_ms, **kwargs)


def _call(name="book_slot", arguments=None, **kwargs):
    return _m().ToolCall(name=name, arguments=BOOK_ARGS if arguments is None else arguments, **kwargs)


def _transcript(turns, system_prompt=SYSTEM_PROMPT):
    return _m().CallTranscript(
        turns=turns,
        duration_ms=5000,
        metadata={},
        scenario_name="clinic_appointment_booking",
        language="en-IN",
        agent_system_prompt=system_prompt,
    )


def _scenario(**kwargs):
    return _m().Scenario(
        name="clinic_appointment_booking",
        language="en-IN",
        domain="healthcare",
        max_turns=12,
        persona="patient who wants the earliest slot on Tuesday",
        **kwargs,
    )


def _ctx(**kwargs):
    from impulse_evals.metrics.base import MetricContext

    return MetricContext(**kwargs)


def _univ023():
    from impulse_evals.metrics.base import get_metric

    return get_metric("UNIV-023")


def _litellm_judge():
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    return LiteLLMJudge(primary_model=PRIMARY, fast_model=FAST, primary_options={}, fast_options={})


def _booking_transcript(success=True, error=None, agent_text="Your appointment is booked for Tuesday."):
    return _transcript(
        [
            _turn("user", "Please book me for Tuesday. I'm Test User, phone 0000000000.", 0),
            _turn("agent", agent_text, 1200, tool_calls=[_call(success=success, error=error)]),
        ]
    )


# ============================================================================================
# Phase 2: format_transcript tool call lines (selected with -k format)
# ============================================================================================


def test_format_transcript_plain_lines():
    transcript = _transcript([_turn("user", "Hi there", 0), _turn("agent", "Hello! How can I help?", 700)])
    assert _format(transcript).splitlines() == [
        "[turn 1] USER: Hi there",
        "[turn 2] AGENT: Hello! How can I help?",
    ]


def test_format_tool_call_line_exact():
    lines = _format(_booking_transcript(success=True)).splitlines()
    assert lines == [
        "[turn 1] USER: Please book me for Tuesday. I'm Test User, phone 0000000000.",
        f"[turn 2] AGENT TOOL CALL: book_slot({BOOK_ARGS_JSON}) -> success=True",
        "[turn 2] AGENT: Your appointment is booked for Tuesday.",
    ]


def test_format_tool_call_error_suffix_when_error_set():
    lines = _format(_booking_transcript(success=False, error="slot unavailable")).splitlines()
    assert lines[1] == f"[turn 2] AGENT TOOL CALL: book_slot({BOOK_ARGS_JSON}) -> success=False error=slot unavailable"


def test_format_success_none_with_visible_error():
    lines = _format(_booking_transcript(success=None, error="timeout after 10 s")).splitlines()
    assert lines[1] == f"[turn 2] AGENT TOOL CALL: book_slot({BOOK_ARGS_JSON}) -> success=None error=timeout after 10 s"


@pytest.mark.parametrize("success, shown", [(True, "True"), (False, "False"), (None, "None")])
def test_format_no_error_suffix_when_error_unset(success, shown):
    lines = _format(_booking_transcript(success=success)).splitlines()
    assert lines[1] == f"[turn 2] AGENT TOOL CALL: book_slot({BOOK_ARGS_JSON}) -> success={shown}"
    assert " error=" not in lines[1]


def test_format_multiple_calls_in_call_order_before_agent_line():
    transcript = _transcript(
        [
            _turn("user", "Move my booking to Wednesday", 0),
            _turn(
                "agent",
                "Done, moved to Wednesday.",
                1500,
                tool_calls=[
                    _call("cancel_slot", {"booking_id": "B-17"}, success=True),
                    _call("book_slot", {"date": "2026-01-02"}, success=True),
                ],
            ),
            _turn("user", "Thanks", 4000),
        ]
    )
    assert _format(transcript).splitlines() == [
        "[turn 1] USER: Move my booking to Wednesday",
        '[turn 2] AGENT TOOL CALL: cancel_slot({"booking_id": "B-17"}) -> success=True',
        '[turn 2] AGENT TOOL CALL: book_slot({"date": "2026-01-02"}) -> success=True',
        "[turn 2] AGENT: Done, moved to Wednesday.",
        "[turn 3] USER: Thanks",
    ]


def test_format_arguments_follow_json_dumps_rule():
    arguments = {"z": [3, 1], "a": {"y": 1, "b": None}, "name": "नमस्ते", "flag": True}
    transcript = _transcript(
        [_turn("user", "hi", 0), _turn("agent", "ok", 500, tool_calls=[_call("note", arguments)])]
    )
    rendered = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    assert rendered == '{"a": {"b": null, "y": 1}, "flag": true, "name": "नमस्ते", "z": [3, 1]}'
    assert _format(transcript).splitlines()[1] == f"[turn 2] AGENT TOOL CALL: note({rendered}) -> success=None"


def test_format_empty_arguments():
    transcript = _transcript(
        [_turn("user", "hi", 0), _turn("agent", "ok", 500, tool_calls=[_call("ping", {}, success=True)])]
    )
    assert _format(transcript).splitlines()[1] == "[turn 2] AGENT TOOL CALL: ping({}) -> success=True"


def test_format_turn_numbers_count_both_speakers_from_one():
    transcript = _transcript(
        [
            _turn("agent", "Hello, this is Sunrise Clinic calling.", 0),
            _turn("user", "Hi", 2000),
            _turn("agent", "Is this a good time?", 2600, tool_calls=[_call("log_call", {"step": 1}, success=True)]),
        ]
    )
    assert _format(transcript).splitlines() == [
        "[turn 1] AGENT: Hello, this is Sunrise Clinic calling.",
        "[turn 2] USER: Hi",
        '[turn 3] AGENT TOOL CALL: log_call({"step": 1}) -> success=True',
        "[turn 3] AGENT: Is this a good time?",
    ]


# ============================================================================================
# Phase 3: UNIV-023 (no marker; excluded at phase 2 by name)
# ============================================================================================


def test_univ023_attributes_and_criterion_word_for_word():
    metric = _univ023()
    assert metric.metric_id == "UNIV-023"
    assert metric.name == "Tool call correctness"
    assert metric.category == "universal"
    assert metric.tier == "primary"
    assert metric.criterion == UNIV023_CRITERION


def test_univ023_criterion_includes_check_5():
    criterion = _univ023().criterion
    assert (
        "(5) Did the agent claim to have performed an action (for example booked, cancelled, paid, "
        "refunded, updated, sent) without a matching AGENT TOOL CALL line" in criterion
    )
    assert "require a tool call that the agent never made?" in criterion


def test_build_criterion_appends_expected_outcome_line():
    from impulse_evals.metrics.base import build_criterion

    with_outcome = _scenario(expected_outcome=EXPECTED_OUTCOME)
    expected = "Is the reply polite?\n\nExpected outcome: " + EXPECTED_OUTCOME
    assert build_criterion("Is the reply polite?", with_outcome, _ctx()) == expected
    assert build_criterion("Is the reply polite?", with_outcome, None) == expected
    without = _scenario()
    assert build_criterion("Is the reply polite?", without, _ctx()) == "Is the reply polite?"
    assert build_criterion("Is the reply polite?", without, None) == "Is the reply polite?"


@pytest.mark.asyncio
async def test_univ023_tools_available_lines_appended_when_agent_tools_set(fake_judge):
    m = _m()
    judge = fake_judge()
    ctx = _ctx(
        agent_tools=[
            m.AgentTool(name="book_slot", description="Book an appointment slot"),
            m.AgentTool(name="cancel_slot"),
        ]
    )
    await _univ023().evaluate(_booking_transcript(), _scenario(), judge, ctx)
    assert len(judge.calls) == 1
    criterion = judge.calls[0]["criterion"]
    assert criterion.rstrip() == (
        UNIV023_CRITERION
        + "\n\nTools available to the agent:\nbook_slot - Book an appointment slot\ncancel_slot"
    )


@pytest.mark.asyncio
async def test_univ023_no_tools_available_lines_when_agent_tools_unset(fake_judge):
    judge = fake_judge()
    await _univ023().evaluate(_booking_transcript(), _scenario(), judge, _ctx())
    criterion = judge.calls[0]["criterion"]
    assert "Tools available to the agent" not in criterion
    assert criterion.rstrip() == UNIV023_CRITERION


@pytest.mark.asyncio
async def test_univ023_context_lines_in_order(fake_judge):
    m = _m()
    judge = fake_judge()
    scenario = _scenario(
        expected_outcome=EXPECTED_OUTCOME,
        expected_tool_calls=[m.ExpectedToolCall(name="book_slot"), m.ExpectedToolCall(name="send_sms")],
    )
    ctx = _ctx(
        agent_tools=[
            m.AgentTool(name="book_slot", description="Book an appointment slot"),
            m.AgentTool(name="send_sms", description="Send a confirmation SMS"),
        ]
    )
    await _univ023().evaluate(_booking_transcript(), scenario, judge, ctx)
    criterion = judge.calls[0]["criterion"]
    assert criterion.startswith(UNIV023_CRITERION + "\n\n")
    lines = criterion.splitlines()
    assert "Expected outcome: " + EXPECTED_OUTCOME in lines
    assert "Tools this scenario expects: book_slot, send_sms" in lines
    header = lines.index("Tools available to the agent:")
    assert lines[header + 1] == "book_slot - Book an appointment slot"
    assert lines[header + 2] == "send_sms - Send a confirmation SMS"
    assert (
        lines.index("Expected outcome: " + EXPECTED_OUTCOME)
        < header
        < lines.index("Tools this scenario expects: book_slot, send_sms")
    )
    assert "Return skip if:" not in criterion


@pytest.mark.asyncio
async def test_univ023_passes_when_tool_called_correctly(mock_llm):
    mock_llm.queue_verdict(
        "pass", reason="book_slot was called with date, name and phone, and it succeeded.", confidence=0.9
    )
    result = await _univ023().evaluate(_booking_transcript(success=True), _scenario(), _litellm_judge(), _ctx())
    assert mock_llm.call_count == 1
    assert mock_llm.calls[0]["model"] == PRIMARY
    assert f"[turn 2] AGENT TOOL CALL: book_slot({BOOK_ARGS_JSON}) -> success=True" in mock_llm.text(0)
    assert result.metric_id == "UNIV-023"
    assert result.category == "universal"
    assert result.skipped is False
    assert result.score == 1.0
    assert result.label == "good"


@pytest.mark.asyncio
async def test_univ023_fails_when_tool_success_false(mock_llm):
    mock_llm.queue_verdict(
        "fail",
        reason="The agent confirmed the booking although book_slot failed.",
        turn_index=2,
        confidence=0.95,
    )
    transcript = _booking_transcript(
        success=False, error="slot unavailable", agent_text="Your appointment is confirmed for Tuesday."
    )
    result = await _univ023().evaluate(transcript, _scenario(), _litellm_judge(), _ctx())
    assert mock_llm.call_count == 1
    assert "-> success=False error=slot unavailable" in mock_llm.text(0)
    assert result.skipped is False
    assert result.score == 0.0
    assert result.label == "poor"
    assert result.turn_index == 2
    assert "book_slot failed" in result.reason


@pytest.mark.asyncio
async def test_univ023_fails_claimed_booking_without_tool_call(fake_judge):
    judge = fake_judge(
        {
            "result": "fail",
            "reason": "The agent said the appointment was booked but never called book_slot.",
            "turn_index": 2,
            "confidence": 0.9,
        }
    )
    transcript = _transcript(
        [
            _turn("user", "Please book me for Tuesday at 10.", 0),
            _turn("agent", "Done, your appointment is booked for Tuesday at 10.", 1000),
        ]
    )
    result = await _univ023().evaluate(transcript, _scenario(), judge, _ctx())
    assert len(judge.calls) == 1, "UNIV-023 must run (not skip) when no tool was called"
    assert judge.calls[0]["tier"] == "primary"
    assert judge.calls[0]["criterion"].startswith(UNIV023_CRITERION)
    assert all(not turn.tool_calls for turn in judge.calls[0]["transcript"].turns)
    assert result.skipped is False
    assert result.score == 0.0
    assert result.label == "poor"
    assert result.turn_index == 2


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 34): SPEC.md section 12 shows the ` error=` suffix only "when
# `error` is set"; an empty error string is not set, so the line has no dangling ` error=` suffix
# (`if call.error:`). Formatting tests (names contain "format").
# ============================================================================================


@pytest.mark.parametrize("success, shown", [(True, "True"), (False, "False"), (None, "None")])
def test_format_no_error_suffix_when_error_is_empty_string(success, shown):
    lines = _format(_booking_transcript(success=success, error="")).splitlines()
    assert lines[1] == f"[turn 2] AGENT TOOL CALL: book_slot({BOOK_ARGS_JSON}) -> success={shown}"
    assert not lines[1].endswith(" error=")


def test_format_empty_error_message_reaches_the_judge_without_suffix():
    from impulse_evals.judge.prompts import build_messages

    transcript = _booking_transcript(success=False, error="")
    text = "\n".join(str(message.get("content", "")) for message in build_messages(transcript, "Is the booking right?"))
    assert f"[turn 2] AGENT TOOL CALL: book_slot({BOOK_ARGS_JSON}) -> success=False\n" in text + "\n"
    assert " error=" not in text
