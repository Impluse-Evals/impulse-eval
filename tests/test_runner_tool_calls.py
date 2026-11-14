# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 6: the runner captures tool calls from adapter replies, UNIV-023 judges them, and
``format_transcript`` shows them (SPEC.md sections 9, 12, 13; METRICS.md section 1)."""

import pytest

SAMPLE_TOOL_CALL = {
    "name": "book_slot",
    "arguments": {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"},
    "result": {"success": True},
    "success": True,
}
BOOK_ARGS_JSON = '{"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}'


def _m():
    from impulse_evals import models

    return models


def _format(transcript):
    from impulse_evals.transcriber import format_transcript

    return format_transcript(transcript)


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


def _metrics(scenario_result):
    return {metric.metric_id: metric for metric in scenario_result.metric_results}


@pytest.mark.asyncio
async def test_sample_agent_tool_call_captured_and_judged_by_univ023(impulse_project, llm_router):
    m = _m()
    project = impulse_project
    project.write_agent()  # copy of examples/sample_agent.py: "book" returns a book_slot call
    project.write_config(
        [
            project.scenario(
                "clinic_booking", opening_user_message="Please book me for Tuesday.", max_turns=1, metrics=["UNIV-023"]
            )
        ]
    )
    result = await _run(project.load())
    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent"]
    assert turns[0].tool_calls == []
    assert turns[1].tool_calls == [m.ToolCall(**SAMPLE_TOOL_CALL)]

    line = f"[turn 2] AGENT TOOL CALL: book_slot({BOOK_ARGS_JSON}) -> success=True"
    formatted = _format(scenario_result.transcript).splitlines()
    assert line in formatted
    assert formatted.index(line) + 1 == formatted.index(f"[turn 2] AGENT: {turns[1].text}")

    univ023_calls = llm_router.judge_calls_for("UNIV-023")
    assert len(univ023_calls) == 1
    _, judged_transcript = llm_router.judge_sections(univ023_calls[0])
    assert line in judged_transcript.splitlines()
    univ023 = _metrics(scenario_result)["UNIV-023"]
    assert univ023.skipped is False
    assert univ023.label == "good"


@pytest.mark.asyncio
async def test_failed_tool_call_reaches_univ023_and_univ025(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [
            project.scenario(
                "failed_booking",
                max_turns=1,
                metrics=["UNIV-023"],
                expected_tool_calls=[{"name": "book_slot", "required_arguments": ["date", "phone"]}],
            )
        ]
    )
    fake_adapter.replies["failed_booking"] = [
        {
            "text": "Done, your appointment is booked.",
            "tool_calls": [
                {
                    "name": "book_slot",
                    "arguments": {"date": "2026-01-01", "phone": "0000000000"},
                    "success": False,
                    "error": "slot taken",
                }
            ],
        }
    ]
    llm_router.on_judge(
        llm_router.verdict("fail", reason="claimed a booking the tool log shows failed", turn_index=2),
        criterion="UNIV-023",
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    line = (
        '[turn 2] AGENT TOOL CALL: book_slot({"date": "2026-01-01", "phone": "0000000000"}) '
        "-> success=False error=slot taken"
    )
    assert line in _format(scenario_result.transcript).splitlines()
    _, judged_transcript = llm_router.judge_sections(llm_router.judge_calls_for("UNIV-023")[0])
    assert line in judged_transcript.splitlines()
    metrics = _metrics(scenario_result)
    assert metrics["UNIV-023"].label == "poor"
    assert metrics["UNIV-023"].turn_index == 2
    assert metrics["UNIV-025"].label == "poor"
    assert metrics["UNIV-025"].reason == "tool book_slot call failed: slot taken"
    assert scenario_result.passed is False


@pytest.mark.asyncio
async def test_several_tool_calls_keep_their_order_and_turns(impulse_project, llm_router, fake_adapter):
    m = _m()
    project = impulse_project
    project.write_config([project.scenario("two_step_booking", max_turns=2, metrics=["UNIV-023"])])
    llm_router.script(project.persona("two_step_booking"), "Yes, book the 10 o'clock slot.")
    lookup = {"name": "lookup_slots", "arguments": {"day": "Tuesday"}, "result": {"slots": ["10:00"]}, "success": True}
    book = {"name": "book_slot", "arguments": {"date": "2026-01-01", "time": "10:00"}, "success": None}
    notify = {"name": "send_sms", "arguments": {"phone": "0000000000"}, "success": True}
    fake_adapter.replies["two_step_booking"] = [
        {"text": "Tuesday has a 10 o'clock slot.", "tool_calls": [lookup]},
        {"text": "Booked, and I sent you an SMS.", "tool_calls": [book, notify]},
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    turns = result.scenario_results[0].transcript.turns
    assert turns[1].tool_calls == [m.ToolCall(**lookup)]
    assert turns[3].tool_calls == [m.ToolCall(**book), m.ToolCall(**notify)]
    assert all(turn.tool_calls == [] for turn in turns if turn.speaker == "user")
    lines = _format(result.scenario_results[0].transcript).splitlines()
    assert lines == [
        f"[turn 1] USER: {turns[0].text}",
        '[turn 2] AGENT TOOL CALL: lookup_slots({"day": "Tuesday"}) -> success=True result={"slots": ["10:00"]}',
        "[turn 2] AGENT: Tuesday has a 10 o'clock slot.",
        "[turn 3] USER: Yes, book the 10 o'clock slot.",
        '[turn 4] AGENT TOOL CALL: book_slot({"date": "2026-01-01", "time": "10:00"}) -> success=None',
        '[turn 4] AGENT TOOL CALL: send_sms({"phone": "0000000000"}) -> success=True',
        "[turn 4] AGENT: Booked, and I sent you an SMS.",
    ]


@pytest.mark.asyncio
async def test_agent_tools_and_expectations_reach_univ023_criterion(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [
            project.scenario(
                "claimed_booking",
                max_turns=1,
                metrics=["UNIV-023"],
                expected_tool_calls=[{"name": "book_slot"}],
            )
        ],
        agent={"tools": [{"name": "book_slot", "description": "Book an appointment slot."}, "cancel_slot"]},
    )
    fake_adapter.replies["claimed_booking"] = ["Done, you are booked for Tuesday."]  # no tool call at all
    llm_router.on_judge(llm_router.verdict("fail", reason="claimed booking without a tool call"), criterion="UNIV-023")
    result = await _run(project.load(), adapter_factory=fake_adapter)
    calls = llm_router.judge_calls_for("UNIV-023")
    assert len(calls) == 1  # UNIV-023 runs although no tool was called
    criterion, _ = llm_router.judge_sections(calls[0])
    assert "Tools available to the agent:\nbook_slot - Book an appointment slot.\ncancel_slot" in criterion
    assert "Tools this scenario expects: book_slot" in criterion
    metrics = _metrics(result.scenario_results[0])
    assert metrics["UNIV-023"].label == "poor"
    assert metrics["UNIV-025"].reason == "expected tool book_slot was never called"


@pytest.mark.asyncio
async def test_agent_tools_empty_skips_univ023_without_judge_call(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [project.scenario("no_tools", max_turns=1, metrics=["UNIV-013", "UNIV-023"])],
        agent={"tools": []},
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    metrics = _metrics(result.scenario_results[0])
    assert metrics["UNIV-023"].skipped is True
    assert metrics["UNIV-023"].label == "skipped"
    assert llm_router.judge_calls_for("UNIV-023") == []
    assert len(llm_router.judge_calls_for("UNIV-013")) == 1


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 14): a tool-call-only reply (no text) is an agent turn with blank
# text. The simulator LLM receives it as "(no speech)", never as an empty message, so a provider
# that rejects empty content (Gemini; stand-in: a LiteLLM BadRequestError for any blank message)
# does not turn a well-behaved agent into a scenario error. The transcript keeps the blank text.
# =============================================================================================

NO_SPEECH = "(no speech)"
TOOL_ONLY_AGENT = """
def respond(user_text, history):
    return {"tool_calls": [{"name": "book_slot", "arguments": {"date": "2026-01-01"}, "success": True}]}
"""


def _reject_blank_content(llm_router):
    """Wrap the router so that any LLM call with a blank non-system message raises the error a
    provider that rejects empty content gives (a stand-in for Gemini's rejection)."""
    import litellm

    route = llm_router.mock.responder
    rejected = []

    def strict(call):
        for message in call.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if message.get("role") != "system" and (not isinstance(content, str) or not content.strip()):
                rejected.append(call)
                raise litellm.BadRequestError(
                    message="GeminiException - contents.parts must not be empty",
                    model=call.get("model") or "mock-model",
                    llm_provider="gemini",
                )
        return route(call)

    llm_router.mock.responder = strict
    return rejected


@pytest.mark.asyncio
async def test_tool_call_only_reply_never_sends_blank_content_to_the_simulator(impulse_project, llm_router):
    m = _m()
    project = impulse_project
    name = "tool_call_only"
    project.write_agent(TOOL_ONLY_AGENT)
    llm_router.script(project.persona(name), "Did the booking go through?")
    rejected = _reject_blank_content(llm_router)
    project.write_config(
        [project.scenario(name, max_turns=2, opening_user_message="Please book me for Tuesday.", metrics=["UNIV-023"])]
    )
    result = await _run(project.load())
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    assert rejected == []
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert turns[1].text == ""   # the transcript keeps the blank text
    assert turns[1].tool_calls == [m.ToolCall(name="book_slot", arguments={"date": "2026-01-01"}, success=True)]
    assert turns[2].text == "Did the booking go through?"
    simulator_calls = llm_router.calls("simulator")
    assert len(simulator_calls) == 1
    assert llm_router.contents(simulator_calls[0])[-1] == NO_SPEECH


@pytest.mark.asyncio
async def test_fake_adapter_tool_call_only_reply_reaches_the_simulator_as_no_speech(
    impulse_project, llm_router, fake_adapter
):
    project = impulse_project
    name = "fake_tool_only"
    fake_adapter.replies[name] = [
        {"text": "", "tool_calls": [SAMPLE_TOOL_CALL], "latency_ms": 120.0},
        "Your appointment is booked for Tuesday.",
    ]
    llm_router.script(project.persona(name), "Did that go through?")
    rejected = _reject_blank_content(llm_router)
    project.write_config(
        [project.scenario(name, max_turns=2, opening_user_message="Please book me for Tuesday.", metrics=["UNIV-023"])]
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    assert rejected == []
    assert [turn.text for turn in scenario_result.transcript.turns] == [
        "Please book me for Tuesday.", "", "Did that go through?", "Your appointment is booked for Tuesday.",
    ]
    contents = llm_router.contents(llm_router.calls("simulator")[0])
    assert contents[-1] == NO_SPEECH
    assert all(text.strip() for text in contents)


# =============================================================================================
# Audit batch 3 (AUDIT.md findings 32 and 34) through ``runner.run``. F32: a generic agent that
# returns an OpenAI-style tool call ends the scenario with the generic adapter's clear TypeError
# text as the scenario error (never a raw pydantic dump), before any judge call. F34: a tool call
# whose ``error`` is "" is shown with no dangling " error=" suffix in the transcript the UNIV-023
# judge reads.
# =============================================================================================

OPENAI_STYLE_TOOL_CALL_AGENT = """
def respond(user_text, history):
    return {
        "text": "Booked.",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "book_slot", "arguments": '{"date": "2026-01-01"}'}}
        ],
    }
"""


@pytest.mark.asyncio
async def test_openai_style_tool_call_from_a_generic_agent_is_a_clear_scenario_error(impulse_project, llm_router):
    project = impulse_project
    entrypoint = project.write_agent(OPENAI_STYLE_TOOL_CALL_AGENT)
    project.write_config([project.scenario("openai_shape", max_turns=1, metrics=["UNIV-023"])])
    result = await _run(project.load())
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == f"agent function {entrypoint} returned a tool call without a string 'name'"
    assert "errors.pydantic.dev" not in result.model_dump_json()
    assert scenario_result.passed is False
    assert scenario_result.metric_results == []
    assert llm_router.calls("judge") == []


@pytest.mark.asyncio
async def test_empty_tool_call_error_has_no_dangling_suffix_in_the_judged_transcript(
    impulse_project, llm_router, fake_adapter
):
    project = impulse_project
    project.write_config([project.scenario("empty_error", max_turns=1, metrics=["UNIV-023"])])
    fake_adapter.replies["empty_error"] = [
        {
            "text": "That slot is not available.",
            "tool_calls": [{"name": "book_slot", "arguments": {"date": "2026-01-01"}, "success": False, "error": ""}],
        }
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    line = '[turn 2] AGENT TOOL CALL: book_slot({"date": "2026-01-01"}) -> success=False'
    assert line in _format(scenario_result.transcript).splitlines()
    calls = llm_router.judge_calls_for("UNIV-023")
    assert len(calls) == 1
    _, judged_transcript = llm_router.judge_sections(calls[0])
    assert line in judged_transcript.splitlines()
    assert " error=" not in judged_transcript
