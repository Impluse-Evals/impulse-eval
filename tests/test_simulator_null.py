# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 5: the simulated caller survives None / empty / whitespace LLM output (SPEC.md section 10).

``Simulator(model, options, persona_style)``; ``await next_turn(scenario, summary, turns)`` makes one
LLM call through ``litellm.acompletion`` / ``litellm.completion`` and returns the user's next text,
``"I see, please continue."`` when the answer is empty, or ``None`` when the answer contains
``<END_CALL>`` (contract logged in DECISIONS.md, Stage B group 4-8).
"""

import pytest

FALLBACK = "I see, please continue."
SUMMARY = "Books appointments at Sunrise Clinic."
SIMULATOR_MODEL = "ollama/simulator"


def _m():
    from impulse_evals import models

    return models


def _simulator(options=None, persona_style="realistic"):
    from impulse_evals.simulator import Simulator

    return Simulator(SIMULATOR_MODEL, options or {}, persona_style)


def _scenario(**kwargs):
    return _m().Scenario(
        name="clinic_appointment_booking",
        language="en-IN",
        domain="healthcare",
        max_turns=12,
        persona="patient who wants the earliest slot on Tuesday",
        **kwargs,
    )


def _turns():
    m = _m()
    return [
        m.Turn(speaker="user", text="Hi, I need an appointment.", timestamp_ms=0),
        m.Turn(speaker="agent", text="Sure, which day suits you?", timestamp_ms=900),
    ]


async def _next(simulator=None, turns=None, scenario=None):
    simulator = simulator or _simulator()
    return await simulator.next_turn(scenario or _scenario(), SUMMARY, _turns() if turns is None else turns)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["none", "empty", "whitespace"])
async def test_empty_answers_use_the_fallback_text(mock_llm, kind):
    {"none": mock_llm.queue_none, "empty": mock_llm.queue_empty, "whitespace": mock_llm.queue_whitespace}[kind]()
    assert await _next() == FALLBACK
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_tool_call_only_answer_uses_the_fallback_text(mock_llm):
    mock_llm.queue_tool_call(name="lookup_slots", arguments={"day": "Tuesday"})
    assert await _next() == FALLBACK
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_every_answer_is_null_checked(mock_llm):
    mock_llm.queue_none().queue("Tuesday morning, please.").queue_empty()
    simulator = _simulator()
    assert await _next(simulator) == FALLBACK
    assert await _next(simulator) == "Tuesday morning, please."
    assert await _next(simulator) == FALLBACK
    assert mock_llm.call_count == 3


@pytest.mark.asyncio
async def test_text_is_stripped(mock_llm):
    mock_llm.queue("  \n Tuesday morning, please.  \n")
    assert await _next() == "Tuesday morning, please."


@pytest.mark.asyncio
async def test_first_turn_with_empty_conversation(mock_llm):
    mock_llm.queue_none()
    assert await _next(turns=[]) == FALLBACK
    mock_llm.queue("Hello, I'd like an appointment.")
    assert await _next(turns=[]) == "Hello, I'd like an appointment."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["<END_CALL>", "  <END_CALL>\n", "Thanks, that's all I needed. <END_CALL>", "<END_CALL> Goodbye."],
)
async def test_end_call_anywhere_ends_the_call(mock_llm, content):
    mock_llm.queue(content)
    assert await _next() is None


@pytest.mark.asyncio
async def test_message_roles_agent_as_user_simulator_as_assistant(mock_llm):
    m = _m()
    mock_llm.queue("Tuesday at 10 works.")
    scenario = _scenario()
    turns = [
        m.Turn(speaker="user", text="Hi, I need an appointment.", timestamp_ms=0),
        m.Turn(speaker="agent", text="Sure, which day suits you?", timestamp_ms=900),
        m.Turn(speaker="user", text="Tuesday, please.", timestamp_ms=3000),
        m.Turn(speaker="agent", text="Morning or afternoon?", timestamp_ms=3900),
    ]
    simulator = _simulator()
    await simulator.next_turn(scenario, SUMMARY, turns)
    messages = mock_llm.messages(0)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == simulator.build_system_prompt(scenario, SUMMARY)
    assert [(msg["role"], msg["content"]) for msg in messages[-4:]] == [
        ("assistant", "Hi, I need an appointment."),
        ("user", "Sure, which day suits you?"),
        ("assistant", "Tuesday, please."),
        ("user", "Morning or afternoon?"),
    ]


@pytest.mark.asyncio
async def test_outbound_conversation_starts_with_the_agent_as_user(mock_llm):
    m = _m()
    mock_llm.queue("Yes, speaking.")
    turns = [m.Turn(speaker="agent", text="Hello, this is Sunrise Clinic. Am I speaking to Test User?", timestamp_ms=0)]
    assert await _next(turns=turns, scenario=_scenario(direction="outbound")) == "Yes, speaking."
    messages = mock_llm.messages(0)
    assert (messages[-1]["role"], messages[-1]["content"]) == (
        "user",
        "Hello, this is Sunrise Clinic. Am I speaking to Test User?",
    )


@pytest.mark.asyncio
async def test_uses_the_simulator_model_and_options(mock_llm):
    mock_llm.queue("Tuesday, please.")
    simulator = _simulator(options={"api_base": "http://localhost:11434"}, persona_style="polite")
    await _next(simulator)
    call = mock_llm.calls[0]
    assert call["model"] == SIMULATOR_MODEL
    assert call["kwargs"].get("api_base") == "http://localhost:11434"


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 14): an agent turn whose text is blank (a tool-call-only reply, a
# voice turn with no audio or a blank STT text) is sent to the simulator LLM as "(no speech)", the
# same placeholder for every blank agent turn; no message with blank content is ever sent (Gemini
# and Anthropic reject empty content). The turn is never dropped and the transcript is unchanged.
# =============================================================================================

NO_SPEECH = "(no speech)"


def _blank_messages(messages):
    """Non-system messages whose content is not a non-blank string."""
    return [
        message
        for message in messages
        if message["role"] != "system"
        and (not isinstance(message.get("content"), str) or not message["content"].strip())
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", "   ", " \n\t "], ids=["empty", "spaces", "whitespace"])
async def test_blank_agent_turn_is_sent_to_the_simulator_as_no_speech(mock_llm, blank):
    m = _m()
    mock_llm.queue("Did that go through?")
    book = m.ToolCall(name="book_slot", arguments={"date": "2026-01-01"}, success=True)
    turns = [
        m.Turn(speaker="user", text="I want to book", timestamp_ms=0),
        m.Turn(speaker="agent", text=blank, timestamp_ms=900, tool_calls=[book]),
    ]
    assert await _next(turns=turns) == "Did that go through?"
    messages = mock_llm.messages(0)
    assert _blank_messages(messages) == [], messages
    assert (messages[-1]["role"], messages[-1]["content"]) == ("user", NO_SPEECH)
    assert (messages[-2]["role"], messages[-2]["content"]) == ("assistant", "I want to book")
    # The transcript itself is unchanged.
    assert turns[1].text == blank
    assert turns[1].tool_calls == [book]


@pytest.mark.asyncio
async def test_every_blank_agent_turn_gets_the_placeholder_and_none_is_dropped(mock_llm):
    m = _m()
    mock_llm.queue("Can you hear me now?")
    turns = [
        m.Turn(speaker="user", text="Hi, I need an appointment.", timestamp_ms=0),
        m.Turn(speaker="agent", text="", timestamp_ms=900),
        m.Turn(speaker="user", text="Hello?", timestamp_ms=3000),
        m.Turn(speaker="agent", text="Sorry, which day suits you?", timestamp_ms=3900),
        m.Turn(speaker="user", text="Tuesday.", timestamp_ms=6000),
        m.Turn(speaker="agent", text="  \n", timestamp_ms=6900),
    ]
    await _next(turns=turns)
    messages = mock_llm.messages(0)
    assert _blank_messages(messages) == [], messages
    assert [(message["role"], message["content"]) for message in messages[-6:]] == [
        ("assistant", "Hi, I need an appointment."),
        ("user", NO_SPEECH),
        ("assistant", "Hello?"),
        ("user", "Sorry, which day suits you?"),
        ("assistant", "Tuesday."),
        ("user", NO_SPEECH),
    ]


@pytest.mark.asyncio
async def test_blank_outbound_opener_is_sent_as_no_speech(mock_llm):
    m = _m()
    mock_llm.queue("Hello? Is anyone there?")
    turns = [m.Turn(speaker="agent", text="", timestamp_ms=0)]   # e.g. a voice opener with no audio
    assert await _next(turns=turns, scenario=_scenario(direction="outbound")) == "Hello? Is anyone there?"
    messages = mock_llm.messages(0)
    assert _blank_messages(messages) == [], messages
    assert [(message["role"], message["content"]) for message in messages[1:]] == [("user", NO_SPEECH)]
