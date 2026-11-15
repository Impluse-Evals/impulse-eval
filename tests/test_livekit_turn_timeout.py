# SPDX-License-Identifier: Apache-2.0
"""Fixes round 2, A6 (DEFECT-2): LiveKit turn timeout; the tool calls of a timed-out turn are kept
(FIXES-ROUND-2-SPEC.md).

* config: ``agent.livekit.turn_timeout_s`` (``LiveKitConfig.turn_timeout_s: Optional[float] = None``);
  when set it must be > 0 (``ConfigError`` on ``agent.livekit.turn_timeout_s``); B-rev (review R3):
  a finite number > 0, booleans and infinity rejected on the same field.
* adapter: the turn timeout is ``turn_timeout_s`` when set (text and voice mode), else
  ``voice_config.timeout_s`` when ``voice_config`` exists, else 30; error text unchanged ("LiveKit
  agent did not finish its turn within {N} s."). The error is both a ``LiveKitError`` and an
  ``AgentTurnError`` (``impulse_evals.adapters.base``, Apache-2.0, in ``__all__``) whose ``reply`` is
  the turn so far: final agent text, the turn's tool calls (``function_tools_executed``),
  ``interrupted``, audio (voice), ``latency_ms`` measured or NaN when no output arrived.
* runner: an ``AgentTurnError`` whose ``reply`` has text, a tool call, or audio is first recorded as
  an agent turn like a normal reply (heard_text on the user turn; a non-finite latency becomes
  None), then the scenario error is set from the exception as before (no evaluation).

Campaign case: run 20261021T192126Z-60e8a2 (SupportBot, LiveKit, text mode) ``damaged_item_ticket``:
the second agent turn called ``lookup_order`` and ``create_ticket`` and its final text came after the
30 s timeout, so run.json lost both calls. Adapter tests run on ``mock_livekit_room`` (virtual clock,
no network, no livekit); runner tests use ``fake_adapter`` / ``llm_router``.
"""

import json
import math

import pytest
from conftest import pcm_tone, wav_bytes

pytestmark = pytest.mark.phase11

LK_URL = "ws://localhost:7880"
AGENT = "agent-neo"
TURN_TIMEOUT = "LiveKit agent did not finish its turn within {n} s."
VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}

# Campaign data (run 20261021T192126Z-60e8a2 damaged_item_ticket; tool shapes from the SupportBot runs).
SCENARIO = "damaged_item_ticket"
OPENING = "Hi, my coffee mugs arrived cracked. Can you log a complaint?"
FIRST_REPLY = (
    "I'm so sorry to hear that your coffee mugs arrived cracked! Could you please provide your order ID "
    "so I can look into this and get a support ticket opened for you?"
)
SECOND_USER = "Haan, order ID hai ORD-1003."
PARTIAL_TEXT = "Let me look up that order for you."
LOOKUP_ARGS = {"order_id": "ORD-1003"}
LOOKUP_OUTPUT = {
    "ok": True, "order_id": "ORD-1003", "item": "Ceramic coffee mug set", "amount_inr": 899,
    "status": "delivered", "delivered_on": "2026-10-19",
}
TICKET_ARGS = {"summary": "Order ORD-1003: Ceramic coffee mug set arrived cracked"}
TICKET_OUTPUT = {"ok": True, "ticket_id": "TCK-50001", "summary": "Order ORD-1003: Ceramic coffee mug set arrived cracked"}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _cfg(project, *, voice=None, scenarios=None, **livekit):
    block = {"url": LK_URL}
    block.update(livekit)
    kwargs = {}
    if voice is not None:
        kwargs["voice_config"] = dict(VOICE_CONFIG, **voice)
    project.write_config(
        scenarios or [project.scenario(SCENARIO, opening_user_message=OPENING)],
        agent={"adapter": "livekit", "livekit": block},
        **kwargs,
    )
    return project.load()


def _adapter(cfg, mock):
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    return LiveKitAdapter(cfg, room_factory=mock.factory, clock=mock.clock)


async def _started(cfg, mock):
    mock.room.on_connect(mock.participant(AGENT, is_agent=True))
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    return adapter


def _plan(room, steps, fired=None):
    def trigger(_record):
        if fired is not None:
            fired.append(room.clock.monotonic())
        for offset, item in steps:
            room.emit(item, after=offset)

    return trigger


def _on_chat(mock, steps, fired=None, text=None):
    room = mock.room
    match = (lambda rec: rec["topic"] == "lk.chat") if text is None else (
        lambda rec: rec["topic"] == "lk.chat" and rec["text"] == text
    )
    room.on_text(match, _plan(room, steps, fired))


def _function_calls_event(mock, name, arguments, output, call_id):
    """One ``function_tools_executed`` event as LiveKit Agents sends it (arguments / output as JSON text)."""
    return mock.tools_executed(
        [{"name": name, "arguments": json.dumps(arguments), "call_id": call_id}],
        [{"call_id": call_id, "name": name, "output": json.dumps(output), "is_error": False}],
    )


def _campaign_steps(mock, *, with_text=False):
    """The timed-out turn: the agent thinks, runs lookup_order then create_ticket, and stays busy; its
    final text would only come at +40 s, after any timeout used here."""
    steps = [(0.1, mock.state("thinking"))]
    if with_text:
        steps += [
            (0.5, mock.state("speaking")),
            (1.0, mock.agent_text(PARTIAL_TEXT)),
            (1.2, mock.state("thinking")),   # a tool is running: still busy
        ]
    steps += [
        (2.0, _function_calls_event(mock, "lookup_order", LOOKUP_ARGS, LOOKUP_OUTPUT, "call_lookup")),
        (4.0, _function_calls_event(mock, "create_ticket", TICKET_ARGS, TICKET_OUTPUT, "call_ticket")),
        (40.0, mock.agent_text("I've opened ticket TCK-50001 for your cracked mugs.")),
    ]
    return steps


def _base():
    from impulse_evals.adapters import base

    return base


# =============================================================================================
# config.yaml: agent.livekit.turn_timeout_s
# =============================================================================================


def test_turn_timeout_s_is_accepted_and_defaults_to_none(impulse_project):
    cfg = _cfg(impulse_project, turn_timeout_s=12.5)
    assert cfg.agent.livekit.turn_timeout_s == 12.5
    cfg = _cfg(impulse_project)
    assert cfg.agent.livekit.turn_timeout_s is None


@pytest.mark.parametrize("value", [0, -5, "soon"])
def test_turn_timeout_s_must_be_a_positive_number(impulse_project, value):
    from impulse_evals.config import ConfigError

    with pytest.raises(ConfigError) as info:
        _cfg(impulse_project, turn_timeout_s=value)
    assert info.value.field == "agent.livekit.turn_timeout_s"


@pytest.mark.parametrize("raw", [".inf", "1e400", "true"])
def test_turn_timeout_s_must_be_finite_and_not_a_boolean(impulse_project, raw):
    """B-rev (review R3 L2): ``.inf`` and ``1e400`` (both infinity: a hung agent would hang the run)
    and ``true`` (a boolean, coerced to 1.0 s) are rejected. The raw YAML text is written as is."""
    from impulse_evals.config import ConfigError

    project = impulse_project
    placeholder = "__TURN_TIMEOUT_S__"
    path = project.write_config(
        [project.scenario(SCENARIO, opening_user_message=OPENING)],
        agent={"adapter": "livekit", "livekit": {"url": LK_URL, "turn_timeout_s": placeholder}},
    )
    text = path.read_text(encoding="utf-8")
    assert text.count(f"turn_timeout_s: {placeholder}") == 1
    path.write_text(text.replace(f"turn_timeout_s: {placeholder}", f"turn_timeout_s: {raw}"), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        project.load()
    assert info.value.field == "agent.livekit.turn_timeout_s"


@pytest.mark.parametrize("raw", ["true", "false"])
def test_boolean_turn_timeout_s_message_has_no_pydantic_prefix(impulse_project, raw):
    """Re-check nit (FIXES-ROUND-2-SPEC.md "Re-check revisions"): the boolean error message is the
    field's own text, without pydantic's ``Value error, `` prefix (the field's other messages have none)."""
    from impulse_evals.config import ConfigError

    project = impulse_project
    placeholder = "__TURN_TIMEOUT_S__"
    path = project.write_config(
        [project.scenario(SCENARIO, opening_user_message=OPENING)],
        agent={"adapter": "livekit", "livekit": {"url": LK_URL, "turn_timeout_s": placeholder}},
    )
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(f"turn_timeout_s: {placeholder}", f"turn_timeout_s: {raw}"), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        project.load()
    assert info.value.field == "agent.livekit.turn_timeout_s"
    assert "Value error, " not in info.value.message, info.value.message
    assert "Value error, " not in str(info.value), str(info.value)
    assert "boolean" in info.value.message, info.value.message


# =============================================================================================
# AgentTurnError (adapters/base.py, Apache-2.0)
# =============================================================================================


def test_agent_turn_error_is_exported_and_carries_the_reply():
    base = _base()
    assert "AgentTurnError" in base.__all__
    error_cls = base.AgentTurnError
    assert issubclass(error_cls, Exception)
    plain = error_cls("turn failed")
    assert str(plain) == "turn failed"
    assert plain.reply is None
    reply = base.AgentReply(text="partial", latency_ms=120.0)
    error = error_cls("turn failed", reply=reply)
    assert error.reply is reply
    assert str(error) == "turn failed"


# =============================================================================================
# Adapter: which timeout applies (fake clock)
# =============================================================================================


@pytest.mark.asyncio
async def test_text_turn_times_out_at_turn_timeout_s(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, turn_timeout_s=12)
    adapter = await _started(cfg, mock)
    fired = []
    _on_chat(mock, [(0.1, mock.state("thinking"))], fired)   # never finishes
    with pytest.raises(Exception) as info:
        await adapter.send("Hello?", None)
    assert str(info.value) == TURN_TIMEOUT.format(n=12)
    assert fired[0] + 12 - 0.01 <= mock.clock.now <= fired[0] + 12 + 0.5
    await adapter.close()


@pytest.mark.asyncio
async def test_turn_timeout_s_wins_over_voice_config_timeout_in_voice_mode(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, voice={"timeout_s": 30}, turn_timeout_s=5)
    adapter = await _started(cfg, mock)
    room = mock.room
    room.on_audio(_plan(room, [(0.1, mock.state("thinking"))]))   # never finishes
    started = mock.clock.now
    with pytest.raises(Exception) as info:
        await adapter.send("Book a slot", wav_bytes(pcm_tone(0.5, 16000)))
    assert str(info.value) == TURN_TIMEOUT.format(n=5)
    assert mock.clock.now < started + 30   # the voice_config timeout was not used
    await adapter.close()


@pytest.mark.asyncio
async def test_voice_config_timeout_applies_without_turn_timeout_s(impulse_project, mock_livekit_room):
    # Regression guard (passes before the fix).
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, voice={"timeout_s": 7})
    adapter = await _started(cfg, mock)
    fired = []
    _on_chat(mock, [(0.1, mock.state("thinking"))], fired)
    with pytest.raises(Exception) as info:
        await adapter.send("Hello?", None)
    assert str(info.value) == TURN_TIMEOUT.format(n=7)
    assert fired[0] + 7 - 0.01 <= mock.clock.now <= fired[0] + 7 + 0.5
    await adapter.close()


@pytest.mark.asyncio
async def test_text_turn_timeout_defaults_to_30_s(impulse_project, mock_livekit_room):
    # Regression guard (passes before the fix).
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    fired = []
    _on_chat(mock, [(0.1, mock.state("thinking"))], fired)
    with pytest.raises(Exception) as info:
        await adapter.send("Hello?", None)
    assert str(info.value) == TURN_TIMEOUT.format(n=30)
    assert fired[0] + 30 - 0.01 <= mock.clock.now <= fired[0] + 30 + 0.5
    await adapter.close()


# =============================================================================================
# Adapter: the timeout error is a LiveKitError AND an AgentTurnError carrying the turn so far
# =============================================================================================


@pytest.mark.asyncio
async def test_timeout_error_carries_the_tool_calls_of_the_turn(impulse_project, mock_livekit_room):
    """The campaign case: lookup_order and create_ticket ran, the final text came after the timeout."""
    from impulse_evals.adapters.livekit_adapter import LiveKitError

    base = _base()
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    _on_chat(mock, [(0.1, mock.state("speaking")), (0.6, mock.agent_text(FIRST_REPLY)),
                    (0.7, mock.state("listening"))], text=OPENING)
    first = await adapter.send(OPENING, None)
    assert first.text == FIRST_REPLY

    _on_chat(mock, _campaign_steps(mock), text=SECOND_USER)
    with pytest.raises(Exception) as info:
        await adapter.send(SECOND_USER, None)
    error = info.value
    assert str(error) == TURN_TIMEOUT.format(n=30)
    assert isinstance(error, LiveKitError)
    assert isinstance(error, base.AgentTurnError)
    reply = error.reply
    assert isinstance(reply, base.AgentReply)
    assert [call.name for call in reply.tool_calls] == ["lookup_order", "create_ticket"]
    assert reply.tool_calls[0].arguments == LOOKUP_ARGS
    assert reply.tool_calls[1].arguments == TICKET_ARGS
    assert reply.tool_calls[0].result == LOOKUP_OUTPUT
    assert reply.tool_calls[1].result == TICKET_OUTPUT
    assert reply.text == ""            # no final text arrived before the timeout
    assert math.isnan(reply.latency_ms)   # no output arrived
    assert reply.interrupted is False
    assert reply.audio_bytes is None
    await adapter.close()


@pytest.mark.asyncio
async def test_timeout_error_carries_the_text_so_far_and_its_latency(impulse_project, mock_livekit_room):
    base = _base()
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, turn_timeout_s=20)
    adapter = await _started(cfg, mock)
    fired = []
    _on_chat(mock, _campaign_steps(mock, with_text=True), fired)
    with pytest.raises(base.AgentTurnError) as info:
        await adapter.send(SECOND_USER, None)
    assert str(info.value) == TURN_TIMEOUT.format(n=20)
    reply = info.value.reply
    assert reply.text == PARTIAL_TEXT
    assert [call.name for call in reply.tool_calls] == ["lookup_order", "create_ticket"]
    assert reply.latency_ms == pytest.approx(1000.0, abs=5.0)   # send -> the first final agent text
    await adapter.close()


# =============================================================================================
# Runner: the partial turn of an AgentTurnError is recorded before the scenario error
# =============================================================================================


def _tool_call(name, arguments, result):
    from impulse_evals.models import ToolCall

    return ToolCall(name=name, arguments=dict(arguments), result=dict(result))


def _turn_error(reply=None, message=None):
    return _base().AgentTurnError(message or TURN_TIMEOUT.format(n=30), reply=reply)


def _partial_reply(**fields):
    values = {
        "text": "",
        "latency_ms": float("nan"),
        "tool_calls": [
            _tool_call("lookup_order", LOOKUP_ARGS, LOOKUP_OUTPUT),
            _tool_call("create_ticket", TICKET_ARGS, TICKET_OUTPUT),
        ],
    }
    values.update(fields)
    return _base().AgentReply(**values)


def _campaign_project(project, llm_router, **scenario):
    project.write_config(
        [project.scenario(SCENARIO, max_turns=3, opening_user_message=OPENING, assertions=["dpdp.consent_before_pii"],
                          metrics=["UNIV-001", "UNIV-023"], **scenario)]
    )
    llm_router.script(project.persona(SCENARIO), SECOND_USER, "Thank you.")
    return project.load()


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


@pytest.mark.asyncio
async def test_runner_keeps_the_partial_turn_and_its_tool_calls_before_the_error(
    impulse_project, llm_router, fake_adapter, tmp_path
):
    from impulse_evals.models import RunResult
    from impulse_evals.reporter import write_json

    cfg = _campaign_project(impulse_project, llm_router)
    fake_adapter.replies[SCENARIO] = [{"text": FIRST_REPLY, "latency_ms": 3890.0}]
    fake_adapter.send_errors[SCENARIO] = (2, _turn_error(_partial_reply()))
    result = await _run(cfg, adapter_factory=fake_adapter)

    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert [turn.text for turn in turns[:3]] == [OPENING, FIRST_REPLY, SECOND_USER]
    partial = turns[3]
    assert partial.text == ""
    assert [call.name for call in partial.tool_calls] == ["lookup_order", "create_ticket"]
    assert partial.tool_calls[0].arguments == LOOKUP_ARGS
    assert partial.tool_calls[1].arguments == TICKET_ARGS
    assert partial.latency_ms is None          # NaN from the adapter becomes None
    assert partial.timestamp_ms >= turns[2].timestamp_ms
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)
    assert scenario_result.assertion_results == []   # no evaluation, as for any error before evaluation
    assert scenario_result.metric_results == []
    assert scenario_result.passed is False
    assert llm_router.calls("judge") == []

    # ... and therefore run.json holds both calls.
    path = tmp_path / "run.json"
    write_json(result, path)
    stored = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    stored_turns = stored.scenario_results[0].transcript.turns
    assert [call.name for call in stored_turns[3].tool_calls] == ["lookup_order", "create_ticket"]
    assert json.loads(path.read_text(encoding="utf-8"))["scenario_results"][0]["error"] == TURN_TIMEOUT.format(n=30)


@pytest.mark.asyncio
async def test_runner_records_partial_text_latency_and_heard_text(impulse_project, llm_router, fake_adapter):
    cfg = _campaign_project(impulse_project, llm_router)
    fake_adapter.replies[SCENARIO] = [{"text": FIRST_REPLY, "latency_ms": 3890.0}]
    reply = _partial_reply(text=PARTIAL_TEXT, latency_ms=1200.0, heard_text="order id hai ORD-1003")
    fake_adapter.send_errors[SCENARIO] = (2, _turn_error(reply))
    result = await _run(cfg, adapter_factory=fake_adapter)

    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert turns[3].text == PARTIAL_TEXT
    assert turns[3].latency_ms == pytest.approx(1200.0)
    assert turns[2].heard_text == "order id hai ORD-1003"
    assert [call.name for call in turns[3].tool_calls] == ["lookup_order", "create_ticket"]
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)
    assert scenario_result.passed is False


@pytest.mark.asyncio
async def test_runner_keeps_a_partial_opening_turn(impulse_project, llm_router, fake_adapter):
    # The very first send times out after a tool call: the partial agent turn is still recorded.
    cfg = _campaign_project(impulse_project, llm_router)
    fake_adapter.send_errors[SCENARIO] = (1, _turn_error(_partial_reply(tool_calls=[
        _tool_call("lookup_order", LOOKUP_ARGS, LOOKUP_OUTPUT),
    ])))
    result = await _run(cfg, adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent"]
    assert [call.name for call in turns[1].tool_calls] == ["lookup_order"]
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", ["blank_reply", "no_reply"])
async def test_runner_adds_no_turn_for_a_reply_without_content(impulse_project, llm_router, fake_adapter, empty):
    cfg = _campaign_project(impulse_project, llm_router)
    fake_adapter.replies[SCENARIO] = [{"text": FIRST_REPLY, "latency_ms": 3890.0}]
    reply = _partial_reply(text="   ", tool_calls=[]) if empty == "blank_reply" else None
    fake_adapter.send_errors[SCENARIO] = (2, _turn_error(reply))
    result = await _run(cfg, adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert [turn.speaker for turn in scenario_result.transcript.turns] == ["user", "agent", "user"]
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)
    assert scenario_result.passed is False


@pytest.mark.asyncio
async def test_runner_plain_exception_behaves_as_before(impulse_project, llm_router, fake_adapter):
    # Regression guard (passes before the fix): a plain adapter exception adds no turn.
    cfg = _campaign_project(impulse_project, llm_router)
    fake_adapter.replies[SCENARIO] = [{"text": FIRST_REPLY, "latency_ms": 3890.0}]
    fake_adapter.send_errors[SCENARIO] = (2, RuntimeError(TURN_TIMEOUT.format(n=30)))
    result = await _run(cfg, adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert [turn.speaker for turn in scenario_result.transcript.turns] == ["user", "agent", "user"]
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)
    assert scenario_result.assertion_results == []
    assert scenario_result.metric_results == []
    assert llm_router.calls("judge") == []


@pytest.mark.asyncio
async def test_runner_with_livekit_adapter_keeps_the_timed_out_turns_tool_calls(
    impulse_project, llm_router, mock_livekit_room
):
    """End to end on the fake room: the campaign run, reproduced."""
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    mock = mock_livekit_room
    cfg = _cfg(impulse_project, scenarios=[impulse_project.scenario(
        SCENARIO, max_turns=3, opening_user_message=OPENING, metrics=["UNIV-001"],
    )])
    llm_router.script(impulse_project.persona(SCENARIO), SECOND_USER, "Thank you.")
    mock.room.on_connect(mock.participant(AGENT, is_agent=True))
    _on_chat(mock, [(0.1, mock.state("speaking")), (0.6, mock.agent_text(FIRST_REPLY)),
                    (0.7, mock.state("listening"))], text=OPENING)
    _on_chat(mock, _campaign_steps(mock), text=SECOND_USER)

    result = await _run(
        cfg, adapter_factory=lambda config: LiveKitAdapter(config, room_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert [call.name for call in turns[3].tool_calls] == ["lookup_order", "create_ticket"]
    assert turns[3].tool_calls[0].arguments == LOOKUP_ARGS
    assert turns[3].tool_calls[1].arguments == TICKET_ARGS
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)
    assert "create_ticket" in result.model_dump_json()


# =============================================================================================
# Runner, voice mode: the partial turn goes through the same voice step
# =============================================================================================


@pytest.mark.asyncio
async def test_runner_voice_mode_records_the_partial_turn_with_its_audio(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    project = impulse_project
    project.write_config(
        [project.scenario(SCENARIO, max_turns=2, opening_user_message=OPENING, metrics=["VOICE-029"])],
        voice_config=dict(VOICE_CONFIG, save_audio=True, save_recording=False),
    )
    agent_audio = wav_bytes(pcm_tone(0.6, 16000, frequency_hz=330.0))
    reply = _partial_reply(text=PARTIAL_TEXT, latency_ms=900.0, audio_bytes=agent_audio)
    fake_adapter.send_errors[SCENARIO] = (1, _turn_error(reply))
    result = await _run(project.load(), adapter_factory=fake_adapter)

    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent"]
    agent = turns[1]
    assert agent.text == PARTIAL_TEXT
    assert [call.name for call in agent.tool_calls] == ["lookup_order", "create_ticket"]
    assert agent.audio_path is not None            # saved like any voice-mode agent turn
    assert agent.end_ms is not None and agent.end_ms > agent.timestamp_ms
    assert agent.timestamp_ms >= (turns[0].end_ms or 0) + 900 - 1
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)
    run_dir = project.root / ".impulse" / "runs" / result.run_id
    assert (run_dir / agent.audio_path.replace("\\", "/")).is_file()
