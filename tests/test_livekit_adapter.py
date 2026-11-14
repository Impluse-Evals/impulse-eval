# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 11: the LiveKit adapter (SPEC.md sections 8.2 and 9, as updated by the pre-Stage-B source read
of livekit 1.1.21 / livekit-api 1.2.2 / livekit-protocol 1.1.27; DECISIONS.md D11-D16).

``LiveKitAdapter(config, *, room_factory=None, clock=None)`` takes the ``ProjectConfig``; every test
injects ``mock_livekit_room`` (a scripted fake ``Room`` plus a virtual ``Clock``), so neither
``livekit`` nor a server is needed. Token creation (grants, ``agent_name`` dispatch through
``RoomConfiguration``) and the ``lk.agent.session`` protobuf codec live inside the real room class
and are not exercised here; the adapter passes ``agent_name`` to ``Room.connect()``.

Covered: room name and connect arguments; the agent participant (``agent_identity`` or the first
agent) and the join timeout; text turns on ``lk.chat`` with the reply from the agent's final
``lk.transcription`` streams; ``heard_text`` from final ``lk.transcription`` streams sent with
Impulse's own identity; the system prompt through ``request_agent_info`` (only without a config
prompt; 3 s wait); ``tool_calls`` from ``function_tools_executed``; barge-in from ``lk.agent.state``
leaving "speaking" while user speech is sent; end of turn after ``lk.agent.state`` leaves "speaking"
with no return within ``end_of_turn_grace_ms``; ``RoomClosed`` -> ``AgentClosed``; the missing-extra
error. Times are virtual seconds; "+x" means x seconds after the trigger.
"""

import json
import subprocess
import sys

import pytest
from conftest import REPO_ROOT, pcm_tone, read_wav, wav_bytes

pytestmark = pytest.mark.phase11

LK_URL = "ws://localhost:7880"
AGENT = "agent-neo"
DEFAULT = object()
JOIN_TIMEOUT = "LiveKit agent did not join within {n} s."
MISSING_EXTRA = "livekit adapter needs the 'livekit' extra: pip install impulse-evals[livekit]"
BOOK_ARGS = {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}
VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _cfg(project, *, voice=False, system_prompt=DEFAULT, scenarios=None, **livekit):
    block = {"url": LK_URL}
    block.update(livekit)
    kwargs = {}
    if voice:
        kwargs["voice_config"] = dict(VOICE_CONFIG)
    if system_prompt is not DEFAULT:
        kwargs["system_prompt"] = system_prompt
    project.write_config(
        scenarios or [project.scenario("livekit_probe")],
        agent={"adapter": "livekit", "livekit": block},
        **kwargs,
    )
    return project.load()


def _adapter(cfg, mock):
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    return LiveKitAdapter(cfg, room_factory=mock.factory, clock=mock.clock)


def _join(mock, room=None, identity=AGENT, is_agent=True, after=0.0):
    room = room or mock.room
    room.on_connect(mock.participant(identity, is_agent=is_agent), after=after)
    return room


async def _started(cfg, mock, after=0.0, scenario=None):
    _join(mock, after=after)
    adapter = _adapter(cfg, mock)
    await adapter.start(scenario or cfg.scenarios[0])
    return adapter


def _plan(room, steps, fired=None):
    """A rule item: when triggered, schedule each ``(offset_s, item)`` relative to the trigger time."""

    def trigger(_record):
        if fired is not None:
            fired.append(room.clock.monotonic())
        for offset, item in steps:
            room.emit(item, after=offset)

    return trigger


def _on_chat(mock, room, steps, fired=None, text=None):
    match = (lambda rec: rec["topic"] == "lk.chat") if text is None else (
        lambda rec: rec["topic"] == "lk.chat" and rec["text"] == text
    )
    room.on_text(match, _plan(room, steps, fired))


def _reply_steps(mock, *finals, start=0.1):
    """thinking, speaking, an interim (non-final) stream, the final streams, then listening."""
    steps = [
        (start, mock.state("thinking")),
        (start + 0.2, mock.state("speaking")),
        (start + 0.3, mock.agent_text("Hel", final=False)),
    ]
    t = start + 0.6
    for text in finals:
        steps.append((t, mock.agent_text(text)))
        t += 0.1
    steps.append((t + 0.1, mock.state("listening")))
    return steps


def _norm(text):
    return " ".join((text or "").split())


def _user_wav(seconds=1.0, sample_rate=16000):
    return wav_bytes(pcm_tone(seconds, sample_rate))


# =============================================================================================
# Connection
# =============================================================================================


@pytest.mark.asyncio
async def test_connect_passes_url_identity_room_and_agent_name(impulse_project, mock_livekit_room):
    cfg = _cfg(impulse_project, agent_name="neo-recruiter")
    adapter = await _started(cfg, mock_livekit_room)
    call = mock_livekit_room.room.connect_call
    assert call["url"] == LK_URL
    assert isinstance(call["identity"], str) and call["identity"]
    assert call["identity"] != AGENT
    assert call["room_name"].startswith("impulse-")  # default room_prefix
    assert call["room_name"].endswith("-livekit_probe")
    assert len(call["room_name"]) > len("impulse-") + len("-livekit_probe")  # the run id sits in between
    assert call["agent_name"] == "neo-recruiter"
    await adapter.close()
    assert mock_livekit_room.room.close_calls >= 1


@pytest.mark.asyncio
async def test_agent_name_defaults_to_none_and_room_prefix_is_used(impulse_project, mock_livekit_room):
    cfg = _cfg(impulse_project, room_prefix="t2n-")
    adapter = await _started(cfg, mock_livekit_room)
    call = mock_livekit_room.room.connect_call
    assert call["agent_name"] is None
    assert call["room_name"].startswith("t2n-")
    assert call["room_name"].endswith("-livekit_probe")
    await adapter.close()


@pytest.mark.asyncio
async def test_one_new_room_per_scenario(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    project = impulse_project
    cfg = _cfg(project, scenarios=[project.scenario("lk_first"), project.scenario("lk_second")])
    second_room = mock.add_room()
    rooms = []
    for scenario, room in zip(cfg.scenarios, (mock.room, second_room)):
        _join(mock, room=room)
        adapter = _adapter(cfg, mock)
        await adapter.start(scenario)
        await adapter.close()
        rooms.append(room.connect_call["room_name"])
    assert rooms[0].endswith("-lk_first") and rooms[1].endswith("-lk_second")
    assert rooms[0] != rooms[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("participant", [None, "non_agent"])
async def test_agent_join_timeout_gives_exact_error(impulse_project, mock_livekit_room, participant):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, handshake_timeout_s=4)
    if participant == "non_agent":
        mock.room.on_connect(mock.participant("someone-else", is_agent=False))
    t0 = mock.clock.now
    adapter = _adapter(cfg, mock)
    with pytest.raises(Exception) as info:
        await adapter.start(cfg.scenarios[0])
    assert str(info.value) == JOIN_TIMEOUT.format(n=4)
    assert mock.clock.now >= t0 + 4 - 1e-9
    await adapter.close()


@pytest.mark.asyncio
async def test_join_timeout_is_a_scenario_error_in_a_run(impulse_project, llm_router, mock_livekit_room):
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    from impulse_evals.runner import run

    mock = mock_livekit_room
    cfg = _cfg(impulse_project, handshake_timeout_s=2)
    result = await run(
        cfg, adapter_factory=lambda config: LiveKitAdapter(config, room_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == JOIN_TIMEOUT.format(n=2)
    assert scenario_result.passed is False
    assert result.agent_entrypoint == LK_URL


@pytest.mark.asyncio
async def test_agent_identity_selects_the_agent_participant(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, agent_identity="neo-agent")
    room = mock.room
    room.on_connect(mock.participant("helper-agent", is_agent=True))
    room.on_connect(mock.participant("neo-agent", is_agent=True), after=0.5)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    steps = [
        (0.1, mock.state("speaking", identity="neo-agent")),
        (0.3, mock.agent_text("Helper text that must be ignored. ", sender="helper-agent")),
        (0.5, mock.agent_text("Neo here.", sender="neo-agent")),
        (0.6, mock.state("listening", identity="neo-agent")),
    ]
    _on_chat(mock, room, steps)
    reply = await adapter.send("Hello", None)
    assert reply.text == "Neo here."
    await adapter.close()


# =============================================================================================
# Text turns and end of turn
# =============================================================================================


@pytest.mark.asyncio
async def test_text_turn_on_lk_chat_and_reply_from_final_agent_transcriptions(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    room = mock.room
    steps = _reply_steps(mock, "Hello! How can I help?") + [
        (0.75, mock.agent_text("Someone else talking.", sender="other-user")),
        (0.76, mock.agent_text("Agent chat message, not a transcription.", topic="lk.chat")),
    ]
    _on_chat(mock, room, steps)
    reply = await adapter.send("Hello", None)
    assert [(entry["text"], entry["topic"]) for entry in room.sent_texts] == [("Hello", "lk.chat")]
    assert reply.text == "Hello! How can I help?"
    assert reply.latency_ms >= 0.0
    assert room.published_audio == []
    await adapter.close()


@pytest.mark.asyncio
async def test_reply_concatenates_final_segments(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    _on_chat(mock, mock.room, _reply_steps(mock, "Sure. ", "You are booked for Tuesday."))
    reply = await adapter.send("Book me", None)
    assert _norm(reply.text) == "Sure. You are booked for Tuesday."
    await adapter.close()


@pytest.mark.asyncio
async def test_turn_ends_after_agent_stops_speaking_plus_grace(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, end_of_turn_grace_ms=500)
    adapter = await _started(cfg, mock)
    fired = []
    steps = _reply_steps(mock, "Done.")  # listening at +0.1 + 0.6 + 0.1 + 0.1 = +0.9
    _on_chat(mock, mock.room, steps, fired)
    reply = await adapter.send("Hi", None)
    assert reply.text == "Done."
    end = fired[0] + 0.9 + 0.5
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


@pytest.mark.asyncio
async def test_pause_shorter_than_grace_stays_one_turn(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)  # grace 800 ms
    adapter = await _started(cfg, mock)
    fired = []
    steps = [
        (0.1, mock.state("thinking")),
        (0.3, mock.state("speaking")),
        (1.0, mock.agent_text("Let me check. ")),
        (1.05, mock.state("listening")),
        (1.55, mock.state("speaking")),  # 0.5 s later: within the grace window
        (2.2, mock.agent_text("You are booked.")),
        (2.25, mock.state("listening")),
    ]
    _on_chat(mock, mock.room, steps, fired)
    reply = await adapter.send("Book me", None)
    assert _norm(reply.text) == "Let me check. You are booked."
    end = fired[0] + 2.25 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


@pytest.mark.asyncio
async def test_pause_longer_than_grace_ends_the_turn(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    fired = []
    steps = [
        (0.1, mock.state("thinking")),
        (0.3, mock.state("speaking")),
        (1.0, mock.agent_text("Let me check.")),
        (1.05, mock.state("listening")),
        (2.5, mock.state("speaking")),  # 1.45 s later: after the grace window
        (3.0, mock.agent_text("Something later.")),
        (3.05, mock.state("listening")),
    ]
    _on_chat(mock, mock.room, steps, fired)
    reply = await adapter.send("Book me", None)
    assert reply.text == "Let me check."
    end = fired[0] + 1.05 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


# =============================================================================================
# heard_text (voice mode): final lk.transcription streams sent with Impulse's identity
# =============================================================================================


async def _voice_turn(project, mock, steps, *, at_start=(), seconds=1.0, **cfg_kwargs):
    """Start a voice-mode adapter and send ``seconds`` of user speech. ``at_start`` items are timed
    from the moment ``send`` is called (during the paced publish); ``steps`` after the publish."""
    cfg = _cfg(project, voice=True, **cfg_kwargs)
    adapter = await _started(cfg, mock)
    room = mock.room
    for offset, item in at_start:
        room.emit(item, after=offset)
    room.on_audio(_plan(room, steps), match=_after_speech(room, seconds))
    reply = await adapter.send("Book a slot for Tuesday", _user_wav(seconds))
    return adapter, reply, room


def _after_speech(room, seconds):
    """Audio-rule predicate: true once ``seconds`` of user audio (at any sample rate) were published."""

    def predicate(record):
        published = sum(len(entry["pcm"]) / 2.0 / entry["sample_rate"] for entry in room.published_audio)
        return published >= seconds * 0.98

    return predicate


def _voice_reply_steps(mock, text="Your slot is booked.", audio=None):
    audio = audio if audio is not None else mock.tone(0.4)
    return [
        (0.05, mock.state("thinking")),
        (0.1, mock.state("speaking")),
        (0.2, mock.audio(audio)),
        (0.3, mock.agent_text(text)),
        (0.7, mock.state("listening")),
    ]


@pytest.mark.asyncio
async def test_voice_turn_publishes_audio_and_collects_agent_audio(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    first, second = mock.tone(0.3), mock.tone(0.2, frequency_hz=550.0)
    steps = [
        (0.05, mock.state("thinking")),
        (0.1, mock.state("speaking")),
        (0.2, mock.audio(first)),
        (0.5, mock.audio(second)),
        (0.6, mock.agent_text("Your slot is booked.")),
        (0.8, mock.state("listening")),
    ]
    adapter, reply, room = await _voice_turn(impulse_project, mock, steps)
    assert room.published_audio, "user speech must be published as audio"
    assert room.sent_texts == []  # voice turns publish audio only
    assert reply.text == "Your slot is booked."
    channels, width, rate, frames = read_wav(reply.audio_bytes)
    assert (channels, width, rate) == (1, 2, 24000)
    assert frames == first + second
    await adapter.close()


@pytest.mark.asyncio
async def test_heard_text_from_final_transcriptions_with_impulse_identity(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    steps = [
        (0.0, mock.user_transcription("book a", final=False)),
        (0.02, mock.user_transcription("book a slot for Tuesday", final=True)),
    ] + _voice_reply_steps(mock)
    adapter, reply, _ = await _voice_turn(impulse_project, mock, steps)
    assert reply.heard_text == "book a slot for Tuesday"
    assert reply.text == "Your slot is booked."  # the agent's own stream is the reply, not heard_text
    await adapter.close()


@pytest.mark.asyncio
async def test_heard_text_none_without_final_user_transcription(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    steps = [(0.0, mock.user_transcription("book a", final=False))] + _voice_reply_steps(mock)
    adapter, reply, _ = await _voice_turn(impulse_project, mock, steps)
    assert reply.heard_text is None
    await adapter.close()


# =============================================================================================
# System prompt via GetAgentInfo (lk.agent.session)
# =============================================================================================


@pytest.mark.asyncio
async def test_no_agent_info_request_when_config_has_a_system_prompt(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    assert mock.room.agent_info_requests == []
    assert adapter.get_system_prompt() in (None, impulse_project.system_prompt)
    await adapter.close()


@pytest.mark.asyncio
async def test_agent_info_instructions_are_the_system_prompt(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, system_prompt=None)
    mock.room.on_agent_info(mock.agent_info("You are Neo, a recruiter.", tools=["schedule_interview"]), after=1.0)
    adapter = await _started(cfg, mock)
    requests = mock.room.agent_info_requests
    assert len(requests) == 1
    assert requests[0]["agent_identity"] == AGENT
    assert adapter.get_system_prompt() == "You are Neo, a recruiter."
    await adapter.close()


@pytest.mark.asyncio
async def test_unanswered_agent_info_gives_none_after_3_s(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project, system_prompt=None)
    t0 = mock.clock.now
    adapter = await _started(cfg, mock)
    elapsed = mock.clock.now - t0
    assert 3.0 - 1e-9 <= elapsed <= 3.5
    assert adapter.get_system_prompt() is None
    await adapter.close()


# =============================================================================================
# Tool calls from function_tools_executed
# =============================================================================================


@pytest.mark.asyncio
async def test_tool_calls_from_function_tools_executed(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    event = mock.tools_executed(
        [
            {"name": "book_slot", "arguments": json.dumps(BOOK_ARGS), "call_id": "c1"},
            {"name": "cancel_slot", "arguments": "not json", "call_id": "c2"},
            {"name": "lookup", "arguments": "[1, 2]", "call_id": "c3"},  # valid JSON, not an object
            {"name": "notify", "arguments": '{"channel": "sms"}', "call_id": "c4"},
            {"name": "pay", "arguments": '{"amount": 100}', "call_id": "c5"},
            {"name": "hold_slot", "arguments": "{}", "call_id": "c6"},
            {"name": "log_call", "arguments": '{"level": "info"}', "call_id": "c7"},  # no output
        ],
        [  # paired by call_id, not by position
            {"call_id": "c4", "name": "notify", "output": "queued", "is_error": False},
            {"call_id": "c1", "name": "book_slot", "output": '{"success": true, "slot": "10:00"}', "is_error": False},
            {"call_id": "c2", "name": "cancel_slot", "output": "slot not found", "is_error": True},
            {"call_id": "c3", "name": "lookup", "output": '{"items": []}', "is_error": False},
            {"call_id": "c5", "name": "pay", "output": '{"success": false}', "is_error": False},
            {"call_id": "c6", "name": "hold_slot", "output": '"held"', "is_error": False},
        ],
    )
    steps = [
        (0.1, mock.state("thinking")),
        (0.4, event),
        (0.6, mock.state("speaking")),
        (0.8, mock.agent_text("Done.")),
        (0.9, mock.state("listening")),
    ]
    _on_chat(mock, mock.room, steps)
    reply = await adapter.send("Do everything", None)
    calls = reply.tool_calls
    assert [call.name for call in calls] == [
        "book_slot", "cancel_slot", "lookup", "notify", "pay", "hold_slot", "log_call",
    ]
    by_name = {call.name: call for call in calls}
    assert by_name["book_slot"].arguments == BOOK_ARGS
    assert by_name["book_slot"].result == {"success": True, "slot": "10:00"}
    assert (by_name["book_slot"].success, by_name["book_slot"].error) == (True, None)
    assert by_name["cancel_slot"].arguments == {"raw": "not json"}
    assert by_name["cancel_slot"].result == {"value": "slot not found"}
    assert (by_name["cancel_slot"].success, by_name["cancel_slot"].error) == (False, "slot not found")
    assert by_name["lookup"].arguments == {"raw": "[1, 2]"}
    assert by_name["lookup"].result == {"items": []}
    assert by_name["lookup"].success is None
    assert by_name["notify"].arguments == {"channel": "sms"}
    assert by_name["notify"].result == {"value": "queued"}
    assert by_name["notify"].success is None
    assert by_name["pay"].result == {"success": False}
    assert (by_name["pay"].success, by_name["pay"].error) == (False, None)
    assert by_name["hold_slot"].arguments == {}
    assert by_name["hold_slot"].result == {"value": '"held"'}
    assert by_name["hold_slot"].success is None
    assert (by_name["log_call"].result, by_name["log_call"].success) == (None, None)
    assert reply.text == "Done."
    await adapter.close()


@pytest.mark.asyncio
async def test_no_tool_event_means_no_tool_calls(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    _on_chat(mock, mock.room, _reply_steps(mock, "You are booked."))
    reply = await adapter.send("Book me", None)
    assert reply.tool_calls == []
    await adapter.close()


# =============================================================================================
# Barge-in: lk.agent.state leaves "speaking" while user speech is being sent
# =============================================================================================


@pytest.mark.asyncio
async def test_barge_in_sets_interrupted(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    during_speech = [
        (0.2, mock.state("speaking")),  # the agent starts talking over the user...
        (0.5, mock.state("listening")),  # ...and stops while Impulse is still sending speech (1.0 s)
    ]
    steps = [
        (0.05, mock.state("thinking")),
        (0.1, mock.state("speaking")),  # +0.1 after the 1.0 s publish: within the grace window
        (0.2, mock.audio(mock.tone(0.4))),
        (0.3, mock.agent_text("Sorry, go ahead.")),
        (0.6, mock.state("listening")),
    ]
    adapter, reply, _ = await _voice_turn(impulse_project, mock, steps, at_start=during_speech)
    assert reply.interrupted is True
    assert reply.text == "Sorry, go ahead."
    await adapter.close()


@pytest.mark.asyncio
async def test_no_barge_in_when_the_agent_speaks_after_user_speech(impulse_project, mock_livekit_room):
    mock = mock_livekit_room
    adapter, reply, _ = await _voice_turn(impulse_project, mock, _voice_reply_steps(mock))
    assert reply.interrupted is False
    await adapter.close()


# =============================================================================================
# RoomClosed -> AgentClosed (a normal end of the conversation)
# =============================================================================================


@pytest.mark.asyncio
async def test_room_closed_mid_conversation_raises_agent_closed(impulse_project, mock_livekit_room):
    from impulse_evals.adapters.base import AgentClosed

    mock = mock_livekit_room
    cfg = _cfg(impulse_project)
    adapter = await _started(cfg, mock)
    steps = [
        (0.1, mock.state("speaking")),
        (0.3, mock.agent_text("Goodbye.")),
        (0.4, mock.closed("agent left the room")),
    ]
    _on_chat(mock, mock.room, steps)
    with pytest.raises(AgentClosed) as info:
        await adapter.send("Bye", None)
    if info.value.reply is not None:
        assert info.value.reply.text == "Goodbye."
    await adapter.close()  # must tolerate the closed room


@pytest.mark.asyncio
async def test_run_with_livekit_adapter_ends_normally_when_the_room_closes(
    impulse_project, llm_router, mock_livekit_room
):
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    from impulse_evals.runner import run

    mock = mock_livekit_room
    project = impulse_project
    name = "lk_run"
    first_line = project.opening(name)
    second_line = "Is Tuesday free?"
    llm_router.script(project.persona(name), second_line, "Thanks!")
    cfg = _cfg(project, scenarios=[project.scenario(name, max_turns=3, metrics=["UNIV-013"])])
    room = _join(mock)
    _on_chat(mock, room, _reply_steps(mock, "Hello, how can I help?"), text=first_line)
    _on_chat(mock, room, [(0.1, mock.state("speaking")), (0.3, mock.closed())], text=second_line)
    result = await run(
        cfg, adapter_factory=lambda config: LiveKitAdapter(config, room_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    texts = [(turn.speaker, turn.text) for turn in scenario_result.transcript.turns]
    assert texts[:3] == [("user", first_line), ("agent", "Hello, how can I help?"), ("user", second_line)]
    # The room closed during the second turn: no third user turn ("Thanks!" is never sent).
    assert [speaker for speaker, _ in texts].count("user") == 2
    assert [entry["text"] for entry in room.sent_texts] == [first_line, second_line]
    assert result.agent_entrypoint == LK_URL


# =============================================================================================
# Optional extra
# =============================================================================================


@pytest.mark.asyncio
async def test_missing_extra_gives_exact_error(impulse_project, mock_livekit_room, extra_imports, monkeypatch):
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    monkeypatch.setenv("LIVEKIT_API_KEY", "lk-test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "lk-test-SECRET-0001")
    extra_imports.missing("livekit")
    cfg = _cfg(impulse_project)
    with pytest.raises(Exception) as info:
        adapter = LiveKitAdapter(cfg, clock=mock_livekit_room.clock)  # no room_factory: the real room
        await adapter.start(cfg.scenarios[0])
    assert MISSING_EXTRA in str(info.value)


def test_adapter_module_imports_without_livekit():
    """``impulse_evals.adapters.livekit_adapter`` (and its Room types) import in a fresh interpreter in
    which ``livekit`` cannot be imported (the extra is lazy-imported)."""
    code = (
        "import sys\n"
        "sys.modules['livekit'] = None\n"
        "from impulse_evals.adapters.livekit_adapter import (\n"
        "    LiveKitAdapter, Room, RoomParticipant, RoomText, RoomAttributes, RoomSessionEvent, RoomAudio,\n"
        "    RoomClosed)\n"
        "RoomParticipant(identity='a', is_agent=True, attributes={})\n"
        "RoomText(topic='lk.chat', sender_identity='a', text='hi', final=True, arrival_s=0.0)\n"
        "RoomAttributes(identity='a', changed={}, arrival_s=0.0)\n"
        "RoomSessionEvent(kind='agent_info', data={}, arrival_s=0.0)\n"
        "RoomAudio(pcm=b'', sample_rate=24000, arrival_s=0.0); RoomClosed(reason='x')\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")


# =============================================================================================
# Audit batch 1 (AUDIT.md findings 1 and 4). F1: the room closing before the agent's first turn is a
# scenario error at the runner level (the adapter's AgentClosed text is embedded). F4: userinfo in
# ``livekit.url`` never reaches ``RunResult.agent_entrypoint`` while the room still gets the full URL.
# =============================================================================================

CLOSED_BEFORE_FIRST_TURN = "the agent closed the conversation before its first turn: "
LK_USERINFO_PASSWORD = "LKPASS789"
LK_USERINFO_URL = f"wss://u:{LK_USERINFO_PASSWORD}@127.0.0.1:7880"


@pytest.mark.asyncio
async def test_run_with_room_closed_before_first_agent_turn_is_a_scenario_error(
    impulse_project, llm_router, mock_livekit_room
):
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    from impulse_evals.runner import run

    mock = mock_livekit_room
    project = impulse_project
    name = "lk_dead_agent"
    cfg = _cfg(project, scenarios=[project.scenario(name, metrics=["UNIV-001"])])
    room = _join(mock)
    _on_chat(mock, room, [(0.1, mock.closed("agent process crashed"))])   # no agent text, audio, or tool call
    result = await run(
        cfg, adapter_factory=lambda config: LiveKitAdapter(config, room_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == CLOSED_BEFORE_FIRST_TURN + "the LiveKit room closed: agent process crashed"
    assert [turn.speaker for turn in scenario_result.transcript.turns] == ["user"]
    assert scenario_result.passed is False
    assert scenario_result.metric_results == []
    assert scenario_result.assertion_results == []
    assert llm_router.calls("judge") == []
    assert [entry["text"] for entry in room.sent_texts] == [project.opening(name)]
    assert room.close_calls >= 1


@pytest.mark.asyncio
async def test_livekit_url_userinfo_is_stripped_from_agent_entrypoint(impulse_project, llm_router, mock_livekit_room):
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    from impulse_evals.runner import run

    mock = mock_livekit_room
    project = impulse_project
    name = "lk_userinfo"
    cfg = _cfg(project, scenarios=[project.scenario(name, max_turns=1, metrics=["UNIV-013"])], url=LK_USERINFO_URL)
    room = _join(mock)
    _on_chat(mock, room, _reply_steps(mock, "Hello, how can I help?"))
    result = await run(
        cfg, adapter_factory=lambda config: LiveKitAdapter(config, room_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    assert room.connect_call["url"] == LK_USERINFO_URL   # the room still connects with the credentials
    assert result.agent_entrypoint == "wss://127.0.0.1:7880"
    assert LK_USERINFO_PASSWORD not in result.model_dump_json()


# =============================================================================================
# Audit batch 1 review fix-ups (AUDIT.md finding 4, follow-up).
# - URL user names: the password of ``user:password@`` is the credential; the user name is
#   registered for redaction only when the password is empty (``wss://TOKEN@host``), so a common word
#   such as "user" in a URL never turns run texts like the scenario name ``user_signup`` into
#   ``[REDACTED]_signup``.
# - The real ``LiveKitRoom`` connect error shows ``livekit.url`` without its credentials (userinfo and
#   credential query parameters stripped, docs/privacy-and-security.md), not a redacted raw URL. The
#   real room runs here against minimal fake ``livekit`` SDK modules (installed through
#   ``extra_imports``) whose ``Room.connect`` fails; no server, no network.
# =============================================================================================

STRIPPED_LK_URL = "wss://127.0.0.1:7880"
CONSENT = "dpdp.consent_before_pii"
NO_VERDICT_WARNING_PREFIX = "Critical checks not evaluated (judge returned no verdict): "
WORD_USER_PASSWORD = "SECRETPW99"
WORD_USER_URL = f"wss://user:{WORD_USER_PASSWORD}@127.0.0.1:7880"
TOKEN_ONLY_USER = "TOKENONLY12345"
TOKEN_ONLY_URL = f"wss://{TOKEN_ONLY_USER}@127.0.0.1:7880"
SHORT_USER = "lkq"   # shorter than the 4-character redaction minimum
CONNECT_PASSWORD = "LKCONNPASS789"
CONNECT_QUERY_TOKEN = "LKQUERYTOKEN4321"
CONNECT_URL = f"wss://{SHORT_USER}:{CONNECT_PASSWORD}@127.0.0.1:7880?access_token={CONNECT_QUERY_TOKEN}"
SDK_CONNECT_ERROR = "connection refused by the fake LiveKit SDK"


@pytest.mark.asyncio
async def test_url_user_name_with_a_password_is_never_redacted_from_run_texts(
    impulse_project, llm_router, mock_livekit_room
):
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    from impulse_evals.runner import run

    mock = mock_livekit_room
    project = impulse_project
    name = "user_signup"
    cfg = _cfg(
        project,
        scenarios=[project.scenario(name, max_turns=1, assertions=[CONSENT], metrics=["UNIV-013"])],
        url=WORD_USER_URL,
    )
    room = _join(mock)
    _on_chat(mock, room, _reply_steps(mock, "Hello, how can I help?"))
    llm_router.on_judge(None, criterion=CONSENT)   # no verdict: a run warning that names the scenario
    result = await run(
        cfg, adapter_factory=lambda config: LiveKitAdapter(config, room_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == f"critical check {CONSENT} could not be evaluated: judge returned no verdict"
    assert result.warnings.count(NO_VERDICT_WARNING_PREFIX + f"{name}:{CONSENT}") == 1, result.warnings
    assert result.agent_entrypoint == STRIPPED_LK_URL
    assert room.connect_call["url"] == WORD_USER_URL   # the room still connects with the credentials
    dumped = result.model_dump_json()
    assert WORD_USER_PASSWORD not in dumped
    assert "[REDACTED]_signup" not in dumped


@pytest.mark.asyncio
async def test_url_user_name_without_a_password_is_still_redacted(impulse_project, llm_router, mock_livekit_room):
    # Regression guard: in wss://TOKEN@host the user name IS the credential.
    from impulse_evals.adapters.livekit_adapter import LiveKitAdapter

    from impulse_evals.runner import run

    mock = mock_livekit_room
    project = impulse_project
    cfg = _cfg(project, scenarios=[project.scenario("lk_token_user", metrics=["UNIV-013"])], url=TOKEN_ONLY_URL)
    mock.room.connect_error = RuntimeError(f"the server rejected {TOKEN_ONLY_URL} (401)")   # echoes the raw URL
    result = await run(
        cfg, adapter_factory=lambda config: LiveKitAdapter(config, room_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is not None
    assert TOKEN_ONLY_USER not in scenario_result.error, scenario_result.error
    assert "[REDACTED]" in scenario_result.error
    assert result.agent_entrypoint == STRIPPED_LK_URL
    assert TOKEN_ONLY_USER not in result.model_dump_json()
    for warning in result.warnings:
        assert TOKEN_ONLY_USER not in warning, warning
    assert mock.room.connect_call["url"] == TOKEN_ONLY_URL


def _fake_livekit_sdk(extra_imports, connect_urls):
    """Minimal ``livekit.rtc`` / ``livekit.api`` / ``livekit.protocol.agent_pb.agent_session`` modules for
    the real room: the access token builds, ``Room.connect`` records the URL and raises."""
    import types

    class AccessToken:
        def __init__(self, api_key=None, api_secret=None):
            pass

        def with_identity(self, identity):
            return self

        def with_name(self, name):
            return self

        def with_grants(self, grants):
            return self

        def with_room_config(self, config):
            return self

        def to_jwt(self):
            return "fake-access-token-for-tests"

    class FakeRtcRoom:
        def __init__(self, *args, **kwargs):
            self.remote_participants = {}

        def on(self, event, callback=None):
            return callback

        def register_text_stream_handler(self, topic, handler):
            pass

        def register_byte_stream_handler(self, topic, handler):
            pass

        async def connect(self, url, token, options=None):
            connect_urls.append(url)
            raise RuntimeError(SDK_CONNECT_ERROR)

        async def disconnect(self):
            pass

    def package(name):
        module = types.ModuleType(name)
        module.__path__ = []
        return module

    livekit = extra_imports.module("livekit", package("livekit"))
    rtc = extra_imports.module("livekit.rtc", types.ModuleType("livekit.rtc"))
    rtc.Room = FakeRtcRoom
    rtc.RoomOptions = lambda **kwargs: dict(kwargs)
    api = extra_imports.module("livekit.api", types.ModuleType("livekit.api"))
    api.AccessToken = AccessToken
    api.VideoGrants = lambda **kwargs: dict(kwargs)
    api.RoomConfiguration = lambda **kwargs: dict(kwargs)
    api.RoomAgentDispatch = lambda **kwargs: dict(kwargs)
    protocol = extra_imports.module("livekit.protocol", package("livekit.protocol"))
    agent_pb = extra_imports.module("livekit.protocol.agent_pb", package("livekit.protocol.agent_pb"))
    agent_session = extra_imports.module(
        "livekit.protocol.agent_pb.agent_session", types.ModuleType("livekit.protocol.agent_pb.agent_session")
    )
    agent_session.AgentSessionMessage = type("AgentSessionMessage", (), {"ParseFromString": lambda self, data: None})
    livekit.rtc, livekit.api, livekit.protocol = rtc, api, protocol
    protocol.agent_pb = agent_pb
    agent_pb.agent_session = agent_session


@pytest.mark.asyncio
async def test_real_room_connect_error_shows_the_livekit_url_without_credentials(
    impulse_project, llm_router, extra_imports, monkeypatch
):
    from impulse_evals.runner import run

    monkeypatch.setenv("LIVEKIT_API_KEY", "lk-test-key-0001")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "lk-test-SECRET-0001")
    connect_urls = []
    _fake_livekit_sdk(extra_imports, connect_urls)
    project = impulse_project
    cfg = _cfg(project, scenarios=[project.scenario("lk_connect_fails", metrics=["UNIV-013"])], url=CONNECT_URL)
    result = await run(cfg)   # the default adapter factory: LiveKitAdapter with the real room
    scenario_result = result.scenario_results[0]
    error = scenario_result.error
    assert error is not None
    assert STRIPPED_LK_URL in error, error
    assert SDK_CONNECT_ERROR in error, error
    for secret in (SHORT_USER, CONNECT_PASSWORD, CONNECT_QUERY_TOKEN, "access_token", "[REDACTED]"):
        assert secret not in error, error
    assert scenario_result.passed is False
    assert llm_router.calls("judge") == []
    assert connect_urls == [CONNECT_URL]   # the SDK still receives the full URL
