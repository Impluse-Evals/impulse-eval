# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 11: the Pipecat adapter (SPEC.md section 9, as updated by the pre-Stage-B source read of
pipecat-ai 1.12.0; DECISIONS.md "Source read (KICKOFF §2b)" and decisions D1-D10).

``PipecatAdapter(config, *, channel_factory=None, clock=None)`` takes the ``ProjectConfig``; every
test injects ``mock_pipecat_channel`` (a scripted fake ``Channel`` plus a virtual ``Clock``), so no
``pipecat-ai`` and no network are needed. Both connection modes are covered unless a test says
otherwise:

* ``eval``: Pipecat's client transport sends ``client-ready`` itself (the fake bot answers
  ``bot-ready`` on connect); query flags ``skip_tts`` (text mode) / ``capture_bot_audio`` (voice
  mode), never ``trigger_disconnect``; after ``bot-ready`` the adapter sends ``eval-configure``
  (report level FULL) then ``eval-context`` (config system prompt) or returns the reset warning.
  The fake channel paces user audio like Pipecat's eval client (``paced_audio = True``).
* ``websocket``: the adapter sends ``client-ready`` (version 2.1.0, library impulse-evals) and
  paces user audio itself (``chunk_ms``, ``realtime_pacing``, ``trailing_silence_ms``).

Times are virtual seconds on the fake clock; "+x" in comments means x seconds after the trigger.
"""

import subprocess
import sys

import pytest
from conftest import REPO_ROOT, pcm_tone, read_wav, wav_bytes

pytestmark = pytest.mark.phase11

URL = "ws://localhost:7860"
MODES = ["eval", "websocket"]
BOT_READY = {"version": "2.1.0", "about": {"library": "pipecat-ai", "library_version": "1.12.0"}}
DEFAULT = object()

HANDSHAKE_TIMEOUT = (
    "Pipecat bot did not send bot-ready within {n} s. Check that the bot is running at {url} "
    "(eval mode: started with -t eval) and RTVI is enabled."
)
TURN_TIMEOUT = "Pipecat bot did not finish its turn within {n} s."
ONE_CLIENT = "Pipecat bot accepts one client at a time. Run with --concurrency 1, or start one bot per connection."
MISSING_EXTRA = "pipecat adapter needs the 'pipecat' extra: pip install impulse-evals[pipecat]"
EVALS_MISSING = (
    "Pipecat eval mode needs a pipecat-ai release that includes pipecat.evals. "
    "Upgrade pipecat-ai, or set pipecat.mode: websocket."
)
RESET_WARNING = (
    "Pipecat eval bots keep their conversation between scenarios. Set agent.system_prompt in config.yaml "
    "so Impulse can reset it, or reset the LLM context in the bot's on_client_connected handler."
)
FUNCTION_CALL_WARNING = (
    "Pipecat bot reports function calls without names or arguments. Set the RTVI function call report "
    "level to FULL so UNIV-023 can check tool arguments."
)
CAP_MESSAGE = "Pipecat bots serve one client at a time; running scenarios one by one."
DAILY_MESSAGE = "pipecat.mode: daily is planned for v2. Use eval or websocket."

VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}
BOOK_ARGS = {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _cfg(project, mode="eval", *, voice=False, system_prompt=DEFAULT, scenarios=None, timeout_s=None, **pipecat):
    block = {"mode": mode, "url": URL}
    block.update(pipecat)
    kwargs = {}
    if voice:
        kwargs["voice_config"] = dict(VOICE_CONFIG, **({"timeout_s": timeout_s} if timeout_s else {}))
    if system_prompt is not DEFAULT:
        kwargs["system_prompt"] = system_prompt
    project.write_config(
        scenarios or [project.scenario("pipecat_probe")],
        agent={"adapter": "pipecat", "pipecat": block},
        **kwargs,
    )
    return project.load()


def _adapter(cfg, mock, **kwargs):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    return PipecatAdapter(cfg, channel_factory=mock.factory, clock=mock.clock, **kwargs)


def _handshake(mock, mode, channel=None, after=0.0):
    """The fake bot's ``bot-ready``: on connect in eval mode (Pipecat's client transport sent
    ``client-ready``), in answer to the adapter's ``client-ready`` in websocket mode."""
    channel = channel or mock.channel
    if mode == "eval":
        channel.paced_audio = True  # Pipecat's eval client transport paces user audio itself
        channel.on_connect(mock.rtvi("bot-ready", BOT_READY, id="client-ready"), after=after)
    else:
        channel.on_send(
            "client-ready", lambda message: mock.rtvi("bot-ready", BOT_READY, id=message.get("id")), after=after
        )
    return channel


async def _started(cfg, mock, mode, after=0.0):
    _handshake(mock, mode, after=after)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    return adapter


def _plan(channel, steps, fired=None):
    """A rule item: when triggered, schedule each ``(offset_s, item)`` relative to the trigger time."""

    def trigger(_record):
        if fired is not None:
            fired.append(channel.clock.monotonic())
        for offset, item in steps:
            channel.emit(item, after=offset)

    return trigger


def _after_speech(channel, samples):
    """Audio-rule predicate: true once the adapter has sent ``samples`` samples of user audio."""

    def predicate(record):
        total = sum(len(entry["pcm"]) // 2 for entry in channel.sent_audio)
        return total >= samples * 0.98

    return predicate


def _llm_steps(mock, *tokens, start=0.2, gap=0.1):
    """``bot-llm-started`` at +start, one ``bot-llm-text`` per token every ``gap``, then ``bot-llm-stopped``."""
    steps = [(start, mock.rtvi("bot-llm-started"))]
    for index, token in enumerate(tokens, start=1):
        steps.append((start + index * gap, mock.rtvi("bot-llm-text", {"text": token})))
    steps.append((start + (len(tokens) + 1) * gap, mock.rtvi("bot-llm-stopped")))
    return steps


def _on_text_turn(mock, channel, steps, fired=None):
    channel.on_send("send-text", _plan(channel, steps, fired))


def _user_wav(seconds=0.5, sample_rate=16000):
    return wav_bytes(pcm_tone(seconds, sample_rate))


def _norm(text):
    return " ".join((text or "").split())


def _flag_on(query, key):
    value = (query or {}).get(key)
    return value is True or str(value).strip().lower() in ("1", "true", "yes")


def _client_messages(channel, t=None):
    """``data`` of every ``client-message`` the adapter sent (optionally only those with ``data.t == t``)."""
    found = []
    for message in channel.sent_messages:
        if message.get("type") != "client-message":
            continue
        data = message.get("data") or {}
        if t is None or data.get("t") == t:
            found.append(data)
    return found


def _bot_output(text, **fields):
    data = {"text": text, "aggregated_by": "sentence"}
    data.update(fields)
    return data


def _voice_steps(mock, reply="Hello there.", start=0.1, audio_seconds=0.3):
    rtvi = mock.rtvi
    return [
        (start, rtvi("bot-started-speaking")),
        (start + 0.1, mock.BotAudio(mock.tone(audio_seconds))),
        (start + 0.15, rtvi("bot-output", _bot_output(reply, spoken=True, spoken_status="completed"))),
        (start + audio_seconds + 0.2, rtvi("bot-stopped-speaking")),
    ]


async def _voice_turn(project, mock, mode, steps, *, cfg=None, seconds=0.5, fired=None, **cfg_kwargs):
    """Start a voice-mode adapter, send ``seconds`` of user speech, play ``steps`` after the speech."""
    cfg = cfg or _cfg(project, mode, voice=True, **cfg_kwargs)
    rate = cfg.agent.pipecat.sample_rate
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    channel.on_audio(_plan(channel, steps, fired), match=_after_speech(channel, int(seconds * rate)))
    reply = await adapter.send("Book a slot for Tuesday", _user_wav(seconds))
    return adapter, reply, channel


# =============================================================================================
# Connection: URL, query flags, headers
# =============================================================================================


@pytest.mark.asyncio
async def test_eval_text_mode_connects_with_skip_tts_only(impulse_project, mock_pipecat_channel):
    cfg = _cfg(impulse_project, "eval")
    adapter = await _started(cfg, mock_pipecat_channel, "eval")
    channel = mock_pipecat_channel.channel
    assert len(channel.connect_calls) == 1
    assert channel.connect_url == URL
    assert _flag_on(channel.connect_query, "skip_tts")
    assert not _flag_on(channel.connect_query, "capture_bot_audio")
    assert not _flag_on(channel.connect_query, "trigger_disconnect")
    await adapter.close()


@pytest.mark.asyncio
async def test_eval_voice_mode_connects_with_capture_bot_audio_only(impulse_project, mock_pipecat_channel):
    cfg = _cfg(impulse_project, "eval", voice=True)
    adapter = await _started(cfg, mock_pipecat_channel, "eval")
    query = mock_pipecat_channel.channel.connect_query
    assert _flag_on(query, "capture_bot_audio")
    assert not _flag_on(query, "skip_tts")
    assert not _flag_on(query, "trigger_disconnect")
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", [False, True])
async def test_websocket_mode_sets_no_eval_query_flags(impulse_project, mock_pipecat_channel, voice):
    cfg = _cfg(impulse_project, "websocket", voice=voice)
    adapter = await _started(cfg, mock_pipecat_channel, "websocket")
    query = mock_pipecat_channel.channel.connect_query
    for flag in ("skip_tts", "capture_bot_audio", "trigger_disconnect"):
        assert not _flag_on(query, flag)
    assert mock_pipecat_channel.channel.connect_url == URL
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_connect_headers_carry_configured_headers_and_bearer_key(
    impulse_project, mock_pipecat_channel, monkeypatch, mode
):
    monkeypatch.setenv("IMPULSE_TEST_PIPECAT_TOKEN", "pc-token-SECRET-77")
    cfg = _cfg(
        impulse_project, mode, headers={"Origin": "http://localhost:3000"}, api_key_env="IMPULSE_TEST_PIPECAT_TOKEN"
    )
    adapter = await _started(cfg, mock_pipecat_channel, mode)
    headers = {key.lower(): value for key, value in mock_pipecat_channel.channel.connect_headers.items()}
    assert headers["origin"] == "http://localhost:3000"
    assert headers["authorization"] == "Bearer pc-token-SECRET-77"
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_one_connection_per_scenario(impulse_project, mock_pipecat_channel, mode):
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock_pipecat_channel, mode)
    channel = mock_pipecat_channel.channel
    assert len(channel.connect_calls) == 1
    await adapter.close()
    assert channel.close_calls >= 1


# =============================================================================================
# Handshake
# =============================================================================================


@pytest.mark.asyncio
async def test_websocket_mode_adapter_sends_client_ready_first(impulse_project, mock_pipecat_channel):
    cfg = _cfg(impulse_project, "websocket")
    adapter = await _started(cfg, mock_pipecat_channel, "websocket")
    first = mock_pipecat_channel.channel.sent_messages[0]
    assert first["label"] == "rtvi-ai"
    assert first["type"] == "client-ready"
    assert isinstance(first["id"], str) and first["id"]
    assert first["data"] == {"version": "2.1.0", "about": {"library": "impulse-evals", "library_version": "0.1.0"}}
    await adapter.close()


@pytest.mark.asyncio
async def test_eval_mode_adapter_sends_no_client_ready(impulse_project, mock_pipecat_channel):
    cfg = _cfg(impulse_project, "eval")
    adapter = await _started(cfg, mock_pipecat_channel, "eval")
    assert "client-ready" not in mock_pipecat_channel.channel.sent_types()
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_bot_ready_is_awaited_before_anything_else_is_sent(impulse_project, mock_pipecat_channel, mode):
    cfg = _cfg(impulse_project, mode, system_prompt=None)  # eval-configure and get_system_prompt follow bot-ready
    clock = mock_pipecat_channel.clock
    t0 = clock.now
    adapter = await _started(cfg, mock_pipecat_channel, mode, after=1.5)
    channel = mock_pipecat_channel.channel
    after_handshake = [entry for entry in channel.sent if entry.get("data", {}).get("type") != "client-ready"]
    assert after_handshake, "the adapter sent nothing after bot-ready"
    assert all(entry["t"] >= t0 + 1.5 - 1e-9 for entry in after_handshake)
    assert clock.now >= t0 + 1.5 - 1e-9
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_handshake_timeout_gives_exact_error(impulse_project, mock_pipecat_channel, mode):
    cfg = _cfg(impulse_project, mode, handshake_timeout_s=4)
    clock = mock_pipecat_channel.clock
    t0 = clock.now
    adapter = _adapter(cfg, mock_pipecat_channel)  # the fake bot never sends bot-ready
    with pytest.raises(Exception) as info:
        await adapter.start(cfg.scenarios[0])
    assert str(info.value) == HANDSHAKE_TIMEOUT.format(n=4, url=URL)
    assert clock.now >= t0 + 4 - 1e-9
    await adapter.close()


@pytest.mark.asyncio
async def test_handshake_timeout_is_a_scenario_error_in_a_run(impulse_project, llm_router, mock_pipecat_channel):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    from impulse_evals.runner import run

    cfg = _cfg(impulse_project, "eval", handshake_timeout_s=3)
    mock = mock_pipecat_channel
    result = await run(
        cfg, adapter_factory=lambda config: PipecatAdapter(config, channel_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == HANDSHAKE_TIMEOUT.format(n=3, url=URL)
    assert scenario_result.passed is False
    assert scenario_result.metric_results == []
    assert result.agent_entrypoint == URL


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_close_code_1013_gives_one_client_error(impulse_project, mock_pipecat_channel, mode):
    cfg = _cfg(impulse_project, mode)
    mock = mock_pipecat_channel
    mock.channel.on_connect(mock.BotClose(1013))
    adapter = _adapter(cfg, mock)
    with pytest.raises(Exception) as info:
        await adapter.start(cfg.scenarios[0])
    assert str(info.value) == ONE_CLIENT
    await adapter.close()


# =============================================================================================
# Message ids and labels
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_every_sent_message_has_label_and_unique_string_id(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, system_prompt=None)
    adapter = await _started(cfg, mock, mode)
    for index in range(2):
        _on_text_turn(mock, mock.channel, _llm_steps(mock, f"Answer {index}."))
        await adapter.send(f"Question {index}", None)
    await adapter.close()
    messages = mock.channel.sent_messages
    assert len(messages) >= 3  # get_system_prompt and two send-text messages at least
    ids = []
    for message in messages:
        assert message.get("label") == "rtvi-ai", message
        assert isinstance(message.get("id"), str) and message["id"], message
        ids.append(message["id"])
    assert len(ids) == len(set(ids))


# =============================================================================================
# Eval mode: eval-configure, eval-context, reset warning
# =============================================================================================


@pytest.mark.asyncio
async def test_eval_mode_sends_eval_configure_then_eval_context_after_bot_ready(impulse_project, mock_pipecat_channel):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "eval")
    t0 = mock.clock.now
    adapter = await _started(cfg, mock, "eval", after=0.5)
    _on_text_turn(mock, mock.channel, _llm_steps(mock, "Hello."))
    await adapter.send("Hi", None)
    sent = [message for message in mock.channel.sent_messages]
    client_messages = [m for m in sent if m.get("type") == "client-message"]
    assert [m["data"].get("t") for m in client_messages[:2]] == ["eval-configure", "eval-context"]
    assert client_messages[0]["data"] == {"t": "eval-configure", "d": {"function_call_report_level": {"*": "full"}}}
    assert client_messages[1]["data"] == {
        "t": "eval-context",
        "d": {"messages": [{"role": "system", "content": impulse_project.system_prompt}]},
    }
    types_in_order = [m.get("type") for m in sent]
    assert types_in_order.index("send-text") > sent.index(client_messages[1])
    records = [entry for entry in mock.channel.sent if entry["kind"] == "message"]
    assert all(entry["t"] >= t0 + 0.5 - 1e-9 for entry in records)
    assert _client_messages(mock.channel, "get_system_prompt") == []  # the config prompt wins
    assert RESET_WARNING not in adapter.get_warnings()
    await adapter.close()


@pytest.mark.asyncio
async def test_eval_mode_without_system_prompt_warns_once_and_sends_no_eval_context(
    impulse_project, mock_pipecat_channel
):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "eval", system_prompt=None)
    adapter = await _started(cfg, mock, "eval")
    for index in range(2):
        _on_text_turn(mock, mock.channel, _llm_steps(mock, f"Answer {index}."))
        await adapter.send(f"Question {index}", None)
    await adapter.close()
    assert _client_messages(mock.channel, "eval-context") == []
    assert _client_messages(mock.channel, "eval-configure") == [
        {"t": "eval-configure", "d": {"function_call_report_level": {"*": "full"}}}
    ]
    assert adapter.get_warnings().count(RESET_WARNING) == 1


@pytest.mark.asyncio
async def test_websocket_mode_sends_no_eval_messages_and_no_reset_warning(impulse_project, mock_pipecat_channel):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "websocket", system_prompt=None)
    adapter = await _started(cfg, mock, "websocket")
    _on_text_turn(mock, mock.channel, _llm_steps(mock, "Hello."))
    await adapter.send("Hi", None)
    await adapter.close()
    assert _client_messages(mock.channel, "eval-configure") == []
    assert _client_messages(mock.channel, "eval-context") == []
    assert RESET_WARNING not in adapter.get_warnings()


@pytest.mark.asyncio
async def test_reset_warning_appears_once_per_run(impulse_project, llm_router, mock_pipecat_channel):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    from impulse_evals.runner import run

    mock = mock_pipecat_channel
    project = impulse_project
    scenarios = [
        project.scenario("eval_reset_1", max_turns=1, metrics=["UNIV-013"]),
        project.scenario("eval_reset_2", max_turns=1, metrics=["UNIV-013"]),
    ]
    cfg = _cfg(project, "eval", system_prompt=None, scenarios=scenarios)
    for channel in (mock.channel, mock.add_channel()):
        _handshake(mock, "eval", channel=channel)
        _on_text_turn(mock, channel, _llm_steps(mock, "Hello."))
    result = await run(
        cfg, adapter_factory=lambda config: PipecatAdapter(config, channel_factory=mock.factory, clock=mock.clock)
    )
    assert [s.error for s in result.scenario_results] == [None, None]
    assert result.warnings.count(RESET_WARNING) == 1


# =============================================================================================
# System prompt
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_config_system_prompt_wins_and_no_request_is_sent(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    assert _client_messages(mock.channel, "get_system_prompt") == []
    assert adapter.get_system_prompt() in (None, impulse_project.system_prompt)
    await adapter.close()


def _answer_system_prompt(mock, prompt, answer_id=None):
    def answer(message):
        return mock.rtvi(
            "server-response",
            {"t": "get_system_prompt", "d": {"system_prompt": prompt}},
            id=answer_id if answer_id is not None else message.get("id"),
        )

    return answer


def _is_prompt_request(message):
    return message.get("type") == "client-message" and (message.get("data") or {}).get("t") == "get_system_prompt"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_get_system_prompt_answer_within_3_s_is_used(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, system_prompt=None)
    mock.channel.on_send(_is_prompt_request, _answer_system_prompt(mock, "You are the Sunrise Clinic bot."), after=1.0)
    adapter = await _started(cfg, mock, mode)
    assert len(_client_messages(mock.channel, "get_system_prompt")) == 1
    assert adapter.get_system_prompt() == "You are the Sunrise Clinic bot."
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_unanswered_get_system_prompt_gives_none_after_3_s(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, system_prompt=None)
    t0 = mock.clock.now
    adapter = await _started(cfg, mock, mode)
    elapsed = mock.clock.now - t0
    assert 3.0 - 1e-9 <= elapsed <= 3.5
    assert adapter.get_system_prompt() is None
    await adapter.close()


@pytest.mark.asyncio
async def test_server_response_with_another_id_is_ignored(impulse_project, mock_pipecat_channel):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "websocket", system_prompt=None)
    mock.channel.on_send(_is_prompt_request, _answer_system_prompt(mock, "Wrong prompt", answer_id="not-yours"), after=0.5)
    adapter = await _started(cfg, mock, "websocket")
    assert adapter.get_system_prompt() is None
    await adapter.close()


# =============================================================================================
# Text turns
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_turn_sends_send_text_and_falls_back_to_bot_llm_text(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    _on_text_turn(mock, mock.channel, _llm_steps(mock, "Sure", ",", " you are", " booked."))
    reply = await adapter.send("Book me for Tuesday", None)
    send_texts = [m for m in mock.channel.sent_messages if m.get("type") == "send-text"]
    assert len(send_texts) == 1
    assert send_texts[0]["data"] == {
        "content": "Book me for Tuesday",
        "options": {"run_immediately": True, "audio_response": False},
    }
    assert reply.text == "Sure, you are booked."  # bot-llm-text tokens concatenated
    assert reply.latency_ms == pytest.approx(300.0, abs=1.0)  # send-text -> first bot-llm-text
    assert mock.channel.sent_audio == []
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_turn_prefers_bot_output_text(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.25, rtvi("bot-llm-text", {"text": "tokens that are not used"})),
        (0.4, rtvi("bot-output", _bot_output("Hello there. "))),
        (0.5, rtvi("bot-output", _bot_output("How can I help?"))),
        (0.6, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, mock.channel, steps)
    reply = await adapter.send("Hi", None)
    assert _norm(reply.text) == "Hello there. How can I help?"
    assert reply.latency_ms == pytest.approx(250.0, abs=1.0)  # first bot-output OR bot-llm-text
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_turn_ends_after_the_grace_window(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, end_of_turn_grace_ms=500)
    adapter = await _started(cfg, mock, mode)
    fired = []
    _on_text_turn(mock, mock.channel, _llm_steps(mock, "Done."), fired)  # stopped at +0.4
    reply = await adapter.send("Hi", None)
    assert reply.text == "Done."
    end = fired[0] + 0.4 + 0.5
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_pause_shorter_than_grace_stays_one_turn(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)  # grace 800 ms
    adapter = await _started(cfg, mock, mode)
    fired = []
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.3, rtvi("bot-llm-text", {"text": "Let me check. "})),
        (0.4, rtvi("bot-llm-stopped")),
        (0.9, rtvi("bot-llm-started")),  # 0.5 s pause < 0.8 s grace: the turn goes on
        (1.0, rtvi("bot-llm-text", {"text": "You are "})),
        (1.5, rtvi("bot-llm-text", {"text": "booked."})),  # after the first grace window would have ended
        (1.6, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, mock.channel, steps, fired)
    reply = await adapter.send("Book me", None)
    assert reply.text == "Let me check. You are booked."
    end = fired[0] + 1.6 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_pause_longer_than_grace_ends_the_turn(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    fired = []
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.3, rtvi("bot-llm-text", {"text": "Let me check."})),
        (0.4, rtvi("bot-llm-stopped")),
        (1.5, rtvi("bot-llm-started")),  # 1.1 s pause > 0.8 s grace
        (1.6, rtvi("bot-llm-text", {"text": " Something later."})),
        (1.7, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, mock.channel, steps, fired)
    reply = await adapter.send("Book me", None)
    assert reply.text == "Let me check."
    end = fired[0] + 0.4 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tool_call_between_two_llm_runs_stays_one_turn(impulse_project, mock_pipecat_channel, mode):
    from impulse_evals.models import ToolCall

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.3, rtvi("llm-function-call-in-progress",
                   {"tool_call_id": "call-1", "function_name": "book_slot", "arguments": BOOK_ARGS})),
        (0.35, rtvi("bot-llm-stopped")),  # a function call is in progress: not the end of the turn
        (2.35, rtvi("llm-function-call-stopped",
                    {"tool_call_id": "call-1", "function_name": "book_slot", "cancelled": False,
                     "result": {"success": True, "slot": "10:00"}})),
        (2.5, rtvi("bot-llm-started")),
        (2.6, rtvi("bot-llm-text", {"text": "You are booked for 10:00."})),
        (2.8, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, mock.channel, steps)
    reply = await adapter.send("Book me for Tuesday", None)
    assert reply.text == "You are booked for 10:00."
    assert reply.tool_calls == [
        ToolCall(name="book_slot", arguments=BOOK_ARGS, result={"success": True, "slot": "10:00"}, success=True)
    ]
    assert reply.latency_ms == pytest.approx(2600.0, abs=1.0)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_turn_timeout_gives_exact_error(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)  # text mode: voice_config.timeout_s default 30
    adapter = await _started(cfg, mock, mode)
    fired = []
    _on_text_turn(mock, mock.channel, [(0.2, mock.rtvi("bot-llm-started"))], fired)  # never stops
    with pytest.raises(Exception) as info:
        await adapter.send("Hello?", None)
    assert str(info.value) == TURN_TIMEOUT.format(n=30)
    assert mock.clock.now >= fired[0] + 30 - 0.01
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_turn_timeout_uses_voice_config_timeout(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, voice=True, timeout_s=7)
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    steps = [(0.1, mock.rtvi("bot-started-speaking"))]  # the bot never stops speaking
    channel.on_audio(_plan(channel, steps), match=_after_speech(channel, 8000))
    with pytest.raises(Exception) as info:
        await adapter.send("Book a slot", _user_wav(0.5))
    assert str(info.value) == TURN_TIMEOUT.format(n=7)
    await adapter.close()


# =============================================================================================
# Voice turns
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_turn_collects_bot_audio_and_measures_ttfa_from_raw_arrival(
    impulse_project, mock_pipecat_channel, mode
):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    silence = mock.silence(0.1)
    quiet = mock.tone(0.1, amplitude=0.005)  # peak below 1% of full scale: not the first audio
    loud = mock.tone(0.3)
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.1, mock.BotAudio(silence)),
        (0.2, mock.BotAudio(quiet)),
        (0.3, mock.BotAudio(loud)),  # first non-silent bot audio: 0.3 s after the user's speech
        (0.35, rtvi("bot-output", _bot_output("Your slot is booked.", spoken=True, spoken_status="completed"))),
        (0.7, rtvi("bot-stopped-speaking")),
    ]
    adapter, reply, channel = await _voice_turn(impulse_project, mock, mode, steps)
    assert reply.latency_ms == pytest.approx(300.0, abs=1.0)
    assert reply.audio_bytes is not None
    channels, width, rate, frames = read_wav(reply.audio_bytes)
    assert (channels, width, rate) == (1, 2, 16000)
    assert frames == silence + quiet + loud
    assert reply.text == "Your slot is booked."
    assert "send-text" not in channel.sent_types()  # voice turns send audio only
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_turn_text_counts_spoken_or_completed_segments_once(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.2, mock.BotAudio(mock.tone(0.5))),
        (0.25, rtvi("bot-output", _bot_output("Hello there. ", segment_id="s1", spoken=False,
                                              will_be_spoken=True, spoken_status="new"))),
        (0.3, rtvi("bot-output", _bot_output("Hello there. ", segment_id="s1", spoken=True,
                                             spoken_status="completed"))),
        (0.33, rtvi("bot-output", _bot_output("Hello there. ", segment_id="s1", spoken=True))),  # same segment again
        # word-timestamp TTS: the segment is reported complete without "spoken": true
        (0.35, rtvi("bot-output", _bot_output("How can I ", segment_id="s2", spoken_status="in-progress"))),
        (0.4, rtvi("bot-output", _bot_output("How can I help? ", segment_id="s2", spoken_status="completed"))),
        (0.5, rtvi("bot-output", _bot_output("Goodbye.", spoken=True))),  # no segment_id
        (0.9, rtvi("bot-stopped-speaking")),
    ]
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps)
    assert _norm(reply.text) == "Hello there. How can I help? Goodbye."
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_pause_shorter_than_grace_stays_one_turn(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    first, second = mock.tone(0.3), mock.tone(0.4, frequency_hz=550.0)
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.2, mock.BotAudio(first)),
        (0.25, rtvi("bot-output", _bot_output("Let me check. ", spoken=True))),
        (0.6, rtvi("bot-stopped-speaking")),
        (1.1, rtvi("bot-started-speaking")),  # 0.5 s pause < 0.8 s grace: the turn goes on
        (1.2, mock.BotAudio(second)),
        (1.6, rtvi("bot-output", _bot_output("You are booked.", spoken=True))),  # after the first grace window
        (1.7, rtvi("bot-stopped-speaking")),
    ]
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps)
    assert _norm(reply.text) == "Let me check. You are booked."
    frames = read_wav(reply.audio_bytes)[3]
    assert frames == first + second
    await adapter.close()


@pytest.mark.asyncio
async def test_websocket_voice_sends_chunk_frames_at_sample_rate_then_trailing_silence(
    impulse_project, mock_pipecat_channel
):
    mock = mock_pipecat_channel
    speech = mock.tone(0.5)
    cfg = _cfg(impulse_project, "websocket", voice=True, chunk_ms=25, trailing_silence_ms=200, realtime_pacing=False)
    adapter, reply, channel = await _voice_turn(impulse_project, mock, "websocket", _voice_steps(mock), cfg=cfg)
    frames = channel.sent_audio
    assert all(frame["sample_rate"] == 16000 for frame in frames)
    assert all(len(frame["pcm"]) == 800 for frame in frames)  # 25 ms at 16 kHz, 16-bit
    assert len(frames) == 20 + 8  # 0.5 s of speech, then 200 ms of trailing silence
    assert b"".join(frame["pcm"] for frame in frames[:20]) == speech
    assert all(frame["pcm"] == b"\x00" * 800 for frame in frames[20:])
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pacing", [True, False])
async def test_websocket_realtime_pacing(impulse_project, mock_pipecat_channel, pacing):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "websocket", voice=True, realtime_pacing=pacing)  # chunk 20 ms, trailing 600 ms
    adapter, reply, channel = await _voice_turn(impulse_project, mock, "websocket", _voice_steps(mock), cfg=cfg)
    times = [frame["t"] for frame in channel.sent_audio]
    assert len(times) == 25 + 30
    spread = times[-1] - times[0]
    if pacing:
        assert spread == pytest.approx(54 * 0.02, abs=0.03)
    else:
        assert spread < 0.005
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_user_audio_is_sent_at_the_configured_sample_rate(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, voice=True, sample_rate=8000)
    adapter, reply, channel = await _voice_turn(impulse_project, mock, mode, _voice_steps(mock), cfg=cfg)
    frames = channel.sent_audio
    assert frames
    assert all(frame["sample_rate"] == 8000 for frame in frames)
    pcm = b"".join(frame["pcm"] for frame in frames)
    samples = len(pcm.rstrip(b"\x00")) // 2  # trailing silence (websocket mode) removed
    assert abs(samples - 4000) <= 80  # 0.5 s of 16 kHz speech resampled to 8 kHz
    if mode == "websocket":
        assert all(len(frame["pcm"]) == 320 for frame in frames)  # 20 ms at 8 kHz
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_interrupted_when_user_speaks_during_bot_speech(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.2, rtvi("user-started-speaking")),  # the user talks over the bot
        (0.3, rtvi("bot-stopped-speaking")),
        (0.6, rtvi("bot-started-speaking")),
        (0.65, mock.BotAudio(mock.tone(0.3))),
        (0.7, rtvi("bot-output", _bot_output("Sorry, go ahead.", spoken=True))),
        (1.0, rtvi("bot-stopped-speaking")),
    ]
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps)
    assert reply.interrupted is True
    assert reply.text == "Sorry, go ahead."
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_not_interrupted_when_user_speech_precedes_bot_speech(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    steps = [(0.0, rtvi("user-started-speaking")), (0.05, rtvi("user-stopped-speaking"))] + _voice_steps(mock, start=0.2)
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps)
    assert reply.interrupted is False
    await adapter.close()


def _transcription(text, final):
    return {"text": text, "user_id": "", "timestamp": "2026-10-19T10:00:00.000Z", "final": final}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_heard_text_uses_final_transcriptions_only(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    steps = [
        (0.0, rtvi("user-transcription", _transcription("book a", False))),
        (0.05, rtvi("user-transcription", _transcription("book a slot ", True))),
        (0.06, rtvi("user-transcription", _transcription("for Tuesday", True))),
    ] + _voice_steps(mock, start=0.2)
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps)
    assert _norm(reply.heard_text) == "book a slot for Tuesday"
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_heard_text_none_without_final_transcription(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    steps = [(0.0, mock.rtvi("user-transcription", _transcription("book a", False)))] + _voice_steps(mock, start=0.2)
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps)
    assert reply.heard_text is None
    await adapter.close()


# =============================================================================================
# Tool calls
# =============================================================================================


def _in_progress(mock, call_id, name=None, arguments=None):
    data = {"tool_call_id": call_id}
    if name is not None:
        data["function_name"] = name
    if arguments is not None:
        data["arguments"] = arguments
    return mock.rtvi("llm-function-call-in-progress", data)


def _stopped(mock, call_id, name, cancelled=False, **result):
    data = {"tool_call_id": call_id, "function_name": name, "cancelled": cancelled}
    data.update(result)
    return mock.rtvi("llm-function-call-stopped", data)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tool_call_mapping(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    steps = [
        (0.1, rtvi("bot-llm-started")),
        (0.15, _in_progress(mock, "c1", "book_slot", BOOK_ARGS)),
        (0.16, _in_progress(mock, "c2", "cancel_slot", {"slot_id": "s-9"})),
        (0.17, _in_progress(mock, "c3", "hold_slot", {"slot_id": "s-1"})),
        (0.18, _in_progress(mock, "c4", "pay", {"amount": 100})),
        (0.19, _in_progress(mock, "c5", "notify", {"channel": "sms"})),
        (0.20, _in_progress(mock, "c6", "lookup", {"q": "tuesday"})),
        (0.21, _in_progress(mock, "c7", "log_call", {"level": "info"})),
        (0.25, rtvi("bot-llm-stopped")),
        (0.50, _stopped(mock, "c1", "book_slot", result={"success": True, "slot": "10:00"})),
        (0.51, _stopped(mock, "c2", "cancel_slot", cancelled=True)),
        (0.52, _stopped(mock, "c3", "hold_slot", result={"error": "slot taken"})),
        (0.53, _stopped(mock, "c4", "pay", result={"success": False, "error": "card declined"})),
        (0.54, _stopped(mock, "c5", "notify", result="queued")),
        (0.55, _stopped(mock, "c6", "lookup")),  # no result reported
        (0.56, _stopped(mock, "c7", "log_call", result={"error": ""})),
        (0.60, rtvi("llm-function-call", {"function_name": "legacy_lookup", "tool_call_id": "d1", "args": {"q": "x"}})),
        (0.70, rtvi("bot-llm-started")),
        (0.80, rtvi("bot-llm-text", {"text": "Done."})),
        (0.90, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, mock.channel, steps)
    reply = await adapter.send("Do everything", None)
    calls = {call.name: call for call in reply.tool_calls}
    assert [call.name for call in reply.tool_calls] == [
        "book_slot", "cancel_slot", "hold_slot", "pay", "notify", "lookup", "log_call", "legacy_lookup",
    ]
    assert calls["book_slot"].arguments == BOOK_ARGS
    assert calls["book_slot"].result == {"success": True, "slot": "10:00"}
    assert (calls["book_slot"].success, calls["book_slot"].error) == (True, None)
    assert (calls["cancel_slot"].success, calls["cancel_slot"].error) == (False, "cancelled")
    assert calls["cancel_slot"].arguments == {"slot_id": "s-9"}
    assert calls["hold_slot"].result == {"error": "slot taken"}
    assert (calls["hold_slot"].success, calls["hold_slot"].error) == (False, "slot taken")
    assert calls["pay"].result == {"success": False, "error": "card declined"}
    assert calls["pay"].success is False
    assert calls["notify"].result == {"value": "queued"}  # a non-dict result is wrapped
    assert (calls["notify"].success, calls["notify"].error) == (None, None)
    assert (calls["lookup"].result, calls["lookup"].success) == (None, None)
    assert calls["log_call"].result == {"error": ""}
    assert (calls["log_call"].success, calls["log_call"].error) == (None, None)  # empty error: not captured
    assert calls["legacy_lookup"].arguments == {"q": "x"}
    assert calls["legacy_lookup"].success is None
    assert reply.text == "Done."
    assert FUNCTION_CALL_WARNING not in adapter.get_warnings()
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_function_calls_without_name_or_arguments_warn_once(impulse_project, mock_pipecat_channel, mode):
    from impulse_evals.models import ToolCall

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    first = [
        (0.1, rtvi("bot-llm-started")),
        (0.15, _in_progress(mock, "c1")),  # report level NONE: only the id
        (0.2, rtvi("bot-llm-stopped")),
        (0.3, rtvi("llm-function-call-stopped", {"tool_call_id": "c1", "cancelled": False})),
        (0.4, rtvi("bot-llm-started")),
        (0.5, rtvi("bot-llm-text", {"text": "Done."})),
        (0.6, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, mock.channel, first)
    reply = await adapter.send("Book me", None)
    assert reply.tool_calls == [ToolCall(name="unknown", arguments={})]
    second = [
        (0.1, rtvi("bot-llm-started")),
        (0.15, _in_progress(mock, "c2", "book_slot")),  # report level NAME: no arguments
        (0.2, rtvi("bot-llm-stopped")),
        (0.3, rtvi("llm-function-call-stopped", {"tool_call_id": "c2", "function_name": "book_slot", "cancelled": False})),
        (0.4, rtvi("bot-llm-started")),
        (0.5, rtvi("bot-llm-text", {"text": "Done again."})),
        (0.6, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, mock.channel, second)
    reply = await adapter.send("Book me again", None)
    assert reply.tool_calls == [ToolCall(name="book_slot", arguments={})]
    assert adapter.get_warnings().count(FUNCTION_CALL_WARNING) == 1
    await adapter.close()


# =============================================================================================
# Per-service latency (informational)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_pipecat_metrics_stored_in_ms_with_turn_index(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    first = _llm_steps(mock, "Hello.") + [
        (0.25, rtvi("metrics", {
            "ttfb": [{"processor": "OpenAILLMService#0", "model": "llm-x", "value": 0.42}],
            "processing": [{"processor": "OpenAILLMService#0", "model": "llm-x", "value": 1.25}],
            "tokens": [{"processor": "OpenAILLMService#0", "model": "llm-x",
                        "value": {"prompt_tokens": 10, "completion_tokens": 5}}],
        })),
    ]
    _on_text_turn(mock, mock.channel, first)
    await adapter.send("Hi", None)
    second = _llm_steps(mock, "Bye.") + [
        (0.25, rtvi("metrics", {"ttfb": [{"processor": "CartesiaTTSService#0", "value": 0.2}]})),
    ]
    _on_text_turn(mock, mock.channel, second)
    await adapter.send("Bye", None)
    await adapter.close()

    entries = adapter.get_metadata()["pipecat_metrics"]
    assert len(entries) == 3  # "tokens" and other kinds are not per-service latency
    simplified = sorted(
        (entry["turn_index"], entry["kind"], entry["processor"], entry.get("model"), round(entry["value_ms"], 6))
        for entry in entries
    )
    assert simplified == [
        (2, "processing", "OpenAILLMService#0", "llm-x", 1250.0),
        (2, "ttfb", "OpenAILLMService#0", "llm-x", 420.0),
        (4, "ttfb", "CartesiaTTSService#0", None, 200.0),
    ]
    for entry in entries:
        assert set(entry) == {"turn_index", "kind", "processor", "model", "value_ms"}


# =============================================================================================
# Bot closes the connection mid-conversation: AgentClosed (normal end)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_bot_close_mid_conversation_raises_agent_closed(impulse_project, mock_pipecat_channel, mode):
    from impulse_evals.adapters.base import AgentClosed

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    steps = [
        (0.1, rtvi("bot-llm-started")),
        (0.2, rtvi("bot-llm-text", {"text": "Goodbye."})),
        (0.3, mock.BotClose(1000)),
    ]
    _on_text_turn(mock, mock.channel, steps)
    with pytest.raises(AgentClosed) as info:
        await adapter.send("Bye", None)
    if info.value.reply is not None:
        assert info.value.reply.text == "Goodbye."
    await adapter.close()  # must tolerate the closed connection


# =============================================================================================
# Outbound opener
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_outbound_opener_sends_nothing_and_waits_for_the_bot(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, scenarios=[impulse_project.scenario("outbound_probe", direction="outbound")])
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    sent_before = len(channel.sent)
    for offset, item in _llm_steps(mock, "Hello, this is ", "Sunrise Clinic.", start=0.5):
        channel.emit(item, after=offset)
    reply = await adapter.send(None, None)
    assert len(channel.sent) == sent_before  # no send-text, no audio
    assert reply.text == "Hello, this is Sunrise Clinic."
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_outbound_voice_opener_collects_bot_audio(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(
        impulse_project, mode, voice=True, scenarios=[impulse_project.scenario("outbound_probe", direction="outbound")]
    )
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    sent_before = len(channel.sent)
    greeting = mock.tone(0.4)
    rtvi = mock.rtvi
    channel.emit(rtvi("bot-started-speaking"), after=0.5)
    channel.emit(mock.BotAudio(greeting), after=0.6)
    channel.emit(rtvi("bot-output", _bot_output("Hello, this is Sunrise Clinic.", spoken=True)), after=0.65)
    channel.emit(rtvi("bot-stopped-speaking"), after=1.0)
    reply = await adapter.send(None, None)
    assert len(channel.sent) == sent_before
    assert reply.text == "Hello, this is Sunrise Clinic."
    assert read_wav(reply.audio_bytes)[3] == greeting
    await adapter.close()


# =============================================================================================
# Configuration and optional-extra errors
# =============================================================================================


def test_mode_daily_is_a_config_error_with_the_v2_message(impulse_project):
    from impulse_evals.config import ConfigError

    with pytest.raises(ConfigError) as info:
        _cfg(impulse_project, "daily")
    assert info.value.message == DAILY_MESSAGE
    assert info.value.field == "agent.pipecat.mode"


@pytest.mark.asyncio
async def test_adapter_rejects_an_unvalidated_v2_mode(impulse_project, mock_pipecat_channel):
    from impulse_evals.config import ConfigError

    cfg = _cfg(impulse_project, "eval")
    cfg.agent.pipecat.mode = "daily"
    with pytest.raises(ConfigError) as info:
        adapter = _adapter(cfg, mock_pipecat_channel)
        await adapter.start(cfg.scenarios[0])
    assert info.value.message == DAILY_MESSAGE


def test_websocket_only_keys_warn_in_eval_mode_but_sample_rate_never(impulse_project):
    cfg = _cfg(impulse_project, "eval", chunk_ms=40, realtime_pacing=False, trailing_silence_ms=300, sample_rate=24000)
    for key in ("chunk_ms", "realtime_pacing", "trailing_silence_ms"):
        assert f"pipecat.{key} is ignored in eval mode." in cfg.warnings
    assert "pipecat.sample_rate is ignored in eval mode." not in cfg.warnings
    assert cfg.agent.pipecat.sample_rate == 24000


@pytest.mark.asyncio
async def test_concurrency_above_one_is_capped_with_exact_message(
    impulse_project, llm_router, mock_pipecat_channel, capsys
):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    from impulse_evals.runner import run

    mock = mock_pipecat_channel
    project = impulse_project
    scenarios = [project.scenario(f"capped_{index}", max_turns=1, metrics=["UNIV-013"]) for index in (1, 2, 3)]
    cfg = _cfg(project, "eval", scenarios=scenarios)
    channels = [mock.channel, mock.add_channel(), mock.add_channel()]
    for channel in channels:
        _handshake(mock, "eval", channel=channel)
        _on_text_turn(mock, channel, _llm_steps(mock, "Hello."))
    returned, overlaps = [], []

    def channel_factory(*args, **kwargs):
        channel = mock.factory(*args, **kwargs)
        if any(previous.connect_calls and not previous.closed for previous in returned):
            overlaps.append(channel.name)
        returned.append(channel)
        return channel

    result = await run(
        cfg,
        concurrency=4,
        adapter_factory=lambda config: PipecatAdapter(config, channel_factory=channel_factory, clock=mock.clock),
    )
    out = capsys.readouterr()
    assert (out.out + out.err).count(CAP_MESSAGE) == 1
    assert CAP_MESSAGE in result.warnings
    assert overlaps == []
    assert [s.error for s in result.scenario_results] == [None, None, None]
    assert all(len(channel.connect_calls) == 1 for channel in channels)


@pytest.mark.asyncio
async def test_run_merges_pipecat_metadata_into_the_transcript(impulse_project, llm_router, mock_pipecat_channel):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    from impulse_evals.runner import run

    mock = mock_pipecat_channel
    project = impulse_project
    cfg = _cfg(project, "websocket", scenarios=[project.scenario("with_metrics", max_turns=1, metrics=["UNIV-013"])])
    _handshake(mock, "websocket")
    steps = _llm_steps(mock, "Hello.") + [
        (0.25, mock.rtvi("metrics", {"ttfb": [{"processor": "OpenAILLMService#0", "model": "llm-x", "value": 0.3}]})),
    ]
    _on_text_turn(mock, mock.channel, steps)
    result = await run(
        cfg, adapter_factory=lambda config: PipecatAdapter(config, channel_factory=mock.factory, clock=mock.clock)
    )
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    entries = scenario_result.transcript.metadata["pipecat_metrics"]
    assert [(e["turn_index"], e["kind"], e["processor"]) for e in entries] == [(2, "ttfb", "OpenAILLMService#0")]
    assert entries[0]["value_ms"] == pytest.approx(300.0)
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent"]
    assert turns[1].text == "Hello."


@pytest.mark.asyncio
async def test_missing_extra_gives_exact_error(impulse_project, mock_pipecat_channel, extra_imports):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    extra_imports.pipecat("missing")
    cfg = _cfg(impulse_project, "websocket")
    with pytest.raises(Exception) as info:
        adapter = PipecatAdapter(cfg, clock=mock_pipecat_channel.clock)  # no channel_factory: the real channel
        await adapter.start(cfg.scenarios[0])
    assert MISSING_EXTRA in str(info.value)


@pytest.mark.asyncio
async def test_eval_mode_without_pipecat_evals_gives_exact_error(impulse_project, mock_pipecat_channel, extra_imports):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    extra_imports.pipecat("no_evals")
    cfg = _cfg(impulse_project, "eval")
    with pytest.raises(Exception) as info:
        adapter = PipecatAdapter(cfg, clock=mock_pipecat_channel.clock)
        await adapter.start(cfg.scenarios[0])
    assert EVALS_MISSING in str(info.value)


def test_adapter_module_imports_without_pipecat():
    """``impulse_evals.adapters.pipecat_adapter`` (and its Channel types) import in a fresh interpreter
    in which pipecat-ai cannot be imported (the extra is lazy-imported)."""
    code = (
        "import sys\n"
        "sys.modules['pipecat'] = None\n"
        "from impulse_evals.adapters.pipecat_adapter import (\n"
        "    PipecatAdapter, Channel, ChannelMessage, ChannelAudio, ChannelClose, Clock)\n"
        "ChannelMessage(data={'type': 'bot-ready'}); ChannelAudio(pcm=b'', sample_rate=16000, arrival_s=0.0)\n"
        "ChannelClose(code=1000)\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")


# =============================================================================================
# AUDIT batch 1 (findings 2, 3, 4, 6): close codes, voice end of turn during a function call,
# credentials in pipecat.url, literal credentials in pipecat.headers.
# =============================================================================================

CONSENT = "dpdp.consent_before_pii"
BOT_CLOSED = "the Pipecat bot closed the connection (close code {code})"
STRIPPED_URL = "ws://127.0.0.1:1/ws"
URL_CREDENTIAL_CASES = [
    pytest.param("ws://127.0.0.1:1/ws?token=SECRETTOKEN123", "SECRETTOKEN123", id="query-token"),
    pytest.param("ws://bot:URLPASS456@127.0.0.1:1/ws", "URLPASS456", id="userinfo"),
]
ORIGIN = "http://localhost:3000"
HEADER_CASES = [
    pytest.param({"X-Api-Key": "LITERALKEY456", "Origin": ORIGIN}, "LITERALKEY456", id="x-api-key"),
    pytest.param({"Authorization": "Bearer LITERALBEARER654", "Origin": ORIGIN}, "LITERALBEARER654", id="authorization"),
]
TOOL_RESULT = {"success": True, "slot": "10:00"}


def _run_factory(mock, channel_factory=None):
    """``adapter_factory`` for ``runner.run``: a ``PipecatAdapter`` on the fake channel and clock."""
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    factory = channel_factory or mock.factory
    return lambda config: PipecatAdapter(config, channel_factory=factory, clock=mock.clock)


def _first_speech(mock, first):
    """The bot's first voice reply: speaking from +0.1, audio at +0.2, "Let me check." at +0.25."""
    rtvi = mock.rtvi
    return [
        (0.1, rtvi("bot-started-speaking")),
        (0.2, mock.BotAudio(first)),
        (0.25, rtvi("bot-output", _bot_output("Let me check.", spoken=True, spoken_status="completed"))),
    ]


def _second_speech(mock, second, at):
    """The bot speaks again after the tool: started at ``at``, audio +0.1, text +0.2, stopped +0.6."""
    rtvi = mock.rtvi
    return [
        (at, rtvi("bot-started-speaking")),
        (at + 0.1, mock.BotAudio(second)),
        (at + 0.2, rtvi("bot-output", _bot_output("Booked for Tuesday.", spoken=True, spoken_status="completed"))),
        (at + 0.6, rtvi("bot-stopped-speaking")),
    ]


def _assert_ended_at(mock, fired, offset):
    end = fired[0] + offset + 0.8  # default end_of_turn_grace_ms 800
    assert end - 1e-9 <= mock.clock.now <= end + 0.25, (
        f"the turn ended {mock.clock.now - fired[0]:.3f} s after the user's speech; expected {offset + 0.8:.3f} s"
    )


class _EchoingHeadersChannel:
    """Wraps a fake channel; ``connect()`` fails like a rejected WebSocket upgrade whose error echoes the
    request headers (as proxies and ``websockets`` do)."""

    def __init__(self, inner):
        self._inner = inner

    async def connect(self, url, headers, query):
        raise RuntimeError(f"server rejected the WebSocket upgrade; request headers were {dict(headers)}")

    def __getattr__(self, name):
        return getattr(self._inner, name)


# -- finding 2: close codes ------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("code", [1006, 1011])
async def test_abnormal_close_during_turn_is_a_scenario_error(impulse_project, mock_pipecat_channel, mode, code):
    from impulse_evals.adapters.base import AgentClosed

    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    _on_text_turn(mock, mock.channel, [(0.1, mock.rtvi("bot-llm-started")), (0.2, mock.BotClose(code))])
    with pytest.raises(Exception) as info:
        await adapter.send("Hello?", None)
    assert not isinstance(info.value, AgentClosed), f"close code {code} must be a scenario error, not a normal end"
    assert str(info.value) == BOT_CLOSED.format(code=code)
    await adapter.close()  # must tolerate the closed connection


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("code", [1000, 1001])
async def test_normal_close_during_turn_stays_agent_closed(impulse_project, mock_pipecat_channel, mode, code):
    from impulse_evals.adapters.base import AgentClosed

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    steps = [
        (0.1, rtvi("bot-llm-started")),
        (0.2, rtvi("bot-llm-text", {"text": "Goodbye."})),
        (0.3, mock.BotClose(code)),
    ]
    _on_text_turn(mock, mock.channel, steps)
    with pytest.raises(AgentClosed) as info:
        await adapter.send("Bye", None)
    assert str(info.value) == BOT_CLOSED.format(code=code)
    if info.value.reply is not None:
        assert info.value.reply.text == "Goodbye."
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_close_during_system_prompt_wait_fails_start(impulse_project, mock_pipecat_channel, mode):
    from impulse_evals.adapters.base import AgentClosed

    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, system_prompt=None)  # start() asks the bot and waits up to 3 s
    _handshake(mock, mode)
    mock.channel.on_send(_is_prompt_request, mock.BotClose(1006), after=0.5)  # the bot dies during the wait
    adapter = _adapter(cfg, mock)
    with pytest.raises(Exception) as info:
        await adapter.start(cfg.scenarios[0])
    assert not isinstance(info.value, AgentClosed)
    assert str(info.value) == BOT_CLOSED.format(code=1006)
    assert mock.channel.sent_types().count("send-text") == 0
    await adapter.close()


@pytest.mark.asyncio
async def test_run_with_bot_closing_after_bot_ready_is_a_scenario_error(
    impulse_project, llm_router, mock_pipecat_channel
):
    from impulse_evals.runner import run

    mock = mock_pipecat_channel
    project = impulse_project
    cfg = _cfg(
        project, "eval", system_prompt=None,
        scenarios=[project.scenario("bot_dies_on_join", assertions=[CONSENT], metrics=["UNIV-001"])],
    )
    _handshake(mock, "eval")
    mock.channel.on_send(_is_prompt_request, mock.BotClose(1006), after=0.5)
    result = await run(cfg, adapter_factory=_run_factory(mock))
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == BOT_CLOSED.format(code=1006)
    assert scenario_result.passed is False
    assert scenario_result.metric_results == []
    assert scenario_result.assertion_results == []
    assert llm_router.calls("judge") == []
    assert result.failed == 1


# -- finding 3: voice end of turn holds while a function call is in progress -------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_tool_call_slower_than_grace_stays_one_turn(impulse_project, mock_pipecat_channel, mode):
    from impulse_evals.models import ToolCall

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    first, second = mock.tone(0.3), mock.tone(0.4, frequency_hz=550.0)
    fired = []
    steps = _first_speech(mock, first) + [
        (0.15, _in_progress(mock, "call-1", "book_slot", BOOK_ARGS)),  # emitted with "Let me check."
        (0.6, rtvi("bot-stopped-speaking")),  # TTS done; the tool is still running
        (2.3, _stopped(mock, "call-1", "book_slot", result=TOOL_RESULT)),  # 1.7 s after the stop: past the grace
    ] + _second_speech(mock, second, at=2.5)
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps, fired=fired)
    assert _norm(reply.text) == "Let me check. Booked for Tuesday."
    assert reply.tool_calls == [ToolCall(name="book_slot", arguments=BOOK_ARGS, result=TOOL_RESULT, success=True)]
    assert read_wav(reply.audio_bytes)[3] == first + second
    assert reply.latency_ms == pytest.approx(200.0, abs=1.0)
    _assert_ended_at(mock, fired, 3.1)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_tool_call_starting_after_the_stop_within_grace_holds_the_turn(
    impulse_project, mock_pipecat_channel, mode
):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    first, second = mock.tone(0.3), mock.tone(0.4, frequency_hz=550.0)
    fired = []
    steps = _first_speech(mock, first) + [
        (0.6, rtvi("bot-stopped-speaking")),
        (1.2, _in_progress(mock, "call-1", "book_slot", BOOK_ARGS)),  # 0.6 s after the stop: inside the grace
        (2.6, _stopped(mock, "call-1", "book_slot", result=TOOL_RESULT)),
    ] + _second_speech(mock, second, at=2.8)
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps, fired=fired)
    assert _norm(reply.text) == "Let me check. Booked for Tuesday."
    assert [(call.name, call.success) for call in reply.tool_calls] == [("book_slot", True)]
    assert read_wav(reply.audio_bytes)[3] == first + second
    _assert_ended_at(mock, fired, 3.4)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_grace_restarts_at_the_last_function_call_stop(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    first = mock.tone(0.3)
    fired = []
    steps = _first_speech(mock, first) + [
        (0.15, _in_progress(mock, "c1", "book_slot", BOOK_ARGS)),
        (0.16, _in_progress(mock, "c2", "notify", {"channel": "sms"})),
        (0.6, rtvi("bot-stopped-speaking")),
        (1.5, _stopped(mock, "c1", "book_slot", result=TOOL_RESULT)),
        (2.4, _stopped(mock, "c2", "notify", result={"success": True})),  # the last call ends here
    ]  # the bot never speaks again: the grace window runs from the last stop
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps, fired=fired)
    assert _norm(reply.text) == "Let me check."
    assert [(call.name, call.success) for call in reply.tool_calls] == [("book_slot", True), ("notify", True)]
    _assert_ended_at(mock, fired, 2.4)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_fast_tool_within_grace_still_one_turn(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    first, second = mock.tone(0.3), mock.tone(0.4, frequency_hz=550.0)
    steps = _first_speech(mock, first) + [
        (0.15, _in_progress(mock, "call-1", "book_slot", BOOK_ARGS)),
        (0.6, rtvi("bot-stopped-speaking")),
        (1.0, _stopped(mock, "call-1", "book_slot", result=TOOL_RESULT)),  # 0.4 s tool
    ] + _second_speech(mock, second, at=1.3)  # 0.7 s pause < 0.8 s grace
    adapter, reply, _ = await _voice_turn(impulse_project, mock, mode, steps)
    assert _norm(reply.text) == "Let me check. Booked for Tuesday."
    assert [(call.name, call.success) for call in reply.tool_calls] == [("book_slot", True)]
    assert read_wav(reply.audio_bytes)[3] == first + second
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_stuck_tool_ends_in_the_turn_timeout(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice=True, timeout_s=7)
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    steps = _first_speech(mock, mock.tone(0.3)) + [
        (0.15, _in_progress(mock, "call-1", "book_slot", BOOK_ARGS)),
        (0.6, rtvi("bot-stopped-speaking")),
    ]  # llm-function-call-stopped never arrives
    channel.on_audio(_plan(channel, steps), match=_after_speech(channel, 8000))
    with pytest.raises(Exception) as info:
        await adapter.send("Book a slot for Tuesday", _user_wav(0.5))
    assert str(info.value) == TURN_TIMEOUT.format(n=7)
    await adapter.close()


# -- finding 4: credentials in pipecat.url ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("url,secret", URL_CREDENTIAL_CASES)
async def test_url_credentials_are_stripped_from_agent_entrypoint_errors_and_reports(
    impulse_project, llm_router, mock_pipecat_channel, tmp_path, url, secret
):
    from impulse_evals.reporter import render_html
    from impulse_evals.runner import run

    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "eval", url=url, handshake_timeout_s=2)  # the fake bot never sends bot-ready
    result = await run(cfg, adapter_factory=_run_factory(mock))
    scenario_result = result.scenario_results[0]
    assert result.agent_entrypoint == STRIPPED_URL
    assert scenario_result.error == HANDSHAKE_TIMEOUT.format(n=2, url=STRIPPED_URL)
    assert secret not in result.model_dump_json()
    assert secret not in render_html(result, embed_audio=True, run_dir=tmp_path)
    assert mock.channel.connect_url == url  # the connection itself still uses the full URL


@pytest.mark.asyncio
@pytest.mark.parametrize("url,secret", URL_CREDENTIAL_CASES)
async def test_dry_run_strips_url_credentials(impulse_project, url, secret):
    from impulse_evals.runner import run

    cfg = _cfg(impulse_project, "eval", url=url)
    result = await run(cfg, dry_run=True)
    assert result.agent_entrypoint == STRIPPED_URL
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("url,secret", URL_CREDENTIAL_CASES)
async def test_close_before_bot_ready_error_strips_url_credentials(impulse_project, mock_pipecat_channel, url, secret):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "websocket", url=url)
    mock.channel.on_send("client-ready", mock.BotClose(code=4003))  # dev runner: bad token / Origin
    adapter = _adapter(cfg, mock)
    with pytest.raises(Exception) as info:
        await adapter.start(cfg.scenarios[0])
    assert secret not in str(info.value)
    assert STRIPPED_URL in str(info.value)
    await adapter.close()


@pytest.mark.asyncio
async def test_other_query_parameters_survive_stripping(impulse_project):
    from impulse_evals.runner import run

    cfg = _cfg(impulse_project, "eval", url="ws://127.0.0.1:1/ws?room=a&token=SECRETTOKEN123&x=1")
    result = await run(cfg, dry_run=True)
    assert result.agent_entrypoint == "ws://127.0.0.1:1/ws?room=a&x=1"


# -- finding 6: literal credentials in pipecat.headers -----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("headers,secret", HEADER_CASES)
async def test_literal_pipecat_header_values_are_redacted_from_scenario_errors(
    impulse_project, llm_router, mock_pipecat_channel, tmp_path, headers, secret
):
    from impulse_evals.reporter import render_html
    from impulse_evals.runner import run

    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "eval", headers=headers)
    result = await run(cfg, adapter_factory=_run_factory(mock, lambda: _EchoingHeadersChannel(mock.factory())))
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is not None
    assert secret not in scenario_result.error, scenario_result.error
    assert "[REDACTED]" in scenario_result.error
    assert ORIGIN in scenario_result.error  # the Origin header is a public origin: never redacted
    assert secret not in result.model_dump_json()
    assert secret not in render_html(result, embed_audio=True, run_dir=tmp_path)


# =============================================================================================
# AUDIT batch 2 (findings 8 and 15): bot output already in flight when the first user turn is sent;
# bot error texts redacted before they are shortened. (Finding 12 and the eval-harness side of
# finding 15 need pipecat-ai and a loopback server: tests/test_pipecat_eval_connect.py.)
# =============================================================================================

GREETING = "Hello, this is Sunrise Clinic."
TUESDAY_REPLY = "Tuesday works."
VOICE_REPLY = "Tuesday at ten works."

F15_KEY_ENV = "IMPULSE_TEST_PIPECAT_TOKEN"
F15_KEY = "pck-test-PIPECAT-SECRET-0123456789abcdef"  # 40 characters
F15_LEAD = "e" * 172 + "invalid token "  # 186 characters: the 200-character cut falls 14 characters into the key
F15_LEAKY = F15_LEAD + F15_KEY
RTVI_ERROR_PREFIX = "Pipecat bot reported an RTVI error: "
F15_SECRET_SOURCES = [
    pytest.param("api_key_env", id="api-key-env-token"),
    pytest.param("header", id="literal-header"),
    pytest.param("url", id="url-token"),
]
F15_ERROR_MESSAGES = [
    pytest.param("error", "message", id="rtvi-error"),
    pytest.param("error-response", "error", id="rtvi-error-response"),
]


def _before_bot_ready(mock, mode, *items, gap=0.05):
    """The fake bot sends ``items`` (``gap`` apart) as soon as the connection opens and ``bot-ready``
    after them, so the adapter reads them during ``start()`` (an inbound greeting fired by
    ``on_client_connected``). ``agent.system_prompt`` set: ``start()`` returns right after ``bot-ready``."""
    channel = mock.channel
    channel.on_connect(*items, gap=gap)
    _handshake(mock, mode, channel=channel, after=gap * len(items))
    return channel


def _secret_settings(monkeypatch, source):
    """``pipecat`` settings that give the adapter ``F15_KEY`` itself: the ``api_key_env`` token, a literal
    header value, or a ``?token=`` in ``pipecat.url``."""
    if source == "api_key_env":
        monkeypatch.setenv(F15_KEY_ENV, F15_KEY)
        return {"api_key_env": F15_KEY_ENV}
    if source == "header":
        return {"headers": {"X-Api-Key": F15_KEY, "Origin": ORIGIN}}
    return {"url": f"{URL}/ws?token={F15_KEY}"}


def _patch_pipecat_class(monkeypatch, mock):
    """Like ``_patch_pipecat`` in tests/test_doctor.py: the runner's default adapter factory builds the
    real ``PipecatAdapter`` class (with every keyword it passes) on the fake channel and clock."""
    from impulse_evals.adapters import pipecat_adapter

    real = pipecat_adapter.PipecatAdapter

    def patched(config, *args, **kwargs):
        kwargs["channel_factory"] = mock.factory
        kwargs["clock"] = mock.clock
        return real(config, *args, **kwargs)

    monkeypatch.setattr(pipecat_adapter, "PipecatAdapter", patched)


# -- finding 8: a greeting LLM run that started before the first text turn --------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_greeting_in_flight_at_first_text_turn_does_not_set_its_latency(
    impulse_project, mock_pipecat_channel, mode
):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    channel = _before_bot_ready(mock, mode, rtvi("bot-llm-started"))  # the greeting run starts during start()
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    fired = []
    steps = [
        (0.1, rtvi("bot-llm-text", {"text": GREETING})),  # the greeting's text, 0.1 s after send-text
        (0.3, rtvi("bot-llm-stopped")),  # the greeting run ends
        (1.0, rtvi("bot-llm-started")),  # the real reply to the user's turn
        (1.1, rtvi("bot-llm-text", {"text": TUESDAY_REPLY})),
        (1.2, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, channel, steps, fired)
    reply = await adapter.send("Is Tuesday free?", None)
    assert reply.latency_ms == pytest.approx(1100.0, abs=1.0)  # send-text -> the reply's first text
    assert reply.text.endswith(TUESDAY_REPLY)
    assert reply.text == f"{GREETING} {TUESDAY_REPLY}"  # the greeting is never dropped, and comes first
    end = fired[0] + 1.2 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("greeting_text", ["during-start", "after-send-text"])
async def test_greeting_stopping_after_send_text_does_not_end_the_first_turn(
    impulse_project, mock_pipecat_channel, mode, greeting_text
):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    early = [rtvi("bot-llm-started")]
    later = [(0.3, rtvi("bot-llm-stopped"))]  # the greeting's late stop: its grace window would end at +1.1
    if greeting_text == "during-start":
        early.append(rtvi("bot-llm-text", {"text": GREETING}))
    else:
        later.insert(0, (0.1, rtvi("bot-llm-text", {"text": GREETING})))
    channel = _before_bot_ready(mock, mode, *early)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    fired = []
    steps = later + [
        (1.5, rtvi("bot-llm-started")),  # the real reply starts after the greeting's grace window
        (1.6, rtvi("bot-llm-text", {"text": TUESDAY_REPLY})),
        (1.7, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, channel, steps, fired)
    first = await adapter.send("Is Tuesday free?", None)
    assert first.text == f"{GREETING} {TUESDAY_REPLY}"  # the real reply is in turn 1
    assert first.latency_ms == pytest.approx(1600.0, abs=1.0)
    end = fired[0] + 1.7 + 0.8  # not at the greeting's stop + grace
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    _on_text_turn(mock, channel, _llm_steps(mock, "Ten o'clock is free."))
    second = await adapter.send("What time?", None)
    assert second.text == "Ten o'clock is free."  # turn 2 gets only its own reply (nothing shifted into it)
    assert second.latency_ms == pytest.approx(300.0, abs=1.0)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_greeting_read_entirely_during_start_is_added_to_the_first_reply_without_its_latency(
    impulse_project, mock_pipecat_channel, mode
):
    """Regression guard (passes before the finding 8 fix; DECISIONS.md: bot output read before the
    first turn is never dropped): a greeting run read completely during ``start()`` is added to the
    first reply and never sets its latency or ends it."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    greeting = [rtvi("bot-llm-started"), rtvi("bot-llm-text", {"text": GREETING}), rtvi("bot-llm-stopped")]
    channel = _before_bot_ready(mock, mode, *greeting)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    fired = []
    _on_text_turn(mock, channel, _llm_steps(mock, TUESDAY_REPLY), fired)  # started +0.2, text +0.3, stopped +0.4
    reply = await adapter.send("Is Tuesday free?", None)
    assert reply.text == f"{GREETING} {TUESDAY_REPLY}"
    assert reply.latency_ms == pytest.approx(300.0, abs=1.0)
    end = fired[0] + 0.4 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


# -- finding 8: greeting speech still playing after the user's last speech frame ----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("greeting_from", ["start", "user-speech"])
async def test_greeting_speech_playing_past_the_user_speech_sets_neither_ttfa_nor_the_end_of_turn(
    impulse_project, mock_pipecat_channel, mode, greeting_from
):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice=True)
    greeting_started = rtvi("bot-started-speaking")
    if greeting_from == "start":
        _before_bot_ready(mock, mode, greeting_started)  # read during start()
    else:
        _handshake(mock, mode)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    channel = mock.channel
    t0 = mock.clock.now  # the user's 0.5 s of speech starts now
    reply_tone = mock.tone(0.4, frequency_hz=660.0)
    steps = [
        (0.3, mock.BotAudio(mock.tone(0.3))),  # greeting audio while the user speaks
        (0.7, mock.BotAudio(mock.tone(0.2, frequency_hz=550.0))),  # greeting tail after the last user speech frame
        (0.75, rtvi("bot-output", _bot_output(GREETING, spoken=True, spoken_status="completed"))),
        (1.0, rtvi("bot-stopped-speaking")),  # the greeting ends; its grace window would end at +1.8
        (2.0, rtvi("bot-started-speaking")),  # the reply
        (2.1, mock.BotAudio(reply_tone)),
        (2.15, rtvi("bot-output", _bot_output(VOICE_REPLY, spoken=True, spoken_status="completed"))),
        (2.6, rtvi("bot-stopped-speaking")),
    ]
    if greeting_from == "user-speech":
        steps.insert(0, (0.2, greeting_started))  # read while the user's speech is being sent
    for offset, item in steps:
        channel.emit(item, at=t0 + offset)
    reply = await adapter.send("Is Tuesday free?", _user_wav(0.5))
    speech_end = 0.5 if mode == "eval" else 0.48  # eval: one paced send; websocket: the last 20 ms speech frame
    assert reply.latency_ms == pytest.approx((2.1 - speech_end) * 1000.0, abs=2.0)  # TTFA from the reply's audio
    assert _norm(reply.text).endswith(VOICE_REPLY)
    assert read_wav(reply.audio_bytes)[3].endswith(reply_tone)
    end = t0 + 2.6 + 0.8  # the reply's bot-stopped-speaking + grace
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


# -- finding 15: bot error texts are redacted before they are shortened -------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("source", F15_SECRET_SOURCES)
@pytest.mark.parametrize("kind,field", F15_ERROR_MESSAGES)
async def test_bot_error_text_is_redacted_before_it_is_shortened(
    impulse_project, mock_pipecat_channel, monkeypatch, mode, source, kind, field
):
    assert F15_LEAKY[:200].endswith(F15_KEY[:14])  # the 200-character cut falls inside the key
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, **_secret_settings(monkeypatch, source))
    adapter = await _started(cfg, mock, mode)
    steps = [(0.1, mock.rtvi(kind, {field: F15_LEAKY}))] + _llm_steps(mock, "Hello.")
    _on_text_turn(mock, mock.channel, steps)
    reply = await adapter.send("Hi", None)
    assert reply.text == "Hello."
    errors = [warning for warning in adapter.get_warnings() if warning.startswith(RTVI_ERROR_PREFIX)]
    assert len(errors) == 1, adapter.get_warnings()
    assert F15_KEY[:8] not in errors[0], errors[0]
    assert errors[0] == RTVI_ERROR_PREFIX + F15_LEAD + "[REDACTED]"
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_bot_error_with_a_key_only_the_runner_knows_is_redacted_before_it_is_shortened(
    impulse_project, llm_router, mock_pipecat_channel, monkeypatch, mode
):
    from impulse_evals.runner import run

    monkeypatch.setenv("ANTHROPIC_API_KEY", F15_KEY)  # a standard key variable: the adapter holds no copy of it
    mock = mock_pipecat_channel
    project = impulse_project
    cfg = _cfg(project, mode, scenarios=[project.scenario("leaky_bot_error", max_turns=1, metrics=["UNIV-013"])])
    _handshake(mock, mode)
    steps = [(0.1, mock.rtvi("error", {"message": F15_LEAKY}))] + _llm_steps(mock, "Hello.")
    _on_text_turn(mock, mock.channel, steps)
    _patch_pipecat_class(monkeypatch, mock)  # the runner's default factory builds the adapter
    result = await run(cfg)
    assert len(mock.channel.connect_calls) == 1  # the patched class (fake channel) was used
    assert result.scenario_results[0].error is None
    errors = [warning for warning in result.warnings if warning.startswith(RTVI_ERROR_PREFIX)]
    assert len(errors) == 1, result.warnings
    assert F15_KEY[:8] not in errors[0], errors[0]
    assert errors[0] == RTVI_ERROR_PREFIX + F15_LEAD + "[REDACTED]"
    assert F15_KEY[:8] not in result.model_dump_json()


# =============================================================================================
# AUDIT batch 2 review fix-ups (finding 8, Stage D review): bot output already QUEUED on the channel,
# unread, when the user's turn is sent (the runner spends time on the summary, the simulator LLM call
# and TTS between start() and send(); nothing reads the channel meanwhile), and a spoken greeting's
# bot-output in text mode. Before a user turn is sent (send-text, or user audio in any mode; never the
# outbound opener) the adapter reads every item the channel already holds into the pre-turn backlog.
# Text mode: the reply text is the pre-turn text first, then the reply part, which uses bot-output only
# when the reply itself produced bot-output, else its LLM runs. (A greeting whose LLM run has not
# started at all when Impulse sends stays a documented limitation: not tested.)
# =============================================================================================

GREETING_HEAD = "Hello, this is"
GREETING_TAIL = " Sunrise Clinic."  # GREETING_HEAD + GREETING_TAIL == GREETING
RUNNER_PREPARES_S = 1.0  # the runner's time between start() and the first send(): nothing reads the channel
VOICE_PACING_CASES = [
    pytest.param("eval", None, 0.5, id="eval"),  # Pipecat's client transport paces: speech ends +0.5
    pytest.param("websocket", True, 0.48, id="websocket-paced"),  # the last 20 ms speech frame at +0.48
    pytest.param("websocket", False, 0.0, id="websocket-unpaced"),  # every frame leaves at once
]


def _queued_after_bot_ready(mock, mode, steps):
    """``bot-ready`` (on connect in eval mode, on ``client-ready`` in websocket mode), then the fake bot's
    ``steps`` (``(offset_s, item)`` after the connection opened). ``agent.system_prompt`` set: ``start()``
    returns at ``bot-ready``, before any of them arrives; the test then advances the clock by
    ``RUNNER_PREPARES_S`` so they sit on the channel, unread, when the user's turn is sent."""
    # The whole queue falls due at once (clock.advance): the fake clock's driver, spinning while due
    # items are unread, could otherwise move virtual time on in the instant between the adapter taking
    # the last of them and the adapter resuming (a fake-clock artefact, not adapter behaviour).
    mock.clock.settle_rounds = 1000
    channel = _handshake(mock, mode)
    channel.on_connect(_plan(channel, steps))
    return channel


def _spoken_greeting(rtvi, kind):
    """A greeting the bot speaks: ``tts-speak`` (a ``TTSSpeakFrame`` greeting, spoken even with
    ``skip_tts``: no LLM events) or ``llm-spoken`` (an LLM greeting that a websocket text-mode bot also
    speaks, since its TTS runs)."""
    spoken = rtvi("bot-output", _bot_output(GREETING, spoken=True, spoken_status="completed"))
    if kind == "tts-speak":
        return [rtvi("bot-started-speaking"), spoken, rtvi("bot-stopped-speaking")]
    return [
        rtvi("bot-llm-started"),
        rtvi("bot-started-speaking"),
        rtvi("bot-llm-text", {"text": GREETING}),
        rtvi("bot-llm-stopped"),
        spoken,
        rtvi("bot-stopped-speaking"),
    ]


# -- review fix-up 1: a greeting queued on the channel, unread, when the user's turn is sent ----------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_greeting_run_queued_unread_at_the_first_text_turn_is_pre_turn_output(
    impulse_project, mock_pipecat_channel, mode
):
    """The greeting's whole LLM run arrived after ``bot-ready`` and sits on the channel, unread, when
    ``send-text`` goes out: it is added to turn 1 (first), never sets its latency, never ends it, and the
    real reply is not shifted to turn 2."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    greeting = [
        (0.3, rtvi("bot-llm-started")),
        (0.5, rtvi("bot-llm-text", {"text": GREETING})),
        (0.7, rtvi("bot-llm-stopped")),
    ]
    channel = _queued_after_bot_ready(mock, mode, greeting)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    assert channel.pending == len(greeting)  # start() returned before the greeting arrived
    mock.clock.advance(RUNNER_PREPARES_S)  # the whole greeting is now queued on the channel, unread
    fired = []
    # The real reply starts after the greeting's grace window would have closed (send + 0.8 s).
    _on_text_turn(mock, channel, _llm_steps(mock, TUESDAY_REPLY, start=1.5), fired)  # text +1.6, stopped +1.7
    first = await adapter.send("Is Tuesday free?", None)
    assert first.text == f"{GREETING} {TUESDAY_REPLY}"  # the real reply is in turn 1, after the greeting
    assert first.latency_ms == pytest.approx(1600.0, abs=1.0)  # send-text -> the reply's first text
    end = fired[0] + 1.7 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    _on_text_turn(mock, channel, _llm_steps(mock, "Ten o'clock is free."))
    second = await adapter.send("What time?", None)
    assert second.text == "Ten o'clock is free."  # nothing shifted into turn 2
    assert second.latency_ms == pytest.approx(300.0, abs=1.0)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("queued", ["started", "started-and-text"])
async def test_greeting_run_started_while_queued_and_finishing_after_send_text_sets_no_latency(
    impulse_project, mock_pipecat_channel, mode, queued
):
    """The greeting's ``bot-llm-started`` (and some of its text) is queued, unread, when ``send-text``
    goes out; the rest of the greeting arrives after it: a pre-turn run (its text kept, first; no
    latency; its stop never starts the grace window)."""
    assert GREETING_HEAD + GREETING_TAIL == GREETING
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    head = [(0.3, rtvi("bot-llm-started"))]
    later = [(0.1, rtvi("bot-llm-text", {"text": GREETING_TAIL})), (0.3, rtvi("bot-llm-stopped"))]
    if queued == "started-and-text":
        head.append((0.6, rtvi("bot-llm-text", {"text": GREETING_HEAD})))
    else:
        later.insert(0, (0.05, rtvi("bot-llm-text", {"text": GREETING_HEAD})))
    channel = _queued_after_bot_ready(mock, mode, head)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    assert channel.pending == len(head)
    mock.clock.advance(RUNNER_PREPARES_S)  # the greeting's start (and head) queued, unread
    fired = []
    # The greeting's stop at +0.3 would end the turn at +1.1; the real reply starts at +1.5.
    _on_text_turn(mock, channel, later + _llm_steps(mock, TUESDAY_REPLY, start=1.5), fired)
    first = await adapter.send("Is Tuesday free?", None)
    assert first.latency_ms == pytest.approx(1600.0, abs=1.0)  # from the real reply, never the greeting
    assert first.text == f"{GREETING} {TUESDAY_REPLY}"
    end = fired[0] + 1.7 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    _on_text_turn(mock, channel, _llm_steps(mock, "Ten o'clock is free."))
    second = await adapter.send("What time?", None)
    assert second.text == "Ten o'clock is free."
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,pacing,speech_end", VOICE_PACING_CASES)
async def test_greeting_speech_queued_unread_before_the_user_audio_sets_neither_ttfa_nor_the_end_of_turn(
    impulse_project, mock_pipecat_channel, mode, pacing, speech_end
):
    """A whole greeting speech (started, audio, output, stopped) is queued, unread, when the user's
    audio is sent. Unpaced websocket audio leaves at once, so nothing is read while it is sent; the eval
    and paced websocket cases read the queue while the speech is being sent (regression guards)."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    extra = {} if pacing is None else {"realtime_pacing": pacing}
    cfg = _cfg(impulse_project, mode, voice=True, **extra)
    greeting = [
        (0.2, rtvi("bot-started-speaking")),
        (0.3, mock.BotAudio(mock.tone(0.3))),
        (0.35, rtvi("bot-output", _bot_output(GREETING, spoken=True, spoken_status="completed"))),
        (0.7, rtvi("bot-stopped-speaking")),  # its grace window must never end turn 1
    ]
    channel = _queued_after_bot_ready(mock, mode, greeting)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    assert channel.pending == len(greeting)
    mock.clock.advance(RUNNER_PREPARES_S)  # simulator call + TTS: the whole greeting queued, unread
    t0 = mock.clock.now  # the user's 0.5 s of speech starts now
    reply_tone = mock.tone(0.4, frequency_hz=660.0)
    reply_steps = [  # the reply, after the user's last speech frame was sent
        (1.0, rtvi("bot-started-speaking")),
        (1.1, mock.BotAudio(reply_tone)),
        (1.15, rtvi("bot-output", _bot_output(VOICE_REPLY, spoken=True, spoken_status="completed"))),
        (1.6, rtvi("bot-stopped-speaking")),
    ]
    fired = []
    channel.on_audio(_plan(channel, reply_steps, fired), match=_after_speech(channel, 8000))  # 0.5 s at 16 kHz
    reply = await adapter.send("Is Tuesday free?", _user_wav(0.5))
    assert fired[0] == pytest.approx(t0 + speech_end, abs=1e-6)  # the last speech frame: the latency start
    assert reply.latency_ms == pytest.approx(1100.0, abs=2.0)  # TTFA from the reply's audio, not the greeting
    assert _norm(reply.text).endswith(VOICE_REPLY)
    assert read_wav(reply.audio_bytes)[3].endswith(reply_tone)
    end = fired[0] + 1.6 + 0.8  # the reply's bot-stopped-speaking + grace, not the greeting's
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


# -- review fix-up 2: a spoken greeting's bot-output never replaces a text reply made of LLM text -----


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("where", ["during-start", "queued"])
@pytest.mark.parametrize("kind", ["tts-speak", "llm-spoken"])
async def test_spoken_greeting_bot_output_never_replaces_a_text_reply_made_of_llm_text(
    impulse_project, mock_pipecat_channel, mode, where, kind
):
    """Text mode: the pre-turn greeting produced ``bot-output``, the reply (``skip_tts`` /
    ``audio_response: false``) only ``bot-llm-text``: the reply text is the greeting, then the reply."""
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    greeting = _spoken_greeting(mock.rtvi, kind)
    if where == "during-start":
        channel = _before_bot_ready(mock, mode, *greeting)  # read during start()
    else:
        steps = [(0.2 + 0.05 * index, item) for index, item in enumerate(greeting)]
        channel = _queued_after_bot_ready(mock, mode, steps)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    if where == "queued":
        assert channel.pending == len(greeting)
        mock.clock.advance(RUNNER_PREPARES_S)  # queued on the channel, unread, when send-text goes out
    _on_text_turn(mock, channel, _llm_steps(mock, TUESDAY_REPLY))  # bot-llm-text only: started +0.2, text +0.3
    reply = await adapter.send("Is Tuesday free?", None)
    assert reply.text.endswith(TUESDAY_REPLY), reply.text  # the real reply is never lost
    assert reply.text.startswith(GREETING), reply.text  # the pre-turn greeting comes first
    assert reply.text == f"{GREETING} {TUESDAY_REPLY}"  # and only once
    assert reply.latency_ms == pytest.approx(300.0, abs=1.0)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["tts-speak", "llm-spoken"])
async def test_text_reply_with_its_own_bot_output_still_uses_its_bot_output_after_a_spoken_greeting(
    impulse_project, mock_pipecat_channel, mode, kind
):
    """Regression guard (passes before the fix): a text-mode reply that produces ``bot-output`` itself
    is still taken from its ``bot-output``, not its LLM tokens; the pre-turn greeting comes first."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    channel = _before_bot_ready(mock, mode, *_spoken_greeting(rtvi, kind))
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.25, rtvi("bot-llm-text", {"text": "tokens that are not used"})),
        (0.4, rtvi("bot-output", _bot_output(TUESDAY_REPLY))),
        (0.6, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, channel, steps)
    reply = await adapter.send("Is Tuesday free?", None)
    assert _norm(reply.text) == f"{GREETING} {TUESDAY_REPLY}"
    assert "tokens that are not used" not in reply.text
    assert reply.latency_ms == pytest.approx(250.0, abs=1.0)  # the reply's first text (bot-llm-text)
    await adapter.close()


# =============================================================================================
# AUDIT batch 2 second review fix-ups (finding 8, re-check): a voice bot that does not let the user
# interrupt its greeting queues the reply's audio right behind the greeting's, so Pipecat sends ONE
# bot-started-speaking (the greeting's, read before the latency start: during start(), queued unread
# when the user's audio is sent, or while it is sent) and ONE bot-stopped-speaking (after the reply),
# with the reply's own LLM run (bot-llm-started / bot-llm-text after the user's last speech frame) and
# its audio in between, no stop/start. The stop of a pre-turn speech ends the turn (starts the grace
# window) once the reply's own LLM run produced text during that speech. A pre-turn speech whose stop
# comes before any reply LLM text still never ends the turn (the finding 8 rule; a pre-turn greeting
# run's own text never counts), and a separate reply speech still ends at its own stop.
# =============================================================================================

SECOND_REPLY = "Ten o'clock is free."
MERGED_SPEECH_CASES = [
    pytest.param(mode, pacing, speech_end, where, id=f"{case}-{where}")
    for mode, pacing, speech_end, case in (
        ("eval", None, 0.5, "eval"),  # Pipecat's client transport paces: speech ends +0.5
        ("websocket", True, 0.48, "websocket-paced"),  # the last 20 ms speech frame at +0.48
        ("websocket", False, 0.0, "websocket-unpaced"),  # every frame leaves at once
    )
    for where in ("start", "queued", "user-speech")
    if not (pacing is False and where == "user-speech")  # unpaced audio: nothing is read while it is sent
]


async def _greeting_speaking_before_the_user_turn(project, mock, mode, pacing, where, *, greeting_run=False):
    """A started voice adapter whose greeting speech began before the user's turn: its
    ``bot-started-speaking`` (with ``greeting_run``, after the greeting's own ``bot-llm-started``) is
    read during ``start()``, queued unread when the user's audio is sent (with 0.3 s of greeting
    audio), or read while the user's audio is sent (+0.2; the greeting's run +0.1). Returns
    ``(adapter, channel, t0)``: the user's 0.5 s of speech starts at ``t0``."""
    rtvi = mock.rtvi
    extra = {} if pacing is None else {"realtime_pacing": pacing}
    cfg = _cfg(project, mode, voice=True, **extra)
    opening = ([rtvi("bot-llm-started")] if greeting_run else []) + [rtvi("bot-started-speaking")]
    queued = []
    if where == "start":
        channel = _before_bot_ready(mock, mode, *opening)
    elif where == "queued":
        queued = [(0.2 + 0.05 * index, item) for index, item in enumerate(opening)]
        queued.append((0.3, mock.BotAudio(mock.tone(0.3))))
        channel = _queued_after_bot_ready(mock, mode, queued)
    else:
        channel = _handshake(mock, mode)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    if where == "queued":
        assert channel.pending == len(queued)  # start() returned before the greeting arrived
        mock.clock.advance(RUNNER_PREPARES_S)  # the greeting's start sits on the channel, unread
    t0 = mock.clock.now
    if where == "user-speech":
        for index, item in enumerate(reversed(opening)):
            channel.emit(item, at=t0 + 0.2 - 0.1 * index)  # read while the user's speech is being sent
    return adapter, channel, t0


def _greeting_speech_tail(mock):
    """The rest of the greeting speech, relative to the start of the user's speech: audio while the
    user speaks (+0.3), its tail after the user's last speech frame (+0.7), its spoken text (+0.75)."""
    return [
        (0.3, mock.BotAudio(mock.tone(0.3))),
        (0.7, mock.BotAudio(mock.tone(0.2, frequency_hz=550.0))),
        (0.75, mock.rtvi("bot-output", _bot_output(GREETING, spoken=True, spoken_status="completed"))),
    ]


def _assert_turn_ended(mock, t0, stop_offset):
    end = t0 + stop_offset + 0.8  # default end_of_turn_grace_ms 800
    assert end - 1e-9 <= mock.clock.now <= end + 0.25, (
        f"the turn ended {mock.clock.now - t0:.3f} s after the user's speech started; expected {stop_offset + 0.8:.3f} s"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,pacing,speech_end,where", MERGED_SPEECH_CASES)
async def test_reply_spoken_right_behind_a_pre_turn_greeting_ends_at_the_single_stop(
    impulse_project, mock_pipecat_channel, mode, pacing, speech_end, where
):
    """One speech holds the greeting and the reply (no stop/start between them); the reply's own LLM
    run produced its text during it. The turn ends at that speech's stop + grace (not in the 30 s turn
    timeout), the reply text is the greeting then the reply (each once), and turn 2 is unaffected."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    adapter, channel, t0 = await _greeting_speaking_before_the_user_turn(impulse_project, mock, mode, pacing, where)
    reply_tone = mock.tone(0.3, frequency_hz=660.0)
    steps = _greeting_speech_tail(mock) + [
        (0.9, rtvi("bot-llm-started")),  # the reply's own LLM run, after the user's last speech frame
        (1.0, rtvi("bot-llm-text", {"text": VOICE_REPLY})),
        (1.1, rtvi("bot-llm-stopped")),
        (1.3, mock.BotAudio(reply_tone)),  # the reply's audio, right behind the greeting's
        (1.4, rtvi("bot-output", _bot_output(VOICE_REPLY, spoken=True, spoken_status="completed"))),
        (1.8, rtvi("bot-stopped-speaking")),  # the only stop: greeting and reply
    ]
    for offset, item in steps:
        channel.emit(item, at=t0 + offset)
    reply = await adapter.send("Is Tuesday free?", _user_wav(0.5))
    _assert_turn_ended(mock, t0, 1.8)
    assert _norm(reply.text) == f"{GREETING} {VOICE_REPLY}"  # the greeting first, and only once
    assert read_wav(reply.audio_bytes)[3].endswith(reply_tone)
    # The latency is the reply's own (its first text +1.0 or its first audio +1.3), never the greeting's.
    assert (1.0 - speech_end) * 1000.0 - 2.0 <= reply.latency_ms <= (1.3 - speech_end) * 1000.0 + 2.0, reply.latency_ms
    t1 = mock.clock.now  # turn 2: a speech of its own, nothing of turn 1 left over
    for offset, item in [
        (1.0, rtvi("bot-started-speaking")),
        (1.1, mock.BotAudio(mock.tone(0.3, frequency_hz=770.0))),
        (1.15, rtvi("bot-output", _bot_output(SECOND_REPLY, spoken=True, spoken_status="completed"))),
        (1.6, rtvi("bot-stopped-speaking")),
    ]:
        channel.emit(item, at=t1 + offset)
    second = await adapter.send("What time?", _user_wav(0.5))
    assert _norm(second.text) == SECOND_REPLY
    assert second.latency_ms == pytest.approx((1.1 - speech_end) * 1000.0, abs=2.0)
    _assert_turn_ended(mock, t1, 1.6)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,pacing,speech_end,where", MERGED_SPEECH_CASES)
@pytest.mark.parametrize("llm", ["greeting-run-text", "reply-run-started"])
async def test_pre_turn_greeting_stop_before_any_reply_llm_text_still_never_ends_the_turn(
    impulse_project, mock_pipecat_channel, mode, pacing, speech_end, where, llm
):
    """Regression guard (passes before the fix; the finding 8 rule): the greeting speech stops (+1.0)
    before the reply's LLM run produced any text, so its stop never ends the turn (its grace window
    would end at +1.8), also when the greeting's own pre-turn LLM run produced its text during the
    speech after the user's last speech frame (``greeting-run-text``), or when the reply's LLM run
    started during the speech but produced its text after the stop (``reply-run-started``)."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    greeting_run = llm == "greeting-run-text"
    adapter, channel, t0 = await _greeting_speaking_before_the_user_turn(
        impulse_project, mock, mode, pacing, where, greeting_run=greeting_run
    )
    steps = _greeting_speech_tail(mock)
    if greeting_run:
        steps += [(0.6, rtvi("bot-llm-text", {"text": GREETING})), (0.65, rtvi("bot-llm-stopped"))]
        reply_run = _llm_steps(mock, VOICE_REPLY, start=1.4)  # started +1.4, text +1.5, stopped +1.6
    else:
        reply_run = [
            (0.9, rtvi("bot-llm-started")),
            (1.2, rtvi("bot-llm-text", {"text": VOICE_REPLY})),
            (1.3, rtvi("bot-llm-stopped")),
        ]
    reply_tone = mock.tone(0.4, frequency_hz=660.0)
    steps += [(1.0, rtvi("bot-stopped-speaking"))] + reply_run + [
        (2.0, rtvi("bot-started-speaking")),  # the reply, a speech of its own
        (2.1, mock.BotAudio(reply_tone)),
        (2.15, rtvi("bot-output", _bot_output(VOICE_REPLY, spoken=True, spoken_status="completed"))),
        (2.6, rtvi("bot-stopped-speaking")),
    ]
    for offset, item in steps:
        channel.emit(item, at=t0 + offset)
    reply = await adapter.send("Is Tuesday free?", _user_wav(0.5))
    _assert_turn_ended(mock, t0, 2.6)
    assert _norm(reply.text) == f"{GREETING} {VOICE_REPLY}"
    assert reply.latency_ms == pytest.approx((2.1 - speech_end) * 1000.0, abs=2.0)  # TTFA from the reply's audio
    assert read_wav(reply.audio_bytes)[3].endswith(reply_tone)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,pacing,speech_end,where", MERGED_SPEECH_CASES)
async def test_separate_reply_speech_after_a_pre_turn_greeting_still_ends_at_its_own_stop(
    impulse_project, mock_pipecat_channel, mode, pacing, speech_end, where
):
    """Regression guard (passes before the fix): the reply's LLM run produced its text while the
    greeting still played, then the greeting stopped (+1.2) and the reply was spoken as a speech of its
    own (+1.5, inside the greeting stop's grace window): the turn ends at the reply's stop + grace."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    adapter, channel, t0 = await _greeting_speaking_before_the_user_turn(impulse_project, mock, mode, pacing, where)
    reply_tone = mock.tone(0.4, frequency_hz=660.0)
    steps = _greeting_speech_tail(mock) + [
        (0.9, rtvi("bot-llm-started")),
        (1.0, rtvi("bot-llm-text", {"text": VOICE_REPLY})),
        (1.1, rtvi("bot-llm-stopped")),
        (1.2, rtvi("bot-stopped-speaking")),  # the greeting stops; a window from here would end at +2.0
        (1.5, rtvi("bot-started-speaking")),  # the reply, a speech of its own
        (1.6, mock.BotAudio(reply_tone)),
        (1.65, rtvi("bot-output", _bot_output(VOICE_REPLY, spoken=True, spoken_status="completed"))),
        (2.4, rtvi("bot-stopped-speaking")),
    ]
    for offset, item in steps:
        channel.emit(item, at=t0 + offset)
    reply = await adapter.send("Is Tuesday free?", _user_wav(0.5))
    _assert_turn_ended(mock, t0, 2.4)
    assert _norm(reply.text) == f"{GREETING} {VOICE_REPLY}"
    assert reply.latency_ms == pytest.approx((1.6 - speech_end) * 1000.0, abs=2.0)  # TTFA from the reply's audio
    assert read_wav(reply.audio_bytes)[3].endswith(reply_tone)
    await adapter.close()


# =============================================================================================
# AUDIT batch 2 re-check round 3 (finding 8): the reply's own LLM text arrives while a pre-turn greeting
# still plays, the greeting stops (+1.2) and slow TTS starts the reply's own speech 1.0 s or 1.5 s
# later, longer than the 800 ms grace window. A pre-turn speech's stop ends the turn only when a SPOKEN
# bot-output of the reply (spoken: true / spoken_status completed, read at or after the latency start,
# after the reply's own LLM text) was read during that pre-turn speech: the merged case above. Here
# none was (or only an unspoken bot-output of the reply text), so the turn waits for the reply's own
# speech and ends at its stop + grace, and turn 2 gets only its own reply.
# =============================================================================================

SLOW_TTS_GAPS = [pytest.param(1.0, id="gap-1.0s"), pytest.param(1.5, id="gap-1.5s")]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,pacing,speech_end,where", MERGED_SPEECH_CASES)
@pytest.mark.parametrize("gap", SLOW_TTS_GAPS)
@pytest.mark.parametrize("reply_output", ["none", "unspoken"])
async def test_reply_text_during_a_pre_turn_greeting_waits_for_the_replys_own_slow_speech(
    impulse_project, mock_pipecat_channel, mode, pacing, speech_end, where, gap, reply_output
):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    adapter, channel, t0 = await _greeting_speaking_before_the_user_turn(impulse_project, mock, mode, pacing, where)
    reply_tone = mock.tone(0.4, frequency_hz=660.0)
    start = 1.2 + gap  # slow TTS: the reply's own speech starts after the greeting stop's grace window
    steps = _greeting_speech_tail(mock) + [
        (0.9, rtvi("bot-llm-started")),  # the reply's own LLM run, while the greeting still plays
        (1.0, rtvi("bot-llm-text", {"text": VOICE_REPLY})),
        (1.1, rtvi("bot-llm-stopped")),
        (1.2, rtvi("bot-stopped-speaking")),  # the greeting stops; a window from here would end at +2.0
        (start, rtvi("bot-started-speaking")),
        (start + 0.1, mock.BotAudio(reply_tone)),
        (start + 0.15, rtvi("bot-output", _bot_output(VOICE_REPLY, spoken=True, spoken_status="completed"))),
        (start + 0.6, rtvi("bot-stopped-speaking")),
    ]
    if reply_output == "unspoken":  # the reply's text as an unspoken bot-output during the greeting
        steps.append((1.05, rtvi("bot-output", _bot_output(VOICE_REPLY, spoken=False))))
    for offset, item in steps:
        channel.emit(item, at=t0 + offset)
    reply = await adapter.send("Is Tuesday free?", _user_wav(0.5))
    _assert_turn_ended(mock, t0, start + 0.6)  # the reply's own stop + grace, not the greeting's
    assert _norm(reply.text) == f"{GREETING} {VOICE_REPLY}"
    assert reply.audio_bytes is not None and read_wav(reply.audio_bytes)[3].endswith(reply_tone)
    assert reply.latency_ms == pytest.approx((start + 0.1 - speech_end) * 1000.0, abs=2.0)  # TTFA: the reply's audio
    t1 = mock.clock.now  # turn 2: a speech of its own, nothing of turn 1 shifted into it
    for offset, item in [
        (1.0, rtvi("bot-started-speaking")),
        (1.1, mock.BotAudio(mock.tone(0.3, frequency_hz=770.0))),
        (1.15, rtvi("bot-output", _bot_output(SECOND_REPLY, spoken=True, spoken_status="completed"))),
        (1.6, rtvi("bot-stopped-speaking")),
    ]:
        channel.emit(item, at=t1 + offset)
    second = await adapter.send("What time?", _user_wav(0.5))
    assert _norm(second.text) == SECOND_REPLY
    assert second.latency_ms == pytest.approx((1.1 - speech_end) * 1000.0, abs=2.0)
    _assert_turn_ended(mock, t1, 1.6)
    await adapter.close()


# =============================================================================================
# AUDIT batch 3 (finding 26): a function-call stop read in a LATER turn (an ``llm-function-call-
# stopped`` repeated, or arriving late, for a call reported in an earlier turn) never changes an
# ``AgentReply`` already returned: ``send()`` returns copies of the turn's tool calls. The late stop
# records nothing in the later turn either (recording a late result on the current turn is left to
# the maintainer).
# =============================================================================================

LATE_RESULT = {"success": False, "error": "slot taken"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_late_function_call_stop_does_not_change_a_returned_text_reply(impulse_project, mock_pipecat_channel, mode):
    from impulse_evals.models import ToolCall

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    first = [
        (0.1, rtvi("bot-llm-started")),
        (0.15, _in_progress(mock, "c1", "book_slot", BOOK_ARGS)),
        (0.2, rtvi("bot-llm-stopped")),
        (0.3, _stopped(mock, "c1", "book_slot")),  # the stop reports no result
        (0.4, rtvi("bot-llm-started")),
        (0.5, rtvi("bot-llm-text", {"text": "Booking..."})),
        (0.6, rtvi("bot-llm-stopped")),
    ]
    _on_text_turn(mock, mock.channel, first)
    reply1 = await adapter.send("Book me for Tuesday", None)
    returned = [ToolCall(name="book_slot", arguments=BOOK_ARGS)]
    assert reply1.tool_calls == returned
    # Turn 2: a stop for c1 again, now with a failed result, then the bot's answer.
    second = [(0.05, _stopped(mock, "c1", "book_slot", result=LATE_RESULT))] + _llm_steps(mock, "Sorry, that slot is taken.")
    _on_text_turn(mock, mock.channel, second)
    reply2 = await adapter.send("Is it booked?", None)
    assert reply2.text == "Sorry, that slot is taken."
    assert reply1.tool_calls == returned, reply1.tool_calls   # the reply already returned is unchanged
    assert (reply1.tool_calls[0].success, reply1.tool_calls[0].result, reply1.tool_calls[0].error) == (None, None, None)
    assert reply2.tool_calls == []   # the late stop records nothing in turn 2
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_late_function_call_stop_does_not_change_a_returned_voice_reply(impulse_project, mock_pipecat_channel, mode):
    from impulse_evals.models import ToolCall

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    first, second = mock.tone(0.3), mock.tone(0.4, frequency_hz=550.0)
    steps = _first_speech(mock, first) + [
        (0.15, _in_progress(mock, "c1", "book_slot", BOOK_ARGS)),
        (0.4, _stopped(mock, "c1", "book_slot")),  # the stop reports no result
        (0.6, rtvi("bot-stopped-speaking")),
    ]
    adapter, reply1, channel = await _voice_turn(impulse_project, mock, mode, steps)
    returned = [ToolCall(name="book_slot", arguments=BOOK_ARGS)]
    assert reply1.tool_calls == returned
    assert _norm(reply1.text) == "Let me check."
    t1 = mock.clock.now  # turn 2: the user's 0.5 s of speech starts now; the bot answers after it
    for offset, item in [(0.7, _stopped(mock, "c1", "book_slot", result=LATE_RESULT))] + _second_speech(mock, second, at=0.8):
        channel.emit(item, at=t1 + offset)
    reply2 = await adapter.send("Is it booked?", _user_wav(0.5))
    assert _norm(reply2.text) == "Booked for Tuesday."
    assert reply1.tool_calls == returned, reply1.tool_calls   # the reply already returned is unchanged
    assert (reply1.tool_calls[0].success, reply1.tool_calls[0].result, reply1.tool_calls[0].error) == (None, None, None)
    assert reply2.tool_calls == []   # the late stop records nothing in turn 2
    await adapter.close()


# =============================================================================================
# AUDIT batch 3 (finding 27): a voice user turn's reply WAV holds the bot audio from the latency
# start (the user's last speech frame) on, so bot audio that arrived while the user's speech was
# still being sent (a greeting tail, a barge-in overlap) is not in ``AgentReply.audio_bytes``, the
# audio Impulse's STT transcribes for VOICE-009/013. TTFA, the reply text rule (an early spoken
# bot-output is still added to the text) and ``interrupted`` are unchanged; the outbound opener
# still collects every bot audio since the connection opened.
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_reply_audio_excludes_bot_audio_heard_during_user_speech(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice=True)  # eval: paced by the fake; websocket: realtime pacing
    rate = cfg.agent.pipecat.sample_rate
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    early = mock.tone(0.1, frequency_hz=330.0)  # the bot still talking while the user speaks
    reply_tone = mock.tone(0.3)
    t0 = mock.clock.now  # the user's 0.5 s of speech starts now
    channel.emit(rtvi("bot-started-speaking"), at=t0 + 0.15)
    channel.emit(mock.BotAudio(early), at=t0 + 0.2)
    channel.emit(rtvi("bot-output", _bot_output("...and welcome.", spoken=True)), at=t0 + 0.22)
    channel.emit(rtvi("user-started-speaking"), at=t0 + 0.25)
    channel.emit(rtvi("bot-stopped-speaking"), at=t0 + 0.3)
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.2, mock.BotAudio(reply_tone)),
        (0.25, rtvi("bot-output", _bot_output("Your slot is booked.", spoken=True))),
        (0.6, rtvi("bot-stopped-speaking")),
    ]
    channel.on_audio(_plan(channel, steps), match=_after_speech(channel, int(0.5 * rate)))
    reply = await adapter.send("Book a slot for Tuesday", _user_wav(0.5))
    speech_end = max(entry["returned"] for entry in channel.sent_audio if entry["has_speech"])
    assert t0 + 0.2 < speech_end  # the early audio arrived before the user's last speech frame was sent
    frames = read_wav(reply.audio_bytes)[3]
    assert len(frames) == len(reply_tone), (len(frames), len(early), len(reply_tone))
    assert frames == reply_tone  # only the bot audio from the latency start on
    assert reply.latency_ms == pytest.approx(200.0, abs=1.0)  # TTFA unchanged
    assert reply.text == "...and welcome. Your slot is booked."  # the text rule is unchanged
    assert reply.interrupted is True
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_outbound_voice_opener_still_collects_bot_audio_read_during_start(
    impulse_project, mock_pipecat_channel, mode
):
    """Regression guard (passes before the fix): the outbound opener has no user speech, so the bot
    audio read during ``start()`` (before the opener's latency start) is still in its reply WAV."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(
        impulse_project, mode, voice=True, scenarios=[impulse_project.scenario("outbound_probe", direction="outbound")]
    )
    head, tail = mock.tone(0.2), mock.tone(0.3, frequency_hz=550.0)
    channel = _before_bot_ready(mock, mode, rtvi("bot-started-speaking"), mock.BotAudio(head))  # read during start()
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    channel.emit(mock.BotAudio(tail), after=0.2)
    channel.emit(rtvi("bot-output", _bot_output(GREETING, spoken=True)), after=0.25)
    channel.emit(rtvi("bot-stopped-speaking"), after=0.6)
    reply = await adapter.send(None, None)
    assert reply.text == GREETING
    assert read_wav(reply.audio_bytes)[3] == head + tail
    await adapter.close()
