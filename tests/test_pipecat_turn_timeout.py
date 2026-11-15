# SPDX-License-Identifier: Apache-2.0
"""Fixes round 3, A1 (DEFECT-8): the Pipecat adapter keeps a timed-out turn; ``agent.pipecat.turn_timeout_s``
(FIXES-ROUND-3-SPEC.md section A1).

* config: ``PipecatConfig.turn_timeout_s: Optional[float] = None`` (config.yaml
  ``agent.pipecat.turn_timeout_s``) with exactly the rules of ``agent.livekit.turn_timeout_s`` (SPEC.md
  section 8.2): a boolean fails at the field ("turn_timeout_s must be a number of seconds, not a
  boolean"); ``0``, a negative value or NaN is the config error "turn_timeout_s {value:g} must be greater
  than 0"; infinity is "turn_timeout_s must be a finite number of seconds". Checked only when
  ``adapter: pipecat``; NOT a websocket-only key (no eval-mode warning).
* adapter: turn timeout = ``pipecat.turn_timeout_s`` when set (text and voice mode), else
  ``voice_config.timeout_s`` when ``voice_config`` exists, else 30 s. Error text unchanged. The timeout
  raises ``PipecatTurnTimeout`` (in ``__all__``), both a ``PipecatError`` and an ``AgentTurnError``, whose
  ``reply`` is the turn so far when it has content (non-blank text, a tool call, or audio), else None.
  Voice-mode text: the spoken ``bot-output`` texts; when none, every ``bot-output`` text received for the
  turn (announced or in progress, latest text per segment); when none, the turn's ``bot-llm-text``
  tokens. ``latency_ms`` measured as for a normal reply, else NaN (never the timeout instant).
* runner: the partial turn is recorded (fixes round 2 A6, no runner change).

Campaign case: run 20261025T141732Z-927a72 (BankBot, Pipecat ``-t eval``, voice mode)
``guaranteed_loan``: after user turn 9 the bot started speaking a raw function call text and never sent
``bot-stopped-speaking``; the turn timed out after 30 s and run.json lost agent turn 10. Trimmed data:
``tests/data/fixes_round3/pipecat_guaranteed_loan_927a72.json``.

Adapter tests run on ``mock_pipecat_channel`` (fake channel + virtual clock, no network, no pipecat-ai);
runner tests use ``llm_router`` (mocked judge and simulator) and, in voice mode, ``mock_voice_server``.
New behaviour tests catch ``PipecatError`` (which exists today) and then assert the new behaviour, so
they fail on an assertion, not on a missing name.
"""

import importlib
import json
import math
from pathlib import Path

import pytest
from conftest import pcm_tone, read_wav, wav_bytes

pytestmark = pytest.mark.phase11

URL = "ws://localhost:7860"
LK_URL = "ws://localhost:7880"
MODES = ["eval", "websocket"]
BOT_READY = {"version": "2.1.0", "about": {"library": "pipecat-ai", "library_version": "1.12.0"}}
TURN_TIMEOUT = "Pipecat bot did not finish its turn within {n} s."
BOT_CLOSED = "the Pipecat bot closed the connection (close code {code})"
VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}
BOOLEAN_MESSAGE = "turn_timeout_s must be a number of seconds, not a boolean"
INFINITE_MESSAGE = "turn_timeout_s must be a finite number of seconds"

DATA = Path(__file__).resolve().parent / "data" / "fixes_round3"
CAMPAIGN = json.loads((DATA / "pipecat_guaranteed_loan_927a72.json").read_text(encoding="utf-8"))
SCENARIO = CAMPAIGN["scenario_name"]                     # guaranteed_loan
CAMPAIGN_TURNS = CAMPAIGN["turns"]                       # turns 1-9 as stored in run.json
USER_TURN_9 = CAMPAIGN_TURNS[8]["text"]
# The text the bot spoke in turn 10 (from the bot log of the campaign; FIXES-ROUND-3-SPEC.md A1 Stage B).
RAW_CALL_TEXT = (
    "SKIPPED verify_identity(customer_id=CUST-2001, dob=1990-04-12)"
    "verifier:default_api:verify_identity{customer_id:CUST-2001,dob:1990-04-12}"
)
VERIFY_ARGS = {"customer_id": "CUST-2001", "dob": "1990-04-12"}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _cfg(project, mode="eval", *, voice=None, scenarios=None, adapter="pipecat", **pipecat):
    """``voice``: None (text mode), True (VOICE_CONFIG) or a dict of VOICE_CONFIG overrides."""
    block = {"mode": mode, "url": URL}
    block.update(pipecat)
    kwargs = {}
    if voice is not None:
        kwargs["voice_config"] = dict(VOICE_CONFIG, **(voice if isinstance(voice, dict) else {}))
    agent = {"adapter": adapter, "pipecat": block}
    project.write_config(scenarios or [project.scenario(SCENARIO)], agent=agent, **kwargs)
    return project.load()


def _raw_config(project, adapter, raw):
    """Write config.yaml with ``turn_timeout_s: <raw>`` as raw YAML text in the adapter's block."""
    placeholder = "__TURN_TIMEOUT_S__"
    if adapter == "pipecat":
        agent = {"adapter": "pipecat", "pipecat": {"mode": "eval", "url": URL, "turn_timeout_s": placeholder}}
    else:
        agent = {"adapter": "livekit", "livekit": {"url": LK_URL, "turn_timeout_s": placeholder}}
    path = project.write_config([project.scenario(SCENARIO)], agent=agent)
    text = path.read_text(encoding="utf-8")
    assert text.count(f"turn_timeout_s: {placeholder}") == 1
    path.write_text(text.replace(f"turn_timeout_s: {placeholder}", f"turn_timeout_s: {raw}"), encoding="utf-8")
    return path


def _module():
    return importlib.import_module("impulse_evals.adapters.pipecat_adapter")


def _base():
    from impulse_evals.adapters import base

    return base


def _adapter(cfg, mock):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    return PipecatAdapter(cfg, channel_factory=mock.factory, clock=mock.clock)


def _handshake(mock, mode, channel=None):
    channel = channel or mock.channel
    if mode == "eval":
        channel.paced_audio = True  # Pipecat's eval client transport paces user audio itself
        channel.on_connect(mock.rtvi("bot-ready", BOT_READY, id="client-ready"))
    else:
        channel.on_send("client-ready", lambda message: mock.rtvi("bot-ready", BOT_READY, id=message.get("id")))
    return channel


async def _started(cfg, mock, mode):
    _handshake(mock, mode)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    return adapter


def _plan(channel, steps, fired=None):
    def trigger(_record):
        if fired is not None:
            fired.append(channel.clock.monotonic())
        for offset, item in steps:
            channel.emit(item, after=offset)

    return trigger


def _after_speech(channel, samples):
    def predicate(_record):
        total = sum(len(entry["pcm"]) // 2 for entry in channel.sent_audio)
        return total >= samples * 0.98

    return predicate


def _user_wav(seconds=0.5):
    return wav_bytes(pcm_tone(seconds, 16000))


def _on_text_turn(channel, steps, fired=None):
    channel.on_send("send-text", _plan(channel, steps, fired))


def _on_voice_turn(channel, steps, fired=None, seconds=0.5):
    channel.on_audio(_plan(channel, steps, fired), match=_after_speech(channel, int(seconds * 16000)))


def _norm(text):
    return " ".join((text or "").split())


def _spoken(text, segment_id):
    """pipecat-ai 1.12.0: the bot-output of a completed (spoken) segment."""
    return {"text": text, "spoken": True, "will_be_spoken": True, "aggregated_by": "sentence",
            "segment_id": segment_id, "spoken_status": "completed"}


def _announced(text, segment_id):
    """pipecat-ai 1.12.0: the bot-output announced before synthesis (not spoken yet)."""
    return {"text": text, "spoken": False, "will_be_spoken": True, "aggregated_by": "sentence",
            "segment_id": segment_id, "spoken_status": "new"}


def _in_progress_output(text, segment_id):
    """pipecat-ai 1.12.0 ``_send_tts_progress`` (word-timestamp TTS): no ``spoken`` field."""
    return {"text": text, "will_be_spoken": True, "aggregated_by": "sentence", "segment_id": segment_id,
            "spoken_status": "in-progress"}


def _defect8_steps(mock, outputs, audio):
    """Turn 10 of the campaign: the bot starts speaking, its audio and the bot-output(s) of the raw
    function call text arrive, and ``bot-stopped-speaking`` never comes."""
    rtvi = mock.rtvi
    steps = [(0.1, rtvi("bot-started-speaking")), (0.2, mock.BotAudio(audio))]
    t = 0.25
    for data in outputs:
        steps.append((t, rtvi("bot-output", data)))
        t += 0.05
    return steps


async def _send_expecting_pipecat_error(adapter, text, audio):
    from impulse_evals.adapters.pipecat_adapter import PipecatError

    with pytest.raises(PipecatError) as info:
        await adapter.send(text, audio)
    return info.value


def _assert_turn_timeout(error, seconds):
    """The error is the new turn timeout: text unchanged, a ``PipecatTurnTimeout`` that is both a
    ``PipecatError`` and an ``AgentTurnError``."""
    from impulse_evals.adapters.pipecat_adapter import PipecatError

    assert str(error) == TURN_TIMEOUT.format(n=seconds)
    assert isinstance(error, PipecatError)
    assert isinstance(error, _base().AgentTurnError), (
        f"the Pipecat turn timeout ({type(error).__name__}) must also be an AgentTurnError carrying the turn so far"
    )
    timeout_cls = getattr(_module(), "PipecatTurnTimeout", None)
    assert timeout_cls is not None, "impulse_evals.adapters.pipecat_adapter.PipecatTurnTimeout is missing"
    assert isinstance(error, timeout_cls)


def _in_progress_call(mock, call_id, name, arguments):
    return mock.rtvi("llm-function-call-in-progress",
                     {"tool_call_id": call_id, "function_name": name, "arguments": arguments})


# =============================================================================================
# config.yaml: agent.pipecat.turn_timeout_s (the agent.livekit.turn_timeout_s rules)
# =============================================================================================


def test_pipecat_turn_timeout_s_is_accepted_and_defaults_to_none(impulse_project):
    cfg = _cfg(impulse_project, turn_timeout_s=90)
    assert cfg.agent.pipecat.turn_timeout_s == 90
    cfg = _cfg(impulse_project, turn_timeout_s=12.5)
    assert cfg.agent.pipecat.turn_timeout_s == 12.5
    cfg = _cfg(impulse_project)
    assert cfg.agent.pipecat.turn_timeout_s is None


@pytest.mark.parametrize("value,message", [
    (0, "turn_timeout_s 0 must be greater than 0"),
    (-5, "turn_timeout_s -5 must be greater than 0"),
    (-0.25, "turn_timeout_s -0.25 must be greater than 0"),
])
def test_pipecat_turn_timeout_s_must_be_greater_than_zero(impulse_project, value, message):
    from impulse_evals.config import ConfigError

    with pytest.raises(ConfigError) as info:
        _cfg(impulse_project, turn_timeout_s=value)
    assert info.value.field == "agent.pipecat.turn_timeout_s"
    assert info.value.message == message


def test_pipecat_turn_timeout_s_must_be_a_number(impulse_project):
    from impulse_evals.config import ConfigError

    with pytest.raises(ConfigError) as info:
        _cfg(impulse_project, turn_timeout_s="soon")
    assert info.value.field == "agent.pipecat.turn_timeout_s"


@pytest.mark.parametrize("adapter", ["pipecat", "livekit"])
@pytest.mark.parametrize("raw,message", [
    (".nan", "turn_timeout_s nan must be greater than 0"),
    (".inf", INFINITE_MESSAGE),
    ("1e400", INFINITE_MESSAGE),
    ("true", BOOLEAN_MESSAGE),
    ("false", BOOLEAN_MESSAGE),
])
def test_turn_timeout_s_errors_are_the_same_for_pipecat_and_livekit(impulse_project, adapter, raw, message):
    """One shared helper (FIXES-ROUND-3-SPEC.md A1): the Pipecat field gives exactly the LiveKit texts
    (the livekit cases pass today and pin the shared texts)."""
    from impulse_evals.config import ConfigError

    _raw_config(impulse_project, adapter, raw)
    with pytest.raises(ConfigError) as info:
        impulse_project.load()
    assert info.value.field == f"agent.{adapter}.turn_timeout_s"
    assert info.value.message == message
    assert "Value error, " not in str(info.value)


def test_pipecat_turn_timeout_s_is_checked_only_with_adapter_pipecat(impulse_project):
    # Regression guard (passes before the fix): the pipecat block is dropped for another adapter.
    cfg = _cfg(impulse_project, adapter="generic", turn_timeout_s=0)
    assert cfg.agent.adapter == "generic"


def test_pipecat_turn_timeout_s_is_not_a_websocket_only_key(impulse_project):
    cfg = _cfg(impulse_project, "eval", turn_timeout_s=90)
    assert cfg.agent.pipecat.turn_timeout_s == 90
    assert "pipecat.turn_timeout_s is ignored in eval mode." not in cfg.warnings
    assert not [warning for warning in cfg.warnings if "turn_timeout_s" in warning], cfg.warnings


def test_pipecat_turn_timeout_s_in_websocket_mode_is_accepted(impulse_project):
    cfg = _cfg(impulse_project, "websocket", turn_timeout_s=45)
    assert cfg.agent.pipecat.turn_timeout_s == 45
    assert not [warning for warning in cfg.warnings if "turn_timeout_s" in warning], cfg.warnings


# =============================================================================================
# Adapter: which timeout applies (fake clock)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_turn_times_out_at_turn_timeout_s(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, turn_timeout_s=90)
    adapter = await _started(cfg, mock, mode)
    fired = []
    _on_text_turn(mock.channel, [(0.2, mock.rtvi("bot-llm-started"))], fired)   # never stops
    with pytest.raises(Exception) as info:
        await adapter.send("Hello?", None)
    assert str(info.value) == TURN_TIMEOUT.format(n=90)
    assert fired[0] + 90 - 0.01 <= mock.clock.now <= fired[0] + 90 + 0.5
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_fractional_turn_timeout_s_is_formatted_with_g(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, turn_timeout_s=12.5)
    adapter = await _started(cfg, mock, mode)
    fired = []
    _on_text_turn(mock.channel, [(0.2, mock.rtvi("bot-llm-started"))], fired)
    with pytest.raises(Exception) as info:
        await adapter.send("Hello?", None)
    assert str(info.value) == TURN_TIMEOUT.format(n=12.5)
    assert fired[0] + 12.5 - 0.01 <= mock.clock.now <= fired[0] + 12.5 + 0.5
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_turn_timeout_s_wins_over_voice_config_timeout_in_voice_mode(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, voice={"timeout_s": 30}, turn_timeout_s=90)
    adapter = await _started(cfg, mock, mode)
    fired = []
    _on_voice_turn(mock.channel, [(0.1, mock.rtvi("bot-started-speaking"))], fired)   # never stops speaking
    with pytest.raises(Exception) as info:
        await adapter.send(USER_TURN_9, _user_wav())
    assert str(info.value) == TURN_TIMEOUT.format(n=90)
    assert mock.clock.now >= fired[0] + 90 - 0.01   # the voice_config timeout (30 s) was not used
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_shorter_turn_timeout_s_also_wins_over_voice_config_timeout(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, voice={"timeout_s": 30}, turn_timeout_s=5)
    adapter = await _started(cfg, mock, mode)
    fired = []
    _on_voice_turn(mock.channel, [(0.1, mock.rtvi("bot-started-speaking"))], fired)
    with pytest.raises(Exception) as info:
        await adapter.send(USER_TURN_9, _user_wav())
    assert str(info.value) == TURN_TIMEOUT.format(n=5)
    assert fired[0] + 5 - 0.01 <= mock.clock.now <= fired[0] + 5 + 0.5
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_config_timeout_applies_without_turn_timeout_s(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix).
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, voice={"timeout_s": 7})
    adapter = await _started(cfg, mock, mode)
    fired = []
    _on_voice_turn(mock.channel, [(0.1, mock.rtvi("bot-started-speaking"))], fired)
    with pytest.raises(Exception) as info:
        await adapter.send(USER_TURN_9, _user_wav())
    assert str(info.value) == TURN_TIMEOUT.format(n=7)
    assert fired[0] + 7 - 0.01 <= mock.clock.now <= fired[0] + 7 + 0.5
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_turn_timeout_defaults_to_30_s(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix).
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    fired = []
    _on_text_turn(mock.channel, [(0.2, mock.rtvi("bot-llm-started"))], fired)
    with pytest.raises(Exception) as info:
        await adapter.send("Hello?", None)
    assert str(info.value) == TURN_TIMEOUT.format(n=30)
    assert fired[0] + 30 - 0.01 <= mock.clock.now <= fired[0] + 30 + 0.5
    await adapter.close()


# =============================================================================================
# PipecatTurnTimeout: exported, a PipecatError AND an AgentTurnError
# =============================================================================================


def test_pipecat_turn_timeout_class_is_exported_and_is_both_error_kinds():
    module = _module()
    assert "PipecatTurnTimeout" in module.__all__
    timeout_cls = getattr(module, "PipecatTurnTimeout", None)
    assert timeout_cls is not None
    assert issubclass(timeout_cls, module.PipecatError)
    assert issubclass(timeout_cls, _base().AgentTurnError)


# =============================================================================================
# DEFECT-8 (voice mode): the bot speaks a raw function call and never stops speaking
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["spoken", "announced-only", "announced-then-in-progress"])
async def test_defect8_timeout_reply_holds_the_spoken_text_and_the_audio(
    impulse_project, mock_pipecat_channel, mode, variant
):
    """Campaign turn 10: ``send()`` raises the turn timeout after 30 s on the fake clock, and its
    ``reply`` holds the raw function call text and the bot audio (above 1 % of full scale)."""
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, voice=True)   # voice_config.timeout_s 30, as in the campaign
    adapter = await _started(cfg, mock, mode)
    audio = mock.tone(0.5)
    outputs = {
        "spoken": [_spoken(RAW_CALL_TEXT, 501)],
        "announced-only": [_announced(RAW_CALL_TEXT, 501)],
        # word-timestamp progress: the latest text of the segment counts
        "announced-then-in-progress": [_announced("SKIPPED", 501), _in_progress_output(RAW_CALL_TEXT, 501)],
    }[variant]
    fired = []
    _on_voice_turn(mock.channel, _defect8_steps(mock, outputs, audio), fired)
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, _user_wav())
    assert fired[0] + 30 - 0.01 <= mock.clock.now <= fired[0] + 30 + 0.5
    _assert_turn_timeout(error, 30)
    reply = error.reply
    assert isinstance(reply, _base().AgentReply)
    assert reply.text == RAW_CALL_TEXT
    assert reply.audio_bytes is not None
    channels, width, rate, frames = read_wav(reply.audio_bytes)
    assert (channels, width, rate) == (1, 2, 16000)
    assert frames == audio
    assert reply.latency_ms == pytest.approx(200.0, abs=1.0)   # the first speech audio, +0.2 s
    assert reply.tool_calls == []
    assert reply.interrupted is False
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_timeout_text_prefers_spoken_outputs_over_announced_ones(impulse_project, mock_pipecat_channel, mode):
    """The normal voice rule first: when a segment was spoken, announced-only segments are not added."""
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, voice={"timeout_s": 10})
    adapter = await _started(cfg, mock, mode)
    outputs = [_spoken("Let me verify your details.", 401), _announced(RAW_CALL_TEXT, 402)]
    _on_voice_turn(mock.channel, _defect8_steps(mock, outputs, mock.tone(0.5)))
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, _user_wav())
    _assert_turn_timeout(error, 10)
    assert error.reply.text == "Let me verify your details."
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_timeout_text_falls_back_to_the_llm_tokens(impulse_project, mock_pipecat_channel, mode):
    """No bot-output at all: the turn's ``bot-llm-text`` tokens are the text (joined as text mode joins a
    run's tokens: with "")."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice={"timeout_s": 10})
    adapter = await _started(cfg, mock, mode)
    tokens = ["SKIPPED ", "verify_identity(customer_id=CUST-2001, ", "dob=1990-04-12)"]
    steps = [(0.1, rtvi("bot-llm-started"))]
    steps += [(0.15 + 0.05 * index, rtvi("bot-llm-text", {"text": token})) for index, token in enumerate(tokens)]
    steps += [(0.3, rtvi("bot-started-speaking")), (0.4, mock.BotAudio(mock.tone(0.5)))]   # never stops
    _on_voice_turn(mock.channel, steps)
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, _user_wav())
    _assert_turn_timeout(error, 10)
    assert error.reply.text == "".join(tokens).strip()
    assert error.reply.latency_ms == pytest.approx(400.0, abs=1.0)   # first speech audio wins in voice mode
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_timeout_with_speech_audio_only_keeps_the_audio(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice={"timeout_s": 10})
    adapter = await _started(cfg, mock, mode)
    audio = mock.tone(0.4)
    _on_voice_turn(mock.channel, [(0.1, rtvi("bot-started-speaking")), (0.3, mock.BotAudio(audio))])
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, _user_wav())
    _assert_turn_timeout(error, 10)
    reply = error.reply
    assert reply is not None
    assert reply.text.strip() == ""
    assert read_wav(reply.audio_bytes)[3] == audio
    assert reply.latency_ms == pytest.approx(300.0, abs=1.0)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_timeout_without_speech_audio_or_text_has_nan_latency(impulse_project, mock_pipecat_channel, mode):
    """Only silent bot audio (below 1 % of full scale) and a tool call arrived: the reply keeps them, and
    its latency is NaN (no speech audio, no text), never the timeout instant."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice={"timeout_s": 10})
    adapter = await _started(cfg, mock, mode)
    steps = [
        (0.1, rtvi("bot-llm-started")),
        (0.2, _in_progress_call(mock, "call-1", "verify_identity", VERIFY_ARGS)),
        (0.3, rtvi("bot-started-speaking")),
        (0.4, mock.BotAudio(mock.silence(0.2))),
    ]   # the call never stops, the bot never stops speaking
    _on_voice_turn(mock.channel, steps)
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, _user_wav())
    _assert_turn_timeout(error, 10)
    reply = error.reply
    assert reply is not None
    assert [(call.name, call.arguments) for call in reply.tool_calls] == [("verify_identity", VERIFY_ARGS)]
    assert math.isnan(reply.latency_ms)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_timeout_reply_carries_heard_text_and_interrupted(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice={"timeout_s": 10})
    adapter = await _started(cfg, mock, mode)
    heard = "yes you have my consent my customer id is cust 2001"
    steps = [
        (0.05, rtvi("user-transcription", {"text": heard, "user_id": "u", "timestamp": "t", "final": True})),
        (0.1, rtvi("bot-started-speaking")),
        (0.2, mock.BotAudio(mock.tone(0.5))),
        (0.25, rtvi("bot-output", _spoken(RAW_CALL_TEXT, 501))),
        (0.5, rtvi("user-started-speaking")),   # barge-in during the bot's speech
    ]
    _on_voice_turn(mock.channel, steps)
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, _user_wav())
    _assert_turn_timeout(error, 10)
    assert error.reply.heard_text == heard
    assert error.reply.interrupted is True
    assert error.reply.text == RAW_CALL_TEXT
    await adapter.close()


# =============================================================================================
# Text mode: the turn so far
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_timeout_reply_holds_the_llm_text_so_far_and_its_latency(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.3, rtvi("bot-llm-text", {"text": "Let me "})),
        (0.4, rtvi("bot-llm-text", {"text": "verify that."})),
    ]   # bot-llm-stopped never arrives
    _on_text_turn(mock.channel, steps)
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, None)
    _assert_turn_timeout(error, 30)
    reply = error.reply
    assert reply.text == "Let me verify that."
    assert reply.latency_ms == pytest.approx(300.0, abs=1.0)   # send-text -> the first kept text
    assert reply.audio_bytes is None
    assert reply.tool_calls == []
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_timeout_with_only_a_tool_call_has_nan_latency(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.3, _in_progress_call(mock, "call-1", "verify_identity", VERIFY_ARGS)),
        (0.35, rtvi("bot-llm-stopped")),
    ]   # the call never stops: the call in progress holds the turn open
    _on_text_turn(mock.channel, steps)
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, None)
    _assert_turn_timeout(error, 30)
    reply = error.reply
    assert reply is not None
    assert reply.text == ""
    assert [(call.name, call.arguments, call.success) for call in reply.tool_calls] == [
        ("verify_identity", VERIFY_ARGS, None)
    ]
    assert math.isnan(reply.latency_ms)   # no output arrived; never the timeout instant
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_timeout_keeps_an_open_announcement(impulse_project, mock_pipecat_channel, mode):
    """A5-rev in text mode: an announced bot-output still waiting for its completion is in the reply;
    a completed one replaces its announcement (once)."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    first, second = "Thank you, Ananya.", "Let me verify your identity."
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.3, rtvi("bot-llm-text", {"text": "tokens that are not the reply"})),
        (0.4, rtvi("bot-output", _announced(first, 11))),
        (0.45, rtvi("bot-output", _spoken(first, 12))),
        (0.5, rtvi("bot-output", _announced(second, 13))),   # never completed
    ]
    _on_text_turn(mock.channel, steps)
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, None)
    _assert_turn_timeout(error, 30)
    assert _norm(error.reply.text) == f"{first} {second}"
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_text_timeout_reply_puts_the_pre_turn_part_first(impulse_project, mock_pipecat_channel, mode):
    """A greeting queued unread before the send is pre-turn output: first in the reply, never its latency."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    mock.clock.settle_rounds = 1000
    cfg = _cfg(impulse_project, mode)
    channel = _handshake(mock, mode)
    greeting = "Welcome to Sahyog Bank."
    channel.on_connect(_plan(channel, [
        (0.3, rtvi("bot-llm-started")),
        (0.4, rtvi("bot-llm-text", {"text": greeting})),
        (0.5, rtvi("bot-llm-stopped")),
    ]))
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    mock.clock.advance(1.0)   # the greeting now sits on the channel, unread
    steps = [(0.2, rtvi("bot-llm-started")), (0.6, rtvi("bot-llm-text", {"text": "Let me check."}))]   # never stops
    _on_text_turn(channel, steps)
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, None)
    _assert_turn_timeout(error, 30)
    assert error.reply.text == f"{greeting} Let me check."
    assert error.reply.latency_ms == pytest.approx(600.0, abs=1.0)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("voice", [False, True])
async def test_timeout_with_no_output_at_all_gives_reply_none(impulse_project, mock_pipecat_channel, mode, voice):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode, voice=True if voice else None)
    adapter = await _started(cfg, mock, mode)
    if voice:
        _on_voice_turn(mock.channel, [(0.1, mock.rtvi("bot-started-speaking"))])   # speaking, no audio, no text
        error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, _user_wav())
    else:
        _on_text_turn(mock.channel, [(0.2, mock.rtvi("bot-llm-started"))])
        error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, None)
    _assert_turn_timeout(error, 30)
    assert error.reply is None
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_timeout_with_blank_text_only_gives_reply_none(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    _on_text_turn(mock.channel, [(0.2, rtvi("bot-llm-started")), (0.3, rtvi("bot-llm-text", {"text": "   "}))])
    error = await _send_expecting_pipecat_error(adapter, USER_TURN_9, None)
    _assert_turn_timeout(error, 30)
    assert error.reply is None
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_turn_ending_in_the_timeout_remembers_no_open_announcement(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix; A5-rev rule unchanged): the completion of an
    # announcement left open by a timed-out turn is added to the next reply.
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    sentence = "Let me verify your identity."
    _on_text_turn(mock.channel, [(0.2, rtvi("bot-llm-started")), (0.3, rtvi("bot-output", _announced(sentence, 21)))])
    with pytest.raises(Exception) as info:
        await adapter.send(USER_TURN_9, None)
    assert str(info.value) == TURN_TIMEOUT.format(n=30)
    _on_text_turn(mock.channel, [
        (0.2, rtvi("bot-llm-started")),
        (0.3, rtvi("bot-output", _spoken(sentence, 22))),
        (0.4, rtvi("bot-llm-stopped")),
    ])
    reply = await adapter.send("Okay.", None)
    assert _norm(reply.text) == sentence
    await adapter.close()


# =============================================================================================
# Guards: behaviour that must not change
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_normal_voice_turn_still_ends_at_bot_stopped_speaking(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix).
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice=True, turn_timeout_s=90)
    adapter = await _started(cfg, mock, mode)
    audio = mock.tone(0.3)
    fired = []
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.2, mock.BotAudio(audio)),
        (0.25, rtvi("bot-output", _spoken(CAMPAIGN_TURNS[7]["text"], 301))),
        (0.6, rtvi("bot-stopped-speaking")),
    ]
    _on_voice_turn(mock.channel, steps, fired)
    reply = await adapter.send(USER_TURN_9, _user_wav())
    assert reply.text == CAMPAIGN_TURNS[7]["text"]
    assert read_wav(reply.audio_bytes)[3] == audio
    assert reply.latency_ms == pytest.approx(200.0, abs=1.0)
    end = fired[0] + 0.6 + 0.8   # bot-stopped-speaking + end_of_turn_grace_ms
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_normal_text_turn_still_ends_after_the_grace_window(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix).
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, turn_timeout_s=90)
    adapter = await _started(cfg, mock, mode)
    fired = []
    steps = [(0.2, rtvi("bot-llm-started")), (0.3, rtvi("bot-llm-text", {"text": "Done."})), (0.4, rtvi("bot-llm-stopped"))]
    _on_text_turn(mock.channel, steps, fired)
    reply = await adapter.send("Hello?", None)
    assert reply.text == "Done."
    end = fired[0] + 0.4 + 0.8
    assert end - 1e-9 <= mock.clock.now <= end + 0.25
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("code", [1006, 1011])
async def test_abnormal_close_during_a_turn_is_unchanged(impulse_project, mock_pipecat_channel, mode, code):
    # Regression guard (passes before the fix): close errors are unchanged (not a turn timeout).
    from impulse_evals.adapters.base import AgentClosed
    from impulse_evals.adapters.pipecat_adapter import PipecatError

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, turn_timeout_s=90)
    adapter = await _started(cfg, mock, mode)
    _on_text_turn(mock.channel, [(0.1, rtvi("bot-llm-started")), (0.15, rtvi("bot-llm-text", {"text": "Hm"})),
                                 (0.2, mock.BotClose(code))])
    with pytest.raises(PipecatError) as info:
        await adapter.send("Hello?", None)
    assert str(info.value) == BOT_CLOSED.format(code=code)
    assert not isinstance(info.value, AgentClosed)
    assert not isinstance(info.value, _base().AgentTurnError)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("code", [1000, 1001])
async def test_normal_close_during_a_turn_stays_agent_closed(impulse_project, mock_pipecat_channel, mode, code):
    # Regression guard (passes before the fix).
    from impulse_evals.adapters.base import AgentClosed

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, turn_timeout_s=90)
    adapter = await _started(cfg, mock, mode)
    _on_text_turn(mock.channel, [(0.1, rtvi("bot-llm-started")), (0.2, rtvi("bot-llm-text", {"text": "Goodbye."})),
                                 (0.3, mock.BotClose(code))])
    with pytest.raises(AgentClosed) as info:
        await adapter.send("Bye", None)
    assert str(info.value) == BOT_CLOSED.format(code=code)
    assert info.value.reply is not None and info.value.reply.text == "Goodbye."
    await adapter.close()


# =============================================================================================
# Runner: the partial turn of the Pipecat turn timeout lands in the run result
# =============================================================================================


def _run_factory(mock):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    return lambda config: PipecatAdapter(config, channel_factory=mock.factory, clock=mock.clock)


async def _run(cfg, mock):
    from impulse_evals.runner import run

    return await run(cfg, adapter_factory=_run_factory(mock))


def _stored(result, tmp_path):
    """``result`` written with ``reporter.write_json`` and read back (what run.json holds)."""
    from impulse_evals.models import RunResult
    from impulse_evals.reporter import write_json

    path = tmp_path / "run.json"
    write_json(result, path)
    return RunResult.model_validate_json(path.read_text(encoding="utf-8")), json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_runner_text_mode_records_the_timed_out_pipecat_turn(impulse_project, llm_router, mock_pipecat_channel, tmp_path):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    project = impulse_project
    opening = CAMPAIGN_TURNS[0]["text"]
    first_reply = CAMPAIGN_TURNS[1]["text"]
    cfg = _cfg(project, "eval", scenarios=[project.scenario(
        SCENARIO, max_turns=3, opening_user_message=opening, metrics=["UNIV-001", "UNIV-023"],
    )])
    llm_router.script(project.persona(SCENARIO), USER_TURN_9, "Thank you.")
    channel = _handshake(mock, "eval")
    channel.on_send(lambda data: data.get("type") == "send-text" and data["data"]["content"] == opening, _plan(channel, [
        (0.2, rtvi("bot-llm-started")), (0.3, rtvi("bot-llm-text", {"text": first_reply})), (0.4, rtvi("bot-llm-stopped")),
    ]))
    channel.on_send(lambda data: data.get("type") == "send-text" and data["data"]["content"] == USER_TURN_9, _plan(channel, [
        (0.2, rtvi("bot-llm-started")),
        (0.3, rtvi("bot-llm-text", {"text": "Let me verify that."})),
        (0.4, _in_progress_call(mock, "call-1", "verify_identity", VERIFY_ARGS)),
        (0.45, rtvi("bot-llm-stopped")),
    ]))   # the call never stops: the turn ends in the 30 s timeout

    result = await _run(cfg, mock)
    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert [turn.text for turn in turns[:3]] == [opening, first_reply, USER_TURN_9]
    partial = turns[3]
    assert partial.text == "Let me verify that."
    assert [(call.name, call.arguments) for call in partial.tool_calls] == [("verify_identity", VERIFY_ARGS)]
    assert partial.latency_ms == pytest.approx(300.0, abs=1.0)
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)
    assert scenario_result.passed is False
    assert scenario_result.metric_results == []
    assert scenario_result.assertion_results == []
    assert llm_router.calls("judge") == []

    stored, raw = _stored(result, tmp_path)
    stored_turns = stored.scenario_results[0].transcript.turns
    assert stored_turns[3].text == "Let me verify that."
    assert [call.name for call in stored_turns[3].tool_calls] == ["verify_identity"]
    assert raw["scenario_results"][0]["error"] == TURN_TIMEOUT.format(n=30)


@pytest.mark.asyncio
async def test_runner_text_mode_partial_turn_with_only_a_tool_call_has_no_latency(
    impulse_project, llm_router, mock_pipecat_channel
):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    project = impulse_project
    opening = CAMPAIGN_TURNS[0]["text"]
    cfg = _cfg(project, "eval", scenarios=[project.scenario(
        SCENARIO, max_turns=2, opening_user_message=opening, metrics=["UNIV-001"],
    )])
    channel = _handshake(mock, "eval")
    _on_text_turn(channel, [
        (0.2, rtvi("bot-llm-started")),
        (0.3, _in_progress_call(mock, "call-1", "verify_identity", VERIFY_ARGS)),
        (0.35, rtvi("bot-llm-stopped")),
    ])
    result = await _run(cfg, mock)
    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent"]
    assert turns[1].text == ""
    assert [call.name for call in turns[1].tool_calls] == ["verify_identity"]
    assert turns[1].latency_ms is None   # NaN from the adapter becomes None
    assert scenario_result.error == TURN_TIMEOUT.format(n=30)


def _nth_user_speech(channel, number):
    """Audio-rule predicate: the adapter just sent its ``number``-th user audio with speech (eval mode
    sends each user turn in one ``send_audio`` call)."""

    def predicate(record):
        return record["has_speech"] and sum(1 for entry in channel.sent_audio if entry["has_speech"]) == number

    return predicate


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["spoken", "announced-only"])
async def test_runner_voice_mode_reproduces_the_campaign_turn_10(
    impulse_project, llm_router, mock_pipecat_channel, mock_voice_server, variant
):
    """Run 20261025T141732Z-927a72 ``guaranteed_loan`` end to end on the fake channel (eval mode, voice
    mode, voice_config.timeout_s 30): agent turn 10 (the raw function call text, never
    ``bot-stopped-speaking``) is in the run result with its text and audio, and the scenario error is the
    30 s turn timeout."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    project = impulse_project
    project.write_config(
        [project.scenario(SCENARIO, max_turns=6, opening_user_message=CAMPAIGN_TURNS[0]["text"], metrics=["UNIV-001"])],
        agent={"adapter": "pipecat", "pipecat": {"mode": "eval", "url": URL}},
        voice_config=dict(VOICE_CONFIG, save_audio=True, save_recording=False),
    )
    cfg = project.load()
    user_texts = [turn["text"] for turn in CAMPAIGN_TURNS if turn["speaker"] == "user"]
    agent_texts = [turn["text"] for turn in CAMPAIGN_TURNS if turn["speaker"] == "agent"]
    assert len(user_texts) == 5 and len(agent_texts) == 4
    llm_router.script(project.persona(SCENARIO), *user_texts[1:], "Thank you.")

    channel = _handshake(mock, "eval")
    for number, text in enumerate(agent_texts, start=1):
        steps = [
            (0.1, rtvi("bot-started-speaking")),
            (0.2, mock.BotAudio(mock.tone(0.3, frequency_hz=330.0))),
            (0.25, rtvi("bot-output", _spoken(text, 100 + number))),
            (0.6, rtvi("bot-stopped-speaking")),
        ]
        channel.on_audio(_plan(channel, steps), match=_nth_user_speech(channel, number))
    turn10_audio = mock.tone(0.5)
    output = _spoken(RAW_CALL_TEXT, 501) if variant == "spoken" else _announced(RAW_CALL_TEXT, 501)
    channel.on_audio(_plan(channel, _defect8_steps(mock, [output], turn10_audio)), match=_nth_user_speech(channel, 5))

    result = await _run(cfg, mock)
    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent"] * 5
    assert [turn.text for turn in turns[:9]] == [turn["text"] for turn in CAMPAIGN_TURNS]
    turn10 = turns[9]
    assert turn10.text == RAW_CALL_TEXT
    assert turn10.audio_path is not None
    run_dir = project.root / ".impulse" / "runs" / result.run_id
    assert (run_dir / turn10.audio_path.replace("\\", "/")).is_file()
    assert turn10.latency_ms == pytest.approx(200.0, abs=1.0)
    assert scenario_result.error == "Pipecat bot did not finish its turn within 30 s."
    assert scenario_result.passed is False
    assert llm_router.calls("judge") == []
