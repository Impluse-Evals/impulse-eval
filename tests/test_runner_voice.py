# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 11: ``runner.run`` in voice mode (SPEC.md sections 7, 9, 13, 14, 15; METRICS.md VOICE-011).

Voice mode is on when ``voice_config`` exists. Per user turn: simulator text -> TTS -> ``send(text,
audio)``; the agent's audio -> STT. ``Turn.text`` of an agent turn is the adapter's text when
non-empty, else Impulse's STT text of the agent audio (the simulator sees the same text). VOICE-011
(harness round trip: Impulse's STT of its own TTS audio for each user turn) runs automatically, is
informational, stores the round-trip confidence on the user turn, and when poor adds the run warning
"Harness round-trip WER is high; audio metrics in this run are unreliable."

Files (the only files ``runner.run`` writes): with ``save_audio`` one WAV per turn,
``.impulse/runs/<run_id>/audio/<scenario_name>_t{N}_{user|agent}.wav`` (name sanitised, N = 1-based
position in the transcript), ``Turn.audio_path`` relative to the run folder; with
``save_recording`` a stereo recording (left user, right agent, placed by ``timestamp_ms``) in
``CallTranscript.recording_path``.

Voice servers are mocked (``mock_voice_server``); adapters are ``fake_adapter``, the generic adapter
with a test agent that returns audio, or the Pipecat adapter on ``mock_pipecat_channel``.
"""

import re
from pathlib import Path

import pytest
from conftest import multipart_fields, pcm_samples, pcm_tone, read_wav, wav_bytes

pytestmark = pytest.mark.phase11

BASE_URL = "http://localhost:8000/v1"
HOSTED_URL = "http://localhost:9000"
OPENAI_TTS = {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": BASE_URL}
OPENAI_STT = {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": BASE_URL}
HOSTED_TTS = {"provider": "impulse_hosted", "base_url": HOSTED_URL}
HOSTED_STT = {"provider": "impulse_hosted", "base_url": HOSTED_URL}
HOSTED_KEY = "hosted-voice-SECRET-5150"
ROUND_TRIP_WARNING = "Harness round-trip WER is high; audio metrics in this run are unreliable."
PIPECAT_CAP_MESSAGE = "Pipecat bots serve one client at a time; running scenarios one by one."
USER_LINE = "please book a slot on tuesday"
SECOND_LINE = "is ten in the morning free"

AUDIO_AGENT = """
import io
import math
import struct
import wave

def _tone(seconds=0.8, rate=16000):
    frames = b"".join(
        struct.pack("<h", int(16000 * math.sin(2 * math.pi * 330 * i / rate)))
        for i in range(int(seconds * rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()

AUDIO = _tone()

def respond(user_text, history):
    if not user_text:
        return {"text": "Hello, this is Sunrise Clinic.", "audio": AUDIO}
    return {"text": "You said: " + user_text, "audio": AUDIO}
"""


def _voice_config(tts=None, stt=None, **extra):
    config = {
        "tts": dict(tts or OPENAI_TTS),
        "stt": dict(stt or OPENAI_STT),
        "save_audio": True,
        "save_recording": True,
        "timeout_s": 30,
    }
    config.update(extra)
    return config


def _agent_audio(seconds=0.8, sample_rate=16000):
    return wav_bytes(pcm_tone(seconds, sample_rate, frequency_hz=330.0))


def _metrics(scenario_result):
    return {metric.metric_id: metric for metric in scenario_result.metric_results}


def _run_dir(project, result):
    return project.root / ".impulse" / "runs" / result.run_id


def _resolve(run_dir, path):
    path = Path(str(path).replace("\\", "/"))
    return path if path.is_absolute() else run_dir / path


def _duration(path_or_bytes):
    channels, width, rate, frames = read_wav(path_or_bytes)
    return len(frames) / (channels * width) / rate


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


# =============================================================================================
# End to end with the generic adapter
# =============================================================================================


@pytest.mark.asyncio
async def test_end_to_end_voice_run_with_generic_agent(impulse_project, llm_router, mock_voice_server):
    from impulse_evals.models import RunResult

    project = impulse_project
    project.write_agent(AUDIO_AGENT)
    llm_router.script(project.persona("voice_booking"), SECOND_LINE)
    project.write_config(
        [project.scenario("voice_booking", max_turns=2, opening_user_message=USER_LINE,
                          metrics=["UNIV-001", "VOICE-001", "VOICE-029"])],
        voice_config=_voice_config(save_recording=False),
    )
    result = await _run(project.load())

    assert RunResult.model_validate_json(result.model_dump_json()) == result
    assert result.voice_mode is True
    assert result.tts_provider == "openai_compatible"
    assert result.stt_provider == "openai_compatible"
    assert result.latency_profile.is_real_audio is True
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    transcript = scenario_result.transcript
    assert transcript.voice_mode is True
    assert [turn.speaker for turn in transcript.turns] == ["user", "agent", "user", "agent"]
    assert [turn.text for turn in transcript.turns] == [
        USER_LINE, "You said: " + USER_LINE, SECOND_LINE, "You said: " + SECOND_LINE,
    ]

    tts_requests = mock_voice_server.requests_to("/audio/speech")
    assert [request["json"]["input"] for request in tts_requests] == [USER_LINE, SECOND_LINE]
    stt_requests = mock_voice_server.requests_to("/audio/transcriptions")
    assert len(stt_requests) == 4  # round trip of each user turn + each agent reply
    for request in stt_requests:
        fields = multipart_fields(request)
        assert [entry["value"].decode() for entry in fields.get("language", [])] == ["en"]

    metrics = _metrics(scenario_result)
    assert {"UNIV-001", "VOICE-001", "VOICE-029", "VOICE-011"} <= set(metrics)
    assert metrics["VOICE-001"].is_proxy is False
    assert metrics["VOICE-011"].informational is True


@pytest.mark.asyncio
async def test_voice_mode_sends_user_text_and_tts_audio_to_adapter(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    project = impulse_project
    llm_router.script(project.persona("audio_in"), SECOND_LINE)
    fake_adapter.replies["audio_in"] = [
        {"text": "Sure.", "audio_bytes": _agent_audio(), "latency_ms": 300.0},
        {"text": "Ten is free.", "audio_bytes": _agent_audio(), "latency_ms": 300.0},
    ]
    project.write_config(
        [project.scenario("audio_in", max_turns=2, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].error is None
    sends = fake_adapter.sends("audio_in")
    assert [text for text, _ in sends] == [USER_LINE, SECOND_LINE]
    for _, audio in sends:
        assert isinstance(audio, bytes) and audio
        assert _duration(audio) == pytest.approx(0.5, abs=0.05)  # the mock TTS returns 0.5 s


# =============================================================================================
# Files: per-turn WAVs, audio paths, stereo recording
# =============================================================================================


@pytest.mark.asyncio
async def test_per_turn_wav_files_and_relative_audio_paths(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    project = impulse_project
    name = "voice booking"  # sanitised to voice_booking in file names
    llm_router.script(project.persona(name), SECOND_LINE)
    fake_adapter.replies[name] = [
        {"text": "Sure.", "audio_bytes": _agent_audio(0.8), "latency_ms": 300.0},
        {"text": "Ten is free.", "audio_bytes": _agent_audio(1.2), "latency_ms": 300.0},
    ]
    project.write_config(
        [project.scenario(name, max_turns=2, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    run_dir = _run_dir(project, result)
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    expected_durations = [0.5, 0.8, 0.5, 1.2]
    for index, (turn, seconds) in enumerate(zip(turns, expected_durations), start=1):
        expected = f"audio/voice_booking_t{index}_{turn.speaker}.wav"
        assert turn.audio_path is not None
        assert turn.audio_path.replace("\\", "/") == expected
        file = run_dir / expected
        assert file.is_file()
        assert _duration(file) == pytest.approx(seconds, abs=0.05)
    assert scenario_result.transcript.recording_path is None
    # The runner writes audio only; run.json and the report are the CLI's job.
    assert not (run_dir / "run.json").exists()


@pytest.mark.asyncio
async def test_outbound_voice_opener_is_turn_one(impulse_project, llm_router, fake_adapter, mock_voice_server):
    project = impulse_project
    fake_adapter.replies["outbound_voice"] = [
        {"text": "Hello, this is Sunrise Clinic.", "audio_bytes": _agent_audio(0.6), "latency_ms": 50.0},
        {"text": "Sure.", "audio_bytes": _agent_audio(0.8), "latency_ms": 300.0},
    ]
    project.write_config(
        [project.scenario("outbound_voice", direction="outbound", max_turns=1, opening_user_message=USER_LINE,
                          metrics=["VOICE-029"])],
        voice_config=_voice_config(save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    assert fake_adapter.sends("outbound_voice")[0] == (None, None)
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["agent", "user", "agent"]
    assert [turn.audio_path.replace("\\", "/") for turn in turns] == [
        "audio/outbound_voice_t1_agent.wav",
        "audio/outbound_voice_t2_user.wav",
        "audio/outbound_voice_t3_agent.wav",
    ]
    assert turns[0].latency_ms is None  # the opener is not a measured latency
    assert result.latency_profile.all_latencies_ms == [300.0]


@pytest.mark.asyncio
async def test_stereo_recording_left_user_right_agent_placed_by_timestamp(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    project = impulse_project
    llm_router.script(project.persona("recorded"), SECOND_LINE)
    fake_adapter.replies["recorded"] = [
        {"text": "Sure.", "audio_bytes": _agent_audio(0.8), "latency_ms": 300.0},
        {"text": "Ten is free.", "audio_bytes": _agent_audio(0.8), "latency_ms": 300.0},
    ]
    project.write_config(
        [project.scenario("recorded", max_turns=2, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_recording=True),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    run_dir = _run_dir(project, result)
    recording = scenario_result.transcript.recording_path
    assert recording is not None
    path = _resolve(run_dir, recording)
    assert path.is_file()
    assert path.resolve().is_relative_to(run_dir.resolve())
    channels, width, rate, frames = read_wav(path)
    assert channels == 2
    assert width == 2
    samples = pcm_samples(frames)
    left, right = samples[0::2], samples[1::2]
    turns = scenario_result.transcript.turns
    stamps = [turn.timestamp_ms for turn in turns]
    assert stamps == sorted(stamps), f"turn start times must not go backwards: {stamps}"
    last = turns[-1]
    assert len(left) / rate >= last.timestamp_ms / 1000.0 + 0.7  # the last agent audio fits in
    window = int(0.02 * rate)
    for turn in turns:
        start = int(turn.timestamp_ms * rate / 1000)
        channel = left if turn.speaker == "user" else right
        peak = max(abs(sample) for sample in channel[start:start + window])
        assert peak > 1000, f"{turn.speaker} turn at {turn.timestamp_ms} ms is not in its channel"


@pytest.mark.asyncio
async def test_no_audio_files_when_saving_is_off(impulse_project, llm_router, fake_adapter, mock_voice_server):
    project = impulse_project
    fake_adapter.replies["unsaved"] = [{"text": "Sure.", "audio_bytes": _agent_audio(), "latency_ms": 300.0}]
    project.write_config(
        [project.scenario("unsaved", max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    cfg = project.load()
    before = project.files()
    result = await _run(cfg, adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    assert project.files() == before
    assert all(turn.audio_path is None for turn in scenario_result.transcript.turns)
    assert scenario_result.transcript.recording_path is None


# =============================================================================================
# Turn.text rule and heard_text placement
# =============================================================================================


@pytest.mark.asyncio
async def test_agent_turn_text_is_adapter_text_else_stt_text(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    project = impulse_project
    stt_text = "stt heard the agent"
    for _ in range(20):
        mock_voice_server.queue_stt(stt_text)
    persona = project.persona("text_rule")
    llm_router.script(persona, SECOND_LINE, "thank you")
    fake_adapter.replies["text_rule"] = [
        {"text": "Adapter says hello.", "audio_bytes": _agent_audio(), "latency_ms": 300.0},
        {"text": "", "audio_bytes": _agent_audio(), "latency_ms": 300.0},  # no text: STT text is used
        {"text": "Goodbye.", "audio_bytes": _agent_audio(), "latency_ms": 300.0},
    ]
    project.write_config(
        [project.scenario("text_rule", max_turns=3, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    agent_texts = [turn.text for turn in scenario_result.transcript.turns if turn.speaker == "agent"]
    assert agent_texts == ["Adapter says hello.", stt_text, "Goodbye."]
    # The simulator sees the same agent texts as the transcript.
    simulator_calls = llm_router.calls("simulator")
    assert simulator_calls
    last_messages = "\n".join(llm_router.contents(simulator_calls[-1]))
    assert "Adapter says hello." in last_messages
    assert stt_text in last_messages


@pytest.mark.asyncio
async def test_heard_text_lands_on_the_user_turn_in_voice_mode(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    project = impulse_project
    fake_adapter.replies["heard"] = [
        {"text": "Sure.", "audio_bytes": _agent_audio(), "latency_ms": 300.0, "heard_text": "please book a slot"}
    ]
    project.write_config(
        [project.scenario("heard", max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    user, agent = result.scenario_results[0].transcript.turns
    assert user.heard_text == "please book a slot"
    assert agent.heard_text is None


# =============================================================================================
# VOICE-011: automatic, informational, round-trip confidence on the user turn, warning when poor
# =============================================================================================


@pytest.mark.asyncio
async def test_voice011_runs_automatically_and_stores_round_trip_confidence(
    impulse_project, llm_router, fake_adapter, mock_voice_server, monkeypatch
):
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    for _ in range(20):
        mock_voice_server.queue_stt(USER_LINE, confidence=0.83)  # the round trip hears the user exactly
    fake_adapter.replies["round_trip_ok"] = [{"text": "Sure.", "audio_bytes": _agent_audio(), "latency_ms": 300.0}]
    project = impulse_project
    project.write_config(
        [project.scenario("round_trip_ok", max_turns=1, opening_user_message=USER_LINE, metrics=["UNIV-001"])],
        voice_config=_voice_config(stt=HOSTED_STT, save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    metrics = _metrics(scenario_result)
    assert "VOICE-011" in metrics  # not listed in the scenario's metrics
    round_trip = metrics["VOICE-011"]
    assert round_trip.informational is True
    assert round_trip.skipped is False
    assert round_trip.raw_value == pytest.approx(0.0)
    assert round_trip.label == "good"
    user_turn = scenario_result.transcript.turns[0]
    assert user_turn.stt_confidence == pytest.approx(0.83)
    assert ROUND_TRIP_WARNING not in result.warnings


@pytest.mark.asyncio
async def test_voice011_poor_is_informational_and_warns_once(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    for _ in range(20):
        mock_voice_server.queue_stt("completely different words here")
    project = impulse_project
    for name in ("round_trip_bad_1", "round_trip_bad_2"):
        fake_adapter.replies[name] = [{"text": "Sure.", "audio_bytes": _agent_audio(), "latency_ms": 300.0}]
    project.write_config(
        [
            project.scenario("round_trip_bad_1", max_turns=1, opening_user_message=USER_LINE, metrics=["UNIV-001"]),
            project.scenario("round_trip_bad_2", max_turns=1, opening_user_message=USER_LINE, metrics=["UNIV-001"]),
        ],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    for scenario_result in result.scenario_results:
        assert scenario_result.error is None
        round_trip = _metrics(scenario_result)["VOICE-011"]
        assert round_trip.label == "poor"
        assert round_trip.informational is True
        assert scenario_result.passed is True  # informational results never fail a scenario
    assert result.warnings.count(ROUND_TRIP_WARNING) == 1
    assert "VOICE-011" not in result.metric_summary


@pytest.mark.asyncio
async def test_metrics_disabled_runs_no_voice011(impulse_project, llm_router, fake_adapter, mock_voice_server):
    project = impulse_project
    fake_adapter.replies["no_metrics"] = [{"text": "Sure.", "audio_bytes": _agent_audio(), "latency_ms": 300.0}]
    project.write_config(
        [project.scenario("no_metrics", max_turns=1, opening_user_message=USER_LINE, metrics=["UNIV-001"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
        metrics={"enabled": False},
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].metric_results == []


# =============================================================================================
# Unsupported language
# =============================================================================================


@pytest.mark.asyncio
async def test_unsupported_language_raises_clear_error(
    impulse_project, llm_router, fake_adapter, mock_voice_server, monkeypatch
):
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    project = impulse_project
    project.write_config(
        [project.scenario("tamil_call", language="ta-IN", max_turns=1, opening_user_message="Vanakkam",
                          metrics=["VOICE-029"])],
        voice_config=_voice_config(tts=HOSTED_TTS, stt=HOSTED_STT, save_audio=False, save_recording=False),
    )
    try:
        result = await _run(project.load(), adapter_factory=fake_adapter)
    except Exception as exc:  # a configuration-level error is acceptable
        message = str(exc)
    else:
        assert result.scenario_results[0].passed is False
        message = result.scenario_results[0].error or ""
    assert "ta-IN" in message
    assert "impulse_hosted" in message
    assert mock_voice_server.requests_to("/v1/tts") == []


# =============================================================================================
# Pipecat in voice mode: the concurrency cap still applies
# =============================================================================================


def _after_user_speech(channel, samples):
    def predicate(record):
        total = sum(len(entry["pcm"]) // 2 for entry in channel.sent_audio)
        return total >= samples * 0.98

    return predicate


def _script_pipecat_voice_scenario(mock, channel, reply_text):
    rtvi = mock.rtvi
    channel.paced_audio = True
    channel.on_connect(rtvi("bot-ready", {"version": "2.1.0", "about": {"library": "pipecat-ai"}}, id="client-ready"))
    channel.on_audio(
        rtvi("bot-started-speaking"),
        mock.BotAudio(mock.tone(0.4)),
        rtvi("bot-output", {"text": reply_text, "aggregated_by": "sentence", "spoken": True,
                            "spoken_status": "completed"}),
        rtvi("bot-stopped-speaking"),
        match=_after_user_speech(channel, 8000),
        after=0.2,  # bot audio 0.3 s after the user's speech ended: TTFA 300 ms
        gap=0.1,
    )


@pytest.mark.asyncio
async def test_pipecat_voice_run_keeps_concurrency_cap(
    impulse_project, llm_router, mock_voice_server, mock_pipecat_channel, capsys
):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    mock = mock_pipecat_channel
    _script_pipecat_voice_scenario(mock, mock.channel, "Your slot is booked.")
    _script_pipecat_voice_scenario(mock, mock.add_channel(), "Tuesday is free.")
    returned, overlaps = [], []

    def channel_factory(*args, **kwargs):
        channel = mock.factory(*args, **kwargs)
        if any(previous.connect_calls and not previous.closed for previous in returned):
            overlaps.append(channel.name)
        returned.append(channel)
        return channel

    project = impulse_project
    project.write_config(
        [
            project.scenario("pipecat_voice_1", max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"]),
            project.scenario("pipecat_voice_2", max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"]),
        ],
        agent={"adapter": "pipecat", "pipecat": {"mode": "eval", "url": "ws://localhost:7860"}},
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    cfg = project.load()
    result = await _run(
        cfg,
        concurrency=3,
        adapter_factory=lambda config: PipecatAdapter(config, channel_factory=channel_factory, clock=mock.clock),
    )
    out = capsys.readouterr()
    assert (out.out + out.err).count(PIPECAT_CAP_MESSAGE) == 1
    assert PIPECAT_CAP_MESSAGE in result.warnings
    assert overlaps == []
    assert len(returned) == 2
    texts = {}
    for scenario_result in result.scenario_results:
        assert scenario_result.error is None, scenario_result.error
        agent_turns = [turn for turn in scenario_result.transcript.turns if turn.speaker == "agent"]
        assert len(agent_turns) == 1
        texts[scenario_result.scenario_name] = agent_turns[0].text
        assert agent_turns[0].latency_ms == pytest.approx(300.0, abs=1.0)
    assert texts == {"pipecat_voice_1": "Your slot is booked.", "pipecat_voice_2": "Tuesday is free."}
    assert result.agent_entrypoint == "ws://localhost:7860"
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", result.run_id)


# =============================================================================================
# Audit batch 1 (AUDIT.md findings 5 and 7). F7: a TTS WAV whose RIFF / data size fields are
# streaming placeholders (0xFFFFFFFF, or 0) is measured by the PCM bytes actually present: the turn
# timeline, the voice metadata, the saved files, the STT upload, and the recording all see a sane
# clip. F5: a Basic-auth password in ``base_url`` never reaches a scenario error, run.json, or the
# HTML report.
# =============================================================================================

BASE_URL_PASSWORD = "PASSWORD123"
USERINFO_BASE_URL = f"http://user:{BASE_URL_PASSWORD}@localhost:8000/v1"
VOICE_METADATA_KEY = "impulse_voice"


def _placeholder_sizes(wav, riff_size=None, data_size=None):
    """``wav`` with its RIFF size (bytes 4:8) and / or ``data`` chunk size overwritten."""
    import struct

    out = bytearray(wav)
    if riff_size is not None:
        out[4:8] = struct.pack("<I", riff_size)
    if data_size is not None:
        index = wav.find(b"data")
        assert index > 0
        out[index + 4:index + 8] = struct.pack("<I", data_size)
    return bytes(out)


def _data_size_field(wav):
    import struct

    index = wav.find(b"data")
    return struct.unpack("<I", wav[index + 4:index + 8])[0]


@pytest.fixture
def no_huge_allocations(monkeypatch):
    """Safety net for the placeholder tests: a numpy buffer of more than 1e8 elements (a timeline
    driven by a 0xFFFFFFFF header: about 37 hours of stereo audio) raises instead of exhausting
    memory. A sane timeline allocates a few hundred thousand elements."""
    import numpy

    real_zeros = numpy.zeros

    def guarded(shape, *args, **kwargs):
        count = shape if isinstance(shape, int) else int(numpy.prod(shape))
        if count > 100_000_000:
            raise MemoryError(f"test guard: numpy.zeros({shape!r}) would allocate {count:,} elements")
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(numpy, "zeros", guarded)


@pytest.mark.asyncio
async def test_tts_wav_with_streaming_size_placeholders_keeps_the_real_duration(
    impulse_project, llm_router, fake_adapter, mock_voice_server, monkeypatch, no_huge_allocations
):
    good = wav_bytes(pcm_tone(1.0, 16000))                                   # a real 1 s clip
    bad = _placeholder_sizes(good, riff_size=0xFFFFFFFF, data_size=0xFFFFFFFF)   # ffmpeg-to-pipe style header
    monkeypatch.setattr(mock_voice_server, "wav", lambda: bad)
    project = impulse_project
    name = "stream_header"
    llm_router.script(project.persona(name), SECOND_LINE)
    fake_adapter.replies[name] = [
        {"text": "Sure.", "audio_bytes": _agent_audio(0.8), "latency_ms": 300.0},
        {"text": "Ten is free.", "audio_bytes": _agent_audio(0.8), "latency_ms": 300.0},
    ]
    project.write_config(
        [project.scenario(name, max_turns=2, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_recording=True),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    transcript = scenario_result.transcript
    turns = transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert (turns[0].timestamp_ms, turns[0].end_ms) == (0, 1000)            # 1 s of PCM, not 2^31 frames
    assert (turns[1].timestamp_ms, turns[1].end_ms) == (1300, 2100)         # user end + 300 ms latency, 0.8 s
    assert turns[2].timestamp_ms == 2100
    assert transcript.duration_ms == 4200                                   # 1000 + 300 + 800 + 1000 + 300 + 800
    records = transcript.metadata[VOICE_METADATA_KEY]["turns"]
    assert (records[0]["speaker"], records[0]["audio_ms"]) == ("user", 1000)
    sent_audio = fake_adapter.sends(name)[0][1]
    assert _duration(sent_audio) == pytest.approx(1.0, abs=0.01)
    run_dir = _run_dir(project, result)
    assert transcript.recording_path is not None
    assert _duration(_resolve(run_dir, transcript.recording_path)) == pytest.approx(4.2, abs=0.1)
    saved = (run_dir / "audio" / f"{name}_t1_user.wav").read_bytes()
    assert len(read_wav(saved)[3]) == 32000
    assert _data_size_field(saved) == 32000


@pytest.mark.asyncio
async def test_tts_wav_with_zero_data_size_uses_the_pcm_bytes_present(
    impulse_project, llm_router, fake_adapter, mock_voice_server, monkeypatch
):
    good = wav_bytes(pcm_tone(1.0, 16000))
    zero = _placeholder_sizes(good, data_size=0)   # data size 0, RIFF size intact: 32000 PCM bytes follow
    monkeypatch.setattr(mock_voice_server, "wav", lambda: zero)
    project = impulse_project
    name = "zero_header"
    fake_adapter.replies[name] = [{"text": "Sure.", "audio_bytes": _agent_audio(), "latency_ms": 300.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_recording=True),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent"]
    assert turns[0].end_ms == 1000
    assert turns[1].timestamp_ms == 1300
    # The round-trip STT upload (the first transcription request) carries the PCM with a sane header.
    upload = multipart_fields(mock_voice_server.requests_to("/audio/transcriptions")[0])["file"][0]["value"]
    assert len(read_wav(upload)[3]) == 32000
    # The user's audio is in the recording's left channel during the first second.
    run_dir = _run_dir(project, result)
    channels, width, rate, frames = read_wav(_resolve(run_dir, scenario_result.transcript.recording_path))
    assert channels == 2
    left = pcm_samples(frames)[0::2]
    assert max(abs(sample) for sample in left[:rate]) > 1000


@pytest.mark.asyncio
async def test_well_formed_tts_wav_reaches_the_adapter_unchanged(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    # Regression guard: a well-formed WAV is passed through byte-identical.
    project = impulse_project
    name = "verbatim_tts"
    fake_adapter.replies[name] = [{"text": "Sure.", "audio_bytes": _agent_audio(), "latency_ms": 300.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].error is None
    assert fake_adapter.sends(name)[0][1] == mock_voice_server.wav()


@pytest.mark.asyncio
async def test_voice_run_error_never_contains_the_base_url_password(
    impulse_project, llm_router, fake_adapter, mock_voice_server, tmp_path
):
    from impulse_evals.reporter import render_html

    mock_voice_server.queue_status(401, count=10, body={"error": "bad credentials"})
    project = impulse_project
    name = "voice_basic_auth"
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(
            tts=dict(OPENAI_TTS, base_url=USERINFO_BASE_URL),
            stt=dict(OPENAI_STT, base_url=USERINFO_BASE_URL),
            save_audio=False,
            save_recording=False,
        ),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is not None
    assert BASE_URL_PASSWORD not in scenario_result.error, scenario_result.error
    assert "localhost" in scenario_result.error
    assert BASE_URL_PASSWORD not in result.model_dump_json()
    assert BASE_URL_PASSWORD not in render_html(result, embed_audio=True, run_dir=tmp_path)


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 18): an IEEE-float WAV (format tag 3, 32- or 64-bit, or
# WAVE_FORMAT_EXTENSIBLE with the float sub-format), as torchaudio and several torch-based TTS
# wrappers write it, is re-encoded to 16-bit PCM (same rate and channels; clipped to [-1, 1], scaled
# by 32767, rounded half to even) before Impulse uses it: no "unknown format: 3" scenario error, and
# the adapter, the STT uploads, the saved files and the recording all get 16-bit PCM of the same
# length. A 16-bit PCM WAV stays byte-identical (guards above and below).
# =============================================================================================

FLOAT_KINDS = ["float32", "float64", "extensible_float32"]
# KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: 00000003-0000-0010-8000-00aa00389b71
IEEE_FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")


def _float_samples(seconds, rate, frequency_hz=440.0, amplitude=0.5):
    import numpy

    count = round(seconds * rate)
    return amplitude * numpy.sin(2 * numpy.pi * frequency_hz * numpy.arange(count) / rate)


def _float_wav(samples, rate, kind):
    """A mono RIFF/WAVE file with IEEE-float samples (``kind`` in ``FLOAT_KINDS``)."""
    import struct

    bits = 64 if kind == "float64" else 32
    data = samples.astype("<f8" if bits == 64 else "<f4").tobytes()
    block_align = bits // 8
    if kind.startswith("extensible"):
        fmt = struct.pack("<HHIIHH", 0xFFFE, 1, rate, rate * block_align, block_align, bits)
        fmt += struct.pack("<HHI", 22, bits, 0x4) + IEEE_FLOAT_GUID   # cbSize, valid bits, front-centre mask
    else:
        fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * block_align, block_align, bits)
    body = (
        b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"fact" + struct.pack("<II", 4, len(samples))
        + b"data" + struct.pack("<I", len(data)) + data
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _expected_pcm16(samples):
    import numpy

    return numpy.rint(numpy.clip(samples, -1.0, 1.0) * 32767.0).astype(numpy.int64)


def _assert_pcm16_of(wav, samples, rate):
    """``wav`` is mono 16-bit PCM at ``rate`` holding ``samples`` (within one step of the rounding)."""
    import numpy

    channels, width, got_rate, frames = read_wav(wav)
    assert (channels, width, got_rate) == (1, 2, rate)
    got = numpy.array(pcm_samples(frames), dtype=numpy.int64)
    assert got.size == samples.size   # the same duration
    assert int(numpy.max(numpy.abs(got - _expected_pcm16(samples)))) <= 1


def _uploads(server):
    """The WAV bytes of every STT upload, in order."""
    return [
        entry["value"]
        for request in server.requests_to("/audio/transcriptions")
        for entry in multipart_fields(request).get("file", [])
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", FLOAT_KINDS)
async def test_float_tts_wav_is_re_encoded_to_16_bit_pcm(
    impulse_project, llm_router, fake_adapter, mock_voice_server, monkeypatch, kind
):
    rate = 24000
    samples = _float_samples(1.0, rate)
    float_wav = _float_wav(samples, rate, kind)
    monkeypatch.setattr(mock_voice_server, "wav", lambda: float_wav)
    project = impulse_project
    name = f"float_tts_{kind}"
    fake_adapter.replies[name] = [{"text": "Sure.", "audio_bytes": _agent_audio(0.8), "latency_ms": 300.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_recording=True),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    sends = fake_adapter.sends(name)
    assert len(sends) == 1
    _assert_pcm16_of(sends[0][1], samples, rate)   # what the adapter receives
    turns = scenario_result.transcript.turns
    assert (turns[0].timestamp_ms, turns[0].end_ms) == (0, 1000)
    assert turns[1].timestamp_ms == 1300
    round_trip_upload = _uploads(mock_voice_server)[0]   # the round trip of the user turn
    _assert_pcm16_of(round_trip_upload, samples, rate)
    run_dir = _run_dir(project, result)
    _assert_pcm16_of((run_dir / "audio" / f"{name}_t1_user.wav").read_bytes(), samples, rate)
    assert scenario_result.transcript.recording_path is not None
    assert _duration(_resolve(run_dir, scenario_result.transcript.recording_path)) == pytest.approx(2.1, abs=0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", FLOAT_KINDS)
async def test_float_agent_wav_is_re_encoded_to_16_bit_pcm(
    impulse_project, llm_router, fake_adapter, mock_voice_server, kind
):
    rate = 16000
    samples = _float_samples(1.0, rate, frequency_hz=330.0)
    project = impulse_project
    name = f"float_agent_{kind}"
    fake_adapter.replies[name] = [
        {"text": "Your slot is booked.", "audio_bytes": _float_wav(samples, rate, kind), "latency_ms": 300.0}
    ]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_recording=True),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    agent = scenario_result.transcript.turns[1]
    assert agent.speaker == "agent"
    assert agent.text == "Your slot is booked."
    assert agent.end_ms - agent.timestamp_ms == 1000
    uploads = _uploads(mock_voice_server)
    assert len(uploads) == 2   # the round trip of the user turn, then the agent's audio
    _assert_pcm16_of(uploads[1], samples, rate)
    run_dir = _run_dir(project, result)
    _assert_pcm16_of((run_dir / "audio" / f"{name}_t2_agent.wav").read_bytes(), samples, rate)


@pytest.mark.asyncio
async def test_pcm16_agent_wav_reaches_stt_unchanged(impulse_project, llm_router, fake_adapter, mock_voice_server):
    # Regression guard: the agent's 16-bit PCM WAV is uploaded to STT byte-identical.
    project = impulse_project
    name = "verbatim_agent_audio"
    agent_wav = _agent_audio(0.8)
    fake_adapter.replies[name] = [{"text": "Sure.", "audio_bytes": agent_wav, "latency_ms": 300.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].error is None
    assert _uploads(mock_voice_server) == [mock_voice_server.wav(), agent_wav]


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 14), voice mode: an agent turn with no text and no audio (or with
# audio that Impulse's STT hears as blank) reaches the simulator LLM as "(no speech)", never as an
# empty message; a provider that rejects empty content (stand-in: a LiteLLM BadRequestError) no
# longer turns the scenario into an error.
# =============================================================================================

NO_SPEECH = "(no speech)"


def _reject_blank_simulator_content(llm_router):
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
@pytest.mark.parametrize("silent", ["no_audio", "blank_stt_text"])
async def test_voice_turn_without_speech_reaches_the_simulator_as_no_speech(
    impulse_project, llm_router, fake_adapter, mock_voice_server, silent
):
    project = impulse_project
    name = f"voice_silent_{silent}"
    if silent == "no_audio":
        reply = {"text": "", "audio_bytes": None, "latency_ms": 300.0}
    else:
        mock_voice_server.queue_stt(USER_LINE)            # the round trip of the user turn
        mock_voice_server.queue_stt("   ", words=[])      # Impulse's STT hears nothing in the agent audio
        reply = {"text": "", "audio_bytes": _agent_audio(0.8), "latency_ms": 300.0}
    fake_adapter.replies[name] = [reply, {"text": "Yes, I can hear you.", "audio_bytes": _agent_audio(), "latency_ms": 300.0}]
    llm_router.script(project.persona(name), "Hello? I cannot hear you.")
    rejected = _reject_blank_simulator_content(llm_router)
    project.write_config(
        [project.scenario(name, max_turns=2, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    assert rejected == []
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert not turns[1].text.strip()   # the transcript keeps the blank text
    contents = llm_router.contents(llm_router.calls("simulator")[0])
    assert contents[-1] == NO_SPEECH


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 19): local_hf TTS needs torch, which the 'local' extra does not
# install. Without it the run stops with a ConfigError at voice_config.tts.provider BEFORE the first
# paid call (no system-prompt summary, no simulator call), instead of a scenario error inside the
# first synthesis of every scenario.
# =============================================================================================

TORCH_MISSING = (
    "local_hf TTS needs PyTorch, which the 'local' extra does not install: pip install torch "
    "(see https://pytorch.org/get-started/locally/)"
)
LOCAL_TTS = {"provider": "local_hf", "model": "hf-test/tts-default", "device": "cpu"}
LOCAL_STT = {"provider": "local_hf", "model": "hf-test/stt-small", "device": "cpu", "compute_type": "int8"}


@pytest.mark.asyncio
async def test_local_hf_tts_without_torch_stops_the_run_before_any_llm_call(
    impulse_project, llm_router, fake_adapter, extra_imports
):
    from impulse_evals.config import ConfigError

    extra_imports.package("transformers")     # the 'local' extra is installed ...
    extra_imports.package("faster_whisper")
    extra_imports.missing("torch")            # ... but torch is not
    project = impulse_project
    names = ["no_torch_1", "no_torch_2"]
    for name in names:
        llm_router.script(project.persona(name), USER_LINE)
    project.write_config(
        # No opening_user_message: the simulator LLM would make the first user turn (a paid call).
        [project.scenario(name, max_turns=1, opening_user_message=None, metrics=["VOICE-029"]) for name in names],
        voice_config=_voice_config(tts=LOCAL_TTS, stt=LOCAL_STT, save_audio=False, save_recording=False),
    )
    with pytest.raises(ConfigError) as info:
        await _run(project.load(), adapter_factory=fake_adapter)
    assert info.value.field == "voice_config.tts.provider"
    assert info.value.message == TORCH_MISSING
    assert llm_router.mock.calls == []    # no LLM call at all
    assert fake_adapter.events == []      # no scenario started


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 20), runner level: a TTS server answering 200 with a JSON error
# body is a scenario error naming the server's answer, not the pydub / 'voice' extra message.
# =============================================================================================


@pytest.mark.asyncio
async def test_json_200_from_tts_is_a_scenario_error_with_the_server_answer(
    impulse_project, llm_router, fake_adapter, monkeypatch, extra_imports
):
    import httpx
    from conftest import MockVoiceServer

    body ='{"error": "model impulse-test-tts not found; voice alloy is unavailable"}'

    class Json200TTSServer(MockVoiceServer):
        def _handle(self, request):
            if request.method == "POST" and request.url.path.rstrip("/").endswith("/audio/speech"):
                self.requests.append({"method": request.method, "url": str(request.url), "path": request.url.path,
                                      "headers": dict(request.headers), "json": None, "content": request.content})
                return httpx.Response(200, content=body.encode(), headers={"content-type": "application/json"})
            return super()._handle(request)

    extra_imports.missing("pydub")   # the default install: no 'voice' extra
    server = Json200TTSServer()
    server.install(monkeypatch)
    project = impulse_project
    name = "json_tts"
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=False),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == (
        f"openai_compatible TTS: POST {BASE_URL}/audio/speech returned application/json, not audio: {body}"
    )
    assert scenario_result.passed is False
    assert fake_adapter.sends(name) == []
    assert len(server.requests_to("/audio/speech")) == 1   # not retried


# =============================================================================================
# Audit batch 3 (AUDIT.md finding 45), runner level: an impulse_hosted STT answer whose word entries
# lack a confidence (null or missing) is used (confidence 0.0 per word), never an "Impulse STT
# failed" turn: the round-trip confidence lands on the user turn and no STT warning is added.
# =============================================================================================


@pytest.mark.asyncio
async def test_hosted_stt_words_without_confidence_are_used_not_an_stt_failure(
    impulse_project, llm_router, fake_adapter, mock_voice_server, monkeypatch
):
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    project = impulse_project
    name = "null_word_confidence"
    fake_adapter.replies[name] = [{"text": "Your slot is booked.", "audio_bytes": _agent_audio(0.5), "latency_ms": 200.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(tts=HOSTED_TTS, stt=HOSTED_STT, save_audio=False, save_recording=False),
    )
    words = [
        {"word": "please", "start_ms": 0, "end_ms": 250, "confidence": None},
        {"word": "book", "start_ms": 250, "end_ms": 500},
    ]
    for _ in range(2):  # the round trip of the user turn, then the agent audio
        mock_voice_server.queue_stt("please book", words=words, confidence=0.8)
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    assert not [warning for warning in result.warnings if "Impulse STT failed" in warning], result.warnings
    turns = scenario_result.transcript.turns
    assert turns[0].speaker == "user"
    assert turns[0].stt_confidence == pytest.approx(0.8)


# =============================================================================================
# Audit batch 3 (AUDIT.md finding 46): the stereo recording is mixed in a preallocated float32 buffer
# and converted to 16-bit PCM once. Heuristic: tracemalloc's peak over a whole voice run whose agent
# returns a 60 s clip stays under 40 bytes per frame of the recording (the float64 mix with its full
# copies peaks at about 74; a float32 mix needs about 20-30); a warm-up run first, so imports do not
# count. Byte-identity guard: the recording is unchanged, every 16-bit sample value k mapped to
# rint(k / 32768 * 32767) (half to even, as numpy.rint and Python's round do).
# =============================================================================================

MIX_PEAK_BYTES_PER_FRAME = 40
LONG_AGENT_SECONDS = 60


def _recording(project, result):
    """``(channels, sample_width, sample_rate, frames, file_bytes)`` of the run's only recording."""
    scenario_result = result.scenario_results[0]
    assert scenario_result.transcript.recording_path is not None
    path = _resolve(_run_dir(project, result), scenario_result.transcript.recording_path)
    data = path.read_bytes()
    return (*read_wav(data), data)


@pytest.mark.asyncio
async def test_stereo_recording_of_a_long_agent_clip_has_a_bounded_memory_peak(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    import tracemalloc

    project = impulse_project
    name = "long_agent_clip"
    second = pcm_tone(1.0, 16000, frequency_hz=330.0)  # 330 whole periods: repeats seamlessly
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=True),
    )
    cfg = project.load()
    fake_adapter.replies[name] = [{"text": "Sure.", "audio_bytes": wav_bytes(second), "latency_ms": 300.0}]
    warm_up = await _run(cfg, adapter_factory=fake_adapter)  # imports and caches happen here
    assert warm_up.scenario_results[0].error is None, warm_up.scenario_results[0].error
    long_clip = wav_bytes(second * LONG_AGENT_SECONDS)  # built before tracing starts
    fake_adapter.replies[name] = [{"text": "Sure.", "audio_bytes": long_clip, "latency_ms": 300.0}]
    tracemalloc.start()
    try:
        result = await _run(cfg, adapter_factory=fake_adapter)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.scenario_results[0].error is None, result.scenario_results[0].error
    channels, width, rate, frames, _ = _recording(project, result)
    assert (channels, width, rate) == (2, 2, 16000)
    count = len(frames) // (channels * width)
    assert count >= LONG_AGENT_SECONDS * rate
    per_frame = peak / count
    assert per_frame < MIX_PEAK_BYTES_PER_FRAME, (
        f"peak {peak / 1e6:.1f} MB for {count} recording frames: {per_frame:.1f} bytes per frame "
        f"(limit {MIX_PEAK_BYTES_PER_FRAME})"
    )


@pytest.mark.asyncio
async def test_stereo_recording_bytes_are_unchanged(impulse_project, llm_router, fake_adapter, mock_voice_server):
    """Regression guard (passes before the fix): left the user's TTS clip, right an agent clip holding
    every 16-bit sample value once; the file equals the 16-bit WAV of rint(k / 32768 * 32767) per
    sample, each clip at its turn's ``timestamp_ms``."""
    import array
    import sys

    project = impulse_project
    name = "every_sample_value"
    values = list(range(-32768, 32768))
    ramp = array.array("h", values)
    if sys.byteorder == "big":
        ramp.byteswap()
    fake_adapter.replies[name] = [{"text": "Sure.", "audio_bytes": wav_bytes(ramp.tobytes()), "latency_ms": 300.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(save_audio=False, save_recording=True),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    channels, width, rate, frames, data = _recording(project, result)
    assert (channels, width, rate) == (2, 2, 16000)
    user_turn, agent_turn = scenario_result.transcript.turns
    user_pcm = pcm_samples(read_wav(mock_voice_server.wav())[3])
    user_at = round(user_turn.timestamp_ms * rate / 1000)
    agent_at = round(agent_turn.timestamp_ms * rate / 1000)
    length = max(user_at + len(user_pcm), agent_at + len(values))
    left, right = [0] * length, [0] * length
    for offset, sample in enumerate(user_pcm):
        left[user_at + offset] += sample
    for offset, sample in enumerate(values):
        right[agent_at + offset] += sample

    def scaled(k):
        return max(-32768, min(32767, round(k * 32767 / 32768)))

    expected = array.array("h")
    for pair in zip(left, right):
        expected.extend(scaled(k) for k in pair)
    if sys.byteorder == "big":
        expected.byteswap()
    assert len(frames) == len(expected) * 2
    assert frames == expected.tobytes()
    assert data == wav_bytes(expected.tobytes(), rate, channels=2)


# =============================================================================================
# Maintainer decision 2026-10-21, item 3 (AUDIT-FIXES.md, logged F19), runner level: a torch that
# is installed but fails to import (here an on-disk fake torch package whose import raises the
# Windows DLL OSError) stops the run with ConfigError("voice_config.tts.provider", TORCH_IMPORT_FAILED
# ...) BEFORE the first paid call, like a torch that is not installed at all. torch is never
# imported in this process (the check imports it in a separate interpreter). On Windows the
# message ends with TORCH_WINDOWS_HINT.
# =============================================================================================

import importlib.util  # noqa: E402
import sys  # noqa: E402

TORCH_IMPORT_FAILED = "local_hf TTS needs PyTorch, which is installed but could not be imported ({detail})."
TORCH_WINDOWS_HINT = (
    " On Windows, a DLL load error (for example a missing vcruntime140_1.dll) means the Microsoft Visual C++ "
    "Redistributable (x64) needs to be installed or updated."
)
TORCH_DLL_OSERROR = (
    '[WinError 126] The specified module could not be found. Error loading "C:\\fake\\torch\\lib\\c10.dll" '
    "or one of its dependencies."
)


def _broken_on_disk_torch(tmp_path, monkeypatch):
    """A ``torch`` package on disk whose import raises the Windows DLL ``OSError``, first on
    ``sys.path``; every ``sys.modules`` entry for torch (a None from ``extra_imports`` included) is
    removed for the test."""
    site = tmp_path / "torch-site"
    (site / "torch").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text(f"raise OSError({TORCH_DLL_OSERROR!r})\n", encoding="utf-8")
    for loaded in [name for name in sys.modules if name == "torch" or name.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, loaded)
    monkeypatch.syspath_prepend(str(site))
    assert importlib.util.find_spec("torch") is not None and "torch" not in sys.modules


def _expected_torch_import_failed():
    message = TORCH_IMPORT_FAILED.format(detail=f"OSError: {TORCH_DLL_OSERROR}")
    return message + (TORCH_WINDOWS_HINT if sys.platform == "win32" else "")


@pytest.mark.asyncio
async def test_local_hf_tts_with_torch_that_fails_to_import_stops_the_run_before_any_llm_call(
    impulse_project, llm_router, fake_adapter, extra_imports, monkeypatch, tmp_path
):
    from impulse_evals.config import ConfigError

    extra_imports.package("transformers")     # the 'local' extra is installed ...
    extra_imports.package("faster_whisper")
    _broken_on_disk_torch(tmp_path, monkeypatch)   # ... and torch is installed, but its import fails
    project = impulse_project
    names = ["broken_torch_1", "broken_torch_2"]
    for name in names:
        llm_router.script(project.persona(name), USER_LINE)
    project.write_config(
        # No opening_user_message: the simulator LLM would make the first user turn (a paid call).
        [project.scenario(name, max_turns=1, opening_user_message=None, metrics=["VOICE-029"]) for name in names],
        voice_config=_voice_config(tts=LOCAL_TTS, stt=LOCAL_STT, save_audio=False, save_recording=False),
    )
    with pytest.raises(ConfigError) as info:
        await _run(project.load(), adapter_factory=fake_adapter)
    assert info.value.field == "voice_config.tts.provider"
    assert info.value.message == _expected_torch_import_failed()
    assert llm_router.mock.call_count == 0   # no LLM call at all
    assert fake_adapter.events == []         # no scenario started
    assert "torch" not in sys.modules
