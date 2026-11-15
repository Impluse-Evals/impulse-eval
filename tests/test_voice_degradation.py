# SPDX-License-Identifier: Apache-2.0
"""Phase 11: graceful degradation and latency honesty in voice mode (SPEC.md sections 14.1, 14.3, 15;
METRICS.md section 0 and VOICE-015).

* STT without word timings: VOICE-015 is skipped with "STT provider returned no word timings",
  never failed.
* STT without confidence: never fails a scenario; a cohort whose turns all have
  ``stt_confidence=None`` has ``mean_stt_confidence=None`` and no ``stt_warning``.
* Latency measures the agent only: Impulse's own TTS / STT time never enters ``Turn.latency_ms``,
  VOICE-001, or the latency profile.

End to end through ``runner.run`` in voice mode with mocked voice servers (``httpx.MockTransport``).
"""

import time

import pytest
from conftest import (
    MockVoiceServer,
    multipart_fields,
    pcm_tone,
    read_wav,
    wav_bytes,
    word_timings,
)

pytestmark = pytest.mark.phase11

BASE_URL = "http://localhost:8000/v1"
HOSTED_URL = "http://localhost:9000"
OPENAI_TTS = {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": BASE_URL}
OPENAI_STT = {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": BASE_URL}
HOSTED_STT = {"provider": "impulse_hosted", "base_url": HOSTED_URL}
NO_WORD_TIMINGS = "STT provider returned no word timings"
USER_LINE = "please book a slot on tuesday"


def _voice_config(stt=None, **extra):
    config = {"tts": dict(OPENAI_TTS), "stt": dict(stt or OPENAI_STT), "save_audio": False,
              "save_recording": False, "timeout_s": 30}
    config.update(extra)
    return config


def _agent_audio(seconds):
    return wav_bytes(pcm_tone(seconds, 16000, frequency_hz=330.0))


def _metrics(scenario_result):
    return {metric.metric_id: metric for metric in scenario_result.metric_results}


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


class DurationRoutedVoiceServer(MockVoiceServer):
    """``MockVoiceServer`` whose transcriptions are chosen by the duration of the uploaded WAV:
    ``by_duration[(seconds)] = stt dict``; other requests behave as in ``MockVoiceServer``."""

    def __init__(self):
        super().__init__()
        self.by_duration = {}

    def _handle(self, request):
        path = request.url.path.rstrip("/")
        if request.method == "POST" and (path.endswith("/audio/transcriptions") or path.endswith("/v1/stt")):
            record = {"headers": dict(request.headers), "content": request.content}
            files = [
                entry
                for entries in multipart_fields(record).values()
                for entry in entries
                if entry["value"].startswith(b"RIFF")
            ]
            if files and self.by_duration:
                channels, width, rate, frames = read_wav(files[0]["value"])
                seconds = len(frames) / (channels * width) / rate
                nearest = min(self.by_duration, key=lambda known: abs(known - seconds))
                self._stt_script.appendleft(dict(self.by_duration[nearest]))
        return super()._handle(request)


class SlowVoiceServer(MockVoiceServer):
    """``MockVoiceServer`` that takes ``delay_s`` of real time per request (a slow TTS / STT)."""

    delay_s = 0.1

    def _handle(self, request):
        time.sleep(self.delay_s)
        return super()._handle(request)


def _stt(text, words=None, confidence=None):
    return {"text": text, "words": word_timings(text) if words is None else words, "confidence": confidence,
            "filler_words": []}


# =============================================================================================
# No word timings: VOICE-015 skipped, never failed
# =============================================================================================


@pytest.mark.asyncio
async def test_empty_word_timings_skip_voice015_with_exact_reason(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    project = impulse_project
    for _ in range(20):
        mock_voice_server.queue_stt(USER_LINE, words=[])
    fake_adapter.replies["no_timings"] = [
        {"text": "Your slot is booked.", "audio_bytes": _agent_audio(1.0), "latency_ms": 400.0}
    ]
    project.write_config(
        [project.scenario("no_timings", max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-015", "VOICE-029"])],
        voice_config=_voice_config(),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    speech_rate = _metrics(scenario_result)["VOICE-015"]
    assert speech_rate.skipped is True
    assert speech_rate.label == "skipped"
    assert speech_rate.score == 0.0
    assert speech_rate.reason == NO_WORD_TIMINGS
    assert _metrics(scenario_result)["VOICE-029"].skipped is False
    assert scenario_result.passed is True
    assert "VOICE-015" not in result.metric_summary


# =============================================================================================
# No confidence: never fails; cohort mean None and no STT warning
# =============================================================================================


@pytest.mark.asyncio
async def test_stt_confidence_none_never_fails_and_cohort_mean_is_none(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    server = DurationRoutedVoiceServer()
    server.install(monkeypatch)
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", "hosted-voice-SECRET-7781")
    # Round trip of the user's TTS audio (0.5 s): no confidence. Agent audio: 1.0 s in the en-IN
    # scenario (no confidence), 1.5 s in the hi-IN scenario (confidence 0.95).
    server.by_duration = {
        0.5: _stt(USER_LINE, confidence=None),
        1.0: _stt("your slot is booked", confidence=None),
        1.5: _stt("aapka slot book ho gaya", confidence=0.95),
    }
    fake_adapter.replies["no_confidence"] = [
        {"text": "Your slot is booked.", "audio_bytes": _agent_audio(1.0), "latency_ms": 400.0}
    ]
    fake_adapter.replies["with_confidence"] = [
        {"text": "Aapka slot book ho gaya.", "audio_bytes": _agent_audio(1.5), "latency_ms": 400.0}
    ]
    project = impulse_project
    project.write_config(
        [
            project.scenario("no_confidence", language="en-IN", max_turns=1, opening_user_message=USER_LINE,
                             metrics=["VOICE-029"]),
            project.scenario("with_confidence", language="hi-IN", max_turns=1, opening_user_message=USER_LINE,
                             metrics=["VOICE-029"]),
        ],
        voice_config=_voice_config(stt=HOSTED_STT),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    by_name = {s.scenario_name: s for s in result.scenario_results}

    no_confidence = by_name["no_confidence"]
    assert no_confidence.error is None
    assert all(turn.stt_confidence is None for turn in no_confidence.transcript.turns)
    assert no_confidence.passed is True  # missing confidence never fails a scenario
    assert not [m for m in no_confidence.metric_results if m.label == "poor" and not m.skipped and not m.informational]

    with_confidence = by_name["with_confidence"]
    agent_turns = [turn for turn in with_confidence.transcript.turns if turn.speaker == "agent"]
    assert [turn.stt_confidence for turn in agent_turns] == [pytest.approx(0.95)]

    cohorts = {cohort.cohort_key: cohort for cohort in result.cohorts}
    assert cohorts["en-IN/default"].mean_stt_confidence is None
    assert cohorts["en-IN/default"].stt_warning is False
    assert cohorts["hi-IN/default"].mean_stt_confidence == pytest.approx(0.95)
    assert cohorts["hi-IN/default"].stt_warning is False


# =============================================================================================
# Latency honesty: TTS / STT time never counts
# =============================================================================================


@pytest.mark.asyncio
async def test_latency_is_the_adapter_latency_not_tts_or_stt_time(impulse_project, llm_router, fake_adapter, monkeypatch):
    server = SlowVoiceServer()
    server.install(monkeypatch)
    for _ in range(20):
        server.queue_stt("your slot is booked", words=[
            {"word": "your", "start_ms": 0, "end_ms": 250, "confidence": 0.9},
            {"word": "slot", "start_ms": 250, "end_ms": 500, "confidence": 0.9},
            {"word": "is", "start_ms": 500, "end_ms": 750, "confidence": 0.9},
            {"word": "booked", "start_ms": 750, "end_ms": 1000, "confidence": 0.9},
        ])
    project = impulse_project
    persona = project.persona("latency_honesty")
    llm_router.script(persona, "And is Tuesday morning free?")
    fake_adapter.replies["latency_honesty"] = [
        {"text": "Your slot is booked.", "audio_bytes": _agent_audio(1.0), "latency_ms": 250.0},
        {"text": "Tuesday morning is free.", "audio_bytes": _agent_audio(1.0), "latency_ms": 250.0},
    ]
    project.write_config(
        [project.scenario("latency_honesty", max_turns=2, metrics=["VOICE-001", "VOICE-005"])],
        voice_config=_voice_config(),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    agent_turns = [turn for turn in scenario_result.transcript.turns if turn.speaker == "agent"]
    assert [turn.latency_ms for turn in agent_turns] == [250.0, 250.0]
    metrics = _metrics(scenario_result)
    assert metrics["VOICE-001"].raw_value == pytest.approx(250.0)
    assert metrics["VOICE-001"].is_proxy is False
    assert metrics["VOICE-005"].raw_value == pytest.approx(250.0)
    assert result.latency_profile.all_latencies_ms == [250.0, 250.0]
    assert result.latency_profile.p50_ms == pytest.approx(250.0)
    assert result.latency_profile.is_real_audio is True
    assert len(server.requests) >= 4  # the slow server was used for TTS and STT


@pytest.mark.asyncio
async def test_generic_adapter_latency_excludes_tts_and_stt_time(impulse_project, llm_router, monkeypatch):
    server = SlowVoiceServer()
    server.install(monkeypatch)
    project = impulse_project
    project.write_agent(
        """
        import io
        import math
        import struct
        import wave

        def _tone(seconds=0.6, rate=16000):
            frames = b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * 330 * i / rate)))
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
            return {"text": "Your slot is booked.", "audio": AUDIO}
        """
    )
    project.write_config(
        [project.scenario("generic_latency", max_turns=1, metrics=["VOICE-001"])], voice_config=_voice_config()
    )
    result = await _run(project.load())
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    agent_turns = [turn for turn in scenario_result.transcript.turns if turn.speaker == "agent"]
    assert len(agent_turns) == 1
    # Every TTS / STT request takes 100 ms; the agent function itself returns at once.
    assert agent_turns[0].latency_ms is not None
    assert agent_turns[0].latency_ms < 50.0
    assert len(server.requests) >= 2


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 17): Impulse's OWN STT failing (the VOICE-011 round trip of the
# user's TTS audio, or the STT of the agent's audio) never drops the agent turn and is never a
# scenario error. The turn is kept (the agent turn with the adapter's text), the run gets the warning
# "{scenario_name}: Impulse STT failed on turn {turn}: {error}" (turn = 1-based transcript
# position), and VOICE-011 (round trip) / VOICE-015 (agent audio) are skipped with
# "Impulse STT failed: {error}". A TTS failure is still a scenario error.
# =============================================================================================

STT_OUTAGE_BODY = "scripted STT outage"
STT_FAILURE = (
    f"openai_compatible STT: POST {BASE_URL}/audio/transcriptions returned HTTP 500 after 3 attempts: "
    f"{STT_OUTAGE_BODY}"
)
AGENT_REPLY = "Your slot is booked."


class STTOutageServer(DurationRoutedVoiceServer):
    """``DurationRoutedVoiceServer`` whose transcription requests fail with HTTP 500 (every attempt,
    retries included) when the uploaded WAV lasts ``fail_seconds`` (within 50 ms): the 0.5 s mock TTS
    audio is the round trip of a user turn, the agent's audio has its own length."""

    def __init__(self, fail_seconds):
        super().__init__()
        self.fail_seconds = tuple(fail_seconds)
        self.failed_requests = 0

    def _handle(self, request):
        import httpx

        path = request.url.path.rstrip("/")
        if request.method == "POST" and path.endswith("/audio/transcriptions"):
            record = {"headers": dict(request.headers), "content": request.content}
            uploads = [
                entry
                for entries in multipart_fields(record).values()
                for entry in entries
                if entry["value"].startswith(b"RIFF")
            ]
            if uploads:
                channels, width, rate, frames = read_wav(uploads[0]["value"])
                seconds = len(frames) / (channels * width) / rate
                if any(abs(seconds - target) <= 0.05 for target in self.fail_seconds):
                    self.failed_requests += 1
                    self.requests.append({"method": request.method, "url": str(request.url),
                                          "path": request.url.path, "headers": dict(request.headers),
                                          "json": None, "content": request.content})
                    return httpx.Response(500, content=STT_OUTAGE_BODY.encode(),
                                          headers={"content-type": "text/plain"})
        return super()._handle(request)


def _stt_outage_server(monkeypatch, fail_seconds):
    server = STTOutageServer(fail_seconds)
    server.by_duration = {
        0.5: _stt(USER_LINE, confidence=0.9),                                   # round trip: hears the user
        0.8: _stt("your slot is booked", words=word_timings("your slot is booked")),   # agent audio
        1.2: _stt("tuesday morning is free", words=word_timings("tuesday morning is free")),
    }
    server.install(monkeypatch)
    return server


@pytest.mark.asyncio
async def test_round_trip_stt_failure_keeps_the_turns_and_skips_voice011(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    server = _stt_outage_server(monkeypatch, fail_seconds=[0.5])
    project = impulse_project
    name = "round_trip_stt_down"
    fake_adapter.replies[name] = [{"text": AGENT_REPLY, "audio_bytes": _agent_audio(0.8), "latency_ms": 400.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["UNIV-001", "VOICE-015"])],
        voice_config=_voice_config(),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert server.failed_requests == 3   # the provider retried twice, then gave up
    assert scenario_result.error is None, scenario_result.error
    assert len(fake_adapter.sends(name)) == 1
    turns = scenario_result.transcript.turns
    assert [(turn.speaker, turn.text) for turn in turns] == [("user", USER_LINE), ("agent", AGENT_REPLY)]
    assert turns[0].stt_confidence is None   # no round-trip result for the user turn
    assert f"{name}: Impulse STT failed on turn 1: {STT_FAILURE}" in result.warnings
    metrics = _metrics(scenario_result)
    assert metrics["VOICE-011"].skipped is True
    assert metrics["VOICE-011"].label == "skipped"
    assert metrics["VOICE-011"].reason == f"Impulse STT failed: {STT_FAILURE}"
    assert metrics["VOICE-015"].skipped is False   # the agent audio was transcribed
    assert metrics["UNIV-001"].skipped is False
    assert scenario_result.passed is True


@pytest.mark.asyncio
async def test_agent_audio_stt_failure_keeps_the_agent_turn_and_skips_voice015(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    server = _stt_outage_server(monkeypatch, fail_seconds=[0.8])
    project = impulse_project
    name = "agent_stt_down"
    fake_adapter.replies[name] = [{"text": AGENT_REPLY, "audio_bytes": _agent_audio(0.8), "latency_ms": 400.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["UNIV-001", "VOICE-015"])],
        voice_config=_voice_config(),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert server.failed_requests == 3
    assert scenario_result.error is None, scenario_result.error
    assert len(fake_adapter.sends(name)) == 1
    turns = scenario_result.transcript.turns
    assert [(turn.speaker, turn.text) for turn in turns] == [("user", USER_LINE), ("agent", AGENT_REPLY)]
    assert turns[1].stt_confidence is None
    assert turns[1].latency_ms == 400.0
    assert f"{name}: Impulse STT failed on turn 2: {STT_FAILURE}" in result.warnings
    metrics = _metrics(scenario_result)
    assert metrics["VOICE-015"].skipped is True
    assert metrics["VOICE-015"].label == "skipped"
    assert metrics["VOICE-015"].reason == f"Impulse STT failed: {STT_FAILURE}"
    assert metrics["VOICE-011"].skipped is False   # the round trip worked
    assert metrics["UNIV-001"].skipped is False
    assert llm_router.judge_calls_for("UNIV-001")   # the conversation was evaluated
    assert scenario_result.passed is True


@pytest.mark.asyncio
async def test_impulse_stt_failure_warning_names_the_transcript_position(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    # Two user turns; only the second agent reply (1.2 s of audio, transcript turn 4) fails in STT.
    _stt_outage_server(monkeypatch, fail_seconds=[1.2])
    project = impulse_project
    name = "second_reply_stt_down"
    llm_router.script(project.persona(name), "is tuesday morning free")
    fake_adapter.replies[name] = [
        {"text": AGENT_REPLY, "audio_bytes": _agent_audio(0.8), "latency_ms": 400.0},
        {"text": "Tuesday morning is free.", "audio_bytes": _agent_audio(1.2), "latency_ms": 400.0},
    ]
    project.write_config(
        [project.scenario(name, max_turns=2, opening_user_message=USER_LINE, metrics=["VOICE-015"])],
        voice_config=_voice_config(),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert turns[3].text == "Tuesday morning is free."
    stt_warnings = [warning for warning in result.warnings if "Impulse STT failed" in warning]
    assert stt_warnings == [f"{name}: Impulse STT failed on turn 4: {STT_FAILURE}"]
    voice015 = _metrics(scenario_result)["VOICE-015"]
    assert voice015.skipped is True
    assert voice015.reason == f"Impulse STT failed: {STT_FAILURE}"


@pytest.mark.asyncio
async def test_tts_failure_is_still_a_scenario_error(impulse_project, llm_router, fake_adapter, monkeypatch):
    # Regression guard: only Impulse's STT is tolerated; without TTS audio the agent cannot be called.
    class TTSOutageServer(MockVoiceServer):
        def _handle(self, request):
            import httpx

            if request.method == "POST" and request.url.path.rstrip("/").endswith("/audio/speech"):
                self.requests.append({"method": request.method, "url": str(request.url), "path": request.url.path,
                                      "headers": dict(request.headers), "json": None, "content": request.content})
                return httpx.Response(400, content=b"scripted TTS outage", headers={"content-type": "text/plain"})
            return super()._handle(request)

    TTSOutageServer().install(monkeypatch)
    project = impulse_project
    name = "tts_down"
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["VOICE-029"])],
        voice_config=_voice_config(),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == (
        f"openai_compatible TTS: POST {BASE_URL}/audio/speech returned HTTP 400: scripted TTS outage"
    )
    assert scenario_result.passed is False
    assert scenario_result.metric_results == []
    assert fake_adapter.sends(name) == []
    assert not [warning for warning in result.warnings if "Impulse STT failed" in warning]


# =============================================================================================
# Audit batch 2 review fix-ups (AUDIT.md finding 17, local_hf STT): an exception from faster-whisper
# INFERENCE (``WhisperModel.transcribe(...)``, or iterating its lazy segments: e.g. CTranslate2's
# "Library cublas64_12.dll is not found or cannot be loaded" at the first decode) is a VoiceError
# "local_hf STT failed with model {model}: {type}: {message}", so a voice run with
# ``stt.provider: local_hf`` keeps the turns like any other failure of Impulse's own STT: no scenario
# error, the "Impulse STT failed on turn N" warning for each failed transcription, VOICE-011 /
# VOICE-015 skipped with "Impulse STT failed: {error}". TTS is the mock openai_compatible server.
# =============================================================================================

LOCAL_HF_STT = {"provider": "local_hf", "model": "hf-test/stt-small", "device": "cpu", "compute_type": "int8"}
CUBLAS_MISSING = "Library cublas64_12.dll is not found or cannot be loaded"
LOCAL_HF_STT_FAILURE = f"local_hf STT failed with model hf-test/stt-small: RuntimeError: {CUBLAS_MISSING}"


LOCAL_HF_HEARD = {0.5: USER_LINE, 0.8: "your slot is booked"}   # audio seconds -> what the fake STT hears


def _local_hf_whisper_module(fail_seconds, where="transcribe"):
    """A fake ``faster_whisper`` module (for ``extra_imports``) whose ``WhisperModel`` loads. Each
    transcription is routed by the length of the 16 kHz samples it gets (0.5 s: the mock TTS audio of
    a user turn, i.e. the VOICE-011 round trip; 0.8 s: the agent's audio). For a length in
    ``fail_seconds`` (within 50 ms) inference raises ``RuntimeError(CUBLAS_MISSING)``, in
    ``transcribe`` itself (``where="transcribe"``) or while the segments are iterated
    (``where="segments"``); otherwise it hears ``LOCAL_HF_HEARD`` with word timings. Returns
    ``(module, calls)``; ``calls`` lists the audio length of every transcription."""
    import types
    from importlib.machinery import ModuleSpec

    calls = []

    def failing_segments():
        raise RuntimeError(CUBLAS_MISSING)
        yield  # pragma: no cover - makes this function a generator, as faster-whisper's is

    def heard(text):
        words, start = [], 0.0
        for word in text.split():
            words.append(types.SimpleNamespace(word=" " + word, start=start, end=start + 0.4, probability=0.9))
            start += 0.4
        return [types.SimpleNamespace(text=" " + text, start=0.0, end=start, words=words)]

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, *args, **kwargs):
            seconds = len(audio) / 16000.0
            calls.append(round(seconds, 2))
            info = types.SimpleNamespace(language="en", language_probability=0.99, duration=seconds)
            if any(abs(seconds - target) <= 0.05 for target in fail_seconds):
                if where == "transcribe":
                    raise RuntimeError(CUBLAS_MISSING)
                return failing_segments(), info
            nearest = min(LOCAL_HF_HEARD, key=lambda known: abs(known - seconds))
            return iter(heard(LOCAL_HF_HEARD[nearest])), info

    module = types.ModuleType("faster_whisper")
    module.__spec__ = ModuleSpec("faster_whisper", None)
    module.WhisperModel = WhisperModel
    return module, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["transcribe", "segments"], ids=["transcribe-raises", "segment-iteration-raises"])
async def test_local_hf_stt_inference_failure_keeps_the_turns_and_skips_voice011_and_voice015(
    impulse_project, llm_router, fake_adapter, mock_voice_server, extra_imports, where
):
    # Every transcription fails: the round trip of the user's TTS audio (turn 1) and the agent's audio (turn 2).
    module, calls = _local_hf_whisper_module(fail_seconds=[0.5, 0.8], where=where)
    extra_imports.module("faster_whisper", module)
    project = impulse_project
    name = "local_stt_down"
    fake_adapter.replies[name] = [{"text": AGENT_REPLY, "audio_bytes": _agent_audio(0.8), "latency_ms": 400.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["UNIV-001", "VOICE-015"])],
        voice_config=_voice_config(stt=LOCAL_HF_STT),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    assert len(fake_adapter.sends(name)) == 1
    turns = scenario_result.transcript.turns
    assert [(turn.speaker, turn.text) for turn in turns] == [("user", USER_LINE), ("agent", AGENT_REPLY)]
    assert turns[0].stt_confidence is None
    assert turns[1].stt_confidence is None
    assert turns[1].latency_ms == 400.0
    assert len(calls) == 2   # the round trip and the agent audio, each tried once
    stt_warnings = [warning for warning in result.warnings if "Impulse STT failed" in warning]
    assert stt_warnings == [
        f"{name}: Impulse STT failed on turn 1: {LOCAL_HF_STT_FAILURE}",
        f"{name}: Impulse STT failed on turn 2: {LOCAL_HF_STT_FAILURE}",
    ]
    metrics = _metrics(scenario_result)
    for metric_id in ("VOICE-011", "VOICE-015"):
        assert metrics[metric_id].skipped is True, metric_id
        assert metrics[metric_id].label == "skipped", metric_id
        assert metrics[metric_id].reason == f"Impulse STT failed: {LOCAL_HF_STT_FAILURE}", metric_id
    assert metrics["UNIV-001"].skipped is False
    assert llm_router.judge_calls_for("UNIV-001")   # the conversation was evaluated
    assert scenario_result.passed is True


@pytest.mark.asyncio
async def test_local_hf_stt_inference_failure_on_the_agent_audio_only_skips_voice015(
    impulse_project, llm_router, fake_adapter, mock_voice_server, extra_imports
):
    # The round trip (0.5 s, turn 1) works; the agent's audio (0.8 s, turn 2) fails.
    module, calls = _local_hf_whisper_module(fail_seconds=[0.8], where="segments")
    extra_imports.module("faster_whisper", module)
    project = impulse_project
    name = "local_agent_stt_down"
    fake_adapter.replies[name] = [{"text": AGENT_REPLY, "audio_bytes": _agent_audio(0.8), "latency_ms": 400.0}]
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message=USER_LINE, metrics=["UNIV-001", "VOICE-015"])],
        voice_config=_voice_config(stt=LOCAL_HF_STT),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    turns = scenario_result.transcript.turns
    assert [(turn.speaker, turn.text) for turn in turns] == [("user", USER_LINE), ("agent", AGENT_REPLY)]
    assert len(calls) == 2
    stt_warnings = [warning for warning in result.warnings if "Impulse STT failed" in warning]
    assert stt_warnings == [f"{name}: Impulse STT failed on turn 2: {LOCAL_HF_STT_FAILURE}"]
    metrics = _metrics(scenario_result)
    assert metrics["VOICE-015"].skipped is True
    assert metrics["VOICE-015"].reason == f"Impulse STT failed: {LOCAL_HF_STT_FAILURE}"
    assert metrics["VOICE-011"].skipped is False   # the round trip worked
    assert scenario_result.passed is True
