# SPDX-License-Identifier: Apache-2.0
"""Noise simulation (EXPERIMENTAL) in ``runner.run`` (SPEC.md sections 9, 14.4 and 15).

Per user turn in voice mode with noise: simulator text -> TTS (clean) -> mix (noisy) -> ``send(text,
noisy)``. The CLEAN audio goes to Impulse's own STT for the VOICE-011 round trip; the NOISY audio is
what the agent gets, what ``Turn.audio_path`` names (``audio/<stem>_t{N}_user.wav``) and what the
stereo recording holds; the clean audio is also saved as ``audio/<stem>_t{N}_user_clean.wav``.

``CallTranscript.metadata["impulse_noise"]`` = ``{"experimental": true, "profile", "snr_db", "label",
"file" (file name only, never a path), "seed"}``; each user turn's ``metadata["impulse_voice"]`` record
adds ``clean_audio_path``, ``noise_snr_db`` (measured), ``noise_scaled`` and ``noise_offset_ms``.
None of these keys exist when the scenario has no noise: with noise off a run is unchanged.

The generic adapter passes the audio to an agent function that declares a parameter named exactly
``user_audio``; a function without it never hears the noise, and a run with noise warns about that.
Run warnings: the experimental note when any scenario used noise; the dry-run note in ``--dry-run``.
"""

import io
import math
import sys
import wave

import numpy as np
import pytest
from conftest import multipart_fields

EXPERIMENTAL_WARNING = (
    "Noise simulation is experimental: listen to the noisy caller audio in the report before trusting the results."
)
DRY_RUN_WARNING = "noise simulation is ignored in --dry-run."
GENERIC_NO_AUDIO_WARNING = (
    "the generic agent function has no user_audio parameter, so it never hears the noise; "
    "noise simulation has no effect on it."
)
BASE_URL = "http://localhost:8000/v1"
USER_LINE = "please book a slot on tuesday"
SECOND_LINE = "is ten in the morning free"

LISTENING_AGENT = """
import io
import math
import struct
import wave

RECEIVED = []

def _tone(seconds=0.6, rate=16000):
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 330 * i / rate))) for i in range(int(seconds * rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()

AUDIO = _tone()

def respond(user_text, history, user_audio=None):
    RECEIVED.append(user_audio)
    if not user_text:
        return {"text": "Hello, this is Sunrise Clinic.", "audio": AUDIO}
    return {"text": "You said: " + user_text, "heard_text": user_text, "audio": AUDIO}
"""

DEAF_AGENT = """
def respond(user_text, history):
    return {"text": "You said: " + (user_text or "")}
"""


def _voice(noise=None, save_audio=True, save_recording=False):
    config = {
        "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": BASE_URL},
        "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": BASE_URL},
        "save_audio": save_audio,
        "save_recording": save_recording,
        "timeout_s": 30,
    }
    if noise is not None:
        config["noise"] = noise
    return config


def _floats(wav_data):
    with wave.open(io.BytesIO(wav_data), "rb") as handle:
        channels = handle.getnchannels()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float64) / 32767.0
    return samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples


def _snr_db(clean_wav, noisy_wav):
    clean, noisy = _floats(clean_wav), _floats(noisy_wav)
    above = np.flatnonzero(np.abs(clean) > 0.01)
    start, end = int(above[0]), int(above[-1]) + 1
    c, n = clean[start:end], noisy[start:end]
    # The mock TTS tone is loud, so a mix may be scaled down to avoid clipping: estimate the speech gain.
    gain = float(np.dot(n, c) / np.dot(c, c))
    return 20 * math.log10(np.sqrt(np.mean((gain * c) ** 2)) / np.sqrt(np.mean((n - gain * c) ** 2)))


def _agent_module(project):
    return sys.modules[project.entrypoint.split(":", 1)[0]]


def _transcription_uploads(server):
    return [multipart_fields(record)["file"][0]["value"] for record in server.requests_to("/audio/transcriptions")]


def _run_dir(project, result):
    return project.root / ".impulse" / "runs" / result.run_id


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


def _setup(project, llm_router, agent=LISTENING_AGENT, noise=None, scenario_noise="unset", **voice):
    project.write_agent(agent)
    llm_router.script(project.persona("noisy_booking"), SECOND_LINE)
    fields = {"max_turns": 2, "opening_user_message": USER_LINE, "metrics": ["UNIV-001", "VOICE-009"]}
    if scenario_noise != "unset":
        fields["noise"] = scenario_noise
    project.write_config([project.scenario("noisy_booking", **fields)], voice_config=_voice(noise, **voice))
    return project.load()


# =============================================================================================
# Integration: generic adapter in voice mode
# =============================================================================================


@pytest.mark.asyncio
async def test_agent_gets_noisy_audio_and_voice_011_gets_clean_audio(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    cfg = _setup(project, llm_router, noise={"profile": "pink", "snr_db": 10})
    result = await _run(cfg)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    clean = mock_voice_server.wav()

    received = [audio for audio in _agent_module(project).RECEIVED]
    assert len(received) == 2
    for audio in received:
        assert isinstance(audio, bytes) and audio[:4] == b"RIFF"
        assert audio != clean
        assert abs(_snr_db(clean, audio) - 10.0) <= 0.5

    # Transcription requests: the VOICE-011 round trip of each user turn (the CLEAN TTS audio), and the
    # agent's own audio. The noisy audio is never sent to Impulse's STT.
    agent_audio = _agent_module(project).AUDIO
    uploads = _transcription_uploads(mock_voice_server)
    round_trips = [upload for upload in uploads if upload != agent_audio]
    assert len(round_trips) == 2
    assert all(upload == clean for upload in round_trips)
    assert not [upload for upload in uploads if upload in received]

    metrics = {metric.metric_id: metric for metric in scenario_result.metric_results}
    assert metrics["VOICE-011"].skipped is False
    assert metrics["VOICE-011"].informational is True
    assert metrics["VOICE-009"].skipped is False   # heard_text from the agent, which heard the noisy audio


@pytest.mark.asyncio
async def test_noisy_and_clean_files_and_metadata(impulse_project, llm_router, mock_voice_server):
    from impulse_evals.voice.noise import noise_seed

    project = impulse_project
    cfg = _setup(project, llm_router, noise={"profile": "pink", "snr_db": 10, "label": "office hum"})
    result = await _run(cfg)
    run_dir = _run_dir(project, result)
    transcript = result.scenario_results[0].transcript
    received = _agent_module(project).RECEIVED
    clean = mock_voice_server.wav()

    user_turns = [(number, turn) for number, turn in enumerate(transcript.turns, start=1) if turn.speaker == "user"]
    assert [number for number, _ in user_turns] == [1, 3]
    for (number, turn), sent in zip(user_turns, received):
        assert turn.audio_path == f"audio/noisy_booking_t{number}_user.wav"
        assert (run_dir / turn.audio_path).read_bytes() == sent
        assert (run_dir / f"audio/noisy_booking_t{number}_user_clean.wav").read_bytes() == clean

    assert transcript.metadata["impulse_noise"] == {
        "experimental": True,
        "profile": "pink",
        "snr_db": 10.0,
        "label": "office hum",
        "file": None,
        "seed": noise_seed("noisy_booking", None),
    }
    records = [record for record in transcript.metadata["impulse_voice"]["turns"] if record["speaker"] == "user"]
    assert [record["turn_index"] for record in records] == [1, 3]
    for record in records:
        assert record["clean_audio_path"] == f"audio/noisy_booking_t{record['turn_index']}_user_clean.wav"
        assert abs(record["noise_snr_db"] - 10.0) <= 0.5
        assert record["noise_scaled"] is False
        assert record["noise_offset_ms"] is None
    assert EXPERIMENTAL_WARNING in result.warnings


@pytest.mark.asyncio
async def test_scenario_seed_is_used_and_reruns_are_identical(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    project.write_agent(LISTENING_AGENT)
    llm_router.script(project.persona("seeded"), SECOND_LINE, SECOND_LINE)
    project.write_config(
        [project.scenario("seeded", max_turns=1, opening_user_message=USER_LINE, seed=99)],
        voice_config=_voice({"profile": "white", "snr_db": 5}),
    )
    first = await _run(project.load())
    second = await _run(project.load())
    received = _agent_module(project).RECEIVED
    assert len(received) == 2 and received[0] == received[1]
    assert first.scenario_results[0].transcript.metadata["impulse_noise"]["seed"] == 99
    assert second.scenario_results[0].transcript.metadata["impulse_noise"]["seed"] == 99


@pytest.mark.asyncio
async def test_recording_holds_the_noisy_audio(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    cfg = _setup(project, llm_router, noise={"profile": "white", "snr_db": 0}, save_recording=True)
    result = await _run(cfg)
    transcript = result.scenario_results[0].transcript
    recording = (_run_dir(project, result) / transcript.recording_path).read_bytes()
    with wave.open(io.BytesIO(recording), "rb") as handle:
        channels = handle.getnchannels()
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").reshape(-1, channels)
    caller = frames[:, 0].astype(np.float64) / 32767.0
    clean = _floats(mock_voice_server.wav())
    # The caller channel starts with the first user turn: noisy, so not equal to the clean clip.
    head = caller[: len(clean)]
    assert not np.allclose(head, clean[: len(head)], atol=1e-3)


@pytest.mark.asyncio
async def test_file_noise_records_the_file_name_and_offset(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    rng = np.random.default_rng(1)
    pcm = np.clip(np.round(rng.uniform(-0.3, 0.3, 44100 * 3) * 32767), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(pcm.tobytes())
    (project.root / "noise").mkdir()
    (project.root / "noise" / "traffic.wav").write_bytes(buffer.getvalue())
    cfg = _setup(project, llm_router, noise={"profile": "file", "file": "noise/traffic.wav", "snr_db": 5, "label": "traffic"})
    result = await _run(cfg)
    transcript = result.scenario_results[0].transcript
    meta = transcript.metadata["impulse_noise"]
    assert meta["file"] == "traffic.wav"
    assert meta["label"] == "traffic"
    records = [record for record in transcript.metadata["impulse_voice"]["turns"] if record["speaker"] == "user"]
    assert all(isinstance(record["noise_offset_ms"], int) for record in records)
    assert str(project.root) not in result.model_dump_json()
    for audio in _agent_module(project).RECEIVED:
        assert abs(_snr_db(mock_voice_server.wav(), audio) - 5.0) <= 0.5


# =============================================================================================
# Noise off: nothing changes
# =============================================================================================


@pytest.mark.asyncio
async def test_without_noise_the_agent_gets_the_clean_tts_bytes(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    cfg = _setup(project, llm_router)
    result = await _run(cfg)
    clean = mock_voice_server.wav()
    assert _agent_module(project).RECEIVED == [clean, clean]
    transcript = result.scenario_results[0].transcript
    assert "impulse_noise" not in transcript.metadata
    for record in transcript.metadata["impulse_voice"]["turns"]:
        assert not [key for key in record if key.startswith("noise_") or key == "clean_audio_path"]
    files = {path.name for path in (_run_dir(project, result) / "audio").iterdir()}
    assert not [name for name in files if name.endswith("_clean.wav")]
    assert not [warning for warning in result.warnings if "noise" in warning.lower()]


@pytest.mark.asyncio
async def test_scenario_noise_false_overrides_the_default(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    cfg = _setup(project, llm_router, noise={"profile": "white", "snr_db": 5}, scenario_noise=False)
    result = await _run(cfg)
    assert _agent_module(project).RECEIVED == [mock_voice_server.wav()] * 2
    assert "impulse_noise" not in result.scenario_results[0].transcript.metadata
    assert EXPERIMENTAL_WARNING not in result.warnings


@pytest.mark.asyncio
async def test_noise_without_save_audio_writes_no_files(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    cfg = _setup(project, llm_router, noise={"profile": "pink", "snr_db": 10}, save_audio=False)
    result = await _run(cfg)
    transcript = result.scenario_results[0].transcript
    assert all(turn.audio_path is None for turn in transcript.turns)
    records = [record for record in transcript.metadata["impulse_voice"]["turns"] if record["speaker"] == "user"]
    assert all(record["clean_audio_path"] is None for record in records)
    assert all(abs(record["noise_snr_db"] - 10.0) <= 0.5 for record in records)


# =============================================================================================
# Other adapters and warnings
# =============================================================================================


@pytest.mark.asyncio
async def test_custom_style_adapter_receives_noisy_audio(impulse_project, llm_router, mock_voice_server, fake_adapter):
    project = impulse_project
    cfg = _setup(project, llm_router, noise={"profile": "brown", "snr_db": 10})
    await _run(cfg, adapter_factory=fake_adapter)
    clean = mock_voice_server.wav()
    sends = fake_adapter.sends("noisy_booking")
    assert len(sends) == 2
    for _, audio in sends:
        assert audio != clean
        assert abs(_snr_db(clean, audio) - 10.0) <= 0.5


@pytest.mark.asyncio
async def test_custom_style_adapter_without_noise_gets_clean_bytes(impulse_project, llm_router, mock_voice_server, fake_adapter):
    project = impulse_project
    cfg = _setup(project, llm_router)
    await _run(cfg, adapter_factory=fake_adapter)
    assert [audio for _, audio in fake_adapter.sends("noisy_booking")] == [mock_voice_server.wav()] * 2


@pytest.mark.asyncio
async def test_generic_function_without_user_audio_warns(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    cfg = _setup(project, llm_router, agent=DEAF_AGENT, noise={"profile": "pink", "snr_db": 10})
    result = await _run(cfg)
    assert result.scenario_results[0].error is None
    assert GENERIC_NO_AUDIO_WARNING in result.warnings


@pytest.mark.asyncio
async def test_generic_function_without_user_audio_and_no_noise_does_not_warn(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    cfg = _setup(project, llm_router, agent=DEAF_AGENT)
    result = await _run(cfg)
    assert GENERIC_NO_AUDIO_WARNING not in result.warnings


@pytest.mark.asyncio
async def test_generic_function_with_kwargs_only_gets_no_audio(impulse_project, llm_router, mock_voice_server):
    project = impulse_project
    agent = """
SEEN = []

def respond(user_text, history, **kwargs):
    SEEN.append(dict(kwargs))
    return {"text": "ok"}
"""
    cfg = _setup(project, llm_router, agent=agent)
    await _run(cfg)
    assert _agent_module(project).SEEN == [{}, {}]


@pytest.mark.asyncio
async def test_generic_function_with_user_audio_in_text_mode_gets_none(impulse_project, llm_router):
    project = impulse_project
    project.write_agent(LISTENING_AGENT)
    project.write_config([project.scenario("text_only", max_turns=1, opening_user_message=USER_LINE)])
    await _run(project.load())
    assert _agent_module(project).RECEIVED == [None]


@pytest.mark.asyncio
async def test_dry_run_ignores_noise_with_one_warning(impulse_project, mock_llm):
    project = impulse_project
    project.write_agent(LISTENING_AGENT)
    project.write_config(
        [project.scenario("a", opening_user_message=USER_LINE), project.scenario("b", opening_user_message=USER_LINE)],
        voice_config=_voice({"profile": "pink", "snr_db": 10}),
    )
    result = await _run(project.load(), dry_run=True)
    assert result.warnings.count(DRY_RUN_WARNING) == 1
    assert EXPERIMENTAL_WARNING not in result.warnings
    for scenario_result in result.scenario_results:
        assert "impulse_noise" not in scenario_result.transcript.metadata
    assert mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_dry_run_without_noise_has_no_noise_warning(impulse_project, mock_llm):
    project = impulse_project
    project.write_agent(LISTENING_AGENT)
    project.write_config([project.scenario("a", opening_user_message=USER_LINE)], voice_config=_voice())
    result = await _run(project.load(), dry_run=True)
    assert DRY_RUN_WARNING not in result.warnings


def test_cli_dry_run_of_the_noise_example(impulse_project, mock_llm, cli):
    from pathlib import Path

    config = Path(__file__).resolve().parent.parent / "examples" / "config.noise.yaml"
    result = cli.invoke("run", config, "--dry-run")
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert "4 scenarios, 4 passed" in text
    assert text.count(DRY_RUN_WARNING) == 1
    assert mock_llm.call_count == 0


def test_cli_dry_run_of_examples_config_is_unchanged(impulse_project, mock_llm, cli):
    from pathlib import Path

    config = Path(__file__).resolve().parent.parent / "examples" / "config.yaml"
    result = cli.invoke("run", config, "--dry-run")
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert "6 scenarios, 6 passed" in text
    assert "noise" not in text.lower()
