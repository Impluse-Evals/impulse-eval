# SPDX-License-Identifier: Apache-2.0
"""Phase 11: voice providers (SPEC.md sections 14, 14.1, 14.2; section 18.1 ``voice.tts`` / ``voice.stt``).

``get_tts_provider(cfg)`` / ``get_stt_provider(cfg)`` take the loaded ``voice_config`` block
(``ProjectConfig.voice_config``, a ``VoiceConfig``: the ``tts`` / ``stt`` sub-blocks plus
``timeout_s``) and return a ``TTSProvider`` / ``STTProvider``:

* ``await synthesize(text, language, voice) -> bytes`` (WAV)
* ``await transcribe(audio, language) -> STTResult`` (``audio`` is WAV bytes)

HTTP providers are served by ``mock_voice_server`` (``httpx.MockTransport``, no socket). ``local_hf``
runs against fake ``transformers`` / ``faster_whisper`` / ``torch`` modules installed in
``sys.modules`` (the real libraries are never imported). Contracts: DECISIONS.md, "Stage B group
9-13 part B: phase 11 (tests)".
"""

import array
import math
import subprocess
import sys
import types
from importlib.machinery import ModuleSpec

import pytest
from conftest import (
    REPO_ROOT,
    multipart_fields,
    pcm_tone,
    read_wav,
    sine_wav,
    wav_bytes,
)

pytestmark = pytest.mark.phase11

BASE_URL = "http://localhost:8000/v1"
HOSTED_URL = "http://localhost:9000"
HOSTED_KEY = "hosted-voice-SECRET-4242"
LOCAL_EXTRA_MISSING = "local_hf needs the 'local' extra: pip install impulse-evals[local]"

OPENAI_TTS = {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": BASE_URL}
OPENAI_STT = {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": BASE_URL}
HOSTED_TTS = {"provider": "impulse_hosted", "base_url": HOSTED_URL}
HOSTED_STT = {"provider": "impulse_hosted", "base_url": HOSTED_URL}
LOCAL_TTS = {"provider": "local_hf", "model": "hf-test/tts-default", "device": "cpu"}
LOCAL_STT = {"provider": "local_hf", "model": "hf-test/stt-small", "device": "cpu", "compute_type": "int8"}

EXACT_WORDS = [  # quarter-second boundaries: exact in binary, so seconds <-> ms never rounds
    {"word": "book", "start_ms": 0, "end_ms": 250, "confidence": 0.9},
    {"word": "a", "start_ms": 250, "end_ms": 500, "confidence": 0.9},
    {"word": "slot", "start_ms": 750, "end_ms": 1250, "confidence": 0.9},
]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _voice_cfg(project, tts=None, stt=None, **extra):
    """Write a config with a ``voice_config`` block and return the loaded ``VoiceConfig``."""
    voice = {
        "tts": dict(tts or OPENAI_TTS),
        "stt": dict(stt or OPENAI_STT),
        "save_audio": False,
        "save_recording": False,
        "timeout_s": 30,
    }
    voice.update(extra)
    project.write_config([project.scenario("voice_probe")], voice_config=voice)
    return project.load().voice_config


def _tts(voice_cfg):
    from impulse_evals.voice.tts import get_tts_provider

    return get_tts_provider(voice_cfg)


def _stt(voice_cfg):
    from impulse_evals.voice.stt import get_stt_provider

    return get_stt_provider(voice_cfg)


def _stereo_wav(seconds, sample_rate, amplitude=0.5):
    """Stereo 16-bit WAV with the same sine tone on both channels."""
    mono = array.array("h")
    mono.frombytes(pcm_tone(seconds, sample_rate, amplitude=amplitude))
    stereo = array.array("h", (sample for sample in mono for _ in range(2)))
    return wav_bytes(stereo.tobytes(), sample_rate, channels=2)


def _field_text(fields, name):
    values = fields.get(name) or []
    return [entry["value"].decode("utf-8") for entry in values]


# ---------------------------------------------------------------------------------------------
# fake local libraries (transformers, faster_whisper, torch)
# ---------------------------------------------------------------------------------------------


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, None)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class FakeTransformers:
    """``transformers.pipeline(task, model=..., ...)`` returning a text-to-speech callable that yields
    ``{"audio": float32 ndarray, "sampling_rate": int}`` (the Hugging Face TTS pipeline output)."""

    def __init__(self, sampling_rate=22050, seconds=0.4, two_dimensional=True):
        import numpy

        self.sampling_rate = sampling_rate
        count = int(round(seconds * sampling_rate))
        wave_ = 0.5 * numpy.sin(2 * math.pi * 440.0 * numpy.arange(count) / sampling_rate)
        self.audio = wave_.astype(numpy.float32)
        if two_dimensional:
            self.audio = self.audio.reshape(1, -1)  # e.g. VITS / MMS-TTS return shape (1, n)
        self.count = count
        self.pipeline_calls = []
        self.synth_calls = []

    def module(self):
        def pipeline(*args, **kwargs):
            task = kwargs.get("task", args[0] if args else None)
            model = kwargs.get("model", args[1] if len(args) > 1 else None)
            self.pipeline_calls.append({"task": task, "model": model, "args": args, "kwargs": kwargs})

            def synthesize(*call_args, **call_kwargs):
                self.synth_calls.append({"model": model, "args": call_args, "kwargs": call_kwargs})
                return {"audio": self.audio.copy(), "sampling_rate": self.sampling_rate}

            return synthesize

        return _module("transformers", pipeline=pipeline)


class FakeFasterWhisper:
    """``faster_whisper.WhisperModel(model, ...)`` whose ``transcribe(audio, **kwargs)`` returns
    ``(segments, info)`` like faster-whisper (segments with ``.text`` and ``.words``)."""

    def __init__(self, segments):
        self.segments = segments
        self.init_calls = []
        self.transcribe_calls = []

    def module(self):
        outer = self

        class WhisperModel:
            def __init__(self, *args, **kwargs):
                model = kwargs.get("model_size_or_path", args[0] if args else None)
                outer.init_calls.append({"model": model, "args": args, "kwargs": kwargs})

            def transcribe(self, audio, *args, **kwargs):
                outer.transcribe_calls.append({"audio": audio, "args": args, "kwargs": kwargs})
                info = types.SimpleNamespace(language="hi", language_probability=0.99, duration=1.0)
                return iter(list(outer.segments)), info

        return _module("faster_whisper", WhisperModel=WhisperModel)


def _word(text, start, end, probability):
    return types.SimpleNamespace(word=text, start=start, end=end, probability=probability)


def _segment(text, words):
    start = words[0].start if words else 0.0
    end = words[-1].end if words else 0.0
    return types.SimpleNamespace(text=text, start=start, end=end, words=words)


def _fake_torch():
    cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    mps = types.SimpleNamespace(is_available=lambda: False)
    return _module("torch", cuda=cuda, backends=types.SimpleNamespace(mps=mps))


@pytest.fixture
def local_libs(extra_imports):
    """Install fake ``transformers``, ``faster_whisper`` and ``torch`` modules; returns a namespace
    with ``tts`` (FakeTransformers) and ``stt`` (FakeFasterWhisper) to adjust or inspect."""
    fakes = types.SimpleNamespace(
        tts=FakeTransformers(),
        stt=FakeFasterWhisper(
            [
                _segment(
                    " book a slot",
                    [_word(" book", 0.0, 0.5, 0.9), _word(" a", 0.5, 0.75, 0.6), _word(" slot", 0.75, 1.5, 0.9)],
                )
            ]
        ),
    )

    def install():
        extra_imports.module("transformers", fakes.tts.module())
        extra_imports.module("faster_whisper", fakes.stt.module())
        extra_imports.module("torch", _fake_torch())

    fakes.install = install
    install()
    return fakes


# =============================================================================================
# Factory
# =============================================================================================


@pytest.mark.parametrize(
    "tts, stt",
    [(OPENAI_TTS, OPENAI_STT), (HOSTED_TTS, HOSTED_STT), (LOCAL_TTS, LOCAL_STT)],
    ids=["openai_compatible", "impulse_hosted", "local_hf"],
)
def test_factory_returns_provider_for_each_provider(impulse_project, local_libs, monkeypatch, tts, stt):
    from impulse_evals.voice.stt import STTProvider
    from impulse_evals.voice.tts import TTSProvider

    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=tts, stt=stt)
    assert isinstance(_tts(cfg), TTSProvider)
    assert isinstance(_stt(cfg), STTProvider)


@pytest.mark.parametrize("side", ["tts", "stt"])
def test_unknown_provider_is_config_error_at_load(impulse_project, side):
    from impulse_evals.config import ConfigError

    blocks = {"tts": dict(OPENAI_TTS), "stt": dict(OPENAI_STT)}
    blocks[side]["provider"] = "acme_voice"
    with pytest.raises(ConfigError) as info:
        _voice_cfg(impulse_project, **blocks)
    assert info.value.field == f"voice_config.{side}.provider"


@pytest.mark.parametrize("side", ["tts", "stt"])
def test_factory_rejects_unknown_provider_with_config_error(impulse_project, side):
    from impulse_evals.config import ConfigError

    cfg = _voice_cfg(impulse_project)
    getattr(cfg, side).provider = "acme_voice"  # an unvalidated value handed to the factory
    factory = _tts if side == "tts" else _stt
    with pytest.raises(ConfigError) as info:
        factory(cfg)
    assert info.value.field == f"voice_config.{side}.provider"


@pytest.mark.parametrize("side", ["tts", "stt"])
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
def test_http_provider_without_base_url_is_config_error(impulse_project, side, provider):
    from impulse_evals.config import ConfigError

    blocks = {"tts": dict(OPENAI_TTS), "stt": dict(OPENAI_STT)}
    blocks[side] = {"provider": provider, "model": "impulse-test-model"}
    with pytest.raises(ConfigError) as info:
        _voice_cfg(impulse_project, **blocks)
    assert info.value.field == f"voice_config.{side}.base_url"


# =============================================================================================
# openai_compatible
# =============================================================================================


@pytest.mark.asyncio
async def test_openai_compatible_tts_request_shape(impulse_project, mock_voice_server):
    cfg = _voice_cfg(impulse_project)
    audio = await _tts(cfg).synthesize("Book a slot for Tuesday", "en-IN", "alloy")
    assert audio == mock_voice_server.wav()
    requests = mock_voice_server.requests
    assert len(requests) == 1
    request = requests[0]
    assert request["method"] == "POST"
    assert request["url"] == BASE_URL + "/audio/speech"
    assert "json" in request["headers"].get("content-type", "")
    body = request["json"]
    assert body["model"] == "impulse-test-tts"
    assert body["input"] == "Book a slot for Tuesday"
    assert body["voice"] == "alloy"
    assert body["response_format"] == "wav"


@pytest.mark.asyncio
async def test_openai_compatible_sends_bearer_key_from_api_key_env(impulse_project, mock_voice_server, monkeypatch):
    monkeypatch.setenv("IMPULSE_TEST_VOICE_KEY", "tts-SECRET-0042")
    tts = dict(OPENAI_TTS, api_key_env="IMPULSE_TEST_VOICE_KEY")
    stt = dict(OPENAI_STT, api_key_env="IMPULSE_TEST_VOICE_KEY")
    cfg = _voice_cfg(impulse_project, tts=tts, stt=stt)
    await _tts(cfg).synthesize("Hello", "en-IN", None)
    await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert len(mock_voice_server.requests) == 2
    for request in mock_voice_server.requests:
        assert request["headers"].get("authorization") == "Bearer tts-SECRET-0042"


@pytest.mark.asyncio
async def test_openai_compatible_stt_request_shape_and_result(impulse_project, mock_voice_server):
    from impulse_evals.models import STTResult

    cfg = _voice_cfg(impulse_project)
    mock_voice_server.queue_stt("book a slot", words=EXACT_WORDS)
    audio = sine_wav(0.5)
    result = await _stt(cfg).transcribe(audio, "hi-IN")

    requests = mock_voice_server.requests
    assert len(requests) == 1
    request = requests[0]
    assert request["method"] == "POST"
    assert request["url"] == BASE_URL + "/audio/transcriptions"
    assert "multipart/form-data" in request["headers"].get("content-type", "")
    fields = multipart_fields(request)
    assert _field_text(fields, "model") == ["impulse-test-stt"]
    assert _field_text(fields, "response_format") == ["verbose_json"]
    assert "word" in _field_text(fields, "timestamp_granularities[]")
    assert _field_text(fields, "language") == ["hi"]  # primary subtag only (SPEC.md section 14.1)
    files = fields.get("file") or []
    assert len(files) == 1 and files[0]["value"].startswith(b"RIFF")  # a WAV upload

    assert isinstance(result, STTResult)
    assert result.text == "book a slot"
    assert [(w.word, w.start_ms, w.end_ms) for w in result.words] == [
        ("book", 0, 250),
        ("a", 250, 500),
        ("slot", 750, 1250),
    ]
    # The verbose_json response carries no confidence values, so none is reported.
    assert result.confidence is None


@pytest.mark.asyncio
@pytest.mark.parametrize("language, expected", [("hi-IN", "hi"), ("en-IN", "en"), ("hi-Latn-IN", "hi")])
async def test_openai_compatible_stt_uses_primary_language_subtag(impulse_project, mock_voice_server, language, expected):
    cfg = _voice_cfg(impulse_project)
    await _stt(cfg).transcribe(sine_wav(0.5), language)
    fields = multipart_fields(mock_voice_server.requests[0])
    assert _field_text(fields, "language") == [expected]


@pytest.mark.asyncio
async def test_openai_compatible_stt_empty_words_degrade_to_empty_list(impulse_project, mock_voice_server):
    cfg = _voice_cfg(impulse_project)
    mock_voice_server.queue_stt("hello there", words=[])
    result = await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert result.text == "hello there"
    assert result.words == []
    assert result.confidence is None


# =============================================================================================
# Retries: 2x on 429 / 5xx (both HTTP providers)
# =============================================================================================


def _http_cfg(project, provider, monkeypatch):
    if provider == "impulse_hosted":
        monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
        return _voice_cfg(project, tts=HOSTED_TTS, stt=HOSTED_STT)
    return _voice_cfg(project)


TTS_PATH = {"openai_compatible": "/audio/speech", "impulse_hosted": "/v1/tts"}
STT_PATH = {"openai_compatible": "/audio/transcriptions", "impulse_hosted": "/v1/stt"}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_tts_retries_twice_then_succeeds(impulse_project, mock_voice_server, monkeypatch, provider, status):
    cfg = _http_cfg(impulse_project, provider, monkeypatch)
    mock_voice_server.queue_status(status, count=2)
    audio = await _tts(cfg).synthesize("Hello", "en-IN", None)
    assert audio == mock_voice_server.wav()
    assert len(mock_voice_server.requests_to(TTS_PATH[provider])) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
async def test_tts_gives_up_after_two_retries(impulse_project, mock_voice_server, monkeypatch, provider):
    cfg = _http_cfg(impulse_project, provider, monkeypatch)
    tts = _tts(cfg)
    mock_voice_server.queue_status(503, count=10)
    with pytest.raises(Exception):
        await tts.synthesize("Hello", "en-IN", None)
    assert len(mock_voice_server.requests_to(TTS_PATH[provider])) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
async def test_stt_retries_twice_on_429_then_succeeds(impulse_project, mock_voice_server, monkeypatch, provider):
    cfg = _http_cfg(impulse_project, provider, monkeypatch)
    mock_voice_server.queue_status(429, count=2)
    mock_voice_server.queue_stt("hello there", words=EXACT_WORDS[:2])
    result = await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert result.text == "hello there"
    assert len(mock_voice_server.requests_to(STT_PATH[provider])) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
async def test_client_error_is_not_retried(impulse_project, mock_voice_server, monkeypatch, provider):
    cfg = _http_cfg(impulse_project, provider, monkeypatch)
    tts = _tts(cfg)
    mock_voice_server.queue_status(400, count=10)
    with pytest.raises(Exception):
        await tts.synthesize("Hello", "en-IN", None)
    assert len(mock_voice_server.requests_to(TTS_PATH[provider])) == 1


# =============================================================================================
# impulse_hosted
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language, expected_voice", [("en-IN", "af_sarah"), ("hi-IN", "hi_aryan"), ("en-US", "af_bella")]
)
async def test_impulse_hosted_tts_shape_auth_and_default_voice(
    impulse_project, mock_voice_server, monkeypatch, language, expected_voice
):
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=HOSTED_TTS, stt=HOSTED_STT)
    audio = await _tts(cfg).synthesize("Namaste, how can I help?", language, None)
    assert audio == mock_voice_server.wav()
    request = mock_voice_server.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == HOSTED_URL + "/v1/tts"
    assert request["headers"].get("authorization") == f"Bearer {HOSTED_KEY}"
    assert request["json"]["text"] == "Namaste, how can I help?"
    assert request["json"]["language"] == language
    assert request["json"]["voice"] == expected_voice


@pytest.mark.asyncio
async def test_impulse_hosted_tts_explicit_voice_wins(impulse_project, mock_voice_server, monkeypatch):
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=HOSTED_TTS, stt=HOSTED_STT)
    await _tts(cfg).synthesize("Hello", "en-IN", "custom_voice")
    assert mock_voice_server.requests[0]["json"]["voice"] == "custom_voice"


@pytest.mark.asyncio
async def test_impulse_hosted_stt_shape_auth_and_result(impulse_project, mock_voice_server, monkeypatch):
    from impulse_evals.models import FillerWord, STTResult, WordTiming

    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=HOSTED_TTS, stt=HOSTED_STT)
    mock_voice_server.queue_stt(
        "um book a slot", words=EXACT_WORDS, confidence=0.77, filler_words=[{"word": "um", "start_ms": 0}]
    )
    result = await _stt(cfg).transcribe(sine_wav(0.5), "hi-IN")
    request = mock_voice_server.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == HOSTED_URL + "/v1/stt"
    assert "multipart/form-data" in request["headers"].get("content-type", "")
    assert request["headers"].get("authorization") == f"Bearer {HOSTED_KEY}"
    uploads = [entry for entries in multipart_fields(request).values() for entry in entries
               if entry["value"].startswith(b"RIFF")]
    assert len(uploads) == 1  # the audio goes up as one WAV part
    assert isinstance(result, STTResult)
    assert result.text == "um book a slot"
    assert result.confidence == pytest.approx(0.77)
    assert result.words == [WordTiming(**word) for word in EXACT_WORDS]
    assert result.filler_words == [FillerWord(word="um", start_ms=0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["ta-IN", "fr-FR"])
async def test_impulse_hosted_unsupported_language_raises_clear_error_without_request(
    impulse_project, mock_voice_server, monkeypatch, language
):
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=HOSTED_TTS, stt=HOSTED_STT)
    tts, stt = _tts(cfg), _stt(cfg)
    with pytest.raises(Exception) as tts_info:
        await tts.synthesize("Vanakkam", language, None)
    with pytest.raises(Exception) as stt_info:
        await stt.transcribe(sine_wav(0.5), language)
    for info in (tts_info, stt_info):
        assert language in str(info.value)
    assert mock_voice_server.requests == []


# =============================================================================================
# local_hf (mocked libraries; the real transformers / faster-whisper / torch are never imported)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("two_dimensional", [True, False], ids=["audio_shape_1xN", "audio_shape_N"])
async def test_local_hf_tts_writes_mono_16_bit_wav(impulse_project, local_libs, two_dimensional):
    local_libs.tts = FakeTransformers(sampling_rate=22050, seconds=0.4, two_dimensional=two_dimensional)
    local_libs.install()
    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    audio = await _tts(cfg).synthesize("hello there", "en-IN", None)
    channels, width, rate, frames = read_wav(audio)
    assert channels == 1
    assert width == 2
    duration = len(frames) / 2 / rate
    assert duration == pytest.approx(0.4, abs=0.01)
    peak = max(abs(sample) for sample in array.array("h", frames))
    assert 0.45 * 32767 <= peak <= 0.55 * 32767  # float [-1, 1] scaled to 16-bit
    calls = local_libs.tts.synth_calls
    assert len(calls) == 1
    assert "hello there" in list(calls[0]["args"]) + list(calls[0]["kwargs"].values())
    assert [call["model"] for call in local_libs.tts.pipeline_calls] == ["hf-test/tts-default"]


@pytest.mark.asyncio
async def test_local_hf_tts_caches_pipeline_and_uses_models_by_language(impulse_project, local_libs):
    tts = dict(LOCAL_TTS, models_by_language={"hi-IN": "hf-test/tts-hindi"})
    cfg = _voice_cfg(impulse_project, tts=tts, stt=LOCAL_STT)
    provider = _tts(cfg)
    await provider.synthesize("hello", "en-IN", None)
    await provider.synthesize("hello again", "en-IN", None)
    await provider.synthesize("namaste", "hi-IN", None)
    await provider.synthesize("namaste phir se", "hi-IN", None)
    models = [call["model"] for call in local_libs.tts.pipeline_calls]
    assert sorted(models) == ["hf-test/tts-default", "hf-test/tts-hindi"]  # one pipeline per model
    assert [call["model"] for call in local_libs.tts.synth_calls] == [
        "hf-test/tts-default",
        "hf-test/tts-default",
        "hf-test/tts-hindi",
        "hf-test/tts-hindi",
    ]


@pytest.mark.asyncio
async def test_local_hf_stt_resamples_and_maps_words(impulse_project, local_libs):
    import numpy

    from impulse_evals.models import STTResult

    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    result = await _stt(cfg).transcribe(_stereo_wav(1.0, 8000), "hi-IN")

    assert [call["model"] for call in local_libs.stt.init_calls] == ["hf-test/stt-small"]
    assert len(local_libs.stt.transcribe_calls) == 1
    call = local_libs.stt.transcribe_calls[0]
    audio = call["audio"]
    assert isinstance(audio, numpy.ndarray)
    assert audio.ndim == 1  # mono
    assert numpy.issubdtype(audio.dtype, numpy.floating)
    assert abs(len(audio) - 16000) <= 16  # 1.0 s resampled to 16 kHz
    assert 0.3 <= float(numpy.max(numpy.abs(audio))) <= 1.0
    assert call["kwargs"].get("word_timestamps") is True
    assert call["kwargs"].get("language") == "hi"  # primary subtag only

    assert isinstance(result, STTResult)
    assert result.text.strip() == "book a slot"
    assert [(w.word.strip(), w.start_ms, w.end_ms) for w in result.words] == [
        ("book", 0, 500),
        ("a", 500, 750),
        ("slot", 750, 1500),
    ]
    assert result.confidence == pytest.approx(0.8)  # mean word probability


@pytest.mark.asyncio
async def test_local_hf_stt_without_words_has_no_confidence(impulse_project, local_libs):
    local_libs.stt = FakeFasterWhisper([_segment(" hello", [])])
    local_libs.install()
    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    result = await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert result.words == []
    assert result.confidence is None


@pytest.mark.asyncio
async def test_local_hf_tts_missing_extra_gives_exact_error(impulse_project, extra_imports):
    extra_imports.missing("transformers")
    extra_imports.module("torch", _fake_torch())
    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    with pytest.raises(Exception) as info:
        provider = _tts(cfg)
        await provider.synthesize("hello", "en-IN", None)
    assert LOCAL_EXTRA_MISSING in str(info.value)


@pytest.mark.asyncio
async def test_local_hf_stt_missing_extra_gives_exact_error(impulse_project, extra_imports):
    extra_imports.missing("faster_whisper")
    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    with pytest.raises(Exception) as info:
        provider = _stt(cfg)
        await provider.transcribe(sine_wav(0.5), "en-IN")
    assert LOCAL_EXTRA_MISSING in str(info.value)


def test_voice_modules_import_without_optional_libraries():
    """``impulse_evals.voice.tts`` / ``.stt`` import in a fresh interpreter in which the optional
    voice libraries cannot be imported (they are lazy-imported, SPEC.md section 5)."""
    code = (
        "import sys\n"
        "for name in ('transformers', 'faster_whisper', 'torch', 'soundfile', 'sounddevice', 'pydub'):\n"
        "    sys.modules[name] = None\n"
        "import impulse_evals.voice.tts, impulse_evals.voice.stt\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")


# =============================================================================================
# Audit batch 1 (AUDIT.md finding 5): a Basic-auth password inside ``base_url`` never appears in a
# provider error (HTTP error status or transport error); the host stays visible and httpx still
# sends the credentials (Basic auth keeps working).
# =============================================================================================

from conftest import MockVoiceServer  # noqa: E402

BASE_URL_PASSWORD = "PASSWORD123"
USERINFO_BASE_URL = f"http://user:{BASE_URL_PASSWORD}@localhost:8000/v1"
USERINFO_HOSTED_URL = f"http://user:{BASE_URL_PASSWORD}@localhost:9000"


class _ConnectErrorServer(MockVoiceServer):
    """Every request fails at the transport level, as an unreachable proxied TTS / STT server does."""

    def _handle(self, request):
        import httpx

        self.requests.append(
            {"method": request.method, "url": str(request.url), "path": request.url.path,
             "headers": dict(request.headers), "json": None, "content": request.content}
        )
        raise httpx.ConnectError("connection refused by upstream proxy")


def _userinfo_cfg(project, provider, monkeypatch):
    if provider == "impulse_hosted":
        monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
        tts, stt = dict(HOSTED_TTS, base_url=USERINFO_HOSTED_URL), dict(HOSTED_STT, base_url=USERINFO_HOSTED_URL)
    else:
        tts, stt = dict(OPENAI_TTS, base_url=USERINFO_BASE_URL), dict(OPENAI_STT, base_url=USERINFO_BASE_URL)
    return _voice_cfg(project, tts=tts, stt=stt)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http-401", "connect-error"])
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
async def test_http_error_messages_never_contain_the_base_url_password(impulse_project, monkeypatch, provider, failure):
    if failure == "http-401":
        server = MockVoiceServer()
        server.queue_status(401, count=10, body={"error": "bad credentials"})   # 4xx: not retried
    else:
        server = _ConnectErrorServer()
    server.install(monkeypatch)
    cfg = _userinfo_cfg(impulse_project, provider, monkeypatch)
    with pytest.raises(Exception) as tts_info:
        await _tts(cfg).synthesize("Hello", "en-IN", None)
    with pytest.raises(Exception) as stt_info:
        await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert server.requests, "the requests must reach the (mock) transport"
    for info in (tts_info, stt_info):
        message = str(info.value)
        assert BASE_URL_PASSWORD not in message, message
        assert "localhost" in message, message   # the host is kept; only the credentials go
        assert HOSTED_KEY not in message


@pytest.mark.asyncio
async def test_tts_provider_returns_server_bytes_verbatim_with_userinfo_base_url(impulse_project, mock_voice_server):
    # Regression guard: the credentials still reach the server as Basic auth and the WAV is untouched.
    cfg = _voice_cfg(
        impulse_project, tts=dict(OPENAI_TTS, base_url=USERINFO_BASE_URL), stt=dict(OPENAI_STT, base_url=USERINFO_BASE_URL)
    )
    audio = await _tts(cfg).synthesize("Hello", "en-IN", None)
    assert audio == mock_voice_server.wav()
    request = mock_voice_server.requests[0]
    assert request["path"].endswith("/audio/speech")
    assert request["headers"].get("authorization", "").startswith("Basic ")


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 19): local_hf TTS needs torch (transformers' text-to-speech
# pipelines run on it), which the 'local' extra deliberately does not install. The provider checks
# ``importlib.util.find_spec("torch")`` (never importing torch) when it is built / its language is
# checked, i.e. before the first paid call, and raises ConfigError("voice_config.tts.provider",
# TORCH_MISSING). STT (faster-whisper) needs no torch.
# =============================================================================================

import importlib.abc
import importlib.machinery
import importlib.util

TORCH_MISSING = (
    "local_hf TTS needs PyTorch, which the 'local' extra does not install: pip install torch "
    "(see https://pytorch.org/get-started/locally/)"
)


@pytest.fixture
def local_libs_without_torch(local_libs, extra_imports):
    """``local_libs`` (fake transformers and faster_whisper) with torch NOT importable."""
    extra_imports.missing("torch")
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("transformers") is not None
    return local_libs


def test_local_hf_tts_without_torch_is_a_config_error_before_any_synthesis(impulse_project, local_libs_without_torch):
    from impulse_evals.config import ConfigError

    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    with pytest.raises(ConfigError) as info:
        provider = _tts(cfg)                  # the runner builds the provider ...
        provider.check_language("en-IN")      # ... and checks every scenario language before the first paid call
    assert info.value.field == "voice_config.tts.provider"
    assert info.value.message == TORCH_MISSING
    assert local_libs_without_torch.tts.pipeline_calls == []   # no model was loaded


@pytest.mark.asyncio
async def test_local_hf_stt_needs_no_torch(impulse_project, local_libs_without_torch):
    # Regression guard: faster-whisper does not use torch, so STT works without it.
    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    provider = _stt(cfg)
    provider.check_language("en-IN")
    result = await provider.transcribe(sine_wav(0.5), "en-IN")
    assert result.text.strip() == "book a slot"


class _TorchSpecOnlyLoader(importlib.abc.Loader):
    """Loader of a torch that may be FOUND but must never be imported."""

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise AssertionError("Impulse imported torch; the check must use importlib.util.find_spec only")


class _TorchSpecOnlyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch":
            return importlib.machinery.ModuleSpec("torch", _TorchSpecOnlyLoader())
        return None


@pytest.mark.asyncio
async def test_local_hf_tts_with_torch_installed_builds_without_importing_torch(
    impulse_project, local_libs, monkeypatch
):
    # Regression guard: with torch installed (found by find_spec) there is no error, and the check
    # never imports torch (device "cpu": nothing needs it before synthesis).
    for loaded in [name for name in sys.modules if name == "torch" or name.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, loaded)
    monkeypatch.setattr(sys, "meta_path", [_TorchSpecOnlyFinder()] + list(sys.meta_path))
    assert importlib.util.find_spec("torch") is not None
    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    provider = _tts(cfg)
    provider.check_language("en-IN")
    audio = await provider.synthesize("hello there", "en-IN", None)
    assert read_wav(audio)[1] == 2
    assert "torch" not in sys.modules


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 20): an HTTP TTS answer with status 200 that is not audio (a JSON
# error body, a proxy's HTML login page, a server that ignored response_format) raises
# VoiceError("{label}: POST {url} returned {content_type}, not audio: {excerpt}") at once (one
# request, not retried): ``content_type`` is the header value or "no content-type", ``url`` has no
# credentials, ``excerpt`` is the body redacted first, then put on one line and shortened to 200
# characters. Audio content types and WAV bytes are returned unchanged.
# =============================================================================================


class _Fixed200TTSServer(MockVoiceServer):
    """``MockVoiceServer`` whose TTS routes answer 200 with ``body`` and the ``content_type`` header
    (no content-type header when it is None)."""

    def __init__(self, body: bytes, content_type):
        super().__init__()
        self.body = body
        self.content_type = content_type

    def _handle(self, request):
        import httpx

        path = request.url.path.rstrip("/")
        if request.method == "POST" and path.endswith(("/audio/speech", "/v1/tts")):
            self.requests.append({"method": request.method, "url": str(request.url), "path": request.url.path,
                                  "headers": dict(request.headers), "json": None, "content": request.content})
            headers ={"content-type": self.content_type} if self.content_type is not None else {}
            return httpx.Response(200, content=self.body, headers=headers)
        return super()._handle(request)


TTS_LABEL = {"openai_compatible": "openai_compatible TTS", "impulse_hosted": "impulse_hosted TTS"}
TTS_URL = {"openai_compatible": BASE_URL + "/audio/speech", "impulse_hosted": HOSTED_URL + "/v1/tts"}

NOT_AUDIO_200 = {
    "json": (b'{"error": "model impulse-test-tts not found; voice alloy is unavailable"}', "application/json"),
    "problem_json": (b'{"type": "about:blank", "title": "Not Found", "status": 404}', "application/problem+json"),
    "html": (b"<html><body><h1>Sign in</h1>\n<p>Your session has expired.</p></body></html>",
             "text/html; charset=utf-8"),
    "plain_text": (b"upstream proxy error: model not loaded", "text/plain"),
    "octet_stream_object": (b'{"error": "voice not found"}', "application/octet-stream"),
    "octet_stream_array": (b'["error", "bad request"]', "application/octet-stream"),
    "no_content_type_xml": (b"<?xml version='1.0'?><error>quota exceeded</error>", None),
}


def _expected_not_audio(provider, body: bytes, content_type, url=None):
    excerpt = " ".join(body.decode("utf-8").split())
    if len(excerpt) > 200:
        excerpt = excerpt[:197] + "..."
    shown = content_type if content_type is not None else "no content-type"
    return f"{TTS_LABEL[provider]}: POST {url or TTS_URL[provider]} returned {shown}, not audio: {excerpt}"


def _is_voice_error(exc):
    return "VoiceError" in [cls.__name__ for cls in type(exc).__mro__]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(NOT_AUDIO_200))
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
async def test_tts_200_answer_that_is_not_audio_is_a_voice_error(impulse_project, monkeypatch, provider, case):
    body, content_type = NOT_AUDIO_200[case]
    server = _Fixed200TTSServer(body, content_type)
    server.install(monkeypatch)
    cfg = _http_cfg(impulse_project, provider, monkeypatch)
    with pytest.raises(Exception) as info:
        await _tts(cfg).synthesize("Hello", "en-IN", None)
    assert _is_voice_error(info.value), repr(info.value)
    assert str(info.value) == _expected_not_audio(provider, body, content_type)
    assert len(server.requests_to(TTS_PATH[provider])) == 1   # not retried


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
async def test_not_audio_excerpt_is_redacted_then_shortened(impulse_project, monkeypatch, provider):
    # The body echoes the bearer token (impulse_hosted) or the base_url password (openai_compatible)
    # and is longer than 200 characters: the secret is redacted first, then the excerpt is cut.
    secret = HOSTED_KEY if provider == "impulse_hosted" else BASE_URL_PASSWORD
    body = ('{"error": "invalid credentials ' + secret + '", "detail": "' + "x" * 300 + '"}').encode()
    server = _Fixed200TTSServer(body, "application/json")
    server.install(monkeypatch)
    cfg = _userinfo_cfg(impulse_project, provider, monkeypatch)
    with pytest.raises(Exception) as info:
        await _tts(cfg).synthesize("Hello", "en-IN", None)
    message = str(info.value)
    assert _is_voice_error(info.value), repr(info.value)
    assert secret not in message
    assert BASE_URL_PASSWORD not in message
    redacted = body.decode().replace(secret, "[REDACTED]").encode()
    assert message == _expected_not_audio(provider, redacted, "application/json")   # URL without credentials
    assert len(server.requests_to(TTS_PATH[provider])) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    ["audio/wav", "audio/x-wav", "application/octet-stream", None],
    ids=["audio_wav", "audio_x_wav", "octet_stream", "no_content_type"],
)
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
async def test_tts_200_wav_answer_is_returned_unchanged(impulse_project, monkeypatch, provider, content_type):
    # Regression guard: WAV bytes (any audio or octet-stream content type, or none) come back verbatim.
    wav = sine_wav(0.3)
    server = _Fixed200TTSServer(wav, content_type)
    server.install(monkeypatch)
    cfg = _http_cfg(impulse_project, provider, monkeypatch)
    assert await _tts(cfg).synthesize("Hello", "en-IN", None) == wav
    assert len(server.requests_to(TTS_PATH[provider])) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "impulse_hosted"])
async def test_tts_200_compressed_audio_answer_is_returned_unchanged(impulse_project, monkeypatch, provider):
    # Regression guard: an audio content type that is not WAV (converted later by the 'voice' extra).
    mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(range(256))
    server = _Fixed200TTSServer(mp3, "audio/mpeg")
    server.install(monkeypatch)
    cfg = _http_cfg(impulse_project, provider, monkeypatch)
    assert await _tts(cfg).synthesize("Hello", "en-IN", None) == mp3


# =============================================================================================
# Audit batch 2 review fix-ups (AUDIT.md finding 17, local_hf STT): an exception raised by
# faster-whisper INFERENCE, in ``WhisperModel.transcribe(...)`` itself or while its lazy segment
# generator is iterated (faster-whisper decodes there; e.g. CTranslate2's "Library
# cublas64_12.dll is not found or cannot be loaded" at the first decode), is
# VoiceError(f"local_hf STT failed with model {model}: {type(exc).__name__}: {exc}"), as a model
# LOAD failure already is ("local_hf STT could not load model ..."; unchanged), so the runner keeps
# the turn with its "Impulse STT failed" warning instead of erroring the scenario.
# =============================================================================================

CUBLAS_MISSING = "Library cublas64_12.dll is not found or cannot be loaded"
LOCAL_STT_MODEL = LOCAL_STT["model"]


class _InferenceFailingWhisper:
    """A fake ``faster_whisper`` whose ``WhisperModel`` loads, but whose inference raises
    ``make_error()``: in ``transcribe(...)`` itself (``where="transcribe"``) or while the returned
    segment generator is iterated (``where="segments"``)."""

    def __init__(self, make_error, where="transcribe"):
        self.make_error = make_error
        self.where = where
        self.init_calls = []
        self.transcribe_calls = []

    def module(self):
        outer = self

        def failing_segments():
            raise outer.make_error()
            yield  # pragma: no cover - makes this function a generator, as faster-whisper's is

        class WhisperModel:
            def __init__(self, *args, **kwargs):
                outer.init_calls.append({"args": args, "kwargs": kwargs})

            def transcribe(self, audio, *args, **kwargs):
                outer.transcribe_calls.append({"args": args, "kwargs": kwargs})
                if outer.where == "transcribe":
                    raise outer.make_error()
                info = types.SimpleNamespace(language="en", language_probability=0.99, duration=0.5)
                return failing_segments(), info

        return _module("faster_whisper", WhisperModel=WhisperModel)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "where, make_error, expected_cause",
    [
        pytest.param("transcribe", lambda: RuntimeError(CUBLAS_MISSING), f"RuntimeError: {CUBLAS_MISSING}",
                     id="transcribe-raises-runtime-error"),
        pytest.param("segments", lambda: RuntimeError(CUBLAS_MISSING), f"RuntimeError: {CUBLAS_MISSING}",
                     id="segment-iteration-raises-runtime-error"),
        pytest.param("transcribe", lambda: OSError("[WinError 126] The specified module could not be found"),
                     "OSError: [WinError 126] The specified module could not be found",
                     id="transcribe-raises-os-error"),
    ],
)
async def test_local_hf_stt_inference_failure_is_a_voice_error_naming_the_model(
    impulse_project, extra_imports, where, make_error, expected_cause
):
    fake = _InferenceFailingWhisper(make_error, where=where)
    extra_imports.module("faster_whisper", fake.module())
    cfg = _voice_cfg(impulse_project, tts=OPENAI_TTS, stt=LOCAL_STT)
    provider = _stt(cfg)
    provider.check_language("en-IN")
    with pytest.raises(Exception) as info:
        await provider.transcribe(sine_wav(0.5), "en-IN")
    assert _is_voice_error(info.value), repr(info.value)
    assert str(info.value) == f"local_hf STT failed with model {LOCAL_STT_MODEL}: {expected_cause}"
    assert len(fake.init_calls) == 1          # the model loaded ...
    assert len(fake.transcribe_calls) == 1    # ... and its inference failed


@pytest.mark.asyncio
async def test_local_hf_stt_load_failure_text_is_unchanged(impulse_project, extra_imports):
    # Regression guard (passes before the fix too): a failing model LOAD keeps its own VoiceError text
    # (never wrapped a second time by the inference rule).
    class WhisperModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")

    extra_imports.module("faster_whisper", _module("faster_whisper", WhisperModel=WhisperModel))
    cfg = _voice_cfg(impulse_project, tts=OPENAI_TTS, stt=LOCAL_STT)
    with pytest.raises(Exception) as info:
        await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert _is_voice_error(info.value), repr(info.value)
    assert str(info.value) == (
        f"local_hf STT could not load model {LOCAL_STT_MODEL}: "
        "RuntimeError: CUDA driver version is insufficient for CUDA runtime version"
    )


@pytest.mark.asyncio
async def test_local_hf_stt_works_again_after_an_inference_failure(impulse_project, extra_imports):
    # Regression guard (passes before the fix too): one failed inference does not break the cached
    # model; the next transcription of the same provider succeeds.
    attempts = []

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, *args, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise RuntimeError(CUBLAS_MISSING)
            words = [_word(" book", 0.0, 0.5, 0.9), _word(" a", 0.5, 0.75, 0.9), _word(" slot", 0.75, 1.5, 0.9)]
            return iter([_segment(" book a slot", words)]), types.SimpleNamespace(language="en")

    extra_imports.module("faster_whisper", _module("faster_whisper", WhisperModel=WhisperModel))
    provider = _stt(_voice_cfg(impulse_project, tts=OPENAI_TTS, stt=LOCAL_STT))
    with pytest.raises(Exception):
        await provider.transcribe(sine_wav(0.5), "en-IN")
    result = await provider.transcribe(sine_wav(0.5), "en-IN")
    assert result.text.strip() == "book a slot"
    assert len(attempts) == 2


# =============================================================================================
# Audit batch 3 (AUDIT.md finding 45): impulse_hosted STT ``words[]``: a missing or null word
# ``confidence`` becomes 0.0 (as openai_compatible keeps a word whose probability is null, with
# 0.0); an entry that is not a mapping, or lacks ``word``, ``start_ms`` or ``end_ms``, is dropped.
# The top-level text and confidence are unchanged; none of this is a VoiceError.
# =============================================================================================

HOSTED_RAGGED_WORDS = [
    {"word": "book", "start_ms": 0, "end_ms": 250, "confidence": None},   # null confidence -> 0.0
    {"word": "a", "start_ms": 250, "end_ms": 500},                         # missing confidence -> 0.0
    "slot",                                                                # not a mapping: dropped
    None,                                                                  # not a mapping: dropped
    {"start_ms": 500, "end_ms": 750, "confidence": 0.9},                   # no word: dropped
    {"word": "for", "end_ms": 900, "confidence": 0.9},                     # no start_ms: dropped
    {"word": "on", "start_ms": 900, "confidence": 0.9},                    # no end_ms: dropped
    {"word": "tuesday", "start_ms": 1000, "end_ms": 1500, "confidence": 0.8},
]


@pytest.mark.asyncio
async def test_impulse_hosted_stt_word_without_confidence_defaults_to_0_and_malformed_entries_are_dropped(
    impulse_project, mock_voice_server, monkeypatch
):
    from impulse_evals.models import WordTiming

    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=HOSTED_TTS, stt=HOSTED_STT)
    mock_voice_server.queue_stt("book a slot for tuesday", words=HOSTED_RAGGED_WORDS, confidence=0.77)
    result = await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert result.text == "book a slot for tuesday"
    assert result.confidence == pytest.approx(0.77)
    assert result.words == [
        WordTiming(word="book", start_ms=0, end_ms=250, confidence=0.0),
        WordTiming(word="a", start_ms=250, end_ms=500, confidence=0.0),
        WordTiming(word="tuesday", start_ms=1000, end_ms=1500, confidence=0.8),
    ]


@pytest.mark.asyncio
async def test_impulse_hosted_stt_null_top_level_confidence_is_kept(impulse_project, mock_voice_server, monkeypatch):
    # Regression guard: the top-level confidence may be null (STTResult.confidence is Optional).
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=HOSTED_TTS, stt=HOSTED_STT)
    mock_voice_server.queue_stt("book a slot", words=EXACT_WORDS, confidence=None)
    result = await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert result.confidence is None
    assert [word.word for word in result.words] == ["book", "a", "slot"]


# =============================================================================================
# Review fix-up of audit batch 3 (group X): after malformed ``words[]`` entries are dropped, a
# validation error in a later entry still names that entry's index in the SERVER's ``words`` list
# (``words.3.word``), not its index among the entries that were kept.
# =============================================================================================

HOSTED_OK_WORD = {"word": "ok", "start_ms": 0, "end_ms": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "words, location, wrong_location",
    [
        pytest.param(
            ["slot", None, HOSTED_OK_WORD, {"word": 7, "start_ms": 5, "end_ms": 9, "confidence": 0.5}],
            "words.3.word",
            "words.1.word",
            id="bad_word_after_two_dropped_entries",
        ),
        pytest.param(
            [HOSTED_OK_WORD, {"start_ms": 5, "end_ms": 7}, {"word": "a", "start_ms": 7, "end_ms": 9, "confidence": "high"}],
            "words.2.confidence",
            "words.1.confidence",
            id="bad_confidence_after_one_dropped_entry",
        ),
    ],
)
async def test_impulse_hosted_stt_error_names_the_server_index_of_the_bad_word_entry(
    impulse_project, mock_voice_server, monkeypatch, words, location, wrong_location
):
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=HOSTED_TTS, stt=HOSTED_STT)
    mock_voice_server.queue_stt("ok a", words=words)
    with pytest.raises(Exception) as info:
        await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert _is_voice_error(info.value), repr(info.value)
    text = str(info.value)
    assert location in text, f"the error must name the server's words[] index ({location}): {text}"
    assert wrong_location not in text, text


@pytest.mark.asyncio
async def test_impulse_hosted_stt_error_index_without_dropped_entries_is_unchanged(
    impulse_project, mock_voice_server, monkeypatch
):
    # Regression guard (passes before the fix too): nothing dropped, so the index is the server's.
    monkeypatch.setenv("IMPULSE_VOICE_API_KEY", HOSTED_KEY)
    cfg = _voice_cfg(impulse_project, tts=HOSTED_TTS, stt=HOSTED_STT)
    mock_voice_server.queue_stt("ok", words=[HOSTED_OK_WORD, {"word": 7, "start_ms": 5, "end_ms": 9, "confidence": 0.5}])
    with pytest.raises(Exception) as info:
        await _stt(cfg).transcribe(sine_wav(0.5), "en-IN")
    assert _is_voice_error(info.value), repr(info.value)
    assert "words.1.word" in str(info.value), str(info.value)


# =============================================================================================
# Maintainer decision 2026-10-21, item 3 (AUDIT-FIXES.md, logged F19): a torch that is installed but
# fails to import must fail the pre-run check (LocalHFTTS.__init__, which the runner runs before the
# first paid call and the doctor runs as its voice TTS row). torch not findable: TORCH_MISSING
# (unchanged); torch already a module in sys.modules: OK; otherwise `import torch` runs in a SEPARATE
# Python interpreter (never in Impulse's own process, so a torch that crashes the interpreter cannot
# kill the run). Any exception there except a ModuleNotFoundError for torch itself is
# ConfigError("voice_config.tts.provider", TORCH_IMPORT_FAILED with detail "{type name}: {str(exc)}");
# a child that ends with a non-zero exit code and no result gives the detail "the import ended the
# Python process with exit code {code}". The detail has its whitespace runs collapsed to one space
# (stripped) and, when longer than 300 characters, is its first 297 characters + "...". On Windows
# (sys.platform == "win32", read when the check runs) TORCH_WINDOWS_HINT follows; elsewhere the
# message ends after the detail. Checked with an ON-DISK fake torch package in tmp_path, first on
# sys.path, with every sys.modules entry for torch (the local_libs fake) removed; the package
# records the PID of each process that imports it.
# =============================================================================================

import os  # noqa: E402
import pathlib  # noqa: E402

TORCH_IMPORT_FAILED = "local_hf TTS needs PyTorch, which is installed but could not be imported ({detail})."
TORCH_WINDOWS_HINT = (
    " On Windows, a DLL load error (for example a missing vcruntime140_1.dll) means the Microsoft Visual C++ "
    "Redistributable (x64) needs to be installed or updated."
)
TORCH_DLL_OSERROR = (
    '[WinError 126] The specified module could not be found. Error loading "C:\\fake\\torch\\lib\\c10.dll" '
    "or one of its dependencies."
)
TORCH_DLL_IMPORTERROR = "DLL load failed while importing _C: The specified procedure could not be found."
TORCH_LONG_MESSAGE = "torch could not start:\n\n" + "".join(
    f"  step {step}:\tloading C:\\fake\\torch\\lib\\part{step}.dll    failed\r\n" for step in range(1, 13)
)
TORCH_IMPORTERS = "torch-importers.txt"   # one PID per import of the fake torch package


def _one_line_detail(text):
    """Whitespace runs collapsed to one space and stripped; longer than 300: first 297 characters + "..."."""
    flat = " ".join(text.split())
    return flat if len(flat) <= 300 else flat[:297] + "..."


BROKEN_TORCH = {   # kind -> (body of torch/__init__.py, expected detail)
    "dll_oserror": (f"raise OSError({TORCH_DLL_OSERROR!r})\n", f"OSError: {TORCH_DLL_OSERROR}"),
    "dll_importerror": (f"raise ImportError({TORCH_DLL_IMPORTERROR!r})\n", f"ImportError: {TORCH_DLL_IMPORTERROR}"),
    "missing_dependency": (
        "import impulse_missing_torch_dependency\n",
        "ModuleNotFoundError: No module named 'impulse_missing_torch_dependency'",
    ),
    "hard_crash": ("_impulse_os._exit(3)\n", "the import ended the Python process with exit code 3"),
    "long_message": (
        f"raise RuntimeError({TORCH_LONG_MESSAGE!r})\n",
        _one_line_detail(f"RuntimeError: {TORCH_LONG_MESSAGE}"),
    ),
}


def _on_disk_torch(tmp_path, monkeypatch, body):
    """A ``torch`` package on disk (``tmp_path/torch-site/torch``) first on ``sys.path``, every
    ``sys.modules`` entry for torch removed for the test. Its ``__init__`` appends the importing
    process's PID to ``tmp_path/TORCH_IMPORTERS``, then runs ``body``. Returns the PID file."""
    site = tmp_path / "torch-site"
    package = site / "torch"
    package.mkdir(parents=True)
    importers = tmp_path / TORCH_IMPORTERS
    (package / "__init__.py").write_text(
        "import os as _impulse_os\n"
        f"with open({str(importers)!r}, 'a', encoding='utf-8') as _impulse_handle:\n"
        "    _impulse_handle.write(str(_impulse_os.getpid()) + '\\n')\n" + body,
        encoding="utf-8",
    )
    for loaded in [name for name in sys.modules if name == "torch" or name.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, loaded)
    monkeypatch.syspath_prepend(str(site))
    spec = importlib.util.find_spec("torch")
    assert spec is not None and spec.origin and pathlib.Path(spec.origin).parent == package, spec
    assert "torch" not in sys.modules
    return importers


def _importer_pids(importers):
    return importers.read_text(encoding="utf-8").split() if importers.exists() else []


def test_broken_torch_expected_details_are_what_the_tests_assume():
    # Premise: the long message really is cut (collapsed first, then 297 characters + "...").
    detail = BROKEN_TORCH["long_message"][1]
    assert len(" ".join(f"RuntimeError: {TORCH_LONG_MESSAGE}".split())) > 300
    assert len(detail) == 300 and detail.endswith("...") and detail.startswith("RuntimeError: torch could not start: step 1: ")
    assert "\n" not in detail and "\t" not in detail and "\r" not in detail and "  " not in detail
    assert len(f"OSError: {TORCH_DLL_OSERROR}") <= 300


@pytest.mark.parametrize("platform, hint", [("win32", TORCH_WINDOWS_HINT), ("linux", "")], ids=["windows", "linux"])
@pytest.mark.parametrize("kind", list(BROKEN_TORCH))
def test_local_hf_tts_with_torch_that_fails_to_import_is_a_config_error(
    impulse_project, local_libs, monkeypatch, tmp_path, kind, platform, hint
):
    from impulse_evals.config import ConfigError

    body, detail = BROKEN_TORCH[kind]
    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    importers = _on_disk_torch(tmp_path, monkeypatch, body)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", platform)   # read when the check runs
        with pytest.raises(ConfigError) as info:
            provider = _tts(cfg)                  # the runner and the doctor build the provider ...
            provider.check_language("en-IN")      # ... before the first paid call
    assert info.value.field == "voice_config.tts.provider"
    assert info.value.message == TORCH_IMPORT_FAILED.format(detail=detail) + hint
    assert local_libs.tts.pipeline_calls == []   # no model was loaded
    assert "torch" not in sys.modules
    pids = _importer_pids(importers)
    assert pids, "the check never imported torch"
    assert str(os.getpid()) not in pids, "torch was imported in Impulse's own process; the check must use another interpreter"


def test_local_hf_tts_with_an_importable_on_disk_torch_builds_without_importing_it_here(
    impulse_project, local_libs, monkeypatch, tmp_path
):
    # Regression guard (passes before the fix too): a torch that imports cleanly is no error, and it is
    # never imported in this process (the check's import, if any, happened elsewhere).
    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    importers = _on_disk_torch(tmp_path, monkeypatch, "VERSION = 'impulse-test'\n")
    provider = _tts(cfg)
    provider.check_language("en-IN")
    assert local_libs.tts.pipeline_calls == []
    assert "torch" not in sys.modules
    assert str(os.getpid()) not in _importer_pids(importers)


# =============================================================================================
# Maintainer decisions 2026-10-21, review fix-up R2 (item 3): a probe child that never reaches
# `import torch` is inconclusive. The probe writes a start line (IMPULSE_TORCH_PROBE_STARTED) right
# after setting sys.path and before `import torch`, and writes the start and result lines with
# os.write(1, ...), not through sys.stdout (which torch's own import can replace). With no result
# line: start line present and a non-zero exit code -> the crash failure (as before); start line
# absent (the interpreter could not start, e.g. a bad PYTHONHOME, or sys.executable is no Python)
# -> inconclusive, no error; exit code 0 -> inconclusive.
# =============================================================================================

TORCH_REPLACES_STDOUT = "import io, sys\nsys.stdout = io.StringIO()\n" + f"raise OSError({TORCH_DLL_OSERROR!r})\n"


def test_local_hf_tts_torch_probe_child_that_cannot_start_python_is_inconclusive(
    impulse_project, local_libs, monkeypatch, tmp_path
):
    from impulse_evals.config import ConfigError

    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    importers = _on_disk_torch(tmp_path, monkeypatch, BROKEN_TORCH["dll_oserror"][0])
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "no-such-python-home"))
    # Premise: an interpreter started with this environment fails before it runs any code.
    premise = subprocess.run(
        [sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL, capture_output=True, timeout=120, check=False
    )
    assert premise.returncode != 0, f"a bad PYTHONHOME did not stop the interpreter: {premise!r}"
    try:
        provider = _tts(cfg)
        provider.check_language("en-IN")
    except ConfigError as exc:
        pytest.fail(f"a probe child that could not start Python is not a torch failure (inconclusive): {exc}")
    assert local_libs.tts.pipeline_calls == []
    assert "torch" not in sys.modules
    assert _importer_pids(importers) == [], "the probe child got as far as importing torch"


def test_local_hf_tts_with_a_torch_that_replaces_sys_stdout_before_failing_is_a_config_error(
    impulse_project, local_libs, monkeypatch, tmp_path
):
    from impulse_evals.config import ConfigError

    cfg = _voice_cfg(impulse_project, tts=LOCAL_TTS, stt=LOCAL_STT)
    importers = _on_disk_torch(tmp_path, monkeypatch, TORCH_REPLACES_STDOUT)
    with pytest.raises(ConfigError) as info:
        provider = _tts(cfg)
        provider.check_language("en-IN")
    hint = TORCH_WINDOWS_HINT if sys.platform == "win32" else ""
    assert info.value.field == "voice_config.tts.provider"
    assert info.value.message == TORCH_IMPORT_FAILED.format(detail=f"OSError: {TORCH_DLL_OSERROR}") + hint
    assert local_libs.tts.pipeline_calls == []
    assert "torch" not in sys.modules
    pids = _importer_pids(importers)
    assert pids and str(os.getpid()) not in pids, pids
