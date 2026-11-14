# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 3, B2: the LiteLLM voice provider (``voice_config.{tts,stt}.provider: litellm``;
FIXES-ROUND-3-SPEC.md section B2).

* ``config.VOICE_PROVIDERS == ("local_hf", "local_light", "openai_compatible", "impulse_hosted",
  "litellm")``; the "unknown provider ... Use one of: ..." text lists them in that order.
* ``provider: litellm`` needs ``model`` (``voice_config.{side}.model``: "required when provider is
  litellm", load_config); ``base_url`` (optional) is passed as ``api_base``.
* ``impulse_evals.voice.litellm_voice``: ``LiteLLMSTT`` / ``LiteLLMTTS``, returned by the factories;
  ``litellm`` is imported lazily inside the methods.
* Keys: ``api_key_env`` when set, else ``config.key_name_for_model(model)``; the value from the
  environment, then config.toml [keys], passed as ``api_key``; nothing when the model needs no key;
  a named key that is not set is a ConfigError at ``voice_config.{side}.model`` raised when the
  provider is built.
* STT: ``litellm.atranscription(model, file=("audio.wav", wav), language=<primary subtag>,
  response_format="verbose_json", timestamp_granularities=["word"], drop_params=True,
  timeout=voice_config.timeout_s, [api_key], [api_base])``; words in ms; confidence rules incl. the
  Deepgram raw ``_hidden_params`` fallback.
* TTS: ``litellm.aspeech(model, input, [voice], response_format="wav", timeout, [api_key],
  [api_base])``; the bytes go through ``audio_utils.ensure_wav``.
* Errors: one-line ``VoiceError("litellm {STT|TTS} failed with model {model}: {text}")``, key values
  redacted; 2 retries (``asyncio.sleep(1.0 * attempt)``) on 429 / >= 500, none on other 4xx; a
  ConfigError is not wrapped.

Mocks only: ``litellm.atranscription`` / ``litellm.aspeech`` are replaced by async fakes that record
their arguments and return LiteLLM's own response types (built offline, no network).
"""

import asyncio
import subprocess
import sys
import types
from importlib.machinery import ModuleSpec

import pytest
from conftest import REPO_ROOT, sine_wav

DG_MODEL = "deepgram/nova-3"
GROQ_STT_MODEL = "groq/impulse-test-stt"
LOCAL_STT_MODEL = "ollama/impulse-test-stt"
TTS_MODEL = "openai/impulse-test-tts"
BASE_URL = "http://localhost:8123/v1"

DG_KEY = "dg-secret-0123456789abcdef"
DG_TOML_KEY = "dg-toml-secret-fedcba9876543210"
OPENAI_KEY = "sk-impulse-test-tts-0123456789"
CUSTOM_STT_KEY = "custom-stt-token-55aa66bb77cc"
CUSTOM_TTS_KEY = "custom-tts-token-11dd22ee33ff"

KEY_VARS = ("DEEPGRAM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "MY_STT_TOKEN", "MY_TTS_TOKEN")
PROVIDERS = ("local_hf", "local_light", "openai_compatible", "impulse_hosted", "litellm")
PROVIDERS_TEXT = "local_hf, local_light, openai_compatible, impulse_hosted, litellm"
KEY_HINT = "set the environment variable or add it to [keys] in .impulse/config.toml"
VOICE_EXTRA_MISSING = (
    "audio that is not WAV needs the 'voice' extra (pydub, plus ffmpeg for most formats) to be "
    "converted: pip install impulse-evals[voice]"
)

WAV = sine_wav(0.25)
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(range(256))


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _mod():
    from impulse_evals.voice import litellm_voice

    return litellm_voice


@pytest.fixture
def project(impulse_project, monkeypatch):
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    return impulse_project


def _voice_cfg(project, *, stt=None, tts=None, timeout_s=30):
    """Write a config with a litellm ``voice_config`` block and return the loaded ``VoiceConfig``."""
    voice = {
        "tts": dict(tts if tts is not None else {"provider": "litellm", "model": TTS_MODEL}),
        "stt": dict(stt if stt is not None else {"provider": "litellm", "model": DG_MODEL}),
        "save_audio": False,
        "save_recording": False,
        "timeout_s": timeout_s,
    }
    project.write_config([project.scenario("voice_probe")], voice_config=voice)
    return project.load().voice_config


def _stt(cfg, toml=None):
    from impulse_evals.voice.stt import get_stt_provider

    return get_stt_provider(cfg, toml=toml)


def _tts(cfg, toml=None):
    from impulse_evals.voice.tts import get_tts_provider

    return get_tts_provider(cfg, toml=toml)


def _toml(project, text):
    from impulse_evals.config import load_toml

    return load_toml(project.write_toml(text))


def transcription(text, words=None, hidden=None):
    """LiteLLM's own ``TranscriptionResponse`` (dict-like: ``.text``, ``["words"]``, ``_hidden_params``)."""
    from litellm.types.utils import TranscriptionResponse

    response = TranscriptionResponse(text=text)
    if words is not None:
        response["words"] = words
    if hidden is not None:
        response._hidden_params = hidden
    return response


def deepgram_raw(word_confidences, alt_confidence=0.95):
    """A Deepgram ``/listen`` answer as LiteLLM keeps it in ``_hidden_params``."""
    words = [
        {"word": f"w{index}", "start": index * 0.25, "end": index * 0.25 + 0.25, "confidence": confidence}
        for index, confidence in enumerate(word_confidences)
    ]
    alternative = {"transcript": " ".join(word["word"] for word in words), "words": words}
    if alt_confidence is not None:
        alternative["confidence"] = alt_confidence
    return {"metadata": {"duration": 1.5}, "results": {"channels": [{"alternatives": [alternative]}]}}


# Response words as LiteLLM's Deepgram transformation builds them: word, start, end only.
PLAIN_WORDS = [
    {"word": "book", "start": 0.0, "end": 0.25},
    {"word": "a", "start": 0.25, "end": 0.5},
    {"word": "slot", "start": 0.75, "end": 1.25},
]
PLAIN_TIMINGS = [("book", 0, 250), ("a", 250, 500), ("slot", 750, 1250)]


def speech(content):
    """LiteLLM's own ``HttpxBinaryResponseContent`` around a local ``httpx.Response``."""
    import httpx
    from litellm.types.llms.openai import HttpxBinaryResponseContent

    return HttpxBinaryResponseContent(httpx.Response(200, content=content))


def litellm_error(name, status_code=None, message="something broke", module="litellm.exceptions"):
    """An exception whose class is defined in a ``litellm.*`` module (as test_provider_errors.py)."""
    cls = type(name, (Exception,), {})
    cls.__module__ = module
    exc = cls(message)
    if status_code is not None:
        exc.status_code = status_code
    return exc


class FakeLiteLLM:
    """Replaces ``litellm.atranscription`` / ``litellm.aspeech``; queued items are returned in order
    (an exception is raised); with an empty queue a default answer is returned."""

    def __init__(self, monkeypatch):
        import litellm

        self.transcription_calls = []
        self.speech_calls = []
        self.transcriptions = []
        self.speeches = []
        monkeypatch.setattr(litellm, "atranscription", self._atranscription)
        monkeypatch.setattr(litellm, "aspeech", self._aspeech)

    async def _atranscription(self, *args, **kwargs):
        self.transcription_calls.append({"args": args, "kwargs": kwargs})
        item = self.transcriptions.pop(0) if self.transcriptions else transcription("book a slot", PLAIN_WORDS)
        if isinstance(item, BaseException):
            raise item
        return item

    async def _aspeech(self, *args, **kwargs):
        self.speech_calls.append({"args": args, "kwargs": kwargs})
        item = self.speeches.pop(0) if self.speeches else speech(WAV)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def fake_litellm(monkeypatch):
    return FakeLiteLLM(monkeypatch)


@pytest.fixture
def sleeps(monkeypatch):
    """``asyncio.sleep`` returns at once and records each delay (retries never wait in tests)."""
    delays = []

    async def fake_sleep(delay, *args, **kwargs):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


def _timings(result):
    return [(word.word, word.start_ms, word.end_ms) for word in result.words]


def _confidences(result):
    return [word.confidence for word in result.words]


# =============================================================================================
# Provider list, config validation, factories
# =============================================================================================


def test_voice_providers_tuple_order():
    from impulse_evals.config import VOICE_PROVIDERS

    assert tuple(VOICE_PROVIDERS) == PROVIDERS


@pytest.mark.parametrize("side", ["tts", "stt"])
def test_unknown_provider_text_at_load_lists_the_new_providers(project, side):
    from impulse_evals.config import ConfigError

    blocks = {"tts": {"provider": "litellm", "model": TTS_MODEL}, "stt": {"provider": "litellm", "model": DG_MODEL}}
    blocks[side] = {"provider": "acme_voice", "model": "acme/voice-1"}
    with pytest.raises(ConfigError) as info:
        _voice_cfg(project, **blocks)
    assert info.value.field == f"voice_config.{side}.provider"
    assert info.value.message == f"unknown provider acme_voice. Use one of: {PROVIDERS_TEXT}."


@pytest.mark.parametrize("side", ["tts", "stt"])
def test_unknown_provider_text_in_factory_lists_the_new_providers(project, monkeypatch, side):
    from impulse_evals.config import ConfigError

    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    cfg = _voice_cfg(project)
    getattr(cfg, side).provider = "acme_voice"   # an unvalidated value handed to the factory
    factory = _tts if side == "tts" else _stt
    with pytest.raises(ConfigError) as info:
        factory(cfg)
    assert info.value.field == f"voice_config.{side}.provider"
    assert info.value.message == f"unknown provider acme_voice. Use one of: {PROVIDERS_TEXT}."


@pytest.mark.parametrize("side", ["tts", "stt"])
def test_litellm_without_model_is_config_error_at_load(project, side):
    from impulse_evals.config import ConfigError

    blocks = {"tts": {"provider": "litellm", "model": TTS_MODEL}, "stt": {"provider": "litellm", "model": DG_MODEL}}
    blocks[side] = {"provider": "litellm"}
    with pytest.raises(ConfigError) as info:
        _voice_cfg(project, **blocks)
    assert info.value.field == f"voice_config.{side}.model"
    assert info.value.message == "required when provider is litellm"


def test_factories_return_the_litellm_classes(project, monkeypatch):
    from impulse_evals.voice.stt import STTProvider
    from impulse_evals.voice.tts import TTSProvider

    mod = _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    cfg = _voice_cfg(project)   # no base_url: litellm does not need one
    stt = _stt(cfg)
    tts = _tts(cfg)
    assert isinstance(stt, mod.LiteLLMSTT)
    assert isinstance(stt, STTProvider)
    assert isinstance(tts, mod.LiteLLMTTS)
    assert isinstance(tts, TTSProvider)


def test_module_imports_litellm_lazily():
    code = (
        "import sys\n"
        "import impulse_evals.voice.litellm_voice\n"
        "print('litellm' in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False", "litellm must be imported inside the methods, not at module import"


# =============================================================================================
# Keys
# =============================================================================================


def test_stt_missing_key_is_config_error_when_built(project, fake_litellm):
    from impulse_evals.config import ConfigError

    _mod()
    cfg = _voice_cfg(project)
    with pytest.raises(ConfigError) as info:
        _stt(cfg)
    assert info.value.field == "voice_config.stt.model"
    assert info.value.message == f"{DG_MODEL} needs the key DEEPGRAM_API_KEY: {KEY_HINT}"
    assert fake_litellm.transcription_calls == []


def test_tts_missing_key_is_config_error_when_built(project, fake_litellm):
    from impulse_evals.config import ConfigError

    _mod()
    cfg = _voice_cfg(project)
    with pytest.raises(ConfigError) as info:
        _tts(cfg)
    assert info.value.field == "voice_config.tts.model"
    assert info.value.message == f"{TTS_MODEL} needs the key OPENAI_API_KEY: {KEY_HINT}"
    assert fake_litellm.speech_calls == []


def test_missing_api_key_env_is_config_error_naming_it(project, monkeypatch, fake_litellm):
    from impulse_evals.config import ConfigError

    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)   # set, but api_key_env names another variable
    cfg = _voice_cfg(project, stt={"provider": "litellm", "model": DG_MODEL, "api_key_env": "MY_STT_TOKEN"})
    with pytest.raises(ConfigError) as info:
        _stt(cfg)
    assert info.value.field == "voice_config.stt.model"
    assert info.value.message == f"{DG_MODEL} needs the key MY_STT_TOKEN: {KEY_HINT}"


@pytest.mark.asyncio
async def test_stt_key_from_config_toml(project, fake_litellm):
    _mod()
    toml = _toml(project, f'[keys]\nDEEPGRAM_API_KEY = "{DG_TOML_KEY}"\n')
    stt = _stt(_voice_cfg(project), toml=toml)
    await stt.transcribe(WAV, "en-US")
    assert fake_litellm.transcription_calls[0]["kwargs"]["api_key"] == DG_TOML_KEY


@pytest.mark.asyncio
async def test_stt_environment_key_wins_over_config_toml(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    toml = _toml(project, f'[keys]\nDEEPGRAM_API_KEY = "{DG_TOML_KEY}"\n')
    stt = _stt(_voice_cfg(project), toml=toml)
    await stt.transcribe(WAV, "en-US")
    assert fake_litellm.transcription_calls[0]["kwargs"]["api_key"] == DG_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["environment", "config_toml"])
async def test_stt_api_key_env_names_the_variable(project, monkeypatch, fake_litellm, source):
    _mod()
    toml = None
    if source == "environment":
        monkeypatch.setenv("MY_STT_TOKEN", CUSTOM_STT_KEY)
    else:
        toml = _toml(project, f'[keys]\nMY_STT_TOKEN = "{CUSTOM_STT_KEY}"\n')
    cfg = _voice_cfg(project, stt={"provider": "litellm", "model": GROQ_STT_MODEL, "api_key_env": "MY_STT_TOKEN"})
    stt = _stt(cfg, toml=toml)   # GROQ_API_KEY is not set: api_key_env replaces it
    await stt.transcribe(WAV, "en-US")
    assert fake_litellm.transcription_calls[0]["kwargs"]["api_key"] == CUSTOM_STT_KEY


@pytest.mark.asyncio
async def test_stt_model_without_key_passes_no_api_key(project, fake_litellm):
    _mod()
    cfg = _voice_cfg(project, stt={"provider": "litellm", "model": LOCAL_STT_MODEL})
    stt = _stt(cfg)   # no ConfigError: the model needs no Impulse key
    await stt.transcribe(WAV, "en-US")
    kwargs = fake_litellm.transcription_calls[0]["kwargs"]
    assert "api_key" not in kwargs
    assert "api_base" not in kwargs


# =============================================================================================
# STT call
# =============================================================================================


@pytest.mark.asyncio
async def test_stt_call_kwargs_exact(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    stt = _stt(_voice_cfg(project, timeout_s=17))
    await stt.transcribe(WAV, "es-MX")
    assert len(fake_litellm.transcription_calls) == 1
    call = fake_litellm.transcription_calls[0]
    assert call["args"] == ()
    assert call["kwargs"] == {
        "model": DG_MODEL,
        "file": ("audio.wav", WAV),
        "language": "es",
        "detect_language": "es",
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
        "drop_params": True,
        "max_retries": 0,
        "timeout": 17,
        "api_key": DG_KEY,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language, subtag", [("hi-IN", "hi"), ("hi-Latn-IN", "hi"), ("en-US", "en"), ("ta-in", "ta"), ("es", "es")]
)
async def test_stt_language_is_the_primary_subtag(project, monkeypatch, fake_litellm, language, subtag):
    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    await _stt(_voice_cfg(project)).transcribe(WAV, language)
    assert fake_litellm.transcription_calls[0]["kwargs"]["language"] == subtag


@pytest.mark.asyncio
async def test_stt_base_url_is_api_base(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    cfg = _voice_cfg(project, stt={"provider": "litellm", "model": DG_MODEL, "base_url": BASE_URL})
    await _stt(cfg).transcribe(WAV, "en-US")
    assert fake_litellm.transcription_calls[0]["kwargs"]["api_base"] == BASE_URL


@pytest.mark.asyncio
async def test_stt_api_base_omitted_without_base_url(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")
    assert "api_base" not in fake_litellm.transcription_calls[0]["kwargs"]


# =============================================================================================
# STT result: text, word timings, confidence
# =============================================================================================


async def _transcribe(project, monkeypatch, fake_litellm, response):
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    fake_litellm.transcriptions.append(response)
    return await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")


@pytest.mark.asyncio
async def test_text_stripped_and_word_timings_in_ms_with_probability(project, monkeypatch, fake_litellm):
    from impulse_evals.models import STTResult

    _mod()
    words = [dict(word, probability=p) for word, p in zip(PLAIN_WORDS, (0.75, 0.5, 1.0))]
    result = await _transcribe(project, monkeypatch, fake_litellm, transcription("  book a slot \n", words))
    assert isinstance(result, STTResult)
    assert result.text == "book a slot"
    assert _timings(result) == PLAIN_TIMINGS
    assert _confidences(result) == pytest.approx([0.75, 0.5, 1.0])
    assert result.confidence == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_word_objects_with_confidence_attribute(project, monkeypatch, fake_litellm):
    _mod()
    words = [
        types.SimpleNamespace(word=word["word"], start=word["start"], end=word["end"], confidence=c)
        for word, c in zip(PLAIN_WORDS, (0.5, 0.25, 0.75))
    ]
    result = await _transcribe(project, monkeypatch, fake_litellm, transcription("book a slot", words))
    assert _timings(result) == PLAIN_TIMINGS
    assert _confidences(result) == pytest.approx([0.5, 0.25, 0.75])
    assert result.confidence == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_text_none_is_empty_string(project, monkeypatch, fake_litellm):
    _mod()
    result = await _transcribe(project, monkeypatch, fake_litellm, transcription(None))
    assert result.text == ""
    assert result.words == []
    assert result.confidence is None


@pytest.mark.asyncio
async def test_no_words_gives_empty_list(project, monkeypatch, fake_litellm):
    _mod()
    result = await _transcribe(project, monkeypatch, fake_litellm, transcription("book a slot"))
    assert result.text == "book a slot"
    assert result.words == []
    assert result.confidence is None


@pytest.mark.asyncio
async def test_empty_words_list_gives_empty_list(project, monkeypatch, fake_litellm):
    _mod()
    result = await _transcribe(project, monkeypatch, fake_litellm, transcription("book a slot", []))
    assert result.words == []
    assert result.confidence is None


@pytest.mark.asyncio
async def test_words_without_any_confidence_are_zero_and_result_confidence_none(project, monkeypatch, fake_litellm):
    _mod()
    result = await _transcribe(project, monkeypatch, fake_litellm, transcription("book a slot", PLAIN_WORDS))
    assert _timings(result) == PLAIN_TIMINGS
    assert _confidences(result) == [0.0, 0.0, 0.0]
    assert result.confidence is None


@pytest.mark.asyncio
async def test_mean_uses_only_confidences_from_the_response(project, monkeypatch, fake_litellm):
    _mod()
    words = [dict(PLAIN_WORDS[0], probability=0.8), dict(PLAIN_WORDS[1]), dict(PLAIN_WORDS[2], probability=0.6)]
    result = await _transcribe(project, monkeypatch, fake_litellm, transcription("book a slot", words))
    assert _confidences(result) == pytest.approx([0.8, 0.0, 0.6])
    assert result.confidence == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_deepgram_raw_word_confidence_fallback(project, monkeypatch, fake_litellm):
    _mod()
    response = transcription("book a slot", PLAIN_WORDS, hidden=deepgram_raw([0.9, 0.5, 0.7], alt_confidence=0.95))
    result = await _transcribe(project, monkeypatch, fake_litellm, response)
    assert _timings(result) == PLAIN_TIMINGS   # timings from the response words
    assert _confidences(result) == pytest.approx([0.9, 0.5, 0.7])
    assert result.confidence == pytest.approx(0.7)   # the mean of the word confidences, not the alternative's


@pytest.mark.asyncio
async def test_deepgram_raw_words_of_another_length_are_not_used(project, monkeypatch, fake_litellm):
    _mod()
    response = transcription("book a slot", PLAIN_WORDS, hidden=deepgram_raw([0.9, 0.5], alt_confidence=0.95))
    result = await _transcribe(project, monkeypatch, fake_litellm, response)
    assert _confidences(result) == [0.0, 0.0, 0.0]
    assert result.confidence == pytest.approx(0.95)   # the alternative-level confidence


@pytest.mark.asyncio
async def test_response_probability_wins_over_deepgram_raw(project, monkeypatch, fake_litellm):
    _mod()
    words = [dict(word, probability=p) for word, p in zip(PLAIN_WORDS, (0.8, 0.6, 0.7))]
    response = transcription("book a slot", words, hidden=deepgram_raw([0.1, 0.1, 0.1], alt_confidence=0.2))
    result = await _transcribe(project, monkeypatch, fake_litellm, response)
    assert _confidences(result) == pytest.approx([0.8, 0.6, 0.7])
    assert result.confidence == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_alternative_confidence_when_no_words(project, monkeypatch, fake_litellm):
    _mod()
    response = transcription("book a slot", hidden=deepgram_raw([], alt_confidence=0.88))
    result = await _transcribe(project, monkeypatch, fake_litellm, response)
    assert result.words == []
    assert result.confidence == pytest.approx(0.88)


@pytest.mark.asyncio
@pytest.mark.parametrize("alt_confidence", ["high", None], ids=["string", "absent"])
async def test_alternative_confidence_that_is_not_a_number_gives_none(
    project, monkeypatch, fake_litellm, alt_confidence
):
    _mod()
    response = transcription("book a slot", PLAIN_WORDS, hidden=deepgram_raw([0.9], alt_confidence=alt_confidence))
    result = await _transcribe(project, monkeypatch, fake_litellm, response)
    assert _confidences(result) == [0.0, 0.0, 0.0]   # raw list of another length: not used
    assert result.confidence is None


@pytest.mark.asyncio
async def test_entries_without_word_start_or_end_are_dropped(project, monkeypatch, fake_litellm):
    _mod()
    words = [
        {"word": "book", "start": 0.0, "end": 0.25, "probability": 0.5},
        {"start": 0.25, "end": 0.5, "probability": 0.1},           # no word
        {"word": "a", "end": 0.75, "probability": 0.1},            # no start
        {"word": "slot", "start": 0.75, "probability": 0.1},       # no end
        {"word": "today", "start": 1.0, "end": 1.5, "probability": 1.0},
    ]
    result = await _transcribe(project, monkeypatch, fake_litellm, transcription("book a slot today", words))
    assert _timings(result) == [("book", 0, 250), ("today", 1000, 1500)]
    assert _confidences(result) == pytest.approx([0.5, 1.0])
    assert result.confidence == pytest.approx(0.75)


# =============================================================================================
# TTS
# =============================================================================================


def _tts_cfg(project, timeout_s=30, **block):
    return _voice_cfg(project, tts={"provider": "litellm", "model": TTS_MODEL, **block}, timeout_s=timeout_s)


@pytest.mark.asyncio
async def test_tts_call_kwargs_exact_and_wav_passthrough(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    tts = _tts(_tts_cfg(project, timeout_s=17, voice="nova"))
    audio = await tts.synthesize("Hola, quiero una cita.", "es-MX", None)
    assert audio == WAV
    assert len(fake_litellm.speech_calls) == 1
    call = fake_litellm.speech_calls[0]
    assert call["args"] == ()
    assert call["kwargs"] == {
        "model": TTS_MODEL,
        "input": "Hola, quiero una cita.",
        "voice": "nova",
        "response_format": "wav",
        "max_retries": 0,
        "timeout": 17,
        "api_key": OPENAI_KEY,
    }


@pytest.mark.asyncio
async def test_tts_voice_argument_is_passed(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", "alloy")
    assert fake_litellm.speech_calls[0]["kwargs"]["voice"] == "alloy"


@pytest.mark.asyncio
async def test_tts_no_voice_is_not_passed(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", None)
    assert "voice" not in fake_litellm.speech_calls[0]["kwargs"]


@pytest.mark.asyncio
async def test_tts_base_url_and_api_key_env(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("MY_TTS_TOKEN", CUSTOM_TTS_KEY)
    tts = _tts(_tts_cfg(project, base_url=BASE_URL, api_key_env="MY_TTS_TOKEN"))
    await tts.synthesize("Hello", "en-US", None)
    kwargs = fake_litellm.speech_calls[0]["kwargs"]
    assert kwargs["api_base"] == BASE_URL
    assert kwargs["api_key"] == CUSTOM_TTS_KEY


@pytest.mark.asyncio
async def test_tts_api_base_omitted_without_base_url(project, monkeypatch, fake_litellm):
    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", None)
    assert "api_base" not in fake_litellm.speech_calls[0]["kwargs"]


def _fake_pydub(converted):
    received = []

    class AudioSegment:
        @staticmethod
        def from_file(handle, *args, **kwargs):
            received.append(handle.read())
            return AudioSegment()

        def export(self, out, format=None, **kwargs):
            assert format == "wav"
            out.write(converted)
            return out

    module = types.ModuleType("pydub")
    module.__spec__ = ModuleSpec("pydub", None)
    module.AudioSegment = AudioSegment
    return module, received


@pytest.mark.asyncio
async def test_tts_mp3_bytes_are_converted_to_wav(project, monkeypatch, fake_litellm, extra_imports):
    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    converted = sine_wav(0.5)
    module, received = _fake_pydub(converted)
    extra_imports.module("pydub", module)
    fake_litellm.speeches.append(speech(MP3))
    audio = await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", None)
    assert audio == converted
    assert received == [MP3]


@pytest.mark.asyncio
async def test_tts_mp3_without_pydub_is_the_voice_extra_error(project, monkeypatch, fake_litellm, extra_imports, sleeps):
    from impulse_evals.voice import VoiceError

    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    extra_imports.missing("pydub")
    fake_litellm.speeches.append(speech(MP3))
    with pytest.raises(VoiceError) as info:
        await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", None)
    assert str(info.value) == VOICE_EXTRA_MISSING
    assert len(fake_litellm.speech_calls) == 1   # a conversion error is not retried
    assert sleeps == []


class _ReadOnlyAsync:
    def __init__(self, content):
        self._content = content

    async def aread(self):
        return self._content


class _ReadOnlySync:
    def __init__(self, content):
        self._content = content

    def read(self):
        return self._content


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", [_ReadOnlyAsync, _ReadOnlySync], ids=["aread", "read"])
async def test_tts_response_without_content_is_read(project, monkeypatch, fake_litellm, wrapper):
    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    fake_litellm.speeches.append(wrapper(WAV))
    audio = await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", None)
    assert audio == WAV


# =============================================================================================
# Errors and retries
# =============================================================================================


@pytest.mark.asyncio
async def test_stt_litellm_error_is_one_line_voice_error(project, monkeypatch, fake_litellm, sleeps):
    from impulse_evals.voice import VoiceError

    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    fake_litellm.transcriptions.append(
        litellm_error("AuthenticationError", 401, "DeepgramException - Invalid credentials\nsecond line")
    )
    with pytest.raises(VoiceError) as info:
        await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")
    assert str(info.value) == (
        f"litellm STT failed with model {DG_MODEL}: AuthenticationError (HTTP 401): Invalid credentials"
    )


@pytest.mark.asyncio
async def test_tts_litellm_error_is_one_line_voice_error(project, monkeypatch, fake_litellm, sleeps):
    from impulse_evals.voice import VoiceError

    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    fake_litellm.speeches.append(litellm_error("BadRequestError", 400, "OpenAIException - unknown voice\nmore"))
    with pytest.raises(VoiceError) as info:
        await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", None)
    assert str(info.value) == (
        f"litellm TTS failed with model {TTS_MODEL}: BadRequestError (HTTP 400): unknown voice"
    )


@pytest.mark.asyncio
async def test_other_exception_text_is_type_and_message(project, monkeypatch, fake_litellm, sleeps):
    from impulse_evals.voice import VoiceError

    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    fake_litellm.transcriptions.append(RuntimeError("socket   closed\nsecond line"))
    with pytest.raises(VoiceError) as info:
        await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")
    assert str(info.value) == f"litellm STT failed with model {DG_MODEL}: RuntimeError: socket closed"
    assert len(fake_litellm.transcription_calls) == 1


@pytest.mark.asyncio
async def test_error_text_is_cut_to_200_characters(project, monkeypatch, fake_litellm, sleeps):
    from impulse_evals.voice import VoiceError

    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    fake_litellm.transcriptions.append(RuntimeError("x" * 500))
    with pytest.raises(VoiceError) as info:
        await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")
    prefix = f"litellm STT failed with model {DG_MODEL}: "
    message = str(info.value)
    assert message.startswith(prefix + "RuntimeError: xxx")
    assert len(message) - len(prefix) <= 200
    assert "\n" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["standard_env", "api_key_env", "config_toml"])
async def test_error_never_shows_the_key_value(project, monkeypatch, fake_litellm, sleeps, source):
    from impulse_evals.voice import VoiceError

    _mod()
    toml = None
    block = {"provider": "litellm", "model": DG_MODEL}
    if source == "standard_env":
        secret = DG_KEY
        monkeypatch.setenv("DEEPGRAM_API_KEY", secret)
    elif source == "api_key_env":
        secret = CUSTOM_STT_KEY
        monkeypatch.setenv("MY_STT_TOKEN", secret)
        block["api_key_env"] = "MY_STT_TOKEN"
    else:
        secret = DG_TOML_KEY
        toml = _toml(project, f'[keys]\nDEEPGRAM_API_KEY = "{secret}"\n')
    stt = _stt(_voice_cfg(project, stt=block), toml=toml)
    fake_litellm.transcriptions.append(
        litellm_error("AuthenticationError", 401, f"DeepgramException - invalid key {secret} for this project")
    )
    with pytest.raises(VoiceError) as info:
        await stt.transcribe(WAV, "en-US")
    message = str(info.value)
    assert message.startswith(f"litellm STT failed with model {DG_MODEL}: AuthenticationError (HTTP 401): invalid key ")
    assert secret not in message
    assert secret[:12] not in message
    assert message.endswith(" for this project")


@pytest.mark.asyncio
async def test_stt_retries_twice_on_429_and_503_then_succeeds(project, monkeypatch, fake_litellm, sleeps):
    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    fake_litellm.transcriptions.extend(
        [
            litellm_error("RateLimitError", 429, "rate limited"),
            litellm_error("ServiceUnavailableError", 503, "busy"),
            transcription("book a slot", PLAIN_WORDS),
        ]
    )
    result = await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")
    assert result.text == "book a slot"
    assert len(fake_litellm.transcription_calls) == 3
    assert sleeps == [1.0, 2.0]
    first = fake_litellm.transcription_calls[0]["kwargs"]
    assert all(call["kwargs"] == first for call in fake_litellm.transcription_calls)


@pytest.mark.asyncio
async def test_stt_gives_up_after_three_attempts(project, monkeypatch, fake_litellm, sleeps):
    from impulse_evals.voice import VoiceError

    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    fake_litellm.transcriptions.extend([litellm_error("RateLimitError", 429, "rate limited")] * 4)
    with pytest.raises(VoiceError) as info:
        await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")
    assert str(info.value) == f"litellm STT failed with model {DG_MODEL}: RateLimitError (HTTP 429): rate limited"
    assert len(fake_litellm.transcription_calls) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_stt_no_retry_on_other_4xx(project, monkeypatch, fake_litellm, sleeps, status):
    from impulse_evals.voice import VoiceError

    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    fake_litellm.transcriptions.extend(
        [litellm_error("APIError", status, "rejected"), transcription("never reached")]
    )
    with pytest.raises(VoiceError):
        await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")
    assert len(fake_litellm.transcription_calls) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_tts_retries_on_5xx_then_succeeds(project, monkeypatch, fake_litellm, sleeps):
    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    fake_litellm.speeches.extend([litellm_error("InternalServerError", 502, "bad gateway"), speech(WAV)])
    audio = await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", None)
    assert audio == WAV
    assert len(fake_litellm.speech_calls) == 2
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_tts_no_retry_on_401(project, monkeypatch, fake_litellm, sleeps):
    from impulse_evals.voice import VoiceError

    _mod()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    fake_litellm.speeches.extend([litellm_error("AuthenticationError", 401, "bad key"), speech(WAV)])
    with pytest.raises(VoiceError) as info:
        await _tts(_tts_cfg(project)).synthesize("Hello", "en-US", None)
    assert str(info.value) == f"litellm TTS failed with model {TTS_MODEL}: AuthenticationError (HTTP 401): bad key"
    assert len(fake_litellm.speech_calls) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_config_error_is_not_wrapped(project, monkeypatch, fake_litellm, sleeps):
    from impulse_evals.config import ConfigError

    _mod()
    monkeypatch.setenv("DEEPGRAM_API_KEY", DG_KEY)
    error = ConfigError("voice_config.stt.model", "a configuration problem")
    fake_litellm.transcriptions.append(error)
    with pytest.raises(ConfigError) as info:
        await _stt(_voice_cfg(project)).transcribe(WAV, "en-US")
    assert info.value is error
    assert len(fake_litellm.transcription_calls) == 1
