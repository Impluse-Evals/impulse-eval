# SPDX-License-Identifier: Apache-2.0
"""Fixes round 3, B2-rev: the LiteLLM voice provider after the Stage D review
(FIXES-ROUND-3-SPEC.md section "B2-rev"; review findings V1, V2, V3, V4, V9).

* V1: for a model whose LiteLLM provider is ``deepgram`` (``litellm.get_llm_provider``),
  ``litellm.atranscription`` also gets ``detect_language=<primary subtag>``; ``language=`` is still
  passed. Not for other providers. One test runs LiteLLM's REAL Deepgram request building and
  captures the outgoing request at ``httpx.AsyncClient.send`` (no network): the URL query has
  ``model=nova-3`` and ``detect_language=hi`` for a hi-IN turn.
* V2: ``max_retries=0`` is passed to ``litellm.atranscription`` and ``litellm.aspeech``.
* V3: ``aspeech`` gets ``response_format="wav"`` only when the provider is ``openai``, ``azure`` or
  in ``litellm.openai_compatible_providers``; otherwise no ``response_format`` and the provider's
  default format (e.g. ElevenLabs MP3) goes through ``ensure_wav``.
* V4: a transcription that fails with HTTP 400 whose one-line text mentions ``response_format``,
  ``verbose_json``, ``timestamp_granularities`` or ``timestamp`` (case-insensitive) is repeated once
  at once (no sleep, not a retry) without ``response_format`` and ``timestamp_granularities``; the
  provider instance then omits them on later calls.
* V9: the ``support_order_status_us`` expected outcome in ``examples/config.yaml`` asks only for what
  ``examples/system_prompt.txt`` allows (it has no refund policy).

Mocks only (``FakeLiteLLM`` from test_litellm_voice.py) except the one real-request test, which
patches httpx and never opens a socket. Key values are dummies and are never printed.
"""

import httpx
import pytest
import yaml
from conftest import REPO_ROOT, sine_wav
from test_litellm_voice import (
    FakeLiteLLM,
    _voice_cfg,
    litellm_error,
    speech,
    transcription,
)

DG_MODEL = "deepgram/nova-3"
GROQ_STT_MODEL = "groq/impulse-test-stt"
OPENAI_STT_MODEL = "openai/impulse-test-transcribe"
AZURE_STT_MODEL = "azure/impulse-test-transcribe"

OPENAI_TTS_MODEL = "openai/impulse-test-tts"
AZURE_TTS_MODEL = "azure/impulse-test-tts"
GROQ_TTS_MODEL = "groq/impulse-test-tts"
ELEVENLABS_TTS_MODEL = "elevenlabs/impulse-test-tts"

DUMMY_KEYS = {
    "DEEPGRAM_API_KEY": "dg-dummy-rev-0123456789",
    "GROQ_API_KEY": "gsk-dummy-rev-0123456789",
    "OPENAI_API_KEY": "sk-dummy-rev-0123456789",
    "AZURE_API_KEY": "az-dummy-rev-0123456789",
    "ELEVENLABS_API_KEY": "el-dummy-rev-0123456789",
}
OTHER_ENV = ("DEEPGRAM_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_BASE", "AZURE_API_BASE", "MY_STT_TOKEN", "MY_TTS_TOKEN")

WAV = sine_wav(0.25)
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(range(256))


@pytest.fixture
def project(impulse_project, monkeypatch):
    """A project with every provider key set to a dummy value (never a real key)."""
    for name in OTHER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)
    return impulse_project


@pytest.fixture
def fake_litellm(monkeypatch):
    return FakeLiteLLM(monkeypatch)


@pytest.fixture
def sleeps(monkeypatch):
    import asyncio

    delays = []

    async def fake_sleep(delay, *args, **kwargs):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


def _stt_for(project, model, timeout_s=30):
    from impulse_evals.voice.stt import get_stt_provider

    cfg = _voice_cfg(project, stt={"provider": "litellm", "model": model}, timeout_s=timeout_s)
    return get_stt_provider(cfg)


def _tts_for(project, model, timeout_s=30, **block):
    from impulse_evals.voice.tts import get_tts_provider

    cfg = _voice_cfg(project, tts={"provider": "litellm", "model": model, **block}, timeout_s=timeout_s)
    return get_tts_provider(cfg)


def _stt_kwargs(fake, index=0):
    return fake.transcription_calls[index]["kwargs"]


def _tts_kwargs(fake, index=0):
    return fake.speech_calls[index]["kwargs"]


# =============================================================================================
# V1: detect_language for Deepgram
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language, subtag", [("hi-IN", "hi"), ("es-MX", "es"), ("en-US", "en"), ("hi-Latn-IN", "hi"), ("ta-in", "ta")]
)
async def test_deepgram_gets_detect_language_and_language(project, fake_litellm, language, subtag):
    await _stt_for(project, DG_MODEL).transcribe(WAV, language)
    kwargs = _stt_kwargs(fake_litellm)
    assert kwargs["detect_language"] == subtag
    assert kwargs["language"] == subtag


@pytest.mark.asyncio
async def test_deepgram_stt_call_kwargs_exact(project, fake_litellm):
    await _stt_for(project, DG_MODEL, timeout_s=17).transcribe(WAV, "hi-IN")
    assert fake_litellm.transcription_calls[0]["args"] == ()
    assert _stt_kwargs(fake_litellm) == {
        "model": DG_MODEL,
        "file": ("audio.wav", WAV),
        "language": "hi",
        "detect_language": "hi",
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
        "drop_params": True,
        "max_retries": 0,
        "timeout": 17,
        "api_key": DUMMY_KEYS["DEEPGRAM_API_KEY"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [GROQ_STT_MODEL, OPENAI_STT_MODEL, AZURE_STT_MODEL])
async def test_other_providers_get_no_detect_language(project, fake_litellm, model):
    await _stt_for(project, model).transcribe(WAV, "hi-IN")
    kwargs = _stt_kwargs(fake_litellm)
    assert "detect_language" not in kwargs
    assert kwargs["language"] == "hi"


DEEPGRAM_ANSWER = {
    "metadata": {"duration": 1.0},
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "namaste",
                        "confidence": 0.9,
                        "words": [{"word": "namaste", "start": 0.1, "end": 0.4, "confidence": 0.8}],
                    }
                ]
            }
        ]
    },
}


@pytest.mark.asyncio
async def test_deepgram_real_request_url_has_detect_language(project, monkeypatch):
    """LiteLLM's own Deepgram request building (no fake ``atranscription``); the outgoing request is
    captured at ``httpx.AsyncClient.send``, which every LiteLLM async HTTP client goes through. No
    socket is opened."""
    import litellm  # imported before the patch (conftest keeps the model cost map local)

    captured = []

    async def fake_async_send(self, request, *args, **kwargs):
        captured.append(request)
        return httpx.Response(200, json=DEEPGRAM_ANSWER, request=request)

    def no_sync_send(self, request, *args, **kwargs):
        raise AssertionError(f"unexpected synchronous HTTP request to {request.url.host}")

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_async_send)
    monkeypatch.setattr(httpx.Client, "send", no_sync_send)
    assert litellm.get_llm_provider(DG_MODEL)[1] == "deepgram"

    result = await _stt_for(project, DG_MODEL).transcribe(WAV, "hi-IN")

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.host == "api.deepgram.com"
    assert request.url.path == "/v1/listen"
    params = dict(request.url.params)
    assert params.get("model") == "nova-3"
    assert params.get("detect_language") == "hi", f"Deepgram query was {sorted(params)}"
    assert request.headers.get("authorization") == "Token " + DUMMY_KEYS["DEEPGRAM_API_KEY"]
    assert result.text == "namaste"


# =============================================================================================
# V2: max_retries=0
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [DG_MODEL, OPENAI_STT_MODEL, GROQ_STT_MODEL])
async def test_stt_passes_max_retries_zero(project, fake_litellm, model):
    await _stt_for(project, model).transcribe(WAV, "en-US")
    assert _stt_kwargs(fake_litellm)["max_retries"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [OPENAI_TTS_MODEL, ELEVENLABS_TTS_MODEL])
async def test_tts_passes_max_retries_zero(project, fake_litellm, extra_imports, model):
    from test_litellm_voice import _fake_pydub

    module, _ = _fake_pydub(sine_wav(0.5))
    extra_imports.module("pydub", module)
    await _tts_for(project, model, voice="nova").synthesize("Hello", "en-US", None)
    assert _tts_kwargs(fake_litellm)["max_retries"] == 0


@pytest.mark.asyncio
async def test_retries_still_three_attempts_with_max_retries_zero(project, fake_litellm, sleeps):
    fake_litellm.transcriptions.extend(
        [litellm_error("RateLimitError", 429, "rate limited"), litellm_error("ServiceUnavailableError", 503, "busy")]
    )
    result = await _stt_for(project, OPENAI_STT_MODEL).transcribe(WAV, "en-US")
    assert result.text == "book a slot"
    assert len(fake_litellm.transcription_calls) == 3
    assert sleeps == [1.0, 2.0]
    assert all(call["kwargs"]["max_retries"] == 0 for call in fake_litellm.transcription_calls)


# =============================================================================================
# V3: response_format="wav" only for openai, azure and OpenAI-compatible providers
# =============================================================================================


def test_groq_is_an_openai_compatible_provider_in_litellm():
    """Precondition of the groq case below (LiteLLM 1.105.1)."""
    import litellm

    assert "groq" in litellm.openai_compatible_providers
    assert "elevenlabs" not in litellm.openai_compatible_providers


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [OPENAI_TTS_MODEL, AZURE_TTS_MODEL, GROQ_TTS_MODEL])
async def test_tts_wav_format_for_openai_azure_and_compatible(project, fake_litellm, model):
    audio = await _tts_for(project, model, voice="nova").synthesize("Hello", "en-US", None)
    assert audio == WAV
    assert _tts_kwargs(fake_litellm)["response_format"] == "wav"


@pytest.mark.asyncio
async def test_tts_openai_call_kwargs_exact(project, fake_litellm):
    await _tts_for(project, OPENAI_TTS_MODEL, timeout_s=17, voice="nova").synthesize("Hola", "es-MX", None)
    assert fake_litellm.speech_calls[0]["args"] == ()
    assert _tts_kwargs(fake_litellm) == {
        "model": OPENAI_TTS_MODEL,
        "input": "Hola",
        "voice": "nova",
        "response_format": "wav",
        "max_retries": 0,
        "timeout": 17,
        "api_key": DUMMY_KEYS["OPENAI_API_KEY"],
    }


@pytest.mark.asyncio
async def test_tts_elevenlabs_no_response_format_and_mp3_converted(project, fake_litellm, extra_imports):
    from test_litellm_voice import _fake_pydub

    converted = sine_wav(0.5)
    module, received = _fake_pydub(converted)
    extra_imports.module("pydub", module)
    fake_litellm.speeches.append(speech(MP3))
    audio = await _tts_for(project, ELEVENLABS_TTS_MODEL, timeout_s=17, voice="Rachel").synthesize(
        "Hello", "en-US", None
    )
    assert audio == converted
    assert received == [MP3]
    assert _tts_kwargs(fake_litellm) == {
        "model": ELEVENLABS_TTS_MODEL,
        "input": "Hello",
        "voice": "Rachel",
        "max_retries": 0,
        "timeout": 17,
        "api_key": DUMMY_KEYS["ELEVENLABS_API_KEY"],
    }


# =============================================================================================
# V4: a 400 about response_format / timestamps is repeated once without those two parameters
# =============================================================================================

FORMAT_400_TEXTS = [
    "OpenAIException - Unsupported value: 'response_format' does not support 'verbose_json' with this model.",
    "OpenAIException - verbose_json is not available for this model",
    "OpenAIException - Timestamp_Granularities is only supported with whisper",
    "OpenAIException - TIMESTAMP options are not available for this model",
]
DROPPED = ("response_format", "timestamp_granularities")


def _without(kwargs, names=DROPPED):
    return {key: value for key, value in kwargs.items() if key not in names}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", FORMAT_400_TEXTS)
async def test_format_400_is_repeated_once_without_format_params(project, fake_litellm, sleeps, text):
    fake_litellm.transcriptions.extend(
        [litellm_error("BadRequestError", 400, text), transcription("hello there")]
    )
    result = await _stt_for(project, OPENAI_STT_MODEL).transcribe(WAV, "en-US")
    assert result.text == "hello there"
    assert result.words == []
    assert len(fake_litellm.transcription_calls) == 2
    first, second = _stt_kwargs(fake_litellm, 0), _stt_kwargs(fake_litellm, 1)
    assert first["response_format"] == "verbose_json"
    assert first["timestamp_granularities"] == ["word"]
    assert "response_format" not in second
    assert "timestamp_granularities" not in second
    assert second == _without(first)   # everything else unchanged (language, max_retries, timeout, key)
    assert sleeps == []                # the repeat is immediate: not a retry


@pytest.mark.asyncio
async def test_after_format_400_later_calls_omit_format_params(project, fake_litellm, sleeps):
    fake_litellm.transcriptions.extend(
        [litellm_error("BadRequestError", 400, FORMAT_400_TEXTS[0]), transcription("one"), transcription("two")]
    )
    stt = _stt_for(project, OPENAI_STT_MODEL)
    await stt.transcribe(WAV, "en-US")
    result = await stt.transcribe(WAV, "es-MX")
    assert result.text == "two"
    assert len(fake_litellm.transcription_calls) == 3
    later = _stt_kwargs(fake_litellm, 2)
    assert "response_format" not in later
    assert "timestamp_granularities" not in later
    assert later["language"] == "es"
    assert sleeps == []


@pytest.mark.asyncio
async def test_format_memory_belongs_to_the_provider_instance(project, fake_litellm, sleeps):
    fake_litellm.transcriptions.extend(
        [litellm_error("BadRequestError", 400, FORMAT_400_TEXTS[0]), transcription("one"), transcription("two")]
    )
    await _stt_for(project, OPENAI_STT_MODEL).transcribe(WAV, "en-US")
    await _stt_for(project, OPENAI_STT_MODEL).transcribe(WAV, "en-US")   # a new provider instance
    fresh = _stt_kwargs(fake_litellm, 2)
    assert fresh["response_format"] == "verbose_json"
    assert fresh["timestamp_granularities"] == ["word"]


@pytest.mark.asyncio
async def test_format_400_repeat_that_fails_again_is_a_voice_error(project, fake_litellm, sleeps):
    from impulse_evals.voice import VoiceError

    fake_litellm.transcriptions.extend(
        [
            litellm_error("BadRequestError", 400, FORMAT_400_TEXTS[0]),
            litellm_error("BadRequestError", 400, FORMAT_400_TEXTS[0]),
            transcription("never reached"),
        ]
    )
    with pytest.raises(VoiceError) as info:
        await _stt_for(project, OPENAI_STT_MODEL).transcribe(WAV, "en-US")
    assert str(info.value).startswith(f"litellm STT failed with model {OPENAI_STT_MODEL}: BadRequestError (HTTP 400)")
    assert len(fake_litellm.transcription_calls) == 2   # repeated once only
    assert sleeps == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, text",
    [
        (400, "OpenAIException - Invalid file format. Supported formats: flac, mp3, wav"),
        (400, "OpenAIException - language 'xx' is not supported"),
        (422, "OpenAIException - response_format is not valid"),
    ],
    ids=["400-file-format", "400-language", "422-response-format"],
)
async def test_other_errors_are_not_repeated(project, fake_litellm, sleeps, status, text):
    from impulse_evals.voice import VoiceError

    fake_litellm.transcriptions.extend([litellm_error("BadRequestError", status, text), transcription("never")])
    stt = _stt_for(project, OPENAI_STT_MODEL)
    with pytest.raises(VoiceError):
        await stt.transcribe(WAV, "en-US")
    assert len(fake_litellm.transcription_calls) == 1
    assert sleeps == []
    await stt.transcribe(WAV, "en-US")   # the instance did not learn to drop the parameters
    assert _stt_kwargs(fake_litellm, 1)["response_format"] == "verbose_json"


@pytest.mark.asyncio
async def test_tts_format_400_is_not_repeated(project, fake_litellm, sleeps):
    from impulse_evals.voice import VoiceError

    fake_litellm.speeches.extend(
        [litellm_error("BadRequestError", 400, "OpenAIException - response_format 'wav' is not supported"), speech(WAV)]
    )
    with pytest.raises(VoiceError):
        await _tts_for(project, OPENAI_TTS_MODEL, voice="nova").synthesize("Hello", "en-US", None)
    assert len(fake_litellm.speech_calls) == 1
    assert sleeps == []


# =============================================================================================
# V9: examples/config.yaml support_order_status_us asks only for what the rubric allows
# =============================================================================================

NOT_IN_RUBRIC = ("refund", "discount", "voucher", "coupon", "compensation")


def test_support_order_status_expected_outcome_fits_the_system_prompt():
    prompt = (REPO_ROOT / "examples" / "system_prompt.txt").read_text(encoding="utf-8").lower()
    config = yaml.safe_load((REPO_ROOT / "examples" / "config.yaml").read_text(encoding="utf-8"))
    matches = [s for s in config["scenarios"] if s.get("name") == "support_order_status_us"]
    assert len(matches) == 1
    outcome = (matches[0].get("expected_outcome") or "").strip().lower()
    assert outcome, "the scenario keeps an expected outcome"
    for word in NOT_IN_RUBRIC:
        assert word not in outcome, f"expected_outcome asks for {word!r}, which system_prompt.txt does not allow"
