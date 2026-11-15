# SPDX-License-Identifier: Apache-2.0
"""``litellm`` voice provider: speech to text and text to speech through LiteLLM (FIXES-ROUND-3-SPEC.md
B2 and B2-rev).

* ``voice_config.{tts,stt}.model`` is a LiteLLM model string (``deepgram/...``, ``groq/...``,
  ``openai/...``, ...); LiteLLM's own provider list is the authority. ``base_url`` (optional) is
  passed as ``api_base``.
* Keys follow LiteLLM's conventions: the variable is ``api_key_env`` when set, else
  ``config.key_name_for_model(model)``; its value (environment, then config.toml [keys]) is passed
  as ``api_key``. A model that needs no Impulse key (local or credential-file providers) gets no
  ``api_key`` and LiteLLM reads its own environment. A named key that is not set is a ConfigError
  at ``voice_config.{side}.model``, raised when the provider is built (before any paid call).
* STT: ``litellm.atranscription`` with the WAV Impulse has (no conversion), the primary language
  subtag, ``verbose_json`` and word timestamps. A model whose LiteLLM provider is ``deepgram`` also
  gets ``detect_language=<primary subtag>`` (LiteLLM does not put ``language`` in the Deepgram URL,
  but forwards Deepgram's own query parameters). An HTTP 400 whose text names ``response_format``,
  ``verbose_json`` or ``timestamp`` is repeated once at once without ``response_format`` and
  ``timestamp_granularities``, and the provider instance then omits both (no word timings).
  Word timings (seconds) become ``WordTiming`` (ms); a word's confidence is its ``probability`` or
  ``confidence``, else the Deepgram raw answer's word confidence (``_hidden_params``, only when its
  word list has the same length), else 0.0. ``STTResult.confidence`` is the mean of the word
  confidences that came from the answer, else the Deepgram alternative's confidence, else None.
* TTS: ``litellm.aspeech`` with ``response_format="wav"`` for ``openai``, ``azure`` and LiteLLM's
  OpenAI-compatible providers; other providers (e.g. ``elevenlabs``, which has no WAV) get no
  ``response_format``. The bytes go through ``audio_utils.ensure_wav`` (other formats are converted
  with pydub, the ``voice`` extra, plus ffmpeg).
* ``max_retries=0`` on every call (the OpenAI SDK under LiteLLM would otherwise retry inside each of
  Impulse's attempts).
* Errors: one line, ``VoiceError("litellm {STT|TTS} failed with model {model}: {text}")``, key values
  redacted; 2 retries (``asyncio.sleep(1.0 * attempt)``) when the error has ``status_code`` 429 or
  >= 500, none on other errors. A ``ConfigError`` is never wrapped.

LiteLLM is imported inside the methods only, and called through the module (``litellm.atranscription``
/ ``litellm.aspeech``) so tests can patch it.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

from impulse_evals.config import (
    ConfigError,
    is_placeholder,
    key_name_for_model,
    redact,
    resolve_key,
)
from impulse_evals.models import STTResult, WordTiming
from impulse_evals.provider_errors import MESSAGE_MAX_CHARS, provider_error_text
from impulse_evals.voice import VoiceError, primary_subtag
from impulse_evals.voice.audio_utils import ensure_wav
from impulse_evals.voice.stt import STTProvider
from impulse_evals.voice.tts import TTSProvider

if TYPE_CHECKING:
    from impulse_evals.config import TomlConfig

__all__ = ["LiteLLMSTT", "LiteLLMTTS", "parse_litellm_transcription"]

MAX_ATTEMPTS = 3                 # the first attempt plus 2 retries on 429 / >= 500
RETRY_DELAY_S = 1.0              # wait before retry n: RETRY_DELAY_S * n
KEY_HINT = "set the environment variable or add it to [keys] in .impulse/config.toml"
# Transcription parameters left out after an HTTP 400 that names them (B2-rev V4).
FORMAT_PARAMS = ("response_format", "timestamp_granularities")
_FORMAT_ERROR_WORDS = ("response_format", "verbose_json", "timestamp_granularities", "timestamp")
# TTS providers asked for WAV (B2-rev V3); LiteLLM's openai_compatible_providers are added at call time.
WAV_TTS_PROVIDERS = ("openai", "azure")


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


def _field(obj: Any, name: str) -> Any:
    """``obj[name]`` for a mapping, else ``obj.name`` (None when absent)."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _ms(value: Any) -> Optional[int]:
    seconds = _number(value)
    return None if seconds is None else round(seconds * 1000.0)


def _status(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    return None if isinstance(status, bool) or not isinstance(status, int) else status


def _is_retryable(exc: BaseException) -> bool:
    status = _status(exc)
    return status is not None and (status == 429 or status >= 500)


def _cut(text: str) -> str:
    return text if len(text) <= MESSAGE_MAX_CHARS else text[: MESSAGE_MAX_CHARS - 3] + "..."


def _first_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return " ".join(lines[0].split()) if lines else ""


def _deepgram_alternative(response: Any) -> Optional[Mapping[str, Any]]:
    """``_hidden_params["results"]["channels"][0]["alternatives"][0]`` of a Deepgram-shaped raw answer."""
    hidden = getattr(response, "_hidden_params", None)
    try:
        alternative = hidden["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError, TypeError):
        return None
    return alternative if isinstance(alternative, Mapping) else None


def parse_litellm_transcription(response: Any) -> STTResult:
    """``STTResult`` from a LiteLLM transcription answer (``TranscriptionResponse`` or a mapping)."""
    text = _field(response, "text")
    text = text.strip() if isinstance(text, str) else ""
    alternative = _deepgram_alternative(response)
    raw_words = _field(response, "words")
    raw_words = list(raw_words) if isinstance(raw_words, Sequence) and not isinstance(raw_words, str) else []
    deepgram_words = alternative.get("words") if alternative is not None else None
    if not isinstance(deepgram_words, Sequence) or len(deepgram_words) != len(raw_words):
        deepgram_words = None

    words: list[WordTiming] = []
    reported: list[float] = []
    for index, item in enumerate(raw_words):
        word, start_ms, end_ms = _field(item, "word"), _ms(_field(item, "start")), _ms(_field(item, "end"))
        if not isinstance(word, str) or not word.strip() or start_ms is None or end_ms is None:
            continue
        confidence = _number(_field(item, "probability"))
        if confidence is None:
            confidence = _number(_field(item, "confidence"))
        if confidence is None and deepgram_words is not None:
            confidence = _number(_field(deepgram_words[index], "confidence"))
        if confidence is not None:
            reported.append(confidence)
        words.append(
            WordTiming(word=word.strip(), start_ms=start_ms, end_ms=end_ms,
                       confidence=confidence if confidence is not None else 0.0)
        )
    if reported:
        overall: Optional[float] = sum(reported) / len(reported)
    else:
        overall = _number(alternative.get("confidence")) if alternative is not None else None
    return STTResult(text=text, confidence=overall, words=words)


async def _speech_bytes(response: Any) -> Any:
    """The audio bytes of a LiteLLM speech answer: ``content``, else ``await aread()`` / ``read()``."""
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)
    content = getattr(response, "content", None)
    if content is None:
        reader = getattr(response, "aread", None) or getattr(response, "read", None)
        if reader is not None:
            content = reader()
            if inspect.isawaitable(content):
                content = await content
    return bytes(content) if isinstance(content, (bytearray, memoryview)) else content


def _litellm():
    import litellm  # lazy: heavy import, only needed when a voice request is made

    litellm.suppress_debug_info = True   # no "Give Feedback / Get Help" banner
    return litellm


# ------------------------------------------------------------------------------------------------
# Providers
# ------------------------------------------------------------------------------------------------


class _LiteLLMVoice:
    """Shared setup of the litellm STT and TTS providers: model, key, api_base, redaction, retries."""

    side = ""
    label = ""

    def __init__(self, block: Any, *, timeout_s: float = 30.0, toml: Optional[TomlConfig] = None) -> None:
        self.block = block
        self.timeout_s = timeout_s
        self._toml = toml
        field = f"voice_config.{self.side}.model"
        if is_placeholder(block.model):
            raise ConfigError(field, "required when provider is litellm")
        self.model: str = block.model.strip()
        self.api_base: Optional[str] = (block.base_url or "").strip() or None
        self._key_names = tuple(name for name in ((block.api_key_env or "").strip(),) if name)
        self._api_key = self._resolve_api_key(field)
        self._provider: Optional[str] = None
        self._provider_known = False

    def _resolve_api_key(self, field: str) -> Optional[str]:
        name = self._key_names[0] if self._key_names else key_name_for_model(self.model, field=field)
        if name is None:
            return None
        value = resolve_key(name, self._toml)
        if value is None:
            raise ConfigError(field, f"{self.model} needs the key {name}: {KEY_HINT}")
        return value

    def provider(self) -> Optional[str]:
        """LiteLLM's provider of ``model`` (``litellm.get_llm_provider``, offline); None when unknown."""
        if not self._provider_known:
            litellm = _litellm()
            try:
                with contextlib.redirect_stdout(io.StringIO()):   # LiteLLM may print a provider banner
                    self._provider = str(litellm.get_llm_provider(self.model)[1])
            except Exception:  # noqa: BLE001 - an unknown provider: no provider-specific parameter
                self._provider = None
            self._provider_known = True
        return self._provider

    def _call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs["max_retries"] = 0
        kwargs["timeout"] = self.timeout_s
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        return kwargs

    def _redact(self, text: str) -> str:
        values = (self._api_key,) if self._api_key else ()
        return redact(text, self._toml, extra_names=self._key_names, extra_values=values)

    def _error_text(self, exc: BaseException) -> str:
        text = provider_error_text(exc, self._redact)
        if text is None:
            try:
                message = _first_line(self._redact(str(exc)))
            except Exception:  # noqa: BLE001 - a broken __str__ must not hide the error
                message = ""
            text = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
        return _cut(_first_line(text))

    def _error(self, exc: BaseException) -> VoiceError:
        return VoiceError(f"litellm {self.label} failed with model {self.model}: {self._error_text(exc)}")

    def _repeat_without_format(self, exc: BaseException) -> bool:
        """Whether a failed call is repeated at once without ``FORMAT_PARAMS`` (STT only, B2-rev V4)."""
        return False

    async def _call(self, function_name: str, kwargs: dict[str, Any]) -> Any:
        """``await litellm.<function_name>(**kwargs)`` with the retries; errors become one-line VoiceErrors."""
        litellm = _litellm()
        attempt = 1
        while True:
            try:
                return await getattr(litellm, function_name)(**kwargs)
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - every LiteLLM failure becomes a VoiceError
                if self._repeat_without_format(exc):
                    kwargs = {key: value for key, value in kwargs.items() if key not in FORMAT_PARAMS}
                    continue   # an immediate repeat, not a retry
                if attempt < MAX_ATTEMPTS and _is_retryable(exc):
                    await asyncio.sleep(RETRY_DELAY_S * attempt)
                    attempt += 1
                    continue
                error = self._error(exc)
            raise error   # outside the except block: no chained exception text that could hold a key


class LiteLLMSTT(_LiteLLMVoice, STTProvider):
    name = "litellm"
    side = "stt"
    label = "STT"

    def __init__(self, block: Any, *, timeout_s: float = 30.0, toml: Optional[TomlConfig] = None) -> None:
        super().__init__(block, timeout_s=timeout_s, toml=toml)
        self.omit_format = False   # set after an HTTP 400 about response_format / timestamps

    def _repeat_without_format(self, exc: BaseException) -> bool:
        if self.omit_format or _status(exc) != 400:
            return False
        text = self._error_text(exc).lower()
        if not any(word in text for word in _FORMAT_ERROR_WORDS):
            return False
        self.omit_format = True
        return True

    async def transcribe(self, audio: bytes, language: str) -> STTResult:
        subtag = primary_subtag(language)
        kwargs: dict[str, Any] = {"model": self.model, "file": ("audio.wav", audio), "language": subtag}
        if self.provider() == "deepgram":
            kwargs["detect_language"] = subtag
        if not self.omit_format:
            kwargs["response_format"] = "verbose_json"
            kwargs["timestamp_granularities"] = ["word"]
        kwargs["drop_params"] = True
        response = await self._call("atranscription", self._call_kwargs(kwargs))
        return parse_litellm_transcription(response)


class LiteLLMTTS(_LiteLLMVoice, TTSProvider):
    name = "litellm"
    side = "tts"
    label = "TTS"

    def asks_for_wav(self) -> bool:
        """True for ``openai``, ``azure`` and LiteLLM's OpenAI-compatible providers (B2-rev V3)."""
        provider = self.provider()
        if provider is None:
            return False
        compatible = getattr(_litellm(), "openai_compatible_providers", None) or ()
        return provider in WAV_TTS_PROVIDERS or provider in compatible

    async def synthesize(self, text: str, language: str, voice: Optional[str]) -> bytes:
        kwargs: dict[str, Any] = {"model": self.model, "input": text}
        chosen = voice or self.block.voice
        if chosen:
            kwargs["voice"] = chosen
        if self.asks_for_wav():
            kwargs["response_format"] = "wav"
        response = await self._call("aspeech", self._call_kwargs(kwargs))
        try:
            content = await _speech_bytes(response)
        except Exception as exc:  # noqa: BLE001 - reading the answer failed: one line, not retried
            error = self._error(exc)
        else:
            error = None
        if error is not None:
            raise error
        if not isinstance(content, bytes) or not content:
            raise VoiceError(f"litellm TTS failed with model {self.model}: the answer holds no audio")
        return ensure_wav(content)
