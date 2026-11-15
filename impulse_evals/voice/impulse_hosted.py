# SPDX-License-Identifier: Apache-2.0
"""``impulse_hosted`` voice provider: optional client for an Impulse-hosted voice API (SPEC.md section 14.2).

Only the client is built (the hosted server is a non-goal). ``base_url`` has no default and must be
set. Every request carries ``Authorization: Bearer <IMPULSE_VOICE_API_KEY>`` (environment first,
then config.toml [keys]; ``api_key_env`` names another variable when set).

* TTS: ``POST {base_url}/v1/tts``, JSON ``{text, language, voice}``, answer ``audio/wav``. Default
  voices: en-IN ``af_sarah``, hi-IN ``hi_aryan``, en-US ``af_bella``. A 200 answer that is not audio
  (a JSON or HTML body) is a ``VoiceError`` with the redacted body (``audio_content``), not retried.
* STT: ``POST {base_url}/v1/stt``, multipart (the audio as ``file`` plus ``language``), answer JSON
  ``{text, confidence, words[], filler_words[]}``. A ``words[]`` entry with a missing or null
  ``confidence`` gets 0.0; an entry that is not an object, or lacks ``word``, ``start_ms`` or
  ``end_ms``, is dropped.
* Supported languages: en-IN, en-US, hi-IN; any other language raises before a request is sent.
* 429 and 5xx answers are retried twice; requests time out after ``voice_config.timeout_s`` (30 s).
* Error texts show the URL without the credentials of a ``user:password@`` in ``base_url`` and redact
  them from the error body (``post_with_retries``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

from pydantic import ValidationError

from impulse_evals.config import ConfigError, require_key
from impulse_evals.models import STTResult
from impulse_evals.voice import VoiceError
from impulse_evals.voice.audio_utils import upload_name
from impulse_evals.voice.openai_compatible import (
    audio_content,
    bearer_headers,
    key_redactor,
    post_with_retries,
    require_base_url,
)
from impulse_evals.voice.stt import STTProvider
from impulse_evals.voice.tts import TTSProvider

if TYPE_CHECKING:
    from impulse_evals.config import TomlConfig

KEY_NAME = "IMPULSE_VOICE_API_KEY"
SUPPORTED_LANGUAGES = ("en-IN", "en-US", "hi-IN")
DEFAULT_VOICES = {"en-IN": "af_sarah", "hi-IN": "hi_aryan", "en-US": "af_bella"}


class UnsupportedLanguageError(VoiceError):
    """The Impulse-hosted voice API does not support a scenario language."""


def supported_language(language: str) -> str:
    """The supported code matching ``language`` (case-insensitive), or ``UnsupportedLanguageError``."""
    for code in SUPPORTED_LANGUAGES:
        if code.lower() == language.strip().lower():
            return code
    raise UnsupportedLanguageError(
        f"impulse_hosted does not support language {language}. Supported languages: "
        f"{', '.join(SUPPORTED_LANGUAGES)}. Use another voice provider (local_hf or openai_compatible) "
        f"for {language}."
    )


def _api_key(block: Any, side: str, toml: Optional[TomlConfig]) -> str:
    name = (block.api_key_env or "").strip() or KEY_NAME
    try:
        return require_key(name, toml)
    except ConfigError as exc:
        field = f"voice_config.{side}.api_key_env" if name != KEY_NAME else exc.field
        raise ConfigError(field, f"impulse_hosted needs a voice API key. {exc.message}") from None


class ImpulseHostedTTS(TTSProvider):
    name = "impulse_hosted"

    def __init__(self, block: Any, *, timeout_s: float = 30.0, toml: Optional[TomlConfig] = None) -> None:
        self.block = block
        self.timeout_s = timeout_s
        self.url = require_base_url(block, "tts") + "/v1/tts"
        self._headers = bearer_headers(_api_key(block, "tts", toml))
        self._redact = key_redactor(toml, (block.api_key_env, KEY_NAME))

    def check_language(self, language: str) -> None:
        supported_language(language)

    async def synthesize(self, text: str, language: str, voice: Optional[str]) -> bytes:
        code = supported_language(language)
        body = {"text": text, "language": code, "voice": voice or self.block.voice or DEFAULT_VOICES[code]}
        response = await post_with_retries(
            self.url, label="impulse_hosted TTS", timeout_s=self.timeout_s, headers=self._headers,
            json_body=body, redact_text=self._redact,
        )
        return audio_content(
            response, label="impulse_hosted TTS", url=self.url, headers=self._headers, redact_text=self._redact
        )


class ImpulseHostedSTT(STTProvider):
    name = "impulse_hosted"

    def __init__(self, block: Any, *, timeout_s: float = 30.0, toml: Optional[TomlConfig] = None) -> None:
        self.block = block
        self.timeout_s = timeout_s
        self.url = require_base_url(block, "stt") + "/v1/stt"
        self._headers = bearer_headers(_api_key(block, "stt", toml))
        self._redact = key_redactor(toml, (block.api_key_env, KEY_NAME))

    def check_language(self, language: str) -> None:
        supported_language(language)

    async def transcribe(self, audio: bytes, language: str) -> STTResult:
        code = supported_language(language)
        filename, content_type = upload_name(audio)
        response = await post_with_retries(
            self.url,
            label="impulse_hosted STT",
            timeout_s=self.timeout_s,
            headers=self._headers,
            data={"language": code},
            files={"file": (filename, audio, content_type)},
            redact_text=self._redact,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise VoiceError("impulse_hosted STT: the answer is not JSON") from exc
        if not isinstance(data, dict):
            raise VoiceError("impulse_hosted STT: the answer is not a JSON object")
        words, server_index = _word_entries(data.get("words") or [])
        payload = {
            "text": (data.get("text") or "").strip() if isinstance(data.get("text"), str) else "",
            "confidence": data.get("confidence"),
            "words": words,
            "filler_words": data.get("filler_words") or [],
        }
        try:
            return STTResult.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in _server_location(first.get("loc", ()), server_index))
            raise VoiceError(f"impulse_hosted STT: unexpected answer at {where}: {first.get('msg')}") from None


_WORD_KEYS = ("word", "start_ms", "end_ms")


def _word_entries(raw: Any) -> tuple[Any, list[int]]:
    """The ``words[]`` entries to validate, with the index each kept entry had in the server's list:
    an entry that is not a mapping, or lacks ``word``, ``start_ms`` or ``end_ms``, is dropped; a
    missing or null ``confidence`` becomes 0.0 (as ``openai_compatible`` keeps a word without a
    probability). A ``words`` that is not a list is passed on unchanged (the validation names it)."""
    if not isinstance(raw, (list, tuple)):
        return raw, []
    entries: list[dict[str, Any]] = []
    indices: list[int] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or any(item.get(key) is None for key in _WORD_KEYS):
            continue
        entry = dict(item)
        if entry.get("confidence") is None:
            entry["confidence"] = 0.0
        entries.append(entry)
        indices.append(index)
    return entries, indices


def _server_location(location: Any, server_index: list[int]) -> tuple[Any, ...]:
    """A validation error location with a ``words`` entry index mapped back to the server's index
    (entries dropped before validation shift the kept ones)."""
    parts = tuple(location)
    if (
        len(parts) >= 2
        and parts[0] == "words"
        and isinstance(parts[1], int)
        and not isinstance(parts[1], bool)
        and 0 <= parts[1] < len(server_index)
    ):
        return ("words", server_index[parts[1]], *parts[2:])
    return parts
