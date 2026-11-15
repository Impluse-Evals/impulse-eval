# SPDX-License-Identifier: Apache-2.0
"""``openai_compatible`` voice provider: any server with the OpenAI audio API shape (SPEC.md section 14.1).

* TTS: ``POST {base_url}/audio/speech`` with JSON ``model``, ``input``, ``voice`` (when set) and
  ``response_format: "wav"``; the response body (WAV bytes) is returned unchanged. A 200 answer that
  is not audio (``application/json``, ``application/problem+json`` or ``text/*``, or a body starting
  with ``{``, ``[`` or ``<`` under ``application/octet-stream`` or no content type) is a
  ``VoiceError`` naming the content type and the redacted body (``audio_content``), not retried.
* STT: ``POST {base_url}/audio/transcriptions``, multipart: the audio as ``file`` plus ``model``,
  ``response_format=verbose_json``, ``timestamp_granularities[]=word`` and ``language`` (primary
  subtag, ``hi-IN`` -> ``hi``). Word timings (seconds) become ``WordTiming`` (ms); ``confidence`` is
  the mean word ``probability`` when the server reports one, else None.
* ``api_key_env`` set: ``Authorization: Bearer <value>`` (environment first, then config.toml [keys]).
* Requests time out after ``voice_config.timeout_s`` (default 30 s); 429 and 5xx answers are
  retried twice (three attempts in all), other errors are not retried.
* An error answer's body goes into ``VoiceRequestError`` redacted first (every key the provider
  knows, including the token it sent and the password of a ``user:password@`` in ``base_url``) and
  only then shortened to 200 characters; error texts show the URL without its ``user:password@``
  (httpx still receives the full URL, so Basic auth keeps working).

The HTTP helpers here (``post_with_retries``, ``require_base_url``, ``key_redactor``) are shared with
``impulse_hosted``. httpx clients are created per call without a fixed transport.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Optional

import httpx

from impulse_evals.config import (
    ConfigError,
    is_placeholder,
    redact,
    resolve_key,
    strip_url_credentials,
    url_credentials,
)
from impulse_evals.models import STTResult, WordTiming
from impulse_evals.voice import VoiceError, primary_subtag
from impulse_evals.voice.audio_utils import is_wav, upload_name
from impulse_evals.voice.stt import STTProvider
from impulse_evals.voice.tts import TTSProvider

if TYPE_CHECKING:
    from impulse_evals.config import TomlConfig

MAX_RETRIES = 2                       # retries after the first attempt (SPEC.md: "retry 2x on 429/5xx")
RETRY_BACKOFF_S = (0.2, 0.4)          # waits before retry 1 and 2 when the server sends no Retry-After
MAX_RETRY_AFTER_S = 10.0              # longest Retry-After honoured


class VoiceRequestError(VoiceError):
    """An HTTP voice request failed (an error status after the retries, or a transport error)."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ------------------------------------------------------------------------------------------------
# Shared HTTP helpers
# ------------------------------------------------------------------------------------------------


def require_base_url(block: Any, side: str) -> str:
    """``block.base_url`` without trailing slashes; missing is a ConfigError at ``voice_config.{side}.base_url``."""
    base_url = (block.base_url or "").strip()
    if not base_url:
        raise ConfigError(f"voice_config.{side}.base_url", f"required when provider is {block.provider}")
    return base_url.rstrip("/")


def resolve_bearer(block: Any, side: str, toml: Optional[TomlConfig] = None) -> Optional[str]:
    """The key named by ``block.api_key_env`` (environment, then config.toml [keys]); None when not set."""
    name = (block.api_key_env or "").strip()
    if not name:
        return None
    value = resolve_key(name, toml)
    if value is None:
        raise ConfigError(
            f"voice_config.{side}.api_key_env",
            f"{name} is not set. Set the environment variable {name} or add it to [keys] in .impulse/config.toml.",
        )
    return value


def is_retryable(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header:
        try:
            seconds = float(header)
        except ValueError:
            seconds = math.nan
        if math.isfinite(seconds) and seconds >= 0:
            return min(seconds, MAX_RETRY_AFTER_S)
    return RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]


def key_redactor(
    toml: Optional[TomlConfig] = None, names: Iterable[Optional[str]] = ()
) -> Callable[[str, Iterable[str]], str]:
    """``(text, extra_values) -> text`` with every key a voice provider knows replaced by
    ``[REDACTED]`` in ONE longest-first pass: config.toml [keys] and the variables named there, the
    standard key variables, the variables ``names`` (e.g. the provider's ``api_key_env``), and
    ``extra_values`` (e.g. the token a request sent)."""
    extra = tuple(name.strip() for name in names if isinstance(name, str) and name.strip())
    return lambda text, extra_values=(): redact(text, toml, extra_names=extra, extra_values=extra_values)


def _header_secrets(headers: Optional[Mapping[str, str]]) -> list[str]:
    """The credential a request sends (the ``Authorization`` token), so a body echoing it is redacted."""
    found: list[str] = []
    for name, value in (headers or {}).items():
        if str(name).lower() == "authorization" and isinstance(value, str) and value.strip():
            scheme, _, token = value.strip().partition(" ")
            found.append(token.strip() or scheme)
    return found


def _body_excerpt(response: httpx.Response, redact_text: Callable[[str], str], limit: int = 200) -> str:
    """The error body on one line, redacted FIRST and shortened after (SPEC.md section 8.1): a key
    echoed by the server that crossed the cut would otherwise leak in part."""
    try:
        text = response.text
    except Exception:  # noqa: BLE001 - an undecodable body only loses the excerpt
        return ""
    text = " ".join(redact_text(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _request_redactor(
    url: str,
    headers: Optional[Mapping[str, str]],
    redact_text: Optional[Callable[[str, Iterable[str]], str]],
) -> Callable[[str], str]:
    """``text -> text`` redacting, in one pass, what ``redact_text`` (the provider's ``key_redactor``;
    else the standard key variables) knows plus the token sent in ``headers`` and the URL's credentials."""
    sent = _header_secrets(headers) + url_credentials(url)

    def redact_body(text: str) -> str:
        if redact_text is not None:
            return redact_text(text, sent)
        return redact(text, extra_values=sent)

    return redact_body


# Content types of a 200 answer that is never audio (SPEC.md section 14.1); ``text/*`` too.
NOT_AUDIO_CONTENT_TYPES = ("application/json", "application/problem+json")
# Content types under which a body that starts like JSON or markup (and is not WAV) is not audio.
_UNTYPED_CONTENT_TYPES = ("", "application/octet-stream")


def _is_not_audio(content: bytes, content_type: str) -> bool:
    """True for a 200 answer that is a JSON / text / HTML body rather than audio. WAV bytes are
    always audio; a body under another (audio) content type is left to the WAV conversion."""
    if is_wav(content):
        return False
    media = content_type.split(";", 1)[0].strip().lower()
    if media in NOT_AUDIO_CONTENT_TYPES or media.startswith("text/"):
        return True
    return media in _UNTYPED_CONTENT_TYPES and content.lstrip()[:1] in (b"{", b"[", b"<")


def audio_content(
    response: httpx.Response,
    *,
    label: str,
    url: str,
    headers: Optional[Mapping[str, str]] = None,
    redact_text: Optional[Callable[[str, Iterable[str]], str]] = None,
) -> bytes:
    """The audio bytes of a successful TTS answer, unchanged.

    An empty body raises ``VoiceError("{label}: POST {url} returned no audio")``; a body that is not
    audio (``_is_not_audio``: a JSON error, a proxy's HTML page, a server that ignored
    ``response_format``) raises ``VoiceError("{label}: POST {url} returned {content_type}, not audio:
    {excerpt}")`` with the URL credential-stripped and the body redacted, then shortened. Not retried.
    """
    shown = strip_url_credentials(url)
    content = response.content
    if not content:
        raise VoiceError(f"{label}: POST {shown} returned no audio")
    content_type = (response.headers.get("content-type") or "").strip()
    if _is_not_audio(content, content_type):
        excerpt = _body_excerpt(response, _request_redactor(url, headers, redact_text))
        raise VoiceError(
            f"{label}: POST {shown} returned {content_type or 'no content-type'}, not audio"
            + (f": {excerpt}" if excerpt else "")
        )
    return content


async def post_with_retries(
    url: str,
    *,
    label: str,
    timeout_s: float,
    headers: Optional[Mapping[str, str]] = None,
    json_body: Any = None,
    data: Optional[Mapping[str, Any]] = None,
    files: Optional[Mapping[str, Any]] = None,
    redact_text: Optional[Callable[[str, Iterable[str]], str]] = None,
) -> httpx.Response:
    """POST with up to ``MAX_RETRIES`` retries on 429 / 5xx; returns the first successful response.

    Raises ``VoiceRequestError`` for a final error status or a transport error (never naming a key).
    The message shows ``url`` without its credentials (a ``user:password@`` in ``base_url``, which
    httpx sends as Basic auth); the error body and the transport error text are redacted before they
    are shortened, in one pass: with ``redact_text`` (the provider's ``key_redactor``; else the
    standard key variables), the token the request sent in ``headers``, and the URL's credentials.
    """
    redact_body = _request_redactor(url, headers, redact_text)
    shown = strip_url_credentials(url)

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.post(
                    url, headers=dict(headers or {}), json=json_body, data=data, files=files
                )
            except httpx.TransportError as exc:
                detail = redact_body(str(exc) or type(exc).__name__)
                raise VoiceRequestError(f"{label}: POST {shown} failed: {type(exc).__name__}: {detail}") from exc
            if response.status_code < 400:
                return response
            if is_retryable(response.status_code) and attempt < MAX_RETRIES:
                await asyncio.sleep(_retry_delay(response, attempt))
                continue
            attempts = f" after {attempt + 1} attempts" if attempt else ""
            excerpt = _body_excerpt(response, redact_body)
            raise VoiceRequestError(
                f"{label}: POST {shown} returned HTTP {response.status_code}{attempts}"
                + (f": {excerpt}" if excerpt else ""),
                status_code=response.status_code,
            )
    raise AssertionError("unreachable")  # pragma: no cover


def bearer_headers(key: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"} if key else {}


def _seconds_to_ms(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value) * 1000.0)


def _probability(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def parse_transcription(response: httpx.Response) -> STTResult:
    """``STTResult`` from an OpenAI-style transcription answer (``verbose_json``, ``json`` or plain text)."""
    try:
        data = response.json()
    except ValueError:
        return STTResult(text=response.text.strip())
    if not isinstance(data, dict):
        raise VoiceError("openai_compatible STT: the server answered with JSON that is not an object")
    text = data.get("text")
    text = text.strip() if isinstance(text, str) else ""
    raw_words = data.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        # Some servers put the word list inside each segment instead of at the top level.
        segments = data.get("segments") if isinstance(data.get("segments"), list) else []
        raw_words = []
        for segment in segments:
            if isinstance(segment, dict) and isinstance(segment.get("words"), list):
                raw_words.extend(segment["words"])
    words: list[WordTiming] = []
    probabilities: list[float] = []
    for item in raw_words:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word") or "").strip()
        start_ms, end_ms = _seconds_to_ms(item.get("start")), _seconds_to_ms(item.get("end"))
        if not word or start_ms is None or end_ms is None:
            continue
        probability = _probability(item.get("probability"))
        if probability is not None:
            probabilities.append(probability)
        words.append(
            WordTiming(word=word, start_ms=start_ms, end_ms=end_ms,
                       confidence=probability if probability is not None else 0.0)
        )
    confidence = sum(probabilities) / len(probabilities) if probabilities else None
    return STTResult(text=text, confidence=confidence, words=words)


def _model(value: Optional[str]) -> Optional[str]:
    return None if is_placeholder(value) else value.strip()


# ------------------------------------------------------------------------------------------------
# Providers
# ------------------------------------------------------------------------------------------------


class OpenAICompatibleTTS(TTSProvider):
    name = "openai_compatible"

    def __init__(self, block: Any, *, timeout_s: float = 30.0, toml: Optional[TomlConfig] = None) -> None:
        self.block = block
        self.timeout_s = timeout_s
        self.url = require_base_url(block, "tts") + "/audio/speech"
        self._headers = bearer_headers(resolve_bearer(block, "tts", toml))
        self._redact = key_redactor(toml, (block.api_key_env,))

    def model_for(self, language: str) -> Optional[str]:
        """``models_by_language[language]`` when set, else ``model`` (None when unset or a placeholder)."""
        return _model(self.block.models_by_language.get(language)) or _model(self.block.model)

    async def synthesize(self, text: str, language: str, voice: Optional[str]) -> bytes:
        body: dict[str, Any] = {}
        model = self.model_for(language)
        if model:
            body["model"] = model
        body["input"] = text
        chosen = voice or self.block.voice
        if chosen:
            body["voice"] = chosen
        body["response_format"] = "wav"
        response = await post_with_retries(
            self.url, label="openai_compatible TTS", timeout_s=self.timeout_s, headers=self._headers,
            json_body=body, redact_text=self._redact,
        )
        return audio_content(
            response, label="openai_compatible TTS", url=self.url, headers=self._headers, redact_text=self._redact
        )


class OpenAICompatibleSTT(STTProvider):
    name = "openai_compatible"

    def __init__(self, block: Any, *, timeout_s: float = 30.0, toml: Optional[TomlConfig] = None) -> None:
        self.block = block
        self.timeout_s = timeout_s
        self.url = require_base_url(block, "stt") + "/audio/transcriptions"
        self._headers = bearer_headers(resolve_bearer(block, "stt", toml))
        self._redact = key_redactor(toml, (block.api_key_env,))

    async def transcribe(self, audio: bytes, language: str) -> STTResult:
        fields: dict[str, Any] = {
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word"],
            "language": primary_subtag(language),
        }
        model = _model(self.block.model)
        if model:
            fields["model"] = model
        filename, content_type = upload_name(audio)
        response = await post_with_retries(
            self.url,
            label="openai_compatible STT",
            timeout_s=self.timeout_s,
            headers=self._headers,
            data=fields,
            files={"file": (filename, audio, content_type)},
            redact_text=self._redact,
        )
        return parse_transcription(response)
