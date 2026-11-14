# SPDX-License-Identifier: AGPL-3.0-only
"""HTTP adapter: ``agent.adapter: http`` (FIXES-ROUND-3-SPEC.md B4).

An agent in any language behind three JSON endpoints (``agent.http`` in config.yaml): ``url`` (one
POST per agent turn, required), ``start_url`` and ``close_url`` (one POST when a scenario starts /
ends, optional). One ``httpx.AsyncClient`` per scenario, created in ``start()`` (which also warms up
anyio's async backend, so its first use is not counted as agent latency) and closed in ``close()``.
The timeout is ``http.turn_timeout_s``, else ``voice_config.timeout_s``, else 30 s: httpx's timeouts
and a deadline for the whole request, body included (``asyncio.wait_for``), for every request. No
retries: a turn can have side effects (a booking, a payment).

Every request: method POST, ``Content-Type: application/json``, the ``http.headers`` mapping (printable
ASCII values, checked by ``load_config``), and ``Authorization: Bearer <value>`` when
``http.api_key_env`` names a variable (environment, then config.toml ``[keys]``; the value is stripped;
not set, or not printable ASCII, is a ``ConfigError`` at ``agent.http.api_key_env`` raised in
``start()``, never showing the value). With ``api_key_env`` set, the URLs' userinfo is not sent (so
httpx builds no Basic auth); without it, userinfo is Basic auth, as httpx does. Bodies are UTF-8 JSON:

- start: ``{"conversation_id", "scenario_name", "language", "direction"}``; a 2xx answer that is a
  JSON object with a string ``system_prompt`` gives the agent's system prompt (used only when
  config.yaml sets none); any other 2xx body is ignored.
- turn: ``{"conversation_id", "user_text", "history", "scenario_name"}``; ``user_text`` is null for
  the outbound opener (``send(None, None)``), which adds no user entry to ``history`` (the prior turns
  of this conversation only, as the generic adapter). Voice mode: the agent receives text and, only
  with ``http.send_audio: true``, an extra ``"user_audio"`` key (base64 of the caller's WAV, standard
  alphabet with padding; with noise simulation, the noisy audio). Without it, in text mode, and for the
  opener the body has exactly the four keys above, so agents that ignore the key keep working.
- turn answer: a JSON object; ``text`` (missing or null: ""), ``tool_calls`` (the generic adapter's
  rules, ``to_tool_calls`` with the source "HTTP agent"), ``heard_text``, ``audio`` (base64 of a WAV
  file, standard or URL-safe alphabet, padding optional). Other keys are ignored.
- close: ``{"conversation_id", "scenario_name"}``; the answer body is ignored.

``conversation_id`` is ``"impulse-"`` plus 12 lowercase hex characters, new for every ``start()``.
Latency is measured by Impulse around the turn request (``time.perf_counter``).

Error texts show the URL without credentials (``strip_url_credentials``); bodies and transport
errors are redacted FIRST (the run's redactor, the bearer token, the ``headers`` values, the URL's
credentials), then put on one line and cut to 200 characters; an empty body is ``(empty body)``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import secrets
import time
from collections.abc import Callable, Mapping
from typing import Any, Optional

from urllib.parse import urlsplit, urlunsplit

import anyio
import httpx

from impulse_evals.adapters.base import AgentInterface, AgentReply
from impulse_evals.adapters.generic_adapter import to_tool_calls
from impulse_evals.config import (
    ConfigError,
    HTTPConfig,
    ProjectConfig,
    is_printable_ascii,
    load_project_toml,
    read_system_prompt,
    redact,
    resolve_key,
    secret_text,
    strip_url_credentials,
    url_credentials,
)
from impulse_evals.models import Scenario

__all__ = ["HTTPAdapter"]

SOURCE = "HTTP agent"   # the tool-call and field error texts name the agent this way
DEFAULT_TIMEOUT_S = 30.0
ERROR_TEXT_CHARS = 200
EMPTY_BODY = "(empty body)"
CONVERSATION_PREFIX = "impulse-"

NON_2XX_ERROR = "HTTP agent returned HTTP {status} for POST {url}: {body}"
TIMEOUT_ERROR = "HTTP agent did not answer within {seconds:g} s (POST {url})."
UNREACHABLE_ERROR = "could not reach the HTTP agent at {url}: {error}"
NOT_OBJECT_ERROR = "HTTP agent returned a body that is not a JSON object (POST {url}): {body}"
MISSING_KEY_ERROR = "{name} is not set: set the environment variable or add it to [keys] in .impulse/config.toml"
TOKEN_CHARACTERS_ERROR = "the value of {name} must be printable ASCII without control characters"

# Every httpx error that is not a timeout: HTTPError (transport, protocol, redirects) and the classes that
# do not derive from it (InvalidURL, CookieConflict, StreamError) give the "could not reach" text (R5).
_HTTPX_ERRORS = (httpx.HTTPError, httpx.InvalidURL, httpx.CookieConflict, httpx.StreamError)


class HTTPAdapter(AgentInterface):
    """Posts every agent turn to ``agent.http.url`` as JSON."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        redact: Optional[Callable[[str], str]] = None,
    ) -> None:
        http = config.agent.http
        if http is None or not http.url:
            raise ConfigError("agent.http.url", "required when agent.adapter is http")
        self._config = config
        self._http: HTTPConfig = http
        self._transport = transport
        self._run_redact = redact
        self._timeout_s = _timeout_s(config, http)
        self._client: Optional[httpx.AsyncClient] = None
        self._headers = httpx.Headers()
        self._secrets: list[str] = []
        self._conversation_id = ""
        self._scenario_name = ""
        self._history: list[dict[str, str]] = []
        self._config_prompt: Optional[str] = None
        self._config_prompt_loaded = False
        self._agent_prompt: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------------------------------

    async def start(self, scenario: Scenario) -> None:
        """Resolve the bearer key, open the client, and POST ``start_url`` (when set)."""
        await self._close_client()
        token = self._bearer_token()   # a missing or unusable key: ConfigError, before any request
        self._headers = httpx.Headers(self._http.headers)   # case-insensitive: the two below replace a configured one
        self._headers["Content-Type"] = "application/json"
        if token is not None:
            self._headers["Authorization"] = f"Bearer {token}"
        self._secrets = _known_secrets(self._http, token)
        self._conversation_id = CONVERSATION_PREFIX + secrets.token_hex(6)
        self._scenario_name = scenario.name
        self._history = []
        self._agent_prompt = None
        self._client = httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport)
        # Warm-up (B3/B4-rev R3): anyio loads its asyncio backend on first use (httpcore triggers it on
        # the first request of the process, about a second on a cold start); loading it here keeps that
        # cost out of the first timed turn.
        await anyio.sleep(0)
        if not self._http.start_url:
            return
        body = {
            "conversation_id": self._conversation_id,
            "scenario_name": scenario.name,
            "language": scenario.language,
            "direction": scenario.direction,
        }
        try:
            response = await self._post(self._http.start_url, body)
        except BaseException:
            await self._close_client()   # a failed start leaves no open client and sends no close
            raise
        self._agent_prompt = _start_prompt(response)

    async def send(self, user_text: Optional[str], user_audio: Optional[bytes]) -> AgentReply:
        """POST one turn. ``user_audio`` is sent (base64, key ``user_audio``) only with ``http.send_audio``."""
        if self._client is None:
            raise RuntimeError("HTTPAdapter.send() was called before start()")
        body: dict[str, Any] = {
            "conversation_id": self._conversation_id,
            "user_text": user_text,
            "history": copy.deepcopy(self._history),
            "scenario_name": self._scenario_name,
        }
        if self._http.send_audio and user_audio:
            body["user_audio"] = base64.b64encode(bytes(user_audio)).decode("ascii")
        url = self._http.url or ""
        started = time.perf_counter()
        response = await self._post(url, body)
        latency_ms = (time.perf_counter() - started) * 1000.0

        reply = self._to_reply(response, url, latency_ms)
        if user_text is not None:
            self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": reply.text})
        return reply

    async def close(self) -> None:
        """POST ``close_url`` (when set; the answer body is ignored), then close the client."""
        if self._client is None:
            return
        try:
            if self._http.close_url:
                body = {"conversation_id": self._conversation_id, "scenario_name": self._scenario_name}
                await self._post(self._http.close_url, body)
        finally:
            await self._close_client()

    async def _close_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # -- system prompt -----------------------------------------------------------------------------

    def get_system_prompt(self) -> Optional[str]:
        """config.yaml ``agent.system_prompt`` text, else the start answer's ``system_prompt``."""
        if self._config.agent.system_prompt is not None:
            if not self._config_prompt_loaded:
                self._config_prompt = read_system_prompt(self._config)
                self._config_prompt_loaded = True
            return self._config_prompt
        return self._agent_prompt

    # -- requests ----------------------------------------------------------------------------------

    def _bearer_token(self) -> Optional[str]:
        name = (self._http.api_key_env or "").strip()
        if not name:
            return None
        value = resolve_key(name, load_project_toml())   # the environment, then ./.impulse/config.toml
        value = value.strip() if value is not None else ""   # e.g. a trailing newline from a file
        if not value:   # unset, or blank after stripping: not set
            raise ConfigError("agent.http.api_key_env", MISSING_KEY_ERROR.format(name=name))
        if not is_printable_ascii(value):   # httpx's header error would echo it (B3/B4-rev R2, R6)
            raise ConfigError("agent.http.api_key_env", TOKEN_CHARACTERS_ERROR.format(name=name))
        return value

    def _request_url(self, url: str) -> str:
        """The URL a request goes to: with ``api_key_env`` set, without its userinfo, so httpx builds no
        Basic auth that would replace the Bearer header (B3/B4-rev R4); otherwise as configured."""
        return _without_userinfo(url) if (self._http.api_key_env or "").strip() else url

    async def _post(self, url: str, body: Mapping[str, Any]) -> httpx.Response:
        """POST ``body`` to ``url``; a non-2xx answer, a timeout or a transport error is a RuntimeError.

        ``turn_timeout_s`` is a deadline for the whole request, body included (``asyncio.wait_for``),
        on top of httpx's own timeouts: a server that trickles bytes cannot hold a turn open (R1)."""
        assert self._client is not None
        shown = strip_url_credentials(url)
        content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        error: Optional[RuntimeError] = None
        try:
            response = await asyncio.wait_for(
                self._client.post(self._request_url(url), content=content, headers=self._headers), self._timeout_s
            )
        except (httpx.TimeoutException, asyncio.TimeoutError):
            error = RuntimeError(TIMEOUT_ERROR.format(seconds=self._timeout_s, url=shown))
        except _HTTPX_ERRORS as exc:   # connection refused, DNS, protocol errors, an invalid URL, ...
            error = RuntimeError(UNREACHABLE_ERROR.format(url=shown, error=self._excerpt(f"{type(exc).__name__}: {exc}")))
        # Raised outside the except block, so the original (whose text may hold a key) is not chained.
        if error is not None:
            raise error
        if not 200 <= response.status_code < 300:
            raise RuntimeError(
                NON_2XX_ERROR.format(status=response.status_code, url=shown, body=self._body_excerpt(response))
            )
        return response

    # -- answers -----------------------------------------------------------------------------------

    def _to_reply(self, response: httpx.Response, url: str, latency_ms: float) -> AgentReply:
        try:
            data = json.loads(response.content) if response.content else None
        except ValueError:   # invalid JSON (UnicodeDecodeError included)
            data = None
        if not isinstance(data, Mapping):
            raise TypeError(NOT_OBJECT_ERROR.format(url=strip_url_credentials(url), body=self._body_excerpt(response)))

        text = data.get("text")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            raise TypeError(f"{SOURCE} returned a non-string 'text'")

        heard_text = data.get("heard_text")
        if heard_text is not None and not isinstance(heard_text, str):
            raise TypeError(f"{SOURCE} returned a non-string 'heard_text'")

        return AgentReply(
            text=text,
            audio_bytes=_decode_audio(data.get("audio")),
            latency_ms=latency_ms,
            heard_text=heard_text,
            tool_calls=to_tool_calls(data.get("tool_calls"), SOURCE),
        )

    # -- error texts -------------------------------------------------------------------------------

    def _redact(self, text: str) -> str:
        """The run's redactor, then the adapter's own credentials (bearer token, header values, URL
        credentials) and the standard key variables, with their encoded forms (``config.redact``)."""
        if self._run_redact is not None:
            text = self._run_redact(text)
        return redact(text, extra_values=self._secrets)

    def _excerpt(self, text: str) -> str:
        """Redacted FIRST, then on one line and cut to 200 characters (a key cut in half could no
        longer be redacted); ``(empty body)`` when nothing is left."""
        text = " ".join(self._redact(text).split())
        if not text:
            return EMPTY_BODY
        return text if len(text) <= ERROR_TEXT_CHARS else text[: ERROR_TEXT_CHARS - 3] + "..."

    def _body_excerpt(self, response: httpx.Response) -> str:
        try:
            text = response.text
        except Exception:  # noqa: BLE001 - an undecodable body only loses the excerpt
            text = ""
        return self._excerpt(text)


def _timeout_s(config: ProjectConfig, http: HTTPConfig) -> float:
    """``http.turn_timeout_s``, else ``voice_config.timeout_s`` (when voice_config exists), else 30 s."""
    if http.turn_timeout_s is not None:
        return float(http.turn_timeout_s)
    if config.voice_config is not None:
        return float(config.voice_config.timeout_s)
    return DEFAULT_TIMEOUT_S


def _known_secrets(http: HTTPConfig, token: Optional[str]) -> list[str]:
    """The credentials the adapter sends: the bearer token, every header value, and the credentials
    in the three URLs (``url_credentials``)."""
    values = [token] if token else []
    values += [secret_text(value) for value in http.headers.values() if isinstance(value, str) and value.strip()]
    for url in (http.url, http.start_url, http.close_url):
        values += url_credentials(url)
    return list(dict.fromkeys(values))


def _without_userinfo(url: str) -> str:
    """``url`` without ``user:password@`` (path, query and fragment kept); unchanged when it has none
    or cannot be parsed (httpx then reports the invalid URL)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    _userinfo, at, host = parts.netloc.rpartition("@")
    if not at:
        return url
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def _start_prompt(response: httpx.Response) -> Optional[str]:
    """The start answer's string ``system_prompt`` when the body is a JSON object, else None."""
    try:
        data = json.loads(response.content) if response.content else None
    except ValueError:
        return None
    prompt = data.get("system_prompt") if isinstance(data, Mapping) else None
    return prompt if isinstance(prompt, str) and prompt.strip() else None


def _decode_audio(value: Any) -> Optional[bytes]:
    """``audio``: base64 text of a WAV file in the standard or URL-safe alphabet, padding optional;
    None or "" is no audio. Anything else is the TypeError "HTTP agent returned 'audio' that is not base64"."""
    if value is None:
        return None
    error = TypeError(f"{SOURCE} returned 'audio' that is not base64")
    if not isinstance(value, str):
        raise error
    text = "".join(value.split()).replace("-", "+").replace("_", "/").rstrip("=")
    if not text:
        return None
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)
    except (binascii.Error, ValueError):
        pass
    raise error
