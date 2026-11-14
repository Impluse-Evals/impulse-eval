# SPDX-License-Identifier: AGPL-3.0-only
"""One-line texts for LLM provider errors (FIXES-ROUND-2-SPEC.md A7, DEFECT-6).

``provider_error_text(exc)`` turns an exception raised by LiteLLM (its class is defined in the
module ``litellm`` or ``litellm.*``) into one line, ``"{type}{details}: {message}{retry}"``:

- ``type``: the class name (``RateLimitError``);
- ``details``: `` (HTTP 429, provider gemini, model m)``, each part only when known (an int
  ``status_code``, a non-empty ``llm_provider`` / ``model``); no parentheses when none is;
- ``message``: ``str(exc)`` (passed through the caller's ``redact_text`` first) without its leading
  ``litellm.<Name>: `` prefixes, a leading ``<ClassName>: `` (the exception's own class name) and a
  leading ``<word>Exception - ``; when the rest holds a JSON object with ``error.message`` (or a
  top-level ``message``) string, that string; then its first line, whitespace collapsed, at most
  ``MESSAGE_MAX_CHARS`` characters (197 + "...");
- ``retry``: ``; retry after {n} s`` (n rounded up) when ``retry_after_seconds(exc)`` knows a delay.

Any other exception gives None (the caller keeps its own text). LiteLLM itself is never imported.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Optional

__all__ = ["MESSAGE_MAX_CHARS", "is_litellm_error", "one_line_error", "provider_error_text", "retry_after_seconds"]

MESSAGE_MAX_CHARS = 200
_LITELLM_PREFIX_RE = re.compile(r"^\s*litellm\.\w+:\s*")
_PROVIDER_PREFIX_RE = re.compile(r"^\s*\w*Exception - ")
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
_NUMBER = r"(\d+(?:\.\d+)?)"
# ``retry in 10h20m14.51s`` (Gemini) and ``try again in 6m30s`` / ``try again in 20.5 s`` (OpenAI).
_RETRY_IN_RE = re.compile(
    rf"(?:retry|try again) in\s+(?=\d)(?:{_NUMBER}h(?![a-z]))?(?:{_NUMBER}m(?![a-z]))?(?:{_NUMBER}\s?s(?![a-z]))?",
    re.IGNORECASE,
)


def is_litellm_error(exc: BaseException) -> bool:
    """True when the exception's class is defined in the module ``litellm`` or a ``litellm.*`` module."""
    module = getattr(type(exc), "__module__", "") or ""
    return module == "litellm" or module.startswith("litellm.")


def _header_seconds(exc: BaseException) -> Optional[float]:
    """The ``retry-after`` header of ``exc.response`` in seconds (a number), else None."""
    try:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        value = headers.get("retry-after") if headers is not None else None
        seconds = float(str(value).strip()) if value is not None else None
    except Exception:  # noqa: BLE001 - an HTTP-date value or an unusual response object: unknown
        return None
    return seconds if seconds is not None and math.isfinite(seconds) and seconds >= 0 else None


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """The retry delay the error names, in seconds: the ``retry-after`` header of ``exc.response``,
    else a ``"retryDelay": "<n>s"`` in its text, else ``retry in [Nh][Nm][N.Ns]`` or ``try again in
    [Nh][Nm][N.Ns]`` / ``try again in N.N s`` in its text; None when none is found."""
    seconds = _header_seconds(exc)
    if seconds is not None:
        return seconds
    text = _exception_str(exc)
    match = _RETRY_DELAY_RE.search(text)
    if match is not None:
        return float(match.group(1))
    match = _RETRY_IN_RE.search(text)
    if match is not None and any(match.groups()):
        hours, minutes, secs = (float(group) if group else 0.0 for group in match.groups())
        return hours * 3600 + minutes * 60 + secs
    return None


def _exception_str(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ must not hide the error
        return ""


def _json_message(text: str) -> Optional[str]:
    """``error.message`` (else a top-level ``message``) of the first JSON object in ``text``, if a string."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    message = data.get("message")
    return message if isinstance(message, str) else None


def _one_line(text: str) -> str:
    """The first non-blank line, whitespace collapsed, cut to ``MESSAGE_MAX_CHARS`` characters."""
    lines = text.strip().splitlines()
    line = " ".join(lines[0].split()) if lines else ""
    if len(line) > MESSAGE_MAX_CHARS:
        line = line[: MESSAGE_MAX_CHARS - 3] + "..."
    return line


def _redacted_str(exc: BaseException, redact_text: Optional[Callable[[str], str]]) -> str:
    """``str(exc)`` passed through ``redact_text`` (when given) BEFORE any parsing or cutting, so a key
    that would straddle the cut leaves no prefix behind."""
    text = _exception_str(exc)
    return redact_text(text) if redact_text is not None else text


def _message(exc: BaseException, redact_text: Optional[Callable[[str], str]] = None) -> str:
    rest = _redacted_str(exc, redact_text)
    while True:
        stripped = _LITELLM_PREFIX_RE.sub("", rest, count=1)
        if stripped == rest:
            break
        rest = stripped
    class_prefix = f"{type(exc).__name__}:"   # LiteLLM's OpenAI-style mapping repeats the class name
    if rest.lstrip().startswith(class_prefix):
        rest = rest.lstrip()[len(class_prefix):].lstrip()
    rest = _PROVIDER_PREFIX_RE.sub("", rest, count=1)
    message = _json_message(rest)
    return _one_line(message if message is not None else rest)


def _details(exc: BaseException) -> str:
    parts: list[str] = []
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        parts.append(f"HTTP {status}")
    for label, attribute in (("provider", "llm_provider"), ("model", "model")):
        value = getattr(exc, attribute, None)
        if isinstance(value, str) and value.strip():
            parts.append(f"{label} {value.strip()}")
    return f" ({', '.join(parts)})" if parts else ""


def provider_error_text(
    exc: BaseException, redact_text: Optional[Callable[[str], str]] = None
) -> Optional[str]:
    """The one-line text of a LiteLLM error (see the module doc), or None for any other exception.

    ``redact_text`` (the run's key redactor) is applied to ``str(exc)`` before it is parsed, put on
    one line and cut (FIXES-ROUND-2-SPEC.md A1/A7-rev R1-M1)."""
    if not is_litellm_error(exc):
        return None
    seconds = retry_after_seconds(exc)
    retry = f"; retry after {math.ceil(seconds)} s" if seconds is not None else ""
    message = _message(exc, redact_text)
    head = f"{type(exc).__name__}{_details(exc)}"
    return f"{head}: {message}{retry}" if message else f"{head}{retry}"


def one_line_error(exc: BaseException, redact_text: Optional[Callable[[str], str]] = None) -> str:
    """``provider_error_text(exc, redact_text)``, else the first line of ``str(exc)`` (redacted first,
    whitespace collapsed, cut), else the class name: a one-line text for any exception."""
    text = provider_error_text(exc, redact_text)
    if text is None:
        text = _one_line(_redacted_str(exc, redact_text))
    return text or type(exc).__name__
