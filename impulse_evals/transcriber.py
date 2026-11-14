# SPDX-License-Identifier: AGPL-3.0-only
"""Transcript formatting and loading (SPEC.md sections 7, 12, and 15).

``format_transcript`` renders the text the judge reads between the ``<transcript>`` tags:

    [turn 1] USER: text
    [turn 2] AGENT TOOL CALL: name({"arg": "value"}) -> success=True error=message result={"id": 1}
    [turn 2] AGENT: text

``N`` is the 1-based position in ``CallTranscript.turns``, counting both speakers. Tool call lines
come one per call, in call order, immediately before the line of the turn that made them. The
``result=`` part has sensitive key values hidden (``redact_tool_result``) and is cut to
``RESULT_MAX_CHARS`` characters.

``json_sorted(value)`` is the JSON text used for tool call arguments (here and in the HTML report).
It never raises: arguments whose keys cannot be sorted or encoded (``{1: "a", "b": 2}``, tuple
keys) fall back to the same JSON layout with every key turned into text and sorted as text.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from impulse_evals.models import CallTranscript, ToolCall

_SPEAKER_LABELS = {"user": "USER", "agent": "AGENT"}
CIRCULAR_REFERENCE = "[circular reference]"   # how a container that contains itself is written
_CIRCULAR = CIRCULAR_REFERENCE

# The ``result=`` part of a tool call line (FIXES-ROUND-2-SPEC.md A1.3).
RESULT_MAX_CHARS = 300
_TRUNCATED = "...(truncated)"
REDACTED = "[REDACTED]"
_FLAG_KEYS = frozenset({"success", "ok"})
SENSITIVE_KEY_TOKENS = frozenset(
    {
        "password", "passwd", "passcode", "pwd", "pin", "otp", "cvv", "cvc", "secret", "token", "apikey",
        "authorization", "cookie", "mpin", "tpin",
    }
)
SENSITIVE_KEY_PAIRS = frozenset(
    {("api", "key"), ("access", "key"), ("private", "key"), ("card", "number"), ("card", "no")}
)
# One camelCase word: an all-capitals run of two or more letters with a lowercase plural ``s`` (``OTPs``,
# ``PINs``), an all-capitals run not followed by a lowercase letter (``XML`` in ``XMLHttp``), or an
# optional capital followed by lowercase letters and digits.
_CAMEL_WORD = re.compile(r"[A-Z]{2,}s(?![a-z])|[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+|[0-9]+")


def _key_text(key: Any) -> str:
    """A dict key as JSON writes it (``1`` -> ``"1"``, ``True`` -> ``"true"``, ``None`` ->
    ``"null"``, ``1.5`` -> ``"1.5"``); any other key type becomes ``str(key)``."""
    if isinstance(key, str):
        return key
    if key is True:
        return "true"
    if key is False:
        return "false"
    if key is None:
        return "null"
    if isinstance(key, (int, float)):
        return json.dumps(key)
    return str(key)


def _encode_text_keys(value: Any, path: frozenset[int]) -> str:
    """``json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)`` layout, with dict
    keys turned into text first and sorted as text (equal texts keep their order); a container
    that contains itself is written as ``"[circular reference]"``."""
    if isinstance(value, (dict, list, tuple)):
        if id(value) in path:
            return json.dumps(_CIRCULAR)
        inner = path | {id(value)}
        if isinstance(value, dict):
            items = sorted(((_key_text(key), item) for key, item in value.items()), key=lambda pair: pair[0])
            body = ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: {_encode_text_keys(item, inner)}" for key, item in items
            )
            return "{" + body + "}"
        return "[" + ", ".join(_encode_text_keys(item, inner) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, default=str)


def json_sorted(value: Any) -> str:
    """``json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)``, which never raises.

    When that call fails (keys of mixed types cannot be sorted, a key type JSON cannot encode, a
    circular reference), the same layout is built with every key as text, sorted as text; if that
    fails too, the result is a JSON string naming the error. String-keyed data is unchanged.
    """
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - formatting must never break the judge prompt or the report
        try:
            return _encode_text_keys(value, frozenset())
        except Exception as exc:  # noqa: BLE001 - e.g. RecursionError for extremely deep nesting
            return json.dumps(f"[could not encode as JSON: {type(exc).__name__}]")


def _key_tokens(key: Any) -> list[str]:
    """A key name as lowercase word tokens: split on every non-alphanumeric character and on camelCase
    boundaries (``apiKey`` -> ``api``, ``key``; ``X-Auth-Token`` -> ``x``, ``auth``, ``token``)."""
    tokens: list[str] = []
    for part in re.split(r"[\W_]+", _key_text(key)):
        if part:
            tokens.extend(word.lower() for word in (_CAMEL_WORD.findall(part) or [part]))
    return tokens


def _singular(token: str) -> str:
    """A token with a plural ``s`` removed (``tokens`` -> ``token``, ``pins`` -> ``pin``)."""
    return token[:-1] if len(token) > 1 and token.endswith("s") else token


def _is_sensitive_token(token: str) -> bool:
    return token in SENSITIVE_KEY_TOKENS or _singular(token) in SENSITIVE_KEY_TOKENS


def _is_sensitive_key(key: Any) -> bool:
    """True when the key's tokens contain a sensitive token (or its plural) or a sensitive consecutive
    token pair (its second token also in the plural: ``api_keys``, ``card_numbers``)."""
    tokens = _key_tokens(key)
    if any(_is_sensitive_token(token) for token in tokens):
        return True
    return any(
        (first, second) in SENSITIVE_KEY_PAIRS or (first, _singular(second)) in SENSITIVE_KEY_PAIRS
        for first, second in itertools.pairwise(tokens)
    )


def _redact_keys(value: Any, path: frozenset[int]) -> Any:
    if isinstance(value, (dict, list, tuple)):
        if id(value) in path:   # a container that contains itself: never walked twice
            return _CIRCULAR
        inner = path | {id(value)}
        if isinstance(value, dict):
            return {
                key: _redact_sensitive(item) if _is_sensitive_key(key) else _redact_keys(item, inner)
                for key, item in value.items()
            }
        items = [_redact_keys(item, inner) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _redact_sensitive(value: Any) -> Any:
    """The value of a sensitive key: ``true`` / ``false`` / ``null`` shown as is (they cannot carry a
    secret, and outcomes such as ``otp_verified`` matter to the judge; re-check N-L3), and so is an
    empty string, list, dict or tuple (no secret either, and ``"[REDACTED]"`` in its place would make
    UNIV-025 count a missing argument as present; fixes round 4 review); every other value
    ``"[REDACTED]"``."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (str, list, dict, tuple)) and not value:
        return value
    return REDACTED


def redact_tool_result(value: Any) -> Any:
    """A copy of a tool result with the value of every sensitive key replaced by ``"[REDACTED]"``, at any
    depth (dicts inside dicts and lists), whatever the value's type (FIXES-ROUND-2-SPEC.md A1.3), except
    ``true``, ``false`` and ``null``, which are shown as is (re-check N-L3), and empty values (``""``,
    ``[]``, ``{}``), which stay empty (fixes round 4 review).

    A key is sensitive when its word tokens (``_key_tokens``) contain one of ``SENSITIVE_KEY_TOKENS``
    or its plural, or one of the consecutive pairs ``SENSITIVE_KEY_PAIRS``: ``otp``, ``OTPs``,
    ``user_pin``, ``mpin``, ``accessToken``, ``card_number``, ``card_no`` and ``pin_code`` are;
    ``pincode``, ``spins`` and ``shipping`` (one token each) are not (A1.3, A1/A7-rev R1-L3).
    ``value`` is never mutated."""
    return _redact_keys(value, frozenset())


def _is_flag_only(result: dict) -> bool:
    """True for a non-empty result whose every key is ``success`` or ``ok`` with a boolean value: it
    adds nothing to ``success=``."""
    return bool(result) and all(
        isinstance(key, str) and key.strip().lower() in _FLAG_KEYS and isinstance(item, bool)
        for key, item in result.items()
    )


def tool_result_text(result: Any) -> str:
    """The ``result=`` text: ``json_sorted`` of the redacted result (``redact_tool_result``), cut to
    ``RESULT_MAX_CHARS`` characters plus ``...(truncated)`` when longer."""
    return _cut_shown(json_sorted(redact_tool_result(result)))


def _cut_shown(text: str) -> str:
    return text[:RESULT_MAX_CHARS] + _TRUNCATED if len(text) > RESULT_MAX_CHARS else text


def tool_error_text(error: Optional[str]) -> str:
    """The shown text of a tool call's ``error`` (AGENT TOOL CALL line and the report's trace table;
    FIXES-ROUND-2-SPEC.md A1/A7-rev R1-M2): an error whose text parses as a JSON object is shown as
    ``json_sorted(redact_tool_result(object))``; any shown error is cut to ``RESULT_MAX_CHARS``
    characters plus ``...(truncated)``; ``""`` for None or an empty string. The stored error is
    never changed."""
    if not isinstance(error, str) or not error:
        return ""
    text = error
    try:
        parsed = json.loads(error)
    except (ValueError, RecursionError):
        parsed = None
    if isinstance(parsed, dict):
        text = json_sorted(redact_tool_result(parsed))
    return _cut_shown(text)


def format_tool_call(turn_number: int, call: ToolCall) -> str:
    """One ``[turn N] AGENT TOOL CALL: ...`` line in the exact SPEC.md section 12 format.

    `` error={text}`` (``tool_error_text``) follows when the error is a non-empty string, then
    `` result={json}`` when the call has a result that is not flag-only (``_is_flag_only``);
    arguments are shown as stored (the runner stores the value of every sensitive key as
    ``"[REDACTED]"``; fixes round 4, L15) and are never cut."""
    arguments = json_sorted(call.arguments)
    line = f"[turn {turn_number}] AGENT TOOL CALL: {call.name}({arguments}) -> success={call.success}"
    if call.error:   # "when error is set": an empty string is not set (no dangling " error=")
        line += f" error={tool_error_text(call.error)}"
    result = call.result
    if result is not None and not (isinstance(result, dict) and _is_flag_only(result)):
        line += f" result={tool_result_text(result)}"
    return line


def format_transcript(t: CallTranscript) -> str:
    """The transcript as ``[turn N] USER: ...`` / ``[turn N] AGENT: ...`` lines joined with newlines.

    Turn text is rendered verbatim. A turn with non-empty ``tool_calls`` is preceded by its
    ``AGENT TOOL CALL`` lines.
    """
    lines: list[str] = []
    for number, turn in enumerate(t.turns, start=1):
        for call in turn.tool_calls:
            lines.append(format_tool_call(number, call))
        lines.append(f"[turn {number}] {_SPEAKER_LABELS[turn.speaker]}: {turn.text}")
    return "\n".join(lines)


def load_transcript(path: str | os.PathLike[str]) -> CallTranscript:
    """Read a ``CallTranscript`` JSON file.

    When the top-level object has a ``transcript`` key (a stored ``ScenarioResult``, or a
    calibration item), that value is used.
    """
    data = json.loads(Path(path).read_bytes().decode("utf-8-sig"))
    if isinstance(data, dict) and "transcript" in data:
        data = data["transcript"]
    return CallTranscript.model_validate(data)
