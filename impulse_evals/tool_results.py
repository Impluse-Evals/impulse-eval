# SPDX-License-Identifier: AGPL-3.0-only
"""Tool call outcome detection from common result shapes (FIXES-ROUND-2-SPEC.md A1.1, DEFECT-1).

``infer_success(call) -> ToolCall`` returns a copy of ``call`` (never a mutation) with ``success``
inferred when the adapter did not capture it. The runner applies it to every tool call of every
``AgentReply`` before storing the agent turn (every adapter, custom ones included). Rules, first
match wins:

1. ``success`` is not None: unchanged.
2. ``error`` is a non-empty string: ``success=False``.
3. ``result`` is a dict (top level only; key names compared case-insensitively after ``strip()``):
   a. a ``success`` or ``ok`` key with a boolean value: that value (the first such key);
   b. otherwise an ``error`` or ``errors`` key with a truthy error value (a non-blank string, a
      non-empty dict or list, or ``True``): ``success=False``;
   c. otherwise a ``status`` or ``state`` key with a failure word (``failed``, ...): False, or a
      success word (``completed``, ...): True.
4. Otherwise ``success=None`` (not captured).

When the inferred ``success`` is False and ``error`` is empty, ``error`` is filled from the result
when it holds an error text (``_result_error``), at most ``ERROR_MAX_CHARS`` characters.
"""

from __future__ import annotations

from typing import Any, Optional

from impulse_evals.models import ToolCall

__all__ = ["ERROR_MAX_CHARS", "infer_success"]

ERROR_MAX_CHARS = 200
_FLAG_KEYS = frozenset({"success", "ok"})
_ERROR_KEYS = frozenset({"error", "errors"})
_STATUS_KEYS = frozenset({"status", "state"})
_FAILURE_WORDS = frozenset({"error", "failed", "failure", "fail"})
_SUCCESS_WORDS = frozenset({"ok", "success", "succeeded", "successful", "completed", "complete", "done"})


def _key_name(key: Any) -> Optional[str]:
    return key.strip().lower() if isinstance(key, str) else None


def _is_error_value(value: Any) -> bool:
    """A "truthy error value": a non-blank string, a non-empty dict or list, or ``True``."""
    if value is True:
        return True
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list)):
        return bool(value)
    return False


def _result_success(result: dict) -> Optional[bool]:
    """Rule 3 on a dict result: True, False, or None when no rule matches."""
    for key, value in result.items():
        if _key_name(key) in _FLAG_KEYS and isinstance(value, bool):
            return value
    for key, value in result.items():
        if _key_name(key) in _ERROR_KEYS and _is_error_value(value):
            return False
    for key, value in result.items():
        if _key_name(key) in _STATUS_KEYS and isinstance(value, str):
            word = value.strip().lower()
            if word in _FAILURE_WORDS:
                return False
            if word in _SUCCESS_WORDS:
                return True
    return None


def _text(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _values(result: dict, name: str) -> list[Any]:
    """The values of every key named ``name`` (case-insensitive, stripped), in dict order."""
    return [value for key, value in result.items() if _key_name(key) == name]


def _result_error(result: dict) -> Optional[str]:
    """The error text a dict result holds, in the A1.1 order (A1/A7-rev R1-L5), whatever the dict
    order: an ``error`` string, else an ``errors`` string, else the first string item of an ``errors``
    list, else a ``message`` string inside an ``error`` dict (strings non-blank, stripped); None when
    there is none."""
    candidates: list[Any] = [*_values(result, "error"), *_values(result, "errors")]
    for value in _values(result, "errors"):
        if isinstance(value, list):
            candidates.extend(value)
    for value in _values(result, "error"):
        if isinstance(value, dict):
            candidates.append(value.get("message"))
    for candidate in candidates:
        text = _text(candidate)
        if text is not None:
            return text
    return None


def _cut(text: str) -> str:
    return text if len(text) <= ERROR_MAX_CHARS else text[: ERROR_MAX_CHARS - 3] + "..."


def infer_success(call: ToolCall) -> ToolCall:
    """A copy of ``call`` with ``success`` (and, for a failure, ``error``) inferred; see the module doc."""
    inferred = call.model_copy(deep=True)
    if call.success is not None:
        return inferred
    if isinstance(call.error, str) and call.error:
        inferred.success = False
        return inferred
    result = call.result
    if not isinstance(result, dict):
        return inferred
    success = _result_success(result)
    inferred.success = success
    if success is False and not call.error:
        error = _result_error(result)
        if error is not None:
            inferred.error = _cut(error)
    return inferred
