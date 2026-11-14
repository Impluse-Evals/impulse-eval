# SPDX-License-Identifier: AGPL-3.0-only
"""LiteLLM judge (SPEC.md section 12).

Behaviour:
- temperature 0; JSON mode requested where LiteLLM reports that the provider supports it (asked with
  the role's ``custom_llm_provider``, ``config.role_custom_llm_provider``; the answer is kept per
  model and provider);
- routing by tier: ``"primary"`` uses ``primary_model``, ``"fast"`` uses ``fast_model``;
- empty content (None, blank, or a tool-call-only answer) is a no-verdict skip, without a retry;
- the answer is parsed tolerantly but fails closed: the verdict must be a top-level JSON object.
  Its top-level JSON objects are read in order (fenced or not, prose around them ignored); exactly
  one of them may hold the key ``result`` and, when that ``result`` is usable, it is the verdict
  (objects without ``result`` are passed over; two or more objects with ``result`` are invalid
  JSON, even when they agree); objects nested inside another object or inside a well-formed JSON
  array are never accepted, and a top-level value of any other JSON type (array, string, number,
  ``true``/``false``/``null``) is not a verdict; a ``[`` with no JSON value after it (e.g.
  ``[turn 2]`` prose) is prose, while the first other ``{`` or ``[`` that does not decode
  (malformed outer JSON, e.g. ``[{...},]``, or a ``[`` followed by ``,`` such as ``[,{...}]``)
  ends the scan, so no verdict inside it or after it is accepted;
  invalid JSON (no verdict found) is retried once with "Return JSON only", then becomes a
  no-verdict skip;
- a ``"skip"`` returned by the judge is a valid verdict (``no_verdict=False``).

LLM errors raised by LiteLLM (authentication, connection, ...) are not caught here; they are not a
"no verdict" (SPEC.md section 13) and the caller decides how to report them. One exception to "not
caught": a quota error (HTTP 429 or ``RateLimitError``) whose retry delay is at least
``QUOTA_STOP_SECONDS`` stops its model for this judge (the run) and is re-raised; every later check
routed to that model raises ``JudgeStopped`` without a provider call, and ``get_warnings()`` holds
one ``JUDGE_STOPPED_WARNING`` per stopped model (FIXES-ROUND-2-SPEC.md A7). Calls to the other model
continue. The stop is keyed by (model, custom_llm_provider) like the JSON-mode cache, is checked again
before every provider call (the "Return JSON only" retry too), and its text is the one-line error
redacted with the run's ``redact`` callable before it is cut (A1/A7-rev R1-M1, R1-L2, R1-N).

The module also holds the run-level warning helpers of SPEC.md section 12 (judge model equal to
``agent.model``, small judge, ``num_ctx``). The ``num_ctx`` warning is raised by the judge itself and
returned by ``get_warnings()``; the other two are computed by the runner, which knows the agent
model and whether the run includes a critical assertion.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Literal, Optional

from impulse_evals.config import role_custom_llm_provider
from impulse_evals.judge.base import Judge, JudgeVerdict
from impulse_evals.judge.prompts import RETRY_JSON_ONLY, build_messages
from impulse_evals.models import CallTranscript
from impulse_evals.provider_errors import one_line_error, retry_after_seconds

EMPTY_REASON = "Judge returned empty (safety filter or rate limit)"
INVALID_JSON_REASON = "Judge returned invalid JSON"
NO_REASON = "Judge gave no reason."

VERDICT_RESULTS = ("pass", "fail", "skip")

# Options keys the judge handles itself; every other key is passed to litellm.acompletion as is.
_OPTION_KEYS = ("api_base", "api_key", "api_key_env", "extra_params")
# Call arguments the judge always sets; options cannot override them.
_FIXED_KWARGS = ("model", "messages", "temperature", "stream")

# Small-judge warning (SPEC.md section 12).
SMALL_JUDGE_MAX_BILLIONS = 14.0
_PARAMETER_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)b\b", re.IGNORECASE)

# Context-length warning (SPEC.md section 12).
CHARS_PER_TOKEN = 4
CONTEXT_WARN_FRACTION = 0.8

# Judge stop after a quota error with a long retry delay (FIXES-ROUND-2-SPEC.md A7).
QUOTA_STOP_SECONDS = 60
JUDGE_STOPPED_MESSAGE = "judge calls to {model} were stopped for this run after a quota error: {error}"
JUDGE_STOPPED_WARNING = (
    "Judge calls to {model} were stopped after a provider quota error; the checks that use it were not "
    "evaluated after that: {error}"
)


class JudgeStopped(Exception):
    """Raised by ``LiteLLMJudge.evaluate`` without a provider call when the check's model was stopped
    for the run after a quota error (``JUDGE_STOPPED_MESSAGE``)."""


def is_quota_stop(exc: BaseException) -> bool:
    """True for a quota error (HTTP 429 or a ``RateLimitError``) whose retry delay is at least
    ``QUOTA_STOP_SECONDS``: waiting for it would hold the run for a minute or more per check."""
    status = getattr(exc, "status_code", None)
    if status != 429 and type(exc).__name__ != "RateLimitError":
        return False
    seconds = retry_after_seconds(exc)
    return seconds is not None and seconds >= QUOTA_STOP_SECONDS


# ------------------------------------------------------------------------------------------------
# Run-level warnings (SPEC.md section 12)
# ------------------------------------------------------------------------------------------------


def agent_model_warning(
    agent_model: Optional[str], primary_model: Optional[str], fast_model: Optional[str]
) -> Optional[str]:
    """The warning text when ``agent.model`` equals the primary or fast judge model, else None."""
    if agent_model is None or not agent_model.strip():
        return None
    model = agent_model.strip()
    judge_models = {(m or "").strip() for m in (primary_model, fast_model)}
    if model not in judge_models:
        return None
    return (
        f"agent.model is the same as the judge model ({model}); "
        "the judge may be biased toward the agent's own outputs."
    )


def judge_parameter_counts(model: str) -> list[float]:
    """Parameter counts (in billions) found in a model string, e.g. ``[7.0]`` for ``...:7b``."""
    return [float(match) for match in _PARAMETER_COUNT_RE.findall(model or "")]


def is_small_judge(model: str) -> bool:
    """True when the model string contains a parameter count of at most 14B."""
    return any(count <= SMALL_JUDGE_MAX_BILLIONS for count in judge_parameter_counts(model))


def small_judge_warning(model: str, has_critical_assertion: bool) -> Optional[str]:
    """The small-judge warning text, or None.

    Call it with the primary judge model (critical assertions route to it) and whether the run
    includes any ``critical`` assertion.
    """
    if not has_critical_assertion or not is_small_judge(model):
        return None
    return (
        f"Judge model {model} looks small (<= 14B parameters). Critical verdicts from small judges "
        "are unreliable; run impulse calibrate before trusting them."
    )


def num_ctx_from_options(options: Optional[Mapping[str, Any]]) -> Optional[int]:
    """``num_ctx`` from a role's options (top level, else ``extra_params``); None when not configured."""
    options = options or {}
    value = options.get("num_ctx")
    if value is None:
        extra = options.get("extra_params")
        if isinstance(extra, Mapping):
            value = extra.get("num_ctx")
    if value is None or isinstance(value, bool):
        return None
    try:
        num_ctx = int(value)
    except (TypeError, ValueError):
        return None
    return num_ctx if num_ctx > 0 else None


def estimate_tokens(messages: list[dict]) -> float:
    """Estimated prompt length in tokens: characters / 4."""
    chars = sum(len(str(message.get("content") or "")) for message in messages)
    return chars / CHARS_PER_TOKEN


def context_length_warning(
    messages: list[dict], model: str, options: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """The ``num_ctx`` warning when the estimated prompt length exceeds 80% of ``num_ctx``, else None."""
    num_ctx = num_ctx_from_options(options)
    if num_ctx is None:
        return None
    estimate = estimate_tokens(messages)
    if estimate <= CONTEXT_WARN_FRACTION * num_ctx:
        return None
    return (
        f"Judge prompt is about {math.ceil(estimate)} tokens, over 80% of num_ctx={num_ctx} for {model}; "
        "the judge may truncate the transcript."
    )


# ------------------------------------------------------------------------------------------------
# Call arguments
# ------------------------------------------------------------------------------------------------


def completion_kwargs(
    options: Optional[Mapping[str, Any]], *, fixed: Iterable[str] = _FIXED_KWARGS
) -> dict[str, Any]:
    """Extra ``litellm.acompletion`` arguments for a role's options.

    ``extra_params`` entries are passed as arguments, then the other top-level keys (they win);
    ``api_base`` is passed when non-empty; ``api_key`` (a resolved key) is passed as is, otherwise
    the variable named by ``api_key_env`` is read from the environment. The ``fixed`` arguments
    (default: ``model``, ``messages``, ``temperature``, and ``stream``, which the judge always sets)
    are dropped from the options; the simulator passes its own list so a configured
    ``temperature`` reaches its calls.
    """
    options = dict(options or {})
    kwargs: dict[str, Any] = {}
    extra = options.get("extra_params")
    if isinstance(extra, Mapping):
        kwargs.update(extra)
    kwargs.update({key: value for key, value in options.items() if key not in _OPTION_KEYS})
    api_base = options.get("api_base")
    if api_base:
        kwargs["api_base"] = api_base
    api_key = options.get("api_key")
    api_key_env = options.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env) or None
    if api_key:
        kwargs["api_key"] = api_key
    for key in fixed:
        kwargs.pop(key, None)
    return kwargs


def supports_json_mode(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """True when LiteLLM lists ``response_format`` among the provider's supported parameters for
    ``model`` with the role's ``custom_llm_provider`` (None: LiteLLM places the model string itself)."""
    import litellm  # lazy: heavy import

    litellm.suppress_debug_info = True   # no "Give Feedback / Get Help" banner (AUDIT.md finding 43)
    try:
        # LiteLLM prints a provider-list banner for unknown model strings; this is only a probe.
        with contextlib.redirect_stdout(io.StringIO()):
            params = litellm.get_supported_openai_params(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001 - unknown providers raise several types
        return False
    return "response_format" in (params or [])


# ------------------------------------------------------------------------------------------------
# Tolerant verdict parsing
# ------------------------------------------------------------------------------------------------


_JSON_OPENER_RE = re.compile(r"[{\[]")
# Whitespace the JSON decoder skips after a "[" (narrower than str.isspace()).
_JSON_WHITESPACE_RE = re.compile(r"[ \t\n\r]*")


def _is_prose_bracket(text: str, index: int, error: ValueError) -> bool:
    """True when the ``[`` at ``index``, which did not decode, starts no JSON value (``[turn 2]``).

    That is the case exactly when the decoder failed with "Expecting value" at the first
    non-whitespace position after the ``[`` (JSON whitespace; or at the end of the text) and the
    ``[`` is not followed by ``,``; every other failure, including a plain ``ValueError`` such as an
    over-long integer, means a malformed array. A ``[`` whose first character after any whitespace
    (``str.isspace``: JSON's four and also form feed, vertical tab, NO-BREAK SPACE, EM SPACE, ...) is
    ``,`` (``[,{...}]``, ``[ ,]``, ``[\\f,]``) starts a malformed JSON array, never prose (the
    maintainer's judge rule); ``[`` + such whitespace + a word (``[ turn 2]``) stays prose.
    """
    if not isinstance(error, json.JSONDecodeError) or error.msg != "Expecting value":
        return False
    after = index + 1
    while after < len(text) and text[after].isspace():
        after += 1
    if text.startswith(",", after):
        return False
    whitespace = _JSON_WHITESPACE_RE.match(text, index + 1)
    first = whitespace.end() if whitespace is not None else index + 1
    return error.pos == first


def _json_objects(text: str) -> Iterator[dict]:
    """The top-level JSON objects of ``text``, in order, up to the first malformed ``{`` or array.

    The text is scanned from the start and each ``{`` or ``[`` is decoded with ``raw_decode``; the
    scan resumes after the decoded value, so nothing nested inside it is ever yielded. A decoded
    object is yielded; a decoded array is skipped whole, so an object inside a well-formed array
    (wherever the array sits in the text) is never yielded. Fence markers and prose are neither
    braces nor brackets, so a value in a fenced block or between sentences is found the same way,
    and a JSON string, number, ``true``, ``false`` or ``null`` holds no top-level object. A ``[``
    that does not decode is prose only when no JSON value starts after it (``[turn 2]``,
    ``[ slot ]``; see ``_is_prose_bracket``): the scan goes on from the next character. Any other
    ``[`` that does not decode (a malformed array such as ``[{...},]``, ``[{...}`` or
    ``[{...} {...}]``, and a ``[`` followed, after optional whitespace, by ``,`` such as
    ``[,{...}]``) and the first ``{`` that does not decode (malformed outer JSON, or a brace in
    prose) end the scan: nothing inside it or after it is yielded, so the reply counts as invalid
    JSON unless a verdict came before it. Nesting too deep to decode (``RecursionError``) ends the
    scan for ``[`` as for ``{``.
    """
    decoder = json.JSONDecoder()
    match = _JSON_OPENER_RE.search(text)
    while match is not None:
        index = match.start()
        try:
            value, end = decoder.raw_decode(text, index)
        except RecursionError:
            return
        except ValueError as error:
            if text[index] == "{" or not _is_prose_bracket(text, index, error):
                return  # malformed outer JSON
            match = _JSON_OPENER_RE.search(text, index + 1)  # a prose "[" such as "[turn 2]"
            continue
        if text[index] == "{":
            yield value  # a value that starts with "{" is always an object
        match = _JSON_OPENER_RE.search(text, end)


def _as_turn_index(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _as_confidence(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _as_reason(value: Any) -> str:
    if value is None:
        return NO_REASON
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return value if value.strip() else NO_REASON


def _verdict_from_object(data: dict) -> Optional[JudgeVerdict]:
    result = data.get("result")
    if not isinstance(result, str) or result.strip().lower() not in VERDICT_RESULTS:
        return None
    return JudgeVerdict(
        result=result.strip().lower(),
        reason=_as_reason(data.get("reason")),
        turn_index=_as_turn_index(data.get("turn_index")),
        confidence=_as_confidence(data.get("confidence")),
    )


def parse_verdict(text: Optional[str]) -> Optional[JudgeVerdict]:
    """The one top-level JSON object of a judge answer that holds the key ``result``, when its
    ``result`` is pass/fail/skip; otherwise None (invalid JSON).

    Objects without a ``result`` key (e.g. ``{"notes": ...}``) are passed over. More than one
    top-level object with a ``result`` key makes the answer invalid, even when the verdicts agree
    (the maintainer's judge rule). Objects inside a well-formed JSON array and top-level values of
    any other JSON type are never verdicts, and only objects before the first ``{`` or array ``[``
    that does not decode are considered (see ``_json_objects``).
    """
    if not text:
        return None
    candidates = [data for data in _json_objects(text) if "result" in data]
    if len(candidates) != 1:
        return None
    return _verdict_from_object(candidates[0])


def _content_text(response: Any) -> str:
    """``response.choices[0].message.content`` as stripped text ("" when missing or None)."""
    try:
        raw = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""
    if raw is None:
        return ""
    if isinstance(raw, list):  # content parts
        raw = "".join(str(part.get("text") or "") for part in raw if isinstance(part, Mapping))
    return str(raw).strip()


def _no_verdict(reason: str) -> JudgeVerdict:
    return JudgeVerdict(result="skip", reason=reason, no_verdict=True)


# ------------------------------------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------------------------------------


class LiteLLMJudge(Judge):
    """Judge backed by any LiteLLM model string (SPEC.md section 12)."""

    def __init__(
        self,
        primary_model: str,
        fast_model: str,
        primary_options: Optional[Mapping[str, Any]] = None,
        fast_options: Optional[Mapping[str, Any]] = None,
        redact: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.primary_model = primary_model
        self.fast_model = fast_model
        self.primary_options: dict[str, Any] = dict(primary_options or {})
        self.fast_options: dict[str, Any] = dict(fast_options or {})
        self._warnings: list[str] = []
        self._context_warned: set[str] = set()
        # JSON-mode support per (model, custom_llm_provider): two roles with the same model string but
        # different providers get their own answers.
        self._json_mode: dict[tuple[str, Optional[str]], bool] = {}
        # Models stopped for the run after a quota error, keyed like ``_json_mode``:
        # (model, custom_llm_provider) -> the error's one-line text (redacted with ``redact``).
        self._stopped: dict[tuple[str, Optional[str]], str] = {}
        # The run's key redactor, applied to the provider's error text before it is cut to one line.
        self._redact = redact

    def get_warnings(self) -> list[str]:
        return list(self._warnings)

    def _route(self, tier: str) -> tuple[str, dict[str, Any]]:
        if tier == "primary":
            return self.primary_model, self.primary_options
        if tier == "fast":
            return self.fast_model, self.fast_options
        raise ValueError(f"unknown judge tier {tier!r}; expected 'primary' or 'fast'")

    def _call_kwargs(self, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        kwargs = completion_kwargs(options)
        key = (model, role_custom_llm_provider(options))
        if key not in self._json_mode:
            self._json_mode[key] = supports_json_mode(*key)
        if self._json_mode[key] and "response_format" not in kwargs:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _check_context(self, messages: list[dict], model: str, options: Mapping[str, Any]) -> None:
        if model in self._context_warned:
            return
        warning = context_length_warning(messages, model, options)
        if warning is not None:
            self._context_warned.add(model)
            self._warnings.append(warning)

    def _raise_if_stopped(self, model: str, stop_key: tuple[str, Optional[str]]) -> None:
        error = self._stopped.get(stop_key)
        if error is not None:
            raise JudgeStopped(JUDGE_STOPPED_MESSAGE.format(model=model, error=error))

    async def _complete(
        self, model: str, stop_key: tuple[str, Optional[str]], messages: list[dict], kwargs: dict[str, Any]
    ) -> str:
        """One provider call, unless ``stop_key`` was stopped (checked before EVERY call, so a check that
        began before the stop never reaches the provider after it). A quota error with a long retry
        delay (``is_quota_stop``) stops ``stop_key`` for this judge (one warning per key) and is
        re-raised."""
        import litellm  # lazy; called through the module so tests can patch litellm.acompletion

        self._raise_if_stopped(model, stop_key)
        litellm.suppress_debug_info = True   # no "Give Feedback / Get Help" banner (AUDIT.md finding 43)
        try:
            response = await litellm.acompletion(model=model, messages=messages, temperature=0, **kwargs)
        except Exception as exc:
            if is_quota_stop(exc) and stop_key not in self._stopped:
                error = one_line_error(exc, redact_text=self._redact)
                self._stopped[stop_key] = error
                self._warnings.append(JUDGE_STOPPED_WARNING.format(model=model, error=error))
            raise
        return _content_text(response)

    async def evaluate(
        self,
        transcript: CallTranscript,
        criterion: str,
        tier: Literal["primary", "fast"],
    ) -> JudgeVerdict:
        model, options = self._route(tier)
        stop_key = (model, role_custom_llm_provider(options))
        self._raise_if_stopped(model, stop_key)
        messages = build_messages(transcript, criterion)
        self._check_context(messages, model, options)
        kwargs = self._call_kwargs(model, options)

        content = await self._complete(model, stop_key, messages, kwargs)
        if not content:
            return _no_verdict(EMPTY_REASON)
        verdict = parse_verdict(content)
        if verdict is not None:
            return verdict

        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": RETRY_JSON_ONLY},
        ]
        content = await self._complete(model, stop_key, retry_messages, kwargs)
        if not content:
            return _no_verdict(EMPTY_REASON)
        verdict = parse_verdict(content)
        if verdict is not None:
            return verdict
        return _no_verdict(INVALID_JSON_REASON)
