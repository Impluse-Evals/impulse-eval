# SPDX-License-Identifier: AGPL-3.0-only
"""Runner: orchestrates the scenarios of one run (SPEC.md sections 7, 9, 10, 12-15, 17).

``await run(config, *, dry_run=False, concurrency=1, adapter_factory=None) -> RunResult``

Per scenario: a new adapter (``adapter_factory(config)``, default: the adapter named by
``agent.adapter``), ``start(scenario)``, the conversation (SPEC.md section 10), ``close()``, then the
adapter's metadata and warnings are collected, then evaluation: the metrics from ``select_metrics``
and the scenario's assertions, one after the other, with the shared judge.

- Conversation: inbound calls start with the simulated user (``opening_user_message`` verbatim when
  set); outbound calls start with ``send(None, None)`` (the agent's opener, not counted in
  ``max_turns`` and not in the latency profile). The call ends after ``max_turns`` USER turns, when
  the simulator answers ``<END_CALL>``, or when the adapter raises ``AgentClosed`` from ``send()``
  (the agent hung up; the turns so far are evaluated). An ``AgentClosed`` before the agent's first
  turn (no agent turn yet and no text, tool call or audio in the reply it carries) is the scenario
  error "the agent closed the conversation before its first turn: {message}" instead: an agent that
  crashed or hung up on join is never evaluated. ``AgentReply.heard_text`` goes on the preceding user
  turn, ``latency_ms`` / ``interrupted`` / ``tool_calls`` on the agent turn. Every tool call is stored
  with its ``result`` and ``error`` redacted (the run's key redactor) and its ``success`` inferred
  (``tool_results.infer_success``). An ``AgentTurnError`` from ``send()`` whose ``reply`` holds text,
  a tool call, or audio has that reply recorded as an agent turn first (dropped if recording it
  fails); the error is then the scenario error (FIXES-ROUND-2-SPEC.md A1, A6).
- System prompt: the ``config.yaml`` file wins; otherwise ``adapter.get_system_prompt()`` after
  ``start()``. The simulator's summary of it is made once per distinct prompt.
- Errors: a ``ConfigError`` (unresolved model, missing key, entrypoint import failure) propagates out
  of ``run``, its message redacted by the run's full redactor (same field). Any other exception before evaluation (adapter, agent, simulator, Impulse's TTS)
  becomes the scenario ``error`` with empty result lists, except a ``VoiceError`` from Impulse's own
  STT (the round trip or the agent audio): the turn is kept and the failure becomes the run warning
  "{scenario_name}: Impulse STT failed on turn {turn}: {error}" (SPEC.md section 14). A critical
  assertion the judge could not evaluate (``ScoreResult.no_verdict``) sets the scenario ``error``
  and keeps every other result (SPEC.md section 13). A check whose evaluation raises (e.g. a LiteLLM error from the judge) is
  recorded as skipped with "evaluation failed: {message}" and the remaining checks still run; the
  first such failure is the scenario ``error`` unless a critical no-verdict error is set, and every
  failure not used as the error becomes a run warning, except a ``JudgeStopped`` one (the judge's own
  stop warning covers those). A LiteLLM error's text is its one-line ``provider_error_text``.
- Every error and warning text the runner produces, and every string of the adapter's
  ``get_metadata()`` in its JSON form (keys and values, recursively), has the known key values redacted (also their
  URL-encoded, base64 and provider-masked forms, ``config.redact``): config.toml ``[keys]``, the
  standard key variables, every ``*_API_KEY`` environment variable, every variable named by the
  config (role options' ``api_key_env``, Pipecat / LiveKit / HTTP / voice key variables), the keys
  resolved for the roles, and the literal credentials the config holds (string option values,
  ``pipecat.headers`` values except ``Origin``, credential-like ``http.headers`` values, and the credentials in the
  Pipecat / LiveKit / HTTP URLs and the voice ``base_url`` values). ``RunResult.agent_entrypoint``
  holds the Pipecat / LiveKit / HTTP URL without its credentials (the entrypoint for generic and
  custom).
- ``inject_events`` are never executed in v1: they are copied into ``ScenarioResult.inject_events``
  and the v2 notice is printed to the console once per affected scenario.
- Dry run (SPEC.md section 17): no LLM call, no model or key resolution; the scripted simulator;
  judge metrics and assertions skipped with "dry run"; deterministic metrics computed; text mode;
  scenarios run only for the generic adapter (a custom adapter's class is imported and checked, never
  created); the run warning ``DRY_RUN_WARNING`` on every dry run.

Voice mode (``voice_config`` present, never in a dry run; SPEC.md section 14): the TTS and STT
providers are built once per run (with ``./.impulse/config.toml`` for their keys) and every scenario
language is checked against them before anything runs. Per user turn: simulator text -> TTS ->
``send(text, audio)``; then Impulse's STT of its own TTS audio (the VOICE-011 round trip; its
confidence goes on the user turn); then, when the agent returned audio, Impulse's STT of that audio
(confidence on the agent turn). The agent turn's text is the adapter's text when non-empty, else the
STT text, and the simulator sees the same text (a blank one is sent to it as "(no speech)", SPEC.md
section 10). ``latency_ms`` is the adapter's measurement only:
Impulse's own TTS / STT time never enters it. Voice-mode turn times follow the call as heard: a user
turn starts where the previous turn ended and lasts as long as its TTS audio; an agent turn starts
``latency_ms`` after the user's audio ended and lasts as long as its audio; Impulse's processing time
between turns is not on this timeline. The STT word timings and round-trip transcriptions are kept in
``CallTranscript.metadata["impulse_voice"]`` for the voice metrics.

Noise simulation (EXPERIMENTAL, SPEC.md section 14.4; ``config.effective_noise``): after TTS the
caller's audio is mixed with background noise (``voice.noise.mix_noise``, seeded with the scenario's
``seed`` or its name, and the turn number). The NOISY audio is sent to the adapter, saved as the user
turn's ``audio_path`` and used in the recording; the CLEAN audio goes to the VOICE-011 round trip and
is saved as ``<stem>_t{N}_user_clean.wav``. ``metadata["impulse_noise"]`` holds the settings and each
user turn's record the measured SNR; nothing of this exists for a scenario without noise. A dry run
ignores noise with ``NOISE_DRY_RUN_WARNING``.

``run`` writes no run files (SPEC.md section 15): in voice mode the per-turn WAV files
(``save_audio``) and the stereo recording (``save_recording``; left user, right agent, placed by
``timestamp_ms``) are the only exception, written to ``.impulse/runs/<run_id>/audio/`` as turns
happen (``.impulse/`` with its ``.gitignore`` is created first). ``run.json``, the scenario files, and
the report are written by the CLI and ``impulse_check``. It prints only the inject_events notice and
the Pipecat concurrency message; every other run warning is returned in ``RunResult.warnings``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import math
import secrets
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from pydantic_core import to_jsonable_python

from impulse_evals.adapters.base import (
    AgentClosed,
    AgentInterface,
    AgentReply,
    AgentTurnError,
)
from impulse_evals.assertions.base import UnknownAssertionError, get_assertion
from impulse_evals.cohort import build_cohorts
from impulse_evals.config import (
    ROLES,
    ConfigError,
    ProjectConfig,
    TomlConfig,
    api_key_for_role,
    effective_noise,
    load_project_toml,
    option_secrets,
    read_system_prompt,
    redact,
    resolve_key,
    resolve_model,
    resolve_model_options,
    secret_text,
    static_key_values,
    strip_url_credentials,
    url_credentials,
)
from impulse_evals.judge.base import Judge, JudgeVerdict
from impulse_evals.judge.litellm_judge import (
    JudgeStopped,
    LiteLLMJudge,
    agent_model_warning,
    small_judge_warning,
)
from impulse_evals.metrics.base import MetricContext, get_metric, select_metrics
from impulse_evals.models import (
    CallTranscript,
    LatencyProfile,
    MetricResult,
    NoiseSettings,
    RunResult,
    Scenario,
    ScenarioResult,
    ScoreResult,
    STTResult,
    ToolCall,
    Turn,
    compute_passed,
)
from impulse_evals.provider_errors import provider_error_text
from impulse_evals.reporter import (
    IMPULSE_DIR,
    RUNS_DIR,
    ensure_impulse_dir,
    scenario_file_stem,
)
from impulse_evals.simulator import (
    UNKNOWN_PURPOSE,
    PromptSummarizer,
    ScriptedSimulator,
    Simulator,
)
from impulse_evals.tool_results import infer_success
from impulse_evals.transcriber import CIRCULAR_REFERENCE, redact_tool_result

if TYPE_CHECKING:  # annotation only; the voice modules are imported when voice mode runs
    from impulse_evals.voice.stt import STTProvider
    from impulse_evals.voice.tts import TTSProvider

__all__ = [
    "INJECT_EVENTS_NOTE",
    "PIPECAT_CONCURRENCY_MESSAGE",
    "ROUND_TRIP_WARNING",
    "build_latency_profile",
    "build_metric_summary",
    "config_redactor",
    "default_adapter_factory",
    "make_run_id",
    "resolved_role_keys",
    "run",
]

AdapterFactory = Callable[..., AgentInterface]

# ------------------------------------------------------------------------------------------------
# Exact texts (SPEC.md sections 7, 9, 13, 17)
# ------------------------------------------------------------------------------------------------

INJECT_EVENTS_NOTE = "inject_events defined but execution reserved for v2; this scenario ran without injection."
PIPECAT_CONCURRENCY_MESSAGE = "Pipecat bots serve one client at a time; running scenarios one by one."
NO_VERDICT_ERROR = "critical check {rule_id} could not be evaluated: judge returned no verdict"
NO_VERDICT_WARNING_PREFIX = "Critical checks not evaluated (judge returned no verdict): "
EVALUATION_FAILED_ERROR = "evaluation of {check_id} failed: {message}"
EVALUATION_FAILED_REASON = "evaluation failed: {message}"
EVALUATION_FAILED_WARNING = "{scenario_name}: " + EVALUATION_FAILED_ERROR
CLOSED_BEFORE_FIRST_TURN_ERROR = "the agent closed the conversation before its first turn: {message}"
DRY_RUN_REASON = "dry run"
# Added to every dry run (any adapter) so run.json, the report header, and `impulse report` show
# that its results are not verdicts (AUDIT.md finding 9).
DRY_RUN_WARNING = "Dry run: judge metrics and assertions were skipped; results are not verdicts."
DRY_RUN_REMOTE_WARNING = (
    "Dry run: scenarios run only with the generic adapter; the {adapter} config was validated without connecting."
)
METRIC_SUMMARY_REASON = "mean over {n} scenarios ({m} skipped)"
ROUND_TRIP_WARNING = "Harness round-trip WER is high; audio metrics in this run are unreliable."
ROUND_TRIP_METRIC = "VOICE-011"
# Impulse's own STT failing (round trip or agent audio) keeps the turn (SPEC.md section 14; AUDIT.md
# finding 17): one run warning per failed turn; ``turn`` is the 1-based transcript position.
IMPULSE_STT_FAILED_WARNING = "{scenario_name}: Impulse STT failed on turn {turn}: {error}"
# Fixes round 4, M2: a failure of close() / get_metadata() / get_warnings() AFTER the conversation ended
# (for example the HTTP adapter's close_url answering 5xx) is a run warning, not a scenario error, so a
# complete transcript is still evaluated.
CLOSE_FAILED_WARNING = "{scenario_name}: the adapter's close failed after the conversation ended; the results are kept: {error}"
AUDIO_FOLDER = "audio"
# Noise simulation (EXPERIMENTAL; SPEC.md section 14.4).
NOISE_METADATA_KEY = "impulse_noise"
NOISE_EXPERIMENTAL_WARNING = (
    "Noise simulation is experimental: listen to the noisy caller audio in the report before trusting the results."
)
NOISE_DRY_RUN_WARNING = "noise simulation is ignored in --dry-run."
NOISE_GENERIC_NO_AUDIO_WARNING = (
    "the generic agent function has no user_audio parameter, so it never hears the noise; "
    "noise simulation has no effect on it."
)
# metrics.voice values stored in metadata["impulse_voice"]["thresholds"] (FIXES-ROUND-2-SPEC.md B4).
VOICE_THRESHOLD_KEYS = ("dead_air_ms", "ttfa_warn_ms", "ttfa_fail_ms")
# Sample rate of the stereo recording when no clip has a higher one (clips are resampled to it).
RECORDING_MIN_SAMPLE_RATE = 16000

# RunResult.metric_summary labels (SPEC.md section 7): the MetricResult threshold defaults.
SUMMARY_THRESHOLD_WARN = 0.6
SUMMARY_THRESHOLD_FAIL = 0.4

_ADAPTER_CLASSES = {
    "pipecat": ("impulse_evals.adapters.pipecat_adapter", "PipecatAdapter"),
    "livekit": ("impulse_evals.adapters.livekit_adapter", "LiveKitAdapter"),
}


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------


class _Warnings:
    """Run warnings, deduplicated, in first-seen order."""

    def __init__(self, initial: Iterable[str] = ()) -> None:
        self._items: list[str] = []
        self.extend(initial)

    def add(self, warning: Optional[str]) -> None:
        if warning and warning not in self._items:
            self._items.append(warning)

    def extend(self, warnings: Iterable[Optional[str]]) -> None:
        for warning in warnings:
            self.add(warning)

    def as_list(self, redact_text: Optional[Callable[[str], str]] = None) -> list[str]:
        """The warnings, each passed through ``redact_text`` first (deduplicated again afterwards)."""
        if redact_text is None:
            return list(self._items)
        return list(dict.fromkeys(redact_text(warning) for warning in self._items))


@dataclass(frozen=True)
class _Redactor:
    """Replaces every known key value in a text with ``[REDACTED]`` (SPEC.md section 8.1).

    Known values: config.toml ``[keys]`` and the variables named there, the standard key variables,
    the variables in ``names`` (named by the config), and the literal ``values`` the runner holds.
    """

    toml: Optional[TomlConfig] = None
    names: tuple[str, ...] = ()
    values: tuple[str, ...] = ()

    def __call__(self, text: str) -> str:
        return redact(text, self.toml, extra_names=self.names, extra_values=self.values)

    def with_values(self, values: Iterable[Any]) -> _Redactor:
        extra = tuple(value for value in values if isinstance(value, str) and value)
        return _Redactor(self.toml, self.names, self.values + extra)


def _config_redactor(config: ProjectConfig, toml: Optional[TomlConfig]) -> _Redactor:
    """A redactor for every key the config names or holds; no key is resolved here (dry runs too).

    Literal values: the string leaves of the config.yaml role options (``option_secrets``; those of
    config.toml ``[model_options.<role>]`` come with ``toml`` through ``config.secret_values``), the
    ``pipecat.headers`` values except ``Origin`` (a public origin), the HTTP bearer token (stripped) and
    the values of credential-like ``http.headers`` (``is_sensitive_header``), and
    the credentials in ``pipecat.url``, ``livekit.url``, the three HTTP URLs and the voice ``base_url``
    values (``url_credentials``)."""
    names: list[Any] = []
    values: list[Any] = []
    yaml_options = {
        "primary": config.judge.primary_options,
        "fast": config.judge.fast_options,
        "simulator": config.simulator.options,
    }
    for role in ROLES:
        options = yaml_options[role]
        if isinstance(options, Mapping):
            names.append(options.get("api_key_env"))
        values.extend(option_secrets(options))
        if toml is not None:
            names.append(getattr(toml.model_options, role).api_key_env)
    agent = config.agent
    if agent.pipecat is not None:
        names.append(agent.pipecat.api_key_env)
        values.extend(url_credentials(agent.pipecat.url))
        values.extend(
            secret_text(value)
            for header, value in (agent.pipecat.headers or {}).items()
            if isinstance(value, str) and str(header).strip().lower() != "origin"
        )
    if agent.livekit is not None:
        names.extend([agent.livekit.api_key_env, agent.livekit.api_secret_env])
        values.extend(url_credentials(agent.livekit.url))
    if agent.http is not None:
        key_env = (agent.http.api_key_env or "").strip()
        if key_env:
            names.append(key_env)
            token = resolve_key(key_env, toml)
            if token:
                values.append(token.strip())   # the adapter sends the stripped value
        for url in (agent.http.url, agent.http.start_url, agent.http.close_url):
            values.extend(url_credentials(url))
        # Only credential-like headers (B3/B4-rev R7): this redactor also runs on stored tool results,
        # which the judge reads; the adapter's own error texts redact every header value it holds.
        values.extend(
            secret_text(value)
            for header, value in (agent.http.headers or {}).items()
            if isinstance(value, str) and is_sensitive_header(str(header))
        )
    if config.voice_config is not None:
        voice = config.voice_config
        names.extend([voice.tts.api_key_env, voice.stt.api_key_env])
        values.extend(url_credentials(voice.tts.base_url) + url_credentials(voice.stt.base_url))
    return _Redactor(
        toml=toml,
        names=tuple(dict.fromkeys(n.strip() for n in names if isinstance(n, str) and n.strip())),
    ).with_values(values)


SENSITIVE_HEADER_NAMES = ("authorization", "proxy-authorization", "cookie")
SENSITIVE_HEADER_PARTS = ("key", "token", "secret", "password")


def is_sensitive_header(name: str) -> bool:
    """A header whose value is a credential: ``Authorization``, ``Proxy-Authorization``, ``Cookie``, or a
    name containing ``key``, ``token``, ``secret`` or ``password`` (case-insensitive)."""
    lowered = name.strip().lower()
    return lowered in SENSITIVE_HEADER_NAMES or any(part in lowered for part in SENSITIVE_HEADER_PARTS)


def config_redactor(config: ProjectConfig, toml: Optional[TomlConfig]) -> Callable[[str], str]:
    """``text -> text`` with every key value the config and config.toml name or hold redacted (the
    CLI uses it for messages it prints itself); no key is resolved."""
    return _config_redactor(config, toml)


class _CallClock:
    """Milliseconds since the call started (``time.perf_counter``)."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def restart(self) -> None:
        self._start = time.perf_counter()

    def now_ms(self) -> int:
        return round((time.perf_counter() - self._start) * 1000.0)


class _DryRunJudge(Judge):
    """Judge used in a dry run: judge metrics and assertions are skipped before any judge call."""

    async def evaluate(self, transcript: CallTranscript, criterion: str, tier: str) -> JudgeVerdict:
        raise RuntimeError("the judge is never called in a dry run")


def make_run_id(when: Optional[datetime] = None) -> str:
    """UTC compact timestamp plus 6 random hex digits, e.g. ``20261004T120000Z-a3f9c2`` (SPEC.md 15)."""
    when = when or datetime.now(UTC)
    return when.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def _iso_timestamp(when: datetime) -> str:
    """ISO 8601 UTC, e.g. ``2026-10-04T12:00:00Z`` (SPEC.md section 7)."""
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _error_text(exc: BaseException, redact_text: Callable[[str], str]) -> str:
    """The exception message (its type name when empty), with every known key value redacted. A
    LiteLLM error is its one-line ``provider_error_text`` (FIXES-ROUND-2-SPEC.md A7), redacted BEFORE
    it is parsed and cut, so a key straddling the cut leaves no prefix behind (A1/A7-rev R1-M1)."""
    provider_text = provider_error_text(exc, redact_text=redact_text)
    if provider_text is not None:
        return redact_text(provider_text)   # again over the whole line: the type and details parts too
    text = str(exc).strip()
    if not text:
        text = type(exc).__name__
    elif isinstance(exc, KeyError):   # str(KeyError) is only the quoted key
        text = f"KeyError: {text}"
    return redact_text(text)


def _redact_tree(value: Any, redact_text: Callable[[str], str]) -> Any:
    """``value`` with every string in it passed through ``redact_text``: mapping keys and values, and
    list and tuple items, recursively (a mapping becomes a dict, a tuple stays a tuple). Numbers,
    booleans, None and other values are returned unchanged."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {_redact_tree(key, redact_text): _redact_tree(item, redact_text) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_redact_tree(item, redact_text) for item in value)
    if isinstance(value, list):
        return [_redact_tree(item, redact_text) for item in value]
    return value


def _redacted_metadata(metadata: Any, redact_text: Callable[[str], str]) -> Any:
    """Adapter metadata as it will be written, redacted: first converted to its JSON form
    (``pydantic_core.to_jsonable_python(..., fallback=str)``: bytes become text, sets and tuples
    lists, pydantic models dicts, mapping keys strings, anything else its ``str()``; when some bytes are
    not UTF-8, every bytes value of it becomes URL-safe base64 text, as pydantic's base64 mode writes
    it, a form ``redact`` knows), then every string in it goes through ``redact_text`` (``_redact_tree``),
    so no value ``model_dump_json`` would write as text can carry a known key (AUDIT.md finding 42)."""
    try:
        jsonable = to_jsonable_python(metadata, fallback=str)
    except UnicodeDecodeError:   # bytes that are not UTF-8: written as base64 (redact knows that form)
        jsonable = to_jsonable_python(metadata, fallback=str, bytes_mode="base64")
    return _redact_tree(jsonable, redact_text)


def _redacted_config_error(exc: ConfigError, redact_text: Callable[[str], str]) -> ConfigError:
    """``exc`` with its message redacted by ``redact_text`` (same type and field), or ``exc`` itself
    when the message holds no known key value. A ``ConfigError`` raised while a scenario runs can
    carry text the run did not write (e.g. the generic adapter's "cannot import module ..." with the
    agent module's own error), and the CLI's redactor may not know every key the run resolved."""
    message = redact_text(exc.message)
    if message == exc.message:
        return exc
    try:
        redacted = type(exc)(exc.field, message)
    except Exception:  # noqa: BLE001 - a subclass with another constructor: the base class
        redacted = ConfigError(exc.field, message)
    return redacted


@contextlib.contextmanager
def _config_errors_redacted(redact_text: Callable[[str], str]) -> Iterator[None]:
    """Re-raise a ``ConfigError`` from the block with its message redacted (``_redacted_config_error``).
    A redacted copy keeps the original's traceback frames (down to, e.g., the generic adapter's
    ``start()``) but no link to the original, which still holds the key: no ``__cause__`` and no
    ``__context__``. Raised here, while ``__exit__`` handles the original, the copy would get it as
    ``__context__`` even with ``from None``, so that is cleared explicitly as the copy leaves."""
    try:
        yield
    except ConfigError as exc:
        redacted = _redacted_config_error(exc, redact_text)
        if redacted is exc:
            raise
        try:
            raise redacted.with_traceback(exc.__traceback__) from None
        finally:
            redacted.__context__ = None


def _print_console(message: str) -> None:
    """Print one line to stdout; never fails on encoding (SPEC.md section 17: a dry run exits 0).

    A console or redirected stream that cannot encode a character (e.g. a Devanagari scenario name
    on a Windows code page) gets that character as a backslash escape instead of an error.
    """
    stream = sys.stdout
    if stream is None:   # pythonw: no console at all
        return
    try:
        print(message, file=stream, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        try:
            safe = message.encode(encoding, "backslashreplace").decode(encoding, "replace")
        except LookupError:   # an unknown codec name on a wrapped stream
            safe = message.encode("ascii", "backslashreplace").decode("ascii")
        try:
            print(safe, file=stream, flush=True)
        except UnicodeEncodeError:
            print(message.encode("ascii", "backslashreplace").decode("ascii"), file=stream, flush=True)


def default_adapter_factory(
    config: ProjectConfig,
    *,
    run_id: Optional[str] = None,
    redact: Optional[Callable[[str], str]] = None,
) -> AgentInterface:
    """A new adapter for ``agent.adapter`` (imported lazily; the optional extras stay optional).

    ``run_id`` (the run in progress) is passed to the LiveKit adapter, which names each scenario's
    room ``{room_prefix}{run_id}-{scenario_name}`` (SPEC.md section 9). ``redact`` (the run's key
    redactor) is passed to the Pipecat and HTTP adapters, which redact the agent's error texts with
    it BEFORE they shorten them (SPEC.md section 9), and is used for a custom adapter's constructor
    error. ``custom``: ``Class(config)`` of the ``agent.entrypoint`` class (loading errors are
    ``ConfigError``s at ``agent.entrypoint``; FIXES-ROUND-3-SPEC.md B3); ``http``:
    ``HTTPAdapter(config, redact=redact)`` (B4).
    """
    adapter = config.agent.adapter
    if adapter == "generic":
        from impulse_evals.adapters.generic_adapter import GenericAdapter

        return GenericAdapter(config)
    if adapter == "custom":
        from impulse_evals.adapters.custom_adapter import create_adapter

        return create_adapter(config, redact_text=redact)
    if adapter == "http":
        from impulse_evals.adapters.http_adapter import HTTPAdapter

        return HTTPAdapter(config, redact=redact)
    module_name, class_name = _ADAPTER_CLASSES[adapter]
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    if adapter == "livekit" and run_id is not None:
        return adapter_class(config, run_id=run_id)
    if adapter == "pipecat" and redact is not None:
        return adapter_class(config, redact=redact)
    return adapter_class(config)


def agent_entrypoint(config: ProjectConfig) -> str:
    """``RunResult.agent_entrypoint``: the generic or custom entrypoint, or the Pipecat / LiveKit / HTTP
    turn URL without its credentials (userinfo and credential query parameters, ``strip_url_credentials``)."""
    agent = config.agent
    if agent.adapter == "pipecat" and agent.pipecat is not None:
        return strip_url_credentials(agent.pipecat.url or "")
    if agent.adapter == "livekit" and agent.livekit is not None:
        return strip_url_credentials(agent.livekit.url or "")
    if agent.adapter == "http" and agent.http is not None:
        return strip_url_credentials(agent.http.url or "")
    return agent.entrypoint or ""


def agent_domain(scenarios: Sequence[Scenario]) -> str:
    """The single distinct scenario domain, otherwise ``"mixed"`` (SPEC.md section 7)."""
    domains = {scenario.domain for scenario in scenarios}
    return domains.pop() if len(domains) == 1 else "mixed"


def effective_concurrency(config: ProjectConfig, concurrency: int) -> tuple[int, Optional[str]]:
    """``(concurrency to use, message)``: Pipecat bots serve one client at a time (SPEC.md section 9)."""
    try:
        value = max(1, int(concurrency or 1))
    except (TypeError, ValueError):
        value = 1
    if config.agent.adapter == "pipecat" and value > 1:
        return 1, PIPECAT_CONCURRENCY_MESSAGE
    return value, None


def _run_files(config: ProjectConfig, run_id: str) -> _RunFiles:
    """The run folder ``./.impulse/runs/<run_id>/`` (absolute, fixed when the run starts) and one file
    stem per scenario: sanitized as in SPEC.md section 15 and unique in the folder, assigned in config
    order exactly as the CLI names ``scenarios/<stem>.json``. Nothing is created here."""
    impulse_dir = Path.cwd() / IMPULSE_DIR
    used: set[str] = set()
    stems = {scenario.name: scenario_file_stem(scenario.name, used) for scenario in config.scenarios}
    return _RunFiles(impulse_dir=impulse_dir, run_dir=Path.cwd() / RUNS_DIR / run_id, stems=stems)


# ------------------------------------------------------------------------------------------------
# Run setup
# ------------------------------------------------------------------------------------------------


@dataclass
class _VoiceIO:
    """The run's voice providers and file settings (``voice_config``)."""

    tts: TTSProvider
    stt: STTProvider
    save_audio: bool
    save_recording: bool


@dataclass
class _RunFiles:
    """Where voice mode writes its audio: ``<impulse_dir>/runs/<run_id>/audio/``, one file stem per
    scenario (sanitized and unique, as the CLI's ``scenarios/<stem>.json``)."""

    impulse_dir: Path
    run_dir: Path
    stems: dict[str, str]


@dataclass
class _Setup:
    judge: Judge
    simulator: Union[Simulator, ScriptedSimulator]
    summarizer: Optional[PromptSummarizer]
    judge_model: str
    ctx: MetricContext
    config_prompt: Optional[str]
    dry_run: bool
    voice_mode: bool
    redact: _Redactor
    voice: Optional[_VoiceIO] = None
    files: Optional[_RunFiles] = None
    # Noise files loaded once per run (absolute path -> (mono samples, rate)), and their copies
    # resampled to a TTS rate ((path, rate) -> samples), shared by every scenario.
    noise_sources: dict[str, Any] = field(default_factory=dict)
    noise_resampled: dict[tuple[str, int], Any] = field(default_factory=dict)

    async def summary_for(self, system_prompt: Optional[str]) -> str:
        if self.summarizer is None:   # dry run: the scripted simulator ignores the summary
            return UNKNOWN_PURPOSE
        return await self.summarizer.summarize(system_prompt)


def _role_options(config: ProjectConfig, role: str, toml: Optional[TomlConfig]) -> dict[str, Any]:
    """Resolved options for ``role``, with the resolved key (env, then config.toml) as ``api_key``."""
    options = resolve_model_options(config, role, toml)
    key = api_key_for_role(config, role, toml)
    if key:
        options["api_key"] = key
    return options


def resolved_role_keys(config: ProjectConfig, toml: Optional[TomlConfig]) -> list[str]:
    """The key values ``run`` resolves for the roles (``api_key_for_role``: environment, then
    config.toml, whatever the variable's name), for a caller's own redactor: the CLI registers them
    before a real run (AUDIT.md finding 31). A role whose model or key cannot be resolved is left out
    (``run`` raises that ``ConfigError`` itself). Offline: no LLM call, no login; never raises. It
    imports LiteLLM (the key-name lookup), so a dry run never calls it: a dry run's redactors use
    ``config.static_key_values()`` instead."""
    keys: list[str] = []
    for role in ROLES:
        # A role that cannot be resolved is reported by run itself: nothing to learn from it here.
        with contextlib.suppress(Exception):
            key = api_key_for_role(config, role, toml)
            if isinstance(key, str) and key and key not in keys:
                keys.append(key)
    return keys


def _has_critical_assertion(config: ProjectConfig) -> bool:
    for scenario in config.scenarios:
        for rule_id in scenario.assertions:
            try:
                if get_assertion(rule_id).severity == "critical":
                    return True
            except UnknownAssertionError:   # a pack not built yet: reported when the scenario runs
                continue
    return False


def _dry_run_judge_model(config: ProjectConfig, toml: Optional[TomlConfig]) -> str:
    try:
        return resolve_model(config, "primary", toml)
    except ConfigError:
        return (config.judge.primary_model or "").strip() or "YOUR_PRIMARY_MODEL"


def _voice_thresholds(config: ProjectConfig) -> dict[str, Any]:
    """``metadata["impulse_voice"]["thresholds"]``: the ``metrics.voice`` values the report's timeline
    uses (FIXES-ROUND-2-SPEC.md B4)."""
    voice = config.metrics.voice
    return {key: getattr(voice, key) for key in VOICE_THRESHOLD_KEYS}


def _metric_context(config: ProjectConfig, *, dry_run: bool, voice_mode: bool) -> MetricContext:
    return MetricContext(
        agent_tools=config.agent.tools,
        agent_type=config.agent.type,
        voice_mode=voice_mode,
        metrics_enabled=config.metrics.enabled,
        voice_thresholds=config.metrics.voice.model_dump(),
        dry_run=dry_run,
    )


def _prepare(
    config: ProjectConfig,
    toml: Optional[TomlConfig],
    *,
    dry_run: bool,
    voice_mode: bool,
    warnings: _Warnings,
    redactor: _Redactor,
) -> _Setup:
    config_prompt = read_system_prompt(config) if config.agent.system_prompt is not None else None
    ctx = _metric_context(config, dry_run=dry_run, voice_mode=voice_mode)
    if dry_run:
        return _Setup(
            judge=_DryRunJudge(),
            simulator=ScriptedSimulator(),
            summarizer=None,
            judge_model=_dry_run_judge_model(config, toml),
            ctx=ctx,
            config_prompt=config_prompt,
            dry_run=True,
            voice_mode=False,
            redact=redactor,
        )

    # Every role is resolved before the first LLM call, so a missing key never follows a paid call.
    models: dict[str, str] = {}
    options: dict[str, dict[str, Any]] = {}
    for role in ("primary", "fast", "simulator"):
        models[role] = resolve_model(config, role, toml)
        options[role] = _role_options(config, role, toml)

    warnings.add(agent_model_warning(config.agent.model, models["primary"], models["fast"]))
    warnings.add(small_judge_warning(models["primary"], _has_critical_assertion(config)))
    # Voice providers too are built (and every scenario language checked) before the first paid call.
    voice = _build_voice(config, toml) if voice_mode else None

    return _Setup(
        judge=LiteLLMJudge(models["primary"], models["fast"], options["primary"], options["fast"], redact=redactor),
        simulator=Simulator(
            models["simulator"], options["simulator"], config.simulator.persona_style, mode=config.simulator.mode
        ),
        summarizer=PromptSummarizer(models["fast"], options["fast"]),
        judge_model=models["primary"],
        ctx=ctx,
        config_prompt=config_prompt,
        dry_run=False,
        voice_mode=voice_mode,
        # The resolved keys may come from variables no config entry names (e.g. a provider's own
        # ``<PROVIDER>_API_KEY``), so their values are redacted as literals.
        redact=redactor.with_values(options[role].get("api_key") for role in ROLES),
        voice=voice,
    )


def _load_noise_files(config: ProjectConfig, setup: _Setup) -> None:
    """Load every noise file the run uses once (noise simulation). ``load_config`` checked them; a file
    removed or changed since is a ``ConfigError`` at the field that names it."""
    from impulse_evals.voice.noise import NoiseFileError, load_noise_file

    for index, scenario in enumerate(config.scenarios):
        noise = effective_noise(config, scenario)
        if noise is None or noise.profile != "file" or not noise.file or noise.file in setup.noise_sources:
            continue
        field_name = f"scenarios[{index}].noise.file" if scenario.noise is not None else "voice_config.noise.file"
        error: Optional[ConfigError] = None
        try:
            setup.noise_sources[noise.file] = load_noise_file(noise.file)
        except NoiseFileError as exc:
            error = ConfigError(field_name, str(exc))
        if error is not None:
            raise error


def _build_voice(config: ProjectConfig, toml: Optional[TomlConfig]) -> _VoiceIO:
    """The run's TTS and STT providers (keys: environment, then config.toml), after checking that both
    support every scenario language (SPEC.md section 14: "language check per provider").

    A language a provider does not support (e.g. ``impulse_hosted`` and ``ta-IN``) is a ``ConfigError``
    at ``scenarios[i].language``; a missing provider setting keeps its own ``ConfigError``.
    """
    from impulse_evals.voice import VoiceError
    from impulse_evals.voice.stt import get_stt_provider
    from impulse_evals.voice.tts import get_tts_provider

    voice_config = config.voice_config
    assert voice_config is not None   # voice mode means voice_config is set
    tts = get_tts_provider(voice_config, toml=toml)
    stt = get_stt_provider(voice_config, toml=toml)
    checked: set[str] = set()
    for index, scenario in enumerate(config.scenarios):
        if scenario.language in checked:
            continue
        checked.add(scenario.language)
        for provider in (tts, stt):
            try:
                provider.check_language(scenario.language)
            except VoiceError as exc:
                raise ConfigError(f"scenarios[{index}].language", str(exc)) from None
    return _VoiceIO(
        tts=tts, stt=stt, save_audio=voice_config.save_audio, save_recording=voice_config.save_recording
    )


# ------------------------------------------------------------------------------------------------
# One scenario
# ------------------------------------------------------------------------------------------------


@dataclass
class _Outcome:
    result: ScenarioResult
    adapter_warnings: list[str] = field(default_factory=list)
    no_verdict_checks: list[str] = field(default_factory=list)   # "{scenario_name}:{rule_id}"
    evaluation_warnings: list[str] = field(default_factory=list)  # evaluation failures not used as the error
    voice_warnings: list[str] = field(default_factory=list)   # Impulse's own STT failures (turns kept)


def _agent_start_ms(user_turn: Turn, reply: AgentReply, returned_ms: int) -> int:
    """When the agent turn started (text mode): the user turn's time plus the reply's latency, i.e.
    the moment the first agent output arrived; kept between the user turn and ``send()`` returning."""
    try:
        latency = float(reply.latency_ms)
    except (TypeError, ValueError):
        return returned_ms
    if not math.isfinite(latency):
        return returned_ms
    start = user_turn.timestamp_ms + max(0, round(latency))
    return max(user_turn.timestamp_ms, min(start, returned_ms))


async def _send(
    adapter: AgentInterface, text: Optional[str], audio: Optional[bytes] = None
) -> tuple[Optional[AgentReply], Optional[AgentClosed]]:
    """``(reply, closed)``: ``closed`` is the ``AgentClosed`` the adapter raised (the agent hung up),
    else None; ``reply`` is then the agent's last turn carried by the exception, if any."""
    try:
        return await adapter.send(text, audio), None
    except AgentClosed as closed:
        reply = closed.reply if isinstance(closed.reply, AgentReply) else None
        return reply, closed


def _has_content(reply: Optional[AgentReply]) -> bool:
    """True when ``reply`` holds agent output: text, a tool call, or audio."""
    return reply is not None and bool(reply.text.strip() or reply.tool_calls or reply.audio_bytes)


def _check_not_closed_before_first_turn(
    closed: Optional[AgentClosed], reply: Optional[AgentReply], turns: Sequence[Turn]
) -> None:
    """An agent that closes the conversation before any agent output (no agent turn so far, and no
    content in the reply the close carries) crashed or hung up on join: a scenario error, never a
    normal end that would be evaluated as a conversation (SPEC.md sections 7 and 10)."""
    if closed is None or _has_content(reply) or any(turn.speaker == "agent" for turn in turns):
        return
    message = str(closed).strip() or type(closed).__name__
    raise RuntimeError(CLOSED_BEFORE_FIRST_TURN_ERROR.format(message=message))


def _without_cycles(value: Any, path: frozenset[int] = frozenset()) -> Any:
    """``value`` with every container that contains itself (mapping, list or tuple on its own path)
    replaced by ``CIRCULAR_REFERENCE``, as ``json_sorted`` writes it; shared, non-circular parts are
    kept. Mappings become dicts; other values are returned unchanged."""
    if isinstance(value, (Mapping, list, tuple)):
        if id(value) in path:
            return CIRCULAR_REFERENCE
        inner = path | {id(value)}
        if isinstance(value, Mapping):
            return {key: _without_cycles(item, inner) for key, item in value.items()}
        items = [_without_cycles(item, inner) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _stored_tool_call(call: ToolCall, redact_text: Callable[[str], str]) -> ToolCall:
    """A tool call as the agent turn stores it (FIXES-ROUND-2-SPEC.md A1.1, A1.2): every string of its
    ``result`` (keys and values, recursively, in its JSON form as ``_redacted_metadata``) and its
    ``error`` redacted first, then ``success`` inferred from the redacted call (``infer_success``), so
    an error text filled from the result is already redacted. The value of every sensitive key in
    ``arguments`` (the tool-result key rule, ``transcriber.redact_tool_result``: ``otp``, ``password``,
    ``pin``, ``token``, ``card_number``, plurals, at any depth; ``true`` / ``false`` / ``null`` kept) is
    stored as ``"[REDACTED]"`` too (fixes round 4, L15), so run.json, the report and the judge transcript
    never carry it; other argument values are unchanged and arguments are never cut.

    The redacted result is stored whatever its type (a custom adapter can assign a string or a list
    after construction; A1/A7-rev R1-L1). A self-referencing result never ends the scenario: its
    circular part becomes ``CIRCULAR_REFERENCE`` (R1-L6)."""
    stored = call.model_copy(deep=True)
    if isinstance(stored.arguments, dict):
        stored.arguments = redact_tool_result(stored.arguments)   # breaks cycles itself
    if stored.result is not None:
        try:
            stored.result = _redacted_metadata(stored.result, redact_text)
        except ValueError:   # "Circular reference detected": break the cycle, then redact
            stored.result = _redacted_metadata(_without_cycles(stored.result), redact_text)
    if isinstance(stored.error, str):
        stored.error = redact_text(stored.error)
    return infer_success(stored)


def _agent_turn(reply: AgentReply, timestamp_ms: int, *, opener: bool, redact_text: Callable[[str], str]) -> Turn:
    return Turn(
        speaker="agent",
        text=reply.text,
        timestamp_ms=timestamp_ms,
        # The outbound opener has no user input to measure from (SPEC.md section 14).
        latency_ms=None if opener else _finite_latency_ms(reply.latency_ms),
        interrupted=reply.interrupted,
        tool_calls=[_stored_tool_call(call, redact_text) for call in reply.tool_calls],
    )


def _finite_latency_ms(value: Any) -> Optional[float]:
    """The adapter's latency, or None when it is not a finite number (NaN or infinity from a custom
    adapter would break the latency profile and the run file, which stores them as null)."""
    try:
        latency = float(value)
    except (TypeError, ValueError):
        return None
    return latency if math.isfinite(latency) else None


def _latency_gap_ms(reply: AgentReply) -> int:
    """The reply's latency in whole milliseconds (0 when missing, negative, or not finite)."""
    try:
        latency = float(reply.latency_ms)
    except (TypeError, ValueError):
        return 0
    return max(0, round(latency)) if math.isfinite(latency) else 0


class _VoiceSession:
    """The voice pipeline of one scenario (SPEC.md section 14): TTS of the user's text, Impulse's STT
    of that audio (VOICE-011 round trip) and of the agent's audio, the turn timeline, the per-turn WAV
    files, the stereo recording, and the data the voice metrics read from the transcript metadata."""

    def __init__(
        self,
        io: _VoiceIO,
        scenario: Scenario,
        files: _RunFiles,
        *,
        redact: Callable[[str], str] = redact,
        thresholds: Optional[Mapping[str, Any]] = None,
        noise: Optional[NoiseSettings] = None,
        noise_sources: Optional[dict[str, Any]] = None,
        noise_resampled: Optional[dict[tuple[str, int], Any]] = None,
    ) -> None:
        from impulse_evals.metrics.voice import VOICE_METADATA_KEY
        from impulse_evals.voice import audio_utils

        self.noise = noise
        self._noise_sources = noise_sources if noise_sources is not None else {}
        self._noise_resampled = noise_resampled if noise_resampled is not None else {}
        self._noise_records: dict[int, dict[str, Any]] = {}   # user turn number -> keys for its record
        self.noise_seed: Optional[int] = None
        if noise is not None:
            from impulse_evals.voice.noise import noise_seed

            self.noise_seed = noise_seed(scenario.name, scenario.seed)
        self.io = io
        self.thresholds: dict[str, Any] = dict(thresholds or {})
        self.audio = audio_utils
        self.metadata_key = VOICE_METADATA_KEY
        self.redact = redact
        self.scenario_name = scenario.name
        self.warnings: list[str] = []                        # IMPULSE_STT_FAILED_WARNING, turn order
        self.language = scenario.language
        self.stem = files.stems.get(scenario.name) or scenario_file_stem(scenario.name, set())
        self.files = files
        self.cursor_ms = 0                                   # end of the last turn on the call timeline
        self.clips: list[tuple[str, int, bytes]] = []        # (speaker, start ms, WAV) for the recording
        self.records: list[dict[str, Any]] = []
        self._folder_ready = False

    # -- files ------------------------------------------------------------------------------------

    def _audio_folder(self) -> Path:
        folder = self.files.run_dir / AUDIO_FOLDER
        if not self._folder_ready:
            ensure_impulse_dir(self.files.impulse_dir)   # .impulse/ (with .gitignore) before any audio file
            folder.mkdir(parents=True, exist_ok=True)
            self._folder_ready = True
        return folder

    def _write(self, name: str, wav: bytes) -> str:
        """Write ``audio/<name>`` in the run folder; returns the path relative to the run folder."""
        (self._audio_folder() / name).write_bytes(wav)
        return f"{AUDIO_FOLDER}/{name}"

    def _save_turn(self, number: int, speaker: str, wav: bytes) -> Optional[str]:
        if not self.io.save_audio:
            return None
        return self._write(f"{self.stem}_t{number}_{speaker}.wav", wav)

    def _clip(self, speaker: str, start_ms: int, wav: bytes) -> None:
        if self.io.save_recording:
            self.clips.append((speaker, start_ms, wav))

    def _duration_ms(self, wav: bytes) -> int:
        return max(0, round(self.audio.duration_ms(wav)))

    def write_recording(self, transcript: CallTranscript) -> None:
        """The stereo recording (left user, right agent, each clip at its turn's ``timestamp_ms``)."""
        if not self.io.save_recording or not self.clips:
            return
        rate = max([RECORDING_MIN_SAMPLE_RATE] + [self.audio.wav_params(wav)[2] for _, _, wav in self.clips])
        left = [(start, wav) for speaker, start, wav in self.clips if speaker == "user"]
        right = [(start, wav) for speaker, start, wav in self.clips if speaker == "agent"]
        transcript.recording_path = self._write(f"{self.stem}_recording.wav", self.audio.mix_stereo(left, right, rate))

    def metadata(self) -> dict[str, Any]:
        """``CallTranscript.metadata["impulse_voice"]``: the per-turn records and the voice thresholds
        the report's timeline uses (FIXES-ROUND-2-SPEC.md B4)."""
        return {"turns": [dict(record) for record in self.records], "thresholds": dict(self.thresholds)}

    def checkpoint(self) -> tuple[int, int, int, int]:
        """The session state before a turn is recorded, for ``rollback`` (records, clips, warnings, cursor)."""
        return len(self.records), len(self.clips), len(self.warnings), self.cursor_ms

    def rollback(self, state: tuple[int, int, int, int]) -> None:
        """Forget what was recorded after ``checkpoint`` (a partial turn that could not be recorded)."""
        records, clips, warnings, cursor_ms = state
        del self.records[records:]
        del self.clips[clips:]
        del self.warnings[warnings:]
        self.cursor_ms = cursor_ms

    # -- turns ------------------------------------------------------------------------------------

    async def synthesize(self, text: str) -> bytes:
        """Impulse's TTS of the user's text, as WAV."""
        return self.audio.ensure_wav(await self.io.tts.synthesize(text, self.language, None))

    def noise_metadata(self) -> Optional[dict[str, Any]]:
        """``metadata["impulse_noise"]`` for a scenario with noise (None without): the settings, the
        noise file's name only (never its folder), and the seed."""
        if self.noise is None:
            return None
        from impulse_evals.voice.noise import default_label

        return {
            "experimental": True,
            "profile": self.noise.profile,
            "snr_db": float(self.noise.snr_db) if self.noise.snr_db is not None else None,
            "label": default_label(self.noise),
            "file": Path(self.noise.file).name if self.noise.file else None,
            "seed": self.noise_seed,
        }

    def _noise_source(self, rate: int) -> Any:
        """The noise file of ``profile: file`` resampled to ``rate`` (cached per run), else None."""
        noise = self.noise
        if noise is None or noise.profile != "file" or not noise.file:
            return None
        key = (noise.file, rate)
        cached = self._noise_resampled.get(key)
        if cached is None:
            source = self._noise_sources.get(noise.file)
            if source is None:
                from impulse_evals.voice.noise import load_noise_file

                source = self._noise_sources[noise.file] = load_noise_file(noise.file)
            samples, source_rate = source
            cached = self._noise_resampled[key] = (self.audio.resample(samples, source_rate, rate), rate)
        return cached

    def add_noise(self, clean: bytes, number: int) -> bytes:
        """The audio the agent hears for user turn ``number``: ``clean`` itself without noise, else the
        noisy mix (its measured SNR is kept for the turn's voice record)."""
        if self.noise is None:
            return clean
        from impulse_evals.voice.noise import mix_noise

        rate = self.audio.wav_params(clean)[2]
        mixed = mix_noise(
            clean, self.noise, seed=self.noise_seed or 0, turn=number, noise_source=self._noise_source(rate)
        )
        self._noise_records[number] = {
            "clean_audio_path": None,
            "noise_snr_db": mixed.measured_snr_db,
            "noise_scaled": mixed.scaled,
            "noise_offset_ms": mixed.offset_ms,
        }
        return mixed.wav

    def user_turn(self, text: str, wav: bytes, number: int, clean: Optional[bytes] = None) -> Turn:
        """The user turn ``number``: it starts where the previous turn ended and lasts as long as its audio.
        ``wav`` is the audio the agent hears; with noise, ``clean`` (the TTS audio) is saved too."""
        start = self.cursor_ms
        end = start + self._duration_ms(wav)
        self.cursor_ms = end
        turn = Turn(speaker="user", text=text, timestamp_ms=start, end_ms=end)
        turn.audio_path = self._save_turn(number, "user", wav)
        if clean is not None and number in self._noise_records:
            self._noise_records[number]["clean_audio_path"] = self._save_turn(number, "user_clean", clean)
        self._clip("user", start, wav)
        return turn

    async def _transcribe(self, wav: bytes, number: int) -> tuple[Optional[STTResult], Optional[str]]:
        """``(result, None)``, or ``(None, error)`` when Impulse's own STT raises a ``VoiceError``: the
        turn is kept (its STT text is None) and the run gets ``IMPULSE_STT_FAILED_WARNING`` (error
        redacted). Any other exception still ends the scenario with an error."""
        from impulse_evals.voice import VoiceError

        try:
            return await self.io.stt.transcribe(wav, self.language), None
        except VoiceError as exc:
            error = _error_text(exc, self.redact)
            self.warnings.append(
                IMPULSE_STT_FAILED_WARNING.format(scenario_name=self.scenario_name, turn=number, error=error)
            )
            return None, error

    async def round_trip(self, turn: Turn, number: int, wav: bytes) -> None:
        """VOICE-011: Impulse's STT of its own TTS audio; the confidence goes on the user turn. A
        failed STT records ``round_trip_error`` (VOICE-011 is then skipped) and keeps the turn."""
        result, error = await self._transcribe(wav, number)
        turn.stt_confidence = result.confidence if result is not None else None
        record: dict[str, Any] = {
            "turn_index": number,
            "speaker": "user",
            "audio_ms": (turn.end_ms or 0) - turn.timestamp_ms,
            "round_trip_text": result.text if result is not None else None,
            "round_trip_confidence": result.confidence if result is not None else None,
        }
        if error is not None:
            record["round_trip_error"] = error
        record.update(self._noise_records.pop(number, {}))   # noise simulation only
        self.records.append(record)

    async def agent_turn(self, reply: AgentReply, number: int, after: Optional[Turn]) -> Turn:
        """The agent turn ``number`` (``after`` = the user turn it answers; None for the opener).

        Text: the adapter's text when non-empty, else Impulse's STT text of the agent audio. It starts
        ``latency_ms`` after the user's audio ended (the opener: at the current end of the call) and
        lasts as long as its audio. A failed STT of the agent audio records ``stt_error`` (VOICE-015
        is then skipped) and keeps the turn with the adapter's text.
        """
        wav = self.audio.ensure_wav(bytes(reply.audio_bytes)) if reply.audio_bytes else None
        stt, stt_error = await self._transcribe(wav, number) if wav is not None else (None, None)
        if after is None:
            start = self.cursor_ms
        else:
            user_end = after.end_ms if after.end_ms is not None else after.timestamp_ms
            start = max(self.cursor_ms, user_end + _latency_gap_ms(reply))
        turn = _agent_turn(reply, start, opener=after is None, redact_text=self.redact)
        if not reply.text.strip() and stt is not None:
            turn.text = stt.text
        duration = self._duration_ms(wav) if wav is not None else None
        if wav is not None:
            turn.end_ms = start + (duration or 0)
            turn.stt_confidence = stt.confidence if stt is not None else None
            turn.audio_path = self._save_turn(number, "agent", wav)
            self._clip("agent", start, wav)
        self.cursor_ms = turn.end_ms if turn.end_ms is not None else start
        record: dict[str, Any] = {
            "turn_index": number,
            "speaker": "agent",
            "audio_ms": duration,
            "stt_text": stt.text if stt is not None else None,
            "stt_confidence": stt.confidence if stt is not None else None,
            "words": [word.model_dump() for word in stt.words] if stt is not None else [],
        }
        if stt_error is not None:
            record["stt_error"] = stt_error
        self.records.append(record)
        return turn


async def _converse(
    adapter: AgentInterface,
    scenario: Scenario,
    setup: _Setup,
    transcript: CallTranscript,
    clock: _CallClock,
    voice: Optional[_VoiceSession] = None,
) -> None:
    """Start the adapter and hold the conversation (SPEC.md sections 9, 10, and 14); turns go into
    ``transcript``. With ``voice`` (voice mode) the user's text is sent with its TTS audio and the
    agent's audio is transcribed; otherwise the text-mode turn times apply."""
    await adapter.start(scenario)
    clock.restart()
    if setup.config_prompt is not None:
        prompt: Optional[str] = setup.config_prompt   # config.yaml always wins
    else:
        prompt = adapter.get_system_prompt()
    transcript.agent_system_prompt = prompt
    summary = await setup.summary_for(prompt)

    turns = transcript.turns
    if scenario.direction == "outbound":
        called_ms = clock.now_ms()   # the opener has no user turn: stamped when send(None, None) is called
        record_opener = functools.partial(_record_opener, setup=setup, turns=turns, called_ms=called_ms, voice=voice)
        try:
            reply, closed = await _send(adapter, None)
        except AgentTurnError as exc:
            await _record_partial(exc, record_opener, turns, voice)
            raise
        _check_not_closed_before_first_turn(closed, reply, turns)
        if reply is not None:
            await record_opener(reply)
        if closed is not None:   # the agent hung up after speaking: a normal end (SPEC.md section 10)
            return

    user_turns = 0
    while user_turns < scenario.max_turns:
        text = await setup.simulator.user_turn(scenario, summary, turns)
        if text is None:   # <END_CALL>
            break
        audio: Optional[bytes] = None   # what the agent hears (noisy with noise simulation)
        clean: Optional[bytes] = None   # Impulse's TTS audio: the VOICE-011 round trip always uses it
        if voice is not None:
            clean = await voice.synthesize(text)   # before send: TTS time is never in the agent's latency
            audio = voice.add_noise(clean, len(turns) + 1)   # also before send; ``clean`` itself without noise
            user_turn = voice.user_turn(text, audio, len(turns) + 1, clean=clean if voice.noise is not None else None)
        else:
            user_turn = Turn(speaker="user", text=text, timestamp_ms=clock.now_ms())
        turns.append(user_turn)
        user_number = len(turns)
        user_turns += 1

        record = functools.partial(
            _record_reply,
            setup=setup,
            turns=turns,
            user_turn=user_turn,
            user_number=user_number,
            audio=clean,
            clock=clock,
            voice=voice,
        )
        try:
            reply, closed = await _send(adapter, text, audio)
        except AgentTurnError as exc:
            await _record_partial(exc, record, turns, voice, user_turn)
            raise
        _check_not_closed_before_first_turn(closed, reply, turns)
        await record(reply)
        if closed is not None:
            break


async def _record_opener(
    reply: AgentReply, setup: _Setup, turns: list[Turn], called_ms: int, voice: Optional[_VoiceSession]
) -> None:
    """Record the outbound opener (no user turn before it)."""
    if voice is not None:
        turns.append(await voice.agent_turn(reply, len(turns) + 1, None))
    else:
        turns.append(_agent_turn(reply, called_ms, opener=True, redact_text=setup.redact))


async def _record_reply(
    reply: Optional[AgentReply],
    setup: _Setup,
    turns: list[Turn],
    user_turn: Turn,
    user_number: int,
    audio: Optional[bytes],
    clock: _CallClock,
    voice: Optional[_VoiceSession],
) -> None:
    """After ``send()`` for ``user_turn``: the VOICE-011 round trip (voice mode; ``audio`` is the CLEAN
    TTS audio, also with noise simulation), then the agent turn of ``reply`` (when there is one) with its
    ``heard_text`` on the user turn."""
    if voice is not None and audio is not None:
        await voice.round_trip(user_turn, user_number, audio)
    if reply is None:
        return
    if reply.heard_text is not None:
        user_turn.heard_text = reply.heard_text
    if voice is not None:
        turns.append(await voice.agent_turn(reply, len(turns) + 1, user_turn))
    else:
        start = _agent_start_ms(user_turn, reply, clock.now_ms())
        turns.append(_agent_turn(reply, start, opener=False, redact_text=setup.redact))


async def _record_partial(
    exc: AgentTurnError,
    record: Callable[[AgentReply], Any],
    turns: list[Turn],
    voice: Optional[_VoiceSession],
    user_turn: Optional[Turn] = None,
) -> None:
    """The turn so far of an ``AgentTurnError`` (FIXES-ROUND-2-SPEC.md A6): when its ``reply`` holds
    text, a tool call, or audio, it is recorded by ``record`` exactly like a normal reply; the caller
    then re-raises ``exc`` (the scenario error). When recording fails, everything it added is undone
    (the turn list, the user turn's ``heard_text`` / ``stt_confidence``, the voice session's records)
    and the original error stands; a WAV file already written stays on disk."""
    reply = exc.reply if isinstance(exc.reply, AgentReply) else None
    if not _has_content(reply):
        return
    assert reply is not None
    count = len(turns)
    saved = (user_turn.heard_text, user_turn.stt_confidence) if user_turn is not None else None
    state = voice.checkpoint() if voice is not None else None
    try:
        await record(reply)
    except Exception:  # noqa: BLE001 - the adapter's error is the one reported
        del turns[count:]
        if user_turn is not None and saved is not None:
            user_turn.heard_text, user_turn.stt_confidence = saved
        if voice is not None and state is not None:
            voice.rollback(state)


async def _close_quietly(adapter: AgentInterface) -> None:
    try:
        await adapter.close()
    except Exception:  # noqa: BLE001, S110 - the original exception is the one reported
        pass


@dataclass
class _Evaluation:
    metrics: list[MetricResult] = field(default_factory=list)
    assertions: list[ScoreResult] = field(default_factory=list)
    no_verdict: list[str] = field(default_factory=list)   # critical rule IDs without a verdict, scenario order
    failures: list[tuple[str, str]] = field(default_factory=list)   # (check ID, redacted message), in order
    # Indexes in ``failures`` of the checks that raised ``JudgeStopped``: no per-check run warning (the
    # judge's own stop warning covers them; FIXES-ROUND-2-SPEC.md A7).
    stopped: set[int] = field(default_factory=set)


async def _evaluate(transcript: CallTranscript, scenario: Scenario, setup: _Setup) -> _Evaluation:
    """Metrics first (``select_metrics`` order), then assertions (scenario order), one at a time.

    A check whose evaluation raises (other than ``ConfigError``; e.g. a LiteLLM rate-limit or auth
    error from the judge) is recorded as skipped with "evaluation failed: {message}" (an assertion
    with ``no_verdict=False``) and the remaining checks still run. A check that cannot even be
    loaded has no result; its failure is still recorded.
    """
    evaluation = _Evaluation()

    def failed(check_id: str, exc: Exception) -> str:
        message = _error_text(exc, setup.redact)
        if isinstance(exc, JudgeStopped):
            evaluation.stopped.add(len(evaluation.failures))
        evaluation.failures.append((check_id, message))
        return EVALUATION_FAILED_REASON.format(message=message)

    for metric_id in select_metrics(scenario, setup.ctx):
        try:
            metric = get_metric(metric_id)
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - a metric module not available in this build
            failed(metric_id, exc)
            continue
        try:
            result = await metric.evaluate(transcript, scenario, setup.judge, setup.ctx)
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. a LiteLLM error from the judge
            result = metric.skipped(failed(metric_id, exc))
        evaluation.metrics.append(result)

    for rule_id in scenario.assertions:
        try:
            assertion = get_assertion(rule_id)
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - a pack not available in this build
            failed(rule_id, exc)
            continue
        if setup.dry_run:
            score = assertion.skipped(DRY_RUN_REASON)
        else:
            try:
                score = await assertion.evaluate(transcript, scenario, setup.judge)
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - e.g. a LiteLLM error from the judge
                score = assertion.skipped(failed(rule_id, exc))   # result "skip", no_verdict False
        evaluation.assertions.append(score)
        if score.result == "skip" and score.no_verdict and score.severity == "critical":
            evaluation.no_verdict.append(rule_id)
    return evaluation


async def _run_scenario(
    config: ProjectConfig, scenario: Scenario, setup: _Setup, factory: AdapterFactory
) -> _Outcome:
    if scenario.inject_events:
        _print_console(f"{scenario.name}: {INJECT_EVENTS_NOTE}")
    inject_events = [event.model_copy(deep=True) for event in scenario.inject_events]
    transcript = CallTranscript(
        turns=[],
        duration_ms=0,
        metadata={},
        scenario_name=scenario.name,
        language=scenario.language,
        accent=scenario.accent,
        voice_mode=setup.voice_mode,
    )
    clock = _CallClock()
    error: Optional[str] = None
    adapter: Optional[AgentInterface] = None
    voice: Optional[_VoiceSession] = None
    try:
        if setup.voice is not None and setup.files is not None:
            voice = _VoiceSession(
                setup.voice,
                scenario,
                setup.files,
                redact=setup.redact,
                thresholds=_voice_thresholds(config),
                noise=effective_noise(config, scenario),
                noise_sources=setup.noise_sources,
                noise_resampled=setup.noise_resampled,
            )
        adapter = factory(config)
        await _converse(adapter, scenario, setup, transcript, clock, voice)
    except ConfigError:
        if adapter is not None:
            await _close_quietly(adapter)
        raise
    except Exception as exc:  # noqa: BLE001 - adapter, agent, or simulator failure: a scenario error
        error = _error_text(exc, setup.redact)
    except BaseException:
        if adapter is not None:
            await _close_quietly(adapter)
        raise

    adapter_warnings: list[str] = []
    close_failures: list[str] = []   # after a complete conversation: run warnings, the results are kept (M2)
    if adapter is not None:
        try:
            await adapter.close()
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            if error is None:
                close_failures.append(_error_text(exc, setup.redact))
        try:
            # Redacted like the warnings, in the form it will be written: a custom adapter may put
            # anything there (AUDIT.md finding 42).
            transcript.metadata.update(_redacted_metadata(adapter.get_metadata() or {}, setup.redact))
            adapter_warnings = [setup.redact(str(w)) for w in (adapter.get_warnings() or [])]
        except Exception as exc:  # noqa: BLE001
            if error is None:
                close_failures.append(_error_text(exc, setup.redact))
    adapter_warnings += [
        CLOSE_FAILED_WARNING.format(scenario_name=scenario.name, error=text) for text in close_failures
    ]
    if voice is not None and voice.noise is not None and getattr(adapter, "accepts_user_audio", None) is False:
        adapter_warnings.append(NOISE_GENERIC_NO_AUDIO_WARNING)
    if voice is not None:
        noise_metadata = voice.noise_metadata()
        if noise_metadata is not None:
            transcript.metadata[NOISE_METADATA_KEY] = noise_metadata   # the runner's own key wins
        # Written also after an error: the audio of the turns so far helps to see what went wrong.
        try:
            voice.write_recording(transcript)
        except Exception as exc:  # noqa: BLE001 - e.g. the run folder cannot be written
            if error is None:
                error = _error_text(exc, setup.redact)
        transcript.metadata[voice.metadata_key] = voice.metadata()   # the runner's own key wins
        transcript.duration_ms = voice.cursor_ms   # the call as heard (Impulse's processing time excluded)
    else:
        transcript.duration_ms = clock.now_ms()
    voice_warnings = list(voice.warnings) if voice is not None else []

    if error is not None:
        # An error before evaluation: no metric or assertion runs (SPEC.md section 7).
        result = ScenarioResult(
            scenario_name=scenario.name,
            transcript=transcript,
            assertion_results=[],
            metric_results=[],
            passed=False,
            error=error,
            inject_events=inject_events,
        )
        return _Outcome(result=result, adapter_warnings=adapter_warnings, voice_warnings=voice_warnings)

    evaluation = await _evaluate(transcript, scenario, setup)
    first_unreported = 0   # failures before this index are used as the error
    if evaluation.no_verdict:
        # SPEC.md section 13: the error is exactly the no-verdict text, whatever else failed.
        error = NO_VERDICT_ERROR.format(rule_id=evaluation.no_verdict[0])
    elif evaluation.failures:
        check_id, message = evaluation.failures[0]
        error = EVALUATION_FAILED_ERROR.format(check_id=check_id, message=message)
        first_unreported = 1
    # Evaluation failures not used as the error become run warnings, except a check stopped by the
    # judge (JudgeStopped): the judge's own warning covers those.
    unreported: list[tuple[str, str]] = [
        failure
        for index, failure in enumerate(evaluation.failures)
        if index >= first_unreported and index not in evaluation.stopped
    ]
    result = ScenarioResult(
        scenario_name=scenario.name,
        transcript=transcript,
        assertion_results=evaluation.assertions,
        metric_results=evaluation.metrics,
        passed=False,
        error=error,
        inject_events=inject_events,
    )
    result.passed = compute_passed(result)
    return _Outcome(
        result=result,
        adapter_warnings=adapter_warnings,
        no_verdict_checks=[f"{scenario.name}:{rule_id}" for rule_id in evaluation.no_verdict],
        evaluation_warnings=[
            EVALUATION_FAILED_WARNING.format(scenario_name=scenario.name, check_id=check_id, message=message)
            for check_id, message in unreported
        ],
        voice_warnings=voice_warnings,
    )


async def _run_scenarios(
    config: ProjectConfig, setup: _Setup, factory: AdapterFactory, concurrency: int
) -> list[_Outcome]:
    """Every scenario, at most ``concurrency`` at a time; outcomes in config order."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(scenario: Scenario) -> _Outcome:
        async with semaphore:
            return await _run_scenario(config, scenario, setup, factory)

    tasks = [asyncio.create_task(one(scenario)) for scenario in config.scenarios]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


# ------------------------------------------------------------------------------------------------
# Run summary
# ------------------------------------------------------------------------------------------------


def _summary_label(score: float) -> Literal["good", "warning", "poor"]:
    score = round(score, 9)   # a mean such as 0.6 must not fall below the threshold by float error
    if score >= SUMMARY_THRESHOLD_WARN:
        return "good"
    if score >= SUMMARY_THRESHOLD_FAIL:
        return "warning"
    return "poor"


def build_metric_summary(scenario_results: Sequence[ScenarioResult]) -> dict[str, MetricResult]:
    """``RunResult.metric_summary`` (SPEC.md section 7): one entry per metric ID with at least one
    non-skipped, non-informational result; first-seen order."""
    groups: dict[str, list[MetricResult]] = {}
    for scenario_result in scenario_results:
        for metric in scenario_result.metric_results:
            groups.setdefault(metric.metric_id, []).append(metric)

    summary: dict[str, MetricResult] = {}
    for metric_id, results in groups.items():
        counted = [r for r in results if not r.skipped and not r.informational]
        if not counted:
            continue
        skipped = sum(1 for r in results if r.skipped)
        score = sum(r.score for r in counted) / len(counted)
        raw_values = [r.raw_value for r in counted if r.raw_value is not None]
        first = counted[0]
        summary[metric_id] = MetricResult(
            metric_id=metric_id,
            name=first.name,
            category=first.category,
            score=score,
            raw_value=sum(raw_values) / len(raw_values) if raw_values else None,
            label=_summary_label(score),
            reason=METRIC_SUMMARY_REASON.format(n=len(counted), m=skipped),
            threshold_warn=SUMMARY_THRESHOLD_WARN,
            threshold_fail=SUMMARY_THRESHOLD_FAIL,
            turn_index=None,
            requires_audio=first.requires_audio,
            is_proxy=any(r.is_proxy for r in counted),
        )
    return summary


def build_latency_profile(scenario_results: Sequence[ScenarioResult], *, is_real_audio: bool) -> LatencyProfile:
    """Latency over every agent turn that follows a user turn (the outbound opener is excluded);
    ``numpy.percentile`` with linear interpolation; all zeros when there is no latency. A latency that
    is not finite (NaN, infinity) is ignored."""
    values: list[float] = []
    for scenario_result in scenario_results:
        seen_user = False
        for turn in scenario_result.transcript.turns:
            if turn.speaker == "user":
                seen_user = True
            elif seen_user and turn.latency_ms is not None and math.isfinite(turn.latency_ms):
                values.append(float(turn.latency_ms))
    if not values:
        return LatencyProfile(
            p50_ms=0.0, p75_ms=0.0, p95_ms=0.0, p99_ms=0.0, all_latencies_ms=[], is_real_audio=is_real_audio
        )
    import numpy  # core dependency; imported here to keep ``import impulse_evals.runner`` light

    p50, p75, p95, p99 = (float(v) for v in numpy.percentile(values, [50, 75, 95, 99]))
    return LatencyProfile(
        p50_ms=p50, p75_ms=p75, p95_ms=p95, p99_ms=p99, all_latencies_ms=values, is_real_audio=is_real_audio
    )


def _assemble(
    config: ProjectConfig,
    *,
    run_id: str,
    started: datetime,
    scenario_results: list[ScenarioResult],
    warnings: _Warnings,
    judge_model: str,
    voice_mode: bool,
    redactor: _Redactor,
) -> RunResult:
    total = len(scenario_results)
    passed = sum(1 for result in scenario_results if result.passed)
    voice = config.voice_config if voice_mode else None
    return RunResult(
        run_id=run_id,
        timestamp=_iso_timestamp(started),
        agent_entrypoint=agent_entrypoint(config),
        agent_type=config.agent.type,
        agent_domain=agent_domain(config.scenarios),
        judge_model=judge_model,
        voice_mode=voice_mode,
        tts_provider=voice.tts.provider if voice is not None else None,
        stt_provider=voice.stt.provider if voice is not None else None,
        total_scenarios=total,
        passed=passed,
        failed=total - passed,
        overall_pass_rate=passed / total if total else 0.0,
        scenario_results=scenario_results,
        cohorts=build_cohorts(scenario_results),
        critical_failures=[
            score
            for result in scenario_results
            for score in result.assertion_results
            if score.result == "fail" and score.severity == "critical"
        ],
        metric_summary=build_metric_summary(scenario_results),
        latency_profile=build_latency_profile(scenario_results, is_real_audio=voice_mode),
        warnings=warnings.as_list(redactor),
    )


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------


async def run(
    config: ProjectConfig,
    *,
    dry_run: bool = False,
    concurrency: int = 1,
    adapter_factory: Optional[AdapterFactory] = None,
) -> RunResult:
    """Run every scenario of ``config`` and return the ``RunResult``. Writes no run.json or report
    (the CLI does); in voice mode the per-turn audio and the call recording are written under
    ``./.impulse/runs/<run_id>/audio/``.

    ``./.impulse/config.toml`` (current directory) is read when present, for models, options, and
    keys. ``adapter_factory(config)`` returns a new ``AgentInterface`` per scenario (default: the
    adapter named by ``agent.adapter``). Raises ``ConfigError`` for configuration problems.
    """
    started = datetime.now(UTC)
    run_id = make_run_id(started)
    toml = load_project_toml()
    redactor = _config_redactor(config, toml)
    warnings = _Warnings(config.warnings)
    voice_mode = config.voice_config is not None and not dry_run
    if dry_run:
        warnings.add(DRY_RUN_WARNING)   # every dry run, every adapter (SPEC.md section 17)
        if any(effective_noise(config, scenario) is not None for scenario in config.scenarios):
            warnings.add(NOISE_DRY_RUN_WARNING)
        # A dry run resolves no key (it never imports LiteLLM): its scenario errors and warnings are
        # redacted with the keys Impulse's own tables and name endings point to (AUDIT.md finding 31).
        redactor = redactor.with_values(static_key_values())

    if dry_run and config.agent.adapter != "generic":
        # SPEC.md section 17: Pipecat, LiveKit and HTTP configs are validated without connecting; a
        # custom adapter's class is imported and checked, never created (FIXES-ROUND-3-SPEC.md B3).
        if config.agent.adapter == "custom":
            from impulse_evals.adapters.custom_adapter import load_adapter_class

            with _config_errors_redacted(redactor):
                load_adapter_class(config)
        warnings.add(DRY_RUN_REMOTE_WARNING.format(adapter=config.agent.adapter))
        return _assemble(
            config,
            run_id=run_id,
            started=started,
            scenario_results=[],
            warnings=warnings,
            judge_model=_dry_run_judge_model(config, toml),
            voice_mode=False,
            redactor=redactor,
        )

    with _config_errors_redacted(redactor):
        setup = _prepare(config, toml, dry_run=dry_run, voice_mode=voice_mode, warnings=warnings, redactor=redactor)
    if setup.voice is not None:
        setup.files = _run_files(config, run_id)
        _load_noise_files(config, setup)   # before the first paid call, like the voice providers

    concurrency, message = effective_concurrency(config, concurrency)
    if message is not None:
        _print_console(message)
        warnings.add(message)

    if adapter_factory is not None:
        factory: AdapterFactory = adapter_factory
    else:
        factory = functools.partial(default_adapter_factory, run_id=run_id, redact=setup.redact)
    # A ConfigError from a scenario (e.g. the agent module failing at import with an error that echoes
    # a key) leaves the run with its message redacted by the run's full redactor (AUDIT.md finding 31).
    with _config_errors_redacted(setup.redact):
        outcomes = await _run_scenarios(config, setup, factory, concurrency)

    if any(NOISE_METADATA_KEY in outcome.result.transcript.metadata for outcome in outcomes):
        warnings.add(NOISE_EXPERIMENTAL_WARNING)
    for outcome in outcomes:
        warnings.extend(outcome.adapter_warnings)
        warnings.extend(outcome.voice_warnings)   # Impulse's own STT failures, one per failed turn
    warnings.extend(setup.judge.get_warnings())
    if any(
        metric.metric_id == ROUND_TRIP_METRIC and metric.label == "poor" and not metric.skipped
        for outcome in outcomes
        for metric in outcome.result.metric_results
    ):
        warnings.add(ROUND_TRIP_WARNING)   # METRICS.md VOICE-011: once per run
    for outcome in outcomes:
        warnings.extend(outcome.evaluation_warnings)
    no_verdict_checks = [check for outcome in outcomes for check in outcome.no_verdict_checks]
    if no_verdict_checks:
        warnings.add(NO_VERDICT_WARNING_PREFIX + ", ".join(no_verdict_checks))

    return _assemble(
        config,
        run_id=run_id,
        started=started,
        scenario_results=[outcome.result for outcome in outcomes],
        warnings=warnings,
        judge_model=setup.judge_model,
        voice_mode=voice_mode,
        redactor=setup.redact,
    )
