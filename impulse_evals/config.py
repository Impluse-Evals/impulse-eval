# SPDX-License-Identifier: AGPL-3.0-only
"""Load and validate ``config.yaml`` and ``.impulse/config.toml`` (SPEC.md sections 8.1 and 8.2).

``load_config`` validates structure and every SPEC.md section 8.2 rule, but it never imports the
generic ``entrypoint``, never imports a compliance pack module, and never resolves model strings
or keys. Resolution (``resolve_model``, ``resolve_model_options``, ``api_key_for_role``) happens
only when a run needs an LLM, so ``impulse run --dry-run`` works with ``YOUR_..._MODEL``
placeholders. Key values are held as ``SecretStr`` and never appear in reprs, logs, or errors.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import re
import tomllib
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Optional, Union, get_args, get_origin
from urllib.parse import quote, unquote_plus, urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator
from pydantic_core import PydanticCustomError

# The metric and assertion ID tables live in the Apache-2.0 plugin API base files (AGPL code may
# import Apache code); PACK_RULE_IDS is re-exported as config.PACK_RULE_IDS.
from impulse_evals.assertions.base import KNOWN_ASSERTION_IDS
from impulse_evals.assertions.base import PACK_RULE_IDS as PACK_RULE_IDS
from impulse_evals.metrics.base import KNOWN_METRIC_IDS
from impulse_evals.models import AgentTool, NoiseSettings, Scenario

# ------------------------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------------------------


class ConfigError(Exception):
    """A configuration error (CLI exit code 2).

    ``field`` is the exact path of the offending value, e.g. ``scenarios[1].expected_tool_calls[0].name``
    (for errors about a whole file, the file path). ``message`` never contains a key value.
    """

    def __init__(self, field: str, message: str) -> None:
        super().__init__(field, message)
        self.field = field
        self.message = message

    def __str__(self) -> str:
        # Messages that start with the tail of the field path (e.g. "pipecat.mode: daily is planned
        # for v2. ..." for field "agent.pipecat.mode") are not prefixed twice; the full path is kept.
        head, sep, _ = self.message.partition(": ")
        if sep and (self.field == head or self.field.endswith("." + head)):
            return self.field + self.message[len(head):]
        return f"{self.field}: {self.message}"


class UnknownProviderError(ConfigError):
    """``key_name_for_model`` cannot determine a model string's LiteLLM provider (so not the key it needs)."""


# ------------------------------------------------------------------------------------------------
# Known values (SPEC.md sections 8.2, 11, 14; METRICS.md). Static lists so that validating a config
# never imports a pack module (packs load lazily, SPEC.md section 13). KNOWN_METRIC_IDS,
# PACK_RULE_IDS, and KNOWN_ASSERTION_IDS are imported from metrics.base / assertions.base (above).
# ------------------------------------------------------------------------------------------------

LANGUAGE_PATTERN = r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$"
_LANGUAGE_RE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*")
LANGUAGE_PLACEHOLDER = "YOUR_LANGUAGE"

KNOWN_DOMAINS = (
    "fsi", "healthcare", "recruitment", "ecommerce", "edtech",
    "telecom", "realestate", "travel", "legal", "general",
)

VOICE_PROVIDERS = ("local_hf", "local_light", "openai_compatible", "impulse_hosted", "litellm")
_PROVIDERS_NEEDING_BASE_URL = ("openai_compatible", "impulse_hosted")
_PROVIDERS_NEEDING_MODEL = ("litellm",)

PIPECAT_MODES = ("eval", "websocket")
# Keys that only websocket mode uses (eval mode relies on Pipecat's own audio pacing). sample_rate is
# used in both modes and is never one of them (SPEC.md section 8.2).
PIPECAT_WEBSOCKET_ONLY_KEYS = ("chunk_ms", "realtime_pacing", "trailing_silence_ms")

# Noise simulation (EXPERIMENTAL; SPEC.md sections 8.2 and 14.4). The profile list and SNR limits are
# also in voice/noise.py (a test keeps them equal); the mixer is imported only to read a noise file.
NOISE_PROFILES = ("white", "pink", "brown", "file")
NOISE_SNR_MIN_DB = -5.0
NOISE_SNR_MAX_DB = 40.0
NOISE_TEXT_MODE_WARNING = "noise simulation needs voice mode (voice_config); noise settings are ignored."
NOISE_SIMULATION_KEY_WARNING = "metrics.noise_simulation is replaced by voice_config.noise; it is ignored."
NOISE_HTTP_TEXT_ONLY_WARNING = (
    "agent.http.send_audio is false: the HTTP agent receives text only, so noise simulation has no effect on it."
)

STANDARD_KEY_NAMES = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "IMPULSE_VOICE_API_KEY")

ROLES = ("primary", "fast", "simulator")
MODEL_ENV_VARS = {
    "primary": "IMPULSE_MODEL_PRIMARY",
    "fast": "IMPULSE_MODEL_FAST",
    "simulator": "IMPULSE_MODEL_SIMULATOR",
}
_MODEL_FIELDS = {"primary": "judge.primary_model", "fast": "judge.fast_model", "simulator": "simulator.model"}
_PLACEHOLDER_RE = re.compile(r"YOUR_[A-Z0-9_]*MODEL")
_LOCAL_MODEL_PREFIXES = ("ollama/", "ollama_chat/")

DEFAULT_TOML_PATH = Path(".impulse") / "config.toml"

# ------------------------------------------------------------------------------------------------
# config.yaml models (SPEC.md section 8.2). Attribute names equal the config.yaml keys.
# ------------------------------------------------------------------------------------------------


def _reject_boolean_seconds(value: Any) -> Any:
    """Field validator (mode "before") shared by every ``turn_timeout_s``: pydantic would read true as 1.0 s."""
    if isinstance(value, bool):
        # PydanticCustomError: the message is kept as is (a ValueError gets a "Value error, " prefix).
        raise PydanticCustomError(
            "turn_timeout_boolean", "turn_timeout_s must be a number of seconds, not a boolean"
        )
    return value


def is_printable_ascii(value: str) -> bool:
    """True when every character is printable ASCII (0x20..0x7E): no control character, no DEL, no
    non-ASCII character. The empty string is accepted. Used for HTTP header values and bearer tokens."""
    return all(" " <= character <= "~" for character in value)


def _check_turn_timeout(value: Optional[float], field: str) -> None:
    """The ``turn_timeout_s`` rules shared by LiveKit, Pipecat and HTTP (SPEC.md section 8.2): None, or
    a finite number greater than 0. ``field`` is the config path (``agent.livekit.turn_timeout_s``, ...)."""
    if value is None:
        return
    if not value > 0:   # also rejects NaN
        raise ConfigError(field, f"turn_timeout_s {value:g} must be greater than 0")
    if not math.isfinite(value):   # a hung agent would hang the run
        raise ConfigError(field, "turn_timeout_s must be a finite number of seconds")


class PipecatConfig(BaseModel):
    mode: str = "eval"                    # eval | websocket (checked by load_config with the v2 message)
    url: Optional[str] = None
    sample_rate: int = 16000              # both modes: user audio is sent at this rate (the bot's input rate)
    chunk_ms: int = 20                    # websocket mode only
    realtime_pacing: bool = True          # websocket mode only
    trailing_silence_ms: int = 600        # websocket mode only
    end_of_turn_grace_ms: int = 800
    handshake_timeout_s: float = 10.0
    headers: dict[str, str] = {}
    api_key_env: Optional[str] = None     # NAME of an env var, never its value
    turn_timeout_s: Optional[float] = None   # both modes; finite and > 0 when set; None: voice_config.timeout_s, else 30 s

    _reject_boolean_timeout = field_validator("turn_timeout_s", mode="before")(_reject_boolean_seconds)


class LiveKitConfig(BaseModel):
    url: Optional[str] = None
    api_key_env: str = "LIVEKIT_API_KEY"
    api_secret_env: str = "LIVEKIT_API_SECRET"
    room_prefix: str = "impulse-"
    agent_identity: Optional[str] = None
    agent_name: Optional[str] = None      # set: the access token dispatches this agent into each scenario room
    end_of_turn_grace_ms: int = 800
    handshake_timeout_s: float = 10.0
    turn_timeout_s: Optional[float] = None   # finite and > 0 when set; None: voice_config.timeout_s, else 30 s

    _reject_boolean_timeout = field_validator("turn_timeout_s", mode="before")(_reject_boolean_seconds)


class HTTPConfig(BaseModel):
    url: Optional[str] = None             # required when adapter is http; http:// or https://
    start_url: Optional[str] = None       # optional: POSTed once when a scenario starts
    close_url: Optional[str] = None       # optional: POSTed once when a scenario ends
    headers: dict[str, str] = {}          # extra request headers (values are redacted in errors)
    api_key_env: Optional[str] = None     # NAME of an env var; its value is sent as "Authorization: Bearer <value>"
    turn_timeout_s: Optional[float] = None   # seconds per request; None: voice_config.timeout_s, else 30 s
    send_audio: bool = False              # voice mode: add "user_audio" (base64 WAV) to each turn request

    _reject_boolean_timeout = field_validator("turn_timeout_s", mode="before")(_reject_boolean_seconds)


class AgentConfig(BaseModel):
    type: Literal["voice", "chat", "multimodal"] = "voice"
    adapter: Literal["generic", "livekit", "pipecat", "custom", "http"] = "generic"
    entrypoint: Optional[str] = None      # generic (module:function) and custom (module:Class); imported when the adapter is created
    system_prompt: Optional[str] = None   # file path; load_config stores it resolved (absolute)
    model: Optional[str] = None
    tools: Optional[list[AgentTool]] = None   # None = unknown; [] = the agent has no tools
    options: dict[str, Any] = {}          # free-form settings for a custom adapter, passed through unchanged
    pipecat: Optional[PipecatConfig] = None   # kept only when adapter is pipecat
    livekit: Optional[LiveKitConfig] = None   # kept only when adapter is livekit
    http: Optional[HTTPConfig] = None         # kept only when adapter is http

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tools(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": item} if isinstance(item, str) else item for item in value]
        return value


class JudgeConfig(BaseModel):
    primary_model: Optional[str] = None
    fast_model: Optional[str] = None
    primary_options: dict[str, Any] = {}
    fast_options: dict[str, Any] = {}


class SimulatorConfig(BaseModel):
    model: Optional[str] = None
    persona_style: Literal["realistic", "adversarial", "polite"] = "realistic"
    mode: str = "cascade"                 # cascade only in v1 (checked by load_config)
    options: dict[str, Any] = {}


class TTSConfig(BaseModel):
    # local_hf | local_light | openai_compatible | impulse_hosted | litellm (checked by load_config)
    provider: str = "local_hf"
    model: Optional[str] = None           # required for litellm (a LiteLLM speech model string)
    models_by_language: dict[str, str] = {}
    voice: Optional[str] = None
    device: str = "auto"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


class STTConfig(BaseModel):
    # local_hf | local_light | openai_compatible | impulse_hosted | litellm (checked by load_config)
    provider: str = "local_hf"
    model: Optional[str] = None           # required for litellm (a LiteLLM transcription model string)
    device: str = "auto"
    compute_type: str = "auto"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


class VoiceConfig(BaseModel):
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    save_audio: bool = True
    save_recording: bool = True
    timeout_s: float = 30.0
    noise: Optional[NoiseSettings] = None   # noise simulation (EXPERIMENTAL): the default for every scenario


class VoiceMetricsConfig(BaseModel):
    ttfa_warn_ms: float = 1500
    ttfa_fail_ms: float = 3000
    dead_air_ms: float = 1500
    wer_warn: float = 0.15
    wer_fail: float = 0.30
    tail_latency_warn_ms: float = 3000
    tail_latency_fail_ms: float = 5000
    speech_rate_wpm: tuple[float, float] = (110, 180)


class MetricsConfig(BaseModel):
    enabled: bool = True
    voice: VoiceMetricsConfig = Field(default_factory=VoiceMetricsConfig)
    noise_simulation: bool = False        # no-op, kept so older configs load; replaced by voice_config.noise


class ProjectConfig(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    simulator: SimulatorConfig
    voice_config: Optional[VoiceConfig] = None   # None = text mode
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    scenarios: list[Scenario] = []
    config_dir: Path                      # folder of the config file
    warnings: list[str] = []              # every config warning, in SPEC.md section 8.2 order


# ------------------------------------------------------------------------------------------------
# .impulse/config.toml models (SPEC.md section 8.1)
# ------------------------------------------------------------------------------------------------


class TomlRoleOptions(BaseModel):
    api_base: str = ""
    api_key_env: str = ""                 # NAME of an env var, never resolved at load
    extra_params: dict[str, Any] = {}


class TomlModelOptions(BaseModel):
    primary: TomlRoleOptions = Field(default_factory=TomlRoleOptions)
    fast: TomlRoleOptions = Field(default_factory=TomlRoleOptions)
    simulator: TomlRoleOptions = Field(default_factory=TomlRoleOptions)


class TomlModels(BaseModel):
    primary: Optional[str] = None
    fast: Optional[str] = None
    simulator: Optional[str] = None


class TomlDefaults(BaseModel):
    reporter: Literal["json", "html", "all"] = "all"
    language: Optional[str] = None        # no built-in default language
    fail_on: Literal["critical", "high", "any"] = "high"   # fixes round 4 (O3): was critical


class TomlConfig(BaseModel):
    keys: dict[str, SecretStr] = {}
    models: TomlModels = Field(default_factory=TomlModels)
    model_options: TomlModelOptions = Field(default_factory=TomlModelOptions)
    defaults: TomlDefaults = Field(default_factory=TomlDefaults)


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


def _loc_to_path(loc: Iterable[Any]) -> str:
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}" if path else str(part)
    return path


def _validate(model_cls: type[BaseModel], data: Mapping[str, Any]) -> Any:
    """Validate ``data``; the first pydantic error becomes a ConfigError (input values never shown)."""
    error: Optional[ConfigError] = None
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_context=False, include_input=False)[0]
        error = ConfigError(_loc_to_path(first["loc"]) or "(root)", first["msg"])
    # Raised outside the except block so the pydantic error (which can echo input values) is not chained.
    raise error


def _split_optional(annotation: Any) -> tuple[Any, bool]:
    """``(inner type, accepts None)`` for a field annotation such as ``Optional[X]``."""
    if get_origin(annotation) in (Union, types.UnionType):
        args = get_args(annotation)
        rest = [arg for arg in args if arg is not type(None)]
        return (rest[0] if len(rest) == 1 else annotation), len(rest) != len(args)
    return annotation, False


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _is_container(annotation: Any) -> bool:
    return _is_model(annotation) or annotation in (list, dict, tuple) or get_origin(annotation) in (list, dict, tuple)


def _drop_null_blocks(model_cls: type[BaseModel], data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``data`` without null blocks, lists, and mappings, so they behave exactly like absent keys.

    A YAML key written with no body (``judge:``) loads as null. Fields that accept None
    (``voice_config``, ``agent.tools``, ``agent.pipecat``, ...) keep the null, since None is their
    meaning; null scalars are left to validation. Recurses into nested blocks and lists of blocks.
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        field = model_cls.model_fields.get(key) if isinstance(key, str) else None
        if field is None:
            out[key] = value
            continue
        inner, accepts_none = _split_optional(field.annotation)
        if value is None and not accepts_none and _is_container(inner):
            continue
        if _is_model(inner) and isinstance(value, Mapping):
            value = _drop_null_blocks(inner, value)
        elif get_origin(inner) is list and isinstance(value, list):
            item_type = (get_args(inner) or (None,))[0]
            if _is_model(item_type):
                value = [_drop_null_blocks(item_type, item) if isinstance(item, Mapping) else item for item in value]
        out[key] = value
    return out


def _is_valid_language(code: str) -> bool:
    return isinstance(code, str) and _LANGUAGE_RE.fullmatch(code) is not None


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _read_text(path: Path) -> str:
    error: Optional[ConfigError] = None
    try:
        return path.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        error = ConfigError(str(path), "file not found")
    except OSError as exc:
        error = ConfigError(str(path), f"cannot read file: {exc.strerror or type(exc).__name__}")
    except UnicodeDecodeError:
        error = ConfigError(str(path), "file is not valid UTF-8")
    raise error


# ------------------------------------------------------------------------------------------------
# config.yaml
# ------------------------------------------------------------------------------------------------


def _yaml_mark_text(mark: Any) -> str:
    """`` (line L, column C)`` (1-based) for a PyYAML mark; its source snippet is never used."""
    line, column = getattr(mark, "line", None), getattr(mark, "column", None)
    if not isinstance(line, int) or not isinstance(column, int):
        return ""
    return f" (line {line + 1}, column {column + 1})"


def yaml_error_text(exc: yaml.YAMLError) -> str:
    """A YAML syntax error as its problem and position only.

    PyYAML's own message quotes the offending source line (cut to a snippet), which could hold a key
    written in ``config.yaml`` or in an LLM answer (SPEC.md section 8.1: keys never appear in errors;
    a cut key cannot be redacted), so it is never used. Also used by ``impulse generate`` and
    ``impulse convert-failure`` for invalid YAML answers.
    """
    if isinstance(exc, yaml.MarkedYAMLError):
        parts: list[str] = []
        context_mark, problem_mark = exc.context_mark, exc.problem_mark
        if exc.context:
            same_place = (
                problem_mark is not None
                and context_mark is not None
                and (context_mark.line, context_mark.column) == (problem_mark.line, problem_mark.column)
            )
            parts.append(exc.context + ("" if same_place or exc.problem is None else _yaml_mark_text(context_mark)))
        if exc.problem:
            parts.append(exc.problem + _yaml_mark_text(problem_mark))
        elif problem_mark is not None or context_mark is not None:
            parts.append("syntax error" + _yaml_mark_text(problem_mark or context_mark))
        return "; ".join(parts) or type(exc).__name__
    if isinstance(exc, yaml.reader.ReaderError):   # a character code and a position, never source text
        code = exc.character[0] if isinstance(exc.character, bytes) and exc.character else exc.character
        where = f" at position {exc.position}" if isinstance(exc.position, int) else ""
        if isinstance(code, int):
            return f"{exc.reason}: character #x{code:04x}{where}"
        return f"{exc.reason}{where}"
    return type(exc).__name__


def load_config(path: str | os.PathLike[str]) -> ProjectConfig:
    """Load and validate a project ``config.yaml`` (SPEC.md section 8.2).

    Raises ConfigError naming the exact field path. Warnings go to ``ProjectConfig.warnings``.
    """
    path = Path(path)
    text = _read_text(path)
    error: Optional[ConfigError] = None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        error = ConfigError(str(path), f"invalid YAML: {yaml_error_text(exc)}")
    if error is not None:
        raise error
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(str(path), "the top level of config.yaml must be a mapping")

    data = {key: value for key, value in raw.items() if key not in ("config_dir", "warnings")}
    # A block or list written with no body (null) is the same as an absent one: it takes its default.
    data = _drop_null_blocks(ProjectConfig, data)
    if data.get("simulator") is None:
        raise ConfigError("simulator", "the simulator block is required")

    agent = data.get("agent")
    if isinstance(agent, Mapping):
        agent = dict(agent)
        adapter = agent.get("adapter", "generic")
        # pipecat / livekit / http blocks are ignored unless that adapter is selected (SPEC.md section 8.2).
        for block in ("pipecat", "livekit", "http"):
            if adapter != block:
                agent.pop(block, None)
        data["agent"] = agent

    data["config_dir"] = path.resolve().parent
    cfg: ProjectConfig = _validate(ProjectConfig, data)

    _check_simulator(cfg)
    _check_agent(cfg)
    _check_voice_config(cfg)
    domain_warnings = _check_scenarios(cfg)

    warnings = list(domain_warnings)
    if not cfg.metrics.enabled:
        warnings.append("metrics.enabled is false; no metrics will run, including expected tool call checks.")
    if cfg.metrics.noise_simulation:
        warnings.append(NOISE_SIMULATION_KEY_WARNING)
    if cfg.voice_config is None:
        if any(isinstance(scenario.noise, NoiseSettings) for scenario in cfg.scenarios):
            warnings.append(NOISE_TEXT_MODE_WARNING)
    elif (
        cfg.agent.adapter == "http"
        and cfg.agent.http is not None
        and not cfg.agent.http.send_audio
        and any(effective_noise(cfg, scenario) is not None for scenario in cfg.scenarios)
    ):
        warnings.append(NOISE_HTTP_TEXT_ONLY_WARNING)
    pipecat = cfg.agent.pipecat
    if cfg.agent.adapter == "pipecat" and pipecat is not None and pipecat.mode == "eval":
        for key in PIPECAT_WEBSOCKET_ONLY_KEYS:
            if key in pipecat.model_fields_set:   # keys present in the raw YAML; defaults never warn
                warnings.append(f"pipecat.{key} is ignored in eval mode.")
    cfg.warnings = _dedupe(warnings)
    return cfg


def _check_simulator(cfg: ProjectConfig) -> None:
    mode = cfg.simulator.mode
    if mode == "audio_native":
        raise ConfigError("simulator.mode", "simulator.mode: audio_native is planned for v2. Use cascade.")
    if mode != "cascade":
        raise ConfigError("simulator.mode", f"simulator.mode: {mode} is not supported. Use cascade.")


def _check_agent(cfg: ProjectConfig) -> None:
    agent = cfg.agent
    if agent.system_prompt is not None:
        prompt_path = cfg.config_dir / agent.system_prompt
        resolved: Optional[Path] = None
        try:
            if prompt_path.is_file():
                resolved = prompt_path.resolve()
        except (OSError, ValueError):
            # Python 3.11/3.12 re-raise e.g. ENAMETOOLONG / WinError 206 when inline prompt text was
            # given instead of a path; that is the same config error as a missing file.
            resolved = None
        if resolved is None:
            raise ConfigError(
                "agent.system_prompt",
                f"system prompt file not found: {prompt_path} "
                "(agent.system_prompt is a file path, relative to the config file's folder)",
            )
        agent.system_prompt = str(resolved)

    if agent.adapter == "pipecat":
        pipecat = agent.pipecat
        if pipecat is None:
            raise ConfigError("agent.pipecat.url", "required when agent.adapter is pipecat")
        if pipecat.mode not in PIPECAT_MODES:
            raise ConfigError(
                "agent.pipecat.mode", f"pipecat.mode: {pipecat.mode} is planned for v2. Use eval or websocket."
            )
        if not pipecat.url:
            raise ConfigError("agent.pipecat.url", "required when agent.adapter is pipecat")
        if not pipecat.url.startswith(("ws://", "wss://")):
            raise ConfigError("agent.pipecat.url", "must start with ws:// or wss://")
        _check_turn_timeout(pipecat.turn_timeout_s, "agent.pipecat.turn_timeout_s")

    if agent.adapter == "livekit" and (agent.livekit is None or not agent.livekit.url):
        raise ConfigError("agent.livekit.url", "required when agent.adapter is livekit")
    if agent.adapter == "livekit":
        _check_turn_timeout(agent.livekit.turn_timeout_s, "agent.livekit.turn_timeout_s")

    if agent.adapter == "http":
        http = agent.http
        if http is None or not (http.url or "").strip():
            raise ConfigError("agent.http.url", "required when agent.adapter is http")
        for key in ("start_url", "close_url"):   # blank = unset, as the adapter reads it
            if getattr(http, key) is not None and not getattr(http, key).strip():
                setattr(http, key, None)
        for key in ("url", "start_url", "close_url"):
            value = getattr(http, key)
            if value is not None and not value.startswith(("http://", "https://")):
                raise ConfigError(f"agent.http.{key}", "must start with http:// or https://")
        for name, value in http.headers.items():
            # A CR / LF / non-ASCII value would make httpx raise an error that echoes it (B3/B4-rev R2, R6).
            if not is_printable_ascii(value):
                raise ConfigError(
                    f"agent.http.headers.{name}", "header values must be printable ASCII without control characters"
                )
            # Leading or trailing whitespace passes the rule above, but h11 refuses to send it and the httpx
            # error text echoes the value as a bytes repr (fixes round 4, M1); the value is never shown.
            if value != value.strip():
                raise ConfigError(f"agent.http.headers.{name}", "header values must not start or end with whitespace")
        _check_turn_timeout(http.turn_timeout_s, "agent.http.turn_timeout_s")

    # custom: the entrypoint is required here; its form and the class are checked when the adapter is
    # created (never imported by load_config).
    if agent.adapter == "custom" and not (agent.entrypoint or "").strip():
        raise ConfigError("agent.entrypoint", "required when agent.adapter is custom")


def _check_voice_config(cfg: ProjectConfig) -> None:
    voice = cfg.voice_config
    if voice is None:
        return
    for side in ("tts", "stt"):
        sub = getattr(voice, side)
        if sub.provider not in VOICE_PROVIDERS:
            raise ConfigError(
                f"voice_config.{side}.provider",
                f"unknown provider {sub.provider}. Use one of: {', '.join(VOICE_PROVIDERS)}.",
            )
        if sub.provider in _PROVIDERS_NEEDING_BASE_URL and not sub.base_url:
            raise ConfigError(f"voice_config.{side}.base_url", f"required when provider is {sub.provider}")
        if sub.provider in _PROVIDERS_NEEDING_MODEL and not (sub.model or "").strip():
            raise ConfigError(f"voice_config.{side}.model", f"required when provider is {sub.provider}")
    # local_light TTS: voice_config.tts.model is a folder holding the Kokoro files; a relative folder
    # is relative to the config file's folder, so it is resolved here (FIXES-ROUND-3-SPEC.md B1).
    tts = voice.tts
    if tts.provider == "local_light" and tts.model is not None and tts.model.strip():
        folder = Path(tts.model.strip()).expanduser()
        if not folder.is_absolute():
            tts.model = str((cfg.config_dir / folder).resolve())
    if voice.noise is not None:
        _check_noise(voice.noise, "voice_config.noise", cfg.config_dir)


def _check_noise(noise: NoiseSettings, field: str, config_dir: Path) -> None:
    """The noise simulation rules (SPEC.md section 8.2). A relative ``file`` is resolved against the
    config file's folder and stored absolute; the file must be a readable, non-silent WAV."""
    if noise.profile not in NOISE_PROFILES:
        raise ConfigError(
            f"{field}.profile", f"unknown noise profile {noise.profile}. Use one of: {', '.join(NOISE_PROFILES)}."
        )
    if noise.snr_db is None:
        raise ConfigError(f"{field}.snr_db", "required")
    if not NOISE_SNR_MIN_DB <= noise.snr_db <= NOISE_SNR_MAX_DB:   # also rejects NaN
        raise ConfigError(
            f"{field}.snr_db", f"snr_db {noise.snr_db:g} is outside {NOISE_SNR_MIN_DB:g}..{NOISE_SNR_MAX_DB:g}"
        )
    if noise.label is not None and not noise.label.strip():
        raise ConfigError(f"{field}.label", "label must not be blank")
    has_file = noise.file is not None and bool(noise.file.strip())
    if noise.profile != "file":
        if noise.file is not None:
            raise ConfigError(f"{field}.file", "only used when profile is file")
        return
    if not has_file:
        raise ConfigError(f"{field}.file", "required when profile is file")
    from impulse_evals.voice.noise import NoiseFileError, load_noise_file

    candidate = Path(noise.file.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    error: Optional[ConfigError] = None
    try:
        load_noise_file(candidate)
    except NoiseFileError as exc:
        error = ConfigError(f"{field}.file", str(exc))
    if error is not None:
        raise error
    noise.file = str(candidate.resolve())


def effective_noise(cfg: ProjectConfig, scenario: Scenario) -> Optional[NoiseSettings]:
    """The noise settings that apply to ``scenario`` (None = no noise): ``scenario.noise`` when it is a
    block, none when it is ``false``, else ``voice_config.noise``. Noise needs voice mode, so text mode
    (no ``voice_config``) always gives None."""
    if cfg.voice_config is None or scenario.noise is False:
        return None
    if isinstance(scenario.noise, NoiseSettings):
        return scenario.noise
    return cfg.voice_config.noise


def _check_scenarios(cfg: ProjectConfig) -> list[str]:
    """Validate every scenario; returns the unknown-domain warnings (domains rewritten to general)."""
    tool_names = None if cfg.agent.tools is None else {tool.name for tool in cfg.agent.tools}
    metric_ids = set(KNOWN_METRIC_IDS)
    assertion_ids = set(KNOWN_ASSERTION_IDS)
    seen_names: set[str] = set()
    warnings: list[str] = []

    for i, scenario in enumerate(cfg.scenarios):
        prefix = f"scenarios[{i}]"
        if not scenario.name.strip():   # before the duplicate rule: two blank names are blank, not duplicates
            raise ConfigError(f"{prefix}.name", "name must not be blank")
        if scenario.name in seen_names:
            raise ConfigError(f"{prefix}.name", f"duplicate scenario name {scenario.name}; names must be unique")
        seen_names.add(scenario.name)

        if not _is_valid_language(scenario.language):
            raise ConfigError(
                f"{prefix}.language",
                f"invalid language code {scenario.language}; it must match {LANGUAGE_PATTERN} "
                "(for example en-IN, hi-IN, hi-Latn-IN)",
            )

        if scenario.domain not in KNOWN_DOMAINS:
            warnings.append(f"Unknown domain {scenario.domain}; using general defaults.")
            scenario.domain = "general"

        for j, rule_id in enumerate(scenario.assertions):
            if "*" in rule_id:
                raise ConfigError(
                    f"{prefix}.assertions[{j}]",
                    f"wildcard {rule_id} is not accepted in config.yaml; list explicit pack.rule_id strings",
                )
            if rule_id not in assertion_ids:
                raise ConfigError(f"{prefix}.assertions[{j}]", f"unknown assertion {rule_id}")

        for j, metric_id in enumerate(scenario.metrics):
            if metric_id not in metric_ids:
                raise ConfigError(f"{prefix}.metrics[{j}]", f"unknown metric ID {metric_id}")

        if scenario.max_turns < 1:   # before the inject_events turns, which are checked against max_turns
            raise ConfigError(f"{prefix}.max_turns", f"max_turns {scenario.max_turns} must be at least 1")

        for k, event in enumerate(scenario.inject_events):
            if not 1 <= event.at_turn <= scenario.max_turns:
                raise ConfigError(
                    f"{prefix}.inject_events[{k}].at_turn",
                    f"at_turn {event.at_turn} is outside 1..{scenario.max_turns} (max_turns)",
                )

        expected_names: set[str] = set()
        for k, expected in enumerate(scenario.expected_tool_calls):
            entry = f"{prefix}.expected_tool_calls[{k}]"
            if tool_names is not None and expected.name not in tool_names:
                raise ConfigError(f"{entry}.name", f"tool {expected.name} is not in agent.tools")
            if expected.min_count < 0:
                raise ConfigError(f"{entry}.min_count", f"min_count {expected.min_count} must not be negative")
            if expected.min_count == 0 and expected.max_count is None:
                raise ConfigError(f"{entry}.min_count", "min_count: 0 is allowed only together with max_count")
            if expected.max_count is not None and expected.max_count < expected.min_count:
                raise ConfigError(
                    f"{entry}.max_count",
                    f"max_count {expected.max_count} is lower than min_count {expected.min_count}",
                )
            expected_names.add(expected.name)

        for k, name in enumerate(scenario.forbidden_tool_calls):
            entry = f"{prefix}.forbidden_tool_calls[{k}]"
            if tool_names is not None and name not in tool_names:
                raise ConfigError(entry, f"tool {name} is not in agent.tools")
            if name in expected_names:
                raise ConfigError(entry, f"tool {name} is both expected and forbidden")

        if isinstance(scenario.noise, NoiseSettings):
            _check_noise(scenario.noise, f"{prefix}.noise", cfg.config_dir)

    return warnings


def check_scenarios(scenarios: list[Scenario]) -> list[str]:
    """Check a scenario list with the ``config.yaml`` scenario rules (SPEC.md section 8.2) for a project
    without ``agent.tools``: non-blank unique names, language codes, explicit known assertion and metric
    IDs, ``max_turns`` of at least 1, ``inject_events`` turns, and the ``expected_tool_calls`` /
    ``forbidden_tool_calls`` rules.

    Raises ConfigError at ``scenarios[i]....`` for the first problem; returns the unknown-domain
    warnings (such scenarios are rewritten to ``general``, as by ``load_config``). Used by
    ``impulse generate`` so a generated file always loads under ``scenarios:``.
    """
    cfg = ProjectConfig(simulator=SimulatorConfig(), config_dir=Path.cwd(), scenarios=list(scenarios))
    return _check_scenarios(cfg)


def read_system_prompt(cfg: ProjectConfig) -> Optional[str]:
    """The text of ``agent.system_prompt`` (None when not set). The file is sent verbatim."""
    if cfg.agent.system_prompt is None:
        return None
    return _read_text(cfg.config_dir / cfg.agent.system_prompt)


# ------------------------------------------------------------------------------------------------
# .impulse/config.toml
# ------------------------------------------------------------------------------------------------


def load_toml(path: str | os.PathLike[str]) -> TomlConfig:
    """Load ``.impulse/config.toml`` (SPEC.md section 8.1). Keys are held as SecretStr."""
    path = Path(path)
    text = _read_text(path)
    error: Optional[ConfigError] = None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        error = ConfigError(str(path), f"invalid TOML: {exc}")
    if error is not None:
        raise error
    cfg: TomlConfig = _validate(TomlConfig, data)
    language = cfg.defaults.language
    if language is not None and language != LANGUAGE_PLACEHOLDER and not _is_valid_language(language):
        raise ConfigError(
            "defaults.language",
            f"invalid language code {language}; it must match {LANGUAGE_PATTERN} (for example en-IN, hi-IN)",
        )
    return cfg


def load_project_toml(base_dir: str | os.PathLike[str] | None = None) -> Optional[TomlConfig]:
    """Load ``.impulse/config.toml`` under ``base_dir`` (default: the current directory), or None."""
    path = Path(base_dir) / DEFAULT_TOML_PATH if base_dir is not None else DEFAULT_TOML_PATH
    return load_toml(path) if path.is_file() else None


# ------------------------------------------------------------------------------------------------
# Precedence (SPEC.md section 8.1)
# ------------------------------------------------------------------------------------------------


def is_placeholder(value: Optional[str]) -> bool:
    """True for None, blank, or a ``YOUR_..._MODEL`` placeholder."""
    return value is None or not value.strip() or _PLACEHOLDER_RE.fullmatch(value.strip()) is not None


def _check_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"unknown model role {role!r}; expected one of {', '.join(ROLES)}")


def resolve_model(cfg: ProjectConfig, role: str, toml: Optional[TomlConfig] = None) -> str:
    """Model string for ``role`` (primary, fast, simulator): config.yaml, then IMPULSE_MODEL_*, then config.toml."""
    _check_role(role)
    yaml_value = {
        "primary": cfg.judge.primary_model,
        "fast": cfg.judge.fast_model,
        "simulator": cfg.simulator.model,
    }[role]
    if not is_placeholder(yaml_value):
        return yaml_value.strip()
    env_value = os.environ.get(MODEL_ENV_VARS[role])
    if not is_placeholder(env_value):
        return env_value.strip()
    toml_value = getattr(toml.models, role) if toml is not None else None
    if not is_placeholder(toml_value):
        return toml_value.strip()
    raise ConfigError(
        _MODEL_FIELDS[role],
        f"no model configured for the {role} role. Set {_MODEL_FIELDS[role]} in config.yaml, "
        f"the environment variable {MODEL_ENV_VARS[role]}, or [models] {role} in .impulse/config.toml.",
    )


def resolve_model_options(cfg: ProjectConfig, role: str, toml: Optional[TomlConfig] = None) -> dict[str, Any]:
    """Options for ``role``: config.toml [model_options.<role>] first, then config.yaml options key by key.

    Empty TOML values are left out, so ``"api_base" in options`` means an api_base is configured. A
    config.yaml value that is None or ``""`` is left out the same way (it never masks the config.toml
    value; AUDIT.md finding 44); ``0`` and ``False`` are values and override.
    """
    _check_role(role)
    merged: dict[str, Any] = {}
    if toml is not None:
        toml_options: TomlRoleOptions = getattr(toml.model_options, role)
        if toml_options.api_base:
            merged["api_base"] = toml_options.api_base
        if toml_options.api_key_env:
            merged["api_key_env"] = toml_options.api_key_env
        if toml_options.extra_params:
            merged["extra_params"] = dict(toml_options.extra_params)
    yaml_options = {
        "primary": cfg.judge.primary_options,
        "fast": cfg.judge.fast_options,
        "simulator": cfg.simulator.options,
    }[role]
    merged.update({key: value for key, value in (yaml_options or {}).items() if not _is_blank_option(value)})
    return merged


def _is_blank_option(value: Any) -> bool:
    """None or the empty string: an option value that counts as unset (``0`` and ``False`` do not)."""
    return value is None or (isinstance(value, str) and value == "")


def role_custom_llm_provider(options: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The ``custom_llm_provider`` a role's merged options pass to LiteLLM, or None: the top-level
    option, else the one inside ``extra_params`` (config.toml ``[model_options.<role>]`` or config.yaml),
    the same precedence ``completion_kwargs`` uses (top-level keys win). Blank values count as unset."""
    options = options or {}
    value = options.get("custom_llm_provider")
    if not (isinstance(value, str) and value.strip()):
        extra = options.get("extra_params")
        value = extra.get("custom_llm_provider") if isinstance(extra, Mapping) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def model_needs_key(model: str, options: Optional[Mapping[str, Any]] = None) -> bool:
    """Local models (``ollama/``, ``ollama_chat/``) and any role with ``api_base`` need no key."""
    if model.startswith(_LOCAL_MODEL_PREFIXES):
        return False
    return not (options or {}).get("api_base")


def resolve_key(name: str, toml: Optional[TomlConfig] = None) -> Optional[str]:
    """Value of key ``name``: the environment variable first, then config.toml [keys]; None when unset."""
    value = os.environ.get(name)
    if value:
        return value
    if toml is not None:
        secret = toml.keys.get(name)
        if secret is not None and secret.get_secret_value():
            return secret.get_secret_value()
    return None


def require_key(name: str, toml: Optional[TomlConfig] = None) -> str:
    """Like resolve_key, but a missing key is a ConfigError naming the key (never a value)."""
    value = resolve_key(name, toml)
    if value is None:
        raise ConfigError(
            f"keys.{name}",
            f"{name} is not set. Set the environment variable {name} or add it to [keys] in .impulse/config.toml.",
        )
    return value


class _NoEnvironmentOs:
    """The ``os`` module as ``_litellm_key_names`` shows it to LiteLLM's ``validate_environment``: an
    empty, read-only environment (every other attribute is the real ``os``). Nothing is mutated."""

    environ: Mapping[str, str] = types.MappingProxyType({})

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


def _litellm_key_names(model: str) -> list[str]:
    """The variables LiteLLM's own offline mapping (``litellm.validate_environment``) names for
    ``model``, as if none of them were set. ``validate_environment`` lists only the MISSING ones, so
    it runs as a copy of the function whose ``os`` sees an empty environment: ``os.environ`` itself is
    never changed (no other code can observe this lookup). No network."""
    import litellm  # lazy: heavy import, only needed when a run resolves keys

    litellm.suppress_debug_info = True   # no "Give Feedback / Get Help" banner (AUDIT.md finding 43)
    validate = litellm.validate_environment
    code = getattr(validate, "__code__", None)
    if code is not None:
        environless = types.FunctionType(
            code,
            {**validate.__globals__, "os": _NoEnvironmentOs()},
            validate.__name__,
            validate.__defaults__,
            validate.__closure__,
        )
        environless.__kwdefaults__ = validate.__kwdefaults__
        validate = environless
    with contextlib.redirect_stdout(io.StringIO()):
        result = validate(model=model)
    return [str(name) for name in (result or {}).get("missing_keys") or []]


# LiteLLM provider aliases: a provider name that only selects another API of a provider reads that
# provider's key, or uses its credentials (checked against LiteLLM 1.105.1: LiteLLM never reads the
# alias's own *_API_KEY).
PROVIDER_KEY_ALIASES = {
    "text-completion-openai": "openai",
    "aiohttp_openai": "openai",
    "anthropic_text": "anthropic",
    "azure_text": "azure",
    "cohere_chat": "cohere",
    "watsonx_text": "watsonx",
    "vertex_ai_beta": "vertex_ai",      # Google Cloud credentials: no Impulse key
    "sagemaker_chat": "sagemaker",      # AWS credentials: no Impulse key
    "sagemaker_nova": "sagemaker",
}
# Providers whose key variable LiteLLM's tables do not name (checked against LiteLLM 1.105.1 by the
# request it builds): the variable LiteLLM reads, or None for request signing with the provider's
# own credentials (behaves like bedrock: no Impulse key).
PROVIDER_KEY_NAMES: dict[str, Optional[str]] = {
    "meta_llama": "LLAMA_API_KEY",
    "lambda_ai": "LAMBDA_API_KEY",
    "snowflake": "SNOWFLAKE_JWT",
    "oci": None,                        # OCI_USER / OCI_FINGERPRINT / OCI_TENANCY / OCI_KEY signing
}
# Fallback for LiteLLM's PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO (litellm.constants): resolving
# these providers through LiteLLM runs an OAuth device-code login.
_AUTHENTICATING_PROVIDERS_FALLBACK = frozenset({"github_copilot", "chatgpt"})
_INVALID_ENV_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")


def _authenticating_providers() -> frozenset[str]:
    try:
        from litellm.constants import PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO as names
    except Exception:  # noqa: BLE001 - an older or newer LiteLLM without the constant
        return _AUTHENTICATING_PROVIDERS_FALLBACK
    try:
        return frozenset(str(name) for name in names)
    except TypeError:
        return _AUTHENTICATING_PROVIDERS_FALLBACK


def authenticating_provider(model: str, custom_llm_provider: Optional[str] = None) -> Optional[str]:
    """The provider ``model`` (or ``custom_llm_provider``) names when LiteLLM authenticates while it
    resolves that provider (``litellm.constants.PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO``:
    github_copilot, chatgpt; an OAuth device-code login that contacts the provider and waits for a
    human), else None. Read from the model string's prefix only, so nothing is resolved: Impulse
    never passes such a model to ``litellm.get_llm_provider`` or ``validate_environment``."""
    declared = custom_llm_provider or (model.split("/", 1)[0] if isinstance(model, str) and "/" in model else None)
    if not declared:
        return None
    return declared if declared in _authenticating_providers() else None


def _openai_like_key_name(provider: str) -> Optional[str]:
    """The ``api_key_env`` LiteLLM's openai_like provider registry
    (``litellm/llms/openai_like/providers.json``, through LiteLLM's own ``JSONProviderRegistry``; a
    local file, no network) names for ``provider``, or None."""
    try:
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        JSONProviderRegistry.load()
        config = JSONProviderRegistry.get(provider)
    except Exception:  # noqa: BLE001 - a LiteLLM without the registry: no name from it
        return None
    name = getattr(config, "api_key_env", None) if config is not None else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def _fallback_key_name(provider: str) -> str:
    """``{PROVIDER}_API_KEY`` as a valid environment variable name (characters other than
    ``[A-Za-z0-9_]`` become ``_``, then upper case: ``nano-gpt`` -> ``NANO_GPT_API_KEY``)."""
    return _INVALID_ENV_NAME_CHARS.sub("_", str(provider)).upper() + "_API_KEY"


def key_name_for_model(
    model: str, field: str = "models", custom_llm_provider: Optional[str] = None
) -> Optional[str]:
    """Name of the key variable a hosted model needs, or None when it needs no Impulse key.

    ``custom_llm_provider`` is the role's option of that name (``role_custom_llm_provider``; maintainer
    decision 2026-10-20): LiteLLM then resolves the provider from it, and the key table is read for
    ``{provider}/{model}``, so ``llama-3.1-8b-instant`` with ``custom_llm_provider: groq`` needs
    ``GROQ_API_KEY``. Without it, the model string alone decides.

    In order:

    - a provider LiteLLM authenticates while resolving it (``authenticating_provider``:
      ``github_copilot/``, ``chatgpt/``, as the prefix or as ``custom_llm_provider``): None, without
      asking LiteLLM anything (no login, no network);
    - the provider is resolved with ``litellm.get_llm_provider`` (offline; with ``custom_llm_provider``
      when given); an unknown provider raises ``UnknownProviderError``;
    - a provider alias (``PROVIDER_KEY_ALIASES``: ``text-completion-openai``, ``aiohttp_openai``,
      ``anthropic_text``, ``azure_text``, ``cohere_chat``, ``watsonx_text``, ``vertex_ai_beta``,
      ``sagemaker_chat``, ``sagemaker_nova``) is looked up as the provider it aliases;
    - a provider in ``PROVIDER_KEY_NAMES`` (``meta_llama`` -> ``LLAMA_API_KEY``, ``lambda_ai`` ->
      ``LAMBDA_API_KEY``, ``snowflake`` -> ``SNOWFLAKE_JWT``; ``oci`` -> None, request signing): that entry;
    - LiteLLM's openai_like registry names an ``api_key_env`` for the provider: that name
      (``nano-gpt`` -> ``NANOGPT_API_KEY``, ``scaleway`` -> ``SCW_SECRET_KEY``, ``poe`` -> ``POE_API_KEY``);
    - LiteLLM's offline mapping (``litellm.validate_environment``, independent of whether the
      variables are set) lists ``*_API_KEY`` names: the one LiteLLM itself reads; when several are
      listed, one in ``STANDARD_KEY_NAMES`` wins (gemini: ``GEMINI_API_KEY``, not ``GOOGLE_API_KEY``),
      else the first (``together_ai`` -> ``TOGETHERAI_API_KEY``, ``perplexity`` -> ``PERPLEXITYAI_API_KEY``);
    - the mapping lists credentials, none of them a ``*_API_KEY`` name (``bedrock/``,
      ``sagemaker/``, ``vertex_ai/``, ``ollama/``, ``datarobot/``): None, LiteLLM uses the
      provider's own credentials;
    - the mapping lists nothing for the provider (``azure_ai/``, ``sambanova/``, ``watsonx/``,
      ``databricks/``, ``hosted_vllm/``, ...): ``{PROVIDER}_API_KEY`` (``_fallback_key_name``: always
      a valid environment variable name; for azure_ai and sambanova exactly what LiteLLM reads). A
      known limitation: for other rare providers whose variable no LiteLLM table names this can be
      a name LiteLLM does not read; the role's ``api_key_env`` overrides it.

    ``field`` is the config path the model string came from, named by the error for an unknown
    provider (``api_key_for_role`` passes ``judge.primary_model``, ``judge.fast_model`` or
    ``simulator.model``; the default is the config.toml ``[models]`` section).
    """
    custom_llm_provider = custom_llm_provider or None
    if authenticating_provider(model, custom_llm_provider) is not None:
        return None

    import litellm  # lazy: heavy import, only needed when a run resolves keys

    litellm.suppress_debug_info = True   # no "Give Feedback / Get Help" banner (AUDIT.md finding 43)
    error: Optional[ConfigError] = None
    try:
        # LiteLLM prints a provider-list banner to stdout for an unknown provider; this is only a lookup.
        with contextlib.redirect_stdout(io.StringIO()):
            model_name, provider = litellm.get_llm_provider(model, custom_llm_provider=custom_llm_provider)[:2]
    except Exception:  # noqa: BLE001 - LiteLLM raises several types for unknown models
        error = UnknownProviderError(field, f"cannot determine the provider of model {model}")
    if error is not None:
        raise error
    provider = str(provider)
    # With custom_llm_provider the model string carries no provider prefix of its own, so LiteLLM's
    # key table is read for "{provider}/{model}" (the bare model string would name no variable).
    lookup_model = f"{provider}/{model_name}" if custom_llm_provider else model
    if provider in PROVIDER_KEY_ALIASES:
        provider = PROVIDER_KEY_ALIASES[provider]
        lookup_model = f"{provider}/{model_name}"
    if provider in PROVIDER_KEY_NAMES:   # None: the provider's own credentials, like bedrock
        return PROVIDER_KEY_NAMES[provider]
    registry_name = _openai_like_key_name(provider)
    if registry_name is not None:
        return registry_name
    listed = _litellm_key_names(lookup_model)
    if not listed:   # LiteLLM's table does not know the provider: the old rule, as a valid name
        return _fallback_key_name(provider)
    names = [name for name in listed if name.endswith("_API_KEY")]
    if not names:    # credentials only (AWS, Google Cloud, a server address): no Impulse key
        return None
    return next((name for name in names if name in STANDARD_KEY_NAMES), names[0])


def api_key_for_role(cfg: ProjectConfig, role: str, toml: Optional[TomlConfig] = None) -> Optional[str]:
    """The key to pass for ``role``, or None when the role needs no key (SPEC.md section 8.1).

    ``api_key_env`` in the role's options names the variable to use; otherwise ``key_name_for_model``
    (with the role's ``custom_llm_provider``, read from its merged options at the top level or inside
    ``extra_params``). Local models and roles with ``api_base`` never require a key, and neither does a
    provider whose LiteLLM table lists only credentials that are not ``*_API_KEY`` variables (bedrock,
    vertex_ai, ...).
    """
    model = resolve_model(cfg, role, toml)
    options = resolve_model_options(cfg, role, toml)
    env_name = options.get("api_key_env") or None
    if not model_needs_key(model, options):
        return resolve_key(env_name, toml) if env_name else None
    if env_name:
        return require_key(env_name, toml)
    name = key_name_for_model(model, _MODEL_FIELDS[role], role_custom_llm_provider(options))
    return require_key(name, toml) if name is not None else None


# ------------------------------------------------------------------------------------------------
# Redaction
# ------------------------------------------------------------------------------------------------

REDACTED = "[REDACTED]"
_MIN_SECRET_LENGTH = 4
_API_KEY_SUFFIX = "_API_KEY"
# Masked and encoded forms of a known value (AUDIT.md finding 30). A provider-masked form is a prefix
# of the value (3+ characters), a run of 3 or more "*", and a suffix of it (2+ characters), as in
# "Incorrect API key provided: sk-test-************cdef"; only values of 12+ characters have one. A
# base64 form shorter than 8 characters is not redacted: so short a run of base64 characters would
# match unrelated text too often (only a value of 6 bytes or fewer has such a form).
_MASKED_MIN_VALUE_LENGTH = 12
_MASKED_MIN_PREFIX = 3
_MASKED_MIN_SUFFIX = 2
_MASK_RUN_RE = re.compile(r"\*{3,}")
_MIN_BASE64_FORM_LENGTH = 8


def secret_values(
    toml: Optional[TomlConfig] = None, extra_names: Iterable[str] = (), extra_values: Iterable[str] = ()
) -> list[str]:
    """Every known key value: config.toml [keys], the env vars named there, standard, or extra, the
    value of every environment variable whose name ends in ``_API_KEY`` (a provider's own key such as
    ``GROQ_API_KEY``, which a run may resolve although no config entry names it; AUDIT.md finding 31),
    the literal credentials in config.toml ``[model_options.<role>]`` (``option_secrets``), and
    ``extra_values`` (key values the caller already holds); longest first. Values of 4+ characters,
    except a value found only through the ``*_API_KEY`` name rule, which needs 12+ characters
    (``_wildcard_values``: a placeholder such as ``MAPS_API_KEY=book`` is no key)."""
    names = set(STANDARD_KEY_NAMES) | {name for name in extra_names if name}
    values: set[str] = {value for value in extra_values if isinstance(value, str)}
    if toml is not None:
        names |= set(toml.keys)
        values |= {secret.get_secret_value() for secret in toml.keys.values()}
        for role in ROLES:
            values |= set(option_secrets(getattr(toml.model_options, role).model_dump()))
    values |= {os.environ.get(name, "") for name in names}
    values = {value for value in values if len(value) >= _MIN_SECRET_LENGTH}
    values |= _wildcard_values((_API_KEY_SUFFIX,))
    return sorted(values, key=len, reverse=True)


# A value found only through a name ending (``*_API_KEY``; in a dry run also ``*_SECRET_KEY`` and
# ``*_JWT``) is a key only when it is this long; a variable named explicitly keeps the 4-character rule.
_MIN_WILDCARD_SECRET_LENGTH = 12


def _wildcard_values(suffixes: tuple[str, ...]) -> set[str]:
    """The values (12+ characters) of the environment variables whose names end in one of ``suffixes``
    (compared in upper case)."""
    return {
        value
        for name, value in os.environ.items()
        if name.upper().endswith(suffixes) and len(value) >= _MIN_WILDCARD_SECRET_LENGTH
    }


# Name endings of key variables a dry run redacts without resolving any key (AUDIT.md finding 31).
STATIC_KEY_SUFFIXES = ("_API_KEY", "_SECRET_KEY", "_JWT")


def static_key_names() -> list[str]:
    """The key variable names in Impulse's own tables, known without LiteLLM: ``STANDARD_KEY_NAMES``,
    the ``{PROVIDER}_API_KEY`` of each ``PROVIDER_KEY_ALIASES`` target, and the ``PROVIDER_KEY_NAMES``
    entries (``LLAMA_API_KEY``, ``LAMBDA_API_KEY``, ``SNOWFLAKE_JWT``)."""
    names = list(STANDARD_KEY_NAMES)
    names += [_fallback_key_name(target) for target in PROVIDER_KEY_ALIASES.values()]
    names += [name for name in PROVIDER_KEY_NAMES.values() if name]
    return list(dict.fromkeys(names))


def static_key_values() -> list[str]:
    """The environment values of ``static_key_names()`` (4+ characters) and of every variable whose
    name ends in ``_API_KEY``, ``_SECRET_KEY`` or ``_JWT`` (``SCW_SECRET_KEY``; 12+ characters when the
    name rule alone finds it, ``_wildcard_values``), longest first. A dry run resolves no key (it never
    imports LiteLLM), so its redactor, and the CLI's for a dry run, know these instead of the keys
    resolved for the roles."""
    values = {os.environ.get(name, "") for name in static_key_names()}
    values = {value for value in values if len(value) >= _MIN_SECRET_LENGTH}
    values |= _wildcard_values(STATIC_KEY_SUFFIXES)
    return sorted(values, key=len, reverse=True)


def redact(
    text: str,
    toml: Optional[TomlConfig] = None,
    extra_names: Iterable[str] = (),
    extra_values: Iterable[str] = (),
) -> str:
    """Replace every known key value in ``text`` with ``[REDACTED]`` (for logs, reports, errors).

    For each value (longest first): the value itself; its URL-encoded form
    (``urllib.parse.quote(value, safe="")``, when it differs); its base64 forms (``_base64_forms``: the
    characters of the base64 text that depend on the value's bytes alone, at each of the three byte
    alignments, so ``Basic base64(user:key)`` keeps the user part); and its provider-masked forms
    (``_redact_masked``). Nothing else in the text changes."""
    for value in secret_values(toml, extra_names, extra_values):
        text = text.replace(value, REDACTED)
        for form in _encoded_forms(value):
            text = text.replace(form, REDACTED)
        text = _redact_masked(text, value)
    return text


def _base64_forms(value: str) -> list[str]:
    """The base64 text of ``value`` (UTF-8) starting 0, 1 or 2 bytes into a 3-byte group, each cut to
    the characters that depend on the value's bytes alone (those that also hold bits of the bytes
    around it differ with the surrounding text), in the standard alphabet and, where it differs, the
    URL-safe one (``-`` and ``_`` for ``+`` and ``/``: pydantic writes bytes that are not UTF-8 that
    way); forms shorter than 8 characters are left out."""
    raw = value.encode("utf-8")
    forms: list[str] = []
    for offset in range(3):
        encoded = base64.b64encode(b"\0" * offset + raw).decode("ascii")
        start = -(-8 * offset // 6)              # the first character with no bit of the padding bytes
        end = 8 * (offset + len(raw)) // 6       # the characters made of the value's bits only
        form = encoded[start:end]
        if len(form) >= _MIN_BASE64_FORM_LENGTH:
            forms.append(form)
            url_safe = form.replace("+", "-").replace("/", "_")   # base64.urlsafe_b64encode's alphabet
            if url_safe != form:
                forms.append(url_safe)
    return forms


def _encoded_forms(value: str) -> list[str]:
    """The URL-encoded form of ``value`` (when it differs from the value), its JSON-escaped form
    (``json.dumps(value)[1:-1]``: a value with ``"`` or ``\\`` echoed inside a JSON body; fixes round 4,
    M1), its Python repr form (``repr(value)[1:-1]``: a str or bytes repr in an exception text, such as
    h11's "Illegal header value b'...'"), and its base64 forms."""
    quoted = quote(value, safe="")
    forms = [quoted] if quoted != value else []
    forms += [json.dumps(value)[1:-1], repr(value)[1:-1]]
    forms += _base64_forms(value)
    return [form for form in dict.fromkeys(forms) if form != value]


def _redact_masked(text: str, value: str) -> str:
    """``text`` with every provider-masked form of ``value`` replaced by ``[REDACTED]``: a prefix of the
    value (at least 3 characters) right before a run of 3 or more ``*`` and a suffix of the value (at
    least 2 characters) right after it, each taken as long as it matches (e.g.
    ``sk-test-************cdef`` for ``sk-test-0123456789abcdef``). Only for values of 12 or more
    characters; a run of stars without such a prefix and suffix is left alone.

    Fast on long texts with many star runs: nothing to do when the value's first 3 or last 2
    characters are not in the text at all, and per run only the prefix (suffix) lengths whose first 3
    (last 2) characters match are compared (``str.find`` / ``str.rfind``)."""
    if len(value) < _MASKED_MIN_VALUE_LENGTH or "***" not in text:
        return text
    if value[:_MASKED_MIN_PREFIX] not in text or value[-_MASKED_MIN_SUFFIX:] not in text:
        return text
    pieces: list[str] = []
    last = 0   # end of the text already copied or replaced
    for match in _MASK_RUN_RE.finditer(text):
        start, end = match.span()
        if start < last:
            continue
        prefix = _masked_prefix_length(text, start, value, last)
        if not prefix:
            continue
        suffix = _masked_suffix_length(text, end, value)
        if suffix:
            pieces += [text[last:start - prefix], REDACTED]
            last = end + suffix
    if not pieces:
        return text
    pieces.append(text[last:])
    return "".join(pieces)


def _masked_prefix_length(text: str, start: int, value: str, floor: int) -> int:
    """The longest ``k`` (3 <= k <= len(value) - 1, not reaching before ``floor``) with
    ``text[start - k:start] == value[:k]``, else 0. Only the positions where the value's first 3
    characters occur are tried, leftmost (longest) first."""
    head = value[:_MASKED_MIN_PREFIX]
    position = text.find(head, max(floor, start - (len(value) - 1)), start)
    while position != -1:
        if text[position:start] == value[:start - position]:
            return start - position
        position = text.find(head, position + 1, start)
    return 0


def _masked_suffix_length(text: str, end: int, value: str) -> int:
    """The longest ``k`` (2 <= k <= len(value) - 1) with ``text[end:end + k] == value[-k:]``, else 0.
    Only the positions where the value's last 2 characters occur are tried, rightmost (longest) first."""
    tail = value[-_MASKED_MIN_SUFFIX:]
    position = text.rfind(tail, end, min(len(text), end + len(value) - 1))
    while position != -1:
        length = position + len(tail) - end
        if text[end:end + length] == value[-length:]:
            return length
        position = text.rfind(tail, end, position + len(tail) - 1)
    return 0


# Query parameters that carry a credential in a URL (compared case-insensitively).
URL_CREDENTIAL_QUERY_KEYS = (
    "token", "key", "api_key", "api-key", "access_token",
    # fixes round 4 (L8)
    "apikey", "api_token", "sig", "signature", "secret", "password", "auth",
)


def _split_url_credentials(url: str) -> tuple[str, list[str]]:
    """``(url without credentials, the removed values)``; ``url`` itself when it holds none."""
    try:
        parts = urlsplit(url)
    except ValueError:   # not a URL that can be parsed (e.g. "http://u:p@[::1/x"): userinfo only
        return _split_unparsed_userinfo(url)
    removed: list[str] = []
    userinfo, at, host = parts.netloc.rpartition("@")
    if at:
        user, _, password = userinfo.partition(":")
        # The password is the credential; without one the user name is (``wss://TOKEN@host``). A user
        # name next to a password (often a common word such as "user") is never a secret.
        removed.append(password or user)
    kept: list[str] = []
    dropped = False
    for item in parts.query.split("&") if parts.query else []:
        name, _, value = item.partition("=")
        if unquote_plus(name).strip().lower() in URL_CREDENTIAL_QUERY_KEYS:
            dropped = True
            removed.append(value)
        else:
            kept.append(item)
    if not at and not dropped:
        return url, []
    stripped = urlunsplit((parts.scheme, host if at else parts.netloc, parts.path, "&".join(kept), parts.fragment))
    # Each value as written in the URL and decoded (a server error may echo either form).
    values = [form for value in removed if value for form in (value, unquote_plus(value))]
    return stripped, list(dict.fromkeys(values))


def _split_unparsed_userinfo(url: str) -> tuple[str, list[str]]:
    """For a URL ``urlsplit`` rejects: the authority is the text between ``://`` and the next ``/``,
    ``?`` or ``#`` (or the end); a ``user:password@`` in it is removed and its credential returned (the
    password; the user name only when there is none). Anything else is returned unchanged."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url, []
    end = min((i for i in (rest.find("/"), rest.find("?"), rest.find("#")) if i != -1), default=len(rest))
    userinfo, at, host = rest[:end].rpartition("@")
    if not at:
        return url, []
    user, _, password = userinfo.partition(":")
    value = password or user
    values = [form for form in (value, unquote_plus(value)) if form] if value else []
    return f"{scheme}://{host}{rest[end:]}", list(dict.fromkeys(values))


def strip_url_credentials(url: str) -> str:
    """``url`` without userinfo (``user:password@``) and without the query parameters named in
    ``URL_CREDENTIAL_QUERY_KEYS`` (other parameters are kept, in order). A URL without credentials is
    returned unchanged. Used wherever a configured URL is stored or shown (SPEC.md sections 7 and 9);
    the connection itself still uses the full URL."""
    return _split_url_credentials(url)[0]


def url_credentials(url: Optional[str]) -> list[str]:
    """The credentials in ``url``, for redaction: the userinfo password (the user name only when there
    is no password) and each query value ``strip_url_credentials`` removes, as written and decoded;
    ``[]`` for None or a URL without any."""
    if not isinstance(url, str) or not url:
        return []
    return _split_url_credentials(url)[1]


# Option keys whose string values are not secrets: URLs (only their credentials are), the model,
# variable NAMES (errors name variables), and LiteLLM provider settings that are not credentials.
URL_OPTION_KEYS = ("api_base", "base_url")
PLAIN_OPTION_KEYS = (
    "model",
    "api_key_env",
    "custom_llm_provider",
    "api_version",
    "region_name",
    "aws_region_name",
    "vertex_project",
    "vertex_location",
)
_AUTH_SCHEME_RE = re.compile(r"(?:bearer|basic|token)\s+(\S.*)", re.IGNORECASE | re.DOTALL)


def secret_text(value: str) -> str:
    """A literal credential as it is redacted: the token of ``Bearer|Basic|Token <token>`` (an echo
    keeps ``Bearer [REDACTED]``), otherwise the whole value."""
    match = _AUTH_SCHEME_RE.fullmatch(value.strip())
    return match.group(1).strip() if match else value


def option_secrets(options: Any, key: Optional[str] = None) -> list[str]:
    """Every string leaf of a role's options (nested mappings and lists such as ``extra_params``,
    ``extra_headers``, ``extra_body``) as a literal secret (``secret_text``), except the
    ``PLAIN_OPTION_KEYS`` and, for the ``URL_OPTION_KEYS``, everything but the URL's credentials.
    Numbers and booleans never. Unknown keys are secrets (fail closed)."""
    if isinstance(options, Mapping):
        return [value for name, item in options.items() for value in option_secrets(item, str(name))]
    if isinstance(options, (list, tuple)):
        return [value for item in options for value in option_secrets(item, key)]
    if not isinstance(options, str):
        return []
    if key in URL_OPTION_KEYS:
        return url_credentials(options)
    if key in PLAIN_OPTION_KEYS:
        return []
    return [secret_text(options)]
