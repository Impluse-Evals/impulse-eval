# SPDX-License-Identifier: Apache-2.0
"""Generic adapter (SPEC.md sections 8.2 and 9): the PRIMARY adapter.

``agent.entrypoint = "module:function"`` names a Python function with the signature
``(user_text: str, history: list[dict]) -> str | dict``, sync or async.

- ``history`` is ``[{"role": "user" | "assistant", "content": str}, ...]`` with the prior turns of
  this conversation only (the current ``user_text`` is not included). Every ``start()`` begins a new
  conversation with an empty history.
- ``send(None, None)`` (outbound opener) calls the function with ``user_text=""`` and adds no user
  entry to the history.
- A returned dict may contain ``text``, ``audio`` (bytes), ``heard_text``, and ``tool_calls`` (a list
  of dicts with the keys ``name``, ``arguments``, ``result``, ``success``, ``error``; a missing
  ``arguments`` becomes ``{}``). A call without a non-empty string ``name`` (an OpenAI-style call too),
  with an ``arguments`` that is not a dict (a JSON string is not parsed), or with any other invalid
  value is a ``TypeError("agent function ... returned ...")``, like every other malformed return.
- Voice mode: the function receives the simulator's text. A function that declares a parameter named
  exactly ``user_audio`` (positional-or-keyword or keyword-only; ``**kwargs`` does not count) also
  gets the caller's audio as WAV bytes (``None`` in text mode and for the outbound opener): with noise
  simulation, the noisy audio. ``accepts_user_audio`` says which (None before the first ``start()``).
- Latency is measured by Impulse around the call, with ``time.perf_counter``.
- System prompt: the ``agent.system_prompt`` file from config.yaml always wins; otherwise the
  module's ``SYSTEM_PROMPT`` constant, otherwise its ``get_system_prompt()`` function, otherwise None.

The entrypoint is imported when the adapter starts (never by ``load_config``), with the config
file's folder on ``sys.path``; any import failure is a ``ConfigError`` at ``agent.entrypoint``.

Shared with the other adapters (FIXES-ROUND-3-SPEC.md B3, B4), so the rules and texts cannot drift:
``split_entrypoint``, ``import_entrypoint_module`` and ``resolve_attribute`` (the custom adapter's
``module:Class`` loading), and ``to_tool_calls`` (the HTTP adapter's answer, with the source label
"HTTP agent" in place of "agent function {entrypoint}").
"""

from __future__ import annotations

import copy
import importlib
import inspect
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from pydantic import ValidationError

from impulse_evals.adapters.base import AgentInterface, AgentReply
from impulse_evals.config import ConfigError, ProjectConfig, read_system_prompt
from impulse_evals.models import Scenario, ToolCall

__all__ = [
    "ENTRYPOINT_FIELD",
    "GenericAdapter",
    "import_entrypoint_module",
    "resolve_attribute",
    "split_entrypoint",
    "to_tool_calls",
]

ENTRYPOINT_FIELD = "agent.entrypoint"
_TOOL_CALL_KEYS = ("name", "arguments", "result", "success", "error")


class GenericAdapter(AgentInterface):
    """Calls the user's ``module:function`` entrypoint in process, one call per agent turn."""

    def __init__(self, config: ProjectConfig) -> None:
        self._config = config
        self._entrypoint: Optional[str] = config.agent.entrypoint
        self._module: Optional[ModuleType] = None
        self._function: Optional[Callable[..., Any]] = None
        self._history: list[dict[str, str]] = []
        self._config_prompt: Optional[str] = None
        self._config_prompt_loaded = False
        self._agent_prompt: Optional[str] = None
        self.accepts_user_audio: Optional[bool] = None   # the function declares ``user_audio``

    # -- lifecycle ---------------------------------------------------------------------------------

    async def start(self, scenario: Scenario) -> None:
        """Import the entrypoint (first start only) and begin a new conversation."""
        if self._function is None:
            self._module, self._function = self._load_entrypoint()
            self.accepts_user_audio = _declares_user_audio(self._function)
        self._history = []
        if self._config.agent.system_prompt is None:
            self._agent_prompt = await self._module_system_prompt()

    async def send(self, user_text: Optional[str], user_audio: Optional[bytes]) -> AgentReply:
        """Call the agent function once. ``user_audio`` is passed only to a function that declares a
        ``user_audio`` parameter; any other function receives text only, as before."""
        if self._function is None:
            raise RuntimeError("GenericAdapter.send() was called before start()")
        text_in = "" if user_text is None else user_text
        history = copy.deepcopy(self._history)   # the agent cannot change the adapter's own history

        started = time.perf_counter()
        if self.accepts_user_audio:
            result = self._function(text_in, history, user_audio=user_audio)
        else:
            result = self._function(text_in, history)
        if inspect.isawaitable(result):
            result = await result
        latency_ms = (time.perf_counter() - started) * 1000.0

        reply = self._to_reply(result, latency_ms)
        if user_text is not None:
            self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": reply.text})
        return reply

    async def close(self) -> None:
        """Nothing to disconnect: the function runs in process. The next ``start()`` resets history."""
        return None

    # -- system prompt -----------------------------------------------------------------------------

    def get_system_prompt(self) -> Optional[str]:
        """config.yaml ``agent.system_prompt`` text, else the module's prompt (read at ``start()``)."""
        if self._config.agent.system_prompt is not None:
            if not self._config_prompt_loaded:
                self._config_prompt = read_system_prompt(self._config)
                self._config_prompt_loaded = True
            return self._config_prompt
        return self._agent_prompt

    async def _module_system_prompt(self) -> Optional[str]:
        """The module's ``SYSTEM_PROMPT`` constant, else its ``get_system_prompt()`` result, else None."""
        module = self._module
        if module is None:
            return None
        constant = getattr(module, "SYSTEM_PROMPT", None)
        if isinstance(constant, str) and constant.strip():
            return constant
        getter = getattr(module, "get_system_prompt", None)
        if callable(getter):
            value = getter()
            if inspect.isawaitable(value):
                value = await value
            if isinstance(value, str) and value.strip():
                return value
        return None

    # -- entrypoint --------------------------------------------------------------------------------

    def _load_entrypoint(self) -> tuple[ModuleType, Callable[..., Any]]:
        entrypoint = (self._entrypoint or "").strip()
        if not entrypoint:
            raise ConfigError(ENTRYPOINT_FIELD, "required when agent.adapter is generic")
        module_name, attr_path = split_entrypoint(
            entrypoint, f"{entrypoint} is not of the form module:function (e.g. my_agent:respond)"
        )
        module = import_entrypoint_module(module_name, Path(self._config.config_dir))
        target = resolve_attribute(module, attr_path)
        if target is None:
            raise ConfigError(ENTRYPOINT_FIELD, f"module {module_name} has no function {attr_path}")
        if not callable(target):
            raise ConfigError(ENTRYPOINT_FIELD, f"{module_name}:{attr_path} is not callable")
        return module, target

    # -- reply conversion --------------------------------------------------------------------------

    def _to_reply(self, result: Any, latency_ms: float) -> AgentReply:
        if isinstance(result, str):
            return AgentReply(text=result, latency_ms=latency_ms)
        if not isinstance(result, Mapping):
            raise TypeError(
                f"agent function {self._entrypoint} returned {type(result).__name__}; expected str or dict"
            )

        source = f"agent function {self._entrypoint}"
        text = result.get("text")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            raise TypeError(f"{source} returned a non-string 'text'")

        audio = result.get("audio")
        if isinstance(audio, (bytearray, memoryview)):
            audio = bytes(audio)
        elif audio is not None and not isinstance(audio, bytes):
            raise TypeError(f"{source} returned non-bytes 'audio'")

        heard_text = result.get("heard_text")
        if heard_text is not None and not isinstance(heard_text, str):
            raise TypeError(f"{source} returned a non-string 'heard_text'")

        return AgentReply(
            text=text,
            audio_bytes=audio,
            latency_ms=latency_ms,
            heard_text=heard_text,
            tool_calls=to_tool_calls(result.get("tool_calls"), source),
        )


# ------------------------------------------------------------------------------------------------
# Shared with the custom and HTTP adapters (FIXES-ROUND-3-SPEC.md B3, B4)
# ------------------------------------------------------------------------------------------------


def _declares_user_audio(function: Callable[..., Any]) -> bool:
    """True when ``function`` has a parameter named exactly ``user_audio`` that can be passed by keyword
    (``**kwargs`` alone does not count, so existing functions never receive a new argument)."""
    try:
        parameter = inspect.signature(function).parameters.get("user_audio")
    except (TypeError, ValueError):   # a builtin or C function without a signature
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def split_entrypoint(entrypoint: str, form_error: str) -> tuple[str, str]:
    """``(module, attribute path)`` of a stripped ``module:attr`` entrypoint (dots allowed after the
    colon for a nested attribute); otherwise ``ConfigError(agent.entrypoint, form_error)``."""
    module_name, sep, attr_path = entrypoint.partition(":")
    module_name, attr_path = module_name.strip(), attr_path.strip()
    if not sep or not module_name or not attr_path:
        raise ConfigError(ENTRYPOINT_FIELD, form_error)
    return module_name, attr_path


def import_entrypoint_module(module_name: str, config_dir: Path) -> ModuleType:
    """Import ``module_name`` with ``config_dir`` (the config file's folder) on ``sys.path``; any import
    failure is ``ConfigError(agent.entrypoint, "cannot import module {module} from {dir} or sys.path:
    {type}: {message}")`` (SPEC.md section 8.2)."""
    _add_to_sys_path(config_dir)
    importlib.invalidate_caches()   # pick up agent modules written after the import system cached the folder
    error: Optional[ConfigError] = None
    try:
        module = importlib.import_module(module_name)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:   # any import failure, SystemExit included, is a config error (SPEC.md section 8.2)
        error = ConfigError(
            ENTRYPOINT_FIELD,
            f"cannot import module {module_name} from {config_dir} or sys.path: "
            f"{type(exc).__name__}: {exc}",
        )
    if error is not None:
        raise error
    return module


def resolve_attribute(module: ModuleType, attr_path: str) -> Any:
    """The attribute ``attr_path`` (``A`` or ``A.B``) of ``module``, or None when a part is missing."""
    target: Any = module
    for part in attr_path.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


def to_tool_calls(raw: Any, source: str) -> list[ToolCall]:
    """Convert a returned ``tool_calls`` list (dicts or ``ToolCall`` objects) into ``ToolCall``s.

    ``source`` names the agent in the ``TypeError`` texts: ``"agent function {entrypoint}"`` for the
    generic adapter, ``"HTTP agent"`` for the HTTP adapter."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{source} returned 'tool_calls' that is not a list")
    calls: list[ToolCall] = []
    for item in raw:
        if isinstance(item, ToolCall):
            calls.append(item.model_copy(deep=True))
            continue
        if not isinstance(item, Mapping):
            raise TypeError(f"{source} returned a tool call that is not a dict")
        values = {key: item[key] for key in _TOOL_CALL_KEYS if key in item}
        name = values.get("name")
        if not isinstance(name, str) or not name:   # also an OpenAI-style call ({"function": {"name": ...}})
            raise TypeError(f"{source} returned a tool call without a string 'name'")
        arguments = values.get("arguments")
        if arguments is None:
            values["arguments"] = {}   # a missing arguments becomes {} (SPEC.md section 9)
        elif not isinstance(arguments, Mapping):   # e.g. a JSON string: the dict contract has no parsing
            raise TypeError(f"{source} returned a tool call whose 'arguments' is not a dict")
        result = values.get("result")
        if result is not None and not isinstance(result, Mapping):
            values["result"] = {"value": result}   # same rule as the Pipecat adapter (SPEC.md section 9)
        try:
            call = ToolCall(**values)
        except ValidationError as exc:
            raise _invalid_tool_call(exc, source) from None
        calls.append(call)
    return calls


def _invalid_tool_call(exc: ValidationError, source: str) -> TypeError:
    """The first validation problem of a tool call as a clear ``TypeError`` (no pydantic URL or input)."""
    details = exc.errors(include_url=False, include_input=False)
    first: Mapping[str, Any] = details[0] if details else {}
    location = ".".join(str(part) for part in first.get("loc", ())) or "tool call"
    message = first.get("msg") or "invalid value"
    return TypeError(f"{source} returned an invalid tool call: {location}: {message}")


def _add_to_sys_path(folder: Path) -> None:
    """Put ``folder`` at the front of ``sys.path`` unless it is already there (as Python does for a script)."""
    target = folder.resolve()
    for entry in sys.path:
        if not entry:
            continue
        try:
            if Path(entry).resolve() == target:
                return
        except (OSError, ValueError):
            continue
    sys.path.insert(0, str(target))
