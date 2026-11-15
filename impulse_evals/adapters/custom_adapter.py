# SPDX-License-Identifier: Apache-2.0
"""Custom adapters from config: ``agent.adapter: custom`` (FIXES-ROUND-3-SPEC.md B3).

``agent.entrypoint = "module:Class"`` (dots allowed after the colon for a nested attribute) names a
subclass of ``impulse_evals.adapters.base.AgentInterface``. The runner's default factory creates one
instance per scenario with ``Class(config)`` (the loaded ``ProjectConfig``; free-form settings for
the adapter are in ``config.agent.options``) and treats it like any built-in adapter.

Loading follows the generic adapter's import rules and texts (the shared helpers in
``generic_adapter``): the config file's folder on ``sys.path``, ``importlib.invalidate_caches()``,
and every problem a ``ConfigError`` at ``agent.entrypoint``:

- not ``module:Class``: "{entrypoint} is not of the form module:Class (e.g. my_adapter:MyAdapter)"
- any import failure: "cannot import module {module} from {dir} or sys.path: {type}: {message}"
- attribute missing: "module {module} has no class {attr}"
- not a class, or not an ``AgentInterface`` subclass: "{module}:{attr} is not a subclass of
  impulse_evals.adapters.base.AgentInterface"
- the constructor raises anything but ``KeyboardInterrupt`` (``SystemExit`` included): "could not
  create {module}:{attr}: {type}: {message}" (the message redacted first, then on one line and cut to
  200 characters); a ``ConfigError`` raised by the constructor itself is re-raised unchanged.

``load_config`` never imports the module (it only requires a non-blank entrypoint); ``impulse
doctor`` and ``impulse run --dry-run`` import and check the class (``load_adapter_class``) without
creating it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Optional

from impulse_evals.adapters.base import AgentInterface
from impulse_evals.adapters.generic_adapter import (
    ENTRYPOINT_FIELD,
    import_entrypoint_module,
    resolve_attribute,
    split_entrypoint,
)
from impulse_evals.config import ConfigError, ProjectConfig, redact

__all__ = ["create_adapter", "entrypoint_parts", "load_adapter_class"]

ERROR_MESSAGE_CHARS = 200   # the constructor's message: one line, cut AFTER redaction


def entrypoint_parts(config: ProjectConfig) -> tuple[str, str]:
    """``(module, attribute path)`` of ``agent.entrypoint``; a missing, blank or malformed entrypoint
    is the ``ConfigError`` at ``agent.entrypoint``."""
    entrypoint = (config.agent.entrypoint or "").strip()
    if not entrypoint:
        raise ConfigError(ENTRYPOINT_FIELD, "required when agent.adapter is custom")
    return split_entrypoint(
        entrypoint, f"{entrypoint} is not of the form module:Class (e.g. my_adapter:MyAdapter)"
    )


def load_adapter_class(config: ProjectConfig) -> type[AgentInterface]:
    """Import and check the class named by ``agent.entrypoint``, without creating it."""
    module_name, attr_path = entrypoint_parts(config)
    module = import_entrypoint_module(module_name, Path(config.config_dir))
    target = resolve_attribute(module, attr_path)
    if target is None:
        raise ConfigError(ENTRYPOINT_FIELD, f"module {module_name} has no class {attr_path}")
    if not (isinstance(target, type) and issubclass(target, AgentInterface)):
        raise ConfigError(
            ENTRYPOINT_FIELD,
            f"{module_name}:{attr_path} is not a subclass of impulse_evals.adapters.base.AgentInterface",
        )
    return target


def create_adapter(
    config: ProjectConfig, *, redact_text: Optional[Callable[[str], str]] = None
) -> AgentInterface:
    """A new instance of the custom adapter class: ``Class(config)``.

    ``redact_text`` (the run's key redactor; default: the standard key variables and every
    ``*_API_KEY`` environment variable) redacts the constructor's error message BEFORE it is cut."""
    # The documented constructor is Class(config); AgentInterface itself declares none.
    adapter_class: Callable[[ProjectConfig], AgentInterface] = load_adapter_class(config)  # type: ignore[assignment]
    module_name, attr_path = entrypoint_parts(config)
    error: Optional[ConfigError] = None
    try:
        return adapter_class(config)
    except (ConfigError, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 - any constructor failure, SystemExit included (B3/B4-rev R8)
        try:
            text = str(exc)
        except Exception:  # noqa: BLE001 - a broken __str__ must not replace the ConfigError
            text = ""
        message = (redact_text or redact)(text)
        message = " ".join(message.split())
        if len(message) > ERROR_MESSAGE_CHARS:
            message = message[: ERROR_MESSAGE_CHARS - 3] + "..."
        error = ConfigError(
            ENTRYPOINT_FIELD, f"could not create {module_name}:{attr_path}: {type(exc).__name__}: {message}"
        )
    # Raised outside the except block, so the original (which may hold a key) is not chained to it.
    raise error
