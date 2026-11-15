# SPDX-License-Identifier: Apache-2.0
"""Plugin API types (SPEC.md sections 5.1, 7, and 18.1).

``AgentTool``, ``ToolCall``, ``MetricResult``, ``ScoreResult``, and ``MetricContext`` are defined in
the Apache-2.0 module ``impulse_evals.plugin_types``; ``impulse_evals.models`` re-exports the first
four and ``impulse_evals.metrics.base`` re-exports ``MetricContext`` (the same class objects), so
every existing import path keeps working.

Field names and required fields of the four section 7 models come from the tables in
``test_models.py`` (loaded by path, not duplicated). The licensing side (SPDX header and the plugin
API import rule) is checked in ``test_license_headers.py``. Impulse modules are imported inside the
tests so collection never fails.
"""

import functools
import importlib.util
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

MODELS_REEXPORTS = ("AgentTool", "ToolCall", "MetricResult", "ScoreResult")
PLUGIN_TYPES = MODELS_REEXPORTS + ("MetricContext",)

# Values for the required fields, used to build an instance and read its defaults.
REQUIRED_VALUES = {
    "AgentTool": {"name": "book_slot"},
    "ToolCall": {"name": "book_slot", "arguments": {}},
    "ScoreResult": {
        "rule_id": "dpdp.consent_before_pii",
        "regulation_ref": "DPDP Act 2023 (notice and consent provisions)",
        "result": "pass",
        "severity": "critical",
        "reason": "r",
    },
    "MetricResult": {
        "metric_id": "UNIV-001",
        "name": "Task completion",
        "category": "universal",
        "score": 1.0,
        "label": "good",
        "reason": "r",
    },
}

# Defaults of the optional fields (SPEC.md section 7). "DISCLAIMER" is the test_models constant.
OPTIONAL_DEFAULTS = {
    "AgentTool": {"description": None},
    "ToolCall": {"result": None, "success": None, "error": None},
    "ScoreResult": {"turn_index": None, "confidence": 0.0, "no_verdict": False, "disclaimer": "DISCLAIMER"},
    "MetricResult": {
        "raw_value": None,
        "threshold_warn": 0.6,
        "threshold_fail": 0.4,
        "turn_index": None,
        "requires_audio": False,
        "is_proxy": False,
        "skipped": False,
        "informational": False,
    },
}

# MetricContext (SPEC.md section 18.1, metrics.base row): every field optional, with these defaults.
METRIC_CONTEXT_DEFAULTS = {
    "agent_tools": None,
    "agent_type": "voice",
    "voice_mode": False,
    "metrics_enabled": True,
    "voice_thresholds": {},
    "dry_run": False,
}


def _plugin_types():
    from impulse_evals import plugin_types

    return plugin_types


@functools.lru_cache(maxsize=1)
def _spec_tables():
    """``test_models.py`` loaded by path: its SPEC.md section 7 tables and ``DISCLAIMER``."""
    spec = importlib.util.spec_from_file_location("_impulse_test_models_tables", TESTS_DIR / "test_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validation_error():
    import pydantic

    return pydantic.ValidationError


# --------------------------------------------------------------------------------------------
# Where the types live
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PLUGIN_TYPES)
def test_plugin_types_defines_type(name):
    import pydantic

    cls = getattr(_plugin_types(), name)
    assert isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)
    assert cls.__module__ == "impulse_evals.plugin_types", (
        f"{name} must be defined in impulse_evals.plugin_types, not imported from {cls.__module__}"
    )


@pytest.mark.parametrize("name", MODELS_REEXPORTS)
def test_models_reexports_plugin_type(name):
    from impulse_evals import models

    assert getattr(models, name) is getattr(_plugin_types(), name)


def test_metrics_base_reexports_metric_context():
    from impulse_evals.metrics import base

    assert base.MetricContext is _plugin_types().MetricContext


def test_from_imports_keep_working():
    from impulse_evals.metrics.base import MetricContext
    from impulse_evals.models import AgentTool, MetricResult, ScoreResult, ToolCall

    plugin_types = _plugin_types()
    assert (AgentTool, ToolCall, MetricResult, ScoreResult, MetricContext) == tuple(
        getattr(plugin_types, name) for name in PLUGIN_TYPES
    )


# --------------------------------------------------------------------------------------------
# Fields and defaults
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS_REEXPORTS)
def test_plugin_type_fields_match_spec(name):
    tables = _spec_tables()
    cls = getattr(_plugin_types(), name)
    assert set(cls.model_fields) == tables.EXPECTED_FIELDS[name]
    required = {field for field, info in cls.model_fields.items() if info.is_required()}
    assert required == tables.EXPECTED_REQUIRED[name]


@pytest.mark.parametrize("name", MODELS_REEXPORTS)
def test_plugin_type_defaults_match_spec(name):
    tables = _spec_tables()
    cls = getattr(_plugin_types(), name)
    instance = cls(**REQUIRED_VALUES[name])
    expected = {
        field: tables.DISCLAIMER if value == "DISCLAIMER" else value
        for field, value in OPTIONAL_DEFAULTS[name].items()
    }
    assert {field: getattr(instance, field) for field in expected} == expected
    assert set(expected) | set(REQUIRED_VALUES[name]) == set(cls.model_fields)


def test_metric_context_fields_and_defaults():
    cls = _plugin_types().MetricContext
    assert set(cls.model_fields) == set(METRIC_CONTEXT_DEFAULTS)
    assert not [field for field, info in cls.model_fields.items() if info.is_required()]
    ctx = cls()
    assert {field: getattr(ctx, field) for field in METRIC_CONTEXT_DEFAULTS} == METRIC_CONTEXT_DEFAULTS


def test_metric_context_agent_tools_are_plugin_agent_tools():
    plugin_types = _plugin_types()
    ctx = plugin_types.MetricContext(agent_tools=[{"name": "book_slot", "description": "Books a slot"}])
    assert isinstance(ctx.agent_tools[0], plugin_types.AgentTool)
    assert ctx.agent_tools[0].name == "book_slot"
    assert ctx.agent_tools[0].description == "Books a slot"
    assert plugin_types.MetricContext(agent_tools=[]).agent_tools == []


def test_metric_context_agent_type_literal():
    cls = _plugin_types().MetricContext
    for agent_type in ("voice", "chat", "multimodal"):
        assert cls(agent_type=agent_type).agent_type == agent_type
    with pytest.raises(_validation_error()):
        cls(agent_type="phone")
