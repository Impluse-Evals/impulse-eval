# SPDX-License-Identifier: Apache-2.0
"""Metric plugin API (SPEC.md sections 12, 13, and 18.1; METRICS.md section 0).

- ``Metric``: base class of every metric. ``JudgeMetric``: base class of judge-based metrics.
- ``MetricContext``: the run settings a metric may depend on (agent tools, agent type, mode, ...);
  defined in ``impulse_evals.plugin_types`` and re-exported here.
- ``get_metric(metric_id)``: registry lookup. Metric modules are imported lazily on first use, so
  ``metrics.voice`` and ``metrics.chat`` are loaded only when one of their IDs is requested.
- ``select_metrics(scenario, ctx)``: the metric IDs to run for a scenario, including the automatic
  ones (UNIV-025 when the scenario defines tool expectations, VOICE-011 in voice mode).
- ``build_criterion(criterion, scenario, ctx)``: the criterion text plus the SPEC.md section 12
  context lines. A built-in metric is recognised by its criterion text (registry lookup), so the
  three-argument call also adds UNIV-023's tool lines and judge-decided ``Return skip if:`` lines.

``KNOWN_METRIC_IDS`` (every metric ID in METRICS.md) is defined here; ``impulse_evals.config``
imports it for ``load_config`` validation, so the registry and config validation always agree.

Licensing: this file is Apache-2.0 (SPEC.md section 5.1) and imports no AGPL core code: the plugin
types come from the Apache-2.0 ``impulse_evals.plugin_types``, every other core name is
annotation-only, and metric modules are loaded by name through ``importlib`` (``get_metric``).
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, Union

# MetricContext is re-exported here (SPEC.md section 18.1, metrics.base row).
from impulse_evals.plugin_types import MetricContext as MetricContext
from impulse_evals.plugin_types import MetricResult

if TYPE_CHECKING:  # annotation only
    from impulse_evals.judge.base import Judge, JudgeVerdict
    from impulse_evals.models import CallTranscript, Scenario


# ------------------------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------------------------


class UnknownMetricError(ValueError):
    """Raised by ``get_metric`` for an ID that no metric module defines."""


# Every metric ID in METRICS.md (sections 1-3). ``impulse_evals.config`` validates config.yaml
# against this tuple, so it never imports a metric module.
KNOWN_METRIC_IDS: tuple[str, ...] = (
    "UNIV-001", "UNIV-003", "UNIV-005", "UNIV-007", "UNIV-009", "UNIV-011", "UNIV-013",
    "UNIV-015", "UNIV-017", "UNIV-019", "UNIV-021", "UNIV-023", "UNIV-025",
    "VOICE-001", "VOICE-003", "VOICE-005", "VOICE-007", "VOICE-009", "VOICE-011", "VOICE-013",
    "VOICE-015", "VOICE-027", "VOICE-029", "VOICE-031", "VOICE-035", "VOICE-037", "VOICE-039",
    "CHAT-001", "CHAT-003", "CHAT-005",
)

_MODULE_BY_PREFIX = {
    "UNIV": "impulse_evals.metrics.universal",
    "VOICE": "impulse_evals.metrics.voice",
    "CHAT": "impulse_evals.metrics.chat",
}

# Metric ID -> module that defines it (imported on first use).
METRIC_MODULES: dict[str, str] = {
    metric_id: _MODULE_BY_PREFIX[metric_id.split("-", 1)[0]] for metric_id in KNOWN_METRIC_IDS
}

# Every universal judge metric (UNIV-001 through UNIV-023): the metrics a scenario with an empty
# ``metrics`` list runs. UNIV-025 is never listed here; it runs automatically (select_metrics).
DEFAULT_UNIVERSAL_METRICS: tuple[str, ...] = tuple(
    metric_id for metric_id in KNOWN_METRIC_IDS if metric_id.startswith("UNIV-") and metric_id != "UNIV-025"
)

_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(cls: type[Metric]) -> type[Metric]:
    """Class decorator: make ``cls`` available through ``get_metric(cls.metric_id)``."""
    _REGISTRY[cls.metric_id] = cls
    return cls


def get_metric(metric_id: str) -> Metric:
    """A new instance of the metric registered under ``metric_id``.

    The module that defines the ID is imported on first use. An unknown ID raises
    ``UnknownMetricError`` (``load_config`` reports unknown IDs as config errors before a run).
    """
    cls = _REGISTRY.get(metric_id)
    if cls is None:
        module = METRIC_MODULES.get(metric_id)
        if module is None:
            raise UnknownMetricError(f"unknown metric ID {metric_id}")
        importlib.import_module(module)
        cls = _REGISTRY.get(metric_id)
        if cls is None:
            raise UnknownMetricError(f"metric {metric_id} is not defined in {module}")
    return cls()


def select_metrics(scenario: Scenario, ctx: Optional[MetricContext] = None) -> list[str]:
    """Metric IDs to run for ``scenario`` (SPEC.md section 13).

    The scenario's ``metrics`` (an empty list means every universal judge metric), then UNIV-025
    when the scenario defines ``expected_tool_calls`` or ``forbidden_tool_calls``, then VOICE-011
    in voice mode; duplicates removed, first occurrence kept. ``metrics.enabled: false`` gives ``[]``.
    """
    ctx = ctx if ctx is not None else MetricContext()
    if not ctx.metrics_enabled:
        return []
    selected = list(scenario.metrics) if scenario.metrics else list(DEFAULT_UNIVERSAL_METRICS)
    if scenario.expected_tool_calls or scenario.forbidden_tool_calls:
        selected.append("UNIV-025")
    if ctx.voice_mode:
        selected.append("VOICE-011")
    return list(dict.fromkeys(selected))


# ------------------------------------------------------------------------------------------------
# Criterion context lines (SPEC.md section 12)
# ------------------------------------------------------------------------------------------------


class _FromRegistry:
    """Default of ``build_criterion``'s keyword arguments: take the value from the metric registry."""

    def __repr__(self) -> str:
        return "FROM_REGISTRY"


FROM_REGISTRY = _FromRegistry()


def _registered_metric_for(criterion: str) -> Optional[Metric]:
    """The registered metric whose ``criterion`` is exactly ``criterion``, or None.

    Only metrics already registered are searched (no module is imported here): a caller holding a
    built-in criterion text got it from its metric class, so that module is already loaded.
    """
    for cls in _REGISTRY.values():
        if cls.criterion is not None and cls.criterion == criterion:
            return cls()
    return None


def build_criterion(
    criterion: str,
    scenario: Optional[Scenario],
    ctx: Optional[MetricContext] = None,
    *,
    include_tools: Union[bool, _FromRegistry] = FROM_REGISTRY,
    skip_when: Union[str, None, _FromRegistry] = FROM_REGISTRY,
) -> str:
    """``criterion`` followed by one blank line and its context lines, in this order:

    1. ``Expected outcome: {expected_outcome}`` when the scenario sets one;
    2. UNIV-023 only: ``Tools available to the agent:`` and one line per tool (``{name} - {description}``,
       or ``{name}``) when ``ctx.agent_tools`` is set, then ``Tools this scenario expects: {names}``
       when the scenario has ``expected_tool_calls``;
    3. ``Return skip if: {Skip-when text}`` for a metric whose skip condition is judge-decided (not
       ``never``, not checked in code).

    With the three SPEC.md section 18.1 arguments, lines 2 and 3 are derived from the registered
    metric whose criterion text equals ``criterion`` (its ``include_tool_context`` and
    ``judge_skip_condition()``); any other text (an assertion criterion, a custom text) gets line 1
    only. The keyword arguments override the lookup: ``include_tools`` (bool) and ``skip_when``
    (the text, or None for no line). Without context lines the criterion is returned unchanged.
    """
    if isinstance(include_tools, _FromRegistry) or isinstance(skip_when, _FromRegistry):
        metric = _registered_metric_for(criterion)
        if isinstance(include_tools, _FromRegistry):
            include_tools = bool(getattr(metric, "include_tool_context", False))
        if isinstance(skip_when, _FromRegistry):
            skip_when = metric.judge_skip_condition() if isinstance(metric, JudgeMetric) else None

    lines: list[str] = []
    outcome = scenario.expected_outcome if scenario is not None else None
    if outcome is not None and outcome.strip():
        lines.append(f"Expected outcome: {outcome}")
    if include_tools:
        tools = ctx.agent_tools if ctx is not None else None
        if tools is not None:
            lines.append("Tools available to the agent:")
            for tool in tools:
                description = (tool.description or "").strip()
                lines.append(f"{tool.name} - {description}" if description else tool.name)
        expected = scenario.expected_tool_calls if scenario is not None else []
        if expected:
            names = dict.fromkeys(entry.name for entry in expected)
            lines.append("Tools this scenario expects: " + ", ".join(names))
    if skip_when is not None:
        lines.append(f"Return skip if: {skip_when}")
    if not lines:
        return criterion
    return criterion + "\n\n" + "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# Metric base classes
# ------------------------------------------------------------------------------------------------

LABEL_SCORES: dict[str, float] = {"good": 1.0, "warning": 0.5, "poor": 0.0}


class Metric(ABC):
    """Base class of every metric (METRICS.md). Subclasses set the class attributes and ``evaluate``."""

    metric_id: ClassVar[str]
    name: ClassVar[str]
    category: ClassVar[Literal["universal", "voice", "chat"]]
    tier: ClassVar[Optional[Literal["primary", "fast"]]] = None   # None for deterministic metrics
    criterion: ClassVar[Optional[str]] = None                      # None for deterministic metrics
    skip_when: ClassVar[str] = "never"                             # the METRICS.md "Skip when" text
    requires_audio: ClassVar[bool] = False
    informational: ClassVar[bool] = False

    @abstractmethod
    async def evaluate(
        self,
        transcript: CallTranscript,
        scenario: Scenario,
        judge: Judge,
        ctx: Optional[MetricContext] = None,
    ) -> MetricResult:
        """Evaluate this metric on one transcript."""

    def result(self, *, score: float, label: str, reason: str, **fields: Any) -> MetricResult:
        """A ``MetricResult`` carrying this metric's ID, name, category, and flags."""
        values: dict[str, Any] = {
            "requires_audio": self.requires_audio,
            "informational": self.informational,
        }
        values.update(fields)
        return MetricResult(
            metric_id=self.metric_id,
            name=self.name,
            category=self.category,
            score=score,
            label=label,
            reason=reason,
            **values,
        )

    def skipped(self, reason: str, **fields: Any) -> MetricResult:
        """A skipped result: ``skipped=True``, ``label="skipped"``, ``score=0.0``."""
        return self.result(score=0.0, label="skipped", reason=reason, skipped=True, **fields)


class JudgeMetric(Metric):
    """A metric decided by the judge: pass = 1.0 / good, fail = 0.0 / poor, skip = skipped.

    Subclasses set ``tier``, ``criterion``, and ``skip_when``. A skip condition that is checked in
    code (METRICS.md section 0) is implemented in ``code_skip_reason`` with ``skip_in_code = True``;
    any other non-``never`` condition is sent to the judge as ``Return skip if: ...``.
    """

    skip_in_code: ClassVar[bool] = False
    include_tool_context: ClassVar[bool] = False   # UNIV-023 only (SPEC.md section 12, context line 2)

    def code_skip_reason(
        self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext
    ) -> Optional[str]:
        """Reason to skip without a judge call, or None to ask the judge."""
        return None

    def judge_skip_condition(self) -> Optional[str]:
        """The ``Return skip if`` text, or None when the condition is ``never`` or checked in code."""
        if self.skip_in_code or not self.skip_when or self.skip_when == "never":
            return None
        return self.skip_when

    def full_criterion(self, scenario: Scenario, ctx: Optional[MetricContext]) -> str:
        """The criterion with its context lines, exactly as sent to the judge."""
        return build_criterion(
            self.criterion or "",
            scenario,
            ctx,
            include_tools=self.include_tool_context,
            skip_when=self.judge_skip_condition(),
        )

    async def evaluate(
        self,
        transcript: CallTranscript,
        scenario: Scenario,
        judge: Judge,
        ctx: Optional[MetricContext] = None,
    ) -> MetricResult:
        ctx = ctx if ctx is not None else MetricContext()
        if ctx.dry_run:
            return self.skipped("dry run")
        reason = self.code_skip_reason(transcript, scenario, ctx)
        if reason is not None:
            return self.skipped(reason)
        verdict = await judge.evaluate(transcript, self.full_criterion(scenario, ctx), self.tier or "primary")
        return self.from_verdict(verdict)

    def from_verdict(self, verdict: JudgeVerdict) -> MetricResult:
        """Map a judge verdict to a ``MetricResult`` (reason and turn index copied from the verdict)."""
        if verdict.result == "skip":
            return self.skipped(verdict.reason, turn_index=verdict.turn_index)
        passed = verdict.result == "pass"
        return self.result(
            score=1.0 if passed else 0.0,
            label="good" if passed else "poor",
            reason=verdict.reason,
            turn_index=verdict.turn_index,
        )
