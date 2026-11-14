# SPDX-License-Identifier: Apache-2.0
"""Plugin API types (SPEC.md sections 5.1, 7, and 18.1).

``AgentTool``, ``ToolCall``, ``MetricResult``, and ``ScoreResult`` (SPEC.md section 7) and
``MetricContext`` (SPEC.md section 18.1) are defined here, in the Apache-2.0 plugin API, so the
plugin API files never import AGPL core code (SPEC.md section 5.1 plugin API import rule).
``impulse_evals.models`` re-exports the first four and ``impulse_evals.metrics.base`` re-exports
``MetricContext``, so every existing import path keeps working.

This module imports only the standard library, pydantic, and typing.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

__all__ = ["AgentTool", "MetricContext", "MetricResult", "ScoreResult", "ToolCall"]


class ToolCall(BaseModel):
    name: str                             # name of the tool / function called
    arguments: dict                       # arguments as the agent passed them
    result: Optional[dict] = None         # raw result returned by the tool, if captured
    success: Optional[bool] = None        # did the downstream action complete? None = not captured
    error: Optional[str] = None           # error message if the call failed


class AgentTool(BaseModel):
    name: str
    description: Optional[str] = None


class ScoreResult(BaseModel):             # result of ONE compliance assertion
    rule_id: str                          # full "pack.rule_id", e.g. "dpdp.consent_before_pii"
    regulation_ref: str
    result: Literal["pass", "fail", "skip"]
    severity: Literal["critical", "high", "medium"]
    reason: str
    turn_index: Optional[int] = None
    confidence: float = 0.0
    no_verdict: bool = False              # True only for a skip caused by an empty or unparseable judge response
    disclaimer: str = ("Technical signal only. Not legal certification. "
                       "Consult qualified counsel before relying on this result "
                       "for regulatory purposes.")


class MetricResult(BaseModel):
    metric_id: str
    name: str
    category: Literal["universal", "voice", "chat"]
    score: float                          # 0.0-1.0
    raw_value: Optional[float] = None
    label: Literal["good", "warning", "poor", "skipped"]  # "skipped" exactly when skipped=True (score 0.0)
    reason: str
    threshold_warn: float = 0.6
    threshold_fail: float = 0.4
    turn_index: Optional[int] = None
    requires_audio: bool = False
    is_proxy: bool = False
    skipped: bool = False
    informational: bool = False


class MetricContext(BaseModel):
    """Run settings passed to every metric (SPEC.md section 18.1)."""

    agent_tools: Optional[list[AgentTool]] = None   # config.yaml agent.tools: None = unknown, [] = no tools
    agent_type: Literal["voice", "chat", "multimodal"] = "voice"
    voice_mode: bool = False
    metrics_enabled: bool = True
    voice_thresholds: dict = {}                       # the metrics.voice block; empty = SPEC.md 8.2 defaults
    dry_run: bool = False
