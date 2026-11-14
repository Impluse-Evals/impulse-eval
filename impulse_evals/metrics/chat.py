# SPDX-License-Identifier: AGPL-3.0-only
"""Chat metrics (METRICS.md section 3; SPEC.md sections 7 and 13). Criterion texts copied word for word.

CHAT metrics run only when a scenario lists them in ``metrics`` (an empty list means the universal
metrics only; ``select_metrics``). They are meant for ``agent.type`` ``chat`` or ``multimodal``: with
``agent.type: voice`` (also the ``MetricContext`` default) every CHAT metric is skipped in code, with
no judge call, with the reason "chat metric; agent.type is voice".

- CHAT-001 Response latency (deterministic, no judge): ``raw_value`` = median ``Turn.latency_ms`` of
  the agent turns that have one (``statistics.median``: the mean of the middle two for an even count);
  good <= 3000 ms, warning <= 8000 ms, poor above; score 1.0 / 0.5 / 0.0; ``threshold_warn`` /
  ``threshold_fail`` record 3000 / 8000. No agent turn with a latency: skipped (the precondition did
  not occur, METRICS.md section 0).
- CHAT-003 Formatting and readability (judge, fast) and CHAT-005 Reference validity (judge, primary):
  pass / fail / skip map to 1.0 good / 0.0 poor / skipped like every judge metric.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Optional

from impulse_evals.metrics.base import (
    LABEL_SCORES,
    JudgeMetric,
    Metric,
    MetricContext,
    register_metric,
)
from impulse_evals.models import CallTranscript, MetricResult, Scenario

if TYPE_CHECKING:  # annotation only
    from impulse_evals.judge.base import Judge

__all__ = ["CHAT_ON_VOICE", "LATENCY_GOOD_MS", "LATENCY_WARNING_MS"]

# Exact skip reason for a voice agent (METRICS.md section 3; SPEC.md section 13).
CHAT_ON_VOICE = "chat metric; agent.type is voice"

# CHAT-001 thresholds in ms (METRICS.md section 3): good <= 3000, warning <= 8000, poor above.
LATENCY_GOOD_MS = 3000.0
LATENCY_WARNING_MS = 8000.0

# METRICS.md section 3 has no "Skip when" column; this text describes the code-decided skip and is
# never sent to the judge.
_SKIP_WHEN = "`agent.type` is `voice` (decided in code)"


def _is_voice_agent(ctx: MetricContext) -> bool:
    return ctx.agent_type == "voice"


def _agent_latencies(transcript: CallTranscript) -> list[float]:
    """Finite ``latency_ms`` values of the agent turns that have one, in transcript order."""
    latencies: list[float] = []
    for turn in transcript.turns:
        if turn.speaker != "agent" or turn.latency_ms is None:
            continue
        value = float(turn.latency_ms)
        if math.isfinite(value):
            latencies.append(value)
    return latencies


@register_metric
class ResponseLatency(Metric):
    metric_id = "CHAT-001"
    name = "Response latency"
    category = "chat"
    tier = None
    criterion = None
    skip_when = _SKIP_WHEN + "; no agent turn has a measured latency"

    async def evaluate(
        self,
        transcript: CallTranscript,
        scenario: Scenario,
        judge: Judge,
        ctx: Optional[MetricContext] = None,
    ) -> MetricResult:
        ctx = ctx if ctx is not None else MetricContext()
        if _is_voice_agent(ctx):
            return self.skipped(CHAT_ON_VOICE)
        latencies = _agent_latencies(transcript)
        if not latencies:
            return self.skipped("no agent turn has a measured latency")
        raw = float(statistics.median(latencies))
        if raw <= LATENCY_GOOD_MS:
            label = "good"
        elif raw <= LATENCY_WARNING_MS:
            label = "warning"
        else:
            label = "poor"
        reason = (
            f"median response latency {raw:.0f} ms over {len(latencies)} agent turns; "
            f"good <= {LATENCY_GOOD_MS:g} ms, warning <= {LATENCY_WARNING_MS:g} ms"
        )
        return self.result(
            score=LABEL_SCORES[label],
            label=label,
            reason=reason,
            raw_value=raw,
            threshold_warn=LATENCY_GOOD_MS,
            threshold_fail=LATENCY_WARNING_MS,
        )


class _ChatJudgeMetric(JudgeMetric):
    """A judge-based CHAT metric: skipped in code (no judge call) for a voice agent."""

    category = "chat"
    skip_when = _SKIP_WHEN
    skip_in_code = True

    def code_skip_reason(
        self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext
    ) -> Optional[str]:
        return CHAT_ON_VOICE if _is_voice_agent(ctx) else None


@register_metric
class FormattingAndReadability(_ChatJudgeMetric):
    metric_id = "CHAT-003"
    name = "Formatting and readability"
    tier = "fast"
    criterion = "Is formatting appropriate for the channel and easy to read (no wall of text, no broken markdown)?"


@register_metric
class ReferenceValidity(_ChatJudgeMetric):
    metric_id = "CHAT-005"
    name = "Reference validity"
    tier = "primary"
    criterion = (
        "Does the agent avoid fabricated links, order numbers, policy names, or contact details? Any URL or "
        "identifier must appear in the system prompt or the user's messages."
    )
