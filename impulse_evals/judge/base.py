# SPDX-License-Identifier: Apache-2.0
"""Judge plugin API (SPEC.md sections 12 and 18.1).

A judge reads one transcript and one criterion and returns a ``JudgeVerdict``. Metrics and
assertions call ``evaluate``; the runner reads ``get_warnings`` once the run is over.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel

if TYPE_CHECKING:  # annotation only: the plugin API has no runtime dependency on the core
    from impulse_evals.models import CallTranscript


class JudgeVerdict(BaseModel):
    """One judge verdict.

    ``no_verdict`` is True only for the skips a judge returns when it got no usable answer
    (empty content, or invalid JSON after the retry). A ``"skip"`` the judge itself chose is a
    valid verdict and keeps ``no_verdict=False``.
    """

    result: Literal["pass", "fail", "skip"]
    reason: str
    turn_index: Optional[int] = None      # 1-based position in CallTranscript.turns (both speakers)
    confidence: float = 0.0
    no_verdict: bool = False


class Judge(ABC):
    """Base class for judges. Subclasses implement ``evaluate``."""

    @abstractmethod
    async def evaluate(
        self,
        transcript: CallTranscript,
        criterion: str,
        tier: Literal["primary", "fast"],
    ) -> JudgeVerdict:
        """Judge ``transcript`` against ``criterion`` with the ``tier`` model (primary or fast)."""

    def get_warnings(self) -> list[str]:
        """Run-level warnings raised while judging; the runner adds them to ``RunResult.warnings``."""
        return []
