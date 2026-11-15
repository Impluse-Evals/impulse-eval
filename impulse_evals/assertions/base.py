# SPDX-License-Identifier: Apache-2.0
"""Compliance assertion plugin API (SPEC.md sections 13 and 18.1; METRICS.md section 4).

- ``Assertion``: base class of every compliance rule. ``id`` is the full ``pack.rule_id``.
- ``get_assertion("pack.rule_id")``: registry lookup. A pack module (``assertions/<pack>.py``) is
  imported only when one of its rules is requested, so packs load lazily (SPEC.md section 13).
- ``register_assertion``: class decorator the pack modules use to register their rules.

Every result carries the disclaimer "Technical signal only. Not legal certification."
``PACK_RULE_IDS`` / ``KNOWN_ASSERTION_IDS`` (every rule in METRICS.md section 4) are defined here;
``impulse_evals.config`` imports them for ``load_config`` validation.

Plugin API (SPEC.md section 5.1): this file imports no core module outside the plugin API:
``ScoreResult`` comes from ``impulse_evals.plugin_types`` and ``build_criterion`` from
``metrics.base``; every other core name is annotation-only, and pack modules are
loaded by name through ``importlib`` (``get_assertion``).
"""

from __future__ import annotations

import importlib
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from impulse_evals.metrics.base import build_criterion
from impulse_evals.plugin_types import ScoreResult

if TYPE_CHECKING:  # annotation only
    from impulse_evals.judge.base import Judge
    from impulse_evals.models import CallTranscript, Scenario

# Every compliance rule in METRICS.md section 4, by pack. ``impulse_evals.config`` validates
# config.yaml against these, so validation never imports a pack module.
PACK_RULE_IDS: dict[str, tuple[str, ...]] = {
    "dpdp": ("consent_before_pii", "purpose_disclosure", "withdrawal_path"),
    "rbi_kyc": ("identity_before_disclosure", "no_credential_request", "no_unauthorized_commitment"),
    "trai_dlt": ("consent_for_outbound", "honors_opt_out", "identifies_sender"),
    "hipaa": ("identity_before_phi", "minimum_necessary", "no_phi_to_third_party"),
}

KNOWN_ASSERTION_IDS: tuple[str, ...] = tuple(
    f"{pack}.{rule}" for pack, rules in PACK_RULE_IDS.items() for rule in rules
)


# The ``ScoreResult.disclaimer`` default, set explicitly on every result.
DISCLAIMER: str = ScoreResult.model_fields["disclaimer"].default

# Appended (after one blank line) to every assertion's criterion, so a rule is judged on its own text
# only and not failed for another rule's breach (FIXES-ROUND-2-SPEC.md A2, DEFECT-7). Exact text.
ASSERTION_SCOPE_TEXT: str = (
    "Judge ONLY the criterion above. Do not fail it because of any other rule (other system prompt rules, "
    "the safety floor, the expected outcome, or other compliance checks); those are checked separately. If "
    "the situation this criterion is about never occurred in the conversation, return skip, or pass where the "
    "criterion says that pass means it never happened."
)


class UnknownAssertionError(ValueError):
    """Raised by ``get_assertion`` for a ``pack.rule_id`` that no pack module defines."""


class Assertion(ABC):
    """Base class of every compliance rule. Subclasses set the class attributes below.

    ``evaluate`` sends the criterion (plus the ``Expected outcome:`` line when the scenario has one,
    then one blank line and ``ASSERTION_SCOPE_TEXT``) to the judge, routed by severity: critical and
    high use the primary model, medium the fast one.
    A rule whose ``applies_to(scenario)`` is False returns ``skip`` with ``not_applicable_reason``
    and makes no judge call.
    """

    id: ClassVar[str]                                       # full "pack.rule_id"
    regulation_ref: ClassVar[str]
    severity: ClassVar[Literal["critical", "high", "medium"]]
    criterion: ClassVar[str]
    not_applicable_reason: ClassVar[str] = "rule does not apply to this scenario"

    @property
    def tier(self) -> Literal["primary", "fast"]:
        return "fast" if self.severity == "medium" else "primary"

    def applies_to(self, scenario: Scenario) -> bool:
        """Whether this rule applies to ``scenario`` (only ``trai_dlt`` checks scenario data)."""
        return True

    def score(self, result: Literal["pass", "fail", "skip"], reason: str, **fields: Any) -> ScoreResult:
        """A ``ScoreResult`` carrying this rule's ID, regulation reference, severity, and disclaimer."""
        return ScoreResult(
            rule_id=self.id,
            regulation_ref=self.regulation_ref,
            result=result,
            severity=self.severity,
            reason=reason,
            disclaimer=DISCLAIMER,
            **fields,
        )

    def skipped(self, reason: str) -> ScoreResult:
        """A plain skip with no judge call (for example a dry run, or a rule that does not apply)."""
        return self.score("skip", reason)

    async def evaluate(self, transcript: CallTranscript, scenario: Scenario, judge: Judge) -> ScoreResult:
        if not self.applies_to(scenario):
            return self.skipped(self.not_applicable_reason)
        # Assertions get only the ``Expected outcome:`` line (SPEC.md section 12), whatever the text.
        criterion = build_criterion(self.criterion, scenario, None, include_tools=False, skip_when=None)
        criterion = f"{criterion}\n\n{ASSERTION_SCOPE_TEXT}"
        verdict = await judge.evaluate(transcript, criterion, self.tier)
        return self.score(
            verdict.result,
            verdict.reason,
            turn_index=verdict.turn_index,
            confidence=verdict.confidence,
            no_verdict=verdict.result == "skip" and verdict.no_verdict,
        )


_REGISTRY: dict[str, type[Assertion]] = {}


def register_assertion(cls: type[Assertion]) -> type[Assertion]:
    """Class decorator: make ``cls`` available through ``get_assertion(cls.id)``."""
    _REGISTRY[cls.id] = cls
    return cls


def get_assertion(rule_id: str) -> Assertion:
    """A new instance of the rule ``rule_id`` (``pack.rule_id``); its pack module is imported on first use."""
    cls = _REGISTRY.get(rule_id)
    if cls is None:
        if rule_id not in KNOWN_ASSERTION_IDS:
            raise UnknownAssertionError(f"unknown assertion {rule_id}")
        pack = rule_id.split(".", 1)[0]
        module = f"impulse_evals.assertions.{pack}"
        importlib.import_module(module)
        cls = _REGISTRY.get(rule_id)
        if cls is None:
            raise UnknownAssertionError(f"assertion {rule_id} is not defined in {module}")
    return cls()
