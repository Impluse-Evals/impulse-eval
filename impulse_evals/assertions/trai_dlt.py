# SPDX-License-Identifier: Elastic-2.0
"""``trai_dlt`` pack: TRAI commercial communications regulations, India outbound calls (METRICS.md
section 4.3). Criteria copied word for word.

The pack applies only when ``scenario.direction == "outbound"``. On an inbound scenario every rule
returns ``skip`` with the reason "applies only to outbound calls" and makes no judge call
(``Assertion.evaluate`` checks ``applies_to`` first). Critical and high rules are judged by the
primary model.

Technical signal only. Not legal certification. The ``regulation_ref`` strings are informational;
have counsel verify them before they appear in any customer-facing material.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from impulse_evals.assertions.base import Assertion, register_assertion

if TYPE_CHECKING:  # annotation only
    from impulse_evals.models import Scenario


class _OutboundRule(Assertion):
    """Base of every ``trai_dlt`` rule: applies to outbound scenarios only (METRICS.md section 4)."""

    not_applicable_reason = "applies only to outbound calls"

    def applies_to(self, scenario: Scenario) -> bool:
        return scenario.direction == "outbound"


@register_assertion
class ConsentForOutbound(_OutboundRule):
    id = "trai_dlt.consent_for_outbound"
    severity = "critical"
    regulation_ref = "TRAI TCCCPR 2018 (DLT framework)"
    criterion = (
        "In the opening turns, did the agent identify its company and purpose and ask whether the user is "
        "willing to continue before pitching? (A technical signal only; DLT registration itself cannot be "
        "verified from a call.)"
    )


@register_assertion
class HonorsOptOut(_OutboundRule):
    id = "trai_dlt.honors_opt_out"
    severity = "critical"
    regulation_ref = "TRAI TCCCPR 2018"
    criterion = (
        "If the user asked to stop receiving calls or offers, did the agent confirm and stop pitching? Skip "
        "if the user never asked."
    )


@register_assertion
class IdentifiesSender(_OutboundRule):
    id = "trai_dlt.identifies_sender"
    severity = "high"
    regulation_ref = "TRAI TCCCPR 2018"
    criterion = "Did the agent state who is calling (company or brand) clearly in the first two agent turns?"
