# SPDX-License-Identifier: Apache-2.0
"""Domain tables (SPEC.md section 11): default assertions, metrics, and ``max_turns`` per domain,
plus the domain-guard keyword lists and ``detect_domain``.

``pack.*`` wildcards in the SPEC.md table are expanded here into explicit ``pack.rule_id`` lists
from ``config.PACK_RULE_IDS``, without importing any pack module. The ``hipaa`` pack is opt-in and
appears in no domain default. The India packs (``INDIA_PACKS``: dpdp, rbi_kyc, trai_dlt) are added
only to scenarios whose language has the region IN (``is_india_locale``) when a language is given;
any pack can still be listed explicitly for any language. UNIV-025 is in no domain default (it runs automatically when a
scenario defines tool expectations). A domain that is not in the table is treated as ``general``
with the warning "Unknown domain {domain}; using general defaults."
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal, Optional

from pydantic import BaseModel

from impulse_evals.config import KNOWN_METRIC_IDS, PACK_RULE_IDS

UNKNOWN_DOMAIN_WARNING = "Unknown domain {domain}; using general defaults."
GENERAL = "general"

# Packs the domain defaults add only to scenarios whose language has the region IN (maintainer
# decision 2026-10-26, "Neutral defaults"); any pack can still be listed explicitly for any language.
INDIA_PACKS: tuple[str, ...] = ("dpdp", "rbi_kyc", "trai_dlt")


def is_india_locale(language: str) -> bool:
    """True when any subtag after the first of ``language`` is ``IN`` (case-insensitive):
    ``en-IN``, ``hi-IN``, ``hi-Latn-IN``, ``ta-in``; False for ``hi``, ``en-US``, ``in``."""
    return any(subtag.upper() == "IN" for subtag in str(language).strip().split("-")[1:])


def expand_assertion_patterns(patterns: Iterable[str]) -> list[str]:
    """Explicit ``pack.rule_id`` strings for ``patterns``; ``pack.*`` becomes every rule of the pack.

    Explicit IDs are kept as given; duplicates are removed (first occurrence kept). An unknown pack in
    a wildcard raises ``ValueError``. No pack module is imported.
    """
    expanded: list[str] = []
    for pattern in patterns:
        pack, sep, rule = pattern.partition(".")
        if sep and rule == "*":
            if pack not in PACK_RULE_IDS:
                raise ValueError(f"unknown compliance pack in {pattern}")
            expanded.extend(f"{pack}.{name}" for name in PACK_RULE_IDS[pack])
        else:
            expanded.append(pattern)
    return list(dict.fromkeys(expanded))


class DomainDefaults(BaseModel):
    """One row of the SPEC.md section 11 table, with wildcards already expanded to explicit IDs."""

    assertions: list[str]            # explicit pack.rule_id added to every scenario of the domain
    outbound_assertions: list[str]   # explicit pack.rule_id added only to outbound scenarios
    metrics: list[str]               # metric IDs always included
    max_turns: int

    def assertions_for(
        self, direction: Literal["inbound", "outbound"] = "inbound", language: Optional[str] = None
    ) -> list[str]:
        """Explicit default assertions for a scenario with this ``direction``.

        ``language`` given and not an India locale (``is_india_locale``): the India packs
        (``INDIA_PACKS``) are left out. ``language=None`` keeps every default of the row."""
        extra = self.outbound_assertions if direction == "outbound" else []
        rules = list(dict.fromkeys([*self.assertions, *extra]))
        if language is not None and not is_india_locale(language):
            rules = [rule for rule in rules if rule.partition(".")[0] not in INDIA_PACKS]
        return rules


# All universal judge metrics (UNIV-001 through UNIV-023; not UNIV-025) + VOICE-037 + VOICE-039.
GENERAL_METRICS: list[str] = [
    metric_id for metric_id in KNOWN_METRIC_IDS if metric_id.startswith("UNIV-") and metric_id != "UNIV-025"
] + ["VOICE-037", "VOICE-039"]

# domain: (auto assertions, outbound-only assertions, always-included metrics, max_turns), as in
# SPEC.md section 11. Rows are in the order of the table.
_TABLE: dict[str, tuple[list[str], list[str], list[str], int]] = {
    "fsi": (["rbi_kyc.*", "dpdp.*"], ["trai_dlt.*"], ["UNIV-001", "UNIV-005", "UNIV-019", "UNIV-023"], 12),
    "healthcare": (["dpdp.*"], [], ["UNIV-005", "UNIV-011", "UNIV-019", "UNIV-023"], 12),
    "recruitment": (["dpdp.*"], [], ["UNIV-013", "UNIV-019", "UNIV-023"], 9),
    "ecommerce": ([], [], ["UNIV-001", "UNIV-003", "UNIV-007", "UNIV-023"], 9),
    "edtech": ([], [], ["UNIV-005", "UNIV-013", "UNIV-017"], 9),
    "telecom": ([], ["trai_dlt.*"], ["UNIV-001", "UNIV-011", "UNIV-023"], 9),
    "realestate": (["dpdp.*"], [], ["UNIV-005", "UNIV-007"], 9),
    "travel": ([], [], ["UNIV-001", "UNIV-003", "UNIV-007", "UNIV-023"], 9),
    "legal": (["dpdp.*"], [], ["UNIV-005", "UNIV-019"], 9),
    GENERAL: ([], [], GENERAL_METRICS, 9),
}

DOMAIN_DEFAULTS: dict[str, DomainDefaults] = {
    domain: DomainDefaults(
        assertions=expand_assertion_patterns(assertions),
        outbound_assertions=expand_assertion_patterns(outbound),
        metrics=list(metrics),
        max_turns=max_turns,
    )
    for domain, (assertions, outbound, metrics, max_turns) in _TABLE.items()
}


def resolve_domain(domain: str) -> tuple[str, Optional[str]]:
    """``(domain, None)`` for a domain in the table, else ``("general", warning)``."""
    if domain in DOMAIN_DEFAULTS:
        return domain, None
    return GENERAL, UNKNOWN_DOMAIN_WARNING.format(domain=domain)


def get_domain_defaults(domain: str) -> DomainDefaults:
    """The defaults row for ``domain`` (an unknown domain gives the ``general`` row)."""
    return DOMAIN_DEFAULTS[resolve_domain(domain)[0]]


def default_assertions(
    domain: str, direction: Literal["inbound", "outbound"] = "inbound", language: Optional[str] = None
) -> list[str]:
    """Explicit default ``pack.rule_id`` list for a scenario of ``domain``, ``direction`` and
    ``language`` (a language whose region is not IN gets no India pack; None keeps every default)."""
    return get_domain_defaults(domain).assertions_for(direction, language=language)


def default_metrics(domain: str) -> list[str]:
    """Metric IDs always included for ``domain``."""
    return list(get_domain_defaults(domain).metrics)


def default_max_turns(domain: str) -> int:
    """Default ``max_turns`` (USER turns) for ``domain``: 12 for fsi and healthcare, otherwise 9."""
    return get_domain_defaults(domain).max_turns


# ------------------------------------------------------------------------------------------------
# Domain guard (SPEC.md section 11)
# ------------------------------------------------------------------------------------------------

# Keyword lists, whole-word and case-insensitive, in tie-breaking order.
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fsi": ("loan", "cibil", "kyc", "pan", "emi", "nach", "banking", "credit", "interest rate"),
    "healthcare": ("patient", "diagnosis", "prescription", "symptom", "doctor", "appointment", "medicine"),
    "recruitment": ("interview", "candidate", "resume", "hiring", "recruiter", "job description", "shortlisting"),
    "ecommerce": ("order", "return", "refund", "product", "delivery", "cart", "shipping"),
    "edtech": ("lesson", "quiz", "student", "course", "tutor", "homework", "curriculum"),
    "telecom": ("recharge", "postpaid", "prepaid", "operator", "data pack", "network", "sim card"),
    "realestate": ("property", "apartment", "rent", "bhk", "site visit", "landlord", "builder"),
    "travel": ("flight", "hotel", "booking", "ticket", "itinerary", "boarding", "visa"),
    "legal": ("legal", "rights", "clause", "contract", "dispute", "lawyer", "notice"),
}

MIN_KEYWORD_HITS = 3


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    words = r"\s+".join(re.escape(word) for word in keyword.split())
    return re.compile(rf"(?<!\w){words}(?!\w)", re.IGNORECASE)


_KEYWORD_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    domain: tuple(_keyword_pattern(keyword) for keyword in keywords)
    for domain, keywords in DOMAIN_KEYWORDS.items()
}


def keyword_hits(text: str) -> dict[str, int]:
    """Number of DISTINCT keywords of each domain found in ``text`` (a repeated keyword counts once)."""
    text = text or ""
    return {
        domain: sum(1 for pattern in patterns if pattern.search(text))
        for domain, patterns in _KEYWORD_PATTERNS.items()
    }


def detect_domain(text: str) -> Optional[str]:
    """The domain with the most distinct keyword hits (at least 3), or None.

    Ties are broken by the order of ``DOMAIN_KEYWORDS`` (fsi first, legal last).
    """
    best: Optional[str] = None
    best_hits = MIN_KEYWORD_HITS - 1
    for domain, hits in keyword_hits(text).items():
        if hits > best_hits:
            best, best_hits = domain, hits
    return best
