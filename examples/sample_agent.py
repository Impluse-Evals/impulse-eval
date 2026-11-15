# SPDX-License-Identifier: Apache-2.0
"""Tiny echo-style agent for `impulse run --dry-run`, the docs, and the tests (SPEC.md section 8.2).

Generic adapter entrypoint: ``sample_agent:respond`` with the signature ``(user_text, history)``.
When the user text contains the whole word "book" (case-insensitive), the reply also carries one
dummy ``book_slot`` tool call with fixed values, so UNIV-023 and UNIV-025 can be exercised without
a real backend.
"""

import copy
import re

SYSTEM_PROMPT = (
    "You are the booking assistant of a demo clinic. Greet the caller and answer briefly. "
    "When the caller asks to book an appointment, call the book_slot tool with the date, "
    "the patient's name, and the phone number, and confirm the booking only if the tool succeeded."
)

_BOOK_WORD = re.compile(r"\bbook\b", re.IGNORECASE)

_BOOK_SLOT_TOOL_CALL = {
    "name": "book_slot",
    "arguments": {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"},
    "result": {"success": True},
    "success": True,
}


def respond(user_text, history):
    """Echo the user. ``history`` holds the prior turns only and is not used by this agent."""
    text = user_text or ""
    if not text:
        return "Hello, this is the demo clinic. How can I help you today?"
    reply = f"You said: {text}"
    if _BOOK_WORD.search(text):
        return {
            "text": reply + " I have booked a slot for you.",
            "tool_calls": [copy.deepcopy(_BOOK_SLOT_TOOL_CALL)],
        }
    return reply
