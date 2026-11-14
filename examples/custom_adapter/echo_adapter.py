# SPDX-License-Identifier: AGPL-3.0-only
"""A tiny custom adapter (``adapter: custom``) with no network: it echoes the user, like
``examples/sample_agent.py``, and returns one dummy ``book_slot`` tool call when the user text
contains the whole word "book".

Entrypoint: ``echo_adapter:EchoAdapter`` (config.yaml in this folder). Impulse creates one instance
per scenario with ``EchoAdapter(config)``, where ``config`` is the loaded ``ProjectConfig``; your
own settings (an assistant ID, the NAME of the environment variable that holds a key, never the key
itself) go under ``agent.options`` in config.yaml and arrive as ``config.agent.options``.

Where a real client goes: to test an agent hosted on a voice platform (Vapi, Retell, ...) or an
in-house service, keep this class shape and replace the bodies with calls to that platform's own
client library or API, as its documentation describes:

- ``__init__``: read ``config.agent.options`` and the key variable it names; create the client.
- ``start(scenario)``: open a conversation (a chat session, a web call, a test call) for the
  scenario, using ``scenario.name`` and ``scenario.language``.
- ``send(user_text, user_audio)``: send the user's turn and wait for the agent's complete reply;
  return an ``AgentReply`` with the text, the latency you measured, and the tool calls the platform
  reports (as ``ToolCall`` objects), so UNIV-023 and UNIV-025 can check them. ``send(None, None)``
  asks for the agent's opener (outbound scenarios).
- ``close()``: end the conversation and release the client.

Raise ``AgentClosed`` from ``send()`` when the agent hung up, and ``AgentTurnError`` (with the turn
so far as its ``reply``) when a turn failed after the agent produced something.
"""

import re
import time

from impulse_evals.adapters.base import AgentInterface, AgentReply
from impulse_evals.models import ToolCall

SYSTEM_PROMPT = (
    "You are the booking assistant of a demo clinic. Greet the caller and answer briefly. "
    "When the caller asks to book an appointment, call the book_slot tool with the date, "
    "the patient's name, and the phone number, and confirm the booking only if the tool succeeded."
)

_BOOK_WORD = re.compile(r"\bbook\b", re.IGNORECASE)

_BOOKING_ARGUMENTS = {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}


class EchoAdapter(AgentInterface):
    """Echoes every user turn; books on "book"."""

    def __init__(self, config):
        self.config = config
        self.options = dict(config.agent.options)   # free-form settings from config.yaml agent.options
        self.scenario = None

    async def start(self, scenario):
        """A real adapter opens the platform's conversation here."""
        self.scenario = scenario

    async def send(self, user_text, user_audio):
        """A real adapter sends the turn here and waits for the agent's complete reply."""
        started = time.perf_counter()
        text = user_text or ""
        if not text:
            reply_text = "Hello, this is the demo clinic. How can I help you today?"
        else:
            reply_text = f"You said: {text}"
        tool_calls = []
        if _BOOK_WORD.search(text):
            reply_text += " I have booked a slot for you."
            tool_calls.append(
                ToolCall(name="book_slot", arguments=dict(_BOOKING_ARGUMENTS), result={"success": True}, success=True)
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return AgentReply(text=reply_text, latency_ms=latency_ms, tool_calls=tool_calls)

    async def close(self):
        """A real adapter ends the conversation and releases its client here."""
        self.scenario = None

    def get_system_prompt(self):
        """Used only when config.yaml sets no agent.system_prompt."""
        return SYSTEM_PROMPT
