# SPDX-License-Identifier: Apache-2.0
"""Adapter plugin API (SPEC.md section 9).

An adapter connects Impulse to the agent under test. The runner calls ``start(scenario)`` once per
scenario (one conversation per scenario), then ``send(...)`` once per user turn, then ``close()``.

- Text mode: ``send(user_text, None)``. Voice mode: ``send(user_text, user_audio)``; adapters that
  consume audio use the audio and ignore the text.
- Outbound first turn: ``send(None, None)`` means "wait for the agent's opener".
- ``AgentReply.heard_text`` is stored on the preceding USER turn; ``AgentReply.interrupted`` and
  ``AgentReply.tool_calls`` are stored on the AGENT turn.
- ``get_system_prompt()`` is consulted only when ``agent.system_prompt`` is not set in config.yaml.
- ``get_warnings()``: run-level warnings; the runner adds them (deduplicated) to ``RunResult.warnings``.
- ``get_metadata()``: per-scenario data; the runner merges it into ``CallTranscript.metadata`` after
  ``close()``.
- ``AgentClosed``: raised by ``send()`` when the agent side closed the conversation (hung up). The
  runner ends the conversation normally and evaluates the turns so far (SPEC.md section 10, "when
  the adapter closes"); before the agent's first turn (no agent turn yet and no text, tool call or
  audio in ``reply``) it is the scenario error "the agent closed the conversation before its first
  turn: {message}" instead. Any other exception from the adapter is a scenario error. ``close()`` is
  still called afterwards and must tolerate the closed connection.
- ``AgentTurnError``: an adapter error during a turn whose ``reply`` carries the turn so far; the
  runner records that partial turn (when it has text, a tool call, or audio) before the scenario
  error.

Plugin API (SPEC.md section 5.1): this file imports no core module outside the plugin API:
``ToolCall`` comes from ``impulse_evals.plugin_types``, and ``Scenario`` is annotation-only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from impulse_evals.plugin_types import ToolCall

if TYPE_CHECKING:  # annotation only: the plugin API has no runtime dependency on the core
    from impulse_evals.models import Scenario

__all__ = ["AgentClosed", "AgentInterface", "AgentReply", "AgentTurnError"]


class AgentReply(BaseModel):
    """One agent turn as the adapter observed it."""

    text: str
    audio_bytes: Optional[bytes] = None
    latency_ms: float
    heard_text: Optional[str] = None
    interrupted: bool = False
    tool_calls: list[ToolCall] = []       # tool calls made during this reply, if the adapter can expose them


class AgentClosed(Exception):
    """Raised by ``send()`` when the agent side closed the conversation (for example, it hung up).

    The runner treats it as a normal end of the conversation, not as a scenario error, once the agent
    has taken a turn (before that, it is a scenario error; SPEC.md section 10). ``reply`` is the
    agent's last turn received before the close (for example a goodbye), or None; the runner records
    it like any other reply.
    """

    def __init__(self, message: str = "the agent closed the conversation", *, reply: Optional[AgentReply] = None):
        super().__init__(message)
        self.reply = reply


class AgentTurnError(Exception):
    """An adapter error during a turn that still carries what the agent produced so far.

    ``reply`` is the turn so far (text received, the turn's tool calls, ``interrupted``, audio,
    ``latency_ms`` or NaN when no output arrived), or None. When ``reply`` has text, a tool call, or
    audio, the runner first records it as an agent turn like a normal reply, then sets the scenario
    error from the exception as for any other adapter error (no evaluation). An adapter may raise a
    subclass that is also its own error class (for example the LiveKit turn timeout).
    """

    def __init__(self, message: str, *, reply: Optional[AgentReply] = None):
        super().__init__(message)
        self.reply = reply


class AgentInterface(ABC):
    """Base class of every adapter. Subclasses implement ``start``, ``send``, and ``close``."""

    @abstractmethod
    async def start(self, scenario: Scenario) -> None:
        """Begin a new conversation for ``scenario``."""

    @abstractmethod
    async def send(self, user_text: Optional[str], user_audio: Optional[bytes]) -> AgentReply:
        """Send one user turn and return the agent's reply (``send(None, None)``: wait for the opener)."""

    @abstractmethod
    async def close(self) -> None:
        """End the conversation started by ``start``."""

    def get_system_prompt(self) -> Optional[str]:
        """The agent's own system prompt, if it exposes one (used only when config.yaml sets none)."""
        return None

    def get_warnings(self) -> list[str]:
        """Run-level warnings; the runner adds them (deduplicated) to ``RunResult.warnings``."""
        return []

    def get_metadata(self) -> dict:
        """Per-scenario data; the runner merges it into ``CallTranscript.metadata`` after ``close()``."""
        return {}
