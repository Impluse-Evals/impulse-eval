# SPDX-License-Identifier: Apache-2.0
"""Retell connector: stub only, planned for v2 (SPEC.md section 2). Every entry point raises NotImplementedError."""

from __future__ import annotations

from typing import Any, NoReturn, Optional

from impulse_evals.adapters.base import AgentInterface

NOT_IMPLEMENTED_MESSAGE = "The Retell adapter is planned for v2. Use agent.adapter: generic, livekit, or pipecat."


class RetellAdapter(AgentInterface):
    """Stub: raises NotImplementedError."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    async def start(self, scenario: Any) -> NoReturn:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    async def send(self, user_text: Optional[str], user_audio: Optional[bytes]) -> NoReturn:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    async def close(self) -> NoReturn:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
