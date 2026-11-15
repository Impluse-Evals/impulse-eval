# SPDX-License-Identifier: Apache-2.0
"""Time seam shared by the network adapters (SPEC.md section 9).

Every timeout and grace window of the LiveKit and Pipecat adapters goes through a ``Clock``, so tests
can pass a virtual clock and run instantly. ``SystemClock`` is the real one (``time.monotonic`` and
``asyncio.sleep``). ``next_before`` races one ``recv()`` against the clock, never against real-time
``asyncio.wait_for``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Optional, Protocol, TypeVar

__all__ = ["Clock", "SystemClock", "next_before"]

T = TypeVar("T")


class Clock(Protocol):
    """Monotonic seconds plus a sleep that honours them (SPEC.md section 9)."""

    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real clock: ``time.monotonic()`` and ``asyncio.sleep()``."""

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, float(seconds)))


async def next_before(recv: Callable[[], Awaitable[T]], clock: Clock, deadline: Optional[float]) -> Optional[T]:
    """The next item from ``recv()``, or None when the clock reaches ``deadline`` first.

    ``deadline`` is a ``clock.monotonic()`` value; None waits without limit. An item that arrives
    together with the deadline is returned, never dropped: ``recv()`` is cancelled only while it has
    not produced an item. ``recv()`` must therefore be cancel-safe (an ``asyncio.Queue.get()`` is).
    """
    receiving = asyncio.ensure_future(recv())
    if deadline is None:
        return await receiving
    timer = asyncio.ensure_future(clock.sleep(max(0.0, deadline - clock.monotonic())))
    try:
        await asyncio.wait({receiving, timer}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        receiving.cancel()
        timer.cancel()
        raise
    if receiving.done():
        timer.cancel()
        return receiving.result()
    receiving.cancel()
    try:
        return await receiving   # it may have produced an item while being cancelled
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise   # our own task is being cancelled, not only the receive
        return None
