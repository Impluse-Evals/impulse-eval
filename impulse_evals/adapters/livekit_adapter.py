# SPDX-License-Identifier: Apache-2.0
"""LiveKit adapter (SPEC.md sections 8.2 and 9; optional extra ``livekit``).

Impulse joins one new LiveKit room per scenario as a plain client participant (never through
``livekit-agents``), named ``{room_prefix}{run_id}-{scenario_name}``. With ``livekit.agent_name`` set,
the access token dispatches that agent into the room.

Testability seam (SPEC.md section 9): all LiveKit SDK, token, and protobuf code lives in the real room
class ``LiveKitRoom``; ``LiveKitAdapter`` talks only to the ``Room`` protocol below and exchanges plain
Python values with it, so the tests drive it with a scripted fake room and a virtual ``Clock``. Every
``livekit`` import is lazy and happens only when the real room is used.

How a conversation maps onto the room:

- Agent: the participant ``livekit.agent_identity``, else the first agent participant that joins. No
  agent within ``handshake_timeout_s`` is a scenario error ("LiveKit agent did not join within {N} s.").
- System prompt: only when ``agent.system_prompt`` is not set in config.yaml, one ``GetAgentInfo``
  request on the agent's ``lk.agent.session`` stream; ``GetAgentInfoResponse.instructions`` within 3 s,
  otherwise None.
- Text turns: the user text goes out on the text-stream topic ``lk.chat``; the reply is the final
  ``lk.transcription`` streams sent by the agent (``lk.transcription_final == "true"``), joined with a
  space. Latency: from sending the text to the first such stream.
- Voice turns: the user audio is published as microphone audio (``Room.publish_audio``; no text is
  sent); the agent's audio, from its first non-silent frame (peak above 1 % of full scale) received
  after the user's last speech sample, is collected into one WAV. Latency (TTFA): from the time
  ``publish_audio`` returned (the last non-silent user sample left Impulse) to that first frame.
- ``heard_text``: final ``lk.transcription`` streams sent with Impulse's own identity during the turn.
- ``tool_calls``: ``function_tools_executed`` events of the agent's ``lk.agent.session`` stream (SPEC.md
  section 9 mapping, see ``tool_calls_from_event``).
- ``interrupted``: the agent's ``lk.agent.state`` attribute leaves ``"speaking"`` while Impulse is
  publishing user speech (between the start of ``publish_audio`` and its return).
- End of turn: the agent produced output after the user's input was sent (its ``lk.agent.state``
  entered or left ``"speaking"``, left ``"thinking"``, or a final reply stream arrived), is neither
  ``"speaking"`` nor ``"thinking"`` (a running tool keeps the agent ``"thinking"``), and stays so for
  ``livekit.end_of_turn_grace_ms``. No end of turn within the turn timeout is a scenario error: the
  timeout is ``livekit.turn_timeout_s`` when set (text and voice mode), else ``voice_config.timeout_s``
  when ``voice_config`` exists, else 30 s. The error (``LiveKitTurnTimeout``) is also an
  ``AgentTurnError`` carrying the turn so far (text, tool calls, interrupted, audio; latency NaN when
  no output arrived), so the runner keeps the turn's tool calls. State changes that arrived while user
  speech was being published never end a turn.
- Items the room delivered before the turn started (for example a greeting on an inbound call, or a
  late segment of the previous reply) update the agent's state; their reply text and tool calls are
  kept in the next reply, their audio and user transcriptions are not.
- ``RoomClosed`` (the agent left, or the room ended) makes ``send()`` raise ``AgentClosed``: a normal end
  of the conversation, with the agent's last words attached when there were any (before the agent's
  first turn the runner makes it a scenario error, SPEC.md section 10).
- The connect error shows ``livekit.url`` without its credentials (``strip_url_credentials``); the room
  itself connects with the full URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import secrets
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional, Protocol, Union

from impulse_evals.adapters._clock import Clock, SystemClock, next_before
from impulse_evals.adapters.base import (
    AgentClosed,
    AgentInterface,
    AgentReply,
    AgentTurnError,
)
from impulse_evals.config import (
    ConfigError,
    LiveKitConfig,
    ProjectConfig,
    TomlConfig,
    load_project_toml,
    read_system_prompt,
    resolve_key,
    strip_url_credentials,
)
from impulse_evals.models import Scenario, ToolCall

__all__ = [
    "AGENT_INFO_TIMEOUT_S",
    "JOIN_TIMEOUT_ERROR",
    "MISSING_EXTRA",
    "TURN_TIMEOUT_ERROR",
    "Clock",
    "LiveKitAdapter",
    "LiveKitError",
    "LiveKitRoom",
    "LiveKitTurnTimeout",
    "Room",
    "RoomAttributes",
    "RoomAudio",
    "RoomClosed",
    "RoomParticipant",
    "RoomSessionEvent",
    "RoomText",
    "tool_calls_from_event",
]

# ------------------------------------------------------------------------------------------------
# Protocol names and exact texts (SPEC.md section 9; DECISIONS.md source read)
# ------------------------------------------------------------------------------------------------

TOPIC_CHAT = "lk.chat"
TOPIC_TRANSCRIPTION = "lk.transcription"
TOPIC_AGENT_SESSION = "lk.agent.session"
ATTRIBUTE_AGENT_STATE = "lk.agent.state"
ATTRIBUTE_TRANSCRIPTION_FINAL = "lk.transcription_final"
SESSION_AGENT_INFO = "agent_info"
SESSION_TOOLS_EXECUTED = "function_tools_executed"

MISSING_EXTRA = "livekit adapter needs the 'livekit' extra: pip install impulse-evals[livekit]"
JOIN_TIMEOUT_ERROR = "LiveKit agent did not join within {seconds:g} s."
TURN_TIMEOUT_ERROR = "LiveKit agent did not finish its turn within {seconds:g} s."

AGENT_INFO_TIMEOUT_S = 3.0
DEFAULT_TURN_TIMEOUT_S = 30.0
SPEECH_PEAK_FRACTION = 0.01          # agent audio counts as speech above 1 % of full scale
_BUSY_STATES = ("speaking", "thinking")
_SPEAKING = "speaking"

# ------------------------------------------------------------------------------------------------
# The Room seam (SPEC.md section 9, LiveKit implementation note)
# ------------------------------------------------------------------------------------------------


@dataclass
class RoomParticipant:
    """A participant joined (or was already in the room when Impulse joined)."""

    identity: str
    is_agent: bool
    attributes: dict


@dataclass
class RoomText:
    """One complete text stream; ``final`` is ``lk.transcription_final == "true"``."""

    topic: str
    sender_identity: str
    text: str
    final: bool
    arrival_s: float


@dataclass
class RoomAttributes:
    """``participant_attributes_changed`` (for example ``lk.agent.state``)."""

    identity: str
    changed: dict
    arrival_s: float


@dataclass
class RoomSessionEvent:
    """A decoded ``lk.agent.session`` message: kind ``"agent_info"`` (data: ``instructions``, ``tools``)
    or ``"function_tools_executed"`` (data: ``function_calls``, ``function_call_outputs``)."""

    kind: str
    data: dict
    arrival_s: float


@dataclass
class RoomAudio:
    """Agent audio: mono 16-bit little-endian PCM."""

    pcm: bytes
    sample_rate: int
    arrival_s: float


@dataclass
class RoomClosed:
    """The room ended for Impulse (the agent left, or the connection closed)."""

    reason: str


RoomItem = Union[RoomParticipant, RoomText, RoomAttributes, RoomSessionEvent, RoomAudio, RoomClosed]


class Room(Protocol):
    """What the adapter needs from a LiveKit room. ``LiveKitRoom`` is the real one."""

    async def connect(self, url: str, identity: str, room_name: str, agent_name: Optional[str]) -> None: ...

    async def send_text(self, text: str, topic: str) -> None: ...

    async def publish_audio(self, pcm: bytes, sample_rate: int) -> float:
        """Publish mono 16-bit PCM as microphone audio, paced in real time; returns the clock time the
        last NON-SILENT sample left Impulse."""
        ...

    async def request_agent_info(self, agent_identity: str) -> None:
        """``GetAgentInfo`` on ``lk.agent.session``; the answer arrives as ``RoomSessionEvent("agent_info")``."""
        ...

    async def recv(self) -> RoomItem: ...

    async def close(self) -> None: ...


class LiveKitError(RuntimeError):
    """A LiveKit failure that is a scenario error (join or turn timeout, connection failure)."""


class LiveKitTurnTimeout(LiveKitError, AgentTurnError):
    """The turn timeout: a ``LiveKitError`` and an ``AgentTurnError`` whose ``reply`` is the turn so far
    (the runner records it before the scenario error, so the turn's tool calls are kept)."""

    def __init__(self, message: str, *, reply: Optional[AgentReply] = None):
        AgentTurnError.__init__(self, message, reply=reply)


# ------------------------------------------------------------------------------------------------
# Tool calls (SPEC.md section 9: function_tools_executed -> ToolCall)
# ------------------------------------------------------------------------------------------------


def _json_object(text: str) -> Optional[dict]:
    try:
        value = json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _decode_arguments(raw: Any) -> dict:
    """The JSON-decoded ``arguments`` string; invalid JSON, or JSON that is not an object, gives
    ``{"raw": <string>}``; a missing or blank string gives ``{}`` (no arguments)."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw is None:
        return {}
    text = raw if isinstance(raw, str) else str(raw)
    if not text.strip():
        return {}
    decoded = _json_object(text)
    return decoded if decoded is not None else {"raw": text}


def _output_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if raw is None:
        return ""
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(raw)


def tool_calls_from_event(data: Mapping[str, Any]) -> list[ToolCall]:
    """``ToolCall`` objects from one ``function_tools_executed`` event (SPEC.md section 9).

    Each entry of ``function_calls`` (in order) is paired by ``call_id`` with its
    ``function_call_outputs`` entry (none: ``result`` and ``success`` stay None). ``arguments`` = the
    JSON-decoded ``arguments`` string; ``result`` = the output parsed as JSON when it is a JSON object,
    else ``{"value": output}``; ``success`` = False with ``error=output`` when ``is_error`` is true, else
    ``result["success"]`` when it is a boolean, else None.
    """
    outputs: dict[Any, Mapping[str, Any]] = {}
    for output in data.get("function_call_outputs") or []:
        if isinstance(output, Mapping):
            outputs.setdefault(output.get("call_id"), output)
    calls: list[ToolCall] = []
    for call in data.get("function_calls") or []:
        if not isinstance(call, Mapping):
            continue
        name = call.get("name")
        tool = ToolCall(
            name=name if isinstance(name, str) and name else "unknown",
            arguments=_decode_arguments(call.get("arguments")),
        )
        output = outputs.get(call.get("call_id"))
        if output is not None:
            raw = output.get("output")
            text = _output_text(raw)
            parsed = dict(raw) if isinstance(raw, Mapping) else _json_object(text)
            tool.result = parsed if parsed is not None else {"value": text}
            if output.get("is_error"):
                tool.success = False
                tool.error = text or None
            elif isinstance(tool.result.get("success"), bool):
                tool.success = tool.result["success"]
        calls.append(tool)
    return calls


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


def make_room_run_id(when: Optional[datetime] = None) -> str:
    """A run ID in the runner's format (``20261004T120000Z-a3f9c2``) for rooms of an adapter that was
    not given the run's own ID."""
    when = when or datetime.now(UTC)
    return when.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def _speech_pcm(audio: bytes) -> tuple[bytes, int]:
    """User audio (WAV, or another format pydub can convert) as mono 16-bit PCM at its own rate."""
    from impulse_evals.voice import audio_utils

    wav = audio_utils.ensure_wav(bytes(audio))
    rate = audio_utils.wav_params(wav)[2]
    return audio_utils.pcm16_frames(audio_utils.convert_wav(wav, rate)), rate


def _is_speech(pcm: bytes) -> bool:
    from impulse_evals.voice import audio_utils

    return audio_utils.peak_fraction(pcm) > SPEECH_PEAK_FRACTION


def _resample_pcm(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    import numpy as np

    from impulse_evals.voice import audio_utils

    usable = len(pcm) - len(pcm) % 2
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
    return audio_utils.float_to_pcm16(audio_utils.resample(samples, from_rate, to_rate)).tobytes()


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    from impulse_evals.voice import audio_utils

    return audio_utils.pcm16_to_wav(pcm, sample_rate)


@dataclass
class _Turn:
    """What one ``send()`` collects. Times are ``Clock`` seconds."""

    window_start: float          # items that arrived before this belong to an earlier moment (stale)
    output_from: float           # agent activity at or after this can end the turn
    latency_start: float         # TTFA / latency is measured from here
    audio_from: float            # agent audio is collected from its first speech frame at or after this
    collect_audio: bool
    publish: Optional[tuple[float, float]] = None   # voice: (publish start, publish_audio returned)
    texts: list[str] = field(default_factory=list)
    heard: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    audio: list[bytes] = field(default_factory=list)
    audio_rate: Optional[int] = None
    first_audio_s: Optional[float] = None
    first_text_s: Optional[float] = None
    produced: bool = False
    idle_since: Optional[float] = None
    interrupted: bool = False

    def has_content(self) -> bool:
        return bool(self.texts or self.tool_calls or self.audio)


# ------------------------------------------------------------------------------------------------
# The adapter
# ------------------------------------------------------------------------------------------------


class LiveKitAdapter(AgentInterface):
    """Joins one LiveKit room per scenario and talks to the agent in it (SPEC.md section 9).

    ``room_factory`` is a plain callable (called with no arguments) that returns one ``Room`` per
    scenario; None means the real ``LiveKitRoom`` (needs the ``livekit`` extra and the API key and
    secret named by ``livekit.api_key_env`` / ``api_secret_env``). ``clock`` defaults to the system
    clock. ``run_id`` is the run's ID for the room names (the runner's default factory passes it);
    without it the adapter makes one in the same format.
    """

    def __init__(
        self,
        config: ProjectConfig,
        *,
        room_factory: Optional[Callable[[], Room]] = None,
        clock: Optional[Clock] = None,
        run_id: Optional[str] = None,
    ) -> None:
        settings = config.agent.livekit
        if settings is None or not settings.url:
            raise ConfigError("agent.livekit.url", "required when agent.adapter is livekit")
        self._config = config
        self._settings: LiveKitConfig = settings
        self._url: str = settings.url
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._run_id = run_id or make_room_run_id()
        self._voice = config.voice_config is not None
        if settings.turn_timeout_s is not None:
            self._turn_timeout_s = float(settings.turn_timeout_s)
        elif config.voice_config is not None:
            self._turn_timeout_s = float(config.voice_config.timeout_s)
        else:
            self._turn_timeout_s = DEFAULT_TURN_TIMEOUT_S
        self._grace_s = max(0.0, float(settings.end_of_turn_grace_ms) / 1000.0)
        self._join_timeout_s = max(0.0, float(settings.handshake_timeout_s))
        wanted = (settings.agent_identity or "").strip()
        self._wanted_agent: Optional[str] = wanted or None
        self._identity = f"impulse-user-{secrets.token_hex(4)}"
        self._room_factory: Callable[[], Room] = (
            room_factory if room_factory is not None else _real_room_factory(config, settings, self._clock)
        )
        self._config_prompt: Optional[str] = None
        self._config_prompt_loaded = False
        self._reset()

    def _reset(self) -> None:
        self._room: Optional[Room] = None
        self._room_name: Optional[str] = None
        self._agent: Optional[str] = None
        self._agent_state: Optional[str] = None
        self._agent_prompt: Optional[str] = None
        self._backlog: deque[RoomItem] = deque()
        self._closed_reason: Optional[str] = None
        self._started = False

    # -- AgentInterface ------------------------------------------------------------------------------

    async def start(self, scenario: Scenario) -> None:
        """Join a new room for ``scenario``, wait for the agent, and (only without a config prompt) ask
        the agent for its instructions."""
        self._reset()
        self._room_name = f"{self._settings.room_prefix}{self._run_id}-{scenario.name}"
        room = self._room_factory()
        self._room = room
        await room.connect(
            url=self._url,
            identity=self._identity,
            room_name=self._room_name,
            agent_name=self._settings.agent_name or None,
        )
        self._started = True
        await self._wait_for_agent(room)
        if self._config.agent.system_prompt is None and self._closed_reason is None:
            self._agent_prompt = await self._request_agent_prompt(room)

    async def send(self, user_text: Optional[str], user_audio: Optional[bytes]) -> AgentReply:
        """One user turn: publish the audio (voice) or send the text on ``lk.chat``; ``send(None, None)``
        sends nothing and waits for the agent's opener."""
        room = self._room
        if room is None or not self._started:
            raise RuntimeError("LiveKitAdapter.send() was called before start()")
        if self._closed_reason is not None:
            raise AgentClosed(self._closed_message())
        clock = self._clock
        if user_audio is not None:
            pcm, rate = _speech_pcm(user_audio)
            started = clock.monotonic()
            speech_end = await room.publish_audio(pcm, rate)
            returned = clock.monotonic()
            last_speech = _finite(speech_end, returned)
            turn = _Turn(
                window_start=started,
                output_from=returned,
                latency_start=last_speech,
                audio_from=last_speech,
                collect_audio=True,
                publish=(started, returned),
            )
        elif user_text is not None:
            started = clock.monotonic()
            await room.send_text(user_text, TOPIC_CHAT)
            turn = _Turn(
                window_start=started,
                output_from=started,
                latency_start=started,
                audio_from=started,
                collect_audio=self._voice,
            )
        else:   # outbound opener: everything since the room was joined belongs to it
            turn = _Turn(
                window_start=-math.inf,
                output_from=-math.inf,
                latency_start=clock.monotonic(),
                audio_from=-math.inf,
                collect_audio=self._voice,
            )
        return await self._collect(room, turn)

    async def close(self) -> None:
        """Leave the room (also after the room closed, or after a failed ``start()``)."""
        room, self._room = self._room, None
        self._started = False
        if room is not None:
            await room.close()

    def get_system_prompt(self) -> Optional[str]:
        if self._config.agent.system_prompt is not None:
            if not self._config_prompt_loaded:
                self._config_prompt = read_system_prompt(self._config)
                self._config_prompt_loaded = True
            return self._config_prompt
        return self._agent_prompt

    # -- start -------------------------------------------------------------------------------------

    async def _wait_for_agent(self, room: Room) -> None:
        clock = self._clock
        deadline = clock.monotonic() + self._join_timeout_s
        while self._agent is None:
            if self._closed_reason is not None:
                raise LiveKitError(f"the LiveKit room closed before the agent joined: {self._closed_reason}")
            if clock.monotonic() >= deadline:
                raise LiveKitError(JOIN_TIMEOUT_ERROR.format(seconds=self._join_timeout_s))
            item = await next_before(room.recv, clock, deadline)
            if item is not None:
                self._intake(item)

    async def _request_agent_prompt(self, room: Room) -> Optional[str]:
        """``GetAgentInfoResponse.instructions`` within 3 s, else None (SPEC.md section 9)."""
        assert self._agent is not None
        try:
            await room.request_agent_info(self._agent)
        except Exception:  # noqa: BLE001 - a request that cannot be sent is an unanswered one
            return None
        clock = self._clock
        deadline = clock.monotonic() + AGENT_INFO_TIMEOUT_S
        while self._closed_reason is None and clock.monotonic() < deadline:
            item = await next_before(room.recv, clock, deadline)
            if item is None:
                continue
            if isinstance(item, RoomSessionEvent) and item.kind == SESSION_AGENT_INFO:
                instructions = (item.data or {}).get("instructions")
                return instructions if isinstance(instructions, str) and instructions.strip() else None
            self._intake(item)
        return None

    def _intake(self, item: RoomItem) -> None:
        """An item read during ``start()``: participants and the close are handled now, an early
        ``agent_info`` answer is dropped, everything else waits for the first turn."""
        if isinstance(item, RoomParticipant):
            self._on_participant(item)
        elif isinstance(item, RoomClosed):
            self._closed_reason = item.reason or "the room closed"
        elif isinstance(item, RoomSessionEvent) and item.kind == SESSION_AGENT_INFO:
            return
        else:
            self._backlog.append(item)

    def _on_participant(self, item: RoomParticipant) -> None:
        if self._agent is not None:
            return
        if self._wanted_agent is not None:
            selected = item.identity == self._wanted_agent
        else:
            selected = bool(item.is_agent) and item.identity != self._identity
        if selected:
            self._agent = item.identity
            state = (item.attributes or {}).get(ATTRIBUTE_AGENT_STATE)
            if isinstance(state, str) and state:
                self._agent_state = state

    def _closed_message(self) -> str:
        return f"the LiveKit room closed: {self._closed_reason}"

    # -- turns -------------------------------------------------------------------------------------

    async def _collect(self, room: Room, turn: _Turn) -> AgentReply:
        clock = self._clock
        timeout_at = clock.monotonic() + self._turn_timeout_s
        while self._backlog:
            self._process(self._backlog.popleft(), turn)
        while True:
            if self._closed_reason is not None:
                reply = self._reply(turn, clock.monotonic()) if turn.has_content() else None
                raise AgentClosed(self._closed_message(), reply=reply)
            now = clock.monotonic()
            end_at = turn.idle_since + self._grace_s if turn.idle_since is not None else None
            if end_at is not None and now >= end_at:
                return self._reply(turn, now)
            if now >= timeout_at:
                raise LiveKitTurnTimeout(
                    TURN_TIMEOUT_ERROR.format(seconds=self._turn_timeout_s), reply=self._partial_reply(turn)
                )
            deadline = timeout_at if end_at is None else min(end_at, timeout_at)
            item = await next_before(room.recv, clock, deadline)
            if item is not None:
                self._process(item, turn)

    def _process(self, item: RoomItem, turn: _Turn) -> None:
        if isinstance(item, RoomAttributes):
            if self._agent is not None and item.identity == self._agent:
                state = (item.changed or {}).get(ATTRIBUTE_AGENT_STATE)
                if isinstance(state, str):
                    self._on_agent_state(state, item.arrival_s, turn)
        elif isinstance(item, RoomText):
            if item.topic != TOPIC_TRANSCRIPTION or not item.final:
                return
            if self._agent is not None and item.sender_identity == self._agent:
                self._on_agent_text(item, turn)
            elif item.sender_identity == self._identity and item.arrival_s >= turn.window_start:
                turn.heard.append(item.text)
        elif isinstance(item, RoomAudio):
            if turn.collect_audio:
                self._on_agent_audio(item, turn)
        elif isinstance(item, RoomSessionEvent):
            if item.kind == SESSION_TOOLS_EXECUTED:
                turn.tool_calls.extend(tool_calls_from_event(item.data or {}))
            # a late agent_info answer (after the 3 s wait) is ignored
        elif isinstance(item, RoomParticipant):
            self._on_participant(item)
        elif isinstance(item, RoomClosed):
            self._closed_reason = item.reason or "the room closed"

    def _busy(self) -> bool:
        return self._agent_state in _BUSY_STATES

    def _on_agent_state(self, new: str, arrival: float, turn: _Turn) -> None:
        old = self._agent_state
        self._agent_state = new
        publishing = turn.publish is not None and turn.publish[0] <= arrival <= turn.publish[1]
        if publishing and old == _SPEAKING and new != _SPEAKING:
            turn.interrupted = True   # barge-in: the agent stops speaking while user speech is being sent
        started_or_finished = new == _SPEAKING or (old in _BUSY_STATES and new not in _BUSY_STATES)
        if arrival >= turn.output_from and old != new and started_or_finished:
            turn.produced = True
        if new in _BUSY_STATES:
            turn.idle_since = None
        elif turn.produced and turn.idle_since is None:
            turn.idle_since = self._clock.monotonic()

    def _on_agent_text(self, item: RoomText, turn: _Turn) -> None:
        turn.texts.append(item.text)
        if item.arrival_s >= turn.latency_start and (turn.first_text_s is None or item.arrival_s < turn.first_text_s):
            turn.first_text_s = item.arrival_s
        if item.arrival_s >= turn.output_from:
            turn.produced = True
            if not self._busy():
                turn.idle_since = self._clock.monotonic()   # new output restarts the grace window

    def _on_agent_audio(self, item: RoomAudio, turn: _Turn) -> None:
        if turn.audio_rate is None:
            if item.arrival_s < turn.audio_from or not _is_speech(item.pcm):
                return
            turn.audio_rate = int(item.sample_rate)
            turn.first_audio_s = item.arrival_s
        pcm = item.pcm
        if int(item.sample_rate) != turn.audio_rate:
            pcm = _resample_pcm(pcm, int(item.sample_rate), turn.audio_rate)
        turn.audio.append(bytes(pcm))

    @staticmethod
    def _first_output(turn: _Turn) -> Optional[float]:
        """When the agent's first output of the turn arrived (speech audio, else final text), or None."""
        if turn.first_audio_s is not None and turn.first_audio_s >= turn.latency_start:
            return turn.first_audio_s
        return turn.first_text_s

    def _reply(self, turn: _Turn, ended_at: float) -> AgentReply:
        first = self._first_output(turn)
        if first is None:
            first = ended_at
        return self._build_reply(turn, max(0.0, (first - turn.latency_start) * 1000.0))

    def _partial_reply(self, turn: _Turn) -> AgentReply:
        """The turn so far, for the turn-timeout error: latency NaN when no output arrived (A6)."""
        first = self._first_output(turn)
        latency = math.nan if first is None else max(0.0, (first - turn.latency_start) * 1000.0)
        return self._build_reply(turn, latency)

    def _build_reply(self, turn: _Turn, latency_ms: float) -> AgentReply:
        text = " ".join(part.strip() for part in turn.texts if part.strip())
        heard = " ".join(part.strip() for part in turn.heard if part.strip())
        audio = _wav(b"".join(turn.audio), turn.audio_rate) if turn.audio_rate is not None else None
        return AgentReply(
            text=text,
            audio_bytes=audio,
            latency_ms=latency_ms,
            heard_text=heard or None,
            interrupted=turn.interrupted,
            tool_calls=list(turn.tool_calls),
        )


def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


# ------------------------------------------------------------------------------------------------
# The real room (livekit.rtc, livekit.api, livekit.protocol); imported lazily
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _LiveKitModules:
    rtc: Any
    api: Any
    agent_session: Any


def _load_livekit() -> _LiveKitModules:
    """The LiveKit SDK modules. A missing package is a config error with the SPEC.md section 9 text;
    an SDK that is installed but cannot load (for example a missing system DLL) is a config error too."""
    try:
        from livekit import api, rtc
        from livekit.protocol.agent_pb import agent_session
    except ModuleNotFoundError:
        raise ConfigError("agent.adapter", MISSING_EXTRA) from None
    except (ImportError, OSError) as exc:
        raise ConfigError(
            "agent.adapter",
            f"the livekit extra is installed but could not be imported ({type(exc).__name__}: {exc}). "
            "On Windows, a missing vcruntime140_1.dll means the Microsoft Visual C++ Redistributable "
            "(x64) needs to be installed or updated.",
        ) from None
    return _LiveKitModules(rtc=rtc, api=api, agent_session=agent_session)


def _livekit_secret(name: str, field_path: str, what: str, toml: Optional[TomlConfig]) -> str:
    """The value of the variable ``name`` (environment, then config.toml ``[keys]``); never printed."""
    value = resolve_key(name, toml) if name else None
    if not value:
        raise ConfigError(
            field_path,
            f"the LiveKit {what} is not set. Set the environment variable {name or '(none named)'} "
            "or add it to [keys] in .impulse/config.toml.",
        )
    return value


def _real_room_factory(config: ProjectConfig, settings: LiveKitConfig, clock: Clock) -> Callable[[], Room]:
    """A factory of ``LiveKitRoom``s; checks the extra and the key and secret before any connection."""
    modules = _load_livekit()
    toml = load_project_toml()
    api_key = _livekit_secret(settings.api_key_env, "agent.livekit.api_key_env", "API key", toml)
    api_secret = _livekit_secret(settings.api_secret_env, "agent.livekit.api_secret_env", "API secret", toml)
    voice = config.voice_config is not None
    agent_identity = (settings.agent_identity or "").strip() or None

    def factory() -> Room:
        return LiveKitRoom(
            modules=modules,
            api_key=api_key,
            api_secret=api_secret,
            clock=clock,
            voice=voice,
            agent_identity=agent_identity,
        )

    return factory


_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*")
_REDACTED = "[REDACTED]"


class LiveKitRoom:
    """The real ``Room``: ``livekit.rtc`` for the connection, ``livekit.api`` for the access token,
    and the ``livekit.protocol`` ``AgentSessionMessage`` protobufs for ``lk.agent.session``.

    - Token: ``AccessToken(key, secret)`` with Impulse's identity and
      ``VideoGrants(room_join=True, room=<room>, can_manage_agent_session=True)``; with ``agent_name``
      also ``RoomConfiguration(agents=[RoomAgentDispatch(agent_name=...)])``. The JWT, key, and secret
      never appear in an error message.
    - The agent (for its audio, its session messages, and its departure) is ``agent_identity``, else
      the first participant of kind AGENT, the same rule as the adapter.
    - Voice mode: Impulse publishes a microphone track at 48 kHz mono and paces user speech in real
      time in 20 ms frames, followed by 1 s of silence so the agent's VAD ends the user's turn; the
      agent's audio track is read at 24 kHz mono, frame by frame.
    - Text streams on ``lk.transcription`` are delivered whole (``arrival_s`` = when the stream
      opened); ``lk.agent.session`` byte streams are decoded in arrival order.

    Not exercised by the test suite (it needs a LiveKit server; see DECISIONS.md).
    """

    MIC_SAMPLE_RATE = 48000
    MIC_FRAME_S = 0.02
    TRAILING_SILENCE_S = 1.0
    AGENT_AUDIO_SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        modules: Optional[_LiveKitModules] = None,
        api_key: str,
        api_secret: str,
        clock: Clock,
        voice: bool,
        agent_identity: Optional[str] = None,
    ) -> None:
        self._lk = modules if modules is not None else _load_livekit()
        self._api_key = api_key
        self._api_secret = api_secret
        self._clock = clock
        self._voice = voice
        self._agent: Optional[str] = agent_identity
        self._fixed_agent = agent_identity is not None
        self._queue: asyncio.Queue[RoomItem] = asyncio.Queue()
        self._session_streams: asyncio.Queue[tuple[Any, str, float]] = asyncio.Queue()
        self._room: Any = None
        self._token: Optional[str] = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._audio_streams: list[Any] = []
        self._source: Any = None
        self._mic_lock = asyncio.Lock()
        self._silence_task: Optional[asyncio.Task[Any]] = None
        self._agent_info_request: Optional[str] = None
        self._ended = False
        self._closing = False

    # -- Room protocol -------------------------------------------------------------------------------

    async def connect(self, url: str, identity: str, room_name: str, agent_name: Optional[str]) -> None:
        rtc, api = self._lk.rtc, self._lk.api
        grants = api.VideoGrants(room_join=True, room=room_name, can_manage_agent_session=True)
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(identity)
            .with_name("Impulse")
            .with_grants(grants)
        )
        if agent_name:
            token = token.with_room_config(
                api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=agent_name)])
            )
        try:
            self._token = token.to_jwt()
        except Exception as exc:  # noqa: BLE001
            raise LiveKitError(f"could not create the LiveKit access token: {self._scrub(exc)}") from None

        room = rtc.Room()
        room.on("participant_connected", self._on_participant_connected)
        room.on("participant_disconnected", self._on_participant_disconnected)
        room.on("participant_attributes_changed", self._on_attributes_changed)
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("disconnected", self._on_disconnected)
        room.register_text_stream_handler(TOPIC_TRANSCRIPTION, self._on_text_stream)
        room.register_byte_stream_handler(TOPIC_AGENT_SESSION, self._on_session_stream)
        self._room = room
        self._spawn(self._read_session_streams())
        try:
            await room.connect(url, self._token, rtc.RoomOptions(auto_subscribe=True))
        except Exception as exc:  # noqa: BLE001
            raise LiveKitError(
                f"could not connect to the LiveKit server at {strip_url_credentials(url)}: {self._scrub(exc)}"
            ) from None
        for participant in list(room.remote_participants.values()):
            self._on_participant_connected(participant)
        if self._voice:
            await self._ensure_microphone()

    async def send_text(self, text: str, topic: str) -> None:
        if self._ended:   # RoomClosed is already queued: the adapter ends the conversation normally
            return
        try:
            await self._room.local_participant.send_text(text, topic=topic)
        except Exception as exc:  # noqa: BLE001
            raise LiveKitError(f"could not send text on {topic}: {self._scrub(exc)}") from None

    async def publish_audio(self, pcm: bytes, sample_rate: int) -> float:
        rtc = self._lk.rtc
        if self._ended:   # RoomClosed is already queued: the adapter ends the conversation normally
            return self._clock.monotonic()
        await self._ensure_microphone()
        if int(sample_rate) != self.MIC_SAMPLE_RATE:
            pcm = _resample_pcm(pcm, int(sample_rate), self.MIC_SAMPLE_RATE)
        pcm = bytes(pcm[: len(pcm) - len(pcm) % 2])
        last = _last_speech_sample(pcm)
        frame_bytes = int(self.MIC_SAMPLE_RATE * self.MIC_FRAME_S) * 2
        async with self._mic_lock:
            await self._stop_silence()
            start = self._clock.monotonic()
            next_at = start
            try:
                for offset in range(0, len(pcm), frame_bytes):
                    chunk = pcm[offset : offset + frame_bytes]
                    if len(chunk) < frame_bytes:
                        chunk = chunk + bytes(frame_bytes - len(chunk))
                    await self._source.capture_frame(
                        rtc.AudioFrame(chunk, self.MIC_SAMPLE_RATE, 1, frame_bytes // 2)
                    )
                    next_at += self.MIC_FRAME_S
                    await self._clock.sleep(max(0.0, next_at - self._clock.monotonic()))
            except Exception as exc:  # noqa: BLE001
                raise LiveKitError(f"could not publish audio: {self._scrub(exc)}") from None
            self._silence_task = self._spawn(self._trailing_silence())
        if last < 0:
            return start
        return start + (last + 1) / float(self.MIC_SAMPLE_RATE)

    async def request_agent_info(self, agent_identity: str) -> None:
        pb = self._lk.agent_session
        request_id = "impulse_" + secrets.token_hex(6)
        message = pb.AgentSessionMessage(
            request=pb.SessionRequest(request_id=request_id, get_agent_info=pb.SessionRequest.GetAgentInfo())
        )
        self._agent_info_request = request_id
        try:
            writer = await self._room.local_participant.stream_bytes(
                name=request_id, topic=TOPIC_AGENT_SESSION, destination_identities=[agent_identity]
            )
            await writer.write(message.SerializeToString())
            await writer.aclose()
        except Exception as exc:  # noqa: BLE001
            raise LiveKitError(f"could not send GetAgentInfo on {TOPIC_AGENT_SESSION}: {self._scrub(exc)}") from None

    async def recv(self) -> RoomItem:
        return await self._queue.get()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream in self._audio_streams:
            with contextlib.suppress(Exception):
                await stream.aclose()
        self._audio_streams.clear()
        if self._source is not None:
            with contextlib.suppress(Exception):
                await self._source.aclose()
        if self._room is not None:
            with contextlib.suppress(Exception):
                await self._room.disconnect()
        self._end("closed by Impulse")

    # -- events ------------------------------------------------------------------------------------

    def _put(self, item: RoomItem) -> None:
        self._queue.put_nowait(item)

    def _end(self, reason: str) -> None:
        if not self._ended:
            self._ended = True
            self._put(RoomClosed(reason=reason))

    def _spawn(self, coroutine: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _is_agent_kind(self, participant: Any) -> bool:
        return participant.kind == self._lk.rtc.ParticipantKind.PARTICIPANT_KIND_AGENT

    def _on_participant_connected(self, participant: Any) -> None:
        is_agent = self._is_agent_kind(participant)
        if self._agent is None and is_agent:
            self._agent = participant.identity
        self._put(RoomParticipant(identity=participant.identity, is_agent=is_agent, attributes=dict(participant.attributes)))

    def _on_participant_disconnected(self, participant: Any) -> None:
        if self._agent is not None and participant.identity == self._agent:
            self._end(f"the agent {participant.identity} left the room")

    def _on_disconnected(self, reason: Any = None) -> None:
        name = reason
        with contextlib.suppress(Exception):
            name = self._lk.rtc.DisconnectReason.Name(reason)
        self._end(f"disconnected from the LiveKit room ({name})")

    def _on_attributes_changed(self, changed: Any, participant: Any) -> None:
        self._put(
            RoomAttributes(identity=participant.identity, changed=dict(changed), arrival_s=self._clock.monotonic())
        )

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        rtc = self._lk.rtc
        if not self._voice or track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if self._agent is None or participant.identity != self._agent:
            return
        stream = rtc.AudioStream(track, sample_rate=self.AGENT_AUDIO_SAMPLE_RATE, num_channels=1)
        self._audio_streams.append(stream)
        self._spawn(self._read_agent_audio(stream))

    async def _read_agent_audio(self, stream: Any) -> None:
        with contextlib.suppress(Exception):
            async for event in stream:
                frame = event.frame
                self._put(
                    RoomAudio(pcm=frame.data.tobytes(), sample_rate=frame.sample_rate, arrival_s=self._clock.monotonic())
                )

    def _on_text_stream(self, reader: Any, participant_identity: str) -> None:
        self._spawn(self._read_text_stream(reader, participant_identity, self._clock.monotonic()))

    async def _read_text_stream(self, reader: Any, identity: str, opened: float) -> None:
        try:
            text = await reader.read_all()
        except Exception:  # noqa: BLE001 - an aborted stream carries no complete text
            with contextlib.suppress(Exception):
                reader.close()
            return
        attributes = reader.info.attributes or {}
        self._put(
            RoomText(
                topic=reader.info.topic or TOPIC_TRANSCRIPTION,
                sender_identity=identity,
                text=text,
                final=attributes.get(ATTRIBUTE_TRANSCRIPTION_FINAL) == "true",
                arrival_s=opened,
            )
        )

    def _on_session_stream(self, reader: Any, participant_identity: str) -> None:
        self._session_streams.put_nowait((reader, participant_identity, self._clock.monotonic()))

    async def _read_session_streams(self) -> None:
        """One reader, so session messages are decoded in the order their streams opened."""
        while True:
            reader, identity, opened = await self._session_streams.get()
            try:
                data = b"".join([chunk async for chunk in reader])
                message = self._lk.agent_session.AgentSessionMessage()
                message.ParseFromString(data)
            except Exception:  # noqa: BLE001, S112 - a broken stream is skipped
                continue
            if self._agent is not None and identity != self._agent:
                continue
            event = self._decode_session_message(message, opened)
            if event is not None:
                self._put(event)

    def _decode_session_message(self, message: Any, arrival: float) -> Optional[RoomSessionEvent]:
        which = message.WhichOneof("message")
        if which == "response":
            response = message.response
            if self._agent_info_request is None or response.request_id != self._agent_info_request:
                return None
            self._agent_info_request = None
            if response.error:
                return RoomSessionEvent(
                    kind=SESSION_AGENT_INFO,
                    data={"instructions": None, "tools": [], "error": response.error},
                    arrival_s=arrival,
                )
            if not response.HasField("get_agent_info"):
                return None
            info = response.get_agent_info
            return RoomSessionEvent(
                kind=SESSION_AGENT_INFO,
                data={"instructions": info.instructions, "tools": list(info.tools)},
                arrival_s=arrival,
            )
        if which == "event" and message.event.WhichOneof("event") == SESSION_TOOLS_EXECUTED:
            executed = message.event.function_tools_executed
            return RoomSessionEvent(
                kind=SESSION_TOOLS_EXECUTED,
                data={
                    "function_calls": [
                        {"name": call.name, "arguments": call.arguments, "call_id": call.call_id}
                        for call in executed.function_calls
                    ],
                    "function_call_outputs": [
                        {"call_id": out.call_id, "name": out.name, "output": out.output, "is_error": out.is_error}
                        for out in executed.function_call_outputs
                    ],
                },
                arrival_s=arrival,
            )
        return None

    # -- microphone ----------------------------------------------------------------------------------

    async def _ensure_microphone(self) -> None:
        if self._source is not None:
            return
        rtc = self._lk.rtc
        try:
            self._source = rtc.AudioSource(self.MIC_SAMPLE_RATE, 1, queue_size_ms=200)
            track = rtc.LocalAudioTrack.create_audio_track("impulse-user", self._source)
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await self._room.local_participant.publish_track(track, options)
        except Exception as exc:  # noqa: BLE001
            raise LiveKitError(f"could not publish the microphone track: {self._scrub(exc)}") from None

    async def _stop_silence(self) -> None:
        task, self._silence_task = self._silence_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _trailing_silence(self) -> None:
        """Silence after user speech, paced in real time, so the agent's VAD ends the user's turn."""
        rtc = self._lk.rtc
        samples = int(self.MIC_SAMPLE_RATE * self.MIC_FRAME_S)
        silence = bytes(samples * 2)
        next_at = self._clock.monotonic()
        with contextlib.suppress(Exception):   # a room that went away needs no more silence
            for _ in range(round(self.TRAILING_SILENCE_S / self.MIC_FRAME_S)):
                await self._source.capture_frame(rtc.AudioFrame(silence, self.MIC_SAMPLE_RATE, 1, samples))
                next_at += self.MIC_FRAME_S
                await self._clock.sleep(max(0.0, next_at - self._clock.monotonic()))

    # -- redaction -----------------------------------------------------------------------------------

    def _scrub(self, exc: BaseException) -> str:
        """An SDK error message without the access token, the API key, or the API secret."""
        text = str(getattr(exc, "message", None) or exc).strip() or type(exc).__name__
        for secret in sorted({self._token or "", self._api_secret, self._api_key}, key=len, reverse=True):
            if secret and len(secret) >= 4:
                text = text.replace(secret, _REDACTED)
        return _JWT_RE.sub(_REDACTED, text)


def _last_speech_sample(pcm: bytes) -> int:
    """Index of the last sample above 1 % of full scale in 16-bit PCM, or -1."""
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.int32)
    loud = np.nonzero(np.abs(samples) > 32768 * SPEECH_PEAK_FRACTION)[0]
    return int(loud[-1]) if loud.size else -1
