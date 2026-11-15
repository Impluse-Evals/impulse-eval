# SPDX-License-Identifier: Apache-2.0
"""Pipecat adapter (SPEC.md sections 8.2 and 9; optional extra ``pipecat``).

Impulse connects to a RUNNING Pipecat bot as a client and speaks RTVI, Pipecat's client/server message
protocol. The user's pipeline is never imported or run in Impulse's process. Two connection modes share
everything except the wire:

- ``eval`` (default): the bot runs Pipecat's eval transport (started with ``-t eval``). The wire is
  Pipecat's own: ``EvalChannel`` runs Pipecat's harness-side ``EvalClientTransport`` with
  ``EvalClientSerializer`` in a small pipeline owned by Impulse (never the user's pipeline), so Pipecat's
  client transport sends ``client-ready`` and paces the user's audio itself.
- ``websocket`` (fallback): any Pipecat WebSocket server transport with ``ProtobufFrameSerializer``.
  ``WebsocketChannel`` uses ``websockets`` plus Pipecat's ``ProtobufFrameSerializer``; the adapter sends
  ``client-ready`` and paces the user's audio itself (``chunk_ms``, ``realtime_pacing``,
  ``trailing_silence_ms``).

Testability seam (SPEC.md section 9): all protocol logic lives in ``PipecatAdapter`` and talks only to the
``Channel`` protocol below (``ChannelMessage`` / ``ChannelAudio`` / ``ChannelClose`` items) and a
``Clock``; every timeout and grace window goes through the clock. Every ``pipecat`` import is lazy and
happens only inside the two real channels, so ``import impulse_evals`` never needs pipecat-ai.

Eval mode overrides private pipecat-ai APIs (tested with 1.12.x): they are checked when the adapter is
built (a missing one is a config error pointing at the websocket fallback), another pipecat-ai version
adds a run warning, and the latency start always comes from Pipecat's send path (never "now").

How a conversation maps onto RTVI (SPEC.md section 9):

- Handshake: ``bot-ready`` within ``handshake_timeout_s``. Eval mode then sends ``eval-configure``
  (function call report level FULL) and, when ``agent.system_prompt`` is set, ``eval-context`` with that
  prompt as the only message (a ``-t eval`` bot keeps its LLM context between connections); without it,
  the reset warning is returned once.
- System prompt: the config.yaml prompt wins; otherwise ``client-message`` ``{"t": "get_system_prompt"}``
  and a ``server-response`` with the same ``id`` within 3 s.
- Text turns: ``send-text`` (``audio_response: false``); reply = ``bot-output`` texts (the latest text
  per segment; since a ``-t eval`` bot speaks the LLM run after a tool call and a push-text TTS
  announces each sentence before it completes it, an announced ``bot-output`` (``will_be_spoken: true``,
  not yet ``spoken`` / ``"completed"``) is dropped when a completed one with the same trimmed text
  arrives in the same turn; one still open when the turn ends is kept, and its later completion, in a
  later turn or as pre-turn output, is not added again), else the ``bot-llm-text`` tokens.
  Voice turns: the user audio at ``pipecat.sample_rate``; reply text = the
  ``bot-output`` segments that were spoken (``spoken: true`` or ``spoken_status: "completed"``), once per
  segment; the bot's audio from the latency start (the user's last speech frame sent) on is collected
  into one WAV (bot audio that arrived while the user was still speaking is not the reply's audio).
- End of turn: voice, ``bot-stopped-speaking`` with no function call in progress and no
  ``bot-started-speaking`` within ``end_of_turn_grace_ms``, counted only after bot output seen once the
  user's last speech frame was sent (the bot's messages are read while the user's audio is sent, so a
  greeting tail or a barge-in stop never ends the turn); text, ``bot-llm-stopped`` with no function call
  in progress and no ``bot-llm-started`` within the grace window. In both modes a call in progress holds
  the turn open and the window restarts at the ``llm-function-call-stopped`` that ends the last call, so
  "speak, call a tool, speak again" is ONE turn. ``pipecat.turn_timeout_s`` (both modes) bounds a turn,
  else ``voice_config.timeout_s`` when ``voice_config`` exists, else 30 s.
- Turn timeout (fixes round 3 A1): ``PipecatTurnTimeout`` (a ``PipecatError`` and an ``AgentTurnError``)
  whose ``reply`` is the turn so far when it has non-blank text, a tool call, or audio, else None. Its text
  is the normal reply text, except in voice mode when no ``bot-output`` was spoken: then every
  ``bot-output`` text of the turn (announced or in progress, the latest text per segment), else the turn's
  LLM tokens (a bot speaking a raw function call never sends ``bot-stopped-speaking``). Its latency is
  measured as for a normal reply when speech audio (voice) or text arrived, else NaN (never the timeout
  instant). A turn that ends in the timeout remembers no open announcement.
- ``heard_text`` (final ``user-transcription``), ``tool_calls`` (function call messages),
  ``interrupted`` (``user-started-speaking`` during bot speech), and per-service ``metrics`` (stored in
  milliseconds in ``get_metadata()["pipecat_metrics"]``).
- A function handler that raised (fixes round 3 A2): pipecat-ai returns the result "The function `NAME`
  failed and returned no result." and pushes the RTVI error "Error executing function call [NAME]: ...",
  in either order, and same-name calls may run in parallel. Per turn and function name (A2-rev), an RTVI
  error only ever enriches a FAILED or NOT-CAPTURED call: the template result gives ``success=False`` with
  the oldest pending RTVI error of that name as ``error`` (consumed), else the template; an RTVI error
  replaces the template error of the most recent template-failed call of that name that has none yet,
  else it is pending; when the reply is built (normal end, timeout, close) each pending error ties to the
  most recent call of that name with ``success`` None and no result, and the rest are dropped (only the
  run warning stays). A successful, dict-result or cancelled call is never changed, and no call ends
  with ``success=True`` and such an error.
- A close from the bot during a turn with close code 1000 or 1001 raises ``AgentClosed`` (the bot hung up,
  a normal end); any other code (1006 abnormal, 1011 server error, ...) is a scenario error, and so is any
  close read during ``start()`` after ``bot-ready``; close code 1013 means the bot already serves another
  client.
- Error texts show ``pipecat.url`` without its credentials (userinfo, ``?token=`` and similar query
  parameters); the connection itself uses the full URL. Error texts from the bot or the eval harness
  are redacted (the runner's ``redact`` plus the credentials the adapter holds) BEFORE they are put on
  one line and cut to 200 characters, so a key that straddles the cut leaves no prefix behind.

Items read during ``start()`` (for example an inbound greeting) are kept for the first turn: their reply
text and tool calls are added to the first reply, but they never end that turn or set its latency, and
their audio and user transcriptions are dropped. Just before a user turn is sent (never for the opener)
the adapter reads every item the channel already holds (no waiting), with the same rule, so a greeting
queued unread while the runner prepared the turn is pre-turn output too. The same holds for bot output
still in flight when the user's turn is sent: an LLM run whose ``bot-llm-started`` was read before the
latency start (a pre-turn run) adds its text to the reply, in order, but never sets the latency, and its
``bot-llm-stopped`` never ends the turn; a bot speech whose ``bot-started-speaking`` was read before the
latency start (pre-turn speech) keeps its audio from the latency start on, but never sets TTFA, and its ``bot-stopped-speaking``
never ends the turn, unless a spoken ``bot-output`` was read during that speech after the reply's own
LLM text (the reply spoken right behind the greeting in one speech). In the reply text the pre-turn part comes first, then the reply's own part (each:
its ``bot-output`` texts, else in text mode its LLM tokens). Known limitation: a greeting whose LLM run or
speech has not started at all when the turn is sent cannot be told apart from the reply. For the
outbound opener (``send(None, None)``) everything since the connection opened belongs to the opener.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import itertools
import json
import math
import re
import secrets
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union

from impulse_evals.adapters._clock import Clock, SystemClock, next_before
from impulse_evals.adapters.base import (
    AgentClosed,
    AgentInterface,
    AgentReply,
    AgentTurnError,
)
from impulse_evals.config import (
    PIPECAT_MODES,
    REDACTED,
    ConfigError,
    PipecatConfig,
    ProjectConfig,
    load_project_toml,
    read_system_prompt,
    resolve_key,
    secret_text,
    strip_url_credentials,
    url_credentials,
)
from impulse_evals.models import Scenario, ToolCall

__all__ = [
    "BOT_CLOSED_ERROR",
    "EVALS_MISSING",
    "EVAL_INTERNALS_ERROR",
    "FUNCTION_CALL_WARNING",
    "HANDSHAKE_TIMEOUT_ERROR",
    "HARNESS_FAILED_ERROR",
    "MISSING_EXTRA",
    "ONE_CLIENT_ERROR",
    "RESET_WARNING",
    "SEND_NOT_REPORTED_ERROR",
    "TURN_TIMEOUT_ERROR",
    "UNTESTED_VERSION_WARNING",
    "Channel",
    "ChannelAudio",
    "ChannelClose",
    "ChannelMessage",
    "Clock",
    "EvalChannel",
    "PipecatAdapter",
    "PipecatError",
    "PipecatTurnTimeout",
    "WebsocketChannel",
    "quiet_pipecat_logging",
]

# ------------------------------------------------------------------------------------------------
# Exact texts and protocol constants (SPEC.md section 9; DECISIONS.md source read, pipecat-ai 1.12.0)
# ------------------------------------------------------------------------------------------------

MISSING_EXTRA = "pipecat adapter needs the 'pipecat' extra: pip install impulse-evals[pipecat]"
EVALS_MISSING = (
    "Pipecat eval mode needs a pipecat-ai release that includes pipecat.evals. "
    "Upgrade pipecat-ai, or set pipecat.mode: websocket."
)
HANDSHAKE_TIMEOUT_ERROR = (
    "Pipecat bot did not send bot-ready within {seconds:g} s. Check that the bot is running at {url} "
    "(eval mode: started with -t eval) and RTVI is enabled."
)
TURN_TIMEOUT_ERROR = "Pipecat bot did not finish its turn within {seconds:g} s."
BOT_CLOSED_ERROR = "the Pipecat bot closed the connection (close code {code})"
ONE_CLIENT_ERROR = (
    "Pipecat bot accepts one client at a time. Run with --concurrency 1, or start one bot per connection."
)
V2_MODE_ERROR = "pipecat.mode: {mode} is planned for v2. Use eval or websocket."
RESET_WARNING = (
    "Pipecat eval bots keep their conversation between scenarios. Set agent.system_prompt in config.yaml "
    "so Impulse can reset it, or reset the LLM context in the bot's on_client_connected handler."
)
FUNCTION_CALL_WARNING = (
    "Pipecat bot reports function calls without names or arguments. Set the RTVI function call report "
    "level to FULL so UNIV-023 can check tool arguments."
)
# Fixes round 4, O1: eval-context replaces the bot's LLM context, so the bot runs the prompt FILE; say so
# once per run (in the final live test a BUGGY bot behaved normally for a whole run because of it).
CONTEXT_RESET_NOTE = (
    "Pipecat eval bot: Impulse reset its conversation with the agent.system_prompt file before each scenario, "
    "so the bot ran that prompt, not its own. Keep the file identical to the bot's prompt."
)
# Eval mode relies on private pipecat-ai APIs (EvalChannel); 1.12.x is the tested release series.
TESTED_PIPECAT_SERIES = "1.12"
EVAL_INTERNALS_ERROR = (
    "Pipecat eval mode relies on internal APIs of pipecat-ai 1.12 (the tested version), and the installed "
    "pipecat-ai {version} does not have: {names}. Install pipecat-ai 1.12.x, or set pipecat.mode: websocket."
)
UNTESTED_VERSION_WARNING = "pipecat-ai {version} is not the tested 1.12.x; eval mode relies on its internal APIs."
SEND_NOT_REPORTED_ERROR = (
    "Pipecat eval transport did not report sending the user audio; check the pipecat-ai version (tested: 1.12)."
)
HARNESS_FAILED_ERROR = "Pipecat eval harness pipeline failed: {error}"
HARNESS_ERROR_WARNING = "Pipecat eval harness reported an error: {error}"
HARNESS_ERROR = "impulse-harness-error"   # EvalChannel -> adapter: an error of Impulse's harness pipeline (not RTVI)

RTVI_LABEL = "rtvi-ai"
RTVI_PROTOCOL_VERSION = "2.1.0"
CLIENT_LIBRARY = "impulse-evals"
CLIENT_LIBRARY_VERSION = "0.1.0"
EVAL_CONFIGURE = "eval-configure"
EVAL_CONTEXT = "eval-context"
EVAL_BOT_AUDIO = "eval-bot-audio"
GET_SYSTEM_PROMPT = "get_system_prompt"
REPORT_LEVEL_FULL = {"*": "full"}
SKIP_TTS_FLAG = "skip_tts"
CAPTURE_BOT_AUDIO_FLAG = "capture_bot_audio"

CLOSE_NORMAL_CODES = (1000, 1001)   # normal closure, going away: the bot hung up (AgentClosed)
CLOSE_TRY_AGAIN_LATER = 1013        # SingleClientWebsocketServer: "Server already has a connected client"
CLOSE_FORBIDDEN_DEV_RUNNER = 4003   # development runner: bad token or Origin
SYSTEM_PROMPT_TIMEOUT_S = 3.0
DEFAULT_TURN_TIMEOUT_S = 30.0
SPEECH_PEAK_FRACTION = 0.01          # audio counts as speech above 1 % of full scale
_MAX_ERROR_WARNINGS = 5
_ERROR_TEXT_CHARS = 200              # bot / harness error texts: one line, cut AFTER redaction
_MIN_SECRET_CHARS = 4                # as config.redact: shorter values are never redacted
# pipecat-ai 1.12.0 llm_service.py, a function handler that raised (fixes round 3 A2): the result returned
# to the LLM (FUNCTION_CALL_ERROR_MESSAGE_TEMPLATE) and the error pushed upstream (an RTVI error).
_FAILED_CALL_RESULT = re.compile(r"^The function `(?P<name>.+)` failed and returned no result\.$")
_FAILED_CALL_ERROR = re.compile(r"^Error executing function call \[(?P<name>[^\]]+)\]: ")

# ------------------------------------------------------------------------------------------------
# The Channel seam (SPEC.md section 9)
# ------------------------------------------------------------------------------------------------


@dataclass
class ChannelMessage:
    """One RTVI or eval JSON message from the bot."""

    data: dict


@dataclass
class ChannelAudio:
    """Raw bot audio (mono 16-bit little-endian PCM); ``arrival_s`` is on the adapter's clock."""

    pcm: bytes
    sample_rate: int
    arrival_s: float


@dataclass
class ChannelClose:
    """The connection closed (WebSocket close code)."""

    code: int


ChannelItem = Union[ChannelMessage, ChannelAudio, ChannelClose]


class Channel(Protocol):
    """What the adapter needs from a connection to a Pipecat bot. ``EvalChannel`` and
    ``WebsocketChannel`` are the real ones."""

    async def connect(self, url: str, headers: dict, query: dict) -> None:
        """Open the connection; eval mode: ``query`` holds the ``skip_tts`` / ``capture_bot_audio`` flags."""
        ...

    async def send_message(self, data: dict) -> None: ...

    async def send_audio(self, pcm: bytes, sample_rate: int) -> float:
        """Send mono 16-bit PCM; returns the clock time the last NON-SILENT sample left Impulse."""
        ...

    async def recv(self) -> ChannelItem: ...

    async def close(self) -> None: ...


class PipecatError(RuntimeError):
    """A Pipecat failure that is a scenario error (handshake or turn timeout, one-client or abnormal close, ...)."""


class PipecatTurnTimeout(PipecatError, AgentTurnError):
    """The turn timeout: a ``PipecatError`` and an ``AgentTurnError`` whose ``reply`` is the turn so far
    (the runner records it before the scenario error, so the turn's text, audio and tool calls are kept)."""

    def __init__(self, message: str, *, reply: AgentReply | None = None):
        AgentTurnError.__init__(self, message, reply=reply)


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _last_speech_index(pcm: bytes) -> int:
    """Index of the last sample above 1 % of full scale in 16-bit PCM, or -1."""
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.int32)
    loud = np.nonzero(np.abs(samples) > 32768 * SPEECH_PEAK_FRACTION)[0]
    return int(loud[-1]) if loud.size else -1


def _is_speech(pcm: bytes) -> bool:
    from impulse_evals.voice import audio_utils

    return audio_utils.peak_fraction(pcm) > SPEECH_PEAK_FRACTION


def _speech_pcm(audio: bytes, sample_rate: int) -> bytes:
    """User audio (WAV, or another format pydub can convert) as mono 16-bit PCM at ``sample_rate``."""
    from impulse_evals.voice import audio_utils

    return audio_utils.pcm16_frames(audio_utils.convert_wav(audio_utils.ensure_wav(bytes(audio)), sample_rate))


def _resample_pcm(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    import numpy as np

    from impulse_evals.voice import audio_utils

    usable = len(pcm) - len(pcm) % 2
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
    return audio_utils.float_to_pcm16(audio_utils.resample(samples, from_rate, to_rate)).tobytes()


def _downmix_pcm(pcm: bytes, channels: int) -> bytes:
    """Interleaved 16-bit PCM with ``channels`` channels as mono."""
    if channels <= 1:
        return bytes(pcm)
    import numpy as np

    from impulse_evals.voice import audio_utils

    usable = len(pcm) - len(pcm) % (2 * channels)
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768.0
    return audio_utils.float_to_pcm16(samples.mean(axis=1)).tobytes()


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    from impulse_evals.voice import audio_utils

    return audio_utils.pcm16_to_wav(pcm, sample_rate)


def _tool_arguments(value: Any) -> dict:
    """``ToolCall.arguments`` from a reported value: a dict as is, a JSON object string decoded, missing
    → ``{}``, anything else ``{"raw": value}``."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            decoded = None
        return dict(decoded) if isinstance(decoded, dict) else {"raw": value}
    return {"raw": value}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass
class _Turn:
    """What one ``send()`` collects. Times are ``Clock`` seconds."""

    voice: bool                  # end-of-turn rule: speaking events (voice) or LLM events (text)
    index: int                   # 1-based transcript position of this agent turn (pipecat_metrics)
    latency_start: float
    audio_from: float            # bot audio that arrived before this is not this turn's audio
    collect_audio: bool
    llm_runs: list[list[str]] = field(default_factory=list)   # the reply's own LLM runs
    run_open: bool = False
    run_pre_turn: bool = False   # the open LLM run started before latency_start (e.g. a greeting in flight)
    speech_pre_turn: bool = False  # the bot speech in progress started before latency_start
    reply_llm_text: bool = False  # the reply's own LLM run (not a pre-turn run) produced text
    # A SPOKEN bot-output read during the pre-turn speech in progress, after the reply's own LLM text: the
    # reply is spoken right behind the greeting in ONE speech, so that speech's stop ends the turn.
    reply_spoken_in_speech: bool = False
    outputs: dict[Any, str] = field(default_factory=dict)     # the reply's own bot-output texts
    # Pre-turn output (read during start(), queued before the send, or of a pre-turn run / speech): kept
    # apart, so a spoken greeting's bot-output never replaces a text reply made of LLM text.
    pre_runs: list[list[str]] = field(default_factory=list)
    pre_outputs: dict[Any, str] = field(default_factory=dict)
    # Text mode (A5-rev): announced bot-outputs (will_be_spoken, not yet completed) still waiting for
    # their completion, as (in pre_outputs, key) -> text; each one is also in outputs / pre_outputs.
    announced: dict[tuple[bool, Any], str] = field(default_factory=dict)
    # Voice mode (fixes round 3 A1): EVERY bot-output text (announced, in progress or spoken; the latest
    # text per segment), own and pre-turn; used only for a timed-out turn with no spoken bot-output.
    all_outputs: dict[Any, str] = field(default_factory=dict)
    pre_all_outputs: dict[Any, str] = field(default_factory=dict)
    heard: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Fixes round 3 A2-rev, keyed by id() of a ToolCall: the calls that failed through the template and
    # still carry the template as their error (an RTVI error may replace it, rule 2), and the calls whose
    # error these rules set (rule 4). Pending RTVI errors: per function name, oldest first, this turn only.
    template_errors: dict[int, str] = field(default_factory=dict)
    enriched: set[int] = field(default_factory=set)
    pending_errors: dict[str, list[str]] = field(default_factory=dict)
    audio: list[bytes] = field(default_factory=list)
    audio_rate: Optional[int] = None
    first_audio_s: Optional[float] = None
    first_text_s: Optional[float] = None
    stop_seen: bool = False      # an end-of-turn event of this turn arrived (bot-llm-stopped / bot-stopped-speaking)
    replied: bool = False        # bot output (speaking, text, speech audio) seen at or after latency_start
    end_since: Optional[float] = None
    interrupted: bool = False

    def has_content(self) -> bool:
        return bool(
            self.outputs or self.pre_outputs or any(self.llm_runs) or any(self.pre_runs) or self.tool_calls or self.audio
        )

    def runs_for(self, pre_turn_run: bool) -> list[list[str]]:
        return self.pre_runs if pre_turn_run else self.llm_runs


# ------------------------------------------------------------------------------------------------
# The adapter
# ------------------------------------------------------------------------------------------------


class PipecatAdapter(AgentInterface):
    """Talks to a running Pipecat bot over RTVI, one connection per scenario (SPEC.md section 9).

    ``channel_factory`` is a plain callable (called with no arguments) that returns one ``Channel`` per
    scenario; None means the real channel of ``pipecat.mode`` (``EvalChannel`` or ``WebsocketChannel``),
    which needs the ``pipecat`` extra. ``clock`` defaults to the system clock. ``redact`` (``text ->
    text``, the runner passes its key redactor) is applied, with the credentials the adapter holds
    itself, to every bot and harness error text before it is shortened.
    """

    def __init__(
        self,
        config: ProjectConfig,
        *,
        channel_factory: Optional[Callable[[], Channel]] = None,
        clock: Optional[Clock] = None,
        redact: Optional[Callable[[str], str]] = None,
    ) -> None:
        settings = config.agent.pipecat
        if settings is None or not settings.url:
            raise ConfigError("agent.pipecat.url", "required when agent.adapter is pipecat")
        if settings.mode not in PIPECAT_MODES:
            raise ConfigError("agent.pipecat.mode", V2_MODE_ERROR.format(mode=settings.mode))
        self._config = config
        self._settings: PipecatConfig = settings
        self._mode: str = settings.mode
        self._url: str = settings.url                              # connect() gets the full URL
        self._shown_url: str = strip_url_credentials(settings.url)  # error texts never show its credentials
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._voice = config.voice_config is not None
        self._sample_rate = int(settings.sample_rate) if settings.sample_rate and settings.sample_rate > 0 else 16000
        # Fixes round 3 A1: pipecat.turn_timeout_s (text and voice mode), else voice_config.timeout_s, else 30 s.
        if settings.turn_timeout_s is not None:
            self._turn_timeout_s = float(settings.turn_timeout_s)
        elif config.voice_config is not None:
            self._turn_timeout_s = float(config.voice_config.timeout_s)
        else:
            self._turn_timeout_s = DEFAULT_TURN_TIMEOUT_S
        self._grace_s = max(0.0, float(settings.end_of_turn_grace_ms) / 1000.0)
        self._handshake_timeout_s = max(0.0, float(settings.handshake_timeout_s))
        self._headers = _connect_headers(settings)
        self._redact_text = redact
        self._secrets = _held_secrets(settings, self._headers)
        self._query: dict[str, str] = {}
        if self._mode == "eval":
            # skip_tts in text mode, capture_bot_audio in voice mode; trigger_disconnect never (SPEC.md section 9).
            self._query = {CAPTURE_BOT_AUDIO_FLAG: "true"} if self._voice else {SKIP_TTS_FLAG: "true"}
        self._warnings: list[str] = []
        self._error_warnings = 0
        self._channel_factory: Callable[[], Channel] = (
            channel_factory if channel_factory is not None else _real_channel_factory(self)
        )
        self._config_prompt: Optional[str] = None
        self._config_prompt_loaded = False
        self._id_prefix = f"impulse-{secrets.token_hex(3)}"
        self._ids = itertools.count(1)
        if self._mode == "eval" and config.agent.system_prompt is None:
            self._warn(RESET_WARNING)
        self._reset()

    def _reset(self) -> None:
        self._channel: Optional[Channel] = None
        self._started = False
        self._closed_code: Optional[int] = None
        self._backlog: deque[ChannelItem] = deque()
        self._bot_speaking = False
        self._llm_active = False
        self._calls_in_progress: set[str] = set()
        self._calls_by_id: dict[str, ToolCall] = {}
        # Text mode (A5-rev, re-check N-M2): (trimmed text, segment_id or None) of announced bot-outputs
        # kept in a reply because their completion had not arrived when the turn ended. A later
        # bot-output of a remembered segment is not added (the entry is forgotten on its completion); a
        # later completion with a remembered text is not added either. Cleared on bot-interrupted.
        self._kept_announcements: list[tuple[str, Any]] = []
        self._position = 0                    # transcript turns so far in this scenario
        self._metrics: list[dict[str, Any]] = []
        self._agent_prompt: Optional[str] = None
        self._silence_task: Optional[asyncio.Task[Any]] = None
        self._harness_failure: Optional[str] = None   # a fatal error of the eval harness pipeline

    # -- AgentInterface ------------------------------------------------------------------------------

    async def start(self, scenario: Scenario) -> None:
        """Connect (one connection per scenario), complete the handshake, reset the conversation (eval
        mode), and ask for the system prompt when config.yaml sets none."""
        if self._channel is not None:
            await self.close()
        self._reset()
        channel = self._channel_factory()
        self._channel = channel
        await channel.connect(self._url, dict(self._headers), dict(self._query))
        if self._mode == "websocket":
            await channel.send_message(
                self._message(
                    "client-ready",
                    {
                        "version": RTVI_PROTOCOL_VERSION,
                        "about": {"library": CLIENT_LIBRARY, "library_version": CLIENT_LIBRARY_VERSION},
                    },
                )
            )
        await self._wait_for_bot_ready(channel)
        self._started = True
        prompt = self._config_system_prompt()
        if self._mode == "eval":
            # Pipecat's own harness order: eval-configure, then eval-context (SPEC.md section 9).
            await channel.send_message(
                self._message("client-message", {"t": EVAL_CONFIGURE, "d": {"function_call_report_level": dict(REPORT_LEVEL_FULL)}})
            )
            if prompt is not None:
                await channel.send_message(
                    self._message(
                        "client-message",
                        {"t": EVAL_CONTEXT, "d": {"messages": [{"role": "system", "content": prompt}]}},
                    )
                )
                self._warn(CONTEXT_RESET_NOTE)
        if self._config.agent.system_prompt is None:
            self._agent_prompt = await self.probe_system_prompt()
        self._raise_harness_failure()
        if self._closed_code is not None:   # closed after bot-ready, before any turn: the bot failed
            raise self._start_closed_error()

    async def send(self, user_text: Optional[str], user_audio: Optional[bytes]) -> AgentReply:
        """One user turn: the user audio (voice) or ``send-text`` (text); ``send(None, None)`` sends nothing
        and waits for the bot's first turn."""
        channel = self._channel
        if channel is None or not self._started:
            raise RuntimeError("PipecatAdapter.send() was called before start()")
        self._raise_harness_failure()
        if self._closed_code is not None:
            raise self._closed_error(None)
        clock = self._clock
        opener = user_text is None and user_audio is None
        if not opener:
            # Bot output queued while the runner prepared this turn (a greeting, nothing read it) is
            # pre-turn output, like the items read during start(); never the reply or its latency.
            await self._read_queued(channel)
            self._raise_harness_failure()
        index = self._position + (1 if opener else 2)
        self._position = index
        during: list[tuple[ChannelItem, float]] = []
        try:
            if user_audio is not None:
                pcm = _speech_pcm(user_audio, self._sample_rate)
                if self._mode == "eval" or self._settings.realtime_pacing:
                    latency_start, during = await self._send_and_read(channel, pcm)
                else:   # unpaced websocket audio leaves at once: nothing can arrive while it is sent
                    latency_start = await self._send_user_audio(channel, pcm)
                # The reply WAV holds the bot audio from the latency start (the last user speech frame)
                # on: a greeting tail or barge-in overlap heard while the user spoke is not the reply.
                turn = _Turn(
                    voice=self._voice,
                    index=index,
                    latency_start=latency_start,
                    audio_from=latency_start,
                    collect_audio=True,
                )
            elif user_text is not None:
                started = clock.monotonic()
                await channel.send_message(
                    self._message(
                        "send-text",
                        {"content": user_text, "options": {"run_immediately": True, "audio_response": False}},
                    )
                )
                turn = _Turn(
                    voice=self._voice,
                    index=index,
                    latency_start=clock.monotonic(),
                    audio_from=started,
                    collect_audio=self._voice,
                )
            else:   # outbound opener: everything since the connection opened belongs to it
                turn = _Turn(
                    voice=self._voice,
                    index=index,
                    latency_start=clock.monotonic(),
                    audio_from=-math.inf,
                    collect_audio=self._voice,
                )
            reply = await self._collect(channel, turn, opener=opener, during=during)
        except BaseException:
            await self._stop_silence(cancel=True)
            raise
        await self._stop_silence(cancel=False)
        return reply

    async def close(self) -> None:
        """Disconnect (also after the bot closed, or after a failed ``start()``)."""
        await self._stop_silence(cancel=True)
        channel, self._channel = self._channel, None
        self._started = False
        if channel is not None:
            with contextlib.suppress(Exception):
                await channel.close()

    def get_system_prompt(self) -> Optional[str]:
        if self._config.agent.system_prompt is not None:
            return self._config_system_prompt()
        return self._agent_prompt

    def get_warnings(self) -> list[str]:
        return list(self._warnings)

    def get_metadata(self) -> dict:
        return {"pipecat_metrics": [dict(entry) for entry in self._metrics]}

    async def probe_system_prompt(self) -> Optional[str]:
        """Send ``client-message`` ``{"t": "get_system_prompt"}`` and wait up to 3 s for the
        ``server-response`` with the same ``id``; its ``data.d.system_prompt`` (a non-blank string), else
        None. Called by ``start()`` only when config.yaml sets no prompt; ``impulse doctor`` also calls it."""
        channel = self._channel
        if channel is None or self._closed_code is not None:
            return None
        message = self._message("client-message", {"t": GET_SYSTEM_PROMPT})
        request_id = message["id"]
        try:
            await channel.send_message(message)
        except Exception:  # noqa: BLE001 - a request that cannot be sent is an unanswered one
            return None
        clock = self._clock
        deadline = clock.monotonic() + SYSTEM_PROMPT_TIMEOUT_S
        while self._closed_code is None and self._harness_failure is None and clock.monotonic() < deadline:
            item = await next_before(channel.recv, clock, deadline)
            if item is None:
                continue
            if isinstance(item, ChannelMessage) and isinstance(item.data, Mapping) and item.data.get("id") == request_id:
                kind = item.data.get("type")
                if kind == "server-response":
                    payload = item.data.get("data")
                    answer = payload.get("d") if isinstance(payload, Mapping) else None
                    prompt = answer.get("system_prompt") if isinstance(answer, Mapping) else None
                    return prompt if isinstance(prompt, str) and prompt.strip() else None
                if kind == "error-response":
                    return None
            self._intake(item)
        return None

    # -- start -------------------------------------------------------------------------------------

    def _config_system_prompt(self) -> Optional[str]:
        if self._config.agent.system_prompt is None:
            return None
        if not self._config_prompt_loaded:
            self._config_prompt = read_system_prompt(self._config)
            self._config_prompt_loaded = True
        return self._config_prompt

    def _message(self, kind: str, data: dict) -> dict:
        """An RTVI message Impulse sends: label ``rtvi-ai`` and a unique string ``id``."""
        return {"label": RTVI_LABEL, "type": kind, "id": f"{self._id_prefix}-{next(self._ids)}", "data": data}

    async def _wait_for_bot_ready(self, channel: Channel) -> None:
        clock = self._clock
        deadline = clock.monotonic() + self._handshake_timeout_s
        while True:
            if clock.monotonic() >= deadline:
                raise PipecatError(
                    HANDSHAKE_TIMEOUT_ERROR.format(seconds=self._handshake_timeout_s, url=self._shown_url)
                )
            item = await next_before(channel.recv, clock, deadline)
            if item is None:
                continue
            if isinstance(item, ChannelClose):
                raise self._handshake_close_error(item.code)
            if _harness_error(item) is not None:
                self._on_harness_error(item)
                self._raise_harness_failure()
                continue
            if isinstance(item, ChannelMessage) and isinstance(item.data, Mapping) and item.data.get("type") == "bot-ready":
                return
            self._backlog.append(item)

    def _handshake_close_error(self, code: int) -> PipecatError:
        if code == CLOSE_TRY_AGAIN_LATER:
            return PipecatError(ONE_CLIENT_ERROR)
        hint = ""
        if code == CLOSE_FORBIDDEN_DEV_RUNNER:
            hint = " The bot rejected the token or the Origin header: check pipecat.api_key_env and pipecat.headers."
        return PipecatError(
            f"Pipecat bot closed the connection before bot-ready (close code {code}) at {self._shown_url}.{hint}"
        )

    def _start_closed_error(self) -> PipecatError:
        """A close read during ``start()`` after ``bot-ready`` (e.g. during the system prompt wait): the
        bot never took a turn, so it is a scenario error whatever the code."""
        if self._closed_code == CLOSE_TRY_AGAIN_LATER:
            return PipecatError(ONE_CLIENT_ERROR)
        return PipecatError(BOT_CLOSED_ERROR.format(code=self._closed_code))

    def _intake(self, item: ChannelItem) -> None:
        """An item read during ``start()`` that is not an answer: a close is noted, the rest waits for
        the first turn."""
        if isinstance(item, ChannelClose):
            if self._closed_code is None:
                self._closed_code = int(item.code)
            return
        if _harness_error(item) is not None:
            self._on_harness_error(item)
            return
        self._backlog.append(item)

    async def _read_queued(self, channel: Channel) -> None:
        """Before a user turn is sent: every item the channel already holds (zero-wait reads; nothing is
        waited for) goes to the pre-turn backlog, as the items read during ``start()`` do. A greeting
        queued unread while the runner prepared the turn (summary, simulator call, TTS) is pre-turn output
        (its text is added to the reply, it never sets the latency or ends the turn), and a greeting run or
        speech that started while queued stays pre-turn after the send. A close is kept in the backlog, in
        order, so the turn ends on it with the bot's output so far, as when it is read after the send."""
        clock = self._clock
        while True:
            item = await next_before(channel.recv, clock, clock.monotonic())
            if item is None:
                return
            if isinstance(item, ChannelClose):
                self._backlog.append(item)
                return   # nothing follows a close
            self._intake(item)

    def _on_harness_error(self, item: ChannelItem) -> None:
        """An error of Impulse's own eval harness pipeline (``EvalChannel``): a fatal one ends the
        scenario with an error, any other one is a run warning."""
        error = _harness_error(item)
        if error is None:
            return
        raw, fatal = error
        text = self._error_text(raw)
        if fatal:
            if self._harness_failure is None:
                self._harness_failure = text
            return
        if self._error_warnings >= _MAX_ERROR_WARNINGS:
            return
        warning = HARNESS_ERROR_WARNING.format(error=text)
        if warning not in self._warnings:
            self._error_warnings += 1
            self._warn(warning)

    def _raise_harness_failure(self) -> None:
        if self._harness_failure is not None:
            raise PipecatError(HARNESS_FAILED_ERROR.format(error=self._harness_failure))

    def _closed_error(self, turn: Optional[_Turn]) -> BaseException:
        """A close during a turn: 1000 / 1001 are the bot hanging up (``AgentClosed``, a normal end, with
        the turn so far when it has content); any other code (1006 abnormal, 1011 server error, ...) is a
        scenario error; 1013 keeps the one-client text."""
        code = self._closed_code
        if code == CLOSE_TRY_AGAIN_LATER:
            return PipecatError(ONE_CLIENT_ERROR)
        if code not in CLOSE_NORMAL_CODES:
            return PipecatError(BOT_CLOSED_ERROR.format(code=code))
        reply = self._reply(turn, self._clock.monotonic()) if turn is not None and turn.has_content() else None
        return AgentClosed(BOT_CLOSED_ERROR.format(code=code), reply=reply)

    def _warn(self, text: str) -> None:
        if text not in self._warnings:
            self._warnings.append(text)

    def _redact(self, text: str) -> str:
        """``text`` with every key the runner knows (``redact``) and every credential the adapter holds
        (``_held_secrets``) replaced by ``[REDACTED]``."""
        if self._redact_text is not None:
            text = self._redact_text(text)
        for value in self._secrets:
            text = text.replace(value, REDACTED)
        return text

    def _error_text(self, text: str) -> str:
        """A bot or harness error text for a warning or a scenario error: redacted FIRST, then one line
        of at most 200 characters (a key cut in half could no longer be redacted)."""
        return " ".join(self._redact(text).split())[:_ERROR_TEXT_CHARS]

    # -- user audio --------------------------------------------------------------------------------

    async def _send_and_read(self, channel: Channel, pcm: bytes) -> tuple[float, list[tuple[ChannelItem, float]]]:
        """Send the user's speech while reading what the bot sends meanwhile. Returns the latency start
        (the time the last speech frame left Impulse) and the items read during sending, each with the
        clock time it was read, so the turn can tell output that came before the end of the user's speech
        (a greeting tail, a barge-in) from the reply."""
        clock = self._clock
        sending = asyncio.ensure_future(self._send_user_audio(channel, pcm))
        receiving: Optional[asyncio.Future[ChannelItem]] = None
        read: list[tuple[ChannelItem, float]] = []
        try:
            while True:
                if receiving is None:
                    receiving = asyncio.ensure_future(channel.recv())
                waiting: set[asyncio.Future[Any]] = {sending, receiving}
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if receiving.done():
                    read.append((receiving.result(), clock.monotonic()))
                    receiving = None
                if sending.done():
                    break
        except BaseException:
            _discard(sending)
            if receiving is not None:
                _discard(receiving)
            raise
        if receiving is not None:   # recv() is cancel-safe (SPEC.md section 9 Channel); keep an item it produced
            receiving.cancel()
            try:
                read.append((await receiving, clock.monotonic()))
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    _discard(sending)
                    raise
        return sending.result(), read

    async def _send_user_audio(self, channel: Channel, pcm: bytes) -> float:
        """Send the user's speech; returns the clock time its last non-silent sample left Impulse (the
        TTFA start). Eval mode: one ``send_audio`` call, Pipecat's client transport paces it; a value
        that is not a time means the send path did not report the speech, a scenario error (never "now").
        Websocket mode: ``chunk_ms`` frames, paced in real time when ``realtime_pacing``, then
        ``trailing_silence_ms`` of silence (sent while the reply is collected)."""
        clock = self._clock
        rate = self._sample_rate
        if self._mode == "eval":
            returned = _number(await channel.send_audio(pcm, rate))
            if returned is None:
                raise PipecatError(SEND_NOT_REPORTED_ERROR)
            return returned
        settings = self._settings
        chunk_ms = max(1, int(settings.chunk_ms))
        chunk_s = chunk_ms / 1000.0
        frame_bytes = max(1, round(rate * chunk_ms / 1000.0)) * 2
        pacing = bool(settings.realtime_pacing)
        last = _last_speech_index(pcm)
        last_speech_frame = (last * 2) // frame_bytes if last >= 0 else -1
        latency_start: Optional[float] = None
        next_at = clock.monotonic()
        for number, offset in enumerate(range(0, len(pcm), frame_bytes)):
            frame = pcm[offset : offset + frame_bytes]
            if len(frame) < frame_bytes:
                frame = frame + bytes(frame_bytes - len(frame))
            returned = await channel.send_audio(frame, rate)
            if number == last_speech_frame:
                latency_start = _finite(returned, clock.monotonic())
            if pacing:
                next_at += chunk_s
                await clock.sleep(max(0.0, next_at - clock.monotonic()))
        if latency_start is None:
            latency_start = clock.monotonic()
        count = math.ceil(max(0, int(settings.trailing_silence_ms)) / chunk_ms)
        if count:
            silence = bytes(frame_bytes)
            if pacing:
                self._silence_task = asyncio.ensure_future(self._trailing_silence(channel, silence, count, next_at, chunk_s))
            else:
                for _ in range(count):
                    await channel.send_audio(silence, rate)
        return latency_start

    async def _trailing_silence(self, channel: Channel, silence: bytes, count: int, next_at: float, chunk_s: float) -> None:
        """Silence after user speech, paced in real time, so the bot's VAD ends the user's turn."""
        clock = self._clock
        with contextlib.suppress(Exception):   # a connection that went away needs no more silence
            for _ in range(count):
                await channel.send_audio(silence, self._sample_rate)
                next_at += chunk_s
                await clock.sleep(max(0.0, next_at - clock.monotonic()))

    async def _stop_silence(self, *, cancel: bool) -> None:
        task, self._silence_task = self._silence_task, None
        if task is None:
            return
        if cancel and not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # -- turns -------------------------------------------------------------------------------------

    async def _collect(
        self, channel: Channel, turn: _Turn, *, opener: bool, during: list[tuple[ChannelItem, float]]
    ) -> AgentReply:
        clock = self._clock
        while self._backlog:
            self._process(self._backlog.popleft(), turn, pre_turn=not opener)
        for early_item, read_at in during:   # read while the user's speech was being sent
            self._process(early_item, turn, at=read_at)
        timeout_at = clock.monotonic() + self._turn_timeout_s
        while True:
            self._raise_harness_failure()
            if self._closed_code is not None:
                raise self._closed_error(turn)
            now = clock.monotonic()
            end_at = turn.end_since + self._grace_s if turn.end_since is not None else None
            if end_at is not None and now >= end_at:
                reply = self._reply(turn, now)
                # A5-rev: an announcement still open at the end of the turn stays in the reply; its
                # completion, when it arrives later, is not added again.
                self._kept_announcements.extend(
                    (text.strip(), key[1] if key[0] == "segment" else None)
                    for (_, key), text in turn.announced.items()
                )
                turn.announced.clear()
                return reply
            if now >= timeout_at:
                # A1: the turn so far goes with the error; an open announcement is not remembered.
                raise self._turn_timeout(turn, now)
            deadline = timeout_at if end_at is None else min(end_at, timeout_at)
            item = await next_before(channel.recv, clock, deadline)
            if item is not None:
                self._process(item, turn)

    def _process(self, item: ChannelItem, turn: _Turn, *, pre_turn: bool = False, at: Optional[float] = None) -> None:
        """One item of the turn. ``pre_turn``: read during ``start()`` (never ends the turn, sets no
        latency). ``at``: the clock time the item was read when that was earlier than now (items read
        while the user's speech was being sent); items read before ``turn.latency_start`` (the last speech
        frame sent) are "early": they update the state and the reply text, but never start the grace
        window and never count as the bot's reply (SPEC.md section 9 end of turn). An LLM run or a bot
        speech that STARTED early stays pre-turn until it stops, also for its items read later (bot
        output in flight when the user's turn was sent): its text and audio are kept, but it never sets
        the latency, never counts as the reply, and its stop never starts the grace window."""
        if isinstance(item, ChannelClose):
            if self._closed_code is None:
                self._closed_code = int(item.code)
            return
        if isinstance(item, ChannelAudio):
            if turn.collect_audio and not pre_turn:
                self._on_audio(item, turn)
            return
        if not isinstance(item, ChannelMessage) or not isinstance(item.data, Mapping):
            return
        if _harness_error(item) is not None:
            self._on_harness_error(item)
            return
        data = item.data
        kind = data.get("type")
        payload = data.get("data")
        payload = payload if isinstance(payload, Mapping) else {}
        now = at if at is not None else self._clock.monotonic()
        early = pre_turn or now < turn.latency_start
        if kind == "bot-llm-started":
            self._llm_active = True
            turn.run_open = True
            turn.run_pre_turn = early
            turn.runs_for(early).append([])
            if not turn.voice:
                turn.end_since = None
        elif kind == "bot-llm-text":
            text = payload.get("text")
            if isinstance(text, str):
                if not turn.run_open:   # text without bot-llm-started opens a run of its own
                    turn.run_open = True
                    turn.run_pre_turn = early
                    turn.runs_for(early).append([])
                turn.runs_for(turn.run_pre_turn)[-1].append(text)   # the open run is the last of its list
                if not turn.run_pre_turn:   # a pre-turn run's text is kept but is never the reply's latency
                    self._first_text(turn, now, pre_turn)
                    turn.replied = turn.replied or not early
                    turn.reply_llm_text = turn.reply_llm_text or not early
        elif kind == "bot-llm-stopped":
            self._llm_active = False
            turn.run_open = False
            pre_turn_run, turn.run_pre_turn = turn.run_pre_turn, False
            if not turn.voice and not pre_turn and not pre_turn_run:
                turn.stop_seen = True
                self._maybe_end_text(turn, now)
        elif kind == "bot-output":
            in_flight = turn.run_pre_turn or turn.speech_pre_turn   # output of a pre-turn run or speech
            self._on_bot_output(payload, turn, now, pre_turn or in_flight)
            if not in_flight:
                turn.replied = turn.replied or (not early and isinstance(payload.get("text"), str))
            elif (
                turn.speech_pre_turn
                and not early
                and turn.reply_llm_text
                and isinstance(payload.get("text"), str)
                and (payload.get("spoken") is True or payload.get("spoken_status") == "completed")
            ):   # the reply's text SPOKEN during the greeting's speech: one speech holds both
                turn.reply_spoken_in_speech = True
        elif kind == "bot-started-speaking":
            self._bot_speaking = True
            turn.speech_pre_turn = early
            turn.reply_spoken_in_speech = False
            if turn.voice:
                turn.end_since = None
            turn.replied = turn.replied or not early
        elif kind == "bot-stopped-speaking":
            self._bot_speaking = False
            pre_turn_speech, turn.speech_pre_turn = turn.speech_pre_turn, False
            merged, turn.reply_spoken_in_speech = turn.reply_spoken_in_speech, False
            # Voice rule: only a stop that follows bot output seen after the user's last speech frame
            # was sent starts the grace window (a greeting tail or a barge-in stop never ends the turn),
            # not the stop of a speech that started before it (pre-turn speech) unless a SPOKEN
            # bot-output of the reply was read during that speech after the reply's own LLM text (the
            # reply spoken right behind the greeting, with no stop / start between them; LLM text alone
            # may still wait for slow TTS), and not while a function call is in progress (its stop
            # starts the window then).
            if turn.voice and not early and turn.replied and (not pre_turn_speech or merged):
                turn.stop_seen = True
                turn.end_since = now if not self._calls_in_progress else None
        elif kind == "bot-interrupted":
            # N-M2: the completions of kept announcements will never come; forget them.
            self._kept_announcements.clear()
        elif kind == "user-started-speaking":
            if self._bot_speaking and not pre_turn:
                turn.interrupted = True
        elif kind == "user-transcription":
            text = payload.get("text")
            if not pre_turn and payload.get("final") is True and isinstance(text, str):
                turn.heard.append(text)
        elif kind == "llm-function-call-in-progress":
            self._on_call_in_progress(payload, turn)
        elif kind == "llm-function-call-stopped":
            self._on_call_stopped(payload, turn, now)
        elif kind == "llm-function-call":
            self._on_deprecated_call(payload, turn)
        elif kind == "metrics":
            self._on_metrics(payload, turn)
        elif kind in ("error", "error-response"):
            self._on_bot_error(payload, turn)

    def _first_text(self, turn: _Turn, now: float, pre_turn: bool) -> None:
        if not pre_turn and turn.first_text_s is None and now >= turn.latency_start:
            turn.first_text_s = now

    def _maybe_end_text(self, turn: _Turn, now: float) -> None:
        """Text rule: the LLM stopped during this turn, it is not running, and no function call is in
        progress; the grace window starts now."""
        if turn.stop_seen and not self._llm_active and not self._calls_in_progress and turn.end_since is None:
            turn.end_since = now

    def _on_bot_output(self, payload: Mapping[str, Any], turn: _Turn, now: float, pre_turn: bool) -> None:
        """``pre_turn``: pre-turn output (read during ``start()`` or queued before the send, or of a
        pre-turn run or speech): kept in ``pre_outputs``, never the latency."""
        text = payload.get("text")
        if not isinstance(text, str):
            return
        spoken = payload.get("spoken") is True or payload.get("spoken_status") == "completed"
        outputs = turn.pre_outputs if pre_turn else turn.outputs
        segment = payload.get("segment_id")
        key: Any = ("segment", segment) if segment is not None else ("message", len(outputs), id(payload))
        if turn.voice:
            self._first_text(turn, now, pre_turn)
            every = turn.pre_all_outputs if pre_turn else turn.all_outputs
            every[key if segment is not None else ("message", len(every), id(payload))] = text
            if spoken and key not in turn.outputs and key not in turn.pre_outputs:
                outputs[key] = text   # each spoken segment counted once
            return
        # Text mode, fixes round 2 A5-rev: a -t eval bot re-enables TTS after a tool call, and a push-text
        # TTS reports each sentence twice under different segment ids (announced before synthesis,
        # completed after it). An announced sentence is kept until its completion (same text) arrives in
        # the same turn, which then takes its place; one still open at the end of the turn stays in that
        # reply, and its later completion is not added again (_kept_announcements). Re-check N-M2: any
        # later bot-output of a kept segment (word-timestamp progress or completion, read after the turn
        # ended) is not added either, checked before the open-announcement match.
        if segment is not None:
            remembered = next(
                (index for index, (_, kept) in enumerate(self._kept_announcements) if kept == segment), None
            )
            if remembered is not None:
                if spoken:
                    del self._kept_announcements[remembered]   # its completion: forget the segment
                return
        if payload.get("will_be_spoken") is True and not spoken:
            self._first_text(turn, now, pre_turn)
            outputs[key] = text       # word-timestamp progress: the latest text of the segment
            turn.announced[(pre_turn, key)] = text
            return
        if spoken:
            trimmed = text.strip()
            place = next((where for where in turn.announced if where[1] == key), None)
            if place is None:
                remembered = next(
                    (index for index, (said, _) in enumerate(self._kept_announcements) if said == trimmed), None
                )
                if remembered is not None:
                    del self._kept_announcements[remembered]   # the late completion of a kept announcement
                    return
            if place is None:
                place = next((where for where, said in turn.announced.items() if said.strip() == trimmed), None)
            if place is not None:
                del turn.announced[place]
                in_pre, old_key = place
                target = turn.pre_outputs if in_pre else turn.outputs
                replaced = {(key if k == old_key else k): (text if k == old_key else v) for k, v in target.items()}
                target.clear()
                target.update(replaced)
                self._first_text(turn, now, pre_turn)
                return
        self._first_text(turn, now, pre_turn)
        turn.announced.pop((pre_turn, key), None)
        outputs[key] = text           # text mode: the latest text of each segment

    def _on_audio(self, item: ChannelAudio, turn: _Turn) -> None:
        if item.arrival_s < turn.audio_from:
            return
        pcm = bytes(item.pcm)
        rate = int(item.sample_rate)
        if turn.audio_rate is None:
            turn.audio_rate = rate
        elif rate != turn.audio_rate and rate > 0:
            pcm = _resample_pcm(pcm, rate, turn.audio_rate)
        turn.audio.append(pcm)
        # TTFA: the first speech audio at or after the latency start that is not part of a pre-turn
        # speech (a greeting still playing); after that speech stopped, bot audio counts again.
        if (
            turn.first_audio_s is None
            and not turn.speech_pre_turn
            and item.arrival_s >= turn.latency_start
            and _is_speech(item.pcm)
        ):
            turn.first_audio_s = item.arrival_s
            turn.replied = True   # bot speech after the user's last speech frame

    # -- tool calls (SPEC.md section 9 mapping) ------------------------------------------------------

    def _check_reported(self, payload: Mapping[str, Any], arguments_key: str) -> None:
        name = payload.get("function_name")
        if not (isinstance(name, str) and name) or payload.get(arguments_key) is None:
            self._warn(FUNCTION_CALL_WARNING)

    def _on_call_in_progress(self, payload: Mapping[str, Any], turn: _Turn) -> None:
        self._check_reported(payload, "arguments")
        name = payload.get("function_name")
        call_id = payload.get("tool_call_id")
        key = str(call_id) if call_id is not None else None
        call = self._calls_by_id.get(key) if key is not None else None
        if call is None:
            call = ToolCall(
                name=name if isinstance(name, str) and name else "unknown",
                arguments=_tool_arguments(payload.get("arguments")),
            )
            turn.tool_calls.append(call)
            if key is not None:
                self._calls_by_id[key] = call
        if key is not None:
            self._calls_in_progress.add(key)
        turn.end_since = None   # both modes: a call in progress holds the turn open

    def _on_call_stopped(self, payload: Mapping[str, Any], turn: _Turn, now: float) -> None:
        call_id = payload.get("tool_call_id")
        key = str(call_id) if call_id is not None else None
        name = payload.get("function_name")
        call = self._calls_by_id.get(key) if key is not None else None
        if call is None:   # the in-progress report was missed: record the call from its stop
            call = ToolCall(name=name if isinstance(name, str) and name else "unknown", arguments={})
            turn.tool_calls.append(call)
            if key is not None:
                self._calls_by_id[key] = call
        elif call.name == "unknown" and isinstance(name, str) and name:
            call.name = name
        result = payload.get("result")
        if result is not None:
            call.result = dict(result) if isinstance(result, Mapping) else {"value": result}
        if payload.get("cancelled") is True:
            call.success = False
            call.error = "cancelled"
        elif isinstance(call.result, Mapping) and isinstance(call.result.get("success"), bool):
            call.success = call.result["success"]   # SPEC.md section 9: no error text on this branch
            if call.success and id(call) in turn.enriched:
                call.error = None   # A2-rev rule 4: never success=True with an error these rules set
        elif isinstance(call.result, Mapping) and isinstance(call.result.get("error"), str) and call.result["error"]:
            call.success = False
            call.error = call.result["error"]
        elif isinstance(result, str) and _FAILED_CALL_RESULT.match(result.strip()):
            # A2-rev rule 1: pipecat-ai's result for a handler that raised: the error is the oldest pending
            # RTVI error of its name in this turn (consumed), else the template (stripped), which a later
            # RTVI error replaces (rule 2). The stored result stays as received.
            pending = turn.pending_errors.get(call.name)
            call.success = False
            turn.enriched.add(id(call))
            if pending:
                call.error = pending.pop(0)
                turn.template_errors.pop(id(call), None)
            else:
                call.error = result.strip()
                turn.template_errors[id(call)] = call.error
        if key is not None:
            self._calls_in_progress.discard(key)
        if not turn.voice:
            self._maybe_end_text(turn, now)
        elif turn.stop_seen and not self._bot_speaking and not self._calls_in_progress and turn.end_since is None:
            turn.end_since = now   # voice: the grace window restarts at the stop of the last call

    def _on_deprecated_call(self, payload: Mapping[str, Any], turn: _Turn) -> None:
        """The deprecated ``llm-function-call`` (older Pipecat): name from ``function_name``, arguments
        from ``args``, ``success`` not captured."""
        self._check_reported(payload, "args")
        name = payload.get("function_name")
        call = ToolCall(name=name if isinstance(name, str) and name else "unknown", arguments=_tool_arguments(payload.get("args")))
        turn.tool_calls.append(call)

    @staticmethod
    def _tie_pending_errors(turn: _Turn) -> None:
        """A2-rev rule 3, when the turn's reply is built (normal end, timeout or close partial reply):
        each pending RTVI error still unconsumed, oldest first, ties to the most recent call of its name
        in this turn that is not captured (``success`` None and ``result`` None: its stop never arrived,
        or arrived without a result): ``success=False``, ``error`` = the text. The rest are dropped (they
        stay only the run warning); pending errors never carry over to another turn."""
        for name, texts in turn.pending_errors.items():
            for text in texts:
                call = next(
                    (
                        call
                        for call in reversed(turn.tool_calls)
                        if call.name == name and call.success is None and call.result is None
                    ),
                    None,
                )
                if call is None:
                    break   # a tied call is no longer "not captured": none is left for the others
                call.success = False
                call.error = text
                turn.enriched.add(id(call))
        turn.pending_errors.clear()

    # -- metrics and errors --------------------------------------------------------------------------

    def _on_metrics(self, payload: Mapping[str, Any], turn: _Turn) -> None:
        """RTVI ``metrics``: ``ttfb`` and ``processing`` entries, reported in seconds, stored in ms."""
        for kind in ("ttfb", "processing"):
            entries = payload.get(kind)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                value = _number(entry.get("value"))
                if value is None:
                    continue
                processor = entry.get("processor")
                model = entry.get("model")
                self._metrics.append(
                    {
                        "turn_index": turn.index,
                        "kind": kind,
                        "processor": str(processor) if processor is not None else None,
                        "model": str(model) if model is not None else None,
                        "value_ms": value * 1000.0,
                    }
                )

    def _on_bot_error(self, payload: Mapping[str, Any], turn: _Turn) -> None:
        """RTVI ``error`` / ``error-response``: a run warning (at most five). A2-rev rule 2: an "Error
        executing function call [NAME]: ..." text (redacted, one line, 200 characters) replaces the
        template error of the most recent call of this turn named NAME that failed through the template
        and still carries it; otherwise it is pending for NAME in this turn (rules 1 and 3). It never
        changes a successful call, a dict result, or a cancelled call (rule 4)."""
        text = payload.get("message") or payload.get("error")
        if not isinstance(text, str) or not text.strip():
            return
        failed = _FAILED_CALL_ERROR.match(text)
        if failed is not None:
            name = failed.group("name")
            shown = self._error_text(text)
            call = next(
                (
                    call
                    for call in reversed(turn.tool_calls)
                    if call.name == name
                    and id(call) in turn.template_errors
                    and call.error == turn.template_errors[id(call)]
                ),
                None,
            )
            if call is not None:
                call.error = shown
                del turn.template_errors[id(call)]
            else:
                turn.pending_errors.setdefault(name, []).append(shown)
        if self._error_warnings >= _MAX_ERROR_WARNINGS:
            return
        warning = f"Pipecat bot reported an RTVI error: {self._error_text(text)}"
        if warning not in self._warnings:
            self._error_warnings += 1
            self._warn(warning)

    # -- reply -------------------------------------------------------------------------------------

    def _turn_timeout(self, turn: _Turn, now: float) -> PipecatTurnTimeout:
        """A1: the turn timeout with the turn so far as ``reply`` when it has content (non-blank text, a
        tool call, or audio), else None."""
        reply = self._reply(turn, now, timed_out=True)
        has_content = bool(reply.text.strip() or reply.tool_calls or reply.audio_bytes)
        return PipecatTurnTimeout(
            TURN_TIMEOUT_ERROR.format(seconds=self._turn_timeout_s), reply=reply if has_content else None
        )

    def _reply(self, turn: _Turn, ended_at: float, *, timed_out: bool = False) -> AgentReply:
        """The reply text is the pre-turn part first, then the reply's own part, each by the same rule:
        its ``bot-output`` texts (voice: the spoken segments only), else (text mode) its LLM runs.
        ``timed_out`` (A1): in voice mode a part with no spoken ``bot-output`` falls back to every
        ``bot-output`` text of that part, else its LLM runs; the latency is NaN when neither speech audio
        (voice) nor text arrived (a normal reply then measures to ``ended_at``). Pending RTVI errors are
        tied first (A2-rev rule 3), so the copies carry them."""
        self._tie_pending_errors(turn)
        if timed_out and turn.voice:
            pre = _reply_part(turn.pre_outputs, [], voice=True) or _reply_part(
                turn.pre_all_outputs, turn.pre_runs, voice=False
            )
            own = _reply_part(turn.outputs, [], voice=True) or _reply_part(turn.all_outputs, turn.llm_runs, voice=False)
        else:
            pre = _reply_part(turn.pre_outputs, turn.pre_runs, voice=turn.voice)
            own = _reply_part(turn.outputs, turn.llm_runs, voice=turn.voice)
        text = " ".join(part for part in (pre, own) if part)
        heard = " ".join(part.strip() for part in turn.heard if part.strip())
        audio = _wav(b"".join(turn.audio), turn.audio_rate) if turn.audio_rate is not None else None
        first = turn.first_audio_s if turn.voice and turn.first_audio_s is not None else None
        if first is None:
            first = turn.first_text_s
        if first is not None:
            latency_ms = max(0.0, (first - turn.latency_start) * 1000.0)
        elif timed_out:
            latency_ms = float("nan")   # never the timeout instant
        else:
            latency_ms = max(0.0, (ended_at - turn.latency_start) * 1000.0)
        return AgentReply(
            text=text,
            audio_bytes=audio,
            latency_ms=latency_ms,
            heard_text=heard or None,
            interrupted=turn.interrupted,
            # Copies: a late llm-function-call-stopped (read in a later turn) updates the adapter's own
            # ToolCall, never one in a reply already returned.
            tool_calls=[call.model_copy(deep=True) for call in turn.tool_calls],
        )


def _reply_part(outputs: Mapping[Any, str], runs: list[list[str]], *, voice: bool) -> str:
    """One part of a reply's text: its ``bot-output`` texts (each stripped, joined with one space) when
    there are any or in voice mode, otherwise its LLM runs (a run's tokens joined with "", stripped; runs
    joined with one space)."""
    texts = [text.strip() for text in outputs.values() if text.strip()]
    if voice or texts:
        return " ".join(texts)
    joined = ["".join(run).strip() for run in runs]
    return " ".join(run for run in joined if run)


def _finite(value: Any, fallback: float) -> float:
    number = _number(value)
    return number if number is not None else fallback


def _harness_error(item: Any) -> Optional[tuple[str, bool]]:
    """``(text, fatal)`` for the ``EvalChannel`` harness-error item (no RTVI label), else None. The text
    is returned whole: the adapter redacts it before shortening it (``PipecatAdapter._error_text``)."""
    if not isinstance(item, ChannelMessage) or not isinstance(item.data, Mapping):
        return None
    if item.data.get("type") != HARNESS_ERROR or item.data.get("label") is not None:
        return None
    payload = item.data.get("data")
    payload = payload if isinstance(payload, Mapping) else {}
    return str(payload.get("error") or "unknown error"), payload.get("fatal") is True


def _discard(task: asyncio.Future[Any]) -> None:
    """Cancel a helper task whose outcome no longer matters (its exception is marked as retrieved)."""
    task.cancel()
    task.add_done_callback(lambda done: done.cancelled() or done.exception())


def _connect_headers(settings: PipecatConfig) -> dict[str, str]:
    """``pipecat.headers`` plus ``Authorization: Bearer <value>`` when ``api_key_env`` names a variable
    (environment first, then config.toml ``[keys]``, SPEC.md section 8.1)."""
    headers = {str(key): str(value) for key, value in (settings.headers or {}).items()}
    name = (settings.api_key_env or "").strip()
    if name:
        value = resolve_key(name, load_project_toml())
        if not value:
            raise ConfigError(
                "agent.pipecat.api_key_env",
                f"the Pipecat bot token is not set. Set the environment variable {name} or add it to [keys] "
                "in .impulse/config.toml.",
            )
        headers = {key: val for key, val in headers.items() if key.lower() != "authorization"}
        headers["Authorization"] = f"Bearer {value}"
    return headers


def _held_secrets(settings: PipecatConfig, headers: Mapping[str, str]) -> tuple[str, ...]:
    """The credentials the adapter holds itself, for redaction (longest first): the ``pipecat.url``
    credentials (``url_credentials``) and every connect header value except ``Origin`` (a public
    origin), which includes the resolved ``api_key_env`` token (``secret_text``: the token of
    ``Bearer <token>``). The same values the runner's redactor knows from the config, without needing it."""
    values = list(url_credentials(settings.url))
    values.extend(
        secret_text(value)
        for name, value in headers.items()
        if isinstance(value, str) and str(name).strip().lower() != "origin"
    )
    unique = {value for value in values if isinstance(value, str) and len(value) >= _MIN_SECRET_CHARS}
    return tuple(sorted(unique, key=len, reverse=True))


# ------------------------------------------------------------------------------------------------
# The real channels; pipecat-ai is imported lazily, only here
# ------------------------------------------------------------------------------------------------


def quiet_pipecat_logging() -> None:
    """Turn pipecat's log output off. Pipecat logs through loguru at DEBUG level to stderr by default
    (and an INFO banner when the package is imported), which would flood Impulse's console; call this
    before anything imports ``pipecat``. Errors of Impulse's own eval harness reach the adapter through
    the harness worker's ``on_pipeline_error`` handler (``EvalChannel``), not through the log."""
    with contextlib.suppress(Exception):
        from loguru import logger

        logger.disable("pipecat")


def _pipecat_version() -> str:
    """The installed pipecat-ai version (package metadata; ``pipecat`` itself is not imported)."""
    from importlib import metadata

    try:
        return metadata.version("pipecat-ai")
    except Exception:  # noqa: BLE001 - metadata missing or unreadable
        return "unknown"


def _is_tested_pipecat(version: str) -> bool:
    return re.match(rf"{re.escape(TESTED_PIPECAT_SERIES)}(\.|$)", version) is not None


def _load_pipecat(mode: str) -> None:
    """Check that the pipecat-ai modules ``mode`` needs import, before any connection. A missing
    package or ``pipecat.evals`` is a config error with the SPEC.md section 9 text."""
    if not _module_available("pipecat"):
        raise ConfigError("agent.adapter", MISSING_EXTRA)
    quiet_pipecat_logging()   # before anything imports the pipecat package (it logs a banner at import)
    if mode == "eval" and not _module_available("pipecat.evals"):
        raise ConfigError("agent.pipecat.mode", EVALS_MISSING)
    names = ["pipecat.frames.frames", "pipecat.serializers.protobuf"]
    if mode == "eval":
        names = [
            "pipecat.frames.frames",
            "pipecat.evals.client_transport",
            "pipecat.evals.serializer",
            "pipecat.pipeline.pipeline",
            "pipecat.pipeline.worker",
            "pipecat.processors.frame_processor",
            "pipecat.transports.websocket.client",
            "pipecat.workers.runner",
        ]
    for name in names:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == "pipecat":
                raise ConfigError("agent.adapter", MISSING_EXTRA) from None
            if missing.startswith("pipecat.evals"):
                raise ConfigError("agent.pipecat.mode", EVALS_MISSING) from None
            raise _import_problem(exc) from None
        except (ImportError, OSError) as exc:
            raise _import_problem(exc) from None


def _import_problem(exc: BaseException) -> ConfigError:
    return ConfigError(
        "agent.adapter",
        f"the pipecat extra is installed but could not be imported ({type(exc).__name__}: {exc}). "
        "Reinstall it with pip install impulse-evals[pipecat]; on Windows, a missing vcruntime140_1.dll means "
        "the Microsoft Visual C++ Redistributable (x64) needs to be installed or updated.",
    )


def _real_channel_factory(adapter: PipecatAdapter) -> Callable[[], Channel]:
    """The real channel of ``pipecat.mode``; checks the extra before any connection. Eval mode also
    builds (and checks) Impulse's subclasses of Pipecat's private eval client classes now, and warns once
    when the installed pipecat-ai is not the tested 1.12.x."""
    mode = adapter._mode
    _load_pipecat(mode)
    clock = adapter._clock
    voice = adapter._voice
    sample_rate = adapter._sample_rate
    redact = adapter._redact
    if mode == "eval":
        _eval_classes()
        version = _pipecat_version()
        if not _is_tested_pipecat(version):
            adapter._warn(UNTESTED_VERSION_WARNING.format(version=version))
        return lambda: EvalChannel(clock=clock, voice=voice, sample_rate=sample_rate, redact=redact)
    return lambda: WebsocketChannel(clock=clock)


class _SpeechTracker:
    """``WebsocketChannel.send_audio`` returns the time the last non-silent sample left (per frame)."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.last_speech_s: Optional[float] = None

    def returned(self, sent_speech_at: Optional[float]) -> float:
        if sent_speech_at is not None:
            self.last_speech_s = sent_speech_at
        return self.last_speech_s if self.last_speech_s is not None else self.clock.monotonic()


class WebsocketChannel:
    """Websocket mode: ``websockets`` plus Pipecat's ``ProtobufFrameSerializer`` (binary frames: PCM audio
    frames, and RTVI JSON inside message frames). The adapter chunks and paces the user's audio.

    Not exercised against a live bot by the test suite (see DECISIONS.md).
    """

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock
        self._queue: asyncio.Queue[ChannelItem] = asyncio.Queue()
        self._speech = _SpeechTracker(clock)
        self._ws: Any = None
        self._reader: Optional[asyncio.Task[Any]] = None
        self._serializer: Any = None
        self._frames: Any = None
        self._closed = False

    async def connect(self, url: str, headers: dict, query: dict) -> None:
        from websockets.asyncio.client import connect as websocket_connect

        quiet_pipecat_logging()
        from pipecat.frames import frames
        from pipecat.serializers.protobuf import ProtobufFrameSerializer

        self._frames = frames
        self._serializer = ProtobufFrameSerializer()
        target = _with_query(url, query)
        try:
            self._ws = await websocket_connect(target, additional_headers=dict(headers) or None, open_timeout=10)
        except Exception as exc:  # noqa: BLE001 - refused, timeout, HTTP status, bad URL, ...
            raise PipecatError(
                f"could not connect to the Pipecat bot at {strip_url_credentials(url)}: {type(exc).__name__}: {exc}"
            ) from None
        self._reader = asyncio.ensure_future(self._read())

    async def send_message(self, data: dict) -> None:
        frame = self._frames.OutputTransportMessageUrgentFrame(message=dict(data))
        await self._send(await self._serializer.serialize(frame))

    async def send_audio(self, pcm: bytes, sample_rate: int) -> float:
        frame = self._frames.OutputAudioRawFrame(audio=bytes(pcm), sample_rate=int(sample_rate), num_channels=1)
        await self._send(await self._serializer.serialize(frame))
        return self._speech.returned(self._clock.monotonic() if _is_speech(pcm) else None)

    async def recv(self) -> ChannelItem:
        return await self._queue.get()

    async def close(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._ws.close(), timeout=5)
        if self._reader is not None:
            if not self._reader.done():
                self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        self._put_close(1000)

    async def _send(self, payload: Any) -> None:
        if not payload or self._ws is None:
            return
        with contextlib.suppress(Exception):   # a closed connection shows up as ChannelClose on recv()
            await self._ws.send(payload)

    def _put_close(self, code: int) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(ChannelClose(code=code))

    async def _read(self) -> None:
        frames = self._frames
        try:
            async for message in self._ws:
                arrival = self._clock.monotonic()
                if isinstance(message, str):
                    with contextlib.suppress(ValueError):
                        data = json.loads(message)
                        if isinstance(data, dict):
                            self._queue.put_nowait(ChannelMessage(data=data))
                    continue
                try:
                    frame = await self._serializer.deserialize(bytes(message))
                except Exception:  # noqa: BLE001, S112 - an undecodable frame is skipped
                    continue
                if isinstance(frame, frames.InputTransportMessageFrame) and isinstance(frame.message, dict):
                    self._queue.put_nowait(ChannelMessage(data=dict(frame.message)))
                elif isinstance(frame, frames.InputAudioRawFrame):
                    pcm = _downmix_pcm(frame.audio, int(frame.num_channels or 1))
                    self._queue.put_nowait(ChannelAudio(pcm=pcm, sample_rate=int(frame.sample_rate), arrival_s=arrival))
        except Exception:  # noqa: BLE001, S110 - ConnectionClosed with an error code ends the stream too
            pass
        code = getattr(self._ws, "close_code", None)
        self._put_close(int(code) if code is not None else 1006)


def _with_query(url: str, query: Mapping[str, Any]) -> str:
    """``url`` with ``query`` flags appended the way Pipecat's ``EvalClient`` builds its URL."""
    flags = [f"{key}={value}" for key, value in (query or {}).items() if value not in (None, False, "")]
    if not flags:
        return url
    return f"{url}{'&' if '?' in url else '?'}{'&'.join(flags)}"


class EvalChannel:
    """Eval mode: Pipecat's ``EvalClientTransport`` with ``EvalClientSerializer`` in a small harness
    pipeline owned by Impulse (``input -> sink -> output``; never the user's pipeline), as Pipecat's own
    ``EvalClient`` builds it.

    - Raw RTVI messages are read before the serializer by overriding ``RTVIClientTransport._on_message``
      (private, version-sensitive: pinned to pipecat-ai 1.12.0, DECISIONS.md), because the client
      serializer drops ``bot-ready``, ``bot-output``, ``metrics``, ``server-response`` and others.
    - Bot audio (``eval-bot-audio``, resampled to 16 kHz by the serializer) arrives through
      ``EvalClientRecorder.add_bot``, before Pipecat's gap-fill, with its arrival time.
    - User audio is injected like Pipecat's ``EvalClient.play`` (TTS-bracketed frames) and paced by
      ``EvalClientOutputTransport``; the time the last speech frame was sent comes from its send path
      (``_send_frame``, private), not from ``EvalClientRecorder.add_user`` (which fires on queueing).
      When the send path does not report the audio, ``send_audio`` raises (never a made-up time).
    - Errors of the harness pipeline reach the adapter through the worker's ``on_pipeline_error``
      handler (as Pipecat's ``EvalClient``), as ``HARNESS_ERROR`` items: fatal ones end the scenario.
      Their texts pass through ``redact`` (the adapter's redaction) before they are shortened. A fatal
      one during ``connect()`` (e.g. a rejected WebSocket upgrade: HTTP 401 / 403 / 404) fails it at
      once, instead of after the connect timeout.
    - Pipecat's client transport sends ``client-ready`` itself on connect.

    Not exercised against a live bot by the test suite (see DECISIONS.md).
    """

    CONNECT_TIMEOUT_S = 12.0     # Pipecat opens the WebSocket with open_timeout=10
    PROBE_TIMEOUT_S = 5.0
    END_TIMEOUT_S = 3.0
    SEND_REPORT_MARGIN_S = 10.0  # beyond the audio's own duration, before the send path counts as silent

    def __init__(
        self, *, clock: Clock, voice: bool, sample_rate: int, redact: Optional[Callable[[str], str]] = None
    ) -> None:
        self._clock = clock
        self._voice = voice
        self._sample_rate = int(sample_rate)
        self._redact = redact
        self._queue: asyncio.Queue[ChannelItem] = asyncio.Queue()
        self._connected = asyncio.Event()
        self._disconnected = asyncio.Event()
        self._failed = asyncio.Event()
        self._failure: Optional[str] = None
        self._worker: Any = None
        self._runner: Any = None
        self._run_task: Optional[asyncio.Task[Any]] = None
        self._sink: Any = None
        self._output: Any = None
        self._frames: Any = None
        self._closing = False
        self._closed = False
        self._audio_waiter: Optional[tuple[int, int, asyncio.Future[float]]] = None
        self._speech_sent_at: Optional[float] = None

    # -- Channel protocol --------------------------------------------------------------------------

    async def connect(self, url: str, headers: dict, query: dict) -> None:
        quiet_pipecat_logging()
        classes = _eval_classes()
        frames = classes.frames
        self._frames = frames
        await self._probe(url)
        voice, rate = self._voice, self._sample_rate
        params = classes.WebsocketClientParams(
            audio_in_enabled=voice,
            audio_out_enabled=voice,
            audio_in_sample_rate=classes.EVAL_STT_SAMPLE_RATE if voice else 0,
            audio_out_sample_rate=rate if voice else 0,
            serializer=classes.EvalClientSerializer(),
            additional_headers=dict(headers) or None,
        )
        recorder = classes.Recorder(rate or classes.EVAL_STT_SAMPLE_RATE, self._on_bot_audio)
        transport = classes.Transport(_with_query(url, query), params, recorder=recorder, channel=self)
        self._sink = classes.Sink()
        self._output = transport.output()
        pipeline = classes.Pipeline([transport.input(), self._sink, self._output])
        self._worker = classes.PipelineWorker(
            pipeline,
            params=classes.PipelineParams(
                audio_in_sample_rate=classes.EVAL_STT_SAMPLE_RATE,
                audio_out_sample_rate=rate or classes.EVAL_STT_SAMPLE_RATE,
            ),
            enable_rtvi=False,
            cancel_on_idle_timeout=False,
        )

        @self._worker.event_handler("on_pipeline_error")
        async def _on_pipeline_error(_worker: Any, frame: Any) -> None:
            self.on_pipeline_error(frame)

        self._runner = classes.WorkerRunner(handle_sigint=False)
        await self._runner.add_workers(self._worker)
        self._run_task = asyncio.ensure_future(self._runner.run())
        # Also wake on a fatal harness error: a rejected upgrade (HTTP 401 / 403 / 404) never connects,
        # and Pipecat reports it through on_pipeline_error (the worker keeps running).
        waiter = asyncio.ensure_future(self._connected.wait())
        failed = asyncio.ensure_future(self._failed.wait())
        try:
            await asyncio.wait(
                {waiter, failed, self._run_task}, timeout=self.CONNECT_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            waiter.cancel()
            failed.cancel()
        if not self._connected.is_set():
            failure = self._failure
            await self.close()
            if failure is not None:
                raise PipecatError(HARNESS_FAILED_ERROR.format(error=failure))
            raise PipecatError(
                f"could not open the WebSocket connection to the Pipecat bot at {strip_url_credentials(url)} within "
                f"{self.CONNECT_TIMEOUT_S:g} s (check pipecat.url, that the bot runs with -t eval, and pipecat.headers)"
            )

    async def send_message(self, data: dict) -> None:
        if self._worker is None or self._closed:
            return
        await self._worker.queue_frame(self._frames.OutputTransportMessageUrgentFrame(message=dict(data)))

    async def send_audio(self, pcm: bytes, sample_rate: int) -> float:
        """Queue the user audio on Pipecat's paced client transport and return when all of it was sent:
        the clock time the frame holding its last speech sample left (audio without speech: the time its
        last frame left), as reported by the send path. Raises ``PipecatError`` when the send path never
        reports it (or the harness pipeline failed); a closed connection returns at once (the queued
        ``ChannelClose`` ends the turn)."""
        if self._failure is not None:
            raise PipecatError(HARNESS_FAILED_ERROR.format(error=self._failure))
        if self._sink is None or self._output is None or self._closed:
            return self._clock.monotonic()   # nothing can be sent; the queued ChannelClose ends the turn
        frames = self._frames
        pcm = bytes(pcm[: len(pcm) - len(pcm) % 2])
        if not pcm:
            return self._clock.monotonic()   # nothing to send
        last = _last_speech_index(pcm)
        base = int(getattr(self._output, "impulse_queued", 0))
        speech_target = base + (last + 1) * 2 if last >= 0 else -1
        done: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        self._speech_sent_at = None
        self._audio_waiter = (speech_target, base + len(pcm), done)
        for frame in (
            frames.TTSStartedFrame(),
            frames.TTSAudioRawFrame(audio=pcm, sample_rate=int(sample_rate), num_channels=1),
            frames.TTSStoppedFrame(),
        ):
            await self._sink.inject(frame)
        closed = asyncio.ensure_future(self._disconnected.wait())
        failed = asyncio.ensure_future(self._failed.wait())
        waiting: set[asyncio.Future[Any]] = {done, closed, failed}
        try:
            await asyncio.wait(
                waiting,
                timeout=len(pcm) / (2.0 * max(1, int(sample_rate))) + self.SEND_REPORT_MARGIN_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            closed.cancel()
            failed.cancel()
            self._audio_waiter = None
        if done.done():
            return done.result()
        if self._failure is not None:
            raise PipecatError(HARNESS_FAILED_ERROR.format(error=self._failure))
        if self._disconnected.is_set() or self._closed:
            # The bot closed while the user was speaking: the queued ChannelClose ends the turn.
            return self._speech_sent_at if self._speech_sent_at is not None else self._clock.monotonic()
        raise PipecatError(SEND_NOT_REPORTED_ERROR)

    async def recv(self) -> ChannelItem:
        return await self._queue.get()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        task = self._run_task
        if task is not None and not task.done():
            with contextlib.suppress(Exception):
                await self._worker.queue_frame(self._frames.EndFrame())
            with contextlib.suppress(Exception):
                await asyncio.wait({task}, timeout=self.END_TIMEOUT_S)
            if not task.done():
                with contextlib.suppress(Exception):
                    await self._runner.cancel(reason="closed by Impulse")
                with contextlib.suppress(Exception):
                    await asyncio.wait({task}, timeout=self.END_TIMEOUT_S)
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._put_close(1000)

    # -- callbacks from the Pipecat classes ----------------------------------------------------------

    def on_connected(self) -> None:
        self._connected.set()

    def on_disconnected(self, websocket: Any) -> None:
        self._disconnected.set()
        code = getattr(websocket, "close_code", None)
        self._put_close(int(code) if code is not None else (1000 if self._closing else 1006))

    def on_raw_message(self, message: Any) -> None:
        """Every raw RTVI message from the bot, before ``EvalClientSerializer`` (``eval-bot-audio`` comes
        through ``on_bot_audio`` instead, after the serializer's resample)."""
        try:
            data = json.loads(message)
        except (TypeError, ValueError):
            return
        if isinstance(data, dict) and data.get("label") == RTVI_LABEL and data.get("type") != EVAL_BOT_AUDIO:
            self._queue.put_nowait(ChannelMessage(data=data))

    def _on_bot_audio(self, audio: bytes, in_rate: int) -> None:
        if audio:
            self._queue.put_nowait(ChannelAudio(pcm=bytes(audio), sample_rate=int(in_rate), arrival_s=self._clock.monotonic()))

    def on_frame_sent(self, sent_total: int) -> None:
        """Called by the output transport after each paced frame left; ``sent_total`` counts the queued
        user-audio bytes sent so far, this frame included."""
        waiter = self._audio_waiter
        if waiter is None:
            return
        speech_target, done_target, done = waiter
        now = self._clock.monotonic()
        if speech_target >= 0 and self._speech_sent_at is None and sent_total >= speech_target:
            self._speech_sent_at = now
        if sent_total >= done_target and not done.done():
            done.set_result(self._speech_sent_at if self._speech_sent_at is not None else now)

    def on_pipeline_error(self, frame: Any) -> None:
        """The harness worker's ``on_pipeline_error`` (an ``ErrorFrame`` from Impulse's own pipeline):
        queued for the adapter; fatal when Pipecat marks it fatal or the processor is no longer usable.
        The text is redacted before it is shortened (the queued text whole, ``_failure`` cut to 200)."""
        if self._closing:
            return
        error = getattr(frame, "error", None)
        text = str(error if error is not None else frame)
        if self._redact is not None:
            text = self._redact(text)
        text = " ".join(text.split()) or type(frame).__name__
        processor = getattr(frame, "processor", None)
        fatal = getattr(frame, "fatal", False) is True or (
            processor is not None and getattr(processor, "is_usable", True) is False
        )
        self._queue.put_nowait(ChannelMessage(data={"type": HARNESS_ERROR, "data": {"error": text, "fatal": fatal}}))
        if fatal and self._failure is None:
            self._failure = text[:_ERROR_TEXT_CHARS]
            self._failed.set()

    # -- internals ---------------------------------------------------------------------------------

    def _put_close(self, code: int) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(ChannelClose(code=code))

    async def _probe(self, url: str) -> None:
        """A plain TCP connect first, as Pipecat's ``EvalClient.wait_for_bot`` does: a bot that is not
        listening fails fast (Pipecat's session logs connection errors instead of raising them)."""
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        host = parts.hostname or "localhost"
        try:
            port = parts.port or (443 if parts.scheme == "wss" else 80)
            _reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=self.PROBE_TIMEOUT_S)
        except (OSError, TimeoutError, ValueError) as exc:
            raise PipecatError(
                f"could not connect to the Pipecat bot at {strip_url_credentials(url)}: {type(exc).__name__}: {exc}. "
                "Check that the bot is running (eval mode: started with -t eval)."
            ) from None
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


@dataclass(frozen=True)
class _EvalClasses:
    frames: Any
    EVAL_STT_SAMPLE_RATE: int
    EvalClientSerializer: Any
    WebsocketClientParams: Any
    Pipeline: Any
    PipelineParams: Any
    PipelineWorker: Any
    WorkerRunner: Any
    Recorder: Any
    Transport: Any
    Sink: Any


_EVAL_CLASSES: Optional[_EvalClasses] = None


def _check_method(cls: Any, owner: str, name: str, positional: int, missing: list[str]) -> None:
    """``cls.name`` must be a coroutine function taking ``positional`` positional parameters (self included)."""
    function = getattr(cls, name, None)
    if function is None or not inspect.iscoroutinefunction(function):
        missing.append(f"{owner}.{name}")
        return
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return   # no signature to check; the name exists
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if sum(1 for parameter in parameters if parameter.kind in kinds) != positional:
        missing.append(f"{owner}.{name} (parameters changed)")


def _source_lacks(function: Any, pattern: str) -> bool:
    """True when ``function``'s source is available and does not match ``pattern`` (unavailable source
    is not held against it)."""
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    return re.search(pattern, source) is None


def _missing_eval_internals(modules: Mapping[str, Any]) -> list[str]:
    """The private pipecat-ai 1.12 names ``EvalChannel`` relies on that the installed release lacks."""
    missing: list[str] = []
    client_transport = modules["pipecat.evals.client_transport"]
    names: dict[str, Any] = {
        "EvalClientTransport": getattr(client_transport, "EvalClientTransport", None),
        "EvalClientOutputTransport": getattr(client_transport, "EvalClientOutputTransport", None),
        "EvalClientInputTransport": getattr(client_transport, "EvalClientInputTransport", None),
        "EvalClientRecorder": getattr(client_transport, "EvalClientRecorder", None),
        "EvalClientSerializer": getattr(modules["pipecat.evals.serializer"], "EvalClientSerializer", None),
        "EVAL_STT_SAMPLE_RATE": getattr(modules["pipecat.evals.serializer"], "EVAL_STT_SAMPLE_RATE", None),
        "WebsocketClientParams": getattr(modules["pipecat.transports.websocket.client"], "WebsocketClientParams", None),
    }
    missing.extend(name for name, value in names.items() if value is None)
    transport_cls = names["EvalClientTransport"]
    output_cls = names["EvalClientOutputTransport"]
    input_cls = names["EvalClientInputTransport"]
    recorder_cls = names["EvalClientRecorder"]
    if transport_cls is not None:
        _check_method(transport_cls, "RTVIClientTransport", "_on_message", 3, missing)
        if not callable(getattr(transport_cls, "output", None)):
            missing.append("EvalClientTransport.output")
    if output_cls is not None:
        _check_method(output_cls, "EvalClientOutputTransport", "_send_frame", 2, missing)
        _check_method(output_cls, "EvalClientOutputTransport", "write_audio_frame", 2, missing)
        _check_method(output_cls, "EvalClientOutputTransport", "_send_task_handler", 1, missing)
        handler = getattr(output_cls, "_send_task_handler", None)
        if handler is not None and _source_lacks(handler, r"self\._send_frame\("):
            missing.append("EvalClientOutputTransport._send_task_handler -> self._send_frame()")
        if handler is not None and _source_lacks(handler, r"self\._pending\b"):
            missing.append("EvalClientOutputTransport._send_task_handler -> self._pending")
    if recorder_cls is not None:
        for name in ("add_user", "add_bot"):
            if not callable(getattr(recorder_cls, name, None)):
                missing.append(f"EvalClientRecorder.{name}")
    if input_cls is not None:
        push = getattr(input_cls, "push_audio_frame", None)
        if push is None or _source_lacks(push, r"\.add_bot\("):
            missing.append("EvalClientInputTransport.push_audio_frame -> recorder.add_bot()")
    if missing:
        return missing
    # Instance attributes (set in __init__): build a throwaway transport and its output; no connection.
    try:
        params = names["WebsocketClientParams"](serializer=names["EvalClientSerializer"]())
        transport = transport_cls("ws://127.0.0.1:9", params, recorder=None)
    except Exception as exc:  # noqa: BLE001 - a changed constructor
        return [f"EvalClientTransport(uri, params, recorder=...) ({type(exc).__name__}: {exc})"]
    for name in ("_session", "_params", "_recorder", "_output"):
        if not hasattr(transport, name):
            missing.append(f"EvalClientTransport.{name}")
    if missing:
        return missing
    try:
        output = output_cls(transport, transport._session, transport._params, recorder=None)
    except Exception as exc:  # noqa: BLE001
        return [f"EvalClientOutputTransport(transport, session, params, recorder=...) ({type(exc).__name__}: {exc})"]
    if not isinstance(getattr(output, "_pending", None), bytearray):
        missing.append("EvalClientOutputTransport._pending")
    return missing


def _eval_classes() -> _EvalClasses:
    """Impulse's subclasses of Pipecat's eval client classes (built once, on first use). They override
    private pipecat-ai 1.12 APIs, so every private name they rely on is checked first: a missing one is a
    ``ConfigError`` at ``agent.pipecat.mode`` (the tested version, and the websocket fallback)."""
    global _EVAL_CLASSES
    if _EVAL_CLASSES is not None:
        return _EVAL_CLASSES
    quiet_pipecat_logging()
    modules = {
        name: importlib.import_module(name)
        for name in (
            "pipecat.evals.client_transport",
            "pipecat.evals.serializer",
            "pipecat.transports.websocket.client",
        )
    }
    missing = _missing_eval_internals(modules)
    if missing:
        raise ConfigError(
            "agent.pipecat.mode",
            EVAL_INTERNALS_ERROR.format(version=_pipecat_version(), names=", ".join(missing)),
        )
    from pipecat.evals.client_transport import (
        EvalClientOutputTransport,
        EvalClientRecorder,
        EvalClientTransport,
    )
    from pipecat.evals.serializer import EVAL_STT_SAMPLE_RATE, EvalClientSerializer
    from pipecat.frames import frames
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.transports.websocket.client import WebsocketClientParams
    from pipecat.workers.runner import WorkerRunner

    class Recorder(EvalClientRecorder):
        """Delivers the bot's raw audio (after the 16 kHz resample, before gap-fill) to the channel; keeps
        no recording."""

        def __init__(self, sample_rate: int, on_bot: Callable[[bytes, int], None]) -> None:
            super().__init__(sample_rate)
            self._impulse_on_bot = on_bot

        def add_user(self, audio: bytes, in_rate: int) -> None:
            return None

        def add_bot(self, audio: bytes, in_rate: int) -> None:
            self._impulse_on_bot(audio, in_rate)

    class Output(EvalClientOutputTransport):
        """Counts the user audio queued for pacing and reports each paced frame as it is sent."""

        def __init__(self, *args: Any, channel: EvalChannel, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.impulse_queued = 0
            self._impulse_channel = channel

        async def write_audio_frame(self, frame: Any) -> bool:
            before = len(self._pending)
            result = await super().write_audio_frame(frame)
            self.impulse_queued += max(0, len(self._pending) - before)
            return result

        async def _send_frame(self, frame: Any) -> None:
            sent_total = self.impulse_queued - len(self._pending)
            await super()._send_frame(frame)
            self._impulse_channel.on_frame_sent(sent_total)

    class Transport(EvalClientTransport):
        """Reads every raw bot message before the client serializer, and reports connect / disconnect."""

        def __init__(self, uri: str, params: Any, *, recorder: Any, channel: EvalChannel) -> None:
            super().__init__(uri, params, recorder=recorder)
            self._impulse_channel = channel

            @self.event_handler("on_connected")
            async def _on_connected(_transport: Any, _websocket: Any) -> None:
                channel.on_connected()

            @self.event_handler("on_disconnected")
            async def _on_disconnected(_transport: Any, websocket: Any) -> None:
                channel.on_disconnected(websocket)

        def output(self) -> Any:
            if not self._output:
                self._output = Output(self, self._session, self._params, recorder=self._recorder, channel=self._impulse_channel)
            return self._output

        async def _on_message(self, websocket: Any, message: Any) -> None:
            self._impulse_channel.on_raw_message(message)
            await super()._on_message(websocket, message)

    bot_frames = (
        frames.InputAudioRawFrame,
        frames.InputTransportMessageFrame,
        frames.InterruptionFrame,
        frames.LLMFullResponseStartFrame,
        frames.LLMTextFrame,
        frames.LLMMarkerResponseFrame,
        frames.LLMFullResponseEndFrame,
        frames.TTSTextFrame,
        frames.BotStartedSpeakingFrame,
        frames.BotStoppedSpeakingFrame,
        frames.FunctionCallInProgressFrame,
        frames.FunctionCallResultFrame,
        frames.FunctionCallCancelFrame,
        frames.TranscriptionFrame,
        frames.InterimTranscriptionFrame,
    )

    class Sink(FrameProcessor):
        """Where the bot's frames stop (Impulse reads the raw messages instead) and the user's audio is
        injected towards the output, as in Pipecat's ``EvalClient``."""

        async def inject(self, frame: Any) -> None:
            await self.push_frame(frame)

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, bot_frames):
                return
            await self.push_frame(frame, direction)

    _EVAL_CLASSES = _EvalClasses(
        frames=frames,
        EVAL_STT_SAMPLE_RATE=int(EVAL_STT_SAMPLE_RATE),
        EvalClientSerializer=EvalClientSerializer,
        WebsocketClientParams=WebsocketClientParams,
        Pipeline=Pipeline,
        PipelineParams=PipelineParams,
        PipelineWorker=PipelineWorker,
        WorkerRunner=WorkerRunner,
        Recorder=Recorder,
        Transport=Transport,
        Sink=Sink,
    )
    return _EVAL_CLASSES
