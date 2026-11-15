# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Impulse Evals test suite (SPEC.md section 18).

Stage B owns this file; builders never edit it. Every test runs offline:

* ``_block_network`` (autouse) refuses every non-loopback connection or DNS lookup and fails the
  test at teardown if one was attempted, even when the code under test swallowed the error.
* ``mock_llm`` replaces ``litellm.completion`` / ``litellm.acompletion`` (module attributes).
* ``mock_voice_server`` serves deterministic TTS / STT through ``httpx.MockTransport``.
* ``sample_agent`` loads ``examples/sample_agent.py``.
* ``mock_pipecat_channel`` is a scripted fake Pipecat ``Channel`` plus a virtual ``Clock``.
* ``fake_judge`` builds a fake ``impulse_evals.judge.base.Judge`` subclass ("mocked judge").

Impulse types are imported inside fixtures or helper methods, never at module level, so test
collection works before the package is built.
"""

import os

# litellm downloads its model cost map from the network at import time unless told to use the
# copy bundled with the package. This must run before anything imports litellm.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import array  # noqa: E402
import asyncio  # noqa: E402
import copy  # noqa: E402
import heapq  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import ipaddress  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import wave  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # `import impulse_evals` must work before the package is pip-installed.
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLES_DIR = REPO_ROOT / "examples"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "phase11: assertion that belongs to build phase 11 (SPEC.md sections 18.2 and 19)"
    )
    config.addinivalue_line(
        "markers", "phase12: assertion that belongs to build phase 12 (SPEC.md sections 18.2 and 19)"
    )


# ---------------------------------------------------------------------------------------------
# Network guard (autouse)
# ---------------------------------------------------------------------------------------------


class NetworkBlockedError(RuntimeError):
    """Raised when code under test tries to reach a non-loopback address."""


_LOCAL_HOST_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


def _is_local_host(host) -> bool:
    """True for loopback hosts, unspecified bind addresses, and the "no host" forms."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    host = str(host).strip().strip("[]").lower()
    if host == "":
        return True
    if host in _LOCAL_HOST_NAMES or host.endswith(".localhost"):
        return True
    host = host.split("%", 1)[0]  # IPv6 zone id
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return ip.is_loopback or ip.is_unspecified


def _address_is_local(address) -> bool:
    if isinstance(address, (str, bytes, bytearray)):
        return True  # AF_UNIX path: never leaves the machine
    if isinstance(address, tuple) and address:
        return _is_local_host(address[0])
    return False


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Block real network access. Loopback stays allowed (asyncio on Windows needs socketpairs)."""
    attempts = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def refuse(what, target):
        attempts.append(f"{what}({target!r})")
        raise NetworkBlockedError(
            f"Real network access is blocked in tests: {what}({target!r}). "
            "Use the offline fixtures in tests/conftest.py."
        )

    def guarded_connect(self, address):
        if not _address_is_local(address):
            refuse("socket.connect", address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not _address_is_local(address):
            refuse("socket.connect_ex", address)
        return real_connect_ex(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        if not _address_is_local(address):
            refuse("socket.create_connection", address)
        return real_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_local_host(host):
            refuse("socket.getaddrinfo", host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield attempts
    if attempts:
        pytest.fail(
            "Test attempted real network access (blocked): " + "; ".join(attempts), pytrace=False
        )


# ---------------------------------------------------------------------------------------------
# mock_llm
# ---------------------------------------------------------------------------------------------


class _AttrDict(dict):
    """dict with attribute access; fallback response shape if litellm's types cannot be built."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _to_attr(value):
    if isinstance(value, dict):
        return _AttrDict({k: _to_attr(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr(v) for v in value]
    return value


def make_llm_response(content, tool_calls=None, model="mock-model"):
    """Build a litellm-shaped response: ``resp.choices[0].message.content`` / ``.tool_calls``.

    ``tool_calls`` is a list of ``{"name": str, "arguments": dict}``.
    """
    tool_calls = tool_calls or None
    finish_reason = "tool_calls" if tool_calls else "stop"
    try:
        from litellm import ModelResponse
        from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message

        lite_calls = None
        if tool_calls:
            lite_calls = [
                ChatCompletionMessageToolCall(
                    id=f"call_{i}",
                    type="function",
                    function=Function(name=tc["name"], arguments=json.dumps(tc.get("arguments", {}))),
                )
                for i, tc in enumerate(tool_calls, start=1)
            ]
        message = Message(content=content, role="assistant", tool_calls=lite_calls)
        response = ModelResponse(
            choices=[Choices(message=message, finish_reason=finish_reason, index=0)], model=model
        )
        if response.choices[0].message.content == content:
            return response
    except Exception:  # pragma: no cover - only if litellm internals change
        pass
    raw_calls = None
    if tool_calls:
        raw_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments", {}))},
            }
            for i, tc in enumerate(tool_calls, start=1)
        ]
    return _to_attr(
        {
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content, "tool_calls": raw_calls},
                }
            ],
        }
    )


_NO_DEFAULT = object()


class MockLLM:
    """Scripted stand-in for ``litellm.completion`` and ``litellm.acompletion``.

    Script responses in order with the ``queue_*`` helpers; each LLM call consumes one. Every call
    is recorded in ``calls`` as ``{"model", "messages", "kwargs", "async", "response"}``. A call
    with nothing left in the script uses ``responder(call)`` or ``set_default(...)`` if given,
    otherwise it raises ``AssertionError`` (an unexpected LLM call).
    """

    WHITESPACE = "   \n\t  \n"
    INVALID_JSON = 'I think the agent did fine. {"result": pass, reason: unquoted'

    def __init__(self):
        self.calls = []
        self._script = deque()
        self._default = _NO_DEFAULT
        self.responder = None

    # -- scripting ------------------------------------------------------------------------
    def queue(self, *contents):
        """Queue raw message contents: ``str`` or ``None``; a ``dict`` is sent as JSON text."""
        for content in contents:
            if isinstance(content, dict):
                content = json.dumps(content)
            self._script.append(("content", content))
        return self

    def queue_none(self):
        return self.queue(None)

    def queue_empty(self):
        return self.queue("")

    def queue_whitespace(self):
        return self.queue(self.WHITESPACE)

    def queue_invalid_json(self, text=None):
        return self.queue(self.INVALID_JSON if text is None else text)

    def queue_json(self, obj):
        self._script.append(("content", json.dumps(obj)))
        return self

    def queue_fenced_json(self, obj, lang="json"):
        self._script.append(("content", f"```{lang}\n{json.dumps(obj, indent=2)}\n```"))
        return self

    def queue_tool_call(self, name="lookup_order", arguments=None, content=None):
        """A response whose ``message.tool_calls`` is set and whose ``content`` is None."""
        call = {"name": name, "arguments": arguments or {}}
        self._script.append(("tool_call", (content, [call])))
        return self

    def queue_error(self, exc):
        """The next call raises ``exc``."""
        self._script.append(("error", exc))
        return self

    @staticmethod
    def verdict(result, reason="mock verdict", turn_index=None, confidence=0.9):
        return {"result": result, "reason": reason, "turn_index": turn_index, "confidence": confidence}

    def queue_verdict(self, result, reason="mock verdict", turn_index=None, confidence=0.9):
        return self.queue_json(self.verdict(result, reason, turn_index, confidence))

    def set_default(self, content):
        """Content (str, None, or dict as JSON) returned once the script is exhausted."""
        self._default = json.dumps(content) if isinstance(content, dict) else content
        return self

    # -- inspection -----------------------------------------------------------------------
    @property
    def call_count(self):
        return len(self.calls)

    @property
    def remaining(self):
        return len(self._script)

    def messages(self, index=-1):
        return self.calls[index]["messages"]

    def text(self, index=-1):
        """All message contents of one call, joined with newlines."""
        parts = []
        for message in self.calls[index]["messages"] or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if content:
                parts.append(str(content))
        return "\n".join(parts)

    # -- patched entry points -------------------------------------------------------------
    def _record(self, args, kwargs, is_async):
        kwargs = dict(kwargs)
        model = kwargs.pop("model", args[0] if len(args) > 0 else None)
        messages = kwargs.pop("messages", args[1] if len(args) > 1 else None)
        call = {
            "model": model,
            "messages": copy.deepcopy(messages),
            "kwargs": kwargs,
            "async": is_async,
            "response": None,
        }
        self.calls.append(call)
        return call

    def _respond(self, call):
        if call["kwargs"].get("stream"):
            raise NotImplementedError("mock_llm does not support stream=True")
        model = call["model"] or "mock-model"
        if self._script:
            kind, payload = self._script.popleft()
        elif self.responder is not None:
            produced = self.responder(call)
            if hasattr(produced, "choices"):
                call["response"] = produced
                return produced
            kind, payload = "content", produced
        elif self._default is not _NO_DEFAULT:
            kind, payload = "content", self._default
        else:
            raise AssertionError(
                f"mock_llm: unexpected LLM call #{len(self.calls)} to {model!r} "
                "(no scripted response left)"
            )
        if kind == "error":
            raise payload
        if kind == "tool_call":
            content, tool_calls = payload
            response = make_llm_response(content, tool_calls=tool_calls, model=model)
        else:
            response = make_llm_response(payload, model=model)
        call["response"] = response
        return response

    def completion(self, *args, **kwargs):
        call = self._record(args, kwargs, is_async=False)
        return self._respond(call)

    async def acompletion(self, *args, **kwargs):
        call = self._record(args, kwargs, is_async=True)
        await asyncio.sleep(0)
        return self._respond(call)


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch ``litellm.completion`` and ``litellm.acompletion`` with a scripted ``MockLLM``.

    Code under test must call them through the module (``litellm.acompletion(...)``), not through
    names bound at import time (``from litellm import acompletion``).
    """
    import litellm

    mock = MockLLM()
    monkeypatch.setattr(litellm, "completion", mock.completion)
    monkeypatch.setattr(litellm, "acompletion", mock.acompletion)
    return mock


# ---------------------------------------------------------------------------------------------
# mock_voice_server
# ---------------------------------------------------------------------------------------------


def pcm_tone(seconds, sample_rate=16000, frequency_hz=440.0, amplitude=0.5) -> bytes:
    """Mono 16-bit little-endian PCM sine tone (deterministic)."""
    count = int(round(seconds * sample_rate))
    peak = int(32767 * amplitude)
    samples = array.array(
        "h",
        (int(round(peak * math.sin(2 * math.pi * frequency_hz * i / sample_rate))) for i in range(count)),
    )
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def pcm_silence(seconds, sample_rate=16000) -> bytes:
    """Mono 16-bit PCM digital silence."""
    return b"\x00\x00" * int(round(seconds * sample_rate))


def wav_bytes(pcm: bytes, sample_rate=16000, channels=1) -> bytes:
    """Wrap 16-bit PCM in a WAV container with the standard library ``wave`` module."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def sine_wav(seconds=0.5, sample_rate=16000, frequency_hz=440.0, amplitude=0.5) -> bytes:
    """Mono 16-bit PCM WAV containing a sine tone."""
    return wav_bytes(pcm_tone(seconds, sample_rate, frequency_hz, amplitude), sample_rate)


def word_timings(text, ms_per_word=300, gap_ms=50, confidence=0.95):
    """Deterministic word timings (``WordTiming`` shape) for ``text``."""
    words, start = [], 0
    for word in text.split():
        words.append(
            {"word": word, "start_ms": start, "end_ms": start + ms_per_word, "confidence": confidence}
        )
        start += ms_per_word + gap_ms
    return words


class MockVoiceServer:
    """Deterministic voice backend served through ``httpx.MockTransport`` (no socket).

    Routes (POST, matched on the end of the path):
      * ``.../audio/speech``         OpenAI-style TTS -> ``audio/wav`` sine tone
      * ``.../audio/transcriptions`` OpenAI-style STT -> ``verbose_json`` with word timings (seconds)
      * ``.../v1/tts``               Impulse-hosted TTS -> ``audio/wav`` sine tone
      * ``.../v1/stt``               Impulse-hosted STT -> ``{text, confidence, words, filler_words}``
    Anything else returns 404. ``queue_status`` makes the next requests fail (retry tests);
    ``queue_stt`` scripts transcriptions in order, otherwise ``stt_default_text`` is used.
    While the fixture is active, every ``httpx.Client`` / ``httpx.AsyncClient`` created without an
    explicit ``transport`` uses this server.
    """

    def __init__(self):
        import httpx

        self.requests = []
        self.tts_sample_rate = 16000
        self.tts_seconds = 0.5
        self.tts_frequency_hz = 440.0
        self.stt_default_text = "hello this is a scripted transcription"
        self.stt_default_confidence = 0.9
        self._stt_script = deque()
        self._status_script = deque()
        self.transport = httpx.MockTransport(self._handle)

    # -- scripting ------------------------------------------------------------------------
    def queue_stt(self, text, words=None, confidence=0.9, filler_words=None):
        """Script the next transcription. ``words`` uses the ``WordTiming`` shape (ms)."""
        self._stt_script.append(
            {
                "text": text,
                "words": word_timings(text) if words is None else words,
                "confidence": confidence,
                "filler_words": filler_words or [],
            }
        )
        return self

    def queue_status(self, status_code, count=1, body=None):
        """The next ``count`` requests return ``status_code`` (e.g. 429 or 503)."""
        for _ in range(count):
            self._status_script.append((status_code, body or {"error": f"scripted {status_code}"}))
        return self

    # -- helpers --------------------------------------------------------------------------
    def wav(self) -> bytes:
        """The exact WAV bytes the TTS routes return."""
        return sine_wav(self.tts_seconds, self.tts_sample_rate, self.tts_frequency_hz)

    def client(self, **kwargs):
        import httpx

        return httpx.Client(transport=self.transport, **kwargs)

    def async_client(self, **kwargs):
        import httpx

        return httpx.AsyncClient(transport=self.transport, **kwargs)

    def requests_to(self, path_suffix):
        return [r for r in self.requests if r["path"].endswith(path_suffix)]

    def install(self, monkeypatch):
        """Make new httpx clients use this server unless a transport is passed explicitly."""
        import httpx

        transport = self.transport
        real_client_init = httpx.Client.__init__
        real_async_init = httpx.AsyncClient.__init__

        def client_init(client, *args, **kwargs):
            if kwargs.get("transport") is None:
                kwargs["transport"] = transport
            real_client_init(client, *args, **kwargs)

        def async_init(client, *args, **kwargs):
            if kwargs.get("transport") is None:
                kwargs["transport"] = transport
            real_async_init(client, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", client_init)
        monkeypatch.setattr(httpx.AsyncClient, "__init__", async_init)

    # -- handler --------------------------------------------------------------------------
    def _next_stt(self):
        if self._stt_script:
            return self._stt_script.popleft()
        text = self.stt_default_text
        return {
            "text": text,
            "words": word_timings(text),
            "confidence": self.stt_default_confidence,
            "filler_words": [],
        }

    def _handle(self, request):
        import httpx

        body_json = None
        content = request.content
        if "json" in request.headers.get("content-type", ""):
            try:
                body_json = json.loads(content or b"null")
            except ValueError:
                body_json = None
        self.requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "headers": dict(request.headers),
                "json": body_json,
                "content": content,
            }
        )
        if self._status_script:
            status, body = self._status_script.popleft()
            return httpx.Response(status, json=body)
        path = request.url.path.rstrip("/")
        if request.method == "POST" and (path.endswith("/audio/speech") or path.endswith("/v1/tts")):
            return httpx.Response(200, content=self.wav(), headers={"content-type": "audio/wav"})
        if request.method == "POST" and path.endswith("/audio/transcriptions"):
            stt = self._next_stt()
            duration = (stt["words"][-1]["end_ms"] / 1000.0) if stt["words"] else 0.0
            return httpx.Response(
                200,
                json={
                    "task": "transcribe",
                    "text": stt["text"],
                    "duration": duration,
                    "words": [
                        {"word": w["word"], "start": w["start_ms"] / 1000.0, "end": w["end_ms"] / 1000.0}
                        for w in stt["words"]
                    ],
                    "segments": [{"id": 0, "start": 0.0, "end": duration, "text": stt["text"]}],
                },
            )
        if request.method == "POST" and path.endswith("/v1/stt"):
            stt = self._next_stt()
            return httpx.Response(200, json=stt)
        return httpx.Response(404, json={"error": f"mock voice server: no route for {request.method} {path}"})


@pytest.fixture
def mock_voice_server(monkeypatch):
    """Deterministic TTS (sine WAV) and STT (scripted text + word timings) via httpx.MockTransport."""
    server = MockVoiceServer()
    server.install(monkeypatch)
    return server


# ---------------------------------------------------------------------------------------------
# sample_agent
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def sample_agent():
    """The ``examples/sample_agent.py`` module (SPEC.md section 8.2), freshly loaded."""
    path = EXAMPLES_DIR / "sample_agent.py"
    spec = importlib.util.spec_from_file_location("impulse_tests_sample_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------
# fake_judge
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def fake_judge():
    """Factory for a fake ``Judge`` (SPEC.md section 18.1, judge.base): the "mocked judge".

    ``fake_judge(*verdicts, default=None)`` returns a judge whose ``evaluate`` pops the queued
    verdicts in order (``JudgeVerdict`` or dict), then returns ``default`` (or a "pass" verdict).
    Calls are recorded in ``judge.calls`` as ``{"transcript", "criterion", "tier"}``.
    """

    def make(*verdicts, default=None):
        from impulse_evals.judge.base import Judge, JudgeVerdict

        class FakeJudge(Judge):
            def __init__(self):
                self.calls = []
                self._queue = deque(verdicts)
                self._default = default

            async def evaluate(self, transcript, criterion, tier):
                self.calls.append({"transcript": transcript, "criterion": criterion, "tier": tier})
                if self._queue:
                    verdict = self._queue.popleft()
                elif self._default is not None:
                    verdict = self._default
                else:
                    verdict = {"result": "pass", "reason": "fake judge: pass", "confidence": 1.0}
                if isinstance(verdict, dict):
                    verdict = JudgeVerdict(**verdict)
                return verdict

        return FakeJudge()

    return make


# ---------------------------------------------------------------------------------------------
# mock_pipecat_channel (fake Channel + fake Clock, SPEC.md section 9)
# ---------------------------------------------------------------------------------------------


def _pipecat_types():
    """The Channel item types from the adapter module (imported lazily: built in phase 11)."""
    from impulse_evals.adapters import pipecat_adapter

    return pipecat_adapter


class FakeClock:
    """Virtual clock implementing the SPEC.md section 9 ``Clock`` protocol.

    ``monotonic()`` returns virtual seconds. ``await sleep(s)`` never waits in real time: a driver
    task moves virtual time to the earliest pending wake-up once the other tasks have settled
    (``settle_rounds`` event-loop iterations with nothing due on a channel). ``advance(s)`` moves
    time by hand. ``sleep_calls`` records every requested sleep.
    """

    settle_rounds = 50
    busy_round_limit = 2000

    def __init__(self, start=1000.0):
        self._now = float(start)
        self._waiters = []
        self._seq = itertools.count()
        self._driver = None
        self._busy_checks = []
        self.sleep_calls = []

    def monotonic(self) -> float:
        return self._now

    @property
    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        seconds = float(seconds)
        self.sleep_calls.append(seconds)
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        await self.wait_until(self._now + seconds)

    async def wait_until(self, when: float) -> None:
        if when <= self._now:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self._add_waiter(when, future)
        await future

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)
        self._wake_due()

    # -- internals ------------------------------------------------------------------------
    def _add_waiter(self, when, future):
        heapq.heappush(self._waiters, (when, next(self._seq), future))
        self._ensure_driver()

    def _wake_due(self):
        while self._waiters and self._waiters[0][0] <= self._now:
            _, _, future = heapq.heappop(self._waiters)
            if not future.done():
                future.set_result(None)

    def _ensure_driver(self):
        loop = asyncio.get_running_loop()
        if self._driver is None or self._driver.done() or self._driver.get_loop() is not loop:
            self._driver = loop.create_task(self._drive())

    async def _drive(self):
        while True:
            for _ in range(self.settle_rounds):
                await asyncio.sleep(0)
            rounds = 0
            while rounds < self.busy_round_limit and any(check() for check in self._busy_checks):
                await asyncio.sleep(0)
                rounds += 1
            while self._waiters and self._waiters[0][2].done():
                heapq.heappop(self._waiters)
            if not self._waiters:
                return
            when = self._waiters[0][0]
            if when > self._now:
                self._now = when
            self._wake_due()


@dataclass
class BotAudio:
    """Bot audio to emit; delivered as ``ChannelAudio`` with ``arrival_s`` = its virtual time."""

    pcm: bytes
    sample_rate: int = 16000


@dataclass
class BotClose:
    """Bot-side close; delivered as ``ChannelClose(code)``."""

    code: int = 1000


class FakePipecatChannel:
    """Scripted fake implementing the SPEC.md section 9 ``Channel`` protocol.

    Script what the fake bot sends:
      * ``emit(*items, at=None, after=0.0, gap=0.0)``: at virtual time ``at`` (absolute) or
        ``now + after``; successive items ``gap`` seconds apart.
      * ``on_send(match, *items, after=0.0, gap=0.0, once=True)``: when the adapter sends a message
        whose RTVI ``type`` equals ``match`` (or ``match(data)`` is true; ``None`` matches all).
      * ``on_connect(*items, ...)`` and ``on_audio(*items, match=None, ...)`` likewise.
    Items: a ``dict`` (sent as ``ChannelMessage(data=...)``), ``BotAudio``, ``BotClose``, an
    existing ``ChannelMessage`` / ``ChannelAudio`` / ``ChannelClose``, or (in ``on_*`` rules) a
    callable receiving the triggering message dict / record and returning item(s) or None.

    Records: ``connect_calls`` (url, headers, query, t), ``sent`` (chronological), ``sent_messages``,
    ``sent_audio`` (pcm, sample_rate, sent_at, returned, has_speech), ``received``, ``close_calls``.

    ``send_audio`` returns the clock time the last non-silent sample left (|sample| > 1% of full
    scale). With ``paced_audio = True`` it streams in virtual real time (sleeps for the audio
    duration), like Pipecat's eval client transport; otherwise it returns immediately.
    ``recv()`` waits in virtual time for the next item; with nothing scheduled it fails after
    ``idle_real_timeout_s`` seconds of real time (instead of hanging the test suite).
    """

    idle_real_timeout_s = 5.0
    silence_threshold = 327

    def __init__(self, clock: FakeClock, name="channel-1"):
        self.clock = clock
        self.name = name
        self.connect_calls = []
        self.sent = []
        self.received = []
        self.close_calls = 0
        self.connected = False
        self.closed = False
        self.connect_error = None
        self.paced_audio = False
        self._queue = []
        self._seq = itertools.count()
        self._rules = []
        self._wake = None
        self._last_speech_s = None
        clock._busy_checks.append(self._has_undelivered_due_items)

    # -- views ----------------------------------------------------------------------------
    @property
    def sent_messages(self):
        return [entry["data"] for entry in self.sent if entry["kind"] == "message"]

    @property
    def sent_audio(self):
        return [entry for entry in self.sent if entry["kind"] == "audio"]

    def sent_types(self):
        return [message.get("type") for message in self.sent_messages]

    @property
    def connect_url(self):
        return self.connect_calls[-1]["url"] if self.connect_calls else None

    @property
    def connect_headers(self):
        return self.connect_calls[-1]["headers"] if self.connect_calls else None

    @property
    def connect_query(self):
        return self.connect_calls[-1]["query"] if self.connect_calls else None

    @property
    def pending(self):
        return len(self._queue)

    # -- scripting ------------------------------------------------------------------------
    def emit(self, *items, at=None, after=0.0, gap=0.0):
        when = (self.clock.monotonic() + after) if at is None else float(at)
        for item in items:
            self._schedule(when, item)
            when += gap
        return self

    def on_send(self, match, *items, after=0.0, gap=0.0, once=True):
        if match is None:
            predicate = lambda data: True  # noqa: E731
        elif isinstance(match, str):
            predicate = lambda data, _t=match: isinstance(data, dict) and data.get("type") == _t  # noqa: E731
        else:
            predicate = match
        self._rules.append(
            {"kind": "message", "match": lambda rec: predicate(rec["data"]), "items": items,
             "after": after, "gap": gap, "once": once}
        )
        return self

    def on_connect(self, *items, after=0.0, gap=0.0):
        self._rules.append(
            {"kind": "connect", "match": lambda rec: True, "items": items,
             "after": after, "gap": gap, "once": True}
        )
        return self

    def on_audio(self, *items, match=None, after=0.0, gap=0.0, once=True):
        predicate = match or (lambda rec: True)
        self._rules.append(
            {"kind": "audio", "match": predicate, "items": items,
             "after": after, "gap": gap, "once": once}
        )
        return self

    # -- Channel protocol -----------------------------------------------------------------
    async def connect(self, url: str, headers: dict, query: dict) -> None:
        record = {
            "url": url,
            "headers": dict(headers or {}),
            "query": dict(query or {}),
            "t": self.clock.monotonic(),
        }
        self.connect_calls.append(record)
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True
        self.closed = False
        self._fire("connect", record)
        await asyncio.sleep(0)

    async def send_message(self, data: dict) -> None:
        if not self.connected:
            raise RuntimeError("FakePipecatChannel: send_message() before connect()")
        if self.closed:
            raise RuntimeError("FakePipecatChannel: send_message() after close()")
        record = {"kind": "message", "data": copy.deepcopy(data), "t": self.clock.monotonic()}
        self.sent.append(record)
        self._fire("message", record)
        await asyncio.sleep(0)

    async def send_audio(self, pcm: bytes, sample_rate: int) -> float:
        if not self.connected:
            raise RuntimeError("FakePipecatChannel: send_audio() before connect()")
        if self.closed:
            raise RuntimeError("FakePipecatChannel: send_audio() after close()")
        start = self.clock.monotonic()
        last_index, count = self._last_speech_index(pcm)
        if self.paced_audio:
            await self.clock.sleep(count / float(sample_rate))
            speech_end = start + (last_index + 1) / float(sample_rate) if last_index >= 0 else None
        else:
            speech_end = start if last_index >= 0 else None
        if speech_end is not None:
            self._last_speech_s = speech_end
        returned = self._last_speech_s if self._last_speech_s is not None else start
        record = {
            "kind": "audio",
            "pcm": bytes(pcm),
            "sample_rate": sample_rate,
            "sent_at": start,
            "returned": returned,
            "has_speech": last_index >= 0,
            "t": self.clock.monotonic(),
        }
        self.sent.append(record)
        self._fire("audio", record)
        await asyncio.sleep(0)
        return returned

    async def recv(self):
        loop = asyncio.get_running_loop()
        while True:
            if self._queue and self._queue[0][0] <= self.clock.monotonic():
                _, _, item = heapq.heappop(self._queue)
                self.received.append(item)
                return item
            if self.closed:
                return _pipecat_types().ChannelClose(code=1000)
            self._wake = loop.create_future()
            try:
                if self._queue:
                    self.clock._add_waiter(self._queue[0][0], self._wake)
                    await self._wake
                else:
                    try:
                        await asyncio.wait_for(self._wake, self.idle_real_timeout_s)
                    except asyncio.TimeoutError:
                        raise AssertionError(
                            f"FakePipecatChannel.recv(): the fake bot had nothing scheduled and "
                            f"nothing woke the adapter for {self.idle_real_timeout_s} s of real time. "
                            "Script more bot items, or check that the adapter's timeouts use the clock."
                        ) from None
            finally:
                self._wake = None

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        self.connected = False
        self._poke()
        await asyncio.sleep(0)

    # -- internals ------------------------------------------------------------------------
    def _has_undelivered_due_items(self):
        return bool(self._queue) and self._queue[0][0] <= self.clock.monotonic() and not self.closed

    def _poke(self):
        if self._wake is not None and not self._wake.done():
            self._wake.set_result(None)

    def _convert(self, item, when):
        types = _pipecat_types()
        if isinstance(item, (types.ChannelMessage, types.ChannelAudio, types.ChannelClose)):
            return item
        if isinstance(item, dict):
            return types.ChannelMessage(data=copy.deepcopy(item))
        if isinstance(item, BotAudio):
            return types.ChannelAudio(pcm=item.pcm, sample_rate=item.sample_rate, arrival_s=when)
        if isinstance(item, BotClose):
            return types.ChannelClose(code=item.code)
        raise TypeError(f"FakePipecatChannel cannot emit {item!r}")

    def _schedule(self, when, item):
        heapq.heappush(self._queue, (float(when), next(self._seq), self._convert(item, float(when))))
        self._poke()

    def _fire(self, kind, record):
        trigger = record.get("data", record) if kind == "message" else record
        for rule in list(self._rules):
            if rule["kind"] != kind or not rule["match"](record):
                continue
            if rule["once"]:
                self._rules.remove(rule)
            when = self.clock.monotonic() + rule["after"]
            for item in rule["items"]:
                produced = item(trigger) if callable(item) else item
                for single in produced if isinstance(produced, (list, tuple)) else [produced]:
                    if single is None:
                        continue
                    self._schedule(when, single)
                    when += rule["gap"]

    def _last_speech_index(self, pcm):
        usable = len(pcm) - (len(pcm) % 2)
        samples = array.array("h")
        samples.frombytes(bytes(pcm[:usable]))
        if sys.byteorder == "big":
            samples.byteswap()
        for index in range(len(samples) - 1, -1, -1):
            if abs(samples[index]) > self.silence_threshold:
                return index, len(samples)
        return -1, len(samples)


class MockPipecat:
    """What ``mock_pipecat_channel`` returns.

    * ``clock``: the shared ``FakeClock``; pass it as ``PipecatAdapter(..., clock=mock.clock)``.
    * ``factory``: pass as ``channel_factory``; any arguments are accepted and recorded in
      ``factory_calls``. Each call returns the next channel (``channels[0]`` first, then channels
      added with ``add_channel()``, then fresh ones), so scenarios get separate connections.
    * ``channel``: ``channels[0]``, ready to script before the adapter starts.
    * ``rtvi(type, data=None, **extra)``: an RTVI message dict with ``"label": "rtvi-ai"``.
    * ``tone`` / ``silence``: raw mono 16-bit PCM; ``BotAudio`` / ``BotClose``: items to emit.
    """

    BotAudio = BotAudio
    BotClose = BotClose

    def __init__(self):
        self.clock = FakeClock()
        self.channels = [FakePipecatChannel(self.clock, name="channel-1")]
        self.factory_calls = []
        self._next_index = 0

    @property
    def channel(self):
        return self.channels[0]

    def add_channel(self):
        channel = FakePipecatChannel(self.clock, name=f"channel-{len(self.channels) + 1}")
        self.channels.append(channel)
        return channel

    def factory(self, *args, **kwargs):
        self.factory_calls.append({"args": args, "kwargs": kwargs})
        if self._next_index >= len(self.channels):
            self.add_channel()
        channel = self.channels[self._next_index]
        self._next_index += 1
        return channel

    channel_factory = property(lambda self: self.factory)

    @staticmethod
    def rtvi(type_, data=None, **extra):
        message = {"label": "rtvi-ai", "type": type_}
        if data is not None:
            message["data"] = data
        message.update(extra)
        return message

    @staticmethod
    def tone(seconds, sample_rate=16000, frequency_hz=440.0, amplitude=0.5) -> bytes:
        return pcm_tone(seconds, sample_rate, frequency_hz, amplitude)

    @staticmethod
    def silence(seconds, sample_rate=16000) -> bytes:
        return pcm_silence(seconds, sample_rate)

    @staticmethod
    def types():
        """The ``impulse_evals.adapters.pipecat_adapter`` module (Channel item types)."""
        return _pipecat_types()


@pytest.fixture
def mock_pipecat_channel():
    """Scripted fake Pipecat ``Channel`` with a virtual ``Clock`` (no network, no pipecat-ai).

    The Impulse channel types are imported lazily (when items are scheduled), so this fixture
    works before ``impulse_evals.adapters.pipecat_adapter`` exists.
    """
    return MockPipecat()


# =============================================================================================
# Stage B group 4-8 (phases 4-8): project folders, LLM routing, fake adapters, result builders,
# and the CLI runner. Added after the group 0-3 fixtures above, which are unchanged.
# =============================================================================================

import re  # noqa: E402
import textwrap  # noqa: E402
import uuid  # noqa: E402

import yaml  # noqa: E402

TEST_PRIMARY_MODEL = "ollama/judge-primary"
TEST_FAST_MODEL = "ollama/judge-fast"
TEST_SIMULATOR_MODEL = "ollama/simulator"
TEST_SYSTEM_PROMPT = (
    "You are the booking assistant of Sunrise Clinic. Greet the caller, ask for consent before "
    "collecting personal data, and book appointments with the book_slot tool. Confirm a booking "
    "only after book_slot succeeds."
)
KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "IMPULSE_VOICE_API_KEY")
MODEL_ENV_VARS = ("IMPULSE_MODEL_PRIMARY", "IMPULSE_MODEL_FAST", "IMPULSE_MODEL_SIMULATOR")

# First sentence of the SPEC.md section 12 judge system message and of the section 10 simulator prompt.
JUDGE_MARK = "You are an expert evaluator for AI voice agents."
SIMULATOR_MARK = "You are playing a realistic user calling a voice agent."

# A distinctive substring of each criterion (METRICS.md), used to route judge calls.
CRITERION_MARKS = {
    "dpdp.consent_before_pii": "Before collecting personal data",
    "dpdp.purpose_disclosure": "Did the agent say for what purpose the data is collected",
    "dpdp.withdrawal_path": "If the user asked to stop, delete, or withdraw data",
    "UNIV-001": "Infer from the system prompt what a successfully completed interaction",
    "UNIV-003": "Did the agent correctly remember and use information",
    "UNIV-013": "Does the agent's tone match",
    "UNIV-019": "Apply the SAFETY FLOOR first",
    "UNIV-023": "Look at every tool call in the transcript",
}

DPDP_RULE_META = {
    "dpdp.consent_before_pii": ("critical", "DPDP Act 2023 (notice and consent provisions)"),
    "dpdp.purpose_disclosure": ("high", "DPDP Act 2023 (notice)"),
    "dpdp.withdrawal_path": ("medium", "DPDP Act 2023 (withdrawal of consent)"),
}

METRIC_NAMES = {
    "UNIV-001": "Task completion",
    "UNIV-003": "Context retention",
    "UNIV-005": "Factual accuracy",
    "UNIV-013": "Tone",
    "UNIV-019": "Harmful or misleading content",
    "UNIV-023": "Tool call correctness",
    "UNIV-025": "Expected tool calls",
    "VOICE-001": "Time to first audio (TTFA)",
    "VOICE-011": "Harness round-trip WER (sanity)",
    "VOICE-029": "Spoken-length fit",
}

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def console_text(text) -> str:
    """Console output with ANSI codes removed and every whitespace run collapsed to one space.

    Rich wraps long lines at the console width (80 columns when not a terminal), so exact messages
    are compared on this normalised form.
    """
    return " ".join(_ANSI_RE.sub("", text or "").split())


@pytest.fixture
def normalize_console():
    """The ``console_text`` helper: strip ANSI codes and collapse whitespace."""
    return console_text


# ---------------------------------------------------------------------------------------------
# llm_router: one mock_llm responder for judge, simulator, and other calls
# ---------------------------------------------------------------------------------------------


class LLMRouter:
    """Routes ``mock_llm`` calls by kind; installed as ``mock_llm.responder``.

    * judge calls (a message contains the SPEC.md section 12 judge system text): the first matching
      ``on_judge`` rule answers, else ``judge_default`` (a "pass" verdict). A rule matches on
      substrings of the criterion section and of the ``<transcript>`` section of the judge message.
    * simulator calls (a message contains the SPEC.md section 10 simulator text): the next scripted
      text for the persona on the ``Your persona:`` line, else ``simulator_default`` (``<END_CALL>``).
    * every other call (the system prompt summary, init / doctor pings): ``other_text``.
    Contents may be ``None`` (an empty answer); a dict is sent as JSON text.
    """

    def __init__(self, mock):
        self.mock = mock
        self.judge_rules = []
        self.scripts = {}
        self.judge_default = MockLLM.verdict("pass", reason="router: pass", confidence=0.9)
        self.simulator_default = "<END_CALL>"
        self.other_text = "Books clinic appointments for callers with the book_slot tool."
        mock.responder = self._respond

    verdict = staticmethod(MockLLM.verdict)

    # -- scripting ------------------------------------------------------------------------
    def on_judge(self, content, criterion=None, transcript=None, times=None):
        """Answer judge calls whose criterion contains ``criterion`` (a CRITERION_MARKS key or any
        substring) and whose transcript contains ``transcript``; ``times`` limits the uses."""
        criterion = CRITERION_MARKS.get(criterion, criterion)
        self.judge_rules.append(
            {"content": content, "criterion": criterion, "transcript": transcript, "times": times, "used": 0}
        )
        return self

    def script(self, persona, *texts):
        """Simulator answers, in order, for the scenario whose persona is ``persona``."""
        self.scripts.setdefault(persona, deque()).extend(texts)
        return self

    # -- inspection -----------------------------------------------------------------------
    @staticmethod
    def contents(call):
        out = []
        for message in call.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
            out.append(content if isinstance(content, str) else "")
        return out

    @classmethod
    def kind(cls, call):
        texts = cls.contents(call)
        if any(JUDGE_MARK in text for text in texts):
            return "judge"
        if any(SIMULATOR_MARK in text for text in texts):
            return "simulator"
        return "other"

    def calls(self, kind=None):
        return [call for call in self.mock.calls if kind is None or self.kind(call) == kind]

    @classmethod
    def judge_sections(cls, call):
        """``(criterion, transcript)`` text of a judge call's user message."""
        for text in cls.contents(call):
            if "EVALUATION CRITERION:" in text:
                criterion = text.split("EVALUATION CRITERION:", 1)[1].split("Respond in JSON only", 1)[0].strip()
                transcript = ""
                if "<transcript>" in text:
                    transcript = text.split("<transcript>", 1)[1].split("</transcript>", 1)[0].strip()
                return criterion, transcript
        return "", ""

    def judge_calls_for(self, criterion, transcript=None):
        criterion = CRITERION_MARKS.get(criterion, criterion)
        found = []
        for call in self.calls("judge"):
            crit, trans = self.judge_sections(call)
            if criterion in crit and (transcript is None or transcript in trans):
                found.append(call)
        return found

    @classmethod
    def simulator_prompt(cls, call):
        for text in cls.contents(call):
            if SIMULATOR_MARK in text:
                return text
        return ""

    @classmethod
    def persona_of(cls, call):
        for line in cls.simulator_prompt(call).splitlines():
            if line.startswith("Your persona:"):
                return line[len("Your persona:"):].strip()
        return None

    # -- responder ------------------------------------------------------------------------
    @staticmethod
    def _content(value):
        return json.dumps(value) if isinstance(value, dict) else value

    def _respond(self, call):
        kind = self.kind(call)
        if kind == "judge":
            criterion, transcript = self.judge_sections(call)
            for rule in self.judge_rules:
                if rule["times"] is not None and rule["used"] >= rule["times"]:
                    continue
                if rule["criterion"] is not None and rule["criterion"] not in criterion:
                    continue
                if rule["transcript"] is not None and rule["transcript"] not in transcript:
                    continue
                rule["used"] += 1
                return self._content(rule["content"])
            return self._content(self.judge_default)
        if kind == "simulator":
            queue = self.scripts.get(self.persona_of(call))
            if queue:
                return self._content(queue.popleft())
            return self._content(self.simulator_default)
        return self._content(self.other_text)


@pytest.fixture
def llm_router(mock_llm):
    """``mock_llm`` with an ``LLMRouter`` responder (judge verdicts, simulator turns, other calls)."""
    return LLMRouter(mock_llm)


# ---------------------------------------------------------------------------------------------
# fake_adapter: an adapter_factory for runner.run (SPEC.md section 18.1)
# ---------------------------------------------------------------------------------------------


class FakeAdapterFactory:
    """``adapter_factory`` for ``runner.run``: every call returns a new fake ``AgentInterface``.

    It accepts any arguments (recorded in ``factory_calls``). Each adapter subclasses
    ``impulse_evals.adapters.base.AgentInterface`` and answers with ``AgentReply`` objects (imported
    when the factory is called). Configure before the run:

    * ``replies[scenario_name]``: reply specs consumed in order (``str``, ``dict`` of ``AgentReply``
      fields, or an ``AgentReply``); when exhausted, ``default_reply`` is used.
    * ``warnings`` (returned by ``get_warnings``), ``metadata[scenario_name]`` (``get_metadata``),
      ``system_prompt`` (``get_system_prompt``).
    * ``start_errors[scenario_name] = exc`` and ``send_errors[scenario_name] = (n, exc)`` (raise on
      the n-th send, 1-based).
    ``events`` records every start / send / close in order.
    """

    OPENER = "Hello, this is the fake agent. How can I help?"

    def __init__(self):
        self.factory_calls = []
        self.adapters = []
        self.events = []
        self.replies = {}
        self.warnings = []
        self.metadata = {}
        self.system_prompt = None
        self.start_errors = {}
        self.send_errors = {}

    @classmethod
    def default_reply(cls, user_text, index):
        if user_text is None:
            return {"text": cls.OPENER, "latency_ms": 50.0}
        return {"text": f"Fake agent reply {index} to: {user_text}", "latency_ms": 100.0 * index}

    def sends(self, scenario_name):
        """``(user_text, user_audio)`` of every ``send`` for one scenario, in order."""
        return [
            (event["user_text"], event["user_audio"])
            for event in self.events
            if event["scenario"] == scenario_name and event["event"] == "send"
        ]

    def _reply(self, reply_cls, scenario_name, user_text, index):
        queue = self.replies.get(scenario_name)
        spec = queue.pop(0) if queue else self.default_reply(user_text, index)
        if isinstance(spec, reply_cls):
            return spec
        if isinstance(spec, str):
            spec = {"text": spec}
        values = {"latency_ms": self.default_reply(user_text, index)["latency_ms"]}
        values.update(copy.deepcopy(spec))
        return reply_cls(**values)

    def __call__(self, *args, **kwargs):
        from impulse_evals.adapters.base import AgentInterface, AgentReply

        factory = self

        class FakeAdapter(AgentInterface):
            def __init__(self):
                self.scenario = None
                self.sent = 0

            async def start(self, scenario):
                self.scenario = scenario
                self.sent = 0
                factory.events.append({"scenario": scenario.name, "event": "start"})
                error = factory.start_errors.get(scenario.name)
                if error is not None:
                    raise error

            async def send(self, user_text, user_audio):
                name = self.scenario.name
                self.sent += 1
                factory.events.append(
                    {"scenario": name, "event": "send", "user_text": user_text, "user_audio": user_audio}
                )
                planned = factory.send_errors.get(name)
                if planned is not None and planned[0] == self.sent:
                    raise planned[1]
                await asyncio.sleep(0)
                return factory._reply(AgentReply, name, user_text, self.sent)

            async def close(self):
                name = self.scenario.name if self.scenario is not None else None
                factory.events.append({"scenario": name, "event": "close"})

            def get_system_prompt(self):
                return factory.system_prompt

            def get_warnings(self):
                return list(factory.warnings)

            def get_metadata(self):
                if self.scenario is None:
                    return {}
                return copy.deepcopy(factory.metadata.get(self.scenario.name, {}))

        adapter = FakeAdapter()
        self.factory_calls.append({"args": args, "kwargs": kwargs})
        self.adapters.append(adapter)
        return adapter


@pytest.fixture
def fake_adapter():
    """A ``FakeAdapterFactory`` to pass as ``runner.run(..., adapter_factory=fake_adapter)``."""
    return FakeAdapterFactory()


# ---------------------------------------------------------------------------------------------
# impulse_project: a project folder in tmp_path (also the current directory)
# ---------------------------------------------------------------------------------------------


class ImpulseProject:
    """Writes ``config.yaml``, ``system_prompt.txt``, and agent modules into ``root``.

    Agent modules get unique names (``impulse_test_agent_<hex>``) so cached imports never leak
    between tests. ``write_config`` writes the default sample agent when no agent was written yet.
    Test model strings are local (``ollama/...``), so no key is needed.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.config_path = self.root / "config.yaml"
        self.entrypoint = None
        self.system_prompt = TEST_SYSTEM_PROMPT

    # -- scenario defaults ------------------------------------------------------------------
    @staticmethod
    def persona(name):
        return f"caller for {name}"

    @staticmethod
    def opening(name):
        return f"Hello, this is the caller for {name}."

    @classmethod
    def scenario(cls, name, **fields):
        data = {
            "name": name,
            "language": "en-IN",
            "domain": "healthcare",
            "direction": "inbound",
            "max_turns": 3,
            "persona": cls.persona(name),
            "opening_user_message": cls.opening(name),
            "metrics": ["UNIV-001"],
        }
        data.update(fields)
        return data

    # -- files ------------------------------------------------------------------------------
    def write_agent(self, code=None, stem=None):
        """Write an agent module (default: a copy of examples/sample_agent.py); returns its entrypoint."""
        stem = stem or f"impulse_test_agent_{uuid.uuid4().hex[:10]}"
        if code is None:
            code = (EXAMPLES_DIR / "sample_agent.py").read_text(encoding="utf-8")
        else:
            code = "# SPDX-License-Identifier: Apache-2.0\n" + textwrap.dedent(code)
        (self.root / f"{stem}.py").write_text(code, encoding="utf-8")
        importlib.invalidate_caches()
        self.entrypoint = f"{stem}:respond"
        return self.entrypoint

    def config_data(self, scenarios, *, agent=None, judge=None, simulator=None, metrics=None,
                    voice_config=None, system_prompt=TEST_SYSTEM_PROMPT, **extra):
        agent_block = {"type": "chat", "adapter": "generic", "model": None, "tools": None}
        overrides = dict(agent or {})
        if overrides.get("adapter", "generic") == "generic" and "entrypoint" not in overrides:
            agent_block["entrypoint"] = self.entrypoint or self.write_agent()
        if system_prompt is not None:
            self.system_prompt = system_prompt
            (self.root / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
            agent_block["system_prompt"] = "system_prompt.txt"
        agent_block.update(overrides)
        data = {
            "agent": agent_block,
            "judge": judge if judge is not None else {
                "primary_model": TEST_PRIMARY_MODEL,
                "fast_model": TEST_FAST_MODEL,
                "primary_options": {},
                "fast_options": {},
            },
            "simulator": simulator if simulator is not None else {
                "model": TEST_SIMULATOR_MODEL,
                "persona_style": "realistic",
                "mode": "cascade",
                "options": {},
            },
            "metrics": metrics if metrics is not None else {"enabled": True},
        }
        if voice_config is not None:
            data["voice_config"] = voice_config
        data.update(extra)
        data["scenarios"] = list(scenarios)
        return data

    def write_config(self, scenarios, *, path=None, **kwargs):
        data = self.config_data(scenarios, **kwargs)
        path = Path(path) if path is not None else self.config_path
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    def write_toml(self, text):
        folder = self.root / ".impulse"
        folder.mkdir(exist_ok=True)
        path = folder / "config.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def load(self, path=None):
        from impulse_evals.config import load_config

        return load_config(path if path is not None else self.config_path)

    # -- inspection -------------------------------------------------------------------------
    def files(self):
        """Every file under the project folder (relative POSIX paths), ``__pycache__`` excluded."""
        return {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }

    @property
    def runs_dir(self):
        return self.root / ".impulse" / "runs"

    def run_dirs(self):
        if not self.runs_dir.is_dir():
            return []
        return sorted(path.parent for path in self.runs_dir.glob("*/run.json"))

    def only_run_dir(self):
        dirs = self.run_dirs()
        assert len(dirs) == 1, f"expected exactly one run folder under .impulse/runs, found {dirs}"
        return dirs[0]

    def read_run(self, run_dir=None):
        from impulse_evals.models import RunResult

        run_dir = run_dir if run_dir is not None else self.only_run_dir()
        return RunResult.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))


@pytest.fixture
def impulse_project(tmp_path, monkeypatch):
    """An ``ImpulseProject`` in ``tmp_path``, which is also the current directory.

    Keys and ``IMPULSE_MODEL_*`` are removed from the environment; ``sys.path`` changes are undone
    after the test, no bytecode is written, and agent modules imported from the project (or from
    ``examples/``) are removed from ``sys.modules`` afterwards.
    """
    monkeypatch.chdir(tmp_path)
    for name in KEY_ENV_VARS + MODEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    before = set(sys.modules)
    yield ImpulseProject(tmp_path)
    roots = (tmp_path.resolve(), EXAMPLES_DIR.resolve())
    for name in set(sys.modules) - before:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None)
        if not file:
            continue
        try:
            resolved = Path(file).resolve()
        except (OSError, ValueError):
            continue
        if any(resolved.is_relative_to(root) for root in roots):
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------------------------
# run_builder: hand-built SPEC.md section 7 results (reporter, compare, cohort, report tests)
# ---------------------------------------------------------------------------------------------


class RunBuilder:
    """Builds ``Turn`` / ``CallTranscript`` / ``MetricResult`` / ``ScoreResult`` / ``ScenarioResult`` /
    ``RunResult`` objects. ``ScenarioResult.passed`` is computed with ``models.compute_passed``."""

    @staticmethod
    def models():
        from impulse_evals import models

        return models

    def turn(self, speaker, text, timestamp_ms=0, **fields):
        return self.models().Turn(speaker=speaker, text=text, timestamp_ms=timestamp_ms, **fields)

    def transcript(self, scenario_name, turns=None, *, language="en-IN", accent=None, **fields):
        if turns is None:
            turns = [
                self.turn("user", "Hello, I need an appointment on Tuesday.", 0),
                self.turn("agent", "Sure, which time suits you?", 900, latency_ms=900.0),
            ]
        values = {"duration_ms": 5000, "metadata": {}, "agent_system_prompt": TEST_SYSTEM_PROMPT}
        values.update(fields)
        return self.models().CallTranscript(
            turns=list(turns), scenario_name=scenario_name, language=language, accent=accent, **values
        )

    def metric(self, metric_id, score=1.0, *, label=None, skipped=False, informational=False, reason=None, **fields):
        if skipped:
            score, label = 0.0, "skipped"
        elif label is None:
            label = "good" if score >= 0.6 else ("warning" if score >= 0.4 else "poor")
        category = {"UNIV": "universal", "VOICE": "voice", "CHAT": "chat"}[metric_id.split("-", 1)[0]]
        return self.models().MetricResult(
            metric_id=metric_id,
            name=METRIC_NAMES.get(metric_id, metric_id),
            category=category,
            score=score,
            label=label,
            reason=reason if reason is not None else f"{metric_id}: {label}",
            skipped=skipped,
            informational=informational,
            **fields,
        )

    def assertion(self, rule_id, result="pass", *, severity=None, reason=None, **fields):
        default_severity, regulation_ref = DPDP_RULE_META.get(rule_id, ("high", "test regulation"))
        return self.models().ScoreResult(
            rule_id=rule_id,
            regulation_ref=regulation_ref,
            result=result,
            severity=severity or default_severity,
            reason=reason if reason is not None else f"{rule_id}: {result}",
            **fields,
        )

    def _metric_from(self, item):
        if isinstance(item, tuple):
            metric_id, score, *rest = item
            return self.metric(metric_id, score, **(rest[0] if rest else {}))
        return item

    def _assertion_from(self, item):
        if isinstance(item, tuple):
            rule_id, result, *rest = item
            return self.assertion(rule_id, result, **(rest[0] if rest else {}))
        return item

    def scenario_result(self, name, *, assertions=(), metrics=(), error=None, transcript=None,
                        language="en-IN", accent=None, inject_events=(), passed=None):
        m = self.models()
        result = m.ScenarioResult(
            scenario_name=name,
            transcript=transcript if transcript is not None else self.transcript(name, language=language, accent=accent),
            assertion_results=[self._assertion_from(item) for item in assertions],
            metric_results=[self._metric_from(item) for item in metrics],
            passed=True,
            error=error,
            inject_events=[m.InjectEvent(**e) if isinstance(e, dict) else e for e in inject_events],
        )
        result.passed = m.compute_passed(result) if passed is None else passed
        return result

    def latency(self, values=()):
        values = list(values)
        if not values:
            return self.models().LatencyProfile(p50_ms=0.0, p75_ms=0.0, p95_ms=0.0, p99_ms=0.0, all_latencies_ms=[])
        ordered = sorted(values)
        return self.models().LatencyProfile(
            p50_ms=ordered[len(ordered) // 2], p75_ms=ordered[-1], p95_ms=ordered[-1], p99_ms=ordered[-1],
            all_latencies_ms=values,
        )

    def run(self, scenario_results, *, run_id="20261004T120000Z-a3f9c2", timestamp="2026-10-04T12:00:00Z",
            warnings=(), metric_summary=None, overall_pass_rate=None, voice_mode=False, cohorts=(),
            agent_domain="healthcare", judge_model=TEST_PRIMARY_MODEL, agent_entrypoint="sample_agent:respond",
            agent_type="voice", latency_profile=None, **fields):
        results = list(scenario_results)
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        critical = [
            score
            for result in results
            for score in result.assertion_results
            if score.result == "fail" and score.severity == "critical"
        ]
        return self.models().RunResult(
            run_id=run_id,
            timestamp=timestamp,
            agent_entrypoint=agent_entrypoint,
            agent_type=agent_type,
            agent_domain=agent_domain,
            judge_model=judge_model,
            voice_mode=voice_mode,
            total_scenarios=total,
            passed=passed,
            failed=total - passed,
            overall_pass_rate=(passed / total if total else 0.0) if overall_pass_rate is None else overall_pass_rate,
            scenario_results=results,
            cohorts=list(cohorts),
            critical_failures=critical,
            metric_summary=dict(metric_summary or {}),
            latency_profile=latency_profile if latency_profile is not None else self.latency(),
            warnings=list(warnings),
            **fields,
        )

    @staticmethod
    def write_run(run, runs_dir):
        """Store ``run`` as ``<runs_dir>/<run_id>/run.json`` (JSON files carry no SPDX header)."""
        folder = Path(runs_dir) / run.run_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "run.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return path


@pytest.fixture
def run_builder():
    """A ``RunBuilder`` for hand-built results (SPEC.md section 7 models)."""
    return RunBuilder()


# ---------------------------------------------------------------------------------------------
# cli: typer.testing.CliRunner against impulse_evals.cli.app
# ---------------------------------------------------------------------------------------------


class CliHelper:
    """Invokes ``impulse_evals.cli.app`` with ``typer.testing.CliRunner``.

    An exception other than ``SystemExit`` raised by the command is re-raised, so an exit code can
    never come from an unhandled error (for example a missing implementation).
    """

    def invoke(self, *args, input=None, env=None):
        from typer.testing import CliRunner

        from impulse_evals.cli import app

        result = CliRunner().invoke(app, [str(arg) for arg in args], input=input, env=env)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception.with_traceback(result.exc_info[2] if result.exc_info else None)
        return result

    @staticmethod
    def text(result):
        """``result.output`` (stdout and stderr) normalised with ``console_text``."""
        return console_text(result.output)


@pytest.fixture
def cli():
    """A ``CliHelper``: ``cli.invoke("run", path, "--fail-on", "high")``; ``cli.text(result)``."""
    return CliHelper()


# =============================================================================================
# Stage B group 9-13 part B (phase 11): the scripted fake LiveKit ``Room`` (SPEC.md sections 9 and
# 18.1), import-state simulation for the optional extras, and WAV helpers. Added after the group
# 0-3 and 4-8 fixtures above, which are unchanged.
# =============================================================================================

import importlib.machinery  # noqa: E402
import types  # noqa: E402
from dataclasses import field  # noqa: E402

LIVEKIT_AGENT_IDENTITY = "agent-neo"


def _livekit_types():
    """The Room item types from the adapter module (imported lazily: built in phase 11)."""
    from impulse_evals.adapters import livekit_adapter

    return livekit_adapter


class _ImpulseIdentity:
    """Placeholder for Impulse's own participant identity, known only after ``Room.connect()``."""

    def __repr__(self):
        return "IMPULSE"


IMPULSE_IDENTITY = _ImpulseIdentity()


@dataclass
class LKParticipant:
    """Emitted as ``RoomParticipant``; ``identity`` may be ``IMPULSE_IDENTITY``."""

    identity: object = LIVEKIT_AGENT_IDENTITY
    is_agent: bool = True
    attributes: dict = field(default_factory=dict)


@dataclass
class LKText:
    """Emitted as ``RoomText`` (one complete text stream) with ``arrival_s`` = its virtual time."""

    text: str
    sender_identity: object = LIVEKIT_AGENT_IDENTITY
    topic: str = "lk.transcription"
    final: bool = True


@dataclass
class LKAttributes:
    """Emitted as ``RoomAttributes`` (``participant_attributes_changed``)."""

    changed: dict
    identity: object = LIVEKIT_AGENT_IDENTITY


@dataclass
class LKSession:
    """Emitted as ``RoomSessionEvent`` (a decoded ``lk.agent.session`` message)."""

    kind: str
    data: dict


@dataclass
class LKAudio:
    """Emitted as ``RoomAudio`` (agent audio) with ``arrival_s`` = its virtual time."""

    pcm: bytes
    sample_rate: int = 24000


@dataclass
class LKClosed:
    """Emitted as ``RoomClosed``."""

    reason: str = "the agent closed the room"


class FakeLiveKitRoom:
    """Scripted fake implementing the SPEC.md section 9 LiveKit ``Room`` protocol (no ``livekit``).

    Script what the fake room delivers (same scheduling model as ``FakePipecatChannel``):
      * ``emit(*items, at=None, after=0.0, gap=0.0)``: at virtual time ``at`` or ``now + after``.
      * ``on_connect(*items, ...)``, ``on_text(match, *items, ...)`` (``match``: None = any text,
        a topic string, or a predicate on the record), ``on_audio(*items, match=None, ...)``,
        ``on_agent_info(*items, ...)``: when the adapter connects / sends text / publishes audio /
        requests the agent info.
    Items: ``LKParticipant``, ``LKText``, ``LKAttributes``, ``LKSession``, ``LKAudio``, ``LKClosed``,
    an existing Room item, or (in ``on_*`` rules) a callable receiving the triggering record and
    returning item(s) or None. ``IMPULSE_IDENTITY`` in an item is replaced, at delivery, by the
    identity the adapter passed to ``connect()``.

    Records: ``connect_calls`` (url, identity, room_name, agent_name, t), ``sent`` (chronological),
    ``sent_texts`` (text, topic, t), ``published_audio`` (pcm, sample_rate, sent_at, returned,
    has_speech, t), ``agent_info_requests`` (agent_identity, t), ``received``, ``close_calls``.

    ``publish_audio`` returns the clock time the last non-silent sample left (|sample| > 1% of full
    scale); with ``paced_audio = True`` (the default, like the real room) it streams in virtual real
    time. ``recv()`` waits in virtual time for the next item; with nothing scheduled it fails after
    ``idle_real_timeout_s`` seconds of real time instead of hanging.
    """

    idle_real_timeout_s = 5.0
    silence_threshold = 327

    def __init__(self, clock: FakeClock, name="room-1"):
        self.clock = clock
        self.name = name
        self.connect_calls = []
        self.sent = []
        self.received = []
        self.close_calls = 0
        self.connected = False
        self.closed = False
        self.connect_error = None
        self.paced_audio = True
        self.identity = None
        self._queue = []
        self._seq = itertools.count()
        self._rules = []
        self._wake = None
        self._last_speech_s = None
        clock._busy_checks.append(self._has_undelivered_due_items)

    # -- views ----------------------------------------------------------------------------
    @property
    def sent_texts(self):
        return [entry for entry in self.sent if entry["kind"] == "text"]

    @property
    def published_audio(self):
        return [entry for entry in self.sent if entry["kind"] == "audio"]

    @property
    def agent_info_requests(self):
        return [entry for entry in self.sent if entry["kind"] == "agent_info"]

    @property
    def connect_call(self):
        return self.connect_calls[-1] if self.connect_calls else None

    @property
    def pending(self):
        return len(self._queue)

    # -- scripting ------------------------------------------------------------------------
    def emit(self, *items, at=None, after=0.0, gap=0.0):
        when = (self.clock.monotonic() + after) if at is None else float(at)
        for item in items:
            self._schedule(when, item)
            when += gap
        return self

    def _add_rule(self, kind, predicate, items, after, gap, once):
        self._rules.append(
            {"kind": kind, "match": predicate, "items": items, "after": after, "gap": gap, "once": once}
        )
        return self

    def on_connect(self, *items, after=0.0, gap=0.0):
        return self._add_rule("connect", lambda rec: True, items, after, gap, True)

    def on_text(self, match, *items, after=0.0, gap=0.0, once=True):
        if match is None:
            predicate = lambda rec: True  # noqa: E731
        elif isinstance(match, str):
            predicate = lambda rec, _topic=match: rec["topic"] == _topic  # noqa: E731
        else:
            predicate = match
        return self._add_rule("text", predicate, items, after, gap, once)

    def on_audio(self, *items, match=None, after=0.0, gap=0.0, once=True):
        return self._add_rule("audio", match or (lambda rec: True), items, after, gap, once)

    def on_agent_info(self, *items, after=0.0, gap=0.0, once=True):
        return self._add_rule("agent_info", lambda rec: True, items, after, gap, once)

    # -- Room protocol --------------------------------------------------------------------
    async def connect(self, url, identity, room_name, agent_name=None):
        record = {
            "url": url,
            "identity": identity,
            "room_name": room_name,
            "agent_name": agent_name,
            "t": self.clock.monotonic(),
        }
        self.connect_calls.append(record)
        if self.connect_error is not None:
            raise self.connect_error
        self.identity = identity
        self.connected = True
        self.closed = False
        self._fire("connect", record)
        await asyncio.sleep(0)

    def _check_open(self, what):
        if not self.connected:
            raise RuntimeError(f"FakeLiveKitRoom: {what}() before connect()")
        if self.closed:
            raise RuntimeError(f"FakeLiveKitRoom: {what}() after close()")

    async def send_text(self, text, topic):
        self._check_open("send_text")
        record = {"kind": "text", "text": text, "topic": topic, "t": self.clock.monotonic()}
        self.sent.append(record)
        self._fire("text", record)
        await asyncio.sleep(0)

    async def publish_audio(self, pcm, sample_rate):
        self._check_open("publish_audio")
        start = self.clock.monotonic()
        last_index, count = FakePipecatChannel._last_speech_index(self, pcm)
        if self.paced_audio:
            await self.clock.sleep(count / float(sample_rate))
            speech_end = start + (last_index + 1) / float(sample_rate) if last_index >= 0 else None
        else:
            speech_end = start if last_index >= 0 else None
        if speech_end is not None:
            self._last_speech_s = speech_end
        returned = self._last_speech_s if self._last_speech_s is not None else start
        record = {
            "kind": "audio",
            "pcm": bytes(pcm),
            "sample_rate": sample_rate,
            "sent_at": start,
            "returned": returned,
            "has_speech": last_index >= 0,
            "t": self.clock.monotonic(),
        }
        self.sent.append(record)
        self._fire("audio", record)
        await asyncio.sleep(0)
        return returned

    async def request_agent_info(self, agent_identity):
        self._check_open("request_agent_info")
        record = {"kind": "agent_info", "agent_identity": agent_identity, "t": self.clock.monotonic()}
        self.sent.append(record)
        self._fire("agent_info", record)
        await asyncio.sleep(0)

    async def recv(self):
        loop = asyncio.get_running_loop()
        while True:
            if self._queue and self._queue[0][0] <= self.clock.monotonic():
                when, _, spec = heapq.heappop(self._queue)
                item = self._convert(spec, when)
                self.received.append(item)
                return item
            if self.closed:
                return _livekit_types().RoomClosed(reason="closed by Impulse")
            self._wake = loop.create_future()
            try:
                if self._queue:
                    self.clock._add_waiter(self._queue[0][0], self._wake)
                    await self._wake
                else:
                    try:
                        await asyncio.wait_for(self._wake, self.idle_real_timeout_s)
                    except asyncio.TimeoutError:
                        raise AssertionError(
                            f"FakeLiveKitRoom.recv(): the fake room had nothing scheduled and nothing "
                            f"woke the adapter for {self.idle_real_timeout_s} s of real time. Script "
                            "more room items, or check that the adapter's timeouts use the clock."
                        ) from None
            finally:
                self._wake = None

    async def close(self):
        self.close_calls += 1
        self.closed = True
        self.connected = False
        self._poke()
        await asyncio.sleep(0)

    # -- internals ------------------------------------------------------------------------
    def _has_undelivered_due_items(self):
        return bool(self._queue) and self._queue[0][0] <= self.clock.monotonic() and not self.closed

    def _poke(self):
        if self._wake is not None and not self._wake.done():
            self._wake.set_result(None)

    def _who(self, identity):
        return self.identity if identity is IMPULSE_IDENTITY else identity

    def _convert(self, spec, when):
        types_ = _livekit_types()
        if isinstance(
            spec,
            (types_.RoomParticipant, types_.RoomText, types_.RoomAttributes, types_.RoomSessionEvent,
             types_.RoomAudio, types_.RoomClosed),
        ):
            return spec
        if isinstance(spec, LKParticipant):
            return types_.RoomParticipant(
                identity=self._who(spec.identity), is_agent=spec.is_agent, attributes=dict(spec.attributes)
            )
        if isinstance(spec, LKText):
            return types_.RoomText(
                topic=spec.topic, sender_identity=self._who(spec.sender_identity), text=spec.text,
                final=spec.final, arrival_s=when,
            )
        if isinstance(spec, LKAttributes):
            return types_.RoomAttributes(
                identity=self._who(spec.identity), changed=dict(spec.changed), arrival_s=when
            )
        if isinstance(spec, LKSession):
            return types_.RoomSessionEvent(kind=spec.kind, data=copy.deepcopy(spec.data), arrival_s=when)
        if isinstance(spec, LKAudio):
            return types_.RoomAudio(pcm=spec.pcm, sample_rate=spec.sample_rate, arrival_s=when)
        if isinstance(spec, LKClosed):
            return types_.RoomClosed(reason=spec.reason)
        raise TypeError(f"FakeLiveKitRoom cannot emit {spec!r}")

    def _schedule(self, when, item):
        if not isinstance(item, (LKParticipant, LKText, LKAttributes, LKSession, LKAudio, LKClosed)):
            self._convert(item, when)  # type check now: fail at scripting time, not delivery
        heapq.heappush(self._queue, (float(when), next(self._seq), item))
        self._poke()

    def _fire(self, kind, record):
        for rule in list(self._rules):
            if rule["kind"] != kind or not rule["match"](record):
                continue
            if rule["once"]:
                self._rules.remove(rule)
            when = self.clock.monotonic() + rule["after"]
            for item in rule["items"]:
                produced = item(record) if callable(item) else item
                for single in produced if isinstance(produced, (list, tuple)) else [produced]:
                    if single is None:
                        continue
                    self._schedule(when, single)
                    when += rule["gap"]


class MockLiveKit:
    """What ``mock_livekit_room`` returns.

    * ``clock``: the shared ``FakeClock``; pass it as ``LiveKitAdapter(..., clock=mock.clock)``.
    * ``factory``: pass as ``room_factory``; a plain callable accepting any arguments (recorded in
      ``factory_calls``) that returns the next room (``rooms[0]`` first, then rooms added with
      ``add_room()``, then fresh ones), so every scenario gets its own room.
    * ``room``: ``rooms[0]``, ready to script before the adapter starts.
    * Item helpers: ``participant``, ``agent_text``, ``user_transcription``, ``state``,
      ``agent_info``, ``tools_executed``, ``audio``, ``closed``; ``tone`` / ``silence`` (raw PCM).
    """

    AGENT = LIVEKIT_AGENT_IDENTITY
    IMPULSE = IMPULSE_IDENTITY
    LKParticipant = LKParticipant
    LKText = LKText
    LKAttributes = LKAttributes
    LKSession = LKSession
    LKAudio = LKAudio
    LKClosed = LKClosed

    def __init__(self):
        self.clock = FakeClock()
        self.rooms = [FakeLiveKitRoom(self.clock, name="room-1")]
        self.factory_calls = []
        self._next_index = 0

    @property
    def room(self):
        return self.rooms[0]

    def add_room(self):
        room = FakeLiveKitRoom(self.clock, name=f"room-{len(self.rooms) + 1}")
        self.rooms.append(room)
        return room

    def factory(self, *args, **kwargs):
        self.factory_calls.append({"args": args, "kwargs": kwargs})
        if self._next_index >= len(self.rooms):
            self.add_room()
        room = self.rooms[self._next_index]
        self._next_index += 1
        return room

    room_factory = property(lambda self: self.factory)

    # -- item helpers -----------------------------------------------------------------------
    @staticmethod
    def participant(identity=LIVEKIT_AGENT_IDENTITY, is_agent=True, attributes=None):
        return LKParticipant(identity=identity, is_agent=is_agent, attributes=dict(attributes or {}))

    @staticmethod
    def agent_text(text, final=True, sender=LIVEKIT_AGENT_IDENTITY, topic="lk.transcription"):
        return LKText(text=text, sender_identity=sender, topic=topic, final=final)

    @staticmethod
    def user_transcription(text, final=True):
        """What the agent says it heard: an ``lk.transcription`` stream sent with Impulse's identity."""
        return LKText(text=text, sender_identity=IMPULSE_IDENTITY, topic="lk.transcription", final=final)

    @staticmethod
    def state(value, identity=LIVEKIT_AGENT_IDENTITY):
        return LKAttributes(changed={"lk.agent.state": value}, identity=identity)

    @staticmethod
    def agent_info(instructions, tools=()):
        return LKSession(kind="agent_info", data={"instructions": instructions, "tools": list(tools)})

    @staticmethod
    def tools_executed(function_calls, function_call_outputs=()):
        return LKSession(
            kind="function_tools_executed",
            data={"function_calls": list(function_calls), "function_call_outputs": list(function_call_outputs)},
        )

    @staticmethod
    def audio(pcm, sample_rate=24000):
        return LKAudio(pcm=pcm, sample_rate=sample_rate)

    @staticmethod
    def closed(reason="the agent closed the room"):
        return LKClosed(reason=reason)

    @staticmethod
    def tone(seconds, sample_rate=24000, frequency_hz=440.0, amplitude=0.5) -> bytes:
        return pcm_tone(seconds, sample_rate, frequency_hz, amplitude)

    @staticmethod
    def silence(seconds, sample_rate=24000) -> bytes:
        return pcm_silence(seconds, sample_rate)

    @staticmethod
    def types():
        """The ``impulse_evals.adapters.livekit_adapter`` module (Room item types)."""
        return _livekit_types()


@pytest.fixture
def mock_livekit_room():
    """Scripted fake LiveKit ``Room`` with a virtual ``Clock`` (no network, no ``livekit``).

    The Impulse room types are imported lazily (when items are delivered), so this fixture works
    before ``impulse_evals.adapters.livekit_adapter`` exists.
    """
    return MockLiveKit()


# ---------------------------------------------------------------------------------------------
# Import-state simulation for the optional extras (no package is installed or removed)
# ---------------------------------------------------------------------------------------------


def _fake_package(name):
    """An empty, importable package module with a real ``__spec__`` (``find_spec`` sees it)."""
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    module.__path__ = []
    return module


class ExtraImports:
    """Simulates which optional packages are importable, through ``sys.modules`` (undone after the
    test). ``None`` in ``sys.modules`` makes ``import name`` raise ``ImportError`` and
    ``importlib.util.find_spec(name)`` return None.

    * ``missing(name)``: ``name`` and its submodules cannot be imported.
    * ``package(name)``: an empty importable package (its parents too).
    * ``pipecat(state)``: ``"missing"`` (no pipecat-ai), ``"no_evals"`` (pipecat importable,
      ``pipecat.evals`` not), ``"with_evals"`` (both importable, empty).
    * ``module(name, module)``: install a fake module object under ``name``.
    """

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch

    def _clear(self, name):
        for loaded in list(sys.modules):
            if loaded == name or loaded.startswith(name + "."):
                self._monkeypatch.delitem(sys.modules, loaded, raising=False)

    def missing(self, name):
        self._clear(name)
        self._monkeypatch.setitem(sys.modules, name, None)

    def package(self, name):
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            dotted = ".".join(parts[:index])
            current = sys.modules.get(dotted)
            if current is None or not hasattr(current, "__path__"):
                package = _fake_package(dotted)
                self._monkeypatch.setitem(sys.modules, dotted, package)
                if index > 1:
                    setattr(sys.modules[".".join(parts[: index - 1])], parts[index - 1], package)
        return sys.modules[name]

    def module(self, name, module):
        self._clear(name)
        self._monkeypatch.setitem(sys.modules, name, module)
        return module

    def pipecat(self, state):
        self._clear("pipecat")
        if state == "missing":
            self._monkeypatch.setitem(sys.modules, "pipecat", None)
        elif state == "no_evals":
            self._monkeypatch.setitem(sys.modules, "pipecat", _fake_package("pipecat"))
            self._monkeypatch.setitem(sys.modules, "pipecat.evals", None)
        elif state == "with_evals":
            self.package("pipecat.evals")
        else:  # pragma: no cover - test bug
            raise ValueError(state)


@pytest.fixture
def extra_imports(monkeypatch):
    """An ``ExtraImports`` helper: simulate missing or present optional extras via ``sys.modules``."""
    return ExtraImports(monkeypatch)


# ---------------------------------------------------------------------------------------------
# WAV helpers for voice tests
# ---------------------------------------------------------------------------------------------


def read_wav(data_or_path):
    """``(channels, sample_width, sample_rate, frames_bytes)`` of a WAV given as bytes or a path."""
    if isinstance(data_or_path, (bytes, bytearray)):
        handle = wave.open(io.BytesIO(bytes(data_or_path)), "rb")
    else:
        handle = wave.open(str(data_or_path), "rb")
    with handle:
        return (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
            handle.readframes(handle.getnframes()),
        )


def pcm_samples(pcm: bytes):
    """16-bit little-endian PCM bytes as a list of ints."""
    usable = len(pcm) - (len(pcm) % 2)
    samples = array.array("h")
    samples.frombytes(bytes(pcm[:usable]))
    if sys.byteorder == "big":
        samples.byteswap()
    return list(samples)


def multipart_fields(record):
    """Form fields of a recorded ``multipart/form-data`` request (``mock_voice_server.requests``).

    Returns ``{name: [{"filename", "content_type", "value": bytes}, ...]}`` (a list per name, in order).
    """
    content_type = record["headers"].get("content-type", "")
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    assert match, f"not a multipart request: {content_type!r}"
    boundary = b"--" + match.group(1).encode("latin-1")
    fields = {}
    for part in record["content"].split(boundary)[1:]:
        if part.startswith(b"--"):
            break
        part = part[2:] if part.startswith(b"\r\n") else part
        head, _, body = part.partition(b"\r\n\r\n")
        if body.endswith(b"\r\n"):
            body = body[:-2]
        headers = head.decode("latin-1")
        name = re.search(r'name="([^"]*)"', headers)
        filename = re.search(r'filename="([^"]*)"', headers)
        part_type = re.search(r"(?im)^content-type:\s*(.+?)\s*$", headers)
        fields.setdefault(name.group(1) if name else "", []).append(
            {
                "filename": filename.group(1) if filename else None,
                "content_type": part_type.group(1) if part_type else None,
                "value": body,
            }
        )
    return fields
