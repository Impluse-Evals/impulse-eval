# SPDX-License-Identifier: Apache-2.0
"""Phase 11, AUDIT batch 2 (findings 12 and 15): eval mode with the real ``EvalChannel``
(``PipecatAdapter(config)`` with no ``channel_factory``) against ``websockets`` servers on 127.0.0.1
(an ephemeral port, started by each test). No other network, no LLM.

Skipped unless ``pipecat-ai`` is installed (optional extra ``pipecat``).

* finding 12: a rejected WebSocket upgrade (HTTP 403 from ``PIPECAT_ALLOWED_ORIGINS``, 401 for a bad
  token, 404 for a wrong path) fails ``start()`` as soon as Pipecat reports it, with the harness-failure
  text naming the HTTP status, instead of after the 12 s connect timeout.
* finding 15 (eval harness side): a harness pipeline error whose text holds a key straddling the
  200-character cut keeps no part of the key (keys are redacted before the text is shortened): the
  adapter's own ``pipecat.api_key_env`` token, and through ``runner.run`` a key only the runner knows.
  The server answers the upgrade with a ``Sec-WebSocket-Accept`` header the client rejects; Pipecat
  reports that as "Error setting up processor: invalid Sec-WebSocket-Accept header: <value>".
* regression guard: a server that accepts the upgrade and answers ``client-ready`` with ``bot-ready``
  still completes ``start()``.
"""

import asyncio
import json
import time
from http import HTTPStatus

import pytest

pytest.importorskip("pipecat", reason="pipecat-ai is not installed (optional extra 'pipecat')")
pytest.importorskip("websockets.asyncio.server", reason="websockets (a pipecat-ai dependency) is not installed")

from websockets.asyncio.server import serve

pytestmark = pytest.mark.phase11

SAFETY_CAP_S = 30.0  # a hang fails the test instead of blocking the suite
FAST_S = 3.0  # "as soon as Pipecat reports it": well under EvalChannel's 12 s connect timeout
HARNESS_FAILED_PREFIX = "Pipecat eval harness pipeline failed: "
BOT_READY = {
    "label": "rtvi-ai",
    "type": "bot-ready",
    "id": "client-ready",
    "data": {"version": "2.1.0", "about": {"library": "pipecat-ai", "library_version": "1.12.0"}},
}

KEY_ENV = "IMPULSE_TEST_PIPECAT_TOKEN"
KEY = "pck-test-PIPECAT-SECRET-0123456789abcdef"  # 40 characters
# Pipecat's text for the rejected header, before its value (websockets InvalidHeaderValue, pipecat-ai 1.12.0).
BAD_ACCEPT_PREFIX = "Error setting up processor: invalid Sec-WebSocket-Accept header: "
TEXT_BEFORE_KEY = 186  # the 200-character cut falls 14 characters into the key; redacted, the text is 196


def _cfg(project, url, **pipecat):
    block = {"mode": "eval", "url": url}
    block.update(pipecat)
    project.write_config(
        [project.scenario("eval_connect", max_turns=1, metrics=["UNIV-013"])],
        agent={"adapter": "pipecat", "pipecat": block},
    )
    return project.load()


def _url(server):
    port = next(iter(server.sockets)).getsockname()[1]
    return f"ws://127.0.0.1:{port}"


async def _never(connection):
    await connection.wait_closed()


async def _failed_start(adapter, scenario):
    """``(real seconds start() took, the exception it raised)``; the adapter is always closed."""
    started = time.perf_counter()
    try:
        with pytest.raises(Exception) as info:
            await asyncio.wait_for(adapter.start(scenario), timeout=SAFETY_CAP_S)
        return time.perf_counter() - started, info.value
    finally:
        await adapter.close()


def _rejected_accept(tail, seen):
    """``process_response`` for ``serve``: answers the upgrade with a ``Sec-WebSocket-Accept`` value the
    client rejects, ending with ``tail(request)`` (which ends with the key), padded so that Pipecat's
    error text has exactly ``TEXT_BEFORE_KEY`` characters before the key."""

    def tamper(connection, request, response):
        seen.append(request)
        end = tail(request)
        filler = "e" * (TEXT_BEFORE_KEY - len(BAD_ACCEPT_PREFIX) - (len(end) - len(KEY)))
        del response.headers["Sec-WebSocket-Accept"]
        response.headers["Sec-WebSocket-Accept"] = filler + end
        return response

    return tamper


# -- finding 12: a rejected upgrade fails start() fast ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        pytest.param(HTTPStatus.FORBIDDEN, id="403"),
        pytest.param(HTTPStatus.UNAUTHORIZED, id="401"),
        pytest.param(HTTPStatus.NOT_FOUND, id="404"),
    ],
)
async def test_rejected_upgrade_fails_start_fast_with_the_http_status(impulse_project, status):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    seen = []

    def reject(connection, request):
        seen.append(request.path)
        return connection.respond(status, f"{status.phrase}\n")

    async with serve(_never, "127.0.0.1", 0, process_request=reject) as server:
        cfg = _cfg(impulse_project, _url(server))
        adapter = PipecatAdapter(cfg)  # no channel_factory: the real EvalChannel
        elapsed, error = await _failed_start(adapter, cfg.scenarios[0])
    message = str(error)
    assert seen, "the upgrade request never reached the loopback server"
    assert elapsed < FAST_S, f"start() took {elapsed:.1f} s to report the rejected upgrade: {message}"
    assert f"HTTP {int(status)}" in message, message
    assert message.startswith(HARNESS_FAILED_PREFIX), message


@pytest.mark.asyncio
async def test_eval_mode_start_completes_against_an_accepting_server(impulse_project):
    """Regression guard (passes before the finding 12 fix): an upgrade the server accepts still
    completes the connect and the handshake; the adapter sends eval-configure and eval-context."""
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    paths, received = [], []

    async def bot(connection):
        paths.append(connection.request.path)
        async for message in connection:
            received.append(json.loads(message))
            if len(received) == 1:  # Pipecat's client transport sent client-ready
                await connection.send(json.dumps(BOT_READY))

    async with serve(bot, "127.0.0.1", 0) as server:
        cfg = _cfg(impulse_project, _url(server))
        adapter = PipecatAdapter(cfg)
        try:
            started = time.perf_counter()
            await asyncio.wait_for(adapter.start(cfg.scenarios[0]), timeout=SAFETY_CAP_S)
            elapsed = time.perf_counter() - started
            deadline = time.perf_counter() + FAST_S
            while len(received) < 3 and time.perf_counter() < deadline:
                await asyncio.sleep(0.02)
        finally:
            await adapter.close()
    assert elapsed < FAST_S
    assert paths and "skip_tts=true" in paths[0]
    assert [message.get("type") for message in received[:3]] == ["client-ready", "client-message", "client-message"]
    assert [message["data"].get("t") for message in received[1:3]] == ["eval-configure", "eval-context"]


# -- finding 15: harness pipeline errors are redacted before they are shortened -----------------------


@pytest.mark.asyncio
async def test_harness_error_with_the_bot_token_is_redacted_before_it_is_shortened(impulse_project, monkeypatch):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    monkeypatch.setenv(KEY_ENV, KEY)
    seen = []
    # The server's rejected header echoes the request's Authorization header (as proxies do).
    tamper = _rejected_accept(lambda request: " rejected credentials " + request.headers["Authorization"], seen)
    async with serve(_never, "127.0.0.1", 0, process_response=tamper) as server:
        cfg = _cfg(impulse_project, _url(server), api_key_env=KEY_ENV)
        adapter = PipecatAdapter(cfg)
        _elapsed, error = await _failed_start(adapter, cfg.scenarios[0])
    message = str(error)
    assert seen and seen[0].headers.get("Authorization") == f"Bearer {KEY}"
    assert message.startswith(HARNESS_FAILED_PREFIX), message
    assert KEY[:8] not in message, message
    assert message.endswith(" rejected credentials Bearer [REDACTED]"), message


@pytest.mark.asyncio
async def test_harness_error_with_a_key_only_the_runner_knows_is_redacted_before_it_is_shortened(
    impulse_project, llm_router, monkeypatch
):
    from impulse_evals.runner import run

    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)  # a standard key variable: the adapter holds no copy of it
    seen = []
    tamper = _rejected_accept(lambda request: " rejected credentials " + KEY, seen)
    async with serve(_never, "127.0.0.1", 0, process_response=tamper) as server:
        cfg = _cfg(impulse_project, _url(server))
        result = await asyncio.wait_for(run(cfg), timeout=SAFETY_CAP_S)  # the default factory: the real EvalChannel
    assert seen
    error = result.scenario_results[0].error
    assert error is not None and error.startswith(HARNESS_FAILED_PREFIX), error
    assert KEY[:8] not in error, error
    assert error.endswith(" rejected credentials [REDACTED]"), error
    assert KEY[:8] not in result.model_dump_json()


# -- finding 15 review fix-up (Stage D): impulse doctor passes its key redactor to the adapter --------

import threading  # noqa: E402


class _LoopbackServerThread:
    """A ``websockets`` server on 127.0.0.1 (ephemeral port) in a thread with its own event loop, for a
    synchronous test whose command runs its own event loop (the ``cli`` fixture: ``impulse doctor``)."""

    def __init__(self, handler, **serve_kwargs):
        self._handler = handler
        self._serve_kwargs = serve_kwargs
        self._ready = threading.Event()
        self._loop = None
        self._stop = None
        self._thread = None
        self._error = None
        self.url = None

    def __enter__(self):
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()
        if not self._ready.wait(SAFETY_CAP_S):
            raise AssertionError("the loopback server did not start")
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, *exc_info):
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(SAFETY_CAP_S)
        return False

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            async with serve(self._handler, "127.0.0.1", 0, **self._serve_kwargs) as server:
                self.url = _url(server)
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - reported to the test thread
            self._error = exc
            self._ready.set()


@pytest.mark.parametrize("source", ["standard-key-variable", "config-toml-keys"])
def test_doctor_harness_error_with_a_key_only_the_cli_knows_is_redacted_before_it_is_shortened(
    impulse_project, mock_llm, cli, monkeypatch, source
):
    """``impulse doctor`` (eval mode, the real ``EvalChannel``): the rejected upgrade's harness error holds
    a key only the CLI knows (the adapter holds no copy of it), with the adapter's 200-character cut
    falling inside it: the Pipecat handshake row shows ``[REDACTED]``, never a prefix of the key."""
    mock_llm.set_default("OK")
    project = impulse_project
    if source == "standard-key-variable":
        monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)  # set after the fixture
    else:
        project.write_toml(f'[keys]\nGROQ_API_KEY = "{KEY}"\n')  # known from .impulse/config.toml only
    seen = []
    tamper = _rejected_accept(lambda request: " rejected credentials " + KEY, seen)
    with _LoopbackServerThread(_never, process_response=tamper) as server:
        config = project.write_config(
            [project.scenario("eval_connect", max_turns=1, metrics=["UNIV-013"])],
            agent={"adapter": "pipecat", "pipecat": {"mode": "eval", "url": server.url}},
        )
        result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert seen, "the upgrade request never reached the loopback server"
    assert result.exit_code == 2, text
    assert HARNESS_FAILED_PREFIX.strip() in text, text  # the doctor's Pipecat handshake FAIL row
    squashed = text.replace(" ", "")  # Rich wraps rows at 80 columns, possibly inside the key
    assert KEY[:8] not in squashed, text
    assert "rejectedcredentials[REDAC" in squashed, text
