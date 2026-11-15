# SPDX-License-Identifier: Apache-2.0
"""Fixes round 3, B3/B4-rev: revisions of the HTTP and custom adapters after the Stage D review
(FIXES-ROUND-3-SPEC.md "B3/B4-rev", findings R1-R8).

* R1: ``turn_timeout_s`` is a deadline for the WHOLE request (start, turn, close): a server that
  trickles bytes (or answers late) ends with "HTTP agent did not answer within {seconds:g} s (POST {url})."
  within about ``turn_timeout_s``.
* R2 + R6: ``agent.http.headers`` values must be printable ASCII (0x20..0x7E): ``ConfigError`` at
  ``agent.http.headers.{name}`` from ``load_config``, the value never shown. The bearer token is stripped,
  then checked in ``start()``: ``ConfigError`` at ``agent.http.api_key_env`` naming the variable only.
* R3: ``start()`` warms up anyio's async backend, so the first timed turn does not pay its first use.
* R4: with ``api_key_env``, the URL's userinfo is not used (the request goes to the URL without it and
  carries the Bearer header); without it, userinfo is Basic auth; error texts never show userinfo.
* R5: ``httpx.InvalidURL`` and any other unmapped httpx error -> "could not reach the HTTP agent at
  {url}: {error}".
* R7: the run-wide redactor gets the bearer token and only credential-like header values.
* R8: a custom adapter constructor raising ``SystemExit`` -> "could not create {module}:{attr}: SystemExit: ..."

No network: ``httpx.MockTransport`` only. Seams: tests/data/fixes_round3/NOTES_stageB_adapters.md, "B3/B4-rev".
"""

import asyncio
import base64
import importlib
import json
import time
import uuid
from collections import deque

import pytest

BASE = "http://localhost:8080"
TURN_URL = f"{BASE}/turn"
START_URL = f"{BASE}/start"
CLOSE_URL = f"{BASE}/close"
SCENARIO = "http_rev"
OPENING = "Hello, I would like to book a slot on Tuesday."

TOKEN_ENV = "IMPULSE_TEST_HTTP_REV_TOKEN"     # not a *_API_KEY name: only the config names it
TOKEN = "tok-impulse-rev-SECRET-7731abcdef"

TIMEOUT = "HTTP agent did not answer within {seconds} s (POST {url})."
UNREACHABLE = "could not reach the HTTP agent at {url}: {error}"
NON_2XX = "HTTP agent returned HTTP {status} for POST {url}: {body}"
HEADER_ERROR = "header values must be printable ASCII without control characters"
TOKEN_ERROR = "the value of {name} must be printable ASCII without control characters"
COULD_NOT_CREATE = "could not create {module}:{attr}: {type}: {message}"


# ---------------------------------------------------------------------------------------------
# Fake HTTP agent (httpx.MockTransport)
# ---------------------------------------------------------------------------------------------


class FakeAgent:
    """``/start``, ``/turn``, ``/close`` behind an ``httpx.MockTransport``; ``answers[path]`` is a deque of
    dicts (sent as JSON), ``httpx.Response`` objects, exceptions (raised) or async callables
    ``request -> answer``; default: ``{"text": "ok"}`` for /turn, ``{}`` otherwise."""

    def __init__(self):
        import httpx

        self.requests = []
        self.answers = {"/start": deque(), "/turn": deque(), "/close": deque()}
        self.transport = httpx.MockTransport(self._handle)

    def to(self, path):
        return [record for record in self.requests if record["path"] == path]

    async def _handle(self, request):
        import httpx

        self.requests.append(
            {"url": str(request.url), "path": request.url.path, "headers": dict(request.headers)}
        )
        queue = self.answers.get(request.url.path)
        if queue is None:
            return httpx.Response(404, text="no route")
        answer = queue.popleft() if queue else ({"text": "ok"} if request.url.path == "/turn" else {})
        if callable(answer) and not isinstance(answer, httpx.Response):
            answer = await answer(request)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)


@pytest.fixture
def agent():
    return FakeAgent()


def _scenario(project, name=SCENARIO, **fields):
    values = {"opening_user_message": OPENING, "max_turns": 1, "metrics": ["UNIV-001", "UNIV-023"]}
    values.update(fields)
    return project.scenario(name, **values)


def _write_config(project, http=None, **kwargs):
    block = {"url": TURN_URL}
    block.update(http or {})
    return project.write_config([_scenario(project)], agent={"adapter": "http", "http": block}, **kwargs)


def _adapter(cfg, agent):
    from impulse_evals.adapters.http_adapter import HTTPAdapter

    return HTTPAdapter(cfg, transport=agent.transport)


async def _started(project, agent, http=None):
    _write_config(project, http)
    cfg = project.load()
    adapter = _adapter(cfg, agent)
    await adapter.start(cfg.scenarios[0])
    return adapter, cfg


# =============================================================================================
# R1: turn_timeout_s is a deadline for the whole request
# =============================================================================================

DEADLINE_S = 0.3
TRICKLE_CHUNKS = 30
TRICKLE_INTERVAL_S = 0.05     # 30 x 0.05 s = 1.5 s of trickling: five times the deadline
MAX_ELAPSED_S = 1.2           # the error must come well before the trickle ends


def _trickling_answer(body):
    """An answer whose body arrives one space every 50 ms, then ``body`` (valid JSON once complete)."""
    import httpx

    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(TRICKLE_CHUNKS):
                await asyncio.sleep(TRICKLE_INTERVAL_S)
                yield b" "
            yield json.dumps(body).encode("utf-8")

        async def aclose(self):
            return None

    async def answer(request):
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=Trickle())

    return answer


def _silent_answer(body):
    """An answer that sends nothing until it is 1.5 s late (no httpx timeout applies to a mock)."""

    async def answer(request):
        await asyncio.sleep(TRICKLE_CHUNKS * TRICKLE_INTERVAL_S)
        return body

    return answer


SLOW_ANSWERS = [pytest.param(_trickling_answer, id="trickle"), pytest.param(_silent_answer, id="silent")]


@pytest.mark.asyncio
@pytest.mark.parametrize("slow", SLOW_ANSWERS)
async def test_turn_deadline_covers_the_whole_request(impulse_project, agent, slow):
    adapter, _cfg = await _started(impulse_project, agent, {"turn_timeout_s": DEADLINE_S})
    agent.answers["/turn"].append(slow({"text": "too late"}))
    started = time.perf_counter()
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    elapsed = time.perf_counter() - started
    assert str(info.value) == TIMEOUT.format(seconds="0.3", url=TURN_URL)
    assert elapsed < MAX_ELAPSED_S, f"the turn took {elapsed:.2f} s with turn_timeout_s {DEADLINE_S}"
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("slow", SLOW_ANSWERS)
async def test_start_deadline_covers_the_whole_request(impulse_project, agent, slow):
    project = impulse_project
    _write_config(project, {"turn_timeout_s": DEADLINE_S, "start_url": START_URL, "close_url": CLOSE_URL})
    cfg = project.load()
    adapter = _adapter(cfg, agent)
    agent.answers["/start"].append(slow({"system_prompt": "late"}))
    started = time.perf_counter()
    with pytest.raises(RuntimeError) as info:
        await adapter.start(cfg.scenarios[0])
    elapsed = time.perf_counter() - started
    assert str(info.value) == TIMEOUT.format(seconds="0.3", url=START_URL)
    assert elapsed < MAX_ELAPSED_S, f"start() took {elapsed:.2f} s with turn_timeout_s {DEADLINE_S}"
    await adapter.close()
    assert agent.to("/close") == []          # a failed start sends no close (unchanged)


@pytest.mark.asyncio
@pytest.mark.parametrize("slow", SLOW_ANSWERS)
async def test_close_deadline_covers_the_whole_request(impulse_project, agent, slow):
    adapter, _cfg = await _started(impulse_project, agent, {"turn_timeout_s": DEADLINE_S, "close_url": CLOSE_URL})
    await adapter.send("Hi", None)
    agent.answers["/close"].append(slow({}))
    started = time.perf_counter()
    with pytest.raises(RuntimeError) as info:
        await adapter.close()
    elapsed = time.perf_counter() - started
    assert str(info.value) == TIMEOUT.format(seconds="0.3", url=CLOSE_URL)
    assert elapsed < MAX_ELAPSED_S, f"close() took {elapsed:.2f} s with turn_timeout_s {DEADLINE_S}"


@pytest.mark.asyncio
async def test_a_turn_inside_the_deadline_still_succeeds(impulse_project, agent):
    """A trickle that ends before the deadline is a normal answer."""
    import httpx

    class Quick(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(3):
                await asyncio.sleep(0.01)
                yield b" "
            yield b'{"text": "in time"}'

    async def answer(request):
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=Quick())

    adapter, _cfg = await _started(impulse_project, agent, {"turn_timeout_s": 2})
    agent.answers["/turn"].append(answer)
    reply = await adapter.send("Hi", None)
    assert reply.text == "in time"
    await adapter.close()


# =============================================================================================
# R2 + R6: header values and the bearer token are printable ASCII
# =============================================================================================

SECRET_HEAD = "hdr-SECRET-head-5521"
SECRET_TAIL = "hdr-SECRET-tail-8834"

BAD_CHARACTERS = [
    pytest.param("\n", id="newline"),
    pytest.param("\r", id="carriage-return"),
    pytest.param("\r\n", id="crlf"),
    pytest.param("\x00", id="nul"),
    pytest.param("\x7f", id="del"),
    pytest.param("\t", id="tab"),
    pytest.param("\x1b", id="escape"),
    pytest.param("é", id="non-ascii-e-acute"),
    pytest.param("न", id="non-ascii-devanagari"),
]


@pytest.mark.parametrize("bad", BAD_CHARACTERS)
def test_header_value_with_a_control_or_non_ascii_character_is_a_config_error(impulse_project, bad):
    from impulse_evals.config import ConfigError

    project = impulse_project
    value = SECRET_HEAD + bad + SECRET_TAIL
    _write_config(project, {"headers": {"X-Tenant": "clinic-7", "X-Key": value}})
    with pytest.raises(ConfigError) as info:
        project.load()
    error = info.value
    assert error.field == "agent.http.headers.X-Key"
    assert error.message == HEADER_ERROR
    for shown in (str(error), repr(error), error.message):
        assert SECRET_HEAD not in shown and SECRET_TAIL not in shown, shown


def test_header_value_with_a_trailing_newline_is_a_config_error(impulse_project):
    """A YAML block scalar (``|``) ends with a newline: rejected, not silently sent."""
    from impulse_evals.config import ConfigError

    project = impulse_project
    _write_config(project, {"headers": {"X-Api-Key": SECRET_HEAD + "\n"}})
    with pytest.raises(ConfigError) as info:
        project.load()
    assert info.value.field == "agent.http.headers.X-Api-Key"
    assert info.value.message == HEADER_ERROR
    assert SECRET_HEAD not in str(info.value)


def test_printable_ascii_header_values_are_accepted(impulse_project):
    project = impulse_project
    value = "Basic abc== ~!@#$%^&*()_+{}|:<>?`-=[]\\;',./ \"q\""   # every character in 0x20..0x7E
    _write_config(project, {"headers": {"X-Tenant": "clinic-7", "X-Odd": value, "X-Empty": ""}})
    http = project.load().agent.http
    assert http.headers == {"X-Tenant": "clinic-7", "X-Odd": value, "X-Empty": ""}


def _set_token(project, monkeypatch, value, source):
    if source == "env":
        monkeypatch.setenv(TOKEN_ENV, value)
    else:
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        escaped = json.dumps(value)        # a TOML basic string: \n, \r, \t, \u00XX escapes
        project.write_toml(f"[keys]\n{TOKEN_ENV} = {escaped}\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["env", "toml"])
@pytest.mark.parametrize("padding", [("  ", "  "), ("", "\n"), ("\t", "\r\n"), (" ", " \n")],
                         ids=["spaces", "newline", "tab-crlf", "space-newline"])
async def test_bearer_token_is_stripped_of_surrounding_whitespace(impulse_project, agent, monkeypatch, source, padding):
    project = impulse_project
    _set_token(project, monkeypatch, padding[0] + TOKEN + padding[1], source)
    adapter, _cfg = await _started(
        project, agent, {"api_key_env": TOKEN_ENV, "start_url": START_URL, "close_url": CLOSE_URL}
    )
    await adapter.send("Hi", None)
    await adapter.close()
    assert [record["path"] for record in agent.requests] == ["/start", "/turn", "/close"]
    for record in agent.requests:
        assert record["headers"].get("authorization") == f"Bearer {TOKEN}", record["headers"].get("authorization")


INNER_BAD = [
    pytest.param("\n", "env", id="newline-env"),
    pytest.param("\r", "env", id="cr-env"),
    pytest.param("\x7f", "env", id="del-env"),
    pytest.param("\x1b", "env", id="escape-env"),
    pytest.param("é", "env", id="non-ascii-env"),
    pytest.param("\x00", "toml", id="nul-toml"),         # an environment variable cannot hold NUL
    pytest.param("\n", "toml", id="newline-toml"),
    pytest.param("é", "toml", id="non-ascii-toml"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad, source", INNER_BAD)
async def test_bearer_token_with_a_control_or_non_ascii_character_is_a_config_error(
    impulse_project, agent, monkeypatch, bad, source
):
    from impulse_evals.config import ConfigError

    project = impulse_project
    head, tail = "tok-SECRET-head-4410", "tok-SECRET-tail-9087"
    _set_token(project, monkeypatch, head + bad + tail, source)
    _write_config(project, {"api_key_env": TOKEN_ENV, "start_url": START_URL})
    cfg = project.load()
    adapter = _adapter(cfg, agent)
    with pytest.raises(ConfigError) as info:
        await adapter.start(cfg.scenarios[0])
    error = info.value
    assert error.field == "agent.http.api_key_env"
    assert error.message == TOKEN_ERROR.format(name=TOKEN_ENV)
    for shown in (str(error), repr(error)):
        assert head not in shown and tail not in shown, shown
    assert agent.requests == []          # raised before any request
    await adapter.close()


# =============================================================================================
# R3: start() warms up the async backend before any turn is timed
# =============================================================================================

SLOW_FIRST_USE_S = 0.5


@pytest.mark.asyncio
async def test_first_turn_latency_does_not_include_the_async_backend_first_use(impulse_project, monkeypatch):
    """Stage B seam: the slow first use is anyio loading its asyncio backend
    (``anyio._core._eventloop.get_async_backend`` -> ``import_module("anyio._backends._asyncio")``,
    which httpcore triggers on the first real request). The test forgets the loaded backend, makes its
    load take 0.5 s, and uses a transport that touches anyio's backend on every request (as httpcore
    does). ``start()`` must load the backend (e.g. ``await anyio.sleep(0)``), so the first turn is fast."""
    import anyio
    import anyio._core._eventloop as eventloop
    import httpx

    from impulse_evals.adapters.http_adapter import HTTPAdapter

    real_import = eventloop.import_module
    loads = []

    def slow_import(name, *args, **kwargs):
        if name.startswith("anyio._backends."):
            loads.append(name)
            time.sleep(SLOW_FIRST_USE_S)   # an import blocks the loop, as the real first use does
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(eventloop.loaded_backends, "asyncio", raising=False)
    monkeypatch.setattr(eventloop, "import_module", slow_import)

    async def handle(request):
        await anyio.sleep(0)               # the transport uses anyio's backend, like httpcore
        return httpx.Response(200, json={"text": "ok"})

    project = impulse_project
    _write_config(project)
    cfg = project.load()
    adapter = HTTPAdapter(cfg, transport=httpx.MockTransport(handle))
    await adapter.start(cfg.scenarios[0])
    assert "asyncio" in eventloop.loaded_backends, "start() did not load anyio's async backend"
    reply = await adapter.send("Hi", None)
    await adapter.close()
    assert reply.text == "ok"
    assert reply.latency_ms < SLOW_FIRST_USE_S * 1000.0 / 2, (
        f"first turn latency {reply.latency_ms:.1f} ms includes the backend's first use"
    )
    assert loads, "the patched backend load was never used (seam not exercised)"


# =============================================================================================
# R4: URL userinfo and api_key_env
# =============================================================================================

USERINFO = "agentuser:agentpass-SECRET-31"
BASIC = "Basic " + base64.b64encode(USERINFO.encode("ascii")).decode("ascii")


def _userinfo_block(**extra):
    block = {
        "url": f"http://{USERINFO}@localhost:8080/turn?lang=en",
        "start_url": f"http://{USERINFO}@localhost:8080/start",
        "close_url": f"http://{USERINFO}@localhost:8080/close",
    }
    block.update(extra)
    return block


@pytest.mark.asyncio
async def test_userinfo_is_not_used_when_api_key_env_is_set(impulse_project, agent, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    adapter, _cfg = await _started(impulse_project, agent, _userinfo_block(api_key_env=TOKEN_ENV))
    await adapter.send("Hi", None)
    await adapter.close()
    assert [record["url"] for record in agent.requests] == [
        "http://localhost:8080/start", "http://localhost:8080/turn?lang=en", "http://localhost:8080/close",
    ]
    for record in agent.requests:
        assert record["headers"].get("authorization") == f"Bearer {TOKEN}", record


@pytest.mark.asyncio
async def test_userinfo_is_basic_auth_without_api_key_env(impulse_project, agent):
    adapter, _cfg = await _started(impulse_project, agent, _userinfo_block())
    await adapter.send("Hi", None)
    await adapter.close()
    assert [record["path"] for record in agent.requests] == ["/start", "/turn", "/close"]
    for record in agent.requests:
        assert record["headers"].get("authorization") == BASIC, record


@pytest.mark.asyncio
@pytest.mark.parametrize("with_key", [True, False], ids=["api_key_env", "basic"])
async def test_error_texts_never_show_userinfo(impulse_project, agent, monkeypatch, with_key):
    import httpx

    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    block = _userinfo_block(api_key_env=TOKEN_ENV) if with_key else _userinfo_block()
    adapter, _cfg = await _started(impulse_project, agent, block)
    agent.answers["/turn"].extend([
        httpx.Response(500, text="denied for this caller"),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("timed out"),
    ])
    messages = []
    for _ in range(3):
        with pytest.raises(RuntimeError) as info:
            await adapter.send("Hi", None)
        messages.append(str(info.value))
    shown_url = "http://localhost:8080/turn?lang=en"
    assert messages[0].startswith(NON_2XX.format(status=500, url=shown_url, body="denied for ")), messages[0]
    assert messages[1] == UNREACHABLE.format(url=shown_url, error="ConnectError: refused")
    assert messages[2] == TIMEOUT.format(seconds="30", url=shown_url)
    for message in messages:
        assert "agentpass" not in message and "agentuser" not in message, message
    await adapter.close()


# =============================================================================================
# R5: httpx.InvalidURL and other unmapped httpx errors
# =============================================================================================


@pytest.mark.asyncio
async def test_invalid_url_is_the_could_not_reach_text(impulse_project, agent):
    project = impulse_project
    url = "http://[::1/turn"            # passes load_config (prefix rule), httpx cannot parse it
    _write_config(project, {"url": url})
    cfg = project.load()
    adapter = _adapter(cfg, agent)
    with pytest.raises(RuntimeError) as info:   # raised by start() or send(), either is fine
        await adapter.start(cfg.scenarios[0])
        await adapter.send("Hi", None)
    message = str(info.value)
    assert message.startswith(UNREACHABLE.format(url=url, error="InvalidURL: ")), message
    assert "\n" not in message
    await adapter.close()


@pytest.mark.asyncio
async def test_invalid_url_with_userinfo_never_shows_it(impulse_project, agent):
    project = impulse_project
    url = "http://user:s3cretpass-SECRET@[::1/turn"
    _write_config(project, {"url": url})
    cfg = project.load()
    adapter = _adapter(cfg, agent)
    with pytest.raises(RuntimeError) as info:
        await adapter.start(cfg.scenarios[0])
        await adapter.send("Hi", None)
    message = str(info.value)
    assert message.startswith("could not reach the HTTP agent at "), message
    assert "InvalidURL: " in message, message
    assert "s3cretpass" not in message, message
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make, error",
    [
        pytest.param(lambda: __import__("httpx").CookieConflict("cookie conflict"), "CookieConflict: cookie conflict",
                     id="cookie-conflict"),
        pytest.param(lambda: __import__("httpx").InvalidURL("bad url from transport"), "InvalidURL: bad url from transport",
                     id="invalid-url-from-transport"),
    ],
)
async def test_other_httpx_errors_are_the_could_not_reach_text(impulse_project, agent, make, error):
    adapter, _cfg = await _started(impulse_project, agent)
    agent.answers["/turn"].append(make())
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    assert str(info.value) == UNREACHABLE.format(url=TURN_URL, error=error)
    await adapter.close()


# =============================================================================================
# R7: the run-wide redactor holds the token and only credential-like header values
# =============================================================================================

TENANT = "acme-clinic-tenant-42"
USER_AGENT = "impulse-test-client/1.0"
SENSITIVE_HEADERS = {
    "Authorization": "Basic cmV2LWF1dGgtU0VDUkVULTAx",
    "Proxy-Authorization": "Basic cmV2LXByb3h5LVNFQ1JFVC0wMg==",
    "Cookie": "session=rev-cookie-SECRET-03",
    "X-Api-Key": "rev-apikey-SECRET-04",
    "X-AUTH-TOKEN": "rev-token-SECRET-05",
    "x-client-secret": "rev-secret-SECRET-06",
    "X-Db-Password": "rev-password-SECRET-07",
    "Keycloak-Realm": "rev-keyname-SECRET-08",      # the name contains "key"
}
PUBLIC_HEADERS = {"X-Tenant": TENANT, "User-Agent": USER_AGENT, "Accept": "application/json"}


def test_run_redactor_holds_the_token_and_only_credential_like_header_values(impulse_project, monkeypatch):
    from impulse_evals.runner import config_redactor

    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    project = impulse_project
    _write_config(project, {"api_key_env": TOKEN_ENV, "headers": {**PUBLIC_HEADERS, **SENSITIVE_HEADERS}})
    cfg = project.load()
    redact = config_redactor(cfg, None)
    values = [TOKEN, *SENSITIVE_HEADERS.values(), TENANT, USER_AGENT, "application/json"]
    text = redact(" | ".join(values))
    for secret in [TOKEN, *SENSITIVE_HEADERS.values()]:
        assert secret not in text, (secret, text)
    for public in (TENANT, USER_AGENT, "application/json"):
        assert public in text, (public, text)


@pytest.mark.asyncio
async def test_tool_result_keeps_a_public_header_value_and_loses_secrets(impulse_project, agent, llm_router, monkeypatch):
    from impulse_evals.adapters.http_adapter import HTTPAdapter
    from impulse_evals.runner import run

    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    project = impulse_project
    _write_config(project, {"api_key_env": TOKEN_ENV,
                            "headers": {"X-Tenant": TENANT, "X-Api-Key": SENSITIVE_HEADERS["X-Api-Key"]}})
    agent.answers["/turn"].append({
        "text": "Booked.",
        "tool_calls": [{
            "name": "book_slot",
            "arguments": {"date": "2026-01-01"},
            "result": {"ok": True, "tenant": TENANT, "echo_key": SENSITIVE_HEADERS["X-Api-Key"], "echo_token": TOKEN},
        }],
    })

    def factory(config, *args, **kwargs):
        return HTTPAdapter(config, transport=agent.transport)

    result = await run(project.load(), adapter_factory=factory)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    (call,) = scenario_result.transcript.turns[1].tool_calls
    stored = json.dumps(call.result)
    assert call.result["tenant"] == TENANT, stored
    assert SENSITIVE_HEADERS["X-Api-Key"] not in stored and TOKEN not in stored, stored


@pytest.mark.asyncio
async def test_adapter_error_texts_still_redact_every_header_value(impulse_project, agent):
    """Unchanged: the adapter's own error texts redact all header values it holds, public ones too."""
    import httpx

    adapter, _cfg = await _started(impulse_project, agent, {"headers": {"X-Tenant": TENANT}})
    agent.answers["/turn"].append(httpx.Response(500, text=f"unknown tenant {TENANT}"))
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    assert TENANT not in str(info.value), str(info.value)
    assert str(info.value).startswith(NON_2XX.format(status=500, url=TURN_URL, body="unknown tenant ")), str(info.value)
    await adapter.close()


# =============================================================================================
# R8: a custom adapter constructor raising SystemExit
# =============================================================================================


def _custom_module(project, body):
    stem = f"impulse_test_custom_rev_{uuid.uuid4().hex[:10]}"
    code = f'''# SPDX-License-Identifier: Apache-2.0
from impulse_evals.adapters.base import AgentInterface


class Quits(AgentInterface):
    def __init__(self, config):
        {body}

    async def start(self, scenario):
        return None

    async def send(self, user_text, user_audio):
        return None

    async def close(self):
        return None
'''
    (project.root / f"{stem}.py").write_text(code, encoding="utf-8")
    importlib.invalidate_caches()
    project.write_config([_scenario(project)], agent={"adapter": "custom", "entrypoint": f"{stem}:Quits"})
    return stem, project.load()


def _create(how, cfg):
    if how == "create_adapter":
        from impulse_evals.adapters.custom_adapter import create_adapter

        return create_adapter(cfg)
    from impulse_evals.runner import default_adapter_factory

    return default_adapter_factory(cfg, redact=lambda text: text)


@pytest.mark.parametrize("how", ["create_adapter", "default_adapter_factory"])
@pytest.mark.parametrize(
    "body, message",
    [("raise SystemExit(3)", "3"), ('raise SystemExit("adapter gave up")', "adapter gave up")],
    ids=["code", "text"],
)
def test_system_exit_in_the_constructor_is_could_not_create(impulse_project, how, body, message):
    from impulse_evals.config import ConfigError

    stem, cfg = _custom_module(impulse_project, body)
    try:
        _create(how, cfg)
    except ConfigError as exc:
        error = exc
    except SystemExit as exc:   # today: the run would end with this exit code
        pytest.fail(f"SystemExit({exc.code!r}) escaped from the constructor instead of a ConfigError", pytrace=False)
    else:
        pytest.fail("no error raised")
    assert error.field == "agent.entrypoint"
    assert error.message == COULD_NOT_CREATE.format(module=stem, attr="Quits", type="SystemExit", message=message)


def test_keyboard_interrupt_in_the_constructor_is_not_converted(impulse_project):
    from impulse_evals.adapters.custom_adapter import create_adapter

    _stem, cfg = _custom_module(impulse_project, "raise KeyboardInterrupt()")
    with pytest.raises(KeyboardInterrupt):
        create_adapter(cfg)
