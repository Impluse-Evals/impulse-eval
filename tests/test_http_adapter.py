# SPDX-License-Identifier: Apache-2.0
"""Fixes round 3, B4: the HTTP adapter, ``agent.adapter: http`` (FIXES-ROUND-3-SPEC.md B4).

* config ``agent.http`` (kept only when ``adapter: http``): ``url`` (required, http:// or https://),
  ``start_url`` / ``close_url`` (optional, same scheme rule), ``headers``, ``api_key_env``,
  ``turn_timeout_s`` (the LiveKit / Pipecat rule, field ``agent.http.turn_timeout_s``).
* ``impulse_evals.adapters.http_adapter.HTTPAdapter(config, *, transport=None, redact=None)`` with one
  ``httpx.AsyncClient`` per scenario. Wire format: POST, ``Content-Type: application/json``, the
  ``headers``, ``Authorization: Bearer <value>`` when ``api_key_env`` is set; start / turn / close
  bodies exactly as the spec; ``conversation_id`` = ``"impulse-"`` + 12 lowercase hex, new per
  ``start()``; latency measured around the turn request; no retries.
* turn answer: ``text`` (missing / null -> ""), ``tool_calls`` (generic-adapter rules, "HTTP agent"
  source label), ``heard_text``, ``audio`` (base64 WAV, standard or URL-safe, padding optional).
* errors (URL without credentials; body / error redacted with the run redactor, the bearer token and
  the header values BEFORE one line + 200 characters; "(empty body)"): non-2xx, timeout, connection
  error, non-object body, field errors; a missing key is a ``ConfigError`` raised in ``start()``.
* runner / CLI: default factory ``HTTPAdapter(config, redact=redact)``; ``agent_entrypoint`` = the URL
  without credentials; doctor rows "agent http" (plain TCP connect, no HTTP request), "HTTP agent
  key", INFO "agent connection"; ``--dry-run`` sends no request; ``--fail-on`` / ``compare`` unchanged.

No network: adapters get an ``httpx.MockTransport`` (``FakeHTTPAgent``); runs through the default
factory and the CLI route every ``httpx.AsyncClient`` created without a transport to the same mock
(as ``mock_voice_server`` does); the doctor's TCP connect is replaced by patching
``asyncio.open_connection`` and ``socket.create_connection`` (stage B seam, see
tests/data/fixes_round3/NOTES_stageB_adapters.md). The judge and simulator are ``llm_router``.
"""

import asyncio
import base64
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import pcm_tone, sine_wav, wav_bytes

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "http"

BASE = "http://localhost:8080"
TURN_URL = f"{BASE}/turn"
START_URL = f"{BASE}/start"
CLOSE_URL = f"{BASE}/close"
CRED_TURN_URL = "http://agentuser:agentpass@localhost:8080/turn?token=abc"

SCENARIO = "http_booking"
OPENING = "Hello, I would like to book a slot on Tuesday."
SECOND_USER = "Thank you, that is all for today."
OPENER = "Hello, this is the HTTP agent. How can I help?"
BOOKING_ARGUMENTS = {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}
CONSENT = "dpdp.consent_before_pii"
PURPOSE = "dpdp.purpose_disclosure"
CONVERSATION_ID = re.compile(r"impulse-[0-9a-f]{12}")

TOKEN_ENV = "IMPULSE_TEST_HTTP_TOKEN"          # not a *_API_KEY name: only the adapter knows its value
TOKEN = "tok-impulse-test-SECRET-http-4417abcdef"
HEADER_SECRET = "hdr-impulse-test-SECRET-9921abcdef"
KNOWN_KEY = "sk-impulse-test-SECRET-known-5150abcdef"
RUN_KEY_ENV = "IMPULSE_TEST_HTTP_API_KEY"      # a *_API_KEY name: the run's redactor knows its value
RUN_KEY = "sk-impulse-test-SECRET-runkey-8086abcdef"

MISSING_KEY = "{name} is not set: set the environment variable or add it to [keys] in .impulse/config.toml"
NON_2XX = "HTTP agent returned HTTP {status} for POST {url}: {body}"
TIMEOUT = "HTTP agent did not answer within {seconds} s (POST {url})."
UNREACHABLE = "could not reach the HTTP agent at {url}: {error}"
NOT_OBJECT = "HTTP agent returned a body that is not a JSON object (POST {url}): {body}"
DRY_RUN_WARNING = "Dry run: judge metrics and assertions were skipped; results are not verdicts."
REMOTE_WARNING = (
    "Dry run: scenarios run only with the generic adapter; the {adapter} config was validated without connecting."
)
VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}


# ---------------------------------------------------------------------------------------------
# Fake HTTP agent (httpx.MockTransport, no socket)
# ---------------------------------------------------------------------------------------------


@dataclass
class Raise:
    """An answer that makes the transport raise ``exc_type(message, request=request)``."""

    exc_type: type
    message: str


class FakeHTTPAgent:
    """The three endpoints (``/start``, ``/turn``, ``/close``) behind an ``httpx.MockTransport``.

    Script answers with ``turn_answers`` / ``start_answers`` / ``close_answers`` (deques; each item a
    dict sent as JSON, an ``httpx.Response``, a ``Raise``, or a callable ``body -> answer``); when
    empty, ``/turn`` echoes (``Echo: {user_text}``, the opener for ``user_text: null``, a ``book_slot``
    call with result ``{"ok": true}`` when the text says "book", ``heard_text`` = the text lowercased),
    and ``/start`` / ``/close`` answer ``{}``. Every request is recorded in ``requests``.
    """

    def __init__(self):
        import httpx

        self.requests = []
        self.turn_answers = deque()
        self.start_answers = deque()
        self.close_answers = deque()
        self.delay_s = 0.0
        self.transport = httpx.MockTransport(self._handle)

    @staticmethod
    def echo(body):
        user_text = body.get("user_text") if isinstance(body, dict) else None
        if user_text is None:
            return {"text": OPENER}
        answer = {"text": f"Echo: {user_text}", "heard_text": user_text.lower()}
        if "book" in user_text.lower():
            answer["tool_calls"] = [{"name": "book_slot", "arguments": dict(BOOKING_ARGUMENTS), "result": {"ok": True}}]
        return answer

    def to(self, path):
        return [record for record in self.requests if record["path"] == path]

    async def _handle(self, request):
        import httpx

        raw = request.content
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            body = None
        self.requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "headers": dict(request.headers),
                "json": body,
                "content": raw,
                "timeout": request.extensions.get("timeout"),
            }
        )
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        path = request.url.path
        if path == "/turn":
            answer = self.turn_answers.popleft() if self.turn_answers else self.echo
        elif path == "/start":
            answer = self.start_answers.popleft() if self.start_answers else {}
        elif path == "/close":
            answer = self.close_answers.popleft() if self.close_answers else {}
        else:
            return httpx.Response(404, text=f"fake HTTP agent: no route for {path}")
        if callable(answer) and not isinstance(answer, (httpx.Response, Raise)):
            answer = answer(body)
        if isinstance(answer, Raise):
            raise answer.exc_type(answer.message, request=request)
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)


@pytest.fixture
def http_agent():
    return FakeHTTPAgent()


def _response(status, text=None, *, content=None, content_type="text/plain"):
    import httpx

    data = content if content is not None else (text or "").encode("utf-8")
    return httpx.Response(status, content=data, headers={"content-type": content_type})


def _route_httpx(monkeypatch, agent):
    """Every ``httpx.AsyncClient`` created without an explicit transport uses the fake agent."""
    import httpx

    real_init = httpx.AsyncClient.__init__

    def init(client, *args, **kwargs):
        if kwargs.get("transport") is None:
            kwargs["transport"] = agent.transport
        real_init(client, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)


# ---------------------------------------------------------------------------------------------
# config / adapter helpers
# ---------------------------------------------------------------------------------------------


def _scenario(project, name=SCENARIO, **fields):
    values = {"opening_user_message": OPENING, "max_turns": 2, "metrics": ["UNIV-001", "UNIV-023"]}
    values.update(fields)
    return project.scenario(name, **values)


def _write_config(project, *, http=None, with_block=True, scenarios=None, **kwargs):
    agent = {"adapter": "http"}
    if with_block:
        block = {"url": TURN_URL}
        block.update(http or {})
        agent["http"] = block
    return project.write_config(scenarios or [_scenario(project)], agent=agent, **kwargs)


def _fail_if_adapter_rejected(exc):
    """Today ``adapter: http`` is not an accepted literal: say so clearly instead of a raw error."""
    if getattr(exc, "field", None) == "agent.adapter":
        pytest.fail(f"config.yaml agent.adapter: http is not accepted: {exc}", pytrace=False)


def _load(project, path=None):
    from impulse_evals.config import ConfigError

    try:
        return project.load(path)
    except ConfigError as exc:
        _fail_if_adapter_rejected(exc)
        raise


def _config_error(project):
    from impulse_evals.config import ConfigError

    with pytest.raises(ConfigError) as info:
        project.load()
    _fail_if_adapter_rejected(info.value)
    return info.value


def _adapter(cfg, agent, redact=None):
    from impulse_evals.adapters.http_adapter import HTTPAdapter

    if redact is None:
        return HTTPAdapter(cfg, transport=agent.transport)
    return HTTPAdapter(cfg, transport=agent.transport, redact=redact)


async def _started(project, agent, *, http=None, redact=None, scenarios=None, **kwargs):
    _write_config(project, http=http, scenarios=scenarios, **kwargs)
    cfg = _load(project)
    adapter = _adapter(cfg, agent, redact=redact)
    await adapter.start(cfg.scenarios[0])
    return adapter, cfg


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


def _factory(agent):
    def factory(config, *args, **kwargs):
        from impulse_evals.adapters.http_adapter import HTTPAdapter

        return HTTPAdapter(config, transport=agent.transport)

    return factory


def _odd_wav():
    """A WAV whose base64 needs padding and differs between the standard and URL-safe alphabets."""
    seconds = 0.0501
    while True:
        wav = sine_wav(seconds)
        standard = base64.b64encode(wav).decode("ascii")
        if len(wav) % 3 and ("+" in standard or "/" in standard):
            return wav
        seconds += 0.0001


def _rows(result, status, check):
    """Doctor table rows with ``status`` and the check label ``check``: their detail texts."""
    details = []
    for line in result.output.splitlines():
        line = line.strip()
        if not line.startswith(status):
            continue
        rest = line[len(status):].strip()
        if rest == check or rest.startswith(check + " "):
            details.append(rest[len(check):].strip())
    return details


def _new_run_dir(project, before):
    (run_dir,) = set(project.run_dirs()) - set(before)
    return run_dir


def _replace_raw(path, placeholder, raw):
    text = path.read_text(encoding="utf-8")
    assert text.count(placeholder) == 1, text
    path.write_text(text.replace(placeholder, raw), encoding="utf-8")


# =============================================================================================
# config.yaml: agent.http
# =============================================================================================


def test_http_block_is_accepted_with_its_defaults(impulse_project):
    project = impulse_project
    _write_config(project)
    cfg = _load(project)
    assert cfg.agent.adapter == "http"
    http = cfg.agent.http
    assert http.url == TURN_URL
    assert http.start_url is None and http.close_url is None
    assert http.headers == {}
    assert http.api_key_env is None
    assert http.turn_timeout_s is None


def test_full_http_block_is_kept(impulse_project):
    project = impulse_project
    block = {
        "url": "https://agent.localhost/turn",
        "start_url": "https://agent.localhost/start",
        "close_url": "http://agent.localhost/close",
        "headers": {"X-Tenant": "clinic-7"},
        "api_key_env": TOKEN_ENV,
        "turn_timeout_s": 12.5,
    }
    _write_config(project, http=block)
    http = _load(project).agent.http
    assert (http.url, http.start_url, http.close_url) == (block["url"], block["start_url"], block["close_url"])
    assert http.headers == {"X-Tenant": "clinic-7"}
    assert http.api_key_env == TOKEN_ENV
    assert http.turn_timeout_s == 12.5


@pytest.mark.parametrize("case", ["no_block", "no_url", "empty_url"])
def test_http_url_is_required(impulse_project, case):
    project = impulse_project
    if case == "no_block":
        _write_config(project, with_block=False)
    elif case == "no_url":
        project.write_config([_scenario(project)], agent={"adapter": "http", "http": {"start_url": START_URL}})
    else:
        _write_config(project, http={"url": ""})
    error = _config_error(project)
    assert error.field == "agent.http.url"
    assert error.message == "required when agent.adapter is http"


@pytest.mark.parametrize("key", ["url", "start_url", "close_url"])
@pytest.mark.parametrize("value", ["ws://localhost:8080/x", "localhost:8080/x", "ftp://localhost/x"])
def test_http_urls_must_be_http_or_https(impulse_project, key, value):
    project = impulse_project
    _write_config(project, http={key: value})
    error = _config_error(project)
    assert error.field == f"agent.http.{key}"
    assert error.message == "must start with http:// or https://"


@pytest.mark.parametrize(
    "raw, message",
    [
        ("0", "turn_timeout_s 0 must be greater than 0"),
        ("-5", "turn_timeout_s -5 must be greater than 0"),
        (".nan", "turn_timeout_s nan must be greater than 0"),
        (".inf", "turn_timeout_s must be a finite number of seconds"),
        ("1e400", "turn_timeout_s must be a finite number of seconds"),
        ("true", "turn_timeout_s must be a number of seconds, not a boolean"),
        ("false", "turn_timeout_s must be a number of seconds, not a boolean"),
        ("soon", None),
    ],
)
def test_http_turn_timeout_s_rules(impulse_project, raw, message):
    """The LiveKit / Pipecat rule (one shared helper), on the field ``agent.http.turn_timeout_s``."""
    project = impulse_project
    placeholder = "__TURN_TIMEOUT_S__"
    path = _write_config(project, http={"turn_timeout_s": placeholder})
    _replace_raw(path, f"turn_timeout_s: {placeholder}", f"turn_timeout_s: {raw}")
    error = _config_error(project)
    assert error.field == "agent.http.turn_timeout_s"
    if message is not None:
        assert error.message == message
    assert "Value error, " not in str(error)


@pytest.mark.parametrize("adapter", ["generic", "custom", "pipecat", "livekit"])
def test_http_block_is_dropped_for_other_adapters(impulse_project, adapter):
    project = impulse_project
    agent = {"adapter": adapter, "http": {"url": "ftp://not-checked/x", "turn_timeout_s": -1}}
    if adapter == "generic":
        agent["entrypoint"] = project.write_agent()
    elif adapter == "custom":
        agent["entrypoint"] = "some_adapter:SomeAdapter"   # never imported by load_config
    elif adapter == "pipecat":
        agent["pipecat"] = {"mode": "eval", "url": "ws://localhost:7860"}
    else:
        agent["livekit"] = {"url": "wss://livekit.localhost"}
    project.write_config([_scenario(project)], agent=agent)
    cfg = _load(project)
    assert cfg.agent.adapter == adapter
    assert getattr(cfg.agent, "http", "missing") is None, "AgentConfig.http must exist and be None here"


# =============================================================================================
# Wire format: bodies, headers, conversation_id
# =============================================================================================


def _content_type(record):
    return record["headers"].get("content-type", "").split(";")[0].strip()


@pytest.mark.asyncio
async def test_start_turn_and_close_bodies(impulse_project, http_agent):
    adapter, _cfg = await _started(
        impulse_project, http_agent, http={"start_url": START_URL, "close_url": CLOSE_URL}
    )
    await adapter.send("Hi there", None)
    await adapter.close()

    assert [record["path"] for record in http_agent.requests] == ["/start", "/turn", "/close"]
    assert all(record["method"] == "POST" for record in http_agent.requests)
    assert all(_content_type(record) == "application/json" for record in http_agent.requests)
    start, turn, close = (record["json"] for record in http_agent.requests)
    conversation_id = start["conversation_id"]
    assert CONVERSATION_ID.fullmatch(conversation_id), conversation_id
    assert start == {
        "conversation_id": conversation_id, "scenario_name": SCENARIO, "language": "en-IN", "direction": "inbound",
    }
    assert turn == {"conversation_id": conversation_id, "user_text": "Hi there", "history": [], "scenario_name": SCENARIO}
    assert close == {"conversation_id": conversation_id, "scenario_name": SCENARIO}
    assert http_agent.requests[0]["url"] == START_URL
    assert http_agent.requests[1]["url"] == TURN_URL
    assert http_agent.requests[2]["url"] == CLOSE_URL


@pytest.mark.asyncio
async def test_no_start_or_close_request_without_their_urls(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent)
    await adapter.send("Hi there", None)
    await adapter.close()
    assert [record["path"] for record in http_agent.requests] == ["/turn"]
    assert CONVERSATION_ID.fullmatch(http_agent.requests[0]["json"]["conversation_id"])


@pytest.mark.asyncio
async def test_start_body_of_an_outbound_scenario(impulse_project, http_agent):
    project = impulse_project
    scenario = _scenario(project, "http_outbound", direction="outbound", opening_user_message=None, language="es-MX")
    adapter, _cfg = await _started(project, http_agent, http={"start_url": START_URL}, scenarios=[scenario])
    start = http_agent.requests[0]["json"]
    assert start == {
        "conversation_id": start["conversation_id"], "scenario_name": "http_outbound", "language": "es-MX",
        "direction": "outbound",
    }
    await adapter.close()


@pytest.mark.asyncio
async def test_conversation_id_is_new_for_every_start(impulse_project, http_agent):
    adapter, cfg = await _started(impulse_project, http_agent, http={"close_url": CLOSE_URL})
    await adapter.send("First conversation", None)
    await adapter.close()
    await adapter.start(cfg.scenarios[0])          # a new conversation (and a new client) on the same adapter
    await adapter.send("Second conversation", None)
    await adapter.close()
    other = _adapter(cfg, http_agent)
    await other.start(cfg.scenarios[0])
    await other.send("Third conversation", None)
    await other.close()

    turns = http_agent.to("/turn")
    ids = [record["json"]["conversation_id"] for record in turns]
    assert all(CONVERSATION_ID.fullmatch(value) for value in ids), ids
    assert len(set(ids)) == 3, ids
    closes = [record["json"]["conversation_id"] for record in http_agent.to("/close")]
    assert closes[:2] == ids[:2]
    assert turns[1]["json"]["history"] == []      # the second conversation starts with an empty history


@pytest.mark.asyncio
async def test_history_grows_across_turns(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent)
    first = await adapter.send("First question", None)
    second = await adapter.send("Second question", None)
    await adapter.send("Third question", None)
    assert (first.text, second.text) == ("Echo: First question", "Echo: Second question")
    histories = [record["json"]["history"] for record in http_agent.to("/turn")]
    assert histories == [
        [],
        [{"role": "user", "content": "First question"}, {"role": "assistant", "content": "Echo: First question"}],
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Echo: First question"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Echo: Second question"},
        ],
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_outbound_opener_sends_null_user_text_and_no_user_history_entry(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent)
    opener = await adapter.send(None, None)
    assert opener.text == OPENER
    await adapter.send("Who is calling?", None)
    first, second = (record["json"] for record in http_agent.to("/turn"))
    assert first["user_text"] is None
    assert first["history"] == []
    assert "user_text" in json.loads(http_agent.to("/turn")[0]["content"].decode("utf-8"))   # null, not omitted
    assert second["user_text"] == "Who is calling?"
    assert second["history"] == [{"role": "assistant", "content": OPENER}]
    await adapter.close()


@pytest.mark.asyncio
async def test_body_is_utf8_json(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent)
    text = "नमस्ते, मुझे book करना है"
    await adapter.send(text, None)
    record = http_agent.to("/turn")[0]
    assert json.loads(record["content"].decode("utf-8"))["user_text"] == text
    await adapter.close()


@pytest.mark.asyncio
async def test_voice_mode_still_sends_text_only(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent, voice_config=VOICE_CONFIG)
    await adapter.send("Book me in", wav_bytes(pcm_tone(0.3, 16000)))
    body = http_agent.to("/turn")[0]["json"]
    assert set(body) == {"conversation_id", "user_text", "history", "scenario_name"}
    assert body["user_text"] == "Book me in"
    await adapter.close()


@pytest.mark.asyncio
async def test_custom_headers_and_bearer_token_from_the_environment(impulse_project, http_agent, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    adapter, _cfg = await _started(
        impulse_project,
        http_agent,
        http={
            "start_url": START_URL, "close_url": CLOSE_URL, "api_key_env": TOKEN_ENV,
            "headers": {"X-Tenant": "clinic-7", "X-Api-Secret": HEADER_SECRET},
        },
    )
    await adapter.send("Hi", None)
    await adapter.close()
    assert len(http_agent.requests) == 3
    for record in http_agent.requests:
        headers = record["headers"]
        assert headers.get("authorization") == f"Bearer {TOKEN}"
        assert headers.get("x-tenant") == "clinic-7"
        assert headers.get("x-api-secret") == HEADER_SECRET
        assert _content_type(record) == "application/json"


@pytest.mark.asyncio
async def test_bearer_token_from_config_toml_keys(impulse_project, http_agent, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    impulse_project.write_toml(f'[keys]\n{TOKEN_ENV} = "{TOKEN}"\n')
    adapter, _cfg = await _started(impulse_project, http_agent, http={"api_key_env": TOKEN_ENV})
    await adapter.send("Hi", None)
    await adapter.close()
    assert http_agent.to("/turn")[0]["headers"].get("authorization") == f"Bearer {TOKEN}"


@pytest.mark.asyncio
async def test_no_authorization_header_without_api_key_env(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent)
    await adapter.send("Hi", None)
    await adapter.close()
    assert "authorization" not in http_agent.to("/turn")[0]["headers"]


@pytest.mark.asyncio
async def test_missing_key_is_a_config_error_raised_in_start(impulse_project, http_agent, monkeypatch):
    from impulse_evals.config import ConfigError

    monkeypatch.delenv(TOKEN_ENV, raising=False)
    project = impulse_project
    _write_config(project, http={"api_key_env": TOKEN_ENV, "start_url": START_URL})
    cfg = _load(project)
    adapter = _adapter(cfg, http_agent)   # creating the adapter does not check the key
    with pytest.raises(ConfigError) as info:
        await adapter.start(cfg.scenarios[0])
    assert info.value.field == "agent.http.api_key_env"
    assert info.value.message == MISSING_KEY.format(name=TOKEN_ENV)
    assert http_agent.requests == []


@pytest.mark.asyncio
async def test_latency_is_measured_around_the_turn_request(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.delay_s = 0.08
    reply = await adapter.send("Hi", None)
    assert 70.0 <= reply.latency_ms < 5000.0, reply.latency_ms
    await adapter.close()


@pytest.mark.asyncio
async def test_turn_timeout_is_passed_to_httpx(impulse_project, http_agent):
    """Stage B seam: the per-request timeout is the httpx request timeout (read) of every request."""
    adapter, _cfg = await _started(impulse_project, http_agent, http={"turn_timeout_s": 12, "start_url": START_URL})
    await adapter.send("Hi", None)
    for record in http_agent.requests:
        assert record["timeout"] is not None and record["timeout"].get("read") == pytest.approx(12.0), record["timeout"]
    await adapter.close()


@pytest.mark.asyncio
async def test_no_retries(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append(_response(503, "busy"))
    with pytest.raises(RuntimeError):
        await adapter.send("Book a slot", None)
    assert len(http_agent.to("/turn")) == 1
    await adapter.close()


# =============================================================================================
# Turn answer parsing
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [{}, {"text": None}, {"tool_calls": None, "heard_text": None, "audio": None}])
async def test_missing_or_null_text_is_empty(impulse_project, http_agent, answer):
    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append(answer)
    reply = await adapter.send("Hi", None)
    assert reply.text == ""
    assert reply.tool_calls == []
    assert reply.heard_text is None
    assert reply.audio_bytes is None
    await adapter.close()


@pytest.mark.asyncio
async def test_answer_fields_are_parsed_and_other_keys_ignored(impulse_project, http_agent):
    from impulse_evals.models import ToolCall

    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append(
        {
            "text": "Booked for Tuesday.",
            "heard_text": "book me for tuesday",
            "tool_calls": [
                {"name": "book_slot", "arguments": dict(BOOKING_ARGUMENTS), "result": {"ok": True}},
                {"name": "send_sms"},
                {"name": "lookup", "arguments": {}, "result": "queued"},
                {"name": "charge_card", "arguments": {"amount": 10}, "success": False, "error": "card declined"},
            ],
            "session": {"ignored": True},
            "debug": "ignored",
        }
    )
    reply = await adapter.send("Book me for Tuesday", None)
    assert reply.text == "Booked for Tuesday."
    assert reply.heard_text == "book me for tuesday"
    assert reply.tool_calls == [
        ToolCall(name="book_slot", arguments=BOOKING_ARGUMENTS, result={"ok": True}),
        ToolCall(name="send_sms", arguments={}),
        ToolCall(name="lookup", arguments={}, result={"value": "queued"}),
        ToolCall(name="charge_card", arguments={"amount": 10}, success=False, error="card declined"),
    ]
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["standard", "urlsafe", "standard-unpadded", "urlsafe-unpadded"])
async def test_audio_is_base64_wav_in_either_alphabet_with_optional_padding(impulse_project, http_agent, encoding):
    wav = _odd_wav()
    encoded = (base64.urlsafe_b64encode if encoding.startswith("urlsafe") else base64.b64encode)(wav).decode("ascii")
    if encoding.endswith("unpadded"):
        encoded = encoded.rstrip("=")
    assert encoded != base64.b64encode(wav).decode("ascii") or encoding == "standard"   # the variants differ
    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append({"text": "Here is my answer.", "audio": encoded})
    reply = await adapter.send("Hi", None)
    assert reply.audio_bytes == wav
    await adapter.close()


AUDIO_ERROR = "HTTP agent returned 'audio' that is not base64"

FIELD_ERRORS = [
    pytest.param({"text": 42}, "HTTP agent returned a non-string 'text'", id="non-string-text"),
    pytest.param({"text": "ok", "heard_text": ["hi"]}, "HTTP agent returned a non-string 'heard_text'", id="non-string-heard"),
    pytest.param({"text": "ok", "audio": "@@ not base64 @@"}, AUDIO_ERROR, id="invalid-base64"),
    pytest.param({"text": "ok", "audio": 12345}, AUDIO_ERROR, id="non-string-audio"),
    pytest.param(
        {"text": "ok", "tool_calls": {"name": "book_slot"}},
        "HTTP agent returned 'tool_calls' that is not a list", id="tool-calls-not-list",
    ),
    pytest.param(
        {"text": "ok", "tool_calls": ["book_slot"]}, "HTTP agent returned a tool call that is not a dict",
        id="tool-call-not-dict",
    ),
    pytest.param(
        {"text": "ok", "tool_calls": [{"type": "function", "function": {"name": "book_slot", "arguments": "{}"}}]},
        "HTTP agent returned a tool call without a string 'name'", id="openai-style",
    ),
    pytest.param(
        {"text": "ok", "tool_calls": [{"name": "", "arguments": {}}]},
        "HTTP agent returned a tool call without a string 'name'", id="empty-name",
    ),
    pytest.param(
        {"text": "ok", "tool_calls": [{"name": "book_slot", "arguments": '{"date": "2026-01-01"}'}]},
        "HTTP agent returned a tool call whose 'arguments' is not a dict", id="json-string-arguments",
    ),
    pytest.param(
        {"text": "ok", "tool_calls": [{"name": "book_slot", "arguments": {}, "error": 123}]},
        "HTTP agent returned an invalid tool call: error: Input should be a valid string", id="non-string-error",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, expected", FIELD_ERRORS)
async def test_field_errors_are_type_errors(impulse_project, http_agent, answer, expected):
    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append(answer)
    with pytest.raises(TypeError) as info:
        await adapter.send("Hi", None)
    message = str(info.value)
    assert message == expected, message
    assert "errors.pydantic.dev" not in message
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, shown",
    [(b"[1, 2]", "[1, 2]"), (b'"just text"', '"just text"'), (b"not json at all", "not json at all"), (b"", "(empty body)")],
    ids=["list", "string", "invalid-json", "empty"],
)
async def test_turn_answer_that_is_not_a_json_object(impulse_project, http_agent, content, shown):
    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append(_response(200, content=content, content_type="application/json"))
    with pytest.raises(TypeError) as info:
        await adapter.send("Hi", None)
    assert str(info.value) == NOT_OBJECT.format(url=TURN_URL, body=shown)
    await adapter.close()


# =============================================================================================
# Errors: non-2xx, timeout, connection; URL credentials; redaction; one line; 200 characters
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("status, body, shown", [(500, "agent exploded", "agent exploded"), (404, "", "(empty body)"),
                                                 (401, "bad token", "bad token"), (429, "slow down", "slow down")])
async def test_non_2xx_answer(impulse_project, http_agent, status, body, shown):
    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append(_response(status, body))
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    assert str(info.value) == NON_2XX.format(status=status, url=TURN_URL, body=shown)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "http, voice, seconds",
    [({"turn_timeout_s": 12}, False, "12"), ({"turn_timeout_s": 2.5}, False, "2.5"), ({}, True, "7"),
     ({"turn_timeout_s": 4}, True, "4"), ({}, False, "30")],
    ids=["turn_timeout_s", "fraction", "voice_config", "turn_timeout_s-wins", "default-30"],
)
async def test_timeout_error(impulse_project, http_agent, http, voice, seconds):
    import httpx

    kwargs = {"voice_config": dict(VOICE_CONFIG, timeout_s=7)} if voice else {}
    adapter, _cfg = await _started(impulse_project, http_agent, http=http, **kwargs)
    http_agent.turn_answers.append(Raise(httpx.ReadTimeout, "timed out"))
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    assert str(info.value) == TIMEOUT.format(seconds=seconds, url=TURN_URL)
    await adapter.close()


@pytest.mark.asyncio
async def test_connection_error(impulse_project, http_agent):
    import httpx

    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append(Raise(httpx.ConnectError, "[Errno 111] Connection refused"))
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    assert str(info.value) == UNREACHABLE.format(url=TURN_URL, error="ConnectError: [Errno 111] Connection refused")
    await adapter.close()


@pytest.mark.asyncio
async def test_error_texts_show_the_url_without_credentials(impulse_project, http_agent):
    import httpx

    adapter, _cfg = await _started(impulse_project, http_agent, http={"url": CRED_TURN_URL})
    http_agent.turn_answers.extend([_response(500, "broken"), Raise(httpx.ReadTimeout, "timed out"),
                                    Raise(httpx.ConnectError, "refused")])
    messages = []
    for _ in range(3):
        with pytest.raises(RuntimeError) as info:
            await adapter.send("Hi", None)
        messages.append(str(info.value))
    assert messages == [
        NON_2XX.format(status=500, url=TURN_URL, body="broken"),
        TIMEOUT.format(seconds="30", url=TURN_URL),
        UNREACHABLE.format(url=TURN_URL, error="ConnectError: refused"),
    ]
    assert all("agentpass" not in message and "token=abc" not in message for message in messages)
    assert http_agent.to("/turn")[0]["path"] == "/turn"   # the request itself went to the full URL
    await adapter.close()


def _known_key_redactor(text):
    return text.replace(KNOWN_KEY, "[REDACTED]")


async def _secrets_adapter(project, agent, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    return await _started(
        project, agent, redact=_known_key_redactor,
        http={"api_key_env": TOKEN_ENV, "headers": {"X-Api-Secret": HEADER_SECRET}},
    )


@pytest.mark.asyncio
async def test_error_body_is_redacted_and_on_one_line(impulse_project, http_agent, monkeypatch):
    adapter, _cfg = await _secrets_adapter(impulse_project, http_agent, monkeypatch)
    http_agent.turn_answers.append(
        _response(500, f"bad auth: bearer {TOKEN}\nheader {HEADER_SECRET}\r\n\tknown {KNOWN_KEY} end")
    )
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    message = str(info.value)
    assert message.startswith(NON_2XX.format(status=500, url=TURN_URL, body="bad auth: bearer ")), message
    for secret in (TOKEN, HEADER_SECRET, KNOWN_KEY):
        assert secret not in message, message
    assert "[REDACTED]" in message, message
    assert "\n" not in message and "\r" not in message and "\t" not in message
    assert message.endswith(" end"), message
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", [TOKEN, HEADER_SECRET, KNOWN_KEY], ids=["bearer", "header", "known-key"])
async def test_error_body_is_redacted_before_it_is_cut(impulse_project, http_agent, monkeypatch, secret):
    adapter, _cfg = await _secrets_adapter(impulse_project, http_agent, monkeypatch)
    http_agent.turn_answers.append(_response(500, "y" * 185 + secret + "z" * 60))
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    assert secret[:8] not in str(info.value), str(info.value)
    await adapter.close()


@pytest.mark.asyncio
async def test_long_error_body_is_cut_to_200_characters(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent)
    http_agent.turn_answers.append(_response(500, "x" * 500))
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    prefix = NON_2XX.format(status=500, url=TURN_URL, body="")
    message = str(info.value)
    assert message.startswith(prefix + "x" * 150), message
    assert len(message) - len(prefix) <= 200, len(message) - len(prefix)
    await adapter.close()


@pytest.mark.asyncio
async def test_connection_error_text_is_redacted(impulse_project, http_agent, monkeypatch):
    import httpx

    adapter, _cfg = await _secrets_adapter(impulse_project, http_agent, monkeypatch)
    http_agent.turn_answers.append(Raise(httpx.ConnectError, f"proxy said {TOKEN} and {KNOWN_KEY}"))
    with pytest.raises(RuntimeError) as info:
        await adapter.send("Hi", None)
    message = str(info.value)
    assert message.startswith(UNREACHABLE.format(url=TURN_URL, error="ConnectError: proxy said ")), message
    assert TOKEN not in message and KNOWN_KEY not in message
    await adapter.close()


@pytest.mark.asyncio
async def test_non_object_body_text_is_redacted(impulse_project, http_agent, monkeypatch):
    adapter, _cfg = await _secrets_adapter(impulse_project, http_agent, monkeypatch)
    http_agent.turn_answers.append(_response(200, f"oops {HEADER_SECRET}", content_type="text/plain"))
    with pytest.raises(TypeError) as info:
        await adapter.send("Hi", None)
    message = str(info.value)
    assert message.startswith(NOT_OBJECT.format(url=TURN_URL, body="oops ")), message
    assert HEADER_SECRET not in message and "[REDACTED]" in message, message
    await adapter.close()


@pytest.mark.asyncio
async def test_start_and_close_errors_name_their_url(impulse_project, http_agent):
    import httpx

    project = impulse_project
    _write_config(project, http={"start_url": START_URL, "close_url": CLOSE_URL})
    cfg = _load(project)

    http_agent.start_answers.append(_response(503, "starting up"))
    adapter = _adapter(cfg, http_agent)
    with pytest.raises(RuntimeError) as info:
        await adapter.start(cfg.scenarios[0])
    assert str(info.value) == NON_2XX.format(status=503, url=START_URL, body="starting up")

    http_agent.start_answers.append(Raise(httpx.ConnectError, "refused"))
    adapter = _adapter(cfg, http_agent)
    with pytest.raises(RuntimeError) as info:
        await adapter.start(cfg.scenarios[0])
    assert str(info.value) == UNREACHABLE.format(url=START_URL, error="ConnectError: refused")

    adapter = _adapter(cfg, http_agent)
    await adapter.start(cfg.scenarios[0])
    await adapter.send("Hi", None)
    http_agent.close_answers.append(_response(500, "close failed"))
    with pytest.raises(RuntimeError) as info:
        await adapter.close()
    assert str(info.value) == NON_2XX.format(status=500, url=CLOSE_URL, body="close failed")


@pytest.mark.asyncio
async def test_close_answer_body_is_ignored(impulse_project, http_agent):
    adapter, _cfg = await _started(impulse_project, http_agent, http={"close_url": CLOSE_URL})
    await adapter.send("Hi", None)
    http_agent.close_answers.append(_response(200, "not json at all"))
    await adapter.close()
    assert len(http_agent.to("/close")) == 1


# =============================================================================================
# System prompt from the start answer
# =============================================================================================

START_PROMPT = "You are the HTTP clinic bot. Book appointments with book_slot."


@pytest.mark.asyncio
async def test_start_answer_system_prompt_when_config_sets_none(impulse_project, http_agent):
    http_agent.start_answers.append({"system_prompt": START_PROMPT, "other": 1})
    adapter, _cfg = await _started(impulse_project, http_agent, http={"start_url": START_URL}, system_prompt=None)
    assert adapter.get_system_prompt() == START_PROMPT
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [{"system_prompt": 42}, {"prompt": START_PROMPT}, "text body", [START_PROMPT]],
    ids=["non-string", "other-key", "text", "list"],
)
async def test_other_start_answers_are_ignored(impulse_project, http_agent, answer):
    import httpx

    if isinstance(answer, str):
        http_agent.start_answers.append(_response(200, answer))
    else:
        http_agent.start_answers.append(httpx.Response(200, json=answer))
    adapter, _cfg = await _started(impulse_project, http_agent, http={"start_url": START_URL}, system_prompt=None)
    assert adapter.get_system_prompt() is None
    await adapter.send("Hi", None)
    await adapter.close()


@pytest.mark.asyncio
async def test_config_system_prompt_wins_over_the_start_answer(impulse_project, http_agent, llm_router):
    project = impulse_project
    _write_config(project, http={"start_url": START_URL}, scenarios=[_scenario(project, max_turns=1)])
    http_agent.start_answers.append({"system_prompt": START_PROMPT})
    result = await _run(_load(project), adapter_factory=_factory(http_agent))
    assert result.scenario_results[0].transcript.agent_system_prompt == project.system_prompt

    _write_config(project, http={"start_url": START_URL}, scenarios=[_scenario(project, max_turns=1)],
                  system_prompt=None)
    http_agent.start_answers.append({"system_prompt": START_PROMPT})
    result = await _run(_load(project), adapter_factory=_factory(http_agent))
    assert result.scenario_results[0].transcript.agent_system_prompt == START_PROMPT


# =============================================================================================
# Through runner.run (mocked judge)
# =============================================================================================


@pytest.mark.asyncio
async def test_http_adapter_runs_end_to_end(impulse_project, http_agent, llm_router):
    project = impulse_project
    _write_config(
        project,
        http={"url": CRED_TURN_URL, "start_url": START_URL, "close_url": CLOSE_URL},
        scenarios=[_scenario(project, assertions=[CONSENT])],
    )
    llm_router.script(project.persona(SCENARIO), SECOND_USER)
    result = await _run(_load(project), adapter_factory=_factory(http_agent))

    assert result.agent_entrypoint == TURN_URL           # the URL without its credentials
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert [turn.text for turn in turns] == [OPENING, f"Echo: {OPENING}", SECOND_USER, f"Echo: {SECOND_USER}"]
    assert turns[0].heard_text == OPENING.lower()        # heard_text on the user turn
    (call,) = turns[1].tool_calls
    assert call.name == "book_slot" and call.arguments == BOOKING_ARGUMENTS
    assert call.success is True                           # inferred by the runner from {"ok": true}
    assert turns[1].latency_ms is not None and turns[1].latency_ms >= 0
    assert llm_router.calls("judge")
    assert scenario_result.passed is True
    assert [record["path"] for record in http_agent.requests] == ["/start", "/turn", "/turn", "/close"]
    ids = {record["json"]["conversation_id"] for record in http_agent.requests}
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_outbound_scenario_through_the_runner(impulse_project, http_agent, llm_router):
    project = impulse_project
    _write_config(project, scenarios=[_scenario(project, "http_outbound", direction="outbound",
                                                opening_user_message=None, max_turns=1)])
    llm_router.script(project.persona("http_outbound"), "Yes, I would like to book.")
    result = await _run(_load(project), adapter_factory=_factory(http_agent))
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    assert scenario_result.transcript.turns[0].speaker == "agent"
    assert scenario_result.transcript.turns[0].text == OPENER
    first = http_agent.to("/turn")[0]["json"]
    assert first["user_text"] is None and first["history"] == []


@pytest.mark.asyncio
async def test_agent_error_is_the_scenario_error(impulse_project, http_agent, llm_router):
    project = impulse_project
    _write_config(project, scenarios=[_scenario(project, max_turns=1)])
    http_agent.turn_answers.append(_response(500, "agent exploded"))
    result = await _run(_load(project), adapter_factory=_factory(http_agent))
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == NON_2XX.format(status=500, url=TURN_URL, body="agent exploded")
    assert scenario_result.passed is False
    assert llm_router.calls("judge") == []


@pytest.mark.asyncio
async def test_default_factory_creates_the_http_adapter(impulse_project, http_agent, llm_router, monkeypatch):
    _route_httpx(monkeypatch, http_agent)
    project = impulse_project
    _write_config(project, http={"url": CRED_TURN_URL}, scenarios=[_scenario(project, max_turns=1)])
    result = await _run(_load(project))
    assert result.agent_entrypoint == TURN_URL
    assert result.scenario_results[0].error is None
    assert [record["path"] for record in http_agent.requests] == ["/turn"]
    assert result.scenario_results[0].transcript.turns[1].text == f"Echo: {OPENING}"


@pytest.mark.asyncio
async def test_default_factory_passes_the_run_redactor(impulse_project, http_agent, llm_router, monkeypatch):
    """The run's key redactor (it knows every *_API_KEY variable) reaches the adapter, which redacts
    BEFORE cutting the body: a key straddling the 200-character cut leaves no prefix."""
    monkeypatch.setenv(RUN_KEY_ENV, RUN_KEY)
    _route_httpx(monkeypatch, http_agent)
    project = impulse_project
    _write_config(project, scenarios=[_scenario(project, max_turns=1)])
    http_agent.turn_answers.append(_response(500, "y" * 185 + RUN_KEY + "z" * 60))
    result = await _run(_load(project))
    error = result.scenario_results[0].error
    assert error.startswith(NON_2XX.format(status=500, url=TURN_URL, body="yyyy")), error
    assert RUN_KEY[:8] not in error, error


@pytest.mark.asyncio
async def test_missing_key_stops_the_run_as_a_config_error(impulse_project, http_agent, llm_router, monkeypatch):
    from impulse_evals.config import ConfigError

    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _route_httpx(monkeypatch, http_agent)
    project = impulse_project
    _write_config(project, http={"api_key_env": TOKEN_ENV}, scenarios=[_scenario(project, max_turns=1)])
    with pytest.raises(ConfigError) as info:
        await _run(_load(project))
    assert info.value.field == "agent.http.api_key_env"
    assert http_agent.requests == []


# =============================================================================================
# CLI: --fail-on, compare, --dry-run
# =============================================================================================

MATRIX = {
    "all_pass": {"critical": 0, "high": 0, "any": 0},
    "critical_failure": {"critical": 1, "high": 1, "any": 1},
    "high_failure": {"critical": 0, "high": 1, "any": 1},
    "scenario_error": {"critical": 1, "high": 1, "any": 1},
}


def _matrix_project(project, llm_router, http_agent, case):
    fail = llm_router.verdict("fail", reason="scripted failure")
    if case == "critical_failure":
        llm_router.on_judge(fail, criterion=CONSENT)
    elif case == "high_failure":
        llm_router.on_judge(fail, criterion=PURPOSE)
    elif case == "scenario_error":
        http_agent.turn_answers.append(_response(500, "agent exploded"))
    return _write_config(
        project,
        scenarios=[_scenario(project, "matrix_case", max_turns=1, assertions=[CONSENT, PURPOSE], metrics=["UNIV-001"])],
    )


@pytest.mark.parametrize("level", ["critical", "high", "any"])
@pytest.mark.parametrize("case", sorted(MATRIX))
def test_fail_on_matrix_with_the_http_adapter(impulse_project, llm_router, http_agent, cli, monkeypatch, case, level):
    _route_httpx(monkeypatch, http_agent)
    config = _matrix_project(impulse_project, llm_router, http_agent, case)
    result = cli.invoke("run", config, "--fail-on", level)
    assert result.exit_code == MATRIX[case][level], cli.text(result)
    assert impulse_project.read_run().agent_entrypoint == TURN_URL
    assert http_agent.to("/turn")


def test_compare_two_runs_of_the_http_adapter(impulse_project, llm_router, http_agent, cli, monkeypatch):
    _route_httpx(monkeypatch, http_agent)
    project = impulse_project
    config = _matrix_project(project, llm_router, http_agent, "all_pass")

    before = project.run_dirs()
    assert cli.invoke("run", config).exit_code == 0
    run_a = _new_run_dir(project, before).name
    before = project.run_dirs()
    assert cli.invoke("run", config).exit_code == 0
    run_b = _new_run_dir(project, before).name
    llm_router.on_judge(llm_router.verdict("fail", reason="scripted failure"), criterion=CONSENT)
    before = project.run_dirs()
    assert cli.invoke("run", config).exit_code == 1
    run_c = _new_run_dir(project, before).name

    assert cli.invoke("compare", run_a, run_b).exit_code == 0
    regressed = cli.invoke("compare", run_a, run_c)
    assert regressed.exit_code == 1, cli.text(regressed)


def test_dry_run_sends_no_request(impulse_project, mock_llm, http_agent, cli, monkeypatch):
    _route_httpx(monkeypatch, http_agent)
    project = impulse_project
    config = _write_config(
        project,
        http={"start_url": START_URL, "close_url": CLOSE_URL},
        judge={"primary_model": "YOUR_PRIMARY_MODEL", "fast_model": "YOUR_FAST_MODEL"},
        simulator={"model": "YOUR_SIMULATOR_MODEL", "persona_style": "realistic", "mode": "cascade"},
    )
    result = cli.invoke("run", config, "--dry-run")
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert http_agent.requests == []
    assert mock_llm.call_count == 0
    run = project.read_run()
    assert run.scenario_results == []
    assert DRY_RUN_WARNING in run.warnings
    assert REMOTE_WARNING.format(adapter="http") in run.warnings
    assert REMOTE_WARNING.format(adapter="http") in text


# =============================================================================================
# impulse doctor (the TCP connect is patched: asyncio.open_connection / socket.create_connection)
# =============================================================================================


class _FakeWriter:
    def close(self):
        return None

    async def wait_closed(self):
        return None


class _FakeSocket:
    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnect:
    """Replaces the doctor's plain TCP connect; records ``(host, port)`` and fails with ``error``."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    async def open_connection(self, *args, **kwargs):
        host = args[0] if args else kwargs.get("host")
        port = args[1] if len(args) > 1 else kwargs.get("port")
        self.calls.append((host, port))
        if self.error is not None:
            raise self.error
        return object(), _FakeWriter()

    def create_connection(self, address, *args, **kwargs):
        self.calls.append((address[0], address[1]))
        if self.error is not None:
            raise self.error
        return _FakeSocket()


def _patch_connect(monkeypatch, error=None):
    import socket

    fake = FakeConnect(error)
    monkeypatch.setattr(asyncio, "open_connection", fake.open_connection)
    monkeypatch.setattr(socket, "create_connection", fake.create_connection)
    return fake


@pytest.mark.parametrize(
    "url, host, port",
    [(TURN_URL, "localhost", 8080), ("http://localhost/turn", "localhost", 80), ("https://localhost/turn", "localhost", 443)],
    ids=["explicit-port", "http-default", "https-default"],
)
def test_doctor_http_rows_pass(impulse_project, mock_llm, http_agent, cli, monkeypatch, url, host, port):
    mock_llm.set_default("OK")
    _route_httpx(monkeypatch, http_agent)
    fake = _patch_connect(monkeypatch)
    config = _write_config(impulse_project, http={"url": url, "start_url": START_URL, "close_url": CLOSE_URL})
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert _rows(result, "PASS", "agent http") == [f"reachable: {url}"], text
    assert _rows(result, "INFO", "agent connection") == ["the turn endpoint is not called by impulse doctor"], text
    assert fake.calls == [(host, port)]
    assert http_agent.requests == []                      # no HTTP request: it could have side effects
    assert not _rows(result, "PASS", "HTTP agent key") and not _rows(result, "FAIL", "HTTP agent key")


def test_doctor_http_row_fails_when_the_port_is_closed(impulse_project, mock_llm, http_agent, cli, monkeypatch):
    mock_llm.set_default("OK")
    _route_httpx(monkeypatch, http_agent)
    _patch_connect(monkeypatch, ConnectionRefusedError(111, "Connection refused"))
    config = _write_config(impulse_project)
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2, text
    rows = _rows(result, "FAIL", "agent http")
    assert len(rows) == 1, text
    detail = rows[0]
    assert detail.startswith(f"cannot connect to {TURN_URL}: "), detail
    assert "Connection refused" in detail
    assert http_agent.requests == []


def test_doctor_http_rows_show_the_url_without_credentials(impulse_project, mock_llm, http_agent, cli, monkeypatch):
    mock_llm.set_default("OK")
    _route_httpx(monkeypatch, http_agent)
    fake = _patch_connect(monkeypatch)
    config = _write_config(impulse_project, http={"url": CRED_TURN_URL})
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert _rows(result, "PASS", "agent http") == [f"reachable: {TURN_URL}"]
    assert "agentpass" not in result.output and "token=abc" not in result.output
    assert fake.calls == [("localhost", 8080)]


@pytest.mark.parametrize("where", ["environment", "toml", "missing"])
def test_doctor_http_key_row(impulse_project, mock_llm, http_agent, cli, monkeypatch, where):
    mock_llm.set_default("OK")
    _route_httpx(monkeypatch, http_agent)
    _patch_connect(monkeypatch)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    if where == "environment":
        monkeypatch.setenv(TOKEN_ENV, TOKEN)
    elif where == "toml":
        impulse_project.write_toml(f'[keys]\n{TOKEN_ENV} = "{TOKEN}"\n')
    config = _write_config(impulse_project, http={"api_key_env": TOKEN_ENV})
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    if where == "environment":
        assert result.exit_code == 0, text
        assert _rows(result, "PASS", "HTTP agent key") == [f"{TOKEN_ENV} is set (environment variable)"], text
    elif where == "toml":
        assert result.exit_code == 0, text
        assert _rows(result, "PASS", "HTTP agent key") == [f"{TOKEN_ENV} is set ([keys] in .impulse/config.toml)"], text
    else:
        assert result.exit_code == 2, text
        assert _rows(result, "FAIL", "HTTP agent key") == [MISSING_KEY.format(name=TOKEN_ENV)], text
    assert TOKEN not in result.output
    assert http_agent.requests == []


# =============================================================================================
# examples/http
# =============================================================================================


def test_example_http_files_and_dry_run(impulse_project, mock_llm, cli):
    from impulse_evals.config import load_config

    config = EXAMPLE_DIR / "config.yaml"
    for name in ("config.yaml", "server.py", "README.md"):
        assert (EXAMPLE_DIR / name).is_file(), f"examples/http/{name} is missing"
    cfg = load_config(config)
    assert cfg.agent.adapter == "http"
    assert cfg.agent.http.url.startswith(("http://", "https://"))
    assert [scenario.language for scenario in cfg.scenarios] == ["en-US"]
    readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
    assert "conversation_id" in readme and "not verified" in readme
    result = cli.invoke("run", config, "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    assert REMOTE_WARNING.format(adapter="http") in impulse_project.read_run().warnings
