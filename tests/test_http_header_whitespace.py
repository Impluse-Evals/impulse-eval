# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 4, M1 (FINAL-TEST.md audit, 2026-11-02): no HTTP header value can leak through an error text.

* ``load_config``: an ``agent.http.headers`` value with leading or trailing whitespace is the config
  error ``agent.http.headers.{name}``: "header values must not start or end with whitespace" (the value
  is never shown). Reason: such a value passes the printable-ASCII rule but h11 refuses to send it, and
  the httpx ``LocalProtocolError`` text echoes the value as a bytes repr.
* ``config.redact``: a known value is also redacted in its JSON-escaped form (``json.dumps(value)[1:-1]``,
  so a value with ``"`` or ``\\`` echoed inside a JSON body is hidden) and in its Python repr form
  (``repr(value)[1:-1]``, the form a bytes / str repr in an exception text uses).
* HTTP adapter end to end (``httpx.MockTransport``, no network): a 401 body that echoes a header value
  holding quotes and backslashes keeps no fragment of it in the scenario error.
"""

import json

import pytest

TURN_URL = "http://localhost:8080/turn"
WHITESPACE_MESSAGE = "header values must not start or end with whitespace"
QUOTED = 'sk-FAKE-"quoted"-value-0123'
BACKSLASH = "sk-FAKE-back\\slash-0123"


def _write_config(project, headers):
    scenario = project.scenario("http_headers", max_turns=1, metrics=["UNIV-001"])
    return project.write_config([scenario], agent={"adapter": "http", "http": {"url": TURN_URL, "headers": headers}})


ASCII_MESSAGE = "header values must be printable ASCII without control characters"


@pytest.mark.parametrize(
    "value, message",
    [
        pytest.param(" sk-FAKE-leading", WHITESPACE_MESSAGE, id="leading-space"),
        pytest.param("sk-FAKE-trailing ", WHITESPACE_MESSAGE, id="trailing-space"),
        pytest.param(" sk-FAKE-both ", WHITESPACE_MESSAGE, id="leading-and-trailing"),
        # A tab is a control character: the existing printable-ASCII rule fires first (its message also
        # never shows the value); a Stage B correction after the first run of this file.
        pytest.param("\tsk-FAKE-tab", ASCII_MESSAGE, id="leading-tab"),
    ],
)
def test_header_value_with_surrounding_whitespace_is_a_config_error_without_the_value(impulse_project, value, message):
    from impulse_evals.config import ConfigError

    _write_config(impulse_project, {"X-Token": value})
    with pytest.raises(ConfigError) as info:
        impulse_project.load()
    assert info.value.field == "agent.http.headers.X-Token"
    assert info.value.message == message
    assert "sk-FAKE" not in str(info.value)


def test_header_value_without_surrounding_whitespace_still_loads(impulse_project):
    _write_config(impulse_project, {"X-Token": "sk-FAKE inner space is fine", "Origin": "http://localhost:3000"})
    cfg = impulse_project.load()
    assert cfg.agent.http.headers == {"X-Token": "sk-FAKE inner space is fine", "Origin": "http://localhost:3000"}


@pytest.mark.parametrize(
    "value, text",
    [
        pytest.param(QUOTED, 'body {"X-Api-Key": "' + json.dumps(QUOTED)[1:-1] + '"} end', id="json-escaped-quotes"),
        pytest.param(BACKSLASH, 'body {"X-Token": "' + json.dumps(BACKSLASH)[1:-1] + '"} end', id="json-escaped-backslash"),
        pytest.param(BACKSLASH, "LocalProtocolError: Illegal header value b'" + repr(BACKSLASH)[1:-1] + "'", id="repr-bytes"),
        pytest.param(QUOTED, "ValueError: bad value " + repr(QUOTED), id="repr-str"),
    ],
)
def test_redact_hides_the_json_escaped_and_repr_forms_of_a_known_value(value, text):
    from impulse_evals.config import REDACTED, redact

    redacted = redact(text, extra_values=[value])
    assert REDACTED in redacted
    assert "sk-FAKE" not in redacted
    assert "quoted" not in redacted and "slash" not in redacted


def test_redact_leaves_a_plain_value_rule_unchanged():
    from impulse_evals.config import REDACTED, redact

    assert redact("key sk-FAKE-plain-0123 here", extra_values=["sk-FAKE-plain-0123"]) == f"key {REDACTED} here"
    assert redact("nothing to hide", extra_values=["sk-FAKE-plain-0123"]) == "nothing to hide"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [QUOTED, BACKSLASH], ids=["quotes", "backslash"])
async def test_401_body_echoing_a_header_with_quotes_or_backslashes_is_redacted(impulse_project, value):
    import httpx

    from impulse_evals.adapters.http_adapter import HTTPAdapter

    def handle(request):
        return httpx.Response(401, json={"got": {"X-Api-Key": request.headers.get("X-Api-Key")}})

    _write_config(impulse_project, {"X-Api-Key": value})
    cfg = impulse_project.load()
    adapter = HTTPAdapter(cfg, transport=httpx.MockTransport(handle))
    await adapter.start(cfg.scenarios[0])
    try:
        with pytest.raises(RuntimeError) as info:
            await adapter.send("Hello", None)
    finally:
        await adapter.close()
    message = str(info.value)
    assert message.startswith("HTTP agent returned HTTP 401 for POST http://localhost:8080/turn: ")
    assert "[REDACTED]" in message
    assert "sk-FAKE" not in message and "quoted" not in message and "slash" not in message
