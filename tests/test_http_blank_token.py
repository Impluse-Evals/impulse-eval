# SPDX-License-Identifier: Apache-2.0
"""Fixes round 3, B4-rev (decided after the fix-up; FIXES-ROUND-3-SPEC.md "B1 decisions", last
sentence): an HTTP adapter bearer token that is blank after stripping (environment variable or
config.toml ``[keys]``) is the missing-key ``ConfigError`` at ``agent.http.api_key_env``:
"{NAME} is not set: set the environment variable or add it to [keys] in .impulse/config.toml",
raised in ``start()`` before any request.

No network: ``httpx.MockTransport`` only.
"""

import json

import pytest

TURN_URL = "http://localhost:8080/turn"
START_URL = "http://localhost:8080/start"
TOKEN_ENV = "IMPULSE_TEST_HTTP_BLANK_TOKEN"
MISSING_KEY = "{name} is not set: set the environment variable or add it to [keys] in .impulse/config.toml"


class FakeAgent:
    def __init__(self):
        import httpx

        self.requests = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request):
        import httpx

        self.requests.append(str(request.url))
        return httpx.Response(200, json={"text": "ok"})


def _write_config(project, http):
    block = {"url": TURN_URL, "api_key_env": TOKEN_ENV}
    block.update(http)
    scenario = project.scenario("http_blank_token", max_turns=1, metrics=["UNIV-001"])
    return project.write_config([scenario], agent={"adapter": "http", "http": block})


BLANK = [
    pytest.param("env", "", id="env-empty"),
    pytest.param("env", "   ", id="env-spaces"),
    pytest.param("env", " \t ", id="env-tab"),
    pytest.param("toml", "", id="toml-empty"),
    pytest.param("toml", "  ", id="toml-spaces"),
    pytest.param("toml", "\n\t \r\n", id="toml-newlines"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("source, value", BLANK)
@pytest.mark.parametrize("with_start_url", [True, False], ids=["start_url", "no-start_url"])
async def test_blank_bearer_token_is_the_missing_key_config_error(impulse_project, monkeypatch, source, value, with_start_url):
    from impulse_evals.adapters.http_adapter import HTTPAdapter
    from impulse_evals.config import ConfigError

    project = impulse_project
    if source == "env":
        monkeypatch.setenv(TOKEN_ENV, value)
    else:
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        project.write_toml(f"[keys]\n{TOKEN_ENV} = {json.dumps(value)}\n")
    _write_config(project, {"start_url": START_URL} if with_start_url else {})
    cfg = project.load()
    agent = FakeAgent()
    adapter = HTTPAdapter(cfg, transport=agent.transport)
    with pytest.raises(ConfigError) as info:
        await adapter.start(cfg.scenarios[0])
    assert info.value.field == "agent.http.api_key_env"
    assert info.value.message == MISSING_KEY.format(name=TOKEN_ENV)
    assert agent.requests == []
    await adapter.close()
