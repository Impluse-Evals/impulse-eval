# SPDX-License-Identifier: Apache-2.0
"""HTTP adapter: the optional ``user_audio`` turn field (SPEC.md section 9; noise simulation).

``agent.http.send_audio: true`` adds ``"user_audio": "<base64 of the WAV>"`` (standard alphabet, with
padding) to each turn request that has caller audio, which in voice mode is the audio the agent
should hear (noisy when noise simulation is on). Without ``send_audio`` (the default), in text mode,
and for the outbound opener, the body is exactly the four keys it always had, so agents that ignore
the field, and every existing agent, keep working.
"""

import base64
import json
from collections import deque

import pytest
from conftest import sine_wav

TURN_URL = "http://localhost:8080/turn"
VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}
BASE_KEYS = {"conversation_id", "user_text", "history", "scenario_name"}


class _Agent:
    def __init__(self):
        import httpx

        self.bodies = []
        self.answers = deque()
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request):
        import httpx

        self.bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"text": "ok"})


def _adapter(project, *, send_audio=None, voice=True):
    from impulse_evals.adapters.http_adapter import HTTPAdapter

    http = {"url": TURN_URL}
    if send_audio is not None:
        http["send_audio"] = send_audio
    project.write_agent()
    kwargs = {"voice_config": VOICE_CONFIG} if voice else {}
    project.write_config([project.scenario("http_noise")], agent={"adapter": "http", "http": http}, **kwargs)
    cfg = project.load()
    agent = _Agent()
    return cfg, HTTPAdapter(cfg, transport=agent.transport), agent


@pytest.mark.asyncio
async def test_send_audio_adds_base64_user_audio(impulse_project):
    cfg, adapter, agent = _adapter(impulse_project, send_audio=True)
    audio = sine_wav(0.3)
    await adapter.start(cfg.scenarios[0])
    await adapter.send("hello", audio)
    await adapter.close()
    body = agent.bodies[0]
    assert set(body) == BASE_KEYS | {"user_audio"}
    assert base64.b64decode(body["user_audio"], validate=True) == audio


@pytest.mark.asyncio
async def test_default_body_is_unchanged_in_voice_mode(impulse_project):
    cfg, adapter, agent = _adapter(impulse_project)
    assert cfg.agent.http.send_audio is False
    await adapter.start(cfg.scenarios[0])
    await adapter.send("hello", sine_wav(0.3))
    await adapter.close()
    assert set(agent.bodies[0]) == BASE_KEYS


@pytest.mark.asyncio
async def test_opener_and_text_turns_carry_no_user_audio(impulse_project):
    cfg, adapter, agent = _adapter(impulse_project, send_audio=True, voice=False)
    await adapter.start(cfg.scenarios[0])
    await adapter.send(None, None)
    await adapter.send("hello", None)
    await adapter.close()
    assert [set(body) for body in agent.bodies] == [BASE_KEYS, BASE_KEYS]


def test_send_audio_must_be_a_boolean(impulse_project):
    from impulse_evals.config import ConfigError

    project = impulse_project
    project.write_agent()
    project.write_config(
        [project.scenario("x")],
        agent={"adapter": "http", "http": {"url": TURN_URL, "send_audio": "sometimes"}},
    )
    with pytest.raises(ConfigError) as info:
        project.load()
    assert info.value.field == "agent.http.send_audio"
