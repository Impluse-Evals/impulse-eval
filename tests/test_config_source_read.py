# SPDX-License-Identifier: Apache-2.0
"""Phase 11: ``config.yaml`` changes from the pre-Stage-B source read (SPEC.md sections 8.2 and 9;
DECISIONS.md decisions D2 and D11).

* ``pipecat.sample_rate`` is used in both modes: present under ``mode: eval`` it gives NO
  "ignored in eval mode" warning, while the websocket-only keys ``chunk_ms``, ``realtime_pacing``,
  ``trailing_silence_ms`` do (only when present in the raw YAML; defaults never warn).
* ``livekit.agent_name`` is accepted (default None).
"""

import pytest

pytestmark = pytest.mark.phase11

WEBSOCKET_ONLY = ("chunk_ms", "realtime_pacing", "trailing_silence_ms")


def _pipecat(project, **pipecat):
    block = {"mode": "eval", "url": "ws://localhost:7860"}
    block.update(pipecat)
    project.write_config([project.scenario("pipecat_config")], agent={"adapter": "pipecat", "pipecat": block})
    return project.load()


def _livekit(project, **livekit):
    block = {"url": "ws://localhost:7880"}
    block.update(livekit)
    project.write_config([project.scenario("livekit_config")], agent={"adapter": "livekit", "livekit": block})
    return project.load()


def _ignored(cfg):
    return [warning for warning in cfg.warnings if warning.endswith("is ignored in eval mode.")]


def test_sample_rate_in_eval_mode_gives_no_warning(impulse_project):
    cfg = _pipecat(impulse_project, sample_rate=24000)
    assert _ignored(cfg) == []
    assert cfg.agent.pipecat.sample_rate == 24000


@pytest.mark.parametrize(
    "key, value", [("chunk_ms", 40), ("realtime_pacing", False), ("trailing_silence_ms", 300)]
)
def test_each_websocket_only_key_warns_in_eval_mode(impulse_project, key, value):
    cfg = _pipecat(impulse_project, **{key: value, "sample_rate": 16000})
    assert _ignored(cfg) == [f"pipecat.{key} is ignored in eval mode."]


def test_all_websocket_only_keys_warn_and_sample_rate_does_not(impulse_project):
    cfg = _pipecat(impulse_project, sample_rate=8000, chunk_ms=40, realtime_pacing=False, trailing_silence_ms=300)
    assert sorted(_ignored(cfg)) == sorted(f"pipecat.{key} is ignored in eval mode." for key in WEBSOCKET_ONLY)
    assert "pipecat.sample_rate is ignored in eval mode." not in cfg.warnings


def test_defaults_never_warn_in_eval_mode(impulse_project):
    cfg = _pipecat(impulse_project)
    assert _ignored(cfg) == []
    assert cfg.agent.pipecat.sample_rate == 16000


def test_websocket_mode_never_warns_about_its_own_keys(impulse_project):
    cfg = _pipecat(
        impulse_project, mode="websocket", sample_rate=8000, chunk_ms=40, realtime_pacing=False, trailing_silence_ms=300
    )
    assert _ignored(cfg) == []
    pipecat = cfg.agent.pipecat
    assert (pipecat.sample_rate, pipecat.chunk_ms, pipecat.realtime_pacing, pipecat.trailing_silence_ms) == (
        8000, 40, False, 300,
    )


def test_livekit_agent_name_defaults_to_none(impulse_project):
    cfg = _livekit(impulse_project)
    assert cfg.agent.livekit.agent_name is None


def test_livekit_agent_name_is_accepted(impulse_project):
    cfg = _livekit(impulse_project, agent_name="neo-recruiter", agent_identity="neo-agent", room_prefix="t2n-")
    livekit = cfg.agent.livekit
    assert livekit.agent_name == "neo-recruiter"
    assert livekit.agent_identity == "neo-agent"
    assert livekit.room_prefix == "t2n-"
    assert not [warning for warning in cfg.warnings if "agent_name" in warning]
