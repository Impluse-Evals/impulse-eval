# SPDX-License-Identifier: Apache-2.0
"""Phase 8: ``simulator.mode`` (SPEC.md sections 8.2, 10, and 17).

``audio_native`` is a config error with the exact message
"simulator.mode: audio_native is planned for v2. Use cascade." (``impulse run`` and ``impulse doctor``
exit 2); ``cascade`` runs the simulated caller.
"""

import pytest

AUDIO_NATIVE = "simulator.mode: audio_native is planned for v2. Use cascade."


def _simulator(mode):
    return {"model": "ollama/simulator", "persona_style": "realistic", "mode": mode, "options": {}}


def _config(project, mode):
    project.write_agent()
    return project.write_config(
        [project.scenario("mode_probe", max_turns=3, metrics=["UNIV-013"])], simulator=_simulator(mode)
    )


def test_load_config_audio_native_exact_message(impulse_project):
    from impulse_evals.config import ConfigError

    _config(impulse_project, "audio_native")
    with pytest.raises(ConfigError) as info:
        impulse_project.load()
    assert info.value.field == "simulator.mode"
    assert info.value.message == AUDIO_NATIVE


def test_run_audio_native_exits_2_with_exact_message(impulse_project, mock_llm, cli):
    result = cli.invoke("run", _config(impulse_project, "audio_native"))
    assert result.exit_code == 2
    assert AUDIO_NATIVE in cli.text(result)
    assert mock_llm.call_count == 0


def test_dry_run_audio_native_exits_2(impulse_project, mock_llm, cli):
    result = cli.invoke("run", _config(impulse_project, "audio_native"), "--dry-run")
    assert result.exit_code == 2
    assert AUDIO_NATIVE in cli.text(result)


def test_doctor_audio_native_exits_2(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    result = cli.invoke("doctor", _config(impulse_project, "audio_native"))
    assert result.exit_code == 2
    assert AUDIO_NATIVE in cli.text(result)


def test_unknown_simulator_mode_exits_2(impulse_project, mock_llm, cli):
    result = cli.invoke("run", _config(impulse_project, "duplex"))
    assert result.exit_code == 2
    assert "simulator.mode" in cli.text(result)


def test_cascade_runs_the_simulated_caller(impulse_project, llm_router, cli):
    project = impulse_project
    config = _config(project, "cascade")
    llm_router.script(project.persona("mode_probe"), "Tuesday morning, please.", "<END_CALL>")
    result = cli.invoke("run", config)
    assert result.exit_code == 0, cli.text(result)
    assert len(llm_router.calls("simulator")) == 2
    run = project.read_run()
    users = [turn.text for turn in run.scenario_results[0].transcript.turns if turn.speaker == "user"]
    assert users == [project.opening("mode_probe"), "Tuesday morning, please."]
