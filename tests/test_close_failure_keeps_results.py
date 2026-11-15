# SPDX-License-Identifier: Apache-2.0
"""Fixes round 4, M2 (FINAL-TEST.md audit, 2026-11-02): a failure AFTER the conversation ended does not
discard the evaluation.

When the conversation finished normally and then ``adapter.close()``, ``get_metadata()`` or
``get_warnings()`` raises, the runner evaluates the transcript as usual (metrics, assertions, ``passed``
computed from them, ``error`` None) and adds the run warning
``runner.CLOSE_FAILED_WARNING``: "{scenario_name}: the adapter's close failed after the conversation
ended; the results are kept: {error}" (the error text redacted and on one line, as scenario errors are).
The HTTP adapter's ``close()`` POSTs ``close_url``, so a flaky close endpoint no longer throws away every
verdict. When the conversation itself errored, that error stays the scenario error (unchanged).

The judge and the simulator are ``mock_llm`` routed by ``llm_router``. No network.
"""

import importlib
import uuid

import pytest

CLOSE_FAILED_WARNING = "{scenario_name}: the adapter's close failed after the conversation ended; the results are kept: {error}"
SECRET_ENV = "IMPULSE_TEST_CLOSE_API_KEY"   # a *_API_KEY variable: the run's redactor knows its value
SECRET = "sk-impulse-test-SECRET-close-4471abcdef"

ADAPTER = '''
import os

from impulse_evals.adapters.base import AgentInterface, AgentReply

EVENTS = []


class CloseFailsAdapter(AgentInterface):
    def __init__(self, config):
        self.options = dict(config.agent.options)
        self.scenario = None

    async def start(self, scenario):
        self.scenario = scenario
        EVENTS.append(("start", scenario.name))

    async def send(self, user_text, user_audio):
        EVENTS.append(("send", user_text))
        if "crash" in (user_text or "").lower():
            raise RuntimeError("agent crashed while answering")
        return AgentReply(text="Echo: " + (user_text or ""), latency_ms=10.0)

    async def close(self):
        EVENTS.append(("close", self.scenario.name))
        if self.options.get("close_fails"):
            raise RuntimeError("close endpoint returned HTTP 500 (token " + os.environ.get("IMPULSE_TEST_CLOSE_API_KEY", "none") + ")")

    def get_metadata(self):
        if self.options.get("metadata_fails"):
            raise RuntimeError("metadata exploded")
        return {"close_test": True}

    def get_warnings(self):
        if self.options.get("warnings_fail"):
            raise RuntimeError("warnings exploded")
        return []
'''


def _write_module(project):
    stem = f"impulse_test_close_{uuid.uuid4().hex[:10]}"
    (project.root / f"{stem}.py").write_text("# SPDX-License-Identifier: Apache-2.0\n" + ADAPTER, encoding="utf-8")
    importlib.invalidate_caches()
    return stem


def _config(project, options, *, opening="Hello, I would like to book a slot on Tuesday."):
    stem = _write_module(project)
    scenario = project.scenario("close_test", max_turns=2, opening_user_message=opening, metrics=["UNIV-001", "UNIV-013"])
    project.write_config([scenario], agent={"adapter": "custom", "entrypoint": f"{stem}:CloseFailsAdapter", "options": options})
    return project.load()


async def _run(cfg):
    from impulse_evals.runner import run

    return await run(cfg)


@pytest.mark.asyncio
async def test_close_failure_after_a_complete_conversation_keeps_the_evaluation(impulse_project, llm_router, monkeypatch):
    monkeypatch.setenv(SECRET_ENV, SECRET)
    cfg = _config(impulse_project, {"close_fails": True})
    result = await _run(cfg)
    scenario = result.scenario_results[0]
    assert scenario.error is None
    assert [turn.speaker for turn in scenario.transcript.turns][:2] == ["user", "agent"]
    assert scenario.metric_results, "the transcript was not evaluated"
    assert {m.metric_id for m in scenario.metric_results} >= {"UNIV-001", "UNIV-013"}
    assert scenario.passed is True                      # the mocked judge answers pass
    assert llm_router.calls("judge"), "the judge was never called"
    expected = CLOSE_FAILED_WARNING.format(
        scenario_name="close_test", error="close endpoint returned HTTP 500 (token [REDACTED])"
    )
    assert expected in result.warnings
    assert SECRET not in " ".join(result.warnings)
    assert result.passed == 1 and result.failed == 0


@pytest.mark.asyncio
async def test_close_failed_warning_text_is_a_module_constant():
    from impulse_evals import runner

    assert runner.CLOSE_FAILED_WARNING == CLOSE_FAILED_WARNING


@pytest.mark.asyncio
@pytest.mark.parametrize("option, text", [("metadata_fails", "metadata exploded"), ("warnings_fail", "warnings exploded")])
async def test_metadata_or_warnings_failure_after_the_conversation_keeps_the_evaluation(
    impulse_project, llm_router, option, text
):
    cfg = _config(impulse_project, {option: True})
    result = await _run(cfg)
    scenario = result.scenario_results[0]
    assert scenario.error is None
    assert scenario.metric_results
    assert CLOSE_FAILED_WARNING.format(scenario_name="close_test", error=text) in result.warnings


@pytest.mark.asyncio
async def test_a_conversation_error_still_wins_over_a_close_failure(impulse_project, llm_router):
    cfg = _config(impulse_project, {"close_fails": True}, opening="Please crash now.")
    result = await _run(cfg)
    scenario = result.scenario_results[0]
    assert scenario.error == "agent crashed while answering"
    assert scenario.metric_results == [] and scenario.assertion_results == []
    assert not any(w.startswith("close_test: the adapter's close failed") for w in result.warnings)


@pytest.mark.asyncio
async def test_a_working_close_adds_no_warning(impulse_project, llm_router):
    cfg = _config(impulse_project, {})
    result = await _run(cfg)
    assert result.scenario_results[0].error is None
    assert not any("close failed" in w for w in result.warnings)
    assert result.scenario_results[0].transcript.metadata.get("close_test") is True
