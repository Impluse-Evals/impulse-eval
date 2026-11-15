# SPDX-License-Identifier: Apache-2.0
"""Phase 12: chat metrics in ``metrics/chat.py`` (METRICS.md section 3; SPEC.md sections 7 and 13).

* CHAT-001 Response latency (deterministic, no judge): ``raw_value`` = median ``Turn.latency_ms``
  of the agent turns that have one; good <= 3000 ms, warning <= 8000 ms, poor above; score
  1.0 / 0.5 / 0.0; ``threshold_warn`` / ``threshold_fail`` hold the raw thresholds used (3000 / 8000).
  With no agent latency at all the metric is skipped (precondition did not occur).
* CHAT-003 Formatting and readability (judge, fast) and CHAT-005 Reference validity (judge,
  primary): criterion sent word for word (plus ``Expected outcome:`` when set); pass / fail / skip
  map to 1.0 good / 0.0 poor / skipped.
* CHAT metrics run only when listed in a scenario's ``metrics``; with ``agent.type: voice`` they skip
  (decided in code, zero judge calls) with the reason "chat metric; agent.type is voice". They run for
  ``chat`` and ``multimodal`` agents.
"""

import pytest

PRIMARY = "ollama/judge-primary"
FAST = "ollama/judge-fast"
CHAT_ON_VOICE = "chat metric; agent.type is voice"
SYSTEM_PROMPT = "You are the support assistant of Sunrise Store. Only share links listed in this prompt."

JUDGE_CHAT = {
    "CHAT-003": (
        "Formatting and readability",
        "fast",
        "Is formatting appropriate for the channel and easy to read (no wall of text, no broken markdown)?",
    ),
    "CHAT-005": (
        "Reference validity",
        "primary",
        "Does the agent avoid fabricated links, order numbers, policy names, or contact details? Any URL or "
        "identifier must appear in the system prompt or the user's messages.",
    ),
}
CHAT_IDS = ["CHAT-001", "CHAT-003", "CHAT-005"]


def _m():
    from impulse_evals import models

    return models


def _metric(metric_id):
    from impulse_evals.metrics.base import get_metric

    return get_metric(metric_id)


def _ctx(**kwargs):
    from impulse_evals.metrics.base import MetricContext

    return MetricContext(**kwargs)


def _scenario(**kwargs):
    values = dict(name="store_chat", language="en-IN", domain="ecommerce", persona="shopper with a question")
    values.update(kwargs)
    return _m().Scenario(**values)


def _transcript(agent_latencies, system_prompt=SYSTEM_PROMPT):
    """Alternating user / agent turns; agent turn i gets ``agent_latencies[i]`` as ``latency_ms``."""
    m = _m()
    turns = []
    clock = 0
    for index, latency in enumerate(agent_latencies, start=1):
        turns.append(m.Turn(speaker="user", text=f"Question {index}?", timestamp_ms=clock))
        clock += 1000
        turns.append(m.Turn(speaker="agent", text=f"Answer {index}.", timestamp_ms=clock, latency_ms=latency))
        clock += 1000
    return m.CallTranscript(
        turns=turns,
        duration_ms=clock,
        metadata={},
        scenario_name="store_chat",
        language="en-IN",
        agent_system_prompt=system_prompt,
    )


def _assert_skipped(result):
    assert result.skipped is True
    assert result.label == "skipped"
    assert result.score == 0.0


# ============================================================================================
# Registry attributes
# ============================================================================================


def test_chat001_attributes():
    metric = _metric("CHAT-001")
    assert metric.metric_id == "CHAT-001"
    assert metric.name == "Response latency"
    assert metric.category == "chat"
    assert metric.tier is None
    assert metric.criterion is None


@pytest.mark.parametrize("metric_id", sorted(JUDGE_CHAT))
def test_judge_chat_metric_attributes(metric_id):
    name, tier, criterion = JUDGE_CHAT[metric_id]
    metric = _metric(metric_id)
    assert metric.metric_id == metric_id
    assert metric.name == name
    assert metric.category == "chat"
    assert metric.tier == tier
    assert metric.criterion == criterion


# ============================================================================================
# CHAT-001 (deterministic)
# ============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "latencies, raw, label, score",
    [
        ([2999.0], 2999.0, "good", 1.0),
        ([3000.0], 3000.0, "good", 1.0),  # boundary: <= 3000 is good
        ([3000.5], 3000.5, "warning", 0.5),
        ([8000.0], 8000.0, "warning", 0.5),  # boundary: <= 8000 is warning
        ([8001.0], 8001.0, "poor", 0.0),
        ([1000.0, 2000.0, 9000.0], 2000.0, "good", 1.0),  # median, not mean
        ([1000.0, 5000.0, 9000.0, 12000.0], 7000.0, "warning", 0.5),  # even count: mean of the middle two
        ([9000.0, 500.0, 9500.0], 9000.0, "poor", 0.0),  # order does not matter
    ],
)
async def test_chat001_median_latency_thresholds(fake_judge, mock_llm, latencies, raw, label, score):
    judge = fake_judge()
    result = await _metric("CHAT-001").evaluate(_transcript(latencies), _scenario(), judge, _ctx(agent_type="chat"))
    assert result.metric_id == "CHAT-001"
    assert result.category == "chat"
    assert result.skipped is False
    assert result.raw_value == pytest.approx(raw)
    assert result.label == label
    assert result.score == score
    assert result.threshold_warn == 3000.0
    assert result.threshold_fail == 8000.0
    assert judge.calls == []
    assert mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_chat001_ignores_turns_without_latency(fake_judge):
    m = _m()
    transcript = _transcript([None, 2000.0, None, 4000.0])
    transcript.turns.insert(0, m.Turn(speaker="agent", text="Welcome!", timestamp_ms=0))  # opener, no latency
    result = await _metric("CHAT-001").evaluate(transcript, _scenario(), fake_judge(), _ctx(agent_type="chat"))
    assert result.raw_value == pytest.approx(3000.0)
    assert result.label == "good"


@pytest.mark.asyncio
async def test_chat001_skips_without_any_agent_latency(fake_judge, mock_llm):
    judge = fake_judge()
    result = await _metric("CHAT-001").evaluate(_transcript([None, None]), _scenario(), judge, _ctx(agent_type="chat"))
    _assert_skipped(result)
    assert judge.calls == []
    assert mock_llm.call_count == 0


# ============================================================================================
# CHAT-003 / CHAT-005 (judge)
# ============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", ["chat", "multimodal"])
@pytest.mark.parametrize("metric_id", sorted(JUDGE_CHAT))
async def test_judge_chat_metric_sends_criterion_verbatim(fake_judge, metric_id, agent_type):
    _name, tier, criterion = JUDGE_CHAT[metric_id]
    judge = fake_judge()
    await _metric(metric_id).evaluate(_transcript([900.0, 1200.0]), _scenario(), judge, _ctx(agent_type=agent_type))
    assert len(judge.calls) == 1
    assert judge.calls[0]["tier"] == tier
    assert judge.calls[0]["criterion"].rstrip() == criterion


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_id", sorted(JUDGE_CHAT))
async def test_judge_chat_metric_appends_expected_outcome(fake_judge, metric_id):
    _name, _tier, criterion = JUDGE_CHAT[metric_id]
    judge = fake_judge()
    scenario = _scenario(expected_outcome="the shopper gets the return policy link")
    await _metric(metric_id).evaluate(_transcript([900.0]), scenario, judge, _ctx(agent_type="chat"))
    assert judge.calls[0]["criterion"].rstrip() == (
        criterion + "\n\nExpected outcome: the shopper gets the return policy link"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_id", sorted(JUDGE_CHAT))
@pytest.mark.parametrize("verdict, score, label", [("pass", 1.0, "good"), ("fail", 0.0, "poor")])
async def test_judge_chat_metric_verdict_mapping(fake_judge, metric_id, verdict, score, label):
    judge = fake_judge({"result": verdict, "reason": "judge says so", "turn_index": 2, "confidence": 0.8})
    result = await _metric(metric_id).evaluate(_transcript([900.0]), _scenario(), judge, _ctx(agent_type="chat"))
    assert result.metric_id == metric_id
    assert result.category == "chat"
    assert result.skipped is False
    assert result.score == score
    assert result.label == label
    assert result.turn_index == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_id", sorted(JUDGE_CHAT))
async def test_judge_chat_metric_skip_verdict(fake_judge, metric_id):
    judge = fake_judge({"result": "skip", "reason": "nothing to judge", "confidence": 0.9})
    result = await _metric(metric_id).evaluate(_transcript([900.0]), _scenario(), judge, _ctx(agent_type="chat"))
    _assert_skipped(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_id, model", [("CHAT-003", FAST), ("CHAT-005", PRIMARY)])
async def test_judge_chat_metric_routes_to_its_model_with_mock_llm(mock_llm, metric_id, model):
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    _name, _tier, criterion = JUDGE_CHAT[metric_id]
    mock_llm.queue_verdict("fail", reason="a link was invented", turn_index=2)
    judge = LiteLLMJudge(PRIMARY, FAST, {}, {})
    result = await _metric(metric_id).evaluate(_transcript([900.0]), _scenario(), judge, _ctx(agent_type="chat"))
    assert mock_llm.call_count == 1
    assert mock_llm.calls[0]["model"] == model
    assert criterion in mock_llm.text(0)
    assert result.label == "poor"
    assert result.reason and "a link was invented" in result.reason


# ============================================================================================
# Voice agents and selection
# ============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_id", CHAT_IDS)
async def test_chat_metrics_skip_for_voice_agents_without_judge(fake_judge, mock_llm, metric_id):
    judge = fake_judge()
    for ctx in (_ctx(agent_type="voice"), _ctx()):  # "voice" is also the MetricContext default
        result = await _metric(metric_id).evaluate(_transcript([900.0]), _scenario(), judge, ctx)
        _assert_skipped(result)
        assert result.reason == CHAT_ON_VOICE
        assert result.category == "chat"
    assert judge.calls == []
    assert mock_llm.call_count == 0


def test_chat_metrics_are_selected_only_when_listed():
    from impulse_evals.metrics.base import select_metrics

    assert not [m for m in select_metrics(_scenario(metrics=[]), _ctx(agent_type="chat")) if m.startswith("CHAT-")]
    listed = select_metrics(_scenario(metrics=["UNIV-001", "CHAT-001", "CHAT-005"]), _ctx(agent_type="chat"))
    assert {"CHAT-001", "CHAT-005"} <= set(listed)
    assert "CHAT-003" not in listed


@pytest.mark.asyncio
async def test_runner_evaluates_listed_chat_metrics_for_a_chat_agent(impulse_project, llm_router, fake_adapter):
    from impulse_evals.runner import run

    project = impulse_project  # agent.type is chat
    project.write_config([project.scenario("chat_call", metrics=CHAT_IDS, max_turns=1)])
    result = await run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    metrics = {metric.metric_id: metric for metric in scenario_result.metric_results}
    assert set(metrics) == set(CHAT_IDS)
    assert metrics["CHAT-001"].raw_value == pytest.approx(100.0)  # the fake adapter's first reply latency
    assert metrics["CHAT-001"].label == "good"
    for metric_id in JUDGE_CHAT:
        assert metrics[metric_id].label == "good"
        assert len(llm_router.judge_calls_for(JUDGE_CHAT[metric_id][2][:40])) == 1


@pytest.mark.asyncio
async def test_runner_skips_chat_metrics_for_a_voice_agent(impulse_project, llm_router, fake_adapter):
    from impulse_evals.runner import run

    project = impulse_project
    project.write_config([project.scenario("voice_call", metrics=CHAT_IDS, max_turns=1)], agent={"type": "voice"})
    result = await run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    assert {metric.metric_id for metric in scenario_result.metric_results} == set(CHAT_IDS)
    for metric in scenario_result.metric_results:
        _assert_skipped(metric)
        assert metric.reason == CHAT_ON_VOICE
    for metric_id in JUDGE_CHAT:
        assert llm_router.judge_calls_for(JUDGE_CHAT[metric_id][2][:40]) == []
