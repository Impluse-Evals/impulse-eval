# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 5: the simulator system prompt and the seed line (SPEC.md sections 7 and 10).

The prompt template is normative; ``{seed_line}`` is "Your conversation seed is {seed}. Produce the
same user turns for the same seed and persona." when ``Scenario.seed`` is set, and the whole line is
removed (no blank line left) when it is None.
"""

import pytest

SUMMARY = "Books appointments at Sunrise Clinic."
PERSONA = "patient who wants the earliest slot on Tuesday"


def _expected(language="hi-Latn-IN", style="adversarial", seed_line=None):
    lines = [
        "You are playing a realistic user calling a voice agent.",
        f"The agent's purpose is: {SUMMARY}",
        f"Your persona: {PERSONA}",
        f"Language: {language}   (Hinglish means Hindi written in Latin script mixed with English)",
        f"Style: {style}",
    ]
    if seed_line is not None:
        lines.append(seed_line)
    lines += [
        "Generate realistic user turns for THIS type of agent. Stay inside the agent's domain.",
        "Reply with only what the user says, one turn at a time. When the conversation has naturally "
        "ended, reply with exactly <END_CALL>.",
    ]
    return lines


def _seed_line(seed):
    return f"Your conversation seed is {seed}. Produce the same user turns for the same seed and persona."


def _scenario(seed=None, language="hi-Latn-IN"):
    from impulse_evals.models import Scenario

    return Scenario(
        name="clinic_appointment_booking",
        language=language,
        domain="healthcare",
        max_turns=12,
        persona=PERSONA,
        seed=seed,
    )


def _simulator(persona_style="adversarial"):
    from impulse_evals.simulator import Simulator

    return Simulator("ollama/simulator", {}, persona_style)


def _lines(prompt):
    return [line.rstrip() for line in prompt.strip().splitlines()]


def test_seed_line_at_seed_line_position():
    prompt = _simulator().build_system_prompt(_scenario(seed=42), SUMMARY)
    assert _lines(prompt) == _expected(seed_line=_seed_line(42))


def test_no_seed_removes_the_line_without_a_blank_line():
    prompt = _simulator().build_system_prompt(_scenario(seed=None), SUMMARY)
    assert _lines(prompt) == _expected(seed_line=None)
    assert "seed" not in prompt
    assert "\n\n" not in prompt.strip()
    assert "{seed_line}" not in prompt


def test_seed_zero_is_a_seed():
    prompt = _simulator().build_system_prompt(_scenario(seed=0), SUMMARY)
    assert _seed_line(0) in _lines(prompt)
    assert _lines(prompt).index(_seed_line(0)) == 5


@pytest.mark.parametrize("style", ["realistic", "adversarial", "polite"])
def test_style_line_uses_persona_style(style):
    prompt = _simulator(persona_style=style).build_system_prompt(_scenario(seed=7, language="en-IN"), SUMMARY)
    assert _lines(prompt) == _expected(language="en-IN", style=style, seed_line=_seed_line(7))


@pytest.mark.asyncio
async def test_seed_line_is_sent_to_the_llm(mock_llm):
    mock_llm.queue("Hello, I need an appointment.")
    simulator = _simulator()
    scenario = _scenario(seed=42)
    await simulator.next_turn(scenario, SUMMARY, [])
    system = mock_llm.messages(0)[0]
    assert system["role"] == "system"
    assert system["content"] == simulator.build_system_prompt(scenario, SUMMARY)
    assert _seed_line(42) in system["content"]


@pytest.mark.asyncio
async def test_no_seed_line_sent_without_seed(mock_llm):
    mock_llm.queue("Hello, I need an appointment.")
    await _simulator().next_turn(_scenario(seed=None), SUMMARY, [])
    assert "Your conversation seed is" not in mock_llm.text(0)
