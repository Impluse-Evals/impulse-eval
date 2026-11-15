# SPDX-License-Identifier: Apache-2.0
"""Phase 6: inject_events are modelled but not executed in v1 (SPEC.md sections 7 and 16).

The runner prints the note once per affected scenario to the console, copies the events into
``ScenarioResult.inject_events``, runs the scenario without injection, and never writes to
``DECISIONS.md``. The note in the HTML report is checked in ``test_reporter_fields.py``.
"""

from pathlib import Path

import pytest

NOTE = "inject_events defined but execution reserved for v2; this scenario ran without injection."
REPO_DECISIONS = Path(__file__).resolve().parent.parent / "DECISIONS.md"
EVENTS = [
    {
        "at_turn": 1,
        "type": "tool_error",
        "tool": "book_slot",
        "error_message": "Booking service unavailable",
        "note": "the first booking attempt fails",
    },
    {"at_turn": 2, "type": "silence", "silence_ms": 3000},
]


def _m():
    from impulse_evals import models

    return models


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


def _project(project, llm_router):
    project.write_agent()  # copy of examples/sample_agent.py
    project.write_config(
        [
            project.scenario(
                "injected_booking",
                opening_user_message="Please book me for Tuesday.",
                max_turns=3,
                metrics=["UNIV-013"],
                inject_events=EVENTS,
            ),
            project.scenario("plain_call", max_turns=2, metrics=["UNIV-013"]),
            project.scenario(
                "injected_transfer",
                max_turns=2,
                metrics=["UNIV-013"],
                inject_events=[{"at_turn": 2, "type": "transfer_fail", "note": "handoff to a human fails"}],
            ),
        ]
    )
    llm_router.script(project.persona("injected_booking"), "Tuesday morning is fine.", "<END_CALL>")
    return project.load()


@pytest.mark.asyncio
async def test_scenario_with_inject_events_runs_without_crashing(impulse_project, llm_router):
    result = await _run(_project(impulse_project, llm_router))
    assert result.total_scenarios == 3
    for scenario_result in result.scenario_results:
        assert scenario_result.error is None
        assert scenario_result.transcript.turns


@pytest.mark.asyncio
async def test_note_printed_once_per_affected_scenario(impulse_project, llm_router, capsys, normalize_console):
    await _run(_project(impulse_project, llm_router))
    captured = capsys.readouterr()
    console = normalize_console(captured.out + "\n" + captured.err)
    assert console.count(NOTE) == 2


@pytest.mark.asyncio
async def test_scenario_ran_without_injection(impulse_project, llm_router):
    m = _m()
    result = await _run(_project(impulse_project, llm_router))
    injected = result.scenario_results[0]
    turns = injected.transcript.turns
    # No tool_error was injected: the sample agent's book_slot call still succeeds.
    assert turns[1].tool_calls == [
        m.ToolCall(
            name="book_slot",
            arguments={"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"},
            result={"success": True},
            success=True,
        )
    ]
    assert [turn.text for turn in turns if turn.speaker == "user"] == [
        "Please book me for Tuesday.",
        "Tuesday morning is fine.",
    ]


@pytest.mark.asyncio
async def test_inject_events_copied_into_scenario_result_and_round_trip(impulse_project, llm_router):
    m = _m()
    result = await _run(_project(impulse_project, llm_router))
    by_name = {scenario.scenario_name: scenario for scenario in result.scenario_results}
    injected = by_name["injected_booking"]
    assert injected.inject_events == [m.InjectEvent(**event) for event in EVENTS]
    assert all(isinstance(event, m.InjectEvent) for event in injected.inject_events)
    assert by_name["plain_call"].inject_events == []
    assert by_name["injected_transfer"].inject_events == [
        m.InjectEvent(at_turn=2, type="transfer_fail", note="handoff to a human fails")
    ]
    restored = m.ScenarioResult.model_validate_json(injected.model_dump_json())
    assert restored == injected
    assert restored.inject_events == injected.inject_events
    assert m.RunResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.asyncio
async def test_nothing_written_to_decisions_md(impulse_project, llm_router):
    cfg = _project(impulse_project, llm_router)
    before = REPO_DECISIONS.read_bytes() if REPO_DECISIONS.exists() else None
    files_before = impulse_project.files()
    await _run(cfg)
    after = REPO_DECISIONS.read_bytes() if REPO_DECISIONS.exists() else None
    assert after == before
    assert not (impulse_project.root / "DECISIONS.md").exists()
    assert impulse_project.files() == files_before
