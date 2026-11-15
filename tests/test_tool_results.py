# SPDX-License-Identifier: Apache-2.0
"""Fixes round 2, A1 (DEFECT-1): tool results reach the judge, and success is detected from the
common result shapes (FIXES-ROUND-2-SPEC.md section A1).

* A1.1 ``impulse_evals.tool_results.infer_success(call) -> ToolCall``: a copy, never a mutation;
  the runner applies it to every tool call of every ``AgentReply`` (every adapter, custom ones too),
  so a developer's unchanged tools (``{"ok": true, ...}``, ``{"error": "..."}``, success None) need
  no ``success`` key any more (the campaign's W1 workaround is not needed).
* A1.2 the runner passes every string of each tool call's ``result`` (keys and values, recursively)
  and its ``error`` through the run's key redactor before storing the turn.
* A1.3 ``transcriber.format_tool_call``: `` result={json}`` after ``success=`` (and `` error=``) when
  the result is shown (not None, not flag-only); key-name redaction with
  ``transcriber.redact_tool_result``; 300-character cut with ``...(truncated)``; the HTML report's
  tool-call lines show it too.
* A1.4 ``judge.prompts.TOOL_CALL_NOTE`` right after ``</transcript>`` (one blank line before it),
  once, only when a turn has a tool call.

A1/A7-rev (review R1, orchestrator decisions; tests at the end of this file):

* R1-M2: on the AGENT TOOL CALL line and in the report's trace-table error column, an ``error``
  whose text parses as a JSON object is shown as ``json_sorted(redact_tool_result(object))``; every
  shown error is cut like the result (300 characters + ``...(truncated)``). The stored
  ``ToolCall.error`` is unchanged.
* R1-L1: the runner stores the redacted tool result whatever its type. R1-L6: a self-referencing
  result never ends the scenario; the circular part becomes ``"[circular reference]"``.
* R1-L3: plural sensitive tokens (``passwords``, ``tokens``, ``secrets``, ``cookies``, ``otps``,
  ``pins``), ``mpin``, ``tpin``, the pair ``card no``; ``OTPs`` -> ``otps``, ``PINs`` -> ``pins``,
  ``userPIN`` -> ``user``, ``pin``.
* R1-L5: the error fill order of A1.1: ``error`` string, ``errors`` string, first string item of an
  ``errors`` list, ``message`` string inside an ``error`` dict.

Re-check revision N-L3 (orchestrator decision): a sensitive key whose value is ``true``, ``false`` or
``null`` is shown as is (outcomes such as ``otp_verified`` matter to UNIV-023 and the KYC rules);
every other value of a sensitive key is ``[REDACTED]``. Superseded:
``test_redact_tool_result_at_any_depth_and_whatever_the_value_type`` expected ``{"password": None}``
to become ``[REDACTED]``; it now stays ``None``.

Campaign data (read-only copies in tests/data/fixes_round2/judge/): SupportBot run
20261021T191701Z-e4b05f (``lookup_order`` result ``{"ok": true, ...}`` with no ``success`` key, stored
with success None) and run 20261023T193004Z-a9be47 (``order_status``: the agent repeats the item and
the delivery date from the ``lookup_order`` result; the judge called that fabricated).
"""

import copy
import html
import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent / "data" / "fixes_round2" / "judge"

TOOL_CALL_NOTE = (
    "About the AGENT TOOL CALL lines: result= is the data the tool returned to the agent (shortened, "
    "secrets removed), so details the agent repeats from it are not fabricated. success=True or "
    "success=False is the outcome Impulse captured; success=None means the outcome was not captured: it is "
    "not a failure and not an empty result."
)
REDACTED = "[REDACTED]"
TRUNCATED = "...(truncated)"

LOOKUP_ARGS_JSON = '{"order_id": "ORD-1004"}'
# json_sorted of the e4b05f lookup_order result (no success key).
E4B05F_RESULT_JSON = (
    '{"amount_inr": 7999, "delivered_on": null, "expected_on": "2026-10-25", "item": "Smart watch", '
    '"ok": true, "order_id": "ORD-1004", "status": "shipped"}'
)
E4B05F_LINE = (
    f"[turn 4] AGENT TOOL CALL: lookup_order({LOOKUP_ARGS_JSON}) -> success=True result={E4B05F_RESULT_JSON}"
)
# json_sorted of the a9be47 lookup_order result (the campaign's W1 workaround added "success": true).
A9BE47_RESULT_JSON = (
    '{"amount_inr": 7999, "delivered_on": null, "expected_on": "2026-10-25", "item": "Smart watch", '
    '"ok": true, "order_id": "ORD-1004", "status": "shipped", "success": true}'
)
A9BE47_LINE = (
    f"[turn 4] AGENT TOOL CALL: lookup_order({LOOKUP_ARGS_JSON}) -> success=True result={A9BE47_RESULT_JSON}"
)


def _m():
    from impulse_evals import models

    return models


def _infer(call):
    from impulse_evals.tool_results import infer_success

    return infer_success(call)


def _call(name="lookup_order", arguments=None, **fields):
    return _m().ToolCall(name=name, arguments={"order_id": "ORD-1004"} if arguments is None else arguments, **fields)


def _line(call, number=2):
    from impulse_evals.transcriber import format_tool_call

    return format_tool_call(number, call)


def _redact_result(value):
    from impulse_evals.transcriber import redact_tool_result

    return redact_tool_result(value)


def _format(transcript):
    from impulse_evals.transcriber import format_transcript

    return format_transcript(transcript)


def _load(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _campaign_transcript(name):
    return _m().CallTranscript.model_validate(_load(name)["transcript"])


def _transcript(turns, system_prompt="You are the support assistant of ShopKart."):
    return _m().CallTranscript(
        turns=turns,
        duration_ms=5000,
        metadata={},
        scenario_name="tool_results_probe",
        language="en-IN",
        agent_system_prompt=system_prompt,
    )


def _turn(speaker, text, timestamp_ms, **fields):
    return _m().Turn(speaker=speaker, text=text, timestamp_ms=timestamp_ms, **fields)


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


def _metrics(scenario_result):
    return {metric.metric_id: metric for metric in scenario_result.metric_results}


def _user_message(llm_router, call):
    return next(text for text in llm_router.contents(call) if "EVALUATION CRITERION:" in text)


# ============================================================================================
# A1.1 infer_success: the spec's examples, then the rules one by one
# ============================================================================================

SPEC_EXAMPLES = [
    pytest.param({"result": {"ok": True, "item": "Smart Watch"}}, True, None, id="ok-true"),
    pytest.param({"result": {"error": "slot taken"}}, False, "slot taken", id="error-string"),
    pytest.param({"result": {"ok": False, "error": "order not found"}}, False, "order not found", id="ok-false-error"),
    pytest.param({"result": {"status": "failed"}}, False, None, id="status-failed"),
    pytest.param({"result": {"status": "pending"}}, None, None, id="status-pending"),
    pytest.param({"result": {"value": "Booked"}}, None, None, id="value-only"),
    pytest.param({"result": {"errors": []}}, None, None, id="errors-empty-list"),
    pytest.param({"result": {"error": None, "ok": True}}, True, None, id="error-null-ok-true"),
    pytest.param({"result": {"ok": True}, "success": False}, False, None, id="explicit-success-wins"),
]


@pytest.mark.parametrize("fields, success, error", SPEC_EXAMPLES)
def test_infer_success_spec_examples(fields, success, error):
    inferred = _infer(_call(**fields))
    assert inferred.success is success
    assert inferred.error == error


RULE_CASES = [
    # rule 1: an explicit success is never changed, whatever the result says
    pytest.param({"success": True, "result": {"error": "slot taken"}}, True, None, id="r1-explicit-true"),
    pytest.param({"success": False, "error": "timeout", "result": {"ok": True}}, False, "timeout", id="r1-explicit-false"),
    # rule 2: a non-empty error string means failure; the error itself is kept as it is
    pytest.param({"error": "connection reset"}, False, "connection reset", id="r2-error-no-result"),
    pytest.param({"error": "timeout", "result": {"ok": True}}, False, "timeout", id="r2-before-result-ok"),
    pytest.param({"error": "timeout", "result": {"error": "other"}}, False, "timeout", id="r2-keeps-own-error"),
    pytest.param({"error": ""}, None, "", id="r2-empty-error-is-not-set"),
    # rule 3a: success / ok with a boolean value, case-insensitive after strip(), first in dict order
    pytest.param({"result": {"Success": True}}, True, None, id="r3a-case"),
    pytest.param({"result": {" OK ": False}}, False, None, id="r3a-strip"),
    pytest.param({"result": {"ok": False, "success": True}}, False, None, id="r3a-first-key-wins"),
    pytest.param({"result": {"success": "yes"}}, None, None, id="r3a-not-boolean"),
    pytest.param({"result": {"ok": 1}}, None, None, id="r3a-int-is-not-boolean"),
    pytest.param({"result": {"data": {"ok": True}}}, None, None, id="r3a-top-level-only"),
    # rule 3b: error / errors with a truthy error value
    pytest.param({"result": {"error": "  slot taken  "}}, False, "slot taken", id="r3b-stripped"),
    pytest.param({"result": {"error": "   "}}, None, None, id="r3b-blank-string"),
    pytest.param({"result": {"error": True}}, False, None, id="r3b-true"),
    pytest.param({"result": {"error": False}}, None, None, id="r3b-false"),
    pytest.param({"result": {"Errors": ["card declined", "retry later"]}}, False, "card declined", id="r3b-errors-list"),
    pytest.param({"result": {"errors": [{"code": 7}, "second"]}}, False, "second", id="r3b-first-string-item"),
    pytest.param({"result": {"error": {"message": "card declined", "code": 402}}}, False, "card declined", id="r3b-dict-message"),
    pytest.param({"result": {"error": {"code": 402}}}, False, None, id="r3b-dict-without-message"),
    pytest.param({"result": {"errors": "quota exceeded"}}, False, "quota exceeded", id="r3b-errors-string"),
    pytest.param({"result": {"error": "service down", "ok": True}}, True, None, id="r3a-before-r3b"),
    pytest.param({"error": "", "result": {"error": "slot taken"}}, False, "slot taken", id="r3b-fills-empty-error"),
    # rule 3c: status / state strings
    pytest.param({"result": {"status": " Completed "}}, True, None, id="r3c-completed"),
    pytest.param({"result": {"state": "FAILURE"}}, False, None, id="r3c-state-failure"),
    pytest.param({"result": {"status": "ok"}}, True, None, id="r3c-ok"),
    pytest.param({"result": {"status": "done"}}, True, None, id="r3c-done"),
    pytest.param({"result": {"status": "error"}}, False, None, id="r3c-error"),
    pytest.param({"result": {"status": "shipped"}}, None, None, id="r3c-other-word"),
    pytest.param({"result": {"status": 200}}, None, None, id="r3c-not-a-string"),
    pytest.param({"result": {"error": "slot taken", "status": "success"}}, False, "slot taken", id="r3b-before-r3c"),
    # rule 4: nothing known
    pytest.param({}, None, None, id="r4-no-result"),
    pytest.param({"result": {}}, None, None, id="r4-empty-result"),
]


@pytest.mark.parametrize("fields, success, error", RULE_CASES)
def test_infer_success_rules(fields, success, error):
    inferred = _infer(_call(**fields))
    assert inferred.success is success
    assert inferred.error == error


def test_infer_success_cuts_a_long_filled_error_to_200_characters():
    long_error = "E" * 250
    inferred = _infer(_call(result={"error": long_error}))
    assert inferred.success is False
    assert inferred.error == "E" * 197 + "..."
    assert len(inferred.error) == 200
    exact = _infer(_call(result={"error": "F" * 200}))
    assert exact.error == "F" * 200


def test_infer_success_returns_a_copy_and_never_mutates():
    m = _m()
    call = m.ToolCall(name="book_slot", arguments={"date": "2026-01-01"}, result={"error": "slot taken"})
    before = call.model_copy(deep=True)
    inferred = _infer(call)
    assert isinstance(inferred, m.ToolCall)
    assert inferred is not call
    assert call == before   # unchanged: success None, error None
    assert (inferred.name, inferred.arguments, inferred.result) == (call.name, call.arguments, call.result)
    assert (inferred.success, inferred.error) == (False, "slot taken")


def test_infer_success_on_the_campaign_lookup_order_call():
    transcript = _campaign_transcript("supportbot_order_status_e4b05f.json")
    call = transcript.turns[3].tool_calls[0]
    assert call.success is None and call.error is None   # as the campaign stored it
    inferred = _infer(call)
    assert inferred.success is True
    assert inferred.error is None
    assert inferred.result == call.result


# ============================================================================================
# A1.1 through the runner: every adapter; a developer's unchanged tools need no success key
# ============================================================================================


@pytest.mark.asyncio
async def test_runner_infers_success_for_a_custom_adapter_replaying_the_campaign(
    impulse_project, llm_router, fake_adapter
):
    """Run 20261021T191701Z-e4b05f replayed: lookup_order returned {"ok": true, ...} and no success key."""
    project = impulse_project
    campaign = _load("supportbot_order_status_e4b05f.json")
    turns = campaign["transcript"]["turns"]
    name = "order_status"
    project.write_config(
        [
            project.scenario(
                name,
                domain="ecommerce",
                max_turns=3,
                opening_user_message=turns[0]["text"],
                expected_outcome=campaign["scenario"]["expected_outcome"],
                metrics=["UNIV-005", "UNIV-023"],
                expected_tool_calls=[{"name": "lookup_order", "required_arguments": ["order_id"]}],
            )
        ],
        system_prompt=campaign["transcript"]["agent_system_prompt"],
    )
    llm_router.script(project.persona(name), turns[2]["text"], turns[4]["text"])
    fake_adapter.replies[name] = [
        {"text": turn["text"], "tool_calls": copy.deepcopy(turn["tool_calls"])} for turn in turns if turn["speaker"] == "agent"
    ]
    assert fake_adapter.replies[name][1]["tool_calls"][0]["success"] is None   # the developer's unchanged tool

    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    stored = scenario_result.transcript.turns[3].tool_calls
    assert len(stored) == 1
    assert stored[0].success is True
    assert stored[0].error is None
    assert stored[0].result == turns[3]["tool_calls"][0]["result"]
    assert E4B05F_LINE in _format(scenario_result.transcript).splitlines()

    metrics = _metrics(scenario_result)
    assert metrics["UNIV-025"].label == "good"
    assert metrics["UNIV-025"].reason == "all 1 tool call expectations met"   # was "success not captured"

    univ023 = llm_router.judge_calls_for("UNIV-023")
    assert len(univ023) == 1
    _, judged = llm_router.judge_sections(univ023[0])
    assert E4B05F_LINE in judged.splitlines()
    user = _user_message(llm_router, univ023[0])
    assert "</transcript>\n\n" + TOOL_CALL_NOTE + "\n\nEVALUATION CRITERION:\n" in user
    assert user.count(TOOL_CALL_NOTE) == 1


@pytest.mark.asyncio
async def test_runner_infers_failure_from_an_error_result_and_univ025_sees_it(
    impulse_project, llm_router, fake_adapter
):
    project = impulse_project
    name = "slot_taken"
    project.write_config(
        [
            project.scenario(
                name,
                max_turns=1,
                metrics=["UNIV-023"],
                expected_tool_calls=[{"name": "book_slot", "required_arguments": ["date"]}],
            )
        ]
    )
    fake_adapter.replies[name] = [
        {
            "text": "That slot is taken, sorry.",
            "tool_calls": [{"name": "book_slot", "arguments": {"date": "2026-01-01"}, "result": {"error": "slot taken"}}],
        }
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    call = scenario_result.transcript.turns[1].tool_calls[0]
    assert (call.success, call.error) == (False, "slot taken")
    line = (
        '[turn 2] AGENT TOOL CALL: book_slot({"date": "2026-01-01"}) -> success=False error=slot taken '
        'result={"error": "slot taken"}'
    )
    assert line in _format(scenario_result.transcript).splitlines()
    _, judged = llm_router.judge_sections(llm_router.judge_calls_for("UNIV-023")[0])
    assert line in judged.splitlines()
    univ025 = _metrics(scenario_result)["UNIV-025"]
    assert univ025.label == "poor"
    assert univ025.reason == "tool book_slot call failed: slot taken"


@pytest.mark.asyncio
async def test_runner_keeps_an_explicit_success_and_unknown_shapes(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    name = "explicit_and_unknown"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-023"])])
    fake_adapter.replies[name] = [
        {
            "text": "Done.",
            "tool_calls": [
                {"name": "pay", "arguments": {"amount": 100}, "result": {"ok": True}, "success": False},
                {"name": "hold_slot", "arguments": {}, "result": {"status": "pending"}},
                {"name": "notify", "arguments": {}, "result": {"status": "Completed"}},
            ],
        }
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    calls = result.scenario_results[0].transcript.turns[1].tool_calls
    assert [(call.name, call.success) for call in calls] == [("pay", False), ("hold_slot", None), ("notify", True)]


GENERIC_AGENT = """
def respond(user_text, history):
    if "ORD-1004" in (user_text or ""):
        return {
            "text": "Your order for the Smart watch has been shipped.",
            "tool_calls": [
                {"name": "lookup_order", "arguments": {"order_id": "ORD-1004"},
                 "result": {"ok": True, "item": "Smart watch", "expected_on": "2026-10-25"}},
                {"name": "create_ticket", "arguments": {"summary": "late order"},
                 "result": {"ok": False, "error": "ticket service down"}},
            ],
        }
    return "Could you give me your order ID?"
"""


@pytest.mark.asyncio
async def test_runner_infers_success_for_the_generic_adapter(impulse_project, llm_router):
    project = impulse_project
    project.write_agent(GENERIC_AGENT)
    name = "generic_order"
    project.write_config(
        [project.scenario(name, max_turns=1, opening_user_message="My order ID is ORD-1004.", metrics=["UNIV-023"])]
    )
    result = await _run(project.load())
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    calls = scenario_result.transcript.turns[1].tool_calls
    assert [(call.name, call.success, call.error) for call in calls] == [
        ("lookup_order", True, None),
        ("create_ticket", False, "ticket service down"),
    ]
    lines = _format(scenario_result.transcript).splitlines()
    assert (
        '[turn 2] AGENT TOOL CALL: lookup_order({"order_id": "ORD-1004"}) -> success=True '
        'result={"expected_on": "2026-10-25", "item": "Smart watch", "ok": true}'
    ) in lines
    assert (
        '[turn 2] AGENT TOOL CALL: create_ticket({"summary": "late order"}) -> success=False '
        'error=ticket service down result={"error": "ticket service down", "ok": false}'
    ) in lines


# ============================================================================================
# A1.2 the runner redacts known key values in tool results and errors before storing the turn
# ============================================================================================

TOOL_SECRET = "sk-impulse-test-SECRET-toolresult-7731"   # ANTHROPIC_API_KEY, a standard key variable


@pytest.mark.asyncio
async def test_runner_redacts_known_keys_in_tool_results_and_errors(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", TOOL_SECRET)
    project = impulse_project
    name = "leaky_tool"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-023"])])
    fake_adapter.replies[name] = [
        {
            "text": "I could not reach the booking system.",
            "tool_calls": [
                {
                    "name": "book_slot",
                    "arguments": {"date": "2026-01-01"},
                    "result": {
                        "echo": f"token {TOOL_SECRET}",
                        f"key {TOOL_SECRET}": 1,
                        "nested": [{"deep": TOOL_SECRET}, 7, None, True],
                    },
                    "success": False,
                    "error": f"denied for {TOOL_SECRET}",
                },
                {
                    "name": "lookup_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"ok": False, "error": f"bad credential {TOOL_SECRET}"},
                },
            ],
        }
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    first, second = scenario_result.transcript.turns[1].tool_calls
    assert first.result == {
        "echo": f"token {REDACTED}",
        f"key {REDACTED}": 1,
        "nested": [{"deep": REDACTED}, 7, None, True],
    }
    assert first.error == f"denied for {REDACTED}"
    assert first.success is False
    assert second.result == {"ok": False, "error": f"bad credential {REDACTED}"}
    assert (second.success, second.error) == (False, f"bad credential {REDACTED}")
    assert TOOL_SECRET not in result.model_dump_json()
    for call in llm_router.calls():
        assert all(TOOL_SECRET not in text for text in llm_router.contents(call))


# ============================================================================================
# A1.3 the AGENT TOOL CALL line
# ============================================================================================


def test_format_line_shows_the_result_after_success():
    call = _call(result={"ok": True, "item": "Smart watch", "expected_on": "2026-10-25"}, success=True)
    assert _line(call) == (
        '[turn 2] AGENT TOOL CALL: lookup_order({"order_id": "ORD-1004"}) -> success=True '
        'result={"expected_on": "2026-10-25", "item": "Smart watch", "ok": true}'
    )


def test_format_line_result_comes_after_the_error():
    call = _call("book_slot", {"date": "2026-01-01"}, result={"error": "slot taken"}, success=False, error="slot taken")
    assert _line(call) == (
        '[turn 2] AGENT TOOL CALL: book_slot({"date": "2026-01-01"}) -> success=False error=slot taken '
        'result={"error": "slot taken"}'
    )


@pytest.mark.parametrize("success, shown", [(True, "True"), (False, "False"), (None, "None")])
def test_format_line_without_result_is_unchanged(success, shown):
    call = _call(success=success)
    assert _line(call) == f'[turn 2] AGENT TOOL CALL: lookup_order({{"order_id": "ORD-1004"}}) -> success={shown}'


@pytest.mark.parametrize(
    "result",
    [
        pytest.param({"success": True}, id="success"),
        pytest.param({"ok": False}, id="ok"),
        pytest.param({"Success": True, "OK": True}, id="both-any-case"),
        pytest.param({" ok ": True}, id="stripped"),
    ],
)
def test_format_line_omits_a_flag_only_result(result):
    call = _call(result=result, success=True)
    line = _line(call)
    assert line == '[turn 2] AGENT TOOL CALL: lookup_order({"order_id": "ORD-1004"}) -> success=True'
    assert " result=" not in line


@pytest.mark.parametrize(
    "result, shown",
    [
        pytest.param({}, "{}", id="empty-dict"),
        pytest.param({"success": "yes"}, '{"success": "yes"}', id="flag-not-boolean"),
        pytest.param({"success": True, "id": 1}, '{"id": 1, "success": true}', id="flag-plus-data"),
        pytest.param({"item": "नमस्ते"}, '{"item": "नमस्ते"}', id="unicode-kept"),
    ],
)
def test_format_line_shows_other_results(result, shown):
    call = _call(result=result, success=None)
    assert _line(call) == (
        f'[turn 2] AGENT TOOL CALL: lookup_order({{"order_id": "ORD-1004"}}) -> success=None result={shown}'
    )


def test_format_line_cuts_a_long_result_at_300_characters():
    long_result = {"note": "x" * 400}   # json_sorted: '{"note": "' + 400 x + '"}' (412 characters)
    line = _line(_call(result=long_result, success=True))
    prefix = '[turn 2] AGENT TOOL CALL: lookup_order({"order_id": "ORD-1004"}) -> success=True result='
    assert line == prefix + '{"note": "' + "x" * 290 + TRUNCATED


@pytest.mark.parametrize("length, cut", [(288, False), (289, True)])
def test_format_line_cut_boundary(length, cut):
    result_text = '{"note": "' + "y" * length + '"}'   # 300 characters for 288, 301 for 289
    line = _line(_call(result={"note": "y" * length}, success=True))
    assert " result=" in line, line
    shown = line.split(" result=", 1)[1]
    if cut:
        assert shown == result_text[:300] + TRUNCATED
    else:
        assert shown == result_text


def test_format_line_redacts_before_cutting():
    line = _line(_call(result={"token": "z" * 400, "status": "sent"}, success=True))
    assert line.endswith(' result={"status": "sent", "token": "[REDACTED]"}')
    assert TRUNCATED not in line


def test_format_line_never_redacts_or_cuts_arguments():
    arguments = {"otp": "123456", "note": "a" * 400}
    line = _line(_call("verify_otp", arguments, result={"otp": "123456", "status": "sent"}, success=True))
    assert line == (
        "[turn 2] AGENT TOOL CALL: verify_otp("
        + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        + ') -> success=True result={"otp": "[REDACTED]", "status": "sent"}'
    )


SENSITIVE_KEYS = [
    "otp", "user_pin", "accessToken", "api-key", "card_number", "pin_code", "PIN", "password", "passwd",
    "passcode", "pwd", "cvv", "CVC", "client_secret", "token", "apikey", "apiKey", "Authorization", "set-cookie",
    "x-api-key", "accessKeyId", "private_key", "privateKey", "cardNumber", "card_number_last4", "X-Auth-Token",
]
SAFE_KEYS = ["shipping", "pincode", "card_last4", "keyId", "spin", "opinion", "monkey", "api", "key", "number", "card"]


@pytest.mark.parametrize("key", SENSITIVE_KEYS)
def test_redact_tool_result_hides_sensitive_key_names(key):
    assert _redact_result({key: "s3cr3t-value", "item": "Smart watch"}) == {key: REDACTED, "item": "Smart watch"}


@pytest.mark.parametrize("key", SAFE_KEYS)
def test_redact_tool_result_keeps_other_key_names(key):
    assert _redact_result({key: "411001"}) == {key: "411001"}


def test_redact_tool_result_at_any_depth_and_whatever_the_value_type():
    result = {
        "customer": {"name": "Ananya", "pin": 4821, "cards": [{"cvv": 123, "last4": "4821"}, {"otp": {"code": 1}}]},
        "items": [[{"password": None}], "plain"],
        "status": "ok",
    }
    original = copy.deepcopy(result)
    assert _redact_result(result) == {
        "customer": {"name": "Ananya", "pin": REDACTED, "cards": [{"cvv": REDACTED, "last4": "4821"}, {"otp": REDACTED}]},
        # Re-check N-L3 (supersedes REDACTED here): a null under a sensitive key is shown as is.
        "items": [[{"password": None}], "plain"],
        "status": "ok",
    }
    assert result == original   # never mutated


def test_format_line_redacts_sensitive_keys_in_the_result():
    call = _call("verify_identity", {"customer_id": "CUST-2001"}, result={"ok": True, "otp": "482913", "user": {"apiKey": "k-1"}})
    assert _line(call, 3) == (
        '[turn 3] AGENT TOOL CALL: verify_identity({"customer_id": "CUST-2001"}) -> success=None '
        'result={"ok": true, "otp": "[REDACTED]", "user": {"apiKey": "[REDACTED]"}}'
    )


def test_html_report_tool_call_line_shows_the_result(run_builder, tmp_path):
    from impulse_evals.reporter import render_html

    call = _m().ToolCall(
        name="lookup_order",
        arguments={"order_id": "ORD-1004"},
        result={"ok": True, "item": "Smart watch", "expected_on": "2026-10-25"},
        success=True,
    )
    transcript = run_builder.transcript(
        "order_status",
        [
            run_builder.turn("user", "Where is my order ORD-1004?", 0),
            run_builder.turn("agent", "It is shipped.", 900, latency_ms=900.0, tool_calls=[call]),
        ],
    )
    run = run_builder.run([run_builder.scenario_result("order_status", transcript=transcript, metrics=[("UNIV-023", 1.0)])])
    page = render_html(run, embed_audio=False, run_dir=tmp_path)
    line = (
        '[turn 2] AGENT TOOL CALL: lookup_order({"order_id": "ORD-1004"}) -> success=True '
        'result={"expected_on": "2026-10-25", "item": "Smart watch", "ok": true}'
    )
    assert html.escape(line, quote=True) in page


def test_html_report_tool_call_trace_table_redacts_sensitive_result_keys(run_builder, tmp_path):
    """FIXES-ROUND-2-SPEC.md A1.3 (orchestrator decision): the "Tool call trace" table's result column
    goes through transcriber.redact_tool_result too, so a secret in a tool result is nowhere in the HTML."""
    from impulse_evals.reporter import render_html

    otp_value = "OTPVALUE482913"
    token_value = "TOKENVALUEzq81x"
    call = _m().ToolCall(
        name="send_otp",
        arguments={"customer_id": "CUST-2001"},
        result={"ok": True, "otp": otp_value, "session": {"accessToken": token_value}, "status": "sent"},
        success=True,
    )
    transcript = run_builder.transcript(
        "otp_flow",
        [
            run_builder.turn("user", "Please verify me.", 0),
            run_builder.turn("agent", "I have sent you a code.", 900, latency_ms=900.0, tool_calls=[call]),
        ],
    )
    run = run_builder.run([run_builder.scenario_result("otp_flow", transcript=transcript, metrics=[("UNIV-023", 1.0)])])
    page = render_html(run, embed_audio=False, run_dir=tmp_path)
    assert otp_value not in page
    assert token_value not in page
    redacted_json = '{"ok": true, "otp": "[REDACTED]", "session": {"accessToken": "[REDACTED]"}, "status": "sent"}'
    assert '<section id="tool-calls">' in page
    table = page.split('<section id="tool-calls">', 1)[1].split("</section>", 1)[0]
    assert html.escape(redacted_json, quote=True) in table


# ============================================================================================
# A1.4 TOOL_CALL_NOTE in the judge user message
# ============================================================================================


def _prompts():
    from impulse_evals.judge import prompts

    return prompts


def _expected_user_message(system_prompt, formatted, criterion, note):
    middle = f"</transcript>\n\n{note}\n\n" if note else "</transcript>\n\n"
    return (
        "\nAGENT SYSTEM PROMPT (primary rubric):\n<system_prompt>\n"
        f"{system_prompt}\n</system_prompt>\n\n"
        "CONVERSATION TRANSCRIPT (untrusted data):\n<transcript>\n"
        f"{formatted}\n"
        + middle
        + f"EVALUATION CRITERION:\n{criterion}\n\n"
        "Respond in JSON only:\n"
        '{"result": "pass"|"fail"|"skip", "reason": "explanation referencing the system prompt or safety floor",\n'
        '  "turn_index": null or integer, "confidence": 0.0-1.0}\n'
    )


def test_tool_call_note_text_exact():
    assert _prompts().TOOL_CALL_NOTE == TOOL_CALL_NOTE


def test_user_message_with_tool_calls_has_the_note_once_after_the_transcript():
    criterion = "Did the agent use its tools correctly?"
    transcript = _transcript(
        [
            _turn("user", "Where is ORD-1004?", 0),
            _turn("agent", "Looking.", 700, tool_calls=[_call(result={"ok": True}, success=True)]),
            _turn("user", "And ORD-1005?", 2000),
            _turn(
                "agent",
                "Both shipped.",
                2600,
                tool_calls=[_call(arguments={"order_id": "ORD-1005"}), _call("log", {}, success=True)],
            ),
        ]
    )
    user = _prompts().build_user_message(transcript, criterion)
    expected = _expected_user_message(transcript.agent_system_prompt, _format(transcript), criterion, TOOL_CALL_NOTE)
    assert user.strip() == expected.strip()
    assert user.count(TOOL_CALL_NOTE) == 1
    assert user.index("</transcript>") < user.index(TOOL_CALL_NOTE) < user.index("EVALUATION CRITERION:")
    assert _prompts().build_messages(transcript, criterion)[1]["content"] == user


def test_user_message_note_for_a_tool_call_without_result():
    transcript = _transcript([_turn("user", "Book me.", 0), _turn("agent", "Booked.", 700, tool_calls=[_call()])])
    user = _prompts().build_user_message(transcript, "Is it right?")
    assert "</transcript>\n\n" + TOOL_CALL_NOTE + "\n\nEVALUATION CRITERION:\nIs it right?" in user


def test_user_message_without_tool_calls_has_no_note():
    criterion = "Was the agent polite?"
    transcript = _transcript([_turn("user", "Hi", 0), _turn("agent", "Hello! How can I help?", 700)])
    user = _prompts().build_user_message(transcript, criterion)
    assert TOOL_CALL_NOTE not in user
    assert "About the AGENT TOOL CALL lines" not in user
    expected = _expected_user_message(transcript.agent_system_prompt, _format(transcript), criterion, None)
    assert user.strip() == expected.strip()


# ============================================================================================
# The campaign order_status transcript (run 20261023T193004Z-a9be47) as the judge now sees it
# ============================================================================================


def test_judge_messages_for_the_campaign_order_status_show_the_lookup_result():
    from impulse_evals.metrics.base import get_metric

    transcript = _campaign_transcript("supportbot_order_status_a9be47.json")
    agent_text = transcript.turns[3].text
    assert "Smart watch" in agent_text and "October 25, 2026" in agent_text   # what the agent repeated
    criterion = get_metric("UNIV-005").criterion
    messages = _prompts().build_messages(transcript, criterion)
    user = messages[1]["content"]
    section = user.split("<transcript>\n", 1)[1].split("\n</transcript>", 1)[0]
    lines = section.splitlines()
    assert A9BE47_LINE in lines
    assert lines.index(A9BE47_LINE) + 1 == lines.index(f"[turn 4] AGENT: {agent_text}")
    assert '"item": "Smart watch"' in section and '"expected_on": "2026-10-25"' in section
    assert "</transcript>\n\n" + TOOL_CALL_NOTE + "\n\nEVALUATION CRITERION:\n" + criterion in user


@pytest.mark.asyncio
async def test_litellm_judge_sends_the_campaign_lookup_result_and_note(mock_llm):
    from impulse_evals.judge.litellm_judge import LiteLLMJudge
    from impulse_evals.metrics.base import get_metric

    transcript = _campaign_transcript("supportbot_order_status_a9be47.json")
    mock_llm.queue_verdict("pass", reason="The details came from the lookup_order result.")
    judge = LiteLLMJudge("ollama/judge-primary", "ollama/judge-fast", {}, {})
    verdict = await judge.evaluate(transcript, get_metric("UNIV-005").criterion, "primary")
    assert verdict.result == "pass"
    user = mock_llm.messages(0)[1]["content"]
    assert A9BE47_LINE in user.splitlines()
    assert user.count(TOOL_CALL_NOTE) == 1


# ============================================================================================
# A1/A7-rev R1-M2: a JSON error is shown redacted; every shown error is cut at 300 characters
# ============================================================================================

OTP_VALUE = "482913"
# LiveKit Agents: an is_error function output; the adapter stores the raw output text as error
# (SPEC.md section 9) and the parsed object as result (review R1 probe p3 #1).
LIVEKIT_OTP_OUTPUT = '{"otp": "' + OTP_VALUE + '", "error": "sms gateway down"}'
REDACTED_OTP_JSON = '{"error": "sms gateway down", "otp": "[REDACTED]"}'
SEND_OTP_LINE = (
    f"[turn 2] AGENT TOOL CALL: send_otp({{}}) -> success=False error={REDACTED_OTP_JSON} result={REDACTED_OTP_JSON}"
)


def _livekit_is_error_call(output=LIVEKIT_OTP_OUTPUT):
    from impulse_evals.adapters.livekit_adapter import tool_calls_from_event

    event = {
        "function_calls": [{"call_id": "c1", "name": "send_otp", "arguments": "{}"}],
        "function_call_outputs": [{"call_id": "c1", "name": "send_otp", "output": output, "is_error": True}],
    }
    (call,) = tool_calls_from_event(event)
    return call


def test_format_line_shows_a_livekit_json_error_redacted():
    call = _livekit_is_error_call()
    assert (call.success, call.error) == (False, LIVEKIT_OTP_OUTPUT)   # the adapter rule is unchanged
    line = _line(call)
    assert line == SEND_OTP_LINE
    assert OTP_VALUE not in line
    assert call.error == LIVEKIT_OTP_OUTPUT   # the stored error is never changed by formatting


@pytest.mark.parametrize(
    "length, cut",
    [pytest.param(300, False, id="300-kept"), pytest.param(301, True, id="301-cut"), pytest.param(400, True, id="400-cut")],
)
def test_format_line_cuts_a_long_error_at_300_characters(length, cut):
    error = "E" * length
    line = _line(_call("book_slot", {"date": "2026-01-01"}, success=False, error=error))
    prefix = '[turn 2] AGENT TOOL CALL: book_slot({"date": "2026-01-01"}) -> success=False error='
    assert line == prefix + ("E" * 300 + TRUNCATED if cut else error)


def test_format_line_cuts_a_long_json_error_after_redacting_it():
    error_object = {"detail": "d" * 400, "token": "tok-" + "9" * 20}
    line = _line(_call("login", {}, success=False, error=json.dumps(error_object)))
    shown = json.dumps({"detail": "d" * 400, "token": REDACTED}, sort_keys=True, ensure_ascii=False)
    assert line == "[turn 2] AGENT TOOL CALL: login({}) -> success=False error=" + shown[:300] + TRUNCATED
    assert "tok-" not in line


@pytest.mark.asyncio
async def test_runner_judge_line_and_report_show_a_livekit_json_error_redacted(
    impulse_project, llm_router, fake_adapter, tmp_path
):
    from impulse_evals.reporter import render_html

    project = impulse_project
    name = "otp_gateway_down"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-023"])])
    fake_adapter.replies[name] = [
        {"text": "I could not send the code right now.", "tool_calls": [_livekit_is_error_call().model_dump()]}
    ]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    stored = scenario_result.transcript.turns[1].tool_calls[0]
    assert (stored.success, stored.error) == (False, LIVEKIT_OTP_OUTPUT)   # run.json keeps the error as stored

    assert SEND_OTP_LINE in _format(scenario_result.transcript).splitlines()
    _, judged = llm_router.judge_sections(llm_router.judge_calls_for("UNIV-023")[0])
    assert SEND_OTP_LINE in judged.splitlines()
    for call in llm_router.calls("judge"):
        assert all(OTP_VALUE not in text for text in llm_router.contents(call))

    page = render_html(result, embed_audio=False, run_dir=tmp_path)
    assert OTP_VALUE not in page
    table = page.split('<section id="tool-calls">', 1)[1].split("</section>", 1)[0]
    assert html.escape(REDACTED_OTP_JSON, quote=True) in table


def test_html_report_trace_table_cuts_a_long_error(run_builder, tmp_path):
    from impulse_evals.reporter import render_html

    call = _m().ToolCall(name="book_slot", arguments={"date": "2026-01-01"}, success=False, error="E" * 400)
    transcript = run_builder.transcript(
        "long_error",
        [
            run_builder.turn("user", "Book Tuesday.", 0),
            run_builder.turn("agent", "Sorry, that failed.", 900, latency_ms=900.0, tool_calls=[call]),
        ],
    )
    run = run_builder.run([run_builder.scenario_result("long_error", transcript=transcript, metrics=[("UNIV-023", 1.0)])])
    page = render_html(run, embed_audio=False, run_dir=tmp_path)
    table = page.split('<section id="tool-calls">', 1)[1].split("</section>", 1)[0]
    assert "E" * 300 + html.escape(TRUNCATED, quote=True) in table
    assert "E" * 301 not in page


# ============================================================================================
# A1/A7-rev R1-L1 and R1-L6: what the runner stores for unusual result values
# ============================================================================================


def _reply_with_result(text, call_name, result_value, arguments=None):
    """An ``AgentReply`` whose tool call gets ``result_value`` AFTER construction (pydantic v2 does not
    validate assignment, so a custom adapter can store a non-dict or self-referencing result)."""
    from impulse_evals.adapters.base import AgentReply

    reply = AgentReply(
        text=text, latency_ms=100.0, tool_calls=[_m().ToolCall(name=call_name, arguments=arguments or {})]
    )
    reply.tool_calls[0].result = result_value
    return reply


def _dump_quietly(model):
    import warnings

    with warnings.catch_warnings():   # a non-dict result makes pydantic warn while serialising
        warnings.simplefilter("ignore")
        return model.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value, stored_value",
    [
        pytest.param(f"token issued: {TOOL_SECRET}", f"token issued: {REDACTED}", id="string"),
        pytest.param(["header", TOOL_SECRET, {"note": f"key {TOOL_SECRET}"}], ["header", REDACTED, {"note": f"key {REDACTED}"}],
                     id="list"),
    ],
)
async def test_runner_redacts_a_non_dict_result_set_after_construction(
    impulse_project, llm_router, fake_adapter, monkeypatch, value, stored_value
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", TOOL_SECRET)
    project = impulse_project
    name = "non_dict_result"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-023"])])
    fake_adapter.replies[name] = [_reply_with_result("Your token is ready.", "issue_token", value)]
    result = await _run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    stored = scenario_result.transcript.turns[1].tool_calls[0]
    assert stored.result == stored_value
    assert TOOL_SECRET not in _dump_quietly(result)
    for call in llm_router.calls():
        assert all(TOOL_SECRET not in text for text in llm_router.contents(call))


@pytest.mark.asyncio
async def test_runner_keeps_a_turn_whose_result_references_itself(impulse_project, llm_router, fake_adapter, tmp_path):
    from impulse_evals.models import RunResult
    from impulse_evals.reporter import write_json

    project = impulse_project
    name = "circular_result"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-023"])])
    circular = {"a": 1}
    circular["self"] = circular
    fake_adapter.replies[name] = [_reply_with_result("Found it.", "lookup_order", circular, {"order_id": "ORD-1"})]
    result = await _run(project.load(), adapter_factory=fake_adapter)

    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error   # the scenario is not ended
    stored = scenario_result.transcript.turns[1].tool_calls[0]
    assert stored.result == {"a": 1, "self": "[circular reference]"}
    line = (
        '[turn 2] AGENT TOOL CALL: lookup_order({"order_id": "ORD-1"}) -> success=None '
        'result={"a": 1, "self": "[circular reference]"}'
    )
    assert line in _format(scenario_result.transcript).splitlines()
    _, judged = llm_router.judge_sections(llm_router.judge_calls_for("UNIV-023")[0])
    assert line in judged.splitlines()

    path = tmp_path / "run.json"
    write_json(result, path)   # the stored turn has no cycle left, so run.json can be written
    reloaded = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded.scenario_results[0].transcript.turns[1].tool_calls[0].result == stored.result


# ============================================================================================
# A1/A7-rev R1-L3: plural tokens, mpin / tpin, card no, acronym plurals
# ============================================================================================

SENSITIVE_KEYS_R1_L3 = [
    "passwords", "tokens", "access_tokens", "refreshTokens", "secrets", "client_secrets", "cookies", "otps",
    "OTPs", "PINs", "user_pins", "userPIN", "mpin", "MPIN", "tpin", "upi_mpin", "card_no", "cardNo", "CARD_NO",
]
# Regression guards (pass before the fix): one token each, or no sensitive token / pair.
SAFE_KEYS_R1_L3 = ["spins", "options", "card_note", "tokenizer", "userIDs", "cardinal_no", "pinsetter"]


@pytest.mark.parametrize("key", SENSITIVE_KEYS_R1_L3)
def test_redact_tool_result_hides_plural_and_banking_key_names(key):
    assert _redact_result({key: "s3cr3t-value", "item": "Smart watch"}) == {key: REDACTED, "item": "Smart watch"}


@pytest.mark.parametrize("key", SAFE_KEYS_R1_L3)
def test_redact_tool_result_still_keeps_other_key_names(key):
    assert _redact_result({key: "411001"}) == {key: "411001"}


# ============================================================================================
# A1/A7-rev R1-L5: the error fill order of A1.1
# ============================================================================================


@pytest.mark.parametrize(
    "result, error",
    [
        pytest.param({"error": {"message": "m"}, "errors": "s"}, "s", id="errors-string-before-error-message"),
        pytest.param({"errors": "b", "error": "a"}, "a", id="error-string-before-errors-string"),
        pytest.param({"error": {"message": "m"}, "errors": ["first"]}, "first", id="errors-item-before-error-message"),
        pytest.param({"errors": ["first"], "error": "a"}, "a", id="error-string-before-errors-item"),
    ],
)
def test_error_fill_follows_the_spec_order_not_the_dict_order(result, error):
    inferred = _infer(_call(result=result))
    assert inferred.success is False
    assert inferred.error == error


# ============================================================================================
# Re-check N-L3: true / false / null under a sensitive key are shown as is
# ============================================================================================

OUTCOME_RESULT = {"otp_verified": True, "otp": "482913", "pin": None, "token": False}
OUTCOME_SHOWN = {"otp_verified": True, "otp": REDACTED, "pin": None, "token": False}
OUTCOME_JSON = '{"otp": "[REDACTED]", "otp_verified": true, "pin": null, "token": false}'


def test_redact_tool_result_keeps_booleans_and_null_under_sensitive_keys():
    original = copy.deepcopy(OUTCOME_RESULT)
    assert _redact_result(OUTCOME_RESULT) == OUTCOME_SHOWN
    assert OUTCOME_RESULT == original   # never mutated


@pytest.mark.parametrize(
    "value, shown",
    [
        pytest.param(True, True, id="true"),
        pytest.param(False, False, id="false"),
        pytest.param(None, None, id="null"),
        # every other value of a sensitive key is still hidden (0 and 1 are numbers, not booleans)
        pytest.param(1, REDACTED, id="one"),
        pytest.param(0, REDACTED, id="zero"),
        # fixes round 4 (Stage D fix-up): an empty value carries no secret and stays empty (superseded: REDACTED)
        pytest.param("", "", id="empty-string"),
        pytest.param([], [], id="empty-list"),
        pytest.param({}, {}, id="empty-dict"),
        pytest.param("true", REDACTED, id="string-true"),
        pytest.param({"verified": True}, REDACTED, id="dict"),
        pytest.param([True], REDACTED, id="list"),
    ],
)
def test_redact_tool_result_sensitive_key_value_types(value, shown):
    assert _redact_result({"otp_sent": value, "item": "x"}) == {"otp_sent": shown, "item": "x"}
    nested = _redact_result({"auth": [{"pin_set": value}]})
    assert nested == {"auth": [{"pin_set": shown}]}


def test_format_line_shows_a_sensitive_boolean_outcome():
    call = _call("verify_otp", {"otp": "482913"}, result=OUTCOME_RESULT)
    line = _line(call)
    assert line == (
        '[turn 2] AGENT TOOL CALL: verify_otp({"otp": "482913"}) -> success=None result=' + OUTCOME_JSON
    )
    assert '"otp_verified": true' in line and '"pin": null' in line and '"token": false' in line


def test_html_report_trace_table_shows_a_sensitive_boolean_outcome(run_builder, tmp_path):
    from impulse_evals.reporter import render_html

    call = _m().ToolCall(name="verify_otp", arguments={"customer_id": "CUST-2001"}, result=dict(OUTCOME_RESULT))
    transcript = run_builder.transcript(
        "otp_outcome",
        [
            run_builder.turn("user", "The code is the one you sent.", 0),
            run_builder.turn("agent", "Thank you, you are verified.", 900, latency_ms=900.0, tool_calls=[call]),
        ],
    )
    run = run_builder.run([run_builder.scenario_result("otp_outcome", transcript=transcript, metrics=[("UNIV-023", 1.0)])])
    page = render_html(run, embed_audio=False, run_dir=tmp_path)
    assert "482913" not in page
    table = page.split('<section id="tool-calls">', 1)[1].split("</section>", 1)[0]
    assert html.escape(OUTCOME_JSON, quote=True) in table
    line = '[turn 2] AGENT TOOL CALL: verify_otp({"customer_id": "CUST-2001"}) -> success=None result=' + OUTCOME_JSON
    assert html.escape(line, quote=True) in page
