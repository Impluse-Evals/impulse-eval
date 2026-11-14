# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 2: the LiteLLM judge survives empty, tool-call-only, fenced, and invalid output (SPEC.md section 12)."""

import pytest

PRIMARY = "ollama/judge-primary"
FAST = "ollama/judge-fast"
CRITERION = "Did the agent greet the user and ask which day suits them?"
EMPTY_REASON = "Judge returned empty (safety filter or rate limit)"
INVALID_REASON = "Judge returned invalid JSON"


def _judge():
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    return LiteLLMJudge(primary_model=PRIMARY, fast_model=FAST, primary_options={}, fast_options={})


def _transcript():
    from impulse_evals.models import CallTranscript, Turn

    return CallTranscript(
        turns=[
            Turn(speaker="user", text="Hi, I need an appointment", timestamp_ms=0),
            Turn(speaker="agent", text="Hello! Which day works for you?", timestamp_ms=900),
        ],
        duration_ms=2000,
        metadata={},
        scenario_name="clinic_appointment_booking",
        language="en-IN",
        agent_system_prompt="You are the booking assistant of a demo clinic.",
    )


async def _evaluate(tier="primary"):
    return await _judge().evaluate(_transcript(), CRITERION, tier)


def _assert_no_verdict(verdict, reason):
    from impulse_evals.judge.base import JudgeVerdict

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.result == "skip"
    assert verdict.reason == reason
    assert verdict.no_verdict is True


# --------------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------------


def test_litellm_judge_is_a_judge():
    from impulse_evals.judge.base import Judge
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    assert issubclass(LiteLLMJudge, Judge)


def test_judge_verdict_defaults():
    from impulse_evals.judge.base import JudgeVerdict

    verdict = JudgeVerdict(result="pass", reason="ok")
    assert verdict.turn_index is None
    assert verdict.confidence == 0.0
    assert verdict.no_verdict is False
    assert set(JudgeVerdict.model_fields) == {"result", "reason", "turn_index", "confidence", "no_verdict"}


# --------------------------------------------------------------------------------------------
# Empty content: skip, no verdict, no retry
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_content_is_no_verdict_skip(mock_llm):
    mock_llm.queue_none()
    verdict = await _evaluate()
    _assert_no_verdict(verdict, EMPTY_REASON)
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_empty_string_content_is_no_verdict_skip(mock_llm):
    mock_llm.queue_empty()
    verdict = await _evaluate()
    _assert_no_verdict(verdict, EMPTY_REASON)
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_whitespace_content_is_no_verdict_skip(mock_llm):
    mock_llm.queue_whitespace()
    verdict = await _evaluate()
    _assert_no_verdict(verdict, EMPTY_REASON)
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_tool_call_response_without_content_is_no_verdict_skip(mock_llm):
    mock_llm.queue_tool_call(name="lookup_policy", arguments={"topic": "refunds"})
    verdict = await _evaluate()
    _assert_no_verdict(verdict, EMPTY_REASON)
    assert mock_llm.call_count == 1


# --------------------------------------------------------------------------------------------
# Tolerant JSON parsing
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_json_verdict_parsed(mock_llm):
    mock_llm.queue_json(
        {"result": "fail", "reason": "The agent never greeted the user.", "turn_index": 2, "confidence": 0.8}
    )
    verdict = await _evaluate()
    assert verdict.result == "fail"
    assert verdict.reason == "The agent never greeted the user."
    assert verdict.turn_index == 2
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.no_verdict is False
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["json", ""])
async def test_fenced_json_parsed(mock_llm, lang):
    mock_llm.queue_fenced_json(
        {"result": "pass", "reason": "Greeted and asked for a day.", "turn_index": None, "confidence": 0.95},
        lang=lang,
    )
    verdict = await _evaluate()
    assert verdict.result == "pass"
    assert verdict.reason == "Greeted and asked for a day."
    assert verdict.turn_index is None
    assert verdict.confidence == pytest.approx(0.95)
    assert verdict.no_verdict is False
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_first_json_object_extracted_from_prose(mock_llm):
    mock_llm.queue(
        'Here is my evaluation: {"result": "fail", "reason": "It said {name} literally.", '
        '"turn_index": 2, "confidence": 0.7} Let me know if you need more.'
    )
    verdict = await _evaluate()
    assert verdict.result == "fail"
    assert verdict.reason == "It said {name} literally."
    assert verdict.turn_index == 2
    assert verdict.no_verdict is False
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_missing_confidence_and_turn_index_default(mock_llm):
    mock_llm.queue_json({"result": "pass", "reason": "fine"})
    verdict = await _evaluate()
    assert verdict.result == "pass"
    assert verdict.confidence == 0.0
    assert verdict.turn_index is None
    assert verdict.no_verdict is False


# --------------------------------------------------------------------------------------------
# Invalid JSON: retry once with "Return JSON only", then skip with no verdict
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_json_retried_once_then_parsed(mock_llm):
    mock_llm.queue_invalid_json()
    mock_llm.queue_verdict("pass", reason="Greeted the user.", turn_index=2, confidence=0.9)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert verdict.result == "pass"
    assert verdict.reason == "Greeted the user."
    assert verdict.no_verdict is False


@pytest.mark.asyncio
async def test_retry_asks_for_json_only(mock_llm):
    mock_llm.queue_invalid_json()
    mock_llm.queue_verdict("pass")
    await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)


@pytest.mark.asyncio
async def test_invalid_json_twice_is_no_verdict_skip(mock_llm):
    mock_llm.queue_invalid_json()
    mock_llm.queue_invalid_json("still not json")
    verdict = await _evaluate()
    _assert_no_verdict(verdict, INVALID_REASON)
    assert mock_llm.call_count == 2


# --------------------------------------------------------------------------------------------
# A valid "skip" from the judge is not a no-verdict
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_skip_verdict_is_not_no_verdict(mock_llm):
    mock_llm.queue_verdict("skip", reason="No personal data was requested.", confidence=0.9)
    verdict = await _evaluate()
    assert verdict.result == "skip"
    assert verdict.reason == "No personal data was requested."
    assert verdict.no_verdict is False
    assert mock_llm.call_count == 1


# --------------------------------------------------------------------------------------------
# Call settings
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_uses_temperature_zero(mock_llm):
    mock_llm.queue_invalid_json()
    mock_llm.queue_verdict("pass")
    await _evaluate()
    for call in mock_llm.calls:
        assert call["kwargs"].get("temperature") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tier, model", [("primary", PRIMARY), ("fast", FAST)])
async def test_tier_selects_model(mock_llm, tier, model):
    mock_llm.queue_verdict("pass")
    await _evaluate(tier)
    assert mock_llm.calls[0]["model"] == model


# --------------------------------------------------------------------------------------------
# Malformed outer JSON: never scanned inside or past, so the attempt is invalid JSON
# (DECISIONS.md, "Maintainer decisions before phase 4": judge parsing)
# --------------------------------------------------------------------------------------------

INNER_VERDICT = '{"result": "pass", "reason": "Greeted the user.", "turn_index": 2, "confidence": 0.9}'
RETRY_REASON = "The agent never asked which day suits them."

# Each reply starts with a malformed outer object that holds a well-formed, verdict-shaped object.
MALFORMED_OUTER_WITH_INNER_VERDICT = [
    pytest.param(
        '{"checks": [{"tool": "book_slot", "result": "pass"}], "overall": "fail", "reason": "missing quote}',
        id="unterminated-string-tool-check",
    ),
    pytest.param(
        '{"checks": [' + INNER_VERDICT + '], "overall": "fail", "reason": "missing quote}',
        id="unterminated-string",
    ),
    pytest.param('{"verdict": ' + INNER_VERDICT + ', "overall": pass}', id="unquoted-value"),
    pytest.param('{"checks": [' + INNER_VERDICT + '], "overall": "fail"', id="missing-closing-brace"),
    pytest.param(
        'My evaluation:\n```json\n{"verdict": ' + INNER_VERDICT + ', "overall": "pass",}\n```',
        id="fenced-trailing-comma",
    ),
]

# Each reply has a malformed outer object, then a separate, well-formed top-level verdict after it.
VERDICT_AFTER_MALFORMED_OUTER = [
    pytest.param(
        '{"result": pass, "reason": "Greeted the user."}\nCorrected: ' + INNER_VERDICT,
        id="unquoted-value",
    ),
    pytest.param('{"result": "pass", "reason": "Greeted the user.",}\n' + INNER_VERDICT, id="trailing-comma"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", MALFORMED_OUTER_WITH_INNER_VERDICT)
async def test_malformed_outer_json_with_inner_verdict_is_no_verdict_skip(mock_llm, reply):
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", MALFORMED_OUTER_WITH_INNER_VERDICT)
async def test_malformed_outer_json_then_valid_retry_uses_retry_verdict(mock_llm, reply):
    mock_llm.queue(reply)
    mock_llm.queue_verdict("fail", reason=RETRY_REASON, turn_index=2, confidence=0.8)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert verdict.result == "fail"
    assert verdict.reason == RETRY_REASON
    assert verdict.turn_index == 2
    assert verdict.no_verdict is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", VERDICT_AFTER_MALFORMED_OUTER)
async def test_malformed_outer_json_not_scanned_past_to_later_verdict(mock_llm, reply):
    mock_llm.queue(reply)
    mock_llm.queue_verdict("fail", reason=RETRY_REASON, turn_index=2, confidence=0.8)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert verdict.result == "fail"
    assert verdict.reason == RETRY_REASON
    assert verdict.no_verdict is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", VERDICT_AFTER_MALFORMED_OUTER)
async def test_malformed_outer_json_then_later_verdict_twice_is_no_verdict_skip(mock_llm, reply):
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
async def test_verdict_nested_in_well_formed_outer_json_is_not_accepted(mock_llm):
    reply = '{"checks": [' + INNER_VERDICT + '], "overall": "looks fine"}'
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply, result, reason",
    [
        pytest.param(
            'Here is my verdict:\n```json\n{"result": "fail", "reason": "The agent never asked which day '
            'suits them.", "turn_index": 2, "confidence": 0.8}\n```\nLet me know if you need more.',
            "fail",
            "The agent never asked which day suits them.",
            id="fenced-with-prose",
        ),
        pytest.param(
            'Verdict below.\n```\n{"result": "pass", "reason": "Greeted and asked for a day.", '
            '"turn_index": null, "confidence": 0.95}\n```\nThanks.',
            "pass",
            "Greeted and asked for a day.",
            id="fenced-no-lang-with-prose",
        ),
        pytest.param(
            '{"result": "fail", "reason": "It read the template {first_name} aloud and left a stray } in '
            'the reply.", "turn_index": 2, "confidence": 0.8}',
            "fail",
            "It read the template {first_name} aloud and left a stray } in the reply.",
            id="braces-in-strings",
        ),
        pytest.param(
            'My evaluation: {"result": "fail", "reason": "It left an unclosed { brace and said {slot}.", '
            '"turn_index": 2, "confidence": 0.7} That is all.',
            "fail",
            "It left an unclosed { brace and said {slot}.",
            id="braces-in-strings-with-prose",
        ),
        pytest.param(
            '{"notes": "checked turn 2"}\n{"result": "pass", "reason": "Greeted and asked for a day.", '
            '"turn_index": 2, "confidence": 0.9}',
            "pass",
            "Greeted and asked for a day.",
            id="non-verdict-object-passed-over",
        ),
    ],
)
async def test_malformed_outer_json_rule_keeps_tolerant_parsing(mock_llm, reply, result, reason):
    mock_llm.queue(reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 1
    assert verdict.result == result
    assert verdict.reason == reason
    assert verdict.no_verdict is False


# --------------------------------------------------------------------------------------------
# The verdict must be a top-level JSON object: a top-level JSON array, or any other JSON type,
# is invalid JSON (DECISIONS.md, "Maintainer decisions on the Python compatibility results").
# An object whose "{" sits inside a well-formed top-level [ ... ] is not top-level.
# --------------------------------------------------------------------------------------------

ARRAY_PASS_VERDICT = '{"result": "pass", "reason": "ok", "turn_index": null, "confidence": 0.9}'
ARRAY_FAIL_VERDICT = (
    '{"result": "fail", "reason": "The agent never greeted the user.", "turn_index": 2, "confidence": 0.8}'
)

# Each reply holds a well-formed top-level JSON array with a verdict-shaped object inside it,
# and no verdict object outside the array.
TOP_LEVEL_ARRAY_REPLIES = [
    pytest.param("[" + ARRAY_PASS_VERDICT + "]", id="bare-array"),
    pytest.param("[" + ARRAY_FAIL_VERDICT + "]", id="bare-array-fail-verdict"),
    pytest.param("\n  [" + ARRAY_PASS_VERDICT + "]  \n", id="bare-array-surrounding-whitespace"),
    pytest.param("```json\n[\n  " + ARRAY_PASS_VERDICT + "\n]\n```", id="fenced-json-array"),
    pytest.param("```\n[" + ARRAY_PASS_VERDICT + "]\n```", id="fenced-no-lang-array"),
    pytest.param(
        "Here is my verdict: [" + ARRAY_PASS_VERDICT + "] Let me know if you need more.",
        id="array-with-prose",
    ),
    pytest.param(
        "My evaluation:\n```json\n[" + ARRAY_PASS_VERDICT + "]\n```\nThat is all.",
        id="fenced-array-with-prose",
    ),
    pytest.param(
        "At [turn 2] the agent greeted the user.\n[" + ARRAY_PASS_VERDICT + "]",
        id="turn-bracket-prose-then-array",
    ),
    pytest.param("[" + ARRAY_PASS_VERDICT + ", " + ARRAY_FAIL_VERDICT + "]", id="two-verdicts"),
    pytest.param('[{"notes": "checked turn 2"}, ' + ARRAY_PASS_VERDICT + "]", id="non-verdict-then-verdict"),
    pytest.param('["verdict", ' + ARRAY_PASS_VERDICT + "]", id="string-then-verdict"),
    pytest.param("[[" + ARRAY_PASS_VERDICT + "]]", id="nested-array"),
]

# Each reply is valid JSON whose top-level value is not an object.
NON_OBJECT_JSON_REPLIES = [
    pytest.param('"pass"', id="string"),
    pytest.param("0.9", id="number"),
    pytest.param("1", id="integer"),
    pytest.param("true", id="true"),
    pytest.param("false", id="false"),
    pytest.param("null", id="null"),
    pytest.param('```json\n"pass"\n```', id="fenced-string"),
    pytest.param("```json\ntrue\n```", id="fenced-true"),
    # A JSON string whose text is an encoded verdict: still a string, never decoded a second time.
    pytest.param(
        '"{\\"result\\": \\"pass\\", \\"reason\\": \\"ok\\", \\"turn_index\\": null, \\"confidence\\": 0.9}"',
        id="string-holding-encoded-verdict",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", TOP_LEVEL_ARRAY_REPLIES)
async def test_top_level_json_array_twice_is_no_verdict_skip(mock_llm, reply):
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", TOP_LEVEL_ARRAY_REPLIES)
async def test_top_level_json_array_then_valid_retry_uses_retry_verdict(mock_llm, reply):
    mock_llm.queue(reply)
    mock_llm.queue_verdict("fail", reason=RETRY_REASON, turn_index=2, confidence=0.8)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    assert verdict.result == "fail"
    assert verdict.reason == RETRY_REASON
    assert verdict.turn_index == 2
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.no_verdict is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", NON_OBJECT_JSON_REPLIES)
async def test_top_level_json_non_object_twice_is_no_verdict_skip(mock_llm, reply):
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", NON_OBJECT_JSON_REPLIES)
async def test_top_level_json_non_object_then_valid_retry_uses_retry_verdict(mock_llm, reply):
    mock_llm.queue(reply)
    mock_llm.queue_verdict("fail", reason=RETRY_REASON, turn_index=2, confidence=0.8)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert verdict.result == "fail"
    assert verdict.reason == RETRY_REASON
    assert verdict.no_verdict is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply, result, reason",
    [
        pytest.param(
            '{"result": "pass", "reason": "Greeted and asked for a day.", "turn_index": 2, '
            '"confidence": 0.9}\nThat is my verdict; the agent did what the prompt asks.',
            "pass",
            "Greeted and asked for a day.",
            id="object-then-trailing-prose",
        ),
        pytest.param(
            '{"result": "fail", "reason": "The agent never asked which day suits them.", "turn_index": 2, '
            '"confidence": 0.8}\nSee [turn 2]: the agent skipped the question.',
            "fail",
            "The agent never asked which day suits them.",
            id="object-then-trailing-prose-with-turn-bracket",
        ),
        pytest.param(
            "At [turn 2] the agent greeted the user and asked for a day.\n"
            '{"result": "pass", "reason": "Greeted and asked for a day.", "turn_index": 2, "confidence": 0.9}',
            "pass",
            "Greeted and asked for a day.",
            id="turn-bracket-prose-before-object",
        ),
        pytest.param(
            '```json\n{"result": "fail", "reason": "At [turn 2] the agent said [slot] aloud.", '
            '"turn_index": 2, "confidence": 0.7}\n```\nThanks.',
            "fail",
            "At [turn 2] the agent said [slot] aloud.",
            id="fenced-object-brackets-in-strings",
        ),
        pytest.param(
            '{"result": "pass", "reason": "Greeted and asked for a day.", "turn_index": 2, '
            '"confidence": 0.9, "evidence_turns": [1, 2]}',
            "pass",
            "Greeted and asked for a day.",
            id="array-nested-in-top-level-object",
        ),
    ],
)
async def test_top_level_object_rule_keeps_tolerant_parsing(mock_llm, reply, result, reason):
    mock_llm.queue(reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 1
    assert verdict.result == result
    assert verdict.reason == reason
    assert verdict.no_verdict is False


# --------------------------------------------------------------------------------------------
# A malformed top-level JSON array is invalid JSON (DECISIONS.md, "Maintainer decisions after
# phase 12": judge CHANGE and its reading). A "[" starts a JSON array when a JSON value begins
# right after it (optional whitespace, then "{", "[", '"', a number, true, false, null) or it
# closes at once; such a "[" that does not decode ends the scan: nothing inside it or after it is
# a verdict. A "[" not followed by a JSON value ("[turn 2]", "[slot]") stays prose.
# --------------------------------------------------------------------------------------------

# Each reply has a "[" that starts a JSON array and does not decode, with no top-level verdict
# object before it; every verdict-shaped object sits inside that array or after it.
MALFORMED_TOP_LEVEL_ARRAY_REPLIES = [
    pytest.param("[" + ARRAY_PASS_VERDICT + ",]", id="trailing-comma"),
    pytest.param("[" + ARRAY_FAIL_VERDICT + ",]", id="trailing-comma-fail-verdict"),
    pytest.param("[" + ARRAY_PASS_VERDICT, id="unterminated"),
    pytest.param("[" + ARRAY_PASS_VERDICT + ",", id="unterminated-after-comma"),
    pytest.param("[" + ARRAY_PASS_VERDICT + " " + ARRAY_FAIL_VERDICT + "]", id="missing-comma"),
    pytest.param("[" + ARRAY_PASS_VERDICT + ", pass]", id="unquoted-element"),
    pytest.param("[ " + ARRAY_PASS_VERDICT + ",]", id="space-after-bracket"),
    pytest.param("[\n  " + ARRAY_PASS_VERDICT + ",\n]", id="newline-after-bracket"),
    pytest.param("```json\n[\n  " + ARRAY_PASS_VERDICT + ",\n]\n```", id="fenced-json-trailing-comma"),
    pytest.param("```\n[" + ARRAY_PASS_VERDICT + "\n```", id="fenced-no-lang-unterminated"),
    pytest.param("```\n[" + ARRAY_PASS_VERDICT + ",]\n```", id="fenced-no-lang-trailing-comma"),
    pytest.param("Here is my verdict: [" + ARRAY_PASS_VERDICT + ",]", id="prose-before"),
    pytest.param(
        "Here is my verdict: [" + ARRAY_PASS_VERDICT + " " + ARRAY_FAIL_VERDICT + "] Let me know.",
        id="prose-around-missing-comma",
    ),
    pytest.param(
        "At [turn 2] the agent greeted the user.\n[" + ARRAY_PASS_VERDICT + ",]",
        id="turn-bracket-prose-then-malformed-array",
    ),
    # A well-formed top-level verdict object after the malformed array is never reached.
    pytest.param(
        "[" + ARRAY_FAIL_VERDICT + ",]\n" + ARRAY_PASS_VERDICT,
        id="malformed-array-then-top-level-verdict",
    ),
    pytest.param(
        '[{"notes": "checked turn 2"},]\n' + ARRAY_PASS_VERDICT,
        id="non-verdict-malformed-array-then-top-level-verdict",
    ),
    pytest.param(
        "Draft:\n```json\n[" + ARRAY_FAIL_VERDICT + ",]\n```\nFinal: " + ARRAY_PASS_VERDICT,
        id="fenced-malformed-array-then-top-level-verdict",
    ),
    pytest.param('["verdict", ' + ARRAY_PASS_VERDICT + ",]", id="string-first"),
    pytest.param('[{"notes": "checked turn 2"}, ' + ARRAY_PASS_VERDICT + ",]", id="non-verdict-object-first"),
    pytest.param("[[" + ARRAY_PASS_VERDICT + "],]", id="nested-outer-malformed"),
    pytest.param("[[" + ARRAY_PASS_VERDICT + ",]]", id="nested-inner-malformed"),
]

# Each reply has prose brackets that are not JSON arrays (no JSON value right after the "["),
# then a well-formed top-level verdict object: (reply, result, reason).
PROSE_BRACKETS_BEFORE_VERDICT_REPLIES = [
    pytest.param(
        "The agent read [slot] aloud instead of a time.\n" + ARRAY_FAIL_VERDICT,
        "fail",
        "The agent never greeted the user.",
        id="slot-bracket",
    ),
    pytest.param(
        "As noted [see above], the agent asked for a day.\n" + ARRAY_PASS_VERDICT,
        "pass",
        "ok",
        id="see-above-bracket",
    ),
    pytest.param(
        "At [ turn 2 ] the agent greeted the user.\n" + ARRAY_PASS_VERDICT,
        "pass",
        "ok",
        id="spaced-turn-bracket",
    ),
    pytest.param(
        "At [turn 2] the agent said [slot] aloud.\n" + ARRAY_FAIL_VERDICT,
        "fail",
        "The agent never greeted the user.",
        id="turn-and-slot-brackets",
    ),
    pytest.param(
        "At [turn 2] the agent said [slot] aloud.\n```json\n" + ARRAY_FAIL_VERDICT + "\n```",
        "fail",
        "The agent never greeted the user.",
        id="turn-and-slot-brackets-then-fenced-object",
    ),
    pytest.param("Verdict [see above]: " + ARRAY_PASS_VERDICT, "pass", "ok", id="bracket-right-before-object"),
    pytest.param(
        'The agent said "your slot is [...]" at [turn 2].\n' + ARRAY_FAIL_VERDICT,
        "fail",
        "The agent never greeted the user.",
        id="ellipsis-bracket",
    ),
    pytest.param(
        "[Summary] The agent greeted the user and asked for a day.\n" + ARRAY_PASS_VERDICT,
        "pass",
        "ok",
        id="bracket-at-start",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", MALFORMED_TOP_LEVEL_ARRAY_REPLIES)
async def test_malformed_top_level_json_array_twice_is_no_verdict_skip(mock_llm, reply):
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", MALFORMED_TOP_LEVEL_ARRAY_REPLIES)
async def test_malformed_top_level_json_array_then_valid_retry_uses_retry_verdict(mock_llm, reply):
    mock_llm.queue(reply)
    mock_llm.queue_verdict("fail", reason=RETRY_REASON, turn_index=2, confidence=0.8)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    assert verdict.result == "fail"
    assert verdict.reason == RETRY_REASON
    assert verdict.turn_index == 2
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.no_verdict is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reply, result, reason", PROSE_BRACKETS_BEFORE_VERDICT_REPLIES)
async def test_prose_brackets_before_top_level_verdict_are_tolerated(mock_llm, reply, result, reason):
    mock_llm.queue(reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 1
    assert verdict.result == result
    assert verdict.reason == reason
    assert verdict.no_verdict is False


# --------------------------------------------------------------------------------------------
# Audit batch 2, the maintainer's judge item:
# (a) a top-level "[" followed (after optional JSON whitespace) by "," starts a malformed JSON
#     array: the reply is invalid JSON (one retry, then no verdict), e.g. [,{"result": "pass", ...}];
# (b) a reply with more than one top-level JSON object holding a "result" verdict is invalid JSON
#     (one retry, then no verdict), even when the verdicts agree.
# A non-verdict object (no "result") before or after the one verdict object is still passed over,
# and "[turn 2]"-style prose stays tolerated.
# --------------------------------------------------------------------------------------------

# (a) Each reply has a "[" whose first non-whitespace character after it is ",".
LEADING_COMMA_ARRAY_REPLIES = [
    pytest.param("[," + ARRAY_PASS_VERDICT + "]", id="comma-right-after-bracket"),
    pytest.param("[ , " + ARRAY_PASS_VERDICT + "]", id="space-comma"),
    pytest.param("[\n  ,\n  " + ARRAY_FAIL_VERDICT + "\n]", id="newline-comma-fail-verdict"),
    pytest.param("[\t," + ARRAY_PASS_VERDICT + "]", id="tab-comma"),
    pytest.param("```json\n[," + ARRAY_PASS_VERDICT + "]\n```", id="fenced"),
    pytest.param("Here is my verdict: [," + ARRAY_PASS_VERDICT + "] Let me know.", id="prose-around"),
    pytest.param(
        "At [turn 2] the agent greeted the user.\n[," + ARRAY_PASS_VERDICT + "]",
        id="turn-bracket-prose-then-comma-array",
    ),
    # A well-formed top-level verdict object after the malformed array is never reached.
    pytest.param("[,]\n" + ARRAY_PASS_VERDICT, id="comma-array-then-top-level-verdict"),
]

# (b) Each reply has two (or more) top-level JSON objects whose "result" is a verdict.
SECOND_VERDICT_OBJECT_REPLIES = [
    pytest.param(ARRAY_PASS_VERDICT + "\n" + ARRAY_PASS_VERDICT, id="same-verdict-twice"),
    pytest.param(
        ARRAY_PASS_VERDICT + '\n{"result": "pass", "reason": "Greeted and asked for a day.", "confidence": 0.7}',
        id="agreeing-verdicts-different-reasons",
    ),
    pytest.param(ARRAY_PASS_VERDICT + " " + ARRAY_FAIL_VERDICT, id="pass-then-fail"),
    pytest.param(ARRAY_FAIL_VERDICT + "\n\n" + ARRAY_PASS_VERDICT, id="fail-then-pass"),
    pytest.param("Draft: " + ARRAY_FAIL_VERDICT + "\nFinal: " + ARRAY_PASS_VERDICT, id="prose-between"),
    pytest.param(
        "```json\n" + ARRAY_PASS_VERDICT + "\n```\n```json\n" + ARRAY_FAIL_VERDICT + "\n```",
        id="two-fenced-blocks",
    ),
    pytest.param(
        '{"notes": "checked turn 2"}\n' + ARRAY_PASS_VERDICT + "\n" + ARRAY_FAIL_VERDICT,
        id="non-verdict-then-two-verdicts",
    ),
    pytest.param(
        ARRAY_PASS_VERDICT + "\n" + ARRAY_PASS_VERDICT + "\n" + ARRAY_FAIL_VERDICT, id="three-verdicts"
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", LEADING_COMMA_ARRAY_REPLIES)
async def test_leading_comma_array_twice_is_no_verdict_skip(mock_llm, reply):
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", LEADING_COMMA_ARRAY_REPLIES)
async def test_leading_comma_array_then_valid_retry_uses_retry_verdict(mock_llm, reply):
    mock_llm.queue(reply)
    mock_llm.queue_verdict("fail", reason=RETRY_REASON, turn_index=2, confidence=0.8)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert verdict.result == "fail"
    assert verdict.reason == RETRY_REASON
    assert verdict.turn_index == 2
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.no_verdict is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", SECOND_VERDICT_OBJECT_REPLIES)
async def test_more_than_one_top_level_verdict_object_twice_is_no_verdict_skip(mock_llm, reply):
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", SECOND_VERDICT_OBJECT_REPLIES)
async def test_more_than_one_top_level_verdict_object_then_valid_retry_uses_retry_verdict(mock_llm, reply):
    mock_llm.queue(reply)
    mock_llm.queue_verdict("fail", reason=RETRY_REASON, turn_index=2, confidence=0.8)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert verdict.result == "fail"
    assert verdict.reason == RETRY_REASON
    assert verdict.turn_index == 2
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.no_verdict is False


@pytest.mark.asyncio
async def test_second_verdict_object_in_the_retry_answer_is_no_verdict_skip(mock_llm):
    # A valid first answer is never retried; an invalid first answer and a retry answer holding two
    # verdict objects end as no verdict.
    mock_llm.queue_invalid_json()
    mock_llm.queue(ARRAY_PASS_VERDICT + "\n" + ARRAY_FAIL_VERDICT)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    _assert_no_verdict(verdict, INVALID_REASON)


# Regression guards (pass before the fix too): exactly one top-level verdict object, with non-verdict
# objects, nested "result" keys, or "[turn 2]" prose around it, is still accepted at once.
ONE_VERDICT_OBJECT_REPLIES = [
    pytest.param(
        ARRAY_PASS_VERDICT + '\n{"notes": "checked turn 2"}', "pass", "ok", id="non-verdict-object-after"
    ),
    pytest.param(
        '{"notes": "checked turn 2"}\n' + ARRAY_FAIL_VERDICT + '\n{"evidence_turns": [1, 2]}',
        "fail",
        "The agent never greeted the user.",
        id="non-verdict-objects-before-and-after",
    ),
    pytest.param(
        '{"result": "pass", "reason": "ok", "turn_index": null, "confidence": 0.9, '
        '"checks": [{"tool": "book_slot", "result": "fail"}], "detail": {"result": "fail"}}',
        "pass",
        "ok",
        id="nested-result-keys-are-not-top-level",
    ),
    pytest.param(
        "At [turn 2] the agent said [slot] aloud.\n" + ARRAY_FAIL_VERDICT + "\nSee [turn 2] above.",
        "fail",
        "The agent never greeted the user.",
        id="turn-bracket-prose-around-one-verdict",
    ),
    pytest.param(
        'The agent said "your slot is [...]".\n```json\n' + ARRAY_PASS_VERDICT + '\n```\n{"notes": "done"}',
        "pass",
        "ok",
        id="ellipsis-bracket-fenced-verdict-then-note",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply, result, reason", ONE_VERDICT_OBJECT_REPLIES)
async def test_exactly_one_top_level_verdict_object_is_still_accepted(mock_llm, reply, result, reason):
    mock_llm.queue(reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 1
    assert verdict.result == result
    assert verdict.reason == reason
    assert verdict.no_verdict is False


# --------------------------------------------------------------------------------------------
# Audit batch 2 review fix-ups (the maintainer's judge item (a), follow-up): the whitespace between
# the "[" and the "," is any whitespace Python's str.isspace() knows, not only JSON's four (space,
# tab, CR, LF): a form feed, vertical tab, NO-BREAK SPACE, EM SPACE, IDEOGRAPHIC SPACE, or LINE
# SEPARATOR before the "," still starts a malformed JSON array (one retry, then no verdict). A "["
# followed by such whitespace and then a word ("[ turn 2]") is still prose.
# --------------------------------------------------------------------------------------------

# Each reply has a "[" whose first non-whitespace character (str.isspace) after it is ",", with at
# least one whitespace character there that JSON does not count as whitespace.
UNICODE_SPACE_COMMA_ARRAY_REPLIES = [
    pytest.param("[\f," + ARRAY_PASS_VERDICT + "]", id="form-feed-comma"),
    pytest.param("[\v," + ARRAY_PASS_VERDICT + "]", id="vertical-tab-comma"),
    pytest.param("[ ," + ARRAY_PASS_VERDICT + "]", id="no-break-space-comma"),
    pytest.param("[ ," + ARRAY_FAIL_VERDICT + "]", id="em-space-comma-fail-verdict"),
    pytest.param("[　 ," + ARRAY_PASS_VERDICT + "]", id="ideographic-space-line-separator-comma"),
    pytest.param("[  \t\f ," + ARRAY_PASS_VERDICT + "]", id="json-and-unicode-spaces-mixed-comma"),
    pytest.param("```json\n[ ," + ARRAY_PASS_VERDICT + "]\n```", id="fenced-no-break-space-comma"),
    pytest.param(
        "At [turn 2] the agent greeted the user.\n[\f," + ARRAY_PASS_VERDICT + "]",
        id="turn-bracket-prose-then-form-feed-comma-array",
    ),
    pytest.param("[\f,]\n" + ARRAY_PASS_VERDICT, id="form-feed-comma-array-then-top-level-verdict"),
]

# Regression guards (pass before the fix too): a "[" followed by non-JSON whitespace and then a word
# is prose, as "[turn 2]" is; the one top-level verdict after it is accepted at once.
UNICODE_SPACE_PROSE_BRACKET_REPLIES = [
    pytest.param(
        "At [ turn 2] the agent greeted the user.\n" + ARRAY_PASS_VERDICT, "pass", "ok",
        id="no-break-space-turn-bracket",
    ),
    pytest.param(
        "The agent read [\f slot ] aloud.\n" + ARRAY_FAIL_VERDICT, "fail", "The agent never greeted the user.",
        id="form-feed-slot-bracket",
    ),
    pytest.param(
        "As noted [ see above ], the agent asked for a day.\n" + ARRAY_PASS_VERDICT, "pass", "ok",
        id="em-space-see-above-bracket",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", UNICODE_SPACE_COMMA_ARRAY_REPLIES)
async def test_unicode_whitespace_comma_array_twice_is_no_verdict_skip(mock_llm, reply):
    mock_llm.queue(reply, reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert "Return JSON only" not in mock_llm.text(0)
    _assert_no_verdict(verdict, INVALID_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", UNICODE_SPACE_COMMA_ARRAY_REPLIES)
async def test_unicode_whitespace_comma_array_then_valid_retry_uses_retry_verdict(mock_llm, reply):
    mock_llm.queue(reply)
    mock_llm.queue_verdict("fail", reason=RETRY_REASON, turn_index=2, confidence=0.8)
    verdict = await _evaluate()
    assert mock_llm.call_count == 2
    assert "Return JSON only" in mock_llm.text(1)
    assert verdict.result == "fail"
    assert verdict.reason == RETRY_REASON
    assert verdict.turn_index == 2
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.no_verdict is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reply, result, reason", UNICODE_SPACE_PROSE_BRACKET_REPLIES)
async def test_unicode_whitespace_prose_bracket_before_verdict_is_tolerated(mock_llm, reply, result, reason):
    mock_llm.queue(reply)
    verdict = await _evaluate()
    assert mock_llm.call_count == 1
    assert verdict.result == result
    assert verdict.reason == reason
    assert verdict.no_verdict is False


# --------------------------------------------------------------------------------------------
# Maintainer decision 2026-10-21, item 5 (AUDIT-FIXES.md, logged): JSON mode honours the role's
# custom_llm_provider. LiteLLM cannot place the bare model string "llama-3.1-8b-instant" (no
# response_format among its supported parameters), but with custom_llm_provider="groq" it lists
# response_format. The judge asks LiteLLM with the role's custom_llm_provider (the top-level option,
# else the one inside extra_params; blank = unset) and keeps the answer per (model, provider), so a
# primary and a fast role with the same model string but different providers get their own answers.
# A response_format in the options is kept; a provider LiteLLM does not know means no JSON mode.
# The real (offline) LiteLLM parameter table; mock_llm records each call's kwargs.
# --------------------------------------------------------------------------------------------

BARE_GROQ_MODEL = "llama-3.1-8b-instant"
JSON_MODE = {"type": "json_object"}


def _judge_with(primary_options, fast_options=None, primary=BARE_GROQ_MODEL, fast=BARE_GROQ_MODEL):
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    return LiteLLMJudge(
        primary_model=primary, fast_model=fast, primary_options=primary_options, fast_options=fast_options or {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [{"custom_llm_provider": "groq"}, {"extra_params": {"custom_llm_provider": "groq"}}],
    ids=["top_level", "inside_extra_params"],
)
async def test_json_mode_follows_the_roles_custom_llm_provider(mock_llm, options):
    mock_llm.queue_verdict("pass")
    verdict = await _judge_with(options).evaluate(_transcript(), CRITERION, "primary")
    assert verdict.result == "pass"
    assert mock_llm.calls[0]["model"] == BARE_GROQ_MODEL
    kwargs = mock_llm.calls[0]["kwargs"]
    assert kwargs.get("response_format") == JSON_MODE, (
        f"no JSON mode for {BARE_GROQ_MODEL} with custom_llm_provider groq: {kwargs}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [("primary", "fast"), ("fast", "primary")], ids=["primary_first", "fast_first"])
async def test_json_mode_is_decided_per_model_and_custom_llm_provider(mock_llm, order):
    # Same model string for both roles; only the primary role names the groq provider.
    judge = _judge_with({"custom_llm_provider": "groq"}, {})
    seen = {}
    for tier in order:
        mock_llm.queue_verdict("pass")
        await judge.evaluate(_transcript(), CRITERION, tier)
        assert mock_llm.calls[-1]["model"] == BARE_GROQ_MODEL
        seen[tier] = mock_llm.calls[-1]["kwargs"].get("response_format")
    assert seen == {"primary": JSON_MODE, "fast": None}, seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [{}, {"custom_llm_provider": "   "}, {"custom_llm_provider": "impulse-no-such-provider"}],
    ids=["no_provider", "blank_provider", "unknown_provider"],
)
async def test_json_mode_stays_off_for_a_model_litellm_cannot_place(mock_llm, options):
    # Regression guard (passes before the fix too): the bare model string without a usable provider.
    mock_llm.queue_verdict("pass")
    verdict = await _judge_with(options).evaluate(_transcript(), CRITERION, "primary")
    assert verdict.result == "pass"
    assert "response_format" not in mock_llm.calls[0]["kwargs"], mock_llm.calls[0]["kwargs"]


@pytest.mark.asyncio
async def test_json_mode_keeps_a_response_format_from_the_options(mock_llm):
    # Regression guard (passes before the fix too): the role's own response_format is kept.
    mock_llm.queue_verdict("pass")
    options = {"custom_llm_provider": "groq", "response_format": {"type": "text"}}
    await _judge_with(options).evaluate(_transcript(), CRITERION, "primary")
    assert mock_llm.calls[0]["kwargs"].get("response_format") == {"type": "text"}
