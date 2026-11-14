# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 4, L15 (FINAL-TEST.md): sensitive keys are redacted in tool-call ARGUMENTS too.

The runner stores every tool call's ``arguments`` with the value of every sensitive key (the same key
rule as tool results: ``transcriber.redact_tool_result``, so ``otp``, ``password``, ``pin``, ``token``,
``card_number``, plurals, at any depth) replaced by ``"[REDACTED]"``; ``true`` / ``false`` / ``null`` stay.
So run.json, the HTML report and the judge transcript (the AGENT TOOL CALL line) never carry them.
Other argument values are unchanged and arguments are never cut. UNIV-025 compares against the stored
(redacted) arguments: a required sensitive argument counts as present; an ``argument_values`` entry for a
sensitive key cannot match (documented).

A custom adapter returns the calls (no network); the judge and the simulator are ``mock_llm`` routed by
``llm_router``.
"""

import importlib
import json
import uuid

import pytest

OTP = "483921"
PIN = "7710"
CARD = "4111111111111111"

ADAPTER = '''
from impulse_evals.adapters.base import AgentInterface, AgentReply
from impulse_evals.models import ToolCall


class OtpAdapter(AgentInterface):
    def __init__(self, config):
        self.scenario = None

    async def start(self, scenario):
        self.scenario = scenario

    async def send(self, user_text, user_audio):
        call = ToolCall(
            name="verify_otp",
            arguments={
                "customer_id": "CUST-2001",
                "otp": "483921",
                "card_number": "4111111111111111",
                "otp_sent": True,
                "pin_code": None,
                "nested": {"user_pin": "7710", "note": "keep"},
                "codes": [{"otp": "483921"}, "plain"],
            },
            result={"ok": True, "verified": True},
        )
        return AgentReply(text="Verified, thank you.", latency_ms=10.0, tool_calls=[call])

    async def close(self):
        return None
'''


def _config(project, **fields):
    stem = f"impulse_test_otp_{uuid.uuid4().hex[:10]}"
    (project.root / f"{stem}.py").write_text("# SPDX-License-Identifier: AGPL-3.0-only\n" + ADAPTER, encoding="utf-8")
    importlib.invalidate_caches()
    values = {"max_turns": 1, "metrics": ["UNIV-023"]}
    values.update(fields)
    project.write_config([project.scenario("otp_check", **values)], agent={"adapter": "custom", "entrypoint": f"{stem}:OtpAdapter"})
    return project.load()


async def _run(cfg):
    from impulse_evals.runner import run

    return await run(cfg)


def _call(result):
    agent_turns = [turn for turn in result.scenario_results[0].transcript.turns if turn.speaker == "agent"]
    return agent_turns[0].tool_calls[0]


@pytest.mark.asyncio
async def test_stored_arguments_have_sensitive_values_redacted(impulse_project, llm_router):
    result = await _run(_config(impulse_project))
    call = _call(result)
    assert call.arguments == {
        "customer_id": "CUST-2001",
        "otp": "[REDACTED]",
        "card_number": "[REDACTED]",
        "otp_sent": True,
        "pin_code": None,
        "nested": {"user_pin": "[REDACTED]", "note": "keep"},
        "codes": [{"otp": "[REDACTED]"}, "plain"],
    }
    assert call.result == {"ok": True, "verified": True} and call.success is True
    dumped = json.dumps(result.model_dump(mode="json"))
    assert OTP not in dumped and PIN not in dumped and CARD not in dumped


@pytest.mark.asyncio
async def test_judge_transcript_line_and_report_carry_no_sensitive_argument(impulse_project, llm_router, tmp_path):
    from impulse_evals.reporter import render_html
    from impulse_evals.transcriber import format_transcript

    result = await _run(_config(impulse_project))
    lines = format_transcript(result.scenario_results[0].transcript).splitlines()
    line = [text for text in lines if "AGENT TOOL CALL" in text][0]
    assert 'verify_otp({' in line and '"otp": "[REDACTED]"' in line and '"customer_id": "CUST-2001"' in line
    assert OTP not in line and PIN not in line and CARD not in line
    judge_texts = "\n".join(text for call in llm_router.calls("judge") for text in llm_router.contents(call))
    assert "verify_otp" in judge_texts
    assert OTP not in judge_texts and PIN not in judge_texts and CARD not in judge_texts
    html = render_html(result, embed_audio=False, run_dir=tmp_path)
    assert "verify_otp" in html
    assert OTP not in html and PIN not in html and CARD not in html


@pytest.mark.asyncio
async def test_univ025_required_sensitive_argument_counts_as_present(impulse_project, llm_router):
    cfg = _config(impulse_project, expected_tool_calls=[{"name": "verify_otp", "required_arguments": ["customer_id", "otp"]}])
    result = await _run(cfg)
    univ025 = [m for m in result.scenario_results[0].metric_results if m.metric_id == "UNIV-025"][0]
    assert univ025.label == "good", univ025.reason


def test_redact_tool_arguments_helper_matches_tool_result_rule():
    """The same function redacts arguments and results (``transcriber.redact_tool_result``)."""
    from impulse_evals.transcriber import redact_tool_result

    assert redact_tool_result({"otp": "1", "otp_verified": False, "x": "2"}) == {"otp": "[REDACTED]", "otp_verified": False, "x": "2"}


def test_empty_sensitive_values_stay_empty():
    """Stage D fix-up (round 4 review, finding 1): an empty value carries no secret, and turning it into
    "[REDACTED]" would make UNIV-025's ``required_arguments`` count a missing OTP as present."""
    from impulse_evals.transcriber import redact_tool_result

    value = {"otp": "", "pin": [], "token": {}, "password": None, "card_number": 0, "secret_code": "x"}
    assert redact_tool_result(value) == {
        "otp": "", "pin": [], "token": {}, "password": None, "card_number": "[REDACTED]", "secret_code": "[REDACTED]"
    }


EMPTY_OTP_ADAPTER = ADAPTER.replace('"otp": "483921",', '"otp": "",')


@pytest.mark.asyncio
async def test_univ025_required_sensitive_argument_that_is_empty_is_still_missing(impulse_project, llm_router):
    stem = f"impulse_test_otp_{uuid.uuid4().hex[:10]}"
    (impulse_project.root / f"{stem}.py").write_text("# SPDX-License-Identifier: AGPL-3.0-only\n" + EMPTY_OTP_ADAPTER, encoding="utf-8")
    importlib.invalidate_caches()
    scenario = impulse_project.scenario(
        "otp_check", max_turns=1, metrics=["UNIV-023"],
        expected_tool_calls=[{"name": "verify_otp", "required_arguments": ["customer_id", "otp"]}],
    )
    impulse_project.write_config([scenario], agent={"adapter": "custom", "entrypoint": f"{stem}:OtpAdapter"})
    result = await _run(impulse_project.load())
    assert _call(result).arguments["otp"] == ""
    univ025 = [m for m in result.scenario_results[0].metric_results if m.metric_id == "UNIV-025"][0]
    assert univ025.label == "poor"
    assert univ025.reason == "tool verify_otp is missing required arguments: otp"


def test_format_tool_call_never_cuts_arguments():
    from impulse_evals.models import ToolCall
    from impulse_evals.transcriber import format_tool_call

    long = "x" * 1000
    line = format_tool_call(2, ToolCall(name="t", arguments={"text": long}))
    assert long in line and "truncated" not in line
