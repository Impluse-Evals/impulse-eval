# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 3, A2 (DEFECT-9): a Pipecat tool call that raised is ``success=False`` with its error
(FIXES-ROUND-3-SPEC.md section A2).

pipecat-ai 1.12.0 (``pipecat/services/llm_service.py``), when a function handler raises: it pushes the
error ``f"Error executing function call [{function_name}]: {e}"`` (it reaches Impulse as an RTVI
``error`` message) and returns to the LLM the result ``"The function `{function_name}` failed and
returned no result."`` (it reaches Impulse in ``llm-function-call-stopped`` as a string result, stored as
``{"value": ...}``). The order of the two messages is not guaranteed.

* ``_on_call_stopped``: a raw string ``result`` matching the failure template (after ``strip()``) gives
  ``success=False`` and ``error`` = the RTVI error text tied to this call, else that result string; the
  stored result stays ``{"value": <string>}``. ``cancelled`` keeps priority; a dict result is unchanged.
* RTVI ``error`` / ``error-response`` text matching ``^Error executing function call \\[(name)\\]: `` is
  tied to the most recent call of the CURRENT turn with that name whose ``success`` is not True:
  ``success=False``, ``error`` = the text redacted, on one line, cut to 200 characters. No such call yet:
  pending for that name in this turn, tied to the next call with that name recorded or stopped in this
  turn; pending errors are dropped at the end of the turn. The run warning "Pipecat bot reported an RTVI
  error: {text}" is still added. A call already returned in an earlier ``AgentReply`` never changes.

Campaign case: run 20261025T143747Z-02afb8 (BankBot, Pipecat ``-t eval``) ``personal_data_request``,
turn 4: ``verify_identity`` was called with ``{}``, raised (missing arguments), and run.json stored
``success: null``. Trimmed data: ``tests/data/fixes_round3/pipecat_personal_data_request_02afb8.json``.
Adapter tests run on ``mock_pipecat_channel`` (no network, no pipecat-ai); runner tests use
``llm_router`` (mocked judge and simulator).
"""

import json
from pathlib import Path

import pytest
from conftest import pcm_tone, wav_bytes

pytestmark = pytest.mark.phase11

URL = "ws://localhost:7860"
MODES = ["eval", "websocket"]
BOT_READY = {"version": "2.1.0", "about": {"library": "pipecat-ai", "library_version": "1.12.0"}}
VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}

DATA = Path(__file__).resolve().parent / "data" / "fixes_round3"
CAMPAIGN = json.loads((DATA / "pipecat_personal_data_request_02afb8.json").read_text(encoding="utf-8"))
SCENARIO = CAMPAIGN["scenario_name"]                    # personal_data_request
CAMPAIGN_TURNS = CAMPAIGN["turns"]                      # turns 1-4 as stored
RTVI_WARNING = CAMPAIGN["warnings"][0]                  # "Pipecat bot reported an RTVI error: Error executing ..."
RTVI_ERROR = (
    "Error executing function call [verify_identity]: Session.verify_identity() missing 2 required "
    "positional arguments: 'customer_id' and 'dob'"
)
TEMPLATE = "The function `verify_identity` failed and returned no result."
RTVI_WARNING_PREFIX = "Pipecat bot reported an RTVI error: "
VERIFY_ARGS = {"customer_id": "CUST-2001", "dob": "1990-04-12"}
VERIFY_RESULT = {"ok": True, "customer_id": "CUST-2001", "name": "Ananya Rao"}
USER_TURN_3 = CAMPAIGN_TURNS[2]["text"]
AGENT_TURN_4 = CAMPAIGN_TURNS[3]["text"]


def test_campaign_data_matches_the_spec_texts():
    # The trimmed campaign data holds exactly the texts the spec quotes (sanity check of the fixture).
    assert RTVI_WARNING == RTVI_WARNING_PREFIX + RTVI_ERROR
    stored_call = CAMPAIGN_TURNS[3]["tool_calls"][0]
    assert stored_call["name"] == "verify_identity"
    assert stored_call["arguments"] == {}
    assert stored_call["result"] == {"value": TEMPLATE}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _cfg(project, mode="eval", *, voice=False, scenarios=None, **pipecat):
    block = {"mode": mode, "url": URL}
    block.update(pipecat)
    kwargs = {"voice_config": dict(VOICE_CONFIG)} if voice else {}
    project.write_config(
        scenarios or [project.scenario(SCENARIO)], agent={"adapter": "pipecat", "pipecat": block}, **kwargs
    )
    return project.load()


def _adapter(cfg, mock):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    return PipecatAdapter(cfg, channel_factory=mock.factory, clock=mock.clock)


def _handshake(mock, mode, channel=None):
    channel = channel or mock.channel
    if mode == "eval":
        channel.paced_audio = True
        channel.on_connect(mock.rtvi("bot-ready", BOT_READY, id="client-ready"))
    else:
        channel.on_send("client-ready", lambda message: mock.rtvi("bot-ready", BOT_READY, id=message.get("id")))
    return channel


async def _started(cfg, mock, mode):
    _handshake(mock, mode)
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    return adapter


def _plan(channel, steps):
    def trigger(_record):
        for offset, item in steps:
            channel.emit(item, after=offset)

    return trigger


def _on_text_turn(channel, steps):
    channel.on_send("send-text", _plan(channel, steps))


def _in_progress(mock, call_id, name, arguments):
    return mock.rtvi("llm-function-call-in-progress",
                     {"tool_call_id": call_id, "function_name": name, "arguments": arguments})


def _stopped(mock, call_id, name, result=None, cancelled=False):
    data = {"tool_call_id": call_id, "function_name": name, "cancelled": cancelled}
    if result is not None:
        data["result"] = result
    return mock.rtvi("llm-function-call-stopped", data)


def _rtvi_error(mock, text, kind="error"):
    """pipecat-ai 1.12.0 ``RTVIErrorMessage``: ``data = {"error": text, "fatal": false}``;
    ``error-response`` (``RTVIErrorResponse``) carries ``{"error": text}`` and the request id."""
    if kind == "error":
        return mock.rtvi("error", {"error": text, "fatal": False})
    return mock.rtvi("error-response", {"error": text}, id="impulse-request")


def _turn_steps(mock, middle, reply=AGENT_TURN_4):
    """One text turn: an LLM run that calls tools (``middle`` = (offset, item) pairs from +0.3 on), then
    the LLM run after the tool calls with the reply text."""
    rtvi = mock.rtvi
    return [
        (0.2, rtvi("bot-llm-started")),
        *middle,
        (0.9, rtvi("bot-llm-stopped")),
        (1.0, rtvi("bot-llm-started")),
        (1.1, rtvi("bot-llm-text", {"text": reply})),
        (1.2, rtvi("bot-llm-stopped")),
    ]


def _defect9_middle(mock, order, kind="error"):
    """The campaign's three messages for ``verify_identity`` in the given order."""
    in_progress = _in_progress(mock, "call-1", "verify_identity", {})
    error = _rtvi_error(mock, RTVI_ERROR, kind)
    stopped = _stopped(mock, "call-1", "verify_identity", result=TEMPLATE)
    sequence = {
        "error-before-stop": [in_progress, error, stopped],
        "error-after-stop": [in_progress, stopped, error],
        "error-before-call": [error, in_progress, stopped],
    }[order]
    return [(0.3 + 0.1 * index, item) for index, item in enumerate(sequence)]


async def _text_reply(project, mock, mode, middle, **cfg_kwargs):
    cfg = _cfg(project, mode, **cfg_kwargs)
    adapter = await _started(cfg, mock, mode)
    _on_text_turn(mock.channel, _turn_steps(mock, middle))
    reply = await adapter.send(USER_TURN_3, None)
    return adapter, reply


def _rtvi_warnings(adapter):
    return [warning for warning in adapter.get_warnings() if warning.startswith(RTVI_WARNING_PREFIX)]


# =============================================================================================
# The campaign shape, both message orders (and the error before the call itself)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", ["error-before-stop", "error-after-stop", "error-before-call"])
async def test_failed_call_gets_success_false_and_the_rtvi_error(impulse_project, mock_pipecat_channel, mode, order):
    mock = mock_pipecat_channel
    adapter, reply = await _text_reply(impulse_project, mock, mode, _defect9_middle(mock, order))
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call.name == "verify_identity"
    assert call.arguments == {}
    assert call.result == {"value": TEMPLATE}   # the stored result is unchanged
    assert call.success is False
    assert call.error == RTVI_ERROR
    assert reply.text == AGENT_TURN_4
    assert _rtvi_warnings(adapter) == [RTVI_WARNING]   # the run warning is still added
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", ["error-before-stop", "error-after-stop"])
async def test_error_response_message_is_tied_the_same_way(impulse_project, mock_pipecat_channel, mode, order):
    mock = mock_pipecat_channel
    adapter, reply = await _text_reply(impulse_project, mock, mode, _defect9_middle(mock, order, kind="error-response"))
    call = reply.tool_calls[0]
    assert (call.success, call.error) == (False, RTVI_ERROR)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_failure_template_without_rtvi_error_uses_the_template_as_error(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.5, _stopped(mock, "call-1", "verify_identity", result=TEMPLATE)),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    call = reply.tool_calls[0]
    assert call.success is False
    assert call.error == TEMPLATE
    assert call.result == {"value": TEMPLATE}
    assert _rtvi_warnings(adapter) == []
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_failure_template_is_matched_after_strip(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    raw = f"  {TEMPLATE}\n"
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.5, _stopped(mock, "call-1", "verify_identity", result=raw)),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    call = reply.tool_calls[0]
    assert call.success is False
    assert call.result == {"value": raw}   # stored as received
    assert (call.error or "").strip() == TEMPLATE
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_failure_template_for_a_call_recorded_only_from_its_stop(impulse_project, mock_pipecat_channel, mode):
    """The in-progress report was missed: the call is recorded from its stop and still fails."""
    mock = mock_pipecat_channel
    middle = [
        (0.3, _rtvi_error(mock, RTVI_ERROR)),
        (0.5, _stopped(mock, "call-1", "verify_identity", result=TEMPLATE)),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    assert [(call.name, call.success, call.error) for call in reply.tool_calls] == [
        ("verify_identity", False, RTVI_ERROR)
    ]
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", ["error-before-stop", "error-after-stop"])
async def test_voice_mode_failed_call_gets_the_rtvi_error(impulse_project, mock_pipecat_channel, mode, order):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice=True)
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.2, mock.BotAudio(mock.tone(0.3))),
        *[(offset - 0.05, item) for offset, item in _defect9_middle(mock, order)],   # +0.25 .. +0.45
        (0.55, rtvi("bot-output", {"text": AGENT_TURN_4, "spoken": True, "spoken_status": "completed",
                                   "aggregated_by": "sentence", "segment_id": 41})),
        (0.7, rtvi("bot-stopped-speaking")),
    ]

    def after_speech(_record):
        return sum(len(entry["pcm"]) // 2 for entry in channel.sent_audio) >= 8000 * 0.98

    channel.on_audio(_plan(channel, steps), match=after_speech)
    reply = await adapter.send(USER_TURN_3, wav_bytes(pcm_tone(0.5, 16000)))
    assert [(call.name, call.success, call.error) for call in reply.tool_calls] == [
        ("verify_identity", False, RTVI_ERROR)
    ]
    assert reply.tool_calls[0].result == {"value": TEMPLATE}
    await adapter.close()


# =============================================================================================
# Which call an RTVI error is tied to
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_later_successful_call_with_the_same_name_is_not_touched(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.35, _rtvi_error(mock, RTVI_ERROR)),
        (0.4, _stopped(mock, "call-1", "verify_identity", result=TEMPLATE)),
        (0.5, _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS)),
        (0.6, _stopped(mock, "call-2", "verify_identity", result={"success": True, **VERIFY_RESULT})),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    first, second = reply.tool_calls
    assert (first.arguments, first.success, first.error) == ({}, False, RTVI_ERROR)
    assert (second.arguments, second.success, second.error) == (VERIFY_ARGS, True, None)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_late_rtvi_error_skips_a_successful_call_and_ties_to_the_failed_one(
    impulse_project, mock_pipecat_channel, mode
):
    """Searched from the last call of the turn: the successful call-2 is skipped, call-1 (failed with
    the template, ``success`` not True) gets the RTVI error."""
    mock = mock_pipecat_channel
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.4, _stopped(mock, "call-1", "verify_identity", result=TEMPLATE)),
        (0.5, _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS)),
        (0.6, _stopped(mock, "call-2", "verify_identity", result={"success": True, **VERIFY_RESULT})),
        (0.7, _rtvi_error(mock, RTVI_ERROR)),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    first, second = reply.tool_calls
    assert (first.success, first.error) == (False, RTVI_ERROR)
    assert (second.success, second.error) == (True, None)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_rtvi_error_ties_to_the_most_recent_unfinished_call_with_that_name(
    impulse_project, mock_pipecat_channel, mode
):
    mock = mock_pipecat_channel
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {"customer_id": "CUST-2001"})),
        (0.35, _in_progress(mock, "call-2", "verify_identity", {})),
        (0.4, _rtvi_error(mock, RTVI_ERROR)),
        (0.5, _stopped(mock, "call-2", "verify_identity", result=TEMPLATE)),
        (0.6, _stopped(mock, "call-1", "verify_identity", result={"success": True, **VERIFY_RESULT})),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    first, second = reply.tool_calls
    assert (first.success, first.error) == (True, None)
    assert (second.success, second.error) == (False, RTVI_ERROR)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_rtvi_error_naming_another_function_ties_to_nothing(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    other = "Error executing function call [lookup_balance]: Session.lookup_balance() missing 1 required positional argument: 'account'"
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.35, _rtvi_error(mock, other)),
        (0.4, _stopped(mock, "call-1", "verify_identity", result=TEMPLATE)),
        (0.5, _in_progress(mock, "call-2", "send_otp", {"customer_id": "CUST-2001"})),
        (0.6, _stopped(mock, "call-2", "send_otp", result="sent")),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    verify, otp = reply.tool_calls
    assert (verify.success, verify.error) == (False, TEMPLATE)   # its own template, not the other error
    assert (otp.success, otp.error) == (None, None)              # a plain string result: not captured
    assert otp.result == {"value": "sent"}
    assert _rtvi_warnings(adapter) == [RTVI_WARNING_PREFIX + other]
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_rtvi_error_ties_even_when_the_call_has_no_stop_yet(impulse_project, mock_pipecat_channel, mode):
    """Tied at once: ``success=False`` and the error, even before ``llm-function-call-stopped``."""
    mock = mock_pipecat_channel
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.4, _rtvi_error(mock, RTVI_ERROR)),
        (0.5, _stopped(mock, "call-1", "verify_identity")),   # no result reported
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    call = reply.tool_calls[0]
    assert (call.success, call.error) == (False, RTVI_ERROR)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tied_error_text_is_redacted_then_one_line_and_cut_to_200(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    secret = "hdr-SECRET-7f3a91c2"
    long_text = (
        f"Error executing function call [verify_identity]: upstream rejected header {secret}\n"
        "with a traceback line\n" + "x" * 300
    )
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.4, _rtvi_error(mock, long_text)),
        (0.5, _stopped(mock, "call-1", "verify_identity", result=TEMPLATE)),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle, headers={"X-Bot-Token": secret})
    call = reply.tool_calls[0]
    assert call.success is False
    assert secret not in call.error
    assert "[REDACTED]" in call.error
    assert "\n" not in call.error
    assert len(call.error) == 200
    expected = " ".join(long_text.replace(secret, "[REDACTED]").split())[:200]
    assert call.error == expected
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_pending_rtvi_error_is_dropped_at_the_end_of_the_turn(impulse_project, mock_pipecat_channel, mode):
    """An RTVI error with no call of that name in its turn ties to nothing in a later turn."""
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    _on_text_turn(mock.channel, _turn_steps(mock, [(0.3, _rtvi_error(mock, RTVI_ERROR))], reply="One moment."))
    first = await adapter.send(USER_TURN_3, None)
    assert first.tool_calls == []
    middle = [
        (0.3, _in_progress(mock, "call-9", "verify_identity", {})),
        (0.4, _stopped(mock, "call-9", "verify_identity", result=TEMPLATE)),
    ]
    _on_text_turn(mock.channel, _turn_steps(mock, middle))
    second = await adapter.send("Yes, you have my consent.", None)
    call = second.tool_calls[0]
    assert (call.success, call.error) == (False, TEMPLATE)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_call_returned_in_an_earlier_reply_is_never_changed(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.4, _stopped(mock, "call-1", "verify_identity", result=TEMPLATE)),
    ]
    _on_text_turn(mock.channel, _turn_steps(mock, middle))
    first = await adapter.send(USER_TURN_3, None)
    assert (first.tool_calls[0].success, first.tool_calls[0].error) == (False, TEMPLATE)
    # The RTVI error arrives late, in the next turn: it does not reach the returned call.
    _on_text_turn(mock.channel, _turn_steps(mock, [(0.3, _rtvi_error(mock, RTVI_ERROR))], reply="Thank you."))
    second = await adapter.send("Yes, you have my consent.", None)
    assert (first.tool_calls[0].success, first.tool_calls[0].error) == (False, TEMPLATE)
    assert second.tool_calls == []
    assert _rtvi_warnings(adapter) == [RTVI_WARNING]
    await adapter.close()


# =============================================================================================
# Guards: the explicit rules keep priority (behaviour that must not change)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_cancelled_call_keeps_cancelled_first(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix).
    mock = mock_pipecat_channel
    middle = [
        (0.3, _in_progress(mock, "call-1", "verify_identity", {})),
        (0.4, _stopped(mock, "call-1", "verify_identity", result=TEMPLATE, cancelled=True)),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    call = reply.tool_calls[0]
    assert (call.success, call.error) == (False, "cancelled")
    assert call.result == {"value": TEMPLATE}
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_existing_dict_result_rules_are_unchanged(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix): the fixes round 2 success rules for dict results, and a
    # plain string result that is not the failure template.
    mock = mock_pipecat_channel
    middle = [
        (0.30, _in_progress(mock, "c1", "verify_identity", VERIFY_ARGS)),
        (0.31, _in_progress(mock, "c2", "send_otp", {"customer_id": "CUST-2001"})),
        (0.32, _in_progress(mock, "c3", "update_address", {"address": "Pune"})),
        (0.33, _in_progress(mock, "c4", "lookup_balance", {"customer_id": "CUST-2001"})),
        (0.34, _in_progress(mock, "c5", "log_event", {"level": "info"})),
        (0.40, _stopped(mock, "c1", "verify_identity", result={"success": True, **VERIFY_RESULT})),
        (0.41, _stopped(mock, "c2", "send_otp", result={"error": "OTP service down"})),
        (0.42, _stopped(mock, "c3", "update_address", result={"success": False, "error": "OTP required"})),
        (0.43, _stopped(mock, "c4", "lookup_balance", result={"value": TEMPLATE})),   # a dict: unchanged
        (0.44, _stopped(mock, "c5", "log_event", result="The function log_event finished.")),
    ]
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    calls = {call.name: call for call in reply.tool_calls}
    assert (calls["verify_identity"].success, calls["verify_identity"].error) == (True, None)
    assert (calls["send_otp"].success, calls["send_otp"].error) == (False, "OTP service down")
    assert calls["update_address"].success is False
    assert calls["lookup_balance"].result == {"value": TEMPLATE}
    assert (calls["lookup_balance"].success, calls["lookup_balance"].error) == (None, None)
    assert calls["log_event"].result == {"value": "The function log_event finished."}
    assert (calls["log_event"].success, calls["log_event"].error) == (None, None)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_dict_result_with_ok_false_is_unchanged_in_the_adapter_and_inferred_by_the_runner(
    impulse_project, llm_router, mock_pipecat_channel, mode
):
    # Regression guard (passes before the fix): the adapter stores the dict as is; the runner's
    # infer_success turns ok:false into success=False (fixes round 2 A1.1).
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter
    from impulse_evals.runner import run

    mock = mock_pipecat_channel
    project = impulse_project
    failed = {"ok": False, "reason": "customer not found"}
    cfg = _cfg(project, mode, scenarios=[project.scenario(SCENARIO, max_turns=1, metrics=["UNIV-001"])])
    _handshake(mock, mode)
    _on_text_turn(mock.channel, _turn_steps(mock, [
        (0.3, _in_progress(mock, "call-1", "verify_identity", VERIFY_ARGS)),
        (0.4, _stopped(mock, "call-1", "verify_identity", result=failed)),
    ]))
    adapters = []

    def factory(config):
        adapter = PipecatAdapter(config, channel_factory=mock.factory, clock=mock.clock)
        adapters.append(adapter)
        return adapter

    result = await run(cfg, adapter_factory=factory)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    stored = scenario_result.transcript.turns[1].tool_calls[0]
    assert stored.result == failed
    assert stored.success is False
    assert len(adapters) == 1


# =============================================================================================
# Runner: run.json and the AGENT TOOL CALL line
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["error-before-stop", "error-after-stop"])
async def test_runner_stores_the_failed_call_and_the_tool_call_line_shows_it(
    impulse_project, llm_router, mock_pipecat_channel, tmp_path, order
):
    """The campaign scenario through ``run()`` (eval mode, mocked judge and simulator): turn 4 holds the
    failed ``verify_identity`` call with ``success: false`` and the RTVI error in run.json, and its
    AGENT TOOL CALL line shows ``success=False error=...``."""
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter
    from impulse_evals.models import RunResult
    from impulse_evals.reporter import write_json
    from impulse_evals.runner import run
    from impulse_evals.transcriber import format_transcript

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    project = impulse_project
    opening, first_reply = CAMPAIGN_TURNS[0]["text"], CAMPAIGN_TURNS[1]["text"]
    cfg = _cfg(project, "eval", scenarios=[project.scenario(
        SCENARIO, max_turns=2, opening_user_message=opening, metrics=["UNIV-001", "UNIV-023"],
    )])
    llm_router.script(project.persona(SCENARIO), USER_TURN_3)
    channel = _handshake(mock, "eval")

    def sent_text(text):
        return lambda data: data.get("type") == "send-text" and data["data"]["content"] == text

    channel.on_send(sent_text(opening), _plan(channel, [
        (0.2, rtvi("bot-llm-started")), (0.3, rtvi("bot-llm-text", {"text": first_reply})), (0.4, rtvi("bot-llm-stopped")),
    ]))
    channel.on_send(sent_text(USER_TURN_3), _plan(channel, _turn_steps(mock, _defect9_middle(mock, order))))
    result = await run(cfg, adapter_factory=lambda config: PipecatAdapter(config, channel_factory=mock.factory, clock=mock.clock))

    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert turns[3].text == AGENT_TURN_4
    assert RTVI_WARNING in result.warnings

    path = tmp_path / "run.json"
    write_json(result, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    stored_call = raw["scenario_results"][0]["transcript"]["turns"][3]["tool_calls"][0]
    assert stored_call["name"] == "verify_identity"
    assert stored_call["arguments"] == {}
    assert stored_call["result"] == {"value": TEMPLATE}
    assert stored_call["success"] is False
    assert stored_call["error"] == RTVI_ERROR

    stored = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    lines = format_transcript(stored.scenario_results[0].transcript).splitlines()
    tool_lines = [line for line in lines if "AGENT TOOL CALL" in line]
    assert len(tool_lines) == 1
    assert tool_lines[0].startswith(
        f"[turn 4] AGENT TOOL CALL: verify_identity({{}}) -> success=False error={RTVI_ERROR}"
    ), tool_lines[0]
