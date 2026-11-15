# SPDX-License-Identifier: Apache-2.0
"""Fixes round 3, A2-rev: the revised tie rule for Pipecat RTVI "Error executing function call" errors
(FIXES-ROUND-3-SPEC.md section A2-rev; Stage D review R3P-1..3). It SUPERSEDES the A2 tie rule where
they differ; the A2 tests in ``test_pipecat_tool_errors.py`` stay valid.

Revised rule (per turn, by function name; an RTVI error only ever enriches a FAILED or NOT-CAPTURED call,
never a successful one):

1. A stop whose result is the failure template: ``success=False``; ``error`` = the oldest pending RTVI
   error for that name in this turn (consumed), else the template string (stripped).
2. An RTVI error for name N: when a call of name N in this turn already failed through the template and
   still carries the template string as its error, the most recent such call gets the RTVI text
   (consumed); otherwise the error is pending for N in this turn.
3. When the turn's reply is built (normal end, timeout partial reply, close partial reply), each pending
   error for N still unconsumed ties to the most recent call of name N in this turn whose ``success`` is
   None and whose ``result`` is None: ``success=False``, ``error`` = the text. Pending errors left after
   that are dropped (they stay only the run warning). Pending errors never carry over to another turn.
4. A call with ``success=True``, a dict result, or ``cancelled`` is never changed by an RTVI error; no
   call ever ends with ``success=True`` and an error tied by these rules.

pipecat-ai runs same-name function calls in parallel by default (``run_in_parallel=True``), so two
``verify_identity`` calls of one LLM run can interleave in any order. Adapter tests run on
``mock_pipecat_channel`` (no network, no pipecat-ai); the runner test uses ``llm_router`` (mocked judge
and simulator).
"""

import itertools
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
CAMPAIGN_TURNS = CAMPAIGN["turns"]
RTVI_ERROR = (
    "Error executing function call [verify_identity]: Session.verify_identity() missing 2 required "
    "positional arguments: 'customer_id' and 'dob'"
)
TEMPLATE = "The function `verify_identity` failed and returned no result."
RTVI_WARNING_PREFIX = "Pipecat bot reported an RTVI error: "
RTVI_WARNING = RTVI_WARNING_PREFIX + RTVI_ERROR
VERIFY_ARGS = {"customer_id": "CUST-2001", "dob": "1990-04-12"}
VERIFY_RESULT = {"ok": True, "customer_id": "CUST-2001", "name": "Ananya Rao"}
SUCCESS_RESULTS = {
    "ok": dict(VERIFY_RESULT),                            # no success key: the adapter leaves success None
    "success": {"success": True, **VERIFY_RESULT},      # explicit success: the adapter sets success True
}
USER_TURN_3 = CAMPAIGN_TURNS[2]["text"]
AGENT_TURN_4 = CAMPAIGN_TURNS[3]["text"]
NEXT_USER = "Yes, you have my consent."


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
    data = {"function_name": name, "arguments": arguments}
    if call_id is not None:
        data["tool_call_id"] = call_id
    return mock.rtvi("llm-function-call-in-progress", data)


def _stopped(mock, call_id, name, result=None, cancelled=False):
    data = {"tool_call_id": call_id, "function_name": name, "cancelled": cancelled}
    if result is not None:
        data["result"] = result
    return mock.rtvi("llm-function-call-stopped", data)


def _deprecated_call(mock, name, args):
    """Older Pipecat: the deprecated ``llm-function-call`` (no stop ever follows, no result)."""
    return mock.rtvi("llm-function-call", {"function_name": name, "tool_call_id": None, "args": args})


def _rtvi_error(mock, text=RTVI_ERROR):
    """pipecat-ai 1.12.0 ``RTVIErrorMessage``: ``data = {"error": text, "fatal": false}``."""
    return mock.rtvi("error", {"error": text, "fatal": False})


def _turn_steps(mock, middle, reply=AGENT_TURN_4):
    """One text turn: an LLM run that calls tools (``middle`` = (offset, item) pairs from +0.3 to
    +0.8), then the LLM run after the tool calls with the reply text."""
    rtvi = mock.rtvi
    return [
        (0.2, rtvi("bot-llm-started")),
        *middle,
        (0.9, rtvi("bot-llm-stopped")),
        (1.0, rtvi("bot-llm-started")),
        (1.1, rtvi("bot-llm-text", {"text": reply})),
        (1.2, rtvi("bot-llm-stopped")),
    ]


def _spaced(items, start=0.3, step=0.1):
    return [(round(start + step * index, 3), item) for index, item in enumerate(items)]


async def _text_reply(project, mock, mode, middle, **cfg_kwargs):
    cfg = _cfg(project, mode, **cfg_kwargs)
    adapter = await _started(cfg, mock, mode)
    _on_text_turn(mock.channel, _turn_steps(mock, middle))
    reply = await adapter.send(USER_TURN_3, None)
    return adapter, reply


def _rtvi_warnings(adapter):
    return [warning for warning in adapter.get_warnings() if warning.startswith(RTVI_WARNING_PREFIX)]


def _assert_no_success_with_error(calls):
    """Rule 4: no call ever ends with ``success=True`` and an error text."""
    for call in calls:
        assert not (call.success is True and call.error), (call.name, call.arguments, call.success, call.error)


def _orders(labels, before):
    """Every permutation of ``labels`` in which each (first, second) pair of ``before`` keeps its order
    (a call's in-progress report before its own stop)."""
    result = []
    for order in itertools.permutations(labels):
        if all(order.index(first) < order.index(second) for first, second in before):
            result.append(order)
    return result


def _order_id(order):
    return "-".join(order)


def _assert_success_untouched(call, variant):
    """The successful call: its result as received, success per its result, no error."""
    assert call.result == SUCCESS_RESULTS[variant]
    assert call.error is None
    if variant == "success":
        assert call.success is True
    else:
        assert call.success is None   # {"ok": true}: not captured by the adapter (the runner infers it)


# =============================================================================================
# R3P-1: two parallel same-name calls, one raising (template + RTVI error), one succeeding,
# in every arrival order
# =============================================================================================

# ip1/stop1: call-1 (arguments {}) raises: its stop carries the failure template; err: its RTVI error.
# ip2/stop2: call-2 (full arguments) succeeds. Each in-progress report comes before its own stop.
PARALLEL_ORDERS = _orders(("ip1", "ip2", "stop1", "stop2", "err"), before=[("ip1", "stop1"), ("ip2", "stop2")])


def _parallel_items(mock, variant):
    return {
        "ip1": _in_progress(mock, "call-1", "verify_identity", {}),
        "ip2": _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS),
        "stop1": _stopped(mock, "call-1", "verify_identity", result=TEMPLATE),
        "stop2": _stopped(mock, "call-2", "verify_identity", result=SUCCESS_RESULTS[variant]),
        "err": _rtvi_error(mock),
    }


def _split(calls):
    failed = [call for call in calls if call.arguments == {}]
    succeeded = [call for call in calls if call.arguments == VERIFY_ARGS]
    assert len(failed) == 1 and len(succeeded) == 1, [(call.arguments, call.success, call.error) for call in calls]
    return failed[0], succeeded[0]


def test_parallel_orders_cover_every_valid_arrival_order():
    # 5 messages, two "in-progress before its own stop" constraints: 5! / 4 = 30 orders.
    assert len(PARALLEL_ORDERS) == 30
    assert len(set(PARALLEL_ORDERS)) == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["ok", "success"])
@pytest.mark.parametrize("order", PARALLEL_ORDERS, ids=_order_id)
async def test_parallel_same_name_calls_failed_one_gets_the_rtvi_error_successful_one_untouched(
    impulse_project, mock_pipecat_channel, mode, variant, order
):
    mock = mock_pipecat_channel
    items = _parallel_items(mock, variant)
    adapter, reply = await _text_reply(impulse_project, mock, mode, _spaced([items[label] for label in order]))
    assert len(reply.tool_calls) == 2
    failed, succeeded = _split(reply.tool_calls)
    assert failed.name == succeeded.name == "verify_identity"
    assert failed.result == {"value": TEMPLATE}
    assert (failed.success, failed.error) == (False, RTVI_ERROR)
    _assert_success_untouched(succeeded, variant)
    _assert_no_success_with_error(reply.tool_calls)
    assert _rtvi_warnings(adapter) == [RTVI_WARNING]   # the run warning is still added, once
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["ok", "success"])
@pytest.mark.parametrize(
    "order",
    [("ip1", "ip2", "err", "stop1", "stop2"), ("ip1", "ip2", "err", "stop2", "stop1"), ("ip2", "ip1", "stop2", "err", "stop1")],
    ids=_order_id,
)
async def test_voice_mode_parallel_same_name_calls(impulse_project, mock_pipecat_channel, mode, variant, order):
    """The same rule in voice mode (the tie logic does not depend on the end-of-turn rule)."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice=True)
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    items = _parallel_items(mock, variant)
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.2, mock.BotAudio(mock.tone(0.3))),
        *_spaced([items[label] for label in order], start=0.25, step=0.05),   # +0.25 .. +0.45
        (0.55, rtvi("bot-output", {"text": AGENT_TURN_4, "spoken": True, "spoken_status": "completed",
                                   "aggregated_by": "sentence", "segment_id": 41})),
        (0.7, rtvi("bot-stopped-speaking")),
    ]

    def after_speech(_record):
        return sum(len(entry["pcm"]) // 2 for entry in channel.sent_audio) >= 8000 * 0.98

    channel.on_audio(_plan(channel, steps), match=after_speech)
    reply = await adapter.send(USER_TURN_3, wav_bytes(pcm_tone(0.5, 16000)))
    failed, succeeded = _split(reply.tool_calls)
    assert (failed.success, failed.error) == (False, RTVI_ERROR)
    _assert_success_untouched(succeeded, variant)
    _assert_no_success_with_error(reply.tool_calls)
    await adapter.close()


# =============================================================================================
# Rule 4: a single successful call is never changed by an RTVI error of its name
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["ok", "success"])
@pytest.mark.parametrize("position", ["error-before-call", "error-before-stop", "error-after-stop"])
async def test_successful_call_is_never_marked_by_an_rtvi_error(
    impulse_project, mock_pipecat_channel, mode, variant, position
):
    """No call of the turn failed or is not captured: the error stays only the run warning."""
    mock = mock_pipecat_channel
    in_progress = _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS)
    stopped = _stopped(mock, "call-2", "verify_identity", result=SUCCESS_RESULTS[variant])
    error = _rtvi_error(mock)
    sequence = {
        "error-before-call": [error, in_progress, stopped],
        "error-before-stop": [in_progress, error, stopped],
        "error-after-stop": [in_progress, stopped, error],
    }[position]
    adapter, reply = await _text_reply(impulse_project, mock, mode, _spaced(sequence))
    assert len(reply.tool_calls) == 1
    _assert_success_untouched(reply.tool_calls[0], variant)
    assert _rtvi_warnings(adapter) == [RTVI_WARNING]
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_pending_error_then_explicit_success_stop_without_in_progress(impulse_project, mock_pipecat_channel, mode):
    """R3P-3 via the pending path: the RTVI error first, then a stop (no in-progress report) whose dict
    result has ``"success": true``: success True and NO error."""
    mock = mock_pipecat_channel
    middle = _spaced([
        _rtvi_error(mock),
        _stopped(mock, "call-2", "verify_identity", result=SUCCESS_RESULTS["success"]),
    ])
    adapter, reply = await _text_reply(impulse_project, mock, mode, middle)
    call = reply.tool_calls[0]
    assert (call.success, call.error) == (True, None)
    _assert_no_success_with_error(reply.tool_calls)
    await adapter.close()


# =============================================================================================
# R3P-2: an RTVI error read in turn 2 for a call of turn 1
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["ok", "success"])
@pytest.mark.parametrize("position", ["backlog", "before-call", "before-stop", "after-stop"])
async def test_rtvi_error_of_a_turn_1_call_read_in_turn_2_marks_nothing_in_turn_2(
    impulse_project, mock_pipecat_channel, mode, variant, position
):
    """Turn 1: call-1 ``verify_identity`` stops with the template (its RTVI error has not arrived).
    The RTVI error is read in turn 2 (``backlog``: it arrived after turn 1 ended and was queued before
    the next send; otherwise it arrives during turn 2 at the given position). Turn 2's call-2 of the
    same name succeeds: it is untouched, and the call returned in turn 1 never changes."""
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    turn_1 = _turn_steps(mock, _spaced([
        _in_progress(mock, "call-1", "verify_identity", {}),
        _stopped(mock, "call-1", "verify_identity", result=TEMPLATE),
    ]))
    if position == "backlog":
        turn_1 = turn_1 + [(5.0, _rtvi_error(mock))]
    _on_text_turn(mock.channel, turn_1)
    first = await adapter.send(USER_TURN_3, None)
    assert [(call.success, call.error) for call in first.tool_calls] == [(False, TEMPLATE)]
    if position == "backlog":
        await mock.clock.sleep(10.0)   # the error is read after turn 1 ended, before the next send

    in_progress = _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS)
    stopped = _stopped(mock, "call-2", "verify_identity", result=SUCCESS_RESULTS[variant])
    error = _rtvi_error(mock)
    sequence = {
        "backlog": [in_progress, stopped],
        "before-call": [error, in_progress, stopped],
        "before-stop": [in_progress, error, stopped],
        "after-stop": [in_progress, stopped, error],
    }[position]
    _on_text_turn(mock.channel, _turn_steps(mock, _spaced(sequence)))
    second = await adapter.send(NEXT_USER, None)
    assert len(second.tool_calls) == 1
    _assert_success_untouched(second.tool_calls[0], variant)
    _assert_no_success_with_error(second.tool_calls)
    assert [(call.success, call.error) for call in first.tool_calls] == [(False, TEMPLATE)]   # copy rule
    assert _rtvi_warnings(adapter) == [RTVI_WARNING]
    await adapter.close()


# =============================================================================================
# Rule 3: an RTVI error whose call's stop never arrives (or arrives without a result)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["deprecated", "no-call-id"])
@pytest.mark.parametrize("position", ["error-before-call", "error-after-call"])
async def test_call_without_a_stop_takes_the_rtvi_error_at_the_end_of_the_turn(
    impulse_project, mock_pipecat_channel, mode, kind, position
):
    """The call's stop never arrives before the turn ends in the grace window (a deprecated
    ``llm-function-call``, or an in-progress report with no ``tool_call_id``, which does not hold the turn
    open): ``success=False`` with the RTVI text."""
    mock = mock_pipecat_channel
    call = (_deprecated_call(mock, "verify_identity", {}) if kind == "deprecated"
            else _in_progress(mock, None, "verify_identity", {}))
    error = _rtvi_error(mock)
    sequence = [error, call] if position == "error-before-call" else [call, error]
    adapter, reply = await _text_reply(impulse_project, mock, mode, _spaced(sequence))
    assert [(c.name, c.arguments, c.success, c.error) for c in reply.tool_calls] == [
        ("verify_identity", {}, False, RTVI_ERROR)
    ]
    assert reply.text == AGENT_TURN_4
    await adapter.close()


# call1: call-1 recorded with no stop (deprecated llm-function-call); ip2/stop2: call-2 succeeds.
NO_STOP_ORDERS = _orders(("call1", "ip2", "stop2", "err"), before=[("ip2", "stop2")])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["ok", "success"])
@pytest.mark.parametrize("order", NO_STOP_ORDERS, ids=_order_id)
async def test_error_goes_to_the_call_without_a_stop_not_to_the_successful_one(
    impulse_project, mock_pipecat_channel, mode, variant, order
):
    mock = mock_pipecat_channel
    items = {
        "call1": _deprecated_call(mock, "verify_identity", {}),
        "ip2": _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS),
        "stop2": _stopped(mock, "call-2", "verify_identity", result=SUCCESS_RESULTS[variant]),
        "err": _rtvi_error(mock),
    }
    adapter, reply = await _text_reply(impulse_project, mock, mode, _spaced([items[label] for label in order]))
    failed, succeeded = _split(reply.tool_calls)
    assert (failed.success, failed.error, failed.result) == (False, RTVI_ERROR, None)
    _assert_success_untouched(succeeded, variant)
    _assert_no_success_with_error(reply.tool_calls)
    await adapter.close()


# ip1/stop1: call-1 whose stop carries NO result; ip2/stop2: call-2 succeeds.
NO_RESULT_ORDERS = _orders(("ip1", "stop1", "ip2", "stop2", "err"), before=[("ip1", "stop1"), ("ip2", "stop2")])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", NO_RESULT_ORDERS, ids=_order_id)
async def test_error_goes_to_the_call_whose_stop_had_no_result(impulse_project, mock_pipecat_channel, mode, order):
    """A stop without a result (not captured) is enriched; the ``{"ok": true}`` call (a dict result,
    success not captured by the adapter) is not."""
    mock = mock_pipecat_channel
    items = {
        "ip1": _in_progress(mock, "call-1", "verify_identity", {}),
        "stop1": _stopped(mock, "call-1", "verify_identity"),
        "ip2": _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS),
        "stop2": _stopped(mock, "call-2", "verify_identity", result=SUCCESS_RESULTS["ok"]),
        "err": _rtvi_error(mock),
    }
    adapter, reply = await _text_reply(impulse_project, mock, mode, _spaced([items[label] for label in order]))
    failed, succeeded = _split(reply.tool_calls)
    assert (failed.success, failed.error, failed.result) == (False, RTVI_ERROR, None)
    _assert_success_untouched(succeeded, "ok")
    await adapter.close()


# ip1: call-1 in progress, never stopped (the turn times out / the bot closes); ip2/stop2: call-2 succeeds.
PARTIAL_ORDERS = _orders(("ip1", "ip2", "stop2", "err"), before=[("ip2", "stop2")])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["ok", "success"])
@pytest.mark.parametrize("order", PARTIAL_ORDERS, ids=_order_id)
async def test_timeout_partial_reply_ties_the_error_to_the_call_still_in_progress(
    impulse_project, mock_pipecat_channel, mode, variant, order
):
    from impulse_evals.adapters.pipecat_adapter import PipecatTurnTimeout

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, turn_timeout_s=5)
    adapter = await _started(cfg, mock, mode)
    items = {
        "ip1": _in_progress(mock, "call-1", "verify_identity", {}),
        "ip2": _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS),
        "stop2": _stopped(mock, "call-2", "verify_identity", result=SUCCESS_RESULTS[variant]),
        "err": _rtvi_error(mock),
    }
    _on_text_turn(mock.channel, [
        (0.2, rtvi("bot-llm-started")),
        *_spaced([items[label] for label in order]),
        (0.9, rtvi("bot-llm-stopped")),   # call-1 is still in progress: the turn stays open
    ])
    with pytest.raises(PipecatTurnTimeout) as info:
        await adapter.send(USER_TURN_3, None)
    reply = info.value.reply
    assert reply is not None
    failed, succeeded = _split(reply.tool_calls)
    assert (failed.success, failed.error, failed.result) == (False, RTVI_ERROR, None)
    _assert_success_untouched(succeeded, variant)
    _assert_no_success_with_error(reply.tool_calls)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("position", ["error-before-call", "error-after-call"])
async def test_timeout_partial_reply_single_call_without_a_stop(impulse_project, mock_pipecat_channel, mode, position):
    from impulse_evals.adapters.pipecat_adapter import PipecatTurnTimeout

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, turn_timeout_s=5)
    adapter = await _started(cfg, mock, mode)
    call = _in_progress(mock, "call-1", "verify_identity", {})
    error = _rtvi_error(mock)
    sequence = [error, call] if position == "error-before-call" else [call, error]
    _on_text_turn(mock.channel, [(0.2, rtvi("bot-llm-started")), *_spaced(sequence), (0.9, rtvi("bot-llm-stopped"))])
    with pytest.raises(PipecatTurnTimeout) as info:
        await adapter.send(USER_TURN_3, None)
    reply = info.value.reply
    assert [(c.success, c.error) for c in reply.tool_calls] == [(False, RTVI_ERROR)]
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["ok", "success"])
@pytest.mark.parametrize(
    "order", [("ip1", "ip2", "stop2", "err"), ("err", "ip2", "stop2", "ip1"), ("ip2", "err", "ip1", "stop2")], ids=_order_id
)
async def test_close_partial_reply_ties_the_error_to_the_call_still_in_progress(
    impulse_project, mock_pipecat_channel, mode, variant, order
):
    """The bot hangs up (close 1000) while call-1 is in progress: the ``AgentClosed`` reply."""
    from impulse_evals.adapters.base import AgentClosed

    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    items = {
        "ip1": _in_progress(mock, "call-1", "verify_identity", {}),
        "ip2": _in_progress(mock, "call-2", "verify_identity", VERIFY_ARGS),
        "stop2": _stopped(mock, "call-2", "verify_identity", result=SUCCESS_RESULTS[variant]),
        "err": _rtvi_error(mock),
    }
    _on_text_turn(mock.channel, [
        (0.2, rtvi("bot-llm-started")),
        *_spaced([items[label] for label in order]),
        (0.9, mock.BotClose(1000)),
    ])
    with pytest.raises(AgentClosed) as info:
        await adapter.send(USER_TURN_3, None)
    reply = info.value.reply
    assert reply is not None
    failed, succeeded = _split(reply.tool_calls)
    assert (failed.success, failed.error) == (False, RTVI_ERROR)
    _assert_success_untouched(succeeded, variant)
    _assert_no_success_with_error(reply.tool_calls)
    await adapter.close()


# =============================================================================================
# An RTVI error naming no call of the turn; pending errors never carry over
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("position", ["first", "middle", "last"])
async def test_rtvi_error_naming_no_call_of_the_turn_marks_nothing(impulse_project, mock_pipecat_channel, mode, position):
    """The turn's calls are of other names and not captured (no result): the verify_identity error ties
    to none of them; it stays only the run warning."""
    mock = mock_pipecat_channel
    calls = [
        _in_progress(mock, "call-1", "send_otp", {"customer_id": "CUST-2001"}),
        _stopped(mock, "call-1", "send_otp"),
        _deprecated_call(mock, "lookup_balance", {"customer_id": "CUST-2001"}),
    ]
    index = {"first": 0, "middle": 2, "last": 3}[position]
    sequence = calls[:index] + [_rtvi_error(mock)] + calls[index:]
    adapter, reply = await _text_reply(impulse_project, mock, mode, _spaced(sequence))
    assert [(c.name, c.success, c.error) for c in reply.tool_calls] == [
        ("send_otp", None, None),
        ("lookup_balance", None, None),
    ]
    assert _rtvi_warnings(adapter) == [RTVI_WARNING]
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("turn_1", ["no-call", "successful-call"])
async def test_pending_error_never_carries_over_to_the_next_turn(impulse_project, mock_pipecat_channel, mode, turn_1):
    """Turn 1: an RTVI error for verify_identity that no call of turn 1 takes (no such call, or only a
    successful one). Turn 2: a verify_identity call that is not captured (stop without a result) is not
    marked by turn 1's error."""
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    if turn_1 == "no-call":
        middle = _spaced([_rtvi_error(mock)])
    else:
        middle = _spaced([
            _in_progress(mock, "call-1", "verify_identity", VERIFY_ARGS),
            _rtvi_error(mock),
            _stopped(mock, "call-1", "verify_identity", result=SUCCESS_RESULTS["ok"]),
        ])
    _on_text_turn(mock.channel, _turn_steps(mock, middle, reply="One moment."))
    first = await adapter.send(USER_TURN_3, None)
    if turn_1 == "successful-call":
        _assert_success_untouched(first.tool_calls[0], "ok")
    else:
        assert first.tool_calls == []
    _on_text_turn(mock.channel, _turn_steps(mock, _spaced([
        _in_progress(mock, "call-9", "verify_identity", {}),
        _stopped(mock, "call-9", "verify_identity"),
    ])))
    second = await adapter.send(NEXT_USER, None)
    assert [(c.success, c.error, c.result) for c in second.tool_calls] == [(None, None, None)]
    await adapter.close()


# =============================================================================================
# Runner: run.json and the AGENT TOOL CALL lines of the parallel case (R3P-1)
# =============================================================================================


@pytest.mark.asyncio
async def test_runner_parallel_calls_successful_call_is_stored_as_successful(
    impulse_project, llm_router, mock_pipecat_channel, tmp_path
):
    """The R3P-1 probe order through ``run()`` (eval mode): call-1 raised, call-2 succeeded with
    ``{"ok": true}``. run.json: call-1 ``success: false`` with the RTVI error, call-2 ``success: true``
    (inferred by the runner) with no error; the AGENT TOOL CALL lines show the same."""
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
    items = _parallel_items(mock, "ok")
    order = ("ip1", "ip2", "err", "stop1", "stop2")
    channel.on_send(sent_text(USER_TURN_3), _plan(channel, _turn_steps(mock, _spaced([items[label] for label in order]))))
    result = await run(cfg, adapter_factory=lambda config: PipecatAdapter(config, channel_factory=mock.factory, clock=mock.clock))

    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    path = tmp_path / "run.json"
    write_json(result, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    stored = raw["scenario_results"][0]["transcript"]["turns"][3]["tool_calls"]
    assert [(call["arguments"], call["success"], call["error"]) for call in stored] == [
        ({}, False, RTVI_ERROR),
        (VERIFY_ARGS, True, None),
    ]
    loaded = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    lines = [line for line in format_transcript(loaded.scenario_results[0].transcript).splitlines()
             if "AGENT TOOL CALL" in line]
    assert len(lines) == 2
    assert lines[0].startswith(f"[turn 4] AGENT TOOL CALL: verify_identity({{}}) -> success=False error={RTVI_ERROR}"), lines[0]
    assert "success=True" in lines[1] and "error=" not in lines[1], lines[1]
