# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 2, A5 (DEFECT-5): Pipecat text-mode duplicate ``bot-output`` (FIXES-ROUND-2-SPEC.md).

In a ``-t eval`` text run, the LLM run after a tool call is spoken, and a Deepgram-style TTS
(``push_text_frames=True``) reports each sentence twice with DIFFERENT ``segment_id``s: an
``AggregatedTextFrame`` before synthesis (``spoken: false``, ``will_be_spoken: true``,
``spoken_status: "new"``) and a ``TTSTextFrame`` after it (``spoken: true``, ``spoken_status:
"completed"``), in the order A(new) A(completed) B(new) B(completed). Campaign run
20261023T195937Z-746285 (BankBot ``balance_with_identity``) stored every sentence twice.

Rule (text mode, ``_on_bot_output``), as revised by A5-rev (review R3; supersedes the A5 "keep only
completed" rule): an announced ``bot-output`` (``will_be_spoken`` true, not ``spoken``,
``spoken_status`` not ``"completed"``) is DROPPED only when a completed ``bot-output`` (``spoken`` true
or ``spoken_status == "completed"``) with the same text (whitespace-trimmed) arrives in the same turn;
an announcement still without its completion when the turn ends is KEPT in the reply. The adapter
remembers (per connection) the texts of announcements it kept that way; a completed ``bot-output``
arriving later (in a later turn, also as pre-turn output) whose text matches one of them is not added
again and the text is forgotten. Word-timestamp progress messages (``in-progress``, same
``segment_id``) keep the latest-text-per-segment rule; any other ``bot-output`` is kept as before.
Voice mode is unchanged.

Re-check revision N-M2 (orchestrator decision): the per-connection memory of kept announcements also
remembers their ``segment_id``s; any later ``bot-output`` (announcement, ``in-progress`` or
``completed``) for a remembered segment is not added again (checked before the open-announcement
match), and the segment is forgotten after its completion; the memory is cleared when
``bot-interrupted`` (RTVI ``BotInterruptedMessage``, also sent when a ``send-text`` with
``run_immediately`` interrupts the bot) arrives. Tests: word-timestamp TTS whose late ``in-progress``
and ``completed`` messages for the kept segment arrive before the next ``send-text``, after it, or one
on each side.

Superseded (A5-rev): ``test_sentence_announced_but_never_completed_is_dropped`` expected the
announced-only sentence to be dropped; it is now ``..._is_kept`` and expects it in the reply.

Payloads are shaped as pipecat-ai 1.12.0 sends them (``RTVIObserver._send_aggregated_llm_text``:
``text``, ``spoken``, ``will_be_spoken``, ``aggregated_by``, integer ``segment_id`` = the frame id,
``spoken_status``, ``spoken_progress``; ``None`` fields are left out by ``exclude_none=True``).
Every test runs on ``mock_pipecat_channel`` (no network, no pipecat-ai).
"""

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

# The BankBot balance_with_identity reply of campaign run 20261023T195937Z-746285 (one sentence each).
SENTENCE_A = "Your identity has been verified, Ananya."
SENTENCE_B = "Your current account balance is ₹45,210.75."
SENTENCE_C = "Is there anything else I can help you with today?"
VERIFY_ARGS = {"customer_id": "CUST-2001", "dob": "1990-04-12"}
VERIFY_RESULT = {"success": True, "ok": True, "customer_id": "CUST-2001", "name": "Ananya Rao"}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _cfg(project, mode, *, voice=False):
    kwargs = {"voice_config": dict(VOICE_CONFIG)} if voice else {}
    project.write_config(
        [project.scenario("balance_with_identity")],
        agent={"adapter": "pipecat", "pipecat": {"mode": mode, "url": URL}},
        **kwargs,
    )
    return project.load()


def _adapter(cfg, mock):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    return PipecatAdapter(cfg, channel_factory=mock.factory, clock=mock.clock)


async def _started(cfg, mock, mode):
    channel = mock.channel
    if mode == "eval":
        channel.paced_audio = True
        channel.on_connect(mock.rtvi("bot-ready", BOT_READY, id="client-ready"))
    else:
        channel.on_send("client-ready", lambda message: mock.rtvi("bot-ready", BOT_READY, id=message.get("id")))
    adapter = _adapter(cfg, mock)
    await adapter.start(cfg.scenarios[0])
    return adapter


def _plan(channel, steps):
    def trigger(_record):
        for offset, item in steps:
            channel.emit(item, after=offset)

    return trigger


def _announced(text, segment_id):
    """pipecat-ai 1.12.0: the ``AggregatedTextFrame`` before synthesis (word-timestamp / push-text TTS)."""
    return {
        "text": text,
        "spoken": False,
        "will_be_spoken": True,
        "aggregated_by": "sentence",
        "segment_id": segment_id,
        "spoken_status": "new",
        "spoken_progress": {"accumulated_text": "", "remaining_text": text},
    }


def _completed(text, segment_id):
    """pipecat-ai 1.12.0: the ``TTSTextFrame`` after synthesis (``push_text_frames=True``)."""
    return {
        "text": text,
        "spoken": True,
        "will_be_spoken": True,
        "aggregated_by": "sentence",
        "segment_id": segment_id,
        "spoken_status": "completed",
        "spoken_progress": {"accumulated_text": text, "remaining_text": ""},
    }


def _progress(text, segment_id, *, done):
    """pipecat-ai 1.12.0 ``_send_tts_progress`` (word-timestamp TTS): no ``spoken`` field."""
    return {
        "text": text,
        "will_be_spoken": True,
        "aggregated_by": "sentence",
        "segment_id": segment_id,
        "spoken_status": "completed" if done else "in-progress",
        "spoken_progress": {
            "accumulated_text": text if done else text.split(" ")[0],
            "remaining_text": "" if done else text.split(" ", 1)[1],
        },
    }


def _unspoken(text, segment_id):
    """pipecat-ai 1.12.0 for text that will not be spoken: ``will_be_spoken: false``, no status."""
    return {"text": text, "spoken": False, "will_be_spoken": False, "aggregated_by": "sentence", "segment_id": segment_id}


def _tool_turn_steps(mock, outputs):
    """The campaign shape: LLM run with a function call, the call's result, then the LLM run after the
    tool call (spoken, so its sentences come as ``bot-output``), all inside the 800 ms grace window."""
    rtvi = mock.rtvi
    steps = [
        (0.2, rtvi("bot-llm-started")),
        (0.3, rtvi("llm-function-call-in-progress",
                   {"tool_call_id": "call-1", "function_name": "verify_identity", "arguments": VERIFY_ARGS})),
        (0.35, rtvi("bot-llm-stopped")),
        (0.6, rtvi("llm-function-call-stopped",
                   {"tool_call_id": "call-1", "function_name": "verify_identity", "cancelled": False,
                    "result": VERIFY_RESULT})),
        (0.7, rtvi("bot-llm-started")),
        (0.75, rtvi("bot-llm-text", {"text": "tokens that are not the reply"})),
    ]
    t = 0.8
    for data in outputs:
        steps.append((t, rtvi("bot-output", data)))
        t += 0.05
    steps.append((t + 0.05, rtvi("bot-llm-stopped")))
    return steps


async def _text_reply(project, mock, mode, outputs):
    cfg = _cfg(project, mode)
    adapter = await _started(cfg, mock, mode)
    mock.channel.on_send("send-text", _plan(mock.channel, _tool_turn_steps(mock, outputs)))
    reply = await adapter.send("My customer ID is CUST-2001 and my date of birth is 12 April 1990.", None)
    await adapter.close()
    return reply


def _norm(text):
    return " ".join((text or "").split())


# =============================================================================================
# Text mode: announced sentences are not counted again
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_new_then_completed_sentences_appear_once_each_in_order(impulse_project, mock_pipecat_channel, mode):
    outputs = [
        _announced(SENTENCE_A, 101), _completed(SENTENCE_A, 102),
        _announced(SENTENCE_B, 103), _completed(SENTENCE_B, 104),
    ]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_B}"
    assert [call.name for call in reply.tool_calls] == ["verify_identity"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_campaign_three_sentence_reply_is_not_doubled(impulse_project, mock_pipecat_channel, mode):
    outputs = [
        _announced(SENTENCE_A, 11), _completed(SENTENCE_A, 12),
        _announced(SENTENCE_B, 13), _completed(SENTENCE_B, 14),
        _announced(SENTENCE_C, 15), _completed(SENTENCE_C, 16),
    ]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_B} {SENTENCE_C}"
    for sentence in (SENTENCE_A, SENTENCE_B, SENTENCE_C):
        assert reply.text.count(sentence) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_sentence_announced_but_never_completed_is_kept(impulse_project, mock_pipecat_channel, mode):
    # A5-rev (supersedes A5's "dropped"): B is announced (new) but not completed when the turn ends.
    outputs = [_announced(SENTENCE_A, 21), _completed(SENTENCE_A, 22), _announced(SENTENCE_B, 23)]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_B}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_sentence_really_said_twice_is_kept_twice_with_announcements(
    impulse_project, mock_pipecat_channel, mode
):
    outputs = [
        _announced(SENTENCE_A, 31), _completed(SENTENCE_A, 32),
        _announced(SENTENCE_A, 33), _completed(SENTENCE_A, 34),
    ]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_A}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_sentence_really_said_twice_is_kept_twice(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix): two completed messages, same text, different segments.
    outputs = [_completed(SENTENCE_A, 41), _completed(SENTENCE_A, 42)]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_A}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_word_timestamp_progress_gives_each_sentence_once(impulse_project, mock_pipecat_channel, mode):
    # Regression guard (passes before the fix): word-timestamp TTS reports progress on the SAME segment.
    outputs = [
        _announced(SENTENCE_A, 51), _progress(SENTENCE_A, 51, done=False), _progress(SENTENCE_A, 51, done=True),
        _announced(SENTENCE_B, 52), _progress(SENTENCE_B, 52, done=False), _progress(SENTENCE_B, 52, done=True),
    ]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_B}"


# =============================================================================================
# A5-rev: an announcement still open at the end of the turn is kept; its late completion (word-
# timestamp TTS releases it on the playback clock) is not repeated in the next turn
# =============================================================================================

PREAMBLE = "Let me look that up for you."
NEXT_REPLY = "Anything else?"


def _late_completion_turn_steps(mock, completion):
    """Review R3 probe p08: a preamble from the first LLM run (skip_tts honoured: will_be_spoken false),
    a tool call, then the post-tool run whose sentence is announced at 0.8 s; the text turn ends at
    LLM stop (0.7 s) + 800 ms grace = 1.5 s; the completion arrives at 2.6 s."""
    rtvi = mock.rtvi
    return [
        (0.1, rtvi("bot-llm-started")),
        (0.15, rtvi("bot-llm-text", {"text": PREAMBLE})),
        (0.2, rtvi("bot-output", _unspoken(PREAMBLE, 1))),
        (0.25, rtvi("llm-function-call-in-progress",
                    {"tool_call_id": "call-1", "function_name": "verify_identity", "arguments": VERIFY_ARGS})),
        (0.3, rtvi("bot-llm-stopped")),
        (0.5, rtvi("llm-function-call-stopped",
                   {"tool_call_id": "call-1", "function_name": "verify_identity", "cancelled": False,
                    "result": VERIFY_RESULT})),
        (0.6, rtvi("bot-llm-started")),
        (0.65, rtvi("bot-llm-text", {"text": SENTENCE_B})),
        (0.7, rtvi("bot-llm-stopped")),
        (0.8, rtvi("bot-started-speaking")),
        (0.8, rtvi("bot-output", _announced(SENTENCE_B, 2))),
        (1.2, rtvi("bot-output", _progress(SENTENCE_B, 2, done=False))),   # word-timestamp progress, same segment
        (2.6, rtvi("bot-output", completion)),
        (2.7, rtvi("bot-stopped-speaking")),
    ]


def _plain_turn_steps(mock, text, segment_id, *, llm_stop_at, completed=False):
    rtvi = mock.rtvi
    output = _completed(text, segment_id) if completed else _unspoken(text, segment_id)
    return [
        (0.1, rtvi("bot-llm-started")),
        (0.15, rtvi("bot-llm-text", {"text": text})),
        (0.2, rtvi("bot-output", output)),
        (llm_stop_at, rtvi("bot-llm-stopped")),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "completion_style",
    [
        pytest.param("push-text", id="push-text-completion"),         # TTSTextFrame: new segment, spoken true
        pytest.param("word-timestamp", id="word-timestamp-completion"),   # progress frame: same segment
    ],
)
@pytest.mark.parametrize(
    "arrival",
    [
        pytest.param("pre-turn", id="arrives-between-turns"),   # read as pre-turn output of turn 2
        pytest.param("in-turn", id="arrives-during-turn-2"),
    ],
)
async def test_announcement_open_at_turn_end_is_kept_and_its_late_completion_not_repeated(
    impulse_project, mock_pipecat_channel, mode, completion_style, arrival
):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    completion = _completed(SENTENCE_B, 4) if completion_style == "push-text" else _progress(SENTENCE_B, 2, done=True)
    plans = [
        _late_completion_turn_steps(mock, completion),
        # Turn 2 stays open until 1.5 s + grace, so an in-turn arrival (about 1.1 s in) falls inside it.
        _plain_turn_steps(mock, NEXT_REPLY, 3, llm_stop_at=1.5),
        # Turn 3: the bot really says SENTENCE_B again (completed): kept, since the remembered text was
        # forgotten once its late completion was absorbed.
        _plain_turn_steps(mock, SENTENCE_B, 5, llm_stop_at=0.3, completed=True),
    ]
    sends = []

    def on_text(record):
        sends.append(record)
        _plan(channel, plans[len(sends) - 1])(record)

    channel.on_send("send-text", on_text, once=False)

    first = await adapter.send("My customer ID is CUST-2001 and my date of birth is 12 April 1990.", None)
    assert _norm(first.text) == f"{PREAMBLE} {SENTENCE_B}", "the announced post-tool sentence stays in its turn"
    assert [call.name for call in first.tool_calls] == ["verify_identity"]

    if arrival == "pre-turn":
        await mock.clock.sleep(3.0)   # the completion (2.6 s) arrives before the next send
    second = await adapter.send("Thanks. What is my balance?", None)
    assert _norm(second.text) == NEXT_REPLY, "the late completion is not added to the next turn"

    third = await adapter.send("Can you repeat that?", None)
    assert _norm(third.text) == SENTENCE_B
    await adapter.close()


# =============================================================================================
# Re-check N-M2: word-timestamp TTS keeps sending progress for the kept segment after the turn ended;
# none of it is added to a later turn. bot-interrupted clears the memory.
# =============================================================================================

KEPT_SEGMENT = 2


def _word_timestamp_turn_steps(mock, *, late_progress_at, completion_at):
    """Review re-check probe test_a1_wordts_late_progress.py: the post-tool sentence is announced at
    0.8 s (segment 2) with one progress message inside the turn; the text turn ends at LLM stop (0.7 s)
    + 800 ms grace = 1.5 s; an ``in-progress`` message (``late_progress_at``) and the ``completed`` one
    (``completion_at``, None: never, the bot is interrupted) for the same segment follow on the playback
    clock, after the turn ended."""
    rtvi = mock.rtvi
    steps = [
        (0.1, rtvi("bot-llm-started")),
        (0.15, rtvi("bot-llm-text", {"text": PREAMBLE})),
        (0.2, rtvi("bot-output", _unspoken(PREAMBLE, 1))),
        (0.25, rtvi("llm-function-call-in-progress",
                    {"tool_call_id": "call-1", "function_name": "verify_identity", "arguments": VERIFY_ARGS})),
        (0.3, rtvi("bot-llm-stopped")),
        (0.5, rtvi("llm-function-call-stopped",
                   {"tool_call_id": "call-1", "function_name": "verify_identity", "cancelled": False,
                    "result": VERIFY_RESULT})),
        (0.6, rtvi("bot-llm-started")),
        (0.65, rtvi("bot-llm-text", {"text": SENTENCE_B})),
        (0.7, rtvi("bot-llm-stopped")),
        (0.8, rtvi("bot-started-speaking")),
        (0.8, rtvi("bot-output", _announced(SENTENCE_B, KEPT_SEGMENT))),
        (1.2, rtvi("bot-output", _progress(SENTENCE_B, KEPT_SEGMENT, done=False))),
        (late_progress_at, rtvi("bot-output", _progress(SENTENCE_B, KEPT_SEGMENT, done=False))),   # after the turn
    ]
    if completion_at is not None:
        steps += [
            (completion_at, rtvi("bot-output", _progress(SENTENCE_B, KEPT_SEGMENT, done=True))),
            (completion_at + 0.1, rtvi("bot-stopped-speaking")),
        ]
    return steps


def _word_timestamp_repeat_steps(mock, segment_id):
    """The bot really says SENTENCE_B again, word-timestamp style, under a NEW segment, then SENTENCE_C
    (a second sentence, so the reply comes from the bot-output texts: with no bot-output left the reply
    would fall back to the LLM text and hide a dropped sentence)."""
    rtvi = mock.rtvi
    steps = [
        (0.1, rtvi("bot-llm-started")),
        (0.15, rtvi("bot-llm-text", {"text": f"{SENTENCE_B} {SENTENCE_C}"})),
        (0.2, rtvi("bot-llm-stopped")),
        (0.25, rtvi("bot-started-speaking")),
    ]
    t = 0.25
    for offset, text in enumerate((SENTENCE_B, SENTENCE_C)):
        segment = segment_id + offset
        steps += [
            (t, rtvi("bot-output", _announced(text, segment))),
            (t + 0.05, rtvi("bot-output", _progress(text, segment, done=False))),
            (t + 0.1, rtvi("bot-output", _progress(text, segment, done=True))),
        ]
        t += 0.15
    steps.append((t, rtvi("bot-stopped-speaking")))
    return steps


def _completed_repeat_steps(mock, segment_id):
    """The bot really says SENTENCE_B again, push-text style (one completed message, a NEW segment),
    then SENTENCE_C (so the reply comes from the bot-output texts, not the LLM-text fallback)."""
    rtvi = mock.rtvi
    return [
        (0.1, rtvi("bot-llm-started")),
        (0.15, rtvi("bot-llm-text", {"text": f"{SENTENCE_B} {SENTENCE_C}"})),
        (0.2, rtvi("bot-output", _completed(SENTENCE_B, segment_id))),
        (0.25, rtvi("bot-output", _completed(SENTENCE_C, segment_id + 1))),
        (0.3, rtvi("bot-llm-stopped")),
    ]


def _on_each_send(channel, plans):
    sends = []

    def on_text(record):
        sends.append(record)
        _plan(channel, plans[len(sends) - 1])(record)

    channel.on_send("send-text", on_text, once=False)
    return sends


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "arrival, wait_s",
    [
        # turn 1 ends at about 1.5 s; the late in-progress is at 1.9 s, the completion at 2.6 s
        pytest.param("before-next-send", 3.0, id="both-before-next-send"),   # read as pre-turn output of turn 2
        pytest.param("split", 0.6, id="progress-before-completion-after"),
        pytest.param("after-next-send", 0.0, id="both-during-turn-2"),
    ],
)
async def test_late_word_timestamp_progress_of_a_kept_segment_is_not_repeated(
    impulse_project, mock_pipecat_channel, mode, arrival, wait_s
):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    _on_each_send(channel, [
        _word_timestamp_turn_steps(mock, late_progress_at=1.9, completion_at=2.6),
        # Turn 2 stays open until about 1.5 s + grace after its send, so the late messages fall inside it
        # when they arrive after the send.
        _plain_turn_steps(mock, NEXT_REPLY, 3, llm_stop_at=1.5),
        # Turn 3: a real repeat under a new segment is kept (the kept segment was forgotten after its
        # completion).
        _word_timestamp_repeat_steps(mock, 6),
    ])

    first = await adapter.send("My customer ID is CUST-2001 and my date of birth is 12 April 1990.", None)
    assert _norm(first.text) == f"{PREAMBLE} {SENTENCE_B}", "the announced post-tool sentence stays in its turn"
    if wait_s:
        await mock.clock.sleep(wait_s)
    second = await adapter.send("Thanks. What is my balance?", None)
    assert _norm(second.text) == NEXT_REPLY, "late progress of the kept segment is not added to the next turn"
    third = await adapter.send("Can you repeat that?", None)
    assert _norm(third.text) == f"{SENTENCE_B} {SENTENCE_C}", "a real repeat under a new segment is kept"
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "late_progress",
    [
        pytest.param(True, id="late-in-progress-then-interrupted"),   # the probe's interrupted_by_next_send
        pytest.param(False, id="interrupted-without-late-progress"),
    ],
)
async def test_bot_interrupted_clears_the_kept_announcement_memory(
    impulse_project, mock_pipecat_channel, mode, late_progress
):
    """The next send-text interrupts the bot (RTVI ``send-text`` with ``run_immediately``): the kept
    sentence's completion never comes, pipecat-ai sends ``bot-interrupted``, and the memory is cleared,
    so a later real repeat of the same sentence (one completed message, a new segment) is kept."""
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode)
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    first_steps = _word_timestamp_turn_steps(mock, late_progress_at=1.9, completion_at=None)
    if not late_progress:
        first_steps = first_steps[:-1]
    second_steps = [(0.05, rtvi("bot-interrupted")), (0.06, rtvi("bot-stopped-speaking"))]
    second_steps += _plain_turn_steps(mock, NEXT_REPLY, 3, llm_stop_at=0.3)
    _on_each_send(channel, [first_steps, second_steps, _completed_repeat_steps(mock, 5)])

    first = await adapter.send("My customer ID is CUST-2001 and my date of birth is 12 April 1990.", None)
    assert _norm(first.text) == f"{PREAMBLE} {SENTENCE_B}"
    await mock.clock.sleep(0.6)   # the late in-progress (1.9 s) arrives before the next send
    second = await adapter.send("Thanks. What is my balance?", None)
    assert _norm(second.text) == NEXT_REPLY
    third = await adapter.send("Can you repeat that?", None)
    assert _norm(third.text) == f"{SENTENCE_B} {SENTENCE_C}", "after bot-interrupted nothing is remembered"
    await adapter.close()


# =============================================================================================
# Text mode: bot-output without will_be_spoken is kept as before (regression guards)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_bot_output_without_will_be_spoken_is_kept(impulse_project, mock_pipecat_channel, mode):
    # An older bot (protocol 1.x fields only): no will_be_spoken, no spoken_status.
    outputs = [
        {"text": SENTENCE_A, "aggregated_by": "sentence", "segment_id": 61},
        {"text": SENTENCE_B, "aggregated_by": "sentence", "segment_id": 62, "spoken": False},
    ]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_B}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_bot_output_that_will_not_be_spoken_is_kept(impulse_project, mock_pipecat_channel, mode):
    # skip_tts honoured (the normal text turn): pipecat-ai 1.12.0 sends will_be_spoken false.
    outputs = [_unspoken(SENTENCE_A, 71), _unspoken(SENTENCE_B, 72)]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_B}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_latest_text_per_segment_is_kept_for_messages_without_will_be_spoken(
    impulse_project, mock_pipecat_channel, mode
):
    outputs = [
        {"text": "Your identity", "aggregated_by": "sentence", "segment_id": 81},
        {"text": SENTENCE_A, "aggregated_by": "sentence", "segment_id": 81},
    ]
    reply = await _text_reply(impulse_project, mock_pipecat_channel, mode, outputs)
    assert _norm(reply.text) == SENTENCE_A


# =============================================================================================
# Voice mode is unchanged (regression guard)
# =============================================================================================


def _after_speech(channel, samples):
    def predicate(record):
        total = sum(len(entry["pcm"]) // 2 for entry in channel.sent_audio)
        return total >= samples * 0.98

    return predicate


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_voice_mode_counts_spoken_segments_once(impulse_project, mock_pipecat_channel, mode):
    mock = mock_pipecat_channel
    rtvi = mock.rtvi
    cfg = _cfg(impulse_project, mode, voice=True)
    adapter = await _started(cfg, mock, mode)
    channel = mock.channel
    steps = [
        (0.1, rtvi("bot-started-speaking")),
        (0.15, rtvi("bot-output", _announced(SENTENCE_A, 91))),
        (0.2, mock.BotAudio(mock.tone(0.5))),
        (0.3, rtvi("bot-output", _completed(SENTENCE_A, 92))),
        (0.35, rtvi("bot-output", _announced(SENTENCE_B, 93))),
        (0.5, rtvi("bot-output", _completed(SENTENCE_B, 94))),
        (0.6, rtvi("bot-output", _announced(SENTENCE_C, 95))),   # interrupted: never spoken
        (0.9, rtvi("bot-stopped-speaking")),
    ]
    channel.on_audio(_plan(channel, steps), match=_after_speech(channel, int(0.5 * cfg.agent.pipecat.sample_rate)))
    reply = await adapter.send("What is my balance?", wav_bytes(pcm_tone(0.5, 16000)))
    assert _norm(reply.text) == f"{SENTENCE_A} {SENTENCE_B}"
    await adapter.close()
