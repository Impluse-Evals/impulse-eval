# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 11: wire round trips through Pipecat's real serializers (SPEC.md sections 9 and 18.2).

Skipped unless ``pipecat-ai`` is installed (it is an optional extra, not in the dev environment).
No socket is opened: frames and messages go through the serializer objects only.

* websocket mode: an RTVI message (the ``send-text`` and ``client-ready`` shapes Impulse sends) and a
  user audio frame round-trip through ``ProtobufFrameSerializer``.
* eval mode: ``EvalClientSerializer`` decodes RTVI messages it maps (``bot-llm-text`` ->
  ``LLMTextFrame``, ``user-transcription`` -> raw message frame), returns None for the messages
  Impulse must read before the serializer (``bot-ready``, ``bot-output``, ``metrics``,
  ``server-response``), and decodes ``eval-bot-audio`` at ``sampleRate: 16000`` unchanged.
Message and class names: DECISIONS.md "Source read (KICKOFF §2b)" (pipecat-ai 1.12.0).
"""

import base64
import inspect
import json

import pytest

pytest.importorskip("pipecat", reason="pipecat-ai is not installed (optional extra 'pipecat')")

from conftest import pcm_tone

pytestmark = pytest.mark.phase11


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _setup(serializer, sample_rate=16000):
    setup = getattr(serializer, "setup", None)
    if setup is None:
        return
    from pipecat.frames.frames import StartFrame

    try:
        frame = StartFrame(audio_in_sample_rate=sample_rate, audio_out_sample_rate=sample_rate)
    except TypeError:  # pragma: no cover - older StartFrame signature
        frame = StartFrame()
    await _maybe_await(setup(frame))


# =============================================================================================
# websocket mode: ProtobufFrameSerializer
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": "1",
            "data": {"version": "2.1.0", "about": {"library": "impulse-evals", "library_version": "0.1.0"}},
        },
        {
            "label": "rtvi-ai",
            "type": "send-text",
            "id": "2",
            "data": {"content": "Book a slot for Tuesday", "options": {"run_immediately": True, "audio_response": False}},
        },
    ],
    ids=["client-ready", "send-text"],
)
async def test_websocket_rtvi_message_round_trips_through_protobuf_serializer(message):
    from pipecat.frames.frames import (
        InputTransportMessageFrame,
        OutputTransportMessageUrgentFrame,
    )
    from pipecat.serializers.protobuf import ProtobufFrameSerializer

    serializer = ProtobufFrameSerializer()
    await _setup(serializer)
    wire = await _maybe_await(serializer.serialize(OutputTransportMessageUrgentFrame(message=message)))
    assert isinstance(wire, (bytes, bytearray)) and wire
    frame = await _maybe_await(serializer.deserialize(wire))
    assert isinstance(frame, InputTransportMessageFrame)
    assert frame.message == message


@pytest.mark.asyncio
async def test_websocket_audio_frame_round_trips_through_protobuf_serializer():
    from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
    from pipecat.serializers.protobuf import ProtobufFrameSerializer

    pcm = pcm_tone(0.02, 16000)  # one 20 ms frame
    serializer = ProtobufFrameSerializer()
    await _setup(serializer)
    wire = await _maybe_await(
        serializer.serialize(OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1))
    )
    assert isinstance(wire, (bytes, bytearray)) and wire
    frame = await _maybe_await(serializer.deserialize(wire))
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.audio == pcm
    assert frame.sample_rate == 16000
    assert frame.num_channels == 1


# =============================================================================================
# eval mode: EvalClientSerializer
# =============================================================================================


async def _eval_decode(message):
    from pipecat.evals.serializer import EvalClientSerializer

    serializer = EvalClientSerializer()
    await _setup(serializer)
    return await _maybe_await(serializer.deserialize(json.dumps(message)))


@pytest.mark.asyncio
async def test_eval_bot_llm_text_decodes_to_a_text_frame():
    frame = await _eval_decode({"label": "rtvi-ai", "type": "bot-llm-text", "data": {"text": "Hello there"}})
    assert frame is not None
    assert getattr(frame, "text", None) == "Hello there"


@pytest.mark.asyncio
async def test_eval_user_transcription_decodes_to_the_raw_message():
    message = {
        "label": "rtvi-ai",
        "type": "user-transcription",
        "data": {"text": "book a slot", "user_id": "", "timestamp": "2026-10-19T10:00:00.000Z", "final": True},
    }
    frame = await _eval_decode(message)
    assert frame is not None
    assert getattr(frame, "message", None) == message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        {"label": "rtvi-ai", "type": "bot-ready", "id": "client-ready",
         "data": {"version": "2.1.0", "about": {"library": "pipecat-ai"}}},
        {"label": "rtvi-ai", "type": "bot-output", "data": {"text": "Hello.", "aggregated_by": "sentence", "spoken": True}},
        {"label": "rtvi-ai", "type": "metrics", "data": {"ttfb": [{"processor": "LLM#0", "value": 0.4}]}},
        {"label": "rtvi-ai", "type": "server-response", "id": "7",
         "data": {"t": "get_system_prompt", "d": {"system_prompt": "You are a bot."}}},
    ],
    ids=["bot-ready", "bot-output", "metrics", "server-response"],
)
async def test_eval_serializer_drops_messages_impulse_reads_before_it(message):
    assert await _eval_decode(message) is None


@pytest.mark.asyncio
async def test_eval_bot_audio_at_16_khz_decodes_unchanged():
    from pipecat.frames.frames import InputAudioRawFrame

    pcm = pcm_tone(0.04, 16000)  # one 40 ms bot output chunk
    message = {
        "label": "rtvi-ai",
        "type": "eval-bot-audio",
        "data": {"audio": base64.b64encode(pcm).decode("ascii"), "sampleRate": 16000},
    }
    frame = await _eval_decode(message)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.audio == pcm
    assert frame.sample_rate == 16000
    assert frame.num_channels == 1
