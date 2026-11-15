# SPDX-License-Identifier: Apache-2.0
"""Phase 11 (whole file): voice metrics, ``metrics/voice.py`` (METRICS.md section 2; SPEC.md
sections 7, 13, 14).

Deterministic metrics: ``raw_value``, label against the thresholds (``MetricContext.voice_thresholds``,
the ``metrics.voice`` block; missing keys take the SPEC.md section 8.2 defaults), score good = 1.0,
warning = 0.5, poor = 0.0. Text-mode proxies (VOICE-001/003/005) carry ``is_proxy=True``; VOICE-029
runs identically in both modes. The outbound opener (an agent turn with no user turn before it) is
excluded from VOICE-001/003/005. Percentiles use ``numpy.percentile`` (linear). Skipped results have
``skipped=True``, ``label="skipped"``, ``score=0.0``.

Hand-built transcripts keep both readings of the voice-mode timing consistent: an agent turn's
``latency_ms`` equals its ``timestamp_ms`` minus the preceding user turn's ``end_ms``.
VOICE-015 needs STT word timings, which only the runner has, so it is checked end to end
(``runner.run`` in voice mode with ``mock_voice_server`` and a fake adapter returning audio).
Judge-based voice metrics: registry attributes, ``Return skip if`` lines, VOICE-035's turn mapping.
"""

import pytest
from conftest import pcm_tone, wav_bytes

pytestmark = pytest.mark.phase11

SYSTEM_PROMPT = "You are the booking assistant of Sunrise Clinic. Book appointments with the book_slot tool."

DETERMINISTIC = {
    "VOICE-001": "Time to first audio (TTFA)",
    "VOICE-003": "Dead air",
    "VOICE-005": "Tail latency",
    "VOICE-009": "Overall WER (agent hearing)",
    "VOICE-011": "Harness round-trip WER (sanity)",
    "VOICE-013": "Domain-term error rate",
    "VOICE-015": "Speech rate",
    "VOICE-029": "Spoken-length fit",
}

# Judge-based voice metrics (METRICS.md section 2.2): (name, tier, criterion or None, skip_when or None)
# criterion None: not pinned (VOICE-035's cell describes its special mapping); skip_when None:
# decided in code, only checked non-empty.
JUDGE_VOICE = {
    "VOICE-007": (
        "Interruption handling",
        "fast",
        "For turns where the user interrupted the agent, did the agent stop, acknowledge, and address what "
        "the user said, rather than talking over or ignoring it?",
        None,
    ),
    "VOICE-027": (
        "Clarifying questions",
        "primary",
        "When the user's request was ambiguous or missing needed details, did the agent ask a relevant, "
        "concise clarifying question instead of guessing or stalling? Penalize unnecessary questions when "
        "the answer was clear.",
        "request was never ambiguous",
    ),
    "VOICE-031": (
        "Code-switch handling",
        "fast",
        "When the user mixed languages (for example Hindi and English), did the agent understand and reply "
        "naturally for that mix?",
        None,
    ),
    "VOICE-035": ("Turns to close", "fast", None, "never"),
    "VOICE-037": (
        "Voice naturalness",
        "fast",
        "Would the agent's replies sound natural when SPOKEN aloud? Penalize markdown formatting, bullet "
        "lists, URLs or long numbers read out verbatim, stiff or robotic phrasing, and run-on sentences. A "
        "pass means the output is suitable for spoken delivery without editing.",
        "never",
    ),
    "VOICE-039": (
        "Responsiveness",
        "fast",
        "Does each agent turn directly respond to the user's immediately preceding utterance, rather than "
        "ignoring it, repeating itself from a previous turn, or answering an earlier question? Penalize "
        "non-sequiturs, scripted pivots that ignore what the user just said, and copy-pasted "
        "acknowledgements.",
        "never",
    ),
}

LABEL_SCORE = {"good": 1.0, "warning": 0.5, "poor": 0.0}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _m():
    from impulse_evals import models

    return models


def _metric(metric_id):
    from impulse_evals.metrics.base import get_metric

    return get_metric(metric_id)


def _ctx(**kwargs):
    from impulse_evals.metrics.base import MetricContext

    return MetricContext(**kwargs)


def _turn(speaker, text, timestamp_ms, **kwargs):
    return _m().Turn(speaker=speaker, text=text, timestamp_ms=timestamp_ms, **kwargs)


def _transcript(turns, voice_mode=False, scenario_name="voice_metric_probe"):
    return _m().CallTranscript(
        turns=turns,
        duration_ms=(turns[-1].end_ms or turns[-1].timestamp_ms) + 1000 if turns else 0,
        metadata={},
        scenario_name=scenario_name,
        language="en-IN",
        voice_mode=voice_mode,
        agent_system_prompt=SYSTEM_PROMPT,
    )


def _scenario(**kwargs):
    values = dict(
        name="voice_metric_probe",
        language="en-IN",
        domain="healthcare",
        max_turns=12,
        persona="patient who wants a Tuesday slot",
    )
    values.update(kwargs)
    return _m().Scenario(**values)


def _latency_turns(latencies, opener_latency=None):
    """Inbound turns (outbound with ``opener_latency``): each agent turn starts ``latency`` ms after
    the end of the preceding user turn and carries ``latency_ms=latency``."""
    turns, t = [], 0
    if opener_latency is not None:
        turns.append(_turn("agent", "Hello, this is Sunrise Clinic.", 0, end_ms=1500, latency_ms=float(opener_latency)))
        t = 2000
    for index, latency in enumerate(latencies, start=1):
        user_end = t + 1000
        turns.append(_turn("user", f"User question number {index}", t, end_ms=user_end))
        agent_start = user_end + int(round(latency))
        turns.append(
            _turn("agent", f"Agent answer number {index}", agent_start, end_ms=agent_start + 1500, latency_ms=float(latency))
        )
        t = agent_start + 2000
    return turns


def _word_turns(word_counts):
    turns, t = [], 0
    for index, count in enumerate(word_counts, start=1):
        turns.append(_turn("user", f"User question number {index}", t, end_ms=t + 1000))
        turns.append(_turn("agent", " ".join(["word"] * count), t + 1500, end_ms=t + 3000, latency_ms=500.0))
        t += 4000
    return turns


async def _evaluate(metric_id, transcript, scenario=None, ctx=None, judge=None):
    from impulse_evals.judge.base import Judge

    class NoJudge(Judge):
        async def evaluate(self, transcript, criterion, tier):  # pragma: no cover - must not be called
            raise AssertionError(f"{metric_id} is deterministic and must not call the judge")

    return await _metric(metric_id).evaluate(
        transcript, scenario or _scenario(), judge or NoJudge(), ctx or _ctx()
    )


def _assert_label(result, label):
    assert result.skipped is False
    assert result.label == label
    assert result.score == LABEL_SCORE[label]


def _assert_skipped(result):
    assert result.skipped is True
    assert result.label == "skipped"
    assert result.score == 0.0


# =============================================================================================
# Registry
# =============================================================================================


@pytest.mark.parametrize("metric_id", sorted(DETERMINISTIC))
def test_deterministic_voice_metric_registry(metric_id):
    metric = _metric(metric_id)
    assert metric.metric_id == metric_id
    assert metric.name == DETERMINISTIC[metric_id]
    assert metric.category == "voice"
    assert metric.tier is None
    assert metric.criterion is None


@pytest.mark.parametrize("metric_id", sorted(JUDGE_VOICE))
def test_judge_voice_metric_registry(metric_id):
    name, tier, criterion, skip_when = JUDGE_VOICE[metric_id]
    metric = _metric(metric_id)
    assert metric.metric_id == metric_id
    assert metric.name == name
    assert metric.category == "voice"
    assert metric.tier == tier
    assert metric.criterion
    if criterion is not None:
        assert metric.criterion == criterion
    if skip_when is not None:
        assert metric.skip_when == skip_when
    else:
        assert metric.skip_when


# =============================================================================================
# VOICE-001 time to first audio
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "latencies, raw, label",
    [
        ([400, 900, 1200], 900.0, "good"),
        ([1000, 2000], 1500.0, "good"),  # median of an even count; the warn threshold is inclusive
        ([1500.5], 1500.5, "warning"),
        ([2000, 3000, 3000], 3000.0, "warning"),  # the fail threshold is inclusive
        ([3000.5], 3000.5, "poor"),
        ([100, 4000, 5000, 6000], 4500.0, "poor"),
    ],
)
async def test_voice001_text_mode_proxy_thresholds(latencies, raw, label):
    result = await _evaluate("VOICE-001", _transcript(_latency_turns(latencies)), ctx=_ctx(voice_mode=False))
    _assert_label(result, label)
    assert result.raw_value == pytest.approx(raw)
    assert result.is_proxy is True
    assert result.informational is False
    assert result.metric_id == "VOICE-001"
    assert result.category == "voice"
    assert result.threshold_warn == 1500
    assert result.threshold_fail == 3000


@pytest.mark.asyncio
async def test_voice001_voice_mode_is_not_a_proxy():
    transcript = _transcript(_latency_turns([800, 1200, 2500]), voice_mode=True)
    result = await _evaluate("VOICE-001", transcript, ctx=_ctx(voice_mode=True))
    _assert_label(result, "good")
    assert result.raw_value == pytest.approx(1200.0)
    assert result.is_proxy is False


@pytest.mark.asyncio
async def test_voice001_uses_configured_thresholds():
    ctx = _ctx(voice_mode=False, voice_thresholds={"ttfa_warn_ms": 500, "ttfa_fail_ms": 1000})
    result = await _evaluate("VOICE-001", _transcript(_latency_turns([800])), ctx=ctx)
    _assert_label(result, "warning")
    assert result.threshold_warn == 500
    assert result.threshold_fail == 1000


# =============================================================================================
# VOICE-003 dead air
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "latencies, count, label",
    [
        ([200, 300], 0, "good"),
        ([1500, 1600, 200], 1, "warning"),  # only gaps strictly longer than dead_air_ms count
        ([1600, 2000, 100], 2, "poor"),
        ([1600, 2000, 2500], 3, "poor"),
    ],
)
async def test_voice003_text_mode_proxy_counts_gaps(latencies, count, label):
    result = await _evaluate("VOICE-003", _transcript(_latency_turns(latencies)), ctx=_ctx(voice_mode=False))
    _assert_label(result, label)
    assert result.raw_value == count
    assert result.is_proxy is True


@pytest.mark.asyncio
async def test_voice003_uses_configured_dead_air():
    ctx = _ctx(voice_mode=False, voice_thresholds={"dead_air_ms": 500})
    result = await _evaluate("VOICE-003", _transcript(_latency_turns([600, 400])), ctx=ctx)
    _assert_label(result, "warning")
    assert result.raw_value == 1


@pytest.mark.asyncio
async def test_voice003_voice_mode_is_not_a_proxy():
    transcript = _transcript(_latency_turns([1700, 1800, 300]), voice_mode=True)
    result = await _evaluate("VOICE-003", transcript, ctx=_ctx(voice_mode=True))
    _assert_label(result, "poor")
    assert result.raw_value == 2
    assert result.is_proxy is False


# =============================================================================================
# VOICE-005 tail latency (p95, numpy.percentile linear interpolation)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "latencies, raw, label",
    [
        ([100, 200, 300], 290.0, "good"),
        ([3000], 3000.0, "good"),  # inclusive warn threshold
        ([100, 200, 300, 400, 5000], 4080.0, "warning"),  # linear: 400 + 0.8 * 4600
        ([1000, 6000], 5750.0, "poor"),
    ],
)
async def test_voice005_p95_linear_interpolation_and_thresholds(latencies, raw, label):
    result = await _evaluate("VOICE-005", _transcript(_latency_turns(latencies)), ctx=_ctx(voice_mode=False))
    _assert_label(result, label)
    assert result.raw_value == pytest.approx(raw)
    assert result.is_proxy is True
    assert result.threshold_warn == 3000
    assert result.threshold_fail == 5000


@pytest.mark.asyncio
async def test_voice005_voice_mode_is_not_a_proxy_and_uses_configured_thresholds():
    ctx = _ctx(voice_mode=True, voice_thresholds={"tail_latency_warn_ms": 1000, "tail_latency_fail_ms": 2000})
    transcript = _transcript(_latency_turns([900, 1100, 1300]), voice_mode=True)
    result = await _evaluate("VOICE-005", transcript, ctx=ctx)
    assert result.raw_value == pytest.approx(1280.0)  # 1100 + 0.9 * 200
    _assert_label(result, "warning")
    assert result.is_proxy is False
    assert result.threshold_warn == 1000
    assert result.threshold_fail == 2000


# =============================================================================================
# Outbound opener excluded from VOICE-001/003/005; skipped when nothing is measurable
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_mode", [False, True])
async def test_outbound_opener_excluded_from_latency_metrics(voice_mode):
    transcript = _transcript(_latency_turns([500], opener_latency=9000), voice_mode=voice_mode)
    ctx = _ctx(voice_mode=voice_mode)
    ttfa = await _evaluate("VOICE-001", transcript, ctx=ctx)
    assert ttfa.raw_value == pytest.approx(500.0)
    _assert_label(ttfa, "good")
    dead_air = await _evaluate("VOICE-003", transcript, ctx=ctx)
    assert dead_air.raw_value == 0
    _assert_label(dead_air, "good")
    tail = await _evaluate("VOICE-005", transcript, ctx=ctx)
    assert tail.raw_value == pytest.approx(500.0)
    _assert_label(tail, "good")


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_id", ["VOICE-001", "VOICE-005"])
async def test_latency_metric_skipped_when_only_the_opener_exists(metric_id):
    transcript = _transcript(_latency_turns([], opener_latency=900))
    result = await _evaluate(metric_id, transcript, ctx=_ctx(voice_mode=False))
    _assert_skipped(result)


# =============================================================================================
# VOICE-029 spoken-length fit (both modes, never a proxy)
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "word_counts, median, maximum, label",
    [
        ([10, 30, 50], 30, 50, "good"),
        ([40, 80, 20], 40, 80, "good"),  # both limits inclusive
        ([45, 45], 45, 45, "warning"),
        ([10, 10, 90], 10, 90, "warning"),  # max over 80 but within 120
        ([60, 120, 30], 60, 120, "warning"),
        ([70, 70], 70, 70, "poor"),
        ([10, 130, 10], 10, 130, "poor"),
    ],
)
@pytest.mark.parametrize("voice_mode", [False, True])
async def test_voice029_spoken_length_fit(word_counts, median, maximum, label, voice_mode):
    transcript = _transcript(_word_turns(word_counts), voice_mode=voice_mode)
    result = await _evaluate("VOICE-029", transcript, ctx=_ctx(voice_mode=voice_mode))
    _assert_label(result, label)
    assert result.raw_value == pytest.approx(median)
    assert str(maximum) in result.reason
    assert result.is_proxy is False


@pytest.mark.asyncio
async def test_voice029_skipped_without_agent_turns():
    transcript = _transcript([_turn("user", "Hello?", 0, end_ms=800)])
    result = await _evaluate("VOICE-029", transcript, ctx=_ctx(voice_mode=False))
    _assert_skipped(result)


# =============================================================================================
# VOICE-009 overall WER (agent hearing): corpus WER, normalised
# =============================================================================================


def _hearing_turns(pairs):
    """``pairs``: (user text, heard_text) per user turn; each followed by an agent turn."""
    turns, t = [], 0
    for index, (said, heard) in enumerate(pairs, start=1):
        turns.append(_turn("user", said, t, end_ms=t + 1500, heard_text=heard))
        turns.append(_turn("agent", f"Agent answer number {index}", t + 2000, end_ms=t + 3000, latency_ms=500.0))
        t += 4000
    return turns


@pytest.mark.asyncio
async def test_voice009_is_corpus_wer_over_turns_with_heard_text():
    pairs = [
        ("Please book me an appointment for Tuesday morning at ten.",
         "please book me an appointment for tuesday morning at ten"),
        ("Yes please", "yes"),  # one deletion
        ("Thank you very much", None),  # no heard_text: not in the corpus
    ]
    transcript = _transcript(_hearing_turns(pairs), voice_mode=True)
    result = await _evaluate("VOICE-009", transcript, ctx=_ctx(voice_mode=True))
    # Corpus WER = 1 error / 12 reference words (a per-turn mean would be 0.25, a warning).
    assert result.raw_value == pytest.approx(1 / 12)
    _assert_label(result, "good")
    assert result.threshold_warn == pytest.approx(0.15)
    assert result.threshold_fail == pytest.approx(0.30)
    assert result.is_proxy is False
    assert result.informational is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pairs, raw, label",
    [
        ([("I need a dental checkup", "i need a mental checkup"),
          ("on Friday after five pm", "on friday after nine pm")], 0.2, "warning"),
        ([("Cancel my old booking", "council by gold looking")], 1.0, "poor"),
        ([("Is Tuesday, at ten, free?", "is tuesday at ten free")], 0.0, "good"),  # case and punctuation ignored
    ],
)
async def test_voice009_thresholds(pairs, raw, label):
    transcript = _transcript(_hearing_turns(pairs), voice_mode=True)
    result = await _evaluate("VOICE-009", transcript, ctx=_ctx(voice_mode=True))
    assert result.raw_value == pytest.approx(raw)
    _assert_label(result, label)


@pytest.mark.asyncio
async def test_voice009_uses_configured_wer_thresholds():
    pairs = [("I need a dental checkup", "i need a mental checkup")]  # WER 0.2
    ctx = _ctx(voice_mode=True, voice_thresholds={"wer_warn": 0.05, "wer_fail": 0.10})
    result = await _evaluate("VOICE-009", _transcript(_hearing_turns(pairs), voice_mode=True), ctx=ctx)
    _assert_label(result, "poor")
    assert result.threshold_warn == pytest.approx(0.05)
    assert result.threshold_fail == pytest.approx(0.10)


# =============================================================================================
# VOICE-013 domain-term error rate
# =============================================================================================

KEY_TERMS = ["EMI", "CIBIL score", "NACH", "KYC", "PAN"]
SAID_1 = "what is my EMI and my CIBIL score"
SAID_2 = "please cancel NACH after KYC"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "heard_1, heard_2, raw, label",
    [
        ("what is my emi and my cibil score", "please cancel nach after kyc", 0.0, "good"),
        ("what is my emi and my civil score", "please cancel nach after kyc", 0.25, "warning"),
        # whole-token: "emis" does not contain the term "emi"
        ("what is my emis and my civil score", "please cancel nach after kyc", 0.5, "poor"),
        # a multi-word term matches only as a whole-token sequence
        ("what is my emi and my cibil and score", "please cancel nach after kyc", 0.25, "warning"),
        # a term must be heard in the turn where the user said it
        ("what is my emi and my cibil score kyc", "please cancel nach after", 0.25, "warning"),
    ],
)
async def test_voice013_fraction_of_said_terms_missing_from_heard_text(heard_1, heard_2, raw, label):
    # "PAN" is never said: it is not in the denominator (4 said terms, not 5).
    transcript = _transcript(_hearing_turns([(SAID_1, heard_1), (SAID_2, heard_2)]), voice_mode=True)
    scenario = _scenario(key_terms=KEY_TERMS)
    result = await _evaluate("VOICE-013", transcript, scenario=scenario, ctx=_ctx(voice_mode=True))
    assert result.raw_value == pytest.approx(raw)
    _assert_label(result, label)
    assert result.threshold_warn == pytest.approx(0.15)
    assert result.threshold_fail == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_voice013_skips_when_user_never_said_a_key_term():
    # "EMIs" is not the whole token "EMI", so no key term was said.
    transcript = _transcript(_hearing_turns([("two EMIs are due", "two emis are due")]), voice_mode=True)
    result = await _evaluate(
        "VOICE-013", transcript, scenario=_scenario(key_terms=["EMI", "KYC"]), ctx=_ctx(voice_mode=True)
    )
    _assert_skipped(result)


@pytest.mark.asyncio
async def test_voice013_skips_without_key_terms():
    transcript = _transcript(_hearing_turns([(SAID_1, SAID_1.lower())]), voice_mode=True)
    result = await _evaluate("VOICE-013", transcript, scenario=_scenario(key_terms=[]), ctx=_ctx(voice_mode=True))
    _assert_skipped(result)


# =============================================================================================
# VOICE-015 speech rate (end to end: STT word timings come from the runner's voice pipeline)
# =============================================================================================

VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}


def _spread_words(count, span_ms):
    """``count`` words from 0 ms to exactly ``span_ms`` (WordTiming shape)."""
    return [
        {
            "word": f"w{index}",
            "start_ms": int(round(index * span_ms / count)),
            "end_ms": int(round((index + 1) * span_ms / count)),
            "confidence": 0.9,
        }
        for index in range(count)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count, span_ms, wpm, label",
    [
        (12, 4100, 12 * 60000 / 4100, "good"),
        (11, 5900, 11 * 60000 / 5900, "good"),
        (10, 4000, 150.0, "good"),
        (13, 4000, 195.0, "warning"),  # within 20 wpm above the range
        (10, 6000, 100.0, "warning"),  # within 20 wpm below the range
        (15, 4000, 225.0, "poor"),
        (8, 6000, 80.0, "poor"),
    ],
)
async def test_voice015_speech_rate_from_stt_word_timings(
    impulse_project, llm_router, fake_adapter, mock_voice_server, count, span_ms, wpm, label
):
    from impulse_evals.runner import run

    project = impulse_project
    words = _spread_words(count, span_ms)
    text = " ".join(word["word"] for word in words)
    for _ in range(20):  # every transcription (round trip and agent audio) returns the same result
        mock_voice_server.queue_stt(text, words=words)
    agent_audio = wav_bytes(pcm_tone(span_ms / 1000.0, 16000))
    fake_adapter.replies["speech_rate"] = [{"text": "Your slot is booked.", "audio_bytes": agent_audio, "latency_ms": 400.0}]
    project.write_config(
        [project.scenario("speech_rate", max_turns=1, metrics=["VOICE-015"])], voice_config=VOICE_CONFIG
    )
    result = await run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    metrics = {metric.metric_id: metric for metric in scenario_result.metric_results}
    speech_rate = metrics["VOICE-015"]
    assert speech_rate.raw_value == pytest.approx(wpm, rel=0.002)
    _assert_label(speech_rate, label)
    assert speech_rate.is_proxy is False


# =============================================================================================
# Judge-based voice metrics
# =============================================================================================


def _conversation(user_turns, interrupted_at=None):
    turns, t = [], 0
    for index in range(1, user_turns + 1):
        turns.append(_turn("user", f"User line {index}", t, end_ms=t + 1000))
        turns.append(
            _turn("agent", f"Agent line {index}", t + 1500, end_ms=t + 3000, latency_ms=500.0,
                  interrupted=(index == interrupted_at))
        )
        t += 4000
    return turns


@pytest.mark.asyncio
async def test_voice027_sends_return_skip_line(fake_judge):
    judge = fake_judge()
    name, tier, criterion, skip_when = JUDGE_VOICE["VOICE-027"]
    await _metric("VOICE-027").evaluate(_transcript(_conversation(2)), _scenario(), judge, _ctx())
    assert len(judge.calls) == 1
    assert judge.calls[0]["tier"] == tier
    assert judge.calls[0]["criterion"].rstrip() == criterion + "\n\nReturn skip if: " + skip_when


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_id", ["VOICE-037", "VOICE-039"])
async def test_never_skip_voice_metrics_send_criterion_only(fake_judge, metric_id):
    judge = fake_judge({"result": "fail", "reason": "reads a URL aloud", "turn_index": 2, "confidence": 0.8})
    name, tier, criterion, skip_when = JUDGE_VOICE[metric_id]
    result = await _metric(metric_id).evaluate(_transcript(_conversation(2)), _scenario(), judge, _ctx())
    assert len(judge.calls) == 1
    assert judge.calls[0]["tier"] == tier
    assert judge.calls[0]["criterion"].rstrip() == criterion
    _assert_label(result, "poor")
    assert result.turn_index == 2


@pytest.mark.asyncio
async def test_voice007_judged_when_a_turn_was_interrupted(fake_judge):
    judge = fake_judge()
    name, tier, criterion, _ = JUDGE_VOICE["VOICE-007"]
    transcript = _transcript(_conversation(2, interrupted_at=1), voice_mode=True)
    result = await _metric("VOICE-007").evaluate(transcript, _scenario(), judge, _ctx(voice_mode=True))
    assert len(judge.calls) == 1
    assert judge.calls[0]["tier"] == tier
    assert judge.calls[0]["criterion"].rstrip() == criterion  # code-decided skip: no Return skip line
    _assert_label(result, "good")


@pytest.mark.asyncio
async def test_voice031_judged_when_code_switching(fake_judge):
    judge = fake_judge()
    name, tier, criterion, _ = JUDGE_VOICE["VOICE-031"]
    scenario = _scenario(code_switching=True, language="hi-Latn-IN")
    result = await _metric("VOICE-031").evaluate(_transcript(_conversation(2)), scenario, judge, _ctx())
    assert len(judge.calls) == 1
    assert judge.calls[0]["tier"] == tier
    assert judge.calls[0]["criterion"].rstrip() == criterion
    _assert_label(result, "good")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict, raw, label",
    [
        ({"result": "pass", "turn_index": 12}, 6, "good"),  # 6 user turns <= 60% of 10
        ({"result": "pass", "turn_index": 14}, 7, "warning"),
        ({"result": "pass", "turn_index": 18}, 9, "warning"),  # <= 90% of 10
        ({"result": "pass", "turn_index": 20}, 10, "poor"),
        ({"result": "fail", "turn_index": None}, 10, "poor"),  # never achieved: max_turns
    ],
)
@pytest.mark.parametrize("voice_mode", [False, True])
async def test_voice035_turns_to_close_mapping(fake_judge, verdict, raw, label, voice_mode):
    judge = fake_judge(dict(verdict, reason="outcome reached", confidence=0.9))
    transcript = _transcript(_conversation(10), voice_mode=voice_mode)
    scenario = _scenario(max_turns=10, expected_outcome="appointment booked")
    result = await _metric("VOICE-035").evaluate(transcript, scenario, judge, _ctx(voice_mode=voice_mode))
    assert len(judge.calls) == 1
    assert judge.calls[0]["tier"] == "fast"
    assert result.raw_value == pytest.approx(raw)
    _assert_label(result, label)
    assert result.is_proxy is False


# =============================================================================================
# Audit batch 2 (AUDIT.md finding 16, maintainer decision #16): a VOICE-035 ``pass`` verdict whose
# ``turn_index`` is missing, not a 1-based position in the transcript, or points at a turn that is
# not an agent turn is SKIPPED with the reason "judge passed without a valid turn_index"
# (``raw_value`` None, ``turn_index`` None), never scored "never reached" / poor. A ``fail`` stays
# "never reached" (poor, ``max_turns``); a ``pass`` at a valid agent turn is unchanged.
# =============================================================================================

VOICE035_NO_VALID_INDEX = "judge passed without a valid turn_index"


async def _voice035(fake_judge, verdict, voice_mode=False, user_turns=10, max_turns=10):
    """VOICE-035 on ``_conversation(user_turns)`` (user turns at odd, agent turns at even positions)."""
    judge = fake_judge(dict({"reason": "the booking was confirmed", "confidence": 0.9}, **verdict))
    transcript = _transcript(_conversation(user_turns), voice_mode=voice_mode)
    scenario = _scenario(max_turns=max_turns, expected_outcome="appointment booked")
    result = await _metric("VOICE-035").evaluate(transcript, scenario, judge, _ctx(voice_mode=voice_mode))
    assert len(judge.calls) == 1
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turn_index",
    [None, 0, -1, 21, 99],
    ids=["null", "zero", "negative", "one_past_the_end", "far_out_of_range"],
)
@pytest.mark.parametrize("voice_mode", [False, True])
async def test_voice035_pass_without_a_valid_turn_index_is_skipped(fake_judge, turn_index, voice_mode):
    result = await _voice035(fake_judge, {"result": "pass", "turn_index": turn_index}, voice_mode)  # 20 turns
    _assert_skipped(result)
    assert result.reason == VOICE035_NO_VALID_INDEX
    assert result.raw_value is None
    assert result.turn_index is None
    assert result.metric_id == "VOICE-035"


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_index", [1, 3, 19], ids=["first_user_turn", "user_turn_3", "last_user_turn"])
async def test_voice035_pass_pointing_at_a_user_turn_is_skipped(fake_judge, turn_index):
    result = await _voice035(fake_judge, {"result": "pass", "turn_index": turn_index})
    _assert_skipped(result)
    assert result.reason == VOICE035_NO_VALID_INDEX
    assert result.raw_value is None
    assert result.turn_index is None


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_index", [None, 4, 3, 0, 99], ids=["null", "agent_turn", "user_turn", "zero", "out_of_range"])
async def test_voice035_fail_stays_never_reached_poor(fake_judge, turn_index):
    # Regression guard: a fail verdict (with or without an index) is "never reached": max_turns, poor.
    result = await _voice035(fake_judge, {"result": "fail", "turn_index": turn_index})
    _assert_label(result, "poor")
    assert result.raw_value == pytest.approx(10.0)
    assert result.turn_index is None


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_index, raw, label", [(2, 1, "good"), (12, 6, "good"), (20, 10, "poor")])
async def test_voice035_pass_at_a_valid_agent_turn_is_unchanged(fake_judge, turn_index, raw, label):
    # Regression guard: a pass at an agent turn maps to the user turns up to it, labelled against max_turns.
    result = await _voice035(fake_judge, {"result": "pass", "turn_index": turn_index})
    _assert_label(result, label)
    assert result.raw_value == pytest.approx(raw)
    assert result.turn_index == turn_index


@pytest.mark.asyncio
async def test_voice035_pass_with_null_turn_index_does_not_fail_the_scenario(impulse_project, llm_router, fake_adapter):
    # End to end: the router's default judge answer is a pass with turn_index null (the normal
    # shape of a pass for every other metric); the scenario must not fail because of it.
    from impulse_evals.models import compute_passed
    from impulse_evals.runner import run

    project = impulse_project
    name = "closes_fast"
    fake_adapter.replies[name] = [{"text": "Booked for Tuesday at ten.", "latency_ms": 100.0}]
    project.write_config(
        [project.scenario(name, max_turns=3, opening_user_message="please book a slot on tuesday", metrics=["VOICE-035"])]
    )
    result = await run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    voice035 = {metric.metric_id: metric for metric in scenario_result.metric_results}["VOICE-035"]
    _assert_skipped(voice035)
    assert voice035.reason == VOICE035_NO_VALID_INDEX
    assert scenario_result.passed is True
    assert compute_passed(scenario_result) is True
    assert result.failed == 0


# =============================================================================================
# Audit batch 3 (AUDIT.md finding 33): a non-finite ``AgentReply.latency_ms`` from an adapter (NaN,
# +inf, -inf) is recorded as None on the agent turn and ignored by the run's latency profile and by
# VOICE-001 (text-mode proxy), so the run file round-trips (``RunResult.model_validate_json``).
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "minus-inf"])
async def test_non_finite_adapter_latency_is_ignored_and_the_run_round_trips(impulse_project, llm_router, fake_adapter, bad):
    from impulse_evals.models import RunResult
    from impulse_evals.runner import run

    project = impulse_project
    name = "non_finite_latency"
    llm_router.script(project.persona(name), "Is ten in the morning free?")
    fake_adapter.replies[name] = [
        {"text": "Let me check the calendar.", "latency_ms": bad},
        {"text": "Ten is free.", "latency_ms": 120.0},
    ]
    project.write_config([project.scenario(name, max_turns=2, metrics=["VOICE-001"])])
    result = await run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    agent_turns = [turn for turn in scenario_result.transcript.turns if turn.speaker == "agent"]
    assert [turn.latency_ms for turn in agent_turns] == [None, 120.0]
    profile = result.latency_profile
    assert profile.all_latencies_ms == [120.0]
    assert (profile.p50_ms, profile.p75_ms, profile.p95_ms, profile.p99_ms) == (120.0, 120.0, 120.0, 120.0)
    ttfa = {metric.metric_id: metric for metric in scenario_result.metric_results}["VOICE-001"]
    assert ttfa.raw_value == pytest.approx(120.0)
    text = result.model_dump_json()
    assert RunResult.model_validate_json(text).model_dump_json() == text   # run.json can be read back


# =============================================================================================
# Audit batch 3 (AUDIT.md finding 47 a): in a VOICE-mode run whose agent returned no audio, the
# VOICE-001/003/005 proxy note reads " (proxy: Turn.latency_ms; the agent returned no audio)",
# never "text-mode proxy"; a text-mode run keeps " (text-mode proxy: Turn.latency_ms)".
# =============================================================================================

VOICE_MODE_PROXY_NOTE = " (proxy: Turn.latency_ms; the agent returned no audio)"
TEXT_MODE_PROXY_NOTE = " (text-mode proxy: Turn.latency_ms)"
LATENCY_METRICS = ["VOICE-001", "VOICE-003", "VOICE-005"]


@pytest.mark.asyncio
async def test_voice_mode_latency_proxy_note_says_the_agent_returned_no_audio(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    from impulse_evals.runner import run

    project = impulse_project
    name = "no_agent_audio"
    llm_router.script(project.persona(name), "Is Tuesday free?")
    project.write_config([project.scenario(name, max_turns=2, metrics=LATENCY_METRICS)], voice_config=VOICE_CONFIG)
    result = await run(project.load(), adapter_factory=fake_adapter)  # replies: text and latency, no audio
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    assert result.voice_mode is True
    metrics = {metric.metric_id: metric for metric in scenario_result.metric_results}
    for metric_id in LATENCY_METRICS:
        metric = metrics[metric_id]
        assert metric.is_proxy is True
        assert VOICE_MODE_PROXY_NOTE in metric.reason, metric.reason
        assert "text-mode" not in metric.reason, metric.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_id", LATENCY_METRICS)
async def test_text_mode_latency_proxy_note_is_unchanged(metric_id):
    # Regression guard: a text-mode run keeps the "text-mode proxy" note.
    result = await _evaluate(metric_id, _transcript(_latency_turns([400, 1600])), ctx=_ctx(voice_mode=False))
    assert result.is_proxy is True
    assert TEXT_MODE_PROXY_NOTE in result.reason, result.reason
    assert VOICE_MODE_PROXY_NOTE not in result.reason


@pytest.mark.asyncio
async def test_voice_mode_with_agent_audio_has_no_proxy_note(impulse_project, llm_router, fake_adapter, mock_voice_server):
    # Regression guard: agent audio in every measured turn: a time to first audio, not a proxy.
    from impulse_evals.runner import run

    project = impulse_project
    name = "with_agent_audio"
    fake_adapter.replies[name] = [{"text": "Booked.", "audio_bytes": wav_bytes(pcm_tone(0.5)), "latency_ms": 300.0}]
    project.write_config([project.scenario(name, max_turns=1, metrics=LATENCY_METRICS)], voice_config=VOICE_CONFIG)
    result = await run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None, scenario_result.error
    metrics = {metric.metric_id: metric for metric in scenario_result.metric_results}
    for metric_id in LATENCY_METRICS:
        assert metrics[metric_id].is_proxy is False
        assert "proxy" not in metrics[metric_id].reason, metrics[metric_id].reason
