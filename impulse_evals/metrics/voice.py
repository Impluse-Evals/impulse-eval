# SPDX-License-Identifier: Apache-2.0
"""Voice metrics (METRICS.md section 2; SPEC.md sections 7, 13, and 14).

Criterion and "Skip when" texts are copied word for word from METRICS.md section 2.2.

Deterministic metrics (METRICS.md section 2.1) compute ``raw_value``, label it against the
thresholds of ``MetricContext.voice_thresholds`` (the ``metrics.voice`` block; a missing key takes
the SPEC.md section 8.2 default), and score good = 1.0, warning = 0.5, poor = 0.0. Every
deterministic result records the raw thresholds it was labelled with in ``threshold_warn`` /
``threshold_fail``.

- VOICE-001 / 003 / 005 read ``Turn.latency_ms`` of the agent turns that follow a user turn (the
  outbound opener is excluded). In voice mode the adapter's latency is the time to first agent audio
  (SPEC.md section 9), so the result is not a proxy; in text mode it is the labelled text-mode proxy
  (``is_proxy=True``, reason note " (text-mode proxy: Turn.latency_ms)"), and in voice mode when the
  agent returned no audio for a measured turn it is a proxy too (note " (proxy: Turn.latency_ms; the
  agent returned no audio)").
- VOICE-009 / 011 / 013 / 015 need audio: in text mode they are skipped with "not available in text
  mode". VOICE-009 / 013 / 015 depend on the agent's audio path: in voice mode, when the pipeline
  data shows that no agent turn of the scenario returned audio (e.g. the generic adapter returning
  text only), they are skipped with "agent returned no audio" (SPEC.md section 9), before any other
  reason. VOICE-009 / 013 use ``Turn.heard_text`` (the agent's own transcription of the user);
  VOICE-011 (informational) and VOICE-015 use the voice pipeline's data that the runner stores in
  ``CallTranscript.metadata["impulse_voice"]`` (round-trip transcriptions of the user turns, and the
  STT word timings of the agent's audio). When Impulse's own STT failed on a user turn's audio
  (VOICE-011) or on any agent turn's audio (VOICE-015), the metric is skipped with "Impulse STT
  failed: {error}".
- VOICE-029 runs identically in both modes.

Judge-based metrics (METRICS.md section 2.2): skips that depend only on structure are decided in
code (VOICE-007: no interrupted turn; VOICE-031: no code switching); VOICE-027's condition is sent to
the judge. VOICE-035 maps a ``pass`` verdict's ``turn_index`` (the first agent turn at which the
outcome was reached; it must be the position of an agent turn) to the number of user turns up to
that turn; a ``fail`` means never reached; a ``pass`` without a valid agent-turn index is skipped.

Text normalisation for key terms (VOICE-013, ``normalize_text``): lowercase, apostrophes removed,
every other punctuation character (Unicode category P) replaced by a space, then split on whitespace.
WER (VOICE-009 / VOICE-011, ``corpus_wer``) normalises both sides with ``normalize_for_wer``
(``metrics/_wer_normalize.py``, FIXES-ROUND-2-SPEC.md A4 and A4-rev): the same, plus numbers,
number words, dates, spelled-out letters, and IDs written with a hyphen or a spoken "dash" brought to
one form; an ambiguous numeric date is resolved per pair.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

from impulse_evals.config import VoiceMetricsConfig
from impulse_evals.metrics._wer_normalize import (
    normalize_for_wer,
    normalize_pair_for_wer,
)
from impulse_evals.metrics.base import (
    LABEL_SCORES,
    JudgeMetric,
    Metric,
    MetricContext,
    register_metric,
)
from impulse_evals.models import CallTranscript, MetricResult, Scenario, Turn

if TYPE_CHECKING:  # annotation only
    from impulse_evals.judge.base import Judge, JudgeVerdict

__all__ = [
    "NOT_IN_TEXT_MODE",
    "NO_AGENT_AUDIO",
    "NO_HEARD_TEXT",
    "NO_WORD_TIMINGS",
    "VOICE_METADATA_KEY",
    "corpus_wer",
    "no_agent_audio",
    "normalize_for_wer",
    "normalize_text",
]

# ------------------------------------------------------------------------------------------------
# Exact texts (SPEC.md sections 9 and 14.3; METRICS.md sections 0 and 2.1)
# ------------------------------------------------------------------------------------------------

NOT_IN_TEXT_MODE = "not available in text mode"
NO_HEARD_TEXT = "agent does not expose its transcription; return heard_text from the adapter"
NO_AGENT_AUDIO = "agent returned no audio"
NO_WORD_TIMINGS = "STT provider returned no word timings"

# Key of the voice pipeline data in ``CallTranscript.metadata`` (written by the runner in voice mode):
# ``{"turns": [record, ...]}``, one record per turn that went through the voice pipeline:
#   user:  {"turn_index", "speaker": "user", "audio_ms", "round_trip_text", "round_trip_confidence",
#           "round_trip_error" (only when Impulse's STT of the user audio failed)}
#   agent: {"turn_index", "speaker": "agent", "audio_ms" (None: no agent audio), "stt_text",
#           "stt_confidence", "words": [WordTiming fields, ...],
#           "stt_error" (only when Impulse's STT of the agent audio failed)}
VOICE_METADATA_KEY = "impulse_voice"

NO_VOICE_DATA = "no voice pipeline data in this transcript (it was not recorded in voice mode)"
# VOICE-011 (round trip) / VOICE-015 (agent audio) when Impulse's own STT failed (SPEC.md section 14).
IMPULSE_STT_FAILED = "Impulse STT failed: {error}"

# VOICE-035 (SPEC.md section 13): a ``pass`` whose ``turn_index`` is not an agent turn's position.
PASS_WITHOUT_TURN_INDEX = "judge passed without a valid turn_index"

_DEFAULT_THRESHOLDS: dict[str, Any] = VoiceMetricsConfig().model_dump()

# VOICE-011 fixed thresholds (METRICS.md section 2.1).
ROUND_TRIP_WARN = 0.10
ROUND_TRIP_FAIL = 0.20

# VOICE-029 limits (METRICS.md section 2.1): (median, max) for good and for warning.
LENGTH_GOOD = (40, 80)
LENGTH_WARNING = (60, 120)

# VOICE-015: warning band outside the speech_rate_wpm range.
SPEECH_RATE_BAND = 20.0


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


def _ctx(ctx: Optional[MetricContext]) -> MetricContext:
    return ctx if ctx is not None else MetricContext()


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def thresholds(ctx: Optional[MetricContext]) -> dict[str, Any]:
    """The ``metrics.voice`` thresholds: ``ctx.voice_thresholds`` over the SPEC.md section 8.2 defaults."""
    values = dict(_DEFAULT_THRESHOLDS)
    for key, value in (_ctx(ctx).voice_thresholds or {}).items():
        if value is not None:
            values[key] = value
    return values


def _threshold(values: Mapping[str, Any], key: str) -> float:
    return _number(values.get(key), float(_DEFAULT_THRESHOLDS[key]))


def _speech_rate_range(values: Mapping[str, Any]) -> tuple[float, float]:
    default = tuple(float(v) for v in _DEFAULT_THRESHOLDS["speech_rate_wpm"])
    raw = values.get("speech_rate_wpm")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        low, high = _number(raw[0], math.nan), _number(raw[1], math.nan)
        if math.isfinite(low) and math.isfinite(high) and low <= high:
            return low, high
    return default[0], default[1]


def _percentile(values: Sequence[float], q: float) -> float:
    import numpy  # core dependency; loaded only when a voice metric computes a statistic

    return float(numpy.percentile(list(values), q))


def _label_upper(raw: float, warn: float, fail: float) -> str:
    """Lower is better: good when ``raw <= warn``, warning when ``raw <= fail``, else poor."""
    if raw <= warn:
        return "good"
    if raw <= fail:
        return "warning"
    return "poor"


def _ms(value: float) -> str:
    return f"{value:.0f} ms"


def _g(value: float) -> str:
    return f"{value:g}"


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, apostrophes removed, other punctuation replaced by spaces, whitespace collapsed."""
    if not text:
        return ""
    out: list[str] = []
    for char in text.lower():
        if char in "'’ʼ":
            continue
        out.append(" " if unicodedata.category(char).startswith("P") else char)
    return " ".join("".join(out).split())


def _tokens(text: Optional[str]) -> list[str]:
    return normalize_text(text).split()


def corpus_wer(pairs: Iterable[tuple[Optional[str], Optional[str]]]) -> Optional[float]:
    """Corpus WER (jiwer) over ``(reference, hypothesis)`` pairs, both normalised with
    ``normalize_pair_for_wer`` (``normalize_for_wer`` on each side, plus the per-pair resolution of an
    ambiguous numeric date): all word errors divided by all reference words. None when the
    references hold no words."""
    references: list[str] = []
    hypotheses: list[str] = []
    for reference, hypothesis in pairs:
        # Per pair, so an ambiguous numeric date is resolved only against its own pair (A4-rev).
        normalized_reference, normalized_hypothesis = normalize_pair_for_wer(reference, hypothesis)
        references.append(normalized_reference)
        hypotheses.append(normalized_hypothesis)
    reference_words = sum(len(reference.split()) for reference in references)
    if reference_words == 0:
        return None
    import jiwer  # core dependency; loaded only when a WER is computed

    output = jiwer.process_words(references, hypotheses)
    errors = output.substitutions + output.deletions + output.insertions
    return errors / reference_words


def _contains_sequence(tokens: Sequence[str], term: Sequence[str]) -> bool:
    size = len(term)
    if size == 0 or size > len(tokens):
        return False
    return any(list(tokens[start : start + size]) == list(term) for start in range(len(tokens) - size + 1))


def _measured_agent_turns(transcript: CallTranscript) -> list[tuple[int, Turn, float]]:
    """``(turn number, turn, latency_ms)`` of every agent turn after a user turn with a finite latency.

    Agent turns with no user turn before them (the outbound opener) are excluded (METRICS.md 2.1).
    """
    measured: list[tuple[int, Turn, float]] = []
    seen_user = False
    for number, turn in enumerate(transcript.turns, start=1):
        if turn.speaker == "user":
            seen_user = True
        elif seen_user and turn.latency_ms is not None:
            latency = _number(turn.latency_ms, math.nan)
            if math.isfinite(latency):
                measured.append((number, turn, latency))
    return measured


def voice_records(transcript: CallTranscript) -> Optional[list[dict[str, Any]]]:
    """The runner's voice pipeline records (``metadata["impulse_voice"]["turns"]``), or None."""
    metadata = transcript.metadata if isinstance(transcript.metadata, dict) else {}
    block = metadata.get(VOICE_METADATA_KEY)
    if not isinstance(block, dict) or not isinstance(block.get("turns"), list):
        return None
    return [record for record in block["turns"] if isinstance(record, dict)]


def _agent_records(transcript: CallTranscript) -> Optional[dict[int, dict[str, Any]]]:
    records = voice_records(transcript)
    if records is None:
        return None
    return {
        int(record["turn_index"]): record
        for record in records
        if record.get("speaker") == "agent" and isinstance(record.get("turn_index"), int)
    }


def _has_audio(record: Optional[Mapping[str, Any]]) -> bool:
    return record is not None and record.get("audio_ms") is not None


def _stt_error(records: Iterable[Mapping[str, Any]], speaker: str, key: str) -> Optional[str]:
    """The first ``key`` error text among ``speaker``'s records (Impulse's own STT failed), or None."""
    for record in records:
        error = record.get(key)
        if record.get("speaker") == speaker and isinstance(error, str):
            return error
    return None


def no_agent_audio(transcript: CallTranscript) -> bool:
    """True when the voice pipeline data shows that no agent turn of the scenario returned audio (e.g.
    the generic adapter returning text only); False without pipeline data (a hand-built transcript)."""
    records = _agent_records(transcript)
    return records is not None and not any(_has_audio(record) for record in records.values())


# ------------------------------------------------------------------------------------------------
# Deterministic voice metrics (METRICS.md section 2.1)
# ------------------------------------------------------------------------------------------------


class _DeterministicVoiceMetric(Metric):
    category = "voice"
    tier = None
    criterion = None
    requires_audio = True
    audio_only = False   # True: no text-mode proxy; skipped with "not available in text mode"
    # True: depends on the agent's audio path; in voice mode skipped with "agent returned no audio"
    # when no agent turn of the scenario returned audio (SPEC.md section 9, generic adapter).
    needs_agent_audio = False

    def labelled(self, raw: float, label: str, reason: str, *, warn: float, fail: float, **fields: Any) -> MetricResult:
        return self.result(
            score=LABEL_SCORES[label],
            label=label,
            reason=reason,
            raw_value=raw,
            threshold_warn=warn,
            threshold_fail=fail,
            **fields,
        )

    async def evaluate(
        self,
        transcript: CallTranscript,
        scenario: Scenario,
        judge: Judge,
        ctx: Optional[MetricContext] = None,
    ) -> MetricResult:
        ctx = _ctx(ctx)
        if self.audio_only and not ctx.voice_mode:
            return self.skipped(NOT_IN_TEXT_MODE)
        if self.needs_agent_audio and no_agent_audio(transcript):
            return self.skipped(NO_AGENT_AUDIO)
        return self.compute(transcript, scenario, ctx)

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        raise NotImplementedError  # pragma: no cover - every concrete metric overrides it


class _LatencyMetric(_DeterministicVoiceMetric):
    """VOICE-001 / 003 / 005: agent latency after user input (text-mode proxy: ``Turn.latency_ms``)."""

    skip_when = "no agent turn with a measured latency follows a user turn (the outbound opener is excluded)"

    def proxy(self, transcript: CallTranscript, measured: Sequence[tuple[int, Turn, float]], ctx: MetricContext) -> bool:
        """A proxy in text mode, and in voice mode when a measured agent turn carried no audio (its
        latency is then not a time to first audio)."""
        if not ctx.voice_mode:
            return True
        records = _agent_records(transcript)
        if records is None:   # no pipeline data (e.g. a hand-built transcript): trust the mode
            return False
        return any(not _has_audio(records.get(number)) for number, _, _ in measured)

    @staticmethod
    def proxy_note(is_proxy: bool, ctx: MetricContext) -> str:
        if not is_proxy:
            return ""
        if ctx.voice_mode:   # a voice run whose agent returned no audio for a measured turn
            return " (proxy: Turn.latency_ms; the agent returned no audio)"
        return " (text-mode proxy: Turn.latency_ms)"


@register_metric
class TimeToFirstAudio(_LatencyMetric):
    metric_id = "VOICE-001"
    name = "Time to first audio (TTFA)"

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        values = thresholds(ctx)
        warn, fail = _threshold(values, "ttfa_warn_ms"), _threshold(values, "ttfa_fail_ms")
        measured = _measured_agent_turns(transcript)
        is_proxy = self.proxy(transcript, measured, ctx)
        if not measured:
            return self.skipped("no agent turn with a measured latency follows a user turn", is_proxy=is_proxy)
        raw = _percentile([latency for _, _, latency in measured], 50)
        label = _label_upper(raw, warn, fail)
        reason = (
            f"median time to first audio {_ms(raw)} over {len(measured)} agent turns{self.proxy_note(is_proxy, ctx)}; "
            f"good <= {_g(warn)} ms, warning <= {_g(fail)} ms"
        )
        return self.labelled(raw, label, reason, warn=warn, fail=fail, is_proxy=is_proxy)


@register_metric
class DeadAir(_LatencyMetric):
    metric_id = "VOICE-003"
    name = "Dead air"

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        dead_air = _threshold(thresholds(ctx), "dead_air_ms")
        measured = _measured_agent_turns(transcript)
        is_proxy = self.proxy(transcript, measured, ctx)
        if not measured:
            return self.skipped("no agent turn with a measured latency follows a user turn", is_proxy=is_proxy)
        gaps = [(number, latency) for number, _, latency in measured if latency > dead_air]
        count = len(gaps)
        label = "good" if count == 0 else ("warning" if count == 1 else "poor")
        reason = f"{count} gaps longer than {_g(dead_air)} ms between user input and agent output{self.proxy_note(is_proxy, ctx)}"
        if gaps:
            reason += " (turns " + ", ".join(str(number) for number, _ in gaps) + ")"
        reason += "; good = 0, warning = 1, poor >= 2"
        return self.labelled(
            float(count), label, reason, warn=0.0, fail=1.0, is_proxy=is_proxy,
            turn_index=gaps[0][0] if gaps else None,
        )


@register_metric
class TailLatency(_LatencyMetric):
    metric_id = "VOICE-005"
    name = "Tail latency"

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        values = thresholds(ctx)
        warn, fail = _threshold(values, "tail_latency_warn_ms"), _threshold(values, "tail_latency_fail_ms")
        measured = _measured_agent_turns(transcript)
        is_proxy = self.proxy(transcript, measured, ctx)
        if not measured:
            return self.skipped("no agent turn with a measured latency follows a user turn", is_proxy=is_proxy)
        raw = _percentile([latency for _, _, latency in measured], 95)
        label = _label_upper(raw, warn, fail)
        slowest = max(measured, key=lambda item: item[2])
        reason = (
            f"p95 agent response latency {_ms(raw)} over {len(measured)} agent turns{self.proxy_note(is_proxy, ctx)} "
            f"(slowest: turn {slowest[0]}, {_ms(slowest[2])}); good <= {_g(warn)} ms, warning <= {_g(fail)} ms"
        )
        return self.labelled(raw, label, reason, warn=warn, fail=fail, is_proxy=is_proxy, turn_index=slowest[0])


@register_metric
class OverallWER(_DeterministicVoiceMetric):
    metric_id = "VOICE-009"
    name = "Overall WER (agent hearing)"
    audio_only = True
    needs_agent_audio = True
    skip_when = "text mode; the agent returned no audio; `heard_text` is None for all user turns"

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        values = thresholds(ctx)
        warn, fail = _threshold(values, "wer_warn"), _threshold(values, "wer_fail")
        pairs = [
            (turn.text, turn.heard_text)
            for turn in transcript.turns
            if turn.speaker == "user" and turn.heard_text is not None
        ]
        if not pairs:
            return self.skipped(NO_HEARD_TEXT)
        raw = corpus_wer(pairs)
        if raw is None:
            return self.skipped("the user turns the agent transcribed hold no words")
        label = _label_upper(raw, warn, fail)
        reason = (
            f"corpus WER {raw:.3f} of the agent's transcription (heard_text) over {len(pairs)} user turns; "
            f"good <= {_g(warn)}, warning <= {_g(fail)}"
        )
        return self.labelled(raw, label, reason, warn=warn, fail=fail)


@register_metric
class HarnessRoundTripWER(_DeterministicVoiceMetric):
    """VOICE-011: Impulse's STT of its own TTS audio; a harness health check, never an agent score."""

    metric_id = "VOICE-011"
    name = "Harness round-trip WER (sanity)"
    audio_only = True
    informational = True
    skip_when = "text mode"

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        records = voice_records(transcript)
        if records is None:
            return self.skipped(NO_VOICE_DATA)
        failed = _stt_error(records, "user", "round_trip_error")
        if failed is not None:
            return self.skipped(IMPULSE_STT_FAILED.format(error=failed))
        turns = transcript.turns
        pairs = []
        for record in records:
            number = record.get("turn_index")
            if record.get("speaker") != "user" or not isinstance(number, int) or not 1 <= number <= len(turns):
                continue
            hypothesis = record.get("round_trip_text")
            pairs.append((turns[number - 1].text, hypothesis if isinstance(hypothesis, str) else ""))
        if not pairs:
            return self.skipped("no user turn went through Impulse's TTS and STT")
        raw = corpus_wer(pairs)
        if raw is None:
            return self.skipped("the user turns sent to TTS hold no words")
        label = _label_upper(raw, ROUND_TRIP_WARN, ROUND_TRIP_FAIL)
        reason = (
            f"round-trip WER {raw:.3f} of Impulse's STT on its own TTS audio over {len(pairs)} user turns "
            f"(harness health check, informational); good <= {_g(ROUND_TRIP_WARN)}, warning <= {_g(ROUND_TRIP_FAIL)}"
        )
        if label == "poor":
            reason += "; audio metrics in this run are unreliable"
        return self.labelled(raw, label, reason, warn=ROUND_TRIP_WARN, fail=ROUND_TRIP_FAIL)


@register_metric
class DomainTermErrorRate(_DeterministicVoiceMetric):
    metric_id = "VOICE-013"
    name = "Domain-term error rate"
    audio_only = True
    needs_agent_audio = True
    skip_when = (
        "text mode; the agent returned no audio; no `key_terms` or `heard_text` is None; "
        "the user never said any of the `key_terms`"
    )

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        values = thresholds(ctx)
        warn, fail = _threshold(values, "wer_warn"), _threshold(values, "wer_fail")
        terms: dict[tuple[str, ...], str] = {}
        for term in scenario.key_terms:
            tokens = tuple(_tokens(term))
            if tokens and tokens not in terms:
                terms[tokens] = term
        if not terms:
            return self.skipped("scenario has no key_terms")
        heard_turns = [turn for turn in transcript.turns if turn.speaker == "user" and turn.heard_text is not None]
        if not heard_turns:
            return self.skipped(NO_HEARD_TEXT)
        said = 0
        missing: list[str] = []
        for turn in heard_turns:
            spoken, heard = _tokens(turn.text), _tokens(turn.heard_text)
            for tokens, term in terms.items():
                if _contains_sequence(spoken, tokens):
                    said += 1
                    if not _contains_sequence(heard, tokens):
                        missing.append(term)
        if said == 0:
            return self.skipped("the user never said any of the key_terms")
        raw = len(missing) / said
        label = _label_upper(raw, warn, fail)
        reason = f"{len(missing)} of {said} key terms the user said are missing from the agent's transcription"
        if missing:
            reason += ": " + ", ".join(dict.fromkeys(missing))
        reason += f"; good <= {_g(warn)}, warning <= {_g(fail)}"
        return self.labelled(raw, label, reason, warn=warn, fail=fail)


@register_metric
class SpeechRate(_DeterministicVoiceMetric):
    metric_id = "VOICE-015"
    name = "Speech rate"
    audio_only = True
    needs_agent_audio = True
    skip_when = "text mode; the agent returned no audio; the STT provider returned no word timings"

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        low, high = _speech_rate_range(thresholds(ctx))
        records = _agent_records(transcript)
        if records is None:
            return self.skipped(NO_VOICE_DATA)
        with_audio = [record for record in records.values() if _has_audio(record)]
        if not with_audio:
            return self.skipped(NO_AGENT_AUDIO)
        failed = _stt_error(with_audio, "agent", "stt_error")
        if failed is not None:   # one agent turn without word timings would bias the rate
            return self.skipped(IMPULSE_STT_FAILED.format(error=failed))
        total_words = 0
        speaking_ms = 0.0
        for record in with_audio:
            spans = []
            for word in record.get("words") or []:
                if not isinstance(word, Mapping):
                    continue
                start, end = _number(word.get("start_ms"), math.nan), _number(word.get("end_ms"), math.nan)
                if math.isfinite(start) and math.isfinite(end):
                    spans.append((start, end))
            if not spans:
                continue
            total_words += len(spans)
            speaking_ms += max(0.0, max(end for _, end in spans) - min(start for start, _ in spans))
        if total_words == 0:
            return self.skipped(NO_WORD_TIMINGS)
        if speaking_ms <= 0:
            return self.skipped("the STT word timings give the agent no speaking time")
        raw = total_words / (speaking_ms / 60000.0)
        if low <= raw <= high:
            label = "good"
        elif low - SPEECH_RATE_BAND <= raw <= high + SPEECH_RATE_BAND:
            label = "warning"
        else:
            label = "poor"
        # The thresholds used are those of the side of the range the value falls on.
        warn, fail = (low, low - SPEECH_RATE_BAND) if raw < low else (high, high + SPEECH_RATE_BAND)
        reason = (
            f"{raw:.1f} words per minute ({total_words} words in {speaking_ms / 1000.0:.1f} s of agent speech); "
            f"good {_g(low)}-{_g(high)} wpm, warning within {_g(SPEECH_RATE_BAND)} wpm outside it"
        )
        return self.labelled(raw, label, reason, warn=warn, fail=fail)


@register_metric
class SpokenLengthFit(_DeterministicVoiceMetric):
    metric_id = "VOICE-029"
    name = "Spoken-length fit"
    requires_audio = False
    skip_when = "no agent turns"

    def compute(self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext) -> MetricResult:
        counts = [
            (number, len(turn.text.split()))
            for number, turn in enumerate(transcript.turns, start=1)
            if turn.speaker == "agent"
        ]
        if not counts:
            return self.skipped("no agent turns")
        raw = _percentile([count for _, count in counts], 50)
        longest_turn, longest = max(counts, key=lambda item: item[1])
        if raw <= LENGTH_GOOD[0] and longest <= LENGTH_GOOD[1]:
            label = "good"
        elif raw <= LENGTH_WARNING[0] and longest <= LENGTH_WARNING[1]:
            label = "warning"
        else:
            label = "poor"
        reason = (
            f"median {raw:g} words per agent turn, max {longest} words (turn {longest_turn}) over {len(counts)} "
            f"agent turns; good: median <= {LENGTH_GOOD[0]} and max <= {LENGTH_GOOD[1]}, warning: median <= "
            f"{LENGTH_WARNING[0]} and max <= {LENGTH_WARNING[1]}"
        )
        return self.labelled(
            raw, label, reason, warn=float(LENGTH_GOOD[0]), fail=float(LENGTH_WARNING[0]), turn_index=longest_turn
        )


# ------------------------------------------------------------------------------------------------
# Judge-based voice metrics (METRICS.md section 2.2)
# ------------------------------------------------------------------------------------------------


@register_metric
class InterruptionHandling(JudgeMetric):
    metric_id = "VOICE-007"
    name = "Interruption handling"
    category = "voice"
    tier = "fast"
    criterion = (
        "For turns where the user interrupted the agent, did the agent stop, acknowledge, and address what "
        "the user said, rather than talking over or ignoring it?"
    )
    skip_when = "no turn has `interrupted=True`"
    skip_in_code = True

    def code_skip_reason(
        self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext
    ) -> Optional[str]:
        if not any(turn.interrupted for turn in transcript.turns):
            return "no turn has interrupted=True"
        return None


@register_metric
class ClarifyingQuestions(JudgeMetric):
    metric_id = "VOICE-027"
    name = "Clarifying questions"
    category = "voice"
    tier = "primary"
    criterion = (
        "When the user's request was ambiguous or missing needed details, did the agent ask a relevant, "
        "concise clarifying question instead of guessing or stalling? Penalize unnecessary questions when "
        "the answer was clear."
    )
    skip_when = "request was never ambiguous"


@register_metric
class CodeSwitchHandling(JudgeMetric):
    metric_id = "VOICE-031"
    name = "Code-switch handling"
    category = "voice"
    tier = "fast"
    criterion = (
        "When the user mixed languages (for example Hindi and English), did the agent understand and reply "
        "naturally for that mix?"
    )
    skip_when = "`Scenario.code_switching` is false"
    skip_in_code = True

    def code_skip_reason(
        self, transcript: CallTranscript, scenario: Scenario, ctx: MetricContext
    ) -> Optional[str]:
        if not scenario.code_switching:
            return "scenario.code_switching is false"
        return None


@register_metric
class TurnsToClose(JudgeMetric):
    """VOICE-035: the judge returns the ``turn_index`` of the first agent turn at which the outcome was
    reached (null if never). Reached = a ``pass`` verdict whose ``turn_index`` is a 1-based agent
    turn; ``raw_value`` = user turns up to and including that turn, labelled against ``max_turns``
    (good <= 60 %, warning <= 90 %); never reached (``fail``): ``max_turns``, poor. A ``pass``
    without a valid agent-turn index is skipped (SPEC.md section 13)."""

    metric_id = "VOICE-035"
    name = "Turns to close"
    category = "voice"
    tier = "fast"
    criterion = (
        "Exception to the pass/fail mapping: the judge returns the `turn_index` of the first agent turn at "
        "which the expected outcome (or the UNIV-001 outcome) was achieved, or null if never. `raw_value` = "
        "number of USER turns up to and including that turn (never achieved: `max_turns`, label poor). Label: "
        "good if ≤ 60% of `max_turns`; warning if ≤ 90%; poor otherwise. Runs identically in text mode (not a proxy)."
    )
    skip_when = "never"

    def from_turns(self, verdict: JudgeVerdict, transcript: CallTranscript, scenario: Scenario) -> MetricResult:
        """The outcome counts as reached for a ``pass`` verdict whose ``turn_index`` is the 1-based
        position of an agent turn in ``transcript.turns``. A ``pass`` without one (None, out of
        range, 0 included, or a user turn) is skipped with ``PASS_WITHOUT_TURN_INDEX``: the judge
        said the outcome was reached but not where. A ``fail`` (any index) means never reached. A
        judge skip (or no verdict) is skipped."""
        if verdict.result == "skip":
            return self.skipped(verdict.reason, turn_index=verdict.turn_index)
        max_turns = max(1, int(scenario.max_turns))
        turn_index = verdict.turn_index
        if verdict.result == "pass":
            valid = (
                isinstance(turn_index, int)
                and not isinstance(turn_index, bool)
                and 1 <= turn_index <= len(transcript.turns)
                and transcript.turns[turn_index - 1].speaker == "agent"
            )
            if not valid:
                return self.skipped(PASS_WITHOUT_TURN_INDEX, turn_index=None)
            user_turns = sum(1 for turn in transcript.turns[:turn_index] if turn.speaker == "user")
            if user_turns * 10 <= 6 * max_turns:
                label = "good"
            elif user_turns * 10 <= 9 * max_turns:
                label = "warning"
            else:
                label = "poor"
            note = f"outcome reached at turn {turn_index}, after {user_turns} of max_turns {max_turns} user turns"
            raw = float(user_turns)
        else:
            label = "poor"
            raw = float(max_turns)
            note = f"outcome never reached; counted as max_turns ({max_turns}) user turns"
            turn_index = None
        return self.result(
            score=LABEL_SCORES[label],
            label=label,
            reason=f"{note}. {verdict.reason}",
            raw_value=raw,
            turn_index=turn_index,
        )

    async def evaluate(
        self,
        transcript: CallTranscript,
        scenario: Scenario,
        judge: Judge,
        ctx: Optional[MetricContext] = None,
    ) -> MetricResult:
        ctx = _ctx(ctx)
        if ctx.dry_run:
            return self.skipped("dry run")
        verdict = await judge.evaluate(transcript, self.full_criterion(scenario, ctx), self.tier or "fast")
        return self.from_turns(verdict, transcript, scenario)


@register_metric
class VoiceNaturalness(JudgeMetric):
    metric_id = "VOICE-037"
    name = "Voice naturalness"
    category = "voice"
    tier = "fast"
    criterion = (
        "Would the agent's replies sound natural when SPOKEN aloud? Penalize markdown formatting, bullet "
        "lists, URLs or long numbers read out verbatim, stiff or robotic phrasing, and run-on sentences. A "
        "pass means the output is suitable for spoken delivery without editing."
    )
    skip_when = "never"


@register_metric
class Responsiveness(JudgeMetric):
    metric_id = "VOICE-039"
    name = "Responsiveness"
    category = "voice"
    tier = "fast"
    criterion = (
        "Does each agent turn directly respond to the user's immediately preceding utterance, rather than "
        "ignoring it, repeating itself from a previous turn, or answering an earlier question? Penalize "
        "non-sequiturs, scripted pivots that ignore what the user just said, and copy-pasted "
        "acknowledgements."
    )
    skip_when = "never"
