# SPDX-License-Identifier: AGPL-3.0-only
"""Data models (SPEC.md section 7). Field names, types, and defaults are normative.

``AgentTool``, ``ToolCall``, ``MetricResult``, and ``ScoreResult`` are defined in the Apache-2.0
module ``impulse_evals.plugin_types`` and re-exported here (the same class objects), so
``from impulse_evals.models import ...`` keeps working (SPEC.md section 5.1 plugin API import rule).
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import PydanticCustomError

from impulse_evals.plugin_types import AgentTool as AgentTool
from impulse_evals.plugin_types import MetricResult as MetricResult
from impulse_evals.plugin_types import ScoreResult as ScoreResult
from impulse_evals.plugin_types import ToolCall as ToolCall


class Turn(BaseModel):
    speaker: Literal["user", "agent"]
    text: str
    timestamp_ms: int                     # when the turn started, relative to call start
    end_ms: Optional[int] = None
    audio_path: Optional[str] = None
    latency_ms: Optional[float] = None    # agent turns: time from end of user input to first agent output
    interrupted: bool = False
    stt_confidence: Optional[float] = None
    heard_text: Optional[str] = None      # what the AGENT says it heard (user turns), if the agent exposes it
    intent_confidence: Optional[float] = None
    classified_intent: Optional[str] = None
    tool_calls: list[ToolCall] = []       # tool calls made during this agent turn (usually agent turns only)


class CallTranscript(BaseModel):
    turns: list[Turn]
    duration_ms: int
    metadata: dict
    scenario_name: str
    language: str
    accent: Optional[str] = None
    voice_mode: bool = False
    recording_path: Optional[str] = None
    agent_system_prompt: Optional[str] = None


class ExpectedToolCall(BaseModel):
    """A tool call a scenario requires, checked by code (UNIV-025, METRICS.md section 1)."""

    name: str                             # tool name, exactly as the agent reports it
    required_arguments: list[str] = []    # keys that must be present and non-empty
    argument_values: dict = {}            # keys whose values must match exactly
    must_succeed: bool = True             # success must not be False (None = not captured, allowed)
    min_count: int = 1                    # 0 allowed only together with max_count
    max_count: Optional[int] = None       # e.g. 1 catches duplicate bookings or payments


class InjectEvent(BaseModel):
    """A synthetic event injected into a scenario at a specific user turn."""

    at_turn: int                          # inject BEFORE this user turn (1-indexed)
    type: Literal[
        "tool_timeout",                   # simulate a tool call timing out
        "tool_error",                     # simulate a tool call returning an error
        "silence",                        # inject N ms of silence from the user
        "transfer_fail",                  # simulate a transfer / handoff failing
        "user_change_of_mind",            # inject a topic/request change from the user
    ]
    tool: Optional[str] = None            # tool name for tool_timeout / tool_error
    error_message: Optional[str] = None   # message for tool_error
    silence_ms: Optional[int] = None      # ms of silence for type=silence
    note: Optional[str] = None            # free-text reason for this injection (appears in report)


class NoiseSettings(BaseModel):
    """Background noise mixed into the caller's audio (noise simulation, EXPERIMENTAL; SPEC.md section 14.4).

    ``voice_config.noise`` (the default for every scenario) and ``Scenario.noise`` (replaces it as a
    whole). ``profile`` and ``snr_db`` are checked by ``load_config`` with exact messages; unknown keys
    are errors.
    """

    model_config = ConfigDict(extra="forbid")

    profile: str                          # white | pink | brown | file (checked by load_config)
    snr_db: Optional[float] = None        # required, -5..40 (checked by load_config)
    file: Optional[str] = None            # profile: file only; relative to the config folder, stored resolved
    label: Optional[str] = None           # shown in the report; default: the profile, or the file's stem

    @field_validator("snr_db", mode="before")
    @classmethod
    def _reject_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):   # pydantic would read true as 1.0 dB
            raise PydanticCustomError("snr_db_boolean", "snr_db must be a number, not a boolean")
        return value


class Scenario(BaseModel):
    name: str
    language: str
    accent: Optional[str] = None
    domain: str = "general"
    direction: Literal["inbound", "outbound"] = "inbound"
    max_turns: int = 9                    # number of USER turns (not total turns). Compliance domains should use 12.
    persona: str
    opening_user_message: Optional[str] = None
    expected_outcome: Optional[str] = None
    key_terms: list[str] = []
    code_switching: bool = False
    assertions: list[str] = []            # "pack.rule_id"
    metrics: list[str] = []               # metric IDs; empty list = all universal metrics (hand-authored only)
    inject_events: list[InjectEvent] = []  # v1: noted in the report and console; execution reserved for v2
    seed: Optional[int] = None            # when set, inserted into the simulator prompt to encourage reproducibility
    expected_tool_calls: list[ExpectedToolCall] = []  # checked by code (UNIV-025)
    forbidden_tool_calls: list[str] = []  # tool names that must never be called in this scenario (UNIV-025)
    # Noise simulation (EXPERIMENTAL): None = voice_config.noise applies; False = no noise for this scenario.
    noise: Union[NoiseSettings, Literal[False], None] = None


class WordTiming(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    confidence: float


class FillerWord(BaseModel):
    word: str
    start_ms: int


class STTResult(BaseModel):
    text: str
    confidence: Optional[float] = None
    words: list[WordTiming] = []
    filler_words: list[FillerWord] = []


class LatencyProfile(BaseModel):
    p50_ms: float
    p75_ms: float
    p95_ms: float
    p99_ms: float
    all_latencies_ms: list[float]
    is_real_audio: bool = False


class CohortResult(BaseModel):
    cohort_key: str
    language: str
    accent: Optional[str] = None
    total: int
    passed: int
    failed: int
    pass_rate: float
    metric_means: dict[str, float]
    mean_stt_confidence: Optional[float] = None
    gap_vs_best: float
    gap_warning: bool
    stt_warning: bool = False


class ScenarioResult(BaseModel):
    scenario_name: str
    transcript: CallTranscript
    assertion_results: list[ScoreResult]
    metric_results: list[MetricResult]
    passed: bool
    error: Optional[str] = None
    inject_events: list[InjectEvent] = []  # copied from the Scenario so v2 can replay them


class RunResult(BaseModel):
    run_id: str
    timestamp: str
    agent_entrypoint: str
    agent_type: Literal["voice", "chat", "multimodal"]
    agent_domain: str
    judge_model: str
    voice_mode: bool = False
    tts_provider: Optional[str] = None
    stt_provider: Optional[str] = None
    total_scenarios: int
    passed: int
    failed: int
    overall_pass_rate: float
    scenario_results: list[ScenarioResult]
    cohorts: list[CohortResult]
    critical_failures: list[ScoreResult]
    metric_summary: dict[str, MetricResult]
    latency_profile: LatencyProfile
    warnings: list[str] = []              # run-level warnings shown in the report header (deduplicated, in first-seen order)


class ScenarioDiff(BaseModel):
    """Diff of one scenario across two runs."""

    scenario_name: str
    status_change: Literal["pass->fail", "fail->pass", "pass->pass", "fail->fail", "added", "removed"]
    metric_deltas: dict[str, float]       # metric_id -> score_b - score_a; omitted for added/removed
    assertion_changes: list[str]          # human-readable list of assertions that changed result
    notes: list[str]                      # e.g. "UNIV-023 newly failing", "VOICE-001 improved 0.3"


class CompareResult(BaseModel):
    """Output of impulse compare RUN_A RUN_B."""

    run_id_a: str
    run_id_b: str
    timestamp_a: str
    timestamp_b: str
    overall_pass_rate_delta: float        # run_b - run_a
    regressions: list[ScenarioDiff]       # pass->fail
    improvements: list[ScenarioDiff]      # fail->pass
    unchanged_pass: int
    unchanged_fail: int
    added_scenarios: list[str]            # in B but not in A
    removed_scenarios: list[str]          # in A but not in B
    metric_summary_deltas: dict[str, float]  # per metric ID, mean delta (b - a)
    critical_regressions: list[str]       # scenario names where a critical assertion newly failed or a scenario error is new in B
    check_regressions: list[ScenarioDiff] = []  # fail->fail scenarios with at least one newly failing check
    high_regressions: list[str] = []      # scenario names where a high assertion newly failed or UNIV-019 / UNIV-025 is newly poor


def compute_passed(scenario_result: ScenarioResult) -> bool:
    """Scenario pass logic (SPEC.md section 7).

    A scenario passes iff it has zero failed assertions of any severity, zero metrics labeled
    ``poor`` that are neither skipped nor informational, and no ``error``. Warnings never fail a
    scenario. The stored ``passed`` value is ignored: the result is computed from the lists.
    """
    if scenario_result.error is not None:
        return False
    if any(score.result == "fail" for score in scenario_result.assertion_results):
        return False
    return not any(
        metric.label == "poor" and not metric.skipped and not metric.informational
        for metric in scenario_result.metric_results
    )
