# SPDX-License-Identifier: AGPL-3.0-only
"""Judge calibration against human labels (SPEC.md section 15; ``impulse calibrate DIR``).

Input: a folder with one JSON file per item (every ``*.json`` file directly in the folder, read in
name order)::

    {"transcript": <CallTranscript>, "check": "UNIV-xxx" | "VOICE-xxx" | "CHAT-xxx" | "pack.rule_id",
     "human_label": "pass" | "fail"}

An item may also carry an optional ``"scenario"`` mapping of ``Scenario`` fields (for example
``expected_outcome``, ``direction``, ``code_switching``, ``max_turns``, ``expected_tool_calls``), so the
check sees the same scenario context it had in the run. Without it the scenario is built from the
transcript: ``name`` = ``scenario_name``, ``language`` and ``accent`` from the transcript, and the
``Scenario`` defaults for everything else (``direction: inbound``; an outbound call needs
``"scenario": {"direction": "outbound"}``, since an inbound agent that greets first also speaks first).

Each item is judged once with the normal path: a metric through ``get_metric(check).evaluate(...)``, a
pack rule through ``get_assertion(check).evaluate(...)``, with the normal tier routing. Only judge-based
checks can be calibrated: a deterministic metric (no judge) makes the item invalid. The judge's label is
the assertion result (``pass`` / ``fail``), or for a metric ``fail`` when its label is ``poor`` and
``pass`` otherwise (good, or warning, which never fails a scenario). Items the judge (or a code-decided
skip) returns as ``skip`` are left out of agreement and kappa and counted. Agreement rate is the share
of judged items where the judge's label equals the human label; Cohen's kappa is computed here in plain
Python (no third-party statistics library): 1.0 when observed and expected agreement are both 1.0, 0.0
when only expected agreement is 1.0.

Metric context: there is no ``config.yaml``, so ``agent_type`` is ``multimodal`` (CHAT metrics are
judged, never skipped as "chat metric; agent.type is voice"), ``voice_mode`` follows the transcript,
and ``agent.tools`` is unknown (None).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from pydantic import ValidationError

from impulse_evals.assertions.base import KNOWN_ASSERTION_IDS, Assertion, get_assertion
from impulse_evals.metrics.base import (
    KNOWN_METRIC_IDS,
    Metric,
    MetricContext,
    get_metric,
)
from impulse_evals.models import CallTranscript, MetricResult, Scenario, ScoreResult

if TYPE_CHECKING:  # annotation only
    from impulse_evals.judge.base import Judge

__all__ = [
    "CalibrationError",
    "CalibrationItem",
    "CalibrationResult",
    "InvalidItems",
    "ItemOutcome",
    "agreement_rate",
    "calibrate",
    "cohen_kappa",
    "load_items",
    "required_tiers",
]

Label = Literal["pass", "fail"]
HUMAN_LABELS: tuple[str, ...] = ("pass", "fail")
CALIBRATION_PERSONA = "calibration item (no persona: the transcript is already recorded)"


# ------------------------------------------------------------------------------------------------
# Agreement and Cohen's kappa
# ------------------------------------------------------------------------------------------------


def _check_pairs(a: Sequence[Any], b: Sequence[Any]) -> int:
    if len(a) != len(b):
        raise ValueError(f"label lists differ in length ({len(a)} and {len(b)})")
    if not a:
        raise ValueError("at least one pair of labels is needed")
    return len(a)


def agreement_rate(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Observed agreement: the share of positions where ``a`` and ``b`` hold the same label."""
    n = _check_pairs(a, b)
    return float(Fraction(sum(1 for x, y in zip(a, b) if x == y), n))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa of two equally long label lists (any hashable labels; categories = the labels of
    both lists). Exact arithmetic, so the result is symmetric in ``a`` and ``b``.

    kappa = (po - pe) / (1 - pe), with po the observed agreement and pe the agreement expected from
    each rater's label frequencies. When pe is 1.0 the formula is 0/0 or undefined: kappa is 1.0 when
    po is also 1.0, else 0.0 (SPEC.md section 15).
    """
    n = _check_pairs(a, b)
    po = Fraction(sum(1 for x, y in zip(a, b) if x == y), n)
    counts_a, counts_b = Counter(a), Counter(b)
    pe = sum(
        (Fraction(counts_a[label], n) * Fraction(counts_b[label], n) for label in counts_a.keys() & counts_b.keys()),
        Fraction(0),
    )
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return float((po - pe) / (1 - pe))


# ------------------------------------------------------------------------------------------------
# Item files
# ------------------------------------------------------------------------------------------------

Check = Union[Metric, Assertion]


@dataclass
class CalibrationItem:
    path: Path
    check_id: str
    check: Check
    human_label: Label
    transcript: CallTranscript
    scenario: Scenario

    @property
    def tier(self) -> str:
        """The judge tier the check routes to (``primary`` or ``fast``)."""
        return str(self.check.tier)


class InvalidItems(ValueError):
    """The folder or one or more item files are invalid (``impulse calibrate`` exits 2).
    ``problems`` holds one message per problem."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


class CalibrationError(Exception):
    """The judge call for an item raised (connection, authentication, rate limit, ...); ``item`` is the
    item and ``error`` the original exception (also the ``__cause__``)."""

    def __init__(self, item: CalibrationItem, error: Exception) -> None:
        super().__init__(f"{item.path.name} ({item.check_id}): {type(error).__name__}: {error}")
        self.item = item
        self.error = error


def _validation_text(exc: ValidationError) -> str:
    """The first pydantic error as ``location: message`` (input values are never shown)."""
    first = exc.errors(include_url=False, include_context=False, include_input=False)[0]
    location = ""
    for part in first["loc"]:
        location += f"[{part}]" if isinstance(part, int) else (f".{part}" if location else str(part))
    return f"{location}: {first['msg']}" if location else str(first["msg"])


def _load_check(check_id: str, cache: dict[str, Check]) -> Check:
    """The metric or pack rule ``check_id``; raises ``ValueError`` with the problem."""
    if check_id in cache:
        return cache[check_id]
    if check_id in KNOWN_METRIC_IDS:
        try:
            check: Check = get_metric(check_id)
        except Exception as exc:
            raise ValueError(f"check {check_id} cannot be loaded: {type(exc).__name__}: {exc}") from None
        if check.tier is None:
            raise ValueError(
                f"check {check_id} is a deterministic metric (computed by code, no judge); "
                "calibration measures the judge, so only judge-based checks can be calibrated"
            )
    elif check_id in KNOWN_ASSERTION_IDS:
        try:
            check = get_assertion(check_id)
        except Exception as exc:
            raise ValueError(f"check {check_id} cannot be loaded: {type(exc).__name__}: {exc}") from None
    else:
        raise ValueError(f"unknown check {check_id}; use a metric ID (UNIV-xxx, VOICE-xxx, CHAT-xxx) or pack.rule_id")
    cache[check_id] = check
    return check


def _scenario_for(transcript: CallTranscript, overrides: Any) -> Scenario:
    """The scenario the check is evaluated with (defaults from the transcript, then ``overrides``)."""
    # No direction default here: the Scenario default (inbound, SPEC.md section 7) applies unless the
    # item's "scenario" sets one. The first speaker is no guide: inbound agents often greet first.
    data: dict[str, Any] = {
        "name": transcript.scenario_name,
        "language": transcript.language,
        "accent": transcript.accent,
        "persona": CALIBRATION_PERSONA,
    }
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("scenario must be a mapping of Scenario fields")
        data.update(overrides)
    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"scenario is not a valid Scenario: {_validation_text(exc)}") from None


def _load_item(path: Path, cache: dict[str, Check]) -> CalibrationItem:
    """One item file; raises ``ValueError`` with the problem (without the file name)."""
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except OSError as exc:
        raise ValueError(f"cannot read file: {exc.strerror or type(exc).__name__}") from None
    except UnicodeDecodeError:
        raise ValueError("file is not valid UTF-8") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from None
    if not isinstance(data, dict):
        raise ValueError('the item must be a JSON object with "transcript", "check", and "human_label"')
    if "transcript" not in data:
        raise ValueError('missing "transcript" (a CallTranscript)')
    try:
        transcript = CallTranscript.model_validate(data["transcript"])
    except ValidationError as exc:
        raise ValueError(f"transcript is not a valid CallTranscript: {_validation_text(exc)}") from None
    check_id = data.get("check")
    if not isinstance(check_id, str) or not check_id.strip():
        raise ValueError('missing "check" (a metric ID or pack.rule_id)')
    check_id = check_id.strip()
    label = data.get("human_label")
    normalized = label.strip().lower() if isinstance(label, str) else None
    if normalized not in HUMAN_LABELS:
        # The whole value: the CLI redacts the message first and shortens it only after that, so a
        # key in the value cannot leak in part (SPEC.md section 8.1).
        raise ValueError(f'"human_label" must be "pass" or "fail", got {json.dumps(label, ensure_ascii=False)}')
    check = _load_check(check_id, cache)
    scenario = _scenario_for(transcript, data.get("scenario"))
    return CalibrationItem(
        path=path, check_id=check_id, check=check, human_label=normalized, transcript=transcript, scenario=scenario
    )


def load_items(directory: Union[str, Path]) -> list[CalibrationItem]:
    """Every ``*.json`` item file directly in ``directory``, in name order, validated before any judge
    call. Raises ``InvalidItems`` listing every problem: a missing folder, a path that is not a folder,
    no item files, or invalid item files."""
    folder = Path(directory)
    try:
        is_dir = folder.is_dir()
        exists = is_dir or folder.exists()
    except (OSError, ValueError):
        is_dir = exists = False
    if not exists:
        raise InvalidItems([f"{folder}: directory not found"])
    if not is_dir:
        raise InvalidItems([f"{folder}: not a directory (impulse calibrate takes a folder of item files)"])
    try:
        paths = sorted(
            (path for path in folder.iterdir() if path.suffix.lower() == ".json" and path.is_file()),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise InvalidItems([f"{folder}: cannot list the folder: {exc.strerror or type(exc).__name__}"]) from None
    if not paths:
        raise InvalidItems([f"{folder}: no *.json item files"])
    cache: dict[str, Check] = {}
    items: list[CalibrationItem] = []
    problems: list[str] = []
    for path in paths:
        try:
            items.append(_load_item(path, cache))
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")
    if problems:
        raise InvalidItems(problems)
    return items


def required_tiers(items: Iterable[CalibrationItem]) -> list[str]:
    """The judge tiers the items need (``primary`` first)."""
    tiers = {item.tier for item in items}
    return [tier for tier in ("primary", "fast") if tier in tiers]


# ------------------------------------------------------------------------------------------------
# Judging
# ------------------------------------------------------------------------------------------------


@dataclass
class ItemOutcome:
    item: CalibrationItem
    judge_label: Optional[Label]          # None: the judge (or a code-decided skip) returned skip
    reason: str
    no_verdict: bool = False

    @property
    def skipped(self) -> bool:
        return self.judge_label is None

    @property
    def agrees(self) -> Optional[bool]:
        return None if self.judge_label is None else self.judge_label == self.item.human_label


@dataclass
class CalibrationResult:
    outcomes: list[ItemOutcome] = field(default_factory=list)

    @property
    def item_count(self) -> int:
        """Every item read, skipped ones included."""
        return len(self.outcomes)

    @property
    def judged(self) -> list[ItemOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.skipped]

    @property
    def skipped_count(self) -> int:
        return self.item_count - len(self.judged)

    @property
    def agreeing_count(self) -> int:
        return sum(1 for outcome in self.judged if outcome.agrees)

    @property
    def agreement_rate(self) -> Optional[float]:
        """None when no item was judged."""
        judged = self.judged
        if not judged:
            return None
        return agreement_rate([o.item.human_label for o in judged], [o.judge_label for o in judged])

    @property
    def kappa(self) -> Optional[float]:
        """Cohen's kappa between the human and judge labels; None when no item was judged."""
        judged = self.judged
        if not judged:
            return None
        return cohen_kappa([o.item.human_label for o in judged], [str(o.judge_label) for o in judged])


def _context(item: CalibrationItem) -> MetricContext:
    return MetricContext(agent_type="multimodal", voice_mode=item.transcript.voice_mode)


def _outcome(item: CalibrationItem, result: Union[MetricResult, ScoreResult]) -> ItemOutcome:
    if isinstance(result, ScoreResult):
        label = None if result.result == "skip" else result.result
        return ItemOutcome(item, label, result.reason, no_verdict=result.no_verdict)
    if result.skipped:
        return ItemOutcome(item, None, result.reason)
    return ItemOutcome(item, "fail" if result.label == "poor" else "pass", result.reason)


async def judge_item(item: CalibrationItem, judge: Judge) -> ItemOutcome:
    """Judge one item with its check; raises ``CalibrationError`` when the evaluation raises."""
    try:
        if isinstance(item.check, Assertion):
            result: Union[MetricResult, ScoreResult] = await item.check.evaluate(item.transcript, item.scenario, judge)
        else:
            result = await item.check.evaluate(item.transcript, item.scenario, judge, _context(item))
    except Exception as exc:
        raise CalibrationError(item, exc) from exc
    return _outcome(item, result)


async def calibrate(items: Sequence[CalibrationItem], judge: Judge) -> CalibrationResult:
    """Judge every item, one after the other (one judge call per item unless a check skips in code)."""
    result = CalibrationResult()
    for item in items:
        result.outcomes.append(await judge_item(item, judge))
    return result
