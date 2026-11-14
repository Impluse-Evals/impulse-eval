# SPDX-License-Identifier: AGPL-3.0-only
"""Regression scenarios from failed calls (SPEC.md section 15; ``impulse convert-failure FILE``).

FILE is a stored ``ScenarioResult`` JSON (``.impulse/runs/<run_id>/scenarios/<name>.json``). The
primary model is asked, in ONE user message, with the SPEC.md section 15 prompt::

    Write one Scenario YAML that reproduces this failure: {failed checks with reasons}

    Transcript:
    {format_transcript(transcript)}

    Agent system prompt:
    {transcript.agent_system_prompt}

The failed checks are, one per line: every failed assertion (``{rule_id} ({severity}) failed:
{reason}``) and every critical assertion the judge could not evaluate (``result`` skip with
``no_verdict``, SPEC.md section 13: ``{rule_id} ({severity}) not evaluated: {reason}``), in assertion
order; every poor metric that is neither skipped nor informational (``{metric_id} ({name}) poor:
{reason}``); and the scenario error when there is one (``scenario error: {error}``); reasons are put
on one line. High / medium no-verdict skips and valid judge skips are not failed checks. A stored
result with none of these passed and has nothing to convert.

The answer must be one ``Scenario`` as YAML (a mapping; a one-item list is accepted too); it is
validated with pydantic and the ``config.yaml`` scenario rules. An invalid answer gets the generator's
validate-and-retry-once rule (``generator.complete_with_retry``): one retry with the error appended,
then ``GenerationError`` (exit 1). Fields set from the stored result, never taken from the model:
``language`` and ``accent`` (``transcript.language`` / ``transcript.accent``, overriding the answer;
an accent-less transcript gives no accent). Dropped from the answer, as ``impulse generate`` does:
``seed`` (None) and ``inject_events`` (``[]``), before the ``config.yaml`` rules run. Filled when the
answer leaves it out or blank: ``name`` (``{scenario_name}_regression``). The failed checks are added when
the answer does not select them: every failed (or not evaluated critical) assertion's rule ID to
``assertions``, and every failed
metric ID (except UNIV-025, which runs automatically with tool expectations) to ``metrics``; an empty
``metrics`` list (which runs UNIV-001 ... UNIV-023) that misses a failed metric becomes that universal
set plus the failed IDs, a non-empty list keeps the answer's IDs and gets the failed IDs appended. A
failed UNIV-025 with no ``expected_tool_calls`` in the answer gives a note (``regression_notes``). An
unknown ``domain`` becomes ``general`` with the warning "Unknown domain {domain}; using general
defaults." (as in ``load_config``).

The scenario is written to ``scenarios/regression/<name>_regression.yaml`` (relative to the current
directory; ``<name>`` is the stored ``scenario_name`` sanitized like run files and cut to at most 200
UTF-8 bytes) as a one-item YAML list
that can be pasted under ``scenarios:`` in a ``config.yaml``. The CLI passes its key redactor: every
string value of the scenario, the header comment lines, and the stored name in the file name are
redacted first, so a known key value is written as ``[REDACTED]``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import ValidationError

from impulse_evals.assertions.base import KNOWN_ASSERTION_IDS
from impulse_evals.config import ConfigError, check_scenarios
from impulse_evals.generator import (
    InvalidAnswer,
    complete_with_retry,
    header_comment,
    load_answer_yaml,
    redact_values,
    scenario_mapping,
    strip_fence,
)
from impulse_evals.metrics.base import DEFAULT_UNIVERSAL_METRICS, KNOWN_METRIC_IDS
from impulse_evals.models import Scenario, ScenarioResult
from impulse_evals.reporter import cut_file_stem, sanitize_name
from impulse_evals.transcriber import format_transcript

__all__ = [
    "PROMPT_START",
    "REGRESSION_DIR",
    "UNIV_025_NOTE",
    "FailedCheck",
    "InvalidResultFile",
    "build_regression_prompt",
    "convert_failure",
    "failed_checks",
    "load_scenario_result",
    "parse_regression_scenario",
    "regression_notes",
    "regression_path",
    "regression_yaml",
]

PROMPT_START = "Write one Scenario YAML that reproduces this failure: "
NO_SYSTEM_PROMPT = "Not provided."
REGRESSION_DIR = Path("scenarios") / "regression"
AUTOMATIC_METRICS = ("UNIV-025",)   # runs automatically when a scenario defines tool expectations
# Model-supplied fields that are dropped, as by impulse generate: seed stays None, inject_events [].
DROPPED_FIELDS = ("seed", "inject_events")
UNIV_025_NOTE = (
    "UNIV-025 failed in the stored run; add expected_tool_calls to the regression scenario so it is checked again."
)

# Appended to the request for the one retry (the generator's rule, with a note for one scenario).
RETRY_NOTE = (
    "Your previous answer could not be used: {error}\n"
    "Fix this and answer again with exactly one scenario as a YAML mapping of Scenario fields. "
    "Output YAML only. No explanation. No preamble."
)


class InvalidResultFile(ValueError):
    """FILE is missing, unreadable, or not a stored ``ScenarioResult`` (exit 2)."""


@dataclass(frozen=True)
class FailedCheck:
    check_id: str        # rule ID, metric ID, or "" for the scenario error
    kind: str            # "assertion" | "metric" | "error"
    line: str            # the line in the prompt


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def load_scenario_result(path: Union[str, os.PathLike[str]]) -> ScenarioResult:
    """The stored ``ScenarioResult`` at ``path``; raises ``InvalidResultFile`` with the problem."""
    target = Path(path)
    try:
        text = target.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        raise InvalidResultFile(f"{target}: file not found") from None
    except IsADirectoryError:
        raise InvalidResultFile(f"{target}: is a folder; give a stored scenario result file (scenarios/<name>.json)") from None
    except OSError as exc:
        raise InvalidResultFile(f"{target}: cannot read file: {exc.strerror or type(exc).__name__}") from None
    except UnicodeDecodeError:
        raise InvalidResultFile(f"{target}: file is not valid UTF-8") from None
    try:
        return ScenarioResult.model_validate_json(text)
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_context=False, include_input=False)[0]
        location = ".".join(str(part) for part in first["loc"]) or "(root)"
        raise InvalidResultFile(
            f"{target} is not a stored scenario result (.impulse/runs/<run_id>/scenarios/<name>.json): "
            f"{location}: {first['msg']}"
        ) from None


def failed_checks(result: ScenarioResult) -> list[FailedCheck]:
    """The failed checks of a stored result, in report order: failed assertions and critical
    assertions skipped with no verdict (in assertion order), poor metrics that are neither skipped nor
    informational, then the scenario error. Empty when the scenario passed."""
    checks: list[FailedCheck] = []
    for score in result.assertion_results:
        if score.result == "fail":
            line = f"{score.rule_id} ({score.severity}) failed: {_one_line(score.reason)}"
            checks.append(FailedCheck(score.rule_id, "assertion", line))
        elif score.result == "skip" and score.no_verdict is True and score.severity == "critical":
            # SPEC.md section 13: a critical check the judge could not evaluate fails the scenario.
            line = f"{score.rule_id} ({score.severity}) not evaluated: {_one_line(score.reason)}"
            checks.append(FailedCheck(score.rule_id, "assertion", line))
    for metric in result.metric_results:
        if metric.label == "poor" and not metric.skipped and not metric.informational:
            line = f"{metric.metric_id} ({_one_line(metric.name)}) poor: {_one_line(metric.reason)}"
            checks.append(FailedCheck(metric.metric_id, "metric", line))
    if result.error is not None:
        checks.append(FailedCheck("", "error", f"scenario error: {_one_line(result.error)}"))
    return checks


def build_regression_prompt(result: ScenarioResult) -> str:
    """The SPEC.md section 15 prompt for one stored result."""
    failures = "\n".join(check.line for check in failed_checks(result))
    system_prompt = result.transcript.agent_system_prompt
    return (
        PROMPT_START
        + failures
        + "\n\nTranscript:\n"
        + format_transcript(result.transcript)
        + "\n\nAgent system prompt:\n"
        + (system_prompt if system_prompt is not None else NO_SYSTEM_PROMPT)
    )


def _validation_errors(exc: ValidationError) -> str:
    """Every pydantic error as ``location: message`` (input values are never shown)."""
    parts = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ""
        for part in error["loc"]:
            location += f"[{part}]" if isinstance(part, int) else (f".{part}" if location else str(part))
        parts.append(f"{location}: {error['msg']}" if location else str(error["msg"]))
    return "; ".join(parts)


def _with_failed_checks(scenario: Scenario, result: ScenarioResult) -> Scenario:
    """Add the failed checks the answer does not select (see the module docstring)."""
    checks = failed_checks(result)
    assertions = list(scenario.assertions)
    for check in checks:
        if check.kind == "assertion" and check.check_id in KNOWN_ASSERTION_IDS and check.check_id not in assertions:
            assertions.append(check.check_id)
    metrics = list(scenario.metrics)
    selected = metrics or list(DEFAULT_UNIVERSAL_METRICS)   # an empty list runs UNIV-001 ... UNIV-023
    missing = [
        check.check_id
        for check in checks
        if check.kind == "metric"
        and check.check_id in KNOWN_METRIC_IDS
        and check.check_id not in AUTOMATIC_METRICS
        and check.check_id not in selected
    ]
    if missing:   # an empty list becomes the explicit universal set, so it keeps running those metrics
        metrics = list(dict.fromkeys([*selected, *missing]))
    return scenario.model_copy(update={"assertions": assertions, "metrics": metrics})


def regression_notes(scenario: Scenario, result: ScenarioResult) -> list[str]:
    """Notes to print about checks the regression scenario cannot run again by itself: a failed
    UNIV-025 (it runs only when a scenario defines tool expectations, and is never added to
    ``metrics``) when the scenario has no ``expected_tool_calls``."""
    failed = {check.check_id for check in failed_checks(result) if check.kind == "metric"}
    if "UNIV-025" in failed and not scenario.expected_tool_calls:
        return [UNIV_025_NOTE]
    return []


def parse_regression_scenario(
    text: Optional[str], result: ScenarioResult, warnings: Optional[list[str]] = None
) -> Scenario:
    """The one scenario in an LLM answer, validated; raises ``InvalidAnswer`` with the error to append
    to the retry request. On success the unknown-domain warning (the domain is rewritten to
    ``general``, as by ``load_config``) is appended to ``warnings`` when given."""
    body = strip_fence((text or "").strip()).strip()
    if not body:
        raise InvalidAnswer("the answer was empty")
    data = load_answer_yaml(body)   # YAML, ValueError and RecursionError failures: InvalidAnswer (retried)
    if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list):
        data = data["scenarios"]
    if isinstance(data, list):
        if len(data) != 1:
            raise InvalidAnswer(f"the answer must hold exactly one scenario, not {len(data)}")
        data = data[0]
    if not isinstance(data, Mapping):
        raise InvalidAnswer("the answer must be one scenario: a YAML mapping of Scenario fields")
    fields = {key: value for key, value in data.items() if key not in DROPPED_FIELDS}
    if not isinstance(fields.get("name"), str) or not fields["name"].strip():
        fields["name"] = f"{result.scenario_name}_regression"
    # The stored call's language and accent, never the model's (SPEC.md section 15).
    fields["language"] = result.transcript.language
    fields["accent"] = result.transcript.accent
    try:
        scenario = Scenario.model_validate(fields)
    except ValidationError as exc:
        raise InvalidAnswer(f"the scenario does not validate: {_validation_errors(exc)}") from None
    scenario = _with_failed_checks(scenario, result)
    try:
        found = check_scenarios([scenario])   # config.yaml rules; an unknown domain becomes general
    except ConfigError as exc:
        raise InvalidAnswer(str(exc).replace("scenarios[0].", "", 1)) from None
    if warnings is not None:
        warnings.extend(found)
    return scenario


async def convert_failure(
    result: ScenarioResult,
    *,
    model: str,
    options: Optional[Mapping[str, Any]] = None,
    warnings: Optional[list[str]] = None,
) -> Scenario:
    """Ask ``model`` for one scenario reproducing the failure of ``result`` (one retry on an invalid
    answer). Warnings about the accepted answer are appended to ``warnings`` when given. Raises
    ``generator.GenerationError`` (still invalid) or ``generator.ModelCallError``."""

    def parse(answer: str) -> Scenario:
        return parse_regression_scenario(answer, result, warnings)

    return await complete_with_retry(
        build_regression_prompt(result), parse, model=model, options=options, retry_note=RETRY_NOTE
    )


def regression_path(result: ScenarioResult, *, redact: Optional[Callable[[str], str]] = None) -> Path:
    """``scenarios/regression/<name>_regression.yaml`` (relative to the current directory). With
    ``redact``, the stored name is redacted before it is sanitized, so no key value is in the file name.
    The sanitized stem is cut to at most 200 UTF-8 bytes (``reporter.cut_file_stem``)."""
    name = result.scenario_name if redact is None else redact(result.scenario_name)
    stem = cut_file_stem(sanitize_name(name)) or "_"
    return REGRESSION_DIR / f"{stem}_regression.yaml"


def regression_yaml(
    scenario: Scenario,
    *,
    header: Optional[str] = None,
    redact: Optional[Callable[[str], str]] = None,
) -> str:
    """The one-item YAML list written to the regression file (pasteable under ``scenarios:``):
    ``Scenario`` field order, unset optional fields and empty tool expectations left out, no ``seed``
    (``parse_regression_scenario`` drops the model's); ``header`` lines become ``#`` comments. With
    ``redact`` (the CLI's key redactor), every string value and header line passes through it before
    ``yaml.safe_dump``."""
    data = redact_values(scenario_mapping(scenario), redact)
    body = yaml.safe_dump([data], sort_keys=False, allow_unicode=True, default_flow_style=False)
    return header_comment(header, redact) + body
