# SPDX-License-Identifier: Apache-2.0
"""Scenario generator and domain guard (SPEC.md section 11).

The primary model reads the agent's system prompt and writes test scenarios as YAML. The answer is
validated (``Scenario`` with pydantic, then the ``config.yaml`` scenario rules); an invalid answer is
retried ONCE with the error appended to the request, and a second invalid answer is a
``GenerationError`` (``impulse generate`` exits 1).

Fields the generator sets itself, never taken from the LLM: ``language`` (``--language``), ``domain``
(``--domain``; an unknown domain is ``general``), ``assertions`` (the domain table, wildcards already
expanded by ``domains.py``; ``trai_dlt`` only for outbound fsi / telecom scenarios; the India packs
only when ``--language`` has the region IN, so ``en-US`` gets ``assertions: []``; never ``hipaa``),
``inject_events: []``, ``seed`` (omitted, i.e. null), and ``max_turns`` when the LLM gives none (12 for
fsi and healthcare, otherwise 9). ``metrics`` = the LLM's IDs that are in the mode's list, then (voice
mode) VOICE-001, VOICE-003, VOICE-005, then the domain defaults, without duplicates; it is always an
explicit, non-empty list.

LiteLLM is called through the module (``litellm.acompletion``) so tests can patch it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, TypeVar

import yaml
from pydantic import ValidationError

from impulse_evals.config import ConfigError, check_scenarios, yaml_error_text
from impulse_evals.domains import (
    GENERAL,
    detect_domain,
    get_domain_defaults,
    keyword_hits,
    resolve_domain,
)
from impulse_evals.models import Scenario

__all__ = [
    "DEFAULT_COUNT",
    "GENERATION_PROMPT_TEMPLATE",
    "GUARD_QUESTION",
    "TEXT_MODE_METRICS",
    "VOICE_MODE_ALWAYS",
    "VOICE_MODE_METRICS",
    "GenerationError",
    "InvalidAnswer",
    "ModelCallError",
    "allocate_mix",
    "build_generation_prompt",
    "complete_with_retry",
    "domain_mismatch",
    "domain_mismatch_warning",
    "generate_scenarios",
    "header_comment",
    "load_answer_yaml",
    "mode_metrics",
    "parse_scenarios",
    "redact_values",
    "scenario_mapping",
    "scenarios_yaml",
    "select_metrics_for_scenario",
    "strip_fence",
]

T = TypeVar("T")

DEFAULT_COUNT = 11

# ------------------------------------------------------------------------------------------------
# Exact texts (SPEC.md section 11)
# ------------------------------------------------------------------------------------------------

GENERATION_PROMPT_TEMPLATE = """\
You are a QA engineer for AI voice agents. Read this agent's system prompt carefully and generate
{count} realistic test scenarios that comprehensively test it.

Agent system prompt:
{system_prompt}

Domain: {domain}   Language: {language}

Scenarios must match what this specific agent is supposed to do, cover happy paths, edge cases and
failure modes, be realistic for real users, and use what users actually say in {language}.
Mix: {mix_happy} happy path, {mix_edge} edge cases (confused, impatient, code-switching),
{mix_adversarial} adversarial (user tries to make the agent do the wrong thing),
{mix_hallucination} hallucination traps (agent likely to invent facts).
For each scenario fill: name, persona, opening_user_message, expected_outcome, key_terms,
code_switching, max_turns, direction, and choose metrics from the list below.
{metric_list}
Output YAML only. No explanation. No preamble."""

TEXT_METRIC_LIST = (
    "UNIV-001, UNIV-003, UNIV-005, UNIV-007, UNIV-009, UNIV-011, UNIV-013,\n"
    "UNIV-015, UNIV-017, UNIV-019, UNIV-021, UNIV-023,\n"
    "VOICE-027, VOICE-029, VOICE-035, VOICE-037, VOICE-039"
)

VOICE_METRIC_LIST = (
    "UNIV-001, UNIV-003, UNIV-005, UNIV-007, UNIV-009, UNIV-011, UNIV-013,\n"
    "UNIV-015, UNIV-017, UNIV-019, UNIV-021, UNIV-023,\n"
    "VOICE-001, VOICE-003, VOICE-005, VOICE-007, VOICE-009, VOICE-013, VOICE-015,\n"
    "VOICE-027, VOICE-029, VOICE-031, VOICE-035, VOICE-037, VOICE-039"
)

_METRIC_ID_RE = re.compile(r"\b(?:UNIV|VOICE|CHAT)-\d{3}\b")

TEXT_MODE_METRICS: tuple[str, ...] = tuple(_METRIC_ID_RE.findall(TEXT_METRIC_LIST))
VOICE_MODE_METRICS: tuple[str, ...] = tuple(_METRIC_ID_RE.findall(VOICE_METRIC_LIST))
# Voice mode adds these deterministic audio metrics to every generated scenario.
VOICE_MODE_ALWAYS: tuple[str, ...] = ("VOICE-001", "VOICE-003", "VOICE-005")

# Domain guard question; typer.confirm(..., default=False) shows it as "Continue with --domain {domain}? [y/N]".
GUARD_QUESTION = "Continue with --domain {domain}?"

# Appended to the request for the one retry (SPEC.md section 11: "retry once with the error appended").
RETRY_NOTE = (
    "Your previous answer could not be used: {error}\n"
    "Fix this and answer again with the complete YAML list of scenarios. "
    "Output YAML only. No explanation. No preamble."
)
_MAX_ERROR_CHARS = 2000

# largest-remainder weights in percent: happy, edge, adversarial, hallucination (also the tie order)
MIX_WEIGHTS: tuple[int, int, int, int] = (30, 25, 25, 20)


class InvalidAnswer(ValueError):
    """An LLM answer that cannot be used (not YAML, not a scenario list, or a scenario that does not
    validate). The message is what the retry request appends."""


class GenerationError(Exception):
    """Generation failed: the answer was still invalid after the one retry (exit code 1)."""


class ModelCallError(Exception):
    """The LLM call itself raised (connection, authentication, rate limit, ...): a generation failure
    (exit code 1), never retried. ``error`` is the original exception (also the ``__cause__``). Errors
    raised outside the call (e.g. while reading the answer) are not wrapped and propagate unchanged."""

    def __init__(self, error: Exception) -> None:
        super().__init__(error)
        self.error = error


# ------------------------------------------------------------------------------------------------
# Mix (SPEC.md section 11)
# ------------------------------------------------------------------------------------------------


def allocate_mix(count: int) -> tuple[int, int, int, int]:
    """Split ``count`` scenarios 30/25/25/20 into (happy, edge, adversarial, hallucination).

    Largest-remainder rounding in integer arithmetic; ties go to the earlier category (happy, edge,
    adversarial, hallucination), so the result is deterministic. ``allocate_mix(11) == (3, 3, 3, 2)``.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an int")
    if count < 0:
        raise ValueError("count must not be negative")
    floors = [count * weight // 100 for weight in MIX_WEIGHTS]
    remainders = [count * weight % 100 for weight in MIX_WEIGHTS]
    left = count - sum(floors)
    for index in sorted(range(len(MIX_WEIGHTS)), key=lambda i: (-remainders[i], i))[:left]:
        floors[index] += 1
    happy, edge, adversarial, hallucination = floors
    return happy, edge, adversarial, hallucination


# ------------------------------------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------------------------------------


def mode_metrics(voice_mode: bool) -> tuple[str, ...]:
    """The metric IDs the generator may use: the voice-mode list or the text-mode list."""
    return VOICE_MODE_METRICS if voice_mode else TEXT_MODE_METRICS


def build_generation_prompt(system_prompt: str, *, domain: str, language: str, count: int, voice_mode: bool) -> str:
    """The SPEC.md section 11 generation prompt (the system prompt is inserted verbatim)."""
    happy, edge, adversarial, hallucination = allocate_mix(count)
    return GENERATION_PROMPT_TEMPLATE.format(
        count=count,
        system_prompt=system_prompt,
        domain=domain,
        language=language,
        mix_happy=happy,
        mix_edge=edge,
        mix_adversarial=adversarial,
        mix_hallucination=hallucination,
        metric_list=VOICE_METRIC_LIST if voice_mode else TEXT_METRIC_LIST,
    )


# ------------------------------------------------------------------------------------------------
# Domain guard (SPEC.md section 11)
# ------------------------------------------------------------------------------------------------


def domain_mismatch(system_prompt: str, domain: str) -> Optional[str]:
    """The domain detected in ``system_prompt`` when it differs from ``domain``, else None.

    Never a mismatch for ``--domain general`` or an unknown domain (treated as general), or when no
    domain is detected (fewer than 3 distinct keyword hits).
    """
    resolved, _ = resolve_domain(domain)
    if resolved == GENERAL:
        return None
    detected = detect_domain(system_prompt)
    if detected is None or detected == resolved:
        return None
    return detected


def domain_mismatch_warning(system_prompt: str, detected: str, domain: str) -> str:
    """The warning printed for a domain-guard mismatch (it names the detected domain)."""
    hits = keyword_hits(system_prompt).get(detected, 0)
    return (
        f"Warning: the system prompt looks like the {detected} domain ({hits} distinct {detected} keywords), "
        f"but --domain is {domain}."
    )


# ------------------------------------------------------------------------------------------------
# Answer validation and the fields the generator sets itself
# ------------------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)(?:```|\Z)", re.DOTALL)
_GENERATOR_FIELDS = ("language", "domain", "assertions", "inject_events", "seed")


def load_answer_yaml(body: str) -> Any:
    """``yaml.safe_load(body)`` of an LLM answer (also used by ``impulse convert-failure``). Every
    loading failure is an ``InvalidAnswer`` (one retry, then exit 1): a YAML error (problem and position
    only: PyYAML's text quotes, and cuts, the source), a ``ValueError`` from a value PyYAML cannot build
    (e.g. the timestamp ``2024-02-30``, or ``!!int abc``, whose message quotes the value: "invalid
    literal for int() with base 10: 'abc'"; that text goes only into the retry note sent to the model
    and into the ``GenerationError`` message, which the CLI redacts and shortens before printing it),
    and a ``RecursionError`` from very deep nesting."""
    try:
        return yaml.safe_load(body)
    except yaml.YAMLError as exc:
        raise InvalidAnswer(f"the answer is not valid YAML: {yaml_error_text(exc)}") from None
    except ValueError as exc:
        raise InvalidAnswer(f"the answer is not valid YAML: a value cannot be loaded ({exc})") from None
    except RecursionError:
        raise InvalidAnswer("the answer is not valid YAML: it is nested too deeply") from None


def _is_yaml_collection(text: str) -> bool:
    """True when ``text`` parses by itself as a YAML list or mapping."""
    try:
        data = yaml.safe_load(text)
    except (yaml.YAMLError, ValueError, RecursionError):
        return False
    return isinstance(data, (list, Mapping))


def strip_fence(text: str) -> str:
    """The YAML body of an answer (also used by ``impulse convert-failure``): when the stripped answer
    starts with ```` ``` ````, the content of that fence; otherwise the answer itself when it parses as a
    YAML list or mapping (so a fenced snippet inside a value, such as a persona quoting an error log,
    is kept as text); otherwise the content of the first fenced block (an answer with a preamble line),
    or the answer itself when it has none."""
    if not text.strip().startswith("```") and _is_yaml_collection(text):
        return text
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def _llm_metric_ids(value: Any) -> list[str]:
    """Metric IDs found (with the metric-ID pattern) in the LLM's ``metrics`` value: a string, or each
    string item of a list (``"UNIV-001 (Task completion)"`` gives UNIV-001); anything else gives none."""
    if isinstance(value, str):
        return _METRIC_ID_RE.findall(value)
    if isinstance(value, (list, tuple)):
        return [metric_id for item in value if isinstance(item, str) for metric_id in _METRIC_ID_RE.findall(item)]
    return []


def select_metrics_for_scenario(llm_metrics: Any, domain: str, voice_mode: bool) -> list[str]:
    """The LLM's metric IDs that are in the mode's list, then (voice mode) VOICE-001/003/005, then the
    domain defaults; first occurrence kept. Never empty (every domain has default metrics)."""
    allowed = set(mode_metrics(voice_mode))
    chosen = [metric_id for metric_id in _llm_metric_ids(llm_metrics) if metric_id in allowed]
    if voice_mode:
        chosen.extend(VOICE_MODE_ALWAYS)
    chosen.extend(get_domain_defaults(domain).metrics)
    return list(dict.fromkeys(chosen))


def _error_location(loc: Sequence[Any]) -> str:
    path = ""
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
    return path


def _finalize(item: Mapping[str, Any], *, domain: str, language: str, voice_mode: bool) -> Scenario:
    """One LLM scenario mapping with the generator's own fields set, validated as a ``Scenario``."""
    defaults = get_domain_defaults(domain)
    data = {key: value for key, value in item.items() if key not in _GENERATOR_FIELDS}
    data["language"] = language
    data["domain"] = domain
    data["inject_events"] = []
    data["assertions"] = []
    data["metrics"] = select_metrics_for_scenario(item.get("metrics"), domain, voice_mode)
    if data.get("max_turns") is None:
        data["max_turns"] = defaults.max_turns
    scenario = Scenario.model_validate(data)
    assertions = defaults.assertions_for(scenario.direction, language=language)
    return scenario.model_copy(update={"assertions": assertions, "seed": None})


def parse_scenarios(text: Optional[str], *, domain: str, language: str, voice_mode: bool) -> list[Scenario]:
    """Validated scenarios from one LLM answer, with the generator's own fields set.

    The answer is a YAML list of scenario mappings (a fenced block, or a mapping whose ``scenarios``
    key holds the list, is accepted too). Raises ``InvalidAnswer`` with the error to append to the
    retry request: empty answer, invalid YAML, no list, an item that is not a mapping, a pydantic
    validation error, or a ``config.yaml`` scenario rule (e.g. duplicate names).
    """
    domain = resolve_domain(domain)[0]
    body = strip_fence((text or "").strip()).strip()
    if not body:
        raise InvalidAnswer("the answer was empty")
    data = load_answer_yaml(body)
    if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list):
        data = data["scenarios"]
    if not isinstance(data, list):
        raise InvalidAnswer("the answer must be a YAML list of scenarios, one mapping per scenario")
    if not data:
        raise InvalidAnswer("the answer holds no scenarios")

    errors: list[str] = []
    scenarios: list[Scenario] = []
    for index, item in enumerate(data):
        prefix = f"scenarios[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{prefix}: each scenario must be a mapping of fields")
            continue
        try:
            scenarios.append(_finalize(item, domain=domain, language=language, voice_mode=voice_mode))
        except ValidationError as exc:
            for error in exc.errors(include_url=False, include_context=False, include_input=False):
                location = _error_location(error["loc"])
                errors.append(f"{prefix}.{location}: {error['msg']}" if location else f"{prefix}: {error['msg']}")
    if errors:
        raise InvalidAnswer("; ".join(errors))
    try:
        check_scenarios(scenarios)
    except ConfigError as exc:
        raise InvalidAnswer(str(exc)) from None
    return scenarios


# ------------------------------------------------------------------------------------------------
# LLM call with one retry
# ------------------------------------------------------------------------------------------------


def _message_content(response: Any) -> Optional[str]:
    """``response.choices[0].message.content``; None when missing (a malformed or tool-call-only answer)."""
    try:
        raw = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    if isinstance(raw, list):  # content parts
        raw = "".join(str(part.get("text") or "") for part in raw if isinstance(part, Mapping))
    return raw if isinstance(raw, str) or raw is None else str(raw)


async def _complete(model: str, prompt: str, options: Mapping[str, Any]) -> str:
    import litellm  # lazy: heavy import; called through the module so tests can patch it

    litellm.suppress_debug_info = True   # no "Give Feedback / Get Help" banner (AUDIT.md finding 43)
    from impulse_evals.judge.litellm_judge import completion_kwargs

    # As for the simulator, only model, messages, and stream are fixed: a configured temperature is used.
    kwargs = completion_kwargs(options, fixed=("model", "messages", "stream"))
    try:
        response = await litellm.acompletion(model=model, messages=[{"role": "user", "content": prompt}], **kwargs)
    except Exception as exc:  # noqa: BLE001 - LiteLLM and provider SDKs raise many types; only the call is wrapped
        raise ModelCallError(exc) from exc
    return (_message_content(response) or "").strip()


def _retry_prompt(prompt: str, error: str, note: str = RETRY_NOTE) -> str:
    error = " ".join(error.split())
    if len(error) > _MAX_ERROR_CHARS:
        error = error[: _MAX_ERROR_CHARS - 3] + "..."
    return prompt + "\n\n" + note.format(error=error)


async def complete_with_retry(
    prompt: str,
    parse: Callable[[str], T],
    *,
    model: str,
    options: Optional[Mapping[str, Any]] = None,
    retry_note: str = RETRY_NOTE,
) -> T:
    """Ask ``model`` with ``prompt`` and return ``parse(answer)``. When ``parse`` raises
    ``InvalidAnswer``, ask once more with the request plus ``retry_note`` (its ``{error}`` filled with
    the error) appended; a second invalid answer raises ``GenerationError``. An exception raised by the
    LiteLLM call itself is raised as ``ModelCallError`` (no retry); any other exception propagates
    unchanged. ``impulse convert-failure`` reuses this rule with its own ``parse`` and note."""
    options = dict(options or {})
    answer = await _complete(model, prompt, options)
    try:
        return parse(answer)
    except InvalidAnswer as exc:
        first_error = str(exc)
    answer = await _complete(model, _retry_prompt(prompt, first_error, retry_note), options)
    try:
        return parse(answer)
    except InvalidAnswer as exc:
        raise GenerationError(f"the model's answer was still invalid after one retry: {exc}") from None


async def generate_scenarios(
    system_prompt: str,
    *,
    domain: str,
    language: str,
    count: int = DEFAULT_COUNT,
    voice_mode: bool = False,
    model: str,
    options: Optional[Mapping[str, Any]] = None,
) -> list[Scenario]:
    """Generate validated scenarios for ``system_prompt`` with the primary ``model`` (SPEC.md section 11).

    An unknown ``domain`` is treated as ``general`` (the caller prints the warning). Raises
    ``GenerationError`` when the answer is invalid after the one retry.
    """
    resolved = resolve_domain(domain)[0]
    prompt = build_generation_prompt(
        system_prompt, domain=resolved, language=language, count=count, voice_mode=voice_mode
    )

    def parse(answer: str) -> list[Scenario]:
        return parse_scenarios(answer, domain=resolved, language=language, voice_mode=voice_mode)

    return await complete_with_retry(prompt, parse, model=model, options=options)


# ------------------------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------------------------


def scenario_mapping(scenario: Scenario) -> dict[str, Any]:
    """A generated scenario as written to the YAML file: ``Scenario`` field order, ``seed`` omitted,
    unset optional fields and empty tool expectations left out; ``assertions``, ``metrics``,
    ``key_terms``, and ``inject_events`` always written."""
    data = scenario.model_dump(mode="json")
    data.pop("seed", None)
    for key in ("expected_tool_calls", "forbidden_tool_calls"):
        if not data.get(key):
            data.pop(key, None)
    return {key: value for key, value in data.items() if value is not None}


def redact_values(value: Any, redact: Optional[Callable[[str], str]]) -> Any:
    """``value`` with every string inside it (mapping values, list items, at any depth) passed through
    ``redact``; mapping keys and non-string values are kept. Applied BEFORE ``yaml.safe_dump``, never
    to the dumped text: the dumper then quotes a value such as ``[REDACTED] is my code`` so the file
    still loads."""
    if redact is None:
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {key: redact_values(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_values(item, redact) for item in value]
    return value


def header_comment(header: Optional[str], redact: Optional[Callable[[str], str]] = None) -> str:
    """``header`` as ``#`` comment lines (each line redacted first when ``redact`` is given)."""
    if not header:
        return ""
    lines = header.splitlines() if redact is None else [redact(line) for line in header.splitlines()]
    return "".join(f"# {line}\n" if line else "#\n" for line in lines)


def scenarios_yaml(
    scenarios: Sequence[Scenario],
    *,
    header: Optional[str] = None,
    redact: Optional[Callable[[str], str]] = None,
) -> str:
    """The YAML list written by ``impulse generate`` (it can be pasted under ``scenarios:``). With
    ``redact`` (the CLI's key redactor), every string value and header line passes through it first,
    so a key value the model repeated is written as ``[REDACTED]``."""
    body = yaml.safe_dump(
        [redact_values(scenario_mapping(scenario), redact) for scenario in scenarios],
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return header_comment(header, redact) + body
