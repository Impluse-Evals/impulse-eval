# SPDX-License-Identifier: Apache-2.0
"""Phase 10: ``impulse convert-failure FILE`` (SPEC.md section 15).

FILE is a stored ``ScenarioResult`` JSON (``.impulse/runs/<run_id>/scenarios/<name>.json``). The
PRIMARY model is asked, with the prompt

    "Write one Scenario YAML that reproduces this failure: {failed checks with reasons}\\n\\n"
    "Transcript:\\n{format_transcript}\\n\\nAgent system prompt:\\n{system prompt}"

for one ``Scenario`` (validated; invalid YAML or a validation error is retried once with the error
appended, then exit 1). The result is written to ``scenarios/regression/<name>_regression.yaml``
(relative to the current directory; ``<name>`` is the stored ``scenario_name``, sanitized as in
SPEC.md section 15) as a one-item YAML list that can be pasted under ``scenarios:``. A FILE that
does not exist or is not a stored ``ScenarioResult`` is a usage error (exit 2).
"""

from pathlib import Path

import pytest
import yaml

PRIMARY = "ollama/judge-primary"
MODEL_ENV = {
    "IMPULSE_MODEL_PRIMARY": PRIMARY,
    "IMPULSE_MODEL_FAST": "ollama/judge-fast",
    "IMPULSE_MODEL_SIMULATOR": "ollama/simulator",
}
SYSTEM_PROMPT = (
    "You are the booking assistant of Sunrise Clinic. Ask for consent before collecting personal data.\n"
    "Book appointments with the book_slot tool and confirm only after it succeeds."
)
CONSENT = "dpdp.consent_before_pii"
CONSENT_REASON = "The agent asked for the phone number before asking for consent."
TOOL_REASON = "The agent said the slot was booked but book_slot was never called."
PASSED_REASON = "The tone was warm and polite throughout."
SKIPPED_REASON = "The agent made no factual claims at all."
PROMPT_START = "Write one Scenario YAML that reproduces this failure: "

LLM_SCENARIO = {
    "name": "clinic_booking",
    "language": "en-IN",
    "domain": "healthcare",
    "direction": "inbound",
    "max_turns": 6,
    "persona": "patient who gives the phone number before any consent is asked",
    "opening_user_message": "Hi, I need a slot on Tuesday and my number is 0000000000.",
    "expected_outcome": "the agent asks for consent before collecting personal data",
    "assertions": [CONSENT],
    "metrics": ["UNIV-023"],
}


@pytest.fixture
def project(impulse_project, mock_llm, monkeypatch):
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    return impulse_project


def _failed_result(run_builder, name="clinic_booking"):
    m = run_builder.models()
    turns = [
        run_builder.turn("user", "Hello, I need an appointment on Tuesday.", 0),
        run_builder.turn("agent", "Sure. What is your phone number?", 900, latency_ms=900.0),
        run_builder.turn("user", "It is 0000000000.", 3000),
        run_builder.turn(
            "agent",
            "Done, you are booked for Tuesday.",
            3900,
            latency_ms=900.0,
            tool_calls=[m.ToolCall(name="lookup_slot", arguments={"day": "Tuesday"}, success=True)],
        ),
    ]
    transcript = run_builder.transcript(name, turns, agent_system_prompt=SYSTEM_PROMPT)
    return run_builder.scenario_result(
        name,
        transcript=transcript,
        assertions=[(CONSENT, "fail", {"reason": CONSENT_REASON, "turn_index": 2})],
        metrics=[
            ("UNIV-023", 0.0, {"reason": TOOL_REASON, "turn_index": 4}),
            ("UNIV-013", 1.0, {"reason": PASSED_REASON}),
            ("UNIV-005", 0.0, {"skipped": True, "reason": SKIPPED_REASON}),
        ],
    )


def _store(root, scenario_result):
    folder = Path(root) / "stored_run" / "scenarios"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "failed_scenario.json"
    path.write_text(scenario_result.model_dump_json(indent=2), encoding="utf-8")
    return path


def _regression_file(root, stem):
    return Path(root) / "scenarios" / "regression" / f"{stem}_regression.yaml"


def _regression_files(root):
    folder = Path(root) / "scenarios" / "regression"
    return sorted(path.name for path in folder.glob("*")) if folder.is_dir() else []


def _one_item(path):
    from impulse_evals.models import Scenario

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 1, "a one-item YAML list"
    return data, Scenario.model_validate(data[0])


# ============================================================================================
# Happy path
# ============================================================================================


def test_convert_failure_writes_a_one_item_list(project, mock_llm, cli, run_builder):
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert _regression_files(project.root) == ["clinic_booking_regression.yaml"]
    data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.persona == LLM_SCENARIO["persona"]
    assert scenario.assertions == [CONSENT]
    assert mock_llm.call_count == 1
    assert mock_llm.calls[0]["model"] == PRIMARY


def test_regression_file_can_be_pasted_under_scenarios(project, mock_llm, cli, run_builder):
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    assert cli.invoke("convert-failure", stored).exit_code == 0
    data, _scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    cfg = project.load(project.write_config(data, path=project.root / "regression_config.yaml"))
    assert [scenario.name for scenario in cfg.scenarios] == [LLM_SCENARIO["name"]]


def test_prompt_has_the_spec_shape(project, mock_llm, cli, run_builder):
    from impulse_evals.transcriber import format_transcript

    scenario_result = _failed_result(run_builder)
    stored = _store(project.root, scenario_result)
    mock_llm.queue(yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    assert cli.invoke("convert-failure", stored).exit_code == 0
    text = mock_llm.text(0)
    assert PROMPT_START in text
    tail = (
        "\n\nTranscript:\n"
        + format_transcript(scenario_result.transcript)
        + "\n\nAgent system prompt:\n"
        + SYSTEM_PROMPT
    )
    assert tail in text, "transcript and system prompt follow the failed checks exactly as in SPEC.md section 15"
    failures = text.split(PROMPT_START, 1)[1].split("\n\nTranscript:\n", 1)[0]
    for expected in (CONSENT, CONSENT_REASON, "UNIV-023", TOOL_REASON):
        assert expected in failures, f"failed checks are missing {expected!r}"
    for unexpected in ("UNIV-013", PASSED_REASON, SKIPPED_REASON):
        assert unexpected not in failures, f"{unexpected!r} did not fail"


def test_file_name_uses_the_stored_scenario_name(project, mock_llm, cli, run_builder):
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(yaml.safe_dump(dict(LLM_SCENARIO, name="consent_regression_case"), sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert _regression_files(project.root) == ["clinic_booking_regression.yaml"]


def test_file_name_is_sanitized(project, mock_llm, cli, run_builder):
    stored = _store(project.root, _failed_result(run_builder, name="clinic booking/v2"))
    mock_llm.queue(yaml.safe_dump(dict(LLM_SCENARIO, name="clinic_booking_v2"), sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert _regression_files(project.root) == ["clinic_booking_v2_regression.yaml"]


# ============================================================================================
# Validate, retry once, then exit 1
# ============================================================================================


@pytest.mark.parametrize(
    "first_answer",
    [
        pytest.param("name: [unclosed\npersona: {", id="invalid_yaml"),
        pytest.param(
            yaml.safe_dump({k: v for k, v in LLM_SCENARIO.items() if k != "persona"}, sort_keys=False),
            id="validation_error",
        ),
        pytest.param(None, id="empty_answer"),
    ],
)
def test_invalid_answer_is_retried_once(project, mock_llm, cli, run_builder, first_answer):
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(first_answer, yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 2
    assert all(call["model"] == PRIMARY for call in mock_llm.calls)
    first, second = (" ".join(mock_llm.text(index).split()) for index in (0, 1))
    assert first in second and len(second) > len(first), "the retry repeats the request with the error appended"
    _data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.persona == LLM_SCENARIO["persona"]


def test_two_invalid_answers_exit_1_and_write_nothing(project, mock_llm, cli, run_builder):
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue("name: [unclosed", "persona: {still broken")
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 1, cli.text(result)
    assert mock_llm.call_count == 2
    assert _regression_files(project.root) == []


# ============================================================================================
# Usage errors
# ============================================================================================


def _assert_usage_error(cli, result):
    assert result.exit_code == 2, cli.text(result)
    assert "No such command" not in cli.text(result), "impulse convert-failure must be a registered command"


def test_missing_file_exits_2(project, mock_llm, cli):
    result = cli.invoke("convert-failure", project.root / "no_such_result.json")
    _assert_usage_error(cli, result)
    assert mock_llm.call_count == 0
    assert _regression_files(project.root) == []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"scenario_name": "x", "transcript": ', id="invalid_json"),
        pytest.param('{"scenario_name": "x"}', id="not_a_scenario_result"),
    ],
)
def test_file_that_is_not_a_scenario_result_exits_2(project, mock_llm, cli, text):
    path = project.root / "broken.json"
    path.write_text(text, encoding="utf-8")
    result = cli.invoke("convert-failure", path)
    _assert_usage_error(cli, result)
    assert mock_llm.call_count == 0
    assert _regression_files(project.root) == []


def test_unresolved_primary_model_exits_2_without_llm_call(impulse_project, mock_llm, cli, run_builder):
    """No IMPULSE_MODEL_* variables and no config.toml: the primary model is a config error (exit 2)."""
    stored = _store(impulse_project.root, _failed_result(run_builder))
    result = cli.invoke("convert-failure", stored)
    _assert_usage_error(cli, result)
    assert mock_llm.call_count == 0
    assert _regression_files(impulse_project.root) == []


# ============================================================================================
# Audit batch 1 review fix-ups (AUDIT.md finding 6, follow-up): impulse convert-failure reads no
# config.yaml, so the literal credentials in config.toml [model_options.primary] must still be
# redacted from what it prints (``Bearer [REDACTED]``) while reaching the LiteLLM call unchanged.
# ============================================================================================

TOML_LITERAL_BEARER = "LITERALTOML321"
TOML_LITERAL_HEADERS = (
    "[model_options.primary]\n"
    f'extra_params = {{ extra_headers = {{ Authorization = "Bearer {TOML_LITERAL_BEARER}" }} }}\n'
)
ECHOED_REDACTED_HEADERS = "401 Unauthorized; request headers: {'Authorization': 'Bearer [REDACTED]'}"


def _raise_echoing_request_headers(call):
    """A LiteLLM error that echoes the request headers it was sent (an HTTP client error)."""
    raise RuntimeError(f"401 Unauthorized; request headers: {call['kwargs'].get('extra_headers')}")


def test_convert_failure_never_prints_literal_toml_option_credentials(project, mock_llm, cli, run_builder):
    project.write_toml(TOML_LITERAL_HEADERS)
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.responder = _raise_echoing_request_headers
    result = cli.invoke("convert-failure", stored)
    text = cli.text(result)
    assert result.exit_code == 1, text
    assert TOML_LITERAL_BEARER not in result.output, text
    assert ECHOED_REDACTED_HEADERS in text
    # The header still reaches the LiteLLM call unchanged (redaction is for printed texts only).
    assert mock_llm.calls[0]["model"] == PRIMARY
    assert mock_llm.calls[0]["kwargs"].get("extra_headers") == {"Authorization": f"Bearer {TOML_LITERAL_BEARER}"}
    assert _regression_files(project.root) == []


# ============================================================================================
# Audit batch 2 (AUDIT.md finding 13): a critical assertion the judge could not evaluate (result
# "skip", no_verdict True, severity critical; SPEC.md section 13) is a failed check of kind
# assertion: the prompt lists "{rule_id} ({severity}) not evaluated: {reason}" in assertion order
# (before metrics and the scenario error), and the rule is added to the regression scenario's
# assertions. High / medium no-verdict skips and valid judge skips are not failed checks.
# ============================================================================================

PURPOSE = "dpdp.purpose_disclosure"
WITHDRAWAL = "dpdp.withdrawal_path"
INVALID_REASON = "Judge returned invalid JSON"
EMPTY_REASON = "Judge returned empty (safety filter or rate limit)"
NO_VERDICT_ERROR = f"critical check {CONSENT} could not be evaluated: judge returned no verdict"
# The model's answer selects no assertion, so only the failed checks can add one.
NO_ASSERTION_SCENARIO = dict(
    LLM_SCENARIO,
    name="clinic_booking_regression",
    assertions=[],
    metrics=["UNIV-013"],
)


def _failures(mock_llm, index=0):
    """The failed-check lines of the convert-failure prompt."""
    text = mock_llm.text(index)
    return text.split(PROMPT_START, 1)[1].split("\n\nTranscript:\n", 1)[0].splitlines()


def _critical_no_verdict_result(run_builder, reason=INVALID_REASON):
    """The SPEC.md section 13 shape: the critical rule skipped with no verdict, the scenario error set."""
    transcript = run_builder.transcript("clinic_booking", agent_system_prompt=SYSTEM_PROMPT)
    return run_builder.scenario_result(
        "clinic_booking",
        transcript=transcript,
        assertions=[
            (CONSENT, "skip", {"no_verdict": True, "reason": reason}),
            (PURPOSE, "pass"),
        ],
        metrics=[("UNIV-013", 1.0)],
        error=NO_VERDICT_ERROR,
    )


@pytest.mark.parametrize("reason", [INVALID_REASON, EMPTY_REASON], ids=["invalid_json", "empty"])
def test_critical_no_verdict_check_is_listed_and_added_to_the_regression_scenario(
    project, mock_llm, cli, run_builder, reason
):
    scenario_result = _critical_no_verdict_result(run_builder, reason)
    assert scenario_result.passed is False
    stored = _store(project.root, scenario_result)
    mock_llm.queue(yaml.safe_dump(NO_ASSERTION_SCENARIO, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 1
    assert _failures(mock_llm) == [
        f"{CONSENT} (critical) not evaluated: {reason}",
        f"scenario error: {NO_VERDICT_ERROR}",
    ]
    _data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.assertions == [CONSENT], "the rule that could not be evaluated is checked again"
    assert scenario.metrics == ["UNIV-013"]
    assert CONSENT in cli.text(result).split("Wrote ", 1)[1]


def test_critical_no_verdict_line_comes_in_assertion_order_before_metrics(project, mock_llm, cli, run_builder):
    transcript = run_builder.transcript("clinic_booking", agent_system_prompt=SYSTEM_PROMPT)
    scenario_result = run_builder.scenario_result(
        "clinic_booking",
        transcript=transcript,
        assertions=[
            (PURPOSE, "fail", {"reason": "No purpose was given."}),
            (CONSENT, "skip", {"no_verdict": True, "reason": INVALID_REASON}),
            (WITHDRAWAL, "fail", {"reason": "The withdrawal path   was\nnever offered."}),
        ],
        metrics=[("UNIV-013", 0.0, {"reason": "Cold tone."})],
        error=NO_VERDICT_ERROR,
    )
    stored = _store(project.root, scenario_result)
    mock_llm.queue(yaml.safe_dump(NO_ASSERTION_SCENARIO, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert _failures(mock_llm) == [
        f"{PURPOSE} (high) failed: No purpose was given.",
        f"{CONSENT} (critical) not evaluated: {INVALID_REASON}",
        f"{WITHDRAWAL} (medium) failed: The withdrawal path was never offered.",
        "UNIV-013 (Tone) poor: Cold tone.",
        f"scenario error: {NO_VERDICT_ERROR}",
    ]
    _data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.assertions == [PURPOSE, CONSENT, WITHDRAWAL]


def test_non_critical_no_verdict_and_valid_skips_are_not_failed_checks(project, mock_llm, cli, run_builder):
    # Regression guard (passes before the fix too): only a CRITICAL no-verdict skip is a failed check.
    transcript = run_builder.transcript("clinic_booking", agent_system_prompt=SYSTEM_PROMPT)
    scenario_result = run_builder.scenario_result(
        "clinic_booking",
        transcript=transcript,
        assertions=[
            (CONSENT, "skip", {"reason": "No personal data was requested."}),           # valid judge skip
            (PURPOSE, "skip", {"no_verdict": True, "reason": INVALID_REASON}),          # high, no verdict
            (WITHDRAWAL, "skip", {"no_verdict": True, "reason": EMPTY_REASON}),         # medium, no verdict
        ],
        metrics=[("UNIV-013", 0.0, {"reason": "Cold tone."})],
    )
    assert scenario_result.error is None and scenario_result.passed is False
    stored = _store(project.root, scenario_result)
    mock_llm.queue(yaml.safe_dump(NO_ASSERTION_SCENARIO, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert _failures(mock_llm) == ["UNIV-013 (Tone) poor: Cold tone."]
    _data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.assertions == []


def test_only_non_critical_no_verdict_skips_means_nothing_to_convert(project, mock_llm, cli, run_builder):
    # Regression guard (passes before the fix too): a high no-verdict skip alone does not fail the
    # scenario (SPEC.md section 13), so there is no failure to convert (exit 2, no LLM call).
    scenario_result = run_builder.scenario_result(
        "clinic_booking",
        assertions=[(PURPOSE, "skip", {"no_verdict": True, "reason": INVALID_REASON})],
        metrics=[("UNIV-013", 1.0)],
    )
    assert scenario_result.passed is True
    stored = _store(project.root, scenario_result)
    result = cli.invoke("convert-failure", stored)
    _assert_usage_error(cli, result)
    assert mock_llm.call_count == 0
    assert _regression_files(project.root) == []


# ============================================================================================
# Audit batch 2 (AUDIT.md finding 21): the regression YAML is redacted before it is written. Every
# string value of the scenario mapping and the header comment lines pass through the CLI's redactor,
# so a known key value repeated by the model (or carried by the stored result) is written as
# [REDACTED]; values are redacted BEFORE yaml.safe_dump, so the file still loads.
# ============================================================================================

SECRET_ENV = "sk-impulse-test-SECRET-env-leak-9020"
SECRET_TOML = "sk-impulse-test-SECRET-toml-leak-4451"


@pytest.fixture
def secret_project(project, monkeypatch):
    """Two key values the CLI redactor knows: a standard key variable and config.toml [keys]."""
    project.write_toml(f'[keys]\nOPENAI_API_KEY = "{SECRET_TOML}"\n')
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET_ENV)
    return project


def _written_regression_files(root):
    folder = Path(root) / "scenarios" / "regression"
    return sorted(folder.glob("*.yaml")) if folder.is_dir() else []


def test_convert_failure_redacts_key_values_in_the_written_scenario(secret_project, mock_llm, cli, run_builder):
    project = secret_project
    stored = _store(project.root, _failed_result(run_builder))
    answer = dict(
        LLM_SCENARIO,
        name=f"regress_{SECRET_TOML}",
        persona=f"patient whose note says {SECRET_ENV}",
        opening_user_message=f"{SECRET_ENV} is my account code; book Tuesday.",
        expected_outcome=f"the agent never repeats {SECRET_TOML}",
        key_terms=[SECRET_ENV, "Tuesday"],
    )
    mock_llm.queue(yaml.safe_dump(answer, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    path = _regression_file(project.root, "clinic_booking")
    raw = path.read_text(encoding="utf-8")
    assert SECRET_ENV not in raw
    assert SECRET_TOML not in raw
    assert "[REDACTED]" in raw
    data, scenario = _one_item(path)   # still YAML that validates
    assert scenario.name == "regress_[REDACTED]"
    assert scenario.persona == "patient whose note says [REDACTED]"
    assert scenario.opening_user_message == "[REDACTED] is my account code; book Tuesday."
    assert scenario.key_terms == ["[REDACTED]", "Tuesday"]
    cfg = project.load(project.write_config(data, path=project.root / "regression_config.yaml"))
    assert [s.name for s in cfg.scenarios] == ["regress_[REDACTED]"]


def test_convert_failure_redacts_the_header_comment(secret_project, mock_llm, cli, run_builder):
    # The header names the stored scenario, which comes from FILE.
    project = secret_project
    stored = _store(project.root, _failed_result(run_builder, name=f"clinic {SECRET_ENV}"))
    mock_llm.queue(yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    (path,) = _written_regression_files(project.root)
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("#")
    assert SECRET_ENV not in raw
    header = [line for line in raw.splitlines() if line.startswith("#")]
    assert any("[REDACTED]" in line for line in header), header
    _one_item(path)


def test_convert_failure_without_known_keys_writes_the_answer_verbatim(project, mock_llm, cli, run_builder):
    # Regression guard (passes before the fix too): nothing is redacted when no key value occurs.
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    assert cli.invoke("convert-failure", stored).exit_code == 0
    path = _regression_file(project.root, "clinic_booking")
    assert "[REDACTED]" not in path.read_text(encoding="utf-8")
    _data, scenario = _one_item(path)
    assert scenario.persona == LLM_SCENARIO["persona"]
    assert scenario.opening_user_message == LLM_SCENARIO["opening_user_message"]


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 35): an answer with a max_turns below 1 breaks a config.yaml
# scenario rule (SPEC.md section 8.2: "max_turns {n} must be at least 1"), so it is invalid like
# any other: one retry with the error appended, then exit 1. (A blank name in the answer is
# replaced by "{scenario_name}_regression", so it is never an error here.)
# ============================================================================================


@pytest.mark.parametrize("max_turns", [0, -2])
def test_answer_with_max_turns_below_1_is_retried(project, mock_llm, cli, run_builder, max_turns):
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(
        yaml.safe_dump(dict(LLM_SCENARIO, max_turns=max_turns), sort_keys=False),
        yaml.safe_dump(LLM_SCENARIO, sort_keys=False),
    )
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 2, "the first answer breaks a config.yaml scenario rule and must be retried"
    second = " ".join(mock_llm.text(1).split())
    assert f"max_turns {max_turns} must be at least 1" in second
    _data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.max_turns == LLM_SCENARIO["max_turns"]


def test_two_answers_with_max_turns_0_exit_1_and_write_nothing(project, mock_llm, cli, run_builder):
    stored = _store(project.root, _failed_result(run_builder))
    bad = yaml.safe_dump(dict(LLM_SCENARIO, max_turns=0), sort_keys=False)
    mock_llm.queue(bad, bad)
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 1, cli.text(result)
    assert mock_llm.call_count == 2
    assert _regression_files(project.root) == []


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 36; maintainer decision #36): the regression scenario's language
# and accent always come from the stored transcript (transcript.language / transcript.accent),
# overriding the model's answer; a model-supplied seed and inject_events are dropped (seed None,
# inject_events []), as impulse generate does (SPEC.md section 15).
# ============================================================================================

HINDI_NAME = "clinic_booking_hi"
SILENCE_EVENT = {"at_turn": 2, "type": "silence", "silence_ms": 800, "note": "model invented"}
FRENCH_ANSWER = dict(
    LLM_SCENARIO,
    name=HINDI_NAME,
    language="fr-FR",
    accent="parisian",
    seed=7,
    inject_events=[SILENCE_EVENT],
)


def _hindi_failed_result(run_builder, accent="delhi"):
    turns = [
        run_builder.turn("user", "Namaste, mujhe Tuesday ko appointment chahiye.", 0),
        run_builder.turn("agent", "Zaroor. Aapka phone number kya hai?", 900, latency_ms=900.0),
    ]
    transcript = run_builder.transcript(
        HINDI_NAME, turns, language="hi-IN", accent=accent, agent_system_prompt=SYSTEM_PROMPT
    )
    return run_builder.scenario_result(
        HINDI_NAME,
        transcript=transcript,
        assertions=[(CONSENT, "fail", {"reason": CONSENT_REASON, "turn_index": 2})],
    )


def _convert_hindi(project, mock_llm, cli, run_builder, answer, accent="delhi"):
    stored = _store(project.root, _hindi_failed_result(run_builder, accent=accent))
    mock_llm.queue(yaml.safe_dump(answer, sort_keys=False, allow_unicode=True))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    data, scenario = _one_item(_regression_file(project.root, HINDI_NAME))
    return result, data[0], scenario


def test_language_and_accent_come_from_the_stored_transcript(project, mock_llm, cli, run_builder):
    _result, _raw, scenario = _convert_hindi(project, mock_llm, cli, run_builder, FRENCH_ANSWER)
    assert (scenario.language, scenario.accent) == ("hi-IN", "delhi"), "a hi-IN / delhi failure stays hi-IN / delhi"


def test_accent_comes_from_the_stored_transcript_when_the_answer_has_none(project, mock_llm, cli, run_builder):
    answer = {key: value for key, value in FRENCH_ANSWER.items() if key not in ("language", "accent")}
    _result, _raw, scenario = _convert_hindi(project, mock_llm, cli, run_builder, answer)
    assert (scenario.language, scenario.accent) == ("hi-IN", "delhi")


def test_no_accent_in_the_stored_transcript_overrides_the_answer_accent(project, mock_llm, cli, run_builder):
    _result, raw, scenario = _convert_hindi(project, mock_llm, cli, run_builder, FRENCH_ANSWER, accent=None)
    assert (scenario.language, scenario.accent) == ("hi-IN", None)
    assert raw.get("accent") is None


def test_model_seed_and_inject_events_are_dropped(project, mock_llm, cli, run_builder):
    result, raw, scenario = _convert_hindi(project, mock_llm, cli, run_builder, FRENCH_ANSWER)
    assert scenario.seed is None
    assert raw.get("seed") is None, "no seed is written to the regression file"
    assert scenario.inject_events == []
    assert raw.get("inject_events") in (None, [])


def test_model_inject_events_outside_max_turns_no_longer_cost_the_retry(project, mock_llm, cli, run_builder):
    # The answer's own inject_events are dropped before the config.yaml rules run, so an event turn
    # beyond max_turns cannot make an otherwise valid answer invalid.
    stored = _store(project.root, _hindi_failed_result(run_builder))
    answer = dict(FRENCH_ANSWER, inject_events=[dict(SILENCE_EVENT, at_turn=20)])
    mock_llm.queue(yaml.safe_dump(answer, sort_keys=False), yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 1, "the dropped inject_events must not make the answer invalid"
    _data, scenario = _one_item(_regression_file(project.root, HINDI_NAME))
    assert scenario.inject_events == []
    assert scenario.language == "hi-IN"


def test_answer_fields_other_than_language_accent_seed_and_inject_events_are_kept(project, mock_llm, cli, run_builder):
    # Regression guard (passes before the fix too).
    _result, _raw, scenario = _convert_hindi(project, mock_llm, cli, run_builder, FRENCH_ANSWER)
    assert scenario.name == HINDI_NAME
    assert scenario.persona == LLM_SCENARIO["persona"]
    assert scenario.opening_user_message == LLM_SCENARIO["opening_user_message"]
    assert scenario.max_turns == LLM_SCENARIO["max_turns"]
    assert scenario.assertions == [CONSENT]


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 37): an unfenced answer that parses as YAML by itself is used
# unchanged, so a fenced snippet inside it (a persona quoting an error log) is not mistaken for the
# answer; an answer starting with ``` still has that fence stripped.
# ============================================================================================

ANSWER_MAPPING_WITH_INNER_FENCE = """\
name: pasted_snippet_regression
language: en-IN
domain: healthcare
direction: inbound
max_turns: 6
persona: |
  patient who pastes the booking error into the chat:
  ```
  Error: slot unavailable
  ```
  and then gives the phone number before any consent is asked
opening_user_message: "Hi, I need a slot on Tuesday."
expected_outcome: the agent asks for consent before collecting personal data
assertions: [dpdp.consent_before_pii]
metrics: [UNIV-023]
"""


def test_unfenced_answer_with_a_fenced_snippet_inside_is_used_whole(project, mock_llm, cli, run_builder):
    expected = yaml.safe_load(ANSWER_MAPPING_WITH_INNER_FENCE)
    assert isinstance(expected, dict)   # the answer is a valid YAML mapping by itself
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(ANSWER_MAPPING_WITH_INNER_FENCE, yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 1, "a valid unfenced answer must be used as it is, not reduced to its inner snippet"
    _data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.name == expected["name"]
    assert scenario.persona == expected["persona"]


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda body: "```yaml\n" + body + "```", id="fenced"),
        pytest.param(lambda body: "Here is the scenario.\n\n```yaml\n" + body + "```\n", id="preamble_then_fence"),
    ],
)
def test_fenced_answer_is_still_unwrapped(project, mock_llm, cli, run_builder, wrap):
    # Regression guard (passes before the fix too).
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(wrap(yaml.safe_dump(LLM_SCENARIO, sort_keys=False)))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 1
    _data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.persona == LLM_SCENARIO["persona"]


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 39): the regression YAML goes to a temporary file in the same
# folder that is then os.replace()d onto the target; on any error the temporary file is removed and
# an existing regression file is left untouched (the CLI exits 2: "cannot write ...").
# ============================================================================================

PREVIOUS_REGRESSION = b"# previous regression scenario (kept)\n- name: previous_case\n"


def _fail_replace_onto(monkeypatch, target):
    """``os.replace`` raises (permission denied) when its destination is ``target``; otherwise it works."""
    import errno
    import os

    real_replace = os.replace
    destination = Path(target).resolve()

    def failing_replace(src, dst, *args, **kwargs):
        if Path(dst).resolve() == destination:
            raise OSError(errno.EACCES, "Permission denied (test fault)", str(dst))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)


def test_convert_failure_keeps_the_previous_file_when_the_final_replace_fails(
    project, mock_llm, cli, run_builder, monkeypatch
):
    stored = _store(project.root, _failed_result(run_builder))
    target = _regression_file(project.root, "clinic_booking")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PREVIOUS_REGRESSION)
    mock_llm.queue(yaml.safe_dump(LLM_SCENARIO, sort_keys=False))
    _fail_replace_onto(monkeypatch, target)
    result = cli.invoke("convert-failure", stored)
    assert target.read_bytes() == PREVIOUS_REGRESSION, "the previous regression file was overwritten in place"
    assert _regression_files(project.root) == ["clinic_booking_regression.yaml"], "no temporary file is left"
    assert result.exit_code == 2, cli.text(result)


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 40; maintainer decision #40): the regression file name keeps
# Unicode letters, digits and combining marks of the stored name (NFC first); every other
# character becomes "_" as before.
# ============================================================================================


@pytest.mark.parametrize(
    "stored_name, stem",
    [
        pytest.param("क्लिनिक बुकिंग", "क्लिनिक_बुकिंग", id="devanagari"),
        pytest.param("नमस्ते", "नमस्ते", id="devanagari_combining_marks"),
        pytest.param("clinic booking/v2", "clinic_booking_v2", id="ascii"),
    ],
)
def test_regression_file_name_keeps_unicode_letters(project, mock_llm, cli, run_builder, stored_name, stem):
    stored = _store(project.root, _failed_result(run_builder, name=stored_name))
    mock_llm.queue(yaml.safe_dump(dict(LLM_SCENARIO, name="clinic_booking_regression"), sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert _regression_files(project.root) == [f"{stem}_regression.yaml"]


# ============================================================================================
# Review fix-up of audit batch 3 (group Z), regression of finding 40: the regression file stem
# (the stored name, sanitised) is cut to at most 200 UTF-8 bytes on a character boundary, never
# splitting a combining mark run, so a long Devanagari name (3 bytes per character) stays under
# the 255-byte file name limit with "_regression.yaml" added. Checked on byte lengths.
# ============================================================================================

REGRESSION_SUFFIX = "_regression.yaml"
STEM_BYTE_LIMIT = 200
LONG_DEVANAGARI_NAME = ("क्लिनिक बुकिंग और भुगतान की पुष्टि " * 5)[:119] + "१"   # 120 characters
SPLIT_RUN_DEVANAGARI_NAME = "क" * 64 + "किं" + "क" * 53   # a naive 200-byte cut splits "िं"


def _spec_sanitised_name(name):
    """SPEC.md section 15 sanitising (NFC, then everything but str.isalnum(), "-", "_" and Mn / Mc
    combining marks becomes "_"), before any cut."""
    import unicodedata

    return "".join(
        char if char.isalnum() or char in "-_" or unicodedata.category(char) in ("Mn", "Mc") else "_"
        for char in unicodedata.normalize("NFC", name)
    )


@pytest.mark.parametrize(
    "stored_name",
    [
        pytest.param(LONG_DEVANAGARI_NAME, id="devanagari_120_chars"),
        pytest.param(SPLIT_RUN_DEVANAGARI_NAME, id="cut_inside_a_combining_mark_run"),
    ],
)
def test_regression_file_stem_of_a_long_unicode_name_is_at_most_200_utf8_bytes(
    project, mock_llm, cli, run_builder, stored_name
):
    import unicodedata

    full = _spec_sanitised_name(stored_name)
    assert len(full.encode("utf-8")) > 255
    stored = _store(project.root, _failed_result(run_builder, name=stored_name))
    mock_llm.queue(yaml.safe_dump(dict(LLM_SCENARIO, name="clinic_booking_regression"), sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    files = _regression_files(project.root)
    assert len(files) == 1 and files[0].endswith(REGRESSION_SUFFIX), files
    stem = files[0][: -len(REGRESSION_SUFFIX)]
    size = len(stem.encode("utf-8"))
    assert size <= STEM_BYTE_LIMIT, f"the regression file stem is {size} UTF-8 bytes (limit {STEM_BYTE_LIMIT})"
    assert size > 150, f"the cut keeps the name up to the limit, not much less ({size} bytes)"
    assert unicodedata.is_normalized("NFC", stem)
    assert full.startswith(stem), "the stem is a prefix of the full sanitised name"
    rest = full[len(stem):]
    marks = ("Mn", "Mc")
    assert not (rest and unicodedata.category(stem[-1]) in marks and unicodedata.category(rest[0]) in marks), (
        "the cut splits a combining mark run between the stem and the dropped rest"
    )


def test_regression_file_stem_of_at_most_200_utf8_bytes_is_unchanged(project, mock_llm, cli, run_builder):
    # Regression guard (passes before the fix too): a stem of exactly 200 bytes is kept whole.
    name = "ab" + "क" * 66
    assert len(name.encode("utf-8")) == STEM_BYTE_LIMIT
    stored = _store(project.root, _failed_result(run_builder, name=name))
    mock_llm.queue(yaml.safe_dump(dict(LLM_SCENARIO, name="clinic_booking_regression"), sort_keys=False))
    result = cli.invoke("convert-failure", stored)
    assert result.exit_code == 0, cli.text(result)
    assert _regression_files(project.root) == [name + REGRESSION_SUFFIX]


# ============================================================================================
# Review fix-up of audit batch 3 (group Z): an answer that makes PyYAML raise ValueError (a
# timestamp with an invalid day, ``2024-02-30``) or RecursionError (very deep nesting) while it is
# loaded is an invalid answer like any other: retried once with the error appended, then exit 1
# ("conversion failed"), never exit 2 with "unexpected error".
# ============================================================================================

_GOOD_ANSWER = yaml.safe_dump(LLM_SCENARIO, sort_keys=False)
_OUTCOME_LINE = f"expected_outcome: {LLM_SCENARIO['expected_outcome']}"
BAD_DATE_ANSWER = _GOOD_ANSWER.replace(_OUTCOME_LINE, "expected_outcome: 2024-02-30")   # PyYAML: ValueError
DEEP_NESTING_ANSWER = "[" * 3000 + "]" * 3000                                        # PyYAML: RecursionError


def _error_head(result):
    """The first ``Error:`` line up to its third ": " (command, failure and cause), for comparing
    the message style of two failures."""
    lines = [line for line in result.output.splitlines() if line.startswith("Error: ")]
    assert lines, result.output
    return lines[0].split(": ")[:3]


@pytest.mark.parametrize(
    "first_answer",
    [pytest.param(BAD_DATE_ANSWER, id="invalid_date_value_error"), pytest.param(DEEP_NESTING_ANSWER, id="recursion_error")],
)
def test_answer_that_makes_yaml_raise_is_retried_once(project, mock_llm, cli, run_builder, first_answer):
    assert "expected_outcome: 2024-02-30" in BAD_DATE_ANSWER
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue(first_answer, _GOOD_ANSWER)
    result = cli.invoke("convert-failure", stored)
    assert "unexpected error" not in cli.text(result), cli.text(result)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 2
    first, second = (" ".join(mock_llm.text(index).split()) for index in (0, 1))
    assert first in second and len(second) > len(first), "the retry repeats the request with the error appended"
    _data, scenario = _one_item(_regression_file(project.root, "clinic_booking"))
    assert scenario.persona == LLM_SCENARIO["persona"]


def test_two_answers_that_make_yaml_raise_value_error_exit_1_like_other_invalid_answers(
    project, mock_llm, cli, run_builder
):
    assert "expected_outcome: 2024-02-30" in BAD_DATE_ANSWER
    stored = _store(project.root, _failed_result(run_builder))
    mock_llm.queue("name: [unclosed", "persona: {still broken")
    reference = cli.invoke("convert-failure", stored)
    assert reference.exit_code == 1, cli.text(reference)
    mock_llm.queue(BAD_DATE_ANSWER, BAD_DATE_ANSWER)
    result = cli.invoke("convert-failure", stored)
    assert "unexpected error" not in cli.text(result), cli.text(result)
    assert result.exit_code == 1, cli.text(result)
    assert mock_llm.call_count == 4
    assert _regression_files(project.root) == []
    assert _error_head(result) == _error_head(reference), "the same message as for any other invalid answer"
