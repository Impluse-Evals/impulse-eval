# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 9: ``impulse generate`` (SPEC.md sections 8.1, 11, and 17).

``impulse generate --prompt F --domain D --language L --count 11 [--out FILE] [--yes] [--voice-mode]``
sends the SPEC.md section 11 generation prompt to the PRIMARY model, validates the YAML answer as a
list of ``Scenario`` objects (retry once with the error appended, then exit 1), and writes it to
``--out`` (default ``scenarios/generated.yaml`` relative to the current directory). The generator
sets ``language``, ``domain``, ``assertions`` (domain table, wildcards expanded), ``inject_events: []``,
``seed`` (null) and the ``max_turns`` default itself, and always writes an explicit, non-empty
``metrics`` list. ``--language`` defaults to ``config.toml [defaults] language``; when that is
missing or still ``YOUR_LANGUAGE`` and ``--language`` is not given, the command exits 2 naming
``--language``.

Commands run through ``typer.testing.CliRunner`` (``cli`` fixture) in ``tmp_path`` (``impulse_project``);
the LLM is ``mock_llm``. Domain guard: ``test_generator_domain_guard.py``; metric lists by mode:
``test_generator_voice_mode.py``; domain table: ``test_domain_defaults.py``.
"""

import copy
from pathlib import Path

import pytest
import yaml

PRIMARY = "ollama/judge-primary"
FAST = "ollama/judge-fast"
SIMULATOR = "ollama/simulator"
MODEL_ENV = {"IMPULSE_MODEL_PRIMARY": PRIMARY, "IMPULSE_MODEL_FAST": FAST, "IMPULSE_MODEL_SIMULATOR": SIMULATOR}
# No domain keyword of SPEC.md section 11, so the domain guard never fires.
PROMPT_TEXT = (
    "You are the assistant of Sunrise Store.\n"
    "Answer questions politely and keep replies short.\n"
    "Never invent facts you were not given."
)
DPDP = {"dpdp.consent_before_pii", "dpdp.purpose_disclosure", "dpdp.withdrawal_path"}
EDTECH_METRICS = {"UNIV-005", "UNIV-013", "UNIV-017"}
LANGUAGE_PLACEHOLDER = "YOUR_LANGUAGE"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def models_env(impulse_project, monkeypatch):
    """Model strings for every role through ``IMPULSE_MODEL_*`` (SPEC.md section 8.1, precedence 2)."""
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    return MODEL_ENV


def _write_prompt(root, text=PROMPT_TEXT, name="agent_prompt.txt"):
    path = Path(root) / name
    path.write_text(text, encoding="utf-8")
    return path


def _items(count, **fields):
    """``count`` scenario mappings as the LLM would write them (no language / domain / assertions)."""
    items = []
    for number in range(1, count + 1):
        item = {
            "name": f"generated_case_{number}",
            "persona": f"caller {number} who wants a quick answer",
            "opening_user_message": f"Hello, this is caller {number}.",
            "expected_outcome": "the caller gets a correct answer",
            "key_terms": [],
            "code_switching": False,
            "direction": "inbound",
            "metrics": ["UNIV-001"],
        }
        item.update(copy.deepcopy(fields))
        items.append(item)
    return items


def _yaml(items):
    return yaml.safe_dump(items, sort_keys=False, allow_unicode=True)


def _generate(cli, prompt, *extra, domain="ecommerce", language="en-IN", count=None, yes=True):
    args = ["generate", "--prompt", prompt, "--domain", domain]
    if language is not None:
        args += ["--language", language]
    if count is not None:
        args += ["--count", str(count)]
    if yes:
        args.append("--yes")
    args += list(extra)
    return cli.invoke(*args)


def _raw_items(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert isinstance(data, list), f"{path} must hold a YAML list of scenarios"
    return data


def _read_scenarios(path):
    from impulse_evals.models import Scenario

    return [Scenario.model_validate(item) for item in _raw_items(path)]


def _default_out(project):
    return project.root / "scenarios" / "generated.yaml"


def _prompt(mock_llm, index=0):
    """All message contents of one LLM call, whitespace runs collapsed (layout is not the contract)."""
    return " ".join(mock_llm.text(index).split())


def _assert_retry_repeats_the_request(mock_llm):
    """The second call carries everything the first one sent, plus the error (whitespace-insensitive)."""
    first, second = _prompt(mock_llm, 0), _prompt(mock_llm, 1)
    assert first in second and len(second) > len(first), "the retry repeats the request with the error appended"


# ---------------------------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------------------------


def test_generate_writes_validated_scenarios_to_default_out(impulse_project, mock_llm, cli, models_env):
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(2)))
    result = _generate(cli, prompt, domain="ecommerce", language="hi-IN", count=2)
    assert result.exit_code == 0, cli.text(result)
    scenarios = _read_scenarios(_default_out(impulse_project))
    assert [scenario.name for scenario in scenarios] == ["generated_case_1", "generated_case_2"]
    for scenario in scenarios:
        assert scenario.language == "hi-IN"
        assert scenario.domain == "ecommerce"
        assert scenario.persona
    assert mock_llm.call_count == 1
    assert mock_llm.calls[0]["model"] == PRIMARY


def test_generate_out_flag_sets_the_output_path(impulse_project, mock_llm, cli, models_env):
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(2)))
    result = _generate(cli, prompt, "--out", "custom/my_scenarios.yaml", count=2)
    assert result.exit_code == 0, cli.text(result)
    assert len(_read_scenarios(impulse_project.root / "custom" / "my_scenarios.yaml")) == 2
    assert not _default_out(impulse_project).exists()


def test_generated_file_can_be_used_as_config_scenarios(impulse_project, mock_llm, cli, models_env):
    """Explicit rule IDs and known metric IDs only: the list loads under ``scenarios:``."""
    prompt = _write_prompt(impulse_project.root)
    items = _items(2)
    items[1]["direction"] = "outbound"
    mock_llm.queue(_yaml(items))
    result = _generate(cli, prompt, domain="fsi", count=2)
    assert result.exit_code == 0, cli.text(result)
    raw = _raw_items(_default_out(impulse_project))
    cfg = impulse_project.load(impulse_project.write_config(raw, path=impulse_project.root / "generated_config.yaml"))
    assert [scenario.name for scenario in cfg.scenarios] == ["generated_case_1", "generated_case_2"]
    assert cfg.warnings == []


# ---------------------------------------------------------------------------------------------
# Generation prompt (SPEC.md section 11)
# ---------------------------------------------------------------------------------------------


def test_generation_prompt_follows_the_spec_template(impulse_project, mock_llm, cli, models_env):
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(11)))
    result = _generate(cli, prompt, domain="ecommerce", language="hi-IN")  # --count defaults to 11
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 1
    raw = mock_llm.text(0)
    assert PROMPT_TEXT in raw, "the system prompt file is inserted verbatim"
    text = _prompt(mock_llm)
    for part in (
        "You are a QA engineer for AI voice agents. Read this agent's system prompt carefully and generate "
        "11 realistic test scenarios that comprehensively test it.",
        "Agent system prompt: " + " ".join(PROMPT_TEXT.split()),
        "Domain: ecommerce Language: hi-IN",
        "Scenarios must match what this specific agent is supposed to do, cover happy paths, edge cases and "
        "failure modes, be realistic for real users, and use what users actually say in hi-IN.",
        "Mix: 3 happy path, 3 edge cases (confused, impatient, code-switching), 3 adversarial (user tries to "
        "make the agent do the wrong thing), 2 hallucination traps (agent likely to invent facts).",
        "For each scenario fill: name, persona, opening_user_message, expected_outcome, key_terms, "
        "code_switching, max_turns, direction, and choose metrics from the list below.",
        "Output YAML only. No explanation. No preamble.",
    ):
        assert part in text, f"generation prompt is missing: {part}"
    assert len(_read_scenarios(_default_out(impulse_project))) == 11


@pytest.mark.parametrize("count, mix", [(2, (1, 1, 0, 0)), (4, (1, 1, 1, 1)), (7, (2, 2, 2, 1))])
def test_generation_prompt_count_and_mix_follow_allocate_mix(impulse_project, mock_llm, cli, models_env, count, mix):
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(count)))
    result = _generate(cli, prompt, count=count)
    assert result.exit_code == 0, cli.text(result)
    text = _prompt(mock_llm)
    happy, edge, adversarial, hallucination = mix
    assert f"generate {count} realistic test scenarios" in text
    assert f"Mix: {happy} happy path, {edge} edge cases" in text
    assert f"{adversarial} adversarial (user tries" in text
    assert f"{hallucination} hallucination traps" in text


# ---------------------------------------------------------------------------------------------
# Fields the generator sets itself (never taken from the LLM)
# ---------------------------------------------------------------------------------------------


def test_generator_overrides_language_domain_assertions_seed_and_inject_events(
    impulse_project, mock_llm, cli, models_env
):
    prompt = _write_prompt(impulse_project.root)
    items = _items(
        2,
        language="fr-FR",
        domain="travel",
        assertions=["hipaa.minimum_necessary", "dpdp.*"],
        seed=42,
        inject_events=[{"at_turn": 1, "type": "silence", "silence_ms": 500}],
    )
    mock_llm.queue(_yaml(items))
    result = _generate(cli, prompt, domain="healthcare", language="en-IN", count=2)
    assert result.exit_code == 0, cli.text(result)
    for scenario in _read_scenarios(_default_out(impulse_project)):
        assert scenario.language == "en-IN"
        assert scenario.domain == "healthcare"
        assert set(scenario.assertions) == DPDP
        assert len(scenario.assertions) == len(set(scenario.assertions))
        assert scenario.seed is None
        assert scenario.inject_events == []


@pytest.mark.parametrize("domain, default", [("fsi", 12), ("healthcare", 12), ("ecommerce", 9), ("general", 9)])
def test_max_turns_default_when_llm_gives_none_and_kept_when_given(
    impulse_project, mock_llm, cli, models_env, domain, default
):
    prompt = _write_prompt(impulse_project.root)
    items = _items(2)
    items[1]["max_turns"] = 7
    mock_llm.queue(_yaml(items))
    result = _generate(cli, prompt, domain=domain, count=2)
    assert result.exit_code == 0, cli.text(result)
    first, second = _read_scenarios(_default_out(impulse_project))
    assert first.max_turns == default
    assert second.max_turns == 7


def test_llm_fields_are_kept(impulse_project, mock_llm, cli, models_env):
    """Fields the generator does not set itself come from the LLM answer (SPEC.md section 11)."""
    prompt = _write_prompt(impulse_project.root)
    items = _items(2)
    items[0].update(code_switching=True, key_terms=["EMI", "refund"], direction="outbound")
    items[0]["expected_outcome"] = "the caller agrees to a refund"
    mock_llm.queue(_yaml(items))
    result = _generate(cli, prompt, domain="ecommerce", count=2)
    assert result.exit_code == 0, cli.text(result)
    first, second = _read_scenarios(_default_out(impulse_project))
    assert first.code_switching is True and second.code_switching is False
    assert first.key_terms == ["EMI", "refund"]
    assert first.direction == "outbound" and second.direction == "inbound"
    assert first.expected_outcome == "the caller agrees to a refund"
    assert first.opening_user_message == "Hello, this is caller 1."


def test_metrics_list_is_explicit_and_non_empty_when_llm_gives_none(impulse_project, mock_llm, cli, models_env):
    prompt = _write_prompt(impulse_project.root)
    items = _items(2)
    del items[0]["metrics"]
    items[1]["metrics"] = []
    mock_llm.queue(_yaml(items))
    result = _generate(cli, prompt, domain="edtech", count=2)
    assert result.exit_code == 0, cli.text(result)
    for item in _raw_items(_default_out(impulse_project)):
        assert isinstance(item.get("metrics"), list) and item["metrics"], "metrics must be written explicitly"
        assert set(item["metrics"]) == EDTECH_METRICS
        assert len(item["metrics"]) == len(set(item["metrics"]))


# ---------------------------------------------------------------------------------------------
# Validate, retry once, then exit 1
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first_answer",
    [
        pytest.param("- name: [unclosed\n  persona: {", id="invalid_yaml"),
        pytest.param(_yaml([{k: v for k, v in item.items() if k != "persona"} for item in _items(2)]), id="validation_error"),
        pytest.param(None, id="empty_answer"),
    ],
)
def test_invalid_answer_is_retried_once_with_the_error_appended(
    impulse_project, mock_llm, cli, models_env, first_answer
):
    prompt = _write_prompt(impulse_project.root)
    good = _items(2, persona="caller from the retry answer")
    mock_llm.queue(first_answer, _yaml(good))
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 2
    assert all(call["model"] == PRIMARY for call in mock_llm.calls)
    _assert_retry_repeats_the_request(mock_llm)
    scenarios = _read_scenarios(_default_out(impulse_project))
    assert [scenario.persona for scenario in scenarios] == ["caller from the retry answer"] * 2


def test_two_invalid_answers_exit_1_and_write_nothing(impulse_project, mock_llm, cli, models_env):
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue("- name: [unclosed", "still: [not yaml")
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 1, cli.text(result)
    assert mock_llm.call_count == 2
    assert not _default_out(impulse_project).exists()


# ---------------------------------------------------------------------------------------------
# --language default (SPEC.md section 8.1), models, and usage errors
# ---------------------------------------------------------------------------------------------


def test_language_defaults_to_config_toml(impulse_project, mock_llm, cli, models_env):
    impulse_project.write_toml('[defaults]\nlanguage = "hi-IN"\n')
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(2)))
    result = _generate(cli, prompt, language=None, count=2)
    assert result.exit_code == 0, cli.text(result)
    assert {scenario.language for scenario in _read_scenarios(_default_out(impulse_project))} == {"hi-IN"}
    assert "Language: hi-IN" in _prompt(mock_llm)


@pytest.mark.parametrize("toml_text", [None, f'[defaults]\nlanguage = "{LANGUAGE_PLACEHOLDER}"\n'], ids=["no_toml", "placeholder"])
def test_missing_language_exits_2_naming_the_flag(impulse_project, mock_llm, cli, models_env, toml_text):
    if toml_text is not None:
        impulse_project.write_toml(toml_text)
    prompt = _write_prompt(impulse_project.root)
    result = _generate(cli, prompt, language=None, count=2)
    _assert_usage_error(cli, result)
    assert "--language" in cli.text(result)
    assert mock_llm.call_count == 0
    assert not _default_out(impulse_project).exists()


def test_flag_language_wins_over_config_toml(impulse_project, mock_llm, cli, models_env):
    impulse_project.write_toml('[defaults]\nlanguage = "hi-IN"\n')
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(2)))
    result = _generate(cli, prompt, language="en-US", count=2)
    assert result.exit_code == 0, cli.text(result)
    assert {scenario.language for scenario in _read_scenarios(_default_out(impulse_project))} == {"en-US"}


def _assert_usage_error(cli, result):
    assert result.exit_code == 2, cli.text(result)
    assert "No such command" not in cli.text(result), "impulse generate must be a registered command"


def test_missing_prompt_file_exits_2(impulse_project, mock_llm, cli, models_env):
    result = _generate(cli, impulse_project.root / "no_such_prompt.txt", count=2)
    _assert_usage_error(cli, result)
    assert mock_llm.call_count == 0


def test_primary_model_from_config_toml(impulse_project, mock_llm, cli, monkeypatch):
    impulse_project.write_toml(
        f'[models]\nprimary = "{PRIMARY}"\nfast = "{FAST}"\nsimulator = "{SIMULATOR}"\n'
    )
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(2)))
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 0, cli.text(result)
    assert [call["model"] for call in mock_llm.calls] == [PRIMARY]


def test_unresolved_primary_model_exits_2_without_llm_call(impulse_project, mock_llm, cli):
    prompt = _write_prompt(impulse_project.root)
    result = _generate(cli, prompt, count=2)
    _assert_usage_error(cli, result)
    assert mock_llm.call_count == 0
    assert not _default_out(impulse_project).exists()


# ---------------------------------------------------------------------------------------------
# Audit batch 1 review fix-ups (AUDIT.md finding 6, follow-up): impulse generate reads no
# config.yaml, so the literal credentials in config.toml [model_options.primary] must still be
# redacted from what it prints (``Bearer [REDACTED]``) while reaching the LiteLLM call unchanged.
# ---------------------------------------------------------------------------------------------

TOML_LITERAL_BEARER = "LITERALTOML321"
TOML_LITERAL_HEADERS = (
    "[model_options.primary]\n"
    f'extra_params = {{ extra_headers = {{ Authorization = "Bearer {TOML_LITERAL_BEARER}" }} }}\n'
)
ECHOED_REDACTED_HEADERS = "401 Unauthorized; request headers: {'Authorization': 'Bearer [REDACTED]'}"


def _raise_echoing_request_headers(call):
    """A LiteLLM error that echoes the request headers it was sent (an HTTP client error)."""
    raise RuntimeError(f"401 Unauthorized; request headers: {call['kwargs'].get('extra_headers')}")


def test_generate_never_prints_literal_toml_option_credentials(impulse_project, mock_llm, cli, models_env):
    impulse_project.write_toml(TOML_LITERAL_HEADERS)
    prompt = _write_prompt(impulse_project.root)
    mock_llm.responder = _raise_echoing_request_headers
    result = _generate(cli, prompt, count=2)
    text = cli.text(result)
    assert result.exit_code == 1, text
    assert TOML_LITERAL_BEARER not in result.output, text
    assert ECHOED_REDACTED_HEADERS in text
    # The header still reaches the LiteLLM call unchanged (redaction is for printed texts only).
    assert mock_llm.calls[0]["model"] == PRIMARY
    assert mock_llm.calls[0]["kwargs"].get("extra_headers") == {"Authorization": f"Bearer {TOML_LITERAL_BEARER}"}
    assert not _default_out(impulse_project).exists()


# ---------------------------------------------------------------------------------------------
# Audit batch 2 (AUDIT.md finding 21): scenarios/generated.yaml is redacted before it is written.
# Every string value of the scenario mappings passes through the CLI's redactor, so a known key
# value repeated by the model is written as [REDACTED]; values are redacted BEFORE yaml.safe_dump
# (a "[REDACTED]" at the start of a plain scalar would not parse), so the file still loads.
# ---------------------------------------------------------------------------------------------

SECRET_ENV = "sk-impulse-test-SECRET-env-leak-9020"
SECRET_TOML = "sk-impulse-test-SECRET-toml-leak-4451"


def test_generate_redacts_key_values_in_the_written_scenarios(impulse_project, mock_llm, cli, models_env, monkeypatch):
    impulse_project.write_toml(f'[keys]\nOPENAI_API_KEY = "{SECRET_TOML}"\n')
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET_ENV)
    prompt = _write_prompt(impulse_project.root)
    items = _items(2)
    items[0].update(
        name=f"case_{SECRET_ENV}",
        persona=f"caller who reads {SECRET_TOML} aloud",
        opening_user_message=f"{SECRET_ENV} is my order code.",
        expected_outcome=f"the agent never repeats {SECRET_ENV}",
        key_terms=[SECRET_TOML, "refund"],
    )
    mock_llm.queue(_yaml(items))
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 0, cli.text(result)
    out = _default_out(impulse_project)
    raw = out.read_text(encoding="utf-8")
    assert SECRET_ENV not in raw
    assert SECRET_TOML not in raw
    assert "[REDACTED]" in raw
    first, second = _read_scenarios(out)   # still YAML that validates
    assert first.name == "case_[REDACTED]"
    assert first.persona == "caller who reads [REDACTED] aloud"
    assert first.opening_user_message == "[REDACTED] is my order code."
    assert first.expected_outcome == "the agent never repeats [REDACTED]"
    assert first.key_terms == ["[REDACTED]", "refund"]
    assert second.persona == items[1]["persona"]   # values without a key are written as they were
    raw_items = _raw_items(out)
    cfg = impulse_project.load(impulse_project.write_config(raw_items, path=impulse_project.root / "generated_config.yaml"))
    assert [scenario.name for scenario in cfg.scenarios] == ["case_[REDACTED]", "generated_case_2"]
    assert SECRET_ENV not in result.output and SECRET_TOML not in result.output


def test_generate_without_known_keys_writes_the_answer_verbatim(impulse_project, mock_llm, cli, models_env):
    # Regression guard (passes before the fix too).
    prompt = _write_prompt(impulse_project.root)
    items = _items(2, persona="caller [who brackets] words like [this]")
    mock_llm.queue(_yaml(items))
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 0, cli.text(result)
    raw = _default_out(impulse_project).read_text(encoding="utf-8")
    assert "[REDACTED]" not in raw
    assert [s.persona for s in _read_scenarios(_default_out(impulse_project))] == [items[0]["persona"]] * 2


# ---------------------------------------------------------------------------------------------
# Audit batch 2 (AUDIT.md finding 10): the key variable a hosted model needs is the one LiteLLM
# reads for its provider (litellm.validate_environment, offline): together_ai -> TOGETHERAI_API_KEY,
# perplexity -> PERPLEXITYAI_API_KEY. impulse generate reads no config.yaml; the key reaches the
# LiteLLM call as api_key.
# ---------------------------------------------------------------------------------------------

LITELLM_KEY_CASES = [
    pytest.param("together_ai", "TOGETHERAI_API_KEY", "TOGETHER_AI_API_KEY", id="together_ai"),
    pytest.param("perplexity", "PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY", id="perplexity"),
]


@pytest.mark.parametrize("provider, key_name, wrong_name", LITELLM_KEY_CASES)
def test_generate_uses_the_key_variable_litellm_reads(
    impulse_project, mock_llm, cli, monkeypatch, provider, key_name, wrong_name
):
    secret = f"impulse-test-SECRET-{provider}-6612"
    model = f"{provider}/impulse-test-model"
    monkeypatch.delenv(wrong_name, raising=False)
    monkeypatch.setenv(key_name, secret)
    monkeypatch.setenv("IMPULSE_MODEL_PRIMARY", model)
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(2)))
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 0, cli.text(result)
    assert wrong_name not in cli.text(result)
    assert [call["model"] for call in mock_llm.calls] == [model]
    assert mock_llm.calls[0]["kwargs"].get("api_key") == secret
    assert secret not in result.output
    assert len(_read_scenarios(_default_out(impulse_project))) == 2


@pytest.mark.parametrize("provider, key_name, wrong_name", LITELLM_KEY_CASES)
def test_generate_without_the_litellm_key_variable_names_it(
    impulse_project, mock_llm, cli, monkeypatch, provider, key_name, wrong_name
):
    monkeypatch.delenv(key_name, raising=False)
    monkeypatch.delenv(wrong_name, raising=False)
    monkeypatch.setenv("IMPULSE_MODEL_PRIMARY", f"{provider}/impulse-test-model")
    prompt = _write_prompt(impulse_project.root)
    result = _generate(cli, prompt, count=2)
    _assert_usage_error(cli, result)
    assert f"{key_name} is not set" in cli.text(result)
    assert wrong_name not in cli.text(result)
    assert mock_llm.call_count == 0


# ---------------------------------------------------------------------------------------------
# Audit batch 3 (AUDIT.md finding 35): a generated scenario with a blank name (empty or whitespace
# only) or a max_turns below 1 breaks a config.yaml scenario rule (SPEC.md section 8.2:
# "name must not be blank", "max_turns {n} must be at least 1"), so the answer is invalid like any
# other: one retry with the error appended, then exit 1.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad, message",
    [
        pytest.param({"max_turns": 0}, "max_turns 0 must be at least 1", id="max_turns_0"),
        pytest.param({"max_turns": -2}, "max_turns -2 must be at least 1", id="max_turns_negative"),
        pytest.param({"name": ""}, "name must not be blank", id="empty_name"),
        pytest.param({"name": "   "}, "name must not be blank", id="blank_name"),
    ],
)
def test_generate_retries_an_answer_with_a_blank_name_or_max_turns_below_1(
    impulse_project, mock_llm, cli, models_env, bad, message
):
    prompt = _write_prompt(impulse_project.root)
    first = _items(2)
    first[0].update(bad)
    mock_llm.queue(_yaml(first), _yaml(_items(2, persona="caller from the retry answer")))
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 2, "the first answer breaks a config.yaml scenario rule and must be retried"
    _assert_retry_repeats_the_request(mock_llm)
    assert message in _prompt(mock_llm, 1)
    scenarios = _read_scenarios(_default_out(impulse_project))
    assert [scenario.persona for scenario in scenarios] == ["caller from the retry answer"] * 2


def test_generate_exits_1_when_both_answers_have_max_turns_0(impulse_project, mock_llm, cli, models_env):
    prompt = _write_prompt(impulse_project.root)
    bad = _yaml(_items(2, max_turns=0))
    mock_llm.queue(bad, bad)
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 1, cli.text(result)
    assert mock_llm.call_count == 2
    assert not _default_out(impulse_project).exists()


# ---------------------------------------------------------------------------------------------
# Audit batch 3 (AUDIT.md finding 37): an answer whose stripped text starts with ``` has that
# fence stripped (as before); otherwise an answer that parses as a YAML list or mapping by itself
# is used unchanged, so a fenced snippet INSIDE an unfenced answer (a persona quoting code) no
# longer reduces the answer to the snippet; anything else falls back to the first fenced block
# (an answer with a preamble line still works).
# ---------------------------------------------------------------------------------------------

ANSWER_WITH_INNER_FENCE = """\
- name: pasted_snippet
  persona: |
    developer who pastes an error log into the chat:
    ```
    Traceback (most recent call last):
      ValueError: invalid literal
    ```
    and asks what it means
  opening_user_message: "Hello, I pasted my error above."
  expected_outcome: the agent explains the error politely
  key_terms: []
  code_switching: false
  direction: inbound
  metrics: [UNIV-001]
- name: plain_case
  persona: caller who wants a quick answer
  opening_user_message: "Hello, this is caller 2."
  expected_outcome: the caller gets a correct answer
  key_terms: []
  code_switching: false
  direction: inbound
  metrics: [UNIV-001]
"""

# Inline backticks on one line: the closing ``` is followed by `"` and a newline.
ANSWER_WITH_INLINE_FENCE = """\
- name: inline_backticks
  persona: "caller who says ```hello``` as a joke"
  opening_user_message: "Hello, this is caller 1."
  expected_outcome: the caller gets a correct answer
  key_terms: []
  code_switching: false
  direction: inbound
  metrics: [UNIV-001]
- name: plain_case
  persona: caller who wants a quick answer
  opening_user_message: "Hello, this is caller 2."
  expected_outcome: the caller gets a correct answer
  key_terms: []
  code_switching: false
  direction: inbound
  metrics: [UNIV-001]
"""


@pytest.mark.parametrize(
    "answer", [ANSWER_WITH_INNER_FENCE, ANSWER_WITH_INLINE_FENCE], ids=["block_scalar_snippet", "inline_backticks"]
)
def test_unfenced_answer_with_a_fenced_snippet_inside_is_used_whole(impulse_project, mock_llm, cli, models_env, answer):
    expected = yaml.safe_load(answer)
    assert isinstance(expected, list) and len(expected) == 2   # the answer is a valid YAML list by itself
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(answer, _yaml(_items(2, persona="caller from the retry answer")))
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 1, "a valid unfenced answer must be used as it is, not reduced to its inner snippet"
    scenarios = _read_scenarios(_default_out(impulse_project))
    assert [scenario.name for scenario in scenarios] == [item["name"] for item in expected]
    assert [scenario.persona for scenario in scenarios] == [item["persona"] for item in expected]
    assert "```" in scenarios[0].persona


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda body: "```yaml\n" + body + "```", id="fenced"),
        pytest.param(lambda body: "```\n" + body + "```\n", id="fenced_no_language"),
        pytest.param(lambda body: "Here are the scenarios.\n\n```yaml\n" + body + "```\n", id="preamble_then_fence"),
    ],
)
def test_fenced_answer_is_still_unwrapped(impulse_project, mock_llm, cli, models_env, wrap):
    # Regression guard (passes before the fix too).
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(wrap(_yaml(_items(2))))
    result = _generate(cli, prompt, count=2)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 1
    assert [scenario.name for scenario in _read_scenarios(_default_out(impulse_project))] == [
        "generated_case_1",
        "generated_case_2",
    ]


# ---------------------------------------------------------------------------------------------
# Audit batch 3 (AUDIT.md finding 39): the YAML file impulse generate writes goes to a temporary
# file in the same folder that is then os.replace()d onto the target; on any error the temporary
# file is removed and an existing file is left untouched (the CLI exits 2: "cannot write ...").
# ---------------------------------------------------------------------------------------------

PREVIOUS_GENERATED = b"# previous generated scenarios (kept)\n- name: previous_case\n"


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


def test_generate_keeps_the_previous_file_when_the_final_replace_fails(
    impulse_project, mock_llm, cli, models_env, monkeypatch
):
    out = _default_out(impulse_project)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(PREVIOUS_GENERATED)
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(_yaml(_items(2)))
    _fail_replace_onto(monkeypatch, out)
    result = _generate(cli, prompt, count=2)
    assert out.read_bytes() == PREVIOUS_GENERATED, "the previous file was overwritten in place"
    assert sorted(path.name for path in out.parent.iterdir()) == ["generated.yaml"], "no temporary file is left"
    assert result.exit_code == 2, cli.text(result)


# ---------------------------------------------------------------------------------------------
# Review fix-up of audit batch 3 (group Z): an answer that makes PyYAML raise ValueError (a
# timestamp with an invalid day, ``2024-02-30``) or RecursionError (very deep nesting) while it is
# loaded is an invalid answer like any other: retried once with the error appended, then exit 1
# ("generation failed"), never exit 2 with "unexpected error".
# ---------------------------------------------------------------------------------------------

_OUTCOME_LINE = "expected_outcome: the caller gets a correct answer"
BAD_DATE_ANSWER = _yaml(_items(2)).replace(_OUTCOME_LINE, "expected_outcome: 2024-02-30")   # PyYAML: ValueError
DEEP_NESTING_ANSWER = "[" * 3000 + "]" * 3000                                              # PyYAML: RecursionError


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
def test_answer_that_makes_yaml_raise_is_retried_once(impulse_project, mock_llm, cli, models_env, first_answer):
    assert "expected_outcome: 2024-02-30" in BAD_DATE_ANSWER
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue(first_answer, _yaml(_items(2, persona="caller from the retry answer")))
    result = _generate(cli, prompt, count=2)
    assert "unexpected error" not in cli.text(result), cli.text(result)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 2
    _assert_retry_repeats_the_request(mock_llm)
    scenarios = _read_scenarios(_default_out(impulse_project))
    assert [scenario.persona for scenario in scenarios] == ["caller from the retry answer"] * 2


def test_two_answers_that_make_yaml_raise_value_error_exit_1_like_other_invalid_answers(
    impulse_project, mock_llm, cli, models_env
):
    assert "expected_outcome: 2024-02-30" in BAD_DATE_ANSWER
    prompt = _write_prompt(impulse_project.root)
    mock_llm.queue("- name: [unclosed", "still: [not yaml")
    reference = _generate(cli, prompt, count=2)
    assert reference.exit_code == 1, cli.text(reference)
    mock_llm.queue(BAD_DATE_ANSWER, BAD_DATE_ANSWER)
    result = _generate(cli, prompt, count=2)
    assert "unexpected error" not in cli.text(result), cli.text(result)
    assert result.exit_code == 1, cli.text(result)
    assert mock_llm.call_count == 4
    assert not _default_out(impulse_project).exists()
    assert _error_head(result) == _error_head(reference), "the same message as for any other invalid answer"
