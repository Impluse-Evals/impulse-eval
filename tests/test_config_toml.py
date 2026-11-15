# SPDX-License-Identifier: Apache-2.0
"""Phase 1: `.impulse/config.toml` loading (SPEC.md section 8.1; load_toml / TomlConfig, section 18.1)."""

import logging
from collections.abc import Mapping

import pytest

SECRET_ANTHROPIC = "sk-impulse-test-SECRET-anthropic-7f3a9c"
SECRET_VOICE = "impulse-test-SECRET-voice-41d2"
SECRET_ENV_VALUE = "sk-impulse-test-SECRET-env-value-90be"

# The section 8.1 template, verbatim.
TEMPLATE_TOML = """\
[keys]
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
IMPULSE_VOICE_API_KEY = ""

[models]
primary = "YOUR_PRIMARY_MODEL"
fast = "YOUR_FAST_MODEL"
simulator = "YOUR_SIMULATOR_MODEL"

[model_options.primary]       # [model_options.fast] and [model_options.simulator] have the same shape
api_base = ""
api_key_env = ""
extra_params = {}

[defaults]
reporter = "all"
language = "YOUR_LANGUAGE"    # written by impulse init from the user's answer or --language; there is no built-in default.
                              # Used only as the default for `impulse generate --language`; if it is still the placeholder
                              # and --language is not given, impulse generate exits 2 naming --language.
fail_on = "critical"          # critical | high | any — which conditions trigger exit code 1 (section 17)
"""

FILLED_TOML = f"""\
[keys]
ANTHROPIC_API_KEY = "{SECRET_ANTHROPIC}"
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
IMPULSE_VOICE_API_KEY = "{SECRET_VOICE}"

[models]
primary = "ollama/judge-primary"
fast = "ollama/judge-fast"
simulator = "ollama/simulator"

[model_options.primary]
api_base = "http://localhost:11434"
api_key_env = "MY_JUDGE_KEY"
extra_params = {{num_ctx = 8192}}

[model_options.fast]
api_base = ""
api_key_env = ""
extra_params = {{}}

[defaults]
reporter = "html"
language = "hi-IN"
fail_on = "high"
"""


def _config():
    from impulse_evals import config

    return config


def _write(tmp_path, text, name="config.toml"):
    folder = tmp_path / ".impulse"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def _get(obj, key):
    """Section or value access that works for pydantic models and plain mappings alike."""
    if isinstance(obj, Mapping):
        return obj[key]
    return getattr(obj, key)


def _plain(value):
    """Secret wrappers (e.g. pydantic SecretStr) are unwrapped; other values pass through."""
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return value


def _section_dict(section):
    if isinstance(section, Mapping):
        return dict(section)
    return section.model_dump()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in (
        "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "IMPULSE_VOICE_API_KEY",
        "IMPULSE_MODEL_PRIMARY", "IMPULSE_MODEL_FAST", "IMPULSE_MODEL_SIMULATOR", "MY_JUDGE_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_template_structure_loads(tmp_path):
    cfg = _config().load_toml(_write(tmp_path, TEMPLATE_TOML))
    keys = _get(cfg, "keys")
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "IMPULSE_VOICE_API_KEY"):
        assert _plain(_get(keys, name)) == ""
    models = _get(cfg, "models")
    assert _get(models, "primary") == "YOUR_PRIMARY_MODEL"
    assert _get(models, "fast") == "YOUR_FAST_MODEL"
    assert _get(models, "simulator") == "YOUR_SIMULATOR_MODEL"
    primary_options = _get(_get(cfg, "model_options"), "primary")
    assert _get(primary_options, "api_base") == ""
    assert _get(primary_options, "api_key_env") == ""
    assert _get(primary_options, "extra_params") == {}
    defaults = _get(cfg, "defaults")
    assert _get(defaults, "reporter") == "all"
    assert _get(defaults, "language") == "YOUR_LANGUAGE"
    assert _get(defaults, "fail_on") == "critical"


def test_filled_values_load(tmp_path):
    cfg = _config().load_toml(_write(tmp_path, FILLED_TOML))
    keys = _get(cfg, "keys")
    assert _plain(_get(keys, "ANTHROPIC_API_KEY")) == SECRET_ANTHROPIC
    assert _plain(_get(keys, "IMPULSE_VOICE_API_KEY")) == SECRET_VOICE
    models = _get(cfg, "models")
    assert _get(models, "primary") == "ollama/judge-primary"
    assert _get(models, "fast") == "ollama/judge-fast"
    assert _get(models, "simulator") == "ollama/simulator"
    model_options = _get(cfg, "model_options")
    primary = _get(model_options, "primary")
    assert _get(primary, "api_base") == "http://localhost:11434"
    assert _get(primary, "extra_params") == {"num_ctx": 8192}
    fast = _get(model_options, "fast")
    assert _get(fast, "extra_params") == {}
    defaults = _get(cfg, "defaults")
    assert _get(defaults, "reporter") == "html"
    assert _get(defaults, "language") == "hi-IN"


@pytest.mark.parametrize("level", ["critical", "high", "any"])
def test_fail_on_values_parsed(tmp_path, level):
    text = TEMPLATE_TOML.replace('fail_on = "critical"', f'fail_on = "{level}"')
    cfg = _config().load_toml(_write(tmp_path, text))
    assert _get(_get(cfg, "defaults"), "fail_on") == level


# Fixes round 4 (O3): the default is high (superseded assertions: it was critical).
def test_fail_on_defaults_to_high_when_missing(tmp_path):
    text = '[defaults]\nreporter = "all"\nlanguage = "en-IN"\n'
    cfg = _config().load_toml(_write(tmp_path, text))
    assert _get(_get(cfg, "defaults"), "fail_on") == "high"


def test_fail_on_defaults_to_high_without_defaults_section(tmp_path):
    text = '[models]\nprimary = "ollama/judge-primary"\n'
    cfg = _config().load_toml(_write(tmp_path, text))
    assert _get(_get(cfg, "defaults"), "fail_on") == "high"


def test_invalid_fail_on_is_config_error(tmp_path):
    config = _config()
    text = TEMPLATE_TOML.replace('fail_on = "critical"', 'fail_on = "sometimes"')
    with pytest.raises(config.ConfigError) as info:
        config.load_toml(_write(tmp_path, text))
    assert "fail_on" in info.value.field


def test_api_key_env_holds_variable_name_not_value(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_JUDGE_KEY", SECRET_ENV_VALUE)
    cfg = _config().load_toml(_write(tmp_path, FILLED_TOML))
    primary = _get(_get(cfg, "model_options"), "primary")
    assert _get(primary, "api_key_env") == "MY_JUDGE_KEY"
    assert SECRET_ENV_VALUE not in repr(cfg)
    assert SECRET_ENV_VALUE not in str(_section_dict(primary))


def test_keys_never_printed_or_logged(tmp_path, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    cfg = _config().load_toml(_write(tmp_path, FILLED_TOML))
    captured = capsys.readouterr()
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for secret in (SECRET_ANTHROPIC, SECRET_VOICE):
        assert secret not in captured.out
        assert secret not in captured.err
        assert secret not in logged
        assert secret not in repr(cfg)
        assert secret not in str(cfg)
        assert secret not in repr(_get(cfg, "keys"))


def test_keys_never_in_config_error(tmp_path, capsys, caplog):
    config = _config()
    caplog.set_level(logging.DEBUG)
    text = FILLED_TOML.replace('fail_on = "high"', 'fail_on = "sometimes"')
    with pytest.raises(config.ConfigError) as info:
        config.load_toml(_write(tmp_path, text))
    captured = capsys.readouterr()
    logged = "\n".join(record.getMessage() for record in caplog.records)
    error = info.value
    for secret in (SECRET_ANTHROPIC, SECRET_VOICE):
        assert secret not in str(error)
        assert secret not in repr(error)
        assert secret not in str(error.message)
        assert secret not in captured.out + captured.err
        assert secret not in logged


def test_config_error_carries_field_and_message():
    config = _config()
    error = config.ConfigError("defaults.fail_on", "must be one of critical, high, any")
    assert isinstance(error, Exception)
    assert error.field == "defaults.fail_on"
    assert error.message == "must be one of critical, high, any"


# ---------------------------------------------------------------------------------------------
# Audit batch 3 (AUDIT.md finding 35): the config.yaml scenario rules (SPEC.md section 8.2) reject
# a blank scenario name (empty or whitespace only; checked before the duplicate-name rule) with
# ConfigError("scenarios[i].name", "name must not be blank"), and a max_turns below 1 (checked
# before the inject_events turns) with ConfigError("scenarios[i].max_turns",
# "max_turns {n} must be at least 1"). load_config, impulse generate, and impulse convert-failure
# share the rule set (generate / convert-failure: test_generate_cli.py and test_regression.py).
# ---------------------------------------------------------------------------------------------

BLANK_NAME_MESSAGE = "name must not be blank"


def _max_turns_message(n):
    return f"max_turns {n} must be at least 1"


def _scenario_config_error(project, scenarios):
    """The ConfigError ``load_config`` raises for a config.yaml with ``scenarios``."""
    config = _config()
    path = project.write_config(scenarios)
    with pytest.raises(config.ConfigError) as info:
        project.load(path)
    return info.value


@pytest.mark.parametrize("max_turns", [0, -3])
def test_load_config_rejects_max_turns_below_1(impulse_project, max_turns):
    project = impulse_project
    error = _scenario_config_error(project, [project.scenario("zero_turns", max_turns=max_turns)])
    assert (error.field, error.message) == ("scenarios[0].max_turns", _max_turns_message(max_turns))


def test_max_turns_error_names_the_scenario_index(impulse_project):
    project = impulse_project
    error = _scenario_config_error(project, [project.scenario("ok_case"), project.scenario("zero_turns", max_turns=0)])
    assert (error.field, error.message) == ("scenarios[1].max_turns", _max_turns_message(0))


def test_max_turns_is_checked_before_the_inject_events_turns(impulse_project):
    project = impulse_project
    scenario = project.scenario(
        "zero_turns", max_turns=0, inject_events=[{"at_turn": 1, "type": "silence", "silence_ms": 500}]
    )
    error = _scenario_config_error(project, [scenario])
    assert (error.field, error.message) == ("scenarios[0].max_turns", _max_turns_message(0))


def test_max_turns_1_is_accepted(impulse_project):
    # Regression guard (passes before the fix too).
    project = impulse_project
    cfg = project.load(project.write_config([project.scenario("one_turn", max_turns=1)]))
    assert cfg.scenarios[0].max_turns == 1


@pytest.mark.parametrize("name", ["", "   ", "\t \n"], ids=["empty", "spaces", "whitespace"])
def test_load_config_rejects_a_blank_scenario_name(impulse_project, name):
    project = impulse_project
    error = _scenario_config_error(project, [project.scenario("ok_case"), project.scenario(name)])
    assert (error.field, error.message) == ("scenarios[1].name", BLANK_NAME_MESSAGE)


def test_blank_name_is_checked_before_the_duplicate_name_rule(impulse_project):
    project = impulse_project
    error = _scenario_config_error(project, [project.scenario("  "), project.scenario("  ")])
    assert (error.field, error.message) == ("scenarios[0].name", BLANK_NAME_MESSAGE)


def test_duplicate_names_are_still_rejected(impulse_project):
    # Regression guard (passes before the fix too).
    project = impulse_project
    error = _scenario_config_error(project, [project.scenario("same_case"), project.scenario("same_case")])
    assert error.field == "scenarios[1].name"
    assert "duplicate" in error.message


@pytest.mark.parametrize("fields, field", [
    pytest.param({"max_turns": 0}, "scenarios[0].max_turns", id="max_turns_0"),
    pytest.param({"name": " "}, "scenarios[0].name", id="blank_name"),
])
def test_dry_run_with_a_blank_name_or_max_turns_0_exits_2_naming_the_field(impulse_project, mock_llm, cli, fields, field):
    project = impulse_project
    project.write_agent()
    scenario = project.scenario("zero_turns")
    scenario.update(fields)
    result = cli.invoke("run", project.write_config([scenario]), "--dry-run")
    assert result.exit_code == 2, cli.text(result)
    assert field in cli.text(result)
    assert mock_llm.call_count == 0
    assert project.run_dirs() == []
