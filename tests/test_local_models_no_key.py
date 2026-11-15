# SPDX-License-Identifier: Apache-2.0
"""Phase 8: local models never require a key (SPEC.md section 8.1).

Model strings starting with ``ollama/`` or ``ollama_chat/``, and any role with an ``api_base``
(``config.yaml`` options or ``config.toml [model_options.<role>]``), need no key. A hosted model
string without ``api_base`` and without its key is a config error naming the key (exit 2).
No key variable is set in these tests and no ``.impulse/config.toml`` holds a key.
"""

import pytest

HOSTED = "openai/impulse-test-model"
API_BASE = "http://localhost:8000/v1"


def _roles(primary, fast, simulator, primary_options=None, fast_options=None, simulator_options=None):
    judge = {
        "primary_model": primary,
        "fast_model": fast,
        "primary_options": primary_options or {},
        "fast_options": fast_options or {},
    }
    simulator_block = {"model": simulator, "persona_style": "realistic", "mode": "cascade", "options": simulator_options or {}}
    return {"judge": judge, "simulator": simulator_block}


def _project(project, **roles):
    project.write_agent()
    return project.write_config(
        [project.scenario("no_key_call", max_turns=2, assertions=["dpdp.consent_before_pii"], metrics=["UNIV-001", "UNIV-013"])],
        **roles,
    )


async def _run(project):
    from impulse_evals.runner import run

    return await run(project.load())


def _models_called(mock_llm):
    return {call["model"] for call in mock_llm.calls}


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["ollama/", "ollama_chat/"])
async def test_ollama_models_run_without_any_key(impulse_project, llm_router, prefix):
    _project(
        impulse_project,
        **_roles(f"{prefix}judge-primary", f"{prefix}judge-fast", f"{prefix}simulator"),
    )
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    assert result.judge_model == f"{prefix}judge-primary"
    assert {f"{prefix}judge-primary", f"{prefix}judge-fast", f"{prefix}simulator"} <= _models_called(llm_router.mock)


@pytest.mark.asyncio
async def test_api_base_in_config_yaml_options_needs_no_key(impulse_project, llm_router):
    options = {"api_base": API_BASE}
    _project(
        impulse_project,
        **_roles(HOSTED, HOSTED, HOSTED, primary_options=options, fast_options=options, simulator_options=options),
    )
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    calls = llm_router.mock.calls
    assert calls
    assert all(call["model"] == HOSTED for call in calls)
    assert all(call["kwargs"].get("api_base") == API_BASE for call in llm_router.calls("judge"))
    assert all(call["kwargs"].get("api_base") == API_BASE for call in llm_router.calls("simulator"))


@pytest.mark.asyncio
async def test_api_base_in_config_toml_needs_no_key(impulse_project, llm_router):
    impulse_project.write_toml(
        "[model_options.primary]\n"
        f'api_base = "{API_BASE}"\n'
        "[model_options.fast]\n"
        f'api_base = "{API_BASE}"\n'
        "[model_options.simulator]\n"
        f'api_base = "{API_BASE}"\n'
    )
    _project(impulse_project, **_roles(HOSTED, HOSTED, HOSTED))
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    assert all(call["kwargs"].get("api_base") == API_BASE for call in llm_router.calls("judge"))


@pytest.mark.asyncio
async def test_hosted_model_without_api_base_needs_its_key(impulse_project, llm_router):
    from impulse_evals.config import ConfigError

    _project(impulse_project, **_roles(HOSTED, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert "OPENAI_API_KEY" in f"{info.value.field} {info.value.message}"
    assert not [call for call in llm_router.mock.calls if call["model"] == HOSTED]


def test_cli_run_with_local_models_and_no_key_exits_0(impulse_project, llm_router, cli):
    config = _project(impulse_project, **_roles("ollama/judge-primary", "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("run", config)
    assert result.exit_code == 0, cli.text(result)


def test_cli_run_hosted_model_without_key_exits_2(impulse_project, llm_router, cli):
    config = _project(impulse_project, **_roles(HOSTED, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("run", config)
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in cli.text(result)


def test_cli_doctor_local_and_api_base_roles_need_no_key(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    config = _project(
        impulse_project,
        **_roles("ollama/judge-primary", HOSTED, "ollama_chat/simulator", fast_options={"api_base": API_BASE}),
    )
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert "OPENAI_API_KEY" not in cli.text(result)


# ============================================================================================
# Audit batch 2 (AUDIT.md finding 10): the key variable a hosted model needs is the one LiteLLM
# itself reads for the provider (litellm.validate_environment, offline): together_ai ->
# TOGETHERAI_API_KEY, perplexity -> PERPLEXITYAI_API_KEY (never TOGETHER_AI_API_KEY /
# PERPLEXITY_API_KEY). A provider for which LiteLLM lists no *_API_KEY variable (credential-based:
# bedrock, vertex_ai, ...) needs no Impulse key: no ConfigError, no api_key passed (an api_key_env,
# when configured, is still used). gemini, openai, and anthropic are unchanged, and an unknown
# provider keeps its error. Everything stays offline (the autouse network guard fails the test
# otherwise).
# ============================================================================================

HOSTED_SUFFIX = "impulse-test-model"
LITELLM_KEY_PROVIDERS = [
    pytest.param("together_ai", "TOGETHERAI_API_KEY", "TOGETHER_AI_API_KEY", id="together_ai"),
    pytest.param("perplexity", "PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY", id="perplexity"),
]
CREDENTIAL_PROVIDERS = ["bedrock", "vertex_ai"]
UNKNOWN_MODEL = "impulse-unknown-provider/impulse-test-model"


def _hosted_calls(mock_llm, model):
    return [call for call in mock_llm.calls if call["model"] == model]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name, wrong_name", LITELLM_KEY_PROVIDERS)
async def test_hosted_key_is_read_from_the_variable_litellm_uses(
    impulse_project, llm_router, monkeypatch, provider, key_name, wrong_name
):
    secret = f"impulse-test-SECRET-{provider}-3301"
    model = f"{provider}/{HOSTED_SUFFIX}"
    monkeypatch.delenv(wrong_name, raising=False)
    monkeypatch.setenv(key_name, secret)
    _project(impulse_project, **_roles(model, model, model))
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    assert llm_router.calls("judge") and llm_router.calls("simulator")
    calls = llm_router.mock.calls
    assert all(call["model"] == model for call in calls)
    assert all(call["kwargs"].get("api_key") == secret for call in calls)
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name, wrong_name", LITELLM_KEY_PROVIDERS)
async def test_missing_hosted_key_names_the_variable_litellm_uses(
    impulse_project, llm_router, monkeypatch, provider, key_name, wrong_name
):
    from impulse_evals.config import ConfigError

    monkeypatch.delenv(key_name, raising=False)
    monkeypatch.setenv(wrong_name, "impulse-test-SECRET-wrong-variable-1188")   # never read
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == f"keys.{key_name}"
    assert f"{key_name} is not set" in info.value.message
    assert wrong_name not in f"{info.value.field} {info.value.message}"
    assert not _hosted_calls(llm_router.mock, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", CREDENTIAL_PROVIDERS)
async def test_credential_provider_runs_without_an_impulse_key(impulse_project, llm_router, monkeypatch, provider):
    monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = await _run(impulse_project)   # no ConfigError: LiteLLM uses the provider's own credentials
    assert result.scenario_results[0].error is None
    hosted = _hosted_calls(llm_router.mock, model)
    assert hosted, "the primary judge model was called"
    assert all("api_key" not in call["kwargs"] for call in hosted)


@pytest.mark.asyncio
async def test_credential_provider_still_uses_a_configured_api_key_env(impulse_project, llm_router, monkeypatch):
    # Regression guard (passes before the fix too): api_key_env, when configured, names the key to pass.
    secret = "impulse-test-SECRET-bedrock-token-5150"
    monkeypatch.setenv("IMPULSE_TEST_BEDROCK_TOKEN", secret)
    model = f"bedrock/{HOSTED_SUFFIX}"
    _project(
        impulse_project,
        **_roles(model, "ollama/judge-fast", "ollama/simulator", primary_options={"api_key_env": "IMPULSE_TEST_BEDROCK_TOKEN"}),
    )
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    hosted = _hosted_calls(llm_router.mock, model)
    assert hosted and all(call["kwargs"].get("api_key") == secret for call in hosted)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, key_name",
    [("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")],
)
async def test_standard_provider_keys_are_unchanged(impulse_project, llm_router, monkeypatch, provider, key_name):
    # Regression guard (passes before the fix too). For gemini LiteLLM lists GOOGLE_API_KEY and
    # GEMINI_API_KEY; Impulse keeps GEMINI_API_KEY (the standard name).
    from impulse_evals.config import ConfigError

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == f"keys.{key_name}"
    assert not _hosted_calls(llm_router.mock, model)

    secret = f"impulse-test-SECRET-{provider}-7731"
    monkeypatch.setenv(key_name, secret)
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    hosted = _hosted_calls(llm_router.mock, model)
    assert hosted and all(call["kwargs"].get("api_key") == secret for call in hosted)


@pytest.mark.asyncio
async def test_unknown_provider_error_is_unchanged(impulse_project, llm_router):
    # Regression guard (passes before the fix too).
    from impulse_evals.config import ConfigError

    _project(impulse_project, **_roles(UNKNOWN_MODEL, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == "judge.primary_model"
    assert info.value.message == f"cannot determine the provider of model {UNKNOWN_MODEL}"
    assert not _hosted_calls(llm_router.mock, UNKNOWN_MODEL)


@pytest.mark.parametrize("provider", CREDENTIAL_PROVIDERS)
def test_cli_run_credential_provider_without_key_exits_0(impulse_project, llm_router, cli, monkeypatch, provider):
    monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("run", config)
    assert result.exit_code == 0, cli.text(result)
    assert f"{provider.upper()}_API_KEY" not in cli.text(result)


@pytest.mark.parametrize("provider, key_name, wrong_name", LITELLM_KEY_PROVIDERS)
def test_cli_doctor_pings_with_the_key_litellm_uses(impulse_project, mock_llm, cli, monkeypatch, provider, key_name, wrong_name):
    mock_llm.set_default("OK")
    secret = f"impulse-test-SECRET-{provider}-doctor-2207"
    monkeypatch.delenv(wrong_name, raising=False)
    monkeypatch.setenv(key_name, secret)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert wrong_name not in cli.text(result)
    assert secret not in result.output
    pings = _hosted_calls(mock_llm, model)
    assert pings and all(call["kwargs"].get("api_key") == secret for call in pings)


@pytest.mark.parametrize("provider", CREDENTIAL_PROVIDERS)
def test_cli_doctor_credential_provider_needs_no_impulse_key(impulse_project, mock_llm, cli, monkeypatch, provider):
    mock_llm.set_default("OK")
    monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert f"{provider.upper()}_API_KEY" not in cli.text(result)
    pings = _hosted_calls(mock_llm, model)
    assert pings and all("api_key" not in call["kwargs"] for call in pings)


def test_cli_doctor_gemini_without_key_still_names_gemini_api_key(impulse_project, mock_llm, cli, monkeypatch):
    # Regression guard (passes before the fix too).
    mock_llm.set_default("OK")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    model = f"gemini/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2, cli.text(result)
    assert "GEMINI_API_KEY is not set" in cli.text(result)
    assert not _hosted_calls(mock_llm, model)


# ============================================================================================
# Audit batch 2 (AUDIT.md finding 11, the reproduced part only): importing the CLI sets
# LITELLM_LOCAL_MODEL_COST_MAP=True (setdefault) before anything can import LiteLLM, so LiteLLM
# uses its bundled cost map and makes no request at import; an explicit user value is kept.
# Each check runs in a fresh interpreter with the variable removed from (or preset in) its
# environment; a socket guard inside it refuses and records every non-loopback connection.
# ============================================================================================

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
COST_MAP_VARIABLE = "LITELLM_LOCAL_MODEL_COST_MAP"
SUBPROCESS_SOCKET_GUARD = r"""
import socket
attempts = []
_real_connect = socket.socket.connect
_real_getaddrinfo = socket.getaddrinfo


def _local(host):
    return host in (None, "", "localhost", "::1") or (isinstance(host, str) and host.startswith("127."))


def _connect(self, address):
    if isinstance(address, tuple) and address and not _local(address[0]):
        attempts.append("connect(%r)" % (address,))
        raise OSError("network blocked by the test")
    return _real_connect(self, address)


def _getaddrinfo(host, *args, **kwargs):
    if not _local(host):
        attempts.append("getaddrinfo(%r)" % (host,))
        raise socket.gaierror("network blocked by the test")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _connect
socket.getaddrinfo = _getaddrinfo
"""


def _fresh_interpreter(body, preset=None):
    """Run ``body`` (after the socket guard) in a fresh interpreter; returns its RESULT= JSON."""
    env = {name: value for name, value in os.environ.items() if name != COST_MAP_VARIABLE}
    if preset is not None:
        env[COST_MAP_VARIABLE] = preset
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = SUBPROCESS_SOCKET_GUARD + "\n" + body
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT=")]
    assert proc.returncode == 0 and lines, f"exit {proc.returncode}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
    return json.loads(lines[-1][len("RESULT="):])


IMPORT_CLI = """
import json, os, sys
import impulse_evals.cli
print("RESULT=" + json.dumps({"value": os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"),
                              "litellm_imported": "litellm" in sys.modules, "attempts": attempts}))
"""

IMPORT_CLI_THEN_LITELLM = """
import contextlib, io, json, os, sys
import impulse_evals.cli
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import litellm
print("RESULT=" + json.dumps({"value": os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"), "attempts": attempts}))
"""


def test_cli_import_sets_the_local_cost_map_variable():
    report = _fresh_interpreter(IMPORT_CLI)
    assert report["value"] == "True", report
    assert report["attempts"] == [], report


def test_litellm_imported_after_the_cli_makes_no_network_request():
    report = _fresh_interpreter(IMPORT_CLI_THEN_LITELLM)
    assert report["attempts"] == [], f"LiteLLM reached the network at import: {report['attempts']}"
    assert report["value"] == "True", report


@pytest.mark.parametrize("preset", ["False", "True"])
def test_cli_import_keeps_an_explicit_cost_map_setting(preset):
    # Regression guard (passes before the fix too): an explicit user value is respected.
    report = _fresh_interpreter(IMPORT_CLI, preset=preset)
    assert report["value"] == preset, report
    assert report["litellm_imported"] is False, "importing the CLI never imports LiteLLM"


# ============================================================================================
# Audit batch 2 review fix-ups (AUDIT.md finding 10, regression): a provider that LiteLLM's key
# table (litellm.validate_environment) does not list at all, so it names no variable (azure_ai,
# sambanova, databricks, hosted_vllm, ...), keeps the old rule {PROVIDER}_API_KEY, which for
# azure_ai and sambanova is exactly the variable LiteLLM itself reads (AZURE_AI_API_KEY,
# SAMBANOVA_API_KEY). That key is required (environment or config.toml [keys]; a missing one is
# ConfigError("keys.{PROVIDER}_API_KEY") before the first LLM call, exit 2), passed as api_key, and
# redacted. Only a provider whose table lists credentials none of which ends in _API_KEY (bedrock,
# vertex_ai: tests above; sagemaker below) needs no Impulse key.
# ============================================================================================

UNLISTED_KEY_PROVIDERS = [
    pytest.param("azure_ai", "AZURE_AI_API_KEY", id="azure_ai"),
    pytest.param("sambanova", "SAMBANOVA_API_KEY", id="sambanova"),
]


def _judge_calls_echo_env_key(llm_router, key_name):
    """Every judge call raises an authentication error that echoes the value of ``key_name`` from
    the environment (as a provider SDK echoes the key it read itself); other calls are unchanged."""
    previous = llm_router.mock.responder

    def responder(call):
        if llm_router.kind(call) == "judge":
            raise RuntimeError(f"401 Unauthorized: invalid api key {os.environ.get(key_name)} for this deployment")
        return previous(call)

    llm_router.mock.responder = responder


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name", UNLISTED_KEY_PROVIDERS)
async def test_unlisted_provider_key_from_config_toml_is_passed(impulse_project, llm_router, monkeypatch, provider, key_name):
    secret = f"impulse-test-SECRET-toml-{provider}-4410"
    monkeypatch.delenv(key_name, raising=False)
    impulse_project.write_toml(f'[keys]\n{key_name} = "{secret}"\n')
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, model, model))
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    assert llm_router.calls("judge") and llm_router.calls("simulator")
    calls = llm_router.mock.calls
    assert all(call["model"] == model for call in calls)
    assert [call["kwargs"].get("api_key") for call in calls] == [secret] * len(calls)
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name", UNLISTED_KEY_PROVIDERS)
async def test_unlisted_provider_env_key_is_redacted_from_an_echoed_judge_error(
    impulse_project, llm_router, monkeypatch, provider, key_name
):
    secret = f"impulse-test-SECRET-env-{provider}-6620"
    monkeypatch.setenv(key_name, secret)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    _judge_calls_echo_env_key(llm_router, key_name)
    result = await _run(impulse_project)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is not None
    assert secret not in scenario_result.error
    assert "invalid api key [REDACTED] for this deployment" in scenario_result.error, scenario_result.error
    for warning in result.warnings:
        assert secret not in warning, warning
    assert secret not in result.model_dump_json()
    hosted = _hosted_calls(llm_router.mock, model)
    assert hosted and all(call["kwargs"].get("api_key") == secret for call in hosted)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name", UNLISTED_KEY_PROVIDERS)
async def test_unlisted_provider_without_key_is_a_config_error_before_any_llm_call(
    impulse_project, llm_router, monkeypatch, provider, key_name
):
    from impulse_evals.config import ConfigError

    monkeypatch.delenv(key_name, raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == f"keys.{key_name}"
    assert f"{key_name} is not set" in info.value.message
    assert llm_router.mock.calls == []   # stopped before the first (paid) call


@pytest.mark.parametrize("provider, key_name", UNLISTED_KEY_PROVIDERS)
def test_cli_run_unlisted_provider_without_key_exits_2(impulse_project, llm_router, cli, monkeypatch, provider, key_name):
    monkeypatch.delenv(key_name, raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("run", config)
    assert result.exit_code == 2, cli.text(result)
    assert f"{key_name} is not set" in cli.text(result)
    assert llm_router.mock.calls == []


@pytest.mark.parametrize("provider, key_name", UNLISTED_KEY_PROVIDERS)
def test_cli_doctor_pings_unlisted_provider_with_its_config_toml_key(
    impulse_project, mock_llm, cli, monkeypatch, provider, key_name
):
    mock_llm.set_default("OK")
    secret = f"impulse-test-SECRET-doctor-{provider}-5521"
    monkeypatch.delenv(key_name, raising=False)
    impulse_project.write_toml(f'[keys]\n{key_name} = "{secret}"\n')
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert secret not in result.output
    pings = _hosted_calls(mock_llm, model)
    assert pings and all(call["kwargs"].get("api_key") == secret for call in pings)


@pytest.mark.parametrize("provider, key_name", UNLISTED_KEY_PROVIDERS)
def test_cli_doctor_unlisted_provider_without_key_names_it(impulse_project, mock_llm, cli, monkeypatch, provider, key_name):
    mock_llm.set_default("OK")
    monkeypatch.delenv(key_name, raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2, cli.text(result)
    assert f"{key_name} is not set" in cli.text(result)
    assert not _hosted_calls(mock_llm, model)


@pytest.mark.asyncio
async def test_sagemaker_still_runs_without_an_impulse_key(impulse_project, llm_router, monkeypatch):
    # Regression guard (passes before the fix too): LiteLLM lists AWS credentials for sagemaker, none
    # of them a *_API_KEY variable, so no Impulse key is required and no api_key is passed.
    monkeypatch.delenv("SAGEMAKER_API_KEY", raising=False)
    model = f"sagemaker/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    hosted = _hosted_calls(llm_router.mock, model)
    assert hosted and all("api_key" not in call["kwargs"] for call in hosted)


# ============================================================================================
# Audit batch 2 second review fix-ups (AUDIT.md finding 10, residue): a LiteLLM provider ALIAS needs
# the key of the provider it is an alias of, which is the variable LiteLLM 1.105.1 itself reads for
# it (checked offline by intercepting the request LiteLLM builds, one candidate variable set at a
# time): text-completion-openai -> OPENAI_API_KEY (Impulse asked for TEXT-COMPLETION-OPENAI_API_KEY,
# a name most shells cannot set), anthropic_text -> ANTHROPIC_API_KEY, azure_text -> AZURE_API_KEY,
# watsonx_text -> WATSONX_API_KEY (the watsonx/ name); LiteLLM never reads the *_TEXT_API_KEY names.
# text-completion-codestral / text-completion-inception already map to CODESTRAL_API_KEY /
# INCEPTION_API_KEY (LiteLLM's table names them): regression guards. A key name Impulse builds itself
# (a provider LiteLLM's table does not list) is always a valid environment variable name, also for a
# provider whose name has a hyphen (nano-gpt, scx-ai).
# ============================================================================================

import re  # noqa: E402

ALIAS_KEY_PROVIDERS = [
    pytest.param("text-completion-openai", "OPENAI_API_KEY", "TEXT-COMPLETION-OPENAI_API_KEY", id="text-completion-openai"),
    pytest.param("anthropic_text", "ANTHROPIC_API_KEY", "ANTHROPIC_TEXT_API_KEY", id="anthropic_text"),
    pytest.param("azure_text", "AZURE_API_KEY", "AZURE_TEXT_API_KEY", id="azure_text"),
    pytest.param("watsonx_text", "WATSONX_API_KEY", "WATSONX_TEXT_API_KEY", id="watsonx_text"),
    # regression guards (pass before the fix): LiteLLM's table already names the aliased provider's key
    pytest.param(
        "text-completion-codestral", "CODESTRAL_API_KEY", "TEXT-COMPLETION-CODESTRAL_API_KEY", id="text-completion-codestral"
    ),
    pytest.param(
        "text-completion-inception", "INCEPTION_API_KEY", "TEXT-COMPLETION-INCEPTION_API_KEY", id="text-completion-inception"
    ),
]
HYPHENATED_PROVIDERS = ["text-completion-openai", "nano-gpt", "scx-ai"]
ENV_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _without_api_key_variables(monkeypatch):
    """Remove every ``*_API_KEY`` variable from the environment (restored after the test)."""
    for name in list(os.environ):
        if name.upper().endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name, wrong_name", ALIAS_KEY_PROVIDERS)
async def test_alias_provider_key_is_read_from_the_aliased_providers_variable(
    impulse_project, llm_router, monkeypatch, provider, key_name, wrong_name
):
    secret = f"impulse-test-SECRET-alias-{provider}-8812"
    monkeypatch.delenv(wrong_name, raising=False)
    monkeypatch.setenv(key_name, secret)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, model, model))
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    assert llm_router.calls("judge") and llm_router.calls("simulator")
    calls = llm_router.mock.calls
    assert all(call["model"] == model for call in calls)
    assert [call["kwargs"].get("api_key") for call in calls] == [secret] * len(calls)
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name, wrong_name", ALIAS_KEY_PROVIDERS)
async def test_alias_provider_without_the_aliased_key_is_a_config_error_naming_it(
    impulse_project, llm_router, monkeypatch, provider, key_name, wrong_name
):
    from impulse_evals.config import ConfigError

    monkeypatch.delenv(key_name, raising=False)
    monkeypatch.delenv(wrong_name, raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == f"keys.{key_name}"
    assert info.value.message.startswith(f"{key_name} is not set"), info.value.message
    assert wrong_name not in f"{info.value.field} {info.value.message}"
    assert llm_router.mock.calls == []   # stopped before the first (paid) call


@pytest.mark.parametrize("provider, key_name, wrong_name", ALIAS_KEY_PROVIDERS)
def test_cli_run_alias_provider_without_key_exits_2_naming_the_aliased_key(
    impulse_project, llm_router, cli, monkeypatch, provider, key_name, wrong_name
):
    monkeypatch.delenv(key_name, raising=False)
    monkeypatch.delenv(wrong_name, raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("run", config)
    assert result.exit_code == 2, cli.text(result)
    assert f"{key_name} is not set" in cli.text(result)
    assert wrong_name not in cli.text(result)
    assert llm_router.mock.calls == []


@pytest.mark.parametrize("provider, key_name, wrong_name", ALIAS_KEY_PROVIDERS)
def test_cli_doctor_pings_alias_provider_with_the_aliased_key(
    impulse_project, mock_llm, cli, monkeypatch, provider, key_name, wrong_name
):
    mock_llm.set_default("OK")
    secret = f"impulse-test-SECRET-alias-doctor-{provider}-3390"
    monkeypatch.delenv(wrong_name, raising=False)
    monkeypatch.setenv(key_name, secret)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert wrong_name not in cli.text(result)
    assert secret not in result.output
    pings = _hosted_calls(mock_llm, model)
    assert pings and all(call["kwargs"].get("api_key") == secret for call in pings)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", HYPHENATED_PROVIDERS)
async def test_key_name_impulse_asks_for_is_a_valid_environment_variable_name(
    impulse_project, llm_router, monkeypatch, provider
):
    from impulse_evals.config import ConfigError

    _without_api_key_variables(monkeypatch)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert llm_router.mock.calls == []
    assert info.value.field.startswith("keys."), info.value.field
    name = info.value.field[len("keys."):]
    assert ENV_VAR_NAME.fullmatch(name), f"{name!r} is not a valid environment variable name"
    assert info.value.message.startswith(f"{name} is not set"), info.value.message
    secret = f"impulse-test-SECRET-hyphenated-{provider}-6071"
    monkeypatch.setenv(name, secret)   # the variable the error names is the one that is read
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    hosted = _hosted_calls(llm_router.mock, model)
    assert hosted and all(call["kwargs"].get("api_key") == secret for call in hosted)


# ============================================================================================
# Audit batch 2 second review fix-ups, part 2 (AUDIT.md finding 10, residue):
# (a) A provider that LiteLLM's openai_like provider registry (litellm/llms/openai_like/providers.json)
#     gives an api_key_env needs exactly that variable, the one LiteLLM 1.105.1 itself reads (checked
#     offline by intercepting the request LiteLLM builds, one candidate set at a time): nano-gpt ->
#     NANOGPT_API_KEY, scx-ai -> SCX_API_KEY, veniceai -> VENICE_AI_API_KEY, scaleway -> SCW_SECRET_KEY,
#     apertis -> STIMA_API_KEY, tensormesh -> TENSORMESH_INFERENCE_API_KEY (never the
#     {PROVIDER}_API_KEY fallback). poe (api_key_env POE_API_KEY) is a regression guard.
# (b) The key lookup never touches the network and never starts a login: resolving github_copilot/
#     and chatgpt/ through LiteLLM (get_llm_provider / validate_environment) runs an OAuth device-code
#     flow (LiteLLM's own PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO); it POSTs to
#     github.com/login/device/code or auth.openai.com, then polls and sleeps waiting for a human.
#     These providers need no Impulse key and no api_key is passed. Checked through runner.run (mocked
#     acompletion) and impulse doctor: no network attempt (the autouse guard's record), no ConfigError.
#     LiteLLM's token folders point at empty folders under tmp_path, so no test reads or writes the
#     developer's ~/.config/litellm, and each call runs under a real-time limit so a waiting login fails
#     the test instead of hanging the suite.
# ============================================================================================

import asyncio  # noqa: E402
import threading  # noqa: E402

OPENAI_LIKE_KEY_PROVIDERS = [
    pytest.param("nano-gpt", "NANOGPT_API_KEY", id="nano-gpt"),
    pytest.param("scx-ai", "SCX_API_KEY", id="scx-ai"),
    pytest.param("veniceai", "VENICE_AI_API_KEY", id="veniceai"),
    pytest.param("scaleway", "SCW_SECRET_KEY", id="scaleway"),
    pytest.param("apertis", "STIMA_API_KEY", id="apertis"),
    pytest.param("tensormesh", "TENSORMESH_INFERENCE_API_KEY", id="tensormesh"),
    pytest.param("poe", "POE_API_KEY", id="poe"),  # regression guard (passes before the fix)
]
AUTH_FLOW_PROVIDERS = ["github_copilot", "chatgpt"]
AUTH_FLOW_TIMEOUT_S = 60.0


def _isolated_litellm_token_dirs(monkeypatch, root):
    """LiteLLM's github_copilot and chatgpt token folders: empty folders under ``root`` (a device-code
    request recorded in the developer's folder would make the chatgpt login wait out a cooldown)."""
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(root / "litellm-token-dir-github-copilot"))
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(root / "litellm-token-dir-chatgpt"))


def _finishes_within(seconds, function):
    """``(value, error)`` of ``function()`` run in a daemon thread. Fails the test, instead of hanging
    the suite, when it has not returned within ``seconds`` of real time (a login waiting for a human)."""
    outcome = {}

    def target():
        try:
            outcome["value"] = function()
        except BaseException as exc:  # noqa: BLE001 - handed back to the test
            outcome["error"] = exc

    worker = threading.Thread(target=target, name="impulse-test-login-guard", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        pytest.fail(f"did not return within {seconds:.0f} s: a login flow is waiting (device-code authentication?)")
    return outcome.get("value"), outcome.get("error")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name", OPENAI_LIKE_KEY_PROVIDERS)
async def test_openai_like_provider_key_is_the_api_key_env_litellm_names(
    impulse_project, llm_router, monkeypatch, provider, key_name
):
    secret = f"impulse-test-SECRET-openai-like-{provider}-2468"
    monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)   # the old fallback name, never read
    monkeypatch.setenv(key_name, secret)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, model, model))
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    assert llm_router.calls("judge") and llm_router.calls("simulator")
    calls = llm_router.mock.calls
    assert all(call["model"] == model for call in calls)
    assert [call["kwargs"].get("api_key") for call in calls] == [secret] * len(calls)
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name", OPENAI_LIKE_KEY_PROVIDERS)
async def test_openai_like_provider_without_its_api_key_env_is_a_config_error_naming_it(
    impulse_project, llm_router, monkeypatch, provider, key_name
):
    from impulse_evals.config import ConfigError

    monkeypatch.delenv(key_name, raising=False)
    monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == f"keys.{key_name}"
    assert info.value.message.startswith(f"{key_name} is not set"), info.value.message
    assert llm_router.mock.calls == []   # stopped before the first (paid) call


@pytest.mark.parametrize("provider", AUTH_FLOW_PROVIDERS)
def test_authenticating_provider_run_starts_no_login_and_needs_no_impulse_key(
    impulse_project, llm_router, monkeypatch, tmp_path, _block_network, provider
):
    _isolated_litellm_token_dirs(monkeypatch, tmp_path)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, model, model))
    result, error = _finishes_within(AUTH_FLOW_TIMEOUT_S, lambda: asyncio.run(_run(impulse_project)))
    assert _block_network == [], f"the run reached the network (a login flow): {_block_network}; run error: {error!r}"
    assert error is None, f"{type(error).__name__}: {error}"
    assert result.scenario_results[0].error is None
    assert llm_router.calls("judge") and llm_router.calls("simulator")
    calls = llm_router.mock.calls
    assert all(call["model"] == model for call in calls)
    assert all("api_key" not in call["kwargs"] for call in calls)   # LiteLLM uses the provider's own login


@pytest.mark.parametrize("provider", AUTH_FLOW_PROVIDERS)
def test_cli_doctor_authenticating_provider_starts_no_login_and_needs_no_impulse_key(
    impulse_project, mock_llm, cli, monkeypatch, tmp_path, _block_network, provider
):
    mock_llm.set_default("OK")
    _isolated_litellm_token_dirs(monkeypatch, tmp_path)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, model, model))
    result, error = _finishes_within(AUTH_FLOW_TIMEOUT_S, lambda: cli.invoke("doctor", config))
    assert _block_network == [], f"impulse doctor reached the network (a login flow): {_block_network}"
    assert error is None, f"{type(error).__name__}: {error}"
    assert result.exit_code == 0, cli.text(result)
    assert "is not set" not in cli.text(result)
    assert "cannot determine the provider" not in cli.text(result)
    pings = _hosted_calls(mock_llm, model)
    assert pings and all("api_key" not in call["kwargs"] for call in pings)


# ============================================================================================
# Audit batch 2 re-check round 3 (AUDIT.md finding 10, residue): more key names LiteLLM 1.105.1
# never reads, each checked offline by intercepting the request LiteLLM builds (httpx, requests and
# aiohttp sends recorded and refused; sockets blocked; one candidate variable set at a time):
#   cohere_chat    -> COHERE_API_KEY      (alias of cohere; never COHERE_CHAT_API_KEY)
#   aiohttp_openai -> OPENAI_API_KEY      (alias of openai; never AIOHTTP_OPENAI_API_KEY)
#   meta_llama     -> LLAMA_API_KEY       (never META_LLAMA_API_KEY)
#   lambda_ai      -> LAMBDA_API_KEY      (never LAMBDA_AI_API_KEY)
#   snowflake      -> SNOWFLAKE_JWT       (never SNOWFLAKE_API_KEY)
# Credential-based, no Impulse key and no api_key passed (LiteLLM sends no *_API_KEY variable for
# them): vertex_ai_beta (as vertex_ai: Google Cloud credentials), sagemaker_chat and sagemaker_nova
# (as sagemaker: the AWS access key signs the request), oci (OCI_USER / OCI_FINGERPRINT / OCI_TENANCY /
# OCI_KEY request signing).
# ============================================================================================

NAMED_KEY_PROVIDERS = [
    pytest.param("cohere_chat", "COHERE_API_KEY", "COHERE_CHAT_API_KEY", id="cohere_chat"),
    pytest.param("aiohttp_openai", "OPENAI_API_KEY", "AIOHTTP_OPENAI_API_KEY", id="aiohttp_openai"),
    pytest.param("meta_llama", "LLAMA_API_KEY", "META_LLAMA_API_KEY", id="meta_llama"),
    pytest.param("lambda_ai", "LAMBDA_API_KEY", "LAMBDA_AI_API_KEY", id="lambda_ai"),
    pytest.param("snowflake", "SNOWFLAKE_JWT", "SNOWFLAKE_API_KEY", id="snowflake"),
]
MORE_CREDENTIAL_PROVIDERS = ["vertex_ai_beta", "sagemaker_chat", "sagemaker_nova", "oci"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name, wrong_name", NAMED_KEY_PROVIDERS)
async def test_named_provider_key_is_read_from_the_variable_litellm_reads(
    impulse_project, llm_router, monkeypatch, provider, key_name, wrong_name
):
    secret = f"impulse-test-SECRET-named-{provider}-5813"
    monkeypatch.delenv(wrong_name, raising=False)
    monkeypatch.setenv(key_name, secret)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, model, model))
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    assert llm_router.calls("judge") and llm_router.calls("simulator")
    calls = llm_router.mock.calls
    assert all(call["model"] == model for call in calls)
    assert [call["kwargs"].get("api_key") for call in calls] == [secret] * len(calls)
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, key_name, wrong_name", NAMED_KEY_PROVIDERS)
async def test_named_provider_without_the_variable_litellm_reads_is_a_config_error_naming_it(
    impulse_project, llm_router, monkeypatch, provider, key_name, wrong_name
):
    from impulse_evals.config import ConfigError

    monkeypatch.delenv(key_name, raising=False)
    monkeypatch.delenv(wrong_name, raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == f"keys.{key_name}"
    assert info.value.message.startswith(f"{key_name} is not set"), info.value.message
    assert wrong_name not in f"{info.value.field} {info.value.message}"
    assert llm_router.mock.calls == []   # stopped before the first (paid) call


@pytest.mark.parametrize("provider, key_name, wrong_name", NAMED_KEY_PROVIDERS)
def test_cli_doctor_pings_named_provider_with_the_variable_litellm_reads(
    impulse_project, mock_llm, cli, monkeypatch, provider, key_name, wrong_name
):
    mock_llm.set_default("OK")
    secret = f"impulse-test-SECRET-named-doctor-{provider}-7702"
    monkeypatch.delenv(wrong_name, raising=False)
    monkeypatch.setenv(key_name, secret)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert wrong_name not in cli.text(result)
    assert secret not in result.output
    pings = _hosted_calls(mock_llm, model)
    assert pings and all(call["kwargs"].get("api_key") == secret for call in pings)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", MORE_CREDENTIAL_PROVIDERS)
async def test_more_credential_providers_run_without_an_impulse_key(impulse_project, llm_router, monkeypatch, provider):
    monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    _project(impulse_project, **_roles(model, model, model))
    result = await _run(impulse_project)   # no ConfigError: LiteLLM uses the provider's own credentials
    assert result.scenario_results[0].error is None
    assert llm_router.calls("judge") and llm_router.calls("simulator")
    calls = llm_router.mock.calls
    assert all(call["model"] == model for call in calls)
    assert all("api_key" not in call["kwargs"] for call in calls)


@pytest.mark.parametrize("provider", MORE_CREDENTIAL_PROVIDERS)
def test_cli_doctor_more_credential_providers_need_no_impulse_key(impulse_project, mock_llm, cli, monkeypatch, provider):
    mock_llm.set_default("OK")
    monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
    model = f"{provider}/{HOSTED_SUFFIX}"
    config = _project(impulse_project, **_roles(model, "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert f"{provider.upper()}_API_KEY" not in cli.text(result)
    pings = _hosted_calls(mock_llm, model)
    assert pings and all("api_key" not in call["kwargs"] for call in pings)


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 44, maintainer decision #44): resolve_model_options skips
# config.yaml option values that are None or "" (not 0 / False), the same as blank config.toml
# values, so a blank YAML value never masks the config.toml one: `primary_options: {api_base: ""}`
# keeps the [model_options.primary] api_base (the role still needs no key, and its calls go to that
# server); a blank YAML api_key_env keeps the config.toml api_key_env. Zero and false YAML values
# still override.
# ============================================================================================

TOML_API_BASE = "http://localhost:11434"
JUDGE_KEY_VARIABLE = "IMPULSE_TEST_JUDGE_KEY"


def _primary_calls(llm_router, model=HOSTED):
    return [call for call in llm_router.calls("judge") if call["model"] == model]


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", None], ids=["empty_string", "null"])
async def test_blank_yaml_api_base_keeps_the_config_toml_api_base(impulse_project, llm_router, monkeypatch, blank):
    monkeypatch.delenv(JUDGE_KEY_VARIABLE, raising=False)
    impulse_project.write_toml(
        f'[model_options.primary]\napi_base = "{TOML_API_BASE}"\napi_key_env = "{JUDGE_KEY_VARIABLE}"\n'
    )
    _project(impulse_project, **_roles(HOSTED, "ollama/judge-fast", "ollama/simulator", primary_options={"api_base": blank}))
    result = await _run(impulse_project)   # no ConfigError: the role has an api_base, so it needs no key
    assert result.scenario_results[0].error is None
    calls = _primary_calls(llm_router)
    assert calls, "the primary judge model was called"
    assert [call["kwargs"].get("api_base") for call in calls] == [TOML_API_BASE] * len(calls)
    assert all("api_key" not in call["kwargs"] for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", None], ids=["empty_string", "null"])
async def test_blank_yaml_api_key_env_keeps_the_config_toml_api_key_env(impulse_project, llm_router, monkeypatch, blank):
    secret = "impulse-test-SECRET-judge-key-4444"
    monkeypatch.setenv(JUDGE_KEY_VARIABLE, secret)
    impulse_project.write_toml(f'[model_options.primary]\napi_key_env = "{JUDGE_KEY_VARIABLE}"\n')
    _project(
        impulse_project, **_roles(HOSTED, "ollama/judge-fast", "ollama/simulator", primary_options={"api_key_env": blank})
    )
    result = await _run(impulse_project)   # the key is read from the config.toml api_key_env variable
    assert result.scenario_results[0].error is None
    calls = _primary_calls(llm_router)
    assert calls and [call["kwargs"].get("api_key") for call in calls] == [secret] * len(calls)
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
async def test_non_blank_yaml_api_base_still_overrides_config_toml(impulse_project, llm_router):
    # Regression guard (passes before the fix too): config.yaml overrides config.toml key by key.
    impulse_project.write_toml(f'[model_options.primary]\napi_base = "{TOML_API_BASE}"\n')
    _project(impulse_project, **_roles(HOSTED, "ollama/judge-fast", "ollama/simulator", primary_options={"api_base": API_BASE}))
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    calls = _primary_calls(llm_router)
    assert calls and [call["kwargs"].get("api_base") for call in calls] == [API_BASE] * len(calls)


@pytest.mark.asyncio
async def test_zero_and_false_yaml_option_values_still_override(impulse_project, llm_router):
    # Regression guard (passes before the fix too): only None and "" are blank; 0 and False are values.
    impulse_project.write_toml("[model_options.simulator]\nextra_params = { temperature = 0.7, drop_params = true }\n")
    _project(
        impulse_project,
        **_roles(
            "ollama/judge-primary", "ollama/judge-fast", "ollama/simulator",
            simulator_options={"temperature": 0, "drop_params": False},
        ),
    )
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    calls = llm_router.calls("simulator")
    assert calls, "the simulator was called"
    for call in calls:
        assert call["kwargs"].get("temperature") == 0 and call["kwargs"].get("temperature") is not None
        assert call["kwargs"].get("drop_params") is False


def test_cli_doctor_blank_yaml_api_base_keeps_the_config_toml_api_base(impulse_project, mock_llm, cli, monkeypatch):
    mock_llm.set_default("OK")
    monkeypatch.delenv(JUDGE_KEY_VARIABLE, raising=False)
    impulse_project.write_toml(
        f'[model_options.primary]\napi_base = "{TOML_API_BASE}"\napi_key_env = "{JUDGE_KEY_VARIABLE}"\n'
    )
    config = _project(
        impulse_project, **_roles(HOSTED, "ollama/judge-fast", "ollama/simulator", primary_options={"api_base": ""})
    )
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert f"{JUDGE_KEY_VARIABLE} is not set" not in cli.text(result)
    pings = _hosted_calls(mock_llm, HOSTED)
    assert pings and all(call["kwargs"].get("api_base") == TOML_API_BASE for call in pings)


# ============================================================================================
# Audit batch 3, maintainer decision CLP (2026-10-20, found by the batch 2 Stage D review): the key
# lookup honours the role's custom_llm_provider option, from config.yaml role options or from
# config.toml [model_options.<role>] extra_params (after the usual merge). The provider comes from
# litellm.get_llm_provider(model, custom_llm_provider=custom) and the key table is read for
# "{custom}/{model}", so `model: impulse-test-model` (no provider prefix) with
# `custom_llm_provider: groq` needs, and passes, GROQ_API_KEY instead of failing with "cannot
# determine the provider of model impulse-test-model". The doctor's offline provider check reads
# the option from the same places, so `impulse run` and `impulse doctor` agree. Without the option,
# behaviour is unchanged.
# ============================================================================================

CLP_MODEL = "impulse-test-model"   # no provider prefix: LiteLLM cannot place it on its own
CLP_KEY = "gsk_impulse_test_SECRET_clp_5151"
CLP_PLACEMENTS = ["yaml_options", "toml_extra_params"]


def _clp_project(project, placement, *, key=CLP_KEY):
    toml_lines = []
    if key is not None:
        toml_lines += ["[keys]", f'GROQ_API_KEY = "{key}"', ""]
    if placement == "toml_extra_params":
        toml_lines += ["[model_options.primary]", 'extra_params = { custom_llm_provider = "groq" }', ""]
    if toml_lines:
        project.write_toml("\n".join(toml_lines))
    primary_options = {"custom_llm_provider": "groq"} if placement == "yaml_options" else {}
    return _project(
        project, **_roles(CLP_MODEL, "ollama/judge-fast", "ollama/simulator", primary_options=primary_options)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", CLP_PLACEMENTS)
async def test_custom_llm_provider_role_uses_that_providers_key(impulse_project, llm_router, monkeypatch, placement):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _clp_project(impulse_project, placement)
    result = await _run(impulse_project)
    assert result.scenario_results[0].error is None
    calls = _primary_calls(llm_router, CLP_MODEL)
    assert calls, "the primary judge model was called"
    assert [call["kwargs"].get("api_key") for call in calls] == [CLP_KEY] * len(calls)
    assert all(call["kwargs"].get("custom_llm_provider") == "groq" for call in calls)
    assert CLP_KEY not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", CLP_PLACEMENTS)
async def test_custom_llm_provider_role_without_its_key_is_a_config_error_naming_it(
    impulse_project, llm_router, monkeypatch, placement
):
    from impulse_evals.config import ConfigError

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _clp_project(impulse_project, placement, key=None)
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == "keys.GROQ_API_KEY", f"{info.value.field}: {info.value.message}"
    assert info.value.message.startswith("GROQ_API_KEY is not set"), info.value.message
    assert llm_router.mock.calls == []   # stopped before the first (paid) call


@pytest.mark.parametrize("placement", CLP_PLACEMENTS)
def test_cli_run_custom_llm_provider_role_exits_0(impulse_project, llm_router, cli, monkeypatch, placement):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = _clp_project(impulse_project, placement)
    result = cli.invoke("run", config)
    assert result.exit_code == 0, cli.text(result)
    assert "cannot determine the provider" not in cli.text(result)
    assert CLP_KEY not in result.output


@pytest.mark.parametrize("placement", CLP_PLACEMENTS)
def test_cli_doctor_custom_llm_provider_role_passes(impulse_project, mock_llm, cli, monkeypatch, placement):
    mock_llm.set_default("OK")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = _clp_project(impulse_project, placement)
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert "FAIL primary model" not in text, text
    assert "cannot determine the provider" not in text and "does not recognise" not in text, text
    pings = _hosted_calls(mock_llm, CLP_MODEL)
    assert pings and all(call["kwargs"].get("api_key") == CLP_KEY for call in pings)
    assert all(call["kwargs"].get("custom_llm_provider") == "groq" for call in pings)
    assert CLP_KEY not in result.output


@pytest.mark.asyncio
async def test_bare_model_without_custom_llm_provider_is_still_an_unknown_provider(impulse_project, llm_router, monkeypatch):
    # Regression guard (passes before the fix too): without the option nothing changes.
    from impulse_evals.config import ConfigError

    monkeypatch.setenv("GROQ_API_KEY", CLP_KEY)
    _project(impulse_project, **_roles(CLP_MODEL, "ollama/judge-fast", "ollama/simulator"))
    with pytest.raises(ConfigError) as info:
        await _run(impulse_project)
    assert info.value.field == "judge.primary_model"
    assert info.value.message == f"cannot determine the provider of model {CLP_MODEL}"
    assert llm_router.mock.calls == []


# ============================================================================================
# Audit batch 3 review fix-ups (AUDIT.md finding 43 x finding 10): the doctor / init 5-token ping
# discards what is printed during the call (LiteLLM's "Give Feedback / Get Help" banner) only for
# roles whose provider does not authenticate itself. For a role whose provider LiteLLM authenticates
# (config.authenticating_provider: github_copilot, chatgpt; named by the model prefix or by the
# role's custom_llm_provider option), LiteLLM prints its OAuth device-code sign-in prompt
# ("Please visit ... and enter code ...") from inside litellm.completion; discarded, the user never
# sees the code and the ping waits for a login nobody can complete. Those pings run without the
# stdout / stderr redirect (litellm.suppress_debug_info = True still stops the banner); every other
# role keeps it. The mocked completion prints the sign-in prompt for the authenticating role (as
# LiteLLM does, to stdout) and answers OK; an ordinary role's ping prints the banner to stdout and
# stderr whatever the flag says, then fails.
# ============================================================================================

SIGN_IN_PROMPTS = {
    "github_copilot": "Please visit https://github.com/login/device and enter code IMPU-7341 to authenticate.",
    "chatgpt": (
        "Sign in with ChatGPT using device code:\n"
        "1) Visit https://auth.openai.com/codex/device\n"
        "2) Enter code: IMPU-7342\n"
        "Device codes are a common phishing target. Never share this code."
    ),
}
PING_BANNER = "Give Feedback / Get Help"
PING_BANNER_LINE = f"\033[1;31m{PING_BANNER}: https://github.com/BerriAI/litellm/issues/new\033[0m"
PING_INFO_LINE = "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'."
ORDINARY_PING_MODEL = "ollama/impulse-test-ordinary"
SIGN_IN_PLACEMENTS = ["model_prefix", "custom_llm_provider"]


def _sign_in_role(provider, placement):
    """``(model, options)`` of a role whose provider authenticates, named by the model prefix or by
    the role's ``custom_llm_provider`` option (bare model string)."""
    if placement == "model_prefix":
        return f"{provider}/{HOSTED_SUFFIX}", {}
    return HOSTED_SUFFIX, {"custom_llm_provider": provider}


def _squeezed(text):
    """``text`` with every whitespace run collapsed to one space (as ``cli.text`` does)."""
    return " ".join(text.split())


def _sign_in_ping_responder(auth_model, prompt):
    """Pings of ``auth_model`` print ``prompt`` to stdout (LiteLLM's device-code prompt) and answer
    OK; every other ping prints LiteLLM's banner to stdout and stderr, then fails."""

    def responder(call):
        if call["model"] == auth_model:
            print(prompt, flush=True)
            return "OK"
        for stream in (sys.stdout, sys.stderr):
            print(file=stream)
            print(PING_BANNER_LINE, file=stream)
            print(PING_INFO_LINE, file=stream)
            print(file=stream)
        raise ConnectionError("model server is not reachable")

    return responder


@pytest.mark.parametrize("placement", SIGN_IN_PLACEMENTS)
@pytest.mark.parametrize("provider", AUTH_FLOW_PROVIDERS)
def test_cli_doctor_shows_the_sign_in_prompt_of_an_authenticating_provider(
    impulse_project, mock_llm, cli, monkeypatch, tmp_path, _block_network, provider, placement
):
    _isolated_litellm_token_dirs(monkeypatch, tmp_path)
    model, options = _sign_in_role(provider, placement)
    prompt = SIGN_IN_PROMPTS[provider]
    mock_llm.responder = _sign_in_ping_responder(model, prompt)
    config = _project(
        impulse_project, **_roles(model, ORDINARY_PING_MODEL, ORDINARY_PING_MODEL, primary_options=options)
    )
    result, error = _finishes_within(AUTH_FLOW_TIMEOUT_S, lambda: cli.invoke("doctor", config))
    assert _block_network == [], f"impulse doctor reached the network (a login flow): {_block_network}"
    assert error is None, f"{type(error).__name__}: {error}"
    text = cli.text(result)
    assert _hosted_calls(mock_llm, model), f"the {provider} role was not pinged: {text}"
    assert _squeezed(prompt) in text, f"LiteLLM's {provider} sign-in prompt was hidden from impulse doctor: {text}"
    # The ordinary roles keep the redirect: their banner stays hidden, their failure is still reported.
    assert "not reachable" in text, text
    assert PING_BANNER not in result.output, result.output
    assert "LiteLLM.Info" not in result.output, result.output


@pytest.mark.parametrize("provider", AUTH_FLOW_PROVIDERS)
def test_cli_init_ping_shows_the_sign_in_prompt_of_an_authenticating_provider(
    impulse_project, mock_llm, cli, monkeypatch, tmp_path, _block_network, provider
):
    # Interactive `impulse init --profile custom`: per role a model string and an api_base (blank).
    _isolated_litellm_token_dirs(monkeypatch, tmp_path)
    model = f"{provider}/{HOSTED_SUFFIX}"
    prompt = SIGN_IN_PROMPTS[provider]
    mock_llm.responder = _sign_in_ping_responder(model, prompt)
    answers = "\n".join(["en-IN", model, "", ORDINARY_PING_MODEL, "", ORDINARY_PING_MODEL, ""] + [""] * 12) + "\n"
    result, error = _finishes_within(
        AUTH_FLOW_TIMEOUT_S, lambda: cli.invoke("init", "--profile", "custom", input=answers)
    )
    assert _block_network == [], f"impulse init reached the network (a login flow): {_block_network}"
    assert error is None, f"{type(error).__name__}: {error}"
    text = cli.text(result)
    assert result.exit_code == 0, text   # a failed ping still exits 0
    assert _hosted_calls(mock_llm, model), f"the {provider} role was not pinged: {text}"
    assert _squeezed(prompt) in text, f"LiteLLM's {provider} sign-in prompt was hidden from impulse init: {text}"
    assert "not reachable" in text, text
    assert PING_BANNER not in result.output, result.output
    assert "LiteLLM.Info" not in result.output, result.output


@pytest.mark.parametrize("provider", AUTH_FLOW_PROVIDERS)
def test_cli_doctor_still_hides_the_banner_of_an_ordinary_role_next_to_an_authenticating_one(
    impulse_project, mock_llm, cli, monkeypatch, tmp_path, _block_network, provider
):
    # Regression guard (passes before the fix too): only the authenticating role's ping prints
    # through; the ordinary roles' pings keep their output discarded.
    _isolated_litellm_token_dirs(monkeypatch, tmp_path)
    model = f"{provider}/{HOSTED_SUFFIX}"
    mock_llm.responder = _sign_in_ping_responder(model, SIGN_IN_PROMPTS[provider])
    config = _project(impulse_project, **_roles(model, ORDINARY_PING_MODEL, ORDINARY_PING_MODEL))
    result, error = _finishes_within(AUTH_FLOW_TIMEOUT_S, lambda: cli.invoke("doctor", config))
    assert _block_network == [], f"impulse doctor reached the network (a login flow): {_block_network}"
    assert error is None, f"{type(error).__name__}: {error}"
    assert result.exit_code == 2, cli.text(result)   # the ordinary roles' pings failed
    assert _hosted_calls(mock_llm, ORDINARY_PING_MODEL), "the ordinary roles were pinged"
    assert "not reachable: ConnectionError: model server is not reachable" in cli.text(result)
    assert PING_BANNER not in result.output, result.output
    assert "LiteLLM.Info" not in result.output, result.output


# ============================================================================================
# Final check N1 (AUDIT.md finding 11, verified on a fresh Windows install): LiteLLM's wheel can ship
# its bundled tiktoken cl100k_base file (litellm/litellm_core_utils/tokenizers/<sha1 of the blob
# URL>) with CRLF line endings; tiktoken then rejects it (sha256 mismatch), deletes it and downloads
# it from openaipublic.blob.core.windows.net, so an ollama/ ping or run needs the internet.
# (a) Unless the user set CUSTOM_TIKTOKEN_CACHE_DIR (to a non-empty value), importing
# impulse_evals.cli (before anything imports LiteLLM) finds LiteLLM's bundled tokenizers folder
# without importing litellm (importlib.util.find_spec), and if any bundled tokenizer file holds CRLF
# line endings it writes a repaired copy of the folder (CRLF -> LF for those files, the others
# copied as they are) under the system temp folder (reused when already complete) and sets
# CUSTOM_TIKTOKEN_CACHE_DIR to it; bundled files without CRLF: nothing written, nothing set; a
# user-set CUSTOM_TIKTOKEN_CACHE_DIR is never overridden. A user-set TIKTOKEN_CACHE_DIR alone does
# not skip the repair (LiteLLM overwrites it at its import; maintainer decision 2026-10-21, item 1)
# and is left as it is. Checked in fresh interpreters (socket guard) with a fake
# `litellm` package first on PYTHONPATH and TMP / TEMP / TMPDIR pointing into tmp_path.
# (b) An error text that mentions openaipublic.blob.core.windows.net gets TIKTOKEN_HINT (doctor and
# init ping rows).
# ============================================================================================

import hashlib  # noqa: E402
import importlib.util  # noqa: E402

CL100K_NAME = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"   # sha1 of the cl100k_base blob URL
CL100K_LF_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
OTHER_TOKENIZER_NAME = "ec7223a39ce59f226a68acc30dc1af2788490e15"
FAKE_CRLF_TOKENIZER = b"IQ== 0\r\nIg== 1\r\nIw== 2\r\nJA== 3\r\n"
FAKE_LF_TOKENIZER = b"JQ== 0\nJg== 1\nJw== 2\n"
TIKTOKEN_CACHE_VARIABLES = ("TIKTOKEN_CACHE_DIR", "CUSTOM_TIKTOKEN_CACHE_DIR")
TIKTOKEN_HINT = (
    "LiteLLM tried to download the tiktoken cl100k_base encoding; this needs internet access once, or set "
    "CUSTOM_TIKTOKEN_CACHE_DIR to a folder holding the file (not a model or provider failure)"
)
TIKTOKEN_DOWNLOAD_ERROR = (
    "litellm.APIConnectionError: OllamaException - HTTPSConnectionPool(host='openaipublic.blob.core.windows.net', "
    "port=443): Max retries exceeded with url: /encodings/cl100k_base.tiktoken (Caused by NameResolutionError("
    "\"<urllib3.connection.HTTPSConnection object at 0x000001>: Failed to resolve 'openaipublic.blob.core.windows.net' "
    "([Errno 11001] getaddrinfo failed)\"))"
)
CLI_IMPORT_REPORT = """
import json, os, sys, tempfile
import impulse_evals.cli
print("RESULT=" + json.dumps({
    "custom": os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR"),
    "tiktoken_cache_dir": os.environ.get("TIKTOKEN_CACHE_DIR"),
    "tempdir": tempfile.gettempdir(),
    "litellm_imported": "litellm" in sys.modules,
    "attempts": attempts,
}))
"""
CLI_IMPORT_THEN_TIKTOKEN = """
import json, os, sys
import impulse_evals.cli
outcome = {"custom": os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR"), "litellm_imported": "litellm" in sys.modules}
if outcome["custom"]:
    os.environ["TIKTOKEN_CACHE_DIR"] = outcome["custom"]   # what LiteLLM does with CUSTOM_TIKTOKEN_CACHE_DIR
    import tiktoken
    try:
        outcome["tokens"] = tiktoken.get_encoding("cl100k_base").encode("hello world")
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = (type(exc).__name__ + ": " + str(exc))[:300]
outcome["attempts"] = attempts
print("RESULT=" + json.dumps(outcome))
"""
TIKTOKEN_ONLY = """
import json, os
import tiktoken
outcome = {}
try:
    outcome["tokens"] = tiktoken.get_encoding("cl100k_base").encode("hello world")
except Exception as exc:  # noqa: BLE001
    outcome["error"] = (type(exc).__name__ + ": " + str(exc))[:300]
outcome["attempts"] = attempts
print("RESULT=" + json.dumps(outcome))
"""
HELLO_WORLD_CL100K = [15339, 1917]


def _fake_litellm_package(root, tokenizer_files):
    """A fake ``litellm`` package under ``root / "fake-site"`` (its ``__init__`` only creates the
    marker file ``root / "fake-litellm-imported"``) with ``litellm_core_utils/tokenizers/`` holding
    ``tokenizer_files`` (name -> bytes). Returns ``(site folder, tokenizers folder, marker path)``."""
    site = root / "fake-site"
    package = site / "litellm"
    tokenizers = package / "litellm_core_utils" / "tokenizers"
    tokenizers.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import os\nopen(os.environ['IMPULSE_TEST_FAKE_LITELLM_MARKER'], 'w').close()\n", encoding="utf-8"
    )
    (package / "litellm_core_utils" / "__init__.py").write_text("", encoding="utf-8")
    (tokenizers / "__init__.py").write_text("", encoding="utf-8")
    for name, data in tokenizer_files.items():
        (tokenizers / name).write_bytes(data)
    return site, tokenizers, root / "fake-litellm-imported"


def _tree_bytes(folder):
    """Relative POSIX path -> bytes of every file under ``folder`` (``{}`` when it does not exist)."""
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def _cli_import_interpreter(root, site, body, *, preset=None):
    """Run ``body`` (after the socket guard) in a fresh interpreter whose ``PYTHONPATH`` starts with
    ``site``, whose system temp folder is ``root / "system-temp"`` (TMP, TEMP, TMPDIR) and whose
    environment holds neither tiktoken cache variable unless ``preset`` sets one. Returns
    ``(RESULT= JSON, system temp folder)``."""
    system_temp = root / "system-temp"
    system_temp.mkdir(exist_ok=True)
    env = {name: value for name, value in os.environ.items() if name not in TIKTOKEN_CACHE_VARIABLES}
    inherited = os.environ.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(site)] + ([inherited] if inherited else []))
    env.update(TMP=str(system_temp), TEMP=str(system_temp), TMPDIR=str(system_temp))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["IMPULSE_TEST_FAKE_LITELLM_MARKER"] = str(root / "fake-litellm-imported")
    env.update(preset or {})
    code = SUBPROCESS_SOCKET_GUARD + "\n" + body
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT=")]
    assert proc.returncode == 0 and lines, f"exit {proc.returncode}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
    return json.loads(lines[-1][len("RESULT="):]), system_temp


def _is_under(path, folder):
    resolved, base = os.path.normcase(str(Path(path).resolve())), os.path.normcase(str(Path(folder).resolve()))
    return resolved.startswith(base + os.sep)


def _installed_cl100k_lf_bytes():
    """The installed LiteLLM's bundled cl100k_base file with LF line endings (located without
    importing anything new); its sha256 is the one tiktoken expects."""
    spec = importlib.util.find_spec("litellm")
    assert spec is not None and spec.submodule_search_locations, "LiteLLM is installed"
    path = Path(next(iter(spec.submodule_search_locations))) / "litellm_core_utils" / "tokenizers" / CL100K_NAME
    data = path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == CL100K_LF_SHA256, "the official cl100k_base file (LF)"
    return data


def test_cli_import_repairs_a_crlf_bundled_tokenizer_into_a_temp_cache(tmp_path):
    site, tokenizers, marker = _fake_litellm_package(
        tmp_path, {CL100K_NAME: FAKE_CRLF_TOKENIZER, OTHER_TOKENIZER_NAME: FAKE_LF_TOKENIZER}
    )
    bundled_before = _tree_bytes(tokenizers)
    report, system_temp = _cli_import_interpreter(tmp_path, site, CLI_IMPORT_REPORT)
    assert report["attempts"] == [], report["attempts"]
    assert report["litellm_imported"] is False and not marker.exists(), "importing the CLI imported litellm"
    custom = report["custom"]
    assert custom, f"CUSTOM_TIKTOKEN_CACHE_DIR was not set for a CRLF bundled tokenizer: {report}"
    assert _is_under(custom, system_temp), f"{custom} is not under the system temp folder {system_temp}"
    repaired = _tree_bytes(custom)
    assert repaired.get(CL100K_NAME) == FAKE_CRLF_TOKENIZER.replace(b"\r\n", b"\n"), sorted(repaired)
    assert repaired.get(OTHER_TOKENIZER_NAME) == FAKE_LF_TOKENIZER, sorted(repaired)
    assert _tree_bytes(tokenizers) == bundled_before, "the bundled folder itself is never changed"
    # A second import reuses the complete repaired folder.
    stamp = (Path(custom) / CL100K_NAME).stat().st_mtime_ns
    again, _ = _cli_import_interpreter(tmp_path, site, CLI_IMPORT_REPORT)
    assert again["custom"] and os.path.normcase(str(Path(again["custom"]).resolve())) == os.path.normcase(
        str(Path(custom).resolve())
    ), (custom, again["custom"])
    assert (Path(custom) / CL100K_NAME).stat().st_mtime_ns == stamp, "the repaired copy was rewritten, not reused"


def test_cli_import_leaves_lf_bundled_tokenizers_alone(tmp_path):
    # Regression guard (passes before the fix too): bundled files without CRLF need no repair.
    site, tokenizers, marker = _fake_litellm_package(
        tmp_path, {CL100K_NAME: FAKE_CRLF_TOKENIZER.replace(b"\r\n", b"\n"), OTHER_TOKENIZER_NAME: FAKE_LF_TOKENIZER}
    )
    bundled_before = _tree_bytes(tokenizers)
    report, system_temp = _cli_import_interpreter(tmp_path, site, CLI_IMPORT_REPORT)
    assert report["custom"] is None, report
    assert _tree_bytes(system_temp) == {}, sorted(_tree_bytes(system_temp))
    assert _tree_bytes(tokenizers) == bundled_before
    assert report["litellm_imported"] is False and not marker.exists()
    assert report["attempts"] == []


@pytest.mark.parametrize("variable", ["CUSTOM_TIKTOKEN_CACHE_DIR"])
def test_cli_import_never_overrides_a_user_set_tiktoken_cache_variable(tmp_path, variable):
    # Regression guard (passes before the fix too): the user's CUSTOM_TIKTOKEN_CACHE_DIR is kept and
    # nothing is written. Its former TIKTOKEN_CACHE_DIR case was dropped by maintainer decision
    # 2026-10-21, item 1 (LiteLLM overwrites that variable, so it no longer skips the repair).
    site, tokenizers, marker = _fake_litellm_package(
        tmp_path, {CL100K_NAME: FAKE_CRLF_TOKENIZER, OTHER_TOKENIZER_NAME: FAKE_LF_TOKENIZER}
    )
    user_folder = tmp_path / "user-tiktoken-cache"
    user_folder.mkdir()
    bundled_before = _tree_bytes(tokenizers)
    report, system_temp = _cli_import_interpreter(tmp_path, site, CLI_IMPORT_REPORT, preset={variable: str(user_folder)})
    assert _tree_bytes(tokenizers) == bundled_before
    assert report["custom"] == str(user_folder), report
    assert _tree_bytes(system_temp) == {}, sorted(_tree_bytes(system_temp))
    assert _tree_bytes(user_folder) == {}
    assert report["litellm_imported"] is False and not marker.exists()


def test_crlf_copy_of_the_bundled_cl100k_file_makes_tiktoken_go_online(tmp_path):
    # Premise (passes before the fix too): tiktoken rejects a CRLF cl100k_base file and tries the network.
    crlf = _installed_cl100k_lf_bytes().replace(b"\n", b"\r\n")
    cache = tmp_path / "crlf-cache"
    cache.mkdir()
    (cache / CL100K_NAME).write_bytes(crlf)
    site = tmp_path / "empty-site"
    site.mkdir()
    report, _ = _cli_import_interpreter(tmp_path, site, TIKTOKEN_ONLY, preset={"TIKTOKEN_CACHE_DIR": str(cache)})
    assert "tokens" not in report, report
    assert report["attempts"], f"tiktoken did not try to download the file: {report}"


def test_repaired_cl100k_copy_loads_in_tiktoken_offline(tmp_path):
    lf = _installed_cl100k_lf_bytes()
    site, tokenizers, marker = _fake_litellm_package(tmp_path, {CL100K_NAME: lf.replace(b"\n", b"\r\n")})
    assert b"\r\n" in (tokenizers / CL100K_NAME).read_bytes()   # the bundled copy has CRLF line endings
    report, system_temp = _cli_import_interpreter(tmp_path, site, CLI_IMPORT_THEN_TIKTOKEN)
    assert report["custom"], f"CUSTOM_TIKTOKEN_CACHE_DIR was not set for the CRLF cl100k_base file: {report}"
    assert _is_under(report["custom"], system_temp), report["custom"]
    assert report["attempts"] == [], f"tiktoken reached for the network: {report['attempts']}; {report.get('error')}"
    assert report.get("tokens") == HELLO_WORLD_CL100K, report
    repaired = Path(report["custom"]) / CL100K_NAME
    assert hashlib.sha256(repaired.read_bytes()).hexdigest() == CL100K_LF_SHA256
    assert report["litellm_imported"] is False and not marker.exists()


def _tiktoken_download_failure(call):
    raise RuntimeError(TIKTOKEN_DOWNLOAD_ERROR)


def test_cli_doctor_explains_a_tiktoken_download_failure(impulse_project, mock_llm, cli):
    mock_llm.responder = _tiktoken_download_failure
    config = _project(impulse_project, **_roles("ollama/judge-primary", "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert "not reachable" in text and "openaipublic.blob.core.windows.net" in text, text
    assert TIKTOKEN_HINT in text, f"the doctor ping row does not explain the tiktoken download: {text}"


def test_cli_init_ping_explains_a_tiktoken_download_failure(impulse_project, mock_llm, cli):
    mock_llm.responder = _tiktoken_download_failure
    answers = "\n".join(["en-IN"] + ["impulse-test-model"] * 12) + "\n"
    result = cli.invoke("init", "--profile", "free", input=answers)
    text = cli.text(result)
    assert result.exit_code == 0, text   # a failed ping still exits 0
    assert mock_llm.call_count >= 1, "the ping ran"
    assert "not reachable" in text, text
    assert TIKTOKEN_HINT in text, f"the init ping row does not explain the tiktoken download: {text}"


def test_cli_doctor_adds_no_tiktoken_hint_to_other_failures(impulse_project, mock_llm, cli):
    # Regression guard (passes before the fix too): only the tiktoken download gets the hint.
    def responder(call):
        raise RuntimeError("litellm.APIConnectionError: OllamaException - [WinError 10061] connection refused")

    mock_llm.responder = responder
    config = _project(impulse_project, **_roles("ollama/judge-primary", "ollama/judge-fast", "ollama/simulator"))
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert "connection refused" in text, text
    assert "tiktoken" not in text, text


# ============================================================================================
# Maintainer decision 2026-10-21, item 1 (AUDIT-FIXES.md, "Still open" 1): LiteLLM sets
# TIKTOKEN_CACHE_DIR at its own import (to CUSTOM_TIKTOKEN_CACHE_DIR when that is set to a non-empty
# value, else to its bundled tokenizers folder), so a user-set TIKTOKEN_CACHE_DIR never reaches
# tiktoken. The repair is skipped only when CUSTOM_TIKTOKEN_CACHE_DIR is set to a non-empty value (an
# empty value counts as unset, as in LiteLLM's own `if custom_cache_dir:`). With only
# TIKTOKEN_CACHE_DIR set, the repair runs exactly as when nothing is set (CRLF found: repaired copy
# under <system temp>/impulse-tiktoken/<hash>/ and CUSTOM_TIKTOKEN_CACHE_DIR set to it; no CRLF:
# nothing written, nothing set); the user's TIKTOKEN_CACHE_DIR value stays as it is in os.environ and
# nothing is ever written into the folder it names. Fresh interpreters, socket guard, fake `litellm`
# package, as in the final check N1 block above.
# ============================================================================================

CLI_IMPORT_THEN_LITELLM_STYLE_TIKTOKEN = """
import json, os, sys
import impulse_evals.cli
outcome = {
    "custom": os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR"),
    "tiktoken_cache_dir": os.environ.get("TIKTOKEN_CACHE_DIR"),
    "litellm_imported": "litellm" in sys.modules,
}
# What LiteLLM does at its import: TIKTOKEN_CACHE_DIR = CUSTOM_TIKTOKEN_CACHE_DIR when that is set,
# otherwise its bundled tokenizers folder (a user-set TIKTOKEN_CACHE_DIR is overwritten either way).
os.environ["TIKTOKEN_CACHE_DIR"] = outcome["custom"] or os.environ["IMPULSE_TEST_BUNDLED_TOKENIZERS"]
import tiktoken
try:
    outcome["tokens"] = tiktoken.get_encoding("cl100k_base").encode("hello world")
except Exception as exc:  # noqa: BLE001
    outcome["error"] = (type(exc).__name__ + ": " + str(exc))[:300]
outcome["attempts"] = attempts
print("RESULT=" + json.dumps(outcome))
"""
CLI_IMPORT_REPORT_WITH_PRESET = """
import json, os, sys
preset_custom = os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR", "<unset>")
import impulse_evals.cli
print("RESULT=" + json.dumps({
    "preset_custom": preset_custom,
    "custom": os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR"),
    "litellm_imported": "litellm" in sys.modules,
    "attempts": attempts,
}))
"""


def test_cli_import_repairs_a_crlf_bundled_tokenizer_when_only_tiktoken_cache_dir_is_set(tmp_path):
    site, tokenizers, marker = _fake_litellm_package(
        tmp_path, {CL100K_NAME: FAKE_CRLF_TOKENIZER, OTHER_TOKENIZER_NAME: FAKE_LF_TOKENIZER}
    )
    user_folder = tmp_path / "user-tiktoken-cache"
    user_folder.mkdir()
    bundled_before = _tree_bytes(tokenizers)
    report, system_temp = _cli_import_interpreter(
        tmp_path, site, CLI_IMPORT_REPORT, preset={"TIKTOKEN_CACHE_DIR": str(user_folder)}
    )
    custom = report["custom"]
    assert custom, (
        "CUSTOM_TIKTOKEN_CACHE_DIR was not set for a CRLF bundled tokenizer although only TIKTOKEN_CACHE_DIR was "
        f"set (LiteLLM overwrites TIKTOKEN_CACHE_DIR at its import, so it must not skip the repair): {report}"
    )
    assert _is_under(custom, system_temp), f"{custom} is not under the system temp folder {system_temp}"
    repaired = _tree_bytes(custom)
    assert repaired.get(CL100K_NAME) == FAKE_CRLF_TOKENIZER.replace(b"\r\n", b"\n"), sorted(repaired)
    assert repaired.get(OTHER_TOKENIZER_NAME) == FAKE_LF_TOKENIZER, sorted(repaired)
    assert report["tiktoken_cache_dir"] == str(user_folder), "the user's TIKTOKEN_CACHE_DIR value is left as it is"
    assert _tree_bytes(user_folder) == {}, "nothing is written into the folder TIKTOKEN_CACHE_DIR names"
    assert _tree_bytes(tokenizers) == bundled_before, "the bundled folder itself is never changed"
    assert report["litellm_imported"] is False and not marker.exists(), "importing the CLI imported litellm"
    assert report["attempts"] == [], report["attempts"]


def test_repaired_cl100k_copy_loads_in_tiktoken_offline_when_only_tiktoken_cache_dir_is_set(tmp_path):
    lf = _installed_cl100k_lf_bytes()
    site, tokenizers, marker = _fake_litellm_package(tmp_path, {CL100K_NAME: lf.replace(b"\n", b"\r\n")})
    user_folder = tmp_path / "user-tiktoken-cache"
    user_folder.mkdir()
    report, system_temp = _cli_import_interpreter(
        tmp_path,
        site,
        CLI_IMPORT_THEN_LITELLM_STYLE_TIKTOKEN,
        preset={"TIKTOKEN_CACHE_DIR": str(user_folder), "IMPULSE_TEST_BUNDLED_TOKENIZERS": str(tokenizers)},
    )
    assert report["attempts"] == [], (
        f"tiktoken reached for the network: {report['attempts']}; {report.get('error')} (CUSTOM_TIKTOKEN_CACHE_DIR "
        f"was {report['custom']!r}: a user-set TIKTOKEN_CACHE_DIR must not skip the repair)"
    )
    assert report.get("tokens") == HELLO_WORLD_CL100K, report
    assert report["custom"] and _is_under(report["custom"], system_temp), report
    assert hashlib.sha256((Path(report["custom"]) / CL100K_NAME).read_bytes()).hexdigest() == CL100K_LF_SHA256
    assert report["tiktoken_cache_dir"] == str(user_folder), "the user's TIKTOKEN_CACHE_DIR value is left as it is"
    assert _tree_bytes(user_folder) == {}
    assert report["litellm_imported"] is False and not marker.exists()


def test_cli_import_with_only_tiktoken_cache_dir_set_leaves_lf_bundled_tokenizers_alone(tmp_path):
    # Regression guard (passes before the fix too): no CRLF, nothing written, nothing set.
    site, tokenizers, marker = _fake_litellm_package(
        tmp_path, {CL100K_NAME: FAKE_CRLF_TOKENIZER.replace(b"\r\n", b"\n"), OTHER_TOKENIZER_NAME: FAKE_LF_TOKENIZER}
    )
    user_folder = tmp_path / "user-tiktoken-cache"
    user_folder.mkdir()
    bundled_before = _tree_bytes(tokenizers)
    report, system_temp = _cli_import_interpreter(
        tmp_path, site, CLI_IMPORT_REPORT, preset={"TIKTOKEN_CACHE_DIR": str(user_folder)}
    )
    assert report["custom"] is None, report
    assert report["tiktoken_cache_dir"] == str(user_folder), report
    assert _tree_bytes(system_temp) == {}, sorted(_tree_bytes(system_temp))
    assert _tree_bytes(user_folder) == {}
    assert _tree_bytes(tokenizers) == bundled_before
    assert report["litellm_imported"] is False and not marker.exists()
    assert report["attempts"] == []


def test_cli_import_treats_an_empty_custom_tiktoken_cache_dir_as_unset(tmp_path):
    # Regression guard (passes before the fix too): CUSTOM_TIKTOKEN_CACHE_DIR="" does not skip the repair.
    site, tokenizers, marker = _fake_litellm_package(
        tmp_path, {CL100K_NAME: FAKE_CRLF_TOKENIZER, OTHER_TOKENIZER_NAME: FAKE_LF_TOKENIZER}
    )
    report, system_temp = _cli_import_interpreter(
        tmp_path, site, CLI_IMPORT_REPORT_WITH_PRESET, preset={"CUSTOM_TIKTOKEN_CACHE_DIR": ""}
    )
    assert report["preset_custom"] == "", f"the child did not start with an empty CUSTOM_TIKTOKEN_CACHE_DIR: {report}"
    assert report["custom"] and _is_under(report["custom"], system_temp), report
    assert _tree_bytes(report["custom"]).get(CL100K_NAME) == FAKE_CRLF_TOKENIZER.replace(b"\r\n", b"\n")
    assert report["litellm_imported"] is False and not marker.exists()
    assert report["attempts"] == []


# ============================================================================================
# Maintainer decision 2026-10-21, item 2 (AUDIT-FIXES.md, "Still open" 2): the first-start race in
# the repair. After writing its staging copy (Path.write_bytes into a tempfile.mkdtemp folder next to
# the target), a start re-checks the target: when another start has meanwhile put a complete copy
# there (every bundled file name present), that copy is kept untouched and the staging copy is
# dropped (no rmtree of the target, no rename); only an existing INCOMPLETE target is removed before
# the rename. The staging folder never survives. Simulated deterministically in a fresh interpreter:
# a hook on pathlib.Path.write_bytes, installed before `import impulse_evals.cli`, plays the faster
# start on the first write under <system temp>/impulse-tiktoken/ outside the target folder (that is,
# after this start's first completeness check and before its rename) by creating the complete
# target folder: the repaired files plus a marker file. The target path is learned from a first,
# hook-free run (then removed), so the test does not depend on the hash formula.
# ============================================================================================

import shutil  # noqa: E402

TIKTOKEN_REPAIR_FOLDER = "impulse-tiktoken"
RACE_MARKER_NAME = "written-by-the-other-start"
RACE_MARKER_BYTES = b"the faster start's complete copy\n"
RACE_HOOK = r"""
import os, pathlib
race = {"fired": False, "mtimes": {}}
_race_target = os.environ["IMPULSE_TEST_RACE_TARGET"]
_race_source = os.environ["IMPULSE_TEST_RACE_SOURCE"]
_real_write_bytes = pathlib.Path.write_bytes


def _race_norm(path):
    return os.path.normcase(os.path.realpath(str(path)))


def _race_under(path, folder):
    return _race_norm(path).startswith(_race_norm(folder) + os.sep)


def _race_write_bytes(self, data):
    if not race["fired"] and _race_under(self, os.path.dirname(_race_target)) and not _race_under(self, _race_target):
        race["fired"] = True
        # The faster start finishes now: its complete copy appears at the target.
        os.makedirs(_race_target)
        for name in sorted(os.listdir(_race_source)):
            with open(os.path.join(_race_source, name), "rb") as source:
                content = source.read()
            with open(os.path.join(_race_target, name), "wb") as handle:
                handle.write(content)
        for name in os.listdir(_race_target):
            race["mtimes"][name] = os.stat(os.path.join(_race_target, name)).st_mtime_ns
    return _real_write_bytes(self, data)


pathlib.Path.write_bytes = _race_write_bytes
"""
CLI_IMPORT_RACE_REPORT = RACE_HOOK + """
import json, sys
import impulse_evals.cli
print("RESULT=" + json.dumps({
    "custom": os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR"),
    "litellm_imported": "litellm" in sys.modules,
    "race": race,
    "attempts": attempts,
}))
"""


def _same_path(first, second):
    return os.path.normcase(str(Path(first).resolve())) == os.path.normcase(str(Path(second).resolve()))


def _learn_repair_target(tmp_path):
    """A fake CRLF ``litellm`` package and one hook-free CLI import to learn the repair folder; returns
    ``(site, tokenizers, marker, target, repaired files, system temp)`` with ``impulse-tiktoken/``
    removed again."""
    site, tokenizers, marker = _fake_litellm_package(
        tmp_path, {CL100K_NAME: FAKE_CRLF_TOKENIZER, OTHER_TOKENIZER_NAME: FAKE_LF_TOKENIZER}
    )
    first, system_temp = _cli_import_interpreter(tmp_path, site, CLI_IMPORT_REPORT)
    repair_root = system_temp / TIKTOKEN_REPAIR_FOLDER
    assert first["custom"] and _same_path(Path(first["custom"]).parent, repair_root), first
    target = Path(first["custom"])
    repaired = _tree_bytes(target)
    assert repaired.get(CL100K_NAME) == FAKE_CRLF_TOKENIZER.replace(b"\r\n", b"\n"), sorted(repaired)
    shutil.rmtree(repair_root)
    return site, tokenizers, marker, target, repaired, system_temp


def test_a_slower_first_start_keeps_the_complete_copy_a_faster_start_just_finished(tmp_path):
    site, tokenizers, marker, target, repaired, system_temp = _learn_repair_target(tmp_path)
    bundled_before = _tree_bytes(tokenizers)
    other_start = tmp_path / "other-start-copy"
    other_start.mkdir()
    for name, data in repaired.items():
        (other_start / name).write_bytes(data)
    (other_start / RACE_MARKER_NAME).write_bytes(RACE_MARKER_BYTES)
    expected = _tree_bytes(other_start)
    report, _ = _cli_import_interpreter(
        tmp_path,
        site,
        CLI_IMPORT_RACE_REPORT,
        preset={"IMPULSE_TEST_RACE_TARGET": str(target), "IMPULSE_TEST_RACE_SOURCE": str(other_start)},
    )
    assert report["race"]["fired"], (
        "the staging copy was not written with Path.write_bytes under impulse-tiktoken/ (the hook never ran)"
    )
    found = _tree_bytes(target)
    assert found == expected, (
        "the other start's complete copy was deleted or replaced (rmtree of a complete target): "
        f"{sorted(found)} instead of {sorted(expected)}"
    )
    mtimes = {name: (target / name).stat().st_mtime_ns for name in expected}
    assert mtimes == report["race"]["mtimes"], "the other start's files were rewritten"
    assert report["custom"] and _same_path(report["custom"], target), (report["custom"], str(target))
    assert sorted(os.listdir(system_temp / TIKTOKEN_REPAIR_FOLDER)) == [target.name], (
        "the staging copy survived next to the target"
    )
    assert _tree_bytes(tokenizers) == bundled_before, "the bundled folder itself is never changed"
    assert report["litellm_imported"] is False and not marker.exists()
    assert report["attempts"] == [], report["attempts"]


def test_an_incomplete_repair_folder_is_still_rebuilt_into_a_complete_copy(tmp_path):
    # Regression guard (passes before the fix too): an existing INCOMPLETE target is replaced.
    site, tokenizers, marker = _fake_litellm_package(
        tmp_path, {CL100K_NAME: FAKE_CRLF_TOKENIZER, OTHER_TOKENIZER_NAME: FAKE_LF_TOKENIZER}
    )
    first, system_temp = _cli_import_interpreter(tmp_path, site, CLI_IMPORT_REPORT)
    target = Path(first["custom"])
    repaired = _tree_bytes(target)
    (target / OTHER_TOKENIZER_NAME).unlink()   # one bundled file missing: incomplete
    report, _ = _cli_import_interpreter(tmp_path, site, CLI_IMPORT_REPORT)
    assert report["custom"] and _same_path(report["custom"], target), (report["custom"], str(target))
    assert _tree_bytes(target) == repaired, sorted(_tree_bytes(target))
    assert sorted(os.listdir(system_temp / TIKTOKEN_REPAIR_FOLDER)) == [target.name]
    assert report["litellm_imported"] is False and not marker.exists()
    assert report["attempts"] == []


# ============================================================================================
# Maintainer decisions 2026-10-21, review fix-up R1 (item 2): rename first, remove only after a
# failed rename. After its staging copy is written, a start keeps a complete target (and drops the
# staging copy); otherwise it calls os.replace(staging, target) at once. Only when that rename fails
# AND the target is found incomplete is the target removed (shutil.rmtree) and the rename tried once
# more. So a complete copy another start renames into place right after this start's completeness
# re-check is never removed. Simulated deterministically in a fresh interpreter: hooks installed
# before `import impulse_evals.cli` arm on the staging-file writes (Path.write_bytes under
# <system temp>/impulse-tiktoken/, outside the target) and land the other start's complete copy
# (repaired files plus a marker) at the target at the first of these operations this start then
# performs ON THE TARGET: Path.exists / os.path.exists, shutil.rmtree, os.replace / os.rename with
# it as destination (not the is_dir / is_file calls of the completeness re-check).
# ============================================================================================

WINDOW_HOOK = r"""
import os, pathlib, shutil
window = {"armed": False, "landed_at": None, "writes": 0, "writes_at_landing": None, "mtimes": {}}
_w_target = os.environ["IMPULSE_TEST_RACE_TARGET"]
_w_source = os.environ["IMPULSE_TEST_RACE_SOURCE"]
_w_real = {
    "write_bytes": pathlib.Path.write_bytes,
    "Path.exists": pathlib.Path.exists,
    "os.path.exists": os.path.exists,
    "shutil.rmtree": shutil.rmtree,
    "os.replace": os.replace,
    "os.rename": os.rename,
}


def _w_norm(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _w_is_target(path):
    try:
        return _w_norm(path) == _w_norm(_w_target)
    except (TypeError, ValueError):
        return False


def _w_under(path, folder):
    return _w_norm(path).startswith(_w_norm(folder) + os.sep)


def _w_land(operation):
    if not window["armed"] or window["landed_at"] is not None:
        return
    window["landed_at"] = operation
    window["writes_at_landing"] = window["writes"]
    # The other start renames its complete copy into place now.
    os.makedirs(_w_target)
    for name in sorted(os.listdir(_w_source)):
        with open(os.path.join(_w_source, name), "rb") as source:
            content = source.read()
        with open(os.path.join(_w_target, name), "wb") as handle:
            handle.write(content)
    for name in os.listdir(_w_target):
        window["mtimes"][name] = os.stat(os.path.join(_w_target, name)).st_mtime_ns


def _w_write_bytes(self, data):
    if _w_under(self, os.path.dirname(_w_target)) and not _w_under(self, _w_target):
        window["armed"] = True   # a staging-file write
        window["writes"] += 1
    return _w_real["write_bytes"](self, data)


def _w_path_exists(self, *args, **kwargs):
    if _w_is_target(self):
        _w_land("Path.exists")
    return _w_real["Path.exists"](self, *args, **kwargs)


def _w_os_path_exists(path):
    if _w_is_target(path):
        _w_land("os.path.exists")
    return _w_real["os.path.exists"](path)


def _w_rmtree(path, *args, **kwargs):
    if _w_is_target(path):
        _w_land("shutil.rmtree")
    return _w_real["shutil.rmtree"](path, *args, **kwargs)


def _w_replace(src, dst, *args, **kwargs):
    if _w_is_target(dst):
        _w_land("os.replace")
    return _w_real["os.replace"](src, dst, *args, **kwargs)


def _w_rename(src, dst, *args, **kwargs):
    if _w_is_target(dst):
        _w_land("os.rename")
    return _w_real["os.rename"](src, dst, *args, **kwargs)


pathlib.Path.write_bytes = _w_write_bytes
pathlib.Path.exists = _w_path_exists
os.path.exists = _w_os_path_exists
shutil.rmtree = _w_rmtree
os.replace = _w_replace
os.rename = _w_rename
"""
CLI_IMPORT_WINDOW_REPORT = WINDOW_HOOK + """
import json, sys
import impulse_evals.cli
print("RESULT=" + json.dumps({
    "custom": os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR"),
    "litellm_imported": "litellm" in sys.modules,
    "window": window,
    "attempts": attempts,
}))
"""
WINDOW_OPERATIONS = ("Path.exists", "os.path.exists", "shutil.rmtree", "os.replace", "os.rename")


def test_a_complete_copy_landing_right_after_the_recheck_is_never_removed(tmp_path):
    site, tokenizers, marker, target, repaired, system_temp = _learn_repair_target(tmp_path)
    bundled_before = _tree_bytes(tokenizers)
    other_start = tmp_path / "other-start-copy"
    other_start.mkdir()
    for name, data in repaired.items():
        (other_start / name).write_bytes(data)
    (other_start / RACE_MARKER_NAME).write_bytes(RACE_MARKER_BYTES)
    expected = _tree_bytes(other_start)
    report, _ = _cli_import_interpreter(
        tmp_path,
        site,
        CLI_IMPORT_WINDOW_REPORT,
        preset={"IMPULSE_TEST_RACE_TARGET": str(target), "IMPULSE_TEST_RACE_SOURCE": str(other_start)},
    )
    window = report["window"]
    assert window["armed"], "the staging copy was not written with Path.write_bytes under impulse-tiktoken/"
    assert window["landed_at"] in WINDOW_OPERATIONS, f"this start performed no operation on the target: {window}"
    assert window["writes_at_landing"] == window["writes"], f"the copy landed before the last staging write: {window}"
    found = _tree_bytes(target)
    assert found == expected, (
        f"the other start's complete copy, landed at this start's {window['landed_at']} on the target, was "
        f"removed or replaced: {sorted(found)} instead of {sorted(expected)}"
    )
    mtimes = {name: (target / name).stat().st_mtime_ns for name in expected}
    assert mtimes == window["mtimes"], "the other start's files were rewritten"
    assert report["custom"] and _same_path(report["custom"], target), (report["custom"], str(target))
    assert sorted(os.listdir(system_temp / TIKTOKEN_REPAIR_FOLDER)) == [target.name], (
        "the staging copy survived next to the target"
    )
    assert _tree_bytes(tokenizers) == bundled_before, "the bundled folder itself is never changed"
    assert report["litellm_imported"] is False and not marker.exists()
    assert report["attempts"] == [], report["attempts"]
