# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 8: ``impulse doctor [CONFIG]`` (SPEC.md section 17): a pass/fail table (each row shows the
word PASS or FAIL, any case; info rows never fail); exit 0 when every check passes, 2 otherwise.
Backends are mocked: ``mock_llm`` answers the model pings.

[phase 11] voice provider rows (``mock_voice_server``) and the Pipecat handshake row: the doctor
builds the adapter through ``impulse_evals.adapters.pipecat_adapter.PipecatAdapter`` looked up when
the check runs, so the tests patch it to inject ``mock_pipecat_channel`` (contract logged in
DECISIONS.md, Stage B group 4-8).
"""

import importlib.util

import pytest

SECRET = "sk-impulse-test-SECRET-doctor-5521"
HOSTED_MODEL = "openai/impulse-test-model"


def _config(project, **kwargs):
    project.write_agent()
    return project.write_config([project.scenario("doctor_probe", metrics=["UNIV-013"])], **kwargs)


def _judge(primary="ollama/judge-primary", fast="ollama/judge-fast", primary_options=None):
    return {
        "primary_model": primary,
        "fast_model": fast,
        "primary_options": primary_options or {},
        "fast_options": {},
    }


def _failing_llm(call):
    raise ConnectionError("model server is not reachable")


def test_doctor_all_checks_pass_exits_0(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    config = _config(impulse_project)
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    text = cli.text(result).lower()
    assert "python" in text
    assert "pass" in text  # each row shows its status (PASS / FAIL)
    assert mock_llm.call_count >= 1  # model reachability is checked


def test_doctor_config_error_exits_2_naming_the_field(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    project = impulse_project
    project.write_agent()
    config = project.write_config([project.scenario("doctor_probe", metrics=["UNIV-999"])])
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2
    assert "scenarios[0].metrics[0]" in cli.text(result)


def test_doctor_missing_config_file_exits_2(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    assert cli.invoke("doctor", impulse_project.root / "no_such_config.yaml").exit_code == 2


def test_doctor_unreachable_model_exits_2(impulse_project, mock_llm, cli):
    mock_llm.responder = _failing_llm
    config = _config(impulse_project)
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2
    assert "fail" in cli.text(result).lower()
    assert mock_llm.call_count >= 1


def test_doctor_unresolved_model_exits_2(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    config = _config(impulse_project, judge=_judge(primary="YOUR_PRIMARY_MODEL"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2
    assert "primary" in cli.text(result).lower()


def test_doctor_missing_key_for_hosted_model_exits_2(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    config = _config(impulse_project, judge=_judge(primary=HOSTED_MODEL))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in cli.text(result)
    assert not [call for call in mock_llm.calls if call["model"] == HOSTED_MODEL]


def test_doctor_with_key_passes_and_never_prints_it(impulse_project, mock_llm, cli, monkeypatch):
    mock_llm.set_default("OK")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    config = _config(impulse_project, judge=_judge(primary=HOSTED_MODEL))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert SECRET not in result.output


def test_doctor_local_models_need_no_key(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    config = _config(
        impulse_project,
        judge=_judge(primary=HOSTED_MODEL, primary_options={"api_base": "http://localhost:8000/v1"}),
    )
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)


# ============================================================================================
# [phase 11] voice providers
# ============================================================================================

VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": True,
    "save_recording": False,
    "timeout_s": 30,
}


@pytest.mark.phase11
def test_doctor_voice_providers_pass(impulse_project, mock_llm, mock_voice_server, cli):
    mock_llm.set_default("OK")
    config = _config(impulse_project, voice_config=VOICE_CONFIG)
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert mock_voice_server.requests


@pytest.mark.phase11
def test_doctor_voice_provider_failure_exits_2(impulse_project, mock_llm, mock_voice_server, cli):
    mock_llm.set_default("OK")
    mock_voice_server.queue_status(500, count=50)
    config = _config(impulse_project, voice_config=VOICE_CONFIG)
    assert cli.invoke("doctor", config).exit_code == 2


# ============================================================================================
# [phase 11] Pipecat handshake row
# ============================================================================================


def _pipecat_config(project, mode="websocket", handshake_timeout_s=10):
    return project.write_config(
        [project.scenario("pipecat_probe", metrics=["UNIV-013"])],
        agent={
            "adapter": "pipecat",
            "pipecat": {"mode": mode, "url": "ws://localhost:7860", "handshake_timeout_s": handshake_timeout_s},
        },
    )


def _patch_pipecat(monkeypatch, mock_pipecat_channel):
    from impulse_evals.adapters import pipecat_adapter

    real = pipecat_adapter.PipecatAdapter

    def patched(config, *args, **kwargs):
        kwargs["channel_factory"] = mock_pipecat_channel.factory
        kwargs["clock"] = mock_pipecat_channel.clock
        return real(config, *args, **kwargs)

    monkeypatch.setattr(pipecat_adapter, "PipecatAdapter", patched)


@pytest.mark.phase11
def test_doctor_pipecat_handshake_passes_on_bot_ready(
    impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch
):
    mock_llm.set_default("OK")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    channel = mock_pipecat_channel.channel
    channel.on_send("client-ready", mock_pipecat_channel.rtvi("bot-ready", {"version": "1.0.0"}))
    result = cli.invoke("doctor", _pipecat_config(impulse_project))
    assert result.exit_code == 0, cli.text(result)
    sent = mock_pipecat_channel.channel.sent_types()
    assert sent and sent[0] == "client-ready"


@pytest.mark.phase11
def test_doctor_pipecat_handshake_fails_on_timeout(impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch):
    mock_llm.set_default("OK")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    result = cli.invoke("doctor", _pipecat_config(impulse_project, handshake_timeout_s=2))
    assert result.exit_code == 2
    assert "Pipecat bot did not send bot-ready within 2 s." in cli.text(result)


@pytest.mark.phase11
def test_doctor_pipecat_system_prompt_answer_is_info_only(
    impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch
):
    mock_llm.set_default("OK")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    channel = mock_pipecat_channel.channel
    channel.on_send("client-ready", mock_pipecat_channel.rtvi("bot-ready", {"version": "1.0.0"}))
    # The bot never answers get_system_prompt: reported as info, never a failure.
    project = impulse_project
    config = project.write_config(
        [project.scenario("pipecat_probe", metrics=["UNIV-013"])],
        agent={"adapter": "pipecat", "pipecat": {"mode": "websocket", "url": "ws://localhost:7860"}},
        system_prompt=None,
    )
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)


@pytest.mark.phase11
@pytest.mark.skipif(importlib.util.find_spec("pipecat") is not None, reason="pipecat-ai is installed")
def test_doctor_pipecat_eval_mode_needs_pipecat_evals(impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch):
    mock_llm.set_default("OK")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    mock_pipecat_channel.channel.on_send("client-ready", mock_pipecat_channel.rtvi("bot-ready", {"version": "1.0.0"}))
    result = cli.invoke("doctor", _pipecat_config(impulse_project, mode="eval"))
    assert result.exit_code == 2
    assert "pipecat.evals" in cli.text(result)


# ============================================================================================
# [phase 11] Pipecat eval mode after the source read (SPEC.md sections 9 and 17): with pipecat.evals
# importable, the doctor connects in eval mode; Pipecat's client transport sends client-ready itself,
# so the adapter sends none and the fake bot answers bot-ready on connect. ``extra_imports`` makes an
# empty ``pipecat.evals`` package importable (the adapter never imports it when a channel is injected).
# ============================================================================================


@pytest.mark.phase11
def test_doctor_pipecat_eval_mode_handshake_passes(
    impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch, extra_imports
):
    mock_llm.set_default("OK")
    extra_imports.pipecat("with_evals")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    channel = mock_pipecat_channel.channel
    channel.on_connect(mock_pipecat_channel.rtvi("bot-ready", {"version": "2.1.0"}, id="client-ready"))
    result = cli.invoke("doctor", _pipecat_config(impulse_project, mode="eval"))
    assert result.exit_code == 0, cli.text(result)
    assert len(channel.connect_calls) == 1
    assert "client-ready" not in channel.sent_types()


@pytest.mark.phase11
def test_doctor_pipecat_eval_mode_handshake_fails_on_timeout(
    impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch, extra_imports
):
    mock_llm.set_default("OK")
    extra_imports.pipecat("with_evals")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    result = cli.invoke("doctor", _pipecat_config(impulse_project, mode="eval", handshake_timeout_s=2))
    assert result.exit_code == 2
    assert "Pipecat bot did not send bot-ready within 2 s." in cli.text(result)


# ============================================================================================
# [phase 11] AUDIT batch 1 (findings 4 and 6): the Pipecat row never prints credentials from
# pipecat.url (query token / userinfo) or literal pipecat.headers values.
# ============================================================================================

TOKEN_URL = "ws://127.0.0.1:1/ws?token=SECRETTOKEN123"
HEADER_SECRET = "LITERALKEY456"


class _EchoingHeadersChannel:
    """Wraps a fake channel; ``connect()`` fails like a rejected WebSocket upgrade whose error echoes the
    request headers."""

    def __init__(self, inner):
        self._inner = inner

    async def connect(self, url, headers, query):
        raise RuntimeError(f"server rejected the WebSocket upgrade; request headers were {dict(headers)}")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _patch_pipecat_factory(monkeypatch, mock_pipecat_channel, channel_factory):
    """Like ``_patch_pipecat`` with a custom ``channel_factory``."""
    from impulse_evals.adapters import pipecat_adapter

    real = pipecat_adapter.PipecatAdapter

    def patched(config, *args, **kwargs):
        kwargs["channel_factory"] = channel_factory
        kwargs["clock"] = mock_pipecat_channel.clock
        return real(config, *args, **kwargs)

    monkeypatch.setattr(pipecat_adapter, "PipecatAdapter", patched)


def _pipecat_url_config(project, url, **pipecat):
    block = {"mode": "websocket", "url": url}
    block.update(pipecat)
    return project.write_config(
        [project.scenario("pipecat_probe", metrics=["UNIV-013"])],
        agent={"adapter": "pipecat", "pipecat": block},
    )


@pytest.mark.phase11
def test_doctor_pipecat_row_strips_url_credentials(impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch):
    mock_llm.set_default("OK")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    channel = mock_pipecat_channel.channel
    channel.on_send("client-ready", mock_pipecat_channel.rtvi("bot-ready", {"version": "1.0.0"}))
    result = cli.invoke("doctor", _pipecat_url_config(impulse_project, TOKEN_URL))
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert "SECRETTOKEN123" not in text
    assert "ws://127.0.0.1" in text  # the host is still shown
    assert channel.connect_url == TOKEN_URL  # the connection itself uses the full URL


@pytest.mark.phase11
def test_doctor_redacts_literal_pipecat_header_values(impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch):
    mock_llm.set_default("OK")
    mock = mock_pipecat_channel
    _patch_pipecat_factory(monkeypatch, mock, lambda: _EchoingHeadersChannel(mock.factory()))
    config = _pipecat_url_config(impulse_project, "ws://localhost:7860", headers={"X-Api-Key": HEADER_SECRET})
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2
    assert HEADER_SECRET not in text


# ============================================================================================
# Audit batch 1 review fix-ups (AUDIT.md finding 6, follow-up): impulse doctor without a
# config.yaml checks the models from the environment and config.toml only; the literal credentials
# in config.toml [model_options.primary] must still be redacted from the model rows
# (``Bearer [REDACTED]``) while the ping still sends them unchanged.
# ============================================================================================

TOML_LITERAL_BEARER = "LITERALTOML321"
TOML_LITERAL_HEADERS = (
    "[model_options.primary]\n"
    f'extra_params = {{ extra_headers = {{ Authorization = "Bearer {TOML_LITERAL_BEARER}" }} }}\n'
)
ECHOED_REDACTED_HEADERS = "401 Unauthorized; request headers: {'Authorization': 'Bearer [REDACTED]'}"
STANDALONE_MODEL_ENV = {
    "IMPULSE_MODEL_PRIMARY": "ollama/judge-primary",
    "IMPULSE_MODEL_FAST": "ollama/judge-fast",
    "IMPULSE_MODEL_SIMULATOR": "ollama/simulator",
}


def _raise_echoing_request_headers(call):
    """A LiteLLM error that echoes the request headers it was sent (an HTTP client error)."""
    raise RuntimeError(f"401 Unauthorized; request headers: {call['kwargs'].get('extra_headers')}")


def test_doctor_without_config_yaml_never_prints_literal_toml_option_credentials(
    impulse_project, mock_llm, cli, monkeypatch
):
    for name, value in STANDALONE_MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    impulse_project.write_toml(TOML_LITERAL_HEADERS)
    assert not (impulse_project.root / "config.yaml").exists()   # the no-config.yaml branch
    mock_llm.responder = _raise_echoing_request_headers
    result = cli.invoke("doctor")
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert TOML_LITERAL_BEARER not in result.output, text
    assert ECHOED_REDACTED_HEADERS in text
    # The ping still sends the header unchanged (redaction is for printed texts only).
    primary_pings = [call for call in mock_llm.calls if call["model"] == "ollama/judge-primary"]
    assert primary_pings
    assert primary_pings[0]["kwargs"].get("extra_headers") == {"Authorization": f"Bearer {TOML_LITERAL_BEARER}"}


# ============================================================================================
# Audit batch 2 (AUDIT.md findings 19 and 20): [phase 11] voice rows.
# F19: local_hf TTS without torch (the 'local' extra does not install it) is a FAIL row with the
# exact ConfigError text, not "PASS ... 'local' extra installed". F20: a TTS server answering 200
# with a JSON body is a FAIL row showing the server's answer, not "PASS ... N bytes of audio".
# ============================================================================================

TORCH_MISSING = (
    "local_hf TTS needs PyTorch, which the 'local' extra does not install: pip install torch "
    "(see https://pytorch.org/get-started/locally/)"
)
LOCAL_VOICE_CONFIG = {
    "tts": {"provider": "local_hf", "model": "hf-test/tts-default", "device": "cpu"},
    "stt": {"provider": "local_hf", "model": "hf-test/stt-small", "device": "cpu", "compute_type": "int8"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}


def _status_lines(result, status):
    """Output lines (one table row each) that start with ``status``."""
    return [line.strip() for line in result.output.splitlines() if line.strip().startswith(status)]


@pytest.mark.phase11
def test_doctor_local_hf_tts_without_torch_is_a_fail_row(impulse_project, mock_llm, cli, extra_imports):
    mock_llm.set_default("OK")
    extra_imports.package("transformers")    # the 'local' extra is installed ...
    extra_imports.package("faster_whisper")
    extra_imports.missing("torch")           # ... but torch is not
    config = _config(impulse_project, voice_config=LOCAL_VOICE_CONFIG)
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2, text
    fail_rows = _status_lines(result, "FAIL")
    assert [row for row in fail_rows if "voice TTS" in row and TORCH_MISSING in row], text
    assert not [row for row in _status_lines(result, "PASS") if "voice TTS" in row], text
    # STT (faster-whisper) needs no torch.
    assert [row for row in _status_lines(result, "PASS") if "voice STT" in row], text


@pytest.mark.phase11
def test_doctor_local_hf_tts_with_torch_passes(impulse_project, mock_llm, cli, extra_imports):
    # Regression guard: with torch installed (found by find_spec) the local_hf rows pass.
    mock_llm.set_default("OK")
    extra_imports.package("transformers")
    extra_imports.package("faster_whisper")
    extra_imports.package("torch")
    config = _config(impulse_project, voice_config=LOCAL_VOICE_CONFIG)
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert [row for row in _status_lines(result, "PASS") if "voice TTS" in row], text
    assert TORCH_MISSING not in text


@pytest.mark.phase11
def test_doctor_json_200_tts_answer_is_a_fail_row_with_the_server_answer(impulse_project, mock_llm, cli, monkeypatch):
    import httpx
    from conftest import MockVoiceServer

    class Json200TTSServer(MockVoiceServer):
        def _handle(self, request):
            if request.method == "POST" and request.url.path.rstrip("/").endswith("/audio/speech"):
                return httpx.Response(
                    200, content=b'{"error": "voice alloy is unavailable"}', headers={"content-type": "application/json"}
                )
            return super()._handle(request)

    mock_llm.set_default("OK")
    Json200TTSServer().install(monkeypatch)
    config = _config(impulse_project, voice_config=VOICE_CONFIG)
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2, text
    rows = [row for row in _status_lines(result, "FAIL") if "voice TTS" in row]
    assert rows, text
    assert "POST http://localhost:8000/v1/audio/speech returned application/json, not audio:" in rows[0]
    assert "voice alloy is unavailable" in rows[0]
    assert "bytes of audio" not in text


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 24): [phase 11] ``impulse doctor`` on ``adapter: pipecat`` in eval
# mode with NO pipecat package at all: the ``pipecat.evals`` row is FAIL with the SPEC.md section 9
# missing-extra text (the text ``impulse run`` gives for the same config), never the "Upgrade
# pipecat-ai" text, which stays for a pipecat-ai release without ``pipecat.evals``. The row label
# stays ``pipecat.evals``, so test_doctor_pipecat_eval_mode_needs_pipecat_evals (skipped while
# pipecat-ai is installed) stays valid; its body is also run here with the package hidden.
# ============================================================================================

PIPECAT_MISSING_EXTRA = "pipecat adapter needs the 'pipecat' extra: pip install impulse-evals[pipecat]"
PIPECAT_EVALS_MISSING = (
    "Pipecat eval mode needs a pipecat-ai release that includes pipecat.evals. "
    "Upgrade pipecat-ai, or set pipecat.mode: websocket."
)


@pytest.mark.phase11
def test_doctor_pipecat_eval_mode_without_the_pipecat_package_names_the_missing_extra(
    impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch, extra_imports
):
    mock_llm.set_default("OK")
    extra_imports.pipecat("missing")   # sys.modules["pipecat"] = None: no pipecat-ai installed
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    result = cli.invoke("doctor", _pipecat_config(impulse_project, mode="eval"))
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert f"FAIL pipecat.evals {PIPECAT_MISSING_EXTRA}" in text, text   # one row: label, then the text
    assert "Upgrade pipecat-ai" not in text, text
    assert mock_pipecat_channel.channel.connect_calls == []   # no handshake is tried without the package


@pytest.mark.phase11
def test_doctor_pipecat_eval_mode_needs_pipecat_evals_with_the_package_hidden(
    impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch, extra_imports
):
    # Regression guard: the body of test_doctor_pipecat_eval_mode_needs_pipecat_evals (skipped while
    # pipecat-ai is installed), run with the pipecat package hidden, passes before and after the fix.
    mock_llm.set_default("OK")
    extra_imports.pipecat("missing")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    mock_pipecat_channel.channel.on_send("client-ready", mock_pipecat_channel.rtvi("bot-ready", {"version": "1.0.0"}))
    result = cli.invoke("doctor", _pipecat_config(impulse_project, mode="eval"))
    assert result.exit_code == 2
    assert "pipecat.evals" in cli.text(result)


@pytest.mark.phase11
def test_doctor_pipecat_release_without_pipecat_evals_still_says_upgrade(
    impulse_project, mock_llm, mock_pipecat_channel, cli, monkeypatch, extra_imports
):
    # Regression guard: pipecat importable but without pipecat.evals keeps the upgrade text.
    mock_llm.set_default("OK")
    extra_imports.pipecat("no_evals")
    _patch_pipecat(monkeypatch, mock_pipecat_channel)
    result = cli.invoke("doctor", _pipecat_config(impulse_project, mode="eval"))
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert f"FAIL pipecat.evals {PIPECAT_EVALS_MISSING}" in text, text
    assert PIPECAT_MISSING_EXTRA not in text, text


# ============================================================================================
# Maintainer decision 2026-10-21, item 3 (AUDIT-FIXES.md, logged F19): [phase 11] local_hf TTS with
# a torch that is installed but fails to import (an on-disk fake torch package whose import raises
# the Windows DLL OSError) is a FAIL voice TTS row holding the exact ConfigError message
# (TORCH_IMPORT_FAILED, plus TORCH_WINDOWS_HINT on Windows), exit 2; STT (faster-whisper) still
# passes; torch is never imported in the doctor's own process.
# ============================================================================================

import sys  # noqa: E402

TORCH_IMPORT_FAILED = "local_hf TTS needs PyTorch, which is installed but could not be imported ({detail})."
TORCH_WINDOWS_HINT = (
    " On Windows, a DLL load error (for example a missing vcruntime140_1.dll) means the Microsoft Visual C++ "
    "Redistributable (x64) needs to be installed or updated."
)
TORCH_DLL_OSERROR = (
    '[WinError 126] The specified module could not be found. Error loading "C:\\fake\\torch\\lib\\c10.dll" '
    "or one of its dependencies."
)


def _broken_on_disk_torch(tmp_path, monkeypatch):
    """A ``torch`` package on disk whose import raises the Windows DLL ``OSError``, first on
    ``sys.path``; every ``sys.modules`` entry for torch is removed for the test."""
    site = tmp_path / "torch-site"
    (site / "torch").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text(f"raise OSError({TORCH_DLL_OSERROR!r})\n", encoding="utf-8")
    for loaded in [name for name in sys.modules if name == "torch" or name.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, loaded)
    monkeypatch.syspath_prepend(str(site))
    assert importlib.util.find_spec("torch") is not None and "torch" not in sys.modules


@pytest.mark.phase11
def test_doctor_local_hf_tts_with_torch_that_fails_to_import_is_a_fail_row(
    impulse_project, mock_llm, cli, extra_imports, monkeypatch, tmp_path
):
    mock_llm.set_default("OK")
    extra_imports.package("transformers")    # the 'local' extra is installed ...
    extra_imports.package("faster_whisper")
    _broken_on_disk_torch(tmp_path, monkeypatch)   # ... and torch is installed, but its import fails
    config = _config(impulse_project, voice_config=LOCAL_VOICE_CONFIG)
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    expected = TORCH_IMPORT_FAILED.format(detail=f"OSError: {TORCH_DLL_OSERROR}")
    expected += TORCH_WINDOWS_HINT if sys.platform == "win32" else ""
    assert [row for row in _status_lines(result, "FAIL") if "voice TTS" in row and row.endswith(expected)], text
    assert not [row for row in _status_lines(result, "PASS") if "voice TTS" in row], text
    assert result.exit_code == 2, text
    # STT (faster-whisper) needs no torch.
    assert [row for row in _status_lines(result, "PASS") if "voice STT" in row], text
    assert TORCH_MISSING not in text
    assert "torch" not in sys.modules
