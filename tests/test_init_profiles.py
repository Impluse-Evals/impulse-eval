# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 8: ``impulse init`` profiles and the language rule (SPEC.md sections 8.1 and 17).

Interactive answers are fed through ``CliRunner`` input, one line per question (contract logged in
DECISIONS.md, Stage B group 4-8): the language is the first question in every profile and has no
default (a blank or invalid code is asked again); every later question takes free text and accepts
a blank answer (the value keeps its ``YOUR_..._MODEL`` placeholder or stays empty). ``hosted`` asks
for the LLM keys right after the language. ``--no-ping`` is used unless a test checks the ping.
"""

import pytest

U = "ollama/impulse-test-model"  # valid as an Ollama tag, a LiteLLM model string, and a model id
SECRET = "sk-impulse-test-SECRET-7788"
PLACEHOLDERS = {"primary": "YOUR_PRIMARY_MODEL", "fast": "YOUR_FAST_MODEL", "simulator": "YOUR_SIMULATOR_MODEL"}
KEY_NAMES = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "IMPULSE_VOICE_API_KEY")


def _toml(project):
    from impulse_evals.config import load_toml

    path = project.root / ".impulse" / "config.toml"
    assert path.is_file(), "impulse init must write .impulse/config.toml"
    return load_toml(path), path.read_text(encoding="utf-8")


def _plain(value):
    return value.get_secret_value() if hasattr(value, "get_secret_value") else value


def _keys(cfg):
    return {name: _plain(value) for name, value in cfg.keys.items()}


def _models(cfg):
    models = cfg.models
    return {role: getattr(models, role) for role in ("primary", "fast", "simulator")}


def _language(cfg):
    return cfg.defaults.language


def _lines(*answers, repeat=None, count=12):
    lines = list(answers)
    if repeat is not None:
        lines += [repeat] * count
    return "\n".join(lines) + "\n"


def _assert_gitignore(project):
    path = project.root / ".impulse" / ".gitignore"
    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip() == "*"


# ============================================================================================
# --non-interactive and --language
# ============================================================================================


@pytest.mark.parametrize("profile", ["free", "hosted", "custom"])
def test_non_interactive_writes_given_language_and_placeholders(impulse_project, mock_llm, cli, profile):
    result = cli.invoke("init", "--profile", profile, "--non-interactive", "--language", "hi-IN")
    assert result.exit_code == 0, cli.text(result)
    cfg, text = _toml(impulse_project)
    assert 'language = "hi-IN"' in text
    assert _language(cfg) == "hi-IN"
    assert _models(cfg) == PLACEHOLDERS
    assert all(value == "" for value in _keys(cfg).values())
    assert set(KEY_NAMES) <= set(_keys(cfg))
    assert cfg.defaults.fail_on == "high"   # fixes round 4 (O3): superseded assertion, was "critical"
    assert cfg.defaults.reporter == "all"
    _assert_gitignore(impulse_project)
    assert mock_llm.call_count == 0  # --non-interactive runs no ping


def test_non_interactive_without_language_exits_2_naming_the_flag(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--non-interactive")
    assert result.exit_code == 2
    assert "--language" in cli.text(result)
    assert not (impulse_project.root / ".impulse" / "config.toml").exists()


@pytest.mark.parametrize("profile", ["free", "hosted", "custom"])
def test_non_interactive_never_writes_a_default_language(impulse_project, mock_llm, cli, profile):
    result = cli.invoke("init", "--profile", profile, "--non-interactive", "--no-ping")
    assert result.exit_code == 2
    assert "--language" in cli.text(result)
    assert not (impulse_project.root / ".impulse" / "config.toml").exists()


def test_non_interactive_invalid_language_exits_2(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--non-interactive", "--language", "English")
    assert result.exit_code == 2
    assert not (impulse_project.root / ".impulse" / "config.toml").exists()


def test_default_profile_is_custom(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--no-ping", input=_lines("en-IN", repeat="", count=30))
    assert result.exit_code == 0, cli.text(result)
    text = cli.text(result).lower()
    for role in ("primary", "fast", "simulator"):
        assert role in text
    cfg, _ = _toml(impulse_project)
    assert _language(cfg) == "en-IN"


# ============================================================================================
# Interactive: every profile asks for the language first
# ============================================================================================


@pytest.mark.parametrize("profile", ["free", "hosted", "custom"])
def test_every_profile_asks_for_the_language(impulse_project, mock_llm, cli, profile):
    result = cli.invoke("init", "--profile", profile, "--no-ping", input=_lines("hi-IN", repeat="", count=30))
    assert result.exit_code == 0, cli.text(result)
    assert "language" in cli.text(result).lower()
    cfg, text = _toml(impulse_project)
    assert _language(cfg) == "hi-IN"
    assert 'language = "hi-IN"' in text
    _assert_gitignore(impulse_project)


def test_blank_and_invalid_language_are_asked_again(impulse_project, mock_llm, cli):
    result = cli.invoke(
        "init", "--profile", "free", "--no-ping", input=_lines("", "English", "hi_IN", "hi-Latn-IN", repeat=U)
    )
    assert result.exit_code == 0, cli.text(result)
    cfg, _ = _toml(impulse_project)
    assert _language(cfg) == "hi-Latn-IN"


# ============================================================================================
# Profiles
# ============================================================================================


def test_free_profile_never_prompts_for_a_key(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--profile", "free", "--no-ping", input=_lines("en-IN", repeat=U))
    assert result.exit_code == 0, cli.text(result)
    cfg, _ = _toml(impulse_project)
    # Every answer after the language was U: had a key been asked for, U would be stored as a key.
    assert all(value == "" for value in _keys(cfg).values())
    for role, model in _models(cfg).items():
        assert "impulse-test-model" in model, f"{role} model not taken from the answers: {model!r}"
    assert mock_llm.call_count == 0


def test_free_profile_writes_starter_config_yaml_with_model_ids(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--profile", "free", "--no-ping", input=_lines("en-IN", repeat=U))
    assert result.exit_code == 0, cli.text(result)
    config_yaml = impulse_project.root / "config.yaml"
    assert config_yaml.is_file()
    text = config_yaml.read_text(encoding="utf-8")
    assert "voice_config" in text
    assert U in text


def test_free_profile_keeps_existing_config_yaml_and_prints_snippet(impulse_project, mock_llm, cli):
    config_yaml = impulse_project.root / "config.yaml"
    config_yaml.write_text("# SPDX-License-Identifier: AGPL-3.0-only\n# my own project config\n", encoding="utf-8")
    before = config_yaml.read_bytes()
    result = cli.invoke("init", "--profile", "free", "--no-ping", input=_lines("en-IN", repeat=U))
    assert result.exit_code == 0, cli.text(result)
    assert config_yaml.read_bytes() == before
    assert "voice_config" in cli.text(result)


def test_hosted_profile_prompts_for_keys_and_never_echoes_them(impulse_project, mock_llm, cli):
    result = cli.invoke(
        "init", "--profile", "hosted", "--no-ping", input=_lines("en-IN", SECRET, repeat="", count=30)
    )
    assert result.exit_code == 0, cli.text(result)
    assert SECRET not in result.output
    assert "key" in cli.text(result).lower()
    cfg, _ = _toml(impulse_project)
    assert _language(cfg) == "en-IN"
    assert SECRET in _keys(cfg).values()


def test_hosted_profile_keys_may_be_left_blank(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--profile", "hosted", "--no-ping", input=_lines("en-IN", repeat="", count=30))
    assert result.exit_code == 0, cli.text(result)
    cfg, _ = _toml(impulse_project)
    assert all(value == "" for value in _keys(cfg).values())


def test_custom_profile_asks_per_role(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--profile", "custom", "--no-ping", input=_lines("en-IN", repeat="", count=30))
    assert result.exit_code == 0, cli.text(result)
    text = cli.text(result).lower()
    for role in ("primary", "fast", "simulator"):
        assert role in text
    cfg, _ = _toml(impulse_project)
    assert _language(cfg) == "en-IN"
    assert _models(cfg) == PLACEHOLDERS  # blank answers keep the placeholders


# ============================================================================================
# Model string validation: live ping (skippable with --no-ping)
# ============================================================================================


def test_free_profile_pings_local_models_with_five_tokens(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    result = cli.invoke("init", "--profile", "free", input=_lines("en-IN", repeat=U))
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count >= 1
    assert all("impulse-test-model" in str(call["model"]) for call in mock_llm.calls)
    assert any(call["kwargs"].get("max_tokens") == 5 for call in mock_llm.calls)


def test_no_ping_skips_the_live_check(impulse_project, mock_llm, cli):
    result = cli.invoke("init", "--profile", "free", "--no-ping", input=_lines("en-IN", repeat=U))
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0


# ============================================================================================
# Audit batch 2 (AUDIT.md finding 22): .impulse/config.toml (it holds [keys]) is private. impulse
# init creates it with os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600) and os.fdopen,
# and on POSIX also chmods it to 0o600 after writing (an overwritten file keeps no wider mode);
# .impulse/.gitignore is still written first.
# ============================================================================================

import os  # noqa: E402
import stat  # noqa: E402
from pathlib import Path  # noqa: E402


def _is_config_toml(path):
    return str(path).replace("\\", "/").endswith(".impulse/config.toml")


@pytest.fixture
def os_open_calls(monkeypatch):
    """Every ``os.open(path, flags, mode)`` call, recorded and then performed."""
    calls = []
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        calls.append({"path": os.fspath(path), "flags": flags, "mode": mode})
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    return calls


@pytest.mark.parametrize(
    "args, answers",
    [
        pytest.param(("--profile", "hosted", "--no-ping"), _lines("en-IN", SECRET, repeat="", count=30), id="hosted"),
        pytest.param(("--profile", "free", "--no-ping"), _lines("en-IN", repeat=U), id="free"),
        pytest.param(("--non-interactive", "--language", "en-IN"), None, id="non-interactive"),
    ],
)
def test_init_creates_config_toml_with_mode_0600(impulse_project, mock_llm, cli, os_open_calls, args, answers):
    result = cli.invoke("init", *args, input=answers)
    assert result.exit_code == 0, cli.text(result)
    toml_path = impulse_project.root / ".impulse" / "config.toml"
    created = [call for call in os_open_calls if _is_config_toml(call["path"])]
    assert created, "config.toml must be created with os.open and an explicit mode"
    for call in created:
        assert Path(call["path"]).resolve() == toml_path.resolve()
        assert stat.S_IMODE(call["mode"]) == 0o600, oct(call["mode"])
        assert call["flags"] & os.O_CREAT, "os.open must create the file (O_CREAT)"
        assert call["flags"] & os.O_TRUNC, "an existing file is truncated (O_TRUNC), not appended to"
        assert call["flags"] & (os.O_WRONLY | os.O_RDWR), "the file is opened for writing"
    # Regression guards: the file is complete and loads, and .gitignore is still written.
    cfg, _text = _toml(impulse_project)
    assert _language(cfg) == "en-IN"
    if "hosted" in args:
        assert SECRET in _keys(cfg).values()
    _assert_gitignore(impulse_project)


def test_init_overwrite_still_writes_the_new_config_toml(impulse_project, mock_llm, cli, os_open_calls):
    # Interactive overwrite (answer y): the old file is replaced, not appended to.
    impulse_project.write_toml('[keys]\nOPENAI_API_KEY = "sk-impulse-test-OLD-0001"\n# old file\n')
    result = cli.invoke("init", "--profile", "hosted", "--no-ping", input="y\n" + _lines("en-IN", SECRET, repeat="", count=30))
    assert result.exit_code == 0, cli.text(result)
    cfg, text = _toml(impulse_project)
    assert "sk-impulse-test-OLD-0001" not in text and "# old file" not in text
    assert SECRET in _keys(cfg).values()
    created = [call for call in os_open_calls if _is_config_toml(call["path"])]
    assert created and all(stat.S_IMODE(call["mode"]) == 0o600 for call in created)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not observable through stat() on Windows")
@pytest.mark.parametrize("existing", [False, True], ids=["new_file", "overwritten_0644_file"])
def test_init_config_toml_mode_is_0600_on_posix(impulse_project, mock_llm, cli, existing):
    toml_path = impulse_project.root / ".impulse" / "config.toml"
    answers = _lines("en-IN", SECRET, repeat="", count=30)
    if existing:
        impulse_project.write_toml('[keys]\nOPENAI_API_KEY = ""\n')
        os.chmod(toml_path, 0o644)
        answers = "y\n" + answers
    result = cli.invoke("init", "--profile", "hosted", "--no-ping", input=answers)
    assert result.exit_code == 0, cli.text(result)
    assert stat.S_IMODE(toml_path.stat().st_mode) == 0o600, oct(toml_path.stat().st_mode)
    assert SECRET in _keys(_toml(impulse_project)[0]).values()


# ============================================================================================
# Audit batch 2 review fix-ups (AUDIT.md finding 22, follow-up): os.open's mode applies only when the
# file is CREATED, so when impulse init overwrites an existing, wider-mode config.toml the new
# [keys] would be written while the file is still e.g. 0644 and only narrowed by the final chmod.
# On POSIX the descriptor is narrowed with os.fchmod(fd, 0o600) right after os.open, before any
# byte of the new file is written; the final os.chmod(path, 0o600) stays.
# ============================================================================================


@pytest.fixture
def config_toml_write_modes(monkeypatch):
    """The permission bits of config.toml (``os.fstat`` of its descriptor) at every write of new
    bytes to it: through the file object ``os.fdopen`` makes from the ``os.open`` descriptor, or
    through ``os.write`` on that descriptor. Writes to any other file are not recorded."""
    modes = []
    toml_fds = {}   # fd -> (st_dev, st_ino) of the config.toml it was opened for
    real_open, real_fdopen, real_write = os.open, os.fdopen, os.write

    def is_toml_fd(fd):
        identity = toml_fds.get(fd)
        if identity is None:
            return False
        try:
            info = os.fstat(fd)
        except OSError:   # closed since: not config.toml any more
            return False
        return (info.st_dev, info.st_ino) == identity

    def record(fd):
        modes.append(stat.S_IMODE(os.fstat(fd).st_mode))

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        fd = real_open(path, flags, mode, *args, **kwargs)
        if _is_config_toml(os.fspath(path)):
            info = os.fstat(fd)
            toml_fds[fd] = (info.st_dev, info.st_ino)
        return fd

    class ModeRecordingFile:
        """A file object whose writes record the descriptor's mode first."""

        def __init__(self, handle, fd):
            self._handle = handle
            self._fd = fd

        def write(self, data):
            if data:
                record(self._fd)
            return self._handle.write(data)

        def writelines(self, lines):
            for line in lines:
                self.write(line)

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *exc_info):
            return self._handle.__exit__(*exc_info)

        def __iter__(self):
            return iter(self._handle)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def recording_fdopen(fd, *args, **kwargs):
        handle = real_fdopen(fd, *args, **kwargs)
        return ModeRecordingFile(handle, fd) if is_toml_fd(fd) else handle

    def recording_write(fd, data):
        if data and is_toml_fd(fd):
            record(fd)
        return real_write(fd, data)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "fdopen", recording_fdopen)
    monkeypatch.setattr(os, "write", recording_write)
    return modes


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not observable through fstat() on Windows")
@pytest.mark.parametrize("old_mode", [0o644, 0o666], ids=["overwritten_0644_file", "overwritten_0666_file"])
def test_init_overwrite_narrows_config_toml_to_0600_before_writing_any_key(
    impulse_project, mock_llm, cli, config_toml_write_modes, old_mode
):
    toml_path = impulse_project.root / ".impulse" / "config.toml"
    impulse_project.write_toml('[keys]\nOPENAI_API_KEY = ""\n')
    os.chmod(toml_path, old_mode)
    assert stat.S_IMODE(toml_path.stat().st_mode) == old_mode
    answers = "y\n" + _lines("en-IN", SECRET, repeat="", count=30)
    result = cli.invoke("init", "--profile", "hosted", "--no-ping", input=answers)
    assert result.exit_code == 0, cli.text(result)
    assert config_toml_write_modes, "the new config.toml must be written through its os.open descriptor"
    assert [oct(mode) for mode in config_toml_write_modes] == [oct(0o600)] * len(config_toml_write_modes), (
        "config.toml was still readable by others when its new bytes ([keys]) were written"
    )
    # Regression guards: the final mode is 0600 and the new keys are in the file.
    assert stat.S_IMODE(toml_path.stat().st_mode) == 0o600, oct(toml_path.stat().st_mode)
    assert SECRET in _keys(_toml(impulse_project)[0]).values()


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 49, maintainer decision #49): re-running
# `impulse init --non-interactive` in a folder whose .impulse/config.toml already exists (a CI job on
# a cached workspace) exits 0. .impulse/.gitignore is (re)written with the single line "*", the CLI
# prints "{TOML_PATH} already exists; left unchanged." (TOML_PATH as the CLI shows it everywhere:
# .impulse/config.toml with the platform's path separator), and nothing else is written: config.toml
# stays byte-identical, no starter config.yaml, no ping. Interactive mode keeps the overwrite
# question, and --language stays mandatory with --non-interactive.
# ============================================================================================

import sys  # noqa: E402

TOML_DISPLAY = str(Path(".impulse") / "config.toml")
LEFT_UNCHANGED = f"{TOML_DISPLAY} already exists; left unchanged."
USER_TOML = (
    "# my own settings, edited by hand\n"
    "[keys]\n"
    'OPENAI_API_KEY = ""\n'
    "\n"
    "[models]\n"
    'primary = "ollama/user-choice"\n'
    "\n"
    "[defaults]\n"
    'language = "hi-IN"\n'
)


def _project_files(project):
    """Every file under the project folder (relative POSIX path -> bytes; ``__pycache__`` excluded)."""
    return {name: (project.root / name).read_bytes() for name in project.files()}


@pytest.mark.parametrize("profile", ["free", "hosted", "custom"])
@pytest.mark.parametrize("gitignore", [None, "!config.toml\n"], ids=["no_gitignore", "stale_gitignore"])
def test_non_interactive_rerun_on_existing_config_toml_exits_0_and_leaves_it_unchanged(
    impulse_project, mock_llm, cli, profile, gitignore
):
    toml_path = impulse_project.write_toml(USER_TOML)
    gitignore_path = impulse_project.root / ".impulse" / ".gitignore"
    if gitignore is not None:
        gitignore_path.write_text(gitignore, encoding="utf-8")
    before = _project_files(impulse_project)
    result = cli.invoke("init", "--profile", profile, "--non-interactive", "--language", "en-IN")
    assert result.exit_code == 0, cli.text(result)
    text = cli.text(result)
    assert LEFT_UNCHANGED in text, text
    assert f"Wrote {TOML_DISPLAY}" not in text, text
    assert toml_path.read_bytes() == before[".impulse/config.toml"]
    _assert_gitignore(impulse_project)   # (re)written with the single line "*"
    after = _project_files(impulse_project)
    assert set(after) == set(before) | {".impulse/.gitignore"}, sorted(set(after) ^ set(before))
    for name, data in before.items():
        if name != ".impulse/.gitignore":
            assert after[name] == data, f"{name} was rewritten"
    assert not (impulse_project.root / "config.yaml").exists(), "nothing else is written (no starter config)"
    assert mock_llm.call_count == 0


@pytest.mark.parametrize("profile", ["free", "custom"])
def test_second_non_interactive_init_exits_0_and_changes_nothing(impulse_project, mock_llm, cli, profile):
    first = cli.invoke("init", "--profile", profile, "--non-interactive", "--language", "en-IN")
    assert first.exit_code == 0, cli.text(first)
    before = _project_files(impulse_project)
    second = cli.invoke("init", "--profile", profile, "--non-interactive", "--language", "hi-IN")
    assert second.exit_code == 0, cli.text(second)
    assert LEFT_UNCHANGED in cli.text(second), cli.text(second)
    assert _project_files(impulse_project) == before
    cfg, _ = _toml(impulse_project)
    assert _language(cfg) == "en-IN"   # the second run's --language changed nothing
    assert mock_llm.call_count == 0


def test_non_interactive_rerun_still_requires_language(impulse_project, mock_llm, cli):
    # Regression guard (passes before the fix too): --language stays mandatory with --non-interactive.
    toml_path = impulse_project.write_toml(USER_TOML)
    before = toml_path.read_bytes()
    result = cli.invoke("init", "--non-interactive")
    assert result.exit_code == 2
    assert "--language" in cli.text(result)
    assert LEFT_UNCHANGED not in cli.text(result)
    assert toml_path.read_bytes() == before


def test_interactive_rerun_keeps_the_overwrite_question(impulse_project, mock_llm, cli):
    # Regression guard (passes before the fix too): answering "n" keeps the file and writes nothing.
    toml_path = impulse_project.write_toml(USER_TOML)
    before = toml_path.read_bytes()
    result = cli.invoke("init", "--profile", "custom", "--no-ping", input="n\n")
    assert result.exit_code == 0, cli.text(result)
    text = cli.text(result)
    assert f"{TOML_DISPLAY} already exists. Overwrite it?" in text, text
    assert "Nothing written." in text
    assert toml_path.read_bytes() == before
    assert mock_llm.call_count == 0


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 43): a failed init ping never shows LiteLLM's coloured
# "Give Feedback / Get Help" banner. LiteLLM prints it to stdout from its exception mapping unless
# litellm.suppress_debug_info is True. Every Impulse module that imports litellm sets
# litellm.suppress_debug_info = True right after the import, and the ping runs under
# contextlib.redirect_stdout / redirect_stderr. The mocked completion below prints the banner to
# stdout and stderr unconditionally (as any library could) and then fails.
# ============================================================================================

LITELLM_BANNER = "Give Feedback / Get Help"
LITELLM_BANNER_LINE = f"\033[1;31m{LITELLM_BANNER}: https://github.com/BerriAI/litellm/issues/new\033[0m"
LITELLM_INFO_LINE = "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'."


def _print_litellm_banner_then_fail(call):
    for stream in (sys.stdout, sys.stderr):
        print(file=stream)
        print(LITELLM_BANNER_LINE, file=stream)
        print(LITELLM_INFO_LINE, file=stream)
        print(file=stream)
    raise ConnectionError("model server is not reachable")


def test_failed_init_ping_prints_no_litellm_banner(impulse_project, mock_llm, cli):
    mock_llm.responder = _print_litellm_banner_then_fail
    result = cli.invoke("init", "--profile", "free", input=_lines("en-IN", repeat=U))
    assert result.exit_code == 0, cli.text(result)   # a failed ping still exits 0
    assert mock_llm.call_count >= 1, "the ping ran"
    assert "not reachable" in cli.text(result)       # the failure itself is still reported
    assert LITELLM_BANNER not in result.output, result.output
    assert "LiteLLM.Info" not in result.output, result.output


def test_init_sets_litellm_suppress_debug_info(impulse_project, mock_llm, cli, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "suppress_debug_info", False)   # restored after the test
    mock_llm.set_default("OK")
    result = cli.invoke("init", "--profile", "free", input=_lines("en-IN", repeat=U))
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count >= 1, "the ping ran"
    assert litellm.suppress_debug_info is True
