# SPDX-License-Identifier: Apache-2.0
"""The ``impulse`` command line (SPEC.md sections 8.1, 15, 16, and 17).

Commands: ``init``, ``doctor``, ``run``, ``report``, ``compare``, ``generate``, ``convert-failure``,
``calibrate``.

Exit codes: 0 success; 1 when the ``--fail-on`` condition is met (``impulse run``), a critical
regression exists (``impulse compare``), or generation / conversion fails (``impulse generate`` /
``impulse convert-failure``: the answer is still invalid after one retry, or the model call fails); 2
for a usage or configuration error (including an invalid ``impulse calibrate`` folder or item file,
and a judge call that raises during calibration), for end of input at a prompt ("Aborted."), and for
any unexpected exception ("Error: unexpected error: ..."; never a traceback). Every command is registered with ``@_command(name)``, which adds that mapping. A config
error prints ``str(ConfigError)`` (the exact field path and message). Every text the CLI prints from an
exception or a library message has every known key value redacted first and is shortened only after
that, so key values are never printed, not even in part.

Output is plain text written with ``click.echo`` (no console markup, no line wrapping), so user text
such as ``[turn 2]`` or a field path like ``scenarios[0].metrics[0]`` is printed exactly as it is. A
character the console cannot encode is printed as a backslash escape, never an error.
``.impulse/`` is relative to the current directory; when a command creates it, it also writes
``.impulse/.gitignore`` with the single line ``*``.
"""

import asyncio
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, NoReturn, Optional

import click
import typer
from pydantic import ValidationError

# LiteLLM fetches its model cost map from the internet at import unless this variable is "True";
# use the copy bundled with LiteLLM instead. Set before anything can import LiteLLM (no import above
# or below loads it at module level: it is imported lazily); an explicit user value is respected.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from impulse_evals.config import (
    LANGUAGE_PATTERN,
    LANGUAGE_PLACEHOLDER,
    MODEL_ENV_VARS,
    ROLES,
    STANDARD_KEY_NAMES,
    ConfigError,
    ProjectConfig,
    SimulatorConfig,
    TomlConfig,
    UnknownProviderError,
    api_key_for_role,
    authenticating_provider,
    is_placeholder,
    key_name_for_model,
    load_config,
    load_project_toml,
    load_toml,
    model_needs_key,
    redact,
    resolve_key,
    resolve_model,
    resolve_model_options,
    role_custom_llm_provider,
    static_key_values,
    strip_url_credentials,
)
from impulse_evals.compare import HIGH_METRICS
from impulse_evals.models import RunResult, Scenario

__all__ = ["app", "fail_on_reasons"]

# ------------------------------------------------------------------------------------------------
# LiteLLM's bundled tiktoken files (SPEC.md section 17; AUDIT.md finding 11, final check N1)
# ------------------------------------------------------------------------------------------------

# LiteLLM points tiktoken at its bundled tokenizers folder (litellm/litellm_core_utils/tokenizers/).
# A LiteLLM wheel whose files there have CRLF line endings (seen on Windows) fails tiktoken's sha256
# check, and tiktoken then downloads cl100k_base from this host: a local-only setup would need the
# internet. Importing the CLI repairs that once, without importing LiteLLM.
TIKTOKEN_HOST = "openaipublic.blob.core.windows.net"
TIKTOKEN_HINT = (
    "LiteLLM tried to download the tiktoken cl100k_base encoding; this needs internet access once, or set "
    "CUSTOM_TIKTOKEN_CACHE_DIR to a folder holding the file (not a model or provider failure)"
)
# The only variable that skips the repair (maintainer decision 2026-10-21, item 1). At its import
# LiteLLM sets TIKTOKEN_CACHE_DIR to this variable's value when it is non-empty, otherwise to its
# bundled folder, so a user-set TIKTOKEN_CACHE_DIR never reaches tiktoken and does not count.
TIKTOKEN_CACHE_VARIABLE = "CUSTOM_TIKTOKEN_CACHE_DIR"
TIKTOKEN_REPAIR_FOLDER = "impulse-tiktoken"
_CRLF_PROBE_BYTES = 65536   # read per bundled file to detect CRLF line endings


def _bundled_tokenizers_folder() -> Optional[Path]:
    """LiteLLM's bundled tokenizers folder, located with ``importlib.util.find_spec`` (which finds a
    top-level package without importing it), or None."""
    try:
        spec = importlib.util.find_spec("litellm")
    except (ImportError, ValueError):
        return None
    for location in (spec.submodule_search_locations or []) if spec is not None else []:
        folder = Path(location) / "litellm_core_utils" / "tokenizers"
        if folder.is_dir():
            return folder
    return None


def _has_crlf(path: Path) -> bool:
    """True when the first 64 KB of ``path`` hold a CRLF line ending (only that much is read)."""
    with path.open("rb") as handle:
        return b"\r\n" in handle.read(_CRLF_PROBE_BYTES)


def _is_complete_copy(folder: Path, names: Iterable[str]) -> bool:
    return folder.is_dir() and all((folder / name).is_file() for name in names)


def _write_repaired_copy(files: list[Path], crlf: set[str], target: Path) -> None:
    """Write the copy into a new staging folder next to ``target``, then rename it into place in one
    step, so a concurrent CLI start sees either no folder or a complete one. Rename first, remove
    only after a failed rename: when ``target`` is complete once the staging copy is written (another
    start finished first), it is kept untouched and the staging copy is dropped; otherwise the
    staging copy is renamed onto ``target`` at once. Only when that rename fails AND ``target`` is
    then found incomplete (an existing incomplete folder, e.g. emptied by hand) is ``target`` removed
    and the rename tried once more; a second failure keeps a complete ``target`` and otherwise
    raises (the caller then leaves LiteLLM's default). So ``target`` is removed only after this
    start's own rename onto it failed and it was then found incomplete; a complete copy another start
    renames into place while this one works is kept, except in that case: a copy landing just before
    the ``rmtree`` is removed too (and normally replaced by this start's identical copy; on Windows a
    file of it held open can leave it incomplete). Closing that window would need a lock file. The
    staging folder never survives."""
    names = [path.name for path in files]
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{target.name}.", dir=target.parent))
    try:
        for path in files:
            data = path.read_bytes()
            (staging / path.name).write_bytes(data.replace(b"\r\n", b"\n") if path.name in crlf else data)
        if _is_complete_copy(target, names):   # another start finished first: keep its copy
            return
        try:
            os.replace(staging, target)
        except OSError:   # a folder is already there (or the rename failed for another reason)
            if _is_complete_copy(target, names):   # another start's complete copy: keep it
                return
            shutil.rmtree(target, ignore_errors=True)   # an incomplete folder: replaced
            try:
                os.replace(staging, target)
            except OSError:
                if not _is_complete_copy(target, names):
                    raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def repair_bundled_tiktoken_cache() -> Optional[Path]:
    """Unless ``CUSTOM_TIKTOKEN_CACHE_DIR`` is set (to a non-empty value), when a file in LiteLLM's
    bundled tokenizers folder has CRLF line endings: a repaired copy of the folder (CRLF -> LF for
    those files, every other file copied as it is) in
    ``<tempfile.gettempdir()>/impulse-tiktoken/<hash of the bundled file names and sizes>/`` (reused
    when already complete), and ``CUSTOM_TIKTOKEN_CACHE_DIR`` set to it (LiteLLM then points
    ``TIKTOKEN_CACHE_DIR`` there). Returns that folder, or None when nothing was needed or possible:
    nothing is written when no file has CRLF, a user-set ``CUSTOM_TIKTOKEN_CACHE_DIR`` is never
    overridden, LiteLLM is never imported, and any error leaves LiteLLM's default behaviour.

    A user-set ``TIKTOKEN_CACHE_DIR`` does not skip the repair: LiteLLM overwrites it at its import
    (with ``CUSTOM_TIKTOKEN_CACHE_DIR``, else its bundled folder). Its value is left as it is in
    ``os.environ`` and nothing is written into the folder it names."""
    if os.environ.get(TIKTOKEN_CACHE_VARIABLE):
        return None
    try:
        bundled = _bundled_tokenizers_folder()
        if bundled is None:
            return None
        files = sorted((path for path in bundled.iterdir() if path.is_file()), key=lambda path: path.name)
        crlf = {path.name for path in files if _has_crlf(path)}
        if not crlf:
            return None
        listing = "\n".join(f"{path.name}:{path.stat().st_size}" for path in files)
        target = Path(tempfile.gettempdir()) / TIKTOKEN_REPAIR_FOLDER / hashlib.sha256(listing.encode()).hexdigest()[:16]
        if not _is_complete_copy(target, [path.name for path in files]):
            _write_repaired_copy(files, crlf, target)
    except Exception:  # noqa: BLE001 - never stops the CLI: LiteLLM's default behaviour stays
        return None
    os.environ[TIKTOKEN_CACHE_VARIABLE] = str(target)
    return target


repair_bundled_tiktoken_cache()   # at import, before anything imports LiteLLM

# ------------------------------------------------------------------------------------------------
# Exact texts and file locations (SPEC.md sections 8.1, 9, 15, 17)
# ------------------------------------------------------------------------------------------------

PRIVACY_NOTICE = "Impulse sends your system prompt and transcripts to the LLM and voice providers you configured."
PIPECAT_EVALS_MISSING = (
    "Pipecat eval mode needs a pipecat-ai release that includes pipecat.evals. "
    "Upgrade pipecat-ai, or set pipecat.mode: websocket."
)
LIVEKIT_EXTRA_MISSING = "livekit adapter needs the 'livekit' extra: pip install impulse-evals[livekit]"

IMPULSE_DIR = Path(".impulse")
RUNS_DIR = IMPULSE_DIR / "runs"
TOML_PATH = IMPULSE_DIR / "config.toml"
GITIGNORE_PATH = IMPULSE_DIR / ".gitignore"
PRIVACY_MARKER = IMPULSE_DIR / ".privacy_notice_shown"
STARTER_CONFIG = Path("config.yaml")
STARTER_TEMPLATE = "free_config.yaml"

MODEL_PLACEHOLDERS = {"primary": "YOUR_PRIMARY_MODEL", "fast": "YOUR_FAST_MODEL", "simulator": "YOUR_SIMULATOR_MODEL"}
TTS_PLACEHOLDER = "YOUR_TTS_MODEL"
STT_PLACEHOLDER = "YOUR_STT_MODEL"
ROLE_LABELS = {"primary": "primary judge", "fast": "fast judge", "simulator": "simulator (simulated caller)"}
LLM_KEY_NAMES = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
VOICE_KEY_NAME = "IMPULSE_VOICE_API_KEY"
OLLAMA_PREFIXES = ("ollama/", "ollama_chat/")
# Doctor model row of a provider for which LiteLLM names no *_API_KEY variable (config.key_name_for_model).
NO_IMPULSE_KEY_NOTE = "needs no Impulse key (LiteLLM uses the provider's own credentials)"

PING_MAX_TOKENS = 5
PING_TIMEOUT_S = 60
PING_MESSAGES = [{"role": "user", "content": "Reply with the single word OK."}]
_MAX_ERROR_CHARS = 300

def _safe_console_streams() -> None:
    """Let stdout / stderr write characters their encoding cannot represent as backslash escapes
    instead of raising ``UnicodeEncodeError`` (e.g. a Devanagari scenario name on a Windows code page
    with redirected output). Only plain text streams in strict mode are changed."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.errors == "strict":
            with contextlib.suppress(Exception):
                stream.reconfigure(errors="backslashreplace")


class _ImpulseTyper(typer.Typer):
    """``typer.Typer`` whose console-script call (``impulse = impulse_evals.cli:app``) first makes the
    console streams encoding-safe, so typer's own usage messages cannot crash on encoding either."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _safe_console_streams()
        return super().__call__(*args, **kwargs)


app = _ImpulseTyper(
    name="impulse",
    help="Impulse Evals: pytest-style evaluation for AI voice and chat agents.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,   # locals could hold key values
)


class Profile(str, Enum):
    free = "free"
    hosted = "hosted"
    custom = "custom"


class Reporter(str, Enum):
    json = "json"
    html = "html"
    all = "all"


class FailOn(str, Enum):
    critical = "critical"
    high = "high"
    any = "any"


class CompareFailOn(str, Enum):
    """``impulse compare --fail-on`` (FIXES-ROUND-2-SPEC.md A3): no ``any`` level."""

    critical = "critical"
    high = "high"


# ------------------------------------------------------------------------------------------------
# Console helpers
# ------------------------------------------------------------------------------------------------


def _echo(message: str = "", *, err: bool = False) -> None:
    """Print one line exactly as given (no markup, no wrapping); never fails on encoding."""
    try:
        click.echo(message, err=err)
    except UnicodeEncodeError:
        click.echo(message.encode("ascii", "backslashreplace").decode("ascii"), err=err)


def _fail(message: str, code: int = 2) -> NoReturn:
    """Print an error to stderr and exit (default: 2, a usage or configuration error)."""
    _echo(f"Error: {message}", err=True)
    raise typer.Exit(code)


_STATUS_COLORS = {"PASS": "green", "FAIL": "red", "INFO": "cyan", "SKIP": "yellow"}


def _status(word: str) -> str:
    """A status word padded to 4 characters and coloured (colour is dropped when not a terminal)."""
    return click.style(word.ljust(4), fg=_STATUS_COLORS.get(word))


# ------------------------------------------------------------------------------------------------
# Redaction (SPEC.md section 8.1: keys never appear in logs, reports, or exceptions)
# ------------------------------------------------------------------------------------------------


class _Secrets:
    """The key values the running command knows about: config.toml ``[keys]`` and the variables named
    there, the standard key variables and every ``*_API_KEY`` environment variable
    (``config.secret_values``), the variables and literal keys the project config names
    (``runner.config_redactor``), keys typed at an ``impulse init`` prompt, and every key the command
    resolved for a role, whatever its variable's name (``learn``: the doctor's model rows, the init
    pings, ``runner.resolved_role_keys`` before a real run, and the standalone commands; for a dry run,
    which resolves no key, ``config.static_key_values()``). Reset when a command starts and when it
    ends."""

    def __init__(self) -> None:
        self.reset()

    def learn(self, value: Optional[str]) -> None:
        """Remember a key value the command resolved, so everything it prints later has it redacted."""
        if isinstance(value, str) and value and value not in self.values:
            self.values.append(value)

    def reset(self) -> None:
        self.toml: Optional[TomlConfig] = None
        self._toml_tried = False
        self.redactors: list[Callable[[str], str]] = []
        self.values: list[str] = []

    def use_toml(self, toml: Optional[TomlConfig]) -> None:
        self._toml_tried = True
        if toml is not None:
            self.toml = toml

    def current_toml(self) -> Optional[TomlConfig]:
        """The command's config.toml; when it has not loaded one, ``./.impulse/config.toml`` is read
        once here (best effort: an unreadable or invalid file only means fewer known values)."""
        if self.toml is None and not self._toml_tried:
            self._toml_tried = True
            try:
                self.toml = load_project_toml()
            except Exception:  # noqa: BLE001 - redaction must never fail
                self.toml = None
        return self.toml


_SECRETS = _Secrets()


def _redact(text: str, extra_values: Iterable[str] = ()) -> str:
    """``text`` with every known key value replaced by ``[REDACTED]``. Always applied BEFORE any
    shortening, so a key that would cross the cut cannot leak in part."""
    text = str(text)
    for redactor in _SECRETS.redactors:
        text = redactor(text)
    values = [value for value in [*_SECRETS.values, *extra_values] if isinstance(value, str)]
    return redact(text, _SECRETS.current_toml(), extra_values=values)


def _one_line(redacted_text: str, limit: int = _MAX_ERROR_CHARS) -> str:
    """Whitespace collapsed and cut to ``limit`` characters. Only ever pass text that already went
    through ``_redact``."""
    text = " ".join(str(redacted_text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _short_text(text: str, limit: int = _MAX_ERROR_CHARS, extra_values: Iterable[str] = ()) -> str:
    """Redacted first, then shortened to one line."""
    return _one_line(_redact(text, extra_values), limit)


def _exception_text(exc: BaseException, extra_values: Iterable[str] = ()) -> str:
    """``"{Type}: {message}"`` on one line, redacted, then shortened. A message that mentions the
    tiktoken download host (``TIKTOKEN_HOST``, anywhere in the full text) gets ``"; {TIKTOKEN_HINT}"``
    after the shortened text, so a failed ping or command is not blamed on the model or provider."""
    try:
        message = str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ must not hide the original error
        message = ""
    text = _short_text(message, extra_values=extra_values)
    result = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return f"{result}; {TIKTOKEN_HINT}" if TIKTOKEN_HOST in message else result


def _config_error_text(exc: ConfigError) -> str:
    return _redact(str(exc))


def _load_config_or_exit(path: Path) -> ProjectConfig:
    try:
        return load_config(path)
    except ConfigError as exc:
        _fail(_config_error_text(exc))


def _load_toml_or_exit() -> Optional[TomlConfig]:
    try:
        toml = load_project_toml()
    except ConfigError as exc:
        _SECRETS.use_toml(None)
        _fail(_config_error_text(exc))
    _SECRETS.use_toml(toml)
    return toml


# ------------------------------------------------------------------------------------------------
# Command registration: every error maps to an exit code (SPEC.md section 17)
# ------------------------------------------------------------------------------------------------

ABORTED_MESSAGE = "Aborted."


def _guarded(func: Callable[..., Any]) -> Callable[..., Any]:
    """Run a command so that no error escapes as a traceback: end of input (or Ctrl+C) at a prompt
    prints "Aborted." and exits 2; any other unexpected exception prints
    "Error: unexpected error: {Type}: {message}" (redacted, then shortened) to stderr and exits 2.
    Exit 1 stays reserved for the conditions of SPEC.md section 17."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _SECRETS.reset()
        try:
            return func(*args, **kwargs)
        except (typer.Exit, click.exceptions.Exit, click.ClickException):
            raise
        except (typer.Abort, click.exceptions.Abort):
            _echo(ABORTED_MESSAGE, err=True)
            raise typer.Exit(2) from None
        except Exception as exc:  # noqa: BLE001 - the last line of defence: one line, exit 2
            _echo(f"Error: unexpected error: {_exception_text(exc)}", err=True)
            raise typer.Exit(2) from None
        finally:
            _SECRETS.reset()

    return wrapper


def _command(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """``@app.command(name)`` with ``_guarded``; later phases register their commands with it too."""

    def register(func: Callable[..., Any]) -> Callable[..., Any]:
        app.command(name)(_guarded(func))
        return func

    return register


def _ensure_impulse_dir() -> None:
    """Create ``.impulse/``; a newly created folder also gets ``.gitignore`` with ``*`` (run output and
    keys never belong in version control)."""
    from impulse_evals.reporter import ensure_impulse_dir

    ensure_impulse_dir(IMPULSE_DIR)


def _write_or_exit(action: Callable[[], Any], path: Path) -> Any:
    try:
        return action()
    except OSError as exc:   # the path can hold a file-supplied name (convert-failure): redacted
        _fail(_redact(f"cannot write {path}: {exc.strerror or type(exc).__name__}"))


# ------------------------------------------------------------------------------------------------
# --fail-on (SPEC.md section 17)
# ------------------------------------------------------------------------------------------------


def fail_on_reasons(run: RunResult, level: str) -> list[str]:
    """Why ``impulse run --fail-on {level}`` exits 1; an empty list means exit 0.

    Each level includes everything above it, and a scenario error counts at every level:
    ``critical``: a failed critical assertion, or a scenario error (including a critical check the
    judge could not evaluate); ``high`` (the default since fixes round 4): also a failed high assertion,
    a high assertion with no judge verdict (L16), or a poor ``HIGH_METRICS`` metric, UNIV-019 or UNIV-025
    (both count as severity high; O3); ``any``: any scenario with ``passed == False``. One line per
    scenario.
    """
    if level not in ("critical", "high", "any"):
        raise ValueError(f"unknown --fail-on level {level!r}")
    reasons: list[str] = []
    for result in run.scenario_results:
        name = result.scenario_name
        if result.error is not None:
            reasons.append(f"{name}: scenario error: {result.error}")
            continue
        failed = [score for score in result.assertion_results if score.result == "fail"]
        critical = [score.rule_id for score in failed if score.severity == "critical"]
        if critical:
            reasons.append(f"{name}: critical assertion failed: {', '.join(critical)}")
            continue
        if level in ("high", "any"):
            high = [score.rule_id for score in failed if score.severity == "high"]
            if high:
                reasons.append(f"{name}: high assertion failed: {', '.join(high)}")
                continue
            no_verdict = [
                score.rule_id
                for score in result.assertion_results
                if score.severity == "high" and score.result == "skip" and score.no_verdict
            ]
            if no_verdict:
                reasons.append(f"{name}: high assertion had no verdict: {', '.join(no_verdict)}")
                continue
            poor = [
                metric_id
                for metric_id in HIGH_METRICS
                if any(
                    metric.metric_id == metric_id and metric.label == "poor" and not metric.skipped and not metric.informational
                    for metric in result.metric_results
                )
            ]
            if len(poor) == 1:
                reasons.append(f"{name}: {poor[0]} is poor (counts as severity high)")
                continue
            if poor:
                reasons.append(f"{name}: {' and '.join(poor)} are poor (count as severity high)")
                continue
        if level == "any" and not result.passed:
            reasons.append(f"{name}: scenario failed")
    return reasons


# ------------------------------------------------------------------------------------------------
# impulse run
# ------------------------------------------------------------------------------------------------


def _privacy_notice_due() -> bool:
    """True when this directory has no privacy marker yet (the notice is shown on the first run)."""
    try:
        return not PRIVACY_MARKER.exists()
    except OSError:
        return True


def _write_privacy_marker() -> None:
    """Record that the privacy notice was shown. Written only after the runner returned, i.e. after
    config, models, and keys resolved and the scenarios ran: a run that stops on a config error leaves
    no marker, so the notice shows again on the next run."""
    with contextlib.suppress(OSError):   # if the marker cannot be written, the notice shows again next time
        _ensure_impulse_dir()
        PRIVACY_MARKER.write_text(PRIVACY_NOTICE + "\n", encoding="utf-8")


def _write_run_outputs(run: RunResult, reporter: str, embed_audio: bool) -> tuple[Path, list[Path]]:
    """``run.json`` and ``scenarios/<name>.json`` (always: ``impulse report`` and ``impulse compare`` read
    them), plus ``report.html`` for ``--reporter html|all``, in ``.impulse/runs/<run_id>/``
    (``reporter.write_run_files``; scenario file names from ``reporter.scenario_file_stem``)."""
    from impulse_evals.reporter import write_run_files

    run_dir = RUNS_DIR / run.run_id
    written: list[Path] = []

    def write_all() -> None:
        _ensure_impulse_dir()
        written.extend(
            write_run_files(run, run_dir, with_html=reporter in ("html", "all"), embed_audio=embed_audio)
        )

    _write_or_exit(write_all, run_dir)
    return run_dir, written


def _scenario_line(result: Any, width: int) -> str:
    details: list[str] = []
    if result.error is not None:
        details.append(f"error: {result.error}")
    for score in result.assertion_results:
        if score.result == "fail":
            details.append(f"{score.rule_id} failed ({score.severity})")
    for metric in result.metric_results:
        if metric.label == "poor" and not metric.skipped and not metric.informational:
            details.append(f"{metric.metric_id} poor")
    status = _status("PASS" if result.passed else "FAIL")
    line = f"  {status}  {result.scenario_name.ljust(width)}"
    return (line + "  " + "; ".join(details)).rstrip() if details else line.rstrip()


def _print_run_summary(run: RunResult, written: list[Path], *, dry_run: bool, already_printed: Iterable[str]) -> None:
    mode = "dry run" if dry_run else ("voice mode" if run.voice_mode else "text mode")
    _echo(
        f"Run {run.run_id} ({mode}): {run.total_scenarios} scenarios, {run.passed} passed, "
        f"{run.failed} failed (pass rate {run.overall_pass_rate:.1%})"
    )
    width = max((len(result.scenario_name) for result in run.scenario_results), default=0)
    for result in run.scenario_results:
        _echo(_scenario_line(result, width))
    if run.critical_failures:
        _echo(f"Critical assertion failures: {len(run.critical_failures)}")
    skip = set(already_printed)
    warnings = [warning for warning in run.warnings if warning not in skip]
    if warnings:
        _echo("Warnings:")
        for warning in warnings:
            _echo(f"  - {warning}")
    for path in written:
        _echo(f"Wrote {path}")


@_command("run")
def run_command(
    config: Annotated[Path, typer.Argument(help="Path to the project config.yaml.", show_default=False)],
    reporter: Annotated[
        Optional[Reporter],
        typer.Option(
            "--reporter",
            # "\[" keeps rich help markup from reading [defaults] as a style tag.
            help="Outputs: json, html, or all (default: \\[defaults] reporter in .impulse/config.toml, else all).",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate the config and run a scripted conversation with zero LLM calls.")
    ] = False,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, help="Scenarios run at the same time.")] = 1,
    fail_on: Annotated[
        Optional[FailOn],
        typer.Option(
            "--fail-on", help="Exit 1 on: critical, high, or any (default: \\[defaults] fail_on, else high)."
        ),
    ] = None,
    embed_audio: Annotated[
        bool,
        typer.Option("--embed-audio/--no-embed-audio", help="Embed WAV audio in report.html, or link the audio/ files."),
    ] = True,
) -> None:
    """Run the scenarios of CONFIG and write .impulse/runs/<run_id>/."""
    # config.toml first: its [keys] must be known before any config.yaml error is printed.
    toml = _load_toml_or_exit()
    cfg = _load_config_or_exit(config)
    reporter_value = reporter.value if reporter is not None else (toml.defaults.reporter if toml else "all")
    fail_on_value = fail_on.value if fail_on is not None else (toml.defaults.fail_on if toml else "high")

    from impulse_evals import runner  # lazy: loads the judge, simulator, and LiteLLM

    _SECRETS.redactors.append(runner.config_redactor(cfg, toml))
    # A real run: the keys it resolves for the roles (offline, never raises). A dry run never imports
    # LiteLLM, so it resolves none; the agent module still loads there, so the CLI knows the values of
    # the variables Impulse's own key tables and name endings point to (SPEC.md sections 8.1 and 17).
    for key in static_key_values() if dry_run else runner.resolved_role_keys(cfg, toml):
        _SECRETS.learn(key)
    show_notice = not dry_run and _privacy_notice_due()
    if show_notice:   # before the runner sends anything
        _echo(PRIVACY_NOTICE)
    try:
        result = asyncio.run(runner.run(cfg, dry_run=dry_run, concurrency=concurrency))
    except ConfigError as exc:
        _fail(_config_error_text(exc))
    if show_notice:
        _write_privacy_marker()

    _, written = _write_run_outputs(result, reporter_value, embed_audio)
    # The runner already printed the Pipecat concurrency message (and the inject_events notes).
    _print_run_summary(result, written, dry_run=dry_run, already_printed=[runner.PIPECAT_CONCURRENCY_MESSAGE])
    if dry_run:
        _echo("Dry run: --fail-on is not applied.")
        return
    reasons = fail_on_reasons(result, fail_on_value)
    if reasons:
        _echo(f"--fail-on {fail_on_value}: exit 1")
        for reason in reasons:
            _echo(f"  - {reason}")
        raise typer.Exit(1)


# ------------------------------------------------------------------------------------------------
# Stored runs: impulse report, impulse compare
# ------------------------------------------------------------------------------------------------


def _resolve_run(argument: str) -> Path:
    """An existing path to a ``run.json`` file or a run folder is used directly; anything else is a
    run ID resolved to ``./.impulse/runs/<id>/run.json``. A run that cannot be found exits 2."""
    candidate = Path(argument)
    try:
        exists = candidate.exists()
    except (OSError, ValueError):
        exists = False
    if exists:
        path = candidate / "run.json" if candidate.is_dir() else candidate
    else:
        path = RUNS_DIR / argument / "run.json"
    try:
        found = path.is_file()
    except (OSError, ValueError):
        found = False
    if not found:
        _fail(f"run not found: {argument} (no run.json at {path})")
    return path


def _load_run(path: Path) -> RunResult:
    error = f"cannot load {path}"
    try:
        return RunResult.model_validate_json(path.read_bytes().decode("utf-8-sig"))
    except OSError as exc:
        error = f"cannot read {path}: {exc.strerror or type(exc).__name__}"
    except UnicodeDecodeError:
        error = f"{path} is not valid UTF-8"
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_context=False, include_input=False)[0]
        location = ".".join(str(part) for part in first["loc"]) or "(root)"
        error = f"{path} is not a valid run.json: {location}: {first['msg']}"
    _fail(error)


@_command("report")
def report_command(
    run_id: Annotated[str, typer.Argument(help="Run ID (or a run folder / run.json path).", show_default=False)],
    embed_audio: Annotated[
        bool,
        typer.Option("--embed-audio/--no-embed-audio", help="Embed WAV audio in report.html, or link the audio/ files."),
    ] = True,
) -> None:
    """Rewrite report.html of a stored run from its run.json (run.json is not changed)."""
    from impulse_evals.reporter import write_html

    path = _resolve_run(run_id)
    run = _load_run(path)
    run_dir = path.parent
    target = run_dir / "report.html"
    _write_or_exit(lambda: write_html(run, target, embed_audio=embed_audio, run_dir=run_dir), target)
    _echo(f"Wrote {target}")


def _print_compare(compare: Any, written: list[Path], notes: Iterable[str] = ()) -> None:
    critical = set(compare.critical_regressions)
    _echo(
        f"Compare {compare.run_id_a} ({compare.timestamp_a}) -> {compare.run_id_b} ({compare.timestamp_b}): "
        f"pass rate {compare.overall_pass_rate_delta * 100:+.1f} points"
    )

    def diff_lines(title: str, diffs: list) -> None:
        _echo(f"{title}: {len(diffs)}")
        width = max((len(diff.scenario_name) for diff in diffs), default=0)
        for diff in diffs:
            details = list(diff.assertion_changes) + list(diff.notes)
            flag = "CRITICAL  " if diff.scenario_name in critical else ""
            line = f"  {flag}{diff.scenario_name.ljust(width)}"
            _echo((line + "  " + "; ".join(details)).rstrip() if details else line.rstrip())

    diff_lines("Regressions (pass -> fail)", compare.regressions)
    diff_lines("Improvements (fail -> pass)", compare.improvements)
    diff_lines("Newly failing checks (fail -> fail)", compare.check_regressions)
    _echo(f"Unchanged: {compare.unchanged_pass} pass, {compare.unchanged_fail} fail")
    if compare.added_scenarios:
        _echo(f"Added (in B only): {', '.join(compare.added_scenarios)}")
    if compare.removed_scenarios:
        _echo(f"Removed (in A only): {', '.join(compare.removed_scenarios)}")
    if compare.critical_regressions:
        _echo(f"Critical regressions: {', '.join(compare.critical_regressions)}")
    else:
        _echo("No critical regressions.")
    if compare.high_regressions:
        _echo(f"High regressions: {', '.join(compare.high_regressions)}")
    for note in notes:   # compare.dry_run_notes: one per dry run, run A first
        _echo(f"Note: {note}")
    for path in written:
        _echo(f"Wrote {path}")


@_command("compare")
def compare_command(
    run_a: Annotated[str, typer.Argument(help="Baseline run: run ID, run folder, or run.json path.", show_default=False)],
    run_b: Annotated[str, typer.Argument(help="New run: run ID, run folder, or run.json path.", show_default=False)],
    reporter: Annotated[
        Reporter, typer.Option("--reporter", help="Outputs: json, html, or all.")
    ] = Reporter.all,
    fail_on: Annotated[
        CompareFailOn,
        typer.Option(
            "--fail-on",
            help="Exit 1 on: critical (critical regressions) or high (also high regressions). Default: critical.",
        ),
    ] = CompareFailOn.critical,
) -> None:
    """Diff two runs; exit 1 when a critical assertion newly failed, or a scenario newly errored, in RUN_B
    (with --fail-on high: also when a high assertion newly failed or UNIV-025 became poor)."""
    from impulse_evals.compare import compare_runs, dry_run_notes
    from impulse_evals.reporter import (
        render_compare_html,
        sanitize_name,
        write_bytes_atomic,
        write_json,
    )

    reporter_value = reporter.value   # SPEC.md section 17: default all (config.toml is not consulted)
    fail_on_value = fail_on.value     # default critical; config.toml [defaults] fail_on is not consulted
    run_a_result = _load_run(_resolve_run(run_a))
    run_b_result = _load_run(_resolve_run(run_b))
    compare = compare_runs(run_a_result, run_b_result)
    notes = dry_run_notes(run_a_result, run_b_result)

    stem = f"compare_{sanitize_name(run_a_result.run_id)}_vs_{sanitize_name(run_b_result.run_id)}"
    json_path, html_path = RUNS_DIR / f"{stem}.json", RUNS_DIR / f"{stem}.html"
    written: list[Path] = []

    def write_all() -> None:
        _ensure_impulse_dir()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        if reporter_value in ("json", "all"):
            written.append(write_json(compare, json_path))
        if reporter_value in ("html", "all"):
            written.append(write_bytes_atomic(html_path, render_compare_html(compare, notes=notes).encode("utf-8")))

    _write_or_exit(write_all, RUNS_DIR)
    _print_compare(compare, written, notes)
    if compare.critical_regressions or (fail_on_value == "high" and compare.high_regressions):
        raise typer.Exit(1)


# ------------------------------------------------------------------------------------------------
# Model checks shared by init and doctor (SPEC.md sections 8.1 and 17)
# ------------------------------------------------------------------------------------------------


@dataclass
class _Row:
    status: str   # PASS | FAIL | INFO | SKIP
    check: str
    detail: str


def _print_rows(rows: list[_Row]) -> None:
    width = max((len(row.check) for row in rows), default=0)
    for row in rows:
        _echo(f"{_status(row.status)}  {row.check.ljust(width)}  {row.detail}".rstrip())


def _offline_model_problem(model: str, options: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """None when LiteLLM recognises the model string offline (``litellm.get_llm_provider``), else why not.
    The role's ``custom_llm_provider`` is read as the key lookup reads it
    (``config.role_custom_llm_provider``: the top-level option, else the one inside ``extra_params``),
    so ``impulse doctor`` and ``impulse run`` agree. A provider LiteLLM authenticates while resolving
    it (``config.authenticating_provider``: github_copilot, chatgpt) is recognised by its name and never
    resolved: that would start a login."""
    options = options or {}
    custom_llm_provider = role_custom_llm_provider(options)
    if authenticating_provider(model, custom_llm_provider) is not None:
        return None

    import litellm  # lazy: heavy import

    litellm.suppress_debug_info = True   # no "Give Feedback / Get Help" banner (AUDIT.md finding 43)
    try:
        # LiteLLM prints a provider-list banner for unknown model strings; this is only a probe.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            litellm.get_llm_provider(
                model,
                custom_llm_provider=custom_llm_provider,
                api_base=options.get("api_base") or None,
            )
    except Exception as exc:  # noqa: BLE001 - LiteLLM raises several types for unknown models
        return f"LiteLLM does not recognise this model string ({_short_text(str(exc), 160)})"
    return None


def _ping_kwargs(options: Mapping[str, Any], api_key: Optional[str]) -> dict[str, Any]:
    from impulse_evals.judge.litellm_judge import completion_kwargs

    merged = dict(options)
    if api_key:
        merged["api_key"] = api_key
    kwargs = completion_kwargs(merged, fixed=("model", "messages", "stream", "max_tokens"))
    kwargs.setdefault("timeout", PING_TIMEOUT_S)
    return kwargs


class _Pinger:
    """5-token pings through ``litellm.completion``; the same model and options are pinged once.
    A failure message is redacted (known keys plus the key sent with the ping), then shortened. The
    call runs with stdout and stderr captured and discarded, so nothing LiteLLM (or a provider SDK)
    prints, such as its "Give Feedback / Get Help" banner, reaches Impulse's output (AUDIT.md
    finding 43); ``litellm.suppress_debug_info`` stops the banner too. Exception: the ping of a role
    whose provider LiteLLM authenticates (``config.authenticating_provider``: github_copilot, chatgpt,
    by the model prefix or ``custom_llm_provider``) is a real call that runs LiteLLM's own sign-in, so
    its output is not redirected: the device-code prompt must stay visible."""

    def __init__(self) -> None:
        self._results: dict[tuple[str, str], tuple[bool, str]] = {}

    def __call__(self, model: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
        cache_key = (model, json.dumps(kwargs, sort_keys=True, default=str))
        if cache_key not in self._results:
            self._results[cache_key] = self._ping(model, kwargs)
        return self._results[cache_key]

    def _ping(self, model: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
        import litellm  # lazy: heavy import

        litellm.suppress_debug_info = True   # no "Give Feedback / Get Help" banner (AUDIT.md finding 43)
        api_key = kwargs.get("api_key")
        extra_values = [api_key] if isinstance(api_key, str) else []
        signs_in = authenticating_provider(model, role_custom_llm_provider(kwargs)) is not None
        try:
            with contextlib.ExitStack() as output:
                if not signs_in:   # the sign-in prompt of github_copilot / chatgpt must stay visible
                    output.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    output.enter_context(contextlib.redirect_stderr(io.StringIO()))
                litellm.completion(model=model, messages=PING_MESSAGES, max_tokens=PING_MAX_TOKENS, **kwargs)
        except Exception as exc:  # noqa: BLE001 - any failure means the model is not reachable
            return False, "not reachable: " + _exception_text(exc, extra_values=extra_values)
        return True, f"reachable ({PING_MAX_TOKENS}-token ping)"


# ------------------------------------------------------------------------------------------------
# impulse doctor
# ------------------------------------------------------------------------------------------------


def _python_row() -> _Row:
    version = platform.python_version()
    if sys.version_info >= (3, 11):  # noqa: UP036 - reported to the user, not a code branch
        return _Row("PASS", "python", f"Python {version}")
    return _Row("FAIL", "python", f"Python {version}; Impulse needs Python 3.11 or newer")


def _toml_row() -> tuple[Optional[TomlConfig], _Row]:
    if not TOML_PATH.is_file():
        return None, _Row("INFO", "config.toml", f"{TOML_PATH} not found (optional; impulse init writes it)")
    try:
        toml = load_toml(TOML_PATH)
    except ConfigError as exc:
        _SECRETS.use_toml(None)
        return None, _Row("FAIL", "config.toml", _config_error_text(exc))
    _SECRETS.use_toml(toml)
    return toml, _Row("PASS", "config.toml", f"{TOML_PATH} loaded")


def _model_rows(cfg: ProjectConfig, toml: Optional[TomlConfig], pinger: _Pinger) -> list[_Row]:
    rows: list[_Row] = []
    for role in ROLES:
        check = f"{role} model"
        try:
            model = resolve_model(cfg, role, toml)
        except ConfigError as exc:
            rows.append(_Row("FAIL", check, _config_error_text(exc)))
            continue
        options = resolve_model_options(cfg, role, toml)
        problem = _offline_model_problem(model, options)
        if problem is not None:
            rows.append(_Row("FAIL", check, f"{model}: {problem}"))
            continue
        try:
            key = api_key_for_role(cfg, role, toml)
        except ConfigError as exc:   # a hosted model without its key: no ping
            rows.append(_Row("FAIL", check, f"{model}: {_config_error_text(exc)}"))
            continue
        _SECRETS.learn(key)   # e.g. SNOWFLAKE_JWT: redacted from every later row (agent, voice)
        ok, detail = pinger(model, _ping_kwargs(options, key))
        if key is None and model_needs_key(model, options) and not options.get("api_key_env"):
            # A provider for which LiteLLM names no *_API_KEY variable (bedrock/, vertex_ai/, ...).
            detail += f"; {NO_IMPULSE_KEY_NOTE}"
        rows.append(_Row("PASS" if ok else "FAIL", check, f"{model}: {detail}"))
    return rows


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _probe_scenario(cfg: ProjectConfig) -> Scenario:
    if cfg.scenarios:
        return cfg.scenarios[0]
    return Scenario(name="impulse_doctor_probe", language="en", persona="impulse doctor connection probe")


async def _generic_rows(cfg: ProjectConfig) -> list[_Row]:
    from impulse_evals.adapters.generic_adapter import GenericAdapter

    adapter = GenericAdapter(cfg)
    try:
        await adapter.start(_probe_scenario(cfg))
        prompt = adapter.get_system_prompt()
    except ConfigError as exc:
        return [_Row("FAIL", "agent entrypoint", _config_error_text(exc))]
    except Exception as exc:  # noqa: BLE001 - e.g. the module's get_system_prompt() raised
        return [_Row("FAIL", "agent entrypoint", _exception_text(exc))]
    finally:
        with contextlib.suppress(Exception):
            await adapter.close()
    rows = [_Row("PASS", "agent entrypoint", f"imported {cfg.agent.entrypoint}")]
    if cfg.agent.system_prompt is not None:
        rows.append(_Row("INFO", "system prompt", f"from {cfg.agent.system_prompt}"))
    elif prompt:
        rows.append(_Row("INFO", "system prompt", "from the agent module (SYSTEM_PROMPT or get_system_prompt())"))
    else:
        rows.append(
            _Row("INFO", "system prompt", "none: the judge falls back to general best practice; set agent.system_prompt")
        )
    return rows


async def _pipecat_rows(cfg: ProjectConfig) -> list[_Row]:
    """The Pipecat handshake check (SPEC.md section 17): eval mode first needs ``pipecat.evals``
    importable; then connect in the configured mode (through the adapter, exactly as a run does) and wait
    for ``bot-ready``; report (info only, never a failure) whether the bot answers ``get_system_prompt``,
    plus the adapter's warnings (for example the eval-mode conversation reset warning)."""
    pipecat = cfg.agent.pipecat
    rows: list[_Row] = []
    mode = pipecat.mode if pipecat is not None else "eval"
    try:
        pipecat_adapter = importlib.import_module("impulse_evals.adapters.pipecat_adapter")
    except ImportError as exc:
        return rows + [_Row("FAIL", "pipecat handshake", _exception_text(exc))]
    if mode == "eval":
        # No pipecat package at all: the missing-extra text `impulse run` gives (SPEC.md section 9), not
        # the upgrade text, which is for a pipecat-ai release without pipecat.evals.
        if not _module_available("pipecat"):
            return [_Row("FAIL", "pipecat.evals", pipecat_adapter.MISSING_EXTRA)]
        # find_spec("pipecat.evals") imports the pipecat package, which logs a banner: quiet it first.
        pipecat_adapter.quiet_pipecat_logging()
        if not _module_available("pipecat.evals"):
            return [_Row("FAIL", "pipecat.evals", PIPECAT_EVALS_MISSING)]
        rows.append(_Row("PASS", "pipecat.evals", "importable"))
    adapter = None
    try:
        # Looked up as a module attribute when the check runs (tests replace it to inject a fake channel).
        adapter = pipecat_adapter.PipecatAdapter(cfg, redact=_redact)   # redacted BEFORE its 200-char cut
        await adapter.start(_probe_scenario(cfg))
        if cfg.agent.system_prompt is not None:
            # start() sends no request when config.yaml sets the prompt; ask once anyway, for the report.
            answered = await adapter.probe_system_prompt()
        else:
            answered = adapter.get_system_prompt()
        warnings = adapter.get_warnings()
    except ConfigError as exc:
        return rows + [_Row("FAIL", "pipecat handshake", _config_error_text(exc))]
    except Exception as exc:  # noqa: BLE001 - handshake timeout, connection refused, one-client close, ...
        return rows + [_Row("FAIL", "pipecat handshake", _short_text(str(exc)) or type(exc).__name__)]
    finally:
        if adapter is not None:
            with contextlib.suppress(Exception):
                await adapter.close()
    shown_url = strip_url_credentials(pipecat.url or "") if pipecat else ""   # never the URL's credentials
    rows.append(_Row("PASS", "pipecat handshake", f"{mode} mode: bot-ready received from {shown_url}"))
    if cfg.agent.system_prompt is not None:
        detail = (
            "the bot answers get_system_prompt" if answered else "the bot does not answer get_system_prompt"
        ) + "; runs use agent.system_prompt from config.yaml"
    elif answered:
        detail = "the bot answers get_system_prompt; runs use its answer"
    else:
        detail = (
            "the bot did not answer get_system_prompt; the judge uses general best practice "
            "(add the handler, or set agent.system_prompt)"
        )
    rows.append(_Row("INFO", "pipecat system prompt", detail))
    for warning in warnings:
        rows.append(_Row("INFO", "pipecat", _short_text(warning)))
    return rows


def _livekit_rows(cfg: ProjectConfig, toml: Optional[TomlConfig]) -> list[_Row]:
    livekit = cfg.agent.livekit
    rows: list[_Row] = []
    if not _module_available("livekit"):
        rows.append(_Row("FAIL", "livekit extra", LIVEKIT_EXTRA_MISSING))
    else:
        rows.append(_Row("PASS", "livekit extra", "installed"))
    if livekit is not None:
        for label, name in (("LiveKit API key", livekit.api_key_env), ("LiveKit API secret", livekit.api_secret_env)):
            # The adapter's lookup (SPEC.md section 8.1): the environment variable, then config.toml [keys].
            rows.append(_key_row(label, name, toml))
    rows.append(_Row("INFO", "livekit connection", "not checked by impulse doctor"))
    return rows


def _key_row(label: str, name: Optional[str], toml: Optional[TomlConfig]) -> _Row:
    """PASS when the variable ``name`` is set (environment, then config.toml [keys], the adapters'
    lookup; SPEC.md section 8.1), else FAIL with the missing-key text (``(none named)`` for no name).
    Never shows the value. Used by the LiveKit and HTTP rows."""
    if name and os.environ.get(name):
        return _Row("PASS", label, f"{name} is set (environment variable)")
    if name and resolve_key(name, toml):
        return _Row("PASS", label, f"{name} is set ([keys] in .impulse/config.toml)")
    return _Row(
        "FAIL",
        label,
        f"{name or '(none named)'} is not set: set the environment variable or add it to [keys] in .impulse/config.toml",
    )


def _custom_rows(cfg: ProjectConfig) -> list[_Row]:
    """The custom adapter's class is imported and checked, never created (FIXES-ROUND-3-SPEC.md B3)."""
    from impulse_evals.adapters.custom_adapter import (
        entrypoint_parts,
        load_adapter_class,
    )

    rows: list[_Row] = []
    try:
        load_adapter_class(cfg)
        module_name, attr_path = entrypoint_parts(cfg)
    except ConfigError as exc:
        rows.append(_Row("FAIL", "agent adapter", _config_error_text(exc)))
    else:
        rows.append(
            _Row("PASS", "agent adapter", f"custom adapter {module_name}:{attr_path} imported (a subclass of AgentInterface)")
        )
    rows.append(_Row("INFO", "agent connection", "not checked by impulse doctor (custom adapter)"))
    return rows


HTTP_CONNECT_TIMEOUT_S = 5.0


async def _http_rows(cfg: ProjectConfig, toml: Optional[TomlConfig]) -> list[_Row]:
    """A plain TCP connect to the turn URL's host and port (default 80 / 443), never an HTTP request
    (a turn can have side effects), and the bearer key row (FIXES-ROUND-3-SPEC.md B4)."""
    from urllib.parse import urlsplit

    http = cfg.agent.http
    rows: list[_Row] = []
    url = http.url if http is not None and http.url else ""
    shown = strip_url_credentials(url)   # never the URL's credentials
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if not host:
            raise ValueError("no host in the URL")
        # Looked up as module attributes when the check runs (tests replace asyncio.open_connection).
        _reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), HTTP_CONNECT_TIMEOUT_S)
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    except Exception as exc:  # noqa: BLE001 - refused, unknown host, timeout, invalid port, ...
        rows.append(_Row("FAIL", "agent http", f"cannot connect to {shown}: {_exception_text(exc)}"))
    else:
        rows.append(_Row("PASS", "agent http", f"reachable: {shown}"))
    name = (http.api_key_env or "").strip() if http is not None else ""
    if name:
        rows.append(_key_row("HTTP agent key", name, toml))
    rows.append(_Row("INFO", "agent connection", "the turn endpoint is not called by impulse doctor"))
    return rows


DOCTOR_VOICE_TEXT = "This is an Impulse voice check."


def _probe_tone_wav() -> bytes:
    """Half a second of a quiet 440 Hz tone (16 kHz mono WAV) for the STT check."""
    import numpy as np

    from impulse_evals.voice.audio_utils import STT_SAMPLE_RATE, encode_wav

    times = np.arange(STT_SAMPLE_RATE // 2) / STT_SAMPLE_RATE
    return encode_wav(0.3 * np.sin(2 * np.pi * 440.0 * times), STT_SAMPLE_RATE)


async def _voice_rows(cfg: ProjectConfig, toml: Optional[TomlConfig]) -> list[_Row]:
    """Voice provider checks: HTTP providers synthesize a short phrase and transcribe it (one round
    trip each); ``local_hf`` checks the ``local`` extra and the model settings without loading a
    model (that downloads weights and is left to the first run); ``local_light`` likewise checks the
    ``light`` extra and whether the Kokoro files are present, never loading or downloading them."""
    voice = cfg.voice_config
    if voice is None:
        return [_Row("INFO", "voice providers", "text mode (no voice_config)")]
    from impulse_evals.voice.audio_utils import is_wav
    from impulse_evals.voice.impulse_hosted import (
        SUPPORTED_LANGUAGES,
        UnsupportedLanguageError,
        supported_language,
    )
    from impulse_evals.voice.stt import get_stt_provider
    from impulse_evals.voice.tts import get_tts_provider

    language = _probe_scenario(cfg).language
    rows: list[_Row] = []
    audio: Optional[bytes] = None
    try:
        tts = get_tts_provider(voice, toml=toml)
        if voice.tts.provider == "local_hf":
            model = tts.model_for(language)
            rows.append(_Row("PASS", "voice TTS", f"local_hf: 'local' extra installed; model {model} loads on the first run"))
        elif voice.tts.provider == "local_light":
            # Never loads the model and never downloads it (that is left to the first run).
            status = tts.model_status()
            rows.append(_Row("PASS", "voice TTS", f"local_light: 'light' extra installed; Kokoro model {status}"))
        else:
            audio = await tts.synthesize(DOCTOR_VOICE_TEXT, language, voice.tts.voice)
            rows.append(_Row("PASS", "voice TTS", f"{voice.tts.provider}: {len(audio)} bytes of audio ({language})"))
    except ConfigError as exc:
        rows.append(_Row("FAIL", "voice TTS", _config_error_text(exc)))
    except Exception as exc:  # noqa: BLE001 - request failed, unsupported language, ...
        rows.append(_Row("FAIL", "voice TTS", _exception_text(exc)))
    try:
        stt = get_stt_provider(voice, toml=toml)
        if voice.stt.provider == "local_hf":
            model = stt.model_id()
            rows.append(_Row("PASS", "voice STT", f"local_hf: 'local' extra installed; model {model} loads on the first run"))
        elif voice.stt.provider == "local_light":
            model = stt.model_id()
            rows.append(
                _Row("PASS", "voice STT", f"local_light: 'light' extra installed; model {model} loads on the first run")
            )
        else:
            probe = audio if audio and is_wav(audio) else _probe_tone_wav()
            result = await stt.transcribe(probe, language)
            timings = f"{len(result.words)} word timings" if result.words else "no word timings (VOICE-015 will skip)"
            rows.append(_Row("PASS", "voice STT", f"{voice.stt.provider}: transcribed the probe audio, {timings}"))
    except ConfigError as exc:
        rows.append(_Row("FAIL", "voice STT", _config_error_text(exc)))
    except Exception as exc:  # noqa: BLE001
        rows.append(_Row("FAIL", "voice STT", _exception_text(exc)))
    if "impulse_hosted" in (voice.tts.provider, voice.stt.provider):
        unsupported = []
        for code in dict.fromkeys(scenario.language for scenario in cfg.scenarios):
            try:
                supported_language(code)
            except UnsupportedLanguageError:
                unsupported.append(code)
        if unsupported:
            rows.append(
                _Row(
                    "FAIL",
                    "voice languages",
                    f"impulse_hosted does not support {', '.join(unsupported)} "
                    f"(supported: {', '.join(SUPPORTED_LANGUAGES)})",
                )
            )
    if voice.tts.provider == "local_light" and not (voice.tts.voice or "").strip():
        from impulse_evals.voice.local_light import default_voice

        no_voice = [code for code in dict.fromkeys(s.language for s in cfg.scenarios) if default_voice(code) is None]
        if no_voice:
            rows.append(_Row("FAIL", "voice languages", f"local_light TTS has no voice for {', '.join(no_voice)}"))
    return rows


NOISE_MIN_SAMPLE_RATE = 8000
NOISE_MAX_SAMPLE_RATE = 192000
NOISE_LOW_SAMPLE_RATE = 16000
NOISE_SHORT_FILE_S = 1.0


def _noise_rows(cfg: ProjectConfig) -> list[_Row]:
    """Noise simulation (EXPERIMENTAL): one INFO row with the levels, and one row per noise file (rate,
    channels, length; FAIL outside 8000..192000 Hz; INFO when below 16000 Hz or shorter than 1 s).
    Nothing when no noise is configured. ``load_config`` already checked that the files are readable."""
    from impulse_evals.config import effective_noise
    from impulse_evals.models import NoiseSettings
    from impulse_evals.voice.noise import default_label, level_name

    configured = [scenario.noise for scenario in cfg.scenarios if isinstance(scenario.noise, NoiseSettings)]
    if cfg.voice_config is not None and cfg.voice_config.noise is not None:
        configured.append(cfg.voice_config.noise)
    if not configured:
        return []
    if cfg.voice_config is None:
        return [_Row("INFO", "noise simulation", "experimental; ignored in text mode (no voice_config)")]

    used = [noise for noise in (effective_noise(cfg, scenario) for scenario in cfg.scenarios) if noise is not None]
    levels = list(dict.fromkeys(level_name(default_label(noise), float(noise.snr_db or 0.0)) for noise in used))
    detail = f"experimental; levels: {'; '.join(levels)}" if levels else "experimental; no scenario uses noise"
    rows = [_Row("INFO", "noise simulation", detail)]
    for path in dict.fromkeys(noise.file for noise in used if noise.profile == "file" and noise.file):
        rows.extend(_noise_file_rows(Path(path)))
    return rows


def _noise_file_rows(path: Path) -> list[_Row]:
    from impulse_evals.voice import audio_utils

    check = f"noise file {path.name}"
    try:
        wav = audio_utils.ensure_wav(path.read_bytes())
        channels, _, rate = audio_utils.wav_params(wav)
        seconds = audio_utils.duration_ms(wav) / 1000.0
    except Exception as exc:  # noqa: BLE001 - removed or changed since the config was loaded
        why = (getattr(exc, "strerror", None) or "").strip() or type(exc).__name__
        return [_Row("FAIL", check, f"cannot read the WAV file ({why})")]
    if not NOISE_MIN_SAMPLE_RATE <= rate <= NOISE_MAX_SAMPLE_RATE:
        return [
            _Row(
                "FAIL",
                check,
                f"sample rate {rate} Hz is outside {NOISE_MIN_SAMPLE_RATE}..{NOISE_MAX_SAMPLE_RATE} Hz; "
                "re-export the recording at 16000 Hz or more",
            )
        ]
    plural = "" if channels == 1 else "s"
    rows = [_Row("PASS", check, f"readable WAV, {rate} Hz, {channels} channel{plural}, {seconds:.1f} s")]
    if rate < NOISE_LOW_SAMPLE_RATE:
        rows.append(
            _Row("INFO", check, f"{rate} Hz is below {NOISE_LOW_SAMPLE_RATE} Hz; it will be upsampled and sound muffled")
        )
    if seconds < NOISE_SHORT_FILE_S:
        rows.append(_Row("INFO", check, f"{seconds:.1f} s is shorter than 1 s; it will loop audibly"))
    return rows


async def _agent_rows(cfg: ProjectConfig, toml: Optional[TomlConfig]) -> list[_Row]:
    if cfg.agent.adapter == "pipecat":
        return await _pipecat_rows(cfg)
    if cfg.agent.adapter == "livekit":
        return _livekit_rows(cfg, toml)
    if cfg.agent.adapter == "custom":
        return _custom_rows(cfg)
    if cfg.agent.adapter == "http":
        return await _http_rows(cfg, toml)
    return await _generic_rows(cfg)


@_command("doctor")
def doctor_command(
    config: Annotated[
        Optional[Path],
        typer.Argument(help="Project config.yaml (default: ./config.yaml when it exists).", show_default=False),
    ] = None,
) -> None:
    """Check Python, the config, model reachability, the agent connection, and voice providers."""
    rows: list[_Row] = [_python_row()]
    toml, toml_row = _toml_row()
    rows.append(toml_row)

    config_path = config if config is not None else (STARTER_CONFIG if STARTER_CONFIG.is_file() else None)
    cfg: Optional[ProjectConfig] = None
    config_ok = True
    if config_path is None:
        rows.append(_Row("INFO", "config", "no config given and no ./config.yaml; checking models from the environment and .impulse/config.toml"))
    else:
        try:
            cfg = load_config(config_path)
        except ConfigError as exc:
            config_ok = False
            rows.append(_Row("FAIL", "config", _config_error_text(exc)))
        else:
            # lazy: loads the judge and LiteLLM
            from impulse_evals.runner import config_redactor

            _SECRETS.redactors.append(config_redactor(cfg, toml))
            rows.append(_Row("PASS", "config", f"{config_path} is valid"))
            rows.extend(_Row("INFO", "config warning", warning) for warning in cfg.warnings)

    pinger = _Pinger()
    if not config_ok:
        rows.append(_Row("INFO", "models", "not checked: fix the config first"))
    else:
        bare = cfg if cfg is not None else ProjectConfig(simulator=SimulatorConfig(), config_dir=Path.cwd())
        rows.extend(_model_rows(bare, toml, pinger))
    if cfg is not None:
        rows.extend(asyncio.run(_agent_rows(cfg, toml)))
        rows.extend(asyncio.run(_voice_rows(cfg, toml)))
        rows.extend(_noise_rows(cfg))

    _echo("impulse doctor")
    _print_rows(rows)
    failed = sum(1 for row in rows if row.status == "FAIL")
    if failed:
        _echo(f"{failed} check(s) failed. Fix every FAIL row, then run impulse doctor again.")
        raise typer.Exit(2)
    _echo("All checks passed.")


# ------------------------------------------------------------------------------------------------
# impulse init (SPEC.md section 8.1)
# ------------------------------------------------------------------------------------------------


def _valid_language(code: Optional[str]) -> bool:
    return isinstance(code, str) and re.fullmatch(LANGUAGE_PATTERN, code) is not None


@dataclass
class _InitAnswers:
    language: Optional[str] = None
    models: dict[str, str] = field(default_factory=lambda: dict(MODEL_PLACEHOLDERS))
    api_base: dict[str, str] = field(default_factory=lambda: {role: "" for role in ROLES})
    keys: dict[str, str] = field(default_factory=lambda: {name: "" for name in STANDARD_KEY_NAMES})
    tts_model: str = TTS_PLACEHOLDER
    stt_model: str = STT_PLACEHOLDER


def _prompt(text: str, *, hidden: bool = False) -> str:
    """One free-text answer; a blank answer is allowed and returned as ``""``."""
    return str(typer.prompt(text, default="", show_default=False, hide_input=hidden)).strip()


def _ask_language() -> str:
    """The default language: asked again while blank or invalid; no language is ever assumed."""
    while True:
        value = _prompt("Default language (a language code such as en-IN, hi-IN, hi-Latn-IN)")
        if _valid_language(value):
            return value
        if not value:
            _echo("A language is required; Impulse never assumes one.")
        else:
            _echo(f"{value} is not a valid language code; it must match {LANGUAGE_PATTERN} (for example en-IN, hi-Latn-IN).")


def _ollama_model(answer: str) -> str:
    """An Ollama tag as a LiteLLM model string (``ollama/<tag>``); ``ollama/`` / ``ollama_chat/`` kept."""
    return answer if answer.startswith(OLLAMA_PREFIXES) else f"ollama/{answer}"


def _ask_model(question: str, placeholder: str, *, ollama: bool = False) -> str:
    """A model string, checked offline with ``litellm.get_llm_provider`` (asked again when LiteLLM does
    not recognise it); a blank answer keeps the placeholder."""
    while True:
        answer = _prompt(question)
        if is_placeholder(answer):
            return placeholder
        model = _ollama_model(answer) if ollama else answer
        problem = _offline_model_problem(model)
        if problem is None:
            return model
        _echo(f"{model}: {problem}. Enter a LiteLLM model string (provider/model), or leave it blank.")


def _ask_key(name: str, note: str = "") -> str:
    value = _prompt(f"{name}{note} (input hidden; leave blank to skip)", hidden=True)
    if value:
        _SECRETS.values.append(value)   # redacted from anything printed later in this command
    return value


def _ask_free(answers: _InitAnswers) -> None:
    for role in ROLES:
        answers.models[role] = _ask_model(
            f"Ollama model for the {ROLE_LABELS[role]} (Ollama tag; blank keeps {MODEL_PLACEHOLDERS[role]})",
            MODEL_PLACEHOLDERS[role],
            ollama=True,
        )
    answers.tts_model = _prompt(f"Hugging Face text-to-speech model id (blank keeps {TTS_PLACEHOLDER})") or TTS_PLACEHOLDER
    answers.stt_model = (
        _prompt(f"Hugging Face speech-to-text (faster-whisper) model id (blank keeps {STT_PLACEHOLDER})") or STT_PLACEHOLDER
    )


def _ask_hosted(answers: _InitAnswers) -> None:
    for name in LLM_KEY_NAMES:
        answers.keys[name] = _ask_key(name)
    answers.keys[VOICE_KEY_NAME] = _ask_key(VOICE_KEY_NAME, ", optional, for the impulse_hosted voice provider")
    for role in ROLES:
        answers.models[role] = _ask_model(
            f"Model for the {ROLE_LABELS[role]} (LiteLLM model string; blank keeps {MODEL_PLACEHOLDERS[role]})",
            MODEL_PLACEHOLDERS[role],
        )


def _ask_custom(answers: _InitAnswers) -> None:
    for role in ROLES:
        answers.models[role] = _ask_model(
            f"Model for the {ROLE_LABELS[role]} (LiteLLM model string; blank keeps {MODEL_PLACEHOLDERS[role]})",
            MODEL_PLACEHOLDERS[role],
        )
        answers.api_base[role] = _prompt(f"api_base for the {ROLE_LABELS[role]} (local or self-hosted server; blank for none)")
    needed: list[str] = []
    for role in ROLES:
        model = answers.models[role]
        options = {"api_base": answers.api_base[role]} if answers.api_base[role] else {}
        if is_placeholder(model) or not model_needs_key(model, options):
            continue
        try:
            name = key_name_for_model(model, f"models.{role}")
        except ConfigError:
            continue
        if name is not None and name not in needed:   # None: LiteLLM uses the provider's own credentials
            needed.append(name)
    for name in needed:
        answers.keys[name] = _ask_key(name)


def _toml_string(value: str) -> str:
    """A TOML basic string (JSON escapes are valid TOML escapes; DEL is escaped too)."""
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _toml_key(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else _toml_string(name)


PRIVATE_FILE_MODE = 0o600


def _write_private_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` for its owner only: ``.impulse/config.toml`` holds ``[keys]``. The file
    is opened with ``os.open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)`` (a new file never exists with
    a wider mode). That mode applies only when the file is created, so on POSIX the descriptor is
    also narrowed with ``os.fchmod(fd, 0o600)`` before any byte is written (an overwritten file never
    holds the new keys with its old, wider mode), and ``os.chmod(path, 0o600)`` follows the write.
    The bytes go through the ``os.open`` descriptor. On Windows the mode only sets the read-only flag
    (0600 leaves the file writable; the folder's default permissions apply)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        if os.name != "nt":
            os.fchmod(fd, PRIVATE_FILE_MODE)
        handle = os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        raise
    with handle:
        handle.write(data)
    if os.name != "nt":
        os.chmod(path, PRIVATE_FILE_MODE)


def _render_toml(answers: _InitAnswers) -> str:
    """``.impulse/config.toml`` from a string template (SPEC.md section 8.1; no TOML writer)."""
    lines = [
        "# Impulse Evals settings for this folder, written by impulse init. Keep it out of version control",
        "# (.impulse/.gitignore ignores the whole folder). Environment variables win over [keys]; local models",
        "# (ollama/..., or any role with api_base) need no key.",
        "",
        "[keys]",
    ]
    lines += [f"{_toml_key(name)} = {_toml_string(value)}" for name, value in answers.keys.items()]
    lines += ["", "[models]"]
    lines += [f"{role} = {_toml_string(answers.models[role])}" for role in ROLES]
    for role in ROLES:
        lines += [
            "",
            f"[model_options.{role}]",
            f"api_base = {_toml_string(answers.api_base[role])}",
            'api_key_env = ""',
            "extra_params = {}",
        ]
    lines += [
        "",
        "[defaults]",
        'reporter = "all"',
        f"language = {_toml_string(answers.language or '')}",
        'fail_on = "high"              # critical | high | any',
    ]
    return "\n".join(lines) + "\n"


def _starter_config_text(answers: _InitAnswers) -> str:
    """The packaged free-profile starter ``config.yaml`` with the language and Hugging Face ids filled
    in (the template's own SPDX line is dropped: it licenses the template, not the user's config)."""
    text = resources.files("impulse_evals").joinpath("templates").joinpath(STARTER_TEMPLATE).read_text(encoding="utf-8")
    first, _, rest = text.partition("\n")
    if first.lstrip("﻿").startswith("# SPDX-License-Identifier:"):
        text = rest
    replacements = {
        f'model: "{TTS_PLACEHOLDER}"': f"model: {json.dumps(answers.tts_model, ensure_ascii=False)}",
        f'model: "{STT_PLACEHOLDER}"': f"model: {json.dumps(answers.stt_model, ensure_ascii=False)}",
        'language: "YOUR_LANGUAGE"': f"language: {json.dumps(answers.language or 'YOUR_LANGUAGE', ensure_ascii=False)}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new, 1)
    return text


def _voice_config_block(config_text: str) -> str:
    """The ``voice_config:`` block of a config text (up to the next top-level key)."""
    lines = config_text.splitlines()
    block: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("voice_config:"):
            inside = True
        elif inside and line and not line[0].isspace() and not line.startswith("#"):
            break
        if inside:
            block.append(line)
    while block and not block[-1].strip():
        block.pop()
    return "\n".join(block)


def _write_starter_config(answers: _InitAnswers) -> None:
    text = _starter_config_text(answers)
    if STARTER_CONFIG.exists():
        _echo(f"{STARTER_CONFIG} already exists and was left unchanged. For local speech models, add this block to it:")
        _echo(_voice_config_block(text))
        return
    _write_or_exit(lambda: STARTER_CONFIG.write_bytes(text.encode("utf-8")), STARTER_CONFIG)
    _echo(f"Wrote {STARTER_CONFIG} (starter config: set agent.entrypoint and agent.system_prompt before the first run)")


def _init_ping_rows(answers: _InitAnswers, toml: TomlConfig) -> list[_Row]:
    """A live 5-token ping per configured model when the model is local or its key is present."""
    pinger = _Pinger()
    rows: list[_Row] = []
    for role in ROLES:
        model = answers.models[role]
        check = f"{role} model"
        if is_placeholder(model):
            continue
        options = {"api_base": answers.api_base[role]} if answers.api_base[role] else {}
        key: Optional[str] = None
        if model_needs_key(model, options):
            try:
                name = key_name_for_model(model, f"models.{role}")
            except ConfigError as exc:
                rows.append(_Row("FAIL", check, _config_error_text(exc)))
                continue
            key = resolve_key(name, toml) if name is not None else None
            if name is not None and key is None:
                rows.append(_Row("SKIP", check, f"{model}: not pinged ({name} is not set)"))
                continue
        _SECRETS.learn(key)   # a key from a variable of any name: redacted from every later line
        ok, detail = pinger(model, _ping_kwargs(options, key))
        rows.append(_Row("PASS" if ok else "FAIL", check, f"{model}: {detail}"))
    return rows


@_command("init")
def init_command(
    profile: Annotated[Profile, typer.Option("--profile", help="free (Ollama + local speech, no keys), hosted, or custom.")] = Profile.custom,
    language: Annotated[
        Optional[str], typer.Option("--language", help="Default language code, e.g. en-IN (required with --non-interactive).")
    ] = None,
    no_ping: Annotated[bool, typer.Option("--no-ping", help="Skip the live 5-token ping of each model.")] = False,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Ask nothing and ping nothing (CI); needs --language.")
    ] = False,
) -> None:
    """Write .impulse/config.toml (and, for --profile free, a starter config.yaml)."""
    if language is not None:
        language = language.strip()
        if not _valid_language(language):
            _fail(
                f"--language {language} is not a valid language code; it must match {LANGUAGE_PATTERN} "
                "(for example en-IN, hi-IN, hi-Latn-IN)"
            )
    if non_interactive and language is None:
        _fail("--language is required with --non-interactive; Impulse never assumes a language (for example --language en-IN).")
    if TOML_PATH.exists():
        if non_interactive:
            # A re-run (e.g. a CI job on a cached workspace) keeps the file, which may hold keys, and
            # writes nothing else: only .gitignore is (re)written so the folder stays ignored. Exit 0.
            _write_or_exit(lambda: GITIGNORE_PATH.write_text("*\n", encoding="utf-8"), GITIGNORE_PATH)
            _echo(f"{TOML_PATH} already exists; left unchanged.")
            return
        if not typer.confirm(f"{TOML_PATH} already exists. Overwrite it?", default=False):
            _echo("Nothing written.")
            return

    answers = _InitAnswers(language=language)
    if not non_interactive:
        if answers.language is None:
            answers.language = _ask_language()
        {Profile.free: _ask_free, Profile.hosted: _ask_hosted, Profile.custom: _ask_custom}[profile](answers)

    def write_files() -> None:
        IMPULSE_DIR.mkdir(parents=True, exist_ok=True)
        GITIGNORE_PATH.write_text("*\n", encoding="utf-8")   # first: the keys are never committed
        _write_private_file(TOML_PATH, _render_toml(answers).encode("utf-8"))

    _write_or_exit(write_files, TOML_PATH)
    try:
        toml = load_toml(TOML_PATH)
    except ConfigError as exc:   # cannot happen with validated answers; reported rather than hidden
        _fail(_config_error_text(exc))
    _SECRETS.use_toml(toml)
    set_keys = [name for name, value in answers.keys.items() if value]
    _echo(f"Wrote {TOML_PATH} (profile {profile.value}, language {answers.language}) and {GITIGNORE_PATH}")
    if set_keys:
        _echo(f"Keys stored in {TOML_PATH}: {', '.join(set_keys)}")
    if profile is Profile.free:
        _write_starter_config(answers)

    if not non_interactive and not no_ping:
        rows = _init_ping_rows(answers, toml)
        if rows:
            _print_rows(rows)
            if any(row.status == "FAIL" for row in rows):
                _echo("Some models did not answer the ping (for example the server is not running or the model is not pulled).")
    _echo("Next: impulse doctor")


# ------------------------------------------------------------------------------------------------
# impulse generate (SPEC.md sections 8.1, 11, and 17)
# ------------------------------------------------------------------------------------------------

DEFAULT_GENERATE_OUT = Path("scenarios") / "generated.yaml"
# Commands without a config.yaml (generate, convert-failure, calibrate) resolve their models from the
# environment and config.toml only (SPEC.md section 8.1 precedence).
STANDALONE_NO_MODEL = (
    "no model configured for the {role} role, which {command} uses. Set the environment variable "
    "{env}, or [models] {role} in .impulse/config.toml."
)
GENERATE_NO_MODEL = STANDALONE_NO_MODEL.format(
    role="primary", command="impulse generate", env=MODEL_ENV_VARS["primary"]
)
GENERATE_NO_LANGUAGE = (
    "--language is required: .impulse/config.toml has no [defaults] language (or still has the "
    f"{LANGUAGE_PLACEHOLDER} placeholder), and Impulse never assumes a language (for example --language en-IN)."
)


def _stdin_is_tty() -> bool:
    """True when the stream the domain-guard question would read from is a terminal."""
    try:
        return sys.stdin is not None and bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):   # closed or replaced stdin: no terminal
        return False


def _generate_language(language: Optional[str], toml: Optional[TomlConfig]) -> str:
    """``--language`` (checked with the language-code rule), else ``[defaults] language``; neither is exit 2."""
    if language is not None:
        language = language.strip()
        if not _valid_language(language):
            _fail(
                f"--language {language} is not a valid language code; it must match {LANGUAGE_PATTERN} "
                "(for example en-IN, hi-IN, hi-Latn-IN)"
            )
        return language
    configured = toml.defaults.language if toml is not None else None
    if configured is None or not configured.strip() or configured.strip() == LANGUAGE_PLACEHOLDER:
        _fail(GENERATE_NO_LANGUAGE)
    return configured.strip()


def _read_prompt_file(path: Path) -> str:
    """The system prompt file, verbatim except that line endings become ``\\n`` (read as UTF-8 text with
    universal newlines, optional BOM); a missing, unreadable, or blank file is exit 2."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        _fail(f"--prompt {path}: file not found")
    except OSError as exc:   # e.g. a folder, or no permission
        _fail(f"--prompt {path}: cannot read file: {exc.strerror or type(exc).__name__}")
    except UnicodeDecodeError:
        _fail(f"--prompt {path}: file is not valid UTF-8")
    if not text.strip():
        _fail(f"--prompt {path}: the system prompt file is empty")
    return text


def _standalone_model_source(role: str) -> str:
    """Where a role's model string came from for a command without a config.yaml: the environment
    variable when it is set to a non-placeholder value, else config.toml (same order as ``resolve_model``)."""
    env_name = MODEL_ENV_VARS[role]
    if not is_placeholder(os.environ.get(env_name)):
        return f"the environment variable {env_name}"
    return f"[models] {role} in .impulse/config.toml"


def _standalone_model(toml: Optional[TomlConfig], role: str, command: str) -> tuple[str, dict[str, Any]]:
    """A role's model and call options for a command that reads no config.yaml (SPEC.md section 8.1
    precedence: ``IMPULSE_MODEL_<ROLE>``, then config.toml); a resolved key is put into the options as
    ``api_key``. Unresolved model, unknown provider, or missing key: exit 2, before any LLM call."""
    bare = ProjectConfig(simulator=SimulatorConfig(), config_dir=Path.cwd())
    try:
        model = resolve_model(bare, role, toml)
    except ConfigError:
        _fail(STANDALONE_NO_MODEL.format(role=role, command=command, env=MODEL_ENV_VARS[role]))
    options = resolve_model_options(bare, role, toml)
    try:
        key = api_key_for_role(bare, role, toml)
    except UnknownProviderError:   # its field would be a config.yaml field, which the command never reads
        _fail(
            _redact(
                f"cannot determine the provider of model {model}, set by {_standalone_model_source(role)} "
                f"(the {role} role, which {command} uses)."
            )
        )
    except ConfigError as exc:   # a hosted model whose key is not set
        _fail(_config_error_text(exc))
    if key:
        _SECRETS.values.append(key)   # redacted from any LiteLLM error printed below
        options["api_key"] = key
    return model, options


def _generate_model(toml: Optional[TomlConfig]) -> tuple[str, dict[str, Any]]:
    """The primary model and its call options for ``impulse generate``."""
    return _standalone_model(toml, "primary", "impulse generate")


@_command("generate")
def generate_command(
    prompt: Annotated[
        Path, typer.Option("--prompt", help="The agent's system prompt file (scenarios are generated from it).", show_default=False)
    ],
    domain: Annotated[
        str,
        typer.Option(
            "--domain",
            help="fsi, healthcare, recruitment, ecommerce, edtech, telecom, realestate, travel, legal, or general "
            "(sets the default assertions, metrics, and max_turns).",
            show_default=False,
        ),
    ],
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            # "\[" keeps rich help markup from reading [defaults] as a style tag.
            help="Language code, e.g. en-IN (default: \\[defaults] language in .impulse/config.toml).",
        ),
    ] = None,
    count: Annotated[int, typer.Option("--count", min=1, help="Number of scenarios to generate.")] = 11,
    out: Annotated[Path, typer.Option("--out", help="Output YAML file (relative to the current directory).")] = (
        DEFAULT_GENERATE_OUT
    ),
    yes: Annotated[
        bool, typer.Option("--yes", help="Do not ask when the system prompt looks like another domain.")
    ] = False,
    voice_mode: Annotated[
        bool,
        typer.Option("--voice-mode", help="Offer the voice metric list; every scenario gets VOICE-001, VOICE-003, VOICE-005."),
    ] = False,
) -> None:
    """Generate test scenarios from an agent's system prompt with the primary model."""
    from impulse_evals import generator
    from impulse_evals.domains import resolve_domain
    from impulse_evals.reporter import write_bytes_atomic

    toml = _load_toml_or_exit()
    language_value = _generate_language(language, toml)
    system_prompt = _read_prompt_file(prompt)
    resolved, unknown_warning = resolve_domain(domain)
    if unknown_warning is not None:
        _echo(f"Warning: {unknown_warning}")
    model, options = _generate_model(toml)

    # Domain guard: warn on a mismatch; ask only on a terminal without --yes (no TTY behaves as --yes).
    detected = generator.domain_mismatch(system_prompt, domain)
    if detected is not None:
        _echo(generator.domain_mismatch_warning(system_prompt, detected, domain))
        if (
            not yes
            and _stdin_is_tty()
            and not typer.confirm(generator.GUARD_QUESTION.format(domain=domain), default=False)
        ):
            _fail(f"nothing generated. Rerun with --domain {detected}, or answer y (or pass --yes) to keep --domain {domain}.")

    try:
        scenarios = asyncio.run(
            generator.generate_scenarios(
                system_prompt,
                domain=resolved,
                language=language_value,
                count=count,
                voice_mode=voice_mode,
                model=model,
                options=options,
            )
        )
    except generator.GenerationError as exc:
        _fail(f"generation failed; nothing written: {_short_text(str(exc))}", code=1)
    except generator.ModelCallError as exc:   # the LiteLLM call itself (connection, authentication, ...)
        _fail(f"generation failed; the primary model call raised {_exception_text(exc.error)}", code=1)
    # Any other exception is a bug, not a generation failure: _guarded prints "unexpected error", exit 2.

    mode = "voice mode" if voice_mode else "text mode"
    header = (
        f"Written by impulse generate: domain {resolved}, language {language_value}, {mode}.\n"
        "Review the scenarios, then add them under scenarios: in a config.yaml."
    )
    # Every string value and header line is redacted before the YAML is dumped (AUDIT.md finding 21).
    text = generator.scenarios_yaml(scenarios, header=header, redact=_redact)

    # A temporary file in the same folder, then os.replace: a failed write keeps the previous file.
    _write_or_exit(lambda: write_bytes_atomic(out, text.encode("utf-8")), out)
    if len(scenarios) != count:
        _echo(f"Note: the model returned {len(scenarios)} scenarios; {count} were requested.")
    # The language can come from config.toml: the line is redacted whole (SPEC.md section 8.1).
    _echo(_redact(f"Wrote {out}: {len(scenarios)} scenarios (domain {resolved}, language {language_value}, {mode})."))


# ------------------------------------------------------------------------------------------------
# impulse convert-failure (SPEC.md section 15)
# ------------------------------------------------------------------------------------------------


@_command("convert-failure")
def convert_failure_command(
    file: Annotated[
        Path,
        typer.Argument(
            help="A stored scenario result: .impulse/runs/<run_id>/scenarios/<name>.json.", show_default=False
        ),
    ],
) -> None:
    """Write a regression scenario that reproduces a failed scenario (scenarios/regression/)."""
    from impulse_evals import generator, regression
    from impulse_evals.reporter import write_bytes_atomic

    toml = _load_toml_or_exit()
    try:
        result = regression.load_scenario_result(file)
    except regression.InvalidResultFile as exc:
        _fail(_redact(str(exc)))
    checks = regression.failed_checks(result)
    stored_name = " ".join(result.scenario_name.split())
    if not checks:
        _fail(_redact(f"{file}: scenario {stored_name} passed (no failed check, no error); there is no failure to convert"))
    model, options = _standalone_model(toml, "primary", "impulse convert-failure")

    warnings: list[str] = []
    try:
        scenario = asyncio.run(regression.convert_failure(result, model=model, options=options, warnings=warnings))
    except generator.GenerationError as exc:
        _fail(f"conversion failed; nothing written: {_short_text(str(exc))}", code=1)
    except generator.ModelCallError as exc:   # the LiteLLM call itself (connection, authentication, ...)
        _fail(f"conversion failed; the primary model call raised {_exception_text(exc.error)}", code=1)
    # Any other exception is a bug, not a conversion failure: _guarded prints "unexpected error", exit 2.
    for warning in dict.fromkeys(warnings):
        _echo(f"Warning: {_redact(warning)}")

    names = ", ".join(check.check_id or "scenario error" for check in checks)
    header = (
        f"Written by impulse convert-failure: reproduces the failure of scenario {stored_name} ({names}).\n"
        "Review it, then add it under scenarios: in a config.yaml."
    )
    # Every string value, header line, and the stored name in the file name are redacted before
    # anything is written (AUDIT.md finding 21).
    text = regression.regression_yaml(scenario, header=header, redact=_redact)
    target = regression.regression_path(result, redact=_redact)

    # A temporary file in the same folder, then os.replace: a failed write keeps the previous file.
    _write_or_exit(lambda: write_bytes_atomic(target, text.encode("utf-8")), target)
    # The stored name and check IDs come from FILE, the scenario name from the LLM: redacted whole
    # (never cut), so a key in any of them cannot be printed (SPEC.md section 8.1).
    new_name = " ".join(scenario.name.split())
    _echo(_redact(f"Wrote {target}: scenario {new_name} reproduces the failure of {stored_name} ({names})."))
    for note in regression.regression_notes(scenario, result):
        _echo(f"Note: {note}")


# ------------------------------------------------------------------------------------------------
# impulse calibrate (SPEC.md section 15)
# ------------------------------------------------------------------------------------------------

_CALIBRATION_REASON_CHARS = 160


def _print_calibration(directory: Path, result: Any) -> None:
    """Per-item lines, then item count, skipped count, agreement rate, and Cohen's kappa."""
    judged = len(result.judged)
    _echo(
        f"Calibration of {directory}: {result.item_count} items, {judged} judged, "
        f"{result.skipped_count} skipped"
    )
    width = max((len(outcome.item.path.name) for outcome in result.outcomes), default=0)
    for outcome in result.outcomes:
        item = outcome.item
        reason = _short_text(outcome.reason, _CALIBRATION_REASON_CHARS)
        if outcome.skipped:
            status, verdict = "SKIP", f"judge returned skip ({reason})"
        elif outcome.agrees:
            status, verdict = "AGREE", f"judge {outcome.judge_label}"
        else:
            status, verdict = "DIFF", f"judge {outcome.judge_label} ({reason})"
        name = _redact(item.path.name).ljust(width)
        _echo(f"  {status.ljust(5)}  {name}  {item.check_id}: human {item.human_label}, {verdict}")
    _echo(f"Items: {result.item_count}")
    _echo(f"Skipped (judge returned skip; left out of agreement and kappa): {result.skipped_count}")
    if result.agreement_rate is None:
        _echo("Agreement rate: n/a (no judged items)")
        _echo("Cohen's kappa: n/a (no judged items)")
        return
    _echo(f"Agreement rate: {result.agreement_rate:.1%} ({result.agreeing_count} of {judged} judged items)")
    _echo(f"Cohen's kappa: {result.kappa:.3f}")


@_command("calibrate")
def calibrate_command(
    directory: Annotated[
        Path,
        typer.Argument(
            help="Folder of labeled items: one JSON file per item with transcript, check, and human_label.",
            show_default=False,
        ),
    ],
) -> None:
    """Measure how often the judge agrees with human labels (agreement rate and Cohen's kappa)."""
    from impulse_evals import calibration
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    toml = _load_toml_or_exit()
    try:
        items = calibration.load_items(directory)
    except calibration.InvalidItems as exc:
        for problem in exc.problems:   # may quote file values (e.g. a human_label): redacted, then shortened
            _echo(f"Error: {_short_text(problem)}", err=True)
        raise typer.Exit(2) from None

    models: dict[str, str] = {}
    options: dict[str, dict[str, Any]] = {}
    for role in calibration.required_tiers(items):   # only the roles the checks route to
        models[role], options[role] = _standalone_model(toml, role, "impulse calibrate")
    judge = LiteLLMJudge(
        models.get("primary", ""), models.get("fast", ""), options.get("primary"), options.get("fast")
    )
    try:
        result = asyncio.run(calibration.calibrate(items, judge))
    except calibration.CalibrationError as exc:
        _fail(
            f"calibration stopped: the judge call for {_redact(exc.item.path.name)} ({exc.item.check_id}) raised "
            f"{_exception_text(exc.error)}"
        )

    _print_calibration(directory, result)
    for warning in judge.get_warnings():
        _echo(f"Warning: {_redact(warning)}")


if __name__ == "__main__":
    app()
