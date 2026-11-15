# SPDX-License-Identifier: Apache-2.0
"""Reports: JSON and one self-contained HTML file (SPEC.md section 16).

- ``write_json(run, path)`` writes ``RunResult.model_dump_json(indent=2)`` (UTF-8, one trailing
  newline). Any pydantic model works the same way, so the CLI also writes the stored
  ``scenarios/<name>.json`` files with it (``sanitize_name`` gives the file name, SPEC.md section 15:
  Unicode letters, digits, ``-``, ``_`` and combining marks kept after NFC, anything else ``_``).
- ``render_html(run, embed_audio, run_dir) -> str`` returns the HTML report; ``write_html`` renders
  and writes it as UTF-8. The run is only read, never changed.
- Every file is written atomically by ``write_bytes_atomic(path, data)`` (a temporary file in the same
  folder, then ``os.replace``; on an error the temporary file is removed and an existing file is left
  as it was), which the CLI also uses for the compare HTML and the generated / regression YAML.
- ``render_compare_html(compare, *, notes=()) -> str`` returns the ``impulse compare`` HTML report:
  critical regression banner, the "High regressions" banner when there are any, the notes (the dry-run
  sentences) above the tables, regressions, improvements, and newly failing checks (fail -> fail)
  tables, and one inline-SVG delta bar per metric of ``metric_summary_deltas``.
- ``write_run_files(run, run_dir, with_html=..., embed_audio=...)`` writes a run folder (``run.json``,
  ``scenarios/<name>.json``, ``report.html``) for ``impulse run`` and ``testing.impulse_check``;
  ``ensure_impulse_dir()`` creates ``.impulse/`` (with ``.gitignore``) when it does not exist.

HTML report layout: a header (run ID, timestamp, agent domain and type, entrypoint, the distinct
scenario languages, scenario counts, judge model, mode, providers, and ``RunResult.warnings``), then
the sections in SPEC.md order: critical failures, scenario table, cohort table (with gap warnings),
metric summary (proxy, informational, and skipped labelled), latency profile (with the
"Per-service latency" tables only when a transcript has ``metadata["pipecat_metrics"]``),
per-scenario collapsible transcripts (the exact inject_events note for each scenario with
``inject_events``, whose block opens expanded and whose scenario-table row carries an "inject_events
noted (v2)" badge; failing turns highlighted, tool call lines before the agent text, an ``<audio>``
player on every turn with ``audio_path``, user and agent), and the tool call trace per turn (its
result column through ``transcriber.redact_tool_result``). Each scenario body starts with the full
call recording player when ``transcript.recording_path`` is set; the turn cell of a failed assertion
or a poor counted metric carries a "Hear turn N" in-page link to that turn's player (each turn's player
sits in ``<div class="turn-audio" id="scenario-S-turn-N-audio">``); voice-mode scenarios with ``end_ms`` get a static
SVG call timeline (caller / agent rows, dead-air gaps, slow replies) before the transcript list.

Noise simulation (EXPERIMENTAL; read from ``metadata["impulse_noise"]`` and the per-turn
``impulse_voice`` records, so a run without noise renders exactly as before): a "noise simulation"
header row with the levels, a "noise: {label}, {snr} dB" badge in the scenario table and the scenario
summary, the clean caller audio in a collapsed "clean caller audio (before noise)" player under the
(noisy) caller player that "Hear turn N" points to, and the "Pass rate by noise level" table after the
cohorts when the run has two or more levels ("clean" counts as one).

Audio: ``embed_audio=True`` embeds each WAV file (``Turn.audio_path`` and
``CallTranscript.recording_path``, relative to ``run_dir``) as ``data:audio/wav;base64,...`` and prints
the SPEC.md section 16 size warning when every embedded player copy in the page (user, agent and
recording players; the "Hear turn N" links embed nothing) totals more than 50 MB of base64 text (MiB; fixes round 4, O4); ``embed_audio=False`` links the relative path. Only ``.wav`` files
inside ``<run_dir>/audio/`` (checked after resolving the path) are
embedded, and only ``audio/<name>.wav`` paths (no ``..``, not absolute; the file is not read) are
linked; any other path gets the same note in both modes ("audio not embedded (only .wav files inside
the run's audio/ folder are embedded): <path>"). ``\\`` in ``audio_path`` is read as ``/`` in both modes.

Security: every text taken from the run (transcripts, reasons, names, config values, warnings, tool
arguments) is HTML-escaped. The page loads nothing from outside: no ``src`` / ``href`` / CSS ``url(``
value with ``http:``, ``https:`` or ``//``, no external CSS or JS, and no script at all (the
template's own inline style only; its Content-Security-Policy also forbids scripts).
"""

from __future__ import annotations

import base64
import contextlib
import functools
import html
import math
import os
import re
import secrets
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import quote

from pydantic import BaseModel

from impulse_evals.models import (
    CompareResult,
    MetricResult,
    RunResult,
    ScenarioDiff,
    ScenarioResult,
    ScoreResult,
    ToolCall,
    Turn,
)
from impulse_evals.transcriber import format_tool_call, json_sorted, redact_tool_result, tool_error_text

__all__ = [
    "EMBEDDED_AUDIO_WARNING",
    "IMPULSE_DIR",
    "INJECT_EVENTS_NOTE",
    "PER_SERVICE_HEADING",
    "RUNS_DIR",
    "STEM_BYTE_LIMIT",
    "cut_file_stem",
    "ensure_impulse_dir",
    "render_compare_html",
    "render_html",
    "sanitize_name",
    "scenario_file_stem",
    "write_bytes_atomic",
    "write_html",
    "write_json",
    "write_run_files",
]

PathArg = Union[str, os.PathLike]

# ------------------------------------------------------------------------------------------------
# Exact texts (SPEC.md sections 7 and 16)
# ------------------------------------------------------------------------------------------------

INJECT_EVENTS_NOTE = "inject_events defined but execution reserved for v2; this scenario ran without injection."
INJECT_EVENTS_MARKER = "inject_events noted (v2)"  # scenario table badge; the full note is in the transcript block
EMBEDDED_AUDIO_WARNING = (
    "Embedded audio is {n} MB; the HTML report will be large. Use --no-embed-audio to link the files instead."
)
EMBEDDED_AUDIO_LIMIT_BYTES = 50 * 1024 * 1024
PER_SERVICE_HEADING = "Per-service latency"
RECORDING_LABEL = "Full call recording (left channel: caller, right channel: agent)"
WAV_DATA_URI_PREFIX = "data:audio/wav;base64,"
AUDIO_FOLDER = "audio"
_AUDIO_NOT_FOUND = "audio file not found in the run folder"
_AUDIO_NOT_EMBEDDED = "audio not embedded (only .wav files inside the run's audio/ folder are embedded)"
NOISE_METADATA_KEY = "impulse_noise"
NOISE_LEVEL_HEADING = "Pass rate by noise level"
CLEAN_AUDIO_LABEL = "clean caller audio (before noise)"
CLEAN_LEVEL = "clean"

TEMPLATE_FILE = "report_template.html"
_SPDX_PREFIX = "<!-- SPDX-License-Identifier:"
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_COMBINING_MARKS = ("Mn", "Mc")   # Devanagari vowel signs, virama, anusvara, ...
_NOT_AVAILABLE = "n/a"

# ------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------


def _file_name_char(char: str) -> str:
    if char.isalnum() or char in "-_" or unicodedata.category(char) in _COMBINING_MARKS:
        return char
    return "_"


def sanitize_name(name: str) -> str:
    """A scenario name for file names (SPEC.md section 15): NFC-normalised, then every character other
    than a Unicode letter or digit (``str.isalnum()``), ``-``, ``_``, or a combining mark (categories
    Mn and Mc, needed for Devanagari vowel signs and the virama) becomes ``_``. For example
    ``"क्लिनिक बुकिंग"`` gives ``"क्लिनिक_बुकिंग"`` and ``"clinic booking/v2"`` gives
    ``"clinic_booking_v2"``."""
    return "".join(_file_name_char(char) for char in unicodedata.normalize("NFC", name))


def write_bytes_atomic(path: PathArg, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically: to a temporary file in the same folder, then
    ``os.replace`` onto ``path``. On any error the temporary file is removed and the error is raised;
    an existing file at ``path`` is then left untouched (never a partial file). Missing parent folders
    are created."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Hidden, unique, and short enough for any file name limit (the name is cut to 64 characters).
    temporary = target.with_name(f".{target.name[:64]}.{secrets.token_hex(4)}.tmp")
    try:
        with open(temporary, "xb") as handle:
            handle.write(data)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return target


def write_json(run: BaseModel, path: PathArg) -> Path:
    """Write ``run.model_dump_json(indent=2)`` plus one newline to ``path`` (UTF-8, ``\\n`` line ends),
    atomically (``write_bytes_atomic``: a failed write leaves the previous file as it was).

    ``run`` is a ``RunResult`` (``run.json``) or any other model, e.g. a ``ScenarioResult``
    (``scenarios/<name>.json``). Missing parent folders are created.
    """
    return write_bytes_atomic(path, (run.model_dump_json(indent=2) + "\n").encode("utf-8"))


def write_html(run: RunResult, path: PathArg, *, embed_audio: bool = True, run_dir: Optional[PathArg] = None) -> Path:
    """Render the HTML report and write it to ``path`` as UTF-8, atomically (``write_bytes_atomic``).

    Audio paths are resolved against ``run_dir`` (default: the folder of ``path``).
    """
    target = Path(path)
    page = render_html(run, embed_audio=embed_audio, run_dir=run_dir if run_dir is not None else target.parent)
    return write_bytes_atomic(target, page.encode("utf-8"))


# ------------------------------------------------------------------------------------------------
# Run storage (SPEC.md section 15): .impulse/runs/<run_id>/run.json, scenarios/<name>.json, report.html
# ------------------------------------------------------------------------------------------------

IMPULSE_DIR = Path(".impulse")   # relative to the current directory
RUNS_DIR = IMPULSE_DIR / "runs"
# Windows device names (compared upper-cased); the superscript digits pass str.isalnum(), and
# COM¹ ... LPT³ are device names too (os.path.isreserved on Python 3.14).
_DEVICE_DIGITS = [str(i) for i in range(1, 10)] + ["¹", "²", "³"]
_WINDOWS_RESERVED = (
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{d}" for d in _DEVICE_DIGITS} | {f"LPT{d}" for d in _DEVICE_DIGITS}
)
# A file stem is cut to at most this many UTF-8 bytes, so "<stem>_2.json", "<stem>_t12_agent.wav" and
# "<stem>_regression.yaml" stay under the 255-byte file name limit of common file systems.
STEM_BYTE_LIMIT = 200


def ensure_impulse_dir(folder: PathArg = IMPULSE_DIR) -> None:
    """Create ``folder`` (default ``.impulse/``); a newly created folder also gets ``.gitignore`` with
    ``*`` (run output and keys never belong in version control). An existing folder is left as it is."""
    target = Path(folder)
    if target.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    (target / ".gitignore").write_text("*\n", encoding="utf-8")


def _is_combining_mark(char: str) -> bool:
    return unicodedata.category(char) in _COMBINING_MARKS


def cut_file_stem(stem: str, limit: int = STEM_BYTE_LIMIT) -> str:
    """``stem`` cut to at most ``limit`` UTF-8 bytes on a character boundary (unchanged when it fits).
    A base character is never separated from its combining marks: when the cut would fall inside
    such a cluster, the whole cluster (base included) is dropped."""
    if len(stem.encode("utf-8")) <= limit:
        return stem
    size = end = 0
    for char in stem:
        size += len(char.encode("utf-8"))
        if size > limit:
            break
        end += 1
    if end < len(stem) and _is_combining_mark(stem[end]):   # the cut falls inside a cluster
        while end > 0 and _is_combining_mark(stem[end - 1]):
            end -= 1
        end = max(end - 1, 0)   # drop the cluster's base character too
    return stem[:end]


def scenario_file_stem(name: str, used: set[str]) -> str:
    """A file stem for a scenario: sanitized (SPEC.md section 15), cut to at most 200 UTF-8 bytes
    (``cut_file_stem``), never a Windows device name, and unique in the folder (case-insensitively),
    e.g. ``a_b`` then ``a_b_2`` for ``a/b`` and ``a b``. ``used`` holds the stems already taken
    (casefolded) and is updated."""
    stem = cut_file_stem(sanitize_name(name)) or "_"
    if stem.upper() in _WINDOWS_RESERVED:
        stem = "_" + stem
    candidate, n = stem, 1
    while candidate.casefold() in used:
        n += 1
        candidate = f"{stem}_{n}"
    used.add(candidate.casefold())
    return candidate


def write_run_files(
    run: RunResult, run_dir: PathArg, *, with_html: bool = True, embed_audio: bool = True
) -> list[Path]:
    """Write ``run.json`` and ``scenarios/<name>.json`` (always: ``impulse report`` and ``impulse compare``
    read them), plus ``report.html`` when ``with_html`` is true, into ``run_dir``. Returns the paths of
    ``run.json`` and (when written) ``report.html``. Used by ``impulse run`` and ``impulse_check``."""
    folder = Path(run_dir)
    written = [write_json(run, folder / "run.json")]
    used: set[str] = set()
    for result in run.scenario_results:
        write_json(result, folder / "scenarios" / f"{scenario_file_stem(result.scenario_name, used)}.json")
    if with_html:
        written.append(write_html(run, folder / "report.html", embed_audio=embed_audio, run_dir=folder))
    return written


# ------------------------------------------------------------------------------------------------
# Template and small HTML helpers
# ------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _template() -> str:
    """The packaged page template without its SPDX header line (that line licenses the template
    file, not the user's report)."""
    text = resources.files("impulse_evals").joinpath("templates").joinpath(TEMPLATE_FILE).read_text(encoding="utf-8")
    first, _, rest = text.partition("\n")
    if first.lstrip("﻿").startswith(_SPDX_PREFIX):
        return rest
    return text


def _page(title: str, body: str) -> str:
    # One pass over the template only: inserted values are never scanned for placeholders.
    values = {"title": _e(title), "body": body}
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], _template())


def _e(value: Any) -> str:
    """HTML-escape any value (quotes included, so the result is safe in attributes too)."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return _NOT_AVAILABLE
    if not _is_number(value):
        return _e(value)
    number = float(value)
    if not math.isfinite(number):
        return _e(number)
    return f"{number:.{digits}f}"


def _signed(value: Any, digits: int = 2) -> str:
    if not _is_number(value) or not math.isfinite(float(value)):
        return _num(value, digits)
    return f"{float(value):+.{digits}f}"


def _pct(value: Any) -> str:
    if not _is_number(value) or not math.isfinite(float(value)):
        return _num(value)
    return f"{float(value) * 100:.1f}%"


def _json_text(value: Any) -> str:
    # Same text as the judge's tool call lines; never raises (mixed key types, tuple keys).
    return json_sorted(value)


def _badge(text: Any, css: str) -> str:
    return f'<span class="badge {css}">{_e(text)}</span>'


def _code(value: Any) -> str:
    return f"<code>{_e(value)}</code>"


def _list(items: Iterable[str]) -> str:
    """A bullet list of already-escaped items, or an empty string."""
    entries = [f"<li>{item}</li>" for item in items]
    return f'<ul class="plain">{"".join(entries)}</ul>' if entries else ""


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], numeric: Iterable[int] = ()) -> str:
    """A table; ``headers`` are plain text, cells are already-escaped HTML."""
    right = set(numeric)

    def cell(tag: str, index: int, content: str) -> str:
        css = ' class="num"' if index in right else ""
        return f"<{tag}{css}>{content}</{tag}>"

    head = "".join(cell("th", index, _e(text)) for index, text in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", index, content) for index, content in enumerate(row)) + "</tr>" for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _distinct(values: Iterable[Any]) -> list[Any]:
    seen: dict[Any, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _scenario_anchor(index: int) -> str:
    return f"scenario-{index}"


def _turn_anchor(scenario_index: int, turn_number: int) -> str:
    return f"scenario-{scenario_index}-turn-{turn_number}"


def _turn_ref(scenario_index: Optional[int], turn_index: Optional[int]) -> str:
    if turn_index is None:
        return ""
    if scenario_index is None:
        return _e(turn_index)
    return f'<a href="#{_turn_anchor(scenario_index, turn_index)}">{_e(turn_index)}</a>'


def _status_badge(result: ScenarioResult) -> str:
    if result.error is not None:
        return _badge("ERROR", "error")
    return _badge("PASS", "pass") if result.passed else _badge("FAIL", "fail")


def _result_badge(score: ScoreResult) -> str:
    css = {"pass": "pass", "fail": "fail"}.get(score.result, "skip")
    badge = _badge(score.result, css)
    if score.no_verdict:
        badge += _badge("no verdict", "flag-warning")
    return badge


def _metric_label(metric: MetricResult) -> str:
    badge = _badge(metric.label, f"label-{_e(metric.label)}")
    if metric.is_proxy:
        badge += _badge("proxy", "flag-proxy")
    if metric.informational:
        badge += _badge("informational", "flag-informational")
    return badge


def _success_badge(success: Optional[bool]) -> str:
    if success is True:
        return _badge("True", "pass")
    if success is False:
        return _badge("False", "fail")
    return _badge("None (not captured)", "skip")


def _disclaimers(scores: Iterable[ScoreResult]) -> str:
    texts = _distinct(score.disclaimer for score in scores if score.disclaimer)
    return "".join(f'<p class="disclaimer">{_e(text)}</p>' for text in texts)


def _is_poor(metric: MetricResult) -> bool:
    return metric.label == "poor" and not metric.skipped and not metric.informational


# ------------------------------------------------------------------------------------------------
# Audio players
# ------------------------------------------------------------------------------------------------


def _slashed(audio_path: str) -> str:
    """``audio_path`` with ``\\`` turned into ``/`` (a path recorded on Windows works everywhere)."""
    return audio_path.replace("\\", "/")


def _relative_src(audio_path: str) -> Optional[str]:
    """``audio_path`` as a relative URL when, with ``\\`` read as ``/``, it names a ``.wav`` file (any
    case) under ``audio/`` of the run folder: not absolute, no scheme or drive, no ``..`` segment, and
    (``.`` and empty segments dropped) first segment ``audio`` (the same ``audio/*.wav`` rule as
    embedding; the file need not exist, since linking reads no file). None otherwise."""
    path = _slashed(audio_path)
    if not path or path.startswith("/") or _URL_SCHEME.match(path):
        return None
    parts = [part for part in path.split("/") if part not in ("", ".")]
    if len(parts) < 2 or parts[0] != AUDIO_FOLDER or ".." in parts:
        return None
    name = parts[-1]
    if not name.lower().endswith(".wav") or len(name) <= len(".wav"):
        return None
    return quote("/".join(parts), safe="/")


class _AudioPlayers:
    """Builds the ``<audio>`` players: data URIs (``embed``) or relative links.

    Embedding reads files from disk into the report, so only ``.wav`` files (any case) inside
    ``<run_dir>/audio/`` are embedded, checked after resolving the path (``..``, absolute paths, and
    symlinks cannot pull in other files such as ``.impulse/config.toml``). Linking reads no file.
    """

    def __init__(self, run_dir: Optional[PathArg], embed: bool) -> None:
        self.embed = embed
        base = Path(run_dir) if run_dir is not None else Path.cwd()
        try:
            self.base: Optional[Path] = base.resolve()
            self.audio_dir: Optional[Path] = (self.base / AUDIO_FOLDER).resolve()
        except (OSError, RuntimeError, ValueError):
            self.base = self.audio_dir = None
        self._uris: dict[Path, tuple[str, int]] = {}   # file -> (data URI, base64 characters)
        # base64 characters of every embedded player copy written so far: what the page really holds
        # (a third more than the files; fixes round 4, O4), counted once per copy.
        self.embedded_bytes = 0

    def _find(self, audio_path: str) -> Union[Path, str]:
        """The file to embed, or the note shown instead of a player. The same syntactic rule as linking
        (``_relative_src``: relative, under ``audio/``, no ``..``, a ``.wav`` name; fixes round 4, L25),
        then the resolved file must be inside ``<run_dir>/audio/``."""
        if self.base is None or self.audio_dir is None:
            return _AUDIO_NOT_FOUND
        if _relative_src(audio_path) is None:
            return _AUDIO_NOT_EMBEDDED
        try:
            candidate = (self.base / _slashed(audio_path)).resolve()
            if not (candidate.is_relative_to(self.audio_dir) and candidate.suffix.lower() == ".wav"):
                return _AUDIO_NOT_EMBEDDED
            if candidate.is_file():
                return candidate
        except (OSError, RuntimeError, ValueError):
            pass
        return _AUDIO_NOT_FOUND

    def player(self, audio_path: str) -> str:
        """One player. Embedded: the base64 characters of the data URI are added to ``embedded_bytes``
        on EVERY call, so the size warning counts each copy written into the page (B-rev, O4). Call it
        only for a player that is written into the page. Notes never carry an absolute path (L25)."""
        shown = _shown_path(audio_path)
        if self.embed:
            path = self._find(audio_path)
            if not isinstance(path, Path):
                return f'<div class="turn-extra">{_e(path)}: {_e(shown)}</div>'
            cached = self._uris.get(path)
            if cached is None:
                try:
                    data = path.read_bytes()
                except OSError as exc:
                    # str(exc) would hold the absolute file name; the OS text or the type only (L25).
                    why = (getattr(exc, "strerror", None) or "").strip() or type(exc).__name__
                    return f'<div class="turn-extra">audio file could not be read: {_e(shown)} ({_e(why)})</div>'
                encoded = base64.b64encode(data).decode("ascii")
                cached = (WAV_DATA_URI_PREFIX + encoded, len(encoded))
                self._uris[path] = cached
            uri, size = cached
            self.embedded_bytes += size
            return f'<audio controls preload="metadata" src="{uri}"></audio>'
        src = _relative_src(audio_path)
        if src is None:   # the note embed mode shows for a path it refuses
            return f'<div class="turn-extra">{_e(_AUDIO_NOT_EMBEDDED)}: {_e(shown)}</div>'
        return f'<audio controls preload="metadata" src="{_e(src)}"></audio>'


_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _shown_path(audio_path: str) -> str:
    """``audio_path`` as a note shows it: unchanged when relative; an absolute path (``/...``, ``\\\\...``
    or a drive letter) or a URL (``file:///...``, ``https://...``) is cut to its last segment, so a report
    never carries a folder of the machine that wrote it (fixes round 4, L25)."""
    path = _slashed(audio_path)
    if path.startswith("/") or _DRIVE_RE.match(path) or _URL_SCHEME.match(path):
        return path.rstrip("/").rsplit("/", 1)[-1] or path
    return audio_path


def _warn_embedded_size(total_bytes: int) -> None:
    """``total_bytes`` is the base64 text of every embedded player copy (what the page holds; O4)."""
    if total_bytes > EMBEDDED_AUDIO_LIMIT_BYTES:
        size = f"{total_bytes / (1024 * 1024):.1f}"
        print(EMBEDDED_AUDIO_WARNING.format(n=size), file=sys.stderr)


# ------------------------------------------------------------------------------------------------
# Run report sections
# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# Noise simulation (EXPERIMENTAL)
# ------------------------------------------------------------------------------------------------


def _noise_level(result: ScenarioResult) -> Optional[tuple[str, float]]:
    """``(label, snr_db)`` of a scenario that ran with noise, else None (also for malformed metadata:
    a hand-edited run.json never breaks the report)."""
    metadata = result.transcript.metadata
    noise = metadata.get(NOISE_METADATA_KEY) if isinstance(metadata, dict) else None
    if not isinstance(noise, dict):
        return None
    raw_snr, label = noise.get("snr_db"), noise.get("label")
    if not isinstance(raw_snr, (int, float)) or isinstance(raw_snr, bool) or not math.isfinite(float(raw_snr)):
        return None
    if not isinstance(label, str) or not label.strip():
        label = str(noise.get("profile") or "noise")
    return label.strip(), float(raw_snr)


def _level_name(level: Optional[tuple[str, float]]) -> str:
    return CLEAN_LEVEL if level is None else f"{level[0]}, {level[1]:g} dB"


def _run_has_noise(run: RunResult) -> bool:
    return any(_noise_level(result) is not None for result in run.scenario_results)


def _noise_badge(result: ScenarioResult) -> str:
    level = _noise_level(result)
    return _badge(f"noise: {_level_name(level)}", "flag-info") if level is not None else ""


def _ordered_levels(run: RunResult) -> list[Optional[tuple[str, float]]]:
    """The distinct levels of the run: clean first, then from the mildest (highest SNR) to the harshest."""
    levels = list(dict.fromkeys(_noise_level(result) for result in run.scenario_results))
    return sorted(levels, key=lambda level: (0, 0.0, "") if level is None else (1, -level[1], level[0]))


def _clean_audio_paths(result: ScenarioResult) -> dict[int, str]:
    """User turn number -> its clean (before noise) audio file, from the ``impulse_voice`` records."""
    metadata = result.transcript.metadata
    voice = metadata.get("impulse_voice") if isinstance(metadata, dict) else None
    records = voice.get("turns") if isinstance(voice, dict) else None
    paths: dict[int, str] = {}
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        number, path = record.get("turn_index"), record.get("clean_audio_path")
        if isinstance(number, int) and isinstance(path, str) and path:
            paths[number] = path
    return paths


def _noise_level_section(run: RunResult) -> str:
    """"Pass rate by noise level": only when the run used noise at two or more levels."""
    if not _run_has_noise(run):
        return ""
    levels = _ordered_levels(run)
    if len(levels) < 2:
        return ""
    rows = []
    for level in levels:
        results = [result for result in run.scenario_results if _noise_level(result) == level]
        passed = sum(1 for result in results if result.passed)
        scores = [
            metric.score
            for result in results
            for metric in result.metric_results
            if metric.metric_id == "VOICE-009" and not metric.skipped and _is_number(metric.score)
        ]
        rows.append(
            [
                _e(_level_name(level)),
                str(len(results)),
                str(passed),
                str(len(results) - passed),
                _pct(passed / len(results)) if results else _NOT_AVAILABLE,
                _num(sum(scores) / len(scores)) if scores else _NOT_AVAILABLE,
            ]
        )
    return "".join(
        [
            f'<section id="noise-levels"><h2>{_e(NOISE_LEVEL_HEADING)}</h2>',
            (
                '<p class="muted">Noise simulation is experimental. Clean scenarios ran without noise; '
                "a lower SNR (signal-to-noise ratio) is noisier.</p>"
            ),
            _table(
                ["noise level", "scenarios", "passed", "failed", "pass rate", "mean VOICE-009"],
                rows,
                numeric=[1, 2, 3, 4, 5],
            ),
            "</section>",
        ]
    )


def _header(run: RunResult) -> str:
    languages = _distinct(result.transcript.language for result in run.scenario_results)
    counts = (
        f"{_e(run.total_scenarios)} ({_e(run.passed)} passed, {_e(run.failed)} failed; "
        f"pass rate {_pct(run.overall_pass_rate)})"
    )
    rows = [
        ("run_id", _code(run.run_id)),
        ("timestamp", _e(run.timestamp)),
        ("agent domain", _e(run.agent_domain)),
        ("agent type", _e(run.agent_type)),
        ("entrypoint", _code(run.agent_entrypoint)),
        ("languages", _e(", ".join(languages)) if languages else "none"),
        ("scenarios", counts),
        ("judge model", _code(run.judge_model)),
        ("mode", "voice" if run.voice_mode else "text"),
    ]
    if run.tts_provider is not None:
        rows.append(("TTS provider", _e(run.tts_provider)))
    if run.stt_provider is not None:
        rows.append(("STT provider", _e(run.stt_provider)))
    if _run_has_noise(run):
        levels = "; ".join(_level_name(level) for level in _ordered_levels(run))
        rows.append(("noise simulation", f"experimental; levels: {_e(levels)}"))
    meta ="".join(f"<dt>{_e(label)}</dt><dd>{value}</dd>" for label, value in rows)
    parts = [f'<header><h1>Impulse report</h1><dl class="meta">{meta}</dl>']
    if run.warnings:
        items = "".join(f"<li>{_e(warning)}</li>" for warning in run.warnings)
        parts.append(f'<h3>Warnings</h3><ul class="warnings">{items}</ul>')
    parts.append("</header>")
    return "".join(parts)


def _critical_owners(run: RunResult) -> list[tuple[Optional[int], Optional[str], ScoreResult]]:
    """Each ``critical_failures`` entry with the scenario (index, name) it came from, when found."""
    pool = [
        (index, result.scenario_name, score)
        for index, result in enumerate(run.scenario_results, start=1)
        for score in result.assertion_results
        if score.result == "fail" and score.severity == "critical"
    ]
    owners: list[tuple[Optional[int], Optional[str], ScoreResult]] = []
    for failure in run.critical_failures:
        found: tuple[Optional[int], Optional[str]] = (None, None)
        for position, (index, name, score) in enumerate(pool):
            if score == failure:
                found = (index, name)
                del pool[position]
                break
        owners.append((found[0], found[1], failure))
    return owners


def _critical_section(run: RunResult) -> str:
    parts = ['<section id="critical-failures"><h2>Critical failures</h2>']
    if run.critical_failures:
        rows = []
        for index, name, score in _critical_owners(run):
            scenario = (
                f'<a href="#{_scenario_anchor(index)}">{_e(name)}</a>' if index is not None else _NOT_AVAILABLE
            )
            rows.append(
                [
                    scenario,
                    _code(score.rule_id),
                    _e(score.regulation_ref),
                    _e(score.reason),
                    _turn_ref(index, score.turn_index),
                    _num(score.confidence),
                ]
            )
        parts.append(
            _table(["scenario_name", "rule_id", "regulation_ref", "reason", "turn", "confidence"], rows, numeric=[5])
        )
        parts.append(_disclaimers(run.critical_failures))
    else:
        parts.append('<p class="muted">No critical assertion failed.</p>')
    errors = [(index, result) for index, result in enumerate(run.scenario_results, start=1) if result.error is not None]
    if errors:
        rows = [
            [f'<a href="#{_scenario_anchor(index)}">{_e(result.scenario_name)}</a>', _e(result.error)]
            for index, result in errors
        ]
        parts.append("<h3>Scenario errors</h3>")
        parts.append(_table(["scenario_name", "error"], rows))
    parts.append("</section>")
    return "".join(parts)


def _scenario_table(run: RunResult) -> str:
    rows = []
    for index, result in enumerate(run.scenario_results, start=1):
        failed_assertions = sum(1 for score in result.assertion_results if score.result == "fail")
        poor_metrics = sum(1 for metric in result.metric_results if _is_poor(metric))
        rows.append(
            [
                str(index),
                f'<a href="#{_scenario_anchor(index)}">{_e(result.scenario_name)}</a>',
                _e(result.transcript.language),
                _e(result.transcript.accent) if result.transcript.accent else "",
                _status_badge(result)
                + (_badge(INJECT_EVENTS_MARKER, "flag-warning") if result.inject_events else "")
                + _noise_badge(result),
                f"{failed_assertions} / {len(result.assertion_results)}",
                f"{poor_metrics} / {len(result.metric_results)}",
                _e(result.error) if result.error is not None else "",
            ]
        )
    parts = ['<section id="scenarios"><h2>Scenarios</h2>']
    if rows:
        parts.append(
            _table(
                ["#", "scenario_name", "language", "accent", "status", "failed assertions", "poor metrics", "error"],
                rows,
                numeric=[0, 5, 6],
            )
        )
    else:
        parts.append('<p class="muted">This run has no scenario results.</p>')
    parts.append("</section>")
    return "".join(parts)


def _cohort_section(run: RunResult) -> str:
    parts = [
        '<section id="cohorts"><h2>Cohorts</h2>',
        '<p class="muted">One cohort per language and accent; cohorts are never blended.</p>',
    ]
    if not run.cohorts:
        parts.append('<p class="muted">No cohorts.</p></section>')
        return "".join(parts)
    notices = []
    for cohort in run.cohorts:
        if cohort.gap_warning:
            notices.append(
                f"Gap warning: cohort {_e(cohort.cohort_key)} pass rate {_pct(cohort.pass_rate)} is "
                f"{_num(cohort.gap_vs_best * 100 if _is_number(cohort.gap_vs_best) else cohort.gap_vs_best, 1)} "
                "percentage points below the best cohort (more than 15)."
            )
        if cohort.stt_warning:
            notices.append(
                f"STT warning: cohort {_e(cohort.cohort_key)} mean STT confidence "
                f"{_num(cohort.mean_stt_confidence)} is more than 0.10 below the best cohort."
            )
    if notices:
        parts.append(f'<ul class="warnings">{"".join(f"<li>{notice}</li>" for notice in notices)}</ul>')
    rows = []
    for cohort in run.cohorts:
        means = ", ".join(f"{metric_id} {_num(value)}" for metric_id, value in cohort.metric_means.items())
        rows.append(
            [
                _code(cohort.cohort_key),
                _e(cohort.language),
                _e(cohort.accent) if cohort.accent else "",
                _e(cohort.total),
                _e(cohort.passed),
                _e(cohort.failed),
                _pct(cohort.pass_rate),
                _pct(cohort.gap_vs_best),
                _badge("gap warning", "flag-warning") if cohort.gap_warning else "",
                _num(cohort.mean_stt_confidence),
                _badge("STT warning", "flag-warning") if cohort.stt_warning else "",
                _e(means),
            ]
        )
    parts.append(
        _table(
            [
                "cohort_key", "language", "accent", "total", "passed", "failed", "pass_rate", "gap_vs_best",
                "gap_warning", "mean_stt_confidence", "stt_warning", "metric_means",
            ],
            rows,
            numeric=[3, 4, 5, 6, 7, 9],
        )
    )
    parts.append("</section>")
    return "".join(parts)


def _metric_summary_section(run: RunResult) -> str:
    rows = []
    for metric_id, metric in run.metric_summary.items():
        rows.append(
            [
                _code(metric_id),
                _e(metric.name),
                _e(metric.category),
                _num(metric.score),
                _num(metric.raw_value),
                _metric_label(metric),
                _e(metric.reason),
            ]
        )
    # Metrics absent from the summary (every result skipped or informational) are still listed.
    others: dict[str, list[MetricResult]] = {}
    for result in run.scenario_results:
        for metric in result.metric_results:
            if metric.metric_id not in run.metric_summary:
                others.setdefault(metric.metric_id, []).append(metric)
    for metric_id, results in others.items():
        first = results[0]
        skipped = [metric for metric in results if metric.skipped]
        informational = [metric for metric in results if metric.informational and not metric.skipped]
        if len(skipped) == len(results):
            label = _badge("skipped", "label-skipped")
            reason = f"skipped in all scenario results ({len(results)}); first reason: {first.reason}"
        elif informational:
            label = _badge("informational", "flag-informational")
            reason = (
                "informational: shown per scenario, never counted in pass/fail "
                f"(results: {len(informational)}, skipped: {len(skipped)})"
            )
        else:
            label = _badge(_NOT_AVAILABLE, "label-skipped")
            reason = "not in the run summary"
        rows.append(
            [_code(metric_id), _e(first.name), _e(first.category), _NOT_AVAILABLE, _NOT_AVAILABLE, label, _e(reason)]
        )
    parts = [
        '<section id="metric-summary"><h2>Metric summary</h2>',
        '<p class="muted">Means over non-skipped, non-informational results. Proxy metrics are text-mode '
        "stand-ins for audio measurements.</p>",
    ]
    if rows:
        parts.append(
            _table(["metric_id", "name", "category", "score", "raw_value", "label", "reason"], rows, numeric=[3, 4])
        )
    else:
        parts.append('<p class="muted">No metric results.</p>')
    parts.append("</section>")
    return "".join(parts)


def _pipecat_rows(run: RunResult) -> Optional[list[tuple[int, str, dict]]]:
    """``(scenario index, scenario name, entry)`` for every ``pipecat_metrics`` entry, or None when
    no transcript has the key."""
    present = False
    rows: list[tuple[int, str, dict]] = []
    for index, result in enumerate(run.scenario_results, start=1):
        metadata = result.transcript.metadata
        if not isinstance(metadata, dict) or "pipecat_metrics" not in metadata:
            continue
        present = True
        entries = metadata["pipecat_metrics"]
        if isinstance(entries, list):
            rows.extend((index, result.scenario_name, entry) for entry in entries if isinstance(entry, dict))
    return rows if present else None


def _per_service_section(rows: list[tuple[int, str, dict]]) -> str:
    parts = [
        f"<h3>{_e(PER_SERVICE_HEADING)}</h3>",
        '<p class="muted">Per-service timings reported by the Pipecat bot (RTVI metrics). '
        "Informational, never scored.</p>",
    ]
    if not rows:
        parts.append('<p class="muted">The bot reported no per-service timings.</p>')
        return "".join(parts)
    groups: dict[tuple[str, str, str], list[float]] = {}
    for _, _, entry in rows:
        key = (str(entry.get("processor", "")), str(entry.get("kind", "")), str(entry.get("model", "")))
        values = groups.setdefault(key, [])
        value = entry.get("value_ms")
        if _is_number(value) and math.isfinite(float(value)):
            values.append(float(value))
    summary = [
        [
            _e(processor),
            _e(kind),
            _e(model),
            str(len(values)),
            _num(sum(values) / len(values), 1) if values else _NOT_AVAILABLE,
            _num(max(values), 1) if values else _NOT_AVAILABLE,
        ]
        for (processor, kind, model), values in groups.items()
    ]
    parts.append(_table(["processor", "kind", "model", "samples", "mean value_ms", "max value_ms"], summary, [3, 4, 5]))
    detail = [
        [
            f'<a href="#{_scenario_anchor(index)}">{_e(name)}</a>',
            _turn_ref(index, entry["turn_index"])
            if isinstance(entry.get("turn_index"), int)
            else _e(entry.get("turn_index")),
            _e(entry.get("kind")),
            _e(entry.get("processor")),
            _e(entry.get("model")),
            _num(entry.get("value_ms"), 1),
        ]
        for index, name, entry in rows
    ]
    parts.append(_table(["scenario_name", "turn", "kind", "processor", "model", "value_ms"], detail, numeric=[1, 5]))
    return "".join(parts)


def _latency_section(run: RunResult) -> str:
    profile = run.latency_profile
    what = (
        "Measured on real audio: time from the end of the user's speech to the agent's first audio."
        if profile.is_real_audio
        else "Text mode: time from sending the user turn to the agent's first output."
    )
    parts = [
        '<section id="latency"><h2>Latency profile</h2>',
        f'<p class="muted">{_e(what)} Impulse\'s own TTS and STT time is never included.</p>',
        _table(
            ["samples", "p50_ms", "p75_ms", "p95_ms", "p99_ms"],
            [
                [
                    str(len(profile.all_latencies_ms)),
                    _num(profile.p50_ms, 1),
                    _num(profile.p75_ms, 1),
                    _num(profile.p95_ms, 1),
                    _num(profile.p99_ms, 1),
                ]
            ],
            numeric=[0, 1, 2, 3, 4],
        ),
    ]
    if not profile.all_latencies_ms:
        parts.append('<p class="muted">No agent latencies were recorded.</p>')
    pipecat = _pipecat_rows(run)
    if pipecat is not None:
        parts.append(_per_service_section(pipecat))
    parts.append("</section>")
    return "".join(parts)


def _failing_turns(result: ScenarioResult) -> dict[int, list[str]]:
    """Turn number -> IDs of the failed assertions and poor metrics that point at it."""
    failing: dict[int, list[str]] = {}
    for score in result.assertion_results:
        if score.result == "fail" and score.turn_index is not None:
            failing.setdefault(score.turn_index, []).append(score.rule_id)
    for metric in result.metric_results:
        if _is_poor(metric) and metric.turn_index is not None:
            failing.setdefault(metric.turn_index, []).append(metric.metric_id)
    return failing


def _tool_call_line(number: int, call: ToolCall) -> str:
    css = "tool-call failed" if call.success is False else "tool-call"
    return f'<div class="{css}">{_e(format_tool_call(number, call))}</div>'


def _turn_item(
    scenario_index: int,
    number: int,
    turn: Turn,
    failing: list[str],
    audio: _AudioPlayers,
    clean_audio_path: Optional[str] = None,
) -> str:
    classes = f"turn {_e(turn.speaker)}" + (" failing" if failing else "")
    meta = [f"t={_e(turn.timestamp_ms)} ms"]
    if turn.latency_ms is not None:
        meta.append(f"latency {_num(turn.latency_ms, 0)} ms")
    if turn.interrupted:
        meta.append("interrupted")
    if turn.stt_confidence is not None:
        meta.append(f"STT confidence {_num(turn.stt_confidence)}")
    head = f'[turn {number}] <span class="speaker">{_e(turn.speaker.upper())}</span> &middot; ' + " &middot; ".join(meta)
    if failing:
        head += " " + _badge("failing: " + ", ".join(failing), "fail")
    parts = [f'<li class="{classes}" id="{_turn_anchor(scenario_index, number)}"><div class="turn-head">{head}</div>']
    parts.extend(_tool_call_line(number, call) for call in turn.tool_calls)
    parts.append(f'<div class="turn-text">{_e(turn.text)}</div>')
    if turn.heard_text is not None:
        parts.append(f'<div class="turn-extra">agent heard: {_e(turn.heard_text)}</div>')
    if turn.classified_intent is not None:
        confidence = f" ({_num(turn.intent_confidence)})" if turn.intent_confidence is not None else ""
        parts.append(f'<div class="turn-extra">intent: {_e(turn.classified_intent)}{confidence}</div>')
    if turn.audio_path:   # user and agent turns alike (B2); the "Hear turn N" links point here (B3)
        anchor = _turn_audio_anchor(scenario_index, number)
        parts.append(f'<div class="turn-audio" id="{anchor}">{audio.player(turn.audio_path)}</div>')
    if clean_audio_path and turn.speaker == "user":   # noise simulation: the audio before the noise
        parts.append(
            f'<details class="clean-audio"><summary>{_e(CLEAN_AUDIO_LABEL)}</summary>'
            f"{audio.player(clean_audio_path)}</details>"
        )
    parts.append("</li>")
    return "".join(parts)


def _turn_audio_anchor(scenario_index: int, turn_number: int) -> str:
    return f"{_turn_anchor(scenario_index, turn_number)}-audio"


def _check_turn_cell(scenario_index: int, turn_index: Optional[int], failed: bool, turns: Sequence[Turn]) -> str:
    """The "turn" cell of an assertion or metric row: the turn link, then (B3) an in-page "Hear turn N"
    link to that turn's player when the check failed (failed assertion, or poor counted metric) and the
    turn has an audio file. A link, not a second player: the audio is embedded once, in the transcript."""
    cell = _turn_ref(scenario_index, turn_index)
    if failed and isinstance(turn_index, int) and 1 <= turn_index <= len(turns) and turns[turn_index - 1].audio_path:
        anchor = _turn_audio_anchor(scenario_index, turn_index)
        cell += f' <a class="hear-turn" href="#{anchor}">Hear turn {_e(turn_index)}</a>'
    return cell


# ------------------------------------------------------------------------------------------------
# Call timeline (B4): a static inline SVG per voice-mode scenario
# ------------------------------------------------------------------------------------------------

# SPEC.md section 8.2 defaults, used when the transcript has no stored thresholds (older runs).
DEFAULT_TIMELINE_THRESHOLDS = {"dead_air_ms": 1500.0, "ttfa_warn_ms": 1500.0, "ttfa_fail_ms": 3000.0}
_TL_WIDTH = 720
_TL_LABEL = 56          # room for the row labels on the left
_TL_PAD = 12
_TL_ROW_H = 16
_TL_ROWS = {"user": ("caller", 6), "agent": ("agent", 28)}   # speaker -> (row label, y)
_TL_AXIS_Y = 60
_TL_LEGEND_Y = 76
_TL_HEIGHT = 88
# Styles of the timeline and the recording label; emitted once, only in reports that use them.
_AUDIO_STYLE = (
    "<style>"
    ".recording { font-weight: 600; margin: 8px 0 0; }"
    "svg.timeline { display: block; width: 100%; max-width: 720px; height: auto; margin: 6px 0 10px; }"
    "svg.timeline text { fill: var(--muted); font-size: 11px; }"
    "svg.timeline .caller { fill: var(--muted); }"
    "svg.timeline .agent { fill: var(--info); }"
    "svg.timeline .agent.slow { fill: var(--bad); }"
    "svg.timeline .dead-air { fill: var(--warn); opacity: 0.55; }"
    "svg.timeline .track { fill: var(--panel); stroke: var(--line); stroke-width: 1; }"
    "</style>"
)


_TL_MAX_MS = 10**12   # B-rev: a timeline value beyond this magnitude is treated as missing


def _ms(value: Any) -> str:
    return f"{float(value):.0f}"


def _timeline_value(value: Any) -> Optional[float]:
    """A timeline time or latency as a float, or None (treated as missing) when it is not a finite number
    or its magnitude exceeds 10^12 ms (a corrupted run.json never stops the report; B-rev). The range is
    checked before ``float()``, so a huge int never raises ``OverflowError``."""
    if not _is_number(value):
        return None
    if isinstance(value, int):
        return float(value) if -_TL_MAX_MS <= value <= _TL_MAX_MS else None
    return float(value) if math.isfinite(value) and abs(value) <= _TL_MAX_MS else None


def _timeline_span(turn: Turn) -> Optional[tuple[float, float]]:
    """``(start, end)`` of a turn drawn on the timeline, or None (no ``end_ms``, or a missing value)."""
    start, end = _timeline_value(turn.timestamp_ms), _timeline_value(turn.end_ms)
    return (start, end) if start is not None and end is not None else None


def _timeline_thresholds(metadata: Any) -> dict[str, float]:
    """``metadata["impulse_voice"]["thresholds"]`` (stored by the runner from ``metrics.voice``), each
    missing or non-numeric value replaced by its SPEC.md section 8.2 default."""
    thresholds = dict(DEFAULT_TIMELINE_THRESHOLDS)
    voice = metadata.get("impulse_voice") if isinstance(metadata, dict) else None
    stored = voice.get("thresholds") if isinstance(voice, dict) else None
    if isinstance(stored, dict):
        for key in thresholds:
            value = _timeline_value(stored.get(key))
            if value is not None:
                thresholds[key] = value
    return thresholds


def _has_timeline(result: ScenarioResult) -> bool:
    transcript = result.transcript
    return bool(transcript.voice_mode) and any(_timeline_span(turn) is not None for turn in transcript.turns)


def _timeline(result: ScenarioResult) -> str:
    """The static call timeline (B4): a caller row (user turns) and an agent row (agent turns), one rect
    per turn with ``end_ms``; a ``dead-air`` rect from a user turn's ``end_ms`` to the next agent turn's
    start when that agent turn's ``latency_ms`` > ``dead_air_ms``; class ``slow`` on an agent turn whose
    ``latency_ms`` > ``ttfa_fail_ms``. Every rect has a ``<title>``; no script, no link, no URL. A time
    that is not a finite number or exceeds 10^12 ms in magnitude is treated as missing (B-rev)."""
    turns = result.transcript.turns
    thresholds = _timeline_thresholds(result.transcript.metadata)
    dead_air_ms, ttfa_fail_ms = thresholds["dead_air_ms"], thresholds["ttfa_fail_ms"]
    starts = [_timeline_value(turn.timestamp_ms) for turn in turns]
    spans = [_timeline_span(turn) for turn in turns]
    times = [value for span in spans if span is not None for value in span]
    total = max(times + [start for start in starts if start is not None] + [1.0])
    plot = _TL_WIDTH - _TL_LABEL - _TL_PAD

    def x(ms: float) -> float:
        return _TL_LABEL + max(0.0, min(float(ms), total)) / total * plot

    def rect(css: str, start: float, end: float, y: int, title: str) -> str:
        left = x(start)
        width = max(1.0, x(end) - left)
        return (
            f'<rect class="{css}" x="{left:.1f}" y="{y}" width="{width:.1f}" height="{_TL_ROW_H}">'
            f"<title>{_e(title)}</title></rect>"
        )

    parts = [
        f'<svg class="timeline" viewBox="0 0 {_TL_WIDTH} {_TL_HEIGHT}" width="{_TL_WIDTH}" height="{_TL_HEIGHT}" '
        'role="img" aria-label="Call timeline">'
    ]
    for label, y in _TL_ROWS.values():
        parts.append(
            f'<text x="0" y="{y + _TL_ROW_H - 4}">{_e(label)}</text>'
            f'<rect class="track" x="{_TL_LABEL}" y="{y}" width="{plot}" height="{_TL_ROW_H}"></rect>'
        )
    previous_user_end: Optional[float] = None   # the previous turn's end, when it is a user turn
    for number, (turn, start, span) in enumerate(zip(turns, starts, spans), start=1):
        row = _TL_ROWS.get(turn.speaker)
        latency = _timeline_value(turn.latency_ms)
        if (
            turn.speaker == "agent"
            and latency is not None
            and latency > dead_air_ms
            and previous_user_end is not None
            and start is not None
            and start > previous_user_end
        ):
            parts.append(
                rect("dead-air", previous_user_end, start, _TL_ROWS["agent"][1],
                     f"dead air before turn {number}: {_ms(latency)} ms")
            )
        if row is not None and span is not None:
            label, y = row
            text = f"{_ms(span[0])}–{_ms(span[1])} ms"
            css, title = label, f"turn {number}, {label}, {text}"
            if turn.speaker == "agent" and latency is not None and latency > ttfa_fail_ms:
                css, title = f"{label} slow", f"slow reply, turn {number}: {_ms(latency)} ms ({text})"
            parts.append(rect(css, span[0], span[1], y, title))
        previous_user_end = _timeline_value(turn.end_ms) if turn.speaker == "user" else None
    parts.append(
        f'<text x="{_TL_LABEL}" y="{_TL_AXIS_Y}">0 s</text>'
        f'<text x="{_TL_WIDTH - _TL_PAD}" y="{_TL_AXIS_Y}" text-anchor="end">{_num(total / 1000, 1)} s</text>'
    )
    legend = [
        ("caller", "caller"),
        ("agent", "agent"),
        ("agent slow", f"slow reply (latency > {_ms(ttfa_fail_ms)} ms)"),
        ("dead-air", f"dead air (> {_ms(dead_air_ms)} ms)"),
    ]
    offset = _TL_LABEL
    for css, text in legend:   # legend swatches carry no <title>
        parts.append(
            f'<rect class="{css}" x="{offset}" y="{_TL_LEGEND_Y - 9}" width="10" height="10"></rect>'
            f'<text x="{offset + 14}" y="{_TL_LEGEND_Y}">{_e(text)}</text>'
        )
        offset += 28 + 6 * len(text)
    parts.append("</svg>")
    return "".join(parts)


def _inject_events_block(result: ScenarioResult) -> str:
    rows = [
        [
            _e(event.at_turn),
            _code(event.type),
            _e(event.tool),
            _e(event.error_message),
            _e(event.silence_ms),
            _e(event.note),
        ]
        for event in result.inject_events
    ]
    return f'<div class="note">{_e(INJECT_EVENTS_NOTE)}</div>' + _table(
        ["at_turn", "type", "tool", "error_message", "silence_ms", "note"], rows, numeric=[0, 4]
    )


def _scenario_details(index: int, result: ScenarioResult, audio: _AudioPlayers) -> str:
    transcript = result.transcript
    where = _e(transcript.language) + (f" / {_e(transcript.accent)}" if transcript.accent else "")
    duration = _num(transcript.duration_ms / 1000, 1) if _is_number(transcript.duration_ms) else _NOT_AVAILABLE
    summary = (
        f"{_status_badge(result)} <strong>{_e(result.scenario_name)}</strong> "
        f'<span class="muted">&middot; {where} &middot; {len(transcript.turns)} turns &middot; {duration} s</span>'
    )
    noise_badge = _noise_badge(result)
    if noise_badge:
        summary += " " + noise_badge
    # Failed scenarios open expanded, and so do scenarios with inject_events, so the note is visible.
    open_attr = "" if result.passed and not result.inject_events else " open"
    parts = [
        f'<details class="scenario" id="{_scenario_anchor(index)}"{open_attr}><summary>{summary}</summary>',
        '<div class="scenario-body">',
    ]
    if transcript.recording_path:   # B1: the full call recording first
        parts.append(f'<div class="recording">{_e(RECORDING_LABEL)}</div>')
        parts.append(audio.player(transcript.recording_path))
    if result.inject_events:
        parts.append(_inject_events_block(result))
    if result.error is not None:
        parts.append(f'<div class="error-box">Error: {_e(result.error)}</div>')
    if result.assertion_results:
        rows = [
            [
                _code(score.rule_id),
                _e(score.severity),
                _result_badge(score),
                _e(score.reason),
                _check_turn_cell(index, score.turn_index, score.result == "fail", transcript.turns),
                _num(score.confidence),
                _e(score.regulation_ref),
            ]
            for score in result.assertion_results
        ]
        parts.append("<h3>Assertions</h3>")
        parts.append(
            _table(["rule_id", "severity", "result", "reason", "turn", "confidence", "regulation_ref"], rows, [4, 5])
        )
        parts.append(_disclaimers(result.assertion_results))
    if result.metric_results:
        rows = [
            [
                _code(metric.metric_id),
                _e(metric.name),
                _metric_label(metric),
                _num(metric.score),
                _num(metric.raw_value),
                _check_turn_cell(index, metric.turn_index, _is_poor(metric), transcript.turns),
                _e(metric.reason),
            ]
            for metric in result.metric_results
        ]
        parts.append("<h3>Metrics</h3>")
        parts.append(_table(["metric_id", "name", "label", "score", "raw_value", "turn", "reason"], rows, [3, 4, 5]))
    parts.append("<h3>Transcript</h3>")
    if _has_timeline(result):
        parts.append(_timeline(result))
    if transcript.turns:
        failing = _failing_turns(result)
        clean_paths = _clean_audio_paths(result) if _noise_level(result) is not None else {}
        items = "".join(
            _turn_item(index, number, turn, failing.get(number, []), audio, clean_paths.get(number))
            for number, turn in enumerate(transcript.turns, start=1)
        )
        parts.append(f'<ol class="turns">{items}</ol>')
    else:
        parts.append('<p class="muted">No turns were recorded.</p>')
    parts.append("</div></details>")
    return "".join(parts)


def _transcripts_section(run: RunResult, audio: _AudioPlayers) -> str:
    parts = ['<section id="transcripts"><h2>Scenario transcripts</h2>']
    if any(result.transcript.recording_path or _has_timeline(result) for result in run.scenario_results):
        parts.append(_AUDIO_STYLE)
    if not run.scenario_results:
        parts.append('<p class="muted">No scenarios.</p>')
    parts.extend(
        _scenario_details(index, result, audio) for index, result in enumerate(run.scenario_results, start=1)
    )
    parts.append("</section>")
    return "".join(parts)


def _tool_trace_section(run: RunResult) -> str:
    rows = []
    for index, result in enumerate(run.scenario_results, start=1):
        for number, turn in enumerate(result.transcript.turns, start=1):
            for call in turn.tool_calls:
                rows.append(
                    [
                        f'<a href="#{_scenario_anchor(index)}">{_e(result.scenario_name)}</a>',
                        _turn_ref(index, number),
                        _code(call.name),
                        _code(_json_text(call.arguments)),
                        _success_badge(call.success),
                        # A JSON error redacted like a result, every error cut at 300 characters (R1-M2).
                        _e(tool_error_text(call.error)),
                        # Sensitive keys redacted (A1.3): the report is meant to be shared.
                        _code(_json_text(redact_tool_result(call.result))) if call.result is not None else "",
                    ]
                )
    parts = ['<section id="tool-calls"><h2>Tool call trace</h2>']
    if rows:
        parts.append(
            _table(["scenario_name", "turn", "tool", "arguments", "success", "error", "result"], rows, numeric=[1])
        )
    else:
        parts.append('<p class="muted">No tool calls were recorded.</p>')
    parts.append("</section>")
    return "".join(parts)


def render_html(run: RunResult, embed_audio: bool = True, run_dir: Optional[PathArg] = None) -> str:
    """The HTML report of ``run`` as one self-contained document (SPEC.md section 16).

    ``run_dir`` is the run folder that ``Turn.audio_path`` is relative to (default: the current
    directory). With ``embed_audio`` every turn's WAV file and the call recording are embedded as data
    URIs (each file read once), and the size warning is printed (stderr) when the embedded player copies
    in the page (every real copy counted, base64 not included) total more than 50 MB; otherwise the relative
    path is linked. The run is not modified.
    """
    audio = _AudioPlayers(run_dir, embed_audio)
    body = "".join(
        [
            _header(run),
            _critical_section(run),
            _scenario_table(run),
            _cohort_section(run),
            _noise_level_section(run),
            _metric_summary_section(run),
            _latency_section(run),
            _transcripts_section(run, audio),
            _tool_trace_section(run),
        ]
    )
    _warn_embedded_size(audio.embedded_bytes)   # after rendering: every embedded player copy in the page counted
    return _page(f"Impulse report {run.run_id}", body)


# ------------------------------------------------------------------------------------------------
# Compare report (impulse compare)
# ------------------------------------------------------------------------------------------------

_BAR_WIDTH = 200
_BAR_HEIGHT = 14


def _delta_bar(metric_id: str, delta: Any) -> str:
    """An inline SVG bar from the centre line: right (green) for a positive delta, left (red) for a
    negative one; scores are 0..1, so the full half-width is a delta of 1.0."""
    value = float(delta) if _is_number(delta) and math.isfinite(float(delta)) else 0.0
    half = _BAR_WIDTH / 2
    length = min(abs(value), 1.0) * half
    x = half if value >= 0 else half - length
    css = "pos" if value >= 0 else "neg"
    return (
        f'<svg class="delta" width="{_BAR_WIDTH}" height="{_BAR_HEIGHT}" viewBox="0 0 {_BAR_WIDTH} {_BAR_HEIGHT}" '
        f'role="img" aria-label="{_e(metric_id)} delta {_signed(delta)}">'
        f'<rect class="{css}" x="{x:.1f}" y="2" width="{length:.1f}" height="{_BAR_HEIGHT - 4}"></rect>'
        f'<line class="axis" x1="{half:.1f}" y1="0" x2="{half:.1f}" y2="{_BAR_HEIGHT}"></line></svg>'
    )


def _diff_table(diffs: Sequence[ScenarioDiff], critical: set[str]) -> str:
    if not diffs:
        return '<p class="muted">None.</p>'
    rows = []
    for diff in diffs:
        name = _e(diff.scenario_name)
        if diff.scenario_name in critical:
            name += " " + _badge("critical regression", "fail")
        deltas = _list(f"{_code(metric_id)} {_signed(delta)}" for metric_id, delta in diff.metric_deltas.items())
        rows.append(
            [
                name,
                _code(diff.status_change),
                _list(_e(change) for change in diff.assertion_changes),
                deltas,
                _list(_e(note) for note in diff.notes),
            ]
        )
    return _table(["scenario_name", "status_change", "assertion_changes", "metric_deltas", "notes"], rows)


def render_compare_html(compare: CompareResult, *, notes: Iterable[str] = ()) -> str:
    """The ``impulse compare`` HTML report (SPEC.md section 16): a critical regression banner, a "High
    regressions: {names}" banner when ``high_regressions`` is non-empty, the ``notes`` (e.g.
    ``compare.dry_run_notes``) above the tables, the regressions, improvements, and newly failing
    checks (fail -> fail, ``check_regressions``) tables, added / removed scenarios, and one inline-SVG delta bar per metric of
    ``metric_summary_deltas``. Every text is escaped; nothing is loaded from outside."""
    critical = set(compare.critical_regressions)
    pass_rate_delta = compare.overall_pass_rate_delta
    if _is_number(pass_rate_delta):
        pass_rate_delta = pass_rate_delta * 100
    meta_rows = [
        ("run A", f"{_code(compare.run_id_a)} {_e(compare.timestamp_a)}"),
        ("run B", f"{_code(compare.run_id_b)} {_e(compare.timestamp_b)}"),
        ("overall pass rate delta", f"{_signed(pass_rate_delta, 1)} percentage points"),
    ]
    meta = "".join(f"<dt>{_e(label)}</dt><dd>{value}</dd>" for label, value in meta_rows)
    parts = [f'<header><h1>Impulse compare</h1><dl class="meta">{meta}</dl></header>']
    if compare.critical_regressions:
        names = ", ".join(_e(name) for name in compare.critical_regressions)
        parts.append(
            f'<div class="banner critical">Critical regressions: a critical assertion newly failed, or the '
            f"scenario newly errored, in {names}.</div>"
        )
    else:
        parts.append('<div class="banner ok">No critical regressions.</div>')
    if compare.high_regressions:
        names = ", ".join(_e(name) for name in compare.high_regressions)
        parts.append(f'<div class="banner critical">High regressions: {names}</div>')
    note_items = "".join(f"<li>{_e(note)}</li>" for note in notes)
    if note_items:
        parts.append(f'<ul class="warnings">{note_items}</ul>')
    counts = [
        [
            str(len(compare.regressions)),
            str(len(compare.improvements)),
            _e(compare.unchanged_pass),
            _e(compare.unchanged_fail),
            str(len(compare.added_scenarios)),
            str(len(compare.removed_scenarios)),
        ]
    ]
    parts.append(
        _table(
            ["regressions", "improvements", "unchanged pass", "unchanged fail", "added", "removed"],
            counts,
            numeric=[0, 1, 2, 3, 4, 5],
        )
    )
    parts.append('<section id="regressions"><h2>Regressions (pass -&gt; fail)</h2>')
    parts.append(_diff_table(compare.regressions, critical))
    parts.append("</section>")
    parts.append('<section id="improvements"><h2>Improvements (fail -&gt; pass)</h2>')
    parts.append(_diff_table(compare.improvements, critical))
    parts.append("</section>")
    parts.append('<section id="check-regressions"><h2>Newly failing checks (fail -&gt; fail)</h2>')
    parts.append(_diff_table(compare.check_regressions, critical))
    parts.append("</section>")
    parts.append('<section id="critical-regressions"><h2>Critical regressions</h2>')
    if compare.critical_regressions:
        parts.append(_list(_e(name) for name in compare.critical_regressions))
    else:
        parts.append('<p class="muted">None.</p>')
    parts.append("</section>")
    parts.append('<section id="added-removed"><h2>Added and removed scenarios</h2>')
    parts.append(
        _table(
            ["added (in B only)", "removed (in A only)"],
            [[_list(_e(name) for name in compare.added_scenarios), _list(_e(name) for name in compare.removed_scenarios)]],
        )
    )
    parts.append("</section>")
    parts.append('<section id="metric-deltas"><h2>Metric summary deltas (B - A)</h2>')
    if compare.metric_summary_deltas:
        rows = [
            [_code(metric_id), _signed(delta), _delta_bar(metric_id, delta)]
            for metric_id, delta in compare.metric_summary_deltas.items()
        ]
        parts.append(_table(["metric_id", "delta", "change"], rows, numeric=[1]))
    else:
        parts.append('<p class="muted">No metric is in both runs\' metric summaries.</p>')
    parts.append("</section>")
    return _page(f"Impulse compare {compare.run_id_a} vs {compare.run_id_b}", "".join(parts))
