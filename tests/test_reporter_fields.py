# SPDX-License-Identifier: Apache-2.0
"""Phase 7: JSON and HTML reports (SPEC.md section 16; reporter.write_json / reporter.render_html).

``write_json(run, path)`` writes ``RunResult.model_dump_json(indent=2)`` to the file ``path``.
``render_html(run, embed_audio, run_dir)`` returns one self-contained HTML document: every
transcript, reason, and config text is HTML-escaped, no ``src`` / ``href`` / ``url(`` value starts
with ``http:``, ``https:``, or ``//``, the header shows ``RunResult.warnings``, scenarios with
``inject_events`` show the exact v2 note, and the per-service latency table (heading
"Per-service latency") appears only when a transcript has ``metadata["pipecat_metrics"]``.
"""

import base64
import json
import re

import pytest

NOTE = "inject_events defined but execution reserved for v2; this scenario ran without injection."
EXTERNAL_ATTR = re.compile(r"""\b(?:src|href)\s*=\s*["']?\s*(?:https?:|//)""", re.IGNORECASE)
EXTERNAL_CSS = re.compile(r"""url\(\s*["']?\s*(?:https?:|//)""", re.IGNORECASE)
PER_SERVICE_HEADING = "Per-service latency"


def _render(run, run_dir, embed_audio=False):
    from impulse_evals.reporter import render_html

    html = render_html(run, embed_audio=embed_audio, run_dir=run_dir)
    assert isinstance(html, str)
    return html


def _payload(n):
    return f"<script>alert({n})</script>"


def _rich_run(run_builder):
    """A run whose text fields carry script payloads, URLs, tool calls, warnings, and inject_events."""
    b = run_builder
    booking_turns = [
        b.turn("user", f"Hi {_payload(1)} see https://example.com/pricing for prices", 0),
        b.turn(
            "agent",
            "Your slot is booked.",
            900,
            latency_ms=900.0,
            tool_calls=[
                {
                    "name": "book_slot",
                    "arguments": {"date": "2026-01-01", "note": _payload(6)},
                    "success": False,
                    "error": "slot taken",
                }
            ],
        ),
    ]
    booking = b.scenario_result(
        "clinic_booking",
        transcript=b.transcript("clinic_booking", booking_turns, language="en-IN"),
        assertions=[("dpdp.consent_before_pii", "fail", {"reason": f"asked for the phone first {_payload(3)}"})],
        metrics=[
            ("UNIV-001", 0.0, {"reason": f"not booked {_payload(2)} //cdn.example.com/x.js"}),
            ("VOICE-001", 1.0, {"is_proxy": True, "reason": "median latency 900 ms"}),
        ],
        inject_events=[{"at_turn": 1, "type": "tool_error", "tool": "book_slot", "error_message": "down"}],
    )
    hindi = b.scenario_result(
        f"hindi_call {_payload(7)}",
        transcript=b.transcript("hindi_call", language="hi-IN", accent="north-indian"),
        metrics=[("UNIV-013", 1.0)],
        inject_events=[{"at_turn": 2, "type": "silence", "silence_ms": 2000}],
    )
    plain = b.scenario_result(
        "plain_call",
        metrics=[("UNIV-013", 1.0), ("UNIV-003", 0.0, {"skipped": True, "reason": "fewer than 4 turns"})],
    )
    return b.run(
        [booking, hindi, plain],
        warnings=[
            "Pipecat bots serve one client at a time; running scenarios one by one.",
            f"Unknown domain insurance; using general defaults. {_payload(4)}",
        ],
        agent_entrypoint=f"clinic_agent:respond{_payload(5)}",
        agent_domain="healthcare",
        judge_model="ollama/judge-primary",
        metric_summary={"VOICE-001": b.metric("VOICE-001", 1.0, is_proxy=True, reason="mean over 1 scenarios (0 skipped)")},
    )


# ============================================================================================
# JSON
# ============================================================================================


def test_write_json_uses_exact_field_names(run_builder, tmp_path):
    from impulse_evals.models import RunResult
    from impulse_evals.reporter import write_json

    run = _rich_run(run_builder)
    path = tmp_path / "run.json"
    write_json(run, path)
    text = path.read_text(encoding="utf-8")
    assert text.rstrip("\n") == run.model_dump_json(indent=2)
    data = json.loads(text)
    assert data["scenario_results"][0]["scenario_name"] == "clinic_booking"
    assert set(data["latency_profile"]) >= {"p50_ms", "p75_ms", "p95_ms", "p99_ms", "all_latencies_ms"}
    assert data["warnings"] == run.warnings
    assert data["scenario_results"][0]["inject_events"][0]["type"] == "tool_error"
    assert "scenario_id" not in text
    assert RunResult.model_validate_json(text) == run


# ============================================================================================
# HTML: escaping and external URLs
# ============================================================================================


def test_html_escapes_script_everywhere(run_builder, tmp_path):
    html = _render(_rich_run(run_builder), tmp_path)
    for n in range(1, 8):
        assert _payload(n) not in html, f"payload {n} is not HTML-escaped"
    assert "alert(1)" in html  # the transcript text is shown, escaped


def test_html_has_no_external_urls(run_builder, tmp_path):
    html = _render(_rich_run(run_builder), tmp_path)
    assert not EXTERNAL_ATTR.search(html), EXTERNAL_ATTR.search(html).group(0)
    assert not EXTERNAL_CSS.search(html), EXTERNAL_CSS.search(html).group(0)
    # A URL said in the transcript is text content, which is allowed.
    assert "https://example.com/pricing" in html


def test_html_is_one_document(run_builder, tmp_path):
    html = _render(_rich_run(run_builder), tmp_path)
    assert "<html" in html.lower()
    assert "</html>" in html.lower()
    assert '<link rel="stylesheet"' not in html.lower()


# ============================================================================================
# HTML: header, sections, warnings, inject_events note
# ============================================================================================


def test_header_shows_run_fields(run_builder, tmp_path):
    html = _render(_rich_run(run_builder), tmp_path)
    assert "ollama/judge-primary" in html
    assert "healthcare" in html
    assert "en-IN" in html
    assert "hi-IN" in html
    assert "clinic_agent:respond" in html


def test_header_warnings_rendered_from_run_warnings(run_builder, tmp_path):
    html = _render(_rich_run(run_builder), tmp_path)
    assert "Pipecat bots serve one client at a time; running scenarios one by one." in html
    assert "Unknown domain insurance; using general defaults." in html


def test_inject_events_note_once_per_scenario_with_events(run_builder, tmp_path):
    html = _render(_rich_run(run_builder), tmp_path)
    assert html.count(NOTE) == 2


def test_no_inject_events_note_without_events(run_builder, tmp_path):
    run = run_builder.run([run_builder.scenario_result("plain_call", metrics=[("UNIV-013", 1.0)])])
    assert NOTE not in _render(run, tmp_path)


def test_failures_tool_trace_and_proxy_label_shown(run_builder, tmp_path):
    html = _render(_rich_run(run_builder), tmp_path)
    assert "dpdp.consent_before_pii" in html  # critical failure
    assert "book_slot" in html  # tool call trace
    assert "slot taken" in html  # tool call error
    assert "clinic_booking" in html
    assert "plain_call" in html
    assert "proxy" in html.lower()  # proxy metrics are labelled
    assert "skipped" in html.lower()  # skipped metrics are labelled, with their reason
    assert "fewer than 4 turns" in html


# ============================================================================================
# HTML: audio players (embedded or linked)
# ============================================================================================


def _voice_run(run_builder, run_dir, wav):
    b = run_builder
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "clinic_booking_t2_agent.wav").write_bytes(wav)
    turns = [
        b.turn("user", "Hello, I need an appointment.", 0),
        b.turn("agent", "Sure, which day?", 900, latency_ms=900.0, audio_path="audio/clinic_booking_t2_agent.wav"),
    ]
    result = b.scenario_result("clinic_booking", transcript=b.transcript("clinic_booking", turns, voice_mode=True))
    return b.run([result], voice_mode=True)


def test_embedded_audio_uses_data_uri(run_builder, tmp_path):
    wav = sine_wav_bytes()
    html = _render(_voice_run(run_builder, tmp_path, wav), tmp_path, embed_audio=True)
    assert "<audio" in html
    assert "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii") in html


def test_linked_audio_uses_relative_src(run_builder, tmp_path):
    html = _render(_voice_run(run_builder, tmp_path, sine_wav_bytes()), tmp_path, embed_audio=False)
    assert "<audio" in html
    assert re.search(r"""src\s*=\s*["']audio/clinic_booking_t2_agent\.wav["']""", html)
    assert "data:audio/wav;base64," not in html


def sine_wav_bytes():
    import io
    import math
    import struct
    import wave

    frames = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(1600))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(frames)
    return buffer.getvalue()


# ============================================================================================
# HTML: per-service latency table (Pipecat metrics)
# ============================================================================================


def _pipecat_run(run_builder, with_metrics):
    b = run_builder
    metadata = {}
    if with_metrics:
        metadata["pipecat_metrics"] = [
            {"turn_index": 2, "kind": "ttfb", "processor": "FakeSTTService#0", "model": "fake-stt", "value_ms": 180.0},
            {"turn_index": 2, "kind": "ttfb", "processor": "FakeLLMService#0", "model": "fake-llm", "value_ms": 412.5},
            {"turn_index": 2, "kind": "processing", "processor": "FakeTTSService#0", "model": "fake-tts", "value_ms": 95.0},
        ]
    result = b.scenario_result("pipecat_call", transcript=b.transcript("pipecat_call", metadata=metadata))
    return b.run([result], agent_entrypoint="ws://localhost:7860")


def test_per_service_latency_table_when_pipecat_metrics_present(run_builder, tmp_path):
    html = _render(_pipecat_run(run_builder, with_metrics=True), tmp_path)
    assert PER_SERVICE_HEADING in html
    for processor in ("FakeSTTService#0", "FakeLLMService#0", "FakeTTSService#0"):
        assert processor in html
    assert "412.5" in html


def test_no_per_service_latency_table_without_pipecat_metrics(run_builder, tmp_path):
    html = _render(_pipecat_run(run_builder, with_metrics=False), tmp_path)
    assert PER_SERVICE_HEADING not in html
    assert "FakeLLMService#0" not in html


@pytest.mark.parametrize("embed_audio", [True, False])
def test_render_html_does_not_modify_the_run(run_builder, tmp_path, embed_audio):
    run = _rich_run(run_builder)
    before = run.model_dump_json()
    _render(run, tmp_path, embed_audio=embed_audio)
    assert run.model_dump_json() == before


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 39): reporter.write_json (and write_html, reached through
# `impulse report`) write to a temporary file in the same folder and os.replace() it onto the
# target; on any error the temporary file is removed and an existing target is left untouched.
# ============================================================================================

RUN_ID_OLD = "20261001T120000Z-aaaaaa"
RUN_ID_NEW = "20261002T120000Z-bbbbbb"


def _small_run(run_builder, run_id):
    return run_builder.run([run_builder.scenario_result("clinic_booking", metrics=[("UNIV-013", 1.0)])], run_id=run_id)


def _fail_replace_onto(monkeypatch, target):
    """``os.replace`` raises (permission denied) when its destination is ``target``; otherwise it works."""
    import errno
    import os
    from pathlib import Path

    real_replace = os.replace
    destination = Path(target).resolve()

    def failing_replace(src, dst, *args, **kwargs):
        if Path(dst).resolve() == destination:
            raise OSError(errno.EACCES, "Permission denied (test fault)", str(dst))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)


def _partial_writes_in(monkeypatch, folder):
    """Every path-based write-mode ``open()`` of a file in ``folder`` writes 10 characters, then
    raises ENOSPC (a full disk). ``io.open`` and ``builtins.open`` are both patched, so
    ``Path.write_bytes`` / ``Path.write_text`` / ``open()`` are covered; file-descriptor opens are not."""
    import builtins
    import errno
    import io
    from pathlib import Path

    real_open = io.open
    root = Path(folder).resolve()
    hits = []

    class PartialWrite(io.RawIOBase):
        def __init__(self, handle):
            self._handle = handle

        def writable(self):
            return True

        def write(self, data):
            self._handle.write(data[:10])
            self._handle.flush()
            raise OSError(errno.ENOSPC, "No space left on device (test fault)")

        def close(self):
            self._handle.close()
            super().close()

    def fake_open(file, mode="r", *args, **kwargs):
        try:
            inside = isinstance(file, (str, Path)) and Path(file).resolve().parent == root
        except (OSError, ValueError):
            inside = False
        if inside and any(flag in mode for flag in "wxa+"):
            hits.append((Path(file).name, mode))
            return PartialWrite(real_open(file, mode, *args, **kwargs))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake_open)
    monkeypatch.setattr(builtins, "open", fake_open)
    return hits


def test_write_json_keeps_the_previous_run_json_when_the_final_replace_fails(run_builder, tmp_path, monkeypatch):
    from impulse_evals.reporter import write_json

    path = tmp_path / "run.json"
    path.write_bytes((_small_run(run_builder, RUN_ID_OLD).model_dump_json(indent=2) + "\n").encode("utf-8"))
    before = path.read_bytes()
    _fail_replace_onto(monkeypatch, path)
    raised = None
    try:
        write_json(_small_run(run_builder, RUN_ID_NEW), path)
    except OSError as exc:
        raised = exc
    assert path.read_bytes() == before, "run.json was rewritten in place"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"], "no temporary file is left"
    assert raised is not None, "the failed write raises OSError"


def test_write_json_never_leaves_a_partial_run_json(run_builder, tmp_path, monkeypatch):
    from impulse_evals.models import RunResult
    from impulse_evals.reporter import write_json

    path = tmp_path / "run.json"
    path.write_bytes((_small_run(run_builder, RUN_ID_OLD).model_dump_json(indent=2) + "\n").encode("utf-8"))
    before = path.read_bytes()
    new_run = _small_run(run_builder, RUN_ID_NEW)
    _partial_writes_in(monkeypatch, tmp_path)
    raised = None
    try:
        write_json(new_run, path)
    except OSError as exc:
        raised = exc
    content = path.read_bytes()
    if raised is not None:
        assert content == before, "a failed write left a partial run.json instead of the previous one"
    else:   # the write did not go through a faulted open: it must be complete
        assert RunResult.model_validate_json(content.decode("utf-8")) == new_run
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"], "no temporary file is left"


def test_write_json_still_writes_the_new_run(run_builder, tmp_path):
    # Regression guard (passes before the fix too): a normal write replaces the previous file.
    from impulse_evals.models import RunResult
    from impulse_evals.reporter import write_json

    path = tmp_path / "nested" / "run.json"
    write_json(_small_run(run_builder, RUN_ID_OLD), path)
    new_run = _small_run(run_builder, RUN_ID_NEW)
    write_json(new_run, path)
    assert path.read_text(encoding="utf-8") == new_run.model_dump_json(indent=2) + "\n"
    assert RunResult.model_validate_json(path.read_text(encoding="utf-8")) == new_run
    assert sorted(p.name for p in path.parent.iterdir()) == ["run.json"]


def test_report_command_keeps_the_previous_report_when_the_final_replace_fails(
    impulse_project, run_builder, cli, monkeypatch
):
    run = _small_run(run_builder, RUN_ID_OLD)
    run_json = run_builder.write_run(run, impulse_project.runs_dir)
    report = run_json.parent / "report.html"
    report.write_bytes(b"<!DOCTYPE html>\n<p>previous report (kept)</p>\n")
    before = report.read_bytes()
    _fail_replace_onto(monkeypatch, report)
    result = cli.invoke("report", run.run_id)
    assert report.read_bytes() == before, "report.html was rewritten in place"
    assert sorted(p.name for p in run_json.parent.iterdir()) == ["report.html", "run.json"], "no temporary file is left"
    assert result.exit_code == 2, cli.text(result)


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 40; maintainer decision #40): scenario file names keep Unicode
# letters and digits (str.isalnum()), "-", "_" and combining marks (Mn / Mc, the Devanagari vowel
# signs and virama), after NFC normalisation; every other character becomes "_" as before. Windows
# reserved names and the case-insensitive clash suffixes are unchanged.
# ============================================================================================


def _dry_run_scenario_files(project, cli, names):
    """The ``scenarios/*.json`` file names of an ``impulse run --dry-run`` of scenarios ``names``."""
    project.write_agent()
    config = project.write_config([project.scenario(name, language="hi-IN", max_turns=2) for name in names])
    result = cli.invoke("run", config, "--dry-run", "--reporter", "json")
    assert result.exit_code == 0, cli.text(result)
    run_dir = project.only_run_dir()
    return sorted(path.name for path in (run_dir / "scenarios").iterdir()), project.read_run(run_dir)


def test_devanagari_scenario_names_keep_their_letters_in_file_names(impulse_project, mock_llm, cli):
    names = ["क्लिनिक बुकिंग", "नमस्ते", "Booking बुकिंग 2", "बुकिंग।"]
    files, run = _dry_run_scenario_files(impulse_project, cli, names)
    assert files == sorted(["क्लिनिक_बुकिंग.json", "नमस्ते.json", "Booking_बुकिंग_2.json", "बुकिंग_.json"])
    assert [result.scenario_name for result in run.scenario_results] == names
    assert mock_llm.call_count == 0


def test_scenario_file_name_is_nfc_normalised(impulse_project, mock_llm, cli):
    import unicodedata

    decomposed = "café booking"   # "e" + COMBINING ACUTE ACCENT (NFD)
    files, run = _dry_run_scenario_files(impulse_project, cli, [decomposed])
    assert files == [unicodedata.normalize("NFC", "café_booking") + ".json"]
    assert run.scenario_results[0].scenario_name == decomposed


def test_devanagari_names_that_sanitise_alike_get_the_clash_suffix(impulse_project, mock_llm, cli):
    files, _run = _dry_run_scenario_files(impulse_project, cli, ["क्लिनिक बुकिंग", "क्लिनिक/बुकिंग"])
    assert files == ["क्लिनिक_बुकिंग.json", "क्लिनिक_बुकिंग_2.json"]


def test_reserved_and_ascii_scenario_names_are_unchanged(impulse_project, mock_llm, cli):
    # Regression guard (passes before the fix too): Windows device names get a "_" prefix, ASCII
    # names are sanitised as before, and case-insensitive clashes get "_2".
    files, _run = _dry_run_scenario_files(impulse_project, cli, ["con", "NUL", "clinic booking/v2", "Clinic", "clinic"])
    assert files == sorted(["_con.json", "_NUL.json", "clinic_booking_v2.json", "Clinic.json", "clinic_2.json"])


# ============================================================================================
# Review fix-up of audit batch 3 (group Z), regression of finding 40: Devanagari is 3 UTF-8 bytes
# per character, so a long Unicode scenario name gave a file name over the 255-byte POSIX limit.
# The sanitised stem is cut to at most 200 UTF-8 bytes on a character boundary (a combining mark
# run is never split: the cut may not leave part of a run on each side) BEFORE the reserved-name
# and clash checks; a stem of at most 200 bytes is unchanged. Checked on byte lengths, not on an
# OS error, so the tests are the same on every platform.
# ============================================================================================

STEM_BYTE_LIMIT = 200
_LONG_BASE = "क्लिनिक बुकिंग और भुगतान की पुष्टि "
LONG_DEVANAGARI_NAME = (_LONG_BASE * 5)[:119] + "१"                     # 120 characters, 322 UTF-8 bytes
# 120 characters; a cut at the last character boundary within 200 bytes would leave the vowel sign
# "ि" in the stem and the anusvara "ं" (the rest of the same combining mark run) outside it.
SPLIT_RUN_DEVANAGARI_NAME = "क" * 64 + "किं" + "क" * 53


def _spec_sanitised(name):
    """SPEC.md section 15 sanitising (NFC, then everything but str.isalnum(), "-", "_" and Mn / Mc
    combining marks becomes "_"), before any cut."""
    import unicodedata

    return "".join(
        char if char.isalnum() or char in "-_" or unicodedata.category(char) in ("Mn", "Mc") else "_"
        for char in unicodedata.normalize("NFC", name)
    )


def _assert_cut_stem(stem, name):
    """``stem`` is the sanitised ``name`` cut to at most 200 UTF-8 bytes on a character boundary."""
    import unicodedata

    full = _spec_sanitised(name)
    size = len(stem.encode("utf-8"))
    assert size <= STEM_BYTE_LIMIT, f"the file stem is {size} UTF-8 bytes (limit {STEM_BYTE_LIMIT}): {stem!r}"
    assert size > 150, f"the cut keeps the name up to the limit, not much less ({size} bytes)"
    assert unicodedata.is_normalized("NFC", stem), "the stem is NFC"
    assert full.startswith(stem), "the stem is a prefix of the full sanitised name"

    def is_mark(char):
        return unicodedata.category(char) in ("Mn", "Mc")

    rest = full[len(stem):]
    assert not (stem and rest and is_mark(stem[-1]) and is_mark(rest[0])), (
        "the cut splits a combining mark run between the stem and the dropped rest"
    )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(LONG_DEVANAGARI_NAME, id="devanagari_120_chars"),
        pytest.param(SPLIT_RUN_DEVANAGARI_NAME, id="cut_inside_a_combining_mark_run"),
    ],
)
def test_long_unicode_scenario_name_gives_a_file_stem_of_at_most_200_utf8_bytes(impulse_project, mock_llm, cli, name):
    assert len(name) == 120 and len(_spec_sanitised(name).encode("utf-8")) > 255
    files, run = _dry_run_scenario_files(impulse_project, cli, [name])
    assert len(files) == 1 and files[0].endswith(".json"), files
    stem = files[0][: -len(".json")]
    _assert_cut_stem(stem, name)
    stored = json.loads((impulse_project.only_run_dir() / "scenarios" / files[0]).read_text(encoding="utf-8"))
    assert stored["scenario_name"] == name, "only the file name is cut, never the stored scenario name"
    assert [result.scenario_name for result in run.scenario_results] == [name]
    assert mock_llm.call_count == 0


def test_long_names_that_share_the_first_200_bytes_get_the_clash_suffix(impulse_project, mock_llm, cli):
    first = LONG_DEVANAGARI_NAME
    second = LONG_DEVANAGARI_NAME[:-1] + "२"   # differs only in the last character (far past 200 bytes)
    files, _run = _dry_run_scenario_files(impulse_project, cli, [first, second])
    assert len(files) == 2, files
    stems = sorted((name[: -len(".json")] for name in files), key=len)
    _assert_cut_stem(stems[0], first)
    assert stems[1] == stems[0] + "_2", f"the second name gets the clash suffix on the cut stem: {files}"
    for stem in stems:
        assert len((stem + ".json").encode("utf-8")) <= 255
    scenarios_dir = impulse_project.only_run_dir() / "scenarios"
    stored = {stem: json.loads((scenarios_dir / f"{stem}.json").read_text(encoding="utf-8"))["scenario_name"] for stem in stems}
    assert stored == {stems[0]: first, stems[1]: second}, "config order: the first scenario keeps the unsuffixed stem"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "clinic booking for a returning patient who asks about the fee, the slot and the doctor and then cancels",
            id="ascii_long_name",
        ),
        pytest.param("ab" + "क" * 66, id="devanagari_exactly_200_bytes"),
    ],
)
def test_scenario_file_stem_of_at_most_200_utf8_bytes_is_unchanged(impulse_project, mock_llm, cli, name):
    # Regression guard (passes before the fix too): nothing is cut at or below the limit.
    expected = _spec_sanitised(name)
    assert len(expected.encode("utf-8")) <= STEM_BYTE_LIMIT
    files, _run = _dry_run_scenario_files(impulse_project, cli, [name])
    assert files == [expected + ".json"]


# ============================================================================================
# Review fix-up of audit batch 3 (group Z): the superscript device names COM¹, COM², COM³, LPT¹,
# LPT², LPT³ pass str.isalnum() but are Windows device names too (os.path.isreserved is True on
# Python 3.14), so they get the "_" prefix like COM1 (case-insensitively).
# ============================================================================================

SUPERSCRIPT_DEVICE_NAMES = ["COM¹", "com²", "COM³", "LPT¹", "lpt²", "LPT³"]


def test_superscript_com_and_lpt_device_names_get_the_reserved_name_prefix(impulse_project, mock_llm, cli):
    files, run = _dry_run_scenario_files(impulse_project, cli, SUPERSCRIPT_DEVICE_NAMES)
    assert files == sorted(f"_{name}.json" for name in SUPERSCRIPT_DEVICE_NAMES)
    assert [result.scenario_name for result in run.scenario_results] == SUPERSCRIPT_DEVICE_NAMES


def test_digit_device_names_and_longer_superscript_names_are_unchanged(impulse_project, mock_llm, cli):
    # Regression guard (passes before the fix too).
    files, _run = _dry_run_scenario_files(impulse_project, cli, ["COM1", "lpt9", "COM10", "COM¹ booking"])
    assert files == sorted(["_COM1.json", "_lpt9.json", "COM10.json", "COM¹_booking.json"])
