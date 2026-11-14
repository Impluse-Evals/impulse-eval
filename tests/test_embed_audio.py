# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 7: audio in the HTML report (SPEC.md section 16; reporter.render_html).

``render_html(run, embed_audio=True, run_dir)`` embeds each WAV file named by ``Turn.audio_path``
(relative to ``run_dir``) as ``data:audio/wav;base64,...``; ``embed_audio=False`` links it with the
relative path. When the embedded players total more than 50 MB of base64 text (fixes round 4, O4), ``render_html``
prints "Embedded audio is {n} MB; the HTML report will be large. Use --no-embed-audio to link the
files instead." The CLI flags (``impulse run`` / ``impulse report``) are tested in
``test_cli_exit_codes.py`` (phase 8).

The size test uses a real 52 MiB WAV file in ``tmp_path`` (its data is a sparse, zero-filled
extension of a valid header), because no test-facing hook exists to fake file sizes.
"""

import base64
import io
import math
import re
import struct
import wave

import pytest

SIZE_WARNING = re.compile(
    r"Embedded audio is \d+(?:\.\d+)? MB; the HTML report will be large\. "
    r"Use --no-embed-audio to link the files instead\."
)
DATA_URI = "data:audio/wav;base64,"


def _render(run, run_dir, embed_audio):
    from impulse_evals.reporter import render_html

    return render_html(run, embed_audio=embed_audio, run_dir=run_dir)


def _tone_wav(frequency_hz, seconds=0.1, sample_rate=16000):
    count = int(seconds * sample_rate)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * frequency_hz * i / sample_rate))) for i in range(count)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def _write_large_wav(path, size):
    """A valid 16 kHz mono WAV header whose data chunk is zero-filled up to ``size`` bytes."""
    data_len = size - 44
    header = (
        b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data" + struct.pack("<I", data_len)
    )
    with open(path, "wb") as handle:
        handle.write(header)
        handle.truncate(size)


def _voice_run(run_builder, run_dir, files):
    """One voice-mode scenario; ``files`` maps agent turn audio paths (relative) to WAV bytes or a size."""
    b = run_builder
    (run_dir / "audio").mkdir(parents=True, exist_ok=True)
    turns, timestamp = [], 0
    for index, (relative, content) in enumerate(files.items()):
        path = run_dir / relative
        if isinstance(content, int):
            _write_large_wav(path, content)
        else:
            path.write_bytes(content)
        turns.append(b.turn("user", f"user turn {index + 1}", timestamp))
        turns.append(b.turn("agent", f"agent turn {index + 1}", timestamp + 900, latency_ms=900.0, audio_path=relative))
        timestamp += 4000
    result = b.scenario_result("clinic_booking", transcript=b.transcript("clinic_booking", turns, voice_mode=True))
    return b.run([result], voice_mode=True)


def _console(capsys, normalize_console):
    captured = capsys.readouterr()
    return normalize_console(captured.out + "\n" + captured.err)


def test_embed_audio_produces_data_uris(run_builder, tmp_path):
    first, second = _tone_wav(440), _tone_wav(660)
    run = _voice_run(
        run_builder, tmp_path,
        {"audio/clinic_booking_t2_agent.wav": first, "audio/clinic_booking_t4_agent.wav": second},
    )
    html = _render(run, tmp_path, embed_audio=True)
    assert html.count("<audio") >= 2
    assert DATA_URI + base64.b64encode(first).decode("ascii") in html
    assert DATA_URI + base64.b64encode(second).decode("ascii") in html
    assert not re.search(r"""src\s*=\s*["']audio/""", html)


def test_no_embed_audio_produces_relative_src(run_builder, tmp_path):
    run = _voice_run(
        run_builder, tmp_path,
        {"audio/clinic_booking_t2_agent.wav": _tone_wav(440), "audio/clinic_booking_t4_agent.wav": _tone_wav(660)},
    )
    html = _render(run, tmp_path, embed_audio=False)
    assert re.search(r"""src\s*=\s*["']audio/clinic_booking_t2_agent\.wav["']""", html)
    assert re.search(r"""src\s*=\s*["']audio/clinic_booking_t4_agent\.wav["']""", html)
    assert DATA_URI not in html


def test_size_warning_when_embedded_audio_exceeds_50_mb(run_builder, tmp_path, capsys, normalize_console):
    run = _voice_run(run_builder, tmp_path, {"audio/clinic_booking_t2_agent.wav": 52 * 1024 * 1024})
    html = _render(run, tmp_path, embed_audio=True)
    assert DATA_URI in html
    assert SIZE_WARNING.search(_console(capsys, normalize_console))


def test_no_size_warning_for_small_audio(run_builder, tmp_path, capsys, normalize_console):
    run = _voice_run(run_builder, tmp_path, {"audio/clinic_booking_t2_agent.wav": _tone_wav(440)})
    _render(run, tmp_path, embed_audio=True)
    assert "Embedded audio is" not in _console(capsys, normalize_console)


def test_no_size_warning_when_audio_is_linked(run_builder, tmp_path, capsys, normalize_console):
    run = _voice_run(run_builder, tmp_path, {"audio/clinic_booking_t2_agent.wav": 52 * 1024 * 1024})
    html = _render(run, tmp_path, embed_audio=False)
    assert DATA_URI not in html
    assert "Embedded audio is" not in _console(capsys, normalize_console)


@pytest.mark.parametrize("embed_audio", [True, False])
def test_text_mode_run_has_no_audio_player(run_builder, tmp_path, embed_audio):
    run = run_builder.run([run_builder.scenario_result("text_call", metrics=[("UNIV-013", 1.0)])])
    html = _render(run, tmp_path, embed_audio=embed_audio)
    assert DATA_URI not in html
    assert "<audio" not in html


# ---------------------------------------------------------------------------------------------
# Audit batch 3 (AUDIT.md finding 38): --no-embed-audio links a path only when, with slashes
# normalised, it is audio/<name>.wav (no "..", not absolute; the file need not exist). Anything
# else gets the same note embed mode shows for a path it refuses, and no src attribute.
# ---------------------------------------------------------------------------------------------

AUDIO_SRC = re.compile(r"""<audio[^>]*\ssrc\s*=\s*["']([^"']*)["']""")
TAG_TEXT = re.compile(r">([^<>]*)<")

# Paths the run folder's audio/ rule refuses (embed mode never embeds them either).
REFUSED_AUDIO_PATHS = [
    "../secret.wav",
    "../../.impulse/config.toml",
    "audio/../../other_run/audio/x.wav",
    "audio/../secret.wav",
    "audio\\..\\..\\secret.wav",
    "notes.txt",
    "audio/notes.txt",
    "audio/clinic_booking_t2_agent.mp3",
    "clinic_booking_t2_agent.wav",
    "/etc/secret.wav",
]


def _run_with_audio_path(run_builder, audio_path):
    """One voice-mode scenario whose agent turn names ``audio_path`` (no file is written)."""
    b = run_builder
    turns = [
        b.turn("user", "Hello, I need an appointment.", 0),
        b.turn("agent", "Sure, which day?", 900, latency_ms=900.0, audio_path=audio_path),
    ]
    result = b.scenario_result("clinic_booking", transcript=b.transcript("clinic_booking", turns, voice_mode=True))
    return b.run([result], voice_mode=True)


def _texts_naming(html_text, audio_path):
    """Every text node of the page that contains ``audio_path`` (HTML-escaped as the report escapes it)."""
    import html as html_module

    escaped = html_module.escape(audio_path, quote=True)
    return [text for text in TAG_TEXT.findall(html_text) if escaped in text]


@pytest.mark.parametrize("audio_path", REFUSED_AUDIO_PATHS)
def test_no_embed_audio_links_only_audio_folder_wav_files(run_builder, tmp_path, audio_path):
    (tmp_path / "audio").mkdir()
    html = _render(_run_with_audio_path(run_builder, audio_path), tmp_path, embed_audio=False)
    assert AUDIO_SRC.findall(html) == [], f"{audio_path!r} must not be linked"


def _shown(audio_path):
    """What a note shows for ``audio_path``: an absolute path is cut to its file name (fixes round 4, L25:
    a report never carries an absolute path; superseded: the note named the whole path)."""
    return audio_path.rsplit("/", 1)[-1] if audio_path.startswith("/") else audio_path


@pytest.mark.parametrize("audio_path", REFUSED_AUDIO_PATHS)
def test_no_embed_audio_shows_the_embed_mode_note_for_a_refused_path(run_builder, tmp_path, audio_path):
    (tmp_path / "audio").mkdir()
    run = _run_with_audio_path(run_builder, audio_path)
    embed_notes = _texts_naming(_render(run, tmp_path, embed_audio=True), _shown(audio_path))
    assert embed_notes, "embed mode shows a note naming the refused path"
    linked = _render(run, tmp_path, embed_audio=False)
    for note in embed_notes:
        assert note in linked, f"--no-embed-audio must show the same note as embed mode: {note!r}"


@pytest.mark.parametrize(
    "audio_path, expected",
    [
        pytest.param("audio/clinic_booking_t2_agent.wav", "audio/clinic_booking_t2_agent.wav", id="slash"),
        pytest.param("audio\\clinic_booking_t2_agent.wav", "audio/clinic_booking_t2_agent.wav", id="backslash"),
        pytest.param("audio/missing_t4_agent.wav", "audio/missing_t4_agent.wav", id="missing_file"),
        pytest.param("audio/क्लिनिक_बुकिंग_t2_agent.wav", "audio/क्लिनिक_बुकिंग_t2_agent.wav", id="devanagari"),
    ],
)
def test_no_embed_audio_still_links_audio_folder_wav_files(run_builder, tmp_path, audio_path, expected):
    # Regression guard (passes before the fix too): audio/<name>.wav is linked whether or not the
    # file exists (link mode reads no file); "\" is read as "/"; a Unicode name (scenario file stems
    # keep Devanagari letters, finding 40) is linked too (percent-encoding allowed).
    import urllib.parse

    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "clinic_booking_t2_agent.wav").write_bytes(_tone_wav(440))
    html = _render(_run_with_audio_path(run_builder, audio_path), tmp_path, embed_audio=False)
    srcs = AUDIO_SRC.findall(html)
    assert [urllib.parse.unquote(src) for src in srcs] == [expected]
