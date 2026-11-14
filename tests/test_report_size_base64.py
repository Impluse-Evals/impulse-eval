# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 4, O4 (FINAL-TEST.md): the embedded-audio size warning counts the base64 text on the page.

``render_html(..., embed_audio=True)`` embeds WAV files as ``data:audio/wav;base64,...``; base64 is a third
larger than the file. The warning "Embedded audio is {n} MB; the HTML report will be large. Use
--no-embed-audio to link the files instead." is printed when the base64 characters of every embedded
player copy (counted once per copy written, as before) exceed 50 MiB; ``{n}`` is that size in MiB, one
decimal. Before, raw bytes were counted: a 58 MiB page printed nothing (live run 56187b).

Large files are sparse, zero-filled extensions of a valid WAV header (as ``test_embed_audio.py``).
"""

import base64
import re
import struct

import pytest

SIZE_WARNING = re.compile(
    r"Embedded audio is (\d+(?:\.\d+)?) MB; the HTML report will be large\. "
    r"Use --no-embed-audio to link the files instead\."
)
MIB = 1024 * 1024


def _write_large_wav(path, size):
    data_len = size - 44
    header = (
        b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data" + struct.pack("<I", data_len)
    )
    with open(path, "wb") as handle:
        handle.write(header)
        handle.truncate(size)


def _voice_run(run_builder, run_dir, files, *, failed_turn=None):
    b = run_builder
    (run_dir / "audio").mkdir(parents=True, exist_ok=True)
    turns, timestamp = [], 0
    for index, (relative, size) in enumerate(files.items()):
        _write_large_wav(run_dir / relative, size)
        turns.append(b.turn("user", f"user turn {index + 1}", timestamp))
        turns.append(b.turn("agent", f"agent turn {index + 1}", timestamp + 500, latency_ms=500.0,
                            audio_path=relative, end_ms=timestamp + 1500))
        timestamp += 2000
    transcript = b.transcript("clinic_booking", turns, voice_mode=True)
    metrics = []
    if failed_turn is not None:
        metrics.append(b.metric("VOICE-005", 0.0, turn_index=failed_turn))
    return b.run([b.scenario_result("clinic_booking", transcript=transcript, metrics=metrics)], voice_mode=True)


def _console(capsys, normalize_console):
    captured = capsys.readouterr()
    return normalize_console(captured.out + captured.err)


def _render(run, run_dir):
    from impulse_evals.reporter import render_html

    return render_html(run, embed_audio=True, run_dir=run_dir)


def _base64_len(size):
    return len(base64.b64encode(b"\0" * size))


def test_warning_when_the_base64_text_exceeds_50_mib_although_the_files_do_not(run_builder, tmp_path, capsys, normalize_console):
    size = 40 * MIB   # 40 MiB of WAV is 53.3 MiB of base64
    run = _voice_run(run_builder, tmp_path, {"audio/clinic_booking_t2_agent.wav": size})
    html = _render(run, tmp_path)
    assert html.count("data:audio/wav;base64,") == 1
    match = SIZE_WARNING.search(_console(capsys, normalize_console))
    assert match, "no size warning for a 53 MiB page"
    assert match.group(1) == f"{_base64_len(size) / MIB:.1f}"


def test_no_warning_when_the_base64_text_stays_under_50_mib(run_builder, tmp_path, capsys, normalize_console):
    run = _voice_run(run_builder, tmp_path, {"audio/clinic_booking_t2_agent.wav": 30 * MIB})   # 40 MiB of base64
    _render(run, tmp_path)
    assert "Embedded audio is" not in _console(capsys, normalize_console)


def test_every_embedded_copy_counts_its_base64_size(run_builder, tmp_path, capsys, normalize_console):
    """The same file embedded twice (turn player and the recording slot are separate files here; a second
    copy comes from a failed check that links the turn, which embeds nothing extra: the copies counted are
    the players written). Two files of 20 MiB each: 26.7 + 26.7 = 53.3 MiB of base64."""
    files = {"audio/clinic_booking_t2_agent.wav": 20 * MIB, "audio/clinic_booking_t4_agent.wav": 20 * MIB}
    run = _voice_run(run_builder, tmp_path, files, failed_turn=2)
    html = _render(run, tmp_path)
    assert html.count("data:audio/wav;base64,") == 2
    match = SIZE_WARNING.search(_console(capsys, normalize_console))
    assert match
    assert match.group(1) == f"{2 * _base64_len(20 * MIB) / MIB:.1f}"


@pytest.mark.parametrize("embed", [True, False])
def test_embedded_bytes_attribute_counts_base64_characters(run_builder, tmp_path, embed):
    """``_AudioPlayers.embedded_bytes`` (the number the warning uses) is the base64 length per copy."""
    from impulse_evals.reporter import _AudioPlayers

    path = tmp_path / "audio" / "x.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_large_wav(path, 3000)
    players = _AudioPlayers(tmp_path, embed=embed)
    players.player("audio/x.wav")
    players.player("audio/x.wav")
    assert players.embedded_bytes == (2 * _base64_len(3000) if embed else 0)
