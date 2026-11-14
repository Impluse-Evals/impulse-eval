# SPDX-License-Identifier: AGPL-3.0-only
"""Noise simulation (EXPERIMENTAL) in the HTML report (SPEC.md section 16).

Read from ``CallTranscript.metadata["impulse_noise"]`` and the per-turn ``impulse_voice`` records, so
older runs and runs without noise render exactly as before (the page holds no "noise" text at all):

- header row "noise simulation" with "experimental" and the levels in the run;
- a badge "noise: {label}, {snr} dB" in the scenario table status cell and the scenario summary line;
- each user turn plays the NOISY audio (``Turn.audio_path``, where "Hear turn N" points); the clean
  audio is in a collapsed ``<details>`` "clean caller audio (before noise)" below it;
- section ``id="noise-levels"`` "Pass rate by noise level" when the run has two or more levels
  ("clean" = a scenario without noise counts as a level): one row per level, clean first, then by
  ``snr_db`` from the mildest (highest) to the harshest, with scenarios, passed, failed, pass rate and
  the mean VOICE-009 score.
"""

import re

import pytest
from conftest import sine_wav


def _noise(profile="pink", snr_db=10.0, label=None, file=None, seed=1):
    return {
        "experimental": True,
        "profile": profile,
        "snr_db": snr_db,
        "label": label if label is not None else profile,
        "file": file,
        "seed": seed,
    }


def _scenario(run_builder, run_dir, name, *, noise=None, passed=True, voice_009=None):
    user_path = f"audio/{name}_t1_user.wav"
    clean_path = f"audio/{name}_t1_user_clean.wav" if noise else None
    agent_path = f"audio/{name}_t2_agent.wav"
    folder = run_dir / "audio"
    folder.mkdir(parents=True, exist_ok=True)
    (run_dir / user_path).write_bytes(sine_wav(0.2, frequency_hz=300.0))
    (run_dir / agent_path).write_bytes(sine_wav(0.2, frequency_hz=500.0))
    record = {"turn_index": 1, "speaker": "user", "audio_ms": 200, "round_trip_text": "hello", "round_trip_confidence": 0.9}
    if noise:
        (run_dir / clean_path).write_bytes(sine_wav(0.2, frequency_hz=700.0))
        record.update({"clean_audio_path": clean_path, "noise_snr_db": noise["snr_db"], "noise_scaled": False,
                       "noise_offset_ms": None})
    metadata = {"impulse_voice": {"turns": [record], "thresholds": {}}}
    if noise:
        metadata["impulse_noise"] = noise
    turns = [
        run_builder.turn("user", "hello", 0, end_ms=200, audio_path=user_path, heard_text="hello"),
        run_builder.turn("agent", "hi there", 900, end_ms=1100, latency_ms=700.0, audio_path=agent_path),
    ]
    transcript = run_builder.transcript(name, turns, voice_mode=True, metadata=metadata)
    metrics = [("UNIV-001", 1.0)]
    if voice_009 is not None:
        metrics.append(("VOICE-009", voice_009))
    if not passed:
        metrics.append(("UNIV-002", 0.1, {"turn_index": 1}))
    return run_builder.scenario_result(name, transcript=transcript, metrics=metrics)


def _render(run, run_dir, embed=False):
    from impulse_evals.reporter import render_html

    return render_html(run, embed_audio=embed, run_dir=run_dir)


def _section(page, section_id):
    match = re.search(rf'<section id="{section_id}">(.*?)</section>', page, re.S)
    return match.group(1) if match else None


def test_run_without_noise_has_no_noise_text(run_builder, tmp_path):
    run = run_builder.run([_scenario(run_builder, tmp_path, "plain")], voice_mode=True)
    page = _render(run, tmp_path)
    assert "noise" not in page.lower()


def test_header_badge_and_summary(run_builder, tmp_path):
    results = [_scenario(run_builder, tmp_path, "street", noise=_noise("file", 5.0, label="traffic", file="traffic.wav"))]
    page = _render(run_builder.run(results, voice_mode=True), tmp_path)
    header = re.search(r"<header>(.*?)</header>", page, re.S).group(1)
    assert "noise simulation" in header
    assert "experimental" in header
    assert "traffic, 5 dB" in header
    table = _section(page, "scenarios")
    assert "noise: traffic, 5 dB" in table
    summary = re.search(r'<details class="scenario"[^>]*><summary>(.*?)</summary>', page, re.S).group(1)
    assert "noise: traffic, 5 dB" in summary


def test_user_turn_plays_noisy_audio_with_clean_version_as_option(run_builder, tmp_path):
    results = [_scenario(run_builder, tmp_path, "office", noise=_noise("pink", 10.0), passed=False)]
    page = _render(run_builder.run(results, voice_mode=True), tmp_path)
    turn_audio = re.search(r'<div class="turn-audio" id="scenario-1-turn-1-audio">(.*?)</div>', page, re.S).group(1)
    assert 'src="audio/office_t1_user.wav"' in turn_audio
    assert "office_t1_user_clean.wav" not in turn_audio
    clean = re.search(r"<details class=\"clean-audio\"><summary>(.*?)</summary>(.*?)</details>", page, re.S)
    assert clean is not None
    assert clean.group(1) == "clean caller audio (before noise)"
    assert 'src="audio/office_t1_user_clean.wav"' in clean.group(2)
    assert page.index('src="audio/office_t1_user.wav"') < page.index('src="audio/office_t1_user_clean.wav"')
    # "Hear turn 1" jumps to the noisy player
    assert 'href="#scenario-1-turn-1-audio">Hear turn 1</a>' in page


def test_clean_player_is_embedded_too(run_builder, tmp_path):
    results = [_scenario(run_builder, tmp_path, "office", noise=_noise("pink", 10.0))]
    page = _render(run_builder.run(results, voice_mode=True), tmp_path, embed=True)
    clean = re.search(r"<details class=\"clean-audio\"><summary>[^<]*</summary>(.*?)</details>", page, re.S).group(1)
    assert 'src="data:audio/wav;base64,' in clean


def test_pass_rate_by_noise_level(run_builder, tmp_path):
    results = [
        _scenario(run_builder, tmp_path, "clean_a", voice_009=1.0),
        _scenario(run_builder, tmp_path, "harsh", noise=_noise("pink", 5.0), passed=False, voice_009=0.2),
        _scenario(run_builder, tmp_path, "mild", noise=_noise("pink", 20.0), voice_009=0.9),
        _scenario(run_builder, tmp_path, "busy_a", noise=_noise("pink", 10.0), voice_009=0.8),
        _scenario(run_builder, tmp_path, "busy_b", noise=_noise("pink", 10.0), passed=False, voice_009=0.4),
    ]
    page = _render(run_builder.run(results, voice_mode=True), tmp_path)
    section = _section(page, "noise-levels")
    assert section is not None
    assert "Pass rate by noise level" in section
    headers = re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", section)   # <th>, not <thead>
    assert headers == ["noise level", "scenarios", "passed", "failed", "pass rate", "mean VOICE-009"]
    rows = [re.findall(r"<td(?:\s[^>]*)?>(.*?)</td>", row) for row in re.findall(r"<tr>(.*?)</tr>", section, re.S)]
    rows = [row for row in rows if row]
    assert [row[0] for row in rows] == ["clean", "pink, 20 dB", "pink, 10 dB", "pink, 5 dB"]
    assert [row[1:4] for row in rows] == [["1", "1", "0"], ["1", "1", "0"], ["2", "1", "1"], ["1", "0", "1"]]
    assert rows[2][4] == "50.0%"
    assert rows[2][5] == "0.60"
    # the section sits after the cohort table
    assert page.index('id="cohorts"') < page.index('id="noise-levels"')


def test_one_noise_level_has_no_level_table(run_builder, tmp_path):
    results = [
        _scenario(run_builder, tmp_path, "a", noise=_noise("pink", 10.0)),
        _scenario(run_builder, tmp_path, "b", noise=_noise("pink", 10.0)),
    ]
    page = _render(run_builder.run(results, voice_mode=True), tmp_path)
    assert _section(page, "noise-levels") is None


def test_same_label_at_two_levels_is_two_rows(run_builder, tmp_path):
    results = [
        _scenario(run_builder, tmp_path, "a", noise=_noise("file", 10.0, label="traffic", file="t.wav")),
        _scenario(run_builder, tmp_path, "b", noise=_noise("file", 5.0, label="traffic", file="t.wav")),
    ]
    section = _section(_render(run_builder.run(results, voice_mode=True), tmp_path), "noise-levels")
    assert "traffic, 10 dB" in section and "traffic, 5 dB" in section


def test_noise_label_is_escaped(run_builder, tmp_path):
    results = [_scenario(run_builder, tmp_path, "x", noise=_noise("white", 10.0, label="<b>loud</b>"))]
    page = _render(run_builder.run(results, voice_mode=True), tmp_path)
    assert "<b>loud</b>" not in page
    assert "&lt;b&gt;loud&lt;/b&gt;" in page


@pytest.mark.parametrize("bad", ["not a dict", {"snr_db": "x"}, {"label": None, "snr_db": None}])
def test_malformed_noise_metadata_never_breaks_the_report(run_builder, tmp_path, bad):
    result = _scenario(run_builder, tmp_path, "odd")
    result.transcript.metadata["impulse_noise"] = bad
    page = _render(run_builder.run([result], voice_mode=True), tmp_path)
    assert "Impulse report" in page
