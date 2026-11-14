# SPDX-License-Identifier: AGPL-3.0-only
"""``impulse doctor`` rows for noise simulation (SPEC.md section 17; EXPERIMENTAL).

- No noise anywhere: no noise row.
- Noise set: an INFO row "noise simulation" saying it is experimental and listing the levels.
- Each ``profile: file`` noise file (once, even when several scenarios use it): a row "noise file
  {name}": PASS with rate, channels and length; FAIL when the sample rate is below 8000 Hz or above
  192000 Hz; INFO when the rate is below 16000 Hz (it will be upsampled) or the file is shorter than
  1 s (it will loop audibly).
- Noise in text mode: an INFO row saying it is ignored.
A missing or unreadable file is already a config error (the "config" FAIL row).
"""

import io
import re
import wave

import numpy as np

VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}


def _wav(seconds, rate, channels=1):
    frames = int(seconds * rate)
    data = np.random.default_rng(0).uniform(-0.3, 0.3, frames * channels)
    pcm = np.round(data * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _voice(noise=None):
    config = dict(VOICE_CONFIG)
    if noise is not None:
        config["noise"] = noise
    return config


_STATUS = r"(?:PASS|FAIL|INFO|SKIP)"


def _rows(text, check):
    """``"STATUS check detail"`` for every row named ``check`` (``cli.text`` collapses whitespace, so
    the output is one line: a row ends where the next status word or the closing sentence starts)."""
    pattern = rf"({_STATUS}) {re.escape(check)} (.*?)(?= {_STATUS} | All checks passed\.| \d+ check\(s\) failed|$)"
    return [f"{status} {check} {detail}" for status, detail in re.findall(pattern, text)]


def _doctor(project, cli, scenarios=None, **kwargs):
    project.write_agent()
    config = project.write_config(scenarios or [project.scenario("doctor_probe", metrics=["UNIV-013"])], **kwargs)
    result = cli.invoke("doctor", config)
    return result, cli.text(result)


def test_no_noise_no_row(impulse_project, mock_llm, mock_voice_server, cli):
    mock_llm.set_default("OK")
    result, text = _doctor(impulse_project, cli, voice_config=_voice())
    assert result.exit_code == 0, text
    # (the tmp folder name contains the test name, so only the rows are checked)
    assert not re.search(rf"{_STATUS} noise", text)


def test_generated_noise_is_shown_as_experimental(impulse_project, mock_llm, mock_voice_server, cli):
    mock_llm.set_default("OK")
    project = impulse_project
    result, text = _doctor(
        project,
        cli,
        [project.scenario("a", noise={"profile": "pink", "snr_db": 10}),
         project.scenario("b", noise={"profile": "white", "snr_db": 5, "label": "fan"}),
         project.scenario("c", noise=False)],
        voice_config=_voice(),
    )
    assert result.exit_code == 0, text
    rows = _rows(text, "noise simulation")
    assert len(rows) == 1
    assert rows[0].lstrip().startswith("INFO")
    assert "experimental" in rows[0]
    assert "pink, 10 dB" in rows[0] and "fan, 5 dB" in rows[0]


def test_good_noise_file_passes(impulse_project, mock_llm, mock_voice_server, cli):
    mock_llm.set_default("OK")
    project = impulse_project
    (project.root / "traffic.wav").write_bytes(_wav(3.0, 44100, channels=2))
    result, text = _doctor(
        project,
        cli,
        [project.scenario("a"), project.scenario("b")],
        voice_config=_voice({"profile": "file", "file": "traffic.wav", "snr_db": 5}),
    )
    assert result.exit_code == 0, text
    rows = _rows(text, "noise file traffic.wav")
    assert len(rows) == 1
    assert rows[0].lstrip().startswith("PASS")
    assert "44100 Hz" in rows[0] and "2 channels" in rows[0] and "3.0 s" in rows[0]


def test_too_low_sample_rate_fails(impulse_project, mock_llm, mock_voice_server, cli):
    mock_llm.set_default("OK")
    project = impulse_project
    (project.root / "hum.wav").write_bytes(_wav(2.0, 4000))
    result, text = _doctor(project, cli, voice_config=_voice({"profile": "file", "file": "hum.wav", "snr_db": 5}))
    assert result.exit_code == 2
    rows = _rows(text, "noise file hum.wav")
    assert rows and rows[0].lstrip().startswith("FAIL")
    assert "4000 Hz" in rows[0]


def test_low_rate_and_short_file_are_info(impulse_project, mock_llm, mock_voice_server, cli):
    mock_llm.set_default("OK")
    project = impulse_project
    (project.root / "phone.wav").write_bytes(_wav(0.5, 8000))
    result, text = _doctor(project, cli, voice_config=_voice({"profile": "file", "file": "phone.wav", "snr_db": 5}))
    assert result.exit_code == 0, text
    rows = _rows(text, "noise file phone.wav")
    statuses = sorted(row.split()[0] for row in rows)
    assert "PASS" in statuses and "INFO" in statuses
    joined = " ".join(rows)
    assert "upsampled" in joined and "loop" in joined


def test_noise_in_text_mode_is_info(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    project = impulse_project
    result, text = _doctor(project, cli, [project.scenario("a", noise={"profile": "pink", "snr_db": 10})])
    assert result.exit_code == 0, text
    rows = _rows(text, "noise simulation")
    assert len(rows) == 1 and rows[0].lstrip().startswith("INFO")
    assert "ignored" in rows[0] and "text mode" in rows[0]


def test_missing_noise_file_is_a_config_failure(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    project = impulse_project
    result, text = _doctor(project, cli, voice_config=_voice({"profile": "file", "file": "gone.wav", "snr_db": 5}))
    assert result.exit_code == 2
    assert "voice_config.noise.file" in text
