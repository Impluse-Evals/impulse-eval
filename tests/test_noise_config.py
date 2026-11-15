# SPDX-License-Identifier: Apache-2.0
"""Noise simulation (EXPERIMENTAL): config.yaml keys and validation (SPEC.md sections 7 and 8.2).

- ``voice_config.noise`` is the default for every scenario; ``scenarios[i].noise`` replaces it as a
  whole, and ``noise: false`` turns it off for that scenario. ``config.effective_noise(cfg, scenario)``
  returns the settings that apply (None = no noise).
- ``NoiseSettings``: ``profile`` (white | pink | brown | file), ``snr_db`` (required, -5..40), ``file``
  (only with ``profile: file``; relative to the config file's folder, stored resolved), ``label``
  (optional, not blank). Unknown keys inside a noise block are errors.
- Text mode (no ``voice_config``): noise is ignored with ONE warning, never an error.
- ``metrics.noise_simulation`` is kept as a no-op: ``true`` warns that it was replaced.
- ``agent.http.send_audio`` (default false): with the HTTP adapter in voice mode and noise set, false
  gives a warning (the agent receives text only).
"""

import io
import math
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_MODE_WARNING = "noise simulation needs voice mode (voice_config); noise settings are ignored."
OLD_KEY_WARNING = "metrics.noise_simulation is replaced by voice_config.noise; it is ignored."
OLD_RESERVED_WARNING = "metrics.noise_simulation is reserved for v2 and is ignored."
HTTP_TEXT_ONLY_WARNING = (
    "agent.http.send_audio is false: the HTTP agent receives text only, so noise simulation has no effect on it."
)
VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}


def _wav(samples, rate=16000, channels=1):
    pcm = np.clip(np.round(np.asarray(samples, dtype=np.float64) * 32767), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _noise_wav(seconds=1.0, rate=16000):
    return _wav(np.random.default_rng(0).uniform(-0.3, 0.3, int(seconds * rate)), rate)


def _voice(noise=None):
    config = dict(VOICE_CONFIG)
    if noise is not None:
        config["noise"] = noise
    return config


def _write(project, scenarios=None, **kwargs):
    project.write_agent()
    return project.write_config(scenarios or [project.scenario("noisy")], **kwargs)


def _load(project, path):
    from impulse_evals.config import load_config

    return load_config(path)


def _error(project, path):
    from impulse_evals.config import ConfigError, load_config

    with pytest.raises(ConfigError) as info:
        load_config(path)
    return info.value


# ---------------------------------------------------------------------------------------------
# Shape: default, override, off
# ---------------------------------------------------------------------------------------------


def test_voice_config_noise_is_the_default_for_every_scenario(impulse_project):
    from impulse_evals.config import effective_noise
    from impulse_evals.models import NoiseSettings

    project = impulse_project
    path = _write(
        project,
        [project.scenario("a"), project.scenario("b")],
        voice_config=_voice({"profile": "pink", "snr_db": 10, "label": "office hum"}),
    )
    cfg = _load(project, path)
    assert isinstance(cfg.voice_config.noise, NoiseSettings)
    for scenario in cfg.scenarios:
        noise = effective_noise(cfg, scenario)
        assert (noise.profile, noise.snr_db, noise.label, noise.file) == ("pink", 10.0, "office hum", None)
    assert cfg.warnings == []


def test_scenario_noise_replaces_the_default_as_a_whole(impulse_project):
    from impulse_evals.config import effective_noise

    project = impulse_project
    path = _write(
        project,
        [project.scenario("override", noise={"profile": "brown", "snr_db": 5}), project.scenario("inherit")],
        voice_config=_voice({"profile": "pink", "snr_db": 20, "label": "office"}),
    )
    cfg = _load(project, path)
    override = effective_noise(cfg, cfg.scenarios[0])
    assert (override.profile, override.snr_db, override.label) == ("brown", 5.0, None)
    assert effective_noise(cfg, cfg.scenarios[1]).snr_db == 20.0


def test_scenario_noise_false_turns_noise_off(impulse_project):
    from impulse_evals.config import effective_noise

    project = impulse_project
    path = _write(
        project,
        [project.scenario("clean", noise=False), project.scenario("noisy")],
        voice_config=_voice({"profile": "white", "snr_db": 10}),
    )
    cfg = _load(project, path)
    assert cfg.scenarios[0].noise is False
    assert effective_noise(cfg, cfg.scenarios[0]) is None
    assert effective_noise(cfg, cfg.scenarios[1]).profile == "white"


def test_scenario_noise_null_inherits_the_default(impulse_project):
    from impulse_evals.config import effective_noise

    project = impulse_project
    path = _write(project, [project.scenario("x", noise=None)], voice_config=_voice({"profile": "white", "snr_db": 10}))
    cfg = _load(project, path)
    assert effective_noise(cfg, cfg.scenarios[0]).profile == "white"


def test_scenario_noise_without_a_default(impulse_project):
    from impulse_evals.config import effective_noise

    project = impulse_project
    path = _write(
        project,
        [project.scenario("noisy", noise={"profile": "pink", "snr_db": 0}), project.scenario("clean")],
        voice_config=_voice(),
    )
    cfg = _load(project, path)
    assert cfg.voice_config.noise is None
    assert effective_noise(cfg, cfg.scenarios[0]).snr_db == 0.0
    assert effective_noise(cfg, cfg.scenarios[1]) is None


def test_no_noise_anywhere_changes_nothing(impulse_project):
    from impulse_evals.config import effective_noise

    project = impulse_project
    path = _write(project, voice_config=_voice())
    cfg = _load(project, path)
    assert cfg.voice_config.noise is None
    assert cfg.scenarios[0].noise is None
    assert effective_noise(cfg, cfg.scenarios[0]) is None
    assert cfg.warnings == []


def test_relative_noise_file_is_resolved_against_the_config_folder(impulse_project):
    project = impulse_project
    folder = project.root / "noise"
    folder.mkdir()
    (folder / "traffic.wav").write_bytes(_noise_wav())
    path = _write(project, voice_config=_voice({"profile": "file", "file": "noise/traffic.wav", "snr_db": 5}))
    cfg = _load(project, path)
    stored = Path(cfg.voice_config.noise.file)
    assert stored.is_absolute()
    assert stored == (folder / "traffic.wav").resolve()


def test_scenario_noise_file_is_resolved_too(impulse_project):
    project = impulse_project
    (project.root / "cafe.wav").write_bytes(_noise_wav())
    path = _write(
        project,
        [project.scenario("cafe", noise={"profile": "file", "file": "cafe.wav", "snr_db": 10, "label": "cafe"})],
        voice_config=_voice(),
    )
    cfg = _load(project, path)
    assert Path(cfg.scenarios[0].noise.file) == (project.root / "cafe.wav").resolve()


def test_snr_limits_are_inclusive(impulse_project):
    project = impulse_project
    path = _write(
        project,
        [project.scenario("low", noise={"profile": "white", "snr_db": -5}),
         project.scenario("high", noise={"profile": "white", "snr_db": 40})],
        voice_config=_voice(),
    )
    cfg = _load(project, path)
    assert [s.noise.snr_db for s in cfg.scenarios] == [-5.0, 40.0]


# ---------------------------------------------------------------------------------------------
# Validation errors: every one, at voice_config.noise and at scenarios[i].noise
# ---------------------------------------------------------------------------------------------

ERROR_CASES = [
    ({"profile": "purple", "snr_db": 10}, "profile", "unknown noise profile purple. Use one of: white, pink, brown, file."),
    ({"profile": "pink"}, "snr_db", "required"),
    ({"profile": "pink", "snr_db": 41}, "snr_db", "snr_db 41 is outside -5..40"),
    ({"profile": "pink", "snr_db": -6}, "snr_db", "snr_db -6 is outside -5..40"),
    ({"profile": "pink", "snr_db": 40.5}, "snr_db", "snr_db 40.5 is outside -5..40"),
    ({"profile": "pink", "snr_db": float("nan")}, "snr_db", "snr_db nan is outside -5..40"),
    ({"profile": "file", "snr_db": 10}, "file", "required when profile is file"),
    ({"profile": "file", "file": "  ", "snr_db": 10}, "file", "required when profile is file"),
    ({"profile": "pink", "file": "x.wav", "snr_db": 10}, "file", "only used when profile is file"),
    ({"profile": "pink", "snr_db": 10, "label": "   "}, "label", "label must not be blank"),
]


@pytest.mark.parametrize(("noise", "key", "message"), ERROR_CASES)
def test_default_noise_errors(impulse_project, noise, key, message):
    project = impulse_project
    error = _error(project, _write(project, voice_config=_voice(noise)))
    assert error.field == f"voice_config.noise.{key}"
    assert error.message == message


@pytest.mark.parametrize(("noise", "key", "message"), ERROR_CASES)
def test_scenario_noise_errors(impulse_project, noise, key, message):
    project = impulse_project
    path = _write(project, [project.scenario("ok"), project.scenario("bad", noise=noise)], voice_config=_voice())
    error = _error(project, path)
    assert error.field == f"scenarios[1].noise.{key}"
    assert error.message == message


@pytest.mark.parametrize("value", ["loud", True, [10]])
def test_snr_must_be_a_number(impulse_project, value):
    project = impulse_project
    error = _error(project, _write(project, voice_config=_voice({"profile": "pink", "snr_db": value})))
    assert error.field == "voice_config.noise.snr_db"


def test_missing_profile_is_an_error(impulse_project):
    project = impulse_project
    error = _error(project, _write(project, voice_config=_voice({"snr_db": 10})))
    assert error.field == "voice_config.noise.profile"


def test_unknown_key_in_a_noise_block_is_an_error(impulse_project):
    project = impulse_project
    error = _error(project, _write(project, voice_config=_voice({"profile": "pink", "snr_db": 10, "volume": 3})))
    assert error.field == "voice_config.noise.volume"


def test_scenario_noise_true_is_an_error(impulse_project):
    project = impulse_project
    error = _error(project, _write(project, [project.scenario("x", noise=True)], voice_config=_voice()))
    assert error.field.startswith("scenarios[0].noise")


def test_missing_noise_file(impulse_project):
    project = impulse_project
    error = _error(project, _write(project, voice_config=_voice({"profile": "file", "file": "nope.wav", "snr_db": 5})))
    assert error.field == "voice_config.noise.file"
    assert error.message == f"noise file not found: {project.root / 'nope.wav'}"


def test_noise_file_that_is_a_folder(impulse_project):
    project = impulse_project
    (project.root / "folder.wav").mkdir()
    error = _error(project, _write(project, voice_config=_voice({"profile": "file", "file": "folder.wav", "snr_db": 5})))
    assert error.field == "voice_config.noise.file"
    assert error.message.startswith("noise file not found: ")


def test_unreadable_noise_file(impulse_project, monkeypatch):
    project = impulse_project
    target = project.root / "locked.wav"
    target.write_bytes(_noise_wav())
    real_read_bytes = Path.read_bytes

    def read_bytes(self):
        if self.name == "locked.wav":
            raise PermissionError(13, "Permission denied")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    error = _error(project, _write(project, voice_config=_voice({"profile": "file", "file": "locked.wav", "snr_db": 5})))
    assert error.field == "voice_config.noise.file"
    assert error.message == f"cannot read noise file {target.resolve()}: Permission denied"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"RIFF but not really a wave file", "noise file is not a WAV file: {path}"),
        (b"ID3\x03\x00\x00\x00 an mp3", "noise file is not a WAV file: {path}"),
        ("empty", "noise file holds no audio: {path}"),
        ("silent", "noise file is silent: {path}"),
    ],
)
def test_bad_noise_file_content(impulse_project, content, message):
    project = impulse_project
    target = project.root / "bad.wav"
    if content == "empty":
        target.write_bytes(_wav(np.zeros(0)))
    elif content == "silent":
        target.write_bytes(_wav(np.zeros(16000)))
    else:
        target.write_bytes(content)
    error = _error(project, _write(project, voice_config=_voice({"profile": "file", "file": "bad.wav", "snr_db": 5})))
    assert error.field == "voice_config.noise.file"
    assert error.message == message.format(path=target.resolve())


def test_noise_errors_never_echo_file_contents(impulse_project):
    project = impulse_project
    secret = b"sk-impulse-test-SECRET-noise-7781"
    (project.root / "secret.wav").write_bytes(secret)
    error = _error(project, _write(project, voice_config=_voice({"profile": "file", "file": "secret.wav", "snr_db": 5})))
    assert secret.decode() not in str(error)


# ---------------------------------------------------------------------------------------------
# Warnings: text mode, the old key, HTTP without audio
# ---------------------------------------------------------------------------------------------


def test_text_mode_noise_is_one_warning(impulse_project):
    project = impulse_project
    path = _write(
        project,
        [project.scenario("a", noise={"profile": "pink", "snr_db": 10}),
         project.scenario("b", noise={"profile": "white", "snr_db": 5})],
    )
    cfg = _load(project, path)
    assert cfg.voice_config is None
    assert cfg.warnings.count(TEXT_MODE_WARNING) == 1


def test_text_mode_noise_false_gives_no_warning(impulse_project):
    project = impulse_project
    cfg = _load(project, _write(project, [project.scenario("a", noise=False)]))
    assert cfg.warnings == []


def test_text_mode_still_validates_noise(impulse_project):
    project = impulse_project
    error = _error(project, _write(project, [project.scenario("a", noise={"profile": "purple", "snr_db": 10})]))
    assert error.field == "scenarios[0].noise.profile"


def test_old_key_true_warns_that_it_was_replaced(impulse_project):
    project = impulse_project
    cfg = _load(project, _write(project, metrics={"enabled": True, "noise_simulation": True}))
    assert OLD_KEY_WARNING in cfg.warnings
    assert OLD_RESERVED_WARNING not in cfg.warnings


def test_old_key_false_is_silent(impulse_project):
    project = impulse_project
    cfg = _load(project, _write(project, metrics={"enabled": True, "noise_simulation": False}))
    assert cfg.warnings == []


def test_http_without_send_audio_warns_when_noise_is_set(impulse_project):
    project = impulse_project
    path = _write(
        project,
        agent={"adapter": "http", "http": {"url": "http://localhost:8080/turn"}},
        voice_config=_voice({"profile": "pink", "snr_db": 10}),
    )
    cfg = _load(project, path)
    assert cfg.agent.http.send_audio is False
    assert HTTP_TEXT_ONLY_WARNING in cfg.warnings


def test_http_with_send_audio_does_not_warn(impulse_project):
    project = impulse_project
    path = _write(
        project,
        agent={"adapter": "http", "http": {"url": "http://localhost:8080/turn", "send_audio": True}},
        voice_config=_voice({"profile": "pink", "snr_db": 10}),
    )
    cfg = _load(project, path)
    assert cfg.agent.http.send_audio is True
    assert HTTP_TEXT_ONLY_WARNING not in cfg.warnings


def test_http_without_noise_does_not_warn(impulse_project):
    project = impulse_project
    path = _write(project, agent={"adapter": "http", "http": {"url": "http://localhost:8080/turn"}}, voice_config=_voice())
    assert _load(project, path).warnings == []


# ---------------------------------------------------------------------------------------------
# examples/config.noise.yaml
# ---------------------------------------------------------------------------------------------


def test_example_noise_config_loads_with_four_levels():
    from impulse_evals.config import effective_noise, load_config

    cfg = load_config(REPO_ROOT / "examples" / "config.noise.yaml")
    assert cfg.voice_config is not None
    assert len(cfg.scenarios) == 4
    levels = [effective_noise(cfg, scenario) for scenario in cfg.scenarios]
    assert levels[0] is None
    assert [level.snr_db for level in levels[1:]] == [20.0, 10.0, 5.0]
    assert cfg.warnings == []


def test_example_configs_no_longer_carry_the_reserved_key():
    for relative in ("examples/config.yaml", "examples/free/config.yaml", "impulse_evals/templates/free_config.yaml"):
        assert "noise_simulation" not in (REPO_ROOT / relative).read_text(encoding="utf-8"), relative


def test_noise_settings_fields():
    from impulse_evals.models import NoiseSettings, Scenario

    assert set(NoiseSettings.model_fields) == {"profile", "snr_db", "file", "label"}
    assert {name for name, field in NoiseSettings.model_fields.items() if field.is_required()} == {"profile"}
    assert Scenario.model_fields["noise"].default is None


def test_config_and_mixer_agree_on_profiles_and_limits():
    from impulse_evals import config
    from impulse_evals.voice import noise

    assert config.NOISE_PROFILES == noise.NOISE_PROFILES
    assert (config.NOISE_SNR_MIN_DB, config.NOISE_SNR_MAX_DB) == (noise.SNR_MIN_DB, noise.SNR_MAX_DB)


def test_snr_constant_text_uses_g_format():
    # 10.0 is shown as 10, 40.5 as 40.5 (the messages above rely on this)
    assert f"{10.0:g}" == "10" and f"{40.5:g}" == "40.5" and f"{math.nan:g}" == "nan"
