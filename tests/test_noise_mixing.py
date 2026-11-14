# SPDX-License-Identifier: AGPL-3.0-only
"""Noise simulation (EXPERIMENTAL): the mixer in ``impulse_evals.voice.noise`` (SPEC.md section 14.4).

``mix_noise(clean_wav, settings, *, seed, turn, noise_source=None) -> NoiseMix`` mixes background
noise into the caller's TTS audio:

- the speech span is the first to the last sample whose magnitude is above 1% of full scale (the rule
  the LiveKit and Pipecat adapters use to find the end of speech);
- the noise level is set by RMS over that span: 20*log10(rms(speech) / rms(noise)) == snr_db;
- the noise runs from the start of the clip to the end of speech, then fades out over 20 ms; the rest
  of the clip is left exactly as it was (digital silence stays silent), so the adapters' end-of-speech
  time, and with it the agent latency, does not move by more than the fade;
- white / pink / brown noise is generated in code; ``profile: file`` noise is mixed down to mono,
  resampled to the speech rate, and looped from a start offset picked from the seed;
- the same ``seed`` and ``turn`` give byte-identical audio; when the mix would clip, the whole mix is
  scaled down (the SNR is kept) and ``scaled`` is True;
- a clip with no sample above the threshold is returned unchanged (``measured_snr_db`` None).

The output is a mono 16-bit PCM WAV at the speech's sample rate. No network, no files outside tmp_path.
"""

import hashlib
import io
import math
import wave

import numpy as np
import pytest

SPEECH_RATE = 16000


# ---------------------------------------------------------------------------------------------
# helpers (independent of the module under test)
# ---------------------------------------------------------------------------------------------


def _wav(samples, rate, channels=1):
    """A 16-bit PCM WAV of float samples in [-1, 1] (1-D mono or (frames, channels))."""
    data = np.asarray(samples, dtype=np.float64)
    pcm = np.clip(np.round(data * 32767), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _read(wav_data):
    """``(mono float64 samples, rate, channels)`` of a 16-bit PCM WAV."""
    with wave.open(io.BytesIO(wav_data), "rb") as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    assert width == 2
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32767.0
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, rate, channels


def _speech(rate=SPEECH_RATE, lead_s=0.2, speech_s=1.0, tail_s=0.3, amplitude=0.4):
    """Speech-like test audio: silence, a 1 s 'utterance' (three tones, syllable envelope), silence."""
    t = np.arange(int(round(speech_s * rate))) / rate
    voice = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 660 * t) + 0.25 * np.sin(2 * np.pi * 1320 * t)
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t) ** 2
    voice = amplitude * voice * envelope / np.max(np.abs(voice * envelope))
    return np.concatenate([np.zeros(int(round(lead_s * rate))), voice, np.zeros(int(round(tail_s * rate)))])


def _span(samples, threshold=0.01):
    above = np.flatnonzero(np.abs(samples) > threshold)
    return int(above[0]), int(above[-1]) + 1


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def _snr_db(clean_wav, noisy_wav, *, least_squares=False):
    """SNR of ``noisy`` against ``clean`` over the clean speech span. Without scaling the noise is
    ``noisy - clean``; with ``least_squares`` the speech gain is estimated first (a scaled mix)."""
    clean, _, _ = _read(clean_wav)
    noisy, _, _ = _read(noisy_wav)
    start, end = _span(clean)
    c, n = clean[start:end], noisy[start:end]
    gain = float(np.dot(n, c) / np.dot(c, c)) if least_squares else 1.0
    return 20 * math.log10(_rms(gain * c) / _rms(n - gain * c))


def _settings(**fields):
    from impulse_evals.models import NoiseSettings

    return NoiseSettings(**fields)


def _mix(clean_wav, *, seed=7, turn=1, noise_source=None, **fields):
    from impulse_evals.voice.noise import mix_noise

    return mix_noise(clean_wav, _settings(**fields), seed=seed, turn=turn, noise_source=noise_source)


def _noise_file(tmp_path, name, seconds, rate, channels=1, seed=3):
    rng = np.random.default_rng(seed)
    frames = int(seconds * rate)
    data = rng.uniform(-0.3, 0.3, size=(frames, channels) if channels > 1 else frames)
    path = tmp_path / name
    path.write_bytes(_wav(data, rate, channels))
    return path


def _source(path):
    from impulse_evals.voice.noise import load_noise_file

    return load_noise_file(path)


# ---------------------------------------------------------------------------------------------
# constants and the public surface
# ---------------------------------------------------------------------------------------------


def test_profiles_and_limits():
    from impulse_evals.voice import noise

    assert noise.NOISE_PROFILES == ("white", "pink", "brown", "file")
    assert noise.SNR_MIN_DB == -5.0
    assert noise.SNR_MAX_DB == 40.0
    assert noise.SPEECH_THRESHOLD == 0.01
    assert noise.FADE_OUT_MS == 20


# ---------------------------------------------------------------------------------------------
# SNR accuracy: within +-0.5 dB of the target
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["white", "pink", "brown"])
@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 5.0, 10.0, 20.0, 40.0])
def test_generated_noise_hits_the_target_snr(profile, snr_db):
    # Quiet speech, so even -5 dB needs no clip protection (scaling is covered by the clipping tests).
    clean = _wav(_speech(amplitude=0.1), SPEECH_RATE)
    mixed = _mix(clean, profile=profile, snr_db=snr_db)
    assert mixed.scaled is False
    assert abs(_snr_db(clean, mixed.wav) - snr_db) <= 0.5
    assert abs(mixed.measured_snr_db - snr_db) <= 0.5


@pytest.mark.parametrize("snr_db", [-5.0, 5.0, 10.0, 20.0, 40.0])
def test_file_noise_hits_the_target_snr(tmp_path, snr_db):
    clean = _wav(_speech(), SPEECH_RATE)
    path = _noise_file(tmp_path, "street.wav", 3.0, SPEECH_RATE)
    mixed = _mix(clean, profile="file", file=str(path), snr_db=snr_db, noise_source=_source(path))
    assert abs(_snr_db(clean, mixed.wav) - snr_db) <= 0.5


def test_output_is_mono_pcm16_at_the_speech_rate_and_same_length():
    clean = _wav(_speech(rate=24000), 24000)
    mixed = _mix(clean, profile="pink", snr_db=10)
    samples, rate, channels = _read(mixed.wav)
    original, _, _ = _read(clean)
    assert (rate, channels) == (24000, 1)
    assert len(samples) == len(original)


def test_stereo_speech_is_mixed_to_mono():
    mono = _speech()
    clean = _wav(np.stack([mono, mono], axis=1), SPEECH_RATE, channels=2)
    mixed = _mix(clean, profile="white", snr_db=10)
    _, rate, channels = _read(mixed.wav)
    assert (rate, channels) == (SPEECH_RATE, 1)
    assert abs(_snr_db(_wav(mono, SPEECH_RATE), mixed.wav) - 10) <= 0.5


# ---------------------------------------------------------------------------------------------
# Latency honesty: noise stops at the end of speech (plus the fade); the tail stays silent
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["white", "pink", "brown"])
def test_tail_after_speech_and_fade_stays_silent(profile):
    clean_samples = _speech(tail_s=0.5)
    clean = _wav(clean_samples, SPEECH_RATE)
    mixed = _mix(clean, profile=profile, snr_db=-5)
    noisy, _, _ = _read(mixed.wav)
    _, end = _span(clean_samples)
    fade = int(SPEECH_RATE * 20 / 1000)
    assert np.all(noisy[end + fade:] == 0.0)
    # the last sample above 1% of full scale moves by at most the fade
    _, noisy_end = _span(noisy)
    assert noisy_end - end <= fade


def test_noise_starts_with_the_clip():
    clean_samples = _speech(lead_s=0.3)
    mixed = _mix(_wav(clean_samples, SPEECH_RATE), profile="white", snr_db=0)
    noisy, _, _ = _read(mixed.wav)
    lead = int(0.3 * SPEECH_RATE)
    assert _rms(noisy[: lead // 2]) > 0.01   # leading silence carries noise (after a short fade-in)


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["white", "pink", "brown"])
def test_same_seed_and_turn_give_identical_bytes(profile):
    clean = _wav(_speech(), SPEECH_RATE)
    first = _mix(clean, profile=profile, snr_db=10, seed=42, turn=3)
    second = _mix(clean, profile=profile, snr_db=10, seed=42, turn=3)
    assert first.wav == second.wav


def test_other_seed_or_turn_gives_other_noise():
    clean = _wav(_speech(), SPEECH_RATE)
    base = _mix(clean, profile="white", snr_db=10, seed=42, turn=1).wav
    assert _mix(clean, profile="white", snr_db=10, seed=43, turn=1).wav != base
    assert _mix(clean, profile="white", snr_db=10, seed=42, turn=2).wav != base


def test_file_offset_comes_from_the_seed(tmp_path):
    clean = _wav(_speech(), SPEECH_RATE)
    path = _noise_file(tmp_path, "cafe.wav", 5.0, SPEECH_RATE)
    source = _source(path)
    a = _mix(clean, profile="file", file=str(path), snr_db=10, seed=1, noise_source=source)
    b = _mix(clean, profile="file", file=str(path), snr_db=10, seed=1, noise_source=source)
    c = _mix(clean, profile="file", file=str(path), snr_db=10, seed=2, noise_source=source)
    assert a.wav == b.wav and a.offset_ms == b.offset_ms
    assert isinstance(a.offset_ms, int) and 0 <= a.offset_ms < 5000
    assert c.offset_ms != a.offset_ms
    assert c.wav != a.wav


def test_generated_noise_has_no_offset():
    mixed = _mix(_wav(_speech(), SPEECH_RATE), profile="pink", snr_db=10)
    assert mixed.offset_ms is None


def test_noise_seed_is_stable_and_uses_the_scenario_seed():
    from impulse_evals.voice.noise import noise_seed

    expected = int.from_bytes(hashlib.sha256("clinic booking".encode("utf-8")).digest()[:4], "big")
    assert noise_seed("clinic booking", None) == expected
    assert noise_seed("clinic booking", 1234) == 1234
    assert noise_seed("other", None) != expected


# ---------------------------------------------------------------------------------------------
# No clipping
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["white", "pink", "brown"])
def test_loud_speech_never_clips(profile):
    clean = _wav(_speech(amplitude=0.98), SPEECH_RATE)
    mixed = _mix(clean, profile=profile, snr_db=-5)
    with wave.open(io.BytesIO(mixed.wav), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert mixed.scaled is True
    assert int(np.max(np.abs(pcm.astype(np.int32)))) <= int(0.99 * 32767) + 1
    assert abs(_snr_db(clean, mixed.wav, least_squares=True) - (-5)) <= 0.5


def test_quiet_mix_is_not_scaled():
    mixed = _mix(_wav(_speech(amplitude=0.2), SPEECH_RATE), profile="white", snr_db=20)
    assert mixed.scaled is False


# ---------------------------------------------------------------------------------------------
# Resampling and recorded noise files
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("noise_rate", [8000, 22050, 44100, 48000])
@pytest.mark.parametrize("speech_rate", [16000, 24000])
def test_noise_file_is_resampled_to_the_speech_rate(tmp_path, noise_rate, speech_rate):
    clean = _wav(_speech(rate=speech_rate), speech_rate)
    path = _noise_file(tmp_path, f"n{noise_rate}.wav", 2.0, noise_rate)
    source = _source(path)
    assert source[1] == noise_rate
    mixed = _mix(clean, profile="file", file=str(path), snr_db=10, noise_source=source)
    samples, rate, channels = _read(mixed.wav)
    assert (rate, channels) == (speech_rate, 1)
    assert len(samples) == len(_read(clean)[0])
    assert abs(_snr_db(clean, mixed.wav) - 10) <= 0.5


def test_short_noise_file_is_looped(tmp_path):
    clean = _wav(_speech(speech_s=2.0), SPEECH_RATE)
    path = _noise_file(tmp_path, "short.wav", 0.25, SPEECH_RATE)
    mixed = _mix(clean, profile="file", file=str(path), snr_db=10, noise_source=_source(path))
    noisy, _, _ = _read(mixed.wav)
    clean_samples, _, _ = _read(clean)
    start, end = _span(clean_samples)
    residual = noisy[start:end] - clean_samples[start:end]
    # noise all the way through the 2 s utterance, not only its first 0.25 s
    last_quarter = residual[-len(residual) // 4:]
    assert _rms(last_quarter) > 0.5 * _rms(residual)
    assert abs(_snr_db(clean, mixed.wav) - 10) <= 0.5


def test_stereo_noise_file_loads_as_mono(tmp_path):
    path = _noise_file(tmp_path, "stereo.wav", 1.0, 44100, channels=2)
    samples, rate = _source(path)
    assert rate == 44100
    assert samples.ndim == 1
    assert len(samples) == 44100


def test_noise_file_is_loaded_without_a_noise_source_argument(tmp_path):
    clean = _wav(_speech(), SPEECH_RATE)
    path = _noise_file(tmp_path, "auto.wav", 2.0, SPEECH_RATE)
    with_source = _mix(clean, profile="file", file=str(path), snr_db=10, noise_source=_source(path))
    without = _mix(clean, profile="file", file=str(path), snr_db=10)
    assert with_source.wav == without.wav


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"this is not audio", "is not a WAV file"),
        ("empty", "holds no audio"),
        ("silent", "is silent"),
    ],
)
def test_bad_noise_files_raise_noise_file_error(tmp_path, content, message):
    from impulse_evals.voice.noise import NoiseFileError, load_noise_file

    path = tmp_path / "bad.wav"
    if content == "empty":
        path.write_bytes(_wav(np.zeros(0), 16000))
    elif content == "silent":
        path.write_bytes(_wav(np.zeros(16000), 16000))
    else:
        path.write_bytes(content)
    with pytest.raises(NoiseFileError) as info:
        load_noise_file(path)
    assert message in str(info.value)


def test_missing_noise_file_raises_noise_file_error(tmp_path):
    from impulse_evals.voice.noise import NoiseFileError, load_noise_file

    with pytest.raises(NoiseFileError) as info:
        load_noise_file(tmp_path / "nope.wav")
    assert "not found" in str(info.value)


# ---------------------------------------------------------------------------------------------
# Edge cases and labels
# ---------------------------------------------------------------------------------------------


def test_silent_clip_is_returned_unchanged():
    clean = _wav(np.zeros(8000), SPEECH_RATE)
    mixed = _mix(clean, profile="white", snr_db=10)
    assert mixed.wav == clean
    assert mixed.measured_snr_db is None
    assert mixed.scaled is False


def test_default_labels():
    from impulse_evals.voice.noise import default_label

    assert default_label(_settings(profile="pink", snr_db=10)) == "pink"
    assert default_label(_settings(profile="file", file="noise/busy street.wav", snr_db=5)) == "busy street"
    assert default_label(_settings(profile="white", snr_db=5, label="office")) == "office"
