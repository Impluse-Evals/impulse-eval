# SPDX-License-Identifier: AGPL-3.0-only
"""Noise simulation (EXPERIMENTAL; SPEC.md section 14.4): background noise mixed into the caller's
TTS audio after TTS and before the adapter sends it.

- ``white`` / ``pink`` / ``brown`` noise is generated here with numpy (pink and brown by shaping white
  noise in the frequency domain, 1/sqrt(f) and 1/f, with nothing below 20 Hz). ``file`` noise comes
  from a user-supplied WAV file only (no audio ships with Impulse): mixed down to mono, resampled to
  the speech's rate, and looped from a start offset picked from the seed.
- The level is set by RMS over the speech span (the first to the last sample above 1% of full scale,
  the rule the LiveKit and Pipecat adapters use to find the end of speech): 20*log10(rms(speech) /
  rms(noise)) == ``snr_db``.
- The noise runs from the start of the clip (10 ms fade-in) to the end of speech, then fades out over
  ``FADE_OUT_MS``; the rest of the clip is left as it was. So the adapters' end-of-speech time, and
  with it the agent latency, moves by at most the fade. Noise that goes on after the caller stops is
  not simulated (a documented limitation).
- The random generator is seeded with ``(seed, turn)``: the same scenario seed (or, without one, a
  SHA-256 of the scenario name) gives byte-identical audio on every run and machine.
- When the mix would clip, the whole mix is scaled down so its peak is 0.99 (the SNR is kept).

Output: a mono 16-bit PCM WAV at the speech's sample rate. A clip with no sample above the
threshold is returned unchanged.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from impulse_evals.models import NoiseSettings
from impulse_evals.voice import audio_utils

__all__ = [
    "CLIP_PEAK",
    "FADE_IN_MS",
    "FADE_OUT_MS",
    "NOISE_PROFILES",
    "SNR_MAX_DB",
    "SNR_MIN_DB",
    "SPEECH_THRESHOLD",
    "NoiseFileError",
    "NoiseMix",
    "default_label",
    "level_name",
    "load_noise_file",
    "mix_noise",
    "noise_seed",
]

NOISE_PROFILES = ("white", "pink", "brown", "file")
SNR_MIN_DB = -5.0
SNR_MAX_DB = 40.0
SPEECH_THRESHOLD = 0.01     # fraction of full scale, as the adapters' end-of-speech rule
FADE_IN_MS = 10
FADE_OUT_MS = 20
CLIP_PEAK = 0.99
LOW_CUT_HZ = 20.0           # pink and brown noise carry nothing below this
_SILENT_PEAK = 1e-4         # a noise file whose peak is below this is "silent"

NoiseSource = tuple[np.ndarray, int]   # (mono float samples, sample rate)


class NoiseFileError(ValueError):
    """A noise file that cannot be used; the message names the file, never its contents."""


@dataclass(frozen=True)
class NoiseMix:
    wav: bytes                          # the noisy audio (the clean audio itself when nothing was mixed)
    measured_snr_db: Optional[float]    # measured on the 16-bit output; None when nothing was mixed
    scaled: bool                        # the mix was scaled down to avoid clipping
    offset_ms: Optional[int]            # profile: file only, where in the file the noise started


def noise_seed(scenario_name: str, seed: Optional[int]) -> int:
    """The scenario's ``seed`` when set, else the first 4 bytes of SHA-256 of its name (stable across
    processes and machines, unlike ``hash()``)."""
    if seed is not None:
        return int(seed)
    return int.from_bytes(hashlib.sha256(scenario_name.encode("utf-8")).digest()[:4], "big")


def default_label(settings: NoiseSettings) -> str:
    """The label shown in reports: ``label`` when set, else the file's stem (``profile: file``), else
    the profile."""
    if settings.label is not None and settings.label.strip():
        return settings.label.strip()
    if settings.profile == "file" and settings.file:
        return Path(settings.file.replace("\\", "/")).stem or settings.profile
    return settings.profile


def level_name(label: str, snr_db: float) -> str:
    """``"{label}, {snr_db:g} dB"``, the name of a noise level in reports and the doctor."""
    return f"{label}, {snr_db:g} dB"


# ------------------------------------------------------------------------------------------------
# Noise files
# ------------------------------------------------------------------------------------------------


def load_noise_file(path: Union[str, Path]) -> NoiseSource:
    """``(mono float32 samples, sample rate)`` of a WAV noise file (PCM 8/16/24/32-bit or IEEE float).

    Raises ``NoiseFileError``: "noise file not found: {path}", "cannot read noise file {path}: {os
    error}", "noise file is not a WAV file: {path}", "noise file holds no audio: {path}", "noise file is
    silent: {path}".
    """
    target = Path(path)
    if not target.is_file():
        raise NoiseFileError(f"noise file not found: {target}")
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise NoiseFileError(f"cannot read noise file {target}: {exc.strerror or type(exc).__name__}") from None
    if not audio_utils.is_wav(data):
        raise NoiseFileError(f"noise file is not a WAV file: {target}")
    error: Optional[NoiseFileError] = None
    try:
        samples, rate = audio_utils.decode_wav(audio_utils.ensure_wav(data))
        mono = audio_utils.to_mono(samples)
    except Exception:  # noqa: BLE001 - any parse failure; the text could quote file bytes, so it is dropped
        error = NoiseFileError(f"noise file is not a WAV file: {target}")
    if error is not None:
        raise error
    if mono.size == 0 or rate <= 0:
        raise NoiseFileError(f"noise file holds no audio: {target}")
    if float(np.max(np.abs(mono))) < _SILENT_PEAK:
        raise NoiseFileError(f"noise file is silent: {target}")
    return mono.astype(np.float32), int(rate)


# ------------------------------------------------------------------------------------------------
# Mixing
# ------------------------------------------------------------------------------------------------


def _rng(seed: int, turn: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed) % 2**64, int(turn) % 2**64]))


def _unit_rms(signal: np.ndarray) -> np.ndarray:
    level = math.sqrt(float(np.mean(np.square(signal)))) if signal.size else 0.0
    return signal / level if level > 0 else signal


def _generate(profile: str, frames: int, rate: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(frames)
    if profile == "white" or frames < 4:
        return _unit_rms(white)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(frames, 1.0 / rate)
    shape = np.zeros_like(freqs)
    band = freqs >= LOW_CUT_HZ
    shape[band] = 1.0 / freqs[band] ** (0.5 if profile == "pink" else 1.0)
    shaped = np.fft.irfft(spectrum * shape, n=frames)
    if not np.any(shaped):   # too short to hold any band above 20 Hz
        return _unit_rms(white)
    return _unit_rms(shaped)


def _from_file(source: NoiseSource, frames: int, rate: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    samples, source_rate = source
    noise = audio_utils.resample(np.asarray(samples, dtype=np.float64), source_rate, rate).astype(np.float64)
    if noise.size == 0:
        raise NoiseFileError("noise file holds no audio")
    offset = int(rng.integers(0, noise.size))
    looped = np.take(noise, (offset + np.arange(frames)) % noise.size)
    return looped, int(offset * 1000 // rate)


def _envelope(length: int, speech_end: int, rate: int) -> np.ndarray:
    envelope = np.ones(length)
    fade_in = min(length, max(1, round(rate * FADE_IN_MS / 1000)))
    envelope[:fade_in] = np.linspace(0.0, 1.0, fade_in + 1)[1:]
    fade_out = length - speech_end
    if fade_out > 0:
        steps = max(1, round(rate * FADE_OUT_MS / 1000))
        envelope[speech_end:] = np.minimum(envelope[speech_end:], 1.0 - np.arange(1, fade_out + 1) / steps)
    return np.clip(envelope, 0.0, 1.0)


def _rms(signal: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(signal)))) if signal.size else 0.0


def mix_noise(
    clean_wav: bytes,
    settings: NoiseSettings,
    *,
    seed: int,
    turn: int,
    noise_source: Optional[NoiseSource] = None,
) -> NoiseMix:
    """Mix the noise of ``settings`` into ``clean_wav`` (see the module docstring).

    ``noise_source`` is the loaded file for ``profile: file`` (``load_noise_file``); when None the file
    is loaded here. ``settings.snr_db`` must be set (``load_config`` guarantees it).
    """
    if settings.snr_db is None:
        raise ValueError("noise settings without snr_db")
    samples, rate = audio_utils.decode_wav(audio_utils.ensure_wav(clean_wav))
    speech = audio_utils.to_mono(samples).astype(np.float64)
    above = np.flatnonzero(np.abs(speech) > SPEECH_THRESHOLD)
    if above.size == 0 or rate <= 0:
        return NoiseMix(clean_wav, None, False, None)
    start, end = int(above[0]), int(above[-1]) + 1
    length = min(speech.size, end + max(1, round(rate * FADE_OUT_MS / 1000)))

    rng = _rng(seed, turn)
    offset_ms: Optional[int] = None
    if settings.profile == "file":
        if noise_source is None:
            if not settings.file:
                raise ValueError("profile file without a noise file")
            noise_source = load_noise_file(settings.file)
        noise, offset_ms = _from_file(noise_source, length, rate, rng)
    else:
        noise = _generate(settings.profile, length, rate, rng)

    shaped = noise[:length] * _envelope(length, end, rate)
    speech_rms, noise_rms = _rms(speech[start:end]), _rms(shaped[start:end])
    if noise_rms == 0.0 or speech_rms == 0.0:
        return NoiseMix(clean_wav, None, False, offset_ms)
    gain = speech_rms / (noise_rms * 10.0 ** (float(settings.snr_db) / 20.0))
    mixed = speech.copy()
    mixed[:length] += gain * shaped

    scale, scaled = 1.0, False
    peak = float(np.max(np.abs(mixed)))
    if peak > CLIP_PEAK:
        scale, scaled = CLIP_PEAK / peak, True
        mixed *= scale
    wav = audio_utils.encode_wav(mixed, rate)

    out, _ = audio_utils.decode_wav(wav)
    reference = speech[start:end] * scale
    residual = audio_utils.to_mono(out).astype(np.float64)[start:end] * (32768.0 / 32767.0) - reference
    residual_rms = _rms(residual)
    measured = 20.0 * math.log10(_rms(reference) / residual_rms) if residual_rms > 0 else None
    return NoiseMix(wav, measured, scaled, offset_ms)
