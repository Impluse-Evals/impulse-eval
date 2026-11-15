# SPDX-License-Identifier: Apache-2.0
"""Audio helpers for the voice pipeline (SPEC.md sections 5 and 14).

WAV files are read and written with the standard library ``wave`` module; mixing down to mono and
resampling (for example to 16 kHz for speech recognition) use numpy. pydub is used only to convert
audio that is not WAV (it needs the ``voice`` extra, and ffmpeg for most formats) and is imported
only then.

Sample arrays are float32 in [-1, 1]: 1-D for mono, ``(frames, channels)`` for several channels.
WAV files written here are 16-bit PCM; an IEEE-float WAV (which ``wave`` cannot read) is re-encoded
to 16-bit PCM by ``ensure_wav``.
"""

from __future__ import annotations

import io
import struct
import wave
from collections.abc import Iterable
from typing import Optional

import numpy as np

from impulse_evals.voice import VoiceError

STT_SAMPLE_RATE = 16000

VOICE_EXTRA_MISSING = (
    "audio that is not WAV needs the 'voice' extra (pydub, plus ffmpeg for most formats) to be "
    "converted: pip install impulse-evals[voice]"
)

# ------------------------------------------------------------------------------------------------
# Format detection and conversion
# ------------------------------------------------------------------------------------------------


def is_wav(data: bytes) -> bool:
    """True when ``data`` starts with a RIFF/WAVE header."""
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def upload_name(data: bytes) -> tuple[str, str]:
    """``(filename, content_type)`` for uploading ``data`` to a transcription server."""
    if is_wav(data):
        return "audio.wav", "audio/wav"
    if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio.mp3", "audio/mpeg"
    if data[:4] == b"OggS":
        return "audio.ogg", "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio.flac", "audio/flac"
    return "audio.wav", "application/octet-stream"


_UNKNOWN_SIZE = 0xFFFFFFFF   # the size streaming WAV writers put in the header (ffmpeg to a pipe, ...)


def _with_real_sizes(data: bytes) -> bytes:
    """A RIFF/WAVE file whose ``data`` chunk size is a streaming placeholder (0 or 0xFFFFFFFF) or larger
    than the bytes present, with the RIFF and ``data`` sizes rewritten to the bytes actually present;
    any other file (a well-formed one included) unchanged. Only a file the ``wave`` module can then read
    (PCM) is rewritten."""
    offset = 12
    while offset + 8 <= len(data):
        name = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if name == b"data":
            present = len(data) - (offset + 8)
            if size not in (0, _UNKNOWN_SIZE) and size <= present:
                return data
            fixed = bytearray(data)
            fixed[4:8] = (len(data) - 8).to_bytes(4, "little")
            fixed[offset + 4 : offset + 8] = present.to_bytes(4, "little")
            try:
                with wave.open(io.BytesIO(bytes(fixed)), "rb") as handle:
                    handle.getnframes()
            except (wave.Error, EOFError, ValueError):
                return data
            return bytes(fixed)
        offset += 8 + size + (size & 1)   # chunks are padded to an even size
    return data


_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
# KSDATAFORMAT_SUBTYPE_IEEE_FLOAT (00000003-0000-0010-8000-00aa00389b71) as stored in the fmt chunk.
_IEEE_FLOAT_SUBFORMAT = bytes.fromhex("0300000000001000800000aa00389b71")
_FLOAT_DTYPES = {32: "<f4", 64: "<f8"}


def _float_wav_to_pcm16(data: bytes) -> Optional[bytes]:
    """A RIFF/WAVE file with IEEE-float samples (format tag 3, or WAVE_FORMAT_EXTENSIBLE with the
    float sub-format; 32- or 64-bit) re-encoded as 16-bit PCM WAV with the same rate and channels
    (clipped to [-1, 1], scaled by 32767, rounded half to even); None for any other WAV. The ``wave``
    module reads PCM only (``unknown format: 3``), and torchaudio and several torch-based TTS
    wrappers write float WAV. A placeholder or oversized ``data`` size takes the bytes present."""
    fmt: Optional[bytes] = None
    offset = 12
    while offset + 8 <= len(data):
        name = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        body = offset + 8
        if name == b"fmt ":
            fmt = data[body : body + size]
        elif name == b"data":
            if fmt is None or len(fmt) < 16:
                return None
            tag, channels, rate, _byte_rate, _block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
            if tag == _WAVE_FORMAT_EXTENSIBLE:
                if len(fmt) < 40 or fmt[24:40] != _IEEE_FLOAT_SUBFORMAT:
                    return None
            elif tag != _WAVE_FORMAT_IEEE_FLOAT:
                return None
            if bits not in _FLOAT_DTYPES or channels < 1 or rate < 1:
                return None
            present = len(data) - body
            length = present if size in (0, _UNKNOWN_SIZE) or size > present else size
            frame_bytes = channels * bits // 8
            length -= length % frame_bytes
            samples = np.frombuffer(data[body : body + length], dtype=_FLOAT_DTYPES[bits]).astype(np.float64)
            samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
            pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
            return pcm16_to_wav(pcm.tobytes(), rate, channels)
        offset = body + size + (size & 1)   # chunks are padded to an even size
    return None


def ensure_wav(data: bytes) -> bytes:
    """``data`` when it is a well-formed PCM WAV; a PCM WAV with placeholder size fields (streaming
    TTS output) with its sizes rewritten to the bytes present; an IEEE-float WAV re-encoded as 16-bit
    PCM; other formats converted to WAV with pydub."""
    if is_wav(data):
        converted = _float_wav_to_pcm16(data)
        if converted is not None:
            return converted
        return _with_real_sizes(data)
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise VoiceError(VOICE_EXTRA_MISSING) from exc
    try:
        segment = AudioSegment.from_file(io.BytesIO(data))
        out = io.BytesIO()
        segment.export(out, format="wav")
    except Exception as exc:  # pydub raises CouldntDecodeError, OSError when ffmpeg is missing, ...
        raise VoiceError(f"cannot convert the audio to WAV: {type(exc).__name__}: {exc}") from exc
    return out.getvalue()


# ------------------------------------------------------------------------------------------------
# WAV read / write
# ------------------------------------------------------------------------------------------------


def wav_params(data: bytes) -> tuple[int, int, int]:
    """``(channels, sample_width_bytes, sample_rate)`` of a WAV file."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            return handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
    except (wave.Error, EOFError, ValueError) as exc:
        raise VoiceError(f"cannot read WAV audio: {exc}") from exc


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """``(samples, sample_rate)`` of a PCM WAV file; samples are float32 ``(frames, channels)``."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, ValueError) as exc:
        raise VoiceError(f"cannot read WAV audio: {exc}") from exc
    samples = _pcm_to_float(frames, width)
    usable = len(samples) - len(samples) % max(channels, 1)
    return samples[:usable].reshape(-1, max(channels, 1)), rate


def _pcm_to_float(frames: bytes, width: int) -> np.ndarray:
    usable = len(frames) - len(frames) % width
    raw = frames[:usable]
    if width == 1:  # 8-bit WAV is unsigned
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if width == 3:
        bytes3 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = bytes3[:, 0] | (bytes3[:, 1] << 8) | (bytes3[:, 2] << 16)
        values = np.where(values >= 1 << 23, values - (1 << 24), values)
        return values.astype(np.float32) / float(1 << 23)
    if width == 4:
        return (np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)).astype(np.float32)
    raise VoiceError(f"cannot read WAV audio with {width * 8}-bit samples")


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Float samples in [-1, 1] (clipped) as little-endian 16-bit integers; int16 input is kept."""
    array = np.asarray(samples)
    if array.dtype == np.int16:
        return array.astype("<i2")
    if np.issubdtype(array.dtype, np.integer):  # other integer PCM: scale by its own full range
        array = array.astype(np.float64) / float(np.iinfo(array.dtype).max)
    values = np.nan_to_num(np.asarray(array, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(np.round(values * 32767.0), -32768, 32767).astype("<i2")


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap 16-bit little-endian PCM bytes in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)
    return buffer.getvalue()


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """16-bit PCM WAV from samples (1-D mono, or ``(frames, channels)``)."""
    array = np.asarray(samples)
    if array.ndim == 1:
        channels = 1
    elif array.ndim == 2:
        channels = array.shape[1]
    else:
        raise ValueError(f"expected 1-D or 2-D samples, got shape {array.shape}")
    return pcm16_to_wav(float_to_pcm16(array).tobytes(), sample_rate, channels)


def duration_ms(data: bytes) -> float:
    """Length of a WAV file in milliseconds, from the sample bytes actually present (a header frame count
    can be a streaming placeholder such as 0xFFFFFFFF)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            frame_size = handle.getsampwidth() * handle.getnchannels()
            frames = len(handle.readframes(handle.getnframes())) // frame_size if frame_size else 0
            return 1000.0 * frames / rate if rate else 0.0
    except (wave.Error, EOFError, ValueError) as exc:
        raise VoiceError(f"cannot read WAV audio: {exc}") from exc


# ------------------------------------------------------------------------------------------------
# Mono, resampling, mixing
# ------------------------------------------------------------------------------------------------


def to_mono(samples: np.ndarray) -> np.ndarray:
    """1-D float32 mono signal: the mean of the channels."""
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.mean(axis=1, dtype=np.float32) if array.shape[1] > 1 else array[:, 0]
    raise ValueError(f"expected 1-D or 2-D samples, got shape {array.shape}")


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a 1-D signal from ``from_rate`` to ``to_rate`` Hz (band-limited, FFT based).

    Downsampling drops the frequencies above the new Nyquist limit, so no aliasing is added; the
    same rate returns the signal unchanged (as float32).
    """
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError(f"invalid sample rates {from_rate} -> {to_rate}")
    if from_rate == to_rate or signal.size == 0:
        return signal.astype(np.float32)
    count = signal.size
    out_count = max(1, round(count * to_rate / from_rate))
    spectrum = np.fft.rfft(signal)
    out_bins = out_count // 2 + 1
    kept = np.zeros(out_bins, dtype=np.complex128)
    shared = min(spectrum.size, out_bins)
    kept[:shared] = spectrum[:shared]
    return (np.fft.irfft(kept, out_count) * (out_count / count)).astype(np.float32)


def to_mono_16k(data: bytes) -> np.ndarray:
    """Audio (WAV, or another format pydub can convert) as a 1-D float32 mono signal at 16 kHz."""
    samples, rate = decode_wav(ensure_wav(data))
    return resample(to_mono(samples), rate, STT_SAMPLE_RATE)


def convert_wav(data: bytes, sample_rate: int) -> bytes:
    """Mono 16-bit PCM WAV at ``sample_rate``; a WAV already in that format is returned unchanged."""
    data = ensure_wav(data)
    if wav_params(data) == (1, 2, int(sample_rate)):
        return data
    samples, rate = decode_wav(data)
    return encode_wav(resample(to_mono(samples), rate, sample_rate), sample_rate)


def pcm16_frames(data: bytes) -> bytes:
    """The raw sample bytes of a 16-bit PCM WAV file (no conversion)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            if handle.getsampwidth() != 2:
                raise VoiceError("expected 16-bit PCM WAV audio")
            return handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, ValueError) as exc:
        raise VoiceError(f"cannot read WAV audio: {exc}") from exc


def peak_fraction(pcm: bytes) -> float:
    """Peak amplitude of 16-bit little-endian PCM bytes as a fraction of full scale (0.0 for none)."""
    usable = len(pcm) - len(pcm) % 2
    if usable == 0:
        return 0.0
    values = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.int32)
    return float(np.max(np.abs(values))) / 32768.0


def mix_stereo(
    left: Iterable[tuple[float, bytes]], right: Iterable[tuple[float, bytes]], sample_rate: int = STT_SAMPLE_RATE
) -> bytes:
    """A stereo 16-bit WAV recording: each ``(start_ms, wav_bytes)`` item of ``left`` / ``right`` is
    mixed down to mono, resampled to ``sample_rate``, and placed on its channel at ``start_ms``.

    Items that overlap on one channel are added (and clipped when written).

    Memory: the channels are accumulated in one preallocated float32 buffer and converted to 16-bit
    PCM once, in float64 chunks of ``_MIX_CHUNK_FRAMES`` frames (scale by 32767, round half to even,
    clip), so a long call never holds a float64 copy of the whole recording. When no two clips overlap
    on one channel, the samples are the same as ``encode_wav`` of a float64 mix; clips that overlap are
    summed in float32 and may differ from the float64 sum by one step (1 LSB) in rare cases. (A pure
    float32 scale and round would change some values even without overlap.)
    """
    tracks: list[list[tuple[int, np.ndarray]]] = []
    for items in (left, right):
        placed = []
        for start_ms, audio in items:
            samples, rate = decode_wav(ensure_wav(audio))
            mono = resample(to_mono(samples), rate, sample_rate)
            del samples
            offset = max(0, round(float(start_ms) * sample_rate / 1000.0))
            placed.append((offset, mono))
        tracks.append(placed)
    length = max((offset + mono.size for track in tracks for offset, mono in track), default=0)
    mixed = np.zeros((length, 2), dtype=np.float32)
    for channel, track in enumerate(tracks):
        for offset, mono in track:
            mixed[offset : offset + mono.size, channel] += mono
    del tracks
    pcm = np.empty((length, 2), dtype="<i2")
    for start in range(0, length, _MIX_CHUNK_FRAMES):
        chunk = mixed[start : start + _MIX_CHUNK_FRAMES].astype(np.float64)
        np.nan_to_num(chunk, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.multiply(chunk, 32767.0, out=chunk)
        np.rint(chunk, out=chunk)
        np.clip(chunk, -32768, 32767, out=chunk)
        np.copyto(pcm[start : start + _MIX_CHUNK_FRAMES], chunk, casting="unsafe")
    del mixed
    return pcm16_to_wav(pcm.tobytes(), sample_rate, 2)


_MIX_CHUNK_FRAMES = 65536   # mix_stereo: frames converted to 16-bit PCM per float64 chunk (1 MB)
