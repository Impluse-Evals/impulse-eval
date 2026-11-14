# SPDX-License-Identifier: AGPL-3.0-only
"""``local_light`` voice provider: Kokoro ONNX TTS and faster-whisper STT, in-process, no PyTorch
(SPEC.md section 14; FIXES-ROUND-3-SPEC.md "B1 decisions").

Free, no server and no key. Needs the ``light`` extra (``pip install impulse-evals[light]``):
``kokoro-onnx`` (with ``onnxruntime``) for TTS, ``faster-whisper`` for STT; TTS and STT are chosen
independently, so each side needs only its own library. Building a provider checks that its library
is importable (``ConfigError`` at ``voice_config.{side}.provider`` otherwise) and loads nothing,
downloads nothing; the libraries are imported, the model files downloaded and the models loaded
on the first ``synthesize`` / ``transcribe``, in a worker thread (``asyncio.to_thread``).

* TTS: ``kokoro_onnx.Kokoro(model_path, voices_path)``, one per provider, reused. The text is
  phonemized with the engine's own tokenizer (``engine.tokenizer.phonemize(text, lang)``), espeak's
  language-switch markers such as ``(en)`` / ``(hi)`` are removed (Kokoro would speak them), and
  ``create(phonemes, voice=..., speed=1.0, lang=..., is_phonemes=True)`` returns float samples and
  their rate, written as a mono 16-bit PCM WAV at that rate. The voice is the ``voice`` argument,
  else ``voice_config.tts.voice``, else the default for the language's primary subtag
  (``DEFAULT_VOICES``; no ``ja`` default); the espeak ``lang`` comes from the voice id's first letter
  (``VOICE_LANGS``). A voice missing from the loaded voices file is
  ``VoiceError("local_light TTS: unknown Kokoro voice {voice}")``.
* Model files (``MODEL_FILES``, read when used): ``voice_config.tts.model`` null = the cache folder
  (``IMPULSE_CACHE_DIR``, else ``%LOCALAPPDATA%\\impulse-evals\\kokoro`` on Windows, else
  ``$XDG_CACHE_HOME/impulse-evals/kokoro`` or ``~/.cache/impulse-evals/kokoro``), downloaded from
  ``MODEL_BASE_URL`` on first use (streamed to ``<name>.part``, size and SHA-256 checked, then
  ``os.replace``); a file already present with the right size is used without hashing it again.
  Otherwise ``model`` is a folder holding both files (``load_config`` resolves a relative folder
  against the config file's folder); it is never downloaded into.
* STT: ``faster_whisper.WhisperModel(<model>, device=..., compute_type=...)`` (``small`` / ``cpu`` /
  ``int8`` by default), one per provider; the audio is decoded to mono 16 kHz numpy samples (never a
  file path); words and confidence exactly as ``local_hf`` (shared code).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import os
import re
import secrets
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from impulse_evals.config import ConfigError, is_placeholder
from impulse_evals.models import STTResult
from impulse_evals.voice import VoiceError, primary_subtag
from impulse_evals.voice.audio_utils import encode_wav, to_mono_16k
from impulse_evals.voice.local_hf import _is_importable, whisper_result, whisper_segments
from impulse_evals.voice.stt import STTProvider
from impulse_evals.voice.tts import TTSProvider

LIGHT_EXTRA_MISSING = "local_light needs the 'light' extra: pip install impulse-evals[light]"
PY314_KOKORO_NOTE = (
    " (kokoro-onnx 0.6 supports Python 3.11 to 3.13; on 3.14 install it with: "
    "pip install kokoro-onnx==0.6.1 --ignore-requires-python)"
)
NO_VOICE_FOR_LANGUAGE = (
    "local_light TTS has no voice for {language}. Kokoro speaks en, es, fr, hi, it, pt and zh; "
    "set voice_config.tts.voice or use another TTS provider (litellm, openai_compatible) for this language."
)
MISSING_MODEL_FILES = "local_light needs kokoro-v1.0.onnx and voices-v1.0.bin in {folder}"
CHECKSUM_MISMATCH = "local_light: downloaded {name} does not match its checksum; delete {folder} and try again"
DOWNLOAD_FAILED = "local_light: could not download {name} from {url}: {error}"
DOWNLOAD_NOTICE = "Downloading the light voice model ({mb} MB) to {folder} (first use only)."
UNKNOWN_VOICE_IN_FILE = "local_light TTS: unknown Kokoro voice {voice}"
NO_CACHE_FOLDER = "local_light cannot find a cache folder: set IMPULSE_CACHE_DIR or voice_config.tts.model"
# espeak language-switch markers such as (en), (hi), (cmn), (en-us), (pt-br): Kokoro would speak them.
LANGUAGE_MARKER_RE = re.compile(r"\([a-z]{2,3}(?:-[a-z0-9]+)*\)")

# Read when used (download, presence check, doctor), never copied at import.
MODEL_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
MODEL_FILES: dict[str, tuple[int, str]] = {
    "kokoro-v1.0.onnx": (325_505_369, "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a"),
    "voices-v1.0.bin": (28_214_398, "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}
MODEL_FILE_NAMES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")   # (model, voices) passed to Kokoro

# Default voice per primary language subtag (ids from hexgrad/Kokoro-82M VOICES.md).
DEFAULT_VOICES = {
    "en": "af_heart",
    "es": "ef_dora",
    "hi": "hf_alpha",
    "fr": "ff_siwis",
    "it": "if_sara",
    "pt": "pf_dora",
    "zh": "zf_xiaobei",
}
# No ``ja`` default (B1-rev L2): espeak's Japanese reads kanji as "Chinese letter"; an explicit ``j*``
# voice still works for kana-only text.
# espeak language passed to Kokoro.create, from the voice id's first letter.
VOICE_LANGS = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
    "j": "ja",
    "z": "cmn",
}

DEFAULT_STT_MODEL = "small"
DEFAULT_STT_DEVICE = "cpu"
DEFAULT_STT_COMPUTE_TYPE = "int8"
ERROR_MAX_CHARS = 200
_CHUNK_BYTES = 1 << 20
_DOWNLOAD_TIMEOUT_S = 60.0   # per read (the file is streamed); connect 30 s


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


def _one_line(text: str, limit: int = ERROR_MAX_CHARS) -> str:
    """Whitespace runs (newlines included) collapsed to one space, cut to ``limit`` characters."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def default_voice(language: str) -> Optional[str]:
    """The default Kokoro voice for ``language`` (by its primary subtag), or None when there is none."""
    return DEFAULT_VOICES.get(primary_subtag(language))


def voice_lang(voice: str) -> str:
    """The espeak language for a Kokoro voice id (from its first letter); an unknown letter is a
    ``ConfigError`` at ``voice_config.tts.voice``."""
    lang = VOICE_LANGS.get(voice[:1])
    if lang is None:
        raise ConfigError("voice_config.tts.voice", f"unknown Kokoro voice {voice}")
    return lang


def _text(value: Any) -> Optional[str]:
    """A stripped non-blank string, else None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def cache_folder() -> Path:
    """The model cache folder, read now: ``IMPULSE_CACHE_DIR`` (the folder itself) when set, else the
    platform cache (``%LOCALAPPDATA%\\impulse-evals\\kokoro``, ``$XDG_CACHE_HOME/impulse-evals/kokoro``
    or ``~/.cache/impulse-evals/kokoro``). No such folder (no home folder either) is the
    ``NO_CACHE_FOLDER`` ConfigError at ``voice_config.tts.model``."""
    try:
        explicit = _text(os.environ.get("IMPULSE_CACHE_DIR"))
        if explicit is not None:
            return Path(explicit).expanduser()
        if sys.platform == "win32":
            local = _text(os.environ.get("LOCALAPPDATA"))
            base = Path(local) if local is not None else Path.home() / "AppData" / "Local"
        else:
            xdg = _text(os.environ.get("XDG_CACHE_HOME"))
            base = Path(xdg) if xdg is not None else Path.home() / ".cache"
    except (RuntimeError, KeyError, OSError) as exc:   # "Could not determine home directory."
        raise ConfigError("voice_config.tts.model", NO_CACHE_FOLDER) from exc
    return base / "impulse-evals" / "kokoro"


def _mb(size: int) -> str:
    return f"{size / 1_000_000:.1f}"


def _missing_files(folder: Path) -> list[str]:
    """The model files not present in ``folder`` with their expected size (``MODEL_FILES``, read now)."""
    return _missing_files_named(folder, list(MODEL_FILES))


def _missing_files_named(folder: Path, names: list[str]) -> list[str]:
    missing = []
    for name in names:
        size = MODEL_FILES[name][0]
        path = folder / name
        try:
            if path.is_file() and path.stat().st_size == size:
                continue
        except OSError:
            pass
        missing.append(name)
    return missing


def _download_file(folder: Path, name: str) -> None:
    """Stream ``MODEL_BASE_URL + name`` to this download's own part file
    (``{name}.{pid}.{8 hex}.part``) in ``folder``, check its size and SHA-256, then ``os.replace`` it
    to ``name``. When that rename fails but ``name`` now has the right size (another process finished
    first), that file is used; any other rename failure is the "could not download" VoiceError. The
    part file never survives a failure."""
    import httpx  # lazy: only needed for a download

    size, sha256 = MODEL_FILES[name]
    url = MODEL_BASE_URL + name
    part = folder / f"{name}.{os.getpid()}.{secrets.token_hex(4)}.part"
    final = folder / name
    digest = hashlib.sha256()
    received = 0
    try:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            timeout = httpx.Timeout(_DOWNLOAD_TIMEOUT_S, connect=30.0)
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream("GET", url) as response:
                    if response.status_code >= 400:
                        reason = response.reason_phrase or ""
                        raise VoiceError(
                            _one_line(DOWNLOAD_FAILED.format(
                                name=name, url=url, error=f"HTTPStatusError: HTTP {response.status_code} {reason}".strip()
                            ))
                        )
                    with open(part, "wb") as handle:
                        for chunk in response.iter_bytes(_CHUNK_BYTES):
                            received += len(chunk)
                            if received > size:   # longer than the file: stop, it cannot match
                                break
                            digest.update(chunk)
                            handle.write(chunk)
        except VoiceError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise VoiceError(
                _one_line(DOWNLOAD_FAILED.format(name=name, url=url, error=_exception_text(exc)))
            ) from exc
        if received != size or digest.hexdigest() != sha256:
            raise VoiceError(CHECKSUM_MISMATCH.format(name=name, folder=str(folder)))
        try:
            os.replace(str(part), str(final))
        except OSError as exc:
            if not _missing_files_named(folder, [name]):   # another process finished first: use its file
                part.unlink(missing_ok=True)
                return
            raise VoiceError(
                _one_line(DOWNLOAD_FAILED.format(name=name, url=url, error=_exception_text(exc)))
            ) from exc
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise


def ensure_model_files(folder: Path) -> None:
    """Download the model files missing from the cache ``folder`` (one stderr line first)."""
    missing = _missing_files(folder)
    if not missing:
        return
    total = sum(MODEL_FILES[name][0] for name in missing)
    print(DOWNLOAD_NOTICE.format(mb=_mb(total), folder=str(folder)), file=sys.stderr, flush=True)
    for name in missing:
        _download_file(folder, name)


def _check_folder(folder: Path) -> None:
    """An explicit model folder must hold both files (no size or hash check, never downloaded into)."""
    if not all((folder / name).is_file() for name in MODEL_FILE_NAMES):
        raise ConfigError("voice_config.tts.model", MISSING_MODEL_FILES.format(folder=str(folder)))


def strip_language_markers(phonemes: str) -> str:
    """``phonemes`` without espeak's language-switch markers (``LANGUAGE_MARKER_RE``), whitespace
    collapsed (Kokoro collapses it too)."""
    return " ".join(LANGUAGE_MARKER_RE.sub("", phonemes).split())


def _voices_in_file(path: Path) -> Optional[set[str]]:
    """The voice ids in a Kokoro voices file (an ``.npz`` archive; only its directory is read), or
    None when it cannot be read."""
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            return set(archive.files)
    except Exception:  # noqa: BLE001 - not an archive / unreadable: the first synthesis reports it
        return None


def _engine_has_voice(engine: Any, voice: str) -> bool:
    """Whether the loaded Kokoro engine knows ``voice`` (kokoro-onnx 0.6.1: ``engine.voices``, the
    ``np.load`` of the voices file; else ``get_voices()``); True when neither can be read."""
    voices = getattr(engine, "voices", None)
    if voices is None and callable(getattr(engine, "get_voices", None)):
        voices = engine.get_voices()
    if voices is None:
        return True
    try:
        return voice in voices
    except TypeError:
        return True


def _import(module: str, side: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ConfigError(f"voice_config.{side}.provider", LIGHT_EXTRA_MISSING) from exc


# ------------------------------------------------------------------------------------------------
# Providers
# ------------------------------------------------------------------------------------------------


class LocalLightTTS(TTSProvider):
    name = "local_light"

    def __init__(self, block: Any) -> None:
        if not _is_importable("kokoro_onnx"):
            note = PY314_KOKORO_NOTE if sys.version_info >= (3, 14) else ""   # read now, not at import
            raise ConfigError("voice_config.tts.provider", LIGHT_EXTRA_MISSING + note)
        if not _is_importable("onnxruntime"):
            raise ConfigError("voice_config.tts.provider", LIGHT_EXTRA_MISSING)
        self.block = block
        self.voice = _text(getattr(block, "voice", None))
        if self.voice is not None:
            voice_lang(self.voice)   # an unknown first letter fails here, before the run
        model = _text(getattr(block, "model", None))
        self.folder: Optional[Path] = None
        if model is not None:
            folder = Path(model).expanduser()
            self.folder = folder if folder.is_absolute() else folder.absolute()
            _check_folder(self.folder)
        self._engine: Any = None
        self._lock = threading.Lock()

    def model_status(self) -> str:
        """For ``impulse doctor`` (nothing loaded or downloaded): ``present in {folder}`` or
        ``downloads on first run ({MB} MB)``; an explicit folder without both files is a ConfigError."""
        if self.folder is not None:
            _check_folder(self.folder)
            self._check_voice_in_file(self.folder)
            return f"present in {self.folder}"
        folder = cache_folder()
        missing = _missing_files(folder)
        if not missing:
            self._check_voice_in_file(folder)
            return f"present in {folder}"
        return f"downloads on first run ({_mb(sum(MODEL_FILES[name][0] for name in missing))} MB)"

    def _check_voice_in_file(self, folder: Path) -> None:
        """Doctor only: a configured voice missing from ``voices-v1.0.bin`` (read with ``np.load``, no
        model load) is a ConfigError at ``voice_config.tts.voice``; an unreadable file is not checked."""
        if self.voice is None:
            return
        voices = _voices_in_file(folder / MODEL_FILE_NAMES[1])
        if voices is not None and self.voice not in voices:
            raise ConfigError("voice_config.tts.voice", f"unknown Kokoro voice {self.voice}")

    def check_language(self, language: str) -> None:
        if self.voice is None and default_voice(language) is None:
            raise VoiceError(NO_VOICE_FOR_LANGUAGE.format(language=language))

    def voice_for(self, language: str, voice: Optional[str]) -> str:
        """The ``voice`` argument, else ``voice_config.tts.voice``, else the language default."""
        chosen = _text(voice) or self.voice or default_voice(language)
        if chosen is None:
            raise VoiceError(NO_VOICE_FOR_LANGUAGE.format(language=language))
        return chosen

    async def synthesize(self, text: str, language: str, voice: Optional[str]) -> bytes:
        chosen = self.voice_for(language, voice)
        lang = voice_lang(chosen)
        return await asyncio.to_thread(self._synthesize, text, chosen, lang)

    def _model_paths(self) -> tuple[Path, Path]:
        if self.folder is not None:
            folder = self.folder
            _check_folder(folder)
        else:
            folder = cache_folder()
            ensure_model_files(folder)
        model_name, voices_name = MODEL_FILE_NAMES
        return folder / model_name, folder / voices_name

    def _synthesize(self, text: str, voice: str, lang: str) -> bytes:
        with self._lock:
            model_path, voices_path = self._model_paths()
            try:
                if self._engine is None:
                    kokoro_onnx = _import("kokoro_onnx", "tts")
                    self._engine = kokoro_onnx.Kokoro(str(model_path), str(voices_path))
                engine = self._engine
                if not _engine_has_voice(engine, voice):
                    raise VoiceError(UNKNOWN_VOICE_IN_FILE.format(voice=voice))
                # Phonemize with kokoro-onnx's own tokenizer, drop espeak's language-switch markers
                # (Kokoro would speak "(en)" ... "(hi)" as syllables), then speak the phonemes.
                phonemes = strip_language_markers(engine.tokenizer.phonemize(text, lang))
                samples, rate = engine.create(phonemes, voice=voice, speed=1.0, lang=lang, is_phonemes=True)
            except (ConfigError, VoiceError):
                raise
            except Exception as exc:  # kokoro / onnxruntime / phonemizer errors
                raise VoiceError(
                    _one_line(f"local_light TTS failed with voice {voice}: {_exception_text(exc)}")
                ) from exc
        audio = np.squeeze(np.asarray(samples, dtype=np.float64))
        if audio.ndim > 1:   # several channels: mix them down
            audio = audio.mean(axis=0 if audio.shape[0] <= audio.shape[1] else 1)
        return encode_wav(audio.reshape(-1), int(rate))


class LocalLightSTT(STTProvider):
    name = "local_light"

    def __init__(self, block: Any) -> None:
        if not _is_importable("faster_whisper"):
            raise ConfigError("voice_config.stt.provider", LIGHT_EXTRA_MISSING)
        self.block = block
        self._model: Any = None
        self._lock = threading.Lock()

    def model_id(self) -> str:
        model = getattr(self.block, "model", None)
        return DEFAULT_STT_MODEL if is_placeholder(model) else model.strip()

    def _setting(self, name: str, default: str) -> str:
        value = _text(getattr(self.block, name, None)) or "auto"
        return default if value.lower() == "auto" else value

    async def transcribe(self, audio: bytes, language: str) -> STTResult:
        return await asyncio.to_thread(self._transcribe, audio, primary_subtag(language))

    def _load(self, model: str) -> Any:
        if self._model is None:
            faster_whisper = _import("faster_whisper", "stt")
            try:
                self._model = faster_whisper.WhisperModel(
                    model,
                    device=self._setting("device", DEFAULT_STT_DEVICE),
                    compute_type=self._setting("compute_type", DEFAULT_STT_COMPUTE_TYPE),
                )
            except Exception as exc:
                raise VoiceError(
                    _one_line(f"local_light STT could not load model {model}: {_exception_text(exc)}")
                ) from exc
        return self._model

    def _transcribe(self, audio: bytes, language: str) -> STTResult:
        model = self.model_id()
        samples = to_mono_16k(audio)
        with self._lock:
            whisper = self._load(model)
            segments = whisper_segments(whisper, samples, language, model, self.name)
        return whisper_result(segments)
