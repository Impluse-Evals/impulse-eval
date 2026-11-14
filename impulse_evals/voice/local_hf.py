# SPDX-License-Identifier: AGPL-3.0-only
"""``local_hf`` voice provider: open Hugging Face speech models running in-process (SPEC.md section 14.1).

Free default: no server and no key. Needs the ``local`` extra (``pip install impulse-evals[local]``);
torch is the user's own install choice, but TTS needs it: building the TTS provider without a
findable torch (``importlib.util.find_spec``) is a ``ConfigError`` at ``voice_config.tts.provider``
(``TORCH_MISSING``), and so is a torch that is installed but fails to import (``TORCH_IMPORT_FAILED``,
plus ``TORCH_WINDOWS_HINT`` on Windows). That import is tried in a separate Python interpreter, never
in Impulse's own process, so a torch that crashes the interpreter cannot end the run or the doctor.
STT (faster-whisper) needs no torch. The libraries are imported only when a provider is used.

* TTS: ``transformers.pipeline("text-to-speech", model=<id>, device=<device>)`` for ``model`` (or
  ``models_by_language[<scenario language>]``), one pipeline per model id per provider, cached. The
  pipeline output (``{"audio", "sampling_rate"}``) is written as a mono 16-bit PCM WAV at the
  model's sampling rate. ``device: auto`` picks CUDA, then Apple MPS, then CPU (through torch).
* STT: ``faster_whisper.WhisperModel(<model>, device=..., compute_type=...)``, one per model id,
  cached; the audio is mixed down to mono and resampled to 16 kHz with numpy and transcribed with
  ``word_timestamps=True`` and the primary language subtag. ``confidence`` = mean word probability,
  None when there are no words. An exception from inference (``transcribe`` or iterating its lazy
  segments) is ``VoiceError("local_hf STT failed with model {model}: {type}: {message}")``, like a
  load failure ("local_hf STT could not load model ..."), so the runner keeps the turn.

Model loading and inference run in a worker thread (one at a time per provider), so they never
block the event loop.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import math
import subprocess
import sys
import threading
import types
from typing import Any, Optional

import numpy as np

from impulse_evals.config import ConfigError, is_placeholder
from impulse_evals.models import STTResult, WordTiming
from impulse_evals.voice import VoiceError, primary_subtag
from impulse_evals.voice.audio_utils import encode_wav, to_mono_16k
from impulse_evals.voice.stt import STTProvider
from impulse_evals.voice.tts import TTSProvider

LOCAL_EXTRA_MISSING = "local_hf needs the 'local' extra: pip install impulse-evals[local]"
TORCH_MISSING = (
    "local_hf TTS needs PyTorch, which the 'local' extra does not install: pip install torch "
    "(see https://pytorch.org/get-started/locally/)"
)
TORCH_IMPORT_FAILED = "local_hf TTS needs PyTorch, which is installed but could not be imported ({detail})."
TORCH_WINDOWS_HINT = (
    " On Windows, a DLL load error (for example a missing vcruntime140_1.dll) means the Microsoft Visual C++ "
    "Redistributable (x64) needs to be installed or updated."
)
TORCH_PROBE_TIMEOUT_S = 120   # the separate interpreter that imports torch (check_torch)
TTS_TASK = "text-to-speech"
_REQUIRED_MODULE = {"tts": "transformers", "stt": "faster_whisper"}


# ------------------------------------------------------------------------------------------------
# Lazy imports
# ------------------------------------------------------------------------------------------------


def _missing_extra(side: str) -> ConfigError:
    return ConfigError(f"voice_config.{side}.provider", LOCAL_EXTRA_MISSING)


def _is_importable(module: str) -> bool:
    """True when ``module`` can be found, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except ValueError:  # already imported without a __spec__: importable
        return True
    except ImportError:
        return False


def check_extra(side: str) -> None:
    """Raise the missing-extra ``ConfigError`` when the library for ``side`` cannot be imported."""
    if not _is_importable(_REQUIRED_MODULE[side]):
        raise _missing_extra(side)


_DETAIL_MAX_CHARS = 300
_PROBE_STARTED = "IMPULSE_TORCH_PROBE_STARTED"
_PROBE_RESULT_PREFIX = "IMPULSE_TORCH_PROBE="
# Run as ``<sys.executable> -P -c _TORCH_PROBE <this process's sys.path as JSON>``: sets that
# sys.path, writes the start line, imports torch, writes one result line, then ends at once (no
# interpreter shutdown work). Both lines are ASCII (json.dumps escapes everything else) and go
# straight to file descriptor 1 with os.write, so a torch import that replaces sys.stdout cannot
# swallow them; each starts on a line of its own.
_TORCH_PROBE = f"""\
import json, os, sys
def _impulse_out(text):
    data = ("\\n" + text + "\\n").encode("ascii")
    while data:
        data = data[os.write(1, data):]
sys.path[:] = json.loads(sys.argv[1])
_impulse_out("{_PROBE_STARTED}")
try:
    import torch
except Exception as exc:
    try:
        message = str(exc)
    except Exception:
        message = ""
    result = {{
        "ok": False,
        "type": type(exc).__name__,
        "message": message,
        "name": exc.name if isinstance(exc, ModuleNotFoundError) else None,
    }}
else:
    result = {{"ok": True}}
_impulse_out("{_PROBE_RESULT_PREFIX}" + json.dumps(result))
os._exit(0)
"""


def _probe_lines(stdout: bytes) -> list[str]:
    return [line.strip() for line in stdout.decode("utf-8", errors="replace").splitlines()]


def _probe_result(lines: list[str]) -> Optional[dict[str, Any]]:
    """The probe's result line (the last one), or None when there is none or it is not a JSON object."""
    for line in reversed(lines):
        if line.startswith(_PROBE_RESULT_PREFIX):
            try:
                result = json.loads(line[len(_PROBE_RESULT_PREFIX):])
            except ValueError:
                return None
            return result if isinstance(result, dict) else None
    return None


def _torch_import_failure() -> Optional[str]:
    """Import torch in a separate Python interpreter (``sys.executable``, this process's ``sys.path``)
    and return what went wrong, or None when the import worked or the answer is inconclusive.

    Inconclusive (None): no ``sys.executable``, a child that cannot be started or does not finish
    within ``TORCH_PROBE_TIMEOUT_S``, a ``ModuleNotFoundError`` for torch itself (the child cannot find
    what this process finds; the import at synthesis decides), a child that ends without a result
    line and without the start line (it never reached ``import torch``: the interpreter could not
    start, e.g. a bad ``PYTHONHOME``, or ``sys.executable`` is no Python, e.g. a frozen or embedded
    host), or one that ends with exit code 0 without a result line. A failure: any other exception
    (``"{type name}: {message}"``), or a child that wrote the start line and then ended with a
    non-zero exit code and no result line (``import torch`` crashed the interpreter).

    Limitation: the child starts from a clean interpreter, so it cannot reproduce runtime-only state
    of this process (``os.add_dll_directory`` calls, custom import hooks, ...); a torch that imports
    only with that state can get a false failure here."""
    if not sys.executable:
        return None
    path = json.dumps([entry for entry in sys.path if isinstance(entry, str)])
    try:
        completed = subprocess.run(
            [sys.executable, "-P", "-c", _TORCH_PROBE, path],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=TORCH_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):   # cannot start, or timed out
        return None
    lines = _probe_lines(completed.stdout or b"")
    result = _probe_result(lines)
    if result is None:
        if _PROBE_STARTED in lines and completed.returncode != 0:
            return f"the import ended the Python process with exit code {completed.returncode}"
        return None   # never reached the import, or exit code 0: inconclusive
    if result.get("ok") is True:
        return None
    kind, message = result.get("type"), result.get("message")
    if kind == "ModuleNotFoundError" and result.get("name") == "torch":
        return None
    return f"{kind}: {message}"


def _torch_import_failed_message(detail: str) -> str:
    """``TORCH_IMPORT_FAILED`` for ``detail`` (whitespace runs collapsed to one space, at most 300
    characters: longer is cut to 297 + "..."), plus ``TORCH_WINDOWS_HINT`` when ``sys.platform`` is
    ``win32`` (read now)."""
    flat = " ".join(detail.split())
    if len(flat) > _DETAIL_MAX_CHARS:
        flat = flat[: _DETAIL_MAX_CHARS - 3] + "..."
    message = TORCH_IMPORT_FAILED.format(detail=flat)
    if sys.platform == "win32":
        message += TORCH_WINDOWS_HINT
    return message


def check_torch() -> None:
    """Raise a ``ConfigError`` at ``voice_config.tts.provider`` when torch cannot be used.

    transformers' text-to-speech pipelines run on torch, which the ``local`` extra does not install;
    without this check the error would surface inside the first synthesis of every scenario, after
    the first paid call. In order:

    1. torch not findable (``importlib.util.find_spec``): ``TORCH_MISSING``.
    2. torch already imported in this process (a module in ``sys.modules``): OK.
    3. otherwise ``import torch`` runs in a separate Python interpreter (``_torch_import_failure``);
       a failure there is ``_torch_import_failed_message(detail)``.

    torch is never imported in Impulse's own process here, so a torch that crashes the interpreter
    (an access violation, an abort) cannot end the run or the doctor. Nothing is cached.
    """
    if not _is_importable("torch"):
        raise ConfigError("voice_config.tts.provider", TORCH_MISSING)
    if isinstance(sys.modules.get("torch"), types.ModuleType):
        return
    detail = _torch_import_failure()
    if detail is not None:
        raise ConfigError("voice_config.tts.provider", _torch_import_failed_message(detail))


def _import(module: str, side: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise _missing_extra(side) from exc
    except Exception as exc:  # ImportError, OSError (e.g. a missing system DLL), ...
        raise VoiceError(f"local_hf could not import {module}: {type(exc).__name__}: {exc}") from exc


def _tts_device(device: Optional[str]) -> str:
    """The transformers ``device`` for a configured value; ``auto`` asks torch (CUDA, then MPS, then CPU)."""
    value = (device or "auto").strip()
    if value.lower() != "auto":
        return value
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001, S110 - torch missing or unusable: transformers reports it when it needs torch
        pass
    return "cpu"


def _mono_float(audio: Any) -> np.ndarray:
    """A TTS pipeline's audio (any shape, float or integer PCM) as a 1-D float signal in [-1, 1]."""
    array = np.asarray(audio)
    if array.dtype == object or array.size == 0:
        raise VoiceError("local_hf TTS: the pipeline returned no audio")
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float64) / float(np.iinfo(array.dtype).max)
    array = np.squeeze(array.astype(np.float64))
    if array.ndim == 2:  # several channels, (channels, frames) or (frames, channels): mix them down
        array = array.mean(axis=0 if array.shape[0] <= array.shape[1] else 1)
    return array.reshape(-1)


def _number(value: Any) -> Optional[float]:
    """A finite float from a number (numpy scalars included); None for anything else."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ------------------------------------------------------------------------------------------------
# Providers
# ------------------------------------------------------------------------------------------------


class LocalHFTTS(TTSProvider):
    name = "local_hf"

    def __init__(self, block: Any) -> None:
        check_extra("tts")
        check_torch()   # built before the first paid call (runner and doctor): fail here, not in synthesize
        self.block = block
        self._pipelines: dict[str, Any] = {}
        self._lock = threading.Lock()

    def model_for(self, language: str) -> str:
        """``models_by_language[language]`` when set, else ``model``; neither set is a ConfigError."""
        model = self.block.models_by_language.get(language)
        if is_placeholder(model):
            model = self.block.model
        if is_placeholder(model):
            raise ConfigError(
                "voice_config.tts.model",
                f"no TTS model for language {language}. Set voice_config.tts.model (a Hugging Face model id) "
                f"or voice_config.tts.models_by_language for {language}.",
            )
        return model.strip()

    def check_language(self, language: str) -> None:
        self.model_for(language)   # no model for the language: ConfigError at voice_config.tts.model

    async def synthesize(self, text: str, language: str, voice: Optional[str]) -> bytes:
        model = self.model_for(language)
        return await asyncio.to_thread(self._synthesize, model, text)

    def _pipeline(self, model: str) -> Any:
        pipeline = self._pipelines.get(model)
        if pipeline is None:
            transformers = _import("transformers", "tts")
            try:
                pipeline = transformers.pipeline(TTS_TASK, model=model, device=_tts_device(self.block.device))
            except Exception as exc:
                raise VoiceError(f"local_hf TTS could not load model {model}: {type(exc).__name__}: {exc}") from exc
            self._pipelines[model] = pipeline
        return pipeline

    def _synthesize(self, model: str, text: str) -> bytes:
        with self._lock:
            output = self._pipeline(model)(text)
        if isinstance(output, list):
            output = output[0] if output else {}
        if not isinstance(output, dict) or output.get("audio") is None:
            raise VoiceError(f"local_hf TTS: model {model} returned no audio")
        rate = _number(output.get("sampling_rate"))
        if rate is None or rate <= 0:
            raise VoiceError(f"local_hf TTS: model {model} returned no sampling rate")
        return encode_wav(_mono_float(output["audio"]), round(rate))


class LocalHFSTT(STTProvider):
    name = "local_hf"

    def __init__(self, block: Any) -> None:
        check_extra("stt")
        self.block = block
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def model_id(self) -> str:
        model = self.block.model
        if is_placeholder(model):
            raise ConfigError(
                "voice_config.stt.model",
                "no STT model set. Set voice_config.stt.model to a faster-whisper model size or Hugging Face model id.",
            )
        return model.strip()

    def check_language(self, language: str) -> None:
        self.model_id()   # no STT model: ConfigError at voice_config.stt.model

    async def transcribe(self, audio: bytes, language: str) -> STTResult:
        model = self.model_id()
        return await asyncio.to_thread(self._transcribe, model, audio, primary_subtag(language))

    def _model(self, model: str) -> Any:
        whisper = self._models.get(model)
        if whisper is None:
            faster_whisper = _import("faster_whisper", "stt")
            try:
                whisper = faster_whisper.WhisperModel(
                    model, device=self.block.device or "auto", compute_type=self.block.compute_type or "auto"
                )
            except Exception as exc:
                raise VoiceError(f"local_hf STT could not load model {model}: {type(exc).__name__}: {exc}") from exc
            self._models[model] = whisper
        return whisper

    def _transcribe(self, model: str, audio: bytes, language: str) -> STTResult:
        samples = to_mono_16k(audio)
        with self._lock:
            whisper = self._model(model)   # a load failure keeps its own VoiceError text
            segments = whisper_segments(whisper, samples, language, model, self.name)
        return whisper_result(segments)


# ------------------------------------------------------------------------------------------------
# faster-whisper transcription, shared with ``local_light`` (same STTResult rules, same error text)
# ------------------------------------------------------------------------------------------------


def whisper_segments(whisper: Any, samples: np.ndarray, language: str, model: str, provider: str) -> list[Any]:
    """Run ``whisper.transcribe`` on mono 16 kHz ``samples`` (never a file path) with word timings and
    return its segments as a list. An exception from inference (``transcribe`` or iterating its lazy
    segments) is ``VoiceError("{provider} STT failed with model {model}: {type}: {message}")``."""
    try:
        segments, _info = whisper.transcribe(samples, language=language, word_timestamps=True)
        return list(segments)  # the generator does the decoding work (and can raise there)
    except VoiceError:
        raise
    except Exception as exc:  # e.g. CTranslate2 "Library cublas64_12.dll is not found" at the first decode
        raise VoiceError(f"{provider} STT failed with model {model}: {type(exc).__name__}: {exc}") from exc


def whisper_result(segments: list[Any]) -> STTResult:
    """``STTResult`` from faster-whisper segments: segment texts stripped and joined; words stripped,
    blank or timing-less words dropped; ``confidence`` = mean word probability, None without words."""
    texts: list[str] = []
    words: list[WordTiming] = []
    probabilities: list[float] = []
    for segment in segments:
        text = (getattr(segment, "text", "") or "").strip()
        if text:
            texts.append(text)
        for item in getattr(segment, "words", None) or []:
            word = (getattr(item, "word", "") or "").strip()
            start, end = _number(getattr(item, "start", None)), _number(getattr(item, "end", None))
            if not word or start is None or end is None:
                continue
            probability = _number(getattr(item, "probability", None))
            if probability is not None:
                probabilities.append(probability)
            words.append(
                WordTiming(
                    word=word,
                    start_ms=round(start * 1000.0),
                    end_ms=round(end * 1000.0),
                    confidence=probability if probability is not None else 0.0,
                )
            )
    confidence = sum(probabilities) / len(probabilities) if probabilities else None
    return STTResult(text=" ".join(texts), confidence=confidence, words=words)
