# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 3, B1: the light local voice provider ``local_light`` and the ``light`` extra
(FIXES-ROUND-3-SPEC.md "B1 decisions").

* ``impulse_evals.voice.local_light``: ``LocalLightTTS`` (Kokoro ONNX) and ``LocalLightSTT``
  (faster-whisper), returned by ``get_tts_provider`` / ``get_stt_provider`` for
  ``provider: local_light`` (TTS and STT independently).
* Missing extra: ``ConfigError("voice_config.{side}.provider", "local_light needs the 'light' extra:
  pip install impulse-evals[light]")`` when the provider is built; on Python >= 3.14 with kokoro_onnx
  missing the TTS text adds the kokoro-onnx 0.6 note.
* Voices: the default voice per primary subtag, any Kokoro voice id by config or argument, the espeak
  ``lang`` from the voice id's first letter; ``check_language``; unknown voice letter.
* Synthesis in a worker thread (``asyncio.to_thread``), mono 16-bit PCM WAV at the engine rate.
* Model files: the cache folder (``IMPULSE_CACHE_DIR``, else the platform cache), a relative
  ``voice_config.tts.model`` folder, the download (``.part`` file, size and SHA-256, ``os.replace``,
  the stderr line, mismatch and network error texts, no second download).
* STT: ``small`` / ``cpu`` / ``int8`` defaults, numpy input (never a path), words and confidence as
  ``local_hf``; inference failure text.
* torch is never imported; ``impulse doctor`` rows; the ``light`` extra in pyproject; SPDX header.

No network, no real model: fake ``kokoro_onnx``, ``onnxruntime`` and ``faster_whisper`` modules are
put in ``sys.modules``; the model download is served by an ``httpx.MockTransport`` installed into every
new ``httpx.Client`` / ``httpx.AsyncClient`` (as ``mock_voice_server`` does).

Seams (tests/data/fixes_round3/NOTES_stageB_local_light.md): ``local_light.MODEL_BASE_URL`` (str) and
``local_light.MODEL_FILES`` (``{file name: (size in bytes, sha256 hex)}``, read when used, so a test can
replace it with small files); ``sys.version_info`` read when the provider is built.
"""

import asyncio
import collections
import contextlib
import hashlib
import inspect
import os
import re
import subprocess
import sys
import threading
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
from conftest import REPO_ROOT, pcm_samples, pcm_tone, read_wav, sine_wav, wav_bytes

MODEL_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
ONNX = "kokoro-v1.0.onnx"
VOICES = "voices-v1.0.bin"
ONNX_SIZE = 325_505_369
VOICES_SIZE = 28_214_398
ONNX_SHA256 = "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a"
VOICES_SHA256 = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"

LIGHT_EXTRA_MISSING = "local_light needs the 'light' extra: pip install impulse-evals[light]"
PY314_NOTE = (
    " (kokoro-onnx 0.6 supports Python 3.11 to 3.13; on 3.14 install it with: "
    "pip install kokoro-onnx==0.6.1 --ignore-requires-python)"
)
NO_VOICE = (
    "local_light TTS has no voice for {language}. Kokoro speaks en, es, fr, hi, it, pt and zh; "
    "set voice_config.tts.voice or use another TTS provider (litellm, openai_compatible) for this language."
)
MISSING_FILES = "local_light needs kokoro-v1.0.onnx and voices-v1.0.bin in {folder}"
CHECKSUM = "local_light: downloaded {name} does not match its checksum; delete {folder} and try again"
COULD_NOT_DOWNLOAD = "local_light: could not download {name} from {url}: {error}"
DOWNLOAD_LINE = re.compile(r"^Downloading the light voice model \((\d+(?:\.\d+)?) MB\) to (.+) \(first use only\)\.$")
TOTAL_MB_RANGE = (337.0, 354.0)   # 353,719,767 bytes: 353.7 MB or 337.3 MiB, rounded or not

ENGINE_RATE = 24000
DEFAULT_VOICES = [
    # (scenario language, default voice, espeak lang)
    ("en-US", "af_heart", "en-us"),
    ("en-GB", "af_heart", "en-us"),
    ("en-IN", "af_heart", "en-us"),
    ("es-MX", "ef_dora", "es"),
    ("es", "ef_dora", "es"),
    ("hi-IN", "hf_alpha", "hi"),
    ("fr-FR", "ff_siwis", "fr-fr"),
    ("it-IT", "if_sara", "it"),
    ("pt-BR", "pf_dora", "pt-br"),
    ("zh-CN", "zf_xiaobei", "cmn"),
]
VOICE_LETTERS = [
    # (any Kokoro voice id, espeak lang from its first letter)
    ("am_adam", "en-us"),
    ("bf_emma", "en-gb"),
    ("em_alex", "es"),
    ("ff_siwis", "fr-fr"),
    ("hm_omega", "hi"),
    ("im_nicola", "it"),
    ("pm_alex", "pt-br"),
    ("jm_kumo", "ja"),
    ("zm_yunjian", "cmn"),
]


# ---------------------------------------------------------------------------------------------
# fake libraries
# ---------------------------------------------------------------------------------------------


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, None)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _sine(seconds, rate, amplitude=0.5, frequency=330.0):
    import numpy

    count = int(round(seconds * rate))
    return (amplitude * numpy.sin(2 * numpy.pi * frequency * numpy.arange(count) / rate)).astype(numpy.float32)


# Every Kokoro voice id these tests use; the fake engine knows them when its voices file is not a
# real .npz archive (the tests' small placeholder files). A real archive: exactly its entries.
KNOWN_VOICES = (
    "af_heart", "af_bella", "ef_dora", "hf_alpha", "hf_beta", "ff_siwis", "if_sara", "pf_dora", "jf_alpha",
    "zf_xiaobei", "am_adam", "bf_emma", "em_alex", "hm_omega", "im_nicola", "pm_alex", "jm_kumo", "zm_yunjian",
)


def _load_voices(voices_path):
    import numpy

    try:
        with numpy.load(str(voices_path), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except Exception:  # noqa: BLE001 - a placeholder file: every known voice
        return {name: numpy.zeros((1, 1, 4), dtype=numpy.float32) for name in KNOWN_VOICES}


class FakeKokoro:
    """``kokoro_onnx.Kokoro(model_path, voices_path)`` shaped like kokoro-onnx 0.6.1 (checked against
    the installed package): ``voices`` (loaded from the voices file, ``np.load``), ``get_voices()``,
    ``tokenizer.phonemize(text, lang="en-us", norm=True) -> str``, and ``create(text, voice, speed=1.0,
    lang="en-us", is_phonemes=False, trim=True, ...) -> (float32 samples, rate)``; ``create`` phonemizes
    the text itself unless ``is_phonemes`` and raises ``ValueError("Voice {voice} not found in available
    voices")`` for an unknown voice, as kokoro-onnx does. ``phonemes[text]`` scripts the tokenizer
    (default: the text unchanged). Records constructor, phonemize and create calls with their threads;
    ``create_error`` / ``init_error`` make them raise."""

    def __init__(self, rate=ENGINE_RATE, seconds=0.25):
        self.rate = rate
        self.samples = _sine(seconds, rate)
        self.inits = []
        self.creates = []
        self.phonemize_calls = []
        self.phonemes = {}
        self.create_error = None
        self.init_error = None

    def module(self):
        outer = self

        class Tokenizer:
            def phonemize(self, text, lang="en-us", norm=True):
                outer.phonemize_calls.append({"text": text, "lang": lang, "thread": threading.get_ident()})
                return outer.phonemes.get(text, text)

        class Kokoro:
            def __init__(self, model_path, voices_path, espeak_config=None, vocab_config=None):
                outer.inits.append(
                    {"model_path": str(model_path), "voices_path": str(voices_path), "thread": threading.get_ident()}
                )
                if outer.init_error is not None:
                    raise outer.init_error
                self.voices = _load_voices(voices_path)
                self.tokenizer = Tokenizer()

            def get_voices(self):
                return sorted(self.voices.keys())

            def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False, trim=True,
                       sentence_pause=0.25, clause_pause=0.1, continuous=False):
                spoken = text if is_phonemes else outer.phonemes.get(text, text)
                outer.creates.append(
                    {
                        "text": text,
                        "voice": voice,
                        "speed": speed,
                        "lang": lang,
                        "is_phonemes": is_phonemes,
                        "spoken": spoken,
                        "thread": threading.get_ident(),
                    }
                )
                if outer.create_error is not None:
                    raise outer.create_error
                if isinstance(voice, str) and voice not in self.voices:
                    raise ValueError(f"Voice {voice} not found in available voices")
                return outer.samples.copy(), outer.rate

        return _module("kokoro_onnx", Kokoro=Kokoro)


def _fake_onnxruntime():
    class InferenceSession:  # never used by the fake Kokoro
        def __init__(self, *args, **kwargs):
            raise AssertionError("the fake onnxruntime session must not be created")

    return _module(
        "onnxruntime",
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=InferenceSession,
    )


def _word(text, start, end, probability):
    return types.SimpleNamespace(word=text, start=start, end=end, probability=probability)


def _segment(text, words):
    start = words[0].start if words else 0.0
    end = words[-1].end if words else 0.0
    return types.SimpleNamespace(text=text, start=start, end=end, words=words)


DEFAULT_SEGMENTS = [
    _segment(" book a slot", [_word(" book", 0.0, 0.5, 0.9), _word(" a", 0.5, 0.75, 0.6), _word(" slot", 0.75, 1.5, 0.9)])
]


class FakeWhisper:
    """``faster_whisper.WhisperModel`` with faster-whisper's constructor signature; records the bound
    constructor arguments, every ``transcribe`` input and the threads they ran on."""

    def __init__(self, segments=None):
        self.segments = list(DEFAULT_SEGMENTS if segments is None else segments)
        self.inits = []
        self.transcribes = []
        self.transcribe_error = None
        self.iterate_error = None

    def module(self):
        outer = self

        class WhisperModel:
            def __init__(
                self,
                model_size_or_path,
                device="auto",
                device_index=0,
                compute_type="default",
                cpu_threads=0,
                num_workers=1,
                download_root=None,
                local_files_only=False,
                files=None,
                **model_kwargs,
            ):
                outer.inits.append(
                    {
                        "model": model_size_or_path,
                        "device": device,
                        "compute_type": compute_type,
                        "thread": threading.get_ident(),
                    }
                )

            def transcribe(self, audio, language=None, task="transcribe", **kwargs):
                outer.transcribes.append(
                    {"audio": audio, "language": language, "kwargs": kwargs, "thread": threading.get_ident()}
                )
                if outer.transcribe_error is not None:
                    raise outer.transcribe_error
                info = types.SimpleNamespace(language=language, language_probability=0.99, duration=1.0)

                def segments():
                    for segment in outer.segments:
                        yield segment
                    if outer.iterate_error is not None:
                        raise outer.iterate_error

                return segments(), info

        return _module("faster_whisper", WhisperModel=WhisperModel)


# ---------------------------------------------------------------------------------------------
# model download server (httpx.MockTransport in every new httpx client)
# ---------------------------------------------------------------------------------------------


class ModelServer:
    """Serves ``routes[url]``: bytes (200, octet-stream), an ``httpx.Response``, or an exception
    (raised as the transport error). Any other URL answers 404. Every request is recorded."""

    def __init__(self):
        import httpx

        self.requests = []
        self.routes = {}
        self.transport = httpx.MockTransport(self._handle)

    def serve(self, files):
        """``files``: ``{name: body}`` served at ``MODEL_BASE_URL + name``."""
        for name, body in files.items():
            self.routes[MODEL_BASE_URL + name] = body

    def urls(self):
        return [record["url"] for record in self.requests]

    def _handle(self, request):
        import httpx

        url = str(request.url)
        self.requests.append({"method": request.method, "url": url})
        answer = self.routes.get(url)
        if answer is None:
            return httpx.Response(404, text="Not Found")
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, content=answer, headers={"content-type": "application/octet-stream"})

    def install(self, monkeypatch):
        import httpx

        transport = self.transport
        real_client_init = httpx.Client.__init__
        real_async_init = httpx.AsyncClient.__init__

        def client_init(client, *args, **kwargs):
            if kwargs.get("transport") is None:
                kwargs["transport"] = transport
            real_client_init(client, *args, **kwargs)

        def async_init(client, *args, **kwargs):
            if kwargs.get("transport") is None:
                kwargs["transport"] = transport
            real_async_init(client, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", client_init)
        monkeypatch.setattr(httpx.AsyncClient, "__init__", async_init)


# ---------------------------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def light(impulse_project, extra_imports, monkeypatch, tmp_path):
    """Fake ``kokoro_onnx`` / ``onnxruntime`` / ``faster_whisper``, ``IMPULSE_CACHE_DIR`` pointed at
    ``tmp_path / "kokoro-cache"`` (not created), and the model download server installed."""
    fakes = types.SimpleNamespace(
        project=impulse_project,
        kokoro=FakeKokoro(),
        whisper=FakeWhisper(),
        server=ModelServer(),
        cache=tmp_path / "kokoro-cache",
    )

    def install():
        extra_imports.module("kokoro_onnx", fakes.kokoro.module())
        extra_imports.module("onnxruntime", _fake_onnxruntime())
        extra_imports.module("faster_whisper", fakes.whisper.module())

    fakes.install = install
    install()
    monkeypatch.setenv("IMPULSE_CACHE_DIR", str(fakes.cache))
    fakes.server.install(monkeypatch)
    return fakes


def _mod():
    from impulse_evals.voice import local_light

    return local_light


def _voice_block(tts=None, stt=None):
    return {
        "tts": dict(tts if tts is not None else {"provider": "local_light"}),
        "stt": dict(stt if stt is not None else {"provider": "local_light"}),
        "save_audio": False,
        "save_recording": False,
        "timeout_s": 30,
    }


def _voice_cfg(project, tts=None, stt=None, path=None, language="en-US"):
    """Write a config with a ``local_light`` voice block and return the loaded ``VoiceConfig``."""
    extra = {} if path is None else {"path": path, "system_prompt": None}   # another folder: no prompt file there
    config = project.write_config(
        [project.scenario("light_probe", language=language)], voice_config=_voice_block(tts, stt), **extra
    )
    return project.load(config).voice_config


def _tts(cfg):
    from impulse_evals.voice.tts import get_tts_provider

    return get_tts_provider(cfg)


def _stt(cfg):
    from impulse_evals.voice.stt import get_stt_provider

    return get_stt_provider(cfg)


_VersionInfo = collections.namedtuple("version_info", "major minor micro releaselevel serial")


@contextlib.contextmanager
def _python(minor):
    """``sys.version_info`` reads as 3.{minor} inside the block (the provider reads it when built)."""
    real = sys.version_info
    sys.version_info = _VersionInfo(3, minor, 0, "final", 0)
    try:
        yield
    finally:
        sys.version_info = real


def _same_path(a, b):
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _small_model_files(monkeypatch):
    """Replace ``MODEL_FILES`` with two small files; returns ``{name: body}`` with the right hashes."""
    bodies = {ONNX: b"fake-kokoro-onnx-model-" * 40, VOICES: b"fake-kokoro-voices-" * 25}
    files = {name: (len(body), hashlib.sha256(body).hexdigest()) for name, body in bodies.items()}
    monkeypatch.setattr(_mod(), "MODEL_FILES", files)
    return bodies


def _place(folder, files):
    folder.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (folder / name).write_bytes(body)


def _download_lines(err):
    return [line for line in err.splitlines() if line.startswith("Downloading the light voice model")]


def _one_line(text, limit=200):
    return "\n" not in text and "\r" not in text and len(text) <= limit


# =============================================================================================
# Module, factories, extras
# =============================================================================================


def test_model_file_constants_are_the_spec_values():
    module = _mod()
    assert module.MODEL_BASE_URL == MODEL_BASE_URL
    assert dict(module.MODEL_FILES) == {ONNX: (ONNX_SIZE, ONNX_SHA256), VOICES: (VOICES_SIZE, VOICES_SHA256)}
    assert list(module.MODEL_FILES) == [ONNX, VOICES]


def test_local_light_module_starts_with_the_agpl_header():
    path = REPO_ROOT / "impulse_evals" / "voice" / "local_light.py"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first == "# SPDX-License-Identifier: AGPL-3.0-only"


def test_factories_return_local_light_providers_without_loading_or_downloading(light):
    from impulse_evals.voice.local_light import LocalLightSTT, LocalLightTTS
    from impulse_evals.voice.stt import STTProvider
    from impulse_evals.voice.tts import TTSProvider

    cfg = _voice_cfg(light.project)
    tts, stt = _tts(cfg), _stt(cfg)
    assert isinstance(tts, LocalLightTTS) and isinstance(tts, TTSProvider)
    assert isinstance(stt, LocalLightSTT) and isinstance(stt, STTProvider)
    assert tts.name == "local_light" and stt.name == "local_light"
    # Loaded lazily and downloaded on first use: building loads nothing and downloads nothing.
    assert light.kokoro.inits == [] and light.whisper.inits == []
    assert light.server.requests == []
    assert not light.cache.exists() or not any(light.cache.iterdir())


def test_tts_and_stt_are_chosen_independently(light, extra_imports):
    from impulse_evals.voice.local_light import LocalLightSTT, LocalLightTTS

    other = {"provider": "openai_compatible", "model": "impulse-test-model", "base_url": "http://localhost:8000/v1"}
    # TTS only: faster-whisper is not needed.
    extra_imports.missing("faster_whisper")
    cfg = _voice_cfg(light.project, stt=other)
    assert isinstance(_tts(cfg), LocalLightTTS)
    assert not isinstance(_stt(cfg), LocalLightSTT)
    # STT only: kokoro-onnx and onnxruntime are not needed.
    light.install()
    extra_imports.missing("kokoro_onnx")
    extra_imports.missing("onnxruntime")
    cfg = _voice_cfg(light.project, tts=other)
    assert isinstance(_stt(cfg), LocalLightSTT)
    assert not isinstance(_tts(cfg), LocalLightTTS)


@pytest.mark.parametrize("missing", ["kokoro_onnx", "onnxruntime"])
def test_tts_without_the_light_extra_is_a_config_error_when_built(light, extra_imports, missing):
    from impulse_evals.config import ConfigError

    _mod()   # imported before sys.version_info is replaced
    extra_imports.missing(missing)
    cfg = _voice_cfg(light.project)
    with _python(12), pytest.raises(ConfigError) as info:
        _tts(cfg)
    assert info.value.field == "voice_config.tts.provider"
    assert info.value.message == LIGHT_EXTRA_MISSING


def test_tts_without_kokoro_on_python_3_14_adds_the_kokoro_note(light, extra_imports):
    from impulse_evals.config import ConfigError

    _mod()
    extra_imports.missing("kokoro_onnx")
    cfg = _voice_cfg(light.project)
    with _python(14), pytest.raises(ConfigError) as info:
        _tts(cfg)
    assert info.value.field == "voice_config.tts.provider"
    assert info.value.message == LIGHT_EXTRA_MISSING + PY314_NOTE


def test_tts_without_onnxruntime_on_python_3_14_has_no_kokoro_note(light, extra_imports):
    from impulse_evals.config import ConfigError

    _mod()
    extra_imports.missing("onnxruntime")
    cfg = _voice_cfg(light.project)
    with _python(14), pytest.raises(ConfigError) as info:
        _tts(cfg)
    assert info.value.message == LIGHT_EXTRA_MISSING


@pytest.mark.parametrize("minor", [12, 14])
def test_stt_without_faster_whisper_is_a_config_error_when_built(light, extra_imports, minor):
    from impulse_evals.config import ConfigError

    _mod()
    extra_imports.missing("faster_whisper")
    cfg = _voice_cfg(light.project)
    with _python(minor), pytest.raises(ConfigError) as info:
        _stt(cfg)
    assert info.value.field == "voice_config.stt.provider"
    assert info.value.message == LIGHT_EXTRA_MISSING


def test_light_module_imports_without_the_extra_and_without_heavy_libraries():
    """``impulse_evals.voice.local_light`` imports in a fresh interpreter in which the light libraries
    cannot be imported, and importing it loads none of them (they are imported inside the methods)."""
    code = (
        "import sys\n"
        "for name in ('kokoro_onnx', 'faster_whisper', 'torch'):\n"
        "    sys.modules[name] = None\n"
        "import impulse_evals.voice.local_light, impulse_evals.voice.tts, impulse_evals.voice.stt\n"
        "loaded = [name for name in ('onnxruntime', 'ctranslate2', 'transformers', 'phonemizer')\n"
        "          if name in sys.modules]\n"
        "print('loaded=' + ','.join(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "loaded="


# =============================================================================================
# Voices, check_language
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("language, voice, lang", DEFAULT_VOICES)
async def test_default_voice_and_espeak_lang_per_language(light, language, voice, lang):
    cfg = _voice_cfg(light.project, language=language)
    tts = _tts(cfg)
    tts.check_language(language)   # every table language is supported
    await _synthesize_with_small_files(light, tts, "Hello there.", language, None)
    assert len(light.kokoro.creates) == 1
    call = light.kokoro.creates[0]
    assert (call["text"], call["voice"], call["lang"]) == ("Hello there.", voice, lang)
    assert call["speed"] == 1.0


async def _synthesize_with_small_files(light, tts, text, language, voice):
    """Synthesize with small model files already in the cache folder (no download)."""
    files = {ONNX: b"o" * 64, VOICES: b"v" * 32}
    _place(light.cache, files)
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(
            _mod(), "MODEL_FILES", {name: (len(body), hashlib.sha256(body).hexdigest()) for name, body in files.items()}
        )
        return await tts.synthesize(text, language, voice)
    finally:
        mp.undo()


@pytest.mark.asyncio
@pytest.mark.parametrize("voice, lang", VOICE_LETTERS)
async def test_configured_voice_is_used_and_lang_comes_from_its_first_letter(light, voice, lang):
    cfg = _voice_cfg(light.project, tts={"provider": "local_light", "voice": voice})
    tts = _tts(cfg)
    await _synthesize_with_small_files(light, tts, "Hello.", "en-US", None)
    call = light.kokoro.creates[0]
    assert (call["voice"], call["lang"]) == (voice, lang)


@pytest.mark.asyncio
async def test_voice_argument_is_used_and_wins_over_the_configured_voice(light):
    tts = _tts(_voice_cfg(light.project))
    await _synthesize_with_small_files(light, tts, "Hola.", "en-US", "ef_dora")
    assert (light.kokoro.creates[-1]["voice"], light.kokoro.creates[-1]["lang"]) == ("ef_dora", "es")

    tts = _tts(_voice_cfg(light.project, tts={"provider": "local_light", "voice": "bf_emma"}))
    await _synthesize_with_small_files(light, tts, "Namaste.", "en-US", "hf_beta")
    assert (light.kokoro.creates[-1]["voice"], light.kokoro.creates[-1]["lang"]) == ("hf_beta", "hi")


@pytest.mark.parametrize("language", ["de-DE", "ko-KR", "ta-IN", "ar"])
def test_check_language_without_a_voice_for_the_language_is_a_voice_error(light, language):
    from impulse_evals.voice import VoiceError

    tts = _tts(_voice_cfg(light.project))
    with pytest.raises(VoiceError) as info:
        tts.check_language(language)
    assert str(info.value) == NO_VOICE.format(language=language)
    assert light.kokoro.inits == [] and light.server.requests == []


@pytest.mark.parametrize("language", ["en-US", "es-MX", "hi-IN", "fr-CA", "it", "pt-PT", "zh-TW", "EN-us"])
def test_check_language_accepts_every_primary_subtag_in_the_table(light, language):
    _tts(_voice_cfg(light.project)).check_language(language)


@pytest.mark.parametrize("language", ["de-DE", "ko-KR"])
def test_check_language_with_a_configured_voice_accepts_any_language(light, language):
    _tts(_voice_cfg(light.project, tts={"provider": "local_light", "voice": "af_heart"})).check_language(language)


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["xx_test", "qf_nobody", "_heart"])
async def test_configured_voice_with_an_unknown_first_letter_is_a_config_error(light, voice):
    from impulse_evals.config import ConfigError

    cfg = _voice_cfg(light.project, tts={"provider": "local_light", "voice": voice})
    with pytest.raises(ConfigError) as info:
        tts = _tts(cfg)
        tts.check_language("en-US")
        await _synthesize_with_small_files(light, tts, "Hello.", "en-US", None)
    assert info.value.field == "voice_config.tts.voice"
    assert info.value.message == f"unknown Kokoro voice {voice}"
    assert light.kokoro.creates == []


@pytest.mark.asyncio
async def test_voice_argument_with_an_unknown_first_letter_is_a_config_error(light):
    from impulse_evals.config import ConfigError

    tts = _tts(_voice_cfg(light.project))
    with pytest.raises(ConfigError) as info:
        await _synthesize_with_small_files(light, tts, "Hello.", "en-US", "xm_unknown")
    assert info.value.field == "voice_config.tts.voice"
    assert info.value.message == "unknown Kokoro voice xm_unknown"
    assert light.kokoro.creates == []


# =============================================================================================
# Synthesis: WAV output, worker thread, one engine per provider, errors
# =============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [24000, 22050])
async def test_synthesize_returns_mono_16_bit_wav_at_the_engine_rate(light, rate):
    import numpy

    light.kokoro = FakeKokoro(rate=rate, seconds=0.25)
    light.install()
    tts = _tts(_voice_cfg(light.project))
    audio = await _synthesize_with_small_files(light, tts, "Hello there.", "en-US", None)
    channels, width, wav_rate, frames = read_wav(audio)
    assert (channels, width, wav_rate) == (1, 2, rate)
    samples = numpy.array(pcm_samples(frames), dtype=numpy.int64)
    expected = numpy.round(light.kokoro.samples.astype(numpy.float64) * 32767.0).astype(numpy.int64)
    assert samples.shape == expected.shape
    assert int(numpy.max(numpy.abs(samples - expected))) <= 1


@pytest.mark.asyncio
async def test_synthesis_runs_in_a_worker_thread_and_reuses_one_engine(light, monkeypatch):
    real_to_thread = asyncio.to_thread
    to_thread_calls = []

    async def spy(func, /, *args, **kwargs):
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy)
    loop_thread = threading.get_ident()
    tts = _tts(_voice_cfg(light.project))
    assert light.kokoro.inits == []                       # loaded lazily
    await _synthesize_with_small_files(light, tts, "One.", "en-US", None)
    await _synthesize_with_small_files(light, tts, "Two.", "es-MX", None)
    assert len(light.kokoro.inits) == 1                   # one Kokoro per provider, reused
    assert [call["text"] for call in light.kokoro.creates] == ["One.", "Two."]
    assert all(call["thread"] != loop_thread for call in light.kokoro.creates)
    assert to_thread_calls, "synthesis must run through asyncio.to_thread"


@pytest.mark.asyncio
async def test_engine_error_is_a_one_line_voice_error_naming_the_voice(light):
    from impulse_evals.voice import VoiceError

    light.kokoro.create_error = RuntimeError("ONNXRuntimeError: invalid input shape")
    tts = _tts(_voice_cfg(light.project))
    with pytest.raises(VoiceError) as info:
        await _synthesize_with_small_files(light, tts, "Hello.", "en-US", None)
    assert str(info.value) == "local_light TTS failed with voice af_heart: RuntimeError: ONNXRuntimeError: invalid input shape"


@pytest.mark.asyncio
async def test_long_multi_line_engine_error_is_one_line_of_at_most_200_characters(light):
    from impulse_evals.voice import VoiceError

    light.kokoro.create_error = ValueError("bad phonemes\n" + "x" * 400)
    tts = _tts(_voice_cfg(light.project, tts={"provider": "local_light", "voice": "ef_dora"}))
    with pytest.raises(VoiceError) as info:
        await _synthesize_with_small_files(light, tts, "Hola.", "es-MX", None)
    text = str(info.value)
    assert text.startswith("local_light TTS failed with voice ef_dora: ValueError: bad phonemes")
    assert _one_line(text), text


@pytest.mark.asyncio
async def test_engine_load_error_is_a_voice_error_naming_the_voice(light):
    from impulse_evals.voice import VoiceError

    light.kokoro.init_error = RuntimeError("cannot open the model")
    tts = _tts(_voice_cfg(light.project))
    with pytest.raises(VoiceError) as info:
        await _synthesize_with_small_files(light, tts, "Hello.", "en-US", None)
    assert str(info.value) == "local_light TTS failed with voice af_heart: RuntimeError: cannot open the model"


# =============================================================================================
# Model folder
# =============================================================================================


@pytest.mark.asyncio
async def test_cache_folder_from_impulse_cache_dir_is_used_without_hashing_again(light, monkeypatch):
    # Files of the right size but with other content: present files are not hashed again.
    sizes = {ONNX: 300, VOICES: 120}
    monkeypatch.setattr(_mod(), "MODEL_FILES", {name: (size, "0" * 64) for name, size in sizes.items()})
    _place(light.cache, {name: b"\x01" * size for name, size in sizes.items()})
    tts = _tts(_voice_cfg(light.project))
    await tts.synthesize("Hello.", "en-US", None)
    assert light.server.requests == []
    assert len(light.kokoro.inits) == 1
    assert _same_path(light.kokoro.inits[0]["model_path"], light.cache / ONNX)
    assert _same_path(light.kokoro.inits[0]["voices_path"], light.cache / VOICES)


@pytest.mark.asyncio
async def test_relative_model_folder_is_resolved_against_the_config_folder(light, tmp_path):
    project = light.project
    config_dir = tmp_path / "proj"
    config_dir.mkdir()
    folder = config_dir / "models"
    _place(folder, {ONNX: b"onnx", VOICES: b"voices"})   # an explicit folder: present files, no size check
    cfg = _voice_cfg(project, tts={"provider": "local_light", "model": "models"}, path=config_dir / "config.yaml")
    assert Path.cwd() != config_dir
    tts = _tts(cfg)
    await tts.synthesize("Hello.", "en-US", None)
    assert light.server.requests == []
    assert _same_path(light.kokoro.inits[0]["model_path"], folder / ONNX)
    assert _same_path(light.kokoro.inits[0]["voices_path"], folder / VOICES)
    assert not light.cache.exists() or not any(light.cache.iterdir())


@pytest.mark.asyncio
async def test_absolute_model_folder_is_used_as_given(light, tmp_path):
    folder = tmp_path / "elsewhere" / "kokoro"
    _place(folder, {ONNX: b"onnx", VOICES: b"voices"})
    tts = _tts(_voice_cfg(light.project, tts={"provider": "local_light", "model": str(folder)}))
    await tts.synthesize("Hello.", "en-US", None)
    assert light.server.requests == []
    assert _same_path(light.kokoro.inits[0]["model_path"], folder / ONNX)


@pytest.mark.asyncio
@pytest.mark.parametrize("present", [[], [ONNX], [VOICES]], ids=["none", "onnx-only", "voices-only"])
async def test_model_folder_without_both_files_is_a_config_error(light, tmp_path, present):
    from impulse_evals.config import ConfigError

    config_dir = tmp_path / "proj"
    config_dir.mkdir()
    folder = config_dir / "models"
    _place(folder, {name: b"x" for name in present})
    cfg = _voice_cfg(light.project, tts={"provider": "local_light", "model": "models"}, path=config_dir / "config.yaml")
    with pytest.raises(ConfigError) as info:
        tts = _tts(cfg)
        await tts.synthesize("Hello.", "en-US", None)
    assert info.value.field == "voice_config.tts.model"
    assert info.value.message == MISSING_FILES.format(folder=str(folder.resolve()))
    assert light.server.requests == []        # an explicit folder is never downloaded into
    assert light.kokoro.creates == []


async def _mismatch_download(light, capsys):
    """Serve wrong bodies for both files (real MODEL_FILES) and return (VoiceError text, stderr)."""
    from impulse_evals.voice import VoiceError

    light.server.serve({ONNX: b"not the model", VOICES: b"not the voices"})
    tts = _tts(_voice_cfg(light.project))
    capsys.readouterr()
    with pytest.raises(VoiceError) as info:
        await tts.synthesize("Hello.", "en-US", None)
    return str(info.value), capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "win32", reason="the Windows cache folder")
async def test_windows_cache_folder_is_under_localappdata(light, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("IMPULSE_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    folder = tmp_path / "lad" / "impulse-evals" / "kokoro"
    text, err = await _mismatch_download(light, capsys)
    assert text in {CHECKSUM.format(name=name, folder=str(folder)) for name in (ONNX, VOICES)}
    lines = _download_lines(err)
    assert len(lines) == 1 and DOWNLOAD_LINE.match(lines[0]).group(2) == str(folder)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="the XDG cache folder")
async def test_posix_cache_folder_is_under_xdg_cache_home(light, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("IMPULSE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    folder = tmp_path / "xdg" / "impulse-evals" / "kokoro"
    text, _err = await _mismatch_download(light, capsys)
    assert text in {CHECKSUM.format(name=name, folder=str(folder)) for name in (ONNX, VOICES)}


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="the ~/.cache folder")
async def test_posix_cache_folder_defaults_to_home_cache(light, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("IMPULSE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = tmp_path / "home" / ".cache" / "impulse-evals" / "kokoro"
    text, _err = await _mismatch_download(light, capsys)
    assert text in {CHECKSUM.format(name=name, folder=str(folder)) for name in (ONNX, VOICES)}


# =============================================================================================
# Download
# =============================================================================================


@pytest.mark.asyncio
async def test_first_use_downloads_both_files_through_part_files(light, monkeypatch, capsys):
    import httpx

    bodies = _small_model_files(monkeypatch)
    # The release asset answers with a redirect to the storage host (GitHub does): it is followed.
    storage = "https://objects.githubusercontent.com/kokoro-onnx/" + ONNX
    light.server.routes[MODEL_BASE_URL + ONNX] = httpx.Response(302, headers={"location": storage})
    light.server.routes[storage] = bodies[ONNX]
    light.server.serve({VOICES: bodies[VOICES]})

    real_replace = os.replace
    replaced = []

    def replace_spy(src, dst, *args, **kwargs):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace_spy)
    capsys.readouterr()
    tts = _tts(_voice_cfg(light.project))
    assert light.server.requests == []
    audio = await tts.synthesize("Hello.", "en-US", None)
    err = capsys.readouterr().err

    assert read_wav(audio)[:3] == (1, 2, ENGINE_RATE)
    assert {record["method"] for record in light.server.requests} == {"GET"}
    assert set(light.server.urls()) == {MODEL_BASE_URL + ONNX, storage, MODEL_BASE_URL + VOICES}
    for name, body in bodies.items():
        assert (light.cache / name).read_bytes() == body
    assert not list(light.cache.glob("*.part"))
    for name in (ONNX, VOICES):   # B1-rev L3: a unique part file per download, {name}.{pid}.{8 hex}.part
        part = re.compile(re.escape(f"{name}.{os.getpid()}.") + r"[0-9a-f]{8}\.part")
        assert any(
            _same_path(Path(src).parent, light.cache) and part.fullmatch(Path(src).name) and _same_path(dst, light.cache / name)
            for src, dst in replaced
        ), replaced
    assert _same_path(light.kokoro.inits[0]["model_path"], light.cache / ONNX)
    assert _same_path(light.kokoro.inits[0]["voices_path"], light.cache / VOICES)
    lines = _download_lines(err)
    assert len(lines) == 1, err
    match = DOWNLOAD_LINE.match(lines[0])
    assert match is not None, lines[0]
    assert match.group(2) == str(light.cache)


@pytest.mark.asyncio
async def test_download_line_names_the_total_size_in_mb(light, capsys):
    text, err = await _mismatch_download(light, capsys)
    lines = _download_lines(err)
    assert len(lines) == 1, err
    match = DOWNLOAD_LINE.match(lines[0])
    assert match is not None, lines[0]
    assert TOTAL_MB_RANGE[0] <= float(match.group(1)) <= TOTAL_MB_RANGE[1], lines[0]
    assert match.group(2) == str(light.cache)


@pytest.mark.asyncio
async def test_size_mismatch_deletes_the_part_file_and_names_the_folder(light, capsys):
    text, _err = await _mismatch_download(light, capsys)
    assert text in {CHECKSUM.format(name=name, folder=str(light.cache)) for name in (ONNX, VOICES)}
    assert not list(light.cache.glob("*.part"))
    assert not (light.cache / ONNX).exists() and not (light.cache / VOICES).exists()
    assert light.kokoro.inits == [] and light.kokoro.creates == []


@pytest.mark.asyncio
async def test_checksum_mismatch_with_the_right_size_deletes_the_part_file(light, monkeypatch):
    from impulse_evals.voice import VoiceError

    bodies = _small_model_files(monkeypatch)
    wrong = {name: bytes(len(body)) for name, body in bodies.items()}   # right size, other bytes
    light.server.serve(wrong)
    tts = _tts(_voice_cfg(light.project))
    with pytest.raises(VoiceError) as info:
        await tts.synthesize("Hello.", "en-US", None)
    assert str(info.value) in {CHECKSUM.format(name=name, folder=str(light.cache)) for name in (ONNX, VOICES)}
    assert not list(light.cache.glob("*.part"))
    assert not (light.cache / ONNX).exists() and not (light.cache / VOICES).exists()


@pytest.mark.asyncio
async def test_network_error_text_names_the_file_and_url(light):
    import httpx

    from impulse_evals.voice import VoiceError

    for name in (ONNX, VOICES):
        light.server.routes[MODEL_BASE_URL + name] = httpx.ConnectError("connection refused")
    tts = _tts(_voice_cfg(light.project))
    with pytest.raises(VoiceError) as info:
        await tts.synthesize("Hello.", "en-US", None)
    expected = {
        COULD_NOT_DOWNLOAD.format(name=name, url=MODEL_BASE_URL + name, error="ConnectError: connection refused")
        for name in (ONNX, VOICES)
    }
    assert str(info.value) in expected
    assert not (light.cache / ONNX).exists() and not (light.cache / VOICES).exists()
    assert light.kokoro.creates == []


@pytest.mark.asyncio
async def test_http_error_status_is_a_download_error_not_a_checksum_error(light):
    from impulse_evals.voice import VoiceError

    tts = _tts(_voice_cfg(light.project))   # no route: every model URL answers 404
    with pytest.raises(VoiceError) as info:
        await tts.synthesize("Hello.", "en-US", None)
    text = str(info.value)
    assert any(text.startswith(f"local_light: could not download {name} from {MODEL_BASE_URL}{name}: ") for name in (ONNX, VOICES)), text
    assert "checksum" not in text
    assert _one_line(text), text
    assert not (light.cache / ONNX).exists() and not (light.cache / VOICES).exists()


@pytest.mark.asyncio
async def test_no_second_download_when_the_files_exist(light, monkeypatch, capsys):
    bodies = _small_model_files(monkeypatch)
    light.server.serve(bodies)
    await _tts(_voice_cfg(light.project)).synthesize("First.", "en-US", None)
    first = list(light.server.requests)
    assert first
    capsys.readouterr()
    fresh = _tts(_voice_cfg(light.project))           # a new provider (e.g. the next run)
    await fresh.synthesize("Second.", "en-US", None)
    assert light.server.requests == first
    assert _download_lines(capsys.readouterr().err) == []


@pytest.mark.asyncio
async def test_cached_file_of_the_wrong_size_is_downloaded_again(light, monkeypatch):
    bodies = _small_model_files(monkeypatch)
    _place(light.cache, {ONNX: b"truncated", VOICES: bodies[VOICES]})
    light.server.serve(bodies)
    await _tts(_voice_cfg(light.project)).synthesize("Hello.", "en-US", None)
    assert MODEL_BASE_URL + ONNX in light.server.urls()
    assert (light.cache / ONNX).read_bytes() == bodies[ONNX]


# =============================================================================================
# STT (faster-whisper)
# =============================================================================================


@pytest.mark.asyncio
async def test_stt_defaults_are_small_cpu_int8(light):
    stt = _stt(_voice_cfg(light.project))
    assert light.whisper.inits == []
    await stt.transcribe(sine_wav(0.5), "en-US")
    assert [(i["model"], i["device"], i["compute_type"]) for i in light.whisper.inits] == [("small", "cpu", "int8")]


@pytest.mark.asyncio
async def test_stt_explicit_model_device_and_compute_type_are_passed(light):
    cfg = _voice_cfg(
        light.project, stt={"provider": "local_light", "model": "base", "device": "cuda", "compute_type": "float16"}
    )
    await _stt(cfg).transcribe(sine_wav(0.5), "en-US")
    assert [(i["model"], i["device"], i["compute_type"]) for i in light.whisper.inits] == [("base", "cuda", "float16")]


@pytest.mark.asyncio
async def test_stt_gets_mono_16k_numpy_samples_never_a_path(light):
    import numpy

    stereo = array_stereo_wav(1.0, 8000)
    await _stt(_voice_cfg(light.project)).transcribe(stereo, "hi-IN")
    assert len(light.whisper.transcribes) == 1
    call = light.whisper.transcribes[0]
    audio = call["audio"]
    assert isinstance(audio, numpy.ndarray), type(audio)
    assert audio.ndim == 1 and audio.dtype == numpy.float32
    assert abs(len(audio) - 16000) <= 16
    assert 0.3 <= float(numpy.max(numpy.abs(audio))) <= 1.0
    assert call["language"] == "hi"
    assert call["kwargs"].get("word_timestamps") is True


def array_stereo_wav(seconds, sample_rate, amplitude=0.5):
    import array

    mono = array.array("h")
    mono.frombytes(pcm_tone(seconds, sample_rate, amplitude=amplitude))
    stereo = array.array("h", (sample for sample in mono for _ in range(2)))
    return wav_bytes(stereo.tobytes(), sample_rate, channels=2)


@pytest.mark.asyncio
async def test_stt_words_and_confidence_follow_the_local_hf_rules(light):
    from impulse_evals.models import STTResult

    light.whisper.segments = [
        _segment(" book a", [_word(" book", 0.0, 0.5, 0.9), _word(" a", 0.5, 0.75, 0.6)]),
        _segment(" slot", [_word(" slot", 0.75, 1.5, 0.9), _word(" ", 1.5, 1.6, 0.1), _word(" x", None, 1.7, 0.1)]),
    ]
    result = await _stt(_voice_cfg(light.project)).transcribe(sine_wav(0.5), "en-US")
    assert isinstance(result, STTResult)
    assert result.text == "book a slot"
    assert [(w.word, w.start_ms, w.end_ms, w.confidence) for w in result.words] == [
        ("book", 0, 500, 0.9),
        ("a", 500, 750, 0.6),
        ("slot", 750, 1500, 0.9),
    ]
    assert result.confidence == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_stt_without_words_has_no_confidence(light):
    light.whisper.segments = [_segment(" hello", [])]
    result = await _stt(_voice_cfg(light.project)).transcribe(sine_wav(0.5), "en-US")
    assert result.text == "hello"
    assert result.words == []
    assert result.confidence is None


@pytest.mark.asyncio
async def test_stt_model_loads_once_per_provider_in_a_worker_thread(light):
    loop_thread = threading.get_ident()
    stt = _stt(_voice_cfg(light.project))
    await stt.transcribe(sine_wav(0.5), "en-US")
    await stt.transcribe(sine_wav(0.5), "es-MX")
    assert len(light.whisper.inits) == 1
    assert light.whisper.inits[0]["thread"] != loop_thread
    assert [call["language"] for call in light.whisper.transcribes] == ["en", "es"]
    assert all(call["thread"] != loop_thread for call in light.whisper.transcribes)


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["transcribe", "segments"])
async def test_stt_inference_failure_is_a_voice_error_naming_the_model(light, where):
    from impulse_evals.voice import VoiceError

    error = RuntimeError("Library cublas64_12.dll is not found")
    if where == "transcribe":
        light.whisper.transcribe_error = error
    else:
        light.whisper.iterate_error = error
    with pytest.raises(VoiceError) as info:
        await _stt(_voice_cfg(light.project)).transcribe(sine_wav(0.5), "en-US")
    assert str(info.value) == "local_light STT failed with model small: RuntimeError: Library cublas64_12.dll is not found"


# =============================================================================================
# torch is never imported
# =============================================================================================

_TORCH_SCRIPT = r'''
import asyncio, importlib.abc, importlib.machinery, json, sys, types
import numpy as np

attempts = []

class TorchTrap(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds torch (as if installed) and records any real import of it."""
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            return importlib.machinery.ModuleSpec(name, self)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        attempts.append(module.__name__)
        raise ImportError("torch import attempted")

sys.meta_path.insert(0, TorchTrap())

def install(name, **attrs):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module

class Tokenizer:
    def phonemize(self, text, lang="en-us", norm=True):
        return text

class Kokoro:
    def __init__(self, model_path, voices_path, espeak_config=None, vocab_config=None):
        self.voices = {"af_heart": np.zeros((1, 1, 4), dtype=np.float32)}
        self.tokenizer = Tokenizer()
    def get_voices(self):
        return sorted(self.voices)
    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False, trim=True):
        t = np.arange(4800) / 24000.0
        return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), 24000

class WhisperModel:
    def __init__(self, model_size_or_path, device="auto", compute_type="default", **kwargs):
        pass
    def transcribe(self, audio, language=None, **kwargs):
        assert isinstance(audio, np.ndarray)
        word = types.SimpleNamespace(word=" hello", start=0.0, end=0.2, probability=0.9)
        segment = types.SimpleNamespace(text=" hello", start=0.0, end=0.2, words=[word])
        return iter([segment]), types.SimpleNamespace(language=language)

install("kokoro_onnx", Kokoro=Kokoro)
install("onnxruntime", get_available_providers=lambda: ["CPUExecutionProvider"])
install("faster_whisper", WhisperModel=WhisperModel)

from impulse_evals.config import load_config
from impulse_evals.voice.stt import get_stt_provider
from impulse_evals.voice.tts import get_tts_provider

voice = load_config(sys.argv[1]).voice_config

async def main():
    tts, stt = get_tts_provider(voice), get_stt_provider(voice)
    tts.check_language("en-US")
    audio = await tts.synthesize("Hello there.", "en-US", None)
    result = await stt.transcribe(audio, "en-US")
    return result.text

text = asyncio.run(main())
print("RESULT=" + json.dumps({"text": text, "torch": "torch" in sys.modules, "attempts": attempts}))
'''


def test_torch_is_never_imported_by_synthesize_and_transcribe(impulse_project, tmp_path):
    import json

    folder = tmp_path / "kokoro-files"
    _place(folder, {ONNX: b"onnx", VOICES: b"voices"})
    config = impulse_project.write_config(
        [impulse_project.scenario("light_probe", language="en-US")],
        voice_config=_voice_block({"provider": "local_light", "model": str(folder)}, {"provider": "local_light"}),
    )
    env = dict(os.environ, IMPULSE_CACHE_DIR=str(tmp_path / "unused-cache"))
    completed = subprocess.run(
        [sys.executable, "-c", _TORCH_SCRIPT, str(config)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT=")]
    assert lines, completed.stdout + completed.stderr
    result = json.loads(lines[-1][len("RESULT="):])
    assert result == {"text": "hello", "torch": False, "attempts": []}


# =============================================================================================
# impulse doctor
# =============================================================================================


def _doctor_config(project, tts=None, stt=None, languages=("en-US",)):
    project.write_agent()
    scenarios = [
        project.scenario(f"doctor_light_{index}", language=language, metrics=["UNIV-013"])
        for index, language in enumerate(languages)
    ]
    return project.write_config(scenarios, voice_config=_voice_block(tts, stt))


def _rows(result, status, check):
    return [
        line.strip()
        for line in result.output.splitlines()
        if line.strip().startswith(status) and f" {check} " in f" {line.strip()} "
    ]


def _detail(row, check):
    return row.split(check, 1)[1].strip()


@pytest.mark.phase11
def test_doctor_rows_before_the_first_download(light, mock_llm, cli):
    mock_llm.set_default("OK")
    result = cli.invoke("doctor", _doctor_config(light.project))
    text = cli.text(result)
    assert result.exit_code == 0, text
    tts_rows = _rows(result, "PASS", "voice TTS")
    assert len(tts_rows) == 1, text
    detail = _detail(tts_rows[0], "voice TTS")
    match = re.fullmatch(
        r"local_light: 'light' extra installed; Kokoro model downloads on first run \((\d+(?:\.\d+)?) MB\)", detail
    )
    assert match is not None, detail
    assert TOTAL_MB_RANGE[0] <= float(match.group(1)) <= TOTAL_MB_RANGE[1]
    stt_rows = _rows(result, "PASS", "voice STT")
    assert [_detail(row, "voice STT") for row in stt_rows] == [
        "local_light: 'light' extra installed; model small loads on the first run"
    ], text
    # No round trip, no model load, no download.
    assert light.kokoro.inits == [] and light.kokoro.creates == []
    assert light.whisper.inits == [] and light.whisper.transcribes == []
    assert light.server.requests == []


@pytest.mark.phase11
def test_doctor_rows_with_the_model_in_the_cache_folder(light, mock_llm, cli, monkeypatch):
    mock_llm.set_default("OK")
    bodies = _small_model_files(monkeypatch)
    _place(light.cache, bodies)
    stt = {"provider": "local_light", "model": "base"}
    result = cli.invoke("doctor", _doctor_config(light.project, stt=stt))
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert [_detail(row, "voice TTS") for row in _rows(result, "PASS", "voice TTS")] == [
        f"local_light: 'light' extra installed; Kokoro model present in {light.cache}"
    ], text
    assert [_detail(row, "voice STT") for row in _rows(result, "PASS", "voice STT")] == [
        "local_light: 'light' extra installed; model base loads on the first run"
    ], text
    assert light.kokoro.inits == [] and light.server.requests == []


@pytest.mark.phase11
def test_doctor_rows_with_an_explicit_model_folder(light, mock_llm, cli, tmp_path):
    mock_llm.set_default("OK")
    folder = tmp_path / "models"
    _place(folder, {ONNX: b"onnx", VOICES: b"voices"})
    result = cli.invoke("doctor", _doctor_config(light.project, tts={"provider": "local_light", "model": "models"}))
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert [_detail(row, "voice TTS") for row in _rows(result, "PASS", "voice TTS")] == [
        f"local_light: 'light' extra installed; Kokoro model present in {folder.resolve()}"
    ], text


@pytest.mark.phase11
def test_doctor_explicit_model_folder_without_the_files_fails(light, mock_llm, cli, tmp_path):
    mock_llm.set_default("OK")
    (tmp_path / "models").mkdir()
    result = cli.invoke("doctor", _doctor_config(light.project, tts={"provider": "local_light", "model": "models"}))
    text = cli.text(result)
    assert result.exit_code == 2, text
    expected = MISSING_FILES.format(folder=str((tmp_path / "models").resolve()))
    assert [row for row in _rows(result, "FAIL", "voice TTS") if expected in row], text
    assert not _rows(result, "PASS", "voice TTS"), text


@pytest.mark.phase11
@pytest.mark.parametrize("missing", ["kokoro_onnx", "onnxruntime"])
def test_doctor_without_the_light_extra_fails_the_tts_row(light, mock_llm, cli, extra_imports, missing):
    mock_llm.set_default("OK")
    extra_imports.missing(missing)
    result = cli.invoke("doctor", _doctor_config(light.project))
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert [row for row in _rows(result, "FAIL", "voice TTS") if LIGHT_EXTRA_MISSING in row], text
    assert not _rows(result, "PASS", "voice TTS"), text
    assert _rows(result, "PASS", "voice STT"), text   # STT needs only faster-whisper


@pytest.mark.phase11
def test_doctor_without_faster_whisper_fails_the_stt_row(light, mock_llm, cli, extra_imports):
    mock_llm.set_default("OK")
    extra_imports.missing("faster_whisper")
    result = cli.invoke("doctor", _doctor_config(light.project))
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert [row for row in _rows(result, "FAIL", "voice STT") if LIGHT_EXTRA_MISSING in row], text
    assert _rows(result, "PASS", "voice TTS"), text


@pytest.mark.phase11
def test_doctor_lists_scenario_languages_without_a_kokoro_voice(light, mock_llm, cli):
    mock_llm.set_default("OK")
    config = _doctor_config(light.project, languages=("en-US", "de-DE", "hi-IN", "ko-KR", "de-DE"))
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert [_detail(row, "voice languages") for row in _rows(result, "FAIL", "voice languages")] == [
        "local_light TTS has no voice for de-DE, ko-KR"
    ], text


@pytest.mark.phase11
def test_doctor_has_no_languages_row_when_a_voice_is_configured(light, mock_llm, cli):
    mock_llm.set_default("OK")
    config = _doctor_config(
        light.project, tts={"provider": "local_light", "voice": "af_heart"}, languages=("en-US", "de-DE")
    )
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert not _rows(result, "FAIL", "voice languages"), text


# =============================================================================================
# pyproject: the light extra
# =============================================================================================


def test_pyproject_light_extra_has_exactly_the_two_requirements():
    import tomllib

    from packaging.requirements import Requirement

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    light = data["project"]["optional-dependencies"]["light"]
    assert len(light) == 2, light
    kokoro, whisper = (Requirement(text) for text in light)
    assert kokoro.name == "kokoro-onnx"
    assert {str(spec) for spec in kokoro.specifier} == {">=0.6.1", "<0.7"}
    assert str(kokoro.marker) == 'python_version < "3.14"'
    assert whisper.name == "faster-whisper"
    assert str(whisper.specifier) == "" and whisper.marker is None
    assert not kokoro.extras and not whisper.extras


def test_fake_kokoro_and_whisper_signatures_match_what_the_tests_assume():
    # Guard for the fakes themselves (passes today): create(text, voice, speed, lang) and the
    # faster-whisper constructor accept the spec's call shapes.
    kokoro = FakeKokoro().module().Kokoro
    assert list(inspect.signature(kokoro.create).parameters)[:5] == ["self", "text", "voice", "speed", "lang"]
    whisper = FakeWhisper().module().WhisperModel
    assert list(inspect.signature(whisper.__init__).parameters)[:3] == ["self", "model_size_or_path", "device"]
