# SPDX-License-Identifier: Apache-2.0
"""Fixes round 3, B1-rev: revisions of the light local voice after the Stage D review
(FIXES-ROUND-3-SPEC.md "B1 decisions", bullet "B1-rev"; findings L1-L7).

* L1: synthesis phonemizes first with kokoro-onnx's own tokenizer (``engine.tokenizer.phonemize(text,
  lang)``, the voice's espeak language), removes every language-switch marker matching
  ``\\([a-z]{2,3}(-[a-z0-9]+)*\\)``, and calls ``create(phonemes, voice=..., speed=1.0, lang=...,
  is_phonemes=True)``.
* L2: ``ja`` is no longer a default language ("Kokoro speaks en, es, fr, hi, it, pt and zh"); an explicit
  ``j*`` voice still works.
* L3: each download writes its own part file ``{name}.{pid}.{8 hex}.part``; the final ``os.replace`` is
  inside the error handling: when it fails and the final file now has the right size (another process
  finished first) that file is used; any other failure is the "could not download" VoiceError; no part
  file is left behind on error.
* L5: model loading runs inside ``asyncio.to_thread`` (never on the event loop thread).
* L6: doctor checks a configured ``tts.voice`` against ``voices-v1.0.bin`` when that file is present
  (FAIL "voice TTS" "unknown Kokoro voice {voice}"); synthesis with a voice missing from the file is
  ``VoiceError("local_light TTS: unknown Kokoro voice {voice}")``.
* L7: no cache folder (no ``IMPULSE_CACHE_DIR`` / ``LOCALAPPDATA`` / ``XDG_CACHE_HOME`` and no home folder)
  is ``ConfigError("voice_config.tts.model", "local_light cannot find a cache folder: set
  IMPULSE_CACHE_DIR or voice_config.tts.model")``.

Fakes and helpers are the ones of ``tests/test_local_light.py`` (fake kokoro-onnx 0.6.1 API: ``voices``,
``tokenizer.phonemize``, ``create(..., is_phonemes=...)``). Seams: tests/data/fixes_round3/
NOTES_stageB_local_light.md, "B1-rev".
"""

import io
import os
import pathlib
import re
import threading

import pytest
from test_local_light import (  # noqa: F401 - light is a fixture
    MODEL_BASE_URL,
    ONNX,
    VOICES,
    _detail,
    _doctor_config,
    _mod,
    _one_line,
    _place,
    _rows,
    _same_path,
    _small_model_files,
    _tts,
    _voice_cfg,
    light,
)

NO_VOICE = (
    "local_light TTS has no voice for {language}. Kokoro speaks en, es, fr, hi, it, pt and zh; "
    "set voice_config.tts.voice or use another TTS provider (litellm, openai_compatible) for this language."
)
COULD_NOT_DOWNLOAD = "local_light: could not download {name} from {url}: {error}"
UNKNOWN_IN_FILE = "local_light TTS: unknown Kokoro voice {voice}"
NO_CACHE_FOLDER = "local_light cannot find a cache folder: set IMPULSE_CACHE_DIR or voice_config.tts.model"


def _npz(*voices):
    """A real ``voices-v1.0.bin``-shaped archive (``np.savez``) holding ``voices``."""
    import numpy

    buffer = io.BytesIO()
    numpy.savez(buffer, **{name: numpy.zeros((2, 1, 4), dtype=numpy.float32) for name in voices})
    return buffer.getvalue()


def _explicit_folder(tmp_path, voices=("af_heart", "hf_alpha")):
    folder = tmp_path / "kokoro-files"
    _place(folder, {ONNX: b"onnx", VOICES: _npz(*voices)})
    return folder


def _flat(text):
    return " ".join(text.split())


# =============================================================================================
# L1: phonemize first, strip espeak language markers, create(..., is_phonemes=True)
# =============================================================================================

MARKER_CASES = [
    pytest.param(
        "hi-IN",
        None,
        "मेरा appointment कल है, please confirm.",
        "mˌeːɾaː (en)ɐpˈɔɪntmənt(hi) kˈʌl hɛː, (en)plˈiːz kənfˈɜːm(hi).",
        "mˌeːɾaː ɐpˈɔɪntmənt kˈʌl hɛː, plˈiːz kənfˈɜːm.",
        "hf_alpha",
        "hi",
        id="hinglish",
    ),
    pytest.param(
        "hi-IN",
        None,
        "Mujhe kal ke liye appointment book karna hai.",
        "(en)mˈʌdʒh kˈæl kˈiː lˈaɪ əpˈɔɪntmənt bˈʊk kˈɑːnə hˈaɪ(hi).",
        "mˈʌdʒh kˈæl kˈiː lˈaɪ əpˈɔɪntmənt bˈʊk kˈɑːnə hˈaɪ.",
        "hf_alpha",
        "hi",
        id="romanized-hindi",
    ),
    pytest.param(
        "zh-CN",
        None,
        "订单号是 ABC123",
        "tˈiŋtˈan xˈɑʊ ʂˈɨ (en)ˈeɪ bˈiː sˈiː(cmn) wˈʌn",
        "tˈiŋtˈan xˈɑʊ ʂˈɨ ˈeɪ bˈiː sˈiː wˈʌn",
        "zf_xiaobei",
        "cmn",
        id="mandarin-latin",
    ),
    pytest.param(
        "en-US",
        "pf_dora",
        "Olá, tudo bem?",
        "(pt-br)olˈa(en-us), (fr-fr)tˈudu(es) bˈẽj?",
        "olˈa, tˈudu bˈẽj?",
        "pf_dora",
        "pt-br",
        id="region-markers",
    ),
    pytest.param(
        "en-US",
        None,
        "Hello there.",
        "həlˈoʊ (x) ðˈɛɹ (EN) (ˈæ) (abcd) (e1).",
        "həlˈoʊ (x) ðˈɛɹ (EN) (ˈæ) (abcd) (e1).",
        "af_heart",
        "en-us",
        id="not-markers-kept",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("language, voice, text, phonemes, spoken, expected_voice, lang", MARKER_CASES)
async def test_language_markers_are_removed_before_kokoro_speaks(
    light, tmp_path, language, voice, text, phonemes, spoken, expected_voice, lang
):
    light.kokoro.phonemes[text] = phonemes
    folder = _explicit_folder(tmp_path, voices=(expected_voice,))
    tts = _tts(_voice_cfg(light.project, tts={"provider": "local_light", "model": str(folder)}))
    await tts.synthesize(text, language, voice)
    assert [(call["text"], call["lang"]) for call in light.kokoro.phonemize_calls] == [(text, lang)]
    assert len(light.kokoro.creates) == 1
    call = light.kokoro.creates[0]
    assert call["is_phonemes"] is True
    assert (call["voice"], call["lang"], call["speed"]) == (expected_voice, lang, 1.0)
    assert _flat(call["spoken"]) == spoken          # what Kokoro actually speaks
    assert _flat(call["text"]) == spoken


@pytest.mark.asyncio
async def test_phonemes_without_markers_reach_create_unchanged(light, tmp_path):
    text = "Your appointment is on Tuesday."
    light.kokoro.phonemes[text] = "jʊɹ əpˈɔɪntmənt ɪz ˌɑːn tˈuːzdeɪ."
    folder = _explicit_folder(tmp_path)
    tts = _tts(_voice_cfg(light.project, tts={"provider": "local_light", "model": str(folder)}))
    await tts.synthesize(text, "en-US", None)
    call = light.kokoro.creates[0]
    assert call["is_phonemes"] is True
    assert _flat(call["text"]) == "jʊɹ əpˈɔɪntmənt ɪz ˌɑːn tˈuːzdeɪ."


# =============================================================================================
# L2: no Japanese default
# =============================================================================================


@pytest.mark.parametrize("language", ["ja-JP", "ja"])
def test_japanese_has_no_default_voice(light, language):
    from impulse_evals.voice import VoiceError

    tts = _tts(_voice_cfg(light.project))
    with pytest.raises(VoiceError) as info:
        tts.check_language(language)
    assert str(info.value) == NO_VOICE.format(language=language)


@pytest.mark.asyncio
async def test_japanese_synthesis_without_a_voice_is_the_no_voice_error(light, tmp_path):
    from impulse_evals.voice import VoiceError

    folder = _explicit_folder(tmp_path)
    tts = _tts(_voice_cfg(light.project, tts={"provider": "local_light", "model": str(folder)}))
    with pytest.raises(VoiceError) as info:
        await tts.synthesize("こんにちは", "ja-JP", None)
    assert str(info.value) == NO_VOICE.format(language="ja-JP")
    assert light.kokoro.creates == []


@pytest.mark.asyncio
async def test_explicit_japanese_voice_still_works(light, tmp_path):
    folder = _explicit_folder(tmp_path, voices=("jf_alpha",))
    tts = _tts(_voice_cfg(light.project, tts={"provider": "local_light", "model": str(folder), "voice": "jf_alpha"}))
    tts.check_language("ja-JP")
    await tts.synthesize("よやくをおねがいします", "ja-JP", None)
    call = light.kokoro.creates[0]
    assert (call["voice"], call["lang"]) == ("jf_alpha", "ja")


@pytest.mark.phase11
def test_doctor_lists_japanese_as_a_language_without_a_voice(light, mock_llm, cli):
    mock_llm.set_default("OK")
    result = cli.invoke("doctor", _doctor_config(light.project, languages=("en-US", "ja-JP")))
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert [_detail(row, "voice languages") for row in _rows(result, "FAIL", "voice languages")] == [
        "local_light TTS has no voice for ja-JP"
    ], text


# =============================================================================================
# L3: unique part files, the final rename inside the error handling
# =============================================================================================


def _replace_spy(monkeypatch, behaviour):
    """Replace ``os.replace``: ``behaviour(src, dst, real)`` decides; returns the recorded calls."""
    real = os.replace
    calls = []

    def spy(src, dst, *args, **kwargs):
        calls.append((str(src), str(dst)))
        return behaviour(src, dst, lambda: real(src, dst, *args, **kwargs))

    monkeypatch.setattr(os, "replace", spy)
    return calls


def _part_pattern(name):
    return re.compile(re.escape(f"{name}.{os.getpid()}.") + r"[0-9a-f]{8}\.part")


@pytest.mark.asyncio
async def test_part_file_names_are_unique_per_download(light, monkeypatch, tmp_path):
    bodies = _small_model_files(monkeypatch)
    light.server.serve(bodies)
    calls = _replace_spy(monkeypatch, lambda src, dst, real: real())
    await _tts(_voice_cfg(light.project)).synthesize("First.", "en-US", None)
    monkeypatch.setenv("IMPULSE_CACHE_DIR", str(tmp_path / "second-cache"))
    await _tts(_voice_cfg(light.project)).synthesize("Second.", "en-US", None)
    onnx_parts = [pathlib.Path(src).name for src, dst in calls if pathlib.Path(dst).name == ONNX]
    assert len(onnx_parts) == 2, calls
    assert all(_part_pattern(ONNX).fullmatch(name) for name in onnx_parts), onnx_parts
    assert onnx_parts[0] != onnx_parts[1]
    voices_parts = [pathlib.Path(src).name for src, dst in calls if pathlib.Path(dst).name == VOICES]
    assert voices_parts and all(_part_pattern(VOICES).fullmatch(name) for name in voices_parts), voices_parts


@pytest.mark.asyncio
async def test_rename_failure_after_another_process_finished_uses_its_file(light, monkeypatch):
    bodies = _small_model_files(monkeypatch)
    light.server.serve(bodies)

    def finished_elsewhere(src, dst, real):
        # Another process renamed its own part file first; ours cannot replace it (Windows: in use).
        pathlib.Path(dst).write_bytes(bodies[pathlib.Path(dst).name])
        raise PermissionError(13, "The process cannot access the file because it is being used by another process")

    _replace_spy(monkeypatch, finished_elsewhere)
    audio = await _tts(_voice_cfg(light.project)).synthesize("Hello.", "en-US", None)
    assert audio[:4] == b"RIFF"
    for name, body in bodies.items():
        assert (light.cache / name).read_bytes() == body
    assert not list(light.cache.glob("*.part"))
    assert _same_path(light.kokoro.inits[0]["model_path"], light.cache / ONNX)


@pytest.mark.asyncio
@pytest.mark.parametrize("final_file", [None, b"short"], ids=["no-final-file", "final-file-wrong-size"])
async def test_other_rename_failure_is_the_download_error_without_a_part_file(light, monkeypatch, final_file):
    from impulse_evals.voice import VoiceError

    bodies = _small_model_files(monkeypatch)
    light.server.serve(bodies)

    def fails(src, dst, real):
        if final_file is not None:
            pathlib.Path(dst).write_bytes(final_file)
        raise PermissionError(13, "denied")

    _replace_spy(monkeypatch, fails)
    with pytest.raises(VoiceError) as info:
        await _tts(_voice_cfg(light.project)).synthesize("Hello.", "en-US", None)
    expected = {
        COULD_NOT_DOWNLOAD.format(name=name, url=MODEL_BASE_URL + name, error="PermissionError: [Errno 13] denied")
        for name in (ONNX, VOICES)
    }
    assert str(info.value) in expected
    assert not list(light.cache.glob("*.part"))
    assert light.kokoro.creates == []


def _broken_stream():
    """A response body that sends one chunk, then the connection drops (sync and async clients)."""
    import httpx

    class BrokenStream(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self):
            yield b"partial-model-bytes"
            raise httpx.ReadError("connection reset by peer")

        async def __aiter__(self):
            yield b"partial-model-bytes"
            raise httpx.ReadError("connection reset by peer")

    return BrokenStream()


@pytest.mark.asyncio
async def test_connection_cut_mid_download_leaves_no_part_file(light, monkeypatch):
    import httpx

    from impulse_evals.voice import VoiceError

    bodies = _small_model_files(monkeypatch)
    for name in bodies:
        light.server.routes[MODEL_BASE_URL + name] = httpx.Response(200, stream=_broken_stream())
    with pytest.raises(VoiceError) as info:
        await _tts(_voice_cfg(light.project)).synthesize("Hello.", "en-US", None)
    text = str(info.value)
    assert any(text.startswith(f"local_light: could not download {name} from {MODEL_BASE_URL}{name}: ") for name in bodies), text
    assert "ReadError" in text and _one_line(text), text
    assert not list(light.cache.glob("*.part"))


# =============================================================================================
# L5: model loading in the worker thread
# =============================================================================================


@pytest.mark.asyncio
async def test_kokoro_loads_and_phonemizes_off_the_event_loop_thread(light, monkeypatch):
    bodies = _small_model_files(monkeypatch)
    light.server.serve(bodies)
    loop_thread = threading.get_ident()
    await _tts(_voice_cfg(light.project)).synthesize("Hello.", "en-US", None)
    assert len(light.kokoro.inits) == 1
    assert light.kokoro.inits[0]["thread"] != loop_thread
    assert light.kokoro.phonemize_calls and all(call["thread"] != loop_thread for call in light.kokoro.phonemize_calls)
    assert all(call["thread"] != loop_thread for call in light.kokoro.creates)


# =============================================================================================
# L6: a voice id missing from voices-v1.0.bin
# =============================================================================================


@pytest.mark.asyncio
async def test_synthesis_with_a_voice_argument_missing_from_the_voices_file_is_a_voice_error(light, tmp_path):
    from impulse_evals.voice import VoiceError

    folder = _explicit_folder(tmp_path, voices=("af_heart",))
    tts = _tts(_voice_cfg(light.project, tts={"provider": "local_light", "model": str(folder)}))
    with pytest.raises(VoiceError) as info:
        await tts.synthesize("Hello.", "en-US", "af_bella")
    assert str(info.value) == UNKNOWN_IN_FILE.format(voice="af_bella")
    assert all(call["voice"] != "af_bella" for call in light.kokoro.creates)
    # The same provider still speaks with a voice that is in the file.
    await tts.synthesize("Hello.", "en-US", "af_heart")
    assert light.kokoro.creates[-1]["voice"] == "af_heart"


@pytest.mark.asyncio
async def test_synthesis_with_a_configured_voice_missing_from_the_voices_file_is_a_voice_error(light, tmp_path):
    from impulse_evals.voice import VoiceError

    folder = _explicit_folder(tmp_path, voices=("af_heart", "hf_alpha"))
    cfg = _voice_cfg(light.project, tts={"provider": "local_light", "model": str(folder), "voice": "hf_beta"})
    tts = _tts(cfg)
    with pytest.raises(VoiceError) as info:
        await tts.synthesize("Namaste.", "hi-IN", None)
    assert str(info.value) == UNKNOWN_IN_FILE.format(voice="hf_beta")


def _doctor_with_voice(project, tmp_path, voice, where):
    if where == "explicit":
        folder = tmp_path / "models"
        _place(folder, {ONNX: b"onnx", VOICES: _npz("af_heart", "ef_dora")})
        tts = {"provider": "local_light", "model": "models", "voice": voice}
    else:
        tts = {"provider": "local_light", "voice": voice}
    return _doctor_config(project, tts=tts)


@pytest.mark.phase11
@pytest.mark.parametrize("where", ["explicit", "cache"])
def test_doctor_fails_a_configured_voice_missing_from_the_voices_file(light, mock_llm, cli, monkeypatch, tmp_path, where):
    mock_llm.set_default("OK")
    if where == "cache":
        bodies = {ONNX: b"onnx-model-bytes", VOICES: _npz("af_heart", "ef_dora")}
        import hashlib

        monkeypatch.setattr(
            _mod(), "MODEL_FILES", {name: (len(body), hashlib.sha256(body).hexdigest()) for name, body in bodies.items()}
        )
        _place(light.cache, bodies)
    result = cli.invoke("doctor", _doctor_with_voice(light.project, tmp_path, "af_hearts", where))
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert [row for row in _rows(result, "FAIL", "voice TTS") if "unknown Kokoro voice af_hearts" in row], text
    assert not _rows(result, "PASS", "voice TTS"), text
    assert light.kokoro.inits == [] and light.server.requests == []   # read from the file, no model load


@pytest.mark.phase11
def test_doctor_passes_a_configured_voice_that_is_in_the_voices_file(light, mock_llm, cli, tmp_path):
    mock_llm.set_default("OK")
    result = cli.invoke("doctor", _doctor_with_voice(light.project, tmp_path, "ef_dora", "explicit"))
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert _rows(result, "PASS", "voice TTS"), text


@pytest.mark.phase11
def test_doctor_does_not_check_the_voice_before_the_voices_file_is_downloaded(light, mock_llm, cli, tmp_path):
    mock_llm.set_default("OK")
    result = cli.invoke("doctor", _doctor_with_voice(light.project, tmp_path, "af_hearts", "cache"))
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert _rows(result, "PASS", "voice TTS"), text
    assert light.server.requests == []


# =============================================================================================
# L7: no cache folder
# =============================================================================================


def _no_home(monkeypatch, blank_cache_dir=False):
    for name in ("IMPULSE_CACHE_DIR", "LOCALAPPDATA", "XDG_CACHE_HOME", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
        monkeypatch.delenv(name, raising=False)
    if blank_cache_dir:
        monkeypatch.setenv("IMPULSE_CACHE_DIR", "   ")

    def no_home(cls=None):
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(pathlib.Path, "home", classmethod(no_home))


@pytest.mark.asyncio
@pytest.mark.parametrize("blank_cache_dir", [False, True], ids=["unset", "blank"])
async def test_no_cache_folder_is_a_config_error(light, monkeypatch, blank_cache_dir):
    from impulse_evals.config import ConfigError

    cfg = _voice_cfg(light.project)
    _no_home(monkeypatch, blank_cache_dir)
    with pytest.raises(ConfigError) as info:
        tts = _tts(cfg)
        await tts.synthesize("Hello.", "en-US", None)
    assert info.value.field == "voice_config.tts.model"
    assert info.value.message == NO_CACHE_FOLDER
    assert light.server.requests == [] and light.kokoro.inits == []


@pytest.mark.asyncio
async def test_explicit_model_folder_needs_no_cache_folder(light, monkeypatch, tmp_path):
    folder = _explicit_folder(tmp_path)
    cfg = _voice_cfg(light.project, tts={"provider": "local_light", "model": str(folder)})
    _no_home(monkeypatch)
    await _tts(cfg).synthesize("Hello.", "en-US", None)
    assert len(light.kokoro.creates) == 1


@pytest.mark.phase11
def test_doctor_without_a_cache_folder_fails_the_tts_row(light, mock_llm, cli, monkeypatch):
    import impulse_evals.cli  # noqa: F401 - imported before the home folder disappears

    mock_llm.set_default("OK")
    config = _doctor_config(light.project)
    _no_home(monkeypatch)
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert [row for row in _rows(result, "FAIL", "voice TTS") if NO_CACHE_FOLDER in row], text
