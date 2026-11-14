<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Fixes round 3, Stage B notes: B1 (light local voice `local_light`, extra `light`) and the blank HTTP bearer token rule

New test files: `tests/test_local_light.py` (B1), `tests/test_http_blank_token.py` (B4-rev blank token). No existing test, `conftest.py` or source file was changed.

## Seams the builder provides

1. `impulse_evals.voice.local_light.MODEL_BASE_URL` (str, trailing slash) and `MODEL_FILES` (`{file name: (size in bytes, sha256 hex)}`, in the order `kokoro-v1.0.onnx`, `voices-v1.0.bin`). Both are read when used (download, presence check, doctor MB), never copied at import, so tests replace `MODEL_FILES` with small files (`monkeypatch.setattr`). A test pins the exact spec values. Download URL = `MODEL_BASE_URL + name`.
2. Download: the tests install an `httpx.MockTransport` into every new `httpx.Client` / `httpx.AsyncClient` created without a `transport` (patched `__init__`, as `mock_voice_server`); `httpx.stream(...)`, a sync client in the worker thread, or an async client all work. GET, `follow_redirects` (a 302 to `objects.githubusercontent.com` is served). `os.replace` (module attribute; `Path.replace` calls it) is spied to check `<name>.part` -> `<name>`.
3. Fakes in `sys.modules`: `kokoro_onnx.Kokoro(model_path, voices_path)` with `create(text, voice, speed=1.0, lang=...)` returning `(float32, 24000)`; `onnxruntime` (stub); `faster_whisper.WhisperModel` with faster-whisper's constructor signature. Missing = `sys.modules[name] = None` (`find_spec` returns None, import raises).
4. `sys.version_info` is replaced (a namedtuple) only around the factory call and read when the provider is built. The module is imported before that.
5. Worker thread: `asyncio.to_thread` (module attribute) is spied, and the fakes record their thread (must differ from the loop's).

## Interpretations (simplest reading)

6. `IMPULSE_CACHE_DIR` IS the cache folder (files directly in it, no `kokoro` subfolder). Windows fallback `%LOCALAPPDATA%\impulse-evals\kokoro`; POSIX `$XDG_CACHE_HOME/impulse-evals/kokoro`, else `~/.cache/impulse-evals/kokoro` (POSIX tests skip on Windows).
7. Building a provider loads no model and downloads nothing (doctor builds it); the download and the Kokoro load happen on the first `synthesize`.
8. An explicit `voice_config.tts.model` folder: both files must exist; no size or hash check, never downloaded into. Relative paths are resolved against the config file's folder (the voice block handed to the factory must already carry that; the tests load a config in another folder than the cwd). The `{folder}` in "local_light needs ... in {folder}" and "present in {folder}" is the resolved absolute folder; for the cache it is `str(Path(IMPULSE_CACHE_DIR))`. The missing-files ConfigError may come from the factory or from `synthesize` (either is accepted); doctor shows it as a FAIL "voice TTS" row.
9. Exactly one stderr line per download, before it, naming the total MB of the files to download; the tests accept a number from 337 to 354 (MB or MiB, rounded or not) for the full model. No line when nothing is downloaded.
10. Size and checksum mismatch both give the checksum text (either file name accepted: the order of downloads is not pinned); no `.part` file and no final file remain.
11. Network error `{error}` = `"{type}: {message}"` (e.g. `ConnectError: connection refused`; the whole text is 183 characters). An HTTP error status (404) is also a "could not download" error, never a checksum error.
12. Voice precedence: the `voice` argument, then `voice_config.tts.voice`, then the language default (as litellm / openai_compatible). Voice letters are case-sensitive as in the table; `Af_heart` is not tested. An unknown letter in the configured voice may raise at build, `check_language` or `synthesize`; in the argument, at `synthesize`. Both use the field `voice_config.tts.voice`.
13. `check_language` uses the primary subtag (lower-cased, so `EN-us` works); `{language}` in the text is the code as passed. With a voice configured, any language passes.
14. Engine errors: constructor and `create` failures both give "local_light TTS failed with voice {voice}: {type}: {message}"; long / multi-line messages: one line, at most 200 characters, prefix checked only.
15. STT: `device`/`compute_type` `auto` -> `cpu` / `int8`, explicit values passed unchanged; `model` null -> `small`. Words stripped, blank or timing-less words dropped, confidence = mean probability of kept words (as local_hf). Inference failure (in `transcribe` or while iterating segments): "local_light STT failed with model {model}: {type}: {message}". The load-failure text is not tested.
16. Doctor: "voice languages" codes are the scenario languages in config order, de-duplicated, joined with ", " (as impulse_hosted); no row when a voice is configured. Doctor never loads a model, never transcribes, never downloads.
17. torch: a subprocess installs the fakes and a meta-path finder that "finds" torch and records any real import of it; after build + `check_language` + `synthesize` + `transcribe` (config loaded with `load_config`), `"torch" not in sys.modules` and no import was attempted (a `find_spec` probe is allowed). Module import alone loads none of onnxruntime, ctranslate2, transformers, phonemizer.
18. pyproject: `light` is exactly two requirements, parsed with `packaging`: `kokoro-onnx` with `{>=0.6.1, <0.7}` and marker `python_version < "3.14"`, then `faster-whisper` (no specifier, marker, extras).

Blank token (`test_http_blank_token.py`): env `""`, `"   "`, `" \t "`, toml `""`, `"  "`, `"\n\t \r\n"`, each with and without `start_url`: `ConfigError("agent.http.api_key_env", "{NAME} is not set: ...")` from `start()`, no request. It already passes (implemented before its test). A blank env value with a non-blank `[keys]` value is not decided.

Validation: both files were run against a throwaway prototype in a scratch copy of the repo (not in this repo): 104 passed, 2 skipped (POSIX-only) there.

## Superseded (B1-rev)

New test file: `tests/test_local_light_rev.py` (FIXES-ROUND-3-SPEC.md "B1-rev": L1, L2, L3, L5, L6, L7; L4 not changed, not tested). In `tests/test_local_light.py` only these assertions changed:

- `NO_VOICE` (used by `test_check_language_without_a_voice_for_the_language_is_a_voice_error`): "Kokoro speaks en, es, fr, hi, it, ja, pt and zh" -> "Kokoro speaks en, es, fr, hi, it, pt and zh" (L2).
- `DEFAULT_VOICES` (`test_default_voice_and_espeak_lang_per_language`): the row `("ja-JP", "jf_alpha", "ja")` removed (L2). `VOICE_LETTERS` keeps `jm_kumo` -> `ja` (an explicit `j*` voice still works).
- `test_check_language_accepts_every_primary_subtag_in_the_table`: `"ja-JP"` removed from its languages (L2).
- `test_first_use_downloads_both_files_through_part_files`: the renamed part file `cache/{name}.part` -> a file in the cache folder named `{name}.{os.getpid()}.{8 lowercase hex}.part` (L3).

Fakes extended to the real kokoro-onnx 0.6.1 API (read in `scratchpad\light_eval\venv_kokoro_312\...\kokoro_onnx\__init__.py` and `tokenizer.py`); no assertion changed by this:
- `FakeKokoro`: `Kokoro.voices` (loaded from the voices file with `np.load`, as kokoro-onnx does; a placeholder file that is not an `.npz` gives every id in `KNOWN_VOICES`), `get_voices()`, `tokenizer.phonemize(text, lang="en-us", norm=True) -> str` (scripted by `phonemes[text]`, default the text unchanged), and `create(text, voice, speed=1.0, lang="en-us", is_phonemes=False, trim=True, sentence_pause, clause_pause, continuous)`, which records `is_phonemes` and `spoken` (the phonemes Kokoro would speak: the text when `is_phonemes`, else its own phonemization) and raises `ValueError("Voice {voice} not found in available voices")` for an unknown voice, as kokoro-onnx does.
- The torch subprocess fake (`_TORCH_SCRIPT`) got `voices`, `get_voices()` and `tokenizer.phonemize` too.

## B1-rev interpretations

19. L1 seam: `engine.tokenizer.phonemize(text, lang)` is called once per synthesis with the voice's espeak language (`hi`, `cmn`, `pt-br`, `en-us`, ...); `create` gets the stripped phonemes as its first argument with `is_phonemes=True`, `voice`, `speed=1.0`, `lang`. The tests compare after collapsing whitespace (kokoro collapses it too), so whether the builder collapses spaces left by a removed marker is not pinned. Removed: `(en)`, `(hi)`, `(cmn)`, `(en-us)`, `(pt-br)`, `(fr-fr)`, `(es)`; kept: `(x)`, `(EN)`, `(ˈæ)`, `(abcd)`, `(e1)`.
20. L3: part file `{name}.{pid}.{8 lowercase hex}.part` in the target folder; two downloads in one process get different names. A failing `os.replace` (spied as a module attribute) with the final file then present at the right size (written by the spy: "another process finished first") is success: that file is used, no part file left. With no final file, or one of the wrong size: `VoiceError("local_light: could not download {name} from {url}: PermissionError: [Errno 13] denied")` (`{type}: {message}`), no part file left. A connection cut mid-body (`httpx.ReadError`) leaves no part file (this already passes).
21. L5: the Kokoro constructor, the phonemization and `create` run off the event-loop thread (thread ids recorded by the fakes). The constructor part already passes; the test fails today only because nothing is phonemized.
22. L6, synthesis: the voice is checked against the LOADED engine's voices (`voice in engine.voices` / `get_voices()`, which kokoro-onnx loads from `voices-v1.0.bin`), so the tests' placeholder voices files keep working; missing -> `VoiceError("local_light TTS: unknown Kokoro voice {voice}")` (not wrapped in "TTS failed with voice"), for the `voice` argument and for a configured voice; `create` is never called with it. Building the provider does not read the voices file.
23. L6, doctor: when a voice is configured and `voices-v1.0.bin` is present (explicit folder: it exists; cache: both files of the right size), the file is read with numpy (`np.load(...).files`, a real `.npz` in the tests) without loading the model; a missing voice is a FAIL "voice TTS" row that contains "unknown Kokoro voice {voice}" (a field prefix such as `voice_config.tts.voice: ` is allowed) and no PASS "voice TTS" row. Before the download (cache empty) the voice is not checked (PASS).
24. L7: "no home folder" is simulated by `pathlib.Path.home` raising `RuntimeError("Could not determine home directory.")` with `IMPULSE_CACHE_DIR` (also blank `"   "`), `LOCALAPPDATA`, `XDG_CACHE_HOME`, `HOME`, `USERPROFILE`, `HOMEDRIVE`, `HOMEPATH` unset. The ConfigError at `voice_config.tts.model` may come from the factory or from `synthesize`; doctor shows it as a FAIL "voice TTS" row. An explicit model folder needs no cache folder (already passes).
25. L2 doctor: `ja-JP` without a voice is listed in the "voice languages" FAIL row.

Validation (B1-rev): `test_local_light.py` + `test_local_light_rev.py` against a throwaway prototype of the revision in a scratch copy of the repo: 117 passed, 2 skipped.
