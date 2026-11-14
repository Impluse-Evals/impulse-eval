<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Fixes round 3, Stage B notes: B2 (LiteLLM voice provider) and B5 (neutral pack defaults)

New test files: `tests/test_litellm_voice.py` (B2), `tests/test_neutral_defaults.py` (B5).

## Superseded assertions

- `tests/test_cli_exit_codes.py::test_dry_run_of_examples_config`: `assert run.total_scenarios == 4` -> `assert run.total_scenarios == 6` (FIXES-ROUND-3-SPEC.md B5: `examples/config.yaml` gains `support_order_status_us` and `cita_clinica_mx`). Nothing else in existing tests was changed.

## Interpretations (spec ambiguities; the simplest reading was chosen)

B2, `test_litellm_voice.py`:

1. Model strings in tests: `deepgram/nova-3`, `groq/impulse-test-stt`, `openai/impulse-test-tts`, `ollama/impulse-test-stt` (needs no key). `YOUR_..._MODEL` strings were avoided because `config._PLACEHOLDER_RE` treats them as placeholders, which a builder could reasonably read as "model missing".
2. The STT and TTS call kwargs are compared as exact dicts: STT `{model, file, language, response_format, timestamp_granularities, drop_params, timeout[, api_key][, api_base]}`; TTS `{model, input[, voice], response_format, timeout[, api_key][, api_base]}` (no `drop_params`, no `language` on TTS, as the spec's call). No positional arguments.
3. `timeout` equals `voice_config.timeout_s` (the test sets 17).
4. TTS voice: `synthesize(..., voice)` with `tts.voice` unset passes the argument; with the argument None, `tts.voice` is passed; both None: no `voice` key. The case where both are set is not tested (openai_compatible uses `voice or tts.voice`). The runner always passes None today.
5. "A named key that is not set" also covers `api_key_env` naming an unset variable: `ConfigError("voice_config.{side}.model", "{model} needs the key {NAME}: set the environment variable or add it to [keys] in .impulse/config.toml")` with NAME = the `api_key_env` value (not openai_compatible's `api_key_env` field and text).
6. Deepgram raw fallback "same length": the raw `alternatives[0].words` list is compared with the response's word list; the tests use lists in which no entry is dropped, so "before or after dropping" is not decided.
7. Alternative-level confidence: a string or an absent value gives None; a boolean is not tested.
8. Error text cut: the tests check one line and at most 200 characters after `"litellm STT failed with model {model}: "`, not the exact "..." form; the exact texts are checked only for short messages (provider_error_text form, and `"{type}: {message}"` for other exceptions, first line, whitespace collapsed).
9. A non-WAV TTS answer without pydub raises the existing `VoiceError` text of `audio_utils.ensure_wav` unchanged (not wrapped in "litellm TTS failed ...") and is not retried.
10. A response without `content`: one test object has only `async aread()`, another only `read()`.
11. Retries: the tests replace `asyncio.sleep` (module attribute) and expect the delays `[1.0, 2.0]` (`1.0 * attempt`); an implementation using `from asyncio import sleep` would wait for real and fail the delay assertion. Retried on 429, 502, 503; not on 400, 401, 403, 404, 422. Every attempt uses the same kwargs.
12. `VOICE_PROVIDERS` and the unknown-provider text include `local_light` (B1), so those tests pass only when B1's provider name is also added.
13. Key redaction is checked for three sources: `DEEPGRAM_API_KEY` in the environment, a custom `api_key_env` variable (not `*_API_KEY`), and config.toml `[keys]`; neither the value nor its first 12 characters may appear.
14. Lazy import: a subprocess imports `impulse_evals.voice.litellm_voice` and `litellm` must not be in `sys.modules`.

B5, `test_neutral_defaults.py`:

15. `is_india_locale`: True for `en-IN`, `hi-IN`, `hi-Latn-IN`, `ta-in`, `en-in`, `HI-IN`, `mr-Deva-IN`; False for `en-US`, `es-MX`, `en-GB`, `hi`, `in`, `id`, `en-Latn-US`, `es-419`. `None` is not tested.
16. For a non-IN language every domain default is `[]` (all auto-added packs are India packs); for an IN language the list equals the `language=None` list (same order).
17. `default_assertions` for an unknown domain with a language is the general row (`[]`).
18. `impulse generate`: fsi / healthcare / telecom with `en-US` or `es-MX` write `assertions: []` for both directions and the text contains no `dpdp`, `rbi_kyc`, `trai_dlt` or `hipaa`; fsi with `en-IN` or `hi-Latn-IN` writes the India packs as today.
19. Examples: the six scenarios in the order the four existing ones, then `support_order_status_us`, `cita_clinica_mx`. `support_order_status_us`: en-US, ecommerce, inbound, a non-empty metrics list without UNIV-023, `assertions == []`. `cita_clinica_mx`: es-MX, healthcare, inbound, non-empty persona and expected_outcome, UNIV-023, one `book_slot` expectation with required arguments {date, patient_name, phone} and `max_count: 1`, no `dpdp.` / `rbi_kyc.` / `trai_dlt.` / `hipaa.` assertion. That the persona is in Spanish is not checked.
20. The dry run of `examples/config.yaml` (marker `phase12`): exit 0, no LLM call, six scenarios, every `passed` True with no error; `cita_clinica_mx` has UNIV-025 `good` (not skipped) and exactly one `book_slot` call in its transcript (so its opening message, if any, must not contain the word "book").

Validation: both files were run against a throwaway prototype in a scratch copy of the repo (not in this repo): all tests passed there, so the fakes and expectations are consistent.

## Superseded assertions (B2-rev)

New test file: `tests/test_litellm_voice_rev.py` (FIXES-ROUND-3-SPEC.md "B2-rev": V1, V2, V3, V4, V9). In `tests/test_litellm_voice.py` only these two exact-kwargs assertions were changed (the docstring and every other line are unchanged):

- `test_stt_call_kwargs_exact` (model `deepgram/nova-3`, language `es-MX`): the expected kwargs dict `{model, file, language: "es", response_format, timestamp_granularities, drop_params, timeout: 17, api_key}` -> the same dict plus `"detect_language": "es"` (V1) and `"max_retries": 0` (V2).
- `test_tts_call_kwargs_exact_and_wav_passthrough` (model `openai/impulse-test-tts`): the expected kwargs dict `{model, input, voice: "nova", response_format: "wav", timeout: 17, api_key}` -> the same dict plus `"max_retries": 0` (V2). `response_format: "wav"` stays (openai provider, V3).

## B2-rev interpretations

21. V1 seam: the real-request test calls `LiteLLMSTT.transcribe` with the real `litellm.atranscription` and replaces `httpx.AsyncClient.send` (class attribute, so LiteLLM's cached `AsyncHTTPHandler` clients and its aiohttp transport are both bypassed) with a fake that records the `httpx.Request` and returns a Deepgram-shaped 200; `httpx.Client.send` raises, so no sync request can slip out. No socket is opened. It asserts exactly one request, `POST https://api.deepgram.com/v1/listen`, query `model=nova-3` and `detect_language=hi` for `hi-IN`, `Authorization: Token <dummy>`, and `text == "namaste"`. Verified in LiteLLM 1.105.1 with a direct call: with `detect_language="hi"` the URL is `?model=nova-3&detect_language=hi`; without it, `?model=nova-3`.
22. V1: `detect_language` is the primary subtag string (`"hi"`, not `True`); it is tested for `deepgram/...` only; `groq/...`, `openai/...`, `azure/...` get no `detect_language` and still get `language`.
23. V2: `max_retries == 0` is checked on STT (deepgram, openai, groq) and TTS (openai, elevenlabs) and on every retried attempt; Impulse's own 3 attempts and `[1.0, 2.0]` sleeps are unchanged. A request-count test through the OpenAI SDK was not written (only the kwarg is pinned).
24. V3: `groq` is in `litellm.openai_compatible_providers` and `elevenlabs` is not (asserted as a precondition). `response_format="wav"` for `openai/`, `azure/`, `groq/` TTS models; for `elevenlabs/...` the exact kwargs are `{model, input, voice, max_retries: 0, timeout, api_key}` and its MP3 answer goes through `ensure_wav` (fake pydub). Other non-compatible providers are not tested.
25. V4: the 400 texts tested mention `response_format`, `verbose_json`, `Timestamp_Granularities`, `TIMESTAMP` (case-insensitive). The repeat has the first call's kwargs minus `response_format` and `timestamp_granularities` (language, detect_language, max_retries, timeout, key unchanged), with no sleep. A repeat that fails again is a VoiceError after 2 calls in all (repeated once only). The memory is per provider instance: a new instance sends both parameters again. Not repeated: a 400 with other text (file format, language), a 422 that mentions `response_format` (only HTTP 400 qualifies), and any TTS 400. Not decided: a repeat that fails with 429 / 5xx (whether Impulse's retries then apply).
26. V9: the `support_order_status_us` `expected_outcome` must not contain `refund`, `discount`, `voucher`, `coupon` or `compensation` (none is in `examples/system_prompt.txt`, asserted as a precondition). Whether it must also name a human transfer (the review's suggested wording) is not checked; the persona may still mention a refund.

## B2-rev corrections

- `tests/test_litellm_voice_rev.py::test_support_order_status_expected_outcome_fits_the_system_prompt`: removed the precondition `assert word not in prompt, f"precondition: system_prompt.txt has no {word} policy"` -> (no precondition). Reason: `examples/system_prompt.txt` says "Never promise discounts or results that are not written here.", so "discount" appears there legitimately. The check that `expected_outcome` contains none of `refund`, `discount`, `voucher`, `coupon`, `compensation` is unchanged. Item 26 above: the "asserted as a precondition" part no longer applies.
