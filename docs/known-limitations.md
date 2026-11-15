<!-- SPDX-License-Identifier: Apache-2.0 -->
# Known limitations

This page lists every documented limitation of Impulse 0.1.0 in one place, each with a one-line workaround and a link to the full explanation.

## Quick version

- **What it is.** Things Impulse 0.1.0 does not do, or does with a catch, that the maintainer knows about. Nothing here is hidden elsewhere: each row links to the page that explains it.
- **The one most people hit.** Voice reports embed every audio clip as WAV, so they get large. Rewrite a stored report with linked audio instead:

  ```bash
  impulse report RUN_ID --no-embed-audio
  ```

- **What success looks like.** `report.html` is rewritten without embedded audio, and its players play the files in the run's `audio/` folder. Keep the HTML next to that folder.

## Details

Terms used below: TTS is text-to-speech, STT is speech-to-text, and WER is word error rate. Each "Workaround" is one line; "More" links to the explanation.

### Reports and audio

| Limitation | Workaround | More |
|---|---|---|
| Reports embed audio as WAV. Compressed audio (Opus, MP3 or AAC) is planned for v1.1, because it needs a new optional dependency. A 6-scenario voice report in the second test campaign was 88.9 MiB. | Use `--no-embed-audio` on `impulse run`, or `impulse report RUN_ID --no-embed-audio` afterwards. | [Embedded or linked audio](reading-the-report.md#embedded-or-linked-audio) |
| With `save_recording: true` (the default), the runner keeps every audio clip of a scenario in memory until the scenario ends, for each scenario running at once. Very long voice calls need a lot of RAM. | Set `voice_config.save_recording: false`, or lower `--concurrency`. | [Voice testing](voice-testing.md#audio-files-and-the-report) |
| VOICE-009 (agent-hearing WER) is one WER over the whole call. One misheard amount can hide inside a "good" score (seen in the second test campaign). | Read the turns' "agent heard" lines; UNIV-023 catches a tool called with the wrong number. | [Metrics](metrics.md#deterministic-voice-metrics) |

### Noise simulation (experimental)

| Limitation | Workaround | More |
|---|---|---|
| Noise simulation is experimental. Offline tests cover the mixer, the config, the generic adapter in voice mode (mocked speech), a custom-style adapter, the HTTP adapter's `user_audio` field, the report and `impulse doctor`. It has not been run against a live LiveKit, Pipecat, HTTP or custom agent. | Listen to the noisy caller audio in the report before you trust a result, and tell us how it went. | [Contributing: help us test noise simulation](contributing.md#help-us-test-noise-simulation) |
| The noise stops 20 ms after the caller's last word, so the agent's end-of-speech detection under continuing noise is not tested. This keeps the LiveKit and Pipecat latency measurement honest. | None yet. | [Voice testing: noise simulation](voice-testing.md#noise-simulation-experimental) |
| A generic handler hears the noise only with a `user_audio` parameter, and an HTTP agent only with `agent.http.send_audio: true`. Otherwise the run warns that the noise has no effect. | Add the parameter, or set `send_audio: true`. | [Adapters](adapters.md#generic-adapter) |
| Recorded noise must be a WAV file; Impulse ships no recordings. | Convert other formats to WAV first, for example with ffmpeg. | [Configuration](configuration.md#noise-experimental) |
| Reports with noise embed two caller clips per turn (noisy and clean), so they are larger. | Use `--no-embed-audio`. | [Reading the report](reading-the-report.md#noise-simulation) |

### Pipecat

| Limitation | Workaround | More |
|---|---|---|
| A tool-call result that arrives after its turn was returned is recorded nowhere. The call keeps the `success` it had when its turn ended, often `None`. | None in Impulse; the bot's own log has the result. | [Adapters: what Impulse captures](adapters.md#what-impulse-captures) |
| A greeting whose LLM run or speech had not started when Impulse sends the first turn counts as the reply. It sets the first turn's latency, which can look short. | Read the first turn's latency with care for a bot that greets late. Impulse does not wait for the bot to go quiet. | [Adapters: what Impulse captures](adapters.md#what-impulse-captures) |
| RTVI (Pipecat's client/server message protocol) does not say which LLM run a `bot-output` belongs to. With a greeting the caller cannot interrupt, and a slow reply TTS, the reply can move into the next turn. | Raise `agent.pipecat.end_of_turn_grace_ms` above your TTS's first-audio delay. | [Adapters: what Impulse captures](adapters.md#what-impulse-captures) |
| Pipecat bots serve one client at a time, so Impulse runs Pipecat scenarios one by one; `--concurrency` has no effect. | Plan for sequential runs; stop any other client of the bot first. | [Adapters: one client at a time](adapters.md#one-client-at-a-time) |
| A `-t eval` bot keeps its conversation between scenarios. Without `agent.system_prompt`, Impulse cannot reset it and warns once. | Set `agent.system_prompt` in `config.yaml`, or reset the LLM context in the bot's `on_client_connected`. | [Adapters: conversations between scenarios](adapters.md#conversations-between-scenarios) |

### Voice providers

| Limitation | Workaround | More |
|---|---|---|
| Light local voice: Ctrl+C during the first Kokoro model download waits for the download thread to end. | Let the first download finish (about 354 MB for Kokoro, about 484 MB for Whisper `small`). | [Voice testing: light local voice](voice-testing.md#light-local-voice-no-pytorch-weak-laptops) |
| Light local voice: the first model load holds Python's global interpreter lock for a few seconds. This only matters with `--concurrency` above 1. | Run with `--concurrency 1`, or accept a short stall once per run. | [Voice testing: light local voice](voice-testing.md#light-local-voice-no-pytorch-weak-laptops) |
| Light local voice: a process killed during a download leaves its `.part` file in the model cache; nothing cleans it up. | Delete the `*.part` files in the cache folder (`IMPULSE_CACHE_DIR`, or `impulse-evals/kokoro` under your user cache folder). | [Voice testing: light local voice](voice-testing.md#light-local-voice-no-pytorch-weak-laptops) |
| Light local voice: Hindi STT with Whisper `small` is poor (round-trip WER 0.88 on the maintainer's PC). A larger Whisper model was not measured. | Keep `local_light` for TTS and use a stronger STT, for example Deepgram through `provider: litellm` (WER 0.22 on the same clip). | [Voice testing: light local voice](voice-testing.md#light-local-voice-no-pytorch-weak-laptops) |
| Light local voice: on Python 3.14 the `light` extra skips `kokoro-onnx`, which declares Python below 3.14. | Use Python 3.11 to 3.13, or run `pip install kokoro-onnx==0.6.1 --ignore-requires-python`. | [Troubleshooting: voice](troubleshooting.md#voice) |
| `provider: litellm`: Groq returns no word timings, so VOICE-015 (speech rate) skips. | Use an STT that returns word timings (Deepgram did in the round-3 check). | [Voice testing: litellm](voice-testing.md#litellm) |
| `provider: litellm`: OpenAI, Azure and ElevenLabs TTS need a voice name. | Set `voice_config.tts.voice`. | [Voice testing: litellm](voice-testing.md#litellm) |
| `provider: litellm`: LiteLLM 1.105.1 drops `language` from Deepgram requests. Impulse passes Deepgram's `detect_language`, limited to the scenario language, instead. | Nothing to do; checked in English, Spanish and Hindi on 2026-10-26. | [Voice testing: litellm](voice-testing.md#litellm) |
| `provider: litellm`: an unreadable provider answer reported as a 5xx error is retried like any 5xx error. | None needed; it costs at most two extra requests. | [Voice testing: litellm](voice-testing.md#litellm) |

### Models, keys and runs

| Limitation | Workaround | More |
|---|---|---|
| A rare LiteLLM provider that none of LiteLLM's tables name falls back to the key variable `{PROVIDER}_API_KEY`. | Set `api_key_env` for that role. | [Configuration](configuration.md#where-models-and-keys-come-from) |
| Only the `impulse` command sets LiteLLM's local cost map and repairs its tokenizer files. The library entry points (`impulse_check`, `runner.run`) do neither. | For an offline run, use the CLI, or set `LITELLM_LOCAL_MODEL_COST_MAP=True` before importing LiteLLM. | [Privacy and security](privacy-and-security.md#network-requests-that-carry-no-user-data) |
| Cold first starts are slow. On the maintainer's PC, importing transformers took about 64 s, and a first `impulse doctor` with Ollama about 100 s. | Wait once; later starts are faster. | [Troubleshooting](troubleshooting.md#install-and-imports) |
| `local_hf` checks PyTorch in a separate Python process. A setup that needs runtime-only state, such as `os.add_dll_directory`, can fail that check falsely. | Make `python -c "import torch"` work on its own. | [Troubleshooting: voice](troubleshooting.md#voice) |
| A key found only through a wildcard name rule, such as any variable ending in `_API_KEY`, is redacted only from 12 characters up. | Name the variable in `api_key_env` or put the key in `[keys]`; those are redacted from 4 characters up. | [Privacy and security: keys](privacy-and-security.md#keys) |
| `.impulse/config.toml` gets mode 0600 on Linux and macOS; on Windows the folder's default permissions apply. | Keep the project folder private on Windows. | [Privacy and security: keys](privacy-and-security.md#keys) |
| A generic agent must return tool-call `arguments` as a dict; a JSON string is rejected with a clear error. | Parse the string with `json.loads` before returning it. | [Adapters: generic adapter](adapters.md#generic-adapter) |
| `impulse compare` exits 1 on any newly errored scenario, environment errors included, such as an agent worker that was down. This is by design. | Re-run the errored scenarios before you trust the gate. | [CI integration](ci-integration.md) |
| `seed` is a hint to the simulated caller; the same seed does not guarantee the same turns. | Use `opening_user_message` when the first words must be exact. | [Writing scenarios: seed](writing-scenarios.md#seed) |
| `inject_events` are stored but not executed; execution is reserved for v2. | None yet; the scenario runs without injection and the report says so. | [Writing scenarios: inject_events](writing-scenarios.md#inject_events-v2) |
| Tool-call arguments store the value of a sensitive key (`otp`, `password`, `pin`, `token`, `card_number`, ...) as `[REDACTED]`, so an `expected_tool_calls` entry with `argument_values` for such a key cannot match; `required_arguments` still works. | Check such arguments through the tool's result instead, or through UNIV-023. | [Writing scenarios: tool calls](writing-scenarios.md#tool-calls-agenttools-expected_tool_calls-forbidden_tool_calls) |
| Smaller findings of the final test (short redaction values, WER residuals, a few reason texts, doctor wording) are open and listed with their fixes. | See the roadmap. | [ROADMAP.md](../ROADMAP.md) |

### Not built in 0.1.0

- No dashboard, database, PDF report or telephony.
- No ready-made Vapi, Retell or Bland adapter. Write one with `adapter: custom` ([Adapters](adapters.md#writing-your-own-adapter)).
- Pipecat over Daily, SmallWebRTC or LiveKit transports, and starting bot processes, are planned for v2.

## Next steps

- [Reading the report](reading-the-report.md): what each part of the HTML report shows.
- [Troubleshooting](troubleshooting.md): what the error messages mean.
- [Verified setups](verified-setups.md): what the maintainer has actually run.
- [FAQ](faq.md): short answers to common questions.

---

<!-- docs-nav -->
[← Previous: FAQ](faq.md) | [Docs home](index.md) | [Next: Privacy and security →](privacy-and-security.md)
