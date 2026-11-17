<!-- SPDX-License-Identifier: Apache-2.0 -->
# Changelog

This page lists the changes in each Impulse release, newest first.

## Quick version

- **What it is.** The list of changes per release. The only release so far is 0.1.0, not yet published to the package index.
- **The command to get it today**, from the root of a clone:

  ```bash
  pip install -e .
  ```

- **What success looks like.** `impulse --help` lists the commands `run`, `report`, `compare`, `doctor`, `init`, `generate`, `convert-failure`, and `calibrate` (verified).

## Details

### Unreleased

#### Licence change: everything is Apache-2.0

- **All of Impulse Evals is now under the Apache License 2.0**, including the plugin API files and the compliance packs (`dpdp`, `rbi_kyc`, `trai_dlt`, `hipaa`). Nothing remains under AGPL-3.0-only or the Elastic License 2.0.
- You may use, modify, and embed Impulse, and offer it as a hosted or managed service, including commercially. Keep the `LICENSE` and `NOTICE` files when you redistribute it.
- Copyright VITAF Technologies. The new `NOTICE` file names the project and the copyright holder; the licence does not grant rights to the name "Impulse" or the Impulse logo.
- `LICENSE` holds the Apache License 2.0 text. `LICENSE-APACHE` and `LICENSE-ELv2` are removed. The package metadata declares `License-Expression: Apache-2.0`, with `LICENSE` and `NOTICE` as its licence files.
- Every source file carries the `Apache-2.0` SPDX header. The plugin API import rule stays, as an architecture rule: the plugin API files import only each other ([Architecture](architecture.md#the-plugin-api-import-rule)).
- Contributions are accepted under the Apache License 2.0 and need a Developer Certificate of Origin sign-off (`git commit -s`) ([Contributing](contributing.md#licensing-of-contributions)).
- **Releases already published on PyPI keep the licence they were published under.** This change applies from the first release that includes it.

### 0.1.0 (unreleased)

The first release under the name Impulse Evals (formerly PULSE).

#### Noise simulation (EXPERIMENTAL, 2026-11-14)

- **New: noise simulation** ([Voice testing](voice-testing.md#noise-simulation-experimental)). In voice mode, Impulse mixes background noise into the simulated caller's audio after its text-to-speech and before your agent hears it. White, pink and brown noise is generated; recorded noise comes from your own WAV files (no audio ships with Impulse). The level is a signal-to-noise ratio (SNR) measured on the caller's speech; the noise is seeded, so reruns give the same audio, and a mix that would clip is turned down as a whole.
- **New config keys** ([Configuration](configuration.md#noise-experimental)): `voice_config.noise` (`profile`, `snr_db` from -5 to 40, `file`, `label`), the default for every scenario; `scenarios[i].noise` replaces it, and `noise: false` turns it off for one scenario. Every key is validated with an exact error. In text mode and in `--dry-run`, noise is ignored with one warning.
- **New optional HTTP field:** `agent.http.send_audio: true` adds `user_audio` (the base64 WAV the agent should hear) to each turn request. Without it the request body is unchanged, so existing HTTP agents keep working ([Adapters](adapters.md#the-exact-json)).
- **Generic adapter:** a handler that declares a `user_audio` parameter receives the caller's audio as WAV bytes; other handlers are called exactly as before.
- **Metrics stay honest:** VOICE-011 (harness round trip) always uses the clean audio; VOICE-009 and VOICE-013 measure what the agent heard from the noisy audio; latency stays the agent's own.
- **Report:** noise badges, the noisy caller audio with the clean version one click away, and a "Pass rate by noise level" table. `run.json` stores the noise settings per scenario. `impulse doctor` checks noise files and marks the feature experimental.
- `metrics.noise_simulation`, the old reserved key, now does nothing; `true` warns that it was replaced by `voice_config.noise`. It was removed from the example configs.
- **Status: experimental.** Covered by offline tests; not yet run against a live LiveKit, Pipecat, HTTP or custom agent ([Verified setups](verified-setups.md), [Contributing](contributing.md#help-us-test-noise-simulation)).

#### Fixes round 4 (2026-11-02, after the final live test)

- HTTP adapter: an `agent.http.headers` value that starts or ends with whitespace is now the config error "header values must not start or end with whitespace" (the value is never shown). Before, such a value passed validation and the httpx error text echoed it.
- Redaction: a known secret is also hidden in its JSON-escaped and Python-repr forms (a value with quotes or backslashes echoed by a JSON body or an exception text).
- Runner: when the conversation finished and only the adapter's close step fails (for the HTTP adapter, the `close_url` request), the scenario is still evaluated; the run gets the warning "{scenario}: the adapter's close failed after the conversation ended; the results are kept: {error}". Before, the whole evaluation was discarded.
- Pipecat eval mode: with `agent.system_prompt` set, the run now says once that the bot was reset with that file and ran its prompt ([Adapters](adapters.md#conversations-between-scenarios)).
- **`impulse run` defaults to `--fail-on high`**, and a poor UNIV-019 (harmful or misleading content: a credential request, false urgency, a misleading claim) counts as severity high for `--fail-on` and for `impulse compare`, like a poor UNIV-025. So a bot that asks for a password exits 1 by default in any language, without a compliance pack. A `config.toml` without `fail_on` means `high`; `impulse init` writes `fail_on = "high"`. `impulse compare` keeps its default `critical`, and `assert_no_critical` keeps its critical rule ([CI integration](ci-integration.md#--fail-on)).
- At `--fail-on high`, a `high` assertion for which the judge returned no verdict counts like a high failure ("high assertion had no verdict").
- Tool-call arguments: the value of every sensitive key (`otp`, `password`, `pin`, `token`, `card_number`, ...) is stored as `[REDACTED]`, as in tool results, so run.json, the report and the judge transcript never carry it.
- The embedded-audio size warning counts the base64 text in the page (a third more than the WAV files); before, a 58 MiB page could print nothing.
- Credential query parameters in URLs: `apikey`, `api_token`, `sig`, `signature`, `secret`, `password` and `auth` are now removed from stored and shown URLs too.
- The HTML report never carries an absolute path: an absolute `audio_path` is refused in embed mode too and shown by its file name; an unreadable audio file's note shows the OS error text, not the file's full path.
- `jiwer>=4` is required (3.x raised on an empty reference).
- Open low findings from the final test are listed in [ROADMAP.md](../ROADMAP.md).

#### Documentation pass (2026-10-26)

- Every docs page now starts with a one-sentence intro and a "Quick version": what it is, the one command, what success looks like. Then come "Details" and, last, "Next steps".
- New page [Reading the report](reading-the-report.md): every part of the HTML report. It covers the call recording and per-turn players, "Hear turn N" links, the call timeline, the tool call trace, and embedded or linked audio.
- New page [Known limitations](known-limitations.md): every documented limitation in one place, each with a one-line workaround.
- A real report walkthrough video, with the call audio, from a voice run of the second test campaign: `docs/images/report-walkthrough.mp4` (the summary, then one scenario). It replaces the earlier screenshots `report.png` and `report-summary.png`.
- The README opens with a 3-command quickstart from a clone, verified in a fresh virtual environment.
- [Verified setups](verified-setups.md) gains rows for the second test campaign (2026-10-25) and the round-3 checks (2026-10-26).

#### Renamed from PULSE

- Distribution `impulse-evals`, import package `impulse_evals`, command `impulse`, local folder `.impulse/`, hosted voice provider `impulse_hosted`, environment variable `IMPULSE_VOICE_API_KEY`, and pytest helper `impulse_check`. The old names are no longer used. The version starts again at 0.1.0 for the new distribution name.

#### Licensing

- The core is AGPL-3.0-only.
- The plugin API files (`plugin_types.py` and the four `base.py` files of `adapters`, `assertions`, `judge`, `metrics`) are Apache-2.0 and import no AGPL code.
- The compliance packs (`dpdp`, `rbi_kyc`, `trai_dlt`, `hipaa`) are under the Elastic License 2.0.
- Every source file carries its SPDX header. See [LICENSING.md](../LICENSING.md).
- `LICENSE`, `LICENSE-ELv2`, and `LICENSE-APACHE` hold the official licence texts. Before the first public release, the licensing of the combined distribution must be reviewed by counsel.

#### Features

- A pytest-style command line: `impulse init`, `impulse doctor`, `impulse generate`, `impulse run`, `impulse report`, `impulse compare`, `impulse convert-failure`, and `impulse calibrate`.
- Judging against the agent's own system prompt, with a fixed safety floor; the transcript reaches the judge as delimited, untrusted data.
- Python 3.11 or newer; tested on 3.11, 3.12, and 3.14.
- Universal, voice, and chat metrics as defined in [METRICS.md](../METRICS.md). They include UNIV-023 (tool call correctness, judged), which also catches claimed actions without a tool call. They also include UNIV-025 (expected and forbidden tool calls, checked by code).
- Optional compliance packs. India's: the Digital Personal Data Protection (DPDP) Act, and Reserve Bank of India Know Your Customer (KYC) guidance. Also the Telecom Regulatory Authority of India (TRAI) commercial communication rules, with their Distributed Ledger Technology (DLT) framework. An opt-in HIPAA pack is added by no domain automatically.
- Voice mode: text-to-speech (TTS) to the agent and speech-to-text (STT) on the way back. Providers: a free in-process default (`local_hf`), any OpenAI-compatible audio server, or the optional `impulse_hosted` client. Latency is measured on the agent only.
- Adapters: generic (any Python function), LiveKit (joins rooms as a client; `agent_name` dispatch), and Pipecat. Pipecat uses its eval transport by default, plain WebSocket as a fallback. Custom adapters through `adapter: custom` in `config.yaml` or `runner.run(config, adapter_factory=...)`, and any web service through `adapter: http`.
- Per-language and per-accent cohorts, never blended.
- `--fail-on critical|high|any` exit-code gates, `impulse compare` with JSON and HTML diff reports, and self-contained HTML reports with embedded audio (`--embed-audio`, `--no-embed-audio`).
- A critical compliance check the judge could not evaluate fails its scenario instead of passing silently.
- `inject_events` and `seed` modeled on scenarios; event execution is reserved for v2.
- A free path with no API key: `impulse init --profile free`.

#### Fixes from the first real-world test campaign

- Tool calls: the judge now sees what each tool returned, as ` result=` on the AGENT TOOL CALL line. Sensitive keys such as OTPs, PINs, and tokens are hidden, and the text is cut at 300 characters. So details the agent repeats from a tool are no longer judged as made up.
- Tool calls: `success` is inferred from common result shapes (`{"ok": true}`, `{"error": "..."}`, `{"status": "failed"}`) for every adapter, so tools need no change. The judge gets a short note on what `result=` and `success=None` mean.
- Tool calls: the ` error=` text gets the same treatment when it is a JSON object. The HTML report's tool call trace hides the same sensitive keys in its result and error columns.
- Compliance rules are judged only on their own criterion. Each rule's judge request ends with a scope paragraph, so one breach no longer fails several unrelated rules.
- `impulse compare` lists two new groups in the terminal, JSON, and HTML; `impulse compare --fail-on high` gates on them:
  - "Newly failing checks (fail -> fail)": scenarios that failed in both runs but have a newly failing check, or newly error;
  - "High regressions": a newly failing high assertion or a newly poor UNIV-025.
- Word error rate (WER): numbers, number words, dates, spelled-out letters, and IDs are brought to one form before VOICE-009 and VOICE-011 count errors. For example, `ORD-1004` = "o r d dash one zero zero four". This removes the false "Harness round-trip WER is high" warning on runs full of IDs and dates. Real mishearings are not hidden.
- Pipecat text mode: replies after a tool call are no longer doubled. A `-t eval` bot speaks them and reports each sentence twice. Such a reply stays in its own turn, even when the bot's completed copy of a sentence arrives after the turn ended.
- LiveKit: `agent.livekit.turn_timeout_s` sets the turn timeout in text and voice mode. The text and tool calls of a timed-out turn are kept in the transcript. Custom adapters can do the same by raising `AgentTurnError` with the turn so far.
- Provider errors are one line: type, HTTP status, provider, model, message, retry delay. A quota error may come with a retry delay of a minute or more. Impulse then stops calling that judge model for the rest of the run, with one warning.
- Report audio:
  - a player for the full call recording, and players on the caller's turns too;
  - a "Hear turn N" link next to each failed check, which jumps to the failing turn's player. The audio is embedded once, so failed checks do not grow the report;
  - a call timeline with dead air and slow replies marked.

  The embedded-audio size warning counts every embedded audio copy in the page.
- Docs: LiveKit on slow machines (explicit dispatch, worker `load_threshold`), and restarting `livekit-server --dev` after a network change. Also a TTS that cannot speak the scenario language, and `max_count` on escalation tools.

#### Fixes and additions from the second real-world test campaign

- Pipecat: `agent.pipecat.turn_timeout_s` sets the turn timeout in both modes, text and voice, as for LiveKit. The text, audio and tool calls of a timed-out turn are now kept, also when the bot never sends `bot-stopped-speaking`.
- Pipecat: a tool call whose handler raised is recorded with `success=False` and the bot's error text, so UNIV-023 sees the failure. With parallel calls of the same name, a successful call is never marked.
- New adapter `adapter: http`: an agent in any language behind JSON endpoints, with an example echo server in `examples/http/` (verified on localhost).
- New adapter `adapter: custom`: your own `AgentInterface` subclass, loaded from `config.yaml`, with free-form `agent.options`. Example in `examples/custom_adapter/`.
- New voice provider `provider: local_light`, the light local voice. It runs Kokoro ONNX text to speech and faster-whisper speech to text in-process, with no PyTorch and no GPU. New extra `light`. Measured on a laptop with 8 GB RAM: good in English and Spanish, poor STT in Hindi.
- New voice provider `provider: litellm`: speech to text and text to speech through LiteLLM. Deepgram speech to text was verified in English, Spanish and Hindi.
- Neutral defaults: `impulse generate` adds the India packs (`dpdp`, `rbi_kyc`, `trai_dlt`) only to scenarios whose language has the region `IN`. Any pack can still be listed for any language.
- `examples/config.yaml` has six scenarios: the four Indian ones, a US support call in English, and a Mexican clinic booking in Spanish.
- Docs: `livekit.turn_timeout_s` for voice agents with long spoken replies, and `impulse compare` exiting 1 on any new scenario error, environment errors included.

#### Planned for v1.1

- Compressed audio in HTML reports. Reports embed WAV today, which browsers play directly. A compressed, browser-playable format (Opus, MP3, or AAC) needs an encoder. Neither Python's standard library nor Impulse's required dependencies provide one, so it waits for a new optional dependency.

#### Not verified yet

- The free path with Ollama models, `local_hf` speech on real hardware, LiveKit Cloud, and the Pipecat websocket mode against a live bot. Real runs against LiveKit and Pipecat test bots with hosted models were verified in the two test campaigns ([Verified setups](verified-setups.md)).
- Groq, OpenAI and ElevenLabs voice through `provider: litellm`, a real Vapi or Retell custom adapter, and a Node.js agent behind `adapter: http`.
- Noise simulation against any live agent.

## Next steps

- [Getting started](getting-started.md): install this release.
- [FAQ](faq.md): including why Python 3.10 is not supported.
- [Verified setups](verified-setups.md): what the maintainer has run.

---

<!-- docs-nav -->
[← Previous: Verified setups](verified-setups.md) | [Docs home](index.md) | [Next: Roadmap →](../ROADMAP.md)
