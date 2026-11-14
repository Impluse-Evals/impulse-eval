<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Verified setups

This dated table lists only the setups the maintainer has actually run end to end, and the TEMPLATE row shows what each entry records.

## Quick version

- **What it is.** The honest record of what was really run: dates, versions, RAM, models, commands and results. Everything not in the table is "not verified" on every page.
- **What has been run so far.** All on one Windows 11 laptop with 8 GB RAM:
  - LiveKit and Pipecat bots with hosted models, in text and voice (2026-10-24 and 2026-10-25);
  - on 2026-10-26: the light local voice, Deepgram speech-to-text through `provider: litellm`, and the HTTP adapter on localhost.
- **The one command anyone can repeat today**, with no model and no key, from the root of a clone:

  ```bash
  impulse run examples/config.yaml --dry-run
  ```

- **What success looks like.** Exit code 0 and `6 scenarios, 6 passed, 0 failed`, with the warning that judge checks were skipped. Offline checks like this one are marked "verified" on the pages that show them; they are not rows of the table.

## Details

### The table

Terms used in the rows: text-to-speech (TTS), speech-to-text (STT), time to first audio (TTFA), word error rate (WER).

| date | Impulse version | OS | Python | hardware (RAM) | path | models | commands run | result | verified by |
|---|---|---|---|---|---|---|---|---|---|
| TEMPLATE: YYYY-MM-DD | 0.1.0 | OS and version | 3.x.y | RAM | free (Ollama + local_hf), hosted, LiveKit, Pipecat eval, or Pipecat websocket | the Ollama tags, Hugging Face ids, or provider model strings actually used | for example `impulse doctor` and `impulse run` with the config path | what passed, what failed, run time, anything that needed a workaround | name |
| 2026-10-24 | 0.1.0 (wheel from commit d272dbd) | Windows 11 | 3.12 | 8 GB RAM | LiveKit (local `livekit-server` 1.13.10 `--dev`, livekit-agents 1.8.8 test agent with explicit dispatch), text and voice; voice through `openai_compatible` text-to-speech (TTS) and speech-to-text (STT) through a local Deepgram proxy | judge `gemini/gemini-3.1-pro-preview`, fast and simulator `gemini/gemini-3.5-flash-lite`; voice Deepgram `aura-2-thalia-en` and `nova-3` (# example, check the current name) | `impulse doctor` (all PASS), `impulse run` text and voice configs, `impulse compare`, `--fail-on critical` | 4 runs × 7 scenarios ran end to end: tool calls, heard_text (100%), time to first audio (TTFA) and VOICE-011 captured; seeded bugs caught. Workarounds: agent `agent_name` set (automatic dispatch failed on a busy CPU), agent audio output off for text-mode runs (30 s turn timeout), tool results carry a boolean `success` key. Defects found are listed in the campaign report | Shrikant (campaign run with Claude Code) |
| 2026-10-24 | 0.1.0 (wheel from commit d272dbd) | Windows 11 | 3.12 | 8 GB RAM | Pipecat eval (pipecat-ai 1.12.0 test bot started with `-t eval`), text and voice; voice through `openai_compatible` text-to-speech (TTS) and speech-to-text (STT) through a local Deepgram proxy | judge `gemini/gemini-3.1-pro-preview`, fast and simulator `gemini/gemini-3.5-flash-lite`; voice Deepgram `aura-2-thalia-en` and `nova-3` (# example, check the current name) | `impulse doctor` (all PASS, live handshake), `impulse run` text and voice configs, `impulse compare`, `--fail-on critical` | 3 runs × 6 scenarios end to end and a 4th (voice) stopped after 4 of 6 scenarios by the Gemini daily quota: tool calls, heard_text (100%), `get_system_prompt`, per-service metrics captured; seeded Reserve Bank of India (RBI) Know Your Customer (KYC) bugs caught as critical. Workaround: tool results carry a boolean `success` key. Defects found are listed in the campaign report | Shrikant (campaign run with Claude Code) |
| 2026-10-25 | 0.1.0 (wheel from commit d1e00c9) | Windows 11 | 3.12 | 8 GB RAM | LiveKit (same server and SupportBot test agent as on 2026-10-24, LiveKit Agents 1.8.8, explicit dispatch), text and voice, normal and BUGGY (seeded bugs); `agent.livekit.turn_timeout_s: 90` in text mode; voice through `openai_compatible` text-to-speech (TTS) and speech-to-text (STT) through the same local Deepgram proxy | same as on 2026-10-24: judge `gemini/gemini-3.1-pro-preview`, fast and simulator `gemini/gemini-3.5-flash-lite`; voice Deepgram `aura-2-thalia-en` and `nova-3` (# example, check the current name) | `impulse run` text and voice configs, normal and BUGGY; `impulse compare` normal vs BUGGY at `--fail-on critical` and `--fail-on high` | 4 runs × 7 scenarios plus one re-run. The campaign 1 workarounds were removed: natural tool results, agent audio on in text mode. Normal text run 7/7 passed. 0 false alarms in 53 failed checks; both seeded bugs caught in text and voice, including UNIV-025. heard_text 100%, VOICE-011 0.006 to 0.011. Voice runs fail on real latency: time to first audio (TTFA) p50 3.2 to 4.0 s. One voice turn hit the 30 s turn timeout; its partial turn was kept. Not Impulse: the worker declined 4 jobs on the busy CPU, so the BUGGY text run was repeated | Shrikant (campaign run with Claude Code) |
| 2026-10-25 | 0.1.0 (wheel from commit d1e00c9) | Windows 11 | 3.12 | 8 GB RAM | Pipecat eval (same pipecat-ai 1.12.0 BankBot test bot as on 2026-10-24, started with `-t eval`), text and voice, normal and BUGGY (seeded bugs); voice through `openai_compatible` text-to-speech (TTS) and speech-to-text (STT) through the same local Deepgram proxy | same as on 2026-10-24: judge `gemini/gemini-3.1-pro-preview`, fast and simulator `gemini/gemini-3.5-flash-lite`; voice Deepgram `aura-2-thalia-en` and `nova-3` (# example, check the current name) | `impulse run` text and voice configs, normal and BUGGY; `impulse compare` normal vs BUGGY at `--fail-on critical` and `--fail-on high` | 4 runs × 6 scenarios, every scenario run, no quota error; one voice scenario ended in the 30 s turn timeout. Natural tool results, no workaround. 0 false alarms in 77 failed checks; both seeded Reserve Bank of India (RBI) Know Your Customer (KYC) bugs caught as critical in text and voice. Text replies no longer doubled. heard_text 100%, VOICE-011 0.010 to 0.022. Two adapter defects found (a timed-out turn dropped, a tool call that raised not marked failed); fixed in fixes round 3 with tests, not re-run live | Shrikant (campaign run with Claude Code) |
| 2026-10-26 | 0.1.0 (editable install from the repository, fixes round 3) | Windows 11 | 3.12 (fresh virtual environment) | 8 GB RAM | light local voice: `pip install -e .[light]` (kokoro-onnx 0.6.1, faster-whisper 1.2.1, onnxruntime 1.31.0, no PyTorch); `local_light` text-to-speech (TTS) and speech-to-text (STT) in voice mode, against a scratch agent that spoke with the same Kokoro engine | Kokoro-82M ONNX voices and Whisper `small`; simulator `gemini/gemini-3.5-flash-lite` (# example, check the current name) | `impulse run` in voice mode, 3 caller turns per language, deterministic metrics only (no judge call) | Round-trip word error rate (WER), VOICE-011: en-US 0.000, es-MX 0.048 ("sí" heard as "si"), hi-IN 0.88 (poor; harness warning printed). Wall time for 3 caller turns: 126 s, 124 s and 172 s; about 40 to 60 s per caller turn, including a model load of about 30 s per run. About 1.3 to 1.4 GB of RAM. First use downloads about 354 MB (Kokoro) and 484 MB (Whisper `small`). Hindi speech-to-text with Whisper `small` is not recommended | Shrikant (run with Claude Code on the maintainer's PC) |
| 2026-10-26 | 0.1.0 (fixes round 3) | Windows 11 | 3.14 (the repository's development virtual environment) | 8 GB RAM | `provider: litellm` speech-to-text (STT) with Deepgram, key from the environment, no proxy; input: the Kokoro text-to-speech (TTS) clips of the light voice check | `deepgram/nova-3` (# example, check the current name) | the `litellm` STT provider on one clip per language: en-US, es-MX, hi-IN | Word error rate (WER): en-US 0.00 (exact, 2.3 s), es-MX 0.00 (exact, 1.0 s), hi-IN 0.22 ("order number" in Latin script, one spelling variant, 1.7 s). Word timings and confidence returned every time. The scenario language reached Deepgram through `detect_language` | Shrikant (run with Claude Code on the maintainer's PC) |
| 2026-10-26 | 0.1.0 (fixes round 3) | Windows 11 | 3.14 (the repository's development virtual environment) | 8 GB RAM | HTTP adapter (`adapter: http`) against the example echo agent `examples/http/server.py` on localhost |  | `python examples/http/server.py`, then the HTTP adapter driven from a short Python script: start request, outbound opener, a turn with the `book_slot` tool call, close request | The adapter worked against the example server. On Windows, `localhost` tried IPv6 first and added about 0.4 s per request, so the docs suggest `127.0.0.1`. A Node.js agent is not verified | Shrikant (run with Claude Code on the maintainer's PC) |

### Rows and what they mean

Until a row exists for a setup, every page that describes it calls it "not verified". Some commands are marked verified in the docs themselves, for example the dry run of `examples/config.yaml`. They were run offline while the docs were written, and are not rows of this table.

The test campaign reports and the notes of the round-3 checks are kept by the maintainer, outside the repository. The fixes they led to are listed in the [Changelog](changelog.md).

Not verified yet:

- the free path with Ollama models, and `local_hf` speech on real hardware;
- LiveKit Cloud, and the Pipecat websocket mode against a live bot;
- Groq, OpenAI or ElevenLabs voice through `provider: litellm`;
- a real Vapi or Retell adapter, and a Node.js agent behind `adapter: http`;
- noise simulation (experimental) against any live agent. Its offline tests cover the mixer (the signal-to-noise ratio within 0.5 dB of the target, determinism, no clipping, resampling), the config, a generic handler in voice mode with mocked speech, a custom-style adapter, the HTTP `user_audio` field against a mock server, the report and `impulse doctor`, and the dry run of `examples/config.noise.yaml` (verified, 2026-11-14). Not run yet: LiveKit, Pipecat (eval and websocket), a real HTTP server, a custom adapter for a hosted platform, and a real speech-to-text model on noisy audio.

### How a row is added

The maintainer runs a setup end to end on real hardware. Examples are the free path of [Run Impulse for free](run-for-free.md), a live LiveKit agent, or a Pipecat bot started with `-t eval`. The row records the date, the exact versions and models, the commands, and the outcome. Rows are never added for setups that were not run. If you have run one of these setups, send the maintainer the details listed in the TEMPLATE row.

## Next steps

- [Run Impulse for free](run-for-free.md): the setup that most needs a row.
- [Adapters](adapters.md): the LiveKit, Pipecat and HTTP setups.
- [Known limitations](known-limitations.md): what the runs above showed Impulse cannot do yet.
- [Contributing](contributing.md): how to report results.

---

<!-- docs-nav -->
[← Previous: Contributing](contributing.md) | [Docs home](index.md) | [Next: Changelog →](changelog.md)
