<!-- SPDX-License-Identifier: Apache-2.0 -->
# FAQ

Short answers to the questions people ask most about Impulse.

## Quick version

- **What it is.** One answer per question, each a few lines long, with a link to the page that has the details.
- **The question most people have first:** can I try it without any key? Yes, a dry run needs no model and no key:

  ```bash
  impulse run examples/config.yaml --dry-run
  ```

- **What success looks like.** Exit code 0 and `6 scenarios, 6 passed, 0 failed` with the warning that judge checks were skipped (verified).

## Details

### Why not Python 3.10?

Pipecat's eval mode needs a newer `pipecat-ai` than Python 3.10 supports (pipecat-ai 1.12 requires Python 3.11 or newer), and Python 3.10 reaches end of life in October 2026.

### Do I need a paid API key?

No. With Ollama serving the judge and simulator models, and local speech models for voice, no key is asked for anywhere: [Run Impulse for free](run-for-free.md). Paid providers work too, through LiteLLM model strings. The path with Ollama models is not verified on real hardware yet.

### Do I need a GPU?

No. The light local voice (`provider: local_light`) runs speech on the CPU, without PyTorch. Its round trip was verified on 2026-10-26 on a laptop with 8 GB RAM. Expect about 40 to 60 s per caller turn in a voice run there ([Voice testing](voice-testing.md#light-local-voice-no-pytorch-weak-laptops)). Ollama can also run the language models on a CPU, slowly (not verified).

### Do I need a paid voice service?

No. Two free voices run on your machine: `local_light` (no PyTorch) and `local_hf` (Hugging Face models with PyTorch).

- `local_light` was verified on 2026-10-26: good in English (en-US) and Spanish (es-MX).
- For Hindi, its Whisper `small` speech-to-text was poor. Keep its text-to-speech and pair it with a stronger speech-to-text, for example Deepgram through `provider: litellm` (paid).
- `local_hf` is not verified on real hardware yet.

### Why Impulse instead of Pipecat's own evals or DeepEval?

You do not have to choose. The same Pipecat bot started with `-t eval` works with both `pipecat eval` and Impulse, one at a time. What Impulse adds:

- Judging against your agent's own system prompt, with no rubric to write.
- Tool-call checks: UNIV-023 (judged) and UNIV-025 (exact expected and forbidden calls, checked by code).
- Optional compliance packs.
- One cohort per language and accent, never blended.
- `impulse compare` with exit codes for CI.
- A report you can listen to: the call recording, a player per turn, and "Hear turn N" links.
- One tool across LiveKit, Pipecat, HTTP and custom adapters.

This list says what Impulse does. It is not a feature-by-feature comparison with other tools.

### Can I test Vapi, Retell or a Node.js agent?

Yes, in two ways.

- An agent behind a web API in any language (Node.js, Go, ...): use `adapter: http`. Impulse posts each turn as JSON to your endpoint ([Adapters](adapters.md#http-adapter)). Verified against the example server `examples/http/server.py` on localhost; the Node.js sketch in the docs is not verified.
- A hosted voice platform such as Vapi or Retell: write a small `AgentInterface` subclass (Apache-2.0) with the platform's own client library. Load it with `adapter: custom` ([Adapters](adapters.md#writing-your-own-adapter)). No ready-made Vapi or Retell adapter exists yet.

### Which models should I use?

Impulse does not recommend specific models, and its templates use `YOUR_..._MODEL` placeholders.

- Use a strong model as the primary judge: it decides critical and high compliance checks.
- Use a faster one for the fast judge and the simulated caller.
- [Calibrate](calibration.md) the judge before you trust critical verdicts, especially from a model of 14B parameters or fewer.

### Do I have to write a rubric?

No. The judge reads your agent's own system prompt and judges the agent against it, plus a fixed safety floor. The floor forbids unqualified medical, legal, or financial advice, false urgency, fabricated facts, credential requests, and claiming to be human. Put the rules you care about in the system prompt ([Concepts](concepts.md)).

### Does Impulse send my data anywhere?

Only to the providers you configure: the judge, simulator, and voice providers receive your system prompt and transcripts. On the free local path nothing leaves your machine. Impulse sends no telemetry ([Privacy and security](privacy-and-security.md)).

### Is a passing compliance check a legal certification?

No. Every result says "Technical signal only. Not legal certification.", and the `regulation_ref` strings are informational ([Compliance packs](compliance-packs.md)).

### Which compliance packs run for my scenarios?

Only the rules listed in a scenario's `assertions` run. When `impulse generate` drafts scenarios, its domain defaults add India's packs (`dpdp`, `rbi_kyc`, `trai_dlt`) only for languages with the region IN, such as `hi-IN` or `en-IN`. An `en-US` or `es-MX` scenario gets no pack by default. You can list any pack's rules for any language ([Compliance packs](compliance-packs.md#packs-in-the-domain-defaults)).

### Why does the hipaa pack not run for my healthcare scenarios?

It is opt-in: no domain default adds it, not even `healthcare`. List its rules explicitly in a scenario's `assertions`, for example `hipaa.identity_before_phi`.

### What is the difference between UNIV-023 and UNIV-025?

UNIV-023 asks the judge three things. Did the agent call the right tools with correct arguments? Did it claim success when a call failed? Did it claim an action ("you're booked") without calling any tool? UNIV-025 is checked by code against the exact `expected_tool_calls` and `forbidden_tool_calls` of a scenario. That includes how many times a tool may be called ([Metrics](metrics.md)).

### Why is a metric "skipped"?

There are three cases:

- its situation never occurred, for example no interruption for VOICE-007;
- it needs something the run does not have: audio in text mode, the agent's `heard_text`, or word timings;
- the run was a dry run.

A skipped metric never fails a scenario; its reason says which case applies.

### Why did my scenario fail with an error instead of a verdict?

An error means the conversation or a critical check could not be completed. For example, the agent could not be reached, a turn timed out, or your agent raised. It also happens when the judge returned no usable verdict for a critical assertion. The scenario fails and `impulse run` exits 1 at every `--fail-on` level ([Troubleshooting](troubleshooting.md)).

### Does inject_events work?

Not yet. `inject_events` is modeled so stored results stay compatible, but execution is reserved for v2. The scenario runs without injection. The console and report say "inject_events defined but execution reserved for v2; this scenario ran without injection." ([Writing scenarios](writing-scenarios.md)).

### Can I test with background noise?

Yes, experimentally. In voice mode, add `noise: {profile: pink, snr_db: 10}` to `voice_config`, or give one scenario its own `noise` block. Impulse mixes generated noise (white, pink or brown) or your own WAV recording into the caller's audio at that signal-to-noise ratio (SNR): 20 is mild, 10 a busy street, 5 harsh. Your agent hears the noisy audio; Impulse's own harness check (VOICE-011) still uses the clean audio. Watch VOICE-009 (how well the agent heard), VOICE-027 (does it ask the caller to repeat) and task completion, and run `impulse compare` between a clean and a noisy run. It is not yet verified against a live LiveKit or Pipecat agent ([Voice testing: noise simulation](voice-testing.md#noise-simulation-experimental)).

### Is the same seed guaranteed to give the same conversation?

No. `seed` adds a reproducibility hint to the simulated caller's prompt; models and providers do not guarantee the same turns. Use `opening_user_message` when the first words must be exact.

### Can I use Impulse with Pipecat's own pipecat eval?

Yes. The same bot started with `-t eval` works with both tools, one at a time. Note that Impulse sets FULL function call reporting on the bot, and the setting stays on after the run ([Adapters](adapters.md#pipecat)).

### Can I test an agent that is not a Python function, LiveKit, or Pipecat?

Yes: use `adapter: http` or `adapter: custom` (above), or subclass `AgentInterface` and pass it to `impulse_evals.runner.run` from Python ([Adapters](adapters.md#writing-your-own-adapter)).

### Why is the HTML report so large?

Audio is embedded as WAV by default, so the report plays anywhere; a long voice run can reach tens of megabytes. Impulse warns when the embedded audio in the page exceeds 50 MB (counted as base64, which is what the page holds).

- Link the audio files instead: `impulse run config.yaml --no-embed-audio`, or `impulse report RUN_ID --no-embed-audio` for a stored run.
- A linked report plays only next to the run's `audio/` folder.
- Compressed audio is planned for v1.1 ([Reading the report](reading-the-report.md#embedded-or-linked-audio)).

### What does Impulse not do yet?

[Known limitations](known-limitations.md) lists every documented limitation in one place, each with a workaround.

### What licence is Impulse under? Can I use it commercially or offer it as a service?

All of Impulse, including the compliance packs, is under the Apache License 2.0. You may use, modify, and embed it, and offer it to others as a hosted or managed service, including commercially; keep the `LICENSE` and `NOTICE` files when you redistribute it. Testing your own agents creates no obligation to publish anything. The licence does not grant rights to the name "Impulse" or its logo. Releases already published on PyPI keep the licence they were published under ([Licensing](../LICENSING.md)).

### Was this called PULSE?

Yes. Impulse Evals was previously named PULSE; the package is now `impulse-evals`, the import package `impulse_evals`, and the command `impulse` ([Changelog](changelog.md)).

## Next steps

- [Getting started](getting-started.md): install and make a first run.
- [Reading the report](reading-the-report.md): what each part of the report shows.
- [Concepts](concepts.md): how Impulse judges a conversation.
- [Troubleshooting](troubleshooting.md): what the error messages mean.
- [Licensing](../LICENSING.md): what the licence means for you.

---

<!-- docs-nav -->
[← Previous: Troubleshooting](troubleshooting.md) | [Docs home](index.md) | [Next: Known limitations →](known-limitations.md)
