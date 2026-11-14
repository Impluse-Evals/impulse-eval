<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Roadmap

This page lists what is planned for Impulse after 0.1.0, so you can see where it is going and tell us what matters to you.

## Quick version

- **What it is.** The features planned after 0.1.0, and the small known issues to fix soon. Nothing planned here is built yet; what 0.1.0 cannot do today is in [Known limitations](docs/known-limitations.md).
- **What comes first.** Repeated samples per scenario, more degraded caller audio (background noise is in 0.1.0 as an experimental feature; packet loss, a low-quality microphone, and noise that continues after the caller stops speaking are still planned), and deliberate barge-in by the simulated caller.
- **How to influence it.** Open an issue or a discussion on GitHub, or upvote an existing one. Contributions are welcome ([Contributing](docs/contributing.md)).

## Details

### Simulation

| Planned | What it adds |
|---|---|
| Repeated samples per scenario | Run each scenario several times in one run, with a pass rate, to tell flaky results from real failures. |
| Degraded caller audio | Background noise is in 0.1.0 (experimental; see [Voice testing](docs/voice-testing.md#noise-simulation-experimental)). Still planned for the simulated caller: packet loss, a low-quality microphone, and noise that continues after the caller stops speaking. |
| Deliberate barge-in | Execute `inject_events`, so the simulated caller interrupts on purpose, plus a turn-taking score. |
| End-state check hook | Let your own code check your backend after a call, for example that a booking really exists. |
| Audio-native simulated caller | `simulator.mode: audio_native`, a caller that talks in audio instead of text turned into speech. |

### Reports and results

| Planned | What it adds |
|---|---|
| Compressed report audio (v1.1) | Smaller HTML reports, with audio in a compressed format browsers play. |
| Per-stage latency for every adapter | Speech-to-text, language model and text-to-speech timings in the report. Today only Pipecat reports them, as information. |
| `impulse summary` | One page across many runs: several agents, text and voice, and trends over time. |
| `impulse import` | Judge real production calls offline with the same checks, and turn failures into regression scenarios. |

### Voices, adapters and packs

| Planned | What it adds |
|---|---|
| Stronger free Hindi speech-to-text | A light local option that hears Hindi well; Whisper `small` is poor at it today. |
| More voice providers | More providers through `provider: litellm`, and a plugin route for others. |
| Ready-made hosted-platform adapters | Adapters for platforms such as Vapi and Retell. Today you write them with `adapter: custom`. |
| More compliance packs | Packs for more regions and industries, including community packs through a plugin route. |
| More Pipecat transports | Pipecat over Daily, SmallWebRTC or LiveKit, and starting bot processes for you. |

### Small known issues

The final live test of 2026-11-02 found these. None of them changes a verdict, and none exposes a secret in normal use. They are planned for an early update.

- **Redaction:**
  - values shorter than 4 characters are never hidden;
  - text a Pipecat or LiveKit bot sends as malformed tool arguments is stored as it came, so a sensitive value inside it is not hidden;
  - the judge and `impulse calibrate` rely on later redaction steps instead of the run's full set of known keys;
  - the `litellm` voice provider's own error text does not hide credentials written into its `base_url` (the runner and `impulse doctor` hide them).
- **Voice providers:**
  - a `litellm` text-to-speech error answer is reported with the wrong advice (as audio that needs the `voice` extra);
  - `provider: litellm` asks for a key even when `base_url` points at a server that needs none;
  - the light local voice drops short words in brackets such as "(no)" from mixed-language text, in rare cases;
  - `impulse doctor` tests speech-to-text with a tone, so its "no word timings" note can be wrong.
- **Reports and texts:**
  - a check pointing at a turn that does not exist gets a link to nowhere;
  - a few UNIV-023 and UNIV-025 reasons read oddly ("expected null, got null", an empty "call failed:");
  - with `agent.tools: []`, the judge sees an empty list header.
- **Compare and gates:**
  - `impulse compare` and `--fail-on any` trust the stored pass/fail instead of recomputing it (only a hand-edited `run.json` can disagree);
  - the "newly failing checks" list includes latency metrics that move between two runs of the same bot (the exit code is not affected).
- **Word error rate:** a few number and date forms count one or two extra errors: `+91` numbers, ranges such as "3-5 days", "may" or "march" next to a number, a misheard word next to a digit, and decimals before "lakh". A real mishearing is never hidden.
- **Pipecat:** a tool result that arrives after its turn ended is dropped silently, and a function call without an ID (not seen with pipecat-ai 1.12) would be recorded twice.

### Not on this list

Ideas not listed here are not ruled out. Open a discussion with your use case.

## Next steps

- [Known limitations](docs/known-limitations.md): what 0.1.0 does not do yet, with workarounds.
- [Contributing](docs/contributing.md): how to send a change.
- [FAQ](docs/faq.md): how Impulse compares with other tools.
- [Changelog](docs/changelog.md): what changed in each release.

---

<!-- docs-nav -->
[← Previous: Changelog](docs/changelog.md) | [Docs home](docs/index.md) | [Next: Claude skill →](docs/skill.md)
