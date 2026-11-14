<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Impulse Evals documentation

These pages show how to install Impulse Evals (formerly PULSE), point it at your agent, run it for free, and read its report.

## Contents

1. [Getting started](getting-started.md)
2. [Run Impulse for free](run-for-free.md)
3. [Concepts](concepts.md)
4. [Configuration](configuration.md)
5. [Writing scenarios](writing-scenarios.md)
6. [Adapters](adapters.md)
7. [Voice testing](voice-testing.md)
8. [Metrics](metrics.md)
9. [Compliance packs](compliance-packs.md)
10. [Reading the report](reading-the-report.md)
11. [CLI reference](cli.md)
12. [CI integration](ci-integration.md)
13. [Calibration](calibration.md)
14. [Cookbook](cookbook.md)
15. [Troubleshooting](troubleshooting.md)
16. [FAQ](faq.md)
17. [Known limitations](known-limitations.md)
18. [Privacy and security](privacy-and-security.md)
19. [Architecture](architecture.md)
20. [Contributing](contributing.md)
21. [Verified setups](verified-setups.md)
22. [Changelog](changelog.md)
23. [Roadmap](../ROADMAP.md)
24. [Claude skill](skill.md): let Claude set up, run and read Impulse for you.

## Quick version

- **What it is.** A pytest-style evaluation framework for AI voice and chat agents. A simulated caller talks to your agent, in text or voice. An LLM judge grades every call against your agent's own system prompt. Any language, any stack, any industry.
- **The first command**, from the root of a clone after `pip install -e .` (no model and no key needed):

  ```bash
  impulse run examples/config.yaml --dry-run
  ```

- **What success looks like.** Exit code 0, `6 scenarios, 6 passed, 0 failed`, and a report at `.impulse/runs/<run_id>/report.html` (verified).
- **Where to go next.** [Getting started](getting-started.md) for a first real run, or [Run Impulse for free](run-for-free.md) with no paid key.

## Details

### What Impulse does

- **Plays a simulated caller against your agent**, in text or in voice. In voice mode the caller's words go to your agent through text-to-speech (TTS), and your agent's audio comes back through speech-to-text (STT).
- **Judges every conversation against your agent's own system prompt**, plus a small universal safety floor. A sales agent is judged on its sales instructions, an interview agent on its interview rules. There is no rubric to write and no domain-specific judge code.
- **Checks tool calls**:
  - the right tool with the right arguments (UNIV-023, judged);
  - exact per-scenario expectations such as "one `book_slot` call with a date and a phone number" (UNIV-025, checked by code);
  - no "you're booked" when the booking call failed or was never made.
- **Measures latency on the agent only**: Impulse's own TTS and STT time never enters the numbers.
- **Reports every language and accent as its own cohort**, never blended. The examples include Indian, US English and Mexican Spanish scenarios.
- **Diffs two runs** with `impulse compare`, with an exit code a CI pipeline can act on.
- **Writes a report you can listen to**: the call recording and a player per turn. Each failed check has a "Hear turn N" link.
- **Adds optional compliance packs.** India's packs are one example:
  - the Digital Personal Data Protection (DPDP) Act;
  - the Reserve Bank of India's Know Your Customer (KYC) guidance;
  - the Telecom Regulatory Authority of India (TRAI) rules for commercial calls, with their Distributed Ledger Technology (DLT) framework.

  An opt-in HIPAA pack covers US healthcare. When `impulse generate` drafts scenarios, it adds the India packs only for languages with the region IN. Every result is a technical signal only, not legal certification.
- **Connects to your agent** in four ways:
  - by calling a Python function (generic adapter);
  - as a client of a running LiveKit agent or Pipecat bot;
  - over JSON HTTP, for an agent in any language (`adapter: http`);
  - through your own adapter class (`adapter: custom`).

You bring the language models as LiteLLM model strings. With Ollama and local speech models, no paid key is ever needed.

### Where to start

| You want to | Read |
|---|---|
| Install, make a dry run, and run the example agent | [Getting started](getting-started.md) |
| Do it all with no API key | [Run Impulse for free](run-for-free.md) |
| Understand what the report shows, and hear a failed turn | [Reading the report](reading-the-report.md) |
| Learn how scenarios, the simulated caller, the judge and pass logic fit together | [Concepts](concepts.md) |
| Set up `config.yaml` and `.impulse/config.toml` | [Configuration](configuration.md) |
| Look up a command, flag or exit code | [CLI reference](cli.md) |
| Write scenarios, tool expectations, `seed` and `inject_events` | [Writing scenarios](writing-scenarios.md) |
| See every check Impulse runs | [Metrics](metrics.md) and [Compliance packs](compliance-packs.md) |
| Test in voice, or connect LiveKit, Pipecat, HTTP or your own adapter | [Voice testing](voice-testing.md) and [Adapters](adapters.md) |
| Gate merges, and measure how far to trust the judge | [CI integration](ci-integration.md) and [Calibration](calibration.md) |
| Copy a recipe: ecommerce, Talk2Neo, healthcare, BFSI | [Cookbook](cookbook.md) |
| Know what Impulse cannot do yet, with workarounds | [Known limitations](known-limitations.md) |
| Fix an error | [Troubleshooting](troubleshooting.md) and [FAQ](faq.md) |
| Check privacy, internals or how to contribute | [Privacy and security](privacy-and-security.md), [Architecture](architecture.md), [Contributing](contributing.md) |
| See what was really run, and what changed | [Verified setups](verified-setups.md) and the [Changelog](changelog.md) |

### What has been verified

The commands, flags, keys, defaults, and outputs on these pages were checked against the code of this release. Please report any difference you find.

- Run offline while writing the docs (Windows, Python 3.14), and marked "verified":
  - the dry run `impulse run examples/config.yaml --dry-run`;
  - `impulse init --profile free --non-interactive --language en-IN`;
  - `impulse doctor` without models;
  - `impulse compare` and `impulse report` on dry runs;
  - a custom adapter in a dry run.
- Real runs on the maintainer's PC: LiveKit and Pipecat test bots with hosted models, in text and voice (2026-10-24 and 2026-10-25). On 2026-10-26: the light local voice round trip, Deepgram speech-to-text through `provider: litellm`, and the HTTP adapter against the example server on localhost.
- Not verified yet: the free path with Ollama models, `local_hf` speech on real hardware, and LiveKit Cloud. The Pipecat websocket mode is not verified against a live bot either.

[Verified setups](verified-setups.md) lists the dates, versions, models and results.

### Licensing

- Core: AGPL-3.0-only. Running Impulse to test your own agents, inside your company or in CI, does not oblige you to publish your agent or your configs.
- Plugin API files (`plugin_types.py` and the four `base.py` files of `adapters`, `assertions`, `judge`, `metrics`): Apache License 2.0.
- Compliance packs (`dpdp`, `rbi_kyc`, `trai_dlt`, `hipaa`): Elastic License 2.0.

[LICENSING.md](../LICENSING.md) explains what each licence means for you in plain language. The licensing of the combined distribution (three licences in one package) must be reviewed by counsel before the first public release.

## Next steps

- [Getting started](getting-started.md): install and make a first run.
- [Run Impulse for free](run-for-free.md): the whole path with no API key.
- [Reading the report](reading-the-report.md): what each part of the report shows.
- [Known limitations](known-limitations.md): what to expect before you rely on a result.
- [README](../README.md): the project overview.

---

<!-- docs-nav -->
[← Previous: Overview](../README.md) | **Docs home** | [Next: Getting started →](getting-started.md)
