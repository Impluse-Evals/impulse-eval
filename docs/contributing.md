<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Contributing

Contributions are welcome, and this page explains how to set up a development environment, the rules every change follows, and how licensing works for contributions.

## Quick version

- **Set up and test,** from the root of your clone, in a virtual environment:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

- **What success looks like.** `impulse --help` lists the commands, and every test passes (skips allowed). The suite makes no real network or LLM call.
- **The main rules.** Write tests first, and never edit a test to make it pass. Put no model names in source code. Copy normative texts word for word. Start every source file with its SPDX licence header.
- **Licence of your change.** A contribution is under the licence of the file it changes. Most files are AGPL-3.0-only. The plugin API files, including `impulse_evals/plugin_types.py`, are Apache-2.0. The compliance pack files are under the Elastic License 2.0.

## Details

### 1. Set up a development environment

Requires Python 3.11 or newer (tested on 3.11, 3.12 and 3.14). Python 3.10 is not supported.

From the root of your clone, in a virtual environment:

```bash
pip install -e ".[dev]"
```

The `dev` extra adds pytest-asyncio, ruff, and mypy. Add `local`, `voice`, `livekit`, or `pipecat` when you work on those parts; the test suite does not need them.

Success: `impulse --help` lists the commands.

### 2. Run the tests

```bash
python -m pytest -q
```

The whole suite runs offline. It mocks LiteLLM, the voice servers, the Pipecat channel, and the LiveKit room, and makes no real network or LLM call. A test that needs an optional package is skipped when that package is missing. The workflow in `.github/workflows/ci.yml` runs the same command on Python 3.11, 3.12, and 3.14 (not verified: CI has not run yet).

Success: every test passes (skips allowed).

### 3. Make your change

The rules every change follows:

- **Tests come first, and tests are not edited to make them pass.** A change in behaviour comes with new or extended tests. If an existing test looks wrong, say why in your pull request instead of changing it to match the code.
- **No real network or LLM calls in tests.** Use the fixtures in `tests/conftest.py` (`mock_llm`, `mock_voice_server`, `mock_pipecat_channel`, the example agent).
- **No hardcoded model names in source code.** Models come from `config.yaml`, the environment, or `config.toml`. Templates use `YOUR_..._MODEL` placeholders. A test scans the source for model strings.
- **Exact field names.** The data models in `models.py` and `plugin_types.py` are a stored format: `impulse report` and `impulse compare` read old `run.json` files. Use their names exactly, never abbreviations.
- **Normative texts are copied word for word.** These must match the specification and [METRICS.md](../METRICS.md) exactly; METRICS.md is frozen for 0.1.0:
  - the judge prompts and safety floor (`judge/prompts.py`);
  - the metric criteria (`metrics/`);
  - the compliance criteria (`assertions/`).
- **Lazy imports.** `import impulse_evals` must work with no extra installed. Import optional libraries inside the function that needs them.
- **Licence headers.** Every source file starts with its SPDX header. A test checks this.
  - `# SPDX-License-Identifier: AGPL-3.0-only` for core Python, YAML, and TOML files;
  - `# SPDX-License-Identifier: Apache-2.0` for the five plugin API files;
  - `# SPDX-License-Identifier: Elastic-2.0` for the four pack files;
  - `<!-- SPDX-License-Identifier: AGPL-3.0-only -->` for HTML files (for example `impulse_evals/templates/report_template.html`) and for Markdown under `docs/` and `examples/`, the README, and `LICENSING.md`;
  - none for JSON files, binary fixtures, and the `LICENSE*` files.
- **Out of scope for v1:**
  - a dashboard or server, a database, PDF reports, and telephony;
  - Vapi / Retell / Bland connectors beyond their stubs;
  - Pipecat over Daily, SmallWebRTC, or LiveKit transports.
- **Spec decisions are explained.** When the specification is ambiguous, choose the simplest reading and give a one-line reason in your pull request.

Success: `python -m pytest -q` passes, and `ruff check impulse_evals` reports nothing new.

### 4. Documentation

The pages in `docs/` follow the same accuracy rules as the code. Every command, flag, key, default, and output shown must match the code. Anything not run is labelled "not verified".

Each page:

- starts with its SPDX header, an H1 title, and a one-sentence introduction;
- spells out acronyms on first use, for example word error rate (WER), time to first audio (TTFA), speech-to-text (STT), and text-to-speech (TTS);
- ends with "Next steps" links.

Config blocks marked `# impulse:validate` must load. Model names appear only on lines marked `# example, check the current name`. `docs/verified-setups.md` lists only setups the maintainer ran. `tests/test_docs_examples.py` checks these rules.

### Licensing of contributions

Contributions are accepted under the licence of the file they change:

| you change | your contribution is under |
|---|---|
| a core file (everything not listed below, including `docs/`, `examples/`, and `tests/`) | AGPL-3.0-only |
| a plugin API file: `impulse_evals/plugin_types.py`, `impulse_evals/adapters/base.py`, `impulse_evals/assertions/base.py`, `impulse_evals/metrics/base.py`, `impulse_evals/judge/base.py` | Apache-2.0 |
| a compliance pack file: `impulse_evals/assertions/dpdp.py`, `rbi_kyc.py`, `trai_dlt.py`, `hipaa.py` | Elastic License 2.0 |

The plugin API files are under the Apache License 2.0, so that adapters, metrics, and judges can be built against them without AGPL obligations. They import no AGPL code. A change to one of them may import only:

- the other plugin API files;
- the standard library;
- third-party packages.

It may name an AGPL `impulse_evals` module only inside an `if TYPE_CHECKING:` block (annotations, never executed). A test checks this rule, including that importing a plugin API file loads no AGPL module. New files take the licence of the part they belong to.

What each licence means in plain language: [LICENSING.md](../LICENSING.md). This page is not legal advice; the LICENSE files govern.

### Help us test noise simulation

[Noise simulation](voice-testing.md#noise-simulation-experimental) is experimental: its offline tests pass, but it has not been run against a live agent yet. If you have a voice agent, a run on it is the most useful contribution right now. Testers who send a report are added to [CONTRIBUTORS.md](../CONTRIBUTORS.md), unless they ask not to be.

**Run it on your own agent:**

1. Take a `config.yaml` that already runs in voice mode, and pick two or three scenarios.
2. Run it once clean: `impulse run config.yaml`.
3. Add noise to `voice_config` and run again, for example `noise: {profile: pink, snr_db: 10}`. Try 20, 10 and 5 dB. [examples/config.noise.yaml](../examples/config.noise.yaml) shows four levels in one run.
4. Compare the runs: `impulse compare RUN_ID_CLEAN RUN_ID_NOISY`.
5. Open the noisy report and listen to a few caller turns: the noisy player, and "clean caller audio (before noise)" under it. Does the noise sound right at that level?

Make sure your agent actually hears the noise: an HTTP agent needs `agent.http.send_audio: true`, and a generic handler needs a `user_audio` parameter. LiveKit, Pipecat and custom adapters get the noisy audio as is.

**Report your results** with the "Noise simulation test report" issue template ([.github/ISSUE_TEMPLATE/noise-simulation-test-report.yml](../.github/ISSUE_TEMPLATE/noise-simulation-test-report.yml)). It asks for:

- the adapter and the agent platform (LiveKit, Pipecat eval or websocket, HTTP, custom, Vapi or Retell through a custom adapter);
- the language, the noise type and the SNR levels you ran;
- pass or fail per level, and the VOICE-009 value from the report;
- a short report snippet, and anything that looked wrong.

Remove personal data, keys and private transcripts before you paste; Impulse redacts keys, but check.

**Contribute noise recordings.** Impulse ships no recordings today. A recording can be added only when:

- you made it yourself, or it has a clear licence that allows redistribution (for example CC0 or CC BY), named in your pull request with its source;
- it contains no voice of an identifiable person: no conversations you can follow, no names, no announcements that identify someone;
- it is a WAV file of 16000 Hz or more and at most a minute long.

Describe where and how it was recorded (for example "street corner, Pune, phone microphone"). Until a recordings folder exists, open an issue first.

### Reporting problems

Open an issue with:

- the command you ran;
- the output (key values are already redacted by Impulse, but check before you paste);
- your Python version and OS;
- the versions of any extras involved.

Do not post keys, transcripts with personal data, or details of a security weakness in a public issue. Contact the maintainer privately first.

## Next steps

- [Architecture](architecture.md): how the package is organised.
- [Adapters](adapters.md): the plugin API for your own adapter.
- [Metrics](metrics.md): the normative metric definitions.
- [Verified setups](verified-setups.md): send the maintainer your results.
- [Voice testing: noise simulation](voice-testing.md#noise-simulation-experimental): what you are testing when you help.

---

<!-- docs-nav -->
[← Previous: Architecture](architecture.md) | [Docs home](index.md) | [Next: Verified setups →](verified-setups.md)
