<!-- SPDX-License-Identifier: Apache-2.0 -->
# Contributing to Impulse Evals

Thank you for helping. The full guide, with every rule a change follows and how licensing works for contributions, is the [Contributing](docs/contributing.md) page of the docs.

## Quick start

From the root of your clone, in a virtual environment, with Python 3.11 or newer:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Success: every test passes (skips allowed). The suite makes no real network or LLM call.

Before you open a pull request:

- Add or extend tests for any change in behaviour, and never edit an existing test to make it pass. If a test looks wrong, say why in the pull request.
- Put no model names in source code; models come from `config.yaml`, the environment, or `config.toml`.
- Start every new source file with the Apache-2.0 SPDX licence header ([Contributing](docs/contributing.md#3-make-your-change)).
- Sign off every commit with `git commit -s` (see below).
- If you changed a page in `docs/`, rebuild the Claude skill reference and commit the result:

  ```bash
  python scripts/build_skill_reference.py
  ```

## Licence and sign-off (DCO)

- **Licence.** All of Impulse, including the compliance packs, is under the Apache License 2.0 ([LICENSE](LICENSE), [LICENSING.md](LICENSING.md)). Contributions are accepted under the Apache License 2.0.
- **Developer Certificate of Origin.** Every commit must carry a `Signed-off-by:` line. By signing off you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/): you wrote the change, or otherwise have the right to submit it under the Apache License 2.0. Sign off with:

  ```bash
  git commit -s
  ```

  The sign-off must match the commit author's name and email. To fix commits that are missing it: `git commit --amend -s` for the last commit, or `git rebase --signoff main` for several. Pull requests with commits that are not signed off are not merged.

## Help us test noise simulation

[Noise simulation](docs/voice-testing.md#noise-simulation-experimental) is experimental: its offline tests pass, but it has not been run against a live agent yet. If you have a voice agent, a run on it is the most useful contribution right now. **Testers who send a report are added to the contributors list in [CONTRIBUTORS.md](CONTRIBUTORS.md)**, unless they ask not to be.

**Run it on your own agent:**

1. Take a `config.yaml` that already runs in voice mode, and pick two or three scenarios.
2. Run it once clean: `impulse run config.yaml`.
3. Add noise to `voice_config` and run again, for example `noise: {profile: pink, snr_db: 10}`. Try 20, 10 and 5 dB. [examples/config.noise.yaml](examples/config.noise.yaml) shows four levels in one run.
4. Compare the runs: `impulse compare RUN_ID_CLEAN RUN_ID_NOISY`.
5. Open the noisy report and listen to a few caller turns: the noisy player, and "clean caller audio (before noise)" under it. Does the noise sound right at that level?

Make sure your agent actually hears the noise: an HTTP agent needs `agent.http.send_audio: true`, and a generic handler needs a `user_audio` parameter. LiveKit, Pipecat and custom adapters get the noisy audio as is.

**Report your results** with the "Noise simulation test report" issue template ([.github/ISSUE_TEMPLATE/noise-simulation-test-report.yml](.github/ISSUE_TEMPLATE/noise-simulation-test-report.yml)). It asks for:

- the adapter and the agent platform (LiveKit, Pipecat eval or websocket, HTTP, custom, Vapi or Retell through a custom adapter);
- the language, the noise type and the signal-to-noise ratio (SNR) levels you ran;
- pass or fail per level, and the VOICE-009 value from the report;
- a short report snippet, and anything that looked wrong.

Remove personal data, keys and private transcripts before you paste; Impulse redacts keys, but check.

**Contribute noise recordings.** Impulse ships no recordings today. A recording can be added only when you made it yourself or it has a clear licence that allows redistribution, it contains no voice of an identifiable person, and it is a WAV file of 16000 Hz or more and at most a minute long. Details and how to propose one: [Contributing](docs/contributing.md#help-us-test-noise-simulation).

## Reporting bugs and asking for features

Use the issue templates: bug report, feature request, or noise simulation test report. Do not post keys, transcripts with personal data, or details of a security weakness in a public issue; see [SECURITY.md](SECURITY.md) for private reporting.

## Code of conduct

Everyone taking part in this project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
