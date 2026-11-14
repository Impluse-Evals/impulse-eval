<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# CLI reference

This page lists every `impulse` command with its arguments, flags, defaults, output, and exit codes, as built in this release.

## Quick version

- `impulse` is Impulse's command line. `impulse --help` lists the commands, and `impulse <command> --help` shows a command's flags (verified).
- The commands most people need, in the order they use them:

```bash
impulse init --profile free --non-interactive --language en-IN   # write .impulse/config.toml
impulse doctor config.yaml                                        # check models, keys, and the agent connection
impulse run config.yaml --dry-run                                 # check the config with zero LLM calls
impulse run config.yaml                                           # run the scenarios, write a report
impulse compare RUN_A RUN_B                                       # what got better or worse between two runs
```

- Exit code 0 means success.
- Exit code 1 means Impulse ran and found a problem. For `impulse run`, that is the `--fail-on` condition.
- Exit code 2 means a usage or setup problem.
- A run writes `.impulse/runs/<run_id>/` with `run.json` and `report.html`. How to read the report: [Reading the report](reading-the-report.md).

Words used on this page: TTS is text-to-speech, and STT is speech-to-text.

## Details

### Running the CLI

- Every command prints plain text, never a traceback. An unexpected error prints `Error: unexpected error: ...` and exits 2.
- Usage errors, such as an unknown flag or value, are printed by the CLI framework in a boxed panel. They exit 2.
- Key values are redacted from everything the CLI prints.
- `python -m impulse_evals` runs the same CLI as `impulse`: the same commands, help, output, and exit codes. Use it when the `impulse` script is not on your `PATH`.
- Like any `python -m`, it puts the current folder first on Python's import path (`sys.path`). A module there with the same name as one Impulse imports is then used instead. The `impulse` script does not do this.

### Commands at a glance

| command | what it does |
|---|---|
| `impulse init` | writes `.impulse/config.toml` (and, for `--profile free`, a starter `config.yaml`) |
| `impulse doctor` | checks Python, the config, models, the agent connection, and voice providers |
| `impulse generate` | drafts test scenarios from your agent's system prompt |
| `impulse run` | runs the scenarios of a config and writes a run folder |
| `impulse report` | rewrites `report.html` of a stored run |
| `impulse compare` | diffs two runs; exits 1 on a critical regression (a critical assertion newly fails, or a scenario newly errors), and with `--fail-on high` also on a high regression |
| `impulse convert-failure` | turns a failed scenario into a regression scenario |
| `impulse calibrate` | measures how often the judge agrees with human labels |

### Exit codes

| exit code | meaning |
|---|---|
| 0 | success |
| 1 | `impulse run`: the `--fail-on` condition is met; `impulse compare`: a critical regression (with `--fail-on high`, also a high regression); `impulse generate` / `impulse convert-failure`: the model's answer was still invalid after one retry, or the model call failed |
| 2 | usage or configuration error (the message names the exact field, for example `scenarios[1].expected_tool_calls[0].name`), any `FAIL` row in `impulse doctor`, a run that cannot be found, end of input at a prompt ("Aborted."), or an unexpected error |

### impulse init

```bash
impulse init [--profile free|hosted|custom] [--language L] [--no-ping] [--non-interactive]
```

`impulse init` writes two files in the current directory:

- `.impulse/.gitignore` first: one line, `*`, so the folder stays out of version control;
- then `.impulse/config.toml`, which holds your keys. It is created readable and writable by you only. On Windows the folder's default permissions apply. Details: [Architecture](architecture.md#key-file-permissions).

An existing `config.toml` is overwritten only after you confirm ("Overwrite it?", default no). With `--non-interactive` it is never overwritten. `init` then rewrites `.impulse/.gitignore`, prints `.impulse/config.toml already exists; left unchanged.`, writes nothing else (no starter `config.yaml`), and exits 0. So a CI job can re-run it on a cached workspace.

| flag | default | meaning |
|---|---|---|
| `--profile` | `custom` | `free`: asks for the language, an Ollama tag per role, and the Hugging Face text-to-speech and speech-to-text model ids; never asks for a key. `hosted`: asks for the language, then the LLM keys (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`; input hidden) and optionally `IMPULSE_VOICE_API_KEY`, then a LiteLLM model string per role. `custom`: asks for the language, then a model string and an optional `api_base` per role, then only the keys those models need (the variable LiteLLM reads for each provider; none for a provider such as `bedrock/` that uses its own credentials). |
| `--language` | none | the default language code (for example `en-IN`); checked against `^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$`. Required with `--non-interactive`. There is no built-in default language. |
| `--no-ping` | off | skip the live 5-token ping of each configured model |
| `--non-interactive` | off | ask nothing and ping nothing (for CI); every value not given by a flag is written as its `YOUR_..._MODEL` placeholder |

- The roles are `primary` (primary judge), `fast` (fast judge), and `simulator` (the simulated caller).
- Each model answer is checked offline with LiteLLM. It is asked again when LiteLLM does not recognise it. A blank answer keeps the placeholder.
- With `--profile free`, an Ollama tag becomes the model string `ollama/<tag>`. A starter `config.yaml` (with your language and speech model ids) is written when the current folder has none. Otherwise the `voice_config` block is printed for you to add.

Output (verified, `impulse init --profile free --non-interactive --language en-IN` in a folder without `config.yaml`; on Windows the paths print with `\`):

```text
Wrote .impulse/config.toml (profile free, language en-IN) and .impulse/.gitignore
Wrote config.yaml (starter config: set agent.entrypoint and agent.system_prompt before the first run)
Next: impulse doctor
```

- Without `--language`, `--non-interactive` exits 2 with `Error: --language is required with --non-interactive; Impulse never assumes a language (for example --language en-IN).` (verified).
- `--language` is required on a re-run too, although the existing file keeps its own language.
- A failed ping is reported on its row ("not reachable: ..."). LiteLLM's own "Give Feedback / Get Help" banner is not printed.

### impulse doctor

```bash
impulse doctor [CONFIG]
```

`CONFIG` defaults to `./config.yaml` when it exists. Without any config, only the models from the environment and `.impulse/config.toml` are checked. It prints one row per check, each `PASS`, `FAIL`, or `INFO` (a note, never a failure):

- `python` (3.11 or newer), `config.toml` (loads, or not found), `config` (valid, plus one `INFO` row per config warning).
- `primary model`, `fast model`, `simulator model`: resolved, recognised by LiteLLM, its key present when it needs one, and answering a 5-token ping.
  - The key is the `*_API_KEY` variable LiteLLM reads for the provider ([Configuration](configuration.md#where-models-and-keys-come-from)).
  - A provider for which LiteLLM names no `*_API_KEY` variable (`bedrock/`, `vertex_ai/`, ...) is pinged without a key. Its row ends with "needs no Impulse key (LiteLLM uses the provider's own credentials)".
- The agent, by adapter:
  - generic: the entrypoint imports, and where the system prompt comes from;
  - `adapter: pipecat`: `pipecat.evals` is importable (eval mode), the connection in the configured mode receives `bot-ready`, and whether the bot answers `get_system_prompt` (an `INFO` row);
  - `adapter: livekit`: the `livekit` extra and the API key and secret variables (the connection itself is not checked);
  - `adapter: custom`: the `agent adapter` row, `PASS custom adapter {module}:{attr} imported (a subclass of AgentInterface)` or `FAIL` with the loading error. The class is imported, never created. An `INFO` row `agent connection` says "not checked by impulse doctor (custom adapter)";
  - `adapter: http`: the `agent http` row, a plain TCP connection to the turn URL's host and port with nothing sent (`PASS reachable: {url}` or `FAIL cannot connect to {url}: {error}`). When `api_key_env` is set, an `HTTP agent key` row says whether that variable is set. An `INFO` row `agent connection` says "the turn endpoint is not called by impulse doctor", since a turn can have side effects.
- Voice providers, when the config has `voice_config`. `litellm` is checked like `openai_compatible`: the TTS row synthesizes a short phrase, and the STT row transcribes it.
- Noise simulation (experimental), only when a `noise` block is set: an `INFO` row `noise simulation` that says the feature is experimental and lists the noise levels (or that noise is ignored in text mode), and one row per noise file, `noise file {name}`. It is `PASS` with the sample rate, channels and length; `FAIL` when the sample rate is outside 8000 to 192000 Hz; and adds an `INFO` row when the rate is below 16000 Hz or the file is shorter than 1 s ([Voice testing](voice-testing.md#noise-simulation-experimental)).

Result:

- No `FAIL` row: `All checks passed.` and exit code 0.
- Otherwise: `{n} check(s) failed. Fix every FAIL row, then run impulse doctor again.` and exit code 2.
- In Pipecat eval mode the check connects to the bot like a run does. So it also sets FULL function call reporting on the bot ([Adapters](adapters.md#pipecat)).

### impulse generate

```bash
impulse generate --prompt FILE --domain DOMAIN [--language L] [--count 11] [--out FILE] [--yes] [--voice-mode]
```

Asks the primary model for scenarios that test the agent described by the system prompt file. Impulse validates them and writes a YAML list. You review it and paste it under `scenarios:`.

| flag | default | meaning |
|---|---|---|
| `--prompt` | required | the agent's system prompt file |
| `--domain` | required | `fsi`, `healthcare`, `recruitment`, `ecommerce`, `edtech`, `telecom`, `realestate`, `travel`, `legal`, or `general`; sets the default assertions, metrics, and `max_turns` (any other value is treated as `general` with the warning "Unknown domain {domain}; using general defaults.") |
| `--language` | `[defaults] language` in `.impulse/config.toml` | the scenarios' language code; exit 2 when neither is set |
| `--count` | 11 | number of scenarios (mix: 30% happy path, 25% edge cases, 25% adversarial, 20% hallucination traps) |
| `--out` | `scenarios/generated.yaml` | output file, relative to the current directory |
| `--yes` | off | do not ask when the system prompt looks like another domain |
| `--voice-mode` | off | offer the voice metric list; every scenario gets VOICE-001, VOICE-003, and VOICE-005 |

- **Domain guard.** Impulse counts the distinct domain keywords in the system prompt (whole words, any case). The domain with the most hits, at least 3, is the detected one.
  - When it differs from `--domain` and `--domain` is not `general`, Impulse warns and asks `Continue with --domain {domain}?`. Answering no exits 2.
  - `--yes`, or no terminal, skips the question.
- **Invalid answers.** An invalid answer from the model is retried once with the error appended. Still invalid exits 1 and writes nothing.
- **The model** comes from `IMPULSE_MODEL_PRIMARY` or `[models] primary` in `.impulse/config.toml`. This command reads no `config.yaml`.
- **Redaction.** Known key values are replaced by `[REDACTED]` before the file is written. That covers every string value of the written scenarios, and the header comment. The file still loads.

### impulse run

```bash
impulse run CONFIG [--reporter json|html|all] [--dry-run] [--concurrency N] [--fail-on critical|high|any] [--embed-audio|--no-embed-audio]
```

Runs every scenario of `CONFIG` and writes `.impulse/runs/<run_id>/`: `run.json`, `scenarios/<name>.json` (always), and `report.html` (for `--reporter html` or `all`).

| flag | default | meaning |
|---|---|---|
| `--reporter` | `[defaults] reporter` in `.impulse/config.toml`, else `all` | `json`, `html`, or `all`; `run.json` and the scenario files are always written |
| `--dry-run` | off | validate the config and run a scripted conversation with zero LLM calls (below) |
| `--concurrency` | 1 | scenarios run at the same time (at least 1); Pipecat configs always run one at a time |
| `--fail-on` | `[defaults] fail_on`, else `high` | which conditions make the run exit 1 (below) |
| `--embed-audio` / `--no-embed-audio` | `--embed-audio` | embed each turn's WAV (caller and agent) and the full call recording in `report.html` as data URIs (self-contained, shareable), or link the files in `audio/` by relative path (the report then plays audio only next to its `audio/` folder; only `audio/<name>.wav` paths are linked, as only `.wav` files in `audio/` are embedded) |

Messages you may see:

- **Privacy notice.** On a folder's first real (not dry) run, Impulse prints this notice:
  "Impulse sends your system prompt and transcripts to the LLM and voice providers you configured."
  It records that it did in `.impulse/.privacy_notice_shown`.
- **Large reports.** When the embedded audio of a report exceeds 50 MB, Impulse prints this warning:
  "Embedded audio is {n} MB; the HTML report will be large. Use --no-embed-audio to link the files instead."
  The count is the base64 text in the page (a third more than the WAV files), over the caller, agent, and call recording players. A failed check only links to its turn's player ("Hear turn N") and embeds no second copy. So the size depends on the call audio, not on the number of failed checks.

Output: a summary line, one line per scenario (failed assertions and poor metrics named), the critical failure count, the run warnings, and the files written. A verified dry run of `examples/config.yaml` (Windows, so the paths use backslashes):

```text
Run 20261019T124757Z-3a7dc3 (dry run): 6 scenarios, 6 passed, 0 failed (pass rate 100.0%)
  PASS  sales_lead_followup_hindi
  PASS  recruitment_interview_booking
  PASS  clinic_appointment_booking
  PASS  loan_emi_reminder_outbound
  PASS  support_order_status_us
  PASS  cita_clinica_mx
Warnings:
  - Dry run: judge metrics and assertions were skipped; results are not verdicts.
Wrote .impulse\runs\20261019T124757Z-3a7dc3\run.json
Wrote .impulse\runs\20261019T124757Z-3a7dc3\report.html
Dry run: --fail-on is not applied.
```

The mode in the summary line is `dry run`, `text mode`, or `voice mode`. A failed scenario looks like `  FAIL  no_cancel_expected      UNIV-025 poor`.

#### --fail-on

Each level includes everything above it, and a scenario error exits 1 at every level:

| level | exit code 1 when |
|---|---|
| `critical` | any `critical` assertion fails, or any scenario has an `error` (including a critical check the judge could not evaluate) |
| `high` (default) | also: any `high` assertion fails, a `high` assertion got no judge verdict ("high assertion had no verdict"), or any scenario's UNIV-019 or UNIV-025 is `poor` (both count as severity high) |
| `any` | any scenario has `passed == False`: any failed assertion of any severity, any poor metric that is neither skipped nor informational, or any error |

- When the condition is met, the run prints `--fail-on {level}: exit 1` and one line per reason, for example `  - clinic_appointment_booking: critical assertion failed: dpdp.consent_before_pii`.
- Set the default for a folder with `fail_on` under `[defaults]` in `.impulse/config.toml`. The flag overrides it.

**Provider errors on one line.** Errors from an LLM provider (raised through LiteLLM) are shown on one line everywhere:

- in scenario errors and run warnings;
- in "evaluation failed: ..." reasons;
- in these `--fail-on` reasons.

The format:

- The line is `{error type} ({HTTP status}, provider {provider}, model {model}): {the provider's message}`.
- `; retry after {n} s` is added at the end when the provider says how long to wait. Example: `RateLimitError (HTTP 429, provider {provider}, model {model}): You exceeded your current quota...; retry after 37214 s`.
- Each part in parentheses appears only when known. A message longer than 200 characters is cut.
- Known key values are hidden before the text is cut, so no part of a key is left at the cut.
- The wait is read from the provider's `retry-after` header, a `retryDelay`, or wording such as "retry in 10h20m14s" or (OpenAI) "try again in 6m30s".

**Quota stop.** A judge call can fail with a quota error (HTTP 429) whose retry delay is 60 seconds or more. Impulse then stops calling that judge model for the rest of the run, instead of failing every remaining check the same way.

- Checks routed to that model are skipped with "evaluation failed: judge calls to {model} were stopped for this run after a quota error: {error}".
- The run gets one warning per stopped model:
  "Judge calls to {model} were stopped after a provider quota error; the checks that use it were not evaluated after that: {error}".
- Calls to the other judge model go on.
- No call reaches the stopped model after the stop, not even the "return JSON only" retry of a check that was already running.
- The stop is per model and `custom_llm_provider`. Two roles with the same model string but different providers stop separately.
- A delay under 60 seconds does not stop anything. A delay of 59.9 s is shown rounded up as `retry after 60 s` but still does not stop.
- Such a scenario has an error, so the run exits 1 at every `--fail-on` level. Wait for the quota to reset, or switch the model, and run again.

#### --dry-run

- Validates the config and forces text mode. A scripted caller replaces the LLM: the first user turn is `opening_user_message` (or "Hello"), the second is "I'd like to book", then the call ends.
- Runs scenarios only for the generic adapter. For `livekit`, `pipecat`, `http` and `custom` it validates the config without connecting and runs no scenario.
  - It adds a warning with the adapter's name, verified for the Pipecat example:
    "Dry run: scenarios run only with the generic adapter; the pipecat config was validated without connecting."
  - `http` sends no request.
  - For `custom` the dry run also imports and checks the class without creating it. A loading error exits 2. The dry run of `examples/custom_adapter/config.yaml` exits 0 (verified).
- Skips every judge metric and assertion with the reason "dry run". Deterministic metrics, UNIV-025 included, are computed.
- Adds the run warning "Dry run: judge metrics and assertions were skipped; results are not verdicts." to every dry run (any adapter). So the console summary, `run.json`, the report header, and `impulse report` show that its pass counts are not verdicts.
- Writes `run.json` (and the HTML report per `--reporter`) and exits 0 unless the config is invalid (exit 2).
- The `--fail-on` matrix is not applied, and the privacy notice is not printed.

### impulse report

```bash
impulse report RUN_ID [--embed-audio|--no-embed-audio]
```

- Rewrites `report.html` of a stored run from its `run.json`. `run.json` is not changed.
- Use it, for example, to switch between embedded and linked audio.
- `RUN_ID` is a run ID under `./.impulse/runs/`, a run folder, or a `run.json` path.
- Verified: `impulse report 20261019T124759Z-8315f8 --no-embed-audio` prints `Wrote .impulse\runs\20261019T124759Z-8315f8\report.html`.
- A run that cannot be found exits 2.

What each part of the report shows: [Reading the report](reading-the-report.md).

### impulse compare

```bash
impulse compare RUN_A RUN_B [--reporter json|html|all] [--fail-on critical|high]
```

Diffs a baseline run (`RUN_A`) and a new run (`RUN_B`).

- Each argument is an existing path to a `run.json` file or a run folder, used directly. Otherwise it is a run ID, resolved to `./.impulse/runs/<id>/run.json`.
- A run that cannot be found exits 2 (`Error: run not found: ...`, verified).
- The files `compare_<A>_vs_<B>.json` and `compare_<A>_vs_<B>.html` are written to `.impulse/runs/`, named with the two run IDs.

| flag | default | meaning |
|---|---|---|
| `--reporter` | `all` | `json`, `html`, or `all`; `.impulse/config.toml` is not consulted |
| `--fail-on` | `critical` | `critical`: exit 1 on a critical regression; `high`: also exit 1 on a high regression (below). `fail_on` in `.impulse/config.toml` is not read here, and `any` is not accepted (exit 2) |

Scenarios are matched by name. The terminal shows, in order:

- regressions (pass to fail) and improvements (fail to pass);
- newly failing checks (fail to fail);
- the unchanged counts, and added and removed scenarios;
- the critical regressions, then the high regressions.

Each changed scenario lists its assertion changes (`dpdp.consent_before_pii: pass -> fail`) and notes such as `UNIV-023 newly failing` or `VOICE-001 improved 0.30`.

**Newly failing checks (fail -> fail).** These are scenarios that failed in both runs but have a check that newly fails in `RUN_B`:

- a metric that became poor (counted, not informational in either run), with the note `{id} newly failing`;
- an assertion that is `fail` in `RUN_B` and was not in `RUN_A`;
- a scenario that newly has an error in `RUN_B`, with the note `scenario error: {error}`, so the critical regression it causes has a row.

This is worked out from the results. An ID that appears twice in one scenario counts by its first entry in each run. Without this block, such a scenario only counted as "unchanged fail". The block is always printed with its count, also when it is 0. A line carries `CRITICAL  ` when the scenario is also a critical regression. Example: `  refund_outside_policy  UNIV-025 newly failing`.

**High regressions.** A scenario present in both runs, whatever its pass/fail change, where:

- an assertion of severity `high` is `fail` in `RUN_B` and was not in `RUN_A`; or
- UNIV-025 (expected tool calls) is poor in `RUN_B` and was not in `RUN_A`.

When there are any, the line `High regressions: {names}` follows the critical line. Otherwise nothing is printed for it.

**Dry runs.** A run is a dry run when any of its metric or assertion results is skipped with the reason "dry run". For each such run the terminal prints `Note: run {run_id} is a dry run: judge metrics and assertions were skipped, so its results are not verdicts.` after the table (run A first). The HTML report shows the same sentence above its tables.

Verified output for two dry runs of the same config:

```text
Compare 20261019T124757Z-3a7dc3 (2026-10-19T12:47:57Z) -> 20261019T124759Z-8315f8 (2026-10-19T12:47:59Z): pass rate +0.0 points
Regressions (pass -> fail): 0
Improvements (fail -> pass): 0
Newly failing checks (fail -> fail): 0
Unchanged: 6 pass, 0 fail
No critical regressions.
Note: run 20261019T124757Z-3a7dc3 is a dry run: judge metrics and assertions were skipped, so its results are not verdicts.
Note: run 20261019T124759Z-8315f8 is a dry run: judge metrics and assertions were skipped, so its results are not verdicts.
Wrote .impulse\runs\compare_20261019T124757Z-3a7dc3_vs_20261019T124759Z-8315f8.json
Wrote .impulse\runs\compare_20261019T124757Z-3a7dc3_vs_20261019T124759Z-8315f8.html
```

**Exit codes of `impulse compare`:**

- **1 on a critical regression.** That is a scenario present in both runs where:
  - an assertion of severity `critical` is `fail` in `RUN_B` and was not `fail` in `RUN_A`; or
  - it has no error in `RUN_A` and an error in `RUN_B` (it newly errors). An example is "critical check {rule_id} could not be evaluated: judge returned no verdict". The same condition makes `impulse run --fail-on critical` exit 1.
- When such a scenario passed in `RUN_A`, its line under the regressions also shows the note `scenario error: {error}`.
- An error in both runs, or only in `RUN_A`, is not a critical regression. Added scenarios never count.
- **Any new error counts, environment errors included.** Examples: a LiveKit worker that declined the job, or a bot that was not running. Either makes `impulse compare` exit 1. On flaky test infrastructure, check the `scenario error:` note first. It may not be a regression of your agent.
- **1 on a high regression** too, with `--fail-on high`.
- **0 otherwise.** A dry-run note never changes the exit code; newly failing checks alone do not either.
- **2** when a run cannot be found or read, or on a usage error.

The HTML report has:

- a critical regression banner, and a "High regressions: {names}" banner when there are any;
- the dry-run notes;
- the regressions, improvements, and "Newly failing checks (fail -> fail)" tables;
- one inline delta bar per metric.

Using it in a pipeline: [CI integration](ci-integration.md).

### impulse convert-failure

```bash
impulse convert-failure .impulse/runs/<run_id>/scenarios/<name>.json
```

Reads a stored scenario result that failed (a failed check or a scenario error). It asks the primary model for one scenario that reproduces the failure. The model gets the failed checks with their reasons, the transcript, and the agent's system prompt.

- A critical assertion the judge could not evaluate (no verdict) is a failed check too: `{rule_id} (critical) not evaluated: {reason}`. So the regression scenario checks that rule again.
- The scenario's `language` and `accent` are always those of the stored call, whatever the model writes.
- A `seed` or `inject_events` in the model's answer is dropped, as `impulse generate` does.
- It writes `scenarios/regression/<name>_regression.yaml` (relative to the current directory) as a one-item YAML list to paste under `scenarios:`.
- Every string value, the header comment, and the name in the file name are redacted first: a known key value becomes `[REDACTED]`.
- The file is written to a temporary file next to it and then moved into place. A failed write (exit 2, `cannot write ...`) leaves an existing file as it was.
- When UNIV-025 failed and the new scenario has no `expected_tool_calls`, a note tells you to add them.
- Exit codes: a scenario that passed exits 2 (nothing to convert). An answer that is still invalid after one retry, or a failed model call, exits 1.

### impulse calibrate

```bash
impulse calibrate DIR
```

- Runs the judge on every labeled item in `DIR` (one JSON file per item).
- Prints, per item, whether the judge agrees with the human label. Then it prints the item count, the skipped count, the agreement rate, and Cohen's kappa.
- Exit code 0, or 2 when the folder or an item file is invalid or a judge call raises.
- The item format and how to read the result: [Calibration](calibration.md).

## Next steps

- [Getting started](getting-started.md): the commands in the order you use them.
- [Reading the report](reading-the-report.md): what `impulse run` and `impulse report` write.
- [CI integration](ci-integration.md): `--fail-on` and `impulse compare` in a pipeline.
- [Configuration](configuration.md): `config.yaml` and `.impulse/config.toml`.
- [Troubleshooting](troubleshooting.md): what the error messages mean.

---

<!-- docs-nav -->
[← Previous: Reading the report](reading-the-report.md) | [Docs home](index.md) | [Next: CI integration →](ci-integration.md)
