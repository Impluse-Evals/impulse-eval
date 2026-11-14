<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Reading the report

This page walks through each part of the HTML report that `impulse run` writes, and how to use it to find and hear what went wrong.

## Quick version

- **What it is.** Every run writes one self-contained page, `.impulse/runs/<run_id>/report.html`. It loads nothing from the internet and runs no script, so you can email it or attach it to a ticket.
- **The one command.** Make a run (a dry run needs no model and no key), then open the report in a browser:

  ```bash
  impulse run examples/config.yaml --dry-run
  ```

- **What success looks like.** The console prints `Wrote .impulse/runs/<run_id>/report.html` (with `\` on Windows; verified), and the exit code is 0. The scenarios table lists six scenarios as PASS. The page's header shows the warning "Dry run: judge metrics and assertions were skipped; results are not verdicts."
- **How to read a real run.** Start at the top: warnings, then critical failures, then the scenarios table. Click a failed scenario, read its assertions and metrics, and click "Hear turn N" to listen to the turn that failed.

## Details

Both screenshots on this page come from a real voice run of the second test campaign (2026-10-25): a Pipecat banking bot with seeded bugs.

Terms used below: TTS is text-to-speech, STT is speech-to-text, WER is word error rate, and TTFA is time to first audio. RTVI is Pipecat's client/server message protocol.

### The summary

![The top of an Impulse report: run header, run warnings, critical failures, scenarios, cohorts, and the start of the metric summary.](images/report-summary.png)

The page opens with the run as a whole, in this order:

| Part | What it shows | How to use it |
|---|---|---|
| Header | Run ID, timestamp, agent domain and type, entrypoint, the scenario languages, scenario counts with the pass rate, judge model, and mode (text or voice). Voice runs also show the TTS and STT providers. | Check that you are reading the run you think you are. |
| Warnings | Run warnings, such as the dry-run warning or a Pipecat bot's RTVI error. | Read them first: they say when results are not verdicts, or when something needs fixing. |
| Critical failures | One row per failed critical assertion: scenario, `rule_id`, `regulation_ref`, the judge's reason, the turn, and the confidence. A "Scenario errors" table follows when a scenario could not finish. | These are what `--fail-on critical` exits 1 on; the default `high` level also exits 1 on high failures and on a poor UNIV-019 or UNIV-025. The scenario and turn are links. |
| Scenarios | One row per scenario: language, accent, status (PASS, FAIL or ERROR), failed assertions out of all, poor metrics out of all, and the error. | Click a name to jump to that scenario. A scenario with `inject_events` carries the badge "inject_events noted (v2)"; one that ran with [noise](#noise-simulation) carries "noise: {label}, {SNR} dB". |
| Cohorts | One row per language and accent, with its pass rate, gap to the best cohort, mean STT confidence and metric means. Cohorts are never blended. | A "gap warning" means a pass rate more than 15 percentage points below the best cohort. An "STT warning" means a mean STT confidence more than 0.10 below the best cohort. |
| Metric summary | Each metric's mean score over the results that were neither skipped nor informational, with its label. | A "proxy" label marks a text-mode stand-in for an audio measurement. A metric skipped everywhere shows "skipped" and its first reason. |
| Latency profile | Agent latency percentiles: p50, p75, p95 and p99, in milliseconds. | Voice runs measure from the end of the caller's speech to the agent's first audio. Text runs measure from sending the turn to the agent's first output. Impulse's own TTS and STT time is never included. |

When a Pipecat bot reports RTVI `metrics`, the latency profile also has a "Per-service latency" part. Its first table gives each service's samples, mean and maximum time (for example the bot's STT, LLM and TTS). Its second table lists every timing by scenario and turn. These timings are informational and never scored.

### A scenario

![One scenario of a voice report: call recording, failed checks with "Hear turn 2" links, the call timeline, and a highlighted failing turn.](images/report.png)

Below the summary, "Scenario transcripts" holds one collapsible block per scenario. Failed scenarios open expanded. The block's title line shows the status, the name, the language and accent, the number of turns, and the call length.

Inside a block, from top to bottom:

1. The full call recording player (voice mode).
2. The `inject_events` note and table, when the scenario has any.
3. The error, when the scenario could not finish.
4. **Assertions**: `rule_id`, severity, result (`pass`, `fail` or `skip`, and "no verdict" when the judge gave none), reason, turn, confidence and `regulation_ref`. Below the table, compliance packs add this line:

   > Technical signal only. Not legal certification. Consult qualified counsel before relying on this result for regulatory purposes.
5. **Metrics**: `metric_id`, name, label (`good`, `warning`, `poor` or `skipped`, plus "proxy" or "informational"), score, raw value, turn and reason.
6. **Transcript**: the call timeline (voice mode), then every turn.

Each turn starts with a line such as `[turn 2] AGENT · t=8221 ms · latency 2781 ms · STT confidence 1.00`. It adds "interrupted" when the caller spoke over the agent. A turn that a failed assertion or a poor metric points at is highlighted, with a "failing:" badge naming those checks.

Under the turn line come the agent's tool calls, then the text. A user turn also shows "agent heard:", the agent's own transcription of the caller (`heard_text`), when the agent returns it. Each tool call is one line, the same line the judge reads:

```text
[turn 4] AGENT TOOL CALL: verify_identity({"customer_id": "CUST-2001", "dob": "1990-04-12"}) -> success=True
```

A failed call's line is marked as failed. The line can also carry ` result=` (what the tool returned, sensitive keys hidden) and ` error=`.

### Full call recording

In voice mode, with `voice_config.save_recording: true` (the default), each scenario starts with a player labelled "Full call recording (left channel: caller, right channel: agent)".

- Listen with headphones to hear who talks over whom.
- The recording follows the call as the agent experienced it. The gap before each agent turn is the agent's measured latency; Impulse's own processing time is left out.
- The file is `audio/<scenario>_recording.wav` in the run folder.

### Caller and agent players

Every turn with an audio file has its own player, caller turns and agent turns alike. Caller turns play the audio Impulse sent to the agent: its TTS audio, with the background noise mixed in when the scenario used [noise simulation](#noise-simulation). Agent turns play the audio the agent returned.

The files are `audio/<scenario>_t<N>_user.wav` and `audio/<scenario>_t<N>_agent.wav`, where N counts both speakers' turns. They are written in voice mode with `voice_config.save_audio: true` (the default). A text-mode run, including a dry run, has no players.

### Hear turn N links

In the assertions and metrics tables, the turn cell holds the turn number as a link to that turn. When the check failed and the turn has audio, a "Hear turn N" link follows.

- It counts as failed for a failed assertion, or for a poor metric that is neither skipped nor informational.
- The link jumps to that turn's player in the transcript below. Press play there.
- It is a link, not a second player, on purpose. The audio is embedded once, so many failed checks on one turn do not make the report bigger.
- With noise simulation, it plays what the agent heard: the noisy audio.

### Noise simulation

A run with [noise simulation](voice-testing.md#noise-simulation-experimental) (experimental) adds, and a run without it shows none of this:

- **Header:** a "noise simulation" row, "experimental", and the noise levels of the run, for example "clean; pink, 20 dB; pink, 10 dB". A level is the label and the signal-to-noise ratio (SNR); "clean" means a scenario without noise.
- **Badges:** "noise: {label}, {SNR} dB" in the scenario table and on the scenario's title line.
- **Two players per caller turn:** the turn's player plays the noisy audio the agent heard. Under it, "clean caller audio (before noise)" opens a second player with the same words without the noise. Compare the two when the agent misheard: was it the noise, or the agent? The files are `audio/<scenario>_t<N>_user.wav` (noisy) and `audio/<scenario>_t<N>_user_clean.wav` (clean).
- **Pass rate by noise level:** after the cohort table, when the run has two or more levels. One row per level, clean first, then from the mildest (highest SNR) to the harshest: scenarios, passed, failed, pass rate, and the mean VOICE-009 score (how well the agent heard). A pass rate that drops sharply at one level shows where your agent stops coping.
- **Warning:** "Noise simulation is experimental: listen to the noisy caller audio in the report before trusting the results."

### Call timeline

Voice-mode scenarios get a static timeline above the transcript. It is drawn from the turns' start and end times, on the same clock as the full call recording.

- **Caller row:** one block per caller turn.
- **Agent row:** one block per agent turn. A block is red ("slow reply") when that turn's latency is above `metrics.voice.ttfa_fail_ms` (default 3000 ms).
- **Dead air:** a shaded block from the end of a caller turn to the start of the agent's reply. It appears when that reply's latency is above `metrics.voice.dead_air_ms` (default 1500 ms).
- The thresholds are the ones from your `config.yaml`, stored with the run. The legend under the timeline prints them.
- Hover a block to see its turn number and times in milliseconds.

Use it to spot a slow agent at a glance, then open the red turn's player.

### Tool call trace

The last section lists every tool call of the run in one table:

| Column | Content |
|---|---|
| scenario_name, turn | Links to the scenario and the turn. |
| tool, arguments | The tool name and its arguments, as the agent sent them. |
| success | `True`, `False`, or `None (not captured)` when the adapter could not tell. |
| error | The error text, cut at 300 characters. A JSON error is shown with sensitive keys hidden. |
| result | What the tool returned. |

In the result and in JSON errors, the values of sensitive keys are shown as `[REDACTED]`. Sensitive keys name a password, PIN, OTP, CVV, secret, token, API key, card number and similar. The report is meant to be shared. Tool arguments are not redacted.

### Embedded or linked audio

By default (`--embed-audio`), every WAV file is embedded into the page, so the report plays anywhere. Each file is embedded once.

| You want | Command |
|---|---|
| A report that plays anywhere (default) | `impulse run config.yaml` |
| A small report next to its audio files | `impulse run config.yaml --no-embed-audio` |
| To rewrite a stored run's report with links | `impulse report RUN_ID --no-embed-audio` |

- When the embedded audio exceeds 50 MB (counted as the base64 text in the page, a third more than the files), Impulse prints: "Embedded audio is {n} MB; the HTML report will be large. Use --no-embed-audio to link the files instead."
- WAV is large. In the second test campaign, a 6-scenario voice report embedded 66.6 MiB of audio, and the page was 88.9 MiB.
- With `--no-embed-audio`, the players link `audio/<file>.wav`. They play only while `report.html` sits next to the run's `audio/` folder. Share the whole run folder, or a zip of it.
- Only `.wav` files inside the run's `audio/` folder are embedded or linked. Any other path shows a note instead of a player.
- `impulse report` rewrites only `report.html`; `run.json` is not changed.
- Compressed audio in reports is planned for v1.1 ([Known limitations](known-limitations.md)).

## Next steps

- [Metrics](metrics.md): what each metric ID and label means.
- [Voice testing](voice-testing.md): turn on voice mode and the audio files.
- [CI integration](ci-integration.md): gate merges with `--fail-on` and `impulse compare`.
- [Known limitations](known-limitations.md): what the report cannot show yet.

---

<!-- docs-nav -->
[← Previous: Compliance packs](compliance-packs.md) | [Docs home](index.md) | [Next: CLI reference →](cli.md)
