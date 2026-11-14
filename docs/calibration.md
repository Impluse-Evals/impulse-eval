<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Calibration

Calibration measures how often the judge agrees with your own pass or fail labels, so you know how far to trust it.

## Quick version

- **What it is.** You label some transcripts yourself (`pass` or `fail` for one check). Impulse asks the judge the same question and counts how often it agrees with you.
- **The command,** on a folder with one JSON file per labelled transcript:

```bash
impulse calibrate labeled/
```

- **What success looks like.** One line per item, then the totals: the agreement rate and Cohen's kappa. The exit code is 0 (not verified with a real judge). Without a judge model configured, it exits with code 2 and says which role is missing (verified).
- **When it matters most.** Before you trust critical compliance verdicts from a small or local judge model.

## Details

### When to calibrate

- Before trusting critical compliance verdicts from a small judge model. The primary model string may name 14B parameters or fewer (for example a tag ending in `:7b`) while a run includes a critical assertion. Then Impulse warns: "Judge model {model} looks small (<= 14B parameters). Critical verdicts from small judges are unreliable; run impulse calibrate before trusting them."
- After you change the judge models, or the agent's domain changes.
- Around 30 hand-labeled transcripts per check you care about is a reasonable start. More items give a steadier number.

### 1. Collect transcripts

Run your scenarios with the judge models you want to calibrate. Every scenario result is stored at `.impulse/runs/<run_id>/scenarios/<name>.json`. Its `transcript` value is a complete transcript you can reuse. Real conversations work too, as long as you write them in the same shape.

Success: you have a set of transcripts that covers both good and bad behaviour for the check you calibrate.

### 2. Write one item file per transcript

Create a folder (for example `labeled/`) with one JSON file per item. Every `*.json` file directly in the folder is read, in name order:

```json
{
  "transcript": {
    "scenario_name": "clinic_booking_claimed_without_tool",
    "language": "en-IN",
    "duration_ms": 0,
    "metadata": {},
    "agent_system_prompt": "You are the booking assistant of a demo clinic. Call book_slot before you confirm a booking.",
    "turns": [
      {"speaker": "user", "text": "Please book me for Tuesday morning.", "timestamp_ms": 0},
      {"speaker": "agent", "text": "Done, you are booked for Tuesday at 10.", "timestamp_ms": 1200}
    ]
  },
  "check": "UNIV-023",
  "human_label": "fail"
}
```

The fields:

- `transcript`: a transcript in the stored format. It has `scenario_name`, `language`, `duration_ms`, `metadata`, and `turns`. Each turn has `speaker`, `text`, `timestamp_ms`, and optionally `tool_calls`, `heard_text`, and the other turn fields. `agent_system_prompt` is the judge's rubric.
- `check`: the check to calibrate. It is a judge-based metric ID (`UNIV-...`, `VOICE-...`, `CHAT-...`) or a compliance rule (`pack.rule_id`). Deterministic metrics such as UNIV-025 need no judge, so they make the item invalid.
- `human_label`: `pass` or `fail`, your own verdict.
- Optional `scenario`: a mapping of scenario fields, so the check sees the same context it had in the run. Examples: `expected_outcome`, `direction`, `code_switching`, `max_turns`, `expected_tool_calls`.
  - Without it, the scenario is built from the transcript with default values, including `direction: inbound`.
  - An outbound call needs `"scenario": {"direction": "outbound"}`.

Success: the example above loads as a valid item (verified with Impulse's item loader).

### 3. Label honestly

Decide each `human_label` from the transcript and the agent's system prompt alone. That is the same information the judge gets. Decide before you look at what the judge said in the run. Include clear passes, clear failures, and borderline cases.

Success: every item has a label you would defend.

### 4. Run impulse calibrate

```bash
impulse calibrate labeled/
```

The command reads no `config.yaml`.

- The judge models come from `IMPULSE_MODEL_PRIMARY` / `IMPULSE_MODEL_FAST` or `[models]` in `.impulse/config.toml`.
- Only the roles your checks route to are needed. Metrics go by their tier; critical and high rules go to the primary model, medium rules to the fast one.
- Each item is judged once, the normal way.

Success (not verified with a real judge): one line per item, then the totals, and exit code 0. The output format:

```text
Calibration of labeled: 30 items, 28 judged, 2 skipped
  AGREE  item_001.json  UNIV-023: human fail, judge fail
  DIFF   item_002.json  UNIV-023: human pass, judge fail (<the judge's reason>)
  SKIP   item_003.json  UNIV-023: human pass, judge returned skip (<the judge's reason>)
  ...
Items: 30
Skipped (judge returned skip; left out of agreement and kappa): 2
Agreement rate: 89.3% (25 of 28 judged items)
Cohen's kappa: 0.786
```

The numbers are illustrative; the line formats are the code's.

Exit code 2 means one of these:

- the folder or an item file is invalid (every problem is listed);
- no model is configured for a needed role;
- a judge call raises.

With no model configured, the example item above gives (verified): `Error: no model configured for the primary role, which impulse calibrate uses. Set the environment variable IMPULSE_MODEL_PRIMARY, or [models] primary in .impulse/config.toml.`

### 5. Read the result

- **The judge's label** for a compliance rule is its result (`pass` or `fail`). For a metric it is `fail` when the label is `poor`, and `pass` otherwise. A `warning` never fails a scenario, so it counts as a pass.
- **Skipped items** (the judge, or a code-decided skip, returned `skip`) are left out of agreement and kappa and counted separately. Many skips usually mean the items do not contain the situation the check is about.
- **Agreement rate** is the share of judged items where the judge's label equals yours.
- **Cohen's kappa** corrects agreement for chance. 1.0 is perfect agreement; 0.0 is what chance alone would give.
  - When both observed and expected agreement are 1.0 (every label the same and all agree), kappa is 1.0.
  - When only expected agreement is 1.0, it is 0.0.
  - Kappa is computed in plain Python, without a statistics library.

Success: you know, per check, whether the judge agrees with you well enough to gate on it. If it does not:

- try a stronger primary model;
- make the system prompt's rules more explicit;
- or keep that check out of your `--fail-on` gate.

## Next steps

- [Metrics](metrics.md): the criteria the judge applies.
- [Compliance packs](compliance-packs.md): the critical rules worth calibrating first.
- [CLI reference](cli.md#impulse-calibrate): the command's exit codes.
- [Run Impulse for free](run-for-free.md): local judge models, where calibration matters most.

---

<!-- docs-nav -->
[← Previous: CI integration](ci-integration.md) | [Docs home](index.md) | [Next: Cookbook →](cookbook.md)
