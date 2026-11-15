<!-- SPDX-License-Identifier: Apache-2.0 -->
# Claude skill

This page shows how to give Claude the Impulse Evals skill, so you can ask Claude to set up, run and read Impulse without reading every page first.

## Quick version

- **What it is.** A Claude skill in `skills/impulse-evals/`. Its `SKILL.md` holds the quickstart, every `config.yaml` key, the five adapters, the voice providers, noise simulation, every metric ID and compliance rule, the command line with its exit codes, and the most common errors. `references/documentation.md` holds these docs as one file, which Claude opens only when a question needs a detail.
- **The one command**, from the root of a clone, to package the skill for upload:

  ```bash
  cd skills && zip -r ../impulse-evals-skill.zip impulse-evals
  ```

- **What success looks like.** `impulse-evals-skill.zip` in the root of the clone, with the folder `impulse-evals/` inside it and `SKILL.md` directly in that folder. Upload it to Claude, start a new conversation and ask "what does UNIV-025 check?". Claude answers with the code-checked `expected_tool_calls` and `forbidden_tool_calls` (not verified).

## Details

Terms used below: TTS is text-to-speech, STT is speech-to-text, and SNR is signal-to-noise ratio.

### What is inside

| file | what it holds |
|---|---|
| `skills/impulse-evals/SKILL.md` | about 260 lines: when Claude should use the skill, how to help (doctor, then dry run, then run), the quickstart, models and keys, every `config.yaml` key and scenario field, the adapters with a custom adapter skeleton, voice providers, noise simulation (experimental), every metric ID, every compliance rule with its severity, the commands, `--fail-on` and exit codes, the most common errors, and the known limitations |
| `skills/impulse-evals/references/documentation.md` | every page of `docs/` in one file, in the order of [Documentation home](index.md). Claude searches it by heading, for example "UNIV-025 rules" or "The exact JSON" |
| `scripts/build_skill_reference.py` | rebuilds `references/documentation.md` from `docs/` |

The folder name `impulse-evals` must match the `name:` line of `SKILL.md`. An upload fails when they differ.

### Add it to Claude

**Claude apps (web, desktop).** Skills need code execution:

1. Turn on code execution: Settings > Capabilities (Free, Pro and Max plans). On Team and Enterprise plans, an owner enables it with skills in the organization settings.
2. Package the skill with the command in the quick version.
3. Open Customize > Skills, click "+", then "Create skill", then "Upload a skill", and choose `impulse-evals-skill.zip`.
4. Check that the skill's toggle is on.

These steps follow Claude's help center in November 2026; menu names can change.

**Claude Code.** Copy the folder, no zip needed:

```bash
# for every project on this machine
cp -r skills/impulse-evals ~/.claude/skills/

# or for this repository only
mkdir -p .claude/skills && cp -r skills/impulse-evals .claude/skills/
```

Start a new Claude Code session afterwards. Skills enabled in the Claude apps also load in Claude Code when you sign in with the same account.

**Any other assistant.** Paste the contents of `SKILL.md` at the start of the conversation. It reads as plain Markdown.

Success: the skill is listed and switched on. A question such as "set up Impulse for my Pipecat bot" gets the eval-mode config, the `-t eval` start command and the `impulse doctor` check, without you pasting any docs (not verified).

### When Claude uses it

Claude reads the skill's `description` and opens the skill when a conversation mentions any of:

- Impulse, `impulse-evals` or PULSE;
- `impulse run`, `impulse doctor` or another `impulse` command;
- a `config.yaml` with `agent`, `judge`, `simulator` or `scenarios` blocks;
- a metric ID such as UNIV-023 or VOICE-009, or a rule such as `dpdp.consent_before_pii` or `rbi_kyc.no_credential_request`;
- testing or regression-gating a voice agent with a simulated caller and an LLM judge.

If Claude does not pick it up, name it: "Use the impulse-evals skill to ...".

### Good first questions

- "Write a `config.yaml` that tests my clinic booking agent in English and Hindi, and catches double bookings."
- "My LiveKit agent never joins the room. Here is the `impulse doctor` output."
- "Which metrics and compliance rules should I use for an outbound loan reminder bot in India?"
- "Add a clean and a noisy run of my scenarios, and compare them."
- "Set up a GitHub Actions job that blocks the merge when a high-severity check newly fails."
- "Explain this failed scenario from my report." (paste the scenario's rows)

### What the skill tells Claude to do

- Suggest the safe order: `impulse doctor CONFIG`, fix every `FAIL` row, then `impulse run CONFIG --dry-run`, then `impulse run CONFIG`.
- Write complete YAML with only real keys, metric IDs, rule IDs and flags; `config.yaml` rejects anything else.
- Say what is not verified, for example the free path with Ollama models, LiveKit Cloud, and noise simulation against a live agent.
- Say that a compliance result is a technical signal only, not legal certification.

### Keep the skill in sync with the docs

The skill is a summary of these pages, so it goes stale when they change.

1. **When a page in `docs/` changes**, rebuild the reference file:

   ```bash
   python scripts/build_skill_reference.py
   ```

   Success: it prints `Wrote skills/impulse-evals/references/documentation.md` and the number of pages, and exits 0.
2. **When a default, a key, a flag, a metric, a rule or an error message changes**, update the matching line of `SKILL.md` too. For example, the default of `impulse run --fail-on` (`high`), the `--fail-on` levels, and the noise keys are written out there.
3. **Package and upload again.** Claude keeps the uploaded version until you upload a new one.

### Licence

The skill folder and the script are under the Apache License 2.0, like the rest of Impulse ([Contributing](contributing.md#licensing-of-contributions)). `SKILL.md` starts with the YAML block Claude needs, so its SPDX header comes right after that block.

## Next steps

- [Getting started](getting-started.md): run Impulse yourself in a few minutes.
- [Writing scenarios](writing-scenarios.md): what to ask Claude to write for you.
- [CI integration](ci-integration.md): gate merges on the results.
- [Contributing](contributing.md): send a fix to the skill or the docs.

---

<!-- docs-nav -->
[← Previous: Roadmap](../ROADMAP.md) | [Docs home](index.md)
