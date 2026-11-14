<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Writing scenarios

A scenario describes one simulated conversation: who calls, in which language, what they want, and which checks Impulse runs on the result.

## Quick version

- **What it is.** Scenarios live in the `scenarios:` list of `config.yaml`. Each one becomes one conversation between the simulated caller and your agent. It is evaluated on its own and reported as passed or failed.
- **The minimum.** A scenario needs a `name`, a `language`, and a `persona`. Everything else has a default:

```yaml
scenarios:
  - name: "order_status_check"
    language: en-US
    persona: "online shopper whose order is two days late and wants a delivery date"
```

- **What success looks like.** `impulse run config.yaml --dry-run` checks the config without any model call. It prints `PASS  order_status_check` and exits with code 0 (verified with this scenario in the example config). A config error exits with code 2 and names the field, for example `scenarios[1].expected_tool_calls[0].name`.
- **Make it useful.** Add an `expected_outcome`, a `metrics` list, compliance `assertions`, and `expected_tool_calls` for the tools the agent must call. The sections below show each one.

## Details

### An example

The healthcare scenario of [examples/config.yaml](../examples/config.yaml), with the tool list it relies on:

```yaml
agent:
  tools:                               # the tools the agent can call (names checked against expected/forbidden calls)
    - name: book_slot
      description: "Book a demo call, interview round, or clinic appointment. Arguments: date, patient_name, phone."
    - name: cancel_slot
      description: "Cancel an existing booking. Arguments: date, phone."

scenarios:
  # Healthcare: dpdp only (the hipaa pack is opt-in), exactly one booking, never a cancellation.
  - name: "clinic_appointment_booking"
    language: en-IN
    domain: healthcare
    direction: inbound
    max_turns: 12
    persona: "patient who wants the earliest slot on Tuesday, gives name and phone when asked"
    opening_user_message: "Hello, I need a doctor's appointment on Tuesday."
    expected_outcome: "appointment booked and confirmed back to the patient"
    assertions: [dpdp.consent_before_pii]
    metrics: [UNIV-001, UNIV-005, UNIV-011, UNIV-019, UNIV-023]
    expected_tool_calls:
      - name: book_slot
        required_arguments: [date, patient_name, phone]
        max_count: 1
    forbidden_tool_calls: [cancel_slot]
```

What this scenario does:

- The simulated caller plays a patient in Indian English who opens with the given sentence.
- The judge checks task completion, factual accuracy, escalation, harmful content, and tool call correctness against the agent's system prompt.
- The compliance check asks whether the agent obtained consent before collecting personal data. It comes from the `dpdp` pack, for India's Digital Personal Data Protection (DPDP) Act.
- The code-checked UNIV-025 requires exactly one `book_slot` call with all three arguments, and no `cancel_slot` call.

#### The same idea outside India

The last two scenarios of [examples/config.yaml](../examples/config.yaml) are a US shop in English and a Mexican clinic in Spanish. Neither lists a compliance pack:

```yaml
scenarios:
  # US ecommerce support: no compliance pack (the India packs are for scenarios whose language has
  # the region IN). system_prompt.txt has no order desk, so the agent should not invent one. The
  # metrics list is explicit and leaves out UNIV-023.
  - name: "support_order_status_us"
    language: en-US
    domain: ecommerce
    direction: inbound
    max_turns: 9
    persona: "online shopper whose order is two days late, a little annoyed, wants a clear delivery date and to know whether a refund is possible"
    opening_user_message: "Hi, my order was supposed to arrive on Monday and it still hasn't shown up. Where is it?"
    expected_outcome: "the agent says honestly that it cannot look up orders, invents no order status or delivery date, and offers to transfer the shopper to a human agent"
    assertions: []
    metrics: [UNIV-001, UNIV-003, UNIV-005, UNIV-007, UNIV-019, VOICE-027, VOICE-035]

  # Mexican clinic, in Spanish: exactly one booking with date, patient name and phone. No pack is
  # added by default for es-MX (hipaa is opt-in; list a pack explicitly to check it).
  - name: "cita_clinica_mx"
    language: es-MX
    domain: healthcare
    direction: inbound
    max_turns: 12
    persona: "paciente que quiere una cita con el médico general el jueves por la mañana; da su nombre y su teléfono cuando se los piden"
    opening_user_message: "Hola, quisiera una cita con el médico general para el jueves, por favor."
    expected_outcome: "la cita queda agendada con book_slot y se le confirma al paciente el día, la hora y su nombre"
    assertions: []
    metrics: [UNIV-001, UNIV-005, UNIV-011, UNIV-019, UNIV-023]
    expected_tool_calls:
      - name: book_slot
        required_arguments: [date, patient_name, phone]
        max_count: 1
```

- The persona and the expected outcome can be written in the scenario's language, as in `cita_clinica_mx`.
- `cita_clinica_mx` is the Spanish twin of the clinic scenario above: the same `book_slot` expectation, with no pack.
- These blocks are fragments of the `scenarios:` list (and the `agent:` block); the full file is [examples/config.yaml](../examples/config.yaml).

### Fields

| field | default | meaning |
|---|---|---|
| `name` | required | unique within the config and not blank (empty or only spaces is a config error); also the file name of the stored result (see the note below) |
| `language` | required | a language code (below) |
| `accent` | null | a free label, for example `north-indian`; scenarios are grouped into cohorts by language and accent |
| `domain` | `general` | one of the ten domains below |
| `direction` | `inbound` | `inbound` (the caller speaks first) or `outbound` (the agent speaks first) |
| `max_turns` | 9 | the maximum number of USER turns (below); at least 1 (0 or a negative value is a config error) |
| `persona` | required | who the caller is and what they want, in plain words |
| `opening_user_message` | null | the caller's first words, sent verbatim |
| `expected_outcome` | null | what a successful conversation achieves |
| `key_terms` | `[]` | domain terms the caller should say, for VOICE-013 |
| `code_switching` | false | true when the caller mixes languages; VOICE-031 runs only then |
| `assertions` | `[]` | compliance checks as explicit `pack.rule_id` strings |
| `metrics` | `[]` | metric IDs; an empty list means every universal judge metric (UNIV-001 through UNIV-023) |
| `expected_tool_calls` | `[]` | tool calls the scenario requires, checked by code (UNIV-025) |
| `forbidden_tool_calls` | `[]` | tool names that must never be called (UNIV-025) |
| `seed` | null | a reproducibility hint for the simulated caller |
| `inject_events` | `[]` | synthetic events, modeled now; execution is reserved for v2 |
| `noise` | null | EXPERIMENTAL, voice mode only: background noise for this scenario, `{profile, snr_db, file, label}`. null uses `voice_config.noise`; `false` turns noise off ([Voice testing: noise simulation](voice-testing.md#noise-simulation-experimental)) |

How `name` becomes a file name:

- After Unicode NFC normalisation, letters and digits of any script are kept, as are `-` and `_`. Devanagari is included, with its vowel signs and virama. Every other character becomes `_`. So `क्लिनिक बुकिंग` is stored as `क्लिनिक_बुकिंग.json`.
- A file name longer than 200 UTF-8 bytes (about 66 Devanagari characters) is cut there, never between a letter and its vowel signs. Only the file name is cut, never the stored scenario name.
- A Windows device name such as `CON` or `COM¹` gets a `_` prefix.
- Names that end up the same (ignoring case) get `_2`, `_3`, ...

### Language and domain

`language` must match `^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$`, for example `en-US`, `es-MX`, `en-IN`, `hi-IN`, or `hi-Latn-IN`. Write Hinglish as `hi-Latn-IN`, or as another code with `code_switching: true`. Every language and accent is reported as its own cohort, never blended with the others.

`domain` is one of `fsi`, `healthcare`, `recruitment`, `ecommerce`, `edtech`, `telecom`, `realestate`, `travel`, `legal`, `general`.

- Any other value is not an error. The scenario is treated as `general`, and the run gets the warning "Unknown domain {domain}; using general defaults."
- In a hand-written `config.yaml` the domain is a label: it adds no metrics or assertions by itself.
- The domain defaults (which packs and metrics each domain gets) are applied by `impulse generate` when it drafts scenarios; see [Metrics](metrics.md#domain-defaults).

The India packs (`dpdp`, `rbi_kyc`, `trai_dlt`) are added by those defaults only when the language has the region `IN`.

- `en-IN`, `hi-IN`, `hi-Latn-IN` and `ta-in` qualify; `hi`, `en-US`, `es-MX` and `en-GB` do not.
- So a generated `fsi` scenario in `en-US` gets `assertions: []`.
- You can still list any pack explicitly for any language. The en-US and es-MX scenarios above show scenarios outside India, with no pack.

### Direction and the first words

- **Inbound** (`direction: inbound`): the simulated caller speaks first. With `opening_user_message` set, that sentence is the first user turn, verbatim, with no LLM call.
- **Outbound** (`direction: outbound`): Impulse waits for the agent's opener first; the opener does not count toward `max_turns`. With `opening_user_message` set, that sentence is the caller's reply to the opener, verbatim.

Without `opening_user_message`, the simulator model writes the caller's turns. It uses the `persona`, the scenario language, the config-wide `simulator.persona_style` (`realistic`, `adversarial`, or `polite`), and a short summary of the agent's system prompt.

### max_turns

`max_turns` counts USER turns only, never agent turns. The conversation ends at the first of:

- `max_turns` user turns;
- the simulated caller ending the call (it answers `<END_CALL>` when the conversation has naturally ended);
- the agent hanging up.

An agent may hang up before its first turn. Then nothing is judged, and the scenario fails with the error "the agent closed the conversation before its first turn: ...".

Guidance:

- The default of 9 suits most service conversations.
- Use `max_turns: 12` for the compliance domains, `fsi` and `healthcare`. A regulated flow (greeting, identity verification, purpose disclosure, consent, the action, wrap-up) needs at least 7 user turns, and edge cases add more. `impulse generate` uses 12 for these domains and 9 for the others.
- A short smoke test can use fewer, for example `max_turns: 6` in [examples/pipecat/config.yaml](../examples/pipecat/config.yaml).
- UNIV-003 (context retention) skips transcripts with fewer than 4 turns, both speakers counted. So it needs at least 2 user turns.

### expected_outcome, key_terms, and code_switching

- **`expected_outcome`** is added to every judge criterion of the scenario as the line `Expected outcome: ...`. UNIV-001 (task completion) and VOICE-035 (turns to close) use it most.
- **`key_terms`** feeds VOICE-013 (domain-term error rate). It needs voice mode and the agent's own transcription of the caller (`heard_text`). Terms the caller never said are not counted.
- **`code_switching: true`** turns on VOICE-031 (code-switch handling); otherwise VOICE-031 skips.

### Choosing metrics and assertions

List metric IDs in `metrics`; an unknown ID is a config error.

- An empty list runs every universal judge metric. So a scenario that should not run UNIV-023 needs an explicit list, as the ecommerce scenarios in `examples/config.yaml` have.
- Two metrics run without being listed: UNIV-025 whenever the scenario has `expected_tool_calls` or `forbidden_tool_calls`, and VOICE-011 in every voice-mode scenario. VOICE-011 checks the harness: Impulse's own text-to-speech then speech-to-text loop, used to check itself.
- `metrics.enabled: false` in the config turns every metric off, these two included.

The metric definitions: [Metrics](metrics.md).

List compliance checks in `assertions` as explicit `pack.rule_id` strings such as `dpdp.consent_before_pii`. Wildcards such as `dpdp.*` are rejected in `config.yaml`, and an unknown rule is a config error. The `hipaa` pack is opt-in: it runs only when you list its rules. See [Compliance packs](compliance-packs.md).

### Tool calls: agent.tools, expected_tool_calls, forbidden_tool_calls

Tool calls are where a voice agent does real damage. Think of a booking with the wrong date, a payment made twice, or "you're booked" with no booking call. Impulse checks them twice:

- **UNIV-023 (tool call correctness, judged)** asks the judge five numbered checks:
  1. Was the right tool called?
  2. Were the arguments correct and complete?
  3. Does a failed or errored call suggest the action itself failed?
  4. Did the agent claim success when the call failed?
  5. Did the agent claim or need an action without a matching tool call?
- **UNIV-025 (expected tool calls, checked by code)** enforces exactly what the scenario lists, with no judge call.

**`agent.tools`** (in the `agent` block, not per scenario) tells Impulse which tools the agent has: a list of names, or `{name, description}` items.

- Leave it out (`null`) when you do not know; the judge then reads the tools from the system prompt.
- `agent.tools: []` declares that the agent has no tools. UNIV-023 then skips without a judge call when no tool was called.
- When `agent.tools` is set, its lines are added to the UNIV-023 criterion. Every name in `expected_tool_calls` and `forbidden_tool_calls` must be in it.

**`expected_tool_calls`** entries:

| key | default | meaning |
|---|---|---|
| `name` | required | the tool name, exactly as the agent reports it |
| `required_arguments` | `[]` | keys that must be present and not null, `""`, `[]`, or `{}` |
| `argument_values` | `{}` | keys whose values must match exactly (JSON equality) |
| `must_succeed` | true | the call's `success` must not be `false` (`null`, not captured, is allowed and noted) |
| `min_count` | 1 | at least this many calls; 0 is allowed only together with `max_count` |
| `max_count` | null | at most this many calls; `max_count: 1` catches duplicate bookings or payments |

**`forbidden_tool_calls`** lists tool names that must never be called in this scenario.

UNIV-025 is `good` when every entry passes and `poor` otherwise. A poor UNIV-025 fails the scenario. Under `impulse run --fail-on high` (the default) it counts as a high-severity failure. Its reasons are exact, for example:

- `expected tool book_slot was never called`
- `tool book_slot is missing required arguments: phone`
- `tool book_slot argument date: expected "2026-01-02", got "2026-01-01"`
- `tool book_slot called 2 times, expected at most 1`
- `tool book_slot call failed: slot no longer available`
- `forbidden tool cancel_slot was called at turn 4`

The full rules: [Metrics](metrics.md#univ-025-rules).

**Put `max_count` on escalation tools.** A handoff tool such as `escalate_to_human` should run once per call. Nothing flags a repeat unless the scenario says so. In a real-world campaign an agent called `escalate_to_human` 4 to 6 times in one conversation, and no check caught it. In a scenario that should escalate, expect it once; in one that may escalate, cap it:

```yaml
    expected_tool_calls:
      - name: escalate_to_human
        max_count: 1               # exactly one handoff (min_count defaults to 1)
      # or, when escalating is allowed but not required:
      # - name: escalate_to_human
      #   min_count: 0
      #   max_count: 1
```

When your config sets `agent.tools`, list `escalate_to_human` there too, or the config is rejected. The same applies to any tool whose repeat costs something: payments, refunds, tickets, bookings.

**`success` without changing your tools.** UNIV-025's `must_succeed` and UNIV-023 read each call's `success`. When your adapter does not report it, Impulse infers it from the tool's result ([Metrics](metrics.md#univ-023-notes)):

- `{"ok": true}` or `{"success": true}` is a success;
- `{"error": "..."}`, `{"ok": false}` or `{"status": "failed"}` is a failure;
- anything else stays "not captured".

The judge also sees the result itself, so details the agent repeats from it are not judged as made up.

Config errors (exit code 2, naming the field):

- a name not in `agent.tools`, when `agent.tools` is set;
- `max_count` lower than `min_count`;
- `min_count: 0` without `max_count`;
- a negative `min_count`;
- a tool name that is both expected and forbidden.

Your agent has to report its tool calls for any of this to work. The generic adapter reads them from the `tool_calls` key of your handler's reply. The LiveKit and Pipecat adapters capture them from the running agent ([Adapters](adapters.md)). `examples/sample_agent.py` returns a dummy `book_slot` call whenever the caller says "book". So you can try both metrics without a backend ([Run Impulse for free](run-for-free.md#9-test-tool-calls-with-the-mock-in-sample_agentpy)).

### seed

```yaml
scenarios:
  - name: "clinic_booking_seeded"
    language: en-IN
    persona: "patient who wants the earliest slot on Tuesday"
    seed: 42
```

When `seed` is set, the simulated caller's system prompt gets the line "Your conversation seed is 42. Produce the same user turns for the same seed and persona." When it is not set, the line is left out entirely.

This is a best-effort reproducibility hint: language models and providers do not guarantee the same turns for the same seed. Use `opening_user_message` when the first words must be exact.

### inject_events (v2)

`inject_events` describes synthetic events to inject before a given user turn. The five kinds: a tool timing out, a tool returning an error, the caller going silent, a transfer failing, or the caller changing their mind. In this release (v1) the events are modeled but never executed:

```yaml
scenarios:
  - name: "booking_with_injection"
    language: en-IN
    persona: "patient who wants to book"
    max_turns: 4
    inject_events:
      - at_turn: 2                     # before the 2nd USER turn (1..max_turns)
        type: tool_timeout             # tool_timeout | tool_error | silence | transfer_fail | user_change_of_mind
        tool: book_slot                # for tool_timeout and tool_error
        note: "booking backend times out"
```

Event fields:

| field | meaning |
|---|---|
| `at_turn` | required; the user turn the event comes before, from 1 to `max_turns` (otherwise a config error) |
| `type` | required; one of the five above |
| `tool` | the tool, for `tool_timeout` and `tool_error` |
| `error_message` | for `tool_error` |
| `silence_ms` | for `silence` |
| `note` | free text shown in the report |

What happens in v1: the scenario runs without injection. The console prints, once per affected scenario (verified with the scenario above in a dry run):

```text
booking_with_injection: inject_events defined but execution reserved for v2; this scenario ran without injection.
```

- The HTML report shows the same note, "inject_events defined but execution reserved for v2; this scenario ran without injection.", for each such scenario.
- The stored result keeps the events (`ScenarioResult.inject_events`), so v2 can replay them.
- Do not rely on `inject_events` to test failure handling yet. To test a failed tool today, make the tool fail in your agent or mock. See the tool-call mock in [Run Impulse for free](run-for-free.md).

### Generating scenarios

`impulse generate` drafts scenarios from your agent's system prompt with the primary model:

```bash
impulse generate --prompt system_prompt.txt --domain healthcare --language en-IN
```

It writes `scenarios/generated.yaml` (or `--out`), with a mix of happy paths, edge cases, adversarial callers, and hallucination traps. It sets some fields itself:

- `language` and `domain` from the flags;
- `assertions` from the domain defaults (wildcards expanded to explicit rule IDs, never `hipaa`);
- the domain's default metrics added to the model's choice, in an explicit non-empty `metrics` list;
- `inject_events: []` and no `seed`;
- `max_turns` 12 for `fsi` and `healthcare` (9 otherwise) when the model gives none.

`--voice-mode` adds VOICE-001, VOICE-003, and VOICE-005 to every scenario. Review the file, then paste the scenarios under `scenarios:`. Flags: [CLI reference](cli.md#impulse-generate).

### Regression scenarios from failed runs

When a scenario fails, `impulse convert-failure` turns its stored result into a scenario that reproduces the failure:

```bash
impulse convert-failure .impulse/runs/<run_id>/scenarios/<name>.json
```

It writes `scenarios/regression/<name>_regression.yaml`, a one-item list to paste under `scenarios:`. When UNIV-025 failed, add `expected_tool_calls` to the new scenario so the check runs again. Impulse prints a note when they are missing.

## Next steps

- [Metrics](metrics.md): what each metric ID checks.
- [Compliance packs](compliance-packs.md): the `pack.rule_id` checks.
- [Cookbook](cookbook.md): the example scenarios adapted to your agent.
- [Configuration](configuration.md): the rest of `config.yaml`.

---

<!-- docs-nav -->
[← Previous: Configuration](configuration.md) | [Docs home](index.md) | [Next: Adapters →](adapters.md)
