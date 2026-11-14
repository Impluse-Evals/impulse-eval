---
name: impulse-evals
description: Set up, run, and interpret Impulse Evals (formerly PULSE), the open-source testing framework for AI voice and chat agents (`pip install impulse-evals`, command `impulse`). Use this skill whenever the user mentions Impulse, impulse-evals, PULSE evals, `impulse run`, `impulse doctor`, a `config.yaml` with `agent`/`judge`/`simulator`/`scenarios` blocks, UNIV-/VOICE-/CHAT- metric IDs, compliance rules like `dpdp.consent_before_pii` or `rbi_kyc.*`, or wants to test, evaluate, regression-gate, or load-test a voice agent (LiveKit, Pipecat, Vapi, Retell, an HTTP agent, or a Python function) with a simulated caller and an LLM judge, even if they don't name Impulse explicitly but are clearly working in an Impulse project.
---
<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Impulse Evals

Impulse Evals 0.1.0 tests AI voice and chat agents. A **simulated caller** (an LLM) talks to the user's agent through an **adapter**, in text or voice. An **LLM judge** then grades each conversation against the agent's **own system prompt** plus a fixed safety floor. There is no rubric to write: the system prompt is the rubric.

- Package `impulse-evals`, command `impulse` (or `python -m impulse_evals`), import package `impulse_evals`, Python 3.11 or newer (3.10 is not supported).
- Models are LiteLLM model strings the user brings. With Ollama and local speech models no paid key is needed.
- Licences: core AGPL-3.0-only; plugin API files Apache-2.0; compliance packs Elastic License 2.0. Testing your own agent, in your company or CI, does not oblige you to publish it.

The complete documentation is in `references/documentation.md` (about 6,000 lines). Read only the section you need; the map at the end of this file tells you which heading to search for.

## How to help

1. **Find where the user is**: installing, configuring, writing scenarios, connecting an adapter, running, reading a report, or wiring CI. Ask for their `config.yaml` and the exact command output when debugging; Impulse error messages name the exact field (for example `scenarios[1].expected_tool_calls[0].name`), so the message usually is the diagnosis.
2. **Always suggest the safe order**: `impulse doctor CONFIG` (fix every `FAIL` row) → `impulse run CONFIG --dry-run` (zero LLM calls) → `impulse run CONFIG`.
3. **Write complete, valid YAML.** Never invent keys, metric IDs, rule IDs, or flags: everything valid is listed below. Unknown metric IDs, unknown rules, and wildcards like `dpdp.*` in `config.yaml` are config errors.
4. **Be honest about verification.** The docs mark what the maintainer actually ran. Not verified in 0.1.0: the free path with real Ollama models, `local_hf` speech on real hardware, LiveKit Cloud, Pipecat websocket mode against a live bot, `pip install impulse-evals` from the index (install from a clone with `pip install -e .`).
5. **Compliance results are technical signals only, never legal certification.** Say so whenever you discuss pack results.

## Quickstart

```bash
pip install -e .                               # from the root of a clone
impulse run examples/config.yaml --dry-run     # no model, no key; exit 0, "6 scenarios, 6 passed"
impulse init                                   # asks language + a model per role; writes .impulse/config.toml
impulse doctor examples/config.yaml            # exit 2 until every FAIL row is fixed
impulse run examples/config.yaml               # exit 0, 1 (--fail-on met) or 2 (config error)
```

Free path: `ollama pull` two models, `pip install -e ".[local]"` (or `.[light]`), `impulse init --profile free`. For CI: `impulse init --non-interactive --language en-IN` (language is required; Impulse never assumes one).

Optional extras: `local` (faster-whisper, transformers; install torch yourself), `light` (Kokoro ONNX + faster-whisper, no PyTorch, no GPU), `voice` (pydub; ffmpeg for non-WAV agent audio), `livekit`, `pipecat` (pipecat-ai 1.12.x), `dev`.

## Models and keys

Three roles: **primary** judge (critical/high compliance checks, primary-tier metrics, `generate`, `convert-failure`), **fast** judge (medium checks, fast-tier metrics, the caller's summary of the system prompt), **simulator** (the caller).

Model resolution per role, first match wins:
1. `config.yaml`: `judge.primary_model`, `judge.fast_model`, `simulator.model` (a `YOUR_..._MODEL` placeholder counts as unset);
2. env `IMPULSE_MODEL_PRIMARY` / `IMPULSE_MODEL_FAST` / `IMPULSE_MODEL_SIMULATOR`;
3. `[models]` in `.impulse/config.toml`;
4. otherwise a config error naming the role (exit 2).

Keys: environment variable first, then `[keys]` in `.impulse/config.toml`. The variable is the one LiteLLM reads for the provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, ...), or the role's `api_key_env`. No key for `ollama/`, `ollama_chat/`, any role with `api_base`, or credential-file providers (`bedrock/`, `vertex_ai/`, `sagemaker/`, `oci/`). Keys are redacted from logs and reports.

A primary model naming 14B parameters or fewer (e.g. `:7b`) with a critical check in the run triggers a warning: recommend `impulse calibrate`.

`.impulse/config.toml` sections: `[keys]`, `[models]` (`primary`, `fast`, `simulator`), `[model_options.<role>]` (`api_base`, `api_key_env`, `extra_params`), `[defaults]` (`reporter` json|html|all, `language`, `fail_on` critical|high|any). `.impulse/` is git-ignored by its own `.gitignore`.

## config.yaml

Minimal valid config (generic Python agent, text mode):

```yaml
agent:
  entrypoint: "my_agent:respond"      # module:function, next to this file
  system_prompt: "system_prompt.txt"  # FILE PATH (never inline text), relative to this file; the judge's rubric
simulator:
  model: "YOUR_SIMULATOR_MODEL"
scenarios:
  - name: "order_status_us"
    language: en-US
    persona: "online shopper whose order is two days late"
```

Required: the `simulator` block; `agent.entrypoint` for `generic`/`custom`; `agent.livekit.url` / `agent.pipecat.url` / `agent.http.url` for those adapters; `name`, `language`, `persona` per scenario.

**agent**: `type` (voice | chat | multimodal; default voice), `adapter` (generic | livekit | pipecat | http | custom), `entrypoint`, `options` (free-form for custom adapters; put key variable NAMES here, never secrets), `system_prompt` (always wins over a prompt the agent exposes), `model` (only used to warn if it equals a judge model), `tools` (null = unknown; `[]` = no tools; list of names or `{name, description}`), and the `livekit` / `pipecat` / `http` blocks.

**judge**: `primary_model`, `fast_model`, `primary_options`, `fast_options`. **simulator**: `model`, `persona_style` (realistic | adversarial | polite), `mode` (only `cascade` in v1), `options`.

**voice_config** (its presence turns voice mode on; `--dry-run` is always text):
`tts.provider` / `stt.provider`: `local_hf`, `local_light`, `litellm`, `openai_compatible`, `impulse_hosted`. Also `model`, `models_by_language` (TTS), `voice`, `device`, `compute_type` (STT), `base_url`, `api_key_env`; plus `save_audio`, `save_recording` (both true), `timeout_s` (30).

**metrics**: `enabled` (true; false disables everything incl. UNIV-025 and VOICE-011), `voice.ttfa_warn_ms` 1500 / `ttfa_fail_ms` 3000, `dead_air_ms` 1500, `wer_warn` 0.15 / `wer_fail` 0.30, `tail_latency_warn_ms` 3000 / `tail_latency_fail_ms` 5000, `speech_rate_wpm` [110, 180]. `noise_simulation` is an old key that does nothing now (`true` warns that `voice_config.noise` replaced it).

### Scenario fields

| field | default | notes |
|---|---|---|
| `name` | required | unique, not blank |
| `language` | required | must match `^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$`; Hinglish as `hi-Latn-IN` |
| `accent` | null | free label; cohorts are language + accent |
| `domain` | general | fsi, healthcare, recruitment, ecommerce, edtech, telecom, realestate, travel, legal, general; a label only in hand-written configs |
| `direction` | inbound | outbound = agent speaks first (opener does not count as a user turn) |
| `max_turns` | 9 | counts USER turns; use 12 for fsi/healthcare |
| `persona` | required | plain words; can be in the scenario language |
| `opening_user_message` | null | sent verbatim, no LLM call |
| `expected_outcome` | null | appended to every judge criterion |
| `key_terms` | [] | for VOICE-013 |
| `code_switching` | false | enables VOICE-031 |
| `assertions` | [] | explicit `pack.rule_id` strings, no wildcards |
| `metrics` | [] | empty = every universal judge metric UNIV-001..UNIV-023 |
| `expected_tool_calls` | [] | UNIV-025: `name`, `required_arguments`, `argument_values`, `must_succeed` (true), `min_count` (1), `max_count` |
| `forbidden_tool_calls` | [] | tool names that must never be called |
| `seed` | null | best-effort hint only |
| `inject_events` | [] | modeled but NOT executed in v1 (the report says so) |

Sensitive tool-call arguments (`otp`, `password`, `pin`, `token`, `card_number`, ...) are stored as `[REDACTED]`, so `argument_values` on such a key can never match; use `required_arguments`, the tool's result, or UNIV-023 instead.

Tool rules: if `agent.tools` is set, every expected/forbidden name must be in it. `max_count < min_count`, `min_count: 0` without `max_count`, negative `min_count`, or a name both expected and forbidden are config errors. Recommend `max_count: 1` on anything costly to repeat (bookings, payments, refunds, `escalate_to_human`).

Drafting: `impulse generate --prompt system_prompt.txt --domain healthcare --language en-IN` writes `scenarios/generated.yaml` (11 scenarios: happy paths, edge cases, adversarial, hallucination traps). Tell the user to review before pasting. Domain defaults add India packs only for languages with region `IN`; `hipaa` is never added automatically.

## Adapters

| adapter | agent is | key config | notes |
|---|---|---|---|
| `generic` | a Python function `(user_text, history) -> str \| dict`, sync or async; add a `user_audio=None` parameter to receive the caller's WAV bytes in voice mode | `entrypoint: "module:function"` | `user_text` is `""` for the outbound opener; dict keys `text`, `audio` (WAV bytes), `heard_text`, `tool_calls` |
| `livekit` | a LiveKit Agents worker | `livekit.url`, `api_key_env`, `api_secret_env`, `agent_name` | joins one room per scenario as a client; token needs the `can_manage_agent_session` grant for tool capture |
| `pipecat` | a Pipecat bot | `pipecat.url`, `mode: eval` (start bot with `python bot.py -t eval`) | one client at a time (scenarios run sequentially); set `agent.system_prompt` so Impulse can reset the bot between scenarios, and keep that file identical to the bot's deployed prompt (the bot then runs the file's prompt) |
| `http` | any language behind JSON | `http.url` (+ optional `start_url`, `close_url`, `headers`, `api_key_env`, `turn_timeout_s`, `send_audio`) | one POST per turn; no retries; on Windows prefer `127.0.0.1` over `localhost`; header values must be printable ASCII with no leading or trailing whitespace |
| `custom` | anything else (Vapi, Retell, in-house) | `entrypoint: "module:Class"` | subclass `AgentInterface`; implement async `start(scenario)`, `send(user_text, user_audio)`, `close()` |

Tool call dict (generic and HTTP): `{"name": str, "arguments": dict, "result": ..., "success": True|False|None, "error": str|None}`. `arguments` must be a dict (parse JSON strings first). If `success` is None, Impulse infers it from `result` (`{"ok": true}` / `{"success": true}` success; `{"error": ...}`, `{"ok": false}`, `{"status": "failed"}` failure).

HTTP turn request: `{"conversation_id", "user_text" (null for the outbound opener), "history", "scenario_name"}, plus `user_audio` (base64 WAV) when `http.send_audio: true``; answer `{"text", "tool_calls", "heard_text", "audio" (base64 WAV)}`.

Custom adapter skeleton:

```python
import time
from impulse_evals.adapters.base import AgentInterface, AgentReply
from impulse_evals.plugin_types import ToolCall

class MyAdapter(AgentInterface):
    def __init__(self, config):
        self.options = dict(config.agent.options)
    async def start(self, scenario):
        ...  # open the platform's conversation
    async def send(self, user_text, user_audio):   # (None, None) = wait for the outbound opener
        t = time.perf_counter()
        text = "..."                                # call the agent
        return AgentReply(text=text, latency_ms=(time.perf_counter() - t) * 1000, tool_calls=[])
    async def close(self):
        ...  # must tolerate an already closed connection
```

Raise `AgentClosed(message, reply=...)` when the agent hangs up, `AgentTurnError(message, reply=...)` on a mid-turn failure (keeps the partial turn).

Turn timeouts: `livekit|pipecat|http.turn_timeout_s`, else `voice_config.timeout_s` in voice mode, else 30 s. Raise to 60–90 for slow tools or long spoken replies.

## Voice

Pick a provider per side:
- `local_light`: free, no key, no PyTorch, no GPU. Good in en-US and es-MX; **poor STT in Hindi** (Whisper `small`). For Hindi: keep `local_light` TTS, use `stt: {provider: litellm, model: "deepgram/nova-3"}` (needs `DEEPGRAM_API_KEY`).
- `local_hf`: free, any Hugging Face TTS model + faster-whisper STT; TTS needs PyTorch.
- `litellm`: hosted speech (Deepgram, Groq, OpenAI, ...); `model` required.
- `openai_compatible`: any server with the OpenAI audio API (`base_url` required).

Latency is always the agent's own, never Impulse's TTS/STT time. VOICE-009/013 need the agent to return `heard_text`. VOICE-011 is the harness round-trip WER (informational): if it is poor, the other audio metrics are unreliable; usually the TTS cannot speak the persona's language.

### Noise simulation (experimental)

In voice mode, Impulse can mix background noise into the simulated caller's audio after its TTS and before the agent hears it. Status: experimental, covered by offline tests only, not yet run against a live LiveKit, Pipecat, HTTP or custom agent. Say so when you recommend it.

```yaml
voice_config:
  tts: {provider: local_light}
  stt: {provider: local_light}
  noise:                       # default for every scenario
    profile: pink              # white | pink | brown | file
    snr_db: 10                 # -5..40; 20 mild, 10 busy street, 5 harsh
scenarios:
  - name: "booking_clean"
    language: en-IN
    persona: "patient who wants a slot on Tuesday"
    noise: false               # clean baseline
  - name: "booking_busy_street"
    language: en-IN
    persona: "patient who wants a slot on Tuesday"
    noise: {profile: brown, snr_db: 5, label: "busy street"}
```

- `profile: file` takes `file:` (a WAV next to the config; Impulse ships no recordings) and an optional `label`.
- A scenario's `noise` block replaces `voice_config.noise` as a whole; `noise: false` turns it off. In text mode and `--dry-run` noise is ignored with one warning.
- The agent must actually receive audio: LiveKit, Pipecat and custom adapters get the noisy audio as is; a generic handler needs a `user_audio` parameter; an HTTP agent needs `agent.http.send_audio: true`. Otherwise the run warns that the noise has no effect.
- Watch VOICE-009 (what the agent heard; needs `heard_text`), VOICE-027 (asks the caller to repeat) and UNIV-001/023/025. VOICE-011 always uses the clean audio, and latency stays the agent's own.
- Workflow: run clean, run noisy, then `impulse compare RUN_CLEAN RUN_NOISY`. The report adds noise badges, a "clean caller audio (before noise)" player under each noisy caller turn, and a "Pass rate by noise level" table.

## Checks

A scenario **fails** on any failed assertion, any `poor` metric that is neither skipped nor informational, or a scenario error. `warning` never fails.

**Universal** (judge unless noted): UNIV-001 task completion, 003 context retention (skips under 4 turns), 005 factual accuracy, 007 hallucination resistance, 009 instruction adherence, 011 escalation correctness, 013 tone, 015 topic focus, 017 clarity, 019 harmful/misleading content (safety floor), 021 language consistency, 023 tool call correctness (also catches "you're booked" without a tool call), 025 expected tool calls (deterministic; added automatically when the scenario has tool expectations).

**Voice**: VOICE-001 TTFA, 003 dead air, 005 tail latency (p95), 009 agent-hearing WER, 011 harness WER (informational, auto in voice mode), 013 domain-term error rate, 015 speech rate, 029 spoken-length fit (deterministic); 007 interruption handling, 027 clarifying questions, 031 code-switch handling, 035 turns to close, 037 voice naturalness, 039 responsiveness (judge). In text mode 001/003/005 run as labelled proxies.

**Chat** (only when listed; skip for `agent.type: voice`): CHAT-001 response latency, 003 formatting, 005 reference validity.

**Compliance packs** (`pack.rule_id`, severity):
- `dpdp` (India DPDP Act): `consent_before_pii` critical, `purpose_disclosure` high, `withdrawal_path` medium.
- `rbi_kyc` (India banking): `identity_before_disclosure` critical, `no_credential_request` critical, `no_unauthorized_commitment` high.
- `trai_dlt` (India outbound only; skips inbound): `consent_for_outbound` critical, `honors_opt_out` critical, `identifies_sender` high.
- `hipaa` (US, opt-in): `identity_before_phi` critical, `minimum_necessary` high, `no_phi_to_third_party` high.

Critical/high go to the primary judge, medium to the fast judge. A critical check the judge could not evaluate becomes a scenario error, never a silent pass.

## CLI and exit codes

| command | purpose |
|---|---|
| `impulse init [--profile free\|hosted\|custom] [--language L] [--no-ping] [--non-interactive]` | write `.impulse/config.toml` |
| `impulse doctor [CONFIG]` | check Python, config, models (5-token ping), agent, voice; exit 2 on any FAIL |
| `impulse generate --prompt F --domain D [--language L] [--count 11] [--out F] [--voice-mode]` | draft scenarios |
| `impulse run CONFIG [--dry-run] [--fail-on critical\|high\|any] [--concurrency N] [--reporter json\|html\|all] [--no-embed-audio]` | run, write `.impulse/runs/<run_id>/` |
| `impulse report RUN_ID [--no-embed-audio]` | rewrite `report.html` |
| `impulse compare RUN_A RUN_B [--fail-on critical\|high]` | diff two runs; default `critical`: exit 1 on a critical regression |
| `impulse convert-failure .impulse/runs/<id>/scenarios/<name>.json` | turn a failure into `scenarios/regression/<name>_regression.yaml` |
| `impulse calibrate DIR` | judge agreement with human labels (agreement rate, Cohen's kappa) |

Exit codes: **0** success; **1** Impulse judged and found a problem (`--fail-on` met, regression, invalid model answer); **2** setup/usage/config error, nothing judged. In CI treat 2 as "fix the pipeline", 1 as "the agent regressed".

`--fail-on` for `impulse run` (default `high`; `impulse init` writes `fail_on = "high"`, and a `config.toml` without `fail_on` means `high`): `critical` = a critical assertion fails or any scenario error; `high` = also a high assertion fails or has no verdict ("high assertion had no verdict"), or a poor UNIV-019 (credential request, false urgency, misleading claim) or UNIV-025, so a bot that asks for a password exits 1 without any compliance pack; `any` = any failed scenario. `impulse compare` keeps the default `critical` (UNIV-019 and UNIV-025 count as high there too), and `assert_no_critical` keeps its critical rule. `impulse compare` exits 1 on any newly errored scenario, environment errors included; check the `scenario error:` note before blaming the agent.

CI recipe: set `IMPULSE_MODEL_*` and provider keys from secrets (no `config.toml` needed), then

```bash
impulse run evals/config.yaml --fail-on high
impulse compare evals/baseline/run.json .impulse/runs/RUN_ID/ --fail-on high
```

From pytest: `from impulse_evals.testing import impulse_check, assert_no_critical`.

## Reading a report

`.impulse/runs/<run_id>/report.html` is self-contained (no network, no script). Read in order: header warnings → critical failures → scenarios table → open a failed scenario → its assertions and metrics → click "Hear turn N" to play the failing turn → call timeline (red = reply slower than `ttfa_fail_ms`, shaded = dead air) → tool call trace. A dry run's "passed" is not a verdict. Large voice reports: `impulse report RUN_ID --no-embed-audio`.

## Common errors

| message | fix |
|---|---|
| `no model configured for the primary role` | `impulse init`, set `IMPULSE_MODEL_PRIMARY`, or write the model in `config.yaml` |
| `agent.entrypoint: cannot import module ...` | the module must sit next to `config.yaml` or be on `PYTHONPATH` |
| `local_hf needs the 'local' extra` / `local_light needs the 'light' extra` | install the extra in the same venv; on Python 3.14: `pip install kokoro-onnx==0.6.1 --ignore-requires-python` |
| `local_hf TTS needs PyTorch` | install the torch build for the machine; on Windows a missing `vcruntime140_1.dll` needs the VC++ Redistributable (x64) |
| `LiveKit agent did not join within N s` | set `livekit.agent_name` matching the worker's `agent_name`; raise the worker's `load_threshold` for local runs |
| `Pipecat bot did not send bot-ready` | start the bot with `-t eval` and copy the printed URL into `pipecat.url` |
| `... did not finish its turn within 30 s` | raise `turn_timeout_s` |
| `Harness round-trip WER is high` | the TTS cannot speak the persona's language; pick a matching voice or say "speaks English only" in the persona |
| `Pipecat bot reports function calls without names or arguments` | set the RTVI function call report level to FULL |
| `header values must not start or end with whitespace` | trim the `agent.http.headers` value |
| `{scenario}: the adapter's close failed after the conversation ended; the results are kept` | not a failed run: the scenario was still evaluated; check the HTTP `close_url` endpoint |
| `Pipecat eval bot: Impulse reset its conversation with the agent.system_prompt file ...` | informational; keep the file identical to the bot's prompt, or set `agent.system_prompt: null` |
| a tool error mentioning `openaipublic.blob.core.windows.net` | LiteLLM's tokenizer needs a one-time download; go online once or set `CUSTOM_TIKTOKEN_CACHE_DIR` |

## Known limitations (0.1.0)

Audio is embedded as WAV (large reports; the size warning counts the base64 page text over 50 MB); `inject_events` are not executed; noise simulation is experimental (noise stops 20 ms after the caller's last word); no packet loss, deliberate interruptions, or repeat-with-pass-rate; no ready-made Vapi/Retell adapter (use `custom`); Pipecat late tool results are recorded nowhere; `seed` is only a hint; no dashboard, database, PDF report, or telephony. Smaller open issues from the 2026-11-02 final test (short redaction values, a few WER number forms, odd UNIV-023/025 reason texts, `provider: litellm` asking for a key with a keyless `base_url`) are listed under Roadmap > Small known issues; none changes a verdict.

## Where to look in references/documentation.md

Search for these `## ` headings: Overview, Getting started, Run Impulse for free, Concepts, Configuration, Writing scenarios, Adapters, Voice testing, Metrics, Compliance packs, Reading the report, CLI reference, CI integration, Calibration, Cookbook (ecommerce, Talk2Neo, healthcare, BFSI recipes), Troubleshooting, FAQ, Known limitations, Privacy and security, Architecture, Contributing, Verified setups, Changelog, Roadmap. Useful sub-headings: "Noise simulation (experimental)", "noise (experimental)" (config keys), "Help us test noise simulation", "UNIV-023 notes", "UNIV-025 rules", "WER normalisation", "The exact JSON" (HTTP adapter), "What Impulse captures" (Pipecat), "Where models and keys come from", "Validation".
