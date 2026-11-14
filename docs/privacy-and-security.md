<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Privacy and security

This page says exactly what data leaves your machine when you run Impulse, how keys are handled, and what the files it writes contain.

## Quick version

- **On the free local path** (Ollama for the language models, `local_hf` or `local_light` for speech), your system prompt and transcripts stay on your machine.
- **With any remote provider**, that provider receives your system prompt and the transcripts. Impulse says so on a folder's first real run: "Impulse sends your system prompt and transcripts to the LLM and voice providers you configured."
- **Impulse itself sends no telemetry.** It has no licence keys or feature gating, and it never pays for, or proxies, your LLM calls.
- **Keys** come from environment variables or `.impulse/config.toml`. They are redacted from everything Impulse prints or writes.
- **Run results** (transcripts, audio, reports) are written under `.impulse/`, which is git-ignored by default.
- **Compliance results are technical signals only**, not legal certification.

The one rule most people need: keep keys in environment variables (or `.impulse/config.toml`), and put only their NAMES in `config.yaml`:

```yaml
agent:
  http:
    url: "https://agent.example.com/turn"
    api_key_env: "MY_AGENT_TOKEN"   # the NAME of the variable, never its value
```

## Details

### What each provider receives

| who | receives |
|---|---|
| the primary and fast judge models | for every judge-based metric and assertion: the agent's system prompt, the safety floor, the full transcript (including tool call names, arguments, results, and errors), the criterion, the scenario's expected outcome, and, for UNIV-023, the `agent.tools` list |
| the fast judge model | once per distinct system prompt: the system prompt, to write the short summary the simulated caller gets |
| the simulator model | that summary, the scenario's persona and language, and the conversation so far |
| the text-to-speech (TTS) provider | the simulated caller's sentences (voice mode) |
| the speech-to-text (STT) provider | Impulse's audio of the caller's turns and your agent's audio (voice mode) |
| your agent (and its LiveKit server or Pipecat bot) | the simulated caller's turns, as text or audio; with Pipecat eval mode and `agent.system_prompt` set, also the system prompt from `config.yaml` (the conversation reset) |
| the primary model, for `impulse generate` | the system prompt file |
| the primary model, for `impulse convert-failure` | the failed checks with their reasons, the transcript, and the system prompt |
| the judge models, for `impulse calibrate` | each item's transcript and system prompt, with the check's criterion |

Whether a provider is local or remote depends on what you configure:

- **On your machine:** `ollama/...` model strings, and `local_hf` or `local_light` speech.
- **Wherever they run:** a hosted model, an `api_base` on another machine, and a LiveKit server.
- **Remote speech:** `openai_compatible`, `impulse_hosted`, and `litellm` speech. For example, `litellm` sends the caller's text and the audio to a hosted speech provider such as Deepgram.

Read each provider's own data and retention terms.

### Network requests that carry no user data

- **Model downloads.** These happen on first use only, or when you pull a model.
  - `local_hf` downloads model weights from the Hugging Face Hub the first time a model is used.
  - `local_light` downloads its Kokoro files (about 354 MB) from the kokoro-onnx GitHub release. Its faster-whisper `small` model comes from Hugging Face.
  - Ollama downloads models when you pull them.
- **LiteLLM's cost map.** The `impulse` CLI makes LiteLLM use the cost map bundled with it. So LiteLLM makes no request for it at start-up (verified by the test suite).
  - To let LiteLLM fetch the current map, set `LITELLM_LOCAL_MODEL_COST_MAP=False` before you run `impulse`.
  - Python code that imports Impulse without the CLI (for example the pytest helpers in `impulse_evals.testing`) does not do this. Set `LITELLM_LOCAL_MODEL_COST_MAP=True` yourself for fully offline runs.
  - That request would carry no user data (not verified at runtime). How it works: [Architecture](architecture.md#litellms-cost-map-at-start-up).
- **tiktoken's cl100k_base file.** LiteLLM counts tokens with tiktoken. On some installs (seen on a fresh Windows install) its bundled copy is unusable, and tiktoken downloads the file once from `openaipublic.blob.core.windows.net`. That happens even for a local `ollama/` model.
  - The `impulse` CLI repairs the bundled copy locally when it starts, so no download is needed.
  - Offline, a failed download looks like an Ollama failure. `impulse doctor` and `impulse init` then add a note naming the tiktoken download to the failed row.
  - To use your own folder, set `CUSTOM_TIKTOKEN_CACHE_DIR`; Impulse never changes it.
  - The download request carries no user data. How the repair works: [Architecture](architecture.md#tiktokens-cl100k_base-file).

### Keys

Where a key comes from, in this order:

| order | source | notes |
|---|---|---|
| 1 | the environment variable | in CI, set it from your CI system's secrets |
| 2 | `[keys]` in `.impulse/config.toml`, under the same name | `impulse init` asks for keys with hidden input and stores them here as plain text: keep the file private |

- **Names, not values, in `config.yaml`.** Every `api_key_env` setting holds the NAME of a variable, never the key: `*_options`, `pipecat.api_key_env`, `livekit.api_key_env` / `api_secret_env`, `voice_config.*.api_key_env`, `http.api_key_env`.
- **`config.toml` is private.** `impulse init` writes `.impulse/.gitignore` first. It then creates `.impulse/config.toml` readable and writable by you only on Linux and macOS. On Windows the file gets the folder's default permissions. Details: [Architecture](architecture.md#key-file-permissions).
- **Local models need no key:** `ollama/...`, `ollama_chat/...`, and any role with an `api_base`.
- **Some providers use their own credentials:** `bedrock/...`, `sagemaker/...`, `vertex_ai/...`. For these, LiteLLM lists no `*_API_KEY` variable. LiteLLM uses the provider's credentials from your environment, and Impulse passes no key ([Configuration](configuration.md#where-models-and-keys-come-from)).
- **Version control.** `.impulse/.gitignore` contains `*`, so neither `config.toml` nor the run results are committed.

### Redaction

Impulse replaces secrets with `[REDACTED]`. What is hidden, where, and when:

| what | where it is hidden | when |
|---|---|---|
| every key value Impulse knows (list below) | what the CLI prints; every scenario error and run warning; the adapter metadata stored with each transcript, which is what `run.json` and the report store of them | always |
| masked and encoded forms of a known key: the form providers print in errors (`sk-test-************cdef`), the URL-encoded form, and the key inside base64 text | the same places | always |
| credentials written into the config: text values of the model options, `pipecat.headers` values except `Origin`, the token of `Bearer <token>`, passwords and credential query values in URLs | the same places | always; the values are still sent where you configured them |
| credentials in `pipecat.url`, `livekit.url`, and a voice `base_url` | `run.json`, the report, error texts, and `impulse doctor` show the URL without its user name, password, and credential query parameters (`token`, `key`, `api_key`, `api-key`, `access_token`, `apikey`, `api_token`, `sig`, `signature`, `secret`, `password`, `auth`) | always; Impulse still connects with the full URL |
| the value of a sensitive key (`otp`, `pin`, `password`, `token`, `card_number`, their plurals and similar) in a tool call's `arguments` or `result` | stored as `[REDACTED]` in `run.json`, so the report and the judge transcript never carry it; `true`, `false` and `null` stay | always |
| keys in generated scenario files | every string value and every header comment line of the files `impulse generate` and `impulse convert-failure` write; a key in the stored scenario name never reaches the regression file name | before the YAML is written (the file still loads) |
| keys in long error texts | the error text | before it is shortened, so a key is never printed in part |

The keys Impulse knows:

- the values under `[keys]` in `.impulse/config.toml`;
- the standard key variables, and every environment variable whose name ends in `_API_KEY`;
- every variable your config names;
- the keys resolved for the run, or by `impulse doctor` / `impulse init`, whatever their variable's name;
- keys typed at an `impulse init` prompt.

What is not redacted:

- Transcripts are stored as the conversation happened. Impulse never puts a key into a prompt.
- A short value found only by its variable's name ending is not treated as a key. For example `MAPS_API_KEY=book` does not hide the word "book".

The exact matching rules (minimum lengths, dry runs, which config values count as secrets): [Architecture](architecture.md#redaction-rules).

### What Impulse writes

`.impulse/runs/<run_id>/` holds:

- `run.json` and `scenarios/<name>.json`: full transcripts, the agent's system prompt, tool calls with their arguments and results, every verdict and reason;
- `report.html`;
- in voice mode, the audio of every turn plus a stereo recording of each call.

Treat the folder like the conversations it records. It can hold whatever personal data your agent or your scenarios put into a conversation. Delete runs you no longer need.

Outside `.impulse/`, Impulse writes what you ask for:

- `impulse init`: a starter `config.yaml` with `--profile free`;
- `impulse generate`: `scenarios/generated.yaml` or `--out`;
- `impulse convert-failure`: `scenarios/regression/`.

It also writes two caches, which hold no user data:

- `local_light` keeps its Kokoro model files in a cache folder. That is `IMPULSE_CACHE_DIR` when set, else `%LOCALAPPDATA%\impulse-evals\kokoro` on Windows.
- On Linux and macOS the default is `$XDG_CACHE_HOME/impulse-evals/kokoro` or `~/.cache/impulse-evals/kokoro`.
- The CLI may write a repaired tiktoken copy under your system temp folder, in `impulse-tiktoken/` ([Architecture](architecture.md#tiktokens-cl100k_base-file)).

It never writes to your agent's code or config.

### The HTML report

- Every text taken from a run (transcripts, reasons, names, config values, warnings, tool arguments) is HTML-escaped. So a transcript containing `<script>` cannot run.
- The page loads nothing from outside: no external CSS or JavaScript, and no script at all. No `src`, `href`, or CSS `url(` starts with `http:`, `https:`, or `//`.
- With `--embed-audio` (the default) the audio is inside the file: sharing the report shares the recordings.

### The judge and adversarial transcripts

Simulated callers include adversarial ones, so transcripts are untrusted data.

- Every judge prompt wraps the transcript in `<transcript>` tags.
- It tells the judge to ignore any instruction inside them.
- A fixed safety floor applies on top of your agent's system prompt ([Concepts](concepts.md#the-judge-your-system-prompt-is-the-rubric)).

### Adapters

- **Pipecat eval mode** sets the bot's RTVI function call report level to FULL. RTVI is Pipecat's client/server message protocol.
  - Impulse sends an `eval-configure` message on every `impulse run` and `impulse doctor` connection.
  - The setting stays on the bot afterwards: the bot keeps reporting tool names, arguments, and results at that level after Impulse disconnects.
  - Run Impulse against development or test bots.
- **Pipecat `api_key_env`** sends the named variable's value as `Authorization: Bearer <value>` to the bot.
- **HTTP `api_key_env`** sends the named variable's value as `Authorization: Bearer <value>` to your endpoint.
- **LiveKit:** Impulse creates its access tokens locally from your API key and secret. It joins one new room per scenario as an ordinary participant.
  - The token carries the `can_manage_agent_session` grant, used for the system prompt request and tool-call capture.
  - When `livekit.agent_name` is set, the token dispatches that agent into the room.
  - Tool-call capture through this grant worked on a local `livekit-server --dev` in the 2026-10-24 and 2026-10-25 campaigns. Other LiveKit servers are not verified.

### Compliance results and licences

- Every compliance result carries "Technical signal only. Not legal certification."
- The `regulation_ref` strings are informational. Have counsel verify them before they appear in customer-facing material ([Compliance packs](compliance-packs.md)).
- The models you choose (Ollama models, Hugging Face speech models, hosted models) come with their own licences and terms of use. Check each model's page before you use it.
- Impulse's own licences: [LICENSING.md](../LICENSING.md).

## Next steps

- [Run Impulse for free](run-for-free.md): the path where nothing leaves your machine.
- [Configuration](configuration.md): where keys and models are configured.
- [Architecture](architecture.md): file permissions, redaction rules, and start-up network details.
- [Compliance packs](compliance-packs.md): what a compliance result does and does not mean.
- [Contributing](contributing.md): report a security problem or send a fix.

---

<!-- docs-nav -->
[← Previous: Known limitations](known-limitations.md) | [Docs home](index.md) | [Next: Architecture →](architecture.md)
