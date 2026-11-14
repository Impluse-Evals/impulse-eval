<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Troubleshooting

This page lists the messages Impulse prints when something is wrong, what causes each one, and how to fix it.

## Quick version

- **Start with `impulse doctor`.** It checks Python, both config files, every model, the agent and the voice providers. Each `FAIL` row says what to fix.
- **What it checks of the agent** depends on the adapter. Generic: the entrypoint import. Pipecat: the handshake. HTTP: a TCP connect only. LiveKit and custom adapter connections are not checked.

```bash
impulse doctor config.yaml
```

- **What success looks like.** The last line is `All checks passed.` and the exit code is 0. While any row says `FAIL`, the exit code is 2.
- **Exit code 2** means the command could not do its job. The message says why. Causes:
  - a usage or configuration problem; a config error names the exact field, for example `scenarios[1].language`;
  - any `FAIL` row of `impulse doctor`, such as an unreachable model;
  - a file Impulse "cannot write";
  - "Aborted." (Ctrl+C or end of input at a prompt), or "Error: unexpected error: ...".
- **Then find your message below.** Each entry gives the problem, its cause, and the fix. Limits with no fix yet are in [Known limitations](known-limitations.md).

## Details

### Install and imports

- **`impulse` is not found.** Cause: the virtual environment is not active, or the install failed. Fix: activate it and run `pip install -e .` from the clone again.
- **pip refuses to install on Python 3.10 or older.** Cause: Impulse needs Python 3.11 or newer. Fix: create the virtual environment with Python 3.11 or newer ([FAQ](faq.md)).
- **Windows: `vcruntime140_1.dll` is missing** when `torch` or `livekit` is imported. Fix: install the latest Microsoft Visual C++ Redistributable (x64), then open a new shell. Impulse's own messages say the same for a failed import of the `livekit` or `pipecat` extra, and of torch for `local_hf` text-to-speech.
- **ffmpeg.** Impulse handles WAV audio itself. ffmpeg is needed only by pydub (the `voice` extra), to convert agent audio that is not WAV.

### Configuration errors

The message follows the field path. A config error exits 2.

| message | fix |
|---|---|
| `system prompt file not found: ... (agent.system_prompt is a file path, relative to the config file's folder)` | point `agent.system_prompt` at a file. The path is relative to the folder of `config.yaml`; inline prompt text is not accepted |
| `the simulator block is required` | add a `simulator:` block (a `model` placeholder is enough for a dry run) |
| `unknown metric ID ...` / `unknown assertion ...` | use an ID from [Metrics](metrics.md) or a rule from [Compliance packs](compliance-packs.md) |
| `wildcard dpdp.* is not accepted in config.yaml; list explicit pack.rule_id strings` | list each rule, for example `dpdp.consent_before_pii` |
| `invalid language code ...; it must match ^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$` | use a code such as `en-US`, `es-MX`, `en-IN`, `hi-IN`, or `hi-Latn-IN` |
| `duplicate scenario name ...; names must be unique` | rename one of the scenarios |
| `tool ... is not in agent.tools` | add the tool to `agent.tools`, or fix the name in `expected_tool_calls` / `forbidden_tool_calls` |
| `min_count: 0 is allowed only together with max_count` / `max_count ... is lower than min_count ...` | fix the expectation ([Writing scenarios](writing-scenarios.md)) |
| `at_turn ... is outside 1..9 (max_turns)` | `inject_events[].at_turn` counts user turns, from 1 to `max_turns` |
| `simulator.mode: audio_native is planned for v2. Use cascade.` | remove `mode` or set `cascade` |
| `pipecat.mode: daily is planned for v2. Use eval or websocket.` | Pipecat over Daily, SmallWebRTC or LiveKit transports is v2; use eval or websocket mode |
| `turn_timeout_s 0 must be greater than 0` | set a positive number of seconds, or remove the key |
| `voice_config.noise.profile: unknown noise profile ...` / `scenarios[0].noise.snr_db: snr_db 41 is outside -5..40` | noise simulation: use `white`, `pink`, `brown` or `file`, and an `snr_db` from -5 to 40 ([Configuration](configuration.md#noise-experimental)) |
| `voice_config.noise.file: noise file not found: ...` / `... is not a WAV file: ...` / `... is silent: ...` | the path is relative to the folder of `config.yaml`; convert other formats to WAV; use a recording with sound in it |

These warnings do not stop a run ([Configuration](configuration.md#validation)):

- "Unknown domain {domain}; using general defaults."
- "metrics.enabled is false; no metrics will run, including expected tool call checks."
- "metrics.noise_simulation is replaced by voice_config.noise; it is ignored." (remove the key; configure noise under `voice_config`)
- "noise simulation needs voice mode (voice_config); noise settings are ignored."
- "agent.http.send_audio is false: the HTTP agent receives text only, so noise simulation has no effect on it."
- "pipecat.{key} is ignored in eval mode."

### Models and keys

- **`no model configured for the primary role`** (or `fast`, or `simulator`). Cause: no model string for that role. Fix: set it in `config.yaml`, in `IMPULSE_MODEL_PRIMARY` / `IMPULSE_MODEL_FAST` / `IMPULSE_MODEL_SIMULATOR`, or under `[models]` in `.impulse/config.toml` (`impulse init` writes it). `impulse generate`, `impulse convert-failure` and `impulse calibrate` read no `config.yaml`; for them only the environment and `config.toml` count.
- **`... is not set. Set the environment variable ... or add it to [keys] in .impulse/config.toml.`** Cause: a hosted model needs a key. Fix: set the named variable, the one LiteLLM reads for that provider (for example `TOGETHERAI_API_KEY` for `together_ai/...`), or put it under `[keys]`. Local models (`ollama/...`, `ollama_chat/...`, or any role with `api_base`) never need one. Neither do providers with their own credentials, such as `bedrock/...` or `vertex_ai/...`.
- **`cannot determine the provider of model ...`.** Cause: LiteLLM does not recognise the model string. Fix: use LiteLLM's `provider/model` form.
- **A doctor model row says `not reachable:`** and the provider's error. Cause: the server is down (for Ollama, start it), the model is not pulled or does not exist, or the key is wrong. Fix it and run `impulse doctor` again.
- **A failed row or error names `openaipublic.blob.core.windows.net`.** It ends with:
  "LiteLLM tried to download the tiktoken cl100k_base encoding; this needs internet access once, or set CUSTOM_TIKTOKEN_CACHE_DIR to a folder holding the file (not a model or provider failure)".
  Cause: LiteLLM's token counter needs its `cl100k_base` file, and the copy bundled with LiteLLM was rejected.
  Fix: connect to the internet once, or set `CUSTOM_TIKTOKEN_CACHE_DIR` to a folder holding the file ([Privacy and security](privacy-and-security.md#network-requests-that-carry-no-user-data)).
  - Why the bundled copy fails: some LiteLLM wheels ship it with Windows line endings. The `impulse` CLI repairs that copy into your system temp folder when it starts, unless `CUSTOM_TIKTOKEN_CACHE_DIR` is set.
  - `TIKTOKEN_CACHE_DIR` alone has no effect under LiteLLM, which sets it when it is imported.
- **"Judge model {model} looks small (<= 14B parameters). Critical verdicts from small judges are unreliable; run impulse calibrate before trusting them."** A warning, not an error. See [Calibration](calibration.md).
- **"Judge prompt is about {n} tokens, over 80% of num_ctx={num_ctx} for {model}; the judge may truncate the transcript."**
  Fix: raise `num_ctx` in the role's options, or shorten the scenarios.
- **"agent.model is the same as the judge model ({model}); the judge may be biased toward the agent's own outputs."**
  Fix: use a different judge model.

### The generic adapter

- **`agent.entrypoint: cannot import module ... from ... or sys.path`.** Cause: the module is not importable from the config file's folder or the Python path. Fix: move the config next to your agent, or add its folder to `PYTHONPATH`. The free example needs `examples/` on `PYTHONPATH`.
- **`agent.entrypoint: ... is not of the form module:function`.** Fix: write it as `module:function`, for example `sample_agent:respond`.
- **`agent.entrypoint: module ... has no function ...`.** Fix: check the function name after the colon.
- **A scenario error with your agent's exception text.** Cause: your function raised. The scenario fails with that error and no metric runs. Fix: the bug in your agent.

### Voice

In voice mode, text-to-speech (TTS) voices the simulated caller and speech-to-text (STT) transcribes audio. The word error rate (WER) metrics compare what was said with what was heard. The "harness" is Impulse's own TTS-then-STT loop, used to check itself.

Install and models:

- **`local_hf needs the 'local' extra: pip install impulse-evals[local]`.** Fix: `pip install -e ".[local]"` from the clone, and install `torch` yourself.
- **`voice_config.tts.provider: local_hf TTS needs PyTorch, which the 'local' extra does not install: pip install torch (see https://pytorch.org/get-started/locally/)`.** Cause: the `local` extra is installed but torch is not. Fix: install the PyTorch build that fits your machine. The run stops before its first model call; `impulse doctor` shows it on the voice TTS row. STT (faster-whisper) works without torch.
- **`voice_config.tts.provider: local_hf TTS needs PyTorch, which is installed but could not be imported (...).`** Cause: importing torch failed; Impulse tries it in a separate Python process. The part in parentheses is the error, for example `OSError: [WinError 126] ...` or `ImportError: DLL load failed ...`. It says "the import ended the Python process with exit code {code}" when Python crashed.
  - Fix on Windows, for a DLL load error such as a missing `vcruntime140_1.dll`: install or update the Microsoft Visual C++ Redistributable (x64). Then open a new shell. The message says so too.
  - Otherwise reinstall the PyTorch build that fits your machine.
- **`local_light needs the 'light' extra: pip install impulse-evals[light]`.** Fix: `pip install -e ".[light]"` from the clone. The run stops before its first model call; `impulse doctor` shows it on the voice TTS or STT row.
- **The same message, with a Python 3.14 note at the end.** The note:
  "(kokoro-onnx 0.6 supports Python 3.11 to 3.13; on 3.14 install it with: pip install kokoro-onnx==0.6.1 --ignore-requires-python)".
  Cause: on Python 3.14 the `light` extra skips kokoro-onnx, because kokoro-onnx 0.6.1 declares Python below 3.14.
  Fix: run the command in the message.
- **The first `local_light` run waits a long time.**
  Cause: it downloads about 354 MB of Kokoro files. It first prints this line:
  "Downloading the light voice model (353.7 MB) to {folder} (first use only)."
  faster-whisper also downloads its `small` model (about 484 MB). Later runs reuse the files.
- **`local_light: downloaded {name} does not match its checksum; delete {folder} and try again`** or **`local_light: could not download {name} from {url}: ...`.** Cause: the download was cut or changed on the way. Fix: delete the folder named in the message and run again. Or download both files yourself and set `voice_config.tts.model` to their folder.
- **`local_light TTS has no voice for {language}. ...`.** Cause: Kokoro has no default voice for that language. Fix: set `voice_config.tts.voice`, or use `litellm` or `openai_compatible` TTS for it.
- **`voice_config.tts.model: no TTS model for language ...`** or **`voice_config.stt.model: no STT model set. ...`.** Cause: a placeholder is still in place. Fix: replace `YOUR_TTS_MODEL` / `YOUR_STT_MODEL`, or add the language to `models_by_language`.
- **`voice_config.stt.model: deepgram/nova-3 needs the key DEEPGRAM_API_KEY: set the environment variable or add it to [keys] in .impulse/config.toml`** (or another `litellm` model and key). Cause: no key for the `litellm` provider. Fix: set the named variable or add it under `[keys]`. The run stops before any paid call.
- **`voice_config.stt.model: required when provider is litellm`.** Fix: set a LiteLLM model string, such as `deepgram/nova-3`.
- **"litellm STT failed with model {model}: ..."** or **"litellm TTS failed with model {model}: ..."**. Cause: the provider refused or failed the request; the rest is its error. Fix: check the model string, the key, and for `openai`, `azure` or `elevenlabs` TTS that `voice_config.tts.voice` is set ([Voice testing](voice-testing.md#litellm)).
- **`impulse_hosted does not support language ...`.** Cause: `impulse_hosted` speaks `en-IN`, `en-US` and `hi-IN` only. Fix: use another provider for other languages.
- **`voice_config.tts.base_url: required when provider is openai_compatible`.** Fix: HTTP providers need `base_url`.
- **`... TTS: POST ... returned application/json, not audio: ...`** (or `text/html; ...`, `no content-type`, ...). Cause: the TTS server answered 200 with an error page or JSON instead of audio; the rest is the server's answer. Fix: check the model and voice names, and any proxy between Impulse and the server.

During a run:

- **"{scenario}: Impulse STT failed on turn {n}: ..."**. Cause: Impulse's own STT failed on that turn's audio. The turn is kept (the agent turn with the adapter's text), and the scenario still runs. VOICE-011 (user audio) or VOICE-015 (agent audio) skips with "Impulse STT failed: ...". Fix: the STT provider, then run again for those metrics.
- **"Harness round-trip WER is high; audio metrics in this run are unreliable."**
  Cause: Impulse's own TTS and STT do not understand each other well in this language. The WER metrics of this run are not trustworthy. Fix: use speech models that support the scenario language.
  Numbers, dates and spelled-out IDs do not cause it on their own ([Metrics](metrics.md#wer-normalisation)).
  For example, `ORD-1004` and "o r d dash one zero zero four" count as the same.
- **The same warning in a Hindi run with `local_light`.** Cause: expected with Whisper `small`. In a measured hi-IN run, VOICE-011 was 0.88; Whisper `small` wrote numbers as misspelled Hindi words. Fix: keep `local_light` TTS and use Deepgram STT through `provider: litellm` ([Voice testing](voice-testing.md#light-local-voice-no-pytorch-weak-laptops)).
- **A poor VOICE-011 and garbled caller audio, but no error.** Cause: the TTS probably cannot speak the scenario's language. Or the simulated caller writes in another language than the TTS speaks, such as Hinglish from an `en-IN` persona into an English-only voice. Impulse cannot detect this for `openai_compatible` or `local_hf`. Fix: check the provider's voice list, pick a voice with `voice` or `models_by_language`, and write personas in a language the TTS speaks ([Voice testing](voice-testing.md#language-codes)).
- **Skips with "agent returned no audio".** Cause: your generic handler returns text only. Fix: return `audio` (WAV bytes) for the audio metrics.
- **Skips with "agent does not expose its transcription; return heard_text from the adapter".**
  Fix: return `heard_text` from your handler, or keep the agent's user transcriptions enabled (LiveKit, Pipecat).
- **Skips with "STT provider returned no word timings".**
  Cause: your STT provider gives no word timestamps (Groq, for example); VOICE-015 cannot run with it. Fix: use a provider with word timings if you need VOICE-015.
- **"the generic agent function has no user_audio parameter, so it never hears the noise; noise simulation has no effect on it."** Cause: noise simulation is on, but your handler is `(user_text, history)`. Fix: add a `user_audio=None` parameter and transcribe it ([Adapters](adapters.md#generic-adapter)).
- **"noise simulation is ignored in --dry-run."** A dry run never makes audio. Run without `--dry-run` to hear the noise.
- **VOICE-009 gets much worse with noise, but VOICE-011 stays good.** That is the expected picture: VOICE-011 checks Impulse's own clean audio, VOICE-009 your agent's hearing of the noisy audio. Listen to the turn's noisy and clean players in the report ([Reading the report](reading-the-report.md#noise-simulation)).

### LiveKit

- **"LiveKit agent did not join within 10 s."** Cause: no agent joined the scenario room. Fix:
  - when your agent's worker is registered with an `agent_name`, set `livekit.agent_name` so Impulse's token dispatches it;
  - otherwise check that the agent is running and connected to the same server, or raise `livekit.handshake_timeout_s`.
- **The join error only now and then, with "no servers available" in the LiveKit server log.**
  Cause: the worker declined the job. The machine was busy (its `load_threshold`), or its idle process was still restarting.
  Fix: use explicit dispatch with `livekit.agent_name`, raise the worker's `load_threshold` for test runs, or keep more idle processes ([Adapters](adapters.md#livekit)).
- **Local LiveKit runs fail after a network change.** Cause: `livekit-server --dev` binds to the network address the machine had when it started. Fix: restart it.
- **"LiveKit agent did not finish its turn within 30 s."** Cause: the agent stayed `"speaking"` or `"thinking"` past the turn timeout. That is `livekit.turn_timeout_s` when set, else `voice_config.timeout_s` (default 30, also in text mode). Fix: raise `livekit.turn_timeout_s`, for example to 60, for agents that run several slow tools in one turn or speak long replies. The turn's text and tool calls so far are kept in the transcript, so you can see which tool was still running.
- **No system prompt or no tool calls captured from a LiveKit agent.**
  Cause: both use the agent's `lk.agent.session` stream, which needs the `can_manage_agent_session` grant on Impulse's token. Your LiveKit server or project must allow it (not verified against a live server). Fix: allow the grant. Setting `agent.system_prompt` in `config.yaml` avoids the system prompt request.
- **`livekit adapter needs the 'livekit' extra: pip install impulse-evals[livekit]`.** Fix: `pip install -e ".[livekit]"`.

### Pipecat

- **"Pipecat bot did not send bot-ready within 10 s. Check that the bot is running at {url} (eval mode: started with -t eval) and RTVI is enabled."** RTVI is Pipecat's client/server message protocol. Fix: start the bot with `-t eval`, copy the URL it prints into `pipecat.url`, and keep RTVI enabled.
- **"Pipecat bot accepts one client at a time. Run with --concurrency 1, or start one bot per connection."** Cause: another client (another run, `impulse doctor`, or `pipecat eval`) is connected to the bot. Fix: stop it and run again.
- **"Pipecat bot did not finish its turn within 30 s."** Cause: the bot did not end its turn within the turn timeout. That is `pipecat.turn_timeout_s` when set, else `voice_config.timeout_s`, else 30 s. Fix: raise `pipecat.turn_timeout_s` for slow tools or long spoken replies. The turn so far (its text, audio and tool calls) is kept in the transcript before the scenario error.
- **Tool-call turns are slow in a Pipecat text run.** Cause: expected with a `-t eval` bot. It turns its TTS back on after each text turn, so the answer after a tool call is spoken, which takes time. The reply text is still taken once ([Adapters](adapters.md#what-impulse-captures)).
- **"Pipecat bot closed the connection before bot-ready (close code 4003) ..."** Cause: the development runner rejected the token or the `Origin` header. Fix: check `pipecat.api_key_env` and `pipecat.headers`.
- **"Pipecat eval harness pipeline failed: Error setting up processor: server rejected WebSocket connection: HTTP 403"** (or HTTP 401, HTTP 404; eval mode). Cause: the bot refused the WebSocket connection. Fix by status:
  - 403: the bot accepts origins only from `PIPECAT_ALLOWED_ORIGINS`; put an allowed `Origin` header in `pipecat.headers`;
  - 401: the token was rejected; check `pipecat.api_key_env`;
  - 404: the path is wrong; copy the URL the `-t eval` bot prints into `pipecat.url`.
- **"the Pipecat bot closed the connection (close code 1006)"** (or 1011, or another code that is not 1000 / 1001). Cause: the bot dropped the connection during the scenario, usually because its process crashed or its pipeline failed. Fix: check the bot's own log.
- **"Pipecat eval bots keep their conversation between scenarios. Set agent.system_prompt in config.yaml so Impulse can reset it, or reset the LLM context in the bot's on_client_connected handler."**
  Fix: set `agent.system_prompt` so each scenario starts fresh.
- **"Pipecat eval bot: Impulse reset its conversation with the agent.system_prompt file before each scenario, so the bot ran that prompt, not its own. Keep the file identical to the bot's prompt."**
  Not an error: with `agent.system_prompt` set, the prompt under test is the file. Keep it identical to the prompt your bot deploys, or leave `agent.system_prompt` null and let the bot answer `get_system_prompt`.
- **"{scenario}: the adapter's close failed after the conversation ended; the results are kept: {error}"**
  The conversation finished, but the adapter's close step failed (for the HTTP adapter: the `close_url` request). The scenario was evaluated as usual; check the close endpoint.
- **"Pipecat bot reports function calls without names or arguments. Set the RTVI function call report level to FULL so UNIV-023 can check tool arguments."** Fix: set the report level in the bot ([Adapters](adapters.md#function-call-report-level-full)).
- **"Pipecat eval mode needs a pipecat-ai release that includes pipecat.evals. Upgrade pipecat-ai, or set pipecat.mode: websocket."** Fix: install the extra (`pipecat-ai` 1.12.x), or use websocket mode.
- **"pipecat-ai {version} is not the tested 1.12.x; eval mode relies on its internal APIs."** Fix: `pip install -e ".[pipecat]"` to get the tested release.
- **`pipecat adapter needs the 'pipecat' extra: pip install impulse-evals[pipecat]`.** Fix: `pip install -e ".[pipecat]"`.
- **A dry run of a Pipecat config runs no scenario.** Expected: it validates the config without connecting. It says:
  "Dry run: scenarios run only with the generic adapter; the pipecat config was validated without connecting."

### Reports and runs

What each part of the report shows: [Reading the report](reading-the-report.md).

- **The audio players in a shared report are silent.** Cause: it was written with `--no-embed-audio`, which links the files in `audio/`. Fix: rebuild it with `impulse report RUN_ID --embed-audio`.
- **"Embedded audio is {n} MB; the HTML report will be large. Use --no-embed-audio to link the files instead."** Cause: the size counts the base64 text of every embedded player (each turn's audio and the call recording, once each; base64 is a third larger than the WAV files). Long calls and many voice scenarios make it large. Fix: keep the report next to its `audio/` folder and use `--no-embed-audio`.
- **`Error: run not found: ...`.** Cause: `impulse report` and `impulse compare` look for `./.impulse/runs/<id>/run.json` in the current folder. Fix: run them from the folder you ran in, or pass a run folder or `run.json` path.
- **`impulse compare` exits 1, but no check got worse.**
  Cause: a scenario that errored in run B but not in run A is a critical regression. Environment errors count too, such as a LiveKit worker that declined a job. Fix: look for "scenario error: ..." in the compare output, fix the environment, and run again.
- **"critical check {rule_id} could not be evaluated: judge returned no verdict".**
  Cause: the judge returned nothing usable for a critical rule, so the scenario fails. Fix: check the provider (rate limits, safety filters) and that the judge model can answer in JSON ([Concepts](concepts.md#when-a-scenario-passes)).
- **"evaluation of {check_id} failed: ..."**. Cause: a judge call raised, for example a provider error. Provider errors are one line: `{error type} (HTTP {status}, provider {provider}, model {model}): {message}`, ending with `; retry after {n} s` when the provider names a wait. Fix: the provider problem, then run again.
- **Quota errors, then a stopped judge model.** The quota error reads "RateLimitError (HTTP 429, ...)". The run warning reads:
  "Judge calls to {model} were stopped after a provider quota error; the checks that use it were not evaluated after that: ..."
  Cause: the judge model's quota ran out with a retry delay of a minute or more. Impulse stopped calling that model for the rest of the run.
  Its remaining checks skip with this reason, and get no warning each:
  "evaluation failed: judge calls to {model} were stopped for this run after a quota error: ..."
  The other judge model goes on. A short delay (under 60 s) does not stop the model. Fix: wait for the quota to reset (the `retry after {n} s` part), raise the quota, or use another model, then run again.
- **"the agent closed the conversation before its first turn: ..."**. Cause: the agent hung up, or its connection closed, before it said anything (for example it crashed on join). The rest is the adapter's reason. Nothing was judged. Fix: check that the agent starts and answers.
- **The privacy notice appears on every run.** Cause: the notice shows until a non-dry run finishes in that folder and writes `.impulse/.privacy_notice_shown`. A run that stops on a config error shows it again next time.

## Next steps

- [CLI reference](cli.md): exit codes per command.
- [Configuration](configuration.md): every key and its default.
- [Adapters](adapters.md): the LiveKit and Pipecat setups.
- [Reading the report](reading-the-report.md): every part of the HTML report.
- [Known limitations](known-limitations.md): documented limits and their workarounds.
- [FAQ](faq.md): short answers to common questions.

---

<!-- docs-nav -->
[← Previous: Cookbook](cookbook.md) | [Docs home](index.md) | [Next: FAQ →](faq.md)
