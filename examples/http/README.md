<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Testing an agent over HTTP with Impulse

`adapter: http` lets Impulse test an agent written in any language: Impulse sends one JSON `POST` per agent turn to your endpoint and reads the agent's reply from the JSON answer. Your agent needs one endpoint (`url`) and, optionally, two more (`start_url`, `close_url`).

## Run it

1. Start the echo agent in this folder (standard library only, for local testing): `python examples/http/server.py`. Success: it prints `Echo agent listening on http://localhost:8080/turn ...`.
2. Check the connection: `impulse doctor examples/http/config.yaml`. Success: the row `agent http` is `PASS reachable: http://localhost:8080/turn`. Doctor only opens a TCP connection to the host and port; it never sends an HTTP request, because a turn can have side effects (a booking, a payment).
3. Validate the config without any request or LLM call: `impulse run examples/http/config.yaml --dry-run`. Success: exit code 0 and the warning "Dry run: scenarios run only with the generic adapter; the http config was validated without connecting."
4. Run it for real once judge and simulator models are configured (`impulse init`): `impulse run examples/http/config.yaml`.

## config.yaml

```yaml
agent:
  adapter: http
  http:
    url: "http://localhost:8080/turn"     # required; http:// or https://
    start_url: null                       # optional: POSTed once when a scenario starts
    close_url: null                       # optional: POSTed once when a scenario ends
    headers: {}                           # extra request headers (values are redacted in errors)
    api_key_env: null                     # NAME of an env var; its value is sent as "Authorization: Bearer <value>"
    turn_timeout_s: null                  # seconds per request; null = voice_config.timeout_s, else 30
```

The key named by `api_key_env` is read from the environment, then from `[keys]` in `.impulse/config.toml`. Put the variable's NAME in config.yaml, never its value. Its value and the `headers` values must be printable ASCII without control characters. When `api_key_env` is set, `impulse doctor` also shows an `HTTP agent key` row saying whether the variable is set.

`turn_timeout_s` is a deadline for each whole request (start, turn and close), body included.

On Windows, `localhost` may try IPv6 first and add about 0.4 s to each request, because `server.py` listens on IPv4 only. Use `http://127.0.0.1:8080/...` in the three URLs to avoid it.

## The exact JSON

Every request is a `POST` with the header `Content-Type: application/json`, the `headers` mapping, and `Authorization: Bearer <value>` when `api_key_env` is set. Bodies are UTF-8 JSON. `conversation_id` is `"impulse-"` followed by 12 lowercase hex characters, new for every scenario.

### start (only when `start_url` is set)

Request:

```json
{
  "conversation_id": "impulse-3f9a1c0b7d2e",
  "scenario_name": "http_agent_booking",
  "language": "en-US",
  "direction": "inbound"
}
```

Answer: any 2xx. When the body is a JSON object with a string `system_prompt`, Impulse uses it as the agent's system prompt (only when config.yaml sets no `agent.system_prompt`); any other body is ignored.

```json
{"system_prompt": "You are the booking assistant of a demo clinic. ..."}
```

### turn (`url`, once per agent turn)

Request:

```json
{
  "conversation_id": "impulse-3f9a1c0b7d2e",
  "user_text": "Hi, I'd like to book an appointment for next Tuesday.",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello, this is the demo clinic. How can I help you today?"}
  ],
  "scenario_name": "http_agent_booking"
}
```

- `history` holds the prior turns of this conversation only (the current `user_text` is not in it).
- `user_text` is `null` for an outbound scenario's first request: the agent speaks first (its opener). That request adds no user entry to `history`.
- In voice mode the agent still receives text; user audio is not sent.

Answer: a JSON object. Only these keys are read; other keys are ignored.

```json
{
  "text": "I have booked Tuesday at 10:00 for you.",
  "tool_calls": [
    {
      "name": "book_slot",
      "arguments": {"date": "2026-01-06", "patient_name": "Asha", "phone": "0000000000"},
      "result": {"success": true},
      "success": true,
      "error": null
    }
  ],
  "heard_text": "hi i'd like to book an appointment for next tuesday",
  "audio": null
}
```

- `text`: a string; missing or `null` is an empty reply.
- `tool_calls`: a list of objects with `name` (a non-empty string, required), `arguments` (an object; missing is `{}`), `result` (any JSON value; a value that is not an object is stored as `{"value": ...}`), `success` (`true`, `false` or `null`; `null` lets Impulse infer it from `result`), and `error` (a string or `null`).
- `heard_text` (optional): what the agent understood from the user, stored on the user turn.
- `audio` (optional): a base64 string of a WAV file (standard or URL-safe alphabet, padding optional).

### close (only when `close_url` is set)

Request:

```json
{"conversation_id": "impulse-3f9a1c0b7d2e", "scenario_name": "http_agent_booking"}
```

The answer body is ignored.

### Errors

Impulse never retries a request. A non-2xx answer, a timeout, a connection error, or a turn answer that is not a JSON object ends the scenario with an error (for example `HTTP agent returned HTTP 500 for POST http://localhost:8080/turn: ...`). Error texts never show URL credentials, the bearer token, or the header values. Every error text is listed in [docs/adapters.md](../../docs/adapters.md#errors).

## A Node.js (Express) handler sketch (not verified)

This sketch has not been run against Impulse (not verified); check it against your Express version.

```js
// not verified
const express = require("express");
const app = express();
app.use(express.json());

app.post("/turn", (req, res) => {
  const { conversation_id, user_text, history, scenario_name } = req.body;
  if (user_text === null) {
    return res.json({ text: "Hello, this is the demo clinic. How can I help you today?" });
  }
  const toolCalls = [];
  if (/\bbook\b/i.test(user_text)) {
    toolCalls.push({
      name: "book_slot",
      arguments: { date: "2026-01-01", patient_name: "Test User", phone: "0000000000" },
      result: { success: true },
      success: true,
    });
  }
  res.json({ text: `You said: ${user_text}`, tool_calls: toolCalls });
});

app.post("/start", (req, res) => res.json({}));
app.post("/close", (req, res) => res.json({}));

app.listen(8080, () => console.log("listening on http://localhost:8080/turn"));
```
