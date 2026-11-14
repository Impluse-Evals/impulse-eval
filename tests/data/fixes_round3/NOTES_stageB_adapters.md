<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Stage B notes: fixes round 3, B3 (custom adapters) and B4 (HTTP adapter)

Test files: `tests/test_custom_adapter.py`, `tests/test_http_adapter.py`. No conftest change.

## Seams the tests assume (builders: provide these)

1. **HTTP client.** `HTTPAdapter` creates its client as `httpx.AsyncClient(..., transport=transport)` and, when
   no transport was given, passes `transport=None` or leaves it out. Runs through the default factory and the
   CLI route every `httpx.AsyncClient` created without a transport to a `httpx.MockTransport` by patching
   `httpx.AsyncClient.__init__` (the same technique as `mock_voice_server`).
2. **Per-request timeout.** The timeout is the httpx request timeout: `request.extensions["timeout"]["read"]`
   equals `turn_timeout_s` on every request (start, turn, close), e.g. `httpx.AsyncClient(timeout=N)`.
   A timeout is any `httpx.TimeoutException` (tests raise `httpx.ReadTimeout` from the transport); a
   connection error is any other `httpx.TransportError` (tests raise `httpx.ConnectError`).
3. **Doctor TCP check.** The "agent http" row connects with `asyncio.open_connection(host, port)` (looked up
   as the attribute of the `asyncio` module when the check runs, e.g. inside `asyncio.wait_for(..., 5)`;
   the writer is closed with `close()` / `await wait_closed()`) OR `socket.create_connection((host, port),
   timeout=5)` (attribute of the `socket` module). Tests patch both and record `(host, port)`; a failed
   connect raises `ConnectionRefusedError(111, "Connection refused")`. `loop.create_connection` or
   `from asyncio import open_connection` would bypass the patch.
4. **Bearer key lookup** reads `./.impulse/config.toml` (current directory, as `runner.run` does) when the
   variable is not in the environment.
5. **Custom loading errors leave `runner.run` as `ConfigError`** (as the generic adapter's import error), so
   `impulse run` and `--dry-run` exit 2. A ConfigError from `HTTPAdapter.start()` (missing key) also
   propagates out of `run`.

## Ambiguities (simplest reading chosen)

- B3 "entrypoint required" and "not of the form module:Class": tests accept the `ConfigError` from either
  `load_config` or `runner.run` (and, for doctor, from the "config" row or the "agent adapter" row). Every
  other loading error must come from the factory: `load_config` never imports the module (tested).
- B3 constructor error "(redacted, one line, 200 characters)": tests only require the whole message to be
  at most `len("could not create {m}:{a}: {type}: ") + 200` characters, one line, the key redacted before the
  cut (a key straddling the cut leaves no 10-character prefix). Both "cut {type}: {message}" and "cut
  {message}" satisfy it.
- B3 doctor FAIL row "with the ConfigError text": tests check that the row detail contains the message and
  the field `agent.entrypoint` (the doctor's usual `str(ConfigError)` form).
- B4 `audio` that is not a string (e.g. a number): tests expect the same text "HTTP agent returned 'audio'
  that is not base64" (the only audio error text the spec names).
- B4 `{body}` of a non-object answer is the raw response text (`[1, 2]`, `"just text"`), not a
  re-serialisation; an empty body is `(empty body)` for this error too.
- B4 key row texts follow the LiveKit rows: PASS "{NAME} is set (environment variable)" / "{NAME} is set
  ([keys] in .impulse/config.toml)" (the spec's "({NAME} is set (..." has a stray parenthesis), FAIL the
  missing-key text. No key row when `api_key_env` is not set.
- B4 doctor FAIL "cannot connect to {url}: {error}": tests check only the prefix and that the OS message
  ("Connection refused") is in it; `{url}` is `http.url` without credentials (PASS "reachable: {url}" too).
- B4 the redaction placeholder of the adapter's own credentials is not named by the spec: tests only check
  that the secret is gone and that "[REDACTED]" appears somewhere (the run redactor's placeholder).
- B4 a 3xx answer is not tested (redirect following is not specified).
- B3/B4 examples: tests check the files exist, `examples/custom_adapter` and `examples/http` dry runs exit 0
  with the remote warning, the example `EchoAdapter` books on "book", has a module docstring, and the HTTP
  README contains `conversation_id` and "not verified". `server.py` is not started (no sockets in tests).

## State on the current code (before the build)

All 171 new tests fail for the right reason: `agent.adapter: custom` / `http` is rejected by the config
literal (reported as an explicit `pytest.fail` / exit-code assertion), `AgentConfig.options` and
`AgentConfig.http` are missing, the example folders are missing. No collection or fixture error.
`tests/test_doctor.py` passes; `tests/test_cli_exit_codes.py` has one failure,
`test_dry_run_of_examples_config` (`total_scenarios == 6`), the B5 superseded assertion written by the B5
Stage B session, not by these files.

## B3/B4-rev

Test file: `tests/test_http_adapter_rev.py` (51 tests; no conftest change, no existing test edited). On the
current code 44 fail for the right reason (DID NOT RAISE, raw httpx/SystemExit exceptions, unstripped
token, Basic instead of Bearer, public header values redacted, backend not warmed) and 7 pass (behaviour
already right: a trickle inside the deadline succeeds, printable header values accepted, userinfo as Basic
auth without `api_key_env`, error texts without userinfo (both cases), the adapter's own error texts redact
every header value, KeyboardInterrupt from a custom constructor is not converted).

Seams:

1. **R1 deadline.** Tests use `httpx.MockTransport` (no httpx timeout applies to a mock), so the deadline
   must be enforced by Impulse itself around the whole POST incl. reading the body
   (`asyncio.wait_for(client.post(...), turn_timeout_s)` or `asyncio.timeout`). Real loop, 0.3 s deadline, a
   1.5 s trickle (one byte every 50 ms) or a 1.5 s silent handler; the error must come within 1.2 s.
2. **R3 warm-up.** `start()` must make anyio load its asyncio backend through
   `anyio._core._eventloop.get_async_backend()` (e.g. `await anyio.sleep(0)`). The test deletes
   `loaded_backends["asyncio"]`, patches `anyio._core._eventloop.import_module` to sleep 0.5 s for
   `anyio._backends.*`, uses a transport that calls `anyio.sleep(0)` (as httpcore does), and asserts
   `"asyncio" in loaded_backends` after `start()` and a first-turn latency below 250 ms. A warm-up that only
   imports `anyio._backends._asyncio` directly would not populate `loaded_backends` and fails the test.
3. **R7.** Checked through the public `runner.config_redactor(cfg, None)` and through the stored tool call
   of a `run()` (adapter factory with MockTransport; the runner's redactor is applied to stored results).
4. **R8.** Checked synchronously through `custom_adapter.create_adapter(cfg)` and
   `runner.default_adapter_factory(cfg, redact=...)` (a SystemExit inside the event loop would end pytest).

Ambiguities (simplest reading chosen):

- R2: the check is every character in 0x20..0x7E, so a tab, ESC and a trailing newline (YAML `|`) in a
  header value are errors; an empty header value is accepted. Header field is `agent.http.headers.{name}`
  with the name as written in config.yaml (e.g. `X-Key`).
- R2/R6: the token is stripped with `str.strip()` (spaces, tab, CR, LF) for env and `[keys]` values alike. A
  token that is blank after stripping is not tested (missing-key text or the ASCII text both plausible).
- R4: "the URL without userinfo" keeps the path and the query (`?lang=en` stays); userinfo is dropped from
  all three URLs when `api_key_env` is set.
- R5: the InvalidURL error may come from `start()` or `send()`. For an unparseable URL with userinfo
  (`http://user:pw@[::1/turn`) the message must still not show the password; today
  `strip_url_credentials` returns such a URL unchanged, so the builder must handle that case too.
- R5: "any other httpx error" is tested with `httpx.CookieConflict` and an `httpx.InvalidURL` raised by the
  transport (neither is an `httpx.HTTPError`).
- R7: sensitive names are matched case-insensitively by substring (`Keycloak-Realm` contains `key`, so it is
  redacted); `Accept` and `User-Agent` values are not.
- R8: `SystemExit(3)` gives the message `3` (`str(exc)`), `SystemExit("adapter gave up")` gives that text.
