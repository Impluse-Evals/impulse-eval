<!-- SPDX-License-Identifier: Apache-2.0 -->
# Stage B notes: fixes round 3, A1 and A2 (Pipecat)

Test files: `tests/test_pipecat_turn_timeout.py` (A1, DEFECT-8) and `tests/test_pipecat_tool_errors.py`
(A2, DEFECT-9). Data: `pipecat_guaranteed_loan_927a72.json` (run 20261025T141732Z-927a72, turns 1-9 and
the scenario error) and `pipecat_personal_data_request_02afb8.json` (run 20261025T143747Z-02afb8, turns
1-4 and the RTVI run warning), trimmed from the campaign run.json files (read only; checked for keys and
tokens: none). The turn 10 text (`SKIPPED verify_identity(...)verifier:default_api:...`) is not in
run.json (the turn was lost); it is copied from the spec.

Choices made where the spec is open (builders: tell the orchestrator if one is wrong; do not edit tests):

1. A1 voice fallback to `bot-llm-text` tokens: the tests expect the tokens joined as text mode joins a
   run's tokens (`""`, then stripped), the existing `_reply_part` rule.
2. A1 latency in voice mode with no speech audio but text: "first speech audio (voice) or first kept
   text" is read as the existing `_reply` rule (speech audio first, else the first text). The tests only
   pin: speech audio wins; neither arrived -> NaN (`test_voice_timeout_without_speech_audio_or_text_has_nan_latency`
   uses silent audio below 1 % plus a tool call: the reply exists, because "audio" counts as content,
   and its latency is NaN).
3. A1 "audio" as content: any collected bot audio counts, also audio below 1 % of full scale (the spec
   says "audio", not "speech audio").
4. A1 close codes: "Close and harness errors are unchanged" is pinned as: an abnormal close (1006, 1011)
   is still a plain `PipecatError` with the same text and is NOT an `AgentTurnError`; 1000/1001 stay
   `AgentClosed` with the reply so far.
5. A1 config: the LiveKit cases of the shared error texts are parametrized next to the Pipecat ones
   (they pass today) so both fields give identical texts; the "one helper" itself is not testable
   through the public API.
6. A2 `error` for a template result without an RTVI error, when the raw string has surrounding
   whitespace: "that result string" is ambiguous (raw or stripped); the test accepts either
   (`error.strip() == template`). The stored result stays the raw string.
7. A2 not tested (ambiguous): a `cancelled: true` call that also gets a matching RTVI error. "cancelled
   first" vs "tied: error = the error text" conflict; only cancelled + template result (no RTVI error)
   is pinned (error stays "cancelled").
8. A2 ordering "error before the call is recorded" (pending) is tested as a third order in addition to
   the two campaign orders, plus a call recorded only from its stop (missed in-progress report).
9. A2 a dict result `{"value": "<template>"}` (a dict, not a string) is pinned as unchanged (success
   None in the adapter), from "a dict result is unchanged".
10. Runner voice end-to-end (A1) uses eval mode only (one `send_audio` per user turn makes the
    per-turn trigger simple); adapter-level tests cover both modes.

Checked before handing over: with a throw-away monkeypatch prototype of the A1/A2 adapter behaviour
(scratchpad only, not committed), every test in both files passes except the 20 that need the
`turn_timeout_s` config field, so the scripted channel timings are satisfiable.

## A2-rev

Stage B for the revised tie rule (`tests/test_pipecat_tool_errors_rev.py`). Choices and ambiguities:

1. "Every arrival order of the four messages" (two parallel `verify_identity` calls): read as the five
   messages in-progress call-1, in-progress call-2, stop call-1 (template), stop call-2 (success), RTVI
   error, with each in-progress before its own stop: 30 orders, both success variants (`{"ok": true}`,
   `{"success": true}`), both modes. The no-stop / timeout / close cases use four messages (12 orders).
2. "Stop never arrives, turn ends in the grace window": an in-progress report WITH a `tool_call_id`
   holds the turn open, so a normal end with an unstopped call is only possible with the deprecated
   `llm-function-call` or an in-progress report with no `tool_call_id`; both are tested. A call with an
   id that never stops is tested through the timeout and the close (1000) partial replies.
3. A `{"ok": true}` call has `success=None` in the adapter but a dict `result`, so rule 3 ("result is
   None") never ties to it; pinned as untouched in the adapter (`success None, error None`), and the
   runner infers `success: true` in run.json (one runner test).
4. Not tested (ambiguous): an RTVI error queued after turn 1 ended and read from the backlog before turn
   2's send (`pre_turn`), when turn 2 has a NOT-CAPTURED same-name call. Literal rule 3 would tie it in
   turn 2; R3P-2's suggested fix would consume it. Only the case "turn 2's same-name call succeeds"
   (untouched either way) is pinned, as the spec asks.
5. Not tested: two pending errors and two eligible calls at reply build (rule 3 "each pending error ...
   the most recent call" does not say how several are paired), and rule 2 with two template-failed calls
   and one error (the existing A2 behaviour already ties the most recent).
