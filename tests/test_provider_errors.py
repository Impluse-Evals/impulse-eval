# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 2, A7 (DEFECT-6): one-line provider errors, and judge calls stop after a quota error
with a long retry delay (FIXES-ROUND-2-SPEC.md section A7).

* ``impulse_evals.provider_errors.provider_error_text(exc)``: ``"{type}{details}: {message}{retry}"``
  for an exception whose class is defined in ``litellm`` / ``litellm.*``, else None;
  ``retry_after_seconds(exc)``: the ``retry-after`` header of ``exc.response``, else ``"retryDelay":
  "<n>s"`` in the text, else ``retry in [Nh][Nm][N.Ns]``.
* The runner's ``_error_text`` uses it, so scenario errors, evaluation-failed reasons and warnings,
  and ``cli.fail_on_reasons`` are one line.
* ``LiteLLMJudge``: a 429 / ``RateLimitError`` whose retry delay is >= 60 s stops its model for the
  run (the error is re-raised); later checks routed to it raise ``JudgeStopped`` without a provider
  call; one judge warning per model; no per-check "evaluation failed" warning for ``JudgeStopped``;
  the other model is still called.

A1/A7-rev (review R1, orchestrator decisions; tests at the end of this file):

* R1-M1: provider error text is redacted BEFORE it is parsed, put on one line and cut:
  ``provider_error_text(exc, redact_text=None)`` applies ``redact_text`` to ``str(exc)`` first; the
  runner passes the run's redactor; ``LiteLLMJudge(..., redact=None)`` uses it for the
  ``JudgeStopped`` message and the stop warning. A key straddling the 200-character cut leaves no
  prefix behind.
* R1-L2: a stopped model is checked again before every provider call (the JSON retry too).
* R1-L4: a leading ``<ClassName>: `` (the exception's own class name) is stripped after the
  ``litellm.<Name>: `` prefixes, then ``<word>Exception - ``; ``try again in [Nh][Nm][N.Ns]`` /
  ``try again in N.N s`` (OpenAI wording) is a retry delay too.
* R1-N: ``one_line_error`` and ``is_litellm_error`` in ``__all__``; the stop keyed by
  (model, custom_llm_provider) like the JSON-mode cache.

Campaign case (read-only copy in tests/data/fixes_round2/judge/): run 20261023T202802Z-088f33,
scenario ``personal_data_request``, a ``litellm.RateLimitError`` from Gemini whose ``str()`` is the
stored text after "evaluation of UNIV-009 failed: ". The real ``litellm.RateLimitError`` is built
offline (its constructor makes a local ``httpx.Response``; no network).
"""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent / "data" / "fixes_round2" / "judge"
PRIMARY = "ollama/judge-primary"   # conftest TEST_PRIMARY_MODEL
FAST = "ollama/judge-fast"         # conftest TEST_FAST_MODEL

# The campaign 429: first line of error.message (239 characters) cut to 197 + "...", then the
# retryDelay of the body (37214 s; the "retry in 10h20m14.51s" text would give 37215).
CAMPAIGN_ONE_LINE = (
    "RateLimitError (HTTP 429, provider gemini, model gemini-3.1-pro-preview): "
    "You exceeded your current quota, please check your plan and billing details. For more information on "
    "this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current ..."
    "; retry after 37214 s"
)
STOPPED = "judge calls to {model} were stopped for this run after a quota error: {error}"
JUDGE_STOP_WARNING = (
    "Judge calls to {model} were stopped after a provider quota error; the checks that use it were not "
    "evaluated after that: {error}"
)


def _pe():
    from impulse_evals import provider_errors

    return provider_errors


def _litellm():
    import litellm

    return litellm


def _campaign_exc():
    data = json.loads((DATA_DIR / "bankbot_quota_error_088f33.json").read_text(encoding="utf-8"))
    text = data["exception_str"]
    prefix = "litellm.RateLimitError: "
    assert text.startswith(prefix)
    exc = _litellm().RateLimitError(message=text[len(prefix):], llm_provider=data["llm_provider"], model=data["model"])
    assert str(exc) == text, "the fake reproduces the campaign's str() exactly"
    assert exc.status_code == 429
    return exc


def _response(status, headers=None):
    import httpx

    return httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://example.invalid/v1/chat"))


def _rate_limit(message="openaiException - Rate limit reached for requests", retry_after=None, model="some-model"):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return _litellm().RateLimitError(
        message=message, llm_provider="openai", model=model, response=_response(429, headers)
    )


# ============================================================================================
# provider_error_text / retry_after_seconds
# ============================================================================================


def test_campaign_quota_error_is_one_exact_line():
    exc = _campaign_exc()
    text = _pe().provider_error_text(exc)
    assert text == CAMPAIGN_ONE_LINE
    assert "\n" not in text
    assert text.endswith("; retry after 37214 s")


def test_campaign_retry_delay_comes_from_retry_delay_before_retry_in():
    assert _pe().retry_after_seconds(_campaign_exc()) == pytest.approx(37214.0)


def test_retry_after_header():
    exc = _rate_limit(retry_after=120)
    assert _pe().retry_after_seconds(exc) == pytest.approx(120.0)
    assert _pe().provider_error_text(exc) == (
        "RateLimitError (HTTP 429, provider openai, model some-model): Rate limit reached for requests; retry after 120 s"
    )


def test_retry_after_header_wins_over_the_text():
    exc = _rate_limit(message='geminiException - quota gone, "retryDelay": "37214s"', retry_after=5)
    assert _pe().retry_after_seconds(exc) == pytest.approx(5.0)
    assert _pe().provider_error_text(exc).endswith("; retry after 5 s")


@pytest.mark.parametrize(
    "phrase, seconds, shown",
    [
        ("Please retry in 1h2m3.5s.", 3723.5, 3724),
        ("Please retry in 45s.", 45.0, 45),
        ("Please retry in 2m.", 120.0, 120),
        ("Please retry in 10h20m14.511250394s.", 37214.511250394, 37215),
    ],
)
def test_retry_in_text(phrase, seconds, shown):
    exc = _litellm().RateLimitError(
        message=f"geminiException - Resource exhausted. {phrase}", llm_provider="gemini", model="m-1"
    )
    assert _pe().retry_after_seconds(exc) == pytest.approx(seconds)
    assert _pe().provider_error_text(exc) == (
        f"RateLimitError (HTTP 429, provider gemini, model m-1): Resource exhausted. {phrase}; retry after {shown} s"
    )


def test_no_retry_part_when_no_delay_is_known():
    exc = _litellm().BadRequestError(
        message="GeminiException - contents.parts must not be empty", model="mock-model", llm_provider="gemini"
    )
    assert _pe().retry_after_seconds(exc) is None
    assert _pe().provider_error_text(exc) == (
        "BadRequestError (HTTP 400, provider gemini, model mock-model): contents.parts must not be empty"
    )


def test_only_the_first_line_of_a_multi_line_message():
    exc = _litellm().APIConnectionError(
        message="OllamaException - [WinError 10061] No connection could be made\n  second line\nthird",
        llm_provider="ollama",
        model="llama3",
    )
    assert _pe().provider_error_text(exc) == (
        "APIConnectionError (HTTP 500, provider ollama, model llama3): [WinError 10061] No connection could be made"
    )


def test_top_level_json_message_and_whitespace_collapsed():
    exc = _litellm().AuthenticationError(
        message='AnthropicException - {"type": "error", "message": "  invalid   x-api-key  "}',
        llm_provider="anthropic",
        model="m-2",
    )
    assert _pe().provider_error_text(exc) == "AuthenticationError (HTTP 401, provider anthropic, model m-2): invalid x-api-key"


def test_long_message_cut_to_200_characters():
    exc = _litellm().BadRequestError(message="GeminiException - " + "w" * 250, model="m-3", llm_provider="gemini")
    assert _pe().provider_error_text(exc) == (
        "BadRequestError (HTTP 400, provider gemini, model m-3): " + "w" * 197 + "..."
    )


def _litellm_like(name, module, **attributes):
    cls = type(name, (Exception,), {})
    cls.__module__ = module
    exc = cls(attributes.pop("message", "something broke"))
    for key, value in attributes.items():
        setattr(exc, key, value)
    return exc


def test_details_only_when_known():
    exc = _litellm_like("WeirdError", "litellm.fake_errors", message="something broke", llm_provider="", model=None)
    assert _pe().provider_error_text(exc) == "WeirdError: something broke"
    partial = _litellm_like("WeirdError", "litellm", message="boom", status_code=503)
    assert _pe().provider_error_text(partial) == "WeirdError (HTTP 503): boom"
    string_status = _litellm_like("WeirdError", "litellm", message="boom", status_code="503", model="m-4")
    assert _pe().provider_error_text(string_status) == "WeirdError (model m-4): boom"


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(ValueError("plain error"), id="builtin"),
        pytest.param(RuntimeError("401 Unauthorized"), id="runtime"),
        pytest.param(_litellm_like("RateLimitError", "openai", status_code=429), id="openai-module"),
        pytest.param(_litellm_like("RateLimitError", "litellmx", status_code=429), id="litellmx-module"),
        pytest.param(_litellm_like("RateLimitError", "my_app.litellm", status_code=429), id="nested-name"),
    ],
)
def test_non_litellm_exceptions_give_none(exc):
    assert _pe().provider_error_text(exc) is None


# ============================================================================================
# LiteLLMJudge: stop a model after a quota error with a long retry delay
# ============================================================================================


def _judge():
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    return LiteLLMJudge(PRIMARY, FAST, {}, {})


def _judge_stopped():
    from impulse_evals.judge.litellm_judge import JudgeStopped

    return JudgeStopped


def _transcript():
    from impulse_evals.models import CallTranscript, Turn

    return CallTranscript(
        turns=[
            Turn(speaker="user", text="Please delete my data.", timestamp_ms=0),
            Turn(speaker="agent", text="You can email privacy@sahyogbank.example.", timestamp_ms=900),
        ],
        duration_ms=3000,
        metadata={},
        scenario_name="data_deletion",
        language="en-IN",
        agent_system_prompt="You are BankBot.",
    )


def test_judge_stopped_is_an_exception():
    assert issubclass(_judge_stopped(), Exception)


@pytest.mark.asyncio
async def test_long_quota_delay_stops_the_model_for_the_judge(mock_llm):
    exc = _campaign_exc()
    judge = _judge()
    mock_llm.queue_error(exc)
    with pytest.raises(_litellm().RateLimitError) as info:
        await judge.evaluate(_transcript(), "Did the agent give the deletion route?", "primary")
    assert info.value is exc   # re-raised
    assert mock_llm.call_count == 1

    expected = STOPPED.format(model=PRIMARY, error=CAMPAIGN_ONE_LINE)
    for _ in range(2):
        with pytest.raises(_judge_stopped()) as stopped:
            await judge.evaluate(_transcript(), "Was the agent polite?", "primary")
        assert str(stopped.value) == expected
    assert mock_llm.call_count == 1, "no further call to the stopped model"

    mock_llm.queue_verdict("pass")
    verdict = await judge.evaluate(_transcript(), "Was the tone right?", "fast")
    assert verdict.result == "pass"
    assert mock_llm.call_count == 2
    assert mock_llm.calls[1]["model"] == FAST

    assert judge.get_warnings() == [JUDGE_STOP_WARNING.format(model=PRIMARY, error=CAMPAIGN_ONE_LINE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", [60, 3600])
async def test_quota_delay_of_60_seconds_or_more_stops(mock_llm, retry_after):
    judge = _judge()
    mock_llm.queue_error(_rate_limit(retry_after=retry_after))
    with pytest.raises(_litellm().RateLimitError):
        await judge.evaluate(_transcript(), "criterion", "fast")
    with pytest.raises(_judge_stopped()):
        await judge.evaluate(_transcript(), "criterion", "fast")
    assert mock_llm.call_count == 1
    assert len(judge.get_warnings()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_factory",
    [
        pytest.param(lambda: _rate_limit(retry_after=30), id="30-s-header"),
        pytest.param(lambda: _rate_limit(retry_after=59), id="59-s-header"),
        pytest.param(lambda: _rate_limit(), id="no-delay-known"),
        pytest.param(
            lambda: _litellm().AuthenticationError(
                message="AnthropicException - invalid key, retry in 2h", llm_provider="anthropic", model="m"
            ),
            id="not-a-quota-error",
        ),
    ],
)
async def test_short_or_unknown_delay_and_other_errors_do_not_stop(mock_llm, exc_factory):
    judge = _judge()
    mock_llm.queue_error(exc_factory())
    with pytest.raises(Exception) as info:
        await judge.evaluate(_transcript(), "criterion", "primary")
    assert not isinstance(info.value, _judge_stopped())
    mock_llm.queue_verdict("pass")
    verdict = await judge.evaluate(_transcript(), "criterion", "primary")
    assert verdict.result == "pass"
    assert mock_llm.call_count == 2
    assert [call["model"] for call in mock_llm.calls] == [PRIMARY, PRIMARY]
    assert not any("were stopped" in warning for warning in judge.get_warnings())


# ============================================================================================
# Through the runner: one-line errors, reasons, warnings and --fail-on reasons; the judge stop
# ============================================================================================


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


def _metrics(scenario_result):
    return {metric.metric_id: metric for metric in scenario_result.metric_results}


def _primary_judge_raises(llm_router, exc):
    """Every call to the primary model raises ``exc``; returns the list of those calls."""
    previous = llm_router.mock.responder
    primary_calls = []

    def responder(call):
        if call["model"] == PRIMARY:
            primary_calls.append(call)
            raise exc
        return previous(call)

    llm_router.mock.responder = responder
    return primary_calls


def _all_one_line(result):
    from impulse_evals.cli import fail_on_reasons

    texts = list(result.warnings)
    for scenario_result in result.scenario_results:
        texts.append(scenario_result.error or "")
        texts.extend(metric.reason for metric in scenario_result.metric_results)
        texts.extend(score.reason for score in scenario_result.assertion_results)
    for level in ("critical", "high", "any"):
        texts.extend(fail_on_reasons(result, level))
    for text in texts:
        assert "\n" not in text, text


@pytest.mark.asyncio
async def test_run_stops_primary_judge_calls_after_the_campaign_quota_error(impulse_project, llm_router, fake_adapter):
    from impulse_evals.cli import fail_on_reasons

    project = impulse_project
    metrics = ["UNIV-001", "UNIV-009", "UNIV-013"]   # primary, primary, fast
    project.write_config(
        [
            project.scenario("first_call", max_turns=1, metrics=metrics),
            project.scenario("second_call", max_turns=1, metrics=metrics),
        ]
    )
    primary_calls = _primary_judge_raises(llm_router, _campaign_exc())
    result = await _run(project.load(), adapter_factory=fake_adapter)

    assert len(primary_calls) == 1, "after the first 429 no further primary-model call is made"
    fast_judge_calls = [call for call in llm_router.calls("judge") if call["model"] == FAST]
    assert len(fast_judge_calls) == 2, "the fast model is still called (UNIV-013 in each scenario)"

    stopped = STOPPED.format(model=PRIMARY, error=CAMPAIGN_ONE_LINE)
    first, second = result.scenario_results
    assert first.error == "evaluation of UNIV-001 failed: " + CAMPAIGN_ONE_LINE
    first_metrics = _metrics(first)
    assert first_metrics["UNIV-001"].skipped is True
    assert first_metrics["UNIV-001"].reason == "evaluation failed: " + CAMPAIGN_ONE_LINE
    assert first_metrics["UNIV-009"].skipped is True
    assert first_metrics["UNIV-009"].reason == "evaluation failed: " + stopped
    assert first_metrics["UNIV-013"].label == "good"

    assert second.error == "evaluation of UNIV-001 failed: " + stopped
    second_metrics = _metrics(second)
    assert second_metrics["UNIV-001"].reason == "evaluation failed: " + stopped
    assert second_metrics["UNIV-009"].reason == "evaluation failed: " + stopped
    assert second_metrics["UNIV-013"].label == "good"

    warning = JUDGE_STOP_WARNING.format(model=PRIMARY, error=CAMPAIGN_ONE_LINE)
    assert result.warnings.count(warning) == 1
    assert not any("evaluation of" in w for w in result.warnings), result.warnings   # no per-check warnings
    assert fail_on_reasons(result, "critical") == [
        "first_call: scenario error: evaluation of UNIV-001 failed: " + CAMPAIGN_ONE_LINE,
        "second_call: scenario error: evaluation of UNIV-001 failed: " + stopped,
    ]
    _all_one_line(result)


@pytest.mark.asyncio
async def test_run_with_a_short_quota_delay_keeps_calling_and_warns_per_check(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    project.write_config(
        [
            project.scenario("first_call", max_turns=1, metrics=["UNIV-001", "UNIV-009", "UNIV-013"]),
            project.scenario("second_call", max_turns=1, metrics=["UNIV-001", "UNIV-009", "UNIV-013"]),
        ]
    )
    primary_calls = _primary_judge_raises(llm_router, _rate_limit(retry_after=30))
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert len(primary_calls) == 4, "a 429 with a delay under 60 s does not stop the model"
    one_line = "RateLimitError (HTTP 429, provider openai, model some-model): Rate limit reached for requests; retry after 30 s"
    for scenario_result in result.scenario_results:
        assert scenario_result.error == "evaluation of UNIV-001 failed: " + one_line
        assert _metrics(scenario_result)["UNIV-009"].reason == "evaluation failed: " + one_line
    assert "first_call: evaluation of UNIV-009 failed: " + one_line in result.warnings
    assert "second_call: evaluation of UNIV-009 failed: " + one_line in result.warnings
    assert not any("were stopped" in w for w in result.warnings)
    _all_one_line(result)


@pytest.mark.asyncio
async def test_adapter_litellm_error_is_a_one_line_scenario_error(impulse_project, llm_router, fake_adapter):
    project = impulse_project
    name = "agent_llm_down"
    project.write_config([project.scenario(name, max_turns=1, metrics=["UNIV-001"])])
    fake_adapter.send_errors[name] = (
        1,
        _litellm().APIConnectionError(
            message="OllamaException - [WinError 10061] No connection could be made\nTraceback ...",
            llm_provider="ollama",
            model="llama3",
        ),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].error == (
        "APIConnectionError (HTTP 500, provider ollama, model llama3): [WinError 10061] No connection could be made"
    )
    _all_one_line(result)


PROVIDER_SECRET = "sk-impulse-test-SECRET-provider-5150"   # ANTHROPIC_API_KEY


@pytest.mark.asyncio
async def test_provider_error_text_is_still_redacted(impulse_project, llm_router, fake_adapter, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", PROVIDER_SECRET)
    project = impulse_project
    project.write_config([project.scenario("bad_key", max_turns=1, metrics=["UNIV-001"])])
    _primary_judge_raises(
        llm_router,
        _litellm().AuthenticationError(
            message=f"AnthropicException - invalid x-api-key {PROVIDER_SECRET}\nrequest id req_1",
            llm_provider="anthropic",
            model="m",
        ),
    )
    result = await _run(project.load(), adapter_factory=fake_adapter)
    assert result.scenario_results[0].error == (
        "evaluation of UNIV-001 failed: AuthenticationError (HTTP 401, provider anthropic, model m): "
        "invalid x-api-key [REDACTED]"
    )
    assert PROVIDER_SECRET not in result.model_dump_json()


# ============================================================================================
# A1/A7-rev R1-M1: redact BEFORE parsing, one line and the 200-character cut
# ============================================================================================

# A fake Gemini key (41 characters) and the URL form Gemini errors carry it in (review R1 probe p7).
GEMINI_KEY = "AIzaSyD-THISISAFAKEKEY-0123456789abcdefgh"
GEMINI_KEY_PREFIX = GEMINI_KEY[:6]   # "AIzaSy": no part of the key may survive the cut
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/m:generateContent?key="
REDACTED = "[REDACTED]"
STRADDLE_TAIL = " (timeout after 30 s; the host could not be reached from this network)"


def _straddling_line(lead="Cannot connect to host"):
    """A message line whose key occupies characters 180-220: it straddles the 200-character cut
    (the old order, cut then redact, kept ``?key=AIzaSyD-THISISAFA...``)."""
    head = f"{lead} "
    pad = 180 - len(head) - 1 - len(GEMINI_URL)
    line = head + "x" * pad + " " + GEMINI_URL + GEMINI_KEY + STRADDLE_TAIL
    assert line.index(GEMINI_KEY) == 180 and len(line) > 220
    return line


def _known_key_redactor(text):
    """The run's key redactor (config.redact) with the Gemini key as a known value."""
    from impulse_evals.config import redact

    return redact(text, extra_values=[GEMINI_KEY])


def _cut_200(text):
    """The A7 message rule: at most 200 characters (197 + "...")."""
    return text if len(text) <= 200 else text[:197] + "..."


def _redacted_message(line):
    """The expected one-line message: the key redacted first, then the cut."""
    return _cut_200(line.replace(GEMINI_KEY, REDACTED))


def _assert_no_key_part(text):
    assert GEMINI_KEY_PREFIX not in text, text


def _straddling_connection_error():
    return _litellm().APIConnectionError(
        message="geminiException - " + _straddling_line(), llm_provider="gemini", model="m"
    )


def _straddling_quota_error():
    """A 429 whose first line carries the key across the cut; the retry delay (37214 s) is on line 2."""
    return _litellm().RateLimitError(
        message="geminiException - " + _straddling_line("Quota exceeded for host") + '\n"retryDelay": "37214s"',
        llm_provider="gemini",
        model="m",
    )


STRADDLING_CONNECTION_ONE_LINE = (
    "APIConnectionError (HTTP 500, provider gemini, model m): " + _redacted_message(_straddling_line())
)
STRADDLING_QUOTA_ONE_LINE = (
    "RateLimitError (HTTP 429, provider gemini, model m): "
    + _redacted_message(_straddling_line("Quota exceeded for host"))
    + "; retry after 37214 s"
)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(lambda line: line, id="plain-text"),
        pytest.param(
            lambda line: json.dumps({"error": {"code": 400, "message": line, "status": "INVALID_ARGUMENT"}}),
            id="json-error-message",
        ),
    ],
)
def test_provider_error_text_redacts_before_cutting(body):
    line = _straddling_line()
    exc = _litellm().APIConnectionError(message="geminiException - " + body(line), llm_provider="gemini", model="m")
    assert GEMINI_KEY in str(exc)
    text = _pe().provider_error_text(exc, redact_text=_known_key_redactor)
    assert text == "APIConnectionError (HTTP 500, provider gemini, model m): " + _redacted_message(line)
    assert REDACTED in text
    _assert_no_key_part(text)


def test_provider_error_text_redacts_a_quota_error_before_cutting():
    text = _pe().provider_error_text(_straddling_quota_error(), redact_text=_known_key_redactor)
    assert text == STRADDLING_QUOTA_ONE_LINE
    _assert_no_key_part(text)


@pytest.mark.asyncio
async def test_judge_stop_message_and_warning_use_the_runs_redactor(mock_llm):
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    judge = LiteLLMJudge(PRIMARY, FAST, {}, {}, redact=_known_key_redactor)
    mock_llm.queue_error(_straddling_quota_error())
    with pytest.raises(_litellm().RateLimitError):
        await judge.evaluate(_transcript(), "Did the agent give the deletion route?", "primary")
    with pytest.raises(_judge_stopped()) as stopped:
        await judge.evaluate(_transcript(), "Was the agent polite?", "primary")
    assert str(stopped.value) == STOPPED.format(model=PRIMARY, error=STRADDLING_QUOTA_ONE_LINE)
    _assert_no_key_part(str(stopped.value))
    assert judge.get_warnings() == [JUDGE_STOP_WARNING.format(model=PRIMARY, error=STRADDLING_QUOTA_ONE_LINE)]
    _assert_no_key_part(" ".join(judge.get_warnings()))
    assert mock_llm.call_count == 1


def _two_scenario_project(project):
    metrics = ["UNIV-001", "UNIV-009", "UNIV-013"]   # primary, primary, fast
    project.write_config(
        [
            project.scenario("first_call", max_turns=1, metrics=metrics),
            project.scenario("second_call", max_turns=1, metrics=metrics),
        ]
    )
    return project.load()


def _assert_run_has_no_key_part(result):
    from impulse_evals.cli import fail_on_reasons

    _assert_no_key_part(result.model_dump_json())
    for level in ("critical", "high", "any"):
        for reason in fail_on_reasons(result, level):
            _assert_no_key_part(reason)


@pytest.mark.asyncio
async def test_run_errors_and_evaluation_warnings_leave_no_key_prefix(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    """Scenario error, evaluation-failed reasons and the per-check run warnings (no stop: HTTP 500)."""
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)   # a key variable the run's redactor knows
    cfg = _two_scenario_project(impulse_project)
    _primary_judge_raises(llm_router, _straddling_connection_error())
    result = await _run(cfg, adapter_factory=fake_adapter)

    one = STRADDLING_CONNECTION_ONE_LINE
    for scenario_result in result.scenario_results:
        assert scenario_result.error == "evaluation of UNIV-001 failed: " + one
        assert _metrics(scenario_result)["UNIV-009"].reason == "evaluation failed: " + one
    assert "first_call: evaluation of UNIV-009 failed: " + one in result.warnings
    assert "second_call: evaluation of UNIV-009 failed: " + one in result.warnings
    _assert_run_has_no_key_part(result)
    _all_one_line(result)


@pytest.mark.asyncio
async def test_run_judge_stop_message_and_warning_leave_no_key_prefix(
    impulse_project, llm_router, fake_adapter, monkeypatch
):
    """The quota stop: the first error, the JudgeStopped reasons and errors, and the stop warning."""
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    cfg = _two_scenario_project(impulse_project)
    primary_calls = _primary_judge_raises(llm_router, _straddling_quota_error())
    result = await _run(cfg, adapter_factory=fake_adapter)

    assert len(primary_calls) == 1
    one = STRADDLING_QUOTA_ONE_LINE
    stopped = STOPPED.format(model=PRIMARY, error=one)
    first, second = result.scenario_results
    assert first.error == "evaluation of UNIV-001 failed: " + one
    assert _metrics(first)["UNIV-009"].reason == "evaluation failed: " + stopped
    assert second.error == "evaluation of UNIV-001 failed: " + stopped
    assert result.warnings.count(JUDGE_STOP_WARNING.format(model=PRIMARY, error=one)) == 1
    _assert_run_has_no_key_part(result)
    _all_one_line(result)


# ============================================================================================
# A1/A7-rev R1-L2: no provider call after the stop, including the "Return JSON only" retry
# ============================================================================================


@pytest.mark.asyncio
async def test_no_provider_call_after_the_stop_even_for_the_json_retry(monkeypatch):
    """Two concurrent checks on one model: check A's first answer (not JSON) arrives after check B's
    long-delay 429 stopped the model, so A's JSON retry must not reach the provider."""
    import asyncio

    import litellm
    from conftest import MockLLM, make_llm_response

    quota = _campaign_exc()
    provider_calls = []
    b_failed = asyncio.Event()

    async def acompletion(*args, **kwargs):
        provider_calls.append(kwargs.get("messages"))
        number = len(provider_calls)
        if number == 1:   # check A: answers late, and not in JSON
            await b_failed.wait()
            return make_llm_response("I think the agent did fine.")
        if number == 2:   # check B: the campaign 429 (retry delay 37214 s)
            asyncio.get_running_loop().call_soon(b_failed.set)
            raise quota
        return make_llm_response(json.dumps(MockLLM.verdict("pass")))

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    judge = _judge()
    check_a, check_b = await asyncio.gather(
        judge.evaluate(_transcript(), "Did the agent give the deletion route?", "primary"),
        judge.evaluate(_transcript(), "Was the agent polite?", "primary"),
        return_exceptions=True,
    )
    assert check_b is quota
    assert len(provider_calls) == 2, "the JSON retry of check A was sent after the model was stopped"
    assert isinstance(check_a, _judge_stopped()), check_a
    assert str(check_a) == STOPPED.format(model=PRIMARY, error=CAMPAIGN_ONE_LINE)


# ============================================================================================
# A1/A7-rev R1-L4: OpenAI-shaped LiteLLM errors; "try again in" delays
# ============================================================================================

OPENAI_TPM = "Rate limit reached for gpt in organization org-x on tokens per min (TPM): Limit 30000, Used 29000."


@pytest.mark.parametrize(
    "provider, message, seconds, shown",
    [
        pytest.param("openai", f"{OPENAI_TPM} Please try again in 6m30s.", 390.0, 390, id="6m30s"),
        pytest.param("groq", f"{OPENAI_TPM} Please try again in 7h12m.", 25920.0, 25920, id="7h12m"),
        pytest.param("openai", f"{OPENAI_TPM} Please try again in 1.5s.", 1.5, 2, id="1.5s"),
        pytest.param("openai", f"{OPENAI_TPM} Please try again in 20.5 s.", 20.5, 21, id="20.5-space-s"),
    ],
)
def test_openai_shaped_rate_limit_error(provider, message, seconds, shown):
    word = "OpenAI" if provider == "openai" else provider.capitalize()
    # LiteLLM's exception mapping for OpenAI-compatible providers: message=f"RateLimitError: {provider} - {message}"
    exc = _litellm().RateLimitError(
        message=f"RateLimitError: {word}Exception - {message}", llm_provider=provider, model="some-model"
    )
    assert str(exc).startswith("litellm.RateLimitError: RateLimitError: ")
    assert _pe().retry_after_seconds(exc) == pytest.approx(seconds)
    assert _pe().provider_error_text(exc) == (
        f"RateLimitError (HTTP 429, provider {provider}, model some-model): {message}; retry after {shown} s"
    )


def test_openai_shaped_error_without_delay_drops_the_doubled_class_name():
    exc = _litellm().BadRequestError(
        message="BadRequestError: OpenAIException - Invalid value for 'messages'.", model="some-model", llm_provider="openai"
    )
    assert str(exc).startswith("litellm.BadRequestError: BadRequestError: ")
    assert _pe().provider_error_text(exc) == (
        "BadRequestError (HTTP 400, provider openai, model some-model): Invalid value for 'messages'."
    )


@pytest.mark.asyncio
async def test_openai_try_again_delay_stops_the_judge(mock_llm):
    judge = _judge()
    message = f"{OPENAI_TPM} Please try again in 6m30s."
    mock_llm.queue_error(
        _litellm().RateLimitError(
            message=f"RateLimitError: OpenAIException - {message}", llm_provider="openai", model="some-model"
        )
    )
    with pytest.raises(_litellm().RateLimitError):
        await judge.evaluate(_transcript(), "criterion", "primary")
    with pytest.raises(_judge_stopped()) as stopped:
        await judge.evaluate(_transcript(), "criterion", "primary")
    one_line = f"RateLimitError (HTTP 429, provider openai, model some-model): {message}; retry after 390 s"
    assert str(stopped.value) == STOPPED.format(model=PRIMARY, error=one_line)
    assert mock_llm.call_count == 1


# ============================================================================================
# A1/A7-rev R1-N: __all__; the stop is keyed by (model, custom_llm_provider)
# ============================================================================================


def test_public_helpers_are_in_all():
    names = _pe().__all__
    for name in ("provider_error_text", "retry_after_seconds", "one_line_error", "is_litellm_error"):
        assert name in names
        assert callable(getattr(_pe(), name))


@pytest.mark.asyncio
async def test_stop_is_keyed_by_model_and_custom_llm_provider(mock_llm):
    """Two roles with the same model string but different providers do not share the stop (as the
    JSON-mode cache already does)."""
    from impulse_evals.judge.litellm_judge import LiteLLMJudge

    shared = "judge-shared"
    judge = LiteLLMJudge(shared, shared, {"custom_llm_provider": "ollama"}, {"custom_llm_provider": "openai"})
    mock_llm.queue_error(_campaign_exc())
    with pytest.raises(_litellm().RateLimitError):
        await judge.evaluate(_transcript(), "criterion", "primary")

    mock_llm.queue_verdict("pass")
    verdict = await judge.evaluate(_transcript(), "criterion", "fast")
    assert verdict.result == "pass", "the other provider's role is still called"
    assert mock_llm.call_count == 2
    assert mock_llm.calls[1]["kwargs"].get("custom_llm_provider") == "openai"

    with pytest.raises(_judge_stopped()):
        await judge.evaluate(_transcript(), "criterion", "primary")
    assert mock_llm.call_count == 2
