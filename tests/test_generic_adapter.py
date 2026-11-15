# SPDX-License-Identifier: Apache-2.0
"""Phase 4: adapters/base.py and the generic adapter (SPEC.md sections 8.2 and 9).

Test-facing names used here beyond the SPEC.md section 18.1 table (contracts logged in DECISIONS.md,
Stage B group 4-8): ``impulse_evals.adapters.base.AgentInterface`` and ``AgentReply`` (names and
fields fixed by SPEC.md sections 6 and 9), and
``impulse_evals.adapters.generic_adapter.GenericAdapter(config)`` (takes the ``ProjectConfig``).

[phase 11] voice mode without returned audio needs the runner, the voice pipeline, and
metrics/voice.py; that test carries the ``phase11`` marker (phase 4 acceptance:
``pytest tests/test_generic_adapter.py -m "not phase11"``).
"""

import json
import sys
from pathlib import Path

import pytest

SAMPLE_TOOL_CALL = {
    "name": "book_slot",
    "arguments": {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"},
    "result": {"success": True},
    "success": True,
}

ECHO_AGENT = """
import json


def respond(user_text, history):
    return json.dumps({"user_text": user_text, "history": history})
"""

ASYNC_ECHO_AGENT = """
import asyncio
import json


async def respond(user_text, history):
    await asyncio.sleep(0)
    return json.dumps({"user_text": user_text, "history": history, "async": True})
"""


def _base():
    from impulse_evals.adapters import base

    return base


def _m():
    from impulse_evals import models

    return models


def _generic_adapter(cfg):
    from impulse_evals.adapters.generic_adapter import GenericAdapter

    return GenericAdapter(cfg)


async def _start(project, code=None, scenarios=None, **config_kwargs):
    """Write an agent and a config, load it, build the generic adapter, and start the first scenario."""
    if code is not None or project.entrypoint is None:
        project.write_agent(code)
    project.write_config(scenarios or [project.scenario("generic_probe")], **config_kwargs)
    cfg = project.load()
    adapter = _generic_adapter(cfg)
    await adapter.start(cfg.scenarios[0])
    return adapter, cfg


# ============================================================================================
# adapters/base.py
# ============================================================================================


def test_agent_reply_fields_and_defaults():
    base = _base()
    assert set(base.AgentReply.model_fields) == {
        "text", "audio_bytes", "latency_ms", "heard_text", "interrupted", "tool_calls",
    }
    reply = base.AgentReply(text="Hello", latency_ms=12.5)
    assert reply.text == "Hello"
    assert reply.latency_ms == 12.5
    assert reply.audio_bytes is None
    assert reply.heard_text is None
    assert reply.interrupted is False
    assert reply.tool_calls == []


def test_agent_reply_requires_latency():
    with pytest.raises(ValueError):
        _base().AgentReply(text="Hello")


def test_agent_reply_tool_calls_are_tool_call_models():
    m = _m()
    reply = _base().AgentReply(text="Booked", latency_ms=1.0, tool_calls=[SAMPLE_TOOL_CALL])
    assert reply.tool_calls == [m.ToolCall(**SAMPLE_TOOL_CALL)]
    assert isinstance(reply.tool_calls[0], m.ToolCall)


def test_agent_interface_default_hooks():
    base = _base()

    class MinimalAgent(base.AgentInterface):
        async def start(self, scenario):
            return None

        async def send(self, user_text, user_audio):
            return base.AgentReply(text="", latency_ms=0.0)

        async def close(self):
            return None

    agent = MinimalAgent()
    assert agent.get_system_prompt() is None
    assert agent.get_warnings() == []
    assert agent.get_metadata() == {}


# ============================================================================================
# Generic adapter: function signature, sync and async
# ============================================================================================


@pytest.mark.asyncio
async def test_sync_function_returning_a_string(impulse_project):
    code = """
    def respond(user_text, history):
        return "Echo: " + user_text
    """
    adapter, _ = await _start(impulse_project, code)
    reply = await adapter.send("Hi there", None)
    assert isinstance(reply, _base().AgentReply)
    assert reply.text == "Echo: Hi there"
    assert reply.tool_calls == []
    assert reply.audio_bytes is None
    assert reply.heard_text is None
    assert reply.interrupted is False
    assert isinstance(reply.latency_ms, float)
    assert reply.latency_ms >= 0.0
    await adapter.close()


@pytest.mark.asyncio
async def test_async_function_is_awaited(impulse_project):
    adapter, _ = await _start(impulse_project, ASYNC_ECHO_AGENT)
    reply = await adapter.send("Hi", None)
    assert json.loads(reply.text) == {"user_text": "Hi", "history": [], "async": True}
    await adapter.close()


@pytest.mark.asyncio
async def test_latency_is_measured_around_the_call(impulse_project):
    code = """
    import time


    def respond(user_text, history):
        time.sleep(0.05)
        return "done"
    """
    adapter, _ = await _start(impulse_project, code)
    reply = await adapter.send("Hi", None)
    assert 40.0 <= reply.latency_ms < 10_000.0
    await adapter.close()


# ============================================================================================
# history format, outbound opener, conversation reset
# ============================================================================================


@pytest.mark.asyncio
async def test_history_holds_prior_turns_only(impulse_project):
    adapter, _ = await _start(impulse_project, ECHO_AGENT)
    first = await adapter.send("first message", None)
    second = await adapter.send("second message", None)
    third = await adapter.send("third message", None)
    assert json.loads(first.text) == {"user_text": "first message", "history": []}
    assert json.loads(second.text) == {
        "user_text": "second message",
        "history": [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": first.text},
        ],
    }
    assert json.loads(third.text)["history"] == [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": first.text},
        {"role": "user", "content": "second message"},
        {"role": "assistant", "content": second.text},
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_outbound_opener_calls_function_with_empty_string(impulse_project):
    scenario = impulse_project.scenario("outbound_probe", direction="outbound", opening_user_message=None)
    adapter, _ = await _start(impulse_project, ECHO_AGENT, scenarios=[scenario])
    opener = await adapter.send(None, None)
    assert json.loads(opener.text) == {"user_text": "", "history": []}
    reply = await adapter.send("Who is calling?", None)
    assert json.loads(reply.text) == {
        "user_text": "Who is calling?",
        "history": [{"role": "assistant", "content": opener.text}],
    }
    await adapter.close()


@pytest.mark.asyncio
async def test_start_begins_a_new_conversation(impulse_project):
    project = impulse_project
    scenarios = [project.scenario("first_call"), project.scenario("second_call")]
    adapter, cfg = await _start(project, ECHO_AGENT, scenarios=scenarios)
    await adapter.send("from the first call", None)
    await adapter.close()
    await adapter.start(cfg.scenarios[1])
    reply = await adapter.send("from the second call", None)
    assert json.loads(reply.text) == {"user_text": "from the second call", "history": []}
    await adapter.close()


# ============================================================================================
# dict returns: text, heard_text, audio, tool_calls
# ============================================================================================


@pytest.mark.asyncio
async def test_dict_reply_tool_calls_round_trip_into_tool_call(impulse_project):
    m = _m()
    code = """
    def respond(user_text, history):
        return {
            "text": "Your slot is booked.",
            "tool_calls": [
                {
                    "name": "book_slot",
                    "arguments": {"date": "2026-01-01", "phone": "0000000000"},
                    "result": {"success": True, "booking_id": "B-1"},
                    "success": True,
                },
                {"name": "send_sms"},
                {"name": "charge_card", "arguments": {"amount": 100}, "success": False, "error": "card declined"},
            ],
        }
    """
    adapter, _ = await _start(impulse_project, code)
    reply = await adapter.send("Book me in", None)
    assert reply.text == "Your slot is booked."
    assert reply.tool_calls == [
        m.ToolCall(
            name="book_slot",
            arguments={"date": "2026-01-01", "phone": "0000000000"},
            result={"success": True, "booking_id": "B-1"},
            success=True,
        ),
        m.ToolCall(name="send_sms", arguments={}),
        m.ToolCall(name="charge_card", arguments={"amount": 100}, success=False, error="card declined"),
    ]
    assert all(isinstance(call, m.ToolCall) for call in reply.tool_calls)
    assert reply.tool_calls[1].arguments == {}
    assert reply.tool_calls[1].success is None
    await adapter.close()


@pytest.mark.asyncio
async def test_dict_reply_heard_text_and_audio(impulse_project):
    code = """
    def respond(user_text, history):
        return {"text": "Sure.", "heard_text": "i want a slot on tuesday", "audio": b"RIFF-fake-agent-audio"}
    """
    adapter, _ = await _start(impulse_project, code)
    reply = await adapter.send("I want a slot on Tuesday", None)
    assert reply.text == "Sure."
    assert reply.heard_text == "i want a slot on tuesday"
    assert reply.audio_bytes == b"RIFF-fake-agent-audio"
    assert reply.tool_calls == []
    await adapter.close()


@pytest.mark.asyncio
async def test_dict_reply_without_text_gives_empty_text(impulse_project):
    code = """
    def respond(user_text, history):
        return {"tool_calls": [{"name": "send_sms", "arguments": {"phone": "0000000000"}}]}
    """
    adapter, _ = await _start(impulse_project, code)
    reply = await adapter.send("Text me the details", None)
    assert reply.text == ""
    assert [call.name for call in reply.tool_calls] == ["send_sms"]
    await adapter.close()


@pytest.mark.asyncio
async def test_voice_mode_function_still_receives_the_text(impulse_project):
    # The generic function has no audio parameter: with user audio, it still gets the simulator's text.
    adapter, _ = await _start(impulse_project, ECHO_AGENT)
    reply = await adapter.send("Hello in voice mode", b"RIFF-fake-user-audio")
    assert json.loads(reply.text) == {"user_text": "Hello in voice mode", "history": []}
    assert reply.audio_bytes is None
    await adapter.close()


@pytest.mark.asyncio
async def test_sample_agent_book_tool_call_through_the_adapter(impulse_project):
    m = _m()
    adapter, _ = await _start(impulse_project)  # a copy of examples/sample_agent.py
    reply = await adapter.send("Please book me for Tuesday", None)
    assert "Please book me for Tuesday" in reply.text
    assert reply.tool_calls == [m.ToolCall(**SAMPLE_TOOL_CALL)]
    plain = await adapter.send("Thanks", None)
    assert plain.tool_calls == []
    await adapter.close()


@pytest.mark.asyncio
async def test_generic_adapter_has_no_warnings_or_metadata(impulse_project):
    adapter, _ = await _start(impulse_project)
    await adapter.send("Hello", None)
    await adapter.close()
    assert adapter.get_warnings() == []
    assert adapter.get_metadata() == {}


# ============================================================================================
# System prompt: SYSTEM_PROMPT / get_system_prompt() extraction; config.yaml wins
# ============================================================================================


@pytest.mark.asyncio
async def test_module_system_prompt_constant_is_extracted(impulse_project):
    code = """
    SYSTEM_PROMPT = "You are the module-level prompt of the probe agent."


    def respond(user_text, history):
        return "ok"
    """
    adapter, _ = await _start(impulse_project, code, system_prompt=None)
    assert adapter.get_system_prompt() == "You are the module-level prompt of the probe agent."
    await adapter.close()


@pytest.mark.asyncio
async def test_get_system_prompt_function_is_extracted(impulse_project):
    code = """
    def get_system_prompt():
        return "You are the prompt returned by get_system_prompt()."


    def respond(user_text, history):
        return "ok"
    """
    adapter, _ = await _start(impulse_project, code, system_prompt=None)
    assert adapter.get_system_prompt() == "You are the prompt returned by get_system_prompt()."
    await adapter.close()


@pytest.mark.asyncio
async def test_no_system_prompt_anywhere_gives_none(impulse_project):
    code = """
    def respond(user_text, history):
        return "ok"
    """
    adapter, _ = await _start(impulse_project, code, system_prompt=None)
    assert adapter.get_system_prompt() is None
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exposed", ["constant", "function"])
async def test_config_system_prompt_wins(impulse_project, exposed):
    if exposed == "constant":
        code = """
        SYSTEM_PROMPT = "Prompt exposed by the agent module."


        def respond(user_text, history):
            return "ok"
        """
    else:
        code = """
        def get_system_prompt():
            return "Prompt exposed by the agent module."


        def respond(user_text, history):
            return "ok"
        """
    config_prompt = "Prompt from the system_prompt file named in config.yaml."
    adapter, _ = await _start(impulse_project, code, system_prompt=config_prompt)
    assert adapter.get_system_prompt() == config_prompt
    await adapter.close()


# ============================================================================================
# Entrypoint import: at adapter start, from the config file's folder; failure is a config error
# ============================================================================================


@pytest.mark.asyncio
async def test_entrypoint_imported_at_start_from_config_folder(impulse_project):
    project = impulse_project
    entrypoint = project.write_agent()
    module_name = entrypoint.split(":", 1)[0]
    project.write_config([project.scenario("import_probe")])
    cfg = project.load()
    assert module_name not in sys.modules, "load_config must not import the generic entrypoint"
    adapter = _generic_adapter(cfg)
    await adapter.start(cfg.scenarios[0])
    entries = [Path(entry).resolve() for entry in sys.path if entry]
    assert cfg.config_dir in entries, "the config file's folder must be on sys.path at adapter start"
    reply = await adapter.send("Hello", None)
    assert reply.text == "You said: Hello"
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing_module", "missing_function", "no_colon"])
async def test_entrypoint_import_failure_is_config_error(impulse_project, kind):
    from impulse_evals.config import ConfigError

    project = impulse_project
    real = project.write_agent()
    module_name = real.split(":", 1)[0]
    entrypoint = {
        "missing_module": "impulse_test_no_such_module_4f2a:respond",
        "missing_function": f"{module_name}:no_such_function",
        "no_colon": module_name,
    }[kind]
    project.write_config([project.scenario("bad_entrypoint")], agent={"entrypoint": entrypoint})
    cfg = project.load()  # load_config never imports the entrypoint
    with pytest.raises(ConfigError) as info:
        adapter = _generic_adapter(cfg)
        await adapter.start(cfg.scenarios[0])
    assert info.value.field == "agent.entrypoint"


# ============================================================================================
# [phase 11] Voice mode: no returned audio skips audio-only metrics
# ============================================================================================


@pytest.mark.phase11
@pytest.mark.asyncio
async def test_voice_mode_without_agent_audio_skips_audio_only_metrics(
    impulse_project, llm_router, mock_voice_server
):
    from impulse_evals.runner import run

    project = impulse_project
    project.write_agent()  # returns text only, never audio
    voice_config = {
        "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
        "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
        "save_audio": True,
        "save_recording": False,
        "timeout_s": 30,
    }
    project.write_config(
        [project.scenario("voice_no_audio", max_turns=1, metrics=["VOICE-015"])],
        voice_config=voice_config,
    )
    result = await run(project.load())
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    metrics = {metric.metric_id: metric for metric in scenario_result.metric_results}
    assert metrics["VOICE-015"].skipped is True
    assert metrics["VOICE-015"].label == "skipped"
    assert metrics["VOICE-015"].reason == "agent returned no audio"


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 32): a malformed tool call in a dict return is a clear
# ``TypeError("agent function {entrypoint} returned ...")``, like every other malformed return,
# never a raw pydantic ``ValidationError`` dump. The SPEC.md section 8.2 dict contract is unchanged:
# an OpenAI-style call (``function.name`` / ``arguments`` as a JSON string) is not parsed.
# ============================================================================================

TOOL_CALL_NAME_ERROR = "agent function {entrypoint} returned a tool call without a string 'name'"
TOOL_CALL_ARGUMENTS_ERROR = "agent function {entrypoint} returned a tool call whose 'arguments' is not a dict"
TOOL_CALL_INVALID_ERROR = "agent function {entrypoint} returned an invalid tool call: {location}: {message}"

MALFORMED_TOOL_CALLS = [
    pytest.param(
        {"id": "call_1", "type": "function", "function": {"name": "book_slot", "arguments": '{"date": "2026-01-01"}'}},
        TOOL_CALL_NAME_ERROR,
        id="openai-style",
    ),
    pytest.param({"arguments": {"date": "2026-01-01"}}, TOOL_CALL_NAME_ERROR, id="no-name"),
    pytest.param({"name": 42, "arguments": {}}, TOOL_CALL_NAME_ERROR, id="non-string-name"),
    pytest.param({"name": "", "arguments": {}}, TOOL_CALL_NAME_ERROR, id="empty-name"),
    pytest.param({"name": "book_slot", "arguments": '{"date": "2026-01-01"}'}, TOOL_CALL_ARGUMENTS_ERROR, id="json-string-arguments"),
    pytest.param({"name": "book_slot", "arguments": ["2026-01-01"]}, TOOL_CALL_ARGUMENTS_ERROR, id="list-arguments"),
    pytest.param(
        {"name": "book_slot", "arguments": {}, "error": 123},
        TOOL_CALL_INVALID_ERROR.format(entrypoint="{entrypoint}", location="error", message="Input should be a valid string"),
        id="non-string-error",
    ),
]


def _tool_call_agent(call):
    return f"""
    def respond(user_text, history):
        return {{"text": "Booked.", "tool_calls": [{call!r}]}}
    """


@pytest.mark.asyncio
@pytest.mark.parametrize("call, expected", MALFORMED_TOOL_CALLS)
async def test_malformed_tool_call_is_a_clear_type_error(impulse_project, call, expected):
    adapter, cfg = await _start(impulse_project, _tool_call_agent(call))
    with pytest.raises(TypeError) as info:
        await adapter.send("Book me", None)
    message = str(info.value)
    assert message == expected.format(entrypoint=cfg.agent.entrypoint), message
    assert "errors.pydantic.dev" not in message
    assert "validation error" not in message
    await adapter.close()


@pytest.mark.asyncio
async def test_well_formed_tool_call_dicts_still_round_trip(impulse_project):
    # Regression guard: the valid shapes of SPEC.md section 8.2 are unchanged (missing arguments -> {},
    # a non-dict result wrapped as {"value": ...}, a ToolCall object accepted).
    m = _m()
    code = """
    from impulse_evals.models import ToolCall


    def respond(user_text, history):
        return {
            "text": "Booked.",
            "tool_calls": [
                {"name": "book_slot", "arguments": {"date": "2026-01-01"}, "success": False, "error": "slot taken"},
                {"name": "send_sms"},
                {"name": "lookup", "arguments": {}, "result": "queued"},
                ToolCall(name="notify", arguments={"channel": "sms"}, success=True),
            ],
        }
    """
    adapter, _ = await _start(impulse_project, code)
    reply = await adapter.send("Book me", None)
    assert reply.tool_calls == [
        m.ToolCall(name="book_slot", arguments={"date": "2026-01-01"}, success=False, error="slot taken"),
        m.ToolCall(name="send_sms", arguments={}),
        m.ToolCall(name="lookup", arguments={}, result={"value": "queued"}),
        m.ToolCall(name="notify", arguments={"channel": "sms"}, success=True),
    ]
    await adapter.close()
