# SPDX-License-Identifier: Apache-2.0
"""Fixes round 3, B3: custom adapters from config, ``agent.adapter: custom`` (FIXES-ROUND-3-SPEC.md B3).

* config: ``adapter: custom`` with ``entrypoint: "module:Class"`` (dots allowed after the colon);
  missing / blank entrypoint -> ``ConfigError("agent.entrypoint", "required when agent.adapter is
  custom")``; new ``AgentConfig.options: dict = {}`` passed through unchanged (built-in adapters ignore it).
* loading (by the runner's default factory, never by ``load_config``; the generic adapter's import
  rules and texts): every error is a ``ConfigError`` at ``agent.entrypoint``: "{entrypoint} is not of
  the form module:Class (e.g. my_adapter:MyAdapter)", "cannot import module {module} from {dir} or
  sys.path: {type}: {message}", "module {module} has no class {attr}", "{module}:{attr} is not a
  subclass of impulse_evals.adapters.base.AgentInterface", "could not create {module}:{attr}: {type}:
  {message}" (redacted, one line, 200 characters; a ``ConfigError`` from the constructor unchanged).
* construction ``Class(config)`` (the loaded ``ProjectConfig``), one instance per scenario; the runner
  treats it like any adapter (system prompt, warnings, metadata, ``AgentClosed``, ``AgentTurnError``,
  tool-call redaction and success inference, ``--fail-on``, ``compare``); ``--concurrency`` as given;
  ``RunResult.agent_entrypoint`` = the entrypoint string.
* ``impulse doctor``: "agent adapter" PASS / FAIL row, INFO "agent connection"; the class is imported,
  never instantiated. ``--dry-run``: the remote warning, no scenario run; the class is imported and
  checked (loading errors exit 2), never instantiated.
* example ``examples/custom_adapter/`` (``echo_adapter.py``, ``config.yaml``) dry-runs with exit 0.

Adapter modules are written into the project folder (``tmp_path``, the config's folder) at test time
with unique names. The judge and the simulator are ``mock_llm`` routed by ``llm_router``. No network.
"""

import importlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "custom_adapter"

SCENARIO = "custom_booking"
OPENING = "Hello, I would like to book a slot on Tuesday."
SECOND_USER = "Thank you, that is all for today."
CONSENT = "dpdp.consent_before_pii"
PURPOSE = "dpdp.purpose_disclosure"
AGENT_INTERFACE = "impulse_evals.adapters.base.AgentInterface"
ENTRYPOINT_FIELD = "agent.entrypoint"
REQUIRED = "required when agent.adapter is custom"
FORM_ERROR = "{entrypoint} is not of the form module:Class (e.g. my_adapter:MyAdapter)"
NO_CLASS = "module {module} has no class {attr}"
NOT_SUBCLASS = "{module}:{attr} is not a subclass of impulse_evals.adapters.base.AgentInterface"
COULD_NOT_CREATE = "could not create {module}:{attr}: {type}: "
DRY_RUN_WARNING = "Dry run: judge metrics and assertions were skipped; results are not verdicts."
REMOTE_WARNING = (
    "Dry run: scenarios run only with the generic adapter; the {adapter} config was validated without connecting."
)
PIPECAT_CONCURRENCY_MESSAGE = "Pipecat bots serve one client at a time; running scenarios one by one."
SECRET_ENV = "IMPULSE_TEST_CUSTOM_API_KEY"   # a *_API_KEY variable: the run's redactor knows its value
SECRET = "sk-impulse-test-SECRET-custom-7731abcdef"
ADAPTER_WARNING = "custom adapter warning: replies are echoes"
BOOKING_ARGUMENTS = {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}

# A working AgentInterface subclass (no network): echoes; books when the user says "book"; scripted
# AgentClosed ("hang up"), AgentTurnError ("fail") and plain errors ("crash"). Module-level lists
# record every instance and call so the tests can look inside after the run.
ECHO_ADAPTER = '''
import asyncio
import os

from impulse_evals.adapters.base import AgentClosed, AgentInterface, AgentReply, AgentTurnError
from impulse_evals.models import ToolCall

INSTANCES = []
EVENTS = []
ACTIVE = [0]
MAX_ACTIVE = [0]

BOOKING_ARGUMENTS = {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"}


class EchoAdapter(AgentInterface):
    def __init__(self, config):
        self.config = config
        self.options = dict(config.agent.options)
        self.scenario = None
        INSTANCES.append(self)

    async def start(self, scenario):
        self.scenario = scenario
        EVENTS.append(("start", scenario.name))

    async def send(self, user_text, user_audio):
        EVENTS.append(("send", self.scenario.name, user_text))
        ACTIVE[0] += 1
        MAX_ACTIVE[0] = max(MAX_ACTIVE[0], ACTIVE[0])
        try:
            await asyncio.sleep(float(self.options.get("delay_s", 0)))
        finally:
            ACTIVE[0] -= 1
        text = user_text or ""
        lowered = text.lower()
        if "hang up" in lowered:
            raise AgentClosed("the caller asked to hang up", reply=AgentReply(text="Goodbye.", latency_ms=5.0))
        if "fail" in lowered:
            raise AgentTurnError(
                "custom agent turn failed",
                reply=AgentReply(
                    text="Let me check that.",
                    latency_ms=7.0,
                    tool_calls=[
                        ToolCall(
                            name="lookup_slot",
                            arguments={"day": "Tuesday"},
                            result={"status": "failed", "error": "calendar offline"},
                        )
                    ],
                ),
            )
        if "crash" in lowered:
            raise RuntimeError("custom agent crashed while answering")
        calls = []
        if "book" in lowered:
            calls.append(
                ToolCall(
                    name="book_slot",
                    arguments=dict(BOOKING_ARGUMENTS),
                    result={
                        "ok": True,
                        "booking_id": "BK-1001",
                        "note": "token " + os.environ.get("IMPULSE_TEST_CUSTOM_API_KEY", "none"),
                    },
                )
            )
        reply_text = ("Echo: " + text) if text else "Hello from the custom adapter."
        return AgentReply(text=reply_text, latency_ms=12.0, heard_text=lowered or None, tool_calls=calls)

    async def close(self):
        EVENTS.append(("close", self.scenario.name if self.scenario is not None else None))

    def get_system_prompt(self):
        return self.options.get("prompt")

    def get_warnings(self):
        return ["custom adapter warning: replies are echoes"]

    def get_metadata(self):
        return {"custom_adapter": {"scenario": self.scenario.name if self.scenario is not None else None}}


class Holder:
    EchoAdapter = EchoAdapter
'''

# A subclass whose constructor always fails: imported (doctor, dry run) but never instantiated there.
EXPLODING_CONSTRUCTOR = '''
from impulse_evals.adapters.base import AgentInterface, AgentReply


class ExplodingAdapter(AgentInterface):
    def __init__(self, config):
        raise RuntimeError("ExplodingAdapter must never be instantiated here")

    async def start(self, scenario):
        return None

    async def send(self, user_text, user_audio):
        return AgentReply(text="never", latency_ms=1.0)

    async def close(self):
        return None
'''


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _write_module(project, code, stem=None):
    """Write an adapter module into the project folder (the config's folder); returns its name."""
    stem = stem or f"impulse_test_custom_{uuid.uuid4().hex[:10]}"
    (project.root / f"{stem}.py").write_text("# SPDX-License-Identifier: Apache-2.0\n" + code, encoding="utf-8")
    importlib.invalidate_caches()
    return stem


def _scenario(project, name=SCENARIO, **fields):
    values = {"opening_user_message": OPENING, "max_turns": 2, "metrics": ["UNIV-001", "UNIV-023"]}
    values.update(fields)
    return project.scenario(name, **values)


def _write_config(project, entrypoint, *, scenarios=None, options=None, extra_agent=None, **kwargs):
    agent = {"adapter": "custom"}
    if entrypoint is not None:
        agent["entrypoint"] = entrypoint
    if options is not None:
        agent["options"] = options
    agent.update(extra_agent or {})
    return project.write_config(scenarios or [_scenario(project)], agent=agent, **kwargs)


def _fail_if_adapter_rejected(exc):
    """Today ``adapter: custom`` is not an accepted literal: say so clearly instead of a raw error."""
    if getattr(exc, "field", None) == "agent.adapter":
        pytest.fail(f"config.yaml agent.adapter: custom is not accepted: {exc}", pytrace=False)


def _load(project, path=None):
    from impulse_evals.config import ConfigError

    try:
        return project.load(path)
    except ConfigError as exc:
        _fail_if_adapter_rejected(exc)
        raise


async def _run(cfg, **kwargs):
    from impulse_evals.runner import run

    return await run(cfg, **kwargs)


async def _run_config_error(project):
    """The ConfigError raised by ``runner.run`` (default adapter factory); ``load_config`` must succeed
    because it never imports the custom module."""
    from impulse_evals.config import ConfigError

    cfg = _load(project)
    with pytest.raises(ConfigError) as info:
        await _run(cfg)
    return info.value


async def _load_or_run_config_error(project):
    """The ConfigError of ``load_config`` when it raises one, else the one ``runner.run`` raises."""
    from impulse_evals.config import ConfigError

    try:
        cfg = project.load()
    except ConfigError as exc:
        _fail_if_adapter_rejected(exc)
        return exc
    with pytest.raises(ConfigError) as info:
        await _run(cfg)
    return info.value


def _module(stem):
    return sys.modules[stem]


def _rows(result, status, check):
    """Doctor table rows with ``status`` and the check label ``check``: their detail texts."""
    details = []
    for line in result.output.splitlines():
        line = line.strip()
        if not line.startswith(status):
            continue
        rest = line[len(status):].strip()
        if rest == check or rest.startswith(check + " "):
            details.append(rest[len(check):].strip())
    return details


def _new_run_dir(project, before):
    (run_dir,) = set(project.run_dirs()) - set(before)
    return run_dir


# =============================================================================================
# config.yaml: adapter custom, entrypoint, options
# =============================================================================================


def test_custom_adapter_is_accepted_and_options_default_to_empty(impulse_project):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    _write_config(project, f"{stem}:EchoAdapter")
    cfg = _load(project)
    assert cfg.agent.adapter == "custom"
    assert cfg.agent.entrypoint == f"{stem}:EchoAdapter"
    assert cfg.agent.options == {}


def test_options_are_kept_unchanged(impulse_project):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    options = {"assistant_id": "asst_42", "key_env": "MY_ASSISTANT_KEY", "nested": {"a": [1, 2.5, None, True]}}
    _write_config(project, f"{stem}:EchoAdapter", options=options)
    assert _load(project).agent.options == options


def test_builtin_adapters_accept_and_ignore_options(impulse_project):
    project = impulse_project
    project.write_agent()   # the generic sample agent
    project.write_config(
        [_scenario(project, max_turns=1, metrics=["UNIV-001"])],
        agent={"options": {"assistant_id": "asst_42"}},
    )
    cfg = project.load()
    assert cfg.agent.adapter == "generic"
    assert getattr(cfg.agent, "options", None) == {"assistant_id": "asst_42"}, "AgentConfig.options is missing"


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", [None, "", "   "], ids=["missing", "empty", "blank"])
async def test_custom_entrypoint_is_required(impulse_project, llm_router, entrypoint):
    project = impulse_project
    _write_config(project, entrypoint)
    error = await _load_or_run_config_error(project)
    assert error.field == ENTRYPOINT_FIELD
    assert error.message == REQUIRED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entrypoint",
    ["echo_adapter", "echo_adapter:", ":EchoAdapter", "echo_adapter.EchoAdapter"],
    ids=["no-colon", "no-class", "no-module", "dotted-no-colon"],
)
async def test_custom_entrypoint_must_be_module_colon_class(impulse_project, llm_router, entrypoint):
    project = impulse_project
    _write_config(project, entrypoint)
    error = await _load_or_run_config_error(project)
    assert error.field == ENTRYPOINT_FIELD
    assert error.message == FORM_ERROR.format(entrypoint=entrypoint)


def test_load_config_never_imports_the_custom_module(impulse_project):
    project = impulse_project
    stem = _write_module(project, 'raise RuntimeError("imported by load_config")\n')
    _write_config(project, f"{stem}:EchoAdapter")
    cfg = _load(project)   # no ConfigError: the module is imported only when the adapter is created
    assert stem not in sys.modules
    assert cfg.agent.entrypoint == f"{stem}:EchoAdapter"


# =============================================================================================
# Loading errors (runner's default factory)
# =============================================================================================


@pytest.mark.asyncio
async def test_missing_module_is_the_generic_import_error(impulse_project, llm_router):
    project = impulse_project
    stem = f"impulse_test_custom_missing_{uuid.uuid4().hex[:8]}"
    _write_config(project, f"{stem}:EchoAdapter")
    cfg = _load(project)
    error = await _run_config_error(project)
    assert error.field == ENTRYPOINT_FIELD
    prefix = f"cannot import module {stem} from {cfg.config_dir} or sys.path: ModuleNotFoundError: "
    assert error.message.startswith(prefix), error.message
    assert llm_router.calls("judge") == []


@pytest.mark.asyncio
async def test_module_failing_at_import_is_the_generic_import_error(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, 'raise RuntimeError("adapter module failed at import")\n')
    _write_config(project, f"{stem}:EchoAdapter")
    cfg = _load(project)
    error = await _run_config_error(project)
    assert error.field == ENTRYPOINT_FIELD
    assert error.message == (
        f"cannot import module {stem} from {cfg.config_dir} or sys.path: RuntimeError: adapter module failed at import"
    )


@pytest.mark.asyncio
async def test_missing_attribute(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    _write_config(project, f"{stem}:MissingAdapter")
    error = await _run_config_error(project)
    assert error.field == ENTRYPOINT_FIELD
    assert error.message == NO_CLASS.format(module=stem, attr="MissingAdapter")


NOT_ADAPTERS = '''
from impulse_evals.adapters.base import AgentReply


def make_adapter(config):
    return None


class NotAnAdapter:
    def __init__(self, config):
        self.config = config

    async def start(self, scenario):
        return None

    async def send(self, user_text, user_audio):
        return AgentReply(text="hi", latency_ms=1.0)

    async def close(self):
        return None


ADAPTER = 42
'''


@pytest.mark.asyncio
@pytest.mark.parametrize("attr", ["make_adapter", "NotAnAdapter", "ADAPTER"], ids=["function", "plain-class", "constant"])
async def test_not_a_subclass_of_agent_interface(impulse_project, llm_router, attr):
    project = impulse_project
    stem = _write_module(project, NOT_ADAPTERS)
    _write_config(project, f"{stem}:{attr}")
    error = await _run_config_error(project)
    assert error.field == ENTRYPOINT_FIELD
    assert error.message == NOT_SUBCLASS.format(module=stem, attr=attr)


@pytest.mark.asyncio
async def test_constructor_error_is_could_not_create_with_the_key_redacted(impulse_project, llm_router, monkeypatch):
    monkeypatch.setenv(SECRET_ENV, SECRET)
    project = impulse_project
    code = f'''
import os

from impulse_evals.adapters.base import AgentInterface


class BrokenAdapter(AgentInterface):
    def __init__(self, config):
        key = os.environ["{SECRET_ENV}"]
        raise RuntimeError("cannot reach the assistant with key " + key + "\\nsecond line of the error " + "x" * 300)

    async def start(self, scenario):
        return None

    async def send(self, user_text, user_audio):
        return None

    async def close(self):
        return None
'''
    stem = _write_module(project, code)
    _write_config(project, f"{stem}:BrokenAdapter")
    error = await _run_config_error(project)
    assert error.field == ENTRYPOINT_FIELD
    prefix = COULD_NOT_CREATE.format(module=stem, attr="BrokenAdapter", type="RuntimeError")
    message = error.message
    assert message.startswith(prefix + "cannot reach the assistant with key "), message
    assert SECRET not in message and SECRET not in str(error)
    assert "[REDACTED]" in message, message
    assert "\n" not in message and "\r" not in message                        # one line
    assert "second line of the error" in message                            # ... the lines joined
    assert len(message) <= len(prefix) + 200, (len(message), message)          # cut to 200 characters


@pytest.mark.asyncio
async def test_constructor_error_is_redacted_before_it_is_cut(impulse_project, llm_router, monkeypatch):
    """A key straddling the 200-character cut leaves no prefix behind (redact first, then cut)."""
    monkeypatch.setenv(SECRET_ENV, SECRET)
    project = impulse_project
    code = f'''
import os

from impulse_evals.adapters.base import AgentInterface


class BrokenAdapter(AgentInterface):
    def __init__(self, config):
        raise RuntimeError("y" * 175 + os.environ["{SECRET_ENV}"] + "z" * 100)

    async def start(self, scenario):
        return None

    async def send(self, user_text, user_audio):
        return None

    async def close(self):
        return None
'''
    stem = _write_module(project, code)
    _write_config(project, f"{stem}:BrokenAdapter")
    error = await _run_config_error(project)
    assert error.message.startswith(COULD_NOT_CREATE.format(module=stem, attr="BrokenAdapter", type="RuntimeError"))
    assert SECRET[:10] not in error.message, error.message


@pytest.mark.asyncio
async def test_config_error_from_the_constructor_is_reraised_unchanged(impulse_project, llm_router):
    project = impulse_project
    code = '''
from impulse_evals.adapters.base import AgentInterface
from impulse_evals.config import ConfigError


class StrictAdapter(AgentInterface):
    def __init__(self, config):
        raise ConfigError("agent.options.assistant_id", "required by the in-house adapter")

    async def start(self, scenario):
        return None

    async def send(self, user_text, user_audio):
        return None

    async def close(self):
        return None
'''
    stem = _write_module(project, code)
    _write_config(project, f"{stem}:StrictAdapter")
    error = await _run_config_error(project)
    assert error.field == "agent.options.assistant_id"
    assert error.message == "required by the in-house adapter"


@pytest.mark.asyncio
async def test_abstract_class_is_could_not_create_type_error(impulse_project, llm_router):
    project = impulse_project
    code = '''
from impulse_evals.adapters.base import AgentInterface


class HalfAdapter(AgentInterface):
    """No send() and no close(): abstract, so creating it raises TypeError."""

    def __init__(self, config):
        self.config = config

    async def start(self, scenario):
        return None
'''
    stem = _write_module(project, code)
    _write_config(project, f"{stem}:HalfAdapter")
    error = await _run_config_error(project)
    assert error.field == ENTRYPOINT_FIELD
    assert error.message.startswith(COULD_NOT_CREATE.format(module=stem, attr="HalfAdapter", type="TypeError")), (
        error.message
    )
    assert "HalfAdapter" in error.message[len(COULD_NOT_CREATE.format(module=stem, attr="HalfAdapter", type="TypeError")):]


# =============================================================================================
# Running a custom adapter through runner.run (default factory, mocked judge)
# =============================================================================================


@pytest.mark.asyncio
async def test_custom_adapter_runs_end_to_end(impulse_project, llm_router, monkeypatch):
    from impulse_evals.config import ProjectConfig

    monkeypatch.setenv(SECRET_ENV, SECRET)
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    entrypoint = f"{stem}:EchoAdapter"
    _write_config(project, entrypoint, scenarios=[_scenario(project, assertions=[CONSENT])])
    llm_router.script(project.persona(SCENARIO), SECOND_USER)
    result = await _run(_load(project))

    assert result.agent_entrypoint == entrypoint
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert [turn.text for turn in turns] == [OPENING, f"Echo: {OPENING}", SECOND_USER, f"Echo: {SECOND_USER}"]
    assert turns[0].heard_text == OPENING.lower()          # AgentReply.heard_text on the user turn
    assert turns[1].latency_ms == pytest.approx(12.0)

    (call,) = turns[1].tool_calls
    assert call.name == "book_slot"
    assert call.arguments == BOOKING_ARGUMENTS
    assert call.success is True                           # inferred from {"ok": true} by the runner
    assert call.result["booking_id"] == "BK-1001"
    assert "[REDACTED]" in call.result["note"]             # the tool result is redacted
    assert SECRET not in result.model_dump_json()

    assert ADAPTER_WARNING in result.warnings
    assert scenario_result.transcript.metadata.get("custom_adapter") == {"scenario": SCENARIO}
    assert scenario_result.assertion_results and scenario_result.metric_results
    assert llm_router.calls("judge")                      # evaluated with the (mocked) judge
    assert scenario_result.passed is True

    module = _module(stem)
    (adapter,) = module.INSTANCES
    assert isinstance(adapter.config, ProjectConfig)
    assert [event[0] for event in module.EVENTS] == ["start", "send", "send", "close"]


@pytest.mark.asyncio
async def test_one_new_instance_per_scenario(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    _write_config(
        project,
        f"{stem}:EchoAdapter",
        scenarios=[_scenario(project, "custom_one", max_turns=1), _scenario(project, "custom_two", max_turns=1)],
    )
    result = await _run(_load(project))
    assert [r.error for r in result.scenario_results] == [None, None]
    instances = _module(stem).INSTANCES
    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert sorted(adapter.scenario.name for adapter in instances) == ["custom_one", "custom_two"]


@pytest.mark.asyncio
async def test_nested_class_attribute_is_allowed(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    entrypoint = f"{stem}:Holder.EchoAdapter"
    _write_config(project, entrypoint, scenarios=[_scenario(project, max_turns=1)])
    result = await _run(_load(project))
    assert result.scenario_results[0].error is None
    assert result.agent_entrypoint == entrypoint
    assert len(_module(stem).INSTANCES) == 1


@pytest.mark.asyncio
async def test_options_reach_the_custom_adapter(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    options = {"assistant_id": "asst_42", "key_env": "MY_ASSISTANT_KEY", "nested": {"a": [1, 2]}}
    _write_config(project, f"{stem}:EchoAdapter", options=options, scenarios=[_scenario(project, max_turns=1)])
    await _run(_load(project))
    (adapter,) = _module(stem).INSTANCES
    assert adapter.options == options
    assert adapter.config.agent.options == options


@pytest.mark.asyncio
async def test_system_prompt_from_the_custom_adapter_only_when_config_sets_none(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    adapter_prompt = "You are the in-house booking bot. Book with book_slot."
    _write_config(
        project, f"{stem}:EchoAdapter", options={"prompt": adapter_prompt},
        scenarios=[_scenario(project, max_turns=1)], system_prompt=None,
    )
    result = await _run(_load(project))
    assert result.scenario_results[0].transcript.agent_system_prompt == adapter_prompt

    _write_config(project, f"{stem}:EchoAdapter", options={"prompt": adapter_prompt},
                  scenarios=[_scenario(project, max_turns=1)])   # config.yaml sets agent.system_prompt
    result = await _run(_load(project))
    assert result.scenario_results[0].transcript.agent_system_prompt == project.system_prompt


@pytest.mark.asyncio
async def test_agent_turn_error_partial_turn_is_recorded(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    _write_config(project, f"{stem}:EchoAdapter", scenarios=[_scenario(project, max_turns=3)])
    llm_router.script(project.persona(SCENARIO), "Please fail on this one.")
    result = await _run(_load(project))
    scenario_result = result.scenario_results[0]
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    partial = turns[3]
    assert partial.text == "Let me check that."
    (call,) = partial.tool_calls
    assert call.name == "lookup_slot"
    assert call.success is False                     # inferred from {"status": "failed", ...}
    assert call.error == "calendar offline"
    assert scenario_result.error == "custom agent turn failed"
    assert scenario_result.assertion_results == [] and scenario_result.metric_results == []
    assert scenario_result.passed is False
    assert [event[0] for event in _module(stem).EVENTS][-1] == "close"


@pytest.mark.asyncio
async def test_agent_closed_is_a_normal_end_of_the_conversation(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    _write_config(project, f"{stem}:EchoAdapter", scenarios=[_scenario(project, max_turns=4)])
    llm_router.script(project.persona(SCENARIO), "Okay, please hang up now.", "This line is never sent.")
    result = await _run(_load(project))
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    turns = scenario_result.transcript.turns
    assert [turn.speaker for turn in turns] == ["user", "agent", "user", "agent"]
    assert turns[3].text == "Goodbye."
    assert scenario_result.metric_results                   # the turns so far were evaluated
    sends = [event for event in _module(stem).EVENTS if event[0] == "send"]
    assert len(sends) == 2


@pytest.mark.asyncio
async def test_plain_exception_from_the_custom_adapter_is_a_scenario_error(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    _write_config(project, f"{stem}:EchoAdapter",
                  scenarios=[_scenario(project, opening_user_message="Please crash now.")])
    result = await _run(_load(project))
    scenario_result = result.scenario_results[0]
    assert scenario_result.error == "custom agent crashed while answering"
    assert [turn.speaker for turn in scenario_result.transcript.turns] == ["user"]
    assert llm_router.calls("judge") == []


@pytest.mark.asyncio
async def test_concurrency_is_used_as_given(impulse_project, llm_router):
    project = impulse_project
    stem = _write_module(project, ECHO_ADAPTER)
    _write_config(
        project, f"{stem}:EchoAdapter", options={"delay_s": 0.2},
        scenarios=[_scenario(project, "custom_one", max_turns=1), _scenario(project, "custom_two", max_turns=1)],
    )
    result = await _run(_load(project), concurrency=2)
    assert [r.error for r in result.scenario_results] == [None, None]
    assert _module(stem).MAX_ACTIVE[0] == 2              # both scenarios were in send() at once
    assert PIPECAT_CONCURRENCY_MESSAGE not in result.warnings


# =============================================================================================
# CLI: --fail-on and impulse compare on a custom-adapter project
# =============================================================================================

# case -> exit code expected for --fail-on critical, high, any
MATRIX = {
    "all_pass": {"critical": 0, "high": 0, "any": 0},
    "critical_failure": {"critical": 1, "high": 1, "any": 1},
    "high_failure": {"critical": 0, "high": 1, "any": 1},
    "scenario_error": {"critical": 1, "high": 1, "any": 1},
}


def _matrix_project(project, llm_router, case):
    stem = _write_module(project, ECHO_ADAPTER)
    opening = "Please crash now." if case == "scenario_error" else OPENING
    fail = llm_router.verdict("fail", reason="scripted failure")
    if case == "critical_failure":
        llm_router.on_judge(fail, criterion=CONSENT)
    elif case == "high_failure":
        llm_router.on_judge(fail, criterion=PURPOSE)
    return _write_config(
        project,
        f"{stem}:EchoAdapter",
        scenarios=[_scenario(project, "matrix_case", opening_user_message=opening, max_turns=1,
                             assertions=[CONSENT, PURPOSE], metrics=["UNIV-001"])],
    )


@pytest.mark.parametrize("level", ["critical", "high", "any"])
@pytest.mark.parametrize("case", sorted(MATRIX))
def test_fail_on_matrix_with_a_custom_adapter(impulse_project, llm_router, cli, case, level):
    config = _matrix_project(impulse_project, llm_router, case)
    result = cli.invoke("run", config, "--fail-on", level)
    assert result.exit_code == MATRIX[case][level], cli.text(result)
    run = impulse_project.read_run()
    assert run.agent_entrypoint.endswith(":EchoAdapter")


def test_compare_two_runs_of_a_custom_adapter(impulse_project, llm_router, cli):
    project = impulse_project
    config = _matrix_project(project, llm_router, "all_pass")

    before = project.run_dirs()
    assert cli.invoke("run", config).exit_code == 0
    run_a = _new_run_dir(project, before).name
    before = project.run_dirs()
    assert cli.invoke("run", config).exit_code == 0
    run_b = _new_run_dir(project, before).name

    llm_router.on_judge(llm_router.verdict("fail", reason="scripted failure"), criterion=CONSENT)
    before = project.run_dirs()
    assert cli.invoke("run", config).exit_code == 1    # a critical failure exits 1 at the default level (high)
    run_c = _new_run_dir(project, before).name

    same = cli.invoke("compare", run_a, run_b)
    assert same.exit_code == 0, cli.text(same)
    regressed = cli.invoke("compare", run_a, run_c)
    assert regressed.exit_code == 1, cli.text(regressed)   # a critical regression


# =============================================================================================
# --dry-run
# =============================================================================================


def _dry_run_config(project, entrypoint):
    return _write_config(
        project,
        entrypoint,
        judge={"primary_model": "YOUR_PRIMARY_MODEL", "fast_model": "YOUR_FAST_MODEL"},
        simulator={"model": "YOUR_SIMULATOR_MODEL", "persona_style": "realistic", "mode": "cascade"},
    )


def test_dry_run_checks_the_class_without_instantiating_it(impulse_project, mock_llm, cli):
    project = impulse_project
    stem = _write_module(project, EXPLODING_CONSTRUCTOR)
    config = _dry_run_config(project, f"{stem}:ExplodingAdapter")
    result = cli.invoke("run", config, "--dry-run")
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert mock_llm.call_count == 0
    run = project.read_run()
    assert run.scenario_results == []                           # no scenario run
    assert DRY_RUN_WARNING in run.warnings
    assert REMOTE_WARNING.format(adapter="custom") in run.warnings
    assert run.warnings.index(DRY_RUN_WARNING) < run.warnings.index(REMOTE_WARNING.format(adapter="custom"))
    assert REMOTE_WARNING.format(adapter="custom") in text
    assert stem in sys.modules                                   # the class was imported and checked


@pytest.mark.parametrize("kind", ["missing_module", "missing_class", "not_a_subclass"])
def test_dry_run_loading_errors_exit_2(impulse_project, mock_llm, cli, kind):
    project = impulse_project
    if kind == "missing_module":
        stem = f"impulse_test_custom_missing_{uuid.uuid4().hex[:8]}"
        entrypoint, expected = f"{stem}:EchoAdapter", f"cannot import module {stem} from "
    elif kind == "missing_class":
        stem = _write_module(project, ECHO_ADAPTER)
        entrypoint, expected = f"{stem}:MissingAdapter", NO_CLASS.format(module=stem, attr="MissingAdapter")
    else:
        stem = _write_module(project, NOT_ADAPTERS)
        entrypoint, expected = f"{stem}:NotAnAdapter", NOT_SUBCLASS.format(module=stem, attr="NotAnAdapter")
    config = _dry_run_config(project, entrypoint)
    result = cli.invoke("run", config, "--dry-run")
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert ENTRYPOINT_FIELD in text and expected in text, text
    assert mock_llm.call_count == 0


# =============================================================================================
# impulse doctor
# =============================================================================================


def test_doctor_custom_adapter_rows(impulse_project, mock_llm, cli):
    mock_llm.set_default("OK")
    project = impulse_project
    stem = _write_module(project, EXPLODING_CONSTRUCTOR)   # imported, never instantiated by doctor
    config = _write_config(project, f"{stem}:ExplodingAdapter")
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 0, text
    assert _rows(result, "PASS", "agent adapter") == [
        f"custom adapter {stem}:ExplodingAdapter imported (a subclass of AgentInterface)"
    ], text
    assert _rows(result, "INFO", "agent connection") == ["not checked by impulse doctor (custom adapter)"], text
    assert "ExplodingAdapter must never be instantiated here" not in text


@pytest.mark.parametrize("kind", ["missing_module", "not_a_subclass", "bad_form"])
def test_doctor_custom_adapter_fail_row(impulse_project, mock_llm, cli, kind):
    mock_llm.set_default("OK")
    project = impulse_project
    if kind == "missing_module":
        stem = f"impulse_test_custom_missing_{uuid.uuid4().hex[:8]}"
        entrypoint, expected = f"{stem}:EchoAdapter", f"cannot import module {stem} from "
    elif kind == "not_a_subclass":
        stem = _write_module(project, NOT_ADAPTERS)
        entrypoint, expected = f"{stem}:make_adapter", NOT_SUBCLASS.format(module=stem, attr="make_adapter")
    else:
        entrypoint, expected = "echo_adapter", FORM_ERROR.format(entrypoint="echo_adapter")
    config = _write_config(project, entrypoint)
    result = cli.invoke("doctor", config)
    text = cli.text(result)
    assert result.exit_code == 2, text
    if kind == "bad_form" and not _rows(result, "FAIL", "agent adapter"):
        # The form may also be checked by load_config: then the config row fails with the same text.
        assert [row for row in _rows(result, "FAIL", "config") if expected in row], text
        return
    rows = _rows(result, "FAIL", "agent adapter")
    assert len(rows) == 1 and expected in rows[0], text
    assert ENTRYPOINT_FIELD in rows[0], text


# =============================================================================================
# examples/custom_adapter
# =============================================================================================


def test_example_custom_adapter_dry_run_exits_0(impulse_project, mock_llm, cli):
    config = EXAMPLE_DIR / "config.yaml"
    assert config.is_file(), "examples/custom_adapter/config.yaml is missing"
    assert (EXAMPLE_DIR / "echo_adapter.py").is_file(), "examples/custom_adapter/echo_adapter.py is missing"
    result = cli.invoke("run", config, "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0
    run = impulse_project.read_run()
    assert REMOTE_WARNING.format(adapter="custom") in run.warnings


@pytest.mark.asyncio
async def test_example_echo_adapter_books_like_the_sample_agent(impulse_project):
    from impulse_evals.adapters.base import AgentInterface
    from impulse_evals.config import load_config

    config = EXAMPLE_DIR / "config.yaml"
    assert config.is_file(), "examples/custom_adapter/config.yaml is missing"
    cfg = load_config(config)
    assert cfg.agent.adapter == "custom"
    assert cfg.agent.entrypoint == "echo_adapter:EchoAdapter"
    assert [scenario.language for scenario in cfg.scenarios] == ["en-US"]

    spec = importlib.util.spec_from_file_location("impulse_tests_example_echo_adapter", EXAMPLE_DIR / "echo_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.__doc__ or "").strip(), "echo_adapter.py needs a module docstring"
    assert issubclass(module.EchoAdapter, AgentInterface)
    adapter = module.EchoAdapter(cfg)
    await adapter.start(cfg.scenarios[0])
    plain = await adapter.send("Hello there", None)
    assert plain.tool_calls == []
    booked = await adapter.send("I'd like to book", None)
    assert [call.name for call in booked.tool_calls] == ["book_slot"]
    await adapter.close()
    json.dumps([call.model_dump() for call in booked.tool_calls])   # plain JSON values
