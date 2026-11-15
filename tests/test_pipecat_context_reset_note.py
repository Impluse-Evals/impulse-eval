# SPDX-License-Identifier: Apache-2.0
"""Fixes round 4, O1 (FINAL-TEST.md audit, 2026-11-02): say so when Impulse resets a Pipecat eval bot.

In eval mode with ``agent.system_prompt`` set, the adapter sends ``eval-context`` with that prompt before
every scenario, which replaces the bot's LLM context: the bot then runs the FILE's prompt, not its own.
In the final live test a BUGGY bot behaved normally for a whole run because of it. The adapter now adds,
once per run, the warning ``pipecat_adapter.CONTEXT_RESET_NOTE``:
"Pipecat eval bot: Impulse reset its conversation with the agent.system_prompt file before each scenario,
so the bot ran that prompt, not its own. Keep the file identical to the bot's prompt."
Without ``agent.system_prompt`` the existing RESET_WARNING is the only note (unchanged); websocket mode
sends no eval-context and gets no note.

``mock_pipecat_channel`` only: no network, no pipecat-ai.
"""

import pytest

pytestmark = pytest.mark.phase11

URL = "ws://localhost:7860"
BOT_READY = {"version": "2.1.0", "about": {"library": "pipecat-ai", "library_version": "1.12.0"}}
DEFAULT = object()
CONTEXT_RESET_NOTE = (
    "Pipecat eval bot: Impulse reset its conversation with the agent.system_prompt file before each scenario, "
    "so the bot ran that prompt, not its own. Keep the file identical to the bot's prompt."
)
RESET_WARNING = (
    "Pipecat eval bots keep their conversation between scenarios. Set agent.system_prompt in config.yaml "
    "so Impulse can reset it, or reset the LLM context in the bot's on_client_connected handler."
)


def _cfg(project, mode="eval", *, system_prompt=DEFAULT):
    kwargs = {}
    if system_prompt is not DEFAULT:
        kwargs["system_prompt"] = system_prompt
    project.write_config(
        [project.scenario("reset_note_one"), project.scenario("reset_note_two")],
        agent={"adapter": "pipecat", "pipecat": {"mode": mode, "url": URL}},
        **kwargs,
    )
    return project.load()


def _handshake(mock, mode, channel):
    if mode == "eval":
        channel.paced_audio = True
        channel.on_connect(mock.rtvi("bot-ready", BOT_READY, id="client-ready"))
    else:
        channel.on_send("client-ready", lambda message: mock.rtvi("bot-ready", BOT_READY, id=message.get("id")))


async def _two_starts(cfg, mock, mode):
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter

    _handshake(mock, mode, mock.channel)
    _handshake(mock, mode, mock.add_channel())
    adapter = PipecatAdapter(cfg, channel_factory=mock.factory, clock=mock.clock)
    await adapter.start(cfg.scenarios[0])
    await adapter.start(cfg.scenarios[1])   # a second scenario on the same adapter: the note stays single
    await adapter.close()
    return adapter


def test_note_text_is_a_module_constant():
    from impulse_evals.adapters import pipecat_adapter

    assert pipecat_adapter.CONTEXT_RESET_NOTE == CONTEXT_RESET_NOTE


@pytest.mark.asyncio
async def test_eval_mode_with_a_prompt_file_notes_the_reset_once(impulse_project, mock_pipecat_channel):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "eval")
    adapter = await _two_starts(cfg, mock, "eval")
    warnings = adapter.get_warnings()
    assert warnings.count(CONTEXT_RESET_NOTE) == 1
    assert RESET_WARNING not in warnings
    sent = [m for channel in mock.channels for m in channel.sent_messages if m.get("type") == "client-message"]
    assert [m["data"].get("t") for m in sent if m["data"].get("t") == "eval-context"] == ["eval-context", "eval-context"]


@pytest.mark.asyncio
async def test_eval_mode_without_a_prompt_file_gets_only_the_reset_warning(impulse_project, mock_pipecat_channel):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "eval", system_prompt=None)
    adapter = await _two_starts(cfg, mock, "eval")
    warnings = adapter.get_warnings()
    assert CONTEXT_RESET_NOTE not in warnings
    assert warnings.count(RESET_WARNING) == 1


@pytest.mark.asyncio
async def test_websocket_mode_sends_no_eval_context_and_no_note(impulse_project, mock_pipecat_channel):
    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "websocket")
    adapter = await _two_starts(cfg, mock, "websocket")
    assert CONTEXT_RESET_NOTE not in adapter.get_warnings()
    sent = [m for channel in mock.channels for m in channel.sent_messages if m.get("type") == "client-message"]
    assert all(m["data"].get("t") != "eval-context" for m in sent)


@pytest.mark.asyncio
async def test_the_note_reaches_the_run_warnings(impulse_project, mock_pipecat_channel, llm_router):
    """Through ``runner.run``: the adapter's warnings land in ``RunResult.warnings`` once."""
    from impulse_evals.adapters.pipecat_adapter import PipecatAdapter
    from impulse_evals.runner import run

    mock = mock_pipecat_channel
    cfg = _cfg(impulse_project, "eval")
    for channel in (mock.channel, mock.add_channel()):
        _handshake(mock, "eval", channel)
        # the bot answers every send-text with one LLM run that says "Hello."
        channel.on_send(
            "send-text",
            mock.rtvi("bot-llm-started"),
            mock.rtvi("bot-llm-text", {"text": "Hello."}),
            mock.rtvi("bot-llm-stopped"),
            once=False,
            gap=0.05,
        )
    result = await run(cfg, adapter_factory=lambda config: PipecatAdapter(config, channel_factory=mock.factory, clock=mock.clock))
    assert [r.error for r in result.scenario_results] == [None, None]
    assert result.warnings.count(CONTEXT_RESET_NOTE) == 1
