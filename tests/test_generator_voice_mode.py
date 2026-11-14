# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 9: generator metric lists by mode (SPEC.md section 11).

The generation prompt lists the text-mode metric IDs, or with ``--voice-mode`` the voice-mode IDs.
In voice mode every generated scenario gets VOICE-001, VOICE-003, and VOICE-005. In both modes the
generator drops every metric ID the LLM returns that is not in that mode's list, then adds the
domain defaults.
"""

import re
from pathlib import Path

import pytest
import yaml

MODEL_ENV = {
    "IMPULSE_MODEL_PRIMARY": "ollama/judge-primary",
    "IMPULSE_MODEL_FAST": "ollama/judge-fast",
    "IMPULSE_MODEL_SIMULATOR": "ollama/simulator",
}
PROMPT_TEXT = "You are the assistant of Sunrise Store. Answer questions politely and keep replies short."

TEXT_LIST = {
    "UNIV-001", "UNIV-003", "UNIV-005", "UNIV-007", "UNIV-009", "UNIV-011", "UNIV-013",
    "UNIV-015", "UNIV-017", "UNIV-019", "UNIV-021", "UNIV-023",
    "VOICE-027", "VOICE-029", "VOICE-035", "VOICE-037", "VOICE-039",
}
VOICE_LIST = TEXT_LIST | {
    "VOICE-001", "VOICE-003", "VOICE-005", "VOICE-007", "VOICE-009", "VOICE-013", "VOICE-015", "VOICE-031",
}
ALWAYS_IN_VOICE = {"VOICE-001", "VOICE-003", "VOICE-005"}
EDTECH_DEFAULTS = {"UNIV-005", "UNIV-013", "UNIV-017"}
METRIC_ID = re.compile(r"\b(?:UNIV|VOICE|CHAT)-\d{3}\b")
# What the LLM answers: IDs from both lists plus IDs in neither (automatic, chat, unknown).
LLM_METRICS = [
    "UNIV-013", "VOICE-001", "VOICE-003", "VOICE-005", "VOICE-007", "VOICE-009", "VOICE-013", "VOICE-015",
    "VOICE-031", "VOICE-029", "VOICE-011", "UNIV-025", "CHAT-001", "NOT-A-METRIC",
]


@pytest.fixture
def project(impulse_project, mock_llm, monkeypatch):
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    return impulse_project


def _items(metrics_per_item):
    return [
        {
            "name": f"mode_case_{number}",
            "persona": f"caller {number}",
            "opening_user_message": f"Hello, this is caller {number}.",
            "direction": "inbound",
            "metrics": list(metrics),
        }
        for number, metrics in enumerate(metrics_per_item, start=1)
    ]


def _generate(cli, project, items, *extra, domain="edtech"):
    prompt = Path(project.root) / "agent_prompt.txt"
    prompt.write_text(PROMPT_TEXT, encoding="utf-8")
    return cli.invoke(
        "generate", "--prompt", prompt, "--domain", domain, "--language", "en-IN",
        "--count", str(len(items)), "--yes", *extra,
    )


def _generated_metrics(project):
    data = yaml.safe_load((Path(project.root) / "scenarios" / "generated.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    metrics = []
    for item in data:
        assert isinstance(item.get("metrics"), list) and item["metrics"], "explicit, non-empty metrics list"
        assert len(item["metrics"]) == len(set(item["metrics"])), f"duplicate metric IDs: {item['metrics']}"
        metrics.append(set(item["metrics"]))
    return metrics


# ============================================================================================
# The metric list in the generation prompt
# ============================================================================================


def test_text_mode_prompt_lists_exactly_the_text_metrics(project, mock_llm, cli):
    mock_llm.queue(yaml.safe_dump(_items([["UNIV-001"]])))
    result = _generate(cli, project, _items([["UNIV-001"]]))
    assert result.exit_code == 0, cli.text(result)
    assert set(METRIC_ID.findall(mock_llm.text(0))) == TEXT_LIST


def test_voice_mode_prompt_lists_exactly_the_voice_metrics(project, mock_llm, cli):
    mock_llm.queue(yaml.safe_dump(_items([["UNIV-001"]])))
    result = _generate(cli, project, _items([["UNIV-001"]]), "--voice-mode")
    assert result.exit_code == 0, cli.text(result)
    assert set(METRIC_ID.findall(mock_llm.text(0))) == VOICE_LIST


# ============================================================================================
# Generated metric lists
# ============================================================================================


def test_voice_mode_adds_voice_001_003_005_to_every_scenario(project, mock_llm, cli):
    items = _items([[], ["UNIV-013"], ["VOICE-001"], ["VOICE-027", "VOICE-035"]])
    mock_llm.queue(yaml.safe_dump(items))
    result = _generate(cli, project, items, "--voice-mode")
    assert result.exit_code == 0, cli.text(result)
    generated = _generated_metrics(project)
    assert len(generated) == 4
    for metrics in generated:
        assert ALWAYS_IN_VOICE <= metrics


def test_text_mode_never_has_voice_001_003_005(project, mock_llm, cli):
    items = _items([["VOICE-001", "VOICE-003", "VOICE-005", "UNIV-001"], []])
    mock_llm.queue(yaml.safe_dump(items))
    result = _generate(cli, project, items)
    assert result.exit_code == 0, cli.text(result)
    for metrics in _generated_metrics(project):
        assert not metrics & ALWAYS_IN_VOICE


def test_text_mode_drops_ids_outside_the_text_list_then_adds_domain_defaults(project, mock_llm, cli):
    items = _items([LLM_METRICS])
    mock_llm.queue(yaml.safe_dump(items))
    result = _generate(cli, project, items, domain="edtech")
    assert result.exit_code == 0, cli.text(result)
    assert _generated_metrics(project) == [{"UNIV-013", "VOICE-029"} | EDTECH_DEFAULTS]


def test_voice_mode_drops_ids_outside_the_voice_list_then_adds_domain_defaults(project, mock_llm, cli):
    items = _items([LLM_METRICS])
    mock_llm.queue(yaml.safe_dump(items))
    result = _generate(cli, project, items, "--voice-mode", domain="edtech")
    assert result.exit_code == 0, cli.text(result)
    kept = {
        "UNIV-013", "VOICE-001", "VOICE-003", "VOICE-005", "VOICE-007", "VOICE-009", "VOICE-013",
        "VOICE-015", "VOICE-031", "VOICE-029",
    }
    assert _generated_metrics(project) == [kept | EDTECH_DEFAULTS]


@pytest.mark.parametrize("voice", [False, True], ids=["text", "voice"])
def test_generated_metrics_stay_inside_the_mode_list(project, mock_llm, cli, voice):
    items = _items([LLM_METRICS, [], ["UNIV-025", "VOICE-011"]])
    mock_llm.queue(yaml.safe_dump(items))
    extra = ("--voice-mode",) if voice else ()
    result = _generate(cli, project, items, *extra, domain="general")
    assert result.exit_code == 0, cli.text(result)
    allowed = VOICE_LIST if voice else TEXT_LIST
    for metrics in _generated_metrics(project):
        assert metrics <= allowed
        assert "UNIV-025" not in metrics and "VOICE-011" not in metrics
