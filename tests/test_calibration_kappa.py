# SPDX-License-Identifier: Apache-2.0
"""Phase 10: calibration (SPEC.md section 15).

``calibration.cohen_kappa(a, b)``: Cohen's kappa of two label lists (no sklearn); 1.0 when observed
and expected agreement are both 1.0.

``impulse calibrate DIR``: one JSON file per item, ``{"transcript": CallTranscript, "check": ID,
"human_label": "pass" | "fail"}``; Impulse runs the judge on each item (metrics through
``get_metric``, pack rules through ``get_assertion``); items where the judge returns ``skip`` (also
a no-verdict skip) are left out of agreement and kappa and reported as a count. Output: agreement
rate, kappa, item count (every item file read), skipped count. Exit 0 unless the directory or a
file is invalid (exit 2). CHAT-xxx items are judged like the others (calibration has no config.yaml,
so they are never skipped for ``agent.type: voice``); that test needs ``metrics/chat.py`` and carries
the ``phase12`` marker, so phase 10 runs this file with ``-m "not phase12"``.
"""

import json
import re

import pytest

MODEL_ENV = {
    "IMPULSE_MODEL_PRIMARY": "ollama/judge-primary",
    "IMPULSE_MODEL_FAST": "ollama/judge-fast",
    "IMPULSE_MODEL_SIMULATOR": "ollama/simulator",
}
SYSTEM_PROMPT = "You are the booking assistant of Sunrise Clinic. Ask for consent before collecting personal data."
CONSENT = "dpdp.consent_before_pii"


def _kappa(a, b):
    from impulse_evals.calibration import cohen_kappa

    return cohen_kappa(list(a), list(b))


# ============================================================================================
# cohen_kappa on hand-computed vectors
# ============================================================================================

P, F = "pass", "fail"


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # po = 3/4, pe = 0.5*0.25 + 0.5*0.75 = 0.5 -> (0.75 - 0.5) / 0.5
        ([P, P, F, F], [P, F, F, F], 0.5),
        # perfect agreement with both labels: po = 1, pe = 0.5
        ([P, F, P, F], [P, F, P, F], 1.0),
        # total disagreement: po = 0, pe = 0.5 -> -1
        ([P, F], [F, P], -1.0),
        # po = 0.75, pe = 0.75*1 + 0.25*0 = 0.75 -> 0
        ([P, P, P, F], [P, P, P, P], 0.0),
        # 5 both pass, 1 pass/fail, 1 fail/pass, 3 both fail: po = 0.8, pe = 0.36 + 0.16 = 0.52
        ([P] * 5 + [P] + [F] + [F] * 3, [P] * 5 + [F] + [P] + [F] * 3, 0.28 / 0.48),
        # each rater uses one different label: po = 0, pe = 0 -> 0
        ([P, P], [F, F], 0.0),
    ],
)
def test_cohen_kappa_known_vectors(a, b, expected):
    value = _kappa(a, b)
    assert isinstance(value, float)
    assert value == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("label", [P, F])
def test_cohen_kappa_is_one_when_observed_and_expected_agreement_are_one(label):
    # Both raters always give the same single label: po = pe = 1 (0/0 by the formula).
    assert _kappa([label] * 4, [label] * 4) == 1.0


def test_cohen_kappa_is_symmetric():
    a = [P, P, F, F, P, F, P]
    b = [P, F, F, P, P, F, F]
    assert _kappa(a, b) == pytest.approx(_kappa(b, a), abs=1e-12)


def test_cohen_kappa_three_labels():
    # po = 0.75; pa = (x .5, y .25, z .25), pb = (x .5, y .5, z 0); pe = .25 + .125 = .375 -> 0.6
    assert _kappa(["x", "y", "z", "x"], ["x", "y", "y", "x"]) == pytest.approx(0.6, abs=1e-9)


def test_cohen_kappa_does_not_use_sklearn():
    from pathlib import Path

    import impulse_evals.calibration as calibration

    assert "sklearn" not in Path(calibration.__file__).read_text(encoding="utf-8")


# ============================================================================================
# impulse calibrate
# ============================================================================================


@pytest.fixture
def project(impulse_project, llm_router, monkeypatch):
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    return impulse_project


def _transcript(marker):
    from impulse_evals.models import CallTranscript, Turn

    return CallTranscript(
        turns=[
            Turn(speaker="user", text=f"Hello, I need an appointment ({marker}).", timestamp_ms=0),
            Turn(speaker="agent", text="Sure. May I have your phone number?", timestamp_ms=900, latency_ms=900.0),
            Turn(speaker="user", text="It is 0000000000.", timestamp_ms=3000),
            Turn(speaker="agent", text="Thank you, you are booked for Tuesday.", timestamp_ms=3900, latency_ms=900.0),
        ],
        duration_ms=5000,
        metadata={},
        scenario_name=f"calibration_{marker}",
        language="en-IN",
        agent_system_prompt=SYSTEM_PROMPT,
    )


def _write_item(folder, marker, check, human_label, *, raw=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"item_{marker}.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    item = {"transcript": _transcript(marker).model_dump(mode="json"), "check": check, "human_label": human_label}
    path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return path


def _numbers_on_lines(output, keyword):
    """Every number on the output lines that contain ``keyword`` (case-insensitive)."""
    numbers = []
    for line in output.splitlines():
        if keyword in line.lower():
            numbers += [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", line)]
    return numbers


def _has_value(numbers, value, *, percent=False):
    candidates = [value] + ([value * 100] if percent else [])
    return any(abs(number - candidate) < 1e-6 for number in numbers for candidate in candidates)


def _route(llm_router, marker, verdict):
    llm_router.on_judge(verdict, transcript=f"({marker})")


def test_calibrate_reports_agreement_kappa_items_and_skipped(project, llm_router, cli):
    folder = project.root / "labeled"
    v = llm_router.verdict
    cases = [
        # marker, check, human label, judge verdict
        ("alpha", "UNIV-013", "pass", v("pass")),
        ("bravo", CONSENT, "pass", v("fail")),
        ("charlie", "UNIV-013", "fail", v("fail")),
        ("delta", CONSENT, "fail", v("fail")),
        ("echo", "UNIV-013", "pass", v("skip", reason="nothing to judge")),
    ]
    for marker, check, label, verdict in cases:
        _write_item(folder, marker, check, label)
        _route(llm_router, marker, verdict)

    result = cli.invoke("calibrate", folder)
    assert result.exit_code == 0, cli.text(result)
    output = result.output
    # Included items: alpha, bravo, charlie, delta -> agreement 3/4; human [p, p, f, f] vs judge [p, f, f, f].
    assert _has_value(_numbers_on_lines(output, "agreement"), 0.75, percent=True), output
    assert _has_value(_numbers_on_lines(output, "kappa"), 0.5), output
    assert _has_value(_numbers_on_lines(output, "item"), 5), output
    assert _has_value(_numbers_on_lines(output, "skip"), 1), output

    judge_calls = llm_router.calls("judge")
    assert len(judge_calls) == 5, "one judge call per item"
    models = sorted(call["model"] for call in judge_calls)
    # UNIV-013 is a fast-tier metric; dpdp.consent_before_pii is critical (primary model).
    assert models == ["ollama/judge-fast"] * 3 + ["ollama/judge-primary"] * 2
    assert all(SYSTEM_PROMPT in "\n".join(llm_router.contents(call)) for call in judge_calls)


def test_calibrate_counts_a_no_verdict_item_as_skipped(project, llm_router, cli):
    folder = project.root / "labeled"
    v = llm_router.verdict
    for marker, label, verdict in (("alpha", "pass", v("pass")), ("bravo", "fail", v("fail")), ("charlie", "pass", None)):
        _write_item(folder, marker, "UNIV-013", label)
        _route(llm_router, marker, verdict)
    result = cli.invoke("calibrate", folder)
    assert result.exit_code == 0, cli.text(result)
    output = result.output
    assert _has_value(_numbers_on_lines(output, "agreement"), 1.0, percent=True), output
    assert _has_value(_numbers_on_lines(output, "kappa"), 1.0), output
    assert _has_value(_numbers_on_lines(output, "item"), 3), output
    assert _has_value(_numbers_on_lines(output, "skip"), 1), output


def _assert_usage_error(cli, result):
    assert result.exit_code == 2, cli.text(result)
    assert "No such command" not in cli.text(result), "impulse calibrate must be a registered command"


@pytest.mark.phase12
def test_calibrate_judges_chat_items(project, llm_router, cli):
    """CHAT-xxx checks are a valid item kind (SPEC.md section 15), so calibration judges them; it has no
    config.yaml, so they are never skipped as "chat metric; agent.type is voice"."""
    folder = project.root / "labeled"
    v = llm_router.verdict
    for marker, check, label, verdict in (
        ("alpha", "CHAT-003", "pass", v("pass")),
        ("bravo", "CHAT-005", "fail", v("fail")),
    ):
        _write_item(folder, marker, check, label)
        _route(llm_router, marker, verdict)
    result = cli.invoke("calibrate", folder)
    assert result.exit_code == 0, cli.text(result)
    output = result.output
    assert _has_value(_numbers_on_lines(output, "agreement"), 1.0, percent=True), output
    assert _has_value(_numbers_on_lines(output, "kappa"), 1.0), output
    assert _has_value(_numbers_on_lines(output, "item"), 2), output
    assert _has_value(_numbers_on_lines(output, "skip"), 0), output
    # CHAT-003 is a fast-tier metric, CHAT-005 a primary-tier one (METRICS.md section 3).
    assert sorted(call["model"] for call in llm_router.calls("judge")) == ["ollama/judge-fast", "ollama/judge-primary"]


def test_calibrate_missing_directory_exits_2(project, cli):
    result = cli.invoke("calibrate", project.root / "no_such_folder")
    _assert_usage_error(cli, result)


def test_calibrate_file_instead_of_directory_exits_2(project, cli):
    path = _write_item(project.root / "labeled", "alpha", "UNIV-013", "pass")
    result = cli.invoke("calibrate", path)
    _assert_usage_error(cli, result)


@pytest.mark.parametrize(
    "bad_item",
    [
        pytest.param({"raw": "{not json"}, id="invalid_json"),
        pytest.param({"check": "UNIV-013", "human_label": "maybe"}, id="bad_human_label"),
        pytest.param({"check": "UNIV-999", "human_label": "pass"}, id="unknown_check"),
        pytest.param({"check": "UNIV-013", "human_label": "pass", "no_transcript": True}, id="missing_transcript"),
    ],
)
def test_calibrate_invalid_item_file_exits_2(project, llm_router, cli, bad_item):
    folder = project.root / "labeled"
    _write_item(folder, "alpha", "UNIV-013", "pass")
    if "raw" in bad_item:
        _write_item(folder, "bravo", None, None, raw=bad_item["raw"])
    elif bad_item.get("no_transcript"):
        _write_item(folder, "bravo", None, None, raw=json.dumps({"check": "UNIV-013", "human_label": "pass"}))
    else:
        _write_item(folder, "bravo", bad_item["check"], bad_item["human_label"])
    result = cli.invoke("calibrate", folder)
    _assert_usage_error(cli, result)


# ============================================================================================
# Audit batch 1 review fix-ups (AUDIT.md finding 6, follow-up): impulse calibrate reads no
# config.yaml, so the literal credentials in config.toml [model_options.primary] must still be
# redacted from what it prints (``Bearer [REDACTED]``) while reaching the judge call unchanged.
# ============================================================================================

TOML_LITERAL_BEARER = "LITERALTOML321"
TOML_LITERAL_HEADERS = (
    "[model_options.primary]\n"
    f'extra_params = {{ extra_headers = {{ Authorization = "Bearer {TOML_LITERAL_BEARER}" }} }}\n'
)
ECHOED_REDACTED_HEADERS = "401 Unauthorized; request headers: {'Authorization': 'Bearer [REDACTED]'}"


def test_calibrate_never_prints_literal_toml_option_credentials(project, llm_router, cli):
    project.write_toml(TOML_LITERAL_HEADERS)
    folder = project.root / "labeled"
    _write_item(folder, "alpha", CONSENT, "pass")   # a critical rule: judged by the primary model

    def responder(call):
        if llm_router.kind(call) == "judge":
            raise RuntimeError(f"401 Unauthorized; request headers: {call['kwargs'].get('extra_headers')}")
        return "unexpected call"

    llm_router.mock.responder = responder
    result = cli.invoke("calibrate", folder)
    text = cli.text(result)
    assert result.exit_code != 0, text
    assert TOML_LITERAL_BEARER not in result.output, text
    assert ECHOED_REDACTED_HEADERS in text
    judge_calls = llm_router.calls("judge")
    assert judge_calls and judge_calls[0]["model"] == "ollama/judge-primary"
    assert judge_calls[0]["kwargs"].get("extra_headers") == {"Authorization": f"Bearer {TOML_LITERAL_BEARER}"}
