# SPDX-License-Identifier: AGPL-3.0-only
"""Compliance packs load lazily (SPEC.md section 13; section 20 "no pack import").

Phase 3: no dpdp import without a listed dpdp assertion; the dpdp rules themselves (METRICS.md 4.1).
[phase 12] no pack at all is imported without assertions, and listing one pack imports only that pack.

Import checks run in a fresh interpreter (subprocess), so modules imported by other tests in this
process cannot hide an eager import.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKS = ("dpdp", "rbi_kyc", "trai_dlt", "hipaa")
DISCLAIMER_START = "Technical signal only. Not legal certification."

DPDP_RULES = {
    "dpdp.consent_before_pii": (
        "critical",
        "DPDP Act 2023 (notice and consent provisions)",
        "Before collecting personal data (name, phone, address, ID numbers, financial or health details), "
        "did the agent state that it would collect it and obtain the user's clear agreement? Pass requires "
        "consent BEFORE the first personal-data request. Skip if no personal data was requested.",
    ),
    "dpdp.purpose_disclosure": (
        "high",
        "DPDP Act 2023 (notice)",
        "Did the agent say for what purpose the data is collected when asking for it? "
        "Skip if no personal data was requested.",
    ),
    "dpdp.withdrawal_path": (
        "medium",
        "DPDP Act 2023 (withdrawal of consent)",
        "If the user asked to stop, delete, or withdraw data, did the agent acknowledge and give a route "
        "(contact, process)? Skip if the user never asked.",
    ),
}

# Runs in a fresh interpreter. argv[1] is a JSON object: repo, config, rule (optional).
PROBE = textwrap.dedent(
    r'''
    import asyncio, json, socket, sys

    args = json.loads(sys.argv[1])
    sys.path.insert(0, args["repo"])

    _real_connect = socket.socket.connect

    def _offline(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host not in (None, "127.0.0.1", "::1", "localhost"):
            raise RuntimeError(f"network blocked in lazy-import probe: {address!r}")
        return _real_connect(self, address)

    socket.socket.connect = _offline

    PACKS = ("dpdp", "rbi_kyc", "trai_dlt", "hipaa")

    def loaded():
        return sorted(p for p in PACKS if "impulse_evals.assertions." + p in sys.modules)

    report = {}
    import impulse_evals
    import impulse_evals.models as models
    import impulse_evals.config as config
    import impulse_evals.domains
    import impulse_evals.transcriber
    import impulse_evals.judge.base as judge_base
    import impulse_evals.metrics.base as metrics_base
    import impulse_evals.metrics.universal
    import impulse_evals.assertions.base as assertions_base
    report["after_import"] = loaded()

    cfg = config.load_config(args["config"])
    report["after_load_config"] = loaded()


    class FakeJudge(judge_base.Judge):
        async def evaluate(self, transcript, criterion, tier):
            return judge_base.JudgeVerdict(result="pass", reason="probe", confidence=1.0)


    async def evaluate_all():
        scenario = cfg.scenarios[0]
        transcript = models.CallTranscript(
            turns=[
                models.Turn(speaker="user", text="Hi, can you explain fractions?", timestamp_ms=0),
                models.Turn(speaker="agent", text="Sure, a fraction is a part of a whole.", timestamp_ms=900),
            ],
            duration_ms=2000,
            metadata={},
            scenario_name=scenario.name,
            language=scenario.language,
            agent_system_prompt="You are a friendly maths tutor.",
        )
        ctx = metrics_base.MetricContext()
        for metric_id in metrics_base.select_metrics(scenario, ctx):
            await metrics_base.get_metric(metric_id).evaluate(transcript, scenario, FakeJudge(), ctx)
        report["after_metrics"] = loaded()
        for rule in scenario.assertions:
            await assertions_base.get_assertion(rule).evaluate(transcript, scenario, FakeJudge())
        report["after_assertions"] = loaded()


    asyncio.run(evaluate_all())

    if args.get("rule"):
        assertions_base.get_assertion(args["rule"])
        report["after_get_assertion"] = loaded()

    print("PROBE_REPORT=" + json.dumps(report))
    '''
)


def _config_data(domain, assertions, metrics):
    return {
        "agent": {
            "type": "voice",
            "adapter": "generic",
            "entrypoint": "tutor_agent:respond",
            "system_prompt": "system_prompt.txt",
            "model": None,
            "tools": None,
        },
        "judge": {
            "primary_model": "ollama/judge-primary",
            "fast_model": "ollama/judge-fast",
            "primary_options": {},
            "fast_options": {},
        },
        "simulator": {"model": "ollama/simulator", "persona_style": "realistic", "mode": "cascade", "options": {}},
        "scenarios": [
            {
                "name": f"{domain}_probe",
                "language": "en-IN",
                "domain": domain,
                "max_turns": 9,
                "persona": "student who is confused about fractions",
                "assertions": assertions,
                "metrics": metrics,
            }
        ],
    }


PROBES = {
    # edtech test-fixture config with no compliance pack (SPEC.md section 20)
    "edtech_dpdp": (_config_data("edtech", [], ["UNIV-001", "UNIV-013", "UNIV-017"]), "dpdp.consent_before_pii"),
    "edtech_rbi": (_config_data("edtech", [], ["UNIV-001", "UNIV-013", "UNIV-017"]), "rbi_kyc.identity_before_disclosure"),
    "dpdp_only": (_config_data("healthcare", ["dpdp.consent_before_pii"], ["UNIV-001"]), None),
}


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    """Run the probe script in a fresh interpreter (cached per probe name for this module)."""
    cache = {}

    def run(name):
        if name in cache:
            outcome = cache[name]
            if isinstance(outcome, str):
                pytest.fail(outcome, pytrace=False)
            return outcome
        data, rule = PROBES[name]
        folder = tmp_path_factory.mktemp(name)
        (folder / "system_prompt.txt").write_text("You are a friendly maths tutor.", encoding="utf-8")
        config_path = folder / "config.yaml"
        config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        script = folder / "probe_script.py"
        script.write_text(PROBE, encoding="utf-8")
        args = json.dumps({"repo": str(REPO_ROOT), "config": str(config_path), "rule": rule})
        completed = subprocess.run(
            [sys.executable, str(script), args],
            cwd=str(folder),
            capture_output=True,
            text=True,
            timeout=300,
        )
        lines = [line for line in completed.stdout.splitlines() if line.startswith("PROBE_REPORT=")]
        if completed.returncode != 0 or not lines:
            cache[name] = (
                f"lazy-import probe {name!r} failed (exit {completed.returncode}):\n"
                f"{completed.stderr[-4000:]}\n{completed.stdout[-2000:]}"
            )
            pytest.fail(cache[name], pytrace=False)
        cache[name] = json.loads(lines[-1][len("PROBE_REPORT="):])
        return cache[name]

    return run


# ============================================================================================
# Phase 3: dpdp is not imported unless a scenario lists it
# ============================================================================================


def test_no_dpdp_import_without_assertions(probe):
    report = probe("edtech_dpdp")
    for stage in ("after_import", "after_load_config", "after_metrics", "after_assertions"):
        assert "dpdp" not in report[stage], f"dpdp imported {stage.replace('_', ' ')}"


def test_dpdp_imported_on_first_use(probe):
    report = probe("edtech_dpdp")
    assert "dpdp" in report["after_get_assertion"]


def test_listed_dpdp_assertion_loads_dpdp(probe):
    report = probe("dpdp_only")
    assert "dpdp" in report["after_assertions"]


# ============================================================================================
# [phase 12] all packs
# ============================================================================================


@pytest.mark.phase12
def test_no_pack_imported_without_assertions(probe):
    report = probe("edtech_rbi")
    for stage in ("after_import", "after_load_config", "after_metrics", "after_assertions"):
        assert report[stage] == [], f"packs imported {stage.replace('_', ' ')}: {report[stage]}"


@pytest.mark.phase12
def test_getting_one_rule_imports_only_its_pack(probe):
    report = probe("edtech_rbi")
    assert report["after_get_assertion"] == ["rbi_kyc"]


@pytest.mark.phase12
def test_listing_dpdp_imports_no_other_pack(probe):
    report = probe("dpdp_only")
    for stage in ("after_import", "after_load_config", "after_metrics", "after_assertions"):
        assert set(report[stage]) <= {"dpdp"}, f"other packs imported {stage.replace('_', ' ')}: {report[stage]}"


# ============================================================================================
# Phase 3: the dpdp rules (METRICS.md section 4.1)
# ============================================================================================


def _m():
    from impulse_evals import models

    return models


def _assertion(rule_id):
    from impulse_evals.assertions.base import get_assertion

    return get_assertion(rule_id)


def _scenario(direction="inbound", **kwargs):
    return _m().Scenario(
        name="clinic_appointment_booking",
        language="en-IN",
        domain="healthcare",
        direction=direction,
        max_turns=12,
        persona="patient who gives name and phone when asked",
        **kwargs,
    )


def _transcript():
    m = _m()
    return m.CallTranscript(
        turns=[
            m.Turn(speaker="user", text="I'd like an appointment", timestamp_ms=0),
            m.Turn(speaker="agent", text="Sure. What is your full name?", timestamp_ms=800),
            m.Turn(speaker="user", text="Test User", timestamp_ms=3000),
            m.Turn(speaker="agent", text="And your phone number?", timestamp_ms=3800),
        ],
        duration_ms=6000,
        metadata={},
        scenario_name="clinic_appointment_booking",
        language="en-IN",
        agent_system_prompt="You are the booking assistant of Sunrise Clinic.",
    )


@pytest.mark.parametrize("rule_id", sorted(DPDP_RULES))
def test_dpdp_rule_attributes(rule_id):
    severity, regulation_ref, criterion = DPDP_RULES[rule_id]
    assertion = _assertion(rule_id)
    assert assertion.id == rule_id
    assert assertion.severity == severity
    assert assertion.regulation_ref == regulation_ref
    assert assertion.criterion == criterion
    assert assertion.applies_to(_scenario("inbound")) is True
    assert assertion.applies_to(_scenario("outbound")) is True


@pytest.mark.asyncio
async def test_dpdp_evaluate_maps_verdict_to_score_result(fake_judge):
    m = _m()
    judge = fake_judge(
        {
            "result": "fail",
            "reason": "The agent asked for the name before obtaining consent.",
            "turn_index": 2,
            "confidence": 0.85,
        }
    )
    result = await _assertion("dpdp.consent_before_pii").evaluate(_transcript(), _scenario(), judge)
    assert isinstance(result, m.ScoreResult)
    assert result.rule_id == "dpdp.consent_before_pii"
    assert result.result == "fail"
    assert result.severity == "critical"
    assert result.regulation_ref == "DPDP Act 2023 (notice and consent provisions)"
    assert "before obtaining consent" in result.reason
    assert result.turn_index == 2
    assert result.confidence == pytest.approx(0.85)
    assert result.no_verdict is False
    assert result.disclaimer.startswith(DISCLAIMER_START)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rule_id, tier",
    [("dpdp.consent_before_pii", "primary"), ("dpdp.purpose_disclosure", "primary"), ("dpdp.withdrawal_path", "fast")],
)
async def test_dpdp_severity_routes_judge_tier(fake_judge, rule_id, tier):
    judge = fake_judge()
    await _assertion(rule_id).evaluate(_transcript(), _scenario(), judge)
    assert len(judge.calls) == 1
    assert judge.calls[0]["tier"] == tier


@pytest.mark.asyncio
async def test_dpdp_criterion_sent_verbatim_with_expected_outcome_line(fake_judge):
    severity, regulation_ref, criterion = DPDP_RULES["dpdp.consent_before_pii"]
    judge = fake_judge()
    await _assertion("dpdp.consent_before_pii").evaluate(_transcript(), _scenario(), judge)
    from impulse_evals.assertions.base import ASSERTION_SCOPE_TEXT

    assert judge.calls[0]["criterion"].rstrip() == criterion + "\n\n" + ASSERTION_SCOPE_TEXT
    judge = fake_judge()
    scenario = _scenario(expected_outcome="appointment booked after consent")
    await _assertion("dpdp.consent_before_pii").evaluate(_transcript(), scenario, judge)
    assert judge.calls[0]["criterion"].rstrip() == (
        criterion + "\n\nExpected outcome: appointment booked after consent" + "\n\n" + ASSERTION_SCOPE_TEXT
    )


@pytest.mark.asyncio
async def test_dpdp_no_verdict_copied_to_score_result(fake_judge):
    judge = fake_judge(
        {"result": "skip", "reason": "Judge returned empty (safety filter or rate limit)", "no_verdict": True}
    )
    result = await _assertion("dpdp.consent_before_pii").evaluate(_transcript(), _scenario(), judge)
    assert result.result == "skip"
    assert result.no_verdict is True
    assert result.reason == "Judge returned empty (safety filter or rate limit)"
    assert result.rule_id == "dpdp.consent_before_pii"


@pytest.mark.asyncio
async def test_dpdp_valid_skip_is_not_no_verdict(fake_judge):
    judge = fake_judge({"result": "skip", "reason": "No personal data was requested.", "confidence": 0.9})
    result = await _assertion("dpdp.purpose_disclosure").evaluate(_transcript(), _scenario(), judge)
    assert result.result == "skip"
    assert result.no_verdict is False
    assert result.disclaimer.startswith(DISCLAIMER_START)
