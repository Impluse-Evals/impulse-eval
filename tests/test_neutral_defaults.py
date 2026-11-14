# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 3, B5: neutral compliance-pack defaults and the two global example scenarios
(FIXES-ROUND-3-SPEC.md section B5; maintainer decision 2026-10-26 "Neutral defaults").

* ``domains.INDIA_PACKS == ("dpdp", "rbi_kyc", "trai_dlt")`` and ``domains.is_india_locale(language)``:
  True when any subtag after the first equals ``IN`` case-insensitively (``en-IN``, ``hi-IN``,
  ``hi-Latn-IN``, ``ta-in``).
* ``DomainDefaults.assertions_for(direction, language=None)`` and ``default_assertions(domain,
  direction, language=None)``: a language whose region is not ``IN`` gets no India pack (so no
  auto-added assertion at all; ``hipaa`` stays opt-in everywhere); ``language=None`` keeps today's
  behaviour. Metrics and ``max_turns`` are unchanged.
* ``impulse generate`` passes the scenario language: fsi en-US gets ``assertions: []``, en-IN the
  India packs as today.
* ``examples/config.yaml`` gains ``support_order_status_us`` (en-US, ecommerce) and
  ``cita_clinica_mx`` (es-MX, healthcare, ``book_slot`` expected once) after the four Indian
  scenarios; its dry run exits 0 with all six scenarios passing.
"""

from pathlib import Path

import pytest
import yaml
from conftest import REPO_ROOT

MODEL_ENV = {
    "IMPULSE_MODEL_PRIMARY": "ollama/judge-primary",
    "IMPULSE_MODEL_FAST": "ollama/judge-fast",
    "IMPULSE_MODEL_SIMULATOR": "ollama/simulator",
}
PROMPT_TEXT = "You are the assistant of Sunrise Store. Answer questions politely and keep replies short."
EXAMPLES_CONFIG = REPO_ROOT / "examples" / "config.yaml"

DPDP = ["dpdp.consent_before_pii", "dpdp.purpose_disclosure", "dpdp.withdrawal_path"]
RBI_KYC = [
    "rbi_kyc.identity_before_disclosure",
    "rbi_kyc.no_credential_request",
    "rbi_kyc.no_unauthorized_commitment",
]
TRAI_DLT = ["trai_dlt.consent_for_outbound", "trai_dlt.honors_opt_out", "trai_dlt.identifies_sender"]
INDIA_PREFIXES = ("dpdp.", "rbi_kyc.", "trai_dlt.")
COMPLIANCE_PREFIXES = INDIA_PREFIXES + ("hipaa.",)
UNIVERSAL_JUDGE = [f"UNIV-{n:03d}" for n in range(1, 24, 2)]
GENERAL_METRICS = UNIVERSAL_JUDGE + ["VOICE-037", "VOICE-039"]

# domain: (assertions for every scenario, extra assertions for outbound only, metrics, max_turns)
# (the SPEC.md section 11 table, as in test_domain_defaults.py)
TABLE = {
    "fsi": (RBI_KYC + DPDP, TRAI_DLT, ["UNIV-001", "UNIV-005", "UNIV-019", "UNIV-023"], 12),
    "healthcare": (DPDP, [], ["UNIV-005", "UNIV-011", "UNIV-019", "UNIV-023"], 12),
    "recruitment": (DPDP, [], ["UNIV-013", "UNIV-019", "UNIV-023"], 9),
    "ecommerce": ([], [], ["UNIV-001", "UNIV-003", "UNIV-007", "UNIV-023"], 9),
    "edtech": ([], [], ["UNIV-005", "UNIV-013", "UNIV-017"], 9),
    "telecom": ([], TRAI_DLT, ["UNIV-001", "UNIV-011", "UNIV-023"], 9),
    "realestate": (DPDP, [], ["UNIV-005", "UNIV-007"], 9),
    "travel": ([], [], ["UNIV-001", "UNIV-003", "UNIV-007", "UNIV-023"], 9),
    "legal": (DPDP, [], ["UNIV-005", "UNIV-019"], 9),
    "general": ([], [], GENERAL_METRICS, 9),
}
DIRECTIONS = ("inbound", "outbound")
INDIA_LANGUAGES = ("hi-IN", "en-IN", "hi-Latn-IN", "ta-in")
OTHER_LANGUAGES = ("en-US", "es-MX", "en-GB", "hi")

ORIGINAL_EXAMPLE_NAMES = [
    "sales_lead_followup_hindi",
    "recruitment_interview_booking",
    "clinic_appointment_booking",
    "loan_emi_reminder_outbound",
]
NEW_EXAMPLE_NAMES = ["support_order_status_us", "cita_clinica_mx"]


def _domains():
    from impulse_evals import domains

    return domains


def _today(domain, direction):
    every, outbound, _metrics, _turns = TABLE[domain]
    return set(every) | (set(outbound) if direction == "outbound" else set())


# ============================================================================================
# INDIA_PACKS and is_india_locale
# ============================================================================================


def test_india_packs_constant():
    assert tuple(_domains().INDIA_PACKS) == ("dpdp", "rbi_kyc", "trai_dlt")


@pytest.mark.parametrize("language", INDIA_LANGUAGES + ("en-in", "HI-IN", "mr-Deva-IN"))
def test_is_india_locale_true_for_region_in(language):
    assert _domains().is_india_locale(language) is True


@pytest.mark.parametrize("language", OTHER_LANGUAGES + ("in", "id", "en-Latn-US", "es-419"))
def test_is_india_locale_false_for_other_regions(language):
    assert _domains().is_india_locale(language) is False


# ============================================================================================
# DomainDefaults.assertions_for / default_assertions with a language
# ============================================================================================


@pytest.mark.parametrize("language", OTHER_LANGUAGES)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("domain", sorted(TABLE))
def test_assertions_for_non_india_language_has_no_india_pack(domain, direction, language):
    row = _domains().DOMAIN_DEFAULTS[domain]
    assertions = list(row.assertions_for(direction, language=language))
    assert assertions == []   # every auto-added pack is an India pack; hipaa is never auto-added


@pytest.mark.parametrize("language", INDIA_LANGUAGES)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("domain", sorted(TABLE))
def test_assertions_for_india_language_keeps_todays_defaults(domain, direction, language):
    row = _domains().DOMAIN_DEFAULTS[domain]
    assertions = list(row.assertions_for(direction, language=language))
    assert set(assertions) == _today(domain, direction)
    assert assertions == list(row.assertions_for(direction))
    assert len(assertions) == len(set(assertions))


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("domain", sorted(TABLE))
def test_assertions_for_language_none_is_todays_behaviour(domain, direction):
    row = _domains().DOMAIN_DEFAULTS[domain]
    assert list(row.assertions_for(direction, language=None)) == list(row.assertions_for(direction))
    assert set(row.assertions_for(direction, language=None)) == _today(domain, direction)


@pytest.mark.parametrize("language", OTHER_LANGUAGES + INDIA_LANGUAGES + (None,))
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("domain", sorted(TABLE))
def test_default_assertions_follows_the_language_rule(domain, direction, language):
    domains = _domains()
    got = domains.default_assertions(domain, direction, language=language)
    expected = set() if language in OTHER_LANGUAGES else _today(domain, direction)
    assert set(got) == expected
    assert list(got) == list(domains.DOMAIN_DEFAULTS[domain].assertions_for(direction, language=language))


@pytest.mark.parametrize("language", OTHER_LANGUAGES + INDIA_LANGUAGES + (None,))
@pytest.mark.parametrize("domain", sorted(TABLE))
def test_hipaa_never_in_a_default(domain, language):
    domains = _domains()
    for direction in DIRECTIONS:
        assert not [
            rule for rule in domains.default_assertions(domain, direction, language=language)
            if rule.startswith("hipaa.")
        ]


def test_unknown_domain_with_language_uses_general():
    domains = _domains()
    assert domains.default_assertions("insurance", "outbound", language="en-US") == []
    assert domains.default_assertions("insurance", "outbound", language="en-IN") == []


@pytest.mark.parametrize("domain", sorted(TABLE))
def test_metrics_and_max_turns_unchanged(domain):
    domains = _domains()
    _every, _outbound, metrics, max_turns = TABLE[domain]
    assert set(domains.default_metrics(domain)) == set(metrics)
    assert domains.default_max_turns(domain) == max_turns


# ============================================================================================
# impulse generate passes the scenario language
# ============================================================================================


@pytest.fixture
def project(impulse_project, mock_llm, monkeypatch):
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    return impulse_project


def _items():
    """One inbound and one outbound scenario as the LLM writes them, trying to set assertions itself."""
    base = {
        "opening_user_message": "Hello, I have a question.",
        "metrics": [],
        "assertions": ["hipaa.identity_before_phi", "dpdp.*", "rbi_kyc.identity_before_disclosure"],
    }
    return [
        dict(base, name="inbound_case", persona="caller who phoned in", direction="inbound"),
        dict(base, name="outbound_case", persona="caller who was phoned", direction="outbound"),
    ]


def _generate(cli, project, domain, language):
    prompt = Path(project.root) / "agent_prompt.txt"
    prompt.write_text(PROMPT_TEXT, encoding="utf-8")
    return cli.invoke(
        "generate", "--prompt", prompt, "--domain", domain, "--language", language, "--count", "2", "--yes"
    )


def _generated(project):
    from impulse_evals.models import Scenario

    path = Path(project.root) / "scenarios" / "generated.yaml"
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, list)
    return text, data, {item["name"]: Scenario.model_validate(item) for item in data}


@pytest.mark.parametrize("language", ["en-US", "es-MX"])
@pytest.mark.parametrize("domain", ["fsi", "healthcare", "telecom"])
def test_generate_non_india_language_writes_no_pack(project, mock_llm, cli, domain, language):
    mock_llm.queue(yaml.safe_dump(_items(), sort_keys=False))
    result = _generate(cli, project, domain, language)
    assert result.exit_code == 0, cli.text(result)
    text, data, scenarios = _generated(project)
    for prefix in ("dpdp", "rbi_kyc", "trai_dlt", "hipaa"):
        assert prefix not in text, f"{prefix} written for a {language} scenario"
    _every, _outbound, metrics, max_turns = TABLE[domain]
    for item in data:
        assert item["assertions"] == []
    for scenario in scenarios.values():
        assert scenario.language == language
        assert scenario.assertions == []
        assert set(scenario.metrics) == set(metrics)
        assert scenario.max_turns == max_turns


def test_generate_fsi_en_us_writes_empty_assertions(project, mock_llm, cli):
    mock_llm.queue(yaml.safe_dump(_items(), sort_keys=False))
    result = _generate(cli, project, "fsi", "en-US")
    assert result.exit_code == 0, cli.text(result)
    _text, data, _scenarios = _generated(project)
    assert [item["assertions"] for item in data] == [[], []]


@pytest.mark.parametrize("language", ["en-IN", "hi-Latn-IN"])
def test_generate_fsi_india_language_writes_the_india_packs(project, mock_llm, cli, language):
    mock_llm.queue(yaml.safe_dump(_items(), sort_keys=False))
    result = _generate(cli, project, "fsi", language)
    assert result.exit_code == 0, cli.text(result)
    text, _data, scenarios = _generated(project)
    assert "hipaa" not in text
    assert set(scenarios["inbound_case"].assertions) == set(RBI_KYC + DPDP)
    assert set(scenarios["outbound_case"].assertions) == set(RBI_KYC + DPDP + TRAI_DLT)
    for scenario in scenarios.values():
        assert scenario.language == language
        assert scenario.max_turns == 12


# ============================================================================================
# examples/config.yaml: the two global scenarios
# ============================================================================================


@pytest.fixture
def examples_cfg(impulse_project):
    return impulse_project.load(EXAMPLES_CONFIG)


def _scenario(cfg, name):
    matches = [scenario for scenario in cfg.scenarios if scenario.name == name]
    assert len(matches) == 1, f"examples/config.yaml has no single scenario named {name}"
    return matches[0]


def test_examples_config_has_six_scenarios_in_order(examples_cfg):
    assert [scenario.name for scenario in examples_cfg.scenarios] == ORIGINAL_EXAMPLE_NAMES + NEW_EXAMPLE_NAMES


def test_support_order_status_us(examples_cfg):
    scenario = _scenario(examples_cfg, "support_order_status_us")
    assert scenario.language == "en-US"
    assert scenario.domain == "ecommerce"
    assert scenario.direction == "inbound"
    assert scenario.metrics, "an explicit metrics list (an empty list would mean every universal metric)"
    assert "UNIV-023" not in scenario.metrics
    assert scenario.assertions == []
    assert scenario.persona.strip()


def test_cita_clinica_mx(examples_cfg):
    scenario = _scenario(examples_cfg, "cita_clinica_mx")
    assert scenario.language == "es-MX"
    assert scenario.domain == "healthcare"
    assert scenario.direction == "inbound"
    assert "UNIV-023" in scenario.metrics
    assert not [rule for rule in scenario.assertions if rule.startswith(COMPLIANCE_PREFIXES)]
    assert scenario.persona.strip()
    assert (scenario.expected_outcome or "").strip()
    expected = [call for call in scenario.expected_tool_calls if call.name == "book_slot"]
    assert len(expected) == 1
    assert set(expected[0].required_arguments) == {"date", "patient_name", "phone"}
    assert expected[0].max_count == 1


@pytest.mark.parametrize("name", NEW_EXAMPLE_NAMES)
def test_new_examples_have_no_india_pack_and_no_hipaa(name):
    data = yaml.safe_load(EXAMPLES_CONFIG.read_text(encoding="utf-8"))
    items = [item for item in data["scenarios"] if item.get("name") == name]
    assert len(items) == 1
    assertions = items[0].get("assertions") or []
    assert not [rule for rule in assertions if str(rule).startswith(COMPLIANCE_PREFIXES)]


@pytest.mark.phase12
def test_dry_run_of_examples_config_runs_six_scenarios_all_passing(impulse_project, mock_llm, cli):
    result = cli.invoke("run", EXAMPLES_CONFIG, "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0
    run = impulse_project.read_run()
    assert run.total_scenarios == 6
    by_name = {scenario_result.scenario_name: scenario_result for scenario_result in run.scenario_results}
    assert set(by_name) == set(ORIGINAL_EXAMPLE_NAMES + NEW_EXAMPLE_NAMES)
    for name, scenario_result in by_name.items():
        assert scenario_result.error is None, name
        assert scenario_result.passed is True, name
    cita = by_name["cita_clinica_mx"]
    univ025 = [metric for metric in cita.metric_results if metric.metric_id == "UNIV-025"]
    assert len(univ025) == 1
    assert univ025[0].skipped is False
    assert univ025[0].label == "good"
    tool_calls = [call for turn in cita.transcript.turns for call in turn.tool_calls]
    assert [call.name for call in tool_calls] == ["book_slot"]
