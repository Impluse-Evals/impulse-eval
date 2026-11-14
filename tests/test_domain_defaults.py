# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 9: domain defaults (SPEC.md section 11; METRICS.md section 4.5).

``domains.DOMAIN_DEFAULTS`` (shape recorded in DECISIONS.md "Phase 3 (builder)"): one row per
domain with ``assertions_for(direction)`` (explicit ``pack.rule_id`` list; ``pack.*`` wildcards
already expanded; ``trai_dlt`` only for outbound fsi / telecom scenarios), ``metrics`` (always
included), and ``max_turns`` (12 for fsi and healthcare, else 9). The ``hipaa`` pack is never in a
default, and UNIV-025 is in no default. ``impulse generate`` applies the row to every generated
scenario. An unknown domain is treated as ``general`` with the warning
"Unknown domain {domain}; using general defaults." (``impulse generate --domain`` and ``config.yaml``).
"""

from pathlib import Path

import pytest
import yaml

MODEL_ENV = {
    "IMPULSE_MODEL_PRIMARY": "ollama/judge-primary",
    "IMPULSE_MODEL_FAST": "ollama/judge-fast",
    "IMPULSE_MODEL_SIMULATOR": "ollama/simulator",
}
PROMPT_TEXT = "You are the assistant of Sunrise Store. Answer questions politely and keep replies short."

DPDP = ["dpdp.consent_before_pii", "dpdp.purpose_disclosure", "dpdp.withdrawal_path"]
RBI_KYC = [
    "rbi_kyc.identity_before_disclosure",
    "rbi_kyc.no_credential_request",
    "rbi_kyc.no_unauthorized_commitment",
]
TRAI_DLT = ["trai_dlt.consent_for_outbound", "trai_dlt.honors_opt_out", "trai_dlt.identifies_sender"]
HIPAA = ["hipaa.identity_before_phi", "hipaa.minimum_necessary", "hipaa.no_phi_to_third_party"]
ALL_RULES = set(DPDP + RBI_KYC + TRAI_DLT + HIPAA)
UNIVERSAL_JUDGE = [f"UNIV-{n:03d}" for n in range(1, 24, 2)]  # UNIV-001 .. UNIV-023, not UNIV-025
GENERAL_METRICS = UNIVERSAL_JUDGE + ["VOICE-037", "VOICE-039"]

# domain: (assertions for every scenario, extra assertions for outbound only, metrics, max_turns)
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


def _defaults():
    from impulse_evals.domains import DOMAIN_DEFAULTS

    return DOMAIN_DEFAULTS


def _row(domain):
    return _defaults()[domain]


def _expected_assertions(domain, direction):
    every, outbound, _metrics, _turns = TABLE[domain]
    return set(every) | (set(outbound) if direction == "outbound" else set())


# ============================================================================================
# The table (domains.DOMAIN_DEFAULTS)
# ============================================================================================


def test_table_has_exactly_the_ten_domains():
    assert set(_defaults()) == set(TABLE)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("domain", sorted(TABLE))
def test_default_assertions_per_domain_and_direction(domain, direction):
    assertions = list(_row(domain).assertions_for(direction))
    assert set(assertions) == _expected_assertions(domain, direction)
    assert len(assertions) == len(set(assertions)), f"duplicates in {assertions}"


@pytest.mark.parametrize("domain", sorted(TABLE))
def test_default_metrics_and_max_turns_per_domain(domain):
    _every, _outbound, metrics, max_turns = TABLE[domain]
    row = _row(domain)
    assert set(row.metrics) == set(metrics)
    assert len(row.metrics) == len(set(row.metrics))
    assert row.max_turns == max_turns


@pytest.mark.parametrize("domain", ["fsi", "telecom"])
def test_trai_dlt_only_for_outbound(domain):
    row = _row(domain)
    assert not [rule for rule in row.assertions_for("inbound") if rule.startswith("trai_dlt.")]
    assert set(TRAI_DLT) <= set(row.assertions_for("outbound"))


@pytest.mark.parametrize("domain", sorted(set(TABLE) - {"fsi", "telecom"}))
def test_no_trai_dlt_outside_fsi_and_telecom(domain):
    for direction in DIRECTIONS:
        assert not [rule for rule in _row(domain).assertions_for(direction) if rule.startswith("trai_dlt.")]


@pytest.mark.parametrize("domain", ["fsi", "healthcare", "ecommerce", "recruitment", "telecom", "travel"])
def test_univ023_included(domain):
    assert "UNIV-023" in _row(domain).metrics


@pytest.mark.parametrize("domain", ["edtech", "realestate", "legal"])
def test_univ023_not_auto_included(domain):
    assert "UNIV-023" not in _row(domain).metrics


def test_general_includes_all_universal_judge_metrics_and_voice_037_039():
    metrics = set(_row("general").metrics)
    assert set(UNIVERSAL_JUDGE) <= metrics
    assert {"VOICE-037", "VOICE-039"} <= metrics


@pytest.mark.parametrize("domain", sorted(TABLE))
def test_univ025_in_no_domain_default(domain):
    assert "UNIV-025" not in _row(domain).metrics


@pytest.mark.parametrize("domain", sorted(TABLE))
def test_no_hipaa_rule_in_any_domain_default(domain):
    for direction in DIRECTIONS:
        assert not [rule for rule in _row(domain).assertions_for(direction) if rule.startswith("hipaa.")]


def test_healthcare_default_is_dpdp_only():
    for direction in DIRECTIONS:
        assert set(_row("healthcare").assertions_for(direction)) == set(DPDP)


@pytest.mark.parametrize("domain", sorted(TABLE))
def test_defaults_are_explicit_rule_ids(domain):
    for direction in DIRECTIONS:
        for rule in _row(domain).assertions_for(direction):
            assert "*" not in rule
            assert rule in ALL_RULES, f"{rule} is not an explicit pack.rule_id from METRICS.md section 4"


@pytest.mark.parametrize("domain", ["fsi", "healthcare"])
def test_compliance_domains_use_12_turns(domain):
    assert _row(domain).max_turns == 12


# ============================================================================================
# impulse generate applies the table
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
        "assertions": ["hipaa.identity_before_phi", "dpdp.*"],
    }
    return [
        dict(base, name="inbound_case", persona="caller who phoned in", direction="inbound"),
        dict(base, name="outbound_case", persona="caller who was phoned", direction="outbound"),
    ]


def _generate(cli, project, domain):
    prompt = Path(project.root) / "agent_prompt.txt"
    prompt.write_text(PROMPT_TEXT, encoding="utf-8")
    return cli.invoke(
        "generate", "--prompt", prompt, "--domain", domain, "--language", "en-IN", "--count", "2", "--yes"
    )


def _generated(project):
    from impulse_evals.models import Scenario

    path = Path(project.root) / "scenarios" / "generated.yaml"
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, list)
    return text, {item["name"]: Scenario.model_validate(item) for item in data}


@pytest.mark.parametrize("domain", sorted(TABLE))
def test_generate_applies_the_domain_row(project, mock_llm, cli, domain):
    mock_llm.queue(yaml.safe_dump(_items(), sort_keys=False))
    result = _generate(cli, project, domain)
    assert result.exit_code == 0, cli.text(result)
    text, scenarios = _generated(project)
    assert "*" not in text, "wildcards are expanded before writing"
    assert "hipaa" not in text, "hipaa is never auto-added (and never taken from the LLM)"
    _every, _outbound, metrics, max_turns = TABLE[domain]
    for name, direction in (("inbound_case", "inbound"), ("outbound_case", "outbound")):
        scenario = scenarios[name]
        assert scenario.domain == domain
        assert scenario.direction == direction
        assert set(scenario.assertions) == _expected_assertions(domain, direction)
        assert set(scenario.metrics) == set(metrics)
        assert "UNIV-025" not in scenario.metrics
        assert scenario.max_turns == max_turns


def test_generate_unknown_domain_uses_general_defaults_with_warning(project, mock_llm, cli, normalize_console):
    mock_llm.queue(yaml.safe_dump(_items(), sort_keys=False))
    result = _generate(cli, project, "insurance")
    assert result.exit_code == 0, cli.text(result)
    assert normalize_console(result.output).count("Unknown domain insurance; using general defaults.") == 1
    _text, scenarios = _generated(project)
    for scenario in scenarios.values():
        assert scenario.domain == "general"
        assert scenario.assertions == []
        assert set(scenario.metrics) == set(GENERAL_METRICS)
        assert scenario.max_turns == 9


def test_config_unknown_domain_becomes_general_with_warning(impulse_project):
    project = impulse_project
    project.write_config([project.scenario("odd_domain", domain="insurance")])
    cfg = project.load()
    assert cfg.scenarios[0].domain == "general"
    assert cfg.warnings.count("Unknown domain insurance; using general defaults.") == 1
