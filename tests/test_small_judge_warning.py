# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 13: judge-size and context-length warnings (SPEC.md section 12).

* Small judge: when the primary judge model string contains a parameter count <= 14B
  (case-insensitive pattern ``(\\d+(?:\\.\\d+)?)b\\b``, e.g. ``:7b``) and the run includes any
  ``critical`` assertion, the warning "Judge model {model} looks small (<= 14B parameters). Critical
  verdicts from small judges are unreliable; run impulse calibrate before trusting them." is added
  once to ``RunResult.warnings`` and printed once by ``impulse run``.
* Context length: ``num_ctx`` comes from the role's options (``config.yaml`` ``*_options`` or
  ``config.toml`` ``extra_params``); when the estimated judge prompt (characters / 4) exceeds 80% of it,
  the warning "Judge prompt is about {n} tokens, over 80% of num_ctx={num_ctx} for {model}; the judge
  may truncate the transcript." is added once; no ``num_ctx`` means no warning.

Runs go through ``runner.run`` with the ``fake_adapter`` and ``llm_router`` fixtures (or the CLI).
"""

import re

import pytest

FAST = "ollama/judge-fast"
DEFAULT_PRIMARY = "ollama/judge-primary"
CONSENT = "dpdp.consent_before_pii"  # critical
PURPOSE = "dpdp.purpose_disclosure"  # high
WITHDRAWAL = "dpdp.withdrawal_path"  # medium


def small_judge_warning(model):
    return (
        f"Judge model {model} looks small (<= 14B parameters). Critical verdicts from small judges are "
        "unreliable; run impulse calibrate before trusting them."
    )


def num_ctx_pattern(num_ctx, model):
    return re.compile(
        rf"Judge prompt is about (\d+) tokens, over 80% of num_ctx={num_ctx} for {re.escape(model)}; "
        r"the judge may truncate the transcript\."
    )


def _judge_block(primary=DEFAULT_PRIMARY, fast=FAST, primary_options=None, fast_options=None):
    return {
        "primary_model": primary,
        "fast_model": fast,
        "primary_options": primary_options or {},
        "fast_options": fast_options or {},
    }


def _write(project, assertions, *, judge=None, scenarios=2, direction="inbound", metrics=("UNIV-001",)):
    return project.write_config(
        [
            project.scenario(
                f"call_{number}", assertions=list(assertions), metrics=list(metrics), max_turns=1, direction=direction
            )
            for number in range(1, scenarios + 1)
        ],
        judge=judge or _judge_block(),
    )


async def _run(project, fake_adapter):
    from impulse_evals.runner import run

    return await run(project.load(), adapter_factory=fake_adapter)


def _small_warnings(warnings):
    return [warning for warning in warnings if warning.startswith("Judge model ") and "looks small" in warning]


def _ctx_warnings(warnings):
    return [warning for warning in warnings if "num_ctx=" in warning]


# ============================================================================================
# Small judge
# ============================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["ollama/judge:7b", "ollama/judge:14b", "ollama/judge:1.5b", "ollama/JUDGE:8B", "ollama/judge:3b-instruct"],
)
async def test_small_primary_judge_with_critical_assertion_warns_once(impulse_project, llm_router, fake_adapter, model):
    _write(impulse_project, [CONSENT], judge=_judge_block(primary=model))
    result = await _run(impulse_project, fake_adapter)
    assert result.judge_model == model
    assert result.warnings.count(small_judge_warning(model)) == 1
    assert _small_warnings(result.warnings) == [small_judge_warning(model)]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["ollama/judge:15b", "ollama/judge:14.5b", "ollama/judge:70b", DEFAULT_PRIMARY])
async def test_no_warning_for_judges_over_14b_or_without_a_size(impulse_project, llm_router, fake_adapter, model):
    _write(impulse_project, [CONSENT], judge=_judge_block(primary=model))
    result = await _run(impulse_project, fake_adapter)
    assert _small_warnings(result.warnings) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "assertions",
    [pytest.param([], id="no_assertions"), pytest.param([PURPOSE, WITHDRAWAL], id="high_and_medium_only")],
)
async def test_no_warning_without_a_critical_assertion(impulse_project, llm_router, fake_adapter, assertions):
    _write(impulse_project, assertions, judge=_judge_block(primary="ollama/judge:7b"))
    result = await _run(impulse_project, fake_adapter)
    assert _small_warnings(result.warnings) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rule, direction",
    [
        ("rbi_kyc.identity_before_disclosure", "inbound"),
        ("rbi_kyc.no_credential_request", "inbound"),
        ("trai_dlt.consent_for_outbound", "outbound"),
        ("trai_dlt.honors_opt_out", "outbound"),
        ("hipaa.identity_before_phi", "inbound"),
    ],
)
async def test_any_critical_rule_of_any_pack_triggers_the_warning(
    impulse_project, llm_router, fake_adapter, rule, direction
):
    model = "ollama/judge:7b"
    _write(impulse_project, [rule], judge=_judge_block(primary=model), scenarios=1, direction=direction)
    result = await _run(impulse_project, fake_adapter)
    assert result.warnings.count(small_judge_warning(model)) == 1


@pytest.mark.asyncio
async def test_one_critical_scenario_among_others_is_enough(impulse_project, llm_router, fake_adapter):
    model = "ollama/judge:7b"
    project = impulse_project
    project.write_config(
        [
            project.scenario("plain_call", metrics=["UNIV-001"], max_turns=1),
            project.scenario("consent_call", assertions=[CONSENT], metrics=["UNIV-001"], max_turns=1),
        ],
        judge=_judge_block(primary=model),
    )
    result = await _run(project, fake_adapter)
    assert result.warnings.count(small_judge_warning(model)) == 1


def test_impulse_run_prints_the_small_judge_warning_once(impulse_project, llm_router, cli, normalize_console):
    model = "ollama/judge:7b"
    impulse_project.write_agent()
    config = _write(impulse_project, [CONSENT], judge=_judge_block(primary=model), scenarios=3)
    result = cli.invoke("run", config)
    assert result.exit_code == 0, cli.text(result)
    assert normalize_console(result.output).count(small_judge_warning(model)) == 1
    run = impulse_project.read_run()
    assert run.warnings.count(small_judge_warning(model)) == 1


# ============================================================================================
# num_ctx
# ============================================================================================


@pytest.mark.asyncio
async def test_num_ctx_warning_from_config_yaml_primary_options(impulse_project, llm_router, fake_adapter):
    _write(impulse_project, [CONSENT], judge=_judge_block(primary_options={"num_ctx": 100}))
    result = await _run(impulse_project, fake_adapter)
    matches = [num_ctx_pattern(100, DEFAULT_PRIMARY).fullmatch(warning) for warning in result.warnings]
    matches = [match for match in matches if match]
    assert len(matches) == 1, result.warnings
    assert int(matches[0].group(1)) > 80, "the estimate is over 80% of num_ctx"
    assert len(_ctx_warnings(result.warnings)) == 1


@pytest.mark.asyncio
async def test_num_ctx_warning_from_config_toml_extra_params(impulse_project, llm_router, fake_adapter):
    impulse_project.write_toml("[model_options.primary]\nextra_params = { num_ctx = 100 }\n")
    _write(impulse_project, [CONSENT])
    result = await _run(impulse_project, fake_adapter)
    assert any(num_ctx_pattern(100, DEFAULT_PRIMARY).fullmatch(warning) for warning in result.warnings), result.warnings


@pytest.mark.asyncio
async def test_num_ctx_warning_names_the_fast_model_for_fast_metrics(impulse_project, llm_router, fake_adapter):
    _write(impulse_project, [], judge=_judge_block(fast_options={"num_ctx": 100}), metrics=("UNIV-013",))
    result = await _run(impulse_project, fake_adapter)
    assert any(num_ctx_pattern(100, FAST).fullmatch(warning) for warning in result.warnings), result.warnings
    assert not any(DEFAULT_PRIMARY in warning for warning in _ctx_warnings(result.warnings))


@pytest.mark.asyncio
async def test_no_num_ctx_warning_below_80_percent(impulse_project, llm_router, fake_adapter):
    _write(impulse_project, [CONSENT], judge=_judge_block(primary_options={"num_ctx": 1_000_000}))
    result = await _run(impulse_project, fake_adapter)
    assert _ctx_warnings(result.warnings) == []


@pytest.mark.asyncio
async def test_no_num_ctx_warning_when_not_configured(impulse_project, llm_router, fake_adapter):
    _write(impulse_project, [CONSENT])
    result = await _run(impulse_project, fake_adapter)
    assert _ctx_warnings(result.warnings) == []
    assert not [warning for warning in result.warnings if warning.startswith("Judge prompt is about")]


def test_impulse_run_prints_the_num_ctx_warning_once(impulse_project, llm_router, cli, normalize_console):
    impulse_project.write_agent()
    config = _write(impulse_project, [CONSENT], judge=_judge_block(primary_options={"num_ctx": 100}), scenarios=3)
    result = cli.invoke("run", config)
    assert result.exit_code == 0, cli.text(result)
    assert normalize_console(result.output).count("over 80% of num_ctx=100 for ollama/judge-primary") == 1
