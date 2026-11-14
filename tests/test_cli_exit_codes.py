# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 8: ``impulse run`` / ``impulse report`` exit codes and outputs (SPEC.md sections 15-17).

Exit codes: 0 success; 1 when the ``--fail-on`` condition is met; 2 usage or config error.
``--fail-on`` (each level includes everything above it; a scenario ``error`` exits 1 at every level):
* ``critical``: any failed critical assertion, or any scenario error (including a critical check the
  judge could not evaluate);
* ``high`` (the default since fixes round 4): also any failed high assertion, a high assertion with no
  verdict, or any poor UNIV-019 or UNIV-025 (both count as high);
* ``any``: any scenario with ``passed == False``.
``--dry-run`` makes zero LLM calls and exits 0 unless the config is invalid.

Commands run through ``typer.testing.CliRunner`` (``cli`` fixture) in ``tmp_path``; the judge and the
simulator are ``mock_llm`` routed by ``llm_router``. [phase 12] the dry run of ``examples/config.yaml``
needs every compliance pack and metrics/voice.py.
"""

import base64
import io
import json
import wave
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSENT = "dpdp.consent_before_pii"
PURPOSE = "dpdp.purpose_disclosure"
WITHDRAWAL = "dpdp.withdrawal_path"
ALL_DPDP = [CONSENT, PURPOSE, WITHDRAWAL]
PRIVACY_NOTICE = "Impulse sends your system prompt and transcripts to the LLM and voice providers you configured."
CRASHING_AGENT = """
def respond(user_text, history):
    raise RuntimeError("agent crashed while answering")
"""

# case -> exit code expected for --fail-on critical, high, any
MATRIX = {
    "all_pass": {"critical": 0, "high": 0, "any": 0},
    "critical_failure": {"critical": 1, "high": 1, "any": 1},
    "high_failure": {"critical": 0, "high": 1, "any": 1},
    "medium_failure": {"critical": 0, "high": 0, "any": 1},
    "poor_metric": {"critical": 0, "high": 0, "any": 1},
    "poor_univ025": {"critical": 0, "high": 1, "any": 1},
    "scenario_error": {"critical": 1, "high": 1, "any": 1},
    "critical_no_verdict": {"critical": 1, "high": 1, "any": 1},
}


def _setup_case(project, llm_router, case):
    """One scenario with all three dpdp rules and UNIV-013; the case decides what fails."""
    fields = {"assertions": ALL_DPDP, "metrics": ["UNIV-013"], "max_turns": 1}
    if case == "poor_univ025":
        fields["expected_tool_calls"] = [{"name": "send_sms"}]  # the sample agent never calls it
    if case == "scenario_error":
        project.write_agent(CRASHING_AGENT)
    else:
        project.write_agent()  # copy of examples/sample_agent.py
    fail = llm_router.verdict("fail", reason="scripted failure")
    if case == "critical_failure":
        llm_router.on_judge(fail, criterion=CONSENT)
    elif case == "high_failure":
        llm_router.on_judge(fail, criterion=PURPOSE)
    elif case == "medium_failure":
        llm_router.on_judge(fail, criterion=WITHDRAWAL)
    elif case == "poor_metric":
        llm_router.on_judge(fail, criterion="UNIV-013")
    elif case == "critical_no_verdict":
        llm_router.on_judge(None, criterion=CONSENT)
    return project.write_config([project.scenario("matrix_case", **fields)])


# ============================================================================================
# The --fail-on matrix
# ============================================================================================


@pytest.mark.parametrize("level", ["critical", "high", "any"])
@pytest.mark.parametrize("case", sorted(MATRIX))
def test_fail_on_matrix(impulse_project, llm_router, cli, case, level):
    config = _setup_case(impulse_project, llm_router, case)
    result = cli.invoke("run", config, "--fail-on", level)
    assert result.exit_code == MATRIX[case][level], cli.text(result)


# Fixes round 4 (O3): the default level is high (superseded assertion: it was critical, so high_failure gave 0).
@pytest.mark.parametrize(
    "case, expected", [("critical_failure", 1), ("high_failure", 1), ("medium_failure", 0), ("poor_metric", 0), ("scenario_error", 1)]
)
def test_default_fail_on_is_high(impulse_project, llm_router, cli, case, expected):
    config = _setup_case(impulse_project, llm_router, case)
    assert cli.invoke("run", config).exit_code == expected


def test_scenario_error_is_stored_in_run_json(impulse_project, llm_router, cli):
    config = _setup_case(impulse_project, llm_router, "scenario_error")
    assert cli.invoke("run", config, "--fail-on", "any").exit_code == 1
    run = impulse_project.read_run()
    scenario_result = run.scenario_results[0]
    assert "agent crashed while answering" in scenario_result.error
    assert scenario_result.assertion_results == []
    assert scenario_result.metric_results == []


def test_critical_no_verdict_is_stored_in_run_json(impulse_project, llm_router, cli):
    config = _setup_case(impulse_project, llm_router, "critical_no_verdict")
    assert cli.invoke("run", config, "--fail-on", "critical").exit_code == 1
    run = impulse_project.read_run()
    scenario_result = run.scenario_results[0]
    assert scenario_result.error == f"critical check {CONSENT} could not be evaluated: judge returned no verdict"
    assert (
        f"Critical checks not evaluated (judge returned no verdict): matrix_case:{CONSENT}" in run.warnings
    )


def test_config_toml_fail_on_is_the_default_and_the_flag_overrides_it(impulse_project, llm_router, cli):
    config = _setup_case(impulse_project, llm_router, "high_failure")
    impulse_project.write_toml('[defaults]\nreporter = "all"\nfail_on = "high"\n')
    assert cli.invoke("run", config).exit_code == 1
    assert cli.invoke("run", config, "--fail-on", "critical").exit_code == 0


def test_invalid_fail_on_value_is_a_usage_error(impulse_project, llm_router, cli):
    config = _setup_case(impulse_project, llm_router, "all_pass")
    assert cli.invoke("run", config, "--fail-on", "sometimes").exit_code == 2


# ============================================================================================
# Exit code 2: config errors name the field
# ============================================================================================


def _bad_config(project, kind):
    scenario = project.scenario("bad_config")
    if kind == "unknown_metric":
        scenario["metrics"] = ["UNIV-999"]
        return project.write_config([scenario]), "scenarios[0].metrics[0]"
    if kind == "bad_language":
        scenario["language"] = "English"
        return project.write_config([scenario]), "scenarios[0].language"
    if kind == "duplicate_name":
        return project.write_config([scenario, dict(scenario)]), "scenarios[1].name"
    if kind == "wildcard_assertion":
        scenario["assertions"] = ["dpdp.*"]
        return project.write_config([scenario]), "scenarios[0].assertions[0]"
    if kind == "missing_simulator":
        path = project.write_config([scenario])
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        del data["simulator"]
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path, "simulator"
    if kind == "missing_system_prompt":
        path = project.write_config([scenario])
        (project.root / "system_prompt.txt").unlink()
        return path, "agent.system_prompt"
    if kind == "entrypoint_import":
        return (
            project.write_config([scenario], agent={"entrypoint": "impulse_test_no_such_module_4f2a:respond"}),
            "agent.entrypoint",
        )
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_metric", "bad_language", "duplicate_name", "wildcard_assertion",
        "missing_simulator", "missing_system_prompt", "entrypoint_import",
    ],
)
def test_config_error_exits_2_naming_the_field(impulse_project, llm_router, cli, kind):
    config, field = _bad_config(impulse_project, kind)
    result = cli.invoke("run", config)
    assert result.exit_code == 2, cli.text(result)
    assert field in cli.text(result)


def test_missing_config_file_exits_2(impulse_project, cli):
    assert cli.invoke("run", impulse_project.root / "no_such_config.yaml").exit_code == 2


def test_invalid_yaml_exits_2(impulse_project, cli):
    path = impulse_project.root / "config.yaml"
    path.write_text("agent: [unclosed\n", encoding="utf-8")
    assert cli.invoke("run", path).exit_code == 2


# ============================================================================================
# --dry-run
# ============================================================================================


def _dry_run_project(project, **scenario_fields):
    project.write_agent()  # copy of examples/sample_agent.py
    expectation = [{"name": "book_slot", "required_arguments": ["date", "phone"], "max_count": 1}]
    fields = {"assertions": ALL_DPDP, "metrics": ["UNIV-001", "UNIV-013"], "max_turns": 4,
              "expected_tool_calls": expectation}
    fields.update(scenario_fields)
    return project.write_config(
        [
            project.scenario("dry_inbound", **fields),
            project.scenario("dry_outbound", direction="outbound", opening_user_message=None, **fields),
        ],
        judge={"primary_model": "YOUR_PRIMARY_MODEL", "fast_model": "YOUR_FAST_MODEL"},
        simulator={"model": "YOUR_SIMULATOR_MODEL", "persona_style": "realistic", "mode": "cascade"},
    )


def test_dry_run_makes_zero_llm_calls_and_exits_0(impulse_project, mock_llm, cli):
    config = _dry_run_project(impulse_project)
    result = cli.invoke("run", config, "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0
    run = impulse_project.read_run()
    assert run.total_scenarios == 2
    for scenario_result in run.scenario_results:
        assert scenario_result.error is None
        users = [turn.text for turn in scenario_result.transcript.turns if turn.speaker == "user"]
        assert users[-1] == "I'd like to book"
        for score in scenario_result.assertion_results:
            assert (score.result, score.reason) == ("skip", "dry run")
        for metric in scenario_result.metric_results:
            if metric.metric_id == "UNIV-025":
                assert metric.label == "good"
            else:
                assert (metric.skipped, metric.reason) == (True, "dry run")
    assert (impulse_project.only_run_dir() / "report.html").is_file()


def test_dry_run_prints_no_privacy_notice(impulse_project, mock_llm, cli):
    config = _dry_run_project(impulse_project)
    result = cli.invoke("run", config, "--dry-run")
    assert result.exit_code == 0
    assert PRIVACY_NOTICE not in cli.text(result)
    assert not (impulse_project.root / ".impulse" / ".privacy_notice_shown").exists()


def test_dry_run_ignores_fail_on(impulse_project, mock_llm, cli):
    # The sample agent never calls send_sms, so UNIV-025 is poor; --fail-on is not applied in a dry run.
    config = _dry_run_project(impulse_project, expected_tool_calls=[{"name": "send_sms"}])
    result = cli.invoke("run", config, "--dry-run", "--fail-on", "any")
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0
    run = impulse_project.read_run()
    assert all(not scenario_result.passed for scenario_result in run.scenario_results)


def test_dry_run_with_invalid_config_exits_2(impulse_project, mock_llm, cli):
    project = impulse_project
    config = project.write_config([project.scenario("bad", metrics=["UNIV-999"])])
    assert cli.invoke("run", config, "--dry-run").exit_code == 2
    assert mock_llm.call_count == 0


@pytest.mark.parametrize("adapter", ["pipecat", "livekit"])
def test_dry_run_validates_pipecat_and_livekit_configs_without_connecting(impulse_project, mock_llm, cli, adapter):
    project = impulse_project
    agent = {"adapter": adapter}
    if adapter == "pipecat":
        agent["pipecat"] = {"mode": "eval", "url": "ws://localhost:7860"}
    else:
        agent["livekit"] = {"url": "wss://livekit.impulse-test.invalid"}
    config = project.write_config(
        [project.scenario("remote_bot", metrics=["UNIV-013"])],
        agent=agent,
        judge={"primary_model": "YOUR_PRIMARY_MODEL", "fast_model": "YOUR_FAST_MODEL"},
        simulator={"model": "YOUR_SIMULATOR_MODEL", "persona_style": "realistic", "mode": "cascade"},
    )
    result = cli.invoke("run", config, "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0


@pytest.mark.phase12
def test_dry_run_of_examples_config(impulse_project, mock_llm, cli):
    result = cli.invoke("run", REPO_ROOT / "examples" / "config.yaml", "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0
    run = impulse_project.read_run()
    assert run.total_scenarios == 6
    for scenario_result in run.scenario_results:
        assert scenario_result.error is None
        assert all(score.reason == "dry run" for score in scenario_result.assertion_results)


# ============================================================================================
# Privacy notice and run outputs
# ============================================================================================


def test_privacy_notice_on_first_run_only(impulse_project, llm_router, cli):
    config = _setup_case(impulse_project, llm_router, "all_pass")
    first = cli.invoke("run", config)
    assert first.exit_code == 0
    assert PRIVACY_NOTICE in cli.text(first)
    assert (impulse_project.root / ".impulse" / ".privacy_notice_shown").is_file()
    second = cli.invoke("run", config)
    assert second.exit_code == 0
    assert PRIVACY_NOTICE not in cli.text(second)


def test_run_writes_run_json_scenario_files_and_report(impulse_project, llm_router, cli):
    from impulse_evals.models import RunResult, ScenarioResult

    project = impulse_project
    project.write_agent()
    config = project.write_config(
        [
            project.scenario("clinic booking/v2", metrics=["UNIV-013"], max_turns=1),
            project.scenario("plain_call", metrics=["UNIV-013"], max_turns=1),
        ]
    )
    result = cli.invoke("run", config)
    assert result.exit_code == 0, cli.text(result)
    run_dir = project.only_run_dir()
    run = RunResult.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_dir.name == run.run_id
    assert [s.scenario_name for s in run.scenario_results] == ["clinic booking/v2", "plain_call"]
    stored = ScenarioResult.model_validate_json(
        (run_dir / "scenarios" / "clinic_booking_v2.json").read_text(encoding="utf-8")
    )
    assert stored == run.scenario_results[0]
    assert (run_dir / "scenarios" / "plain_call.json").is_file()
    assert (run_dir / "report.html").is_file()


@pytest.mark.parametrize(
    "flags", [["--embed-audio"], ["--no-embed-audio"], ["--concurrency", "1"], ["--concurrency", "3"]]
)
def test_run_accepts_audio_and_concurrency_flags(impulse_project, llm_router, cli, flags):
    config = _setup_case(impulse_project, llm_router, "all_pass")
    result = cli.invoke("run", config, *flags)
    assert result.exit_code == 0, cli.text(result)
    assert (impulse_project.only_run_dir() / "report.html").is_file()


@pytest.mark.parametrize("reporter, html_expected", [("json", False), ("html", True), ("all", True)])
def test_reporter_flag_controls_report_html(impulse_project, llm_router, cli, reporter, html_expected):
    config = _setup_case(impulse_project, llm_router, "all_pass")
    result = cli.invoke("run", config, "--reporter", reporter)
    assert result.exit_code == 0, cli.text(result)
    runs = list((impulse_project.root / ".impulse" / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "report.html").is_file() is html_expected
    if reporter in ("json", "all"):
        assert (runs[0] / "run.json").is_file()


# ============================================================================================
# impulse report RUN_ID (rewrites report.html from run.json; --embed-audio is the default)
# ============================================================================================


def _stored_voice_run(project, run_builder):
    b = run_builder
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(range(256)) * 25)
    wav = buffer.getvalue()
    turns = [
        b.turn("user", "Hello, I need an appointment.", 0),
        b.turn("agent", "Sure, which day?", 900, latency_ms=900.0, audio_path="audio/clinic_booking_t2_agent.wav"),
    ]
    result = b.scenario_result("clinic_booking", transcript=b.transcript("clinic_booking", turns, voice_mode=True))
    run = b.run([result], voice_mode=True, warnings=["Pipecat bots serve one client at a time; running scenarios one by one."])
    path = b.write_run(run, project.runs_dir)
    (path.parent / "audio").mkdir()
    (path.parent / "audio" / "clinic_booking_t2_agent.wav").write_bytes(wav)
    return run, path, wav


def test_report_embeds_audio_by_default_and_keeps_run_json(impulse_project, run_builder, cli):
    run, path, wav = _stored_voice_run(impulse_project, run_builder)
    before = path.read_bytes()
    result = cli.invoke("report", run.run_id)
    assert result.exit_code == 0, cli.text(result)
    html = (path.parent / "report.html").read_text(encoding="utf-8")
    assert "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii") in html
    assert "Pipecat bots serve one client at a time; running scenarios one by one." in html
    assert path.read_bytes() == before


def test_report_no_embed_audio_links_files(impulse_project, run_builder, cli):
    run, path, _ = _stored_voice_run(impulse_project, run_builder)
    result = cli.invoke("report", run.run_id, "--no-embed-audio")
    assert result.exit_code == 0, cli.text(result)
    html = (path.parent / "report.html").read_text(encoding="utf-8")
    assert "audio/clinic_booking_t2_agent.wav" in html
    assert "data:audio/wav;base64," not in html
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == run.run_id


def test_report_unknown_run_exits_2(impulse_project, cli):
    assert cli.invoke("report", "20261003T120000Z-cccccc").exit_code == 2


# ============================================================================================
# Audit batch 2 (AUDIT.md finding 9): a dry run is marked by the run warning DRY_RUN_WARNING, so the
# console summary, run.json, report.html (header warnings), and `impulse report` on a stored dry
# run all show that its results are not verdicts. A real run's report never shows it.
# ============================================================================================

DRY_RUN_WARNING = "Dry run: judge metrics and assertions were skipped; results are not verdicts."


def _report_header(run_dir):
    html = (Path(run_dir) / "report.html").read_text(encoding="utf-8")
    return html.split("</header>", 1)[0]


def test_dry_run_console_run_json_and_report_show_the_dry_run_warning(impulse_project, mock_llm, cli):
    config = _dry_run_project(impulse_project)
    result = cli.invoke("run", config, "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0
    text = cli.text(result)
    assert "Warnings:" in text, text
    assert f"- {DRY_RUN_WARNING}" in text, text
    assert text.count(DRY_RUN_WARNING) == 1, text
    run = impulse_project.read_run()
    assert run.warnings.count(DRY_RUN_WARNING) == 1, run.warnings
    header = _report_header(impulse_project.only_run_dir())
    assert "<h3>Warnings</h3>" in header
    assert DRY_RUN_WARNING in header


def test_impulse_report_on_a_stored_dry_run_shows_the_dry_run_warning(impulse_project, mock_llm, cli):
    config = _dry_run_project(impulse_project)
    assert cli.invoke("run", config, "--dry-run", "--reporter", "json").exit_code == 0
    run = impulse_project.read_run()
    run_dir = impulse_project.only_run_dir()
    assert not (run_dir / "report.html").exists()
    result = cli.invoke("report", run.run_id)
    assert result.exit_code == 0, cli.text(result)
    header = _report_header(run_dir)
    assert DRY_RUN_WARNING in header, "report.html header warnings do not show the dry run"


def test_real_run_report_shows_no_dry_run_warning(impulse_project, llm_router, cli):
    # Regression guard (passes before the fix too).
    config = _setup_case(impulse_project, llm_router, "all_pass")
    result = cli.invoke("run", config)
    assert result.exit_code == 0, cli.text(result)
    assert DRY_RUN_WARNING not in cli.text(result)
    run = impulse_project.read_run()
    assert DRY_RUN_WARNING not in run.warnings
    assert DRY_RUN_WARNING not in _report_header(impulse_project.only_run_dir())


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 30): redaction also catches the masked and encoded forms of a
# known key value: the URL-encoded form (urllib.parse.quote(value, safe="")), the base64 text of the
# value at each of the three byte alignments (only the characters that depend on the value's bytes
# alone, so "Basic base64(user:key)" loses the key part), and the provider-masked form (a prefix of
# at least 3 characters, 3 or more "*", a suffix of at least 2 characters; values of 12+ characters).
# Checked through what `impulse run` and `impulse doctor` print and write.
# ============================================================================================

import base64 as _b64  # noqa: E402
import urllib.parse  # noqa: E402

F30_KEY = "sk-test-0123456789abcdef"            # OPENAI_API_KEY (a standard key variable)
F30_SPECIAL_KEY = "sk-test/ab+cd=ef0123456789"  # GEMINI_API_KEY; URL-reserved characters
F30_BASIC_USER = "impulse-user:"                 # 13 bytes: the key starts at byte alignment 1
F30_PROXY_USER = "user:"                         # 5 bytes: the key starts at byte alignment 2


def _b64_key_part(value, offset):
    """The base64 characters that depend only on ``value``'s bytes when ``value`` starts ``offset``
    bytes (0, 1 or 2) into a 3-byte group."""
    text = _b64.b64encode(b"\0" * offset + value.encode()).decode()
    start = -(-8 * offset // 6)
    end = (8 * offset + 8 * len(value.encode())) // 6
    return text[start:end]


F30_MASKED = ["sk-test-************cdef", "sk-t***ef"]
F30_QUOTED = urllib.parse.quote(F30_SPECIAL_KEY, safe="")
F30_BASIC = _b64.b64encode((F30_BASIC_USER + F30_KEY).encode()).decode()
F30_PROXY = _b64.b64encode((F30_PROXY_USER + F30_SPECIAL_KEY).encode()).decode()
F30_RAW = _b64.b64encode(F30_KEY.encode()).decode()
# A surviving run of any of these means a known key leaked in a masked or encoded form (the base64
# parts are trimmed by two characters at each end so the check does not depend on the boundary).
F30_LEAK_FORMS = F30_MASKED + [
    F30_QUOTED,
    _b64_key_part(F30_KEY, 0)[2:-2],
    _b64_key_part(F30_KEY, 1)[2:-2],
    _b64_key_part(F30_SPECIAL_KEY, 2)[2:-2],
]
F30_MESSAGE = (
    f"Incorrect API key provided: {F30_MASKED[0]} (also {F30_MASKED[1]}); GET /v1/models?key={F30_QUOTED}; "
    f"Authorization: Basic {F30_BASIC}; Proxy-Authorization: Basic {F30_PROXY}; X-Raw: {F30_RAW}"
)
F30_DOCTOR_MESSAGE = f"Incorrect API key provided: {F30_MASKED[0]}; key={F30_QUOTED}; Basic {F30_BASIC}"


def _f30_leaks(text):
    return [form for form in F30_LEAK_FORMS if form in text]


def _f30_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", F30_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", F30_SPECIAL_KEY)


def test_f30_probe_values_are_what_the_tests_assume():
    # Sanity of the test data itself (passes before the fix too).
    assert F30_QUOTED != F30_SPECIAL_KEY
    assert _b64_key_part(F30_KEY, 1) in F30_BASIC
    assert _b64_key_part(F30_SPECIAL_KEY, 2) in F30_PROXY
    assert _b64_key_part(F30_KEY, 0) == F30_RAW   # 24 bytes: every character depends on the key
    for masked in F30_MASKED:
        prefix, _, suffix = masked.partition("***")
        assert F30_KEY.startswith(prefix) and F30_KEY.endswith(suffix.lstrip("*"))
    assert len(F30_DOCTOR_MESSAGE) < 200   # the doctor row keeps 300 characters: nothing is cut off


def test_run_redacts_masked_and_encoded_key_forms_from_a_scenario_error(impulse_project, llm_router, cli, monkeypatch):
    _f30_keys(monkeypatch)
    project = impulse_project
    project.write_agent(
        f"""
        def respond(user_text, history):
            raise RuntimeError({F30_MESSAGE!r})
        """
    )
    config = project.write_config([project.scenario("leaky_agent", metrics=["UNIV-013"], max_turns=1)])
    result = cli.invoke("run", config, "--fail-on", "any")
    assert result.exit_code == 1, cli.text(result)
    assert _f30_leaks(result.output) == [], result.output
    run = project.read_run()
    error = run.scenario_results[0].error
    assert error is not None and "Incorrect API key provided: [REDACTED]" in error, error
    assert "GET /v1/models?key=[REDACTED];" in error, error
    assert f"Authorization: Basic {F30_BASIC[:16]}" in error, error   # the user part is kept
    assert "X-Raw: [REDACTED]" in error, error
    for path in sorted(project.only_run_dir().rglob("*")):
        if path.is_file():
            assert _f30_leaks(path.read_text(encoding="utf-8")) == [], path.name


def test_run_still_redacts_the_plain_key(impulse_project, llm_router, cli, monkeypatch):
    # Regression guard (passes before the fix too): the plain value is still replaced.
    _f30_keys(monkeypatch)
    project = impulse_project
    project.write_agent(
        f"""
        def respond(user_text, history):
            raise RuntimeError("Incorrect API key provided: " + {F30_KEY!r} + " and " + {F30_SPECIAL_KEY!r})
        """
    )
    config = project.write_config([project.scenario("plain_leak", metrics=["UNIV-013"], max_turns=1)])
    result = cli.invoke("run", config, "--fail-on", "any")
    assert result.exit_code == 1, cli.text(result)
    assert F30_KEY not in result.output and F30_SPECIAL_KEY not in result.output
    error = project.read_run().scenario_results[0].error
    assert "Incorrect API key provided: [REDACTED] and [REDACTED]" in error, error


def test_doctor_redacts_masked_and_encoded_key_forms_from_a_failed_ping(impulse_project, mock_llm, cli, monkeypatch):
    _f30_keys(monkeypatch)

    def responder(call):
        raise RuntimeError(F30_DOCTOR_MESSAGE)

    mock_llm.responder = responder
    impulse_project.write_agent()
    config = impulse_project.write_config([impulse_project.scenario("doctor_probe", metrics=["UNIV-013"])])
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2, cli.text(result)
    text = cli.text(result)
    assert "not reachable: RuntimeError: Incorrect API key provided: [REDACTED]; key=[REDACTED];" in text, text
    assert _f30_leaks(result.output) == [], result.output


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 31, re-verified on a realistic path): a key resolved from a
# provider-specific variable (GROQ_API_KEY, set only in the environment, named by no config entry)
# is redacted everywhere. Realistic trigger: the user's agent module fails at import (e.g. a warm-up
# LiteLLM call raising an authentication error that echoes the key); the generic adapter wraps it in a
# ConfigError ("cannot import module ..."), which `impulse run` (exit 2) and the doctor's "agent
# entrypoint" row print. Fix: config.secret_values also includes the value of every environment
# variable whose name ends in _API_KEY, and the runner redacts a ConfigError message with its full
# redactor (which also knows the keys resolved for the roles) before re-raising it (same field).
# ============================================================================================

GROQ_KEY = "gsk_impulse_test_SECRET_groq_31aa77"
GROQ_MODEL = "groq/impulse-test-model"
SNOWFLAKE_TOKEN = "impulse-test-SECRET-snowflake-jwt-3131"
SNOWFLAKE_MODEL = "snowflake/impulse-test-model"
IMPORT_TIME_KEY_ECHO = """
import os

# A warm-up call at import time that fails with an error echoing the key it used.
raise RuntimeError("GroqException - AuthenticationError: Invalid API Key: " + os.environ[{name!r}])


def respond(user_text, history):
    return "never reached"
"""
TURN_TIME_KEY_ECHO = """
import os


def respond(user_text, history):
    raise RuntimeError("GroqException - AuthenticationError: Invalid API Key: " + os.environ["GROQ_API_KEY"])
"""


def _provider_project(project, model, agent_code):
    project.write_agent(agent_code)
    return project.write_config(
        [project.scenario("provider_case", metrics=["UNIV-013"], max_turns=1)],
        judge={"primary_model": model, "fast_model": model, "primary_options": {}, "fast_options": {}},
        simulator={"model": model, "persona_style": "realistic", "mode": "cascade", "options": {}},
    )


def test_run_never_prints_a_provider_specific_key_from_an_agent_import_error(impulse_project, llm_router, cli, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    config = _provider_project(impulse_project, GROQ_MODEL, IMPORT_TIME_KEY_ECHO.format(name="GROQ_API_KEY"))
    result = cli.invoke("run", config)
    assert result.exit_code == 2, cli.text(result)
    text = cli.text(result)
    assert "cannot import module" in text, text
    assert "Invalid API Key: [REDACTED]" in text, text
    assert GROQ_KEY not in result.output
    assert llm_router.mock.calls == []


def test_doctor_never_prints_a_provider_specific_key_from_an_agent_import_error(impulse_project, mock_llm, cli, monkeypatch):
    mock_llm.set_default("OK")
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    config = _provider_project(impulse_project, GROQ_MODEL, IMPORT_TIME_KEY_ECHO.format(name="GROQ_API_KEY"))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2, cli.text(result)
    text = cli.text(result)
    assert "agent entrypoint" in text and "cannot import module" in text, text
    assert "Invalid API Key: [REDACTED]" in text, text
    assert GROQ_KEY not in result.output


def test_run_never_prints_a_role_key_only_the_runner_resolved(impulse_project, llm_router, cli, monkeypatch):
    # SNOWFLAKE_JWT is not a *_API_KEY name and no config entry names it: only the runner, which
    # resolved it for the roles, knows the value, so the runner must redact the ConfigError itself.
    monkeypatch.setenv("SNOWFLAKE_JWT", SNOWFLAKE_TOKEN)
    config = _provider_project(impulse_project, SNOWFLAKE_MODEL, IMPORT_TIME_KEY_ECHO.format(name="SNOWFLAKE_JWT"))
    result = cli.invoke("run", config)
    assert result.exit_code == 2, cli.text(result)
    text = cli.text(result)
    assert "cannot import module" in text and "Invalid API Key: [REDACTED]" in text, text
    assert SNOWFLAKE_TOKEN not in result.output
    assert llm_router.mock.calls == []


def test_unexpected_error_never_prints_a_provider_specific_key(impulse_project, llm_router, cli, monkeypatch):
    # The CLI's last line of defence ("unexpected error", exit 2) knows every *_API_KEY value too.
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    config = _provider_project(impulse_project, GROQ_MODEL, TURN_TIME_KEY_ECHO)
    from impulse_evals import runner

    async def exploding_run(*args, **kwargs):
        raise RuntimeError(f"provider said: invalid key {GROQ_KEY}")

    monkeypatch.setattr(runner, "run", exploding_run)
    result = cli.invoke("run", config)
    assert result.exit_code == 2, cli.text(result)
    text = cli.text(result)
    assert "unexpected error" in text and "invalid key [REDACTED]" in text, text
    assert GROQ_KEY not in result.output


def test_turn_time_provider_key_echo_is_redacted(impulse_project, llm_router, cli, monkeypatch):
    # Regression guard (passes before the fix too): the runner's redactor knows the resolved key.
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    config = _provider_project(impulse_project, GROQ_MODEL, TURN_TIME_KEY_ECHO)
    result = cli.invoke("run", config, "--fail-on", "any")
    assert result.exit_code == 1, cli.text(result)
    assert GROQ_KEY not in result.output
    run_text = (impulse_project.only_run_dir() / "run.json").read_text(encoding="utf-8")
    assert GROQ_KEY not in run_text and "Invalid API Key: [REDACTED]" in run_text


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 43): LiteLLM's coloured "Give Feedback / Get Help" banner never
# reaches Impulse output. LiteLLM prints it to stdout from its exception mapping on ANY failed call
# unless litellm.suppress_debug_info is True. Every Impulse module that imports litellm sets
# litellm.suppress_debug_info = True right after the import, and the doctor / init ping runs under
# contextlib.redirect_stdout / redirect_stderr. The flag is forced back to False before each test
# (monkeypatch restores it), so each test sees what the command itself sets.
# ============================================================================================

import sys  # noqa: E402

LITELLM_BANNER = "Give Feedback / Get Help"
LITELLM_BANNER_LINE = f"\033[1;31m{LITELLM_BANNER}: https://github.com/BerriAI/litellm/issues/new\033[0m"
LITELLM_INFO_LINE = "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'."


def _print_banner(stream):
    print(file=stream)
    print(LITELLM_BANNER_LINE, file=stream)
    print(LITELLM_INFO_LINE, file=stream)
    print(file=stream)


def _print_litellm_banner_then_fail(call):
    """Prints the banner to stdout and stderr whatever the flag says, then fails."""
    _print_banner(sys.stdout)
    _print_banner(sys.stderr)
    raise ConnectionError("model server is not reachable")


def _litellm_style_failure(call):
    """What litellm's exception mapping does: the banner on stdout unless suppress_debug_info, then raise."""
    import litellm

    if litellm.suppress_debug_info is False:
        _print_banner(sys.stdout)
    raise ConnectionError("model server is not reachable")


@pytest.fixture
def litellm_debug_info_on(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    return litellm


def test_failed_doctor_ping_prints_no_litellm_banner(impulse_project, mock_llm, cli, litellm_debug_info_on):
    mock_llm.responder = _print_litellm_banner_then_fail
    impulse_project.write_agent()
    config = impulse_project.write_config([impulse_project.scenario("doctor_probe", metrics=["UNIV-013"])])
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2, cli.text(result)
    assert "not reachable: ConnectionError: model server is not reachable" in cli.text(result)
    assert LITELLM_BANNER not in result.output, result.output
    assert "LiteLLM.Info" not in result.output, result.output


def test_doctor_sets_litellm_suppress_debug_info(impulse_project, mock_llm, cli, litellm_debug_info_on):
    mock_llm.set_default("OK")
    impulse_project.write_agent()
    config = impulse_project.write_config([impulse_project.scenario("doctor_probe", metrics=["UNIV-013"])])
    result = cli.invoke("doctor", config)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count >= 1, "the pings ran"
    assert litellm_debug_info_on.suppress_debug_info is True


def test_failed_judge_call_in_a_run_prints_no_litellm_banner(impulse_project, llm_router, cli, litellm_debug_info_on):
    previous = llm_router.mock.responder

    def responder(call):
        if llm_router.kind(call) == "judge":
            return _litellm_style_failure(call)
        return previous(call)

    llm_router.mock.responder = responder
    impulse_project.write_agent()
    config = impulse_project.write_config([impulse_project.scenario("judge_down", metrics=["UNIV-013"], max_turns=1)])
    result = cli.invoke("run", config, "--fail-on", "any")
    assert result.exit_code == 1, cli.text(result)
    assert llm_router.calls("judge"), "the judge was called"
    assert "model server is not reachable" in cli.text(result)   # the failure itself is still reported
    assert LITELLM_BANNER not in result.output, result.output
    assert "LiteLLM.Info" not in result.output, result.output


def test_failed_generate_call_prints_no_litellm_banner(impulse_project, mock_llm, cli, monkeypatch, litellm_debug_info_on):
    # The audit saw the banner after an `impulse generate` failure too.
    monkeypatch.setenv("IMPULSE_MODEL_PRIMARY", "ollama/judge-primary")
    prompt = impulse_project.root / "agent_prompt.txt"
    prompt.write_text(
        "You are the booking assistant of a clinic. Book doctor appointments for patients.", encoding="utf-8"
    )
    mock_llm.responder = _litellm_style_failure
    result = cli.invoke(
        "generate", "--prompt", prompt, "--domain", "healthcare", "--language", "en-IN", "--count", "2", "--yes"
    )
    assert result.exit_code == 1, cli.text(result)
    assert mock_llm.call_count >= 1, "the primary model was called"
    assert "generation failed" in cli.text(result)
    assert LITELLM_BANNER not in result.output, result.output
    assert "LiteLLM.Info" not in result.output, result.output


# ============================================================================================
# Audit batch 3 (AUDIT.md finding 31, extended scope): every key the CLI resolves for a role (the
# doctor's model rows and pings, the init pings, and the keys runner.run resolves) is registered
# with the CLI's redactor, whatever the variable's name, in addition to the *_API_KEY environment
# rule. Here the key variable is not a *_API_KEY name (snowflake reads SNOWFLAKE_JWT, scaleway
# SCW_SECRET_KEY), it is set only in the environment, and no config entry names it: only the
# doctor's own key resolution knows the value when the "agent entrypoint" row prints the import error.
# ============================================================================================

NON_API_KEY_PROVIDERS = [
    pytest.param("snowflake/impulse-test-model", "SNOWFLAKE_JWT", "impulse-test-SECRET-snowflake-doctor-3141", id="snowflake"),
    pytest.param("scaleway/impulse-test-model", "SCW_SECRET_KEY", "impulse-test-SECRET-scaleway-doctor-2718", id="scaleway"),
]


@pytest.mark.parametrize("model, key_name, secret", NON_API_KEY_PROVIDERS)
def test_doctor_never_prints_a_role_key_named_by_no_api_key_variable(
    impulse_project, mock_llm, cli, monkeypatch, model, key_name, secret
):
    mock_llm.set_default("OK")
    monkeypatch.setenv(key_name, secret)
    config = _provider_project(impulse_project, model, IMPORT_TIME_KEY_ECHO.format(name=key_name))
    result = cli.invoke("doctor", config)
    assert result.exit_code == 2, cli.text(result)
    text = cli.text(result)
    pings = [call for call in mock_llm.calls if call["model"] == model]
    assert pings and all(call["kwargs"].get("api_key") == secret for call in pings), "the roles resolved the key"
    assert "agent entrypoint" in text and "cannot import module" in text, text
    assert "Invalid API Key: [REDACTED]" in text, text
    assert secret not in result.output


# ============================================================================================
# Audit batch 3 review fix-ups (AUDIT.md finding 31, dry run): `impulse run --dry-run` also
# registers the keys runner.run would resolve for the roles (runner.resolved_role_keys: offline, no
# LLM call, never raises) with the CLI's redactor, so a generic agent that fails at import with an
# error echoing a role key whose variable is not a *_API_KEY name (SNOWFLAKE_JWT, SCW_SECRET_KEY)
# prints "[REDACTED]" instead of the key (exit 2, as any config error).
# ============================================================================================

DRY_RUN_NON_API_KEY_PROVIDERS = [
    pytest.param("snowflake/impulse-test-model", "SNOWFLAKE_JWT", "impulse-test-SECRET-snowflake-dry-run-1618", id="snowflake"),
    pytest.param("scaleway/impulse-test-model", "SCW_SECRET_KEY", "impulse-test-SECRET-scaleway-dry-run-1414", id="scaleway"),
]


@pytest.mark.parametrize("model, key_name, secret", DRY_RUN_NON_API_KEY_PROVIDERS)
def test_dry_run_never_prints_a_role_key_named_by_no_api_key_variable(
    impulse_project, mock_llm, cli, monkeypatch, model, key_name, secret
):
    monkeypatch.setenv(key_name, secret)
    config = _provider_project(impulse_project, model, IMPORT_TIME_KEY_ECHO.format(name=key_name))
    result = cli.invoke("run", config, "--dry-run")
    text = cli.text(result)
    assert result.exit_code == 2, text
    assert "cannot import module" in text, text
    assert secret not in result.output, result.output
    assert secret not in "".join(result.output.split())   # not even wrapped across lines
    assert "Invalid API Key: [REDACTED]" in text, text
    assert mock_llm.call_count == 0   # a dry run makes no LLM call


@pytest.mark.parametrize("model, key_name, secret", DRY_RUN_NON_API_KEY_PROVIDERS)
def test_dry_run_with_an_importable_agent_still_exits_0(impulse_project, mock_llm, cli, monkeypatch, model, key_name, secret):
    # Regression guard (passes before the fix too): learning the role keys changes nothing else in a
    # dry run (no LLM call, exit 0, the key never printed).
    monkeypatch.setenv(key_name, secret)
    config = _provider_project(impulse_project, model, "def respond(user_text, history):\n    return 'fine'\n")
    result = cli.invoke("run", config, "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    assert secret not in result.output
    assert mock_llm.call_count == 0


# ============================================================================================
# Audit batch 3 round 2 (AUDIT.md finding 31, dry run; SPEC.md section 17): a dry run never imports
# LiteLLM, so it resolves no role key through LiteLLM. Its redactor instead knows the environment
# values of every variable in Impulse's own static key tables (standard names, alias targets,
# explicit names such as SNOWFLAKE_JWT, LLAMA_API_KEY, LAMBDA_API_KEY) and of every variable whose
# name ends in _API_KEY, _SECRET_KEY or _JWT; the dry run's scenario errors and warnings, and what
# the CLI prints, use it. (a) A generic agent whose respond() (not its import) raises with a role key
# read from SNOWFLAKE_JWT or SCW_SECRET_KEY: the dry run exits 0, prints "[REDACTED]", and the key is
# in none of the console output, run.json, scenarios/*.json and report.html. (b) In a fresh
# interpreter (socket guard), `impulse run --dry-run` on a config with hosted models (groq/,
# snowflake/), their keys set, exits 0 and LiteLLM is not in sys.modules afterwards.
# ============================================================================================

import os  # noqa: E402
import subprocess  # noqa: E402

DRY_RUN_RESPOND_KEY_ECHO = """
import os


def respond(user_text, history):
    raise RuntimeError("backend said: Invalid API Key: " + os.environ[{name!r}])
"""
DRY_RUN_RESPOND_PROVIDERS = [
    pytest.param("snowflake/impulse-test-model", "SNOWFLAKE_JWT", "impulse-test-SECRET-snowflake-dry-respond-5353", id="snowflake"),
    pytest.param("scaleway/impulse-test-model", "SCW_SECRET_KEY", "impulse-test-SECRET-scaleway-dry-respond-5454", id="scaleway"),
]


@pytest.mark.parametrize("model, key_name, secret", DRY_RUN_RESPOND_PROVIDERS)
def test_dry_run_never_prints_or_writes_a_role_key_echoed_by_respond(
    impulse_project, mock_llm, cli, monkeypatch, model, key_name, secret
):
    monkeypatch.setenv(key_name, secret)
    config = _provider_project(impulse_project, model, DRY_RUN_RESPOND_KEY_ECHO.format(name=key_name))
    result = cli.invoke("run", config, "--dry-run")
    text = cli.text(result)
    assert result.exit_code == 0, text   # a dry run exits 0 unless the config is invalid
    assert "Invalid API Key: [REDACTED]" in text, text
    assert secret not in result.output, result.output
    assert secret not in "".join(result.output.split())   # not even wrapped across lines
    run_dir = impulse_project.only_run_dir()
    written = {path.relative_to(run_dir).as_posix(): path for path in run_dir.rglob("*") if path.is_file()}
    assert "run.json" in written and "report.html" in written, sorted(written)
    assert any(name.startswith("scenarios/") and name.endswith(".json") for name in written), sorted(written)
    leaking = [name for name, path in written.items() if secret.encode() in path.read_bytes()]
    assert leaking == [], f"files holding the key: {leaking}"
    error = impulse_project.read_run(run_dir).scenario_results[0].error
    assert error == "backend said: Invalid API Key: [REDACTED]", error
    assert mock_llm.call_count == 0   # a dry run makes no LLM call


DRY_RUN_SUBPROCESS_GUARD = r"""
import socket
attempts = []
_real_connect = socket.socket.connect
_real_getaddrinfo = socket.getaddrinfo


def _local(host):
    return host in (None, "", "localhost", "::1") or (isinstance(host, str) and host.startswith("127."))


def _connect(self, address):
    if isinstance(address, tuple) and address and not _local(address[0]):
        attempts.append("connect(%r)" % (address,))
        raise OSError("network blocked by the test")
    return _real_connect(self, address)


def _getaddrinfo(host, *args, **kwargs):
    if not _local(host):
        attempts.append("getaddrinfo(%r)" % (host,))
        raise socket.gaierror("network blocked by the test")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _connect
socket.getaddrinfo = _getaddrinfo
"""
DRY_RUN_SUBPROCESS_BODY = """
import json, sys
sys.path.insert(0, {repo!r})
from typer.testing import CliRunner
from impulse_evals.cli import app
result = CliRunner().invoke(app, ["run", "config.yaml", "--dry-run"])
print("RESULT=" + json.dumps({{
    "exit": result.exit_code,
    "litellm_modules": sorted(name for name in sys.modules if name == "litellm" or name.startswith("litellm.")),
    "attempts": attempts,
    "exception": None if result.exception is None or isinstance(result.exception, SystemExit) else repr(result.exception),
    "output": result.output[-3000:],
}}))
"""
DRY_RUN_HOSTED_KEYS = {
    "GROQ_API_KEY": "gsk_impulse_test_SECRET_dry_import_8080",
    "SNOWFLAKE_JWT": "impulse-test-SECRET-snowflake-dry-import-8181",
}


def test_cli_dry_run_never_imports_litellm(impulse_project):
    project = impulse_project
    project.write_agent("def respond(user_text, history):\n    return 'Sure, which day suits you?'\n")
    project.write_config(
        [project.scenario("dry_import", metrics=["UNIV-013"], max_turns=1)],
        judge={
            "primary_model": "groq/impulse-test-model",
            "fast_model": "snowflake/impulse-test-model",
            "primary_options": {},
            "fast_options": {},
        },
        simulator={"model": "groq/impulse-test-model", "persona_style": "realistic", "mode": "cascade", "options": {}},
    )
    env = dict(os.environ)
    env.update(DRY_RUN_HOSTED_KEYS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"   # were LiteLLM imported, it would not fetch its cost map
    code = DRY_RUN_SUBPROCESS_GUARD + "\n" + DRY_RUN_SUBPROCESS_BODY.format(repo=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(project.root), env=env, capture_output=True, text=True, timeout=300,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT=")]
    assert proc.returncode == 0 and lines, f"exit {proc.returncode}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
    report = json.loads(lines[-1][len("RESULT="):])
    assert report["exception"] is None, report
    assert report["exit"] == 0, report["output"]
    assert report["attempts"] == [], report["attempts"]
    assert report["litellm_modules"] == [], (
        f"impulse run --dry-run imported LiteLLM ({report['litellm_modules'][:5]}); a dry run never does"
    )
    for secret in DRY_RUN_HOSTED_KEYS.values():
        assert secret not in report["output"]


# ============================================================================================
# Maintainer decision 2026-10-21, item 6 (AUDIT-FIXES.md, logged): `python -m impulse_evals` runs the
# same CLI as the `impulse` console script (impulse_evals/__main__.py calls the CLI's app with
# prog_name="impulse" under `if __name__ == "__main__":`), so help and usage texts read
# "Usage: impulse ..." and exit codes are the console script's. Importing impulse_evals.__main__
# runs nothing. Each check runs in a fresh interpreter whose current directory is in tmp_path; a
# sitecustomize module first on PYTHONPATH refuses and logs every non-loopback connection.
# ============================================================================================

import re  # noqa: E402

CLI_COMMANDS = ["init", "doctor", "run", "report", "compare", "generate", "convert-failure", "calibrate"]
OFFLINE_SITECUSTOMIZE = r'''
import os
import socket

_log = os.environ["IMPULSE_TEST_NETWORK_LOG"]
open(os.environ["IMPULSE_TEST_GUARD_LOADED"], "w").close()
_real_connect = socket.socket.connect
_real_getaddrinfo = socket.getaddrinfo


def _local(host):
    return host in (None, "", "localhost", "::1") or (isinstance(host, str) and host.startswith("127."))


def _record(text):
    with open(_log, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _connect(self, address):
    if isinstance(address, tuple) and address and not _local(address[0]):
        _record("connect(%r)" % (address,))
        raise OSError("network blocked by the test")
    return _real_connect(self, address)


def _getaddrinfo(host, *args, **kwargs):
    if not _local(host):
        _record("getaddrinfo(%r)" % (host,))
        raise socket.gaierror("network blocked by the test")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _connect
socket.getaddrinfo = _getaddrinfo
'''
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _fresh_impulse(tmp_path, cwd, *args, code=None):
    """``python -m impulse_evals *args`` (or ``python -c code``) in a fresh interpreter with ``cwd``
    as the current directory, the repository on PYTHONPATH, and the socket guard loaded. Returns
    ``(returncode, stdout, stderr, network attempts)``; output decoded as UTF-8, ANSI codes removed."""
    guard = tmp_path / "offline-site"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(OFFLINE_SITECUSTOMIZE, encoding="utf-8")
    log, loaded = tmp_path / "network-attempts.log", tmp_path / "guard-loaded"
    for path in (log, loaded):
        path.unlink(missing_ok=True)
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(guard), str(REPO_ROOT)] + ([inherited] if inherited else []))
    env.update(
        PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", NO_COLOR="1", LITELLM_LOCAL_MODEL_COST_MAP="True",
        IMPULSE_TEST_NETWORK_LOG=str(log), IMPULSE_TEST_GUARD_LOADED=str(loaded),
    )
    command = [sys.executable, "-m", "impulse_evals", *args] if code is None else [sys.executable, "-c", code]
    proc = subprocess.run(
        command, cwd=str(cwd), env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=300,
        check=False,
    )
    assert loaded.exists(), "the socket guard (sitecustomize) was not loaded in the fresh interpreter"
    attempts = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return proc.returncode, _ANSI.sub("", proc.stdout or ""), _ANSI.sub("", proc.stderr or ""), attempts


def _shown(code, out, err):
    return f"exit {code}\nSTDOUT:\n{out[-2000:]}\nSTDERR:\n{err[-2000:]}"


def test_python_m_impulse_evals_help_is_the_impulse_help(tmp_path):
    code, out, err, attempts = _fresh_impulse(tmp_path, tmp_path, "--help")
    assert code == 0, _shown(code, out, err)
    assert out.lstrip().startswith("Usage: impulse "), _shown(code, out, err)
    tokens = set(re.split(r"[\s│┃|]+", out))
    missing = [name for name in CLI_COMMANDS if name not in tokens]
    assert missing == [], f"subcommands missing from the help: {missing}\n{_shown(code, out, err)}"
    assert attempts == [], attempts


@pytest.mark.parametrize(
    "args, usage",
    [(["run"], "Usage: impulse run "), (["no-such-command"], "Usage: impulse ")],
    ids=["run_without_config", "unknown_command"],
)
def test_python_m_impulse_evals_usage_errors_exit_2(tmp_path, args, usage):
    code, out, err, attempts = _fresh_impulse(tmp_path, tmp_path, *args)
    assert code == 2, _shown(code, out, err)
    assert usage in out + err, _shown(code, out, err)
    assert "Traceback" not in out + err, _shown(code, out, err)
    assert attempts == [], attempts


def test_python_m_impulse_evals_dry_run_exits_0_like_the_console_script(impulse_project):
    config = _dry_run_project(impulse_project)
    code, out, err, attempts = _fresh_impulse(impulse_project.root, impulse_project.root, "run", str(config), "--dry-run")
    assert code == 0, _shown(code, out, err)
    assert DRY_RUN_WARNING in out + err, _shown(code, out, err)
    run = impulse_project.read_run()
    assert run.total_scenarios == 2
    assert all(scenario_result.error is None for scenario_result in run.scenario_results), [
        scenario_result.error for scenario_result in run.scenario_results
    ]
    assert attempts == [], attempts


def test_importing_impulse_evals_main_runs_nothing(tmp_path):
    code, out, err, attempts = _fresh_impulse(tmp_path, tmp_path, code="import impulse_evals.__main__")
    assert code == 0, _shown(code, out, err)
    assert out == "", out
    assert err == "", err
    assert attempts == [], attempts
