# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 9: domain guard (SPEC.md section 11).

``domains.detect_domain(text)``: the domain with the most DISTINCT keyword hits (whole-word,
case-insensitive), at least 3; one keyword repeated counts once; ties go to the earlier domain in
the SPEC.md list (fsi, healthcare, recruitment, ecommerce, edtech, telecom, realestate, travel, legal).

``impulse generate``: when the domain detected in the ``--prompt`` file differs from ``--domain`` and
``--domain`` is not ``general``, it warns and asks "Continue with --domain {domain}? [y/N]"; answering
no exits 2 (before any LLM call); ``--yes`` skips the prompt; no TTY behaves as ``--yes``.

A terminal is simulated with the ``tty_stdin`` fixture: ``sys.stdin.isatty()`` is True inside
``cli.invoke`` (typer's CliRunner); without it, the CliRunner's stdin is not a TTY.
"""

import io
from pathlib import Path

import pytest
import yaml

PRIMARY = "ollama/judge-primary"
MODEL_ENV = {
    "IMPULSE_MODEL_PRIMARY": PRIMARY,
    "IMPULSE_MODEL_FAST": "ollama/judge-fast",
    "IMPULSE_MODEL_SIMULATOR": "ollama/simulator",
}
# 4 distinct healthcare keywords (patient, doctor, appointment, prescription), none of another domain.
HEALTHCARE_PROMPT = (
    "You are the receptionist of Sunrise Clinic. Help each patient choose a doctor and an "
    "appointment time, and answer questions about a prescription."
)
# 2 healthcare keywords only: nothing is detected.
WEAK_PROMPT = "You are the receptionist of Sunrise Clinic. Help each patient reach a doctor."


def _detect(text):
    from impulse_evals.domains import detect_domain

    return detect_domain(text)


# ============================================================================================
# detect_domain
# ============================================================================================


def test_three_distinct_hits_detect_a_domain():
    assert _detect("Help the patient get an appointment with a doctor.") == "healthcare"


def test_two_distinct_hits_detect_nothing():
    assert _detect("Help the patient reach a doctor.") is None


def test_a_repeated_keyword_counts_once():
    assert _detect("patient patient patient, doctor doctor doctor, patient and doctor") is None


def test_keywords_are_case_insensitive():
    assert _detect("PATIENT, Doctor and APPOINTMENT") == "healthcare"


def test_keywords_match_whole_words_only():
    # "patients", "doctors", "appointments" are not the whole words patient / doctor / appointment.
    assert _detect("Our patients see doctors for appointments.") is None
    # "company" contains "pan" and "accredit" contains "credit", but neither is the whole word.
    assert _detect("The company is accredited. loan cibil") is None


def test_multi_word_keywords_count_as_one_hit():
    assert _detect("A loan with a low interest rate and a fixed EMI.") == "fsi"
    # "interest" and "rate" apart are not the keyword "interest rate".
    assert _detect("A loan with interest at a fixed EMI rate.") is None


def test_most_distinct_hits_win():
    text = "loan cibil kyc; patient doctor appointment medicine"
    assert _detect(text) == "healthcare"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("loan, cibil and kyc questions from a patient about a doctor appointment", "fsi"),
        ("flight hotel visa contract dispute lawyer", "travel"),
        ("interview candidate resume, lesson quiz student", "recruitment"),
    ],
)
def test_ties_go_to_the_earlier_domain_in_the_list(text, expected):
    assert _detect(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Check the loan, the CIBIL score and the KYC status.", "fsi"),
        ("Schedule the interview, review the resume, update the candidate.", "recruitment"),
        ("Track the order, the refund and the delivery.", "ecommerce"),
        ("Plan the lesson, the quiz and the homework.", "edtech"),
        ("Help with the recharge, a postpaid plan or a new sim card.", "telecom"),
        ("Book a site visit for the apartment and talk to the landlord.", "realestate"),
        ("Change the flight, the hotel and the itinerary.", "travel"),
        ("Explain the clause, the contract and the dispute.", "legal"),
    ],
)
def test_each_domain_is_detected(text, expected):
    assert _detect(text) == expected


@pytest.mark.parametrize("text", ["", "Hello there."])
def test_no_keywords_detect_nothing(text):
    assert _detect(text) is None


# ============================================================================================
# impulse generate: the prompt
# ============================================================================================


class _TtyBytesIO(io.BytesIO):
    def isatty(self):
        return True


@pytest.fixture
def tty_stdin(monkeypatch):
    """Inside ``cli.invoke`` the command's ``sys.stdin.isatty()`` is True (a simulated terminal)."""
    import typer.testing

    original = typer.testing.make_input_stream

    def make_input_stream(input, charset):
        return _TtyBytesIO(original(input, charset).getvalue())

    monkeypatch.setattr(typer.testing, "make_input_stream", make_input_stream)


@pytest.fixture
def project(impulse_project, mock_llm, monkeypatch):
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    return impulse_project


def _items(count):
    return [
        {
            "name": f"guard_case_{number}",
            "persona": f"caller {number}",
            "opening_user_message": f"Hello, this is caller {number}.",
            "direction": "inbound",
            "metrics": ["UNIV-001"],
        }
        for number in range(1, count + 1)
    ]


def _prompt_file(root, text):
    path = Path(root) / "agent_prompt.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _generate(cli, root, text, domain, *extra, input=None):
    prompt = _prompt_file(root, text)
    return cli.invoke(
        "generate", "--prompt", prompt, "--domain", domain, "--language", "en-IN", "--count", "2", *extra, input=input
    )


def _out(root):
    return Path(root) / "scenarios" / "generated.yaml"


def _question(domain):
    return f"Continue with --domain {domain}? [y/N]"


def test_tty_fixture_makes_stdin_a_terminal(tty_stdin):
    """Self-check of the simulated terminal used below."""
    import sys

    import typer
    from typer.testing import CliRunner

    probe = typer.Typer()

    @probe.command()
    def main():
        print(f"isatty={sys.stdin.isatty()}")

    @probe.command()
    def other():  # a second command keeps typer in multi-command mode
        pass

    assert "isatty=True" in CliRunner().invoke(probe, ["main"]).output


@pytest.mark.parametrize("answer", ["n\n", "N\n", "\n"], ids=["n", "N", "default"])
def test_mismatch_on_a_terminal_asks_and_no_exits_2(project, mock_llm, cli, normalize_console, tty_stdin, answer):
    result = _generate(cli, project.root, HEALTHCARE_PROMPT, "fsi", input=answer)
    assert result.exit_code == 2, cli.text(result)
    text = normalize_console(result.output)
    assert _question("fsi") in text
    assert "healthcare" in text, "the warning names the detected domain"
    assert mock_llm.call_count == 0
    assert not _out(project.root).exists()


def test_mismatch_on_a_terminal_yes_continues(project, mock_llm, cli, normalize_console, tty_stdin):
    mock_llm.queue(yaml.safe_dump(_items(2)))
    result = _generate(cli, project.root, HEALTHCARE_PROMPT, "fsi", input="y\n")
    assert result.exit_code == 0, cli.text(result)
    assert _question("fsi") in normalize_console(result.output)
    assert mock_llm.call_count == 1
    assert _out(project.root).is_file()


def test_yes_flag_skips_the_prompt(project, mock_llm, cli, normalize_console, tty_stdin):
    mock_llm.queue(yaml.safe_dump(_items(2)))
    result = _generate(cli, project.root, HEALTHCARE_PROMPT, "fsi", "--yes")
    assert result.exit_code == 0, cli.text(result)
    text = normalize_console(result.output)
    assert "Continue with --domain" not in text
    assert "healthcare" in text, "the mismatch warning is still printed"
    assert mock_llm.call_count == 1
    assert _out(project.root).is_file()


def test_no_tty_behaves_as_yes(project, mock_llm, cli, normalize_console):
    mock_llm.queue(yaml.safe_dump(_items(2)))
    result = _generate(cli, project.root, HEALTHCARE_PROMPT, "fsi")  # CliRunner stdin is not a TTY
    assert result.exit_code == 0, cli.text(result)
    text = normalize_console(result.output)
    assert "Continue with --domain" not in text
    assert "healthcare" in text, "the mismatch warning is still printed"
    assert mock_llm.call_count == 1
    assert _out(project.root).is_file()


@pytest.mark.parametrize(
    "text, domain",
    [
        (HEALTHCARE_PROMPT, "healthcare"),  # detected == --domain
        (HEALTHCARE_PROMPT, "general"),  # --domain general never asks
        (HEALTHCARE_PROMPT, "insurance"),  # an unknown --domain is treated as general: never asks
        (WEAK_PROMPT, "fsi"),  # fewer than 3 hits: nothing detected
    ],
    ids=["same_domain", "general", "unknown_domain_is_general", "nothing_detected"],
)
def test_no_prompt_without_a_mismatch(project, mock_llm, cli, normalize_console, tty_stdin, text, domain):
    mock_llm.queue(yaml.safe_dump(_items(2)))
    result = _generate(cli, project.root, text, domain, input="")
    assert result.exit_code == 0, cli.text(result)
    assert "Continue with --domain" not in normalize_console(result.output)
    assert mock_llm.call_count == 1
    assert _out(project.root).is_file()
