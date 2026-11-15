# SPDX-License-Identifier: AGPL-3.0-only
"""``impulse --version`` prints the installed package version and exits 0."""

from importlib.metadata import version


def test_version_option_prints_the_installed_version(cli):
    result = cli.invoke("--version")
    assert result.exit_code == 0, cli.text(result)
    assert cli.text(result).strip() == f"impulse-evals {version('impulse-evals')}"


def test_help_still_lists_the_commands(cli):
    result = cli.invoke("--help")
    assert result.exit_code == 0, cli.text(result)
    assert "run" in cli.text(result) and "--version" in cli.text(result)
