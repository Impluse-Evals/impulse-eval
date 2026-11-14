# SPDX-License-Identifier: AGPL-3.0-only
"""The pytest API (SPEC.md sections 15 and 18.1).

Use it inside your own pytest tests::

    from impulse_evals.testing import assert_no_critical, impulse_check

    def test_booking_agent():
        run = impulse_check("config.yaml")                       # or scenario_name="clinic_booking"
        assert_no_critical(run)

- ``impulse_check(config_path, scenario_name=None) -> RunResult`` runs the config the way
  ``impulse run`` does (text or voice mode, models and keys from ``config.yaml``, ``IMPULSE_MODEL_*``,
  and ``./.impulse/config.toml``) and writes the run folder ``./.impulse/runs/<run_id>/`` (``run.json``,
  ``scenarios/<name>.json``, ``report.html`` with embedded audio). It is a plain synchronous function;
  called while an event loop is running in the same thread (an async test), the run happens in a
  worker thread. ``scenario_name`` runs only that scenario; a name the config does not have raises
  ``UnknownScenarioError`` naming it, before anything runs. A configuration problem raises
  ``ConfigError``. ``impulse_check`` never applies ``--fail-on``: it returns the result.
- ``assert_no_critical(run_result)`` is the gate: it raises ``AssertionError`` under the
  ``--fail-on critical`` rule (SPEC.md section 17; ``impulse run`` itself defaults to ``high`` since
  fixes round 4): a failed ``critical`` assertion, or any scenario
  ``error`` (which includes a critical check the judge could not evaluate). The message names each
  scenario with its failed critical rules (and their reasons) or its error. Failed high or medium
  assertions, poor metrics, and skips pass.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from collections.abc import Callable, Coroutine
from typing import Any, Optional, TypeVar, Union

from impulse_evals.config import load_config  # raises impulse_evals.config.ConfigError
from impulse_evals.models import RunResult

__all__ = ["UnknownScenarioError", "assert_no_critical", "impulse_check"]

T = TypeVar("T")


class UnknownScenarioError(LookupError):
    """``impulse_check(..., scenario_name=...)`` named a scenario the config does not have."""


def _run_sync(make_coroutine: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine to completion from synchronous code. When an event loop is already running in
    this thread (``asyncio.run`` would refuse), it runs on its own loop in a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coroutine())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="impulse_check") as pool:
        return pool.submit(lambda: asyncio.run(make_coroutine())).result()


def impulse_check(
    config_path: Union[str, os.PathLike[str]], scenario_name: Optional[str] = None
) -> RunResult:
    """Run ``config_path`` (all scenarios, or only ``scenario_name``) and return the ``RunResult``.

    Writes ``./.impulse/runs/<run_id>/`` like ``impulse run`` (``run.json``, ``scenarios/<name>.json``,
    ``report.html``). Raises ``ConfigError`` for an invalid config, unresolved model, or missing key,
    and ``UnknownScenarioError`` when ``scenario_name`` is not in the config. Never applies
    ``--fail-on``; check the result with ``assert_no_critical``.
    """
    config = load_config(config_path)
    if scenario_name is not None:
        selected = [scenario for scenario in config.scenarios if scenario.name == scenario_name]
        if not selected:
            names = ", ".join(scenario.name for scenario in config.scenarios) or "none"
            raise UnknownScenarioError(
                f"scenario_name {scenario_name!r} is not a scenario in {os.fspath(config_path)} "
                f"(its scenarios: {names})"
            )
        config = config.model_copy(update={"scenarios": selected})

    from impulse_evals import runner  # lazy: loads the judge, simulator, and LiteLLM
    from impulse_evals.reporter import RUNS_DIR, ensure_impulse_dir, write_run_files

    result = _run_sync(lambda: runner.run(config))
    ensure_impulse_dir()
    write_run_files(result, RUNS_DIR / result.run_id, with_html=True, embed_audio=True)
    return result


def _critical_problems(run_result: RunResult) -> list[str]:
    """One line per scenario that fails the ``--fail-on critical`` rule (SPEC.md section 17; the rule
    this helper keeps although ``impulse run`` defaults to ``high`` since fixes round 4)."""
    problems: list[str] = []
    for result in run_result.scenario_results:
        if result.error is not None:
            problems.append(f"{result.scenario_name}: scenario error: {result.error}")
        for score in result.assertion_results:
            if score.result == "fail" and score.severity == "critical":
                reason = " ".join(str(score.reason).split())
                problems.append(f"{result.scenario_name}: critical assertion {score.rule_id} failed: {reason}")
    return problems


def assert_no_critical(run_result: RunResult) -> None:
    """Raise ``AssertionError`` when ``run_result`` has a failed critical assertion or a scenario error
    (the ``--fail-on critical`` rule); return None otherwise."""
    problems = _critical_problems(run_result)
    if problems:
        lines = "\n".join(f"  - {problem}" for problem in problems)
        raise AssertionError(
            f"run {run_result.run_id}: {len(problems)} critical problem(s) "
            f"(the --fail-on critical rule):\n{lines}"
        )
