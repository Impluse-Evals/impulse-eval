# SPDX-License-Identifier: Apache-2.0
"""SPDX licence headers and licence metadata (SPEC.md section 5.1).

Everything in the repository is under the Apache License 2.0: every Python file starts with the
Apache-2.0 SPDX header, no tracked file carries another SPDX identifier, and ``pyproject.toml``
declares ``Apache-2.0`` with ``LICENSE`` and ``NOTICE`` as its licence files.

Also the plugin API import rule for the five plugin API files: no import statement naming a
non-plugin-API ``impulse_evals`` module outside ``if TYPE_CHECKING:`` (AST check), and importing
each file in a fresh interpreter loads no other ``impulse_evals`` module than the plugin API files
and the package ``__init__`` modules on its import path (subprocess check). The rule keeps the
plugin API a small, stable surface that adapters, metrics, and judges can build on without the core.
"""

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

APACHE = "# SPDX-License-Identifier: Apache-2.0"

# Exactly the five plugin API files (SPEC.md section 5.1): plugin_types.py and the four base files.
PLUGIN_API_FILES = {
    "impulse_evals/plugin_types.py",
    "impulse_evals/assertions/base.py",
    "impulse_evals/metrics/base.py",
    "impulse_evals/adapters/base.py",
    "impulse_evals/judge/base.py",
}
SCANNED_DIRS = ("impulse_evals", "tests", "examples")

PACKAGE = "impulse_evals"
# Package __init__ modules a plugin API file may load at runtime (only those on the import path of
# a loaded plugin API module; Python always executes them). SPEC.md section 5.1.
ALLOWED_PACKAGE_INITS = {
    "impulse_evals",
    "impulse_evals.metrics",
    "impulse_evals.assertions",
    "impulse_evals.adapters",
    "impulse_evals.judge",
}


def _python_files():
    files = []
    for top in SCANNED_DIRS:
        root = REPO_ROOT / top
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _first_line(path: Path) -> str:
    text = path.read_bytes().decode("utf-8-sig")
    lines = text.splitlines()
    return lines[0].rstrip() if lines else ""


def test_python_files_are_found():
    relatives = {_relative(p) for p in _python_files()}
    assert "impulse_evals/__init__.py" in relatives
    assert "tests/conftest.py" in relatives
    assert "examples/sample_agent.py" in relatives


def test_every_python_file_starts_with_the_apache_header():
    wrong = []
    for path in _python_files():
        actual = _first_line(path)
        if actual != APACHE:
            wrong.append(f"{_relative(path)}: expected {APACHE!r}, found {actual!r}")
    assert not wrong, "Wrong or missing SPDX header:\n" + "\n".join(wrong)


def _tracked_files() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "ls-files"], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    if completed.returncode != 0:
        pytest.skip("not a git checkout")
    return completed.stdout.splitlines()


def test_no_file_carries_another_spdx_identifier():
    # Only a real licence ID counts (code that matches SPDX lines, like this one, is not a header).
    pattern = re.compile(r"SPDX-License-Identifier:[ \t]*([A-Za-z0-9][A-Za-z0-9.+-]*)")
    wrong = []
    for relative in _tracked_files():
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:4096]:
            continue
        for number, line in enumerate(data.decode("utf-8-sig", errors="replace").splitlines(), 1):
            if any(match.group(1) != "Apache-2.0" for match in pattern.finditer(line)):
                wrong.append(f"{relative}:{number}: {line.strip()}")
    assert not wrong, "Only Apache-2.0 SPDX identifiers are allowed:\n" + "\n".join(wrong)


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE", "LICENSING.md"])
def test_licence_files_exist(name):
    assert (REPO_ROOT / name).is_file(), f"{name} must exist in the repository root"


@pytest.mark.parametrize("name", ["LICENSE-ELv2", "LICENSE-APACHE"])
def test_old_licence_files_are_gone(name):
    assert not (REPO_ROOT / name).exists(), f"{name} must not exist: everything is under LICENSE"


def test_licence_file_is_the_apache_licence():
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    assert "END OF TERMS AND CONDITIONS" in text
    assert "Copyright 2026 VITAF Technologies" in text


def test_notice_names_the_project_and_copyright_holder():
    text = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert text.startswith("Impulse Evals\n")
    assert "Copyright 2026 VITAF Technologies" in text


def test_pyproject_declares_licence_expression():
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.is_file(), "pyproject.toml must exist"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["license"] == "Apache-2.0"


def test_pyproject_declares_licence_files():
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.is_file(), "pyproject.toml must exist"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert set(data["project"]["license-files"]) == {"LICENSE", "NOTICE"}


def test_pyproject_has_no_licence_classifier():
    # PEP 639: a licence expression replaces the "License ::" trove classifiers.
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert not [c for c in data["project"].get("classifiers", []) if c.startswith("License ::")]


# ============================================================================================
# Plugin API import rule (SPEC.md section 5.1)
# ============================================================================================


def test_plugin_types_module_exists_with_apache_header():
    path = REPO_ROOT / "impulse_evals" / "plugin_types.py"
    assert path.is_file(), "impulse_evals/plugin_types.py must exist (SPEC.md sections 5.1 and 6)"
    assert _first_line(path) == APACHE


def _module_name(relative: str) -> str:
    """``impulse_evals/metrics/base.py`` -> ``impulse_evals.metrics.base`` (``__init__`` dropped)."""
    parts = relative[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


PLUGIN_API_MODULES = {_module_name(relative) for relative in PLUGIN_API_FILES}


def _in_package(module: str) -> bool:
    return module == PACKAGE or module.startswith(PACKAGE + ".")


def _module_file(module: str) -> Optional[Path]:
    """The repository file of an ``impulse_evals`` module (package ``__init__.py`` first), or None."""
    base = REPO_ROOT.joinpath(*module.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    candidate = base.parent / (base.name + ".py")
    return candidate if candidate.is_file() else None


def _is_type_checking(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` or ``if typing.TYPE_CHECKING:``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING" and isinstance(test.value, ast.Name) and test.value.id == "typing"
    return False


def _runtime_import_nodes(tree: ast.AST):
    """Every import statement except those in the body of an ``if TYPE_CHECKING:`` block.

    Imports inside functions, methods, ``try`` blocks, and the ``else`` branch of a TYPE_CHECKING
    test all count: they run when that code runs.
    """
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _import_targets(node, module: str, is_package: bool) -> list[str]:
    """Absolute module names an import statement names (relative imports resolved).

    ``import a.b`` names ``a.b``; ``from M import x`` names ``M.x`` when that is an ``impulse_evals``
    module file, otherwise ``M`` (``x`` is an attribute of ``M``).
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        parts = module.split(".") if is_package else module.split(".")[:-1]
        if node.level > 1:
            parts = parts[: len(parts) - (node.level - 1)]
        base = ".".join(parts + ([node.module] if node.module else []))
    else:
        base = node.module or ""
    targets = []
    for alias in node.names:
        submodule = f"{base}.{alias.name}"
        if alias.name != "*" and _in_package(submodule) and _module_file(submodule) is not None:
            targets.append(submodule)
        else:
            targets.append(base)
    return targets


def _non_plugin_api_imports(source: str, relative: str) -> list[tuple[int, str]]:
    """``(line, module)`` for each runtime import statement in ``source`` (the text of the file at
    ``relative``) that names an ``impulse_evals`` module outside the plugin API."""
    module = _module_name(relative)
    is_package = relative.endswith("/__init__.py")
    found = []
    for node in _runtime_import_nodes(ast.parse(source, filename=relative)):
        for target in _import_targets(node, module, is_package):
            if _in_package(target) and target not in PLUGIN_API_MODULES:
                found.append((node.lineno, target))
    return sorted(found)


def test_import_rule_checker_flags_runtime_imports_only():
    """The AST check itself: flags function-level, ``else``-branch, relative, and pack imports;
    ignores TYPE_CHECKING-only imports, plugin API targets, and non-``impulse_evals`` modules."""
    source = textwrap.dedent(
        """
        from __future__ import annotations
        import importlib
        import typing
        from typing import TYPE_CHECKING
        import pydantic
        from impulse_evals.judge.base import Judge
        from ..judge import base
        if TYPE_CHECKING:
            from impulse_evals.models import Scenario
            import impulse_evals.config
        if typing.TYPE_CHECKING:
            from ..models import CallTranscript
        else:
            from ..models import Turn
        def lazy():
            from impulse_evals.models import MetricResult
            import impulse_evals.domains
            return importlib.import_module("impulse_evals.metrics.universal")
        from . import universal
        from impulse_evals.assertions import dpdp
        """
    )
    found = _non_plugin_api_imports(source, "impulse_evals/metrics/base.py")
    assert sorted(target for _, target in found) == [
        "impulse_evals.assertions.dpdp",
        "impulse_evals.domains",
        "impulse_evals.metrics.universal",
        "impulse_evals.models",
        "impulse_evals.models",
    ]


@pytest.mark.parametrize("relative", sorted(PLUGIN_API_FILES))
def test_plugin_api_file_imports_only_the_plugin_api(relative):
    path = REPO_ROOT / relative
    assert path.is_file(), f"{relative} must exist"
    source = path.read_bytes().decode("utf-8-sig")
    found = _non_plugin_api_imports(source, relative)
    assert not found, (
        f"{relative} (plugin API) must not import a non-plugin-API impulse_evals module outside "
        "`if TYPE_CHECKING:` (SPEC.md section 5.1 plugin API import rule):\n"
        + "\n".join(f"  line {line}: {target}" for line, target in found)
    )


# Runs in a fresh interpreter: argv[1] = repository root, argv[2] = module to import. Prints every
# loaded impulse_evals module with its file. Non-loopback network is refused.
IMPORT_PROBE = textwrap.dedent(
    r"""
    import importlib, json, socket, sys

    repo, target = sys.argv[1], sys.argv[2]
    sys.path.insert(0, repo)

    _real_connect = socket.socket.connect

    def _offline(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host not in (None, "127.0.0.1", "::1", "localhost"):
            raise RuntimeError(f"network blocked in plugin API import probe: {address!r}")
        return _real_connect(self, address)

    socket.socket.connect = _offline

    importlib.import_module(target)
    loaded = {
        name: getattr(sys.modules[name], "__file__", None)
        for name in sorted(sys.modules)
        if name == "impulse_evals" or name.startswith("impulse_evals.")
    }
    print("IMPORT_PROBE=" + json.dumps(loaded))
    """
)


def _loaded_after_import(module: str) -> dict:
    env = dict(os.environ)
    env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE, str(REPO_ROOT), module],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("IMPORT_PROBE=")]
    if completed.returncode != 0 or not lines:
        pytest.fail(
            f"importing {module} in a fresh interpreter failed (exit {completed.returncode}):\n"
            f"{completed.stderr[-4000:]}\n{completed.stdout[-2000:]}",
            pytrace=False,
        )
    return json.loads(lines[-1][len("IMPORT_PROBE="):])


@pytest.mark.parametrize("relative", sorted(PLUGIN_API_FILES))
def test_importing_plugin_api_file_loads_only_the_plugin_api(relative):
    module = _module_name(relative)
    loaded = _loaded_after_import(module)
    assert module in loaded, f"{module} was not loaded by the import probe"
    plugin_api = [name for name in loaded if name in PLUGIN_API_MODULES]
    wrong = []
    for name, file in sorted(loaded.items()):
        if name in PLUGIN_API_MODULES:
            continue
        is_allowed_init = (
            name in ALLOWED_PACKAGE_INITS
            and file is not None
            and Path(file).name == "__init__.py"
            and any(other.startswith(name + ".") for other in plugin_api)
        )
        if not is_allowed_init:
            wrong.append(f"  {name}")
    assert not wrong, (
        f"importing {module} (plugin API) in a fresh interpreter loaded impulse_evals modules "
        "outside the plugin API other than the package __init__ modules on its import path "
        "(SPEC.md section 5.1 plugin API import rule):\n" + "\n".join(wrong)
    )
