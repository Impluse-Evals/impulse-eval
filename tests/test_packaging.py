# SPDX-License-Identifier: Apache-2.0
"""Audit batch 3 (AUDIT.md finding 49, maintainer decision #49): what the built distributions hold.

The wheel and the sdist are built once per module, offline, from the repository into ``tmp_path``
with hatchling (the build backend, SPEC.md section 5), in a fresh interpreter whose sockets refuse
every non-loopback connection. Skipped when hatchling is not installed.

* ``click`` is a declared dependency: ``cli.py`` imports it, and typer vendors its own copy, so it
  would otherwise arrive only through other packages.
* The wheel ships ``LICENSING.md`` (force-included into the package) and the ``license-files`` set
  in ``pyproject.toml`` (``LICENSE`` and ``NOTICE``, pinned by ``tests/test_license_headers.py``).
* The sdist leaves out the root planning files and ``.github/``.
"""

import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANNING_FILES = (
    "SPEC.md",
    "METRICS.md",
    "KICKOFF.md",
    "CLAUDE.md",
    "REVIEW.md",
    "REVIEW-2.md",
    "DECISIONS.md",
    "PROGRESS.md",
    "FINAL-REPORT.md",
    "AUDIT.md",
    "AUDIT-FIXES.md",
)
LICENCE_FILES = {"LICENSE", "NOTICE"}
BUILD_TIMEOUT_S = 300

BUILD_SCRIPT = r"""
import json, socket, sys
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

from hatchling.cli import hatchling

sys.argv = ["hatchling", "build", "-t", "wheel", "-t", "sdist", "-d", sys.argv[1]]
try:
    code = hatchling()
except SystemExit as exc:
    code = exc.code
print("RESULT=" + json.dumps({"code": code, "attempts": attempts}))
"""


@pytest.fixture(scope="module")
def dists(tmp_path_factory):
    """``{"wheel": Path, "sdist": Path}`` built from the repository (once per module)."""
    pytest.importorskip("hatchling")
    out = tmp_path_factory.mktemp("impulse-dist")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-c", BUILD_SCRIPT, str(out)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_S,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT=")]
    assert proc.returncode == 0 and lines, f"exit {proc.returncode}\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
    report = json.loads(lines[-1][len("RESULT="):])
    assert report["attempts"] == [], f"the build reached the network: {report['attempts']}"
    assert report["code"] in (0, None), f"hatchling build failed ({report['code']}):\n{proc.stderr[-3000:]}"
    wheels = sorted(out.glob("*.whl"))
    sdists = sorted(out.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, sorted(p.name for p in out.iterdir())
    return {"wheel": wheels[0], "sdist": sdists[0]}


def _wheel_names(dists):
    with zipfile.ZipFile(dists["wheel"]) as wheel:
        return wheel.namelist()


def _wheel_read(dists, name):
    with zipfile.ZipFile(dists["wheel"]) as wheel:
        return wheel.read(name)


def _metadata(dists):
    name = next(n for n in _wheel_names(dists) if re.fullmatch(r"[^/]+\.dist-info/METADATA", n))
    return _wheel_read(dists, name).decode("utf-8")


def _sdist_names(dists):
    """Member paths of the sdist without its top folder (``impulse_evals-0.1.0/``)."""
    with tarfile.open(dists["sdist"]) as sdist:
        names = [member.name for member in sdist.getmembers() if member.isfile()]
    return {str(PurePosixPath(*PurePosixPath(name).parts[1:])) for name in names}


def _requirement_name(line):
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else ""


# ============================================================================================
# The wheel
# ============================================================================================


def test_wheel_declares_click_as_a_dependency(dists):
    requires = [
        line[len("Requires-Dist:"):].strip()
        for line in _metadata(dists).splitlines()
        if line.startswith("Requires-Dist:")
    ]
    click = [req for req in requires if _requirement_name(req) == "click"]
    assert click, f"click is not a declared dependency: {requires}"
    assert all("extra ==" not in req for req in click), f"click must be a core dependency, not an extra: {click}"


def test_wheel_ships_licensing_md(dists):
    found = [name for name in _wheel_names(dists) if PurePosixPath(name).name == "LICENSING.md"]
    assert found, "the wheel does not contain LICENSING.md"
    assert _wheel_read(dists, found[0]) == (REPO_ROOT / "LICENSING.md").read_bytes()


def test_wheel_licence_metadata(dists):
    # The PEP 639 licence files and expression (Apache-2.0 for the whole package).
    names = _wheel_names(dists)
    licences = {PurePosixPath(n).name for n in names if re.fullmatch(r"[^/]+\.dist-info/licenses/[^/]+", n)}
    assert licences == LICENCE_FILES
    metadata = _metadata(dists)
    assert "License-Expression: Apache-2.0" in metadata.splitlines()
    assert {line.split(":", 1)[1].strip() for line in metadata.splitlines() if line.startswith("License-File:")} == LICENCE_FILES


def test_wheel_still_holds_the_package(dists):
    # Regression guard (passes before the fix too).
    names = set(_wheel_names(dists))
    for name in ("impulse_evals/__init__.py", "impulse_evals/cli.py", "impulse_evals/templates/free_config.yaml"):
        assert name in names, name
    assert not any(name.startswith(("tests/", "docs/")) for name in names)


# ============================================================================================
# The sdist
# ============================================================================================


def test_sdist_leaves_out_the_planning_files(dists):
    names = _sdist_names(dists)
    shipped = sorted(name for name in PLANNING_FILES if name in names)
    assert shipped == [], f"planning files in the sdist: {shipped}"


def test_sdist_leaves_out_github_folder(dists):
    shipped = sorted(name for name in _sdist_names(dists) if name.startswith(".github/"))
    assert shipped == [], f".github/ files in the sdist: {shipped}"


def test_sdist_keeps_what_a_build_needs(dists):
    # Regression guard (passes before the fix too).
    names = _sdist_names(dists)
    for name in ("pyproject.toml", "README.md", "LICENSING.md", "PKG-INFO", "impulse_evals/__init__.py",
                 "impulse_evals/cli.py", *sorted(LICENCE_FILES)):
        assert name in names, name
