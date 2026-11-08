"""notebooks/tutorial.ipynb cannot rot (issue #15).

Two kinds of test. The *static* ones read the committed file, need nothing but the standard
library, and run in the default suite: they keep the notebook small, stripped of outputs
(so a diff shows only what was edited), valid, and written against the public API only.

The *executing* ones carry the ``notebook`` marker, which the default ``addopts`` deselect: they
run the notebook top to bottom in a fresh kernel, from a directory outside the repo, with the
network switched off, and are what the ``notebook`` CI job runs (``pytest -m notebook``).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

import pmlab

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks" / "tutorial.ipynb"
RUNNER = REPO / "tests" / "e2e" / "run_notebook.py"
MAX_BYTES = 60_000  # a tutorial, not a data dump
MAX_SECONDS = 120  # the budget on a CI runner; the notebook takes a fraction of it

STEPS = ("Discover", "Panel", "Tune", "Backtest", "Gate", "Paper trading", "Report")


def _nb() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text())


def _code_cells() -> list[str]:
    return ["".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"]


# --------------------------------------------------------------------------- static


def test_the_notebook_is_valid_nbformat_4_with_unique_cell_ids() -> None:
    nb = _nb()
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    assert nb["metadata"]["kernelspec"]["name"] == "python3"


def test_the_notebook_is_committed_with_its_outputs_stripped() -> None:
    """Decision: no outputs in git. Outputs carry timings, temp paths and library-version noise
    that would churn every diff; the docs page and the CI run show the real output instead."""
    for cell in _nb()["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [], f"cell {cell['id']} has outputs: re-save it stripped"
        assert cell["execution_count"] is None, f"cell {cell['id']} has an execution count"
        assert not (set(cell["metadata"]) - {"tags"}), f"cell {cell['id']} has run metadata"


def test_the_notebook_stays_small() -> None:
    assert NOTEBOOK.stat().st_size < MAX_BYTES


def test_every_code_cell_is_plain_python() -> None:
    for src in _code_cells():
        ast.parse(src)  # no magics, no shell escapes: it runs the same in any kernel


def test_the_notebook_walks_the_whole_workflow_in_order() -> None:
    headings = [
        line
        for c in _nb()["cells"]
        if c["cell_type"] == "markdown"
        for line in "".join(c["source"]).splitlines()
        if line.startswith("## ")
    ]
    positions = [next(i for i, h in enumerate(headings) if step in h) for step in STEPS]
    assert positions == sorted(positions), headings


def test_the_notebook_uses_only_the_public_api() -> None:
    """Everything comes from `pmlab` (or `pmlab.datasets`), and nothing reaches into a private name."""
    imported: set[str] = set()
    for src in _code_cells():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pmlab"):
                assert node.module in {"pmlab", "pmlab.datasets"}, node.module
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pmlab.") or "._" not in alias.name
            elif isinstance(node, ast.Attribute):
                dunder = node.attr.startswith("__") and node.attr.endswith("__")
                assert dunder or not node.attr.startswith("_"), f"private attribute .{node.attr}"
    assert imported, "the notebook imports nothing from pmlab"
    exported = set(pmlab.__all__) | set(pmlab.datasets.__all__)
    assert imported <= exported, sorted(imported - exported)


def test_the_notebook_cannot_reach_the_network() -> None:
    source = "\n".join(_code_cells())
    for needle in ("httpx", "requests", "urllib", "socket", "GammaClient", "ClobClient"):
        assert needle not in source, needle


# --------------------------------------------------------------------------- executing


def _run_notebook(
    cwd: Path, *extra: str, offline: bool = True, notebook: Path = NOTEBOOK
) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(cwd), "PYTHONDONTWRITEBYTECODE": "1"}
    if offline:
        hook = cwd / "_offline"
        hook.mkdir(parents=True, exist_ok=True)
        (hook / "sitecustomize.py").write_text(_OFFLINE)
        env["PYTHONPATH"] = str(hook)
    return subprocess.run(
        [sys.executable, str(RUNNER), str(notebook), "--cwd", str(cwd / "kernel"), *extra],
        capture_output=True,
        text=True,
        timeout=MAX_SECONDS * 3,
        env=env,
        cwd=cwd,
        check=False,
    )


# Loaded into the kernel at start-up: anything but loopback fails (the kernel talks to the
# notebook over loopback; resolving a name or reaching another host is what an API call needs).
_OFFLINE = textwrap.dedent(
    """
    import socket

    def _refuse(*args, **kwargs):
        raise OSError("network disabled: the tutorial must run offline")

    _connect = socket.socket.connect

    def _connect_local(self, address, *args):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in ("127.0.0.1", "::1", "localhost"):
            _refuse()
        return _connect(self, address, *args)

    socket.socket.connect = _connect_local
    socket.create_connection = _refuse
    socket.getaddrinfo = _refuse
    """
)


@pytest.mark.notebook
def test_the_tutorial_runs_top_to_bottom_offline_from_outside_the_repo(tmp_path: Path) -> None:
    started = time.perf_counter()
    proc = _run_notebook(tmp_path)
    elapsed = time.perf_counter() - started
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NOTEBOOK OK" in proc.stdout
    assert elapsed < MAX_SECONDS, f"{elapsed:.0f}s: over the {MAX_SECONDS}s budget"
    assert REPO not in (tmp_path / "kernel").parents


@pytest.mark.notebook
def test_a_corrupt_bundled_dataset_turns_the_notebook_red(tmp_path: Path) -> None:
    """The mutation, kept as a test: damage the bundled file inside the kernel, run the same
    notebook, and the job must fail, saying why, not limp through on partial data."""
    corrupt = tmp_path / "corrupt.py"
    corrupt.write_text(
        textwrap.dedent(
            """
            import pmlab.datasets.sample as _s

            _read = _s._read
            _s._read = lambda name: b"x" + _read(name) if name.endswith(".gz") else _read(name)
            """
        )
    )
    proc = _run_notebook(tmp_path, "--prepend", str(corrupt))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "SampleDataError" in proc.stdout and "checksum" in proc.stdout


@pytest.mark.notebook
def test_a_notebook_that_breaks_a_safety_rule_turns_the_job_red(tmp_path: Path) -> None:
    """The notebook's own checks are live: lift the segment filter and the Lisbon assert fires."""
    nb = _nb()
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and "allowed_segments=loaded.get_allowed_segments()" in src:
            cell["source"] = src.replace(
                "allowed_segments=loaded.get_allowed_segments()", "allowed_segments=None"
            )
            break
    else:  # pragma: no cover - the cell moved; this test must be updated with it
        pytest.fail("could not find the PaperBroker cell")
    broken = tmp_path / "broken.ipynb"
    broken.write_text(json.dumps(nb))
    proc = _run_notebook(tmp_path, notebook=broken)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "AssertionError" in proc.stdout


@pytest.mark.notebook
def test_the_offline_hook_really_blocks_the_network(tmp_path: Path) -> None:
    """Without this the offline claim of the tests above would be an assumption."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {
                "cell_type": "code",
                "id": "c0",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=3)",
            }
        ],
    }
    probe = tmp_path / "probe.ipynb"
    probe.write_text(json.dumps(nb))
    proc = _run_notebook(tmp_path, notebook=probe)
    assert proc.returncode == 1 and "network disabled" in proc.stdout
