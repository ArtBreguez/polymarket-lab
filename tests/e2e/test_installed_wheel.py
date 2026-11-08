"""End-to-end, as a user: build the wheel, install it in a FRESH venv, run the public API there.

Opt-in (``pytest -m e2e``; deselected by the default ``addopts``) because it needs ``uv`` and the
network and takes a couple of minutes. Nothing here imports ``pmlab`` from the checkout: every
check runs with the python of the fresh venv, from a directory outside the repository, in isolated
mode (``-I``), so only the installed, public API is exercised.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "e2e_transforms.py"
DOC_RUNNER = REPO / "tests" / "e2e" / "run_doc_examples.py"
OK_LINE = "pmlab e2e (as a user): OK"


def _clean_env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"}
    }
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1", "TERM": "dumb", **extra})
    return env


def _run(cmd: list[str | Path], cwd: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=_clean_env(**extra_env),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


class Installed:
    def __init__(self, venv: Path, workdir: Path) -> None:
        self.python = venv / "bin" / "python"
        self.pmlab = venv / "bin" / "pmlab"
        self.workdir = workdir

    def py(self, *args: str | Path) -> subprocess.CompletedProcess[str]:
        return _run([self.python, "-I", *args], cwd=self.workdir)


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> Installed:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build and install the wheel")
    root = tmp_path_factory.mktemp("pmlab-e2e")
    dist, venv, workdir = root / "dist", root / ".venv", root / "work"
    workdir.mkdir()
    _ok(_run([uv, "build", "--wheel", "--out-dir", dist], cwd=REPO))
    wheel = next(dist.glob("pmlab-*.whl"))
    _ok(_run([uv, "venv", "-q", venv, "--python", sys.executable], cwd=workdir))
    _ok(_run([uv, "pip", "install", "-q", "--python", venv / "bin" / "python", wheel], cwd=workdir))
    yield Installed(venv, workdir)
    shutil.rmtree(venv, ignore_errors=True)  # a venv with lightgbm/scipy is hundreds of MB


def test_the_wheel_is_installed_for_real_not_the_checkout(installed: Installed) -> None:
    out = _ok(installed.py("-c", "import pmlab; print(pmlab.__version__); print(pmlab.__file__)"))
    version, location = out.split("\n")[:2]
    assert "site-packages" in location
    assert REPO not in Path(location).parents
    pyproject = (REPO / "pyproject.toml").read_text()
    assert f'version = "{version}"' in pyproject


def test_user_workflow_script(installed: Installed) -> None:
    panel = installed.workdir / "panel.parquet"
    out = _ok(installed.py(EXAMPLE, "--panel-out", panel))
    assert OK_LINE in out
    assert "backtest (with new features)" in out and panel.exists()


def test_docs_examples_run_unmodified(installed: Installed) -> None:
    out = _ok(installed.py(DOC_RUNNER, REPO / "docs" / "features.md"))
    assert "FAIL" not in out and "blocks ran from" in out
    n_ok = int(out.strip().splitlines()[-1].split("/")[0])
    assert n_ok >= 10
    readme = _ok(
        installed.py(DOC_RUNNER, REPO / "README.md", "--contains", "add_missing_indicators")
    )
    assert "1/1 blocks ran" in readme


def test_cli_backtests_a_panel_built_with_the_new_features(installed: Installed) -> None:
    panel = installed.workdir / "panel_cli.parquet"
    _ok(installed.py(EXAMPLE, "--panel-out", panel))
    assert (
        _ok(_run([installed.pmlab, "version"], cwd=installed.workdir)).strip().startswith("pmlab")
    )
    metrics = installed.workdir / "metrics.json"
    out = _ok(
        _run(
            [
                installed.pmlab,
                "backtest",
                "--plugin",
                "weather_tmax",
                "--panel",
                panel,
                "--stride",
                "10",
                "--min-train-rows",
                "200",
                "--policy",
                "top-bin-yes",
                "--output",
                metrics,
            ],  # fmt: skip
            cwd=installed.workdir,
            PMLAB_ARTIFACTS_DIR=str(installed.workdir / "artifacts"),
        )
    )
    assert "Backtest Results" in out
    data = json.loads(metrics.read_text())
    assert data["total_trades"] > 0
    for key in ("total_pnl", "hit_rate", "avg_edge"):
        assert data[key] == data[key] and abs(data[key]) != float("inf"), key
