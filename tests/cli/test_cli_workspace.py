"""`scripts/pmlab-workspace` and `-a/-w` reach the same files through the real commands.

The script scopes the environment to one workspace (PMLAB_ARTIFACTS_DIR is the workspace's
own artifacts, PMLAB_WORKSPACE_NAME names it). The commands used to look for the trade log
and the signals in `workspaces/<name>` *under* that directory, a second time.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from tests.cli.test_cli_costs import _gate, _model, _signals, _write

runner = CliRunner()
_REPO = Path(__file__).resolve().parents[2]


def _publish(champion_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "promote-champion",
            str(_model(tmp_path)),
            "--gate-path",
            str(_gate(tmp_path)),
            "--plugin",
            "politics",
            "--output-dir",
            str(champion_dir),
        ],
    )
    assert result.exit_code == 0, result.output


def _scripted_env(repo: Path, workspace: str) -> dict[str, str]:
    """What scripts/pmlab-workspace puts in the environment, before its fix."""
    return {
        "PMLAB_WORKSPACE_NAME": workspace,
        "PMLAB_ARTIFACTS_DIR": str(repo / "artifacts" / "workspaces" / workspace),
    }


def _trade_log(repo: Path, workspace: str) -> Path:
    return (
        repo
        / "artifacts"
        / "workspaces"
        / workspace
        / "signals"
        / "v2"
        / "forward_paper_trades.json"
    )


def test_record_trades_under_the_script_writes_where_the_direct_style_does(tmp_path: Path) -> None:
    scripted_repo, direct_repo = tmp_path / "scripted", tmp_path / "direct"
    for repo in (scripted_repo, direct_repo):
        _publish(
            repo / "artifacts" / "workspaces" / "paper" / "public_models"
            if repo is scripted_repo
            else repo / "artifacts" / "public_models",
            tmp_path,
        )
        _signals(repo / "artifacts", "paper")

    scripted = runner.invoke(
        app, ["record-trades", "--plugin", "politics"], env=_scripted_env(scripted_repo, "paper")
    )
    direct = runner.invoke(
        app,
        [
            "record-trades",
            "--plugin",
            "politics",
            "-a",
            str(direct_repo / "artifacts"),
            "-w",
            "paper",
        ],
    )
    assert scripted.exit_code == 0, scripted.output
    assert direct.exit_code == 0, direct.output
    assert "Recorded 12 new trades" in scripted.output
    assert "Recorded 12 new trades" in direct.output
    assert _trade_log(scripted_repo, "paper").exists()
    assert _trade_log(direct_repo, "paper").exists()
    # And nothing was written to the doubled path.
    assert not (scripted_repo / "artifacts" / "workspaces" / "paper" / "workspaces").exists()


def test_status_and_monitor_read_the_same_log_under_the_script(tmp_path: Path) -> None:
    _publish(tmp_path / "artifacts" / "workspaces" / "paper" / "public_models", tmp_path)
    log = _trade_log(tmp_path, "paper")
    _write(log, {"trades": []})
    env = _scripted_env(tmp_path, "paper")

    status = runner.invoke(app, ["status"], env=env)
    assert status.exit_code == 0, status.output
    assert "No trades file found" not in status.output
    assert "Paper Trades" in status.output

    monitor = runner.invoke(
        app, ["monitor", "--skip", "drift", "--as-of", "2100-01-01T00:00:00Z"], env=env
    )
    assert monitor.exit_code == 30, monitor.output  # an empty book: insufficient, not "not found"
    assert "trade log not found" not in monitor.output + (monitor.stderr or "")


def test_the_old_doubled_path_is_not_where_it_looks(tmp_path: Path) -> None:
    _publish(tmp_path / "artifacts" / "workspaces" / "paper" / "public_models", tmp_path)
    result = runner.invoke(app, ["status"], env=_scripted_env(tmp_path, "paper"))
    assert "workspaces/paper/workspaces" not in result.output.replace("\n", "")


def test_the_script_exports_the_workspace_for_the_commands(tmp_path: Path) -> None:
    """Run a copy of the real script (it creates directories next to itself)."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(_REPO / "scripts" / "pmlab-workspace", scripts / "pmlab-workspace")
    result = subprocess.run(  # noqa: S603 - a copy of our own script
        ["bash", str(scripts / "pmlab-workspace"), "paper", "env"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    exported: dict[str, Any] = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if line.startswith("PMLAB_")
    )
    assert exported["PMLAB_WORKSPACE"] == "paper"
    assert exported["PMLAB_WORKSPACE_NAME"] == "paper"
    assert exported["PMLAB_ARTIFACTS_DIR"] == str(tmp_path / "artifacts" / "workspaces" / "paper")


def test_the_script_and_the_commands_agree_end_to_end(tmp_path: Path) -> None:
    """The script's real environment, fed to the real command in a subprocess."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(_REPO / "scripts" / "pmlab-workspace", scripts / "pmlab-workspace")
    _publish(tmp_path / "artifacts" / "workspaces" / "paper" / "public_models", tmp_path / "in")
    _signals(tmp_path / "artifacts", "paper")
    code = "from pmlab.cli.main import app; app()"
    result = subprocess.run(  # noqa: S603
        [
            "bash",
            str(scripts / "pmlab-workspace"),
            "paper",
            sys.executable,
            "-c",
            code,
            "record-trades",
            "--plugin",
            "politics",
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "COLUMNS": "200"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Recorded 12 new trades" in result.stdout
    assert json.loads(_trade_log(tmp_path, "paper").read_text())["trades"]


@pytest.mark.parametrize("command", ["status", "record-trades", "settle-trades", "monitor"])
def test_every_command_that_takes_a_workspace_reads_the_scripts_name(command: str) -> None:
    out = runner.invoke(app, [command, "--help"]).output
    assert "PMLAB_WORKSPACE" in out
