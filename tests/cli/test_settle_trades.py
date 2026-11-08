"""Issue #43: ``pmlab settle-trades`` must not look like it works when it settles nothing.

The command cannot settle: settlement needs a plugin instance wired to real data, which a bare
command cannot build from a name. It used to say so in four lines of stdout and exit 0, so a cron
entry such as ``pmlab record-trades ... && pmlab settle-trades ...`` looked healthy forever while
every trade stayed open. It now exits non-zero, says on stderr what it cannot do, and prints the
exact Python call that does settle (`SettlementEngine`), for the workspace it was asked about.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.plugins.base import MarketPlugin

runner = CliRunner()


def _trade_log(artifacts: Path, workspace: str) -> Path:
    """Where ``record-trades`` writes a workspace's trades (and so where settlement must read)."""
    return artifacts / "workspaces" / workspace / "signals" / "v2" / "forward_paper_trades.json"


def _open_trade() -> dict[str, Any]:
    return {
        "recorded_at": "2024-01-14T10:00:00+00:00",
        "city_or_segment": "Chicago",
        "target_date": "2024-01-15",
        "outcome_label": "YES",
        "direction": "yes",
        "gamma_price": 0.6,
        "edge_after_fee": 0.12,
        "horizon": "morning_of",
        "flat_stake": 1.0,
        "size": 1.666667,
        "fee_paid": 0.003,
        "outcome": None,
        "realized_pnl": None,
    }


class _Final(MarketPlugin):
    """A plugin whose truth is final and says ``YES`` won."""

    family = "fake"

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> str | None:
        return "YES"

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None


def _spec() -> MarketSpec:
    return MarketSpec(
        market_id="mkt-001",
        slug="will-it-be-hot",
        question="Will max temp exceed 90F?",
        outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
        close_time="2024-01-15T20:00:00Z",
        market_family="binary",
        metadata={"city": "Chicago", "target_date": "2024-01-15"},
    )


@pytest.fixture
def log(tmp_path: Path) -> Path:
    path = _trade_log(tmp_path, "ws")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"trades": [_open_trade()]}))
    return path


def _invoke(tmp_path: Path, *extra: str, workspace: str = "ws"):
    return runner.invoke(
        app,
        ["settle-trades", "--plugin", "fake", "-a", str(tmp_path), "-w", workspace, *extra],
    )


def test_exits_non_zero_so_a_pipeline_cannot_mistake_it_for_a_settlement(
    tmp_path: Path, log: Path
) -> None:
    result = _invoke(tmp_path)
    assert result.exit_code == 1, result.output


def test_says_on_stderr_what_it_cannot_do_and_nothing_on_stdout(tmp_path: Path, log: Path) -> None:
    result = _invoke(tmp_path)
    assert result.stdout == ""  # a success-looking line on stdout is what hid the problem
    err = " ".join(result.stderr.split())
    assert "settle-trades" in err
    assert "cannot settle" in err
    assert "nothing was settled" in err.lower()
    assert "plugin" in err and "real data" in err


def test_shows_the_exact_call_with_the_workspace_s_own_trade_log(tmp_path: Path, log: Path) -> None:
    err = _invoke(tmp_path).stderr
    assert "from pmlab import SettlementEngine" in err
    assert "SettlementEngine(plugin," in err
    assert ".settle_all(specs)" in err
    assert str(log) in err


def test_the_path_follows_the_workspace_and_the_artifacts_dir(tmp_path: Path) -> None:
    result = _invoke(tmp_path, workspace="other")
    assert str(_trade_log(tmp_path, "other")) in result.stderr


def test_reads_the_workspace_from_the_environment_like_its_siblings(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["settle-trades", "--plugin", "fake", "-a", str(tmp_path)],
        env={"PMLAB_WORKSPACE": "from_env"},
    )
    assert result.exit_code == 1
    assert str(_trade_log(tmp_path, "from_env")) in result.stderr


def test_touches_nothing(tmp_path: Path, log: Path) -> None:
    before = log.read_bytes()
    files_before = sorted(p for p in tmp_path.rglob("*"))
    _invoke(tmp_path)
    assert log.read_bytes() == before
    assert sorted(p for p in tmp_path.rglob("*")) == files_before
    assert json.loads(log.read_text())["trades"][0]["realized_pnl"] is None


def test_does_not_need_the_trade_log_to_exist(tmp_path: Path) -> None:
    result = _invoke(tmp_path)
    assert result.exit_code == 1
    assert not tmp_path.joinpath("workspaces").exists()


def test_the_command_surface_is_unchanged() -> None:
    """The snapshot (tests/api) pins this too: a change of exit code is not a change of options."""
    out = runner.invoke(app, ["settle-trades", "--help"]).output
    for flag in ("--plugin", "--workspace", "--artifacts-dir"):
        assert flag in out


def _printed_call(stderr: str) -> str:
    """The python block of the message: the indented lines that open with its first import."""
    lines = stderr.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "from pathlib import Path")
    block = []
    for line in lines[start:]:
        if line.strip() and not line.startswith("    "):
            break
        block.append(line)
    return textwrap.dedent("\n".join(block))


def test_the_printed_call_really_settles_the_trades(tmp_path: Path, log: Path) -> None:
    """Run what the message says, as written, with the two names it tells the user to supply."""
    code = _printed_call(_invoke(tmp_path).stderr)
    assert "SettlementEngine" in code
    assert not re.search(r"\.\.\.", code), "the example must be runnable, not an ellipsis"

    namespace: dict[str, Any] = {"plugin": _Final(), "specs": [_spec()]}
    exec(compile(code, "<settle-trades message>", "exec"), namespace)  # noqa: S102

    trade = json.loads(log.read_text())["trades"][0]
    assert trade["realized_pnl"] is not None
    assert trade["outcome"] == "won"


def test_an_unfinished_truth_leaves_the_trade_open_through_the_printed_call(
    tmp_path: Path, log: Path
) -> None:
    """The other side: the engine, not the message, holds a trade back until the truth is final."""

    class _NotFinal(_Final):
        def is_truth_final(self, spec: MarketSpec, **kwargs: Any) -> bool:
            return False

    code = _printed_call(_invoke(tmp_path).stderr)
    namespace: dict[str, Any] = {"plugin": _NotFinal(), "specs": [_spec()]}
    exec(compile(code, "<settle-trades message>", "exec"), namespace)  # noqa: S102
    assert json.loads(log.read_text())["trades"][0]["realized_pnl"] is None
