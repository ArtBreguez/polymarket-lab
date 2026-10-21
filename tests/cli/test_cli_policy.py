"""The CLI carries one trade policy from the backtest to the champion to the paper trades.

`record-trades` buys the better of YES and NO, above `--min-edge`. The backtest has to
simulate that, or the champion is promoted on a strategy that is not the one trading and
the watchdog has nothing to hold the live NO trades against. These tests drive the real
commands: `backtest` runs the live policy by default, the champion records it, and
`record-trades` takes its threshold from there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from pmlab.core.edge import DEFAULT_MIN_EDGE
from tests.cli.test_cli_costs import (
    _FAMILY,
    _gate,
    _model,
    _monitor,
    _record,
    _settle,
    _trades_file,
    _write,
)

runner = CliRunner()


def _policy_panel(tmp_path: Path, *, n_dates: int = 150, n_markets: int = 8) -> Path:
    """A feature that says exactly whether the label wins, and prices that ignore it.

    Where the feature says "wins" the model sees ~1.0 and YES is the trade; where it says
    "loses" the model sees ~0.0 and NO is. Both sides clear any sensible threshold.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", periods=n_dates).strftime("%Y-%m-%d")
    rows = []
    for day in dates:
        for market in range(n_markets):
            wins = bool(rng.integers(0, 2))
            rows.append(
                {
                    "market_id": f"m{day}-{market}",
                    "decision_date": day,
                    "outcome_label": "hit",
                    "winning_label": "hit" if wins else "miss",
                    "market_price": float(rng.uniform(0.3, 0.7)),
                    "feature_x": (1.0 if wins else -1.0) + float(rng.normal(0, 0.05)),
                    "segment": "A",
                }
            )
    path = tmp_path / "policy_panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _backtest(
    tmp_path: Path, *extra: str, name: str = "metrics.json"
) -> tuple[Any, dict[str, Any]]:
    out = tmp_path / name
    result = runner.invoke(
        app,
        [
            "backtest",
            "--plugin",
            "x",
            "--panel",
            str(_policy_panel(tmp_path)),
            "--stride",
            "10",
            "--min-train-rows",
            "20",
            "--output",
            str(out),
            *extra,
        ],  # fmt: skip
    )
    metrics: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
    return result, metrics


class TestBacktestRunsTheLivePolicyByDefault:
    def test_both_sides_are_traded(self, tmp_path: Path) -> None:
        result, metrics = _backtest(tmp_path)
        assert result.exit_code == 0, result.output
        split = metrics["expectation"]["by_direction"]
        assert set(split) == {"yes", "no"}
        assert split["yes"]["n_trades"] >= 30
        assert split["no"]["n_trades"] >= 30

    def test_the_policy_is_recorded_with_the_expectation(self, tmp_path: Path) -> None:
        _, metrics = _backtest(tmp_path)
        assert metrics["expectation"]["policy"] == {
            "selection": "best_edge",
            "directions": ["yes", "no"],
            "min_edge": DEFAULT_MIN_EDGE,
        }

    def test_a_stricter_threshold_trades_less_and_is_recorded(self, tmp_path: Path) -> None:
        _, loose = _backtest(tmp_path, "--min-edge", "0.1", name="loose.json")
        _, strict = _backtest(tmp_path, "--min-edge", "0.6", name="strict.json")
        assert strict["total_trades"] < loose["total_trades"]
        assert strict["expectation"]["policy"]["min_edge"] == 0.6

    def test_the_output_names_the_policy(self, tmp_path: Path) -> None:
        result, _ = _backtest(tmp_path, "--min-edge", "0.15")
        assert "live" in result.output
        assert "0.15" in result.output

    def test_the_historical_policy_is_one_flag_away(self, tmp_path: Path) -> None:
        result, metrics = _backtest(tmp_path, "--policy", "top-bin-yes")
        assert result.exit_code == 0, result.output
        assert set(metrics["expectation"]["by_direction"]) == {"yes"}
        assert metrics["expectation"]["policy"]["selection"] == "top_prob"
        # One YES trade per market on every evaluated date, edge or not.
        assert metrics["total_trades"] % 8 == 0

    def test_the_historical_policy_has_no_threshold_to_set(self, tmp_path: Path) -> None:
        result, _ = _backtest(tmp_path, "--policy", "top-bin-yes", "--min-edge", "0.1")
        assert result.exit_code == 1
        assert "min-edge" in result.output

    def test_an_unknown_policy_is_refused(self, tmp_path: Path) -> None:
        result, _ = _backtest(tmp_path, "--policy", "martingale")
        assert result.exit_code == 1
        assert "martingale" in result.output

    def test_a_threshold_nothing_clears_says_so(self, tmp_path: Path) -> None:
        result, _ = _backtest(tmp_path, "--min-edge", "5")
        assert result.exit_code == 0
        assert "No trades" in result.output
        assert "--min-edge" in result.output


def _publish(tmp_path: Path, artifacts: Path, metrics: Path) -> None:
    result = runner.invoke(
        app,
        [
            "promote-champion",
            str(_model(tmp_path)),
            "--gate-path",
            str(_gate(tmp_path)),
            "--plugin",
            _FAMILY,
            "--output-dir",
            str(artifacts / "public_models"),
            "--expectation",
            str(metrics),
        ],  # fmt: skip
    )
    assert result.exit_code == 0, result.output


class TestThePolicyReachesTheChampion:
    def test_champion_json_records_the_policy_and_the_split(self, tmp_path: Path) -> None:
        _backtest(tmp_path, "--min-edge", "0.1")
        artifacts = tmp_path / "art"
        _publish(tmp_path, artifacts, tmp_path / "metrics.json")
        recorded = json.loads((artifacts / "public_models" / "champion.json").read_text())
        expectation = recorded["backtest_expectation"]
        assert expectation["policy"]["min_edge"] == 0.1
        assert set(expectation["by_direction"]) == {"yes", "no"}


def _champion_with_policy(tmp_path: Path, min_edge: float | None) -> Path:
    """Publish a champion whose recorded policy has this threshold (None: no policy)."""
    artifacts = tmp_path / "art"
    _backtest(tmp_path, "--min-edge", "0.1" if min_edge is None else str(min_edge))
    _publish(tmp_path, artifacts, tmp_path / "metrics.json")
    if min_edge is None:
        path = artifacts / "public_models" / "champion.json"
        data = json.loads(path.read_text())
        del data["backtest_expectation"]["policy"]
        path.write_text(json.dumps(data))
    return artifacts


def _signals_at(artifacts: Path, best_edge: float, n: int = 12) -> None:
    signals = [
        {
            "market_id": f"m{i}",
            "city_or_segment": "A",
            "target_date": f"2099-01-{i + 1:02d}",
            "horizon": "morning_of",
            "outcome_label": "YES",
            "direction": "yes",
            "gamma_price": 0.40,
            "model_prob": 0.70,
            "best_edge": best_edge,
            "yes_edge": best_edge,
            "no_edge": -best_edge,
            "question": "q",
        }
        for i in range(n)
    ]
    _write(
        artifacts / "workspaces" / "ws" / "signals" / "v2" / "scan_edge_signals.json",
        {"signals": signals},
    )


def _recorded(artifacts: Path) -> int:
    path = _trades_file(artifacts, "ws")
    return len(json.loads(path.read_text())["trades"]) if path.exists() else 0


class TestRecordTradesTakesTheThresholdFromTheChampion:
    def test_the_champions_threshold_is_the_default(self, tmp_path: Path) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.30)
        _signals_at(artifacts, 0.25)  # clears the CLI's old default of 0.20, not 0.30
        result = _record(artifacts, "ws")
        assert result.exit_code == 0, result.output
        assert _recorded(artifacts) == 0
        assert "0.3" in result.output

    def test_a_signal_above_the_champions_threshold_is_recorded(self, tmp_path: Path) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.05)
        _signals_at(artifacts, 0.10)  # below the old default of 0.20
        assert _record(artifacts, "ws").exit_code == 0
        assert _recorded(artifacts) == 12

    def test_an_explicit_threshold_wins_but_is_called_out(self, tmp_path: Path) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.30)
        _signals_at(artifacts, 0.25)
        result = _record(artifacts, "ws", "--min-edge", "0.2")
        assert result.exit_code == 0, result.output
        assert _recorded(artifacts) == 12
        assert "differs from the champion's policy" in result.output.replace("\n", " ")

    def test_an_explicit_threshold_equal_to_the_champions_is_quiet(self, tmp_path: Path) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.30)
        _signals_at(artifacts, 0.35)
        result = _record(artifacts, "ws", "--min-edge", "0.3")
        assert "differs" not in result.output

    def test_a_champion_with_no_recorded_policy_keeps_the_old_default(self, tmp_path: Path) -> None:
        artifacts = _champion_with_policy(tmp_path, None)
        _signals_at(artifacts, 0.15)
        assert _record(artifacts, "ws").exit_code == 0
        assert _recorded(artifacts) == 0  # 0.15 < 0.20
        _signals_at(artifacts, 0.25)
        assert _record(artifacts, "ws").exit_code == 0
        assert _recorded(artifacts) == 12

    def test_an_edge_equal_to_the_threshold_up_to_round_off_is_recorded(
        self, tmp_path: Path
    ) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.1)
        _signals_at(artifacts, 0.6 - 0.5)  # 0.09999999999999998
        assert _record(artifacts, "ws").exit_code == 0
        assert _recorded(artifacts) == 12


class TestTheWholeChainJudgesNoTrades:
    """backtest (both sides) -> promote -> record -> settle -> monitor: a NO trade is
    held to a NO expectation, not left out."""

    def test_no_trades_reach_the_pnl_check(self, tmp_path: Path) -> None:
        _backtest(tmp_path, "--min-edge", "0.1")
        artifacts = tmp_path / "art"
        _publish(tmp_path, artifacts, tmp_path / "metrics.json")
        signals = []
        for i in range(12):
            no = i % 2 == 1
            signals.append(
                {
                    "market_id": f"m{i}",
                    "city_or_segment": "A",
                    "target_date": f"2099-01-{i + 1:02d}",
                    "horizon": "morning_of",
                    "outcome_label": "YES",
                    "direction": "no" if no else "yes",
                    "gamma_price": 0.60 if no else 0.40,
                    "model_prob": 0.15 if no else 0.70,
                    "best_edge": 0.4,
                    "yes_edge": -0.4 if no else 0.3,
                    "no_edge": 0.4 if no else -0.3,
                    "question": "q",
                }
            )
        _write(
            artifacts / "workspaces" / "ws" / "signals" / "v2" / "scan_edge_signals.json",
            {"signals": signals},
        )
        assert _record(artifacts, "ws").exit_code == 0
        _settle(_trades_file(artifacts, "ws"))

        report = _monitor(artifacts, "ws", tmp_path / "health.json")
        pnl = next(c for c in report["checks"] if c["name"] == "pnl")["metrics"]
        calibration = next(c for c in report["checks"] if c["name"] == "calibration")["metrics"]

        assert pnl["n_by_direction"] == {"yes": 6, "no": 6}
        assert pnl["n_excluded_no_side"] == 0
        assert pnl["n_excluded_unfixed_no"] == 0
        assert pnl["n_trades"] == 12
        assert set(pnl["directions"]) == {"yes", "no"}
        assert calibration["n_by_direction"] == {"yes": 6, "no": 6}


@pytest.mark.parametrize("flag", ["--policy", "--min-edge"])
def test_the_options_are_documented_in_help(flag: str) -> None:
    assert flag in runner.invoke(app, ["backtest", "--help"]).output


def test_a_nan_threshold_is_refused_not_traded(tmp_path: Path) -> None:
    result, _ = _backtest(tmp_path, "--min-edge", "nan")
    assert result.exit_code == 1
    assert "Not a valid --min-edge" in result.output


def _market_signals(artifacts: Path, signals: list[dict[str, Any]]) -> None:
    _write(
        artifacts / "workspaces" / "ws" / "signals" / "v2" / "scan_edge_signals.json",
        {"signals": signals},
    )


def _signal(label: str, edge: float, *, direction: str = "yes", day: int = 1) -> dict[str, Any]:
    return {
        "market_id": f"m-{label}",
        "city_or_segment": "A",
        "target_date": f"2099-01-{day:02d}",
        "horizon": "morning_of",
        "outcome_label": label,
        "direction": direction,
        "gamma_price": 0.40,
        "model_prob": 0.70,
        "best_edge": edge,
        "yes_edge": edge if direction == "yes" else -edge,
        "no_edge": edge if direction == "no" else -edge,
        "question": "q",
    }


class TestOneTradePerMarketIsTheBestOne:
    """The backtest takes, per market, the (bin, side) with the best edge. The broker records
    the first signal it is given for a (segment, date, horizon), so the best has to come first."""

    def test_the_best_signal_wins_whatever_order_the_file_lists_them_in(
        self, tmp_path: Path
    ) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.1)
        _market_signals(artifacts, [_signal("weak", 0.217), _signal("strong", 0.397)])
        assert _record(artifacts, "ws").exit_code == 0
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert [t["outcome_label"] for t in trades] == ["strong"]

    def test_equal_edges_keep_the_files_order(self, tmp_path: Path) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.1)
        _market_signals(artifacts, [_signal("first", 0.3), _signal("second", 0.3)])
        _record(artifacts, "ws")
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert [t["outcome_label"] for t in trades] == ["first"]

    def test_separate_markets_are_all_recorded(self, tmp_path: Path) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.1)
        _market_signals(artifacts, [_signal("a", 0.2, day=1), _signal("b", 0.3, day=2)])
        _record(artifacts, "ws")
        assert _recorded(artifacts) == 2


class TestOnlyTheSidesThePolicyTrades:
    def _yes_only(self, tmp_path: Path) -> Path:
        artifacts = _champion_with_policy(tmp_path, 0.1)
        path = artifacts / "public_models" / "champion.json"
        data = json.loads(path.read_text())
        data["backtest_expectation"]["policy"]["directions"] = ["yes"]
        path.write_text(json.dumps(data))
        return artifacts

    def test_a_no_signal_is_skipped_for_a_yes_only_policy_and_counted(self, tmp_path: Path) -> None:
        artifacts = self._yes_only(tmp_path)
        _market_signals(
            artifacts,
            [_signal("a", 0.3, direction="no", day=1), _signal("b", 0.3, direction="yes", day=2)],
        )
        result = _record(artifacts, "ws")
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert [t["direction"] for t in trades] == ["yes"]
        assert "1 signal on a side the champion's policy does not trade" in result.output.replace(
            "\n", " "
        )

    def test_both_sides_are_recorded_for_the_default_live_policy(self, tmp_path: Path) -> None:
        artifacts = _champion_with_policy(tmp_path, 0.1)
        _market_signals(
            artifacts,
            [_signal("a", 0.3, direction="no", day=1), _signal("b", 0.3, direction="yes", day=2)],
        )
        _record(artifacts, "ws")
        assert _recorded(artifacts) == 2

    def test_a_champion_with_no_recorded_policy_trades_what_it_is_given(
        self, tmp_path: Path
    ) -> None:
        artifacts = _champion_with_policy(tmp_path, None)
        _market_signals(artifacts, [_signal("a", 0.3, direction="no")])
        _record(artifacts, "ws")
        assert _recorded(artifacts) == 1
