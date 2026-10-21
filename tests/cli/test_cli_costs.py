"""The CLI carries one cost model from the backtest to the champion to the paper trades.

`CostModel` was reachable only from Python: `pmlab backtest` ran at the default
costs, `promote-champion` published `costs=None`, and `record-trades` built a
`PaperBroker` without them. A CLI user therefore got a champion whose cost basis the
watchdog could not check, and live fills that did not match whatever the backtest was
run with. These tests drive the real commands, end to end, through to `pmlab monitor`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from pmlab.core.costs import CostModel
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.settlement import SettlementEngine
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.plugins.base import MarketPlugin

runner = CliRunner()

_COSTS = CostModel(taker_bps=30.0, slippage_bps=100.0, slippage_fixed=0.002)
_FAMILY = "politics"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _costs_file(tmp_path: Path, costs: CostModel = _COSTS, name: str = "costs.json") -> Path:
    return _write(tmp_path / name, costs.to_dict())


def _panel(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for d in range(60):
        winner = rng.choice(["YES", "NO"])
        day = f"2026-01-{d + 1:02d}" if d < 28 else f"2026-02-{(d % 28) + 1:02d}"
        for label in ("YES", "NO"):
            rows.append(
                {
                    "market_id": f"m{d}",
                    "decision_date": day,
                    "outcome_label": label,
                    "winning_label": winner,
                    "market_price": 0.45,
                    "feature_x": float(rng.normal()),
                    "segment": "A",
                }
            )
    path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _model(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"feature_x": rng.normal(size=80)})  # noqa: N806
    model = LGBMForecaster(n_estimators=5, num_leaves=4, min_child_samples=5)
    model.fit(X, pd.Series((X["feature_x"] > 0).astype(int)))
    path = tmp_path / "model.pkl"
    model.save(path)
    return path


def _gate(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "gate.json",
        {
            "decision": "GO",
            "segment_results": [
                {
                    "segment": "A",
                    "num_trades": 100,
                    "total_pnl": 10.0,
                    "passes": True,
                    "reason": "ok",
                }
            ],
            "aggregate_pnl": 10.0,
            "aggregate_trades": 100,
        },
    )


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
            str(_panel(tmp_path)),
            "--stride",
            "10",
            "--min-train-rows",
            "10",
            "--output",
            str(out),
            *extra,
        ],  # fmt: skip
    )
    metrics: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
    return result, metrics


def _promote(tmp_path: Path, out_dir: Path, *extra: str) -> Any:
    return runner.invoke(
        app,
        [
            "promote-champion",
            str(_model(tmp_path)),
            "--gate-path",
            str(_gate(tmp_path)),
            "--plugin",
            _FAMILY,
            "--output-dir",
            str(out_dir),
            *extra,
        ],  # fmt: skip
    )


def _signals(artifacts: Path, workspace: str, n: int = 12) -> None:
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
            "best_edge": 0.30,
            "yes_edge": 0.30,
            "no_edge": -0.30,
            "question": "q",
        }
        for i in range(n)
    ]
    _write(
        artifacts / "workspaces" / workspace / "signals" / "v2" / "scan_edge_signals.json",
        {"signals": signals},
    )


def _trades_file(artifacts: Path, workspace: str) -> Path:
    return artifacts / "workspaces" / workspace / "signals" / "v2" / "forward_paper_trades.json"


def _record(artifacts: Path, workspace: str, *extra: str) -> Any:
    return runner.invoke(
        app,
        [
            "record-trades",
            "--plugin",
            _FAMILY,
            "--artifacts-dir",
            str(artifacts),
            "--workspace",
            workspace,
            *extra,
        ],  # fmt: skip
    )


# --------------------------------------------------------------------------
# --costs file handling
# --------------------------------------------------------------------------


class TestTheCostsFile:
    @pytest.mark.parametrize("command", ["backtest", "promote", "record"])
    def test_a_missing_costs_file_is_a_clean_error(self, tmp_path: Path, command: str) -> None:
        missing = str(tmp_path / "nope.json")
        result = self._run(tmp_path, command, missing)
        assert result.exit_code == 1
        assert "nope.json" in result.output
        assert "Traceback" not in result.output

    @pytest.mark.parametrize(
        ("payload", "needle"),
        [
            ("{not json", "costs"),
            ('{"taker_bps": -1}', "taker_bps"),
            ('{"slippage_pips": 3}', "slippage_pips"),
            ('{"depth_reference_stake": 0}', "depth_reference_stake"),
            ("[1, 2]", "costs"),
        ],
    )
    def test_an_unusable_costs_file_is_a_clean_error(
        self, tmp_path: Path, payload: str, needle: str
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(payload)
        result = self._run(tmp_path, "backtest", str(bad))
        assert result.exit_code == 1
        assert needle in result.output
        assert not (tmp_path / "metrics.json").exists()

    def test_a_costs_object_nested_under_costs_is_accepted(self, tmp_path: Path) -> None:
        """So the metrics file `backtest --output` writes can be handed straight back."""
        wrapped = _write(tmp_path / "wrapped.json", {"costs": _COSTS.to_dict(), "total_pnl": 1.0})
        result, metrics = _backtest(tmp_path, "--costs", str(wrapped))
        assert result.exit_code == 0, result.output
        assert metrics["costs"] == _COSTS.to_dict()

    @staticmethod
    def _run(tmp_path: Path, command: str, costs: str) -> Any:
        if command == "backtest":
            return _backtest(tmp_path, "--costs", costs)[0]
        if command == "promote":
            return _promote(tmp_path, tmp_path / "out", "--costs", costs)
        champion = _published_champion(tmp_path / "art", costs=None)
        _signals(champion, "ws")
        return _record(champion, "ws", "--costs", costs)


def _published_champion(artifacts: Path, *, costs: CostModel | None) -> Path:
    """Publish a champion into <artifacts>/public_models through the real CLI."""
    scratch = artifacts.parent / f"{artifacts.name}-inputs"
    scratch.mkdir(parents=True, exist_ok=True)
    extra = ["--costs", str(_costs_file(scratch, costs))] if costs is not None else []
    result = _promote(scratch, artifacts / "public_models", *extra)
    assert result.exit_code == 0, result.output
    return artifacts


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------


class TestBacktestCosts:
    def test_the_costs_change_the_backtest(self, tmp_path: Path) -> None:
        default_run, default = _backtest(tmp_path, name="default.json")
        costly_run, costly = _backtest(
            tmp_path, "--costs", str(_costs_file(tmp_path)), name="costly.json"
        )
        assert default_run.exit_code == 0 and costly_run.exit_code == 0
        assert costly["total_trades"] == default["total_trades"]
        assert costly["total_pnl"] < default["total_pnl"]

    def test_the_metrics_record_the_costs_that_produced_them(self, tmp_path: Path) -> None:
        _, metrics = _backtest(tmp_path, "--costs", str(_costs_file(tmp_path)))
        assert metrics["costs"] == _COSTS.to_dict()

    def test_without_costs_the_metrics_record_the_default_model(self, tmp_path: Path) -> None:
        """So a later promote can see what the expectation was computed under."""
        _, metrics = _backtest(tmp_path)
        assert metrics["costs"] == CostModel().to_dict()

    def test_the_expectation_is_computed_net_of_the_costs(self, tmp_path: Path) -> None:
        _, default = _backtest(tmp_path, name="default.json")
        _, costly = _backtest(tmp_path, "--costs", str(_costs_file(tmp_path)), name="costly.json")
        assert (
            costly["expectation"]["mean_pnl_per_stake"]
            < default["expectation"]["mean_pnl_per_stake"]
        )


# --------------------------------------------------------------------------
# promote-champion
# --------------------------------------------------------------------------


class TestPromoteRecordsTheCosts:
    def test_costs_flag_is_written_to_champion_json(self, tmp_path: Path) -> None:
        result = _promote(tmp_path, tmp_path / "out", "--costs", str(_costs_file(tmp_path)))
        assert result.exit_code == 0, result.output
        on_disk = json.loads((tmp_path / "out" / "champion.json").read_text())
        assert on_disk["costs"] == _COSTS.to_dict()
        assert ChampionManifest.load(tmp_path / "out" / "champion.json").costs == _COSTS

    def test_costs_come_from_the_backtest_metrics_when_not_given(self, tmp_path: Path) -> None:
        """One flag at the backtest is enough: the promotion cannot forget it."""
        _, metrics = _backtest(tmp_path, "--costs", str(_costs_file(tmp_path)))
        result = _promote(
            tmp_path, tmp_path / "out", "--expectation", str(tmp_path / "metrics.json")
        )
        assert result.exit_code == 0, result.output
        assert ChampionManifest.load(tmp_path / "out" / "champion.json").costs == _COSTS
        assert metrics["costs"] == _COSTS.to_dict()

    def test_the_default_backtest_costs_are_recorded_too(self, tmp_path: Path) -> None:
        _backtest(tmp_path)
        _promote(tmp_path, tmp_path / "out", "--expectation", str(tmp_path / "metrics.json"))
        assert ChampionManifest.load(tmp_path / "out" / "champion.json").costs == CostModel()

    def test_a_costs_flag_that_contradicts_the_expectation_is_refused(self, tmp_path: Path) -> None:
        _backtest(tmp_path, "--costs", str(_costs_file(tmp_path)))
        other = _costs_file(tmp_path, CostModel(taker_bps=80.0), name="other.json")
        result = _promote(
            tmp_path, tmp_path / "out", "--expectation", str(tmp_path / "metrics.json"),
            "--costs", str(other),
        )  # fmt: skip
        assert result.exit_code == 1
        assert "differ" in result.output
        assert not (tmp_path / "out" / "champion.json").exists()

    def test_the_same_costs_given_twice_are_fine(self, tmp_path: Path) -> None:
        _backtest(tmp_path, "--costs", str(_costs_file(tmp_path)))
        result = _promote(
            tmp_path, tmp_path / "out", "--expectation", str(tmp_path / "metrics.json"),
            "--costs", str(_costs_file(tmp_path, name="again.json")),
        )  # fmt: skip
        assert result.exit_code == 0, result.output

    def test_a_bare_expectation_with_no_costs_leaves_the_champion_without_them(
        self, tmp_path: Path
    ) -> None:
        """Nothing is guessed: no costs anywhere means 'unknown', as before."""
        _, metrics = _backtest(tmp_path)
        bare = _write(tmp_path / "bare.json", metrics["expectation"])
        result = _promote(tmp_path, tmp_path / "out", "--expectation", str(bare))
        assert result.exit_code == 0, result.output
        assert ChampionManifest.load(tmp_path / "out" / "champion.json").costs is None

    def test_without_any_costs_the_champion_has_none(self, tmp_path: Path) -> None:
        assert _promote(tmp_path, tmp_path / "out").exit_code == 0
        assert ChampionManifest.load(tmp_path / "out" / "champion.json").costs is None


# --------------------------------------------------------------------------
# record-trades
# --------------------------------------------------------------------------


class TestRecordTradesUsesTheChampionsCosts:
    def test_the_champions_costs_are_the_default(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=_COSTS)
        _signals(artifacts, "ws")
        result = _record(artifacts, "ws")
        assert result.exit_code == 0, result.output
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert len(trades) == 12
        expected_fill = round(_COSTS.fill_price(0.40, stake=1.0), 8)
        assert expected_fill > 0.40
        assert {t["fill_price"] for t in trades} == {expected_fill}
        assert {t["fee_paid"] for t in trades} == {round(_COSTS.fee(1.0), 6)}

    def test_a_champion_without_costs_records_at_the_default_model(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=None)
        _signals(artifacts, "ws")
        assert _record(artifacts, "ws").exit_code == 0
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert {t["fill_price"] for t in trades} == {0.4}

    def test_a_costs_flag_is_used_when_the_champion_records_none(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=None)
        _signals(artifacts, "ws")
        result = _record(artifacts, "ws", "--costs", str(_costs_file(tmp_path)))
        assert result.exit_code == 0, result.output
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert {t["fill_price"] for t in trades} == {round(_COSTS.fill_price(0.40, stake=1.0), 8)}

    def test_a_costs_flag_that_contradicts_the_champion_is_refused(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=_COSTS)
        _signals(artifacts, "ws")
        other = _costs_file(tmp_path, CostModel(taker_bps=80.0), name="other.json")
        result = _record(artifacts, "ws", "--costs", str(other))
        assert result.exit_code == 1
        assert "differ" in result.output
        assert not _trades_file(artifacts, "ws").exists()

    def test_the_same_costs_as_the_champion_are_accepted(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=_COSTS)
        _signals(artifacts, "ws")
        result = _record(artifacts, "ws", "--costs", str(_costs_file(tmp_path)))
        assert result.exit_code == 0, result.output

    def test_the_stake_is_part_of_the_fill_through_a_depth_model(self, tmp_path: Path) -> None:
        depth = CostModel(taker_bps=30.0, depth_bps_per_unit=100.0, depth_reference_stake=1.0)
        artifacts = _published_champion(tmp_path / "art", costs=depth)
        _signals(artifacts, "ws")
        _record(artifacts, "ws", "--flat-stake", "5.0")
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert {t["fill_price"] for t in trades} == {round(depth.fill_price(0.40, stake=5.0), 8)}


# --------------------------------------------------------------------------
# The whole chain, into the watchdog
# --------------------------------------------------------------------------


class _Truths(MarketPlugin):
    family = "dummy"

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> str | None:
        day = int(spec.metadata["target_date"][-2:])
        return "YES" if day % 3 else "NO"

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None


def _settle(trades_path: Path) -> None:
    specs = [
        MarketSpec(
            market_id=f"m{i}",
            slug="s",
            question="q",
            outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
            close_time="2099-12-31T00:00:00Z",
            market_family="binary",
            metadata={"city": "A", "target_date": f"2099-01-{i + 1:02d}"},
        )
        for i in range(12)
    ]
    SettlementEngine(plugin=_Truths(), trades_path=trades_path).settle_all(
        specs, today_str="2099-12-31", now_utc=datetime(2099, 12, 31, 12, tzinfo=UTC)
    )


def _monitor(artifacts: Path, workspace: str, out: Path) -> dict[str, Any]:
    result = runner.invoke(
        app,
        [
            "monitor",
            "--artifacts-dir",
            str(artifacts),
            "--workspace",
            workspace,
            "--trades",
            str(_trades_file(artifacts, workspace)),
            "--skip",
            "drift",
            "--min-trades",
            "5",
            "--as-of",
            "2100-01-01T00:00:00Z",
            "--json",
            str(out),
        ],  # fmt: skip
    )
    assert result.exit_code in (0, 10, 20, 30), (result.exit_code, result.output)
    report: dict[str, Any] = json.loads(out.read_text())
    return report


class TestTheChainReachesTheWatchdog:
    def test_a_cli_published_champion_gives_the_pnl_check_a_known_cost_basis(
        self, tmp_path: Path
    ) -> None:
        artifacts = tmp_path / "art"
        # backtest -> promote (costs picked up from the metrics) -> record -> settle -> monitor
        result, _ = _backtest(tmp_path, "--costs", str(_costs_file(tmp_path)))
        assert result.exit_code == 0, result.output
        promoted = runner.invoke(
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
                str(tmp_path / "metrics.json"),
            ],  # fmt: skip
        )
        assert promoted.exit_code == 0, promoted.output
        _signals(artifacts, "ws")
        assert _record(artifacts, "ws").exit_code == 0
        _settle(_trades_file(artifacts, "ws"))

        report = _monitor(artifacts, "ws", tmp_path / "health.json")

        pnl = next(c for c in report["checks"] if c["name"] == "pnl")
        assert report["champion"]["costs"] == _COSTS.to_dict()
        assert pnl["metrics"]["cost_basis"] == "match"
        assert pnl["metrics"]["n_excluded_cost_basis"] == 0
        assert pnl["metrics"]["n_cost_unverified"] == 0
        assert pnl["metrics"]["n_excluded_legacy_settlement"] == 0
        assert pnl["metrics"]["n_trades"] == 12

    def test_a_champion_published_without_costs_is_still_unknown(self, tmp_path: Path) -> None:
        """The control: what every CLI user got before."""
        artifacts = _published_champion(tmp_path / "art", costs=None)
        _signals(artifacts, "ws")
        assert _record(artifacts, "ws").exit_code == 0
        _settle(_trades_file(artifacts, "ws"))

        report = _monitor(artifacts, "ws", tmp_path / "health.json")

        pnl = next(c for c in report["checks"] if c["name"] == "pnl")
        assert report["champion"]["costs"] is None
        assert pnl["metrics"]["cost_basis"] == "unknown"

    def test_trades_filled_under_other_costs_would_have_been_caught(self, tmp_path: Path) -> None:
        """Paper trades recorded without the champion's costs show up as a mismatch.

        Written straight into the log (the CLI now refuses to record them), this is
        what the watchdog sees from a book filled at the quote under a costed champion.
        """
        artifacts = _published_champion(tmp_path / "art", costs=_COSTS)
        _signals(artifacts, "ws")
        assert _record(artifacts, "ws").exit_code == 0
        path = _trades_file(artifacts, "ws")
        data = json.loads(path.read_text())
        for trade in data["trades"]:
            trade["fill_price"] = trade["gamma_price"]
        path.write_text(json.dumps(data))
        _settle(path)

        report = _monitor(artifacts, "ws", tmp_path / "health.json")

        pnl = next(c for c in report["checks"] if c["name"] == "pnl")
        assert pnl["metrics"]["cost_basis"] == "mismatch"
        assert pnl["metrics"]["n_excluded_cost_basis"] == 12
