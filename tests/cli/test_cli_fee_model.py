"""`--fee-model {polymarket,flat-bps}` on the commands that take `--costs` (#72).

`polymarket` is the new default: a backtest charges the price-dependent taker fee. `flat-bps` is the
model every release before 1.5.0 used (0.30% of the stake), kept so a run can be repeated. A costs
file that says which model it is wins over nothing: naming a different one on the command line is a
contradiction and exits 1. A costs file written before 1.5.0 has a `taker_bps` and no `fee_model`:
it is the flat model, and the CLI says that spelling is deprecated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from pmlab.core.costs import CostModel
from tests.cli.test_cli_costs import (
    _backtest,
    _costs_file,
    _promote,
    _published_champion,
    _record,
    _signals,
    _trades_file,
    _unwrapped,
    _write,
)

runner = CliRunner()

FLAT = CostModel(fee_model="flat-bps")
OLD_COSTS = {
    "taker_bps": 30.0,
    "slippage_bps": 0.0,
    "slippage_fixed": 0.0,
    "depth_bps_per_unit": 0.0,
    "depth_reference_stake": 1.0,
}


class TestBacktestFeeModel:
    def test_the_default_is_the_polymarket_model(self, tmp_path: Path) -> None:
        result, metrics = _backtest(tmp_path)
        assert result.exit_code == 0, result.output
        assert metrics["costs"] == CostModel().to_dict()
        assert metrics["costs"]["fee_model"] == "polymarket"

    def test_naming_polymarket_is_the_default(self, tmp_path: Path) -> None:
        _, default = _backtest(tmp_path, name="a.json")
        result, named = _backtest(tmp_path, "--fee-model", "polymarket", name="b.json")
        assert result.exit_code == 0, result.output
        assert named["costs"] == default["costs"]
        assert named["total_pnl"] == default["total_pnl"]

    def test_flat_bps_is_the_old_model(self, tmp_path: Path) -> None:
        result, metrics = _backtest(tmp_path, "--fee-model", "flat-bps")
        assert result.exit_code == 0, result.output
        assert metrics["costs"] == FLAT.to_dict()

    def test_the_fee_model_changes_the_pnl_the_way_the_fees_say(self, tmp_path: Path) -> None:
        _, polymarket = _backtest(tmp_path, name="p.json")
        _, flat = _backtest(tmp_path, "--fee-model", "flat-bps", name="f.json")
        assert polymarket["total_trades"] == flat["total_trades"]
        # a price of 0.45: 0.05 x 0.55 = 2.75% of the stake against 0.30%
        expected = (0.05 * 0.55 - 0.003) * flat["total_trades"]
        assert flat["total_pnl"] - polymarket["total_pnl"] == pytest.approx(expected, abs=1e-6)

    def test_an_unknown_fee_model_is_a_clean_error_naming_the_choices(self, tmp_path: Path) -> None:
        result, _ = _backtest(tmp_path, "--fee-model", "kalshi")
        assert result.exit_code == 1
        assert "polymarket" in result.output and "flat-bps" in result.output
        assert "Traceback" not in result.output

    def test_the_summary_shows_total_fees_and_their_share_of_the_stake(
        self, tmp_path: Path
    ) -> None:
        result, metrics = _backtest(tmp_path)
        out = _unwrapped(result.output)
        assert "Total fees" in out
        assert "% of stake" in out
        n = metrics["total_trades"]
        assert metrics["total_fees"] == pytest.approx(0.05 * 0.55 * n, abs=1e-6)
        assert metrics["fees_pct_of_stake"] == pytest.approx(5.0 * 0.55, abs=1e-6)

    def test_the_flat_summary_shows_the_flat_share(self, tmp_path: Path) -> None:
        _, metrics = _backtest(tmp_path, "--fee-model", "flat-bps")
        assert metrics["fees_pct_of_stake"] == pytest.approx(0.30, abs=1e-6)


class TestTheCostsFileAndTheFlag:
    def test_a_file_of_only_slippage_gets_the_flags_fee_model(self, tmp_path: Path) -> None:
        costs = _write(tmp_path / "s.json", {"slippage_bps": 25.0})
        _, default = _backtest(tmp_path, "--costs", str(costs), name="a.json")
        _, flat = _backtest(
            tmp_path, "--costs", str(costs), "--fee-model", "flat-bps", name="b.json"
        )
        assert default["costs"] == CostModel(slippage_bps=25.0).to_dict()
        assert flat["costs"] == CostModel(fee_model="flat-bps", slippage_bps=25.0).to_dict()

    def test_a_file_that_names_its_model_needs_no_flag(self, tmp_path: Path) -> None:
        costs = _costs_file(tmp_path, FLAT)
        result, metrics = _backtest(tmp_path, "--costs", str(costs))
        assert result.exit_code == 0, result.output
        assert metrics["costs"] == FLAT.to_dict()

    def test_the_same_model_named_twice_is_fine(self, tmp_path: Path) -> None:
        costs = _costs_file(tmp_path, FLAT)
        result, _ = _backtest(tmp_path, "--costs", str(costs), "--fee-model", "flat-bps")
        assert result.exit_code == 0, result.output

    def test_a_flag_that_contradicts_the_file_is_refused(self, tmp_path: Path) -> None:
        costs = _costs_file(tmp_path, FLAT)
        result, metrics = _backtest(tmp_path, "--costs", str(costs), "--fee-model", "polymarket")
        assert result.exit_code == 1
        assert "flat-bps" in result.output and "polymarket" in result.output
        assert metrics == {}

    def test_a_polymarket_file_contradicts_flat_bps(self, tmp_path: Path) -> None:
        costs = _costs_file(tmp_path, CostModel())
        result, _ = _backtest(tmp_path, "--costs", str(costs), "--fee-model", "flat-bps")
        assert result.exit_code == 1

    def test_a_file_written_before_1_5_0_is_the_flat_model_and_the_cli_says_so(
        self, tmp_path: Path
    ) -> None:
        costs = _write(tmp_path / "old.json", OLD_COSTS)
        result, metrics = _backtest(tmp_path, "--costs", str(costs))
        assert result.exit_code == 0, result.output
        assert metrics["costs"] == FLAT.to_dict()
        out = _unwrapped(result.output)
        assert "warning:" in out and "taker_bps" in out and "deprecated" in out
        assert "flat-bps" in out

    def test_a_file_that_names_its_model_does_not_warn(self, tmp_path: Path) -> None:
        costs = _costs_file(tmp_path, FLAT)
        result, _ = _backtest(tmp_path, "--costs", str(costs))
        assert "deprecated" not in result.output

    def test_an_old_file_with_the_polymarket_flag_is_a_contradiction(self, tmp_path: Path) -> None:
        costs = _write(tmp_path / "old.json", OLD_COSTS)
        result, _ = _backtest(tmp_path, "--costs", str(costs), "--fee-model", "polymarket")
        assert result.exit_code == 1

    def test_an_old_file_with_the_flat_flag_is_what_it_says(self, tmp_path: Path) -> None:
        costs = _write(tmp_path / "old.json", OLD_COSTS)
        result, metrics = _backtest(tmp_path, "--costs", str(costs), "--fee-model", "flat-bps")
        assert result.exit_code == 0, result.output
        assert metrics["costs"] == FLAT.to_dict()


class TestPromoteAndRecordFollowTheFeeModel:
    def test_promote_records_the_named_model(self, tmp_path: Path) -> None:
        out = tmp_path / "art"
        result = _promote(tmp_path, out, "--fee-model", "flat-bps")
        assert result.exit_code == 0, result.output
        champion = json.loads((out / "champion.json").read_text())
        assert CostModel.from_dict(champion["costs"]) == FLAT

    def test_record_trades_charges_the_polymarket_fee_by_default(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=None)
        _signals(artifacts, "ws")
        assert _record(artifacts, "ws").exit_code == 0
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        # a YES at 0.40: 0.05 x 0.60
        assert {t["fee_paid"] for t in trades} == {0.03}
        assert {t["fee_source"] for t in trades} == {"category_default"}

    def test_record_trades_with_flat_bps_charges_the_old_fee(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=None)
        _signals(artifacts, "ws")
        assert _record(artifacts, "ws", "--fee-model", "flat-bps").exit_code == 0
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert {t["fee_paid"] for t in trades} == {0.003}
        assert {t["fee_source"] for t in trades} == {"flat_bps"}

    def test_a_champion_promoted_under_flat_fees_keeps_charging_them(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=FLAT)
        _signals(artifacts, "ws")
        assert _record(artifacts, "ws").exit_code == 0
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert {t["fee_paid"] for t in trades} == {0.003}

    def test_a_champion_json_written_before_1_5_0_is_read_as_flat_bps(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=FLAT)
        path = artifacts / "public_models" / "champion.json"
        written: dict[str, Any] = json.loads(path.read_text())
        written["costs"] = OLD_COSTS
        path.write_text(json.dumps(written))
        _signals(artifacts, "ws")
        result = _record(artifacts, "ws")
        assert result.exit_code == 0, result.output
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert {t["fee_paid"] for t in trades} == {0.003}

    def test_a_flag_that_contradicts_the_champion_is_refused(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=FLAT)
        _signals(artifacts, "ws")
        result = _record(artifacts, "ws", "--fee-model", "polymarket")
        assert result.exit_code == 1
        assert "differ" in _unwrapped(result.output)
        assert not _trades_file(artifacts, "ws").exists()


class TestReportCommandShowsFees:
    def test_the_report_command_prints_the_fee_total(self, tmp_path: Path) -> None:
        trades = tmp_path / "trades.json"
        _write(
            trades,
            {
                "trades": [
                    {"realized_pnl": 1.0, "fee_paid": 0.25, "flat_stake": 5.0},
                    {"realized_pnl": -5.0, "fee_paid": 0.05, "flat_stake": 5.0},
                ]
            },
        )
        result = runner.invoke(
            app, ["report", "--trades", str(trades), "--output", str(tmp_path / "r.html")]
        )
        assert result.exit_code == 0, result.output
        out = _unwrapped(result.output)
        assert "Total fees: $0.30" in out
        assert "3.00% of stake" in out

    def test_a_log_without_fees_prints_no_fee_line(self, tmp_path: Path) -> None:
        trades = tmp_path / "trades.json"
        _write(trades, {"trades": [{"realized_pnl": 1.0, "flat_stake": 5.0}]})
        result = runner.invoke(
            app, ["report", "--trades", str(trades), "--output", str(tmp_path / "r.html")]
        )
        assert result.exit_code == 0, result.output
        assert "Total fees" not in result.output
