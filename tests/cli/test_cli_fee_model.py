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
from pmlab.core.fee_schedule import FeeSchedule
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
# a champion that charges the weather rate with slippage: its model, but not the default's numbers
WEATHER = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="user", slippage_bps=50.0)
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

    def test_a_champion_with_no_costs_says_which_fee_model_it_is_traded_under(
        self, tmp_path: Path
    ) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=None)
        _signals(artifacts, "ws")
        out = _unwrapped(_record(artifacts, "ws").output)
        assert "records no costs" in out
        assert "polymarket" in out and "--fee-model flat-bps" in out

    def test_a_champion_that_records_its_costs_needs_no_such_note(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=FLAT)
        _signals(artifacts, "ws")
        assert "records no costs" not in _record(artifacts, "ws").output

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
        out = _unwrapped(result.output)
        # it says which two models disagree, not only that "the costs differ"
        assert "--fee-model polymarket" in out and "flat-bps" in out and "contradicts" in out
        assert not _trades_file(artifacts, "ws").exists()


class TestAFileThatNamesNoFeeModel:
    """1.4.0 read a costs file with only slippage as flat 30 bps plus that slippage; 1.5.0 reads it as
    the polymarket model. The file is hand-written, so the CLI says which it did (review r1, I3)."""

    NOTICE = "gives no fee model"

    def test_a_file_of_only_slippage_says_it_is_read_as_polymarket(self, tmp_path: Path) -> None:
        costs = _write(tmp_path / "s.json", {"slippage_bps": 25.0})
        result, metrics = _backtest(tmp_path, "--costs", str(costs))
        assert result.exit_code == 0, result.output
        assert metrics["costs"] == CostModel(slippage_bps=25.0).to_dict()
        out = _unwrapped(result.output)
        assert self.NOTICE in out
        assert "polymarket" in out and "1.4.0" in out and "flat-bps" in out

    def test_an_empty_file_says_so_too(self, tmp_path: Path) -> None:
        result, _ = _backtest(tmp_path, "--costs", str(_write(tmp_path / "e.json", {})))
        assert result.exit_code == 0, result.output
        assert self.NOTICE in _unwrapped(result.output)

    def test_a_file_under_a_costs_key_says_so_too(self, tmp_path: Path) -> None:
        costs = _write(tmp_path / "s.json", {"costs": {"slippage_bps": 25.0}})
        result, _ = _backtest(tmp_path, "--costs", str(costs))
        assert self.NOTICE in _unwrapped(result.output)

    def test_the_notice_names_the_file_and_how_to_pick_the_other_model(
        self, tmp_path: Path
    ) -> None:
        costs = _write(tmp_path / "s.json", {"slippage_bps": 25.0})
        out = _unwrapped(_backtest(tmp_path, "--costs", str(costs))[0].output)
        assert "s.json" in out
        assert "--fee-model" in out or '"fee_model"' in out

    def test_naming_the_model_on_the_command_line_needs_no_notice(self, tmp_path: Path) -> None:
        costs = _write(tmp_path / "s.json", {"slippage_bps": 25.0})
        result, _ = _backtest(tmp_path, "--costs", str(costs), "--fee-model", "flat-bps")
        assert self.NOTICE not in _unwrapped(result.output)

    @pytest.mark.parametrize(
        "fields",
        [
            {"fee_model": "polymarket", "slippage_bps": 25.0},
            {"fee_schedule": {"rate": 0.01}, "slippage_bps": 25.0},
            {"fee_model": "flat-bps", "slippage_bps": 25.0},
        ],
        ids=["model", "schedule", "flat"],
    )
    def test_a_file_that_says_its_model_gets_none(
        self, tmp_path: Path, fields: dict[str, Any]
    ) -> None:
        result, _ = _backtest(tmp_path, "--costs", str(_write(tmp_path / "s.json", fields)))
        assert result.exit_code == 0, result.output
        assert self.NOTICE not in _unwrapped(result.output)

    def test_a_pre_1_5_0_file_gets_the_deprecation_and_not_this_notice(
        self, tmp_path: Path
    ) -> None:
        result, _ = _backtest(tmp_path, "--costs", str(_write(tmp_path / "old.json", OLD_COSTS)))
        out = _unwrapped(result.output)
        assert "deprecated" in out and self.NOTICE not in out

    def test_no_costs_option_no_notice(self, tmp_path: Path) -> None:
        assert self.NOTICE not in _unwrapped(_backtest(tmp_path)[0].output)


class TestTheFlagNamingTheModelOfTheCostsOnRecord:
    """`--fee-model polymarket` against a polymarket champion names its own model: that is not a
    different cost model, and it must not be refused as one (review r1, minor 3)."""

    def test_record_trades_accepts_the_champions_own_model_and_keeps_its_costs(
        self, tmp_path: Path
    ) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=WEATHER)
        _signals(artifacts, "ws")
        result = _record(artifacts, "ws", "--fee-model", "polymarket")
        assert result.exit_code == 0, result.output
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        assert trades
        # the champion's schedule and slippage, not the default model's
        assert {t["fee_rate"] for t in trades} == {0.01}
        assert {t["fee_source"] for t in trades} == {"user"}
        assert [t["fill_price"] for t in trades] == pytest.approx([0.402] * len(trades))

    def test_record_trades_refuses_another_model_and_names_both(self, tmp_path: Path) -> None:
        artifacts = _published_champion(tmp_path / "art", costs=WEATHER)
        _signals(artifacts, "ws")
        result = _record(artifacts, "ws", "--fee-model", "flat-bps")
        assert result.exit_code == 1
        out = _unwrapped(result.output)
        assert "--fee-model flat-bps" in out
        assert "champion" in out and "polymarket" in out
        assert not _trades_file(artifacts, "ws").exists()

    def test_the_flat_champion_accepts_the_flat_flag_and_refuses_the_other(
        self, tmp_path: Path
    ) -> None:
        flat = CostModel(fee_model="flat-bps", taker_bps=45.0, slippage_bps=50.0)
        artifacts = _published_champion(tmp_path / "art", costs=flat)
        _signals(artifacts, "ws")
        assert _record(artifacts, "ws", "--fee-model", "flat-bps").exit_code == 0
        trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
        # its 45 bps, not 30
        assert [t["fee_rate"] for t in trades] == pytest.approx([0.0045] * len(trades))
        again = _record(artifacts, "ws", "--fee-model", "polymarket")
        assert again.exit_code == 1

    def test_a_costs_file_with_the_flag_is_still_compared_whole(self, tmp_path: Path) -> None:
        # the file is another set of numbers for the same model: that IS a different cost model
        artifacts = _published_champion(tmp_path / "art", costs=WEATHER)
        _signals(artifacts, "ws")
        other = _costs_file(tmp_path, CostModel(slippage_bps=10.0), name="other.json")
        result = _record(artifacts, "ws", "--costs", str(other), "--fee-model", "polymarket")
        assert result.exit_code == 1
        assert "differs" in _unwrapped(result.output)

    def test_promote_accepts_the_expectations_own_model(self, tmp_path: Path) -> None:
        _backtest(tmp_path, "--costs", str(_costs_file(tmp_path, WEATHER)))
        out = tmp_path / "out"
        result = _promote(
            tmp_path, out, "--expectation", str(tmp_path / "metrics.json"),
            "--fee-model", "polymarket",
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert CostModel.from_dict(json.loads((out / "champion.json").read_text())["costs"]) == (
            WEATHER
        )

    def test_promote_refuses_another_model_than_the_expectations_and_names_both(
        self, tmp_path: Path
    ) -> None:
        _backtest(tmp_path, "--costs", str(_costs_file(tmp_path, WEATHER)))
        result = _promote(
            tmp_path, tmp_path / "out", "--expectation", str(tmp_path / "metrics.json"),
            "--fee-model", "flat-bps",
        )  # fmt: skip
        assert result.exit_code == 1
        out = _unwrapped(result.output)
        assert "--fee-model flat-bps" in out and "polymarket" in out
        assert not (tmp_path / "out" / "champion.json").exists()


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
