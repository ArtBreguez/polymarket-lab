"""The watchdog checks a trade's fee against the champion's fee model, whichever model it is (#72).

A flat-bps fee depends on the stake alone; a Polymarket fee depends on the price too, so the check
reads the price the trade was filled at. A champion promoted under one model must not have its live
trades compared with a backtest that paid the other.
"""

from __future__ import annotations

import json

import pytest

from pmlab.core.costs import CostModel
from pmlab.core.fee_schedule import FeeSchedule
from pmlab.monitoring.watchdog import HealthReport, _cost_basis
from tests.monitoring.test_watchdog import _champion, _check, _report, _trades

POLY = CostModel(slippage_bps=50.0)
FLAT = CostModel(fee_model="flat-bps", slippage_bps=50.0)


def _stamped(n: int, costs: CostModel, price: float = 0.40) -> list[dict]:
    fill = round(costs.fill_price(price, stake=1.0), 8)
    fee = round(costs.fee(1.0, fill), 6)
    return _trades(
        n,
        gamma_price=price,
        fill_price=fill,
        fee_paid=fee,
        settled_basis="fill_price",
    )


class TestCostBasisOfAPolymarketTrade:
    def test_a_fee_charged_by_the_curve_at_the_fill_matches(self) -> None:
        fill = round(POLY.fill_price(0.40, stake=1.0), 8)
        trade = {
            "direction": "yes",
            "gamma_price": 0.40,
            "fill_price": fill,
            "fee_paid": round(POLY.fee(1.0, fill), 6),
        }
        assert _cost_basis(trade, POLY, 1.0) == "match"

    def test_the_flat_fee_does_not_match_the_curve(self) -> None:
        fill = round(POLY.fill_price(0.40, stake=1.0), 8)
        trade = {"direction": "yes", "gamma_price": 0.40, "fill_price": fill, "fee_paid": 0.003}
        assert _cost_basis(trade, POLY, 1.0) == "mismatch"

    def test_the_curve_fee_does_not_match_the_flat_model(self) -> None:
        trade = {"direction": "yes", "gamma_price": 0.40, "fill_price": 0.402, "fee_paid": 0.0299}
        assert _cost_basis(trade, FLAT, 1.0) == "mismatch"

    def test_a_no_trade_is_priced_on_the_complement(self) -> None:
        fill = round(POLY.fill_price(0.60, stake=1.0), 8)  # 1 - 0.40
        trade = {
            "direction": "no",
            "gamma_price": 0.40,
            "fill_price": fill,
            "fee_paid": round(POLY.fee(1.0, fill), 6),
        }
        assert _cost_basis(trade, POLY, 1.0) == "match"

    def test_a_fee_one_rounding_unit_from_the_rebuilt_one_still_matches(self) -> None:
        # The trade rounded its fee at the unrounded fill; the check rebuilds it from the fill as
        # recorded (8 decimals), which can land one unit of the schedule (1e-5) away. That is not a
        # different fee model, so it must not leave the trade out of the PnL comparison.
        fill = round(POLY.fill_price(0.40, stake=1.0), 8)
        rebuilt = POLY.fee(1.0, fill)
        unit = POLY.fee_schedule.rounding_unit  # type: ignore[union-attr]
        assert unit == pytest.approx(1e-5)
        base = {"direction": "yes", "gamma_price": 0.40, "fill_price": fill}
        for step in (-1, 1):
            trade = {**base, "fee_paid": round(rebuilt + step * unit, 6)}
            assert _cost_basis(trade, POLY, 1.0) == "match"

    def test_two_units_away_is_a_different_fee(self) -> None:
        fill = round(POLY.fill_price(0.40, stake=1.0), 8)
        rebuilt = POLY.fee(1.0, fill)
        unit = POLY.fee_schedule.rounding_unit  # type: ignore[union-attr]
        base = {"direction": "yes", "gamma_price": 0.40, "fill_price": fill}
        for step in (-2, 2):
            trade = {**base, "fee_paid": round(rebuilt + step * unit, 6)}
            assert _cost_basis(trade, POLY, 1.0) == "mismatch"

    def test_a_fee_with_no_fill_is_checked_at_the_quote(self) -> None:
        costs = CostModel()
        trade = {"direction": "yes", "gamma_price": 0.40, "fee_paid": 0.03}
        assert _cost_basis(trade, costs, 1.0) == "match"
        assert _cost_basis({**trade, "fee_paid": 0.003}, costs, 1.0) == "mismatch"

    @pytest.mark.parametrize(
        "costs",
        [
            CostModel(slippage_bps=50.0),
            CostModel(slippage_fixed=0.01),
            CostModel(depth_bps_per_unit=20.0),
        ],
    )
    def test_a_fee_with_no_fill_is_not_checked_at_a_quote_the_model_says_is_not_the_fill(
        self, costs: CostModel
    ) -> None:
        # with slippage or depth the quote is not the price the fee was charged at, so the fee
        # rebuilt from it would call a correctly priced trade a mismatch
        trade = {"direction": "yes", "gamma_price": 0.40, "fee_paid": 0.0298}
        assert _cost_basis(trade, costs, 1.0) == "unverified"

    def test_a_fee_with_no_price_at_all_cannot_be_checked(self) -> None:
        assert _cost_basis({"fee_paid": 0.03}, CostModel(), 1.0) == "unverified"

    def test_the_stake_scales_the_expected_fee(self) -> None:
        costs = CostModel()
        trade = {"direction": "yes", "gamma_price": 0.40, "fill_price": 0.40, "fee_paid": 0.30}
        assert _cost_basis(trade, costs, 10.0) == "match"

    def test_a_schedule_a_champion_was_promoted_under_is_the_one_checked(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="gamma")
        trade = {"direction": "yes", "gamma_price": 0.40, "fill_price": 0.40, "fee_paid": 0.006}
        assert _cost_basis(trade, costs, 1.0) == "match"
        assert _cost_basis({**trade, "fee_paid": 0.03}, costs, 1.0) == "mismatch"


class TestThePnlCheckWithAPolymarketChampion:
    def test_trades_under_the_champions_fee_model_are_compared(self) -> None:
        champ = _champion(costs=POLY)
        check = _check(_report(trades=_stamped(60, POLY), champion=champ), "pnl")
        assert check.status == "ok"
        assert check.metrics["cost_basis"] == "match"
        assert check.metrics["n_excluded_cost_basis"] == 0

    def test_trades_that_paid_the_flat_fee_are_left_out_of_a_polymarket_comparison(self) -> None:
        champ = _champion(costs=POLY)
        check = _check(
            _report(trades=_stamped(60, POLY) + _stamped(60, FLAT), champion=champ), "pnl"
        )
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_excluded_cost_basis"] == 60

    def test_a_flat_champion_leaves_out_trades_that_paid_the_curve(self) -> None:
        champ = _champion(costs=FLAT)
        check = _check(
            _report(trades=_stamped(60, FLAT) + _stamped(60, POLY), champion=champ), "pnl"
        )
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_excluded_cost_basis"] == 60


class TestTheReportCarriesTheFeeModel:
    def test_the_champions_costs_in_the_report_name_the_model_and_survive_json(self) -> None:
        report = _report(champion=_champion(costs=CostModel()))
        costs = report.champion.costs
        assert costs is not None
        assert costs["fee_model"] == "polymarket"
        assert costs["fee_schedule"]["rate"] == 0.05
        again = HealthReport.from_dict(json.loads(json.dumps(report.to_dict())))
        assert again.champion.costs == costs

    def test_a_report_written_before_1_5_0_still_loads(self) -> None:
        report = _report(champion=_champion(costs=FLAT))
        written = report.to_dict()
        written["champion"]["costs"] = {
            "taker_bps": 30.0,
            "slippage_bps": 50.0,
            "slippage_fixed": 0.0,
            "depth_bps_per_unit": 0.0,
            "depth_reference_stake": 1.0,
        }
        loaded = HealthReport.from_dict(written)
        assert loaded.champion.costs == written["champion"]["costs"]
        assert all(isinstance(v, float) for v in (loaded.champion.costs or {}).values())

    def test_a_champion_without_costs_has_none_in_the_report(self) -> None:
        assert _report(champion=_champion(costs=None)).champion.costs is None

    @pytest.mark.parametrize("costs", [CostModel(), FLAT])
    def test_the_costs_recorded_rebuild_the_champions_model(self, costs: CostModel) -> None:
        report = _report(champion=_champion(costs=costs))
        assert CostModel.from_dict(report.champion.costs or {}) == costs
