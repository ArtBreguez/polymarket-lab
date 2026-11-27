"""`rolling_origin_eval` charges the Polymarket fee per fill and records what it charged (#72).

Every trade of a backtest carries ``fee_paid``, ``fee_rate``, ``liquidity`` and ``fee_source``, so a
number in a report can be traced to the schedule that produced it. The panel here has one price, so
every fee is a known figure: a unit stake bought at 0.50 pays ``0.05 x 0.50 = 0.025``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab import PmlabDeprecationWarning
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.fee_schedule import FeeSchedule
from pmlab.core.policy import TradePolicy
from pmlab.markets.fees import FeeScheduleStore, FeeSnapshot

FLAT = CostModel(fee_model="flat-bps")
FEE_COLUMNS = ["fee_paid", "fee_rate", "liquidity", "fee_source", "fee_model"]


class _Confident:
    """A stub forecaster: the same probability for every row, no fitting."""

    def __init__(self, prob: float = 0.9) -> None:
        self.prob = prob

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


def _panel(price: float = 0.5, n_dates: int = 12) -> pd.DataFrame:
    rows = []
    for d in range(n_dates):
        for market in range(3):
            for label in ("YES", "NO"):
                rows.append(
                    {
                        "market_id": f"m{market}",
                        "decision_date": f"2026-01-{d + 1:02d}",
                        "outcome_label": label,
                        "winning_label": "YES",
                        "market_price": price,
                        "feature_x": 1.0 if label == "YES" else 0.0,
                        "segment": "all",
                    }
                )
    return pd.DataFrame(rows)


def _run(price: float = 0.5, **kwargs: object) -> pd.DataFrame:
    return rolling_origin_eval(
        _panel(price),
        _Confident(),
        min_train_rows=8,
        stride=2,
        **kwargs,  # type: ignore[arg-type]
    ).trades


class TestTheDefaultChargesThePolymarketFee:
    def test_a_unit_stake_at_half_pays_two_and_a_half_percent(self) -> None:
        trades = _run()
        assert not trades.empty
        assert trades["fee_paid"].tolist() == pytest.approx([0.025] * len(trades), abs=1e-12)

    def test_pnl_is_the_gross_win_less_that_fee(self) -> None:
        # 2 shares at 0.50 win 1.00 each on a unit stake: +1.00 gross
        trades = _run()
        assert trades["realized_pnl"].tolist() == pytest.approx([1.0 - 0.025] * len(trades))

    def test_the_fee_follows_the_price(self) -> None:
        trades = _run(price=0.20)
        # 5 shares at 0.20: fee 0.05 x 0.80 = 0.04; gross win 5 x 0.80 = 4.00
        assert trades["fee_paid"].tolist() == pytest.approx([0.04] * len(trades), abs=1e-12)
        assert trades["realized_pnl"].tolist() == pytest.approx([4.0 - 0.04] * len(trades))

    def test_a_dearer_fee_than_the_flat_model_costs_pnl(self) -> None:
        assert _run()["realized_pnl"].sum() < _run(costs=FLAT)["realized_pnl"].sum()

    def test_the_trade_carries_the_fee_fields(self) -> None:
        trades = _run()
        assert set(FEE_COLUMNS) <= set(trades.columns)
        assert set(trades["fee_rate"]) == {0.05}
        assert set(trades["liquidity"]) == {"taker"}
        assert set(trades["fee_source"]) == {"category_default"}
        assert set(trades["fee_model"]) == {"polymarket"}

    def test_a_run_with_no_trades_still_has_the_columns(self) -> None:
        trades = rolling_origin_eval(_panel(), _Confident(), min_train_rows=10_000, stride=2).trades
        assert trades.empty
        assert set(FEE_COLUMNS) <= set(trades.columns)

    def test_a_token_at_zero_buys_nothing_and_pays_nothing(self) -> None:
        trades = _run(price=0.0)
        assert not trades.empty
        assert set(trades["fee_paid"]) == {0.0}


class TestAnExplicitModel:
    def test_the_flat_model_charges_what_it_always_charged(self) -> None:
        trades = _run(costs=FLAT)
        assert trades["fee_paid"].tolist() == pytest.approx([0.003] * len(trades))
        assert trades["realized_pnl"].tolist() == pytest.approx([1.0 - 0.003] * len(trades))
        assert set(trades["fee_source"]) == {"flat_bps"}
        assert set(trades["fee_model"]) == {"flat-bps"}
        assert trades["fee_rate"].tolist() == pytest.approx([0.003] * len(trades))

    def test_a_schedule_the_caller_gives_is_charged_and_named(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="gamma")
        trades = _run(costs=costs)
        assert trades["fee_paid"].tolist() == pytest.approx([0.005] * len(trades), abs=1e-12)
        assert set(trades["fee_source"]) == {"gamma"}
        assert set(trades["fee_rate"]) == {0.01}

    def test_a_fee_free_schedule_pays_nothing_and_reports_a_zero_rate(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.05, enabled=False))
        trades = _run(costs=costs)
        assert set(trades["fee_paid"]) == {0.0}
        assert set(trades["fee_rate"]) == {0.0}


class TestTheLivePolicyDecidesOnTheAfterFeeEdge:
    def test_the_logged_edge_is_net_of_the_curve_fee(self) -> None:
        trades = _run(policy=TradePolicy.live(min_edge=0.05))
        assert not trades.empty
        # p = 0.90, fill 0.50, fee 0.025 of the stake
        assert trades["best_edge"].tolist() == pytest.approx([0.9 - 0.5 - 0.025] * len(trades))

    def test_a_threshold_between_the_two_models_splits_them(self) -> None:
        # edge is 0.397 under the flat fee and 0.375 under the curve: 0.38 clears one only
        flat = _run(costs=FLAT, policy=TradePolicy.live(min_edge=0.38))
        curve = _run(policy=TradePolicy.live(min_edge=0.38))
        assert not flat.empty
        assert curve.empty


class TestTakerBpsIsDeprecated:
    def test_taker_bps_still_works_as_the_flat_model_and_warns(self) -> None:
        with pytest.warns(PmlabDeprecationWarning, match="taker_bps"):
            trades = _run(taker_bps=100.0)
        assert trades["fee_paid"].tolist() == pytest.approx([0.01] * len(trades))
        assert set(trades["fee_source"]) == {"flat_bps"}

    def test_the_warning_names_the_caller_not_pmlab(self) -> None:
        with pytest.warns(PmlabDeprecationWarning) as caught:
            _run(taker_bps=10.0)
        assert caught[0].filename.endswith("test_rolling_origin_fees.py")

    def test_taker_bps_with_costs_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="taker_bps"):
            _run(taker_bps=10.0, costs=CostModel())


class TestTheFeeIsChargedAtTheFillNotTheQuote:
    """Slippage and depth move the fill; the fee follows it (review r1, I2).

    Every other test here has fill == quote, where the fee at the quote and at the fill are the
    same number, so none of them can tell the two apart. These do.
    """

    @pytest.mark.parametrize(
        ("prob", "side"),
        [(0.9, "yes"), (0.1, "no")],
        ids=["a-yes-trade", "a-no-trade"],
    )
    @pytest.mark.parametrize(
        ("costs", "stake"),
        [
            (CostModel(slippage_bps=100.0), 1.0),
            (CostModel(slippage_bps=50.0, depth_bps_per_unit=100.0), 3.0),
            (CostModel(slippage_fixed=0.01), 1.0),
        ],
        ids=["slippage", "slippage-and-depth", "fixed-slippage"],
    )
    def test_fee_paid_is_the_fee_of_the_fill_price(
        self, prob: float, side: str, costs: CostModel, stake: float
    ) -> None:
        trades = rolling_origin_eval(
            _panel(),
            _Confident(prob),
            min_train_rows=8,
            stride=2,
            flat_stake=stake,
            costs=costs,
            policy=TradePolicy.live(min_edge=0.05),
        ).trades
        assert not trades.empty
        assert set(trades["direction"]) == {side}

        at_the_fill = [costs.fee(stake, p) for p in trades["fill_price"]]
        quote = trades["market_price"].where(
            trades["direction"] == "yes", 1.0 - trades["market_price"]
        )
        at_the_quote = [costs.fee(stake, p) for p in quote]
        assert trades["fee_paid"].tolist() == pytest.approx(at_the_fill, abs=1e-12)
        # not vacuous: the fill is dearer than the quote, so the two fees are not the same number
        assert (trades["fill_price"] > quote).all()
        assert all(f != q for f, q in zip(at_the_fill, at_the_quote, strict=True))
        assert trades["fee_paid"].tolist() != pytest.approx(at_the_quote, abs=1e-9)

    def test_the_pnl_nets_the_fee_of_the_fill(self) -> None:
        costs = CostModel(slippage_bps=100.0)
        trades = rolling_origin_eval(
            _panel(), _Confident(), min_train_rows=8, stride=2, costs=costs
        ).trades
        # a YES filled at 0.505 wins 1 / 0.505 shares x 1 and pays the fee of that fill
        gross = 1.0 / 0.505 - 1.0
        fee = costs.fee(1.0, 0.505)
        assert fee == pytest.approx(0.05 * 0.495, abs=1e-5)
        assert trades["realized_pnl"].tolist() == pytest.approx([gross - fee] * len(trades))


SEEN_AFTER_JAN_5 = datetime(2026, 1, 8, tzinfo=UTC)


def _snapshot(seen_at: datetime, rate: float = 0.01) -> FeeSnapshot:
    return FeeSnapshot(
        market_id="m0",
        seen_at=seen_at,
        schedule=FeeSchedule(rate=rate),
        source="gamma",
    )


class TestAFeeNotKnownYetPricesNoTrade:
    """A schedule seen on day S prices no fold that decided at or before S (review r1, I1).

    #89 keeps each schedule with the moment it was seen, because Polymarket changes its rates
    without notice. The folds of this panel decide on 2026-01-05, 07, 09 and 11 (at the start of the
    day, the panel has no `decision_ts`), the first one that has rows to train on.
    """

    def test_a_schedule_seen_after_the_first_fold_refuses_the_run(self) -> None:
        costs = CostModel().with_fee_snapshot(_snapshot(SEEN_AFTER_JAN_5))
        with pytest.raises(ValueError, match="2026-01-05") as caught:
            _run(costs=costs)
        message = str(caught.value)
        assert "2026-01-08" in message  # when it was seen
        assert "FeeScheduleStore" in message and "as_of" in message  # what to do instead

    def test_it_refuses_before_a_model_is_fitted(self) -> None:
        class _Counting(_Confident):
            fits = 0

            def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
                type(self).fits += 1

        costs = CostModel().with_fee_snapshot(_snapshot(SEEN_AFTER_JAN_5))
        with pytest.raises(ValueError, match="fee"):
            rolling_origin_eval(_panel(), _Counting(), min_train_rows=8, stride=2, costs=costs)
        assert _Counting.fits == 0

    def test_a_schedule_seen_exactly_when_the_fold_decides_is_not_yet_known(self) -> None:
        # the store's rule: the latest snapshot seen strictly BEFORE the cutoff
        costs = CostModel().with_fee_snapshot(_snapshot(datetime(2026, 1, 5, tzinfo=UTC)))
        with pytest.raises(ValueError, match="2026-01-05"):
            _run(costs=costs)

    def test_a_schedule_seen_a_second_before_the_first_fold_prices_every_trade(self) -> None:
        seen = datetime(2026, 1, 4, 23, 59, 59, tzinfo=UTC)
        trades = _run(costs=CostModel().with_fee_snapshot(_snapshot(seen)))
        assert not trades.empty
        assert set(trades["fee_source"]) == {"gamma"}
        assert set(trades["fee_rate"]) == {0.01}

    def test_the_decision_time_of_the_panel_is_what_counts(self) -> None:
        panel = _panel()
        panel["decision_ts"] = panel["decision_date"] + "T12:00:00+00:00"
        run = lambda seen: rolling_origin_eval(  # noqa: E731
            panel,
            _Confident(),
            min_train_rows=8,
            stride=2,
            costs=CostModel().with_fee_snapshot(_snapshot(seen)),
        )
        # seen at 06:00 on the day of the first fold, which decides at 12:00: known by then
        assert not run(datetime(2026, 1, 5, 6, tzinfo=UTC)).trades.empty
        with pytest.raises(ValueError, match="2026-01-05"):
            run(datetime(2026, 1, 5, 13, tzinfo=UTC))

    def test_a_schedule_given_without_a_moment_claims_none_and_is_not_checked(self) -> None:
        # the explicit approximation: today's schedule over the whole history, named as the user's
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="user")
        trades = _run(costs=costs)
        assert set(trades["fee_source"]) == {"user"}

    def test_a_panel_with_no_fold_to_price_never_trips_on_a_late_schedule(self) -> None:
        costs = CostModel().with_fee_snapshot(_snapshot(SEEN_AFTER_JAN_5))
        trades = rolling_origin_eval(
            _panel(), _Confident(), min_train_rows=10_000, stride=2, costs=costs
        ).trades
        assert trades.empty

    def test_the_store_hands_each_date_the_schedule_known_before_it(self, tmp_path: Path) -> None:
        # the flow docs/costs.md describes: FeeScheduleStore.as_of, then with_fee_snapshot
        store = FeeScheduleStore(tmp_path)
        store.record(
            [
                _snapshot(datetime(2026, 1, 2, tzinfo=UTC), rate=0.05),
                _snapshot(SEEN_AFTER_JAN_5, rate=0.01),  # a change of rate seen on January 8
            ]
        )
        known_on_jan_5 = store.as_of("2026-01-05")["m0"]
        assert known_on_jan_5.schedule.rate == 0.05
        trades = _run(costs=CostModel().with_fee_snapshot(known_on_jan_5))
        assert set(trades["fee_rate"]) == {0.05}

        # the schedule seen on January 8 is not one the folds of January 5 and 7 could have used
        known_on_jan_9 = store.as_of("2026-01-09")["m0"]
        assert known_on_jan_9.schedule.rate == 0.01
        with pytest.raises(ValueError, match="2026-01-05"):
            _run(costs=CostModel().with_fee_snapshot(known_on_jan_9))
