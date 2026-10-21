"""`rolling_origin_eval(policy=...)`: the backtest takes the decision live trading takes.

Live trading buys, per market, the (bin, side) with the best after-fee edge, YES or NO,
if that edge clears `--min-edge`. The historical backtest bought YES on the most
probable bin of every market. These tests pin both, and that the second is what runs
when no policy is named (published numbers must reproduce).

The model here is a stub that reads its probability straight from `feature_p`, so each
test says exactly what the model believed and the arithmetic can be done by hand.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.policy import TradePolicy

_FRICTIONLESS = CostModel(taker_bps=0.0)


class _FeatureProb:
    """A forecaster whose predicted probability is the `feature_p` column."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["feature_p"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


def _row(
    market: str, date: str, label: str, winner: str, price: float, p: float, **extra: Any
) -> dict[str, Any]:
    return {
        "market_id": market,
        "decision_date": date,
        "outcome_label": label,
        "winning_label": winner,
        "market_price": price,
        "feature_p": p,
        **extra,
    }


def _panel(rows: list[dict[str, Any]], *, warmup: int = 3) -> pd.DataFrame:
    """`warmup` filler rows on earlier dates so the first real date has training data."""
    filler = [_row("warmup", f"2023-12-{d + 1:02d}", "x", "x", 0.5, 0.5) for d in range(warmup)]
    return pd.DataFrame(filler + rows)


def _run(rows: list[dict[str, Any]], policy: TradePolicy | None, **kwargs: Any) -> pd.DataFrame:
    kwargs.setdefault("costs", _FRICTIONLESS)
    result = rolling_origin_eval(
        _panel(rows), _FeatureProb(), min_train_rows=3, stride=1, policy=policy, **kwargs
    )
    trades = result.trades
    return trades[trades["market_id"] != "warmup"].reset_index(drop=True)


class TestTheBacktestBuysNo:
    """A market the model thinks is overpriced: the live rule buys the NO token."""

    # YES on "hot" is quoted 0.60 and the model gives it 0.10. NO costs 0.40 and is
    # worth 0.90: an edge of 0.50, against a YES edge of -0.50.
    ROWS = [
        _row("m1", "2024-01-01", "hot", "cold", 0.60, 0.10),
        _row("m1", "2024-01-01", "cold", "cold", 0.40, 0.90),
    ]

    def test_the_live_policy_buys_no_on_the_overpriced_label(self) -> None:
        trades = _run(self.ROWS, TradePolicy.live(min_edge=0.0))
        assert len(trades) == 1
        trade = trades.iloc[0]
        assert trade["direction"] == "no"
        assert trade["outcome_label"] == "hot"

    def test_a_won_no_trade_pays_one_over_the_no_price_minus_one(self) -> None:
        # "hot" does not win, so the NO token pays 1: 1 / 0.40 - 1 = +1.5 per unit stake.
        trade = _run(self.ROWS, TradePolicy.live(min_edge=0.0)).iloc[0]
        assert trade["fill_price"] == pytest.approx(0.40)
        assert trade["market_price"] == pytest.approx(0.60)  # still the YES quote
        assert trade["realized_pnl"] == pytest.approx(1.5)
        assert bool(trade["label_occurred"]) is False

    def test_a_lost_no_trade_loses_the_whole_stake(self) -> None:
        rows = [{**r, "winning_label": "hot"} for r in self.ROWS]
        trade = _run(rows, TradePolicy.live(min_edge=0.0)).iloc[0]
        assert trade["realized_pnl"] == pytest.approx(-1.0)
        assert bool(trade["label_occurred"]) is True
        # `outcome` of a backtest trade is the profit flag, as it always was.
        assert trade["outcome"] == "lost"

    def test_the_forecast_is_still_the_probability_of_the_label(self) -> None:
        # Calibration scores predicted_prob against label_occurred whichever side was
        # bought, so a NO trade must not flip the forecast to 1 - p.
        trade = _run(self.ROWS, TradePolicy.live(min_edge=0.0)).iloc[0]
        assert trade["predicted_prob"] == pytest.approx(0.10)

    def test_the_edges_are_those_of_a_live_signal(self) -> None:
        trade = _run(self.ROWS, TradePolicy.live(min_edge=0.0)).iloc[0]
        assert trade["no_edge"] == pytest.approx(0.50)
        assert trade["yes_edge"] == pytest.approx(0.10 - 0.60)
        assert trade["best_edge"] == pytest.approx(0.50)
        assert trade["edge"] == pytest.approx(0.50)  # token probability minus fill

    def test_the_fee_is_charged_and_leaves_the_edge_after_fee(self) -> None:
        trade = _run(self.ROWS, TradePolicy.live(min_edge=0.0), costs=CostModel()).iloc[0]
        assert trade["best_edge"] == pytest.approx(0.50 - 0.003)
        assert trade["edge"] == pytest.approx(0.50)  # `edge` has never included the fee
        assert trade["realized_pnl"] == pytest.approx(1.5 - 0.003)

    def test_the_historical_policy_buys_yes_on_the_most_probable_bin(self) -> None:
        trades = _run(self.ROWS, None)
        assert len(trades) == 1
        assert trades.iloc[0]["outcome_label"] == "cold"  # p = 0.90
        assert trades.iloc[0]["direction"] == "yes"

    def test_a_yes_only_edge_policy_does_not_buy_no(self) -> None:
        trades = _run(self.ROWS, TradePolicy.live(min_edge=0.0, directions=("yes",)))
        assert (trades["direction"] == "yes").all()
        # The best YES edge is "cold": 0.90 - 0.40 = 0.50.
        assert trades.iloc[0]["outcome_label"] == "cold"


class TestTheEdgeThreshold:
    ROWS = [
        _row("m1", "2024-01-01", "a", "a", 0.50, 0.60),  # yes edge 0.10
        _row("m1", "2024-01-01", "b", "a", 0.50, 0.40),  # no edge 0.10
    ]

    def test_an_edge_exactly_at_the_threshold_is_traded(self) -> None:
        assert len(_run(self.ROWS, TradePolicy.live(min_edge=0.10))) == 1

    def test_an_edge_below_the_threshold_is_not(self) -> None:
        assert _run(self.ROWS, TradePolicy.live(min_edge=0.11)).empty

    def test_every_market_is_traded_at_a_threshold_of_minus_one(self) -> None:
        rows = [_row("m1", "2024-01-01", "a", "a", 0.5, 0.5)]  # no edge at all
        trades = _run(rows, TradePolicy.live(min_edge=-1.0))
        assert len(trades) == 1

    def test_the_default_live_threshold_is_the_one_record_trades_uses(self) -> None:
        # 0.10 edge < 0.20: nothing is traded under the default live policy.
        assert _run(self.ROWS, TradePolicy.live()).empty


class TestTheBestBinAndSide:
    def test_the_bin_with_the_best_edge_wins_not_the_most_probable(self) -> None:
        rows = [
            _row("m1", "2024-01-01", "likely", "likely", 0.80, 0.85),  # yes edge 0.05
            _row("m1", "2024-01-01", "cheap", "likely", 0.10, 0.40),  # yes edge 0.30
            _row("m1", "2024-01-01", "dear", "likely", 0.30, 0.05),  # no edge 0.65 - 0.70 < 0
        ]
        trade = _run(rows, TradePolicy.live(min_edge=0.0)).iloc[0]
        assert (trade["outcome_label"], trade["direction"]) == ("cheap", "yes")

    def test_one_trade_per_market_and_date(self) -> None:
        rows = [
            _row("m1", "2024-01-01", "a", "a", 0.5, 0.9),
            _row("m1", "2024-01-01", "b", "a", 0.5, 0.1),
            _row("m2", "2024-01-01", "a", "a", 0.5, 0.9),
            _row("m1", "2024-01-02", "a", "a", 0.5, 0.9),
        ]
        trades = _run(rows, TradePolicy.live(min_edge=0.0))
        assert sorted(zip(trades["market_id"], trades["eval_date"], strict=True)) == [
            ("m1", "2024-01-01"),
            ("m1", "2024-01-02"),
            ("m2", "2024-01-01"),
        ]

    def test_a_tie_between_bins_goes_to_the_first(self) -> None:
        rows = [
            _row("m1", "2024-01-01", "a", "a", 0.5, 0.9),
            _row("m1", "2024-01-01", "b", "a", 0.5, 0.9),
        ]
        assert _run(rows, TradePolicy.live(min_edge=0.0)).iloc[0]["outcome_label"] == "a"

    def test_a_market_priced_at_zero_or_one_is_skipped_not_a_free_lunch(self) -> None:
        rows = [
            _row("m1", "2024-01-01", "a", "a", 0.0, 0.9),
            _row("m2", "2024-01-01", "a", "a", 1.0, 0.1),
        ]
        # m1: YES at 0 is stake / 0 shares and its NO costs 1, which can only lose the fee.
        # m2 is the mirror image. Neither side of either market is a trade, whatever
        # threshold is asked for.
        assert _run(rows, TradePolicy.live(min_edge=-1.0)).empty

    def test_a_nan_forecast_is_refused_not_traded(self) -> None:
        rows = [_row("m1", "2024-01-01", "a", "a", 0.5, math.nan)]
        with pytest.raises(ValueError, match="model_prob"):
            _run(rows, TradePolicy.live(min_edge=0.0))


class TestCostsOnTheNoSide:
    def test_slippage_is_charged_on_the_no_token(self) -> None:
        costs = CostModel(taker_bps=0.0, slippage_bps=1000.0)  # 10% of the price
        rows = [
            _row("m1", "2024-01-01", "hot", "cold", 0.60, 0.10),
            _row("m1", "2024-01-01", "cold", "cold", 0.40, 0.90),
        ]
        # The NO tokens are quoted 0.40 ("hot") and 0.60 ("cold") and fill at 0.44 and
        # 0.66; their edges are 0.90 - 0.44 = 0.46 and 0.10 - 0.66 = -0.56.
        trades = _run(rows, TradePolicy.live(min_edge=-1.0, directions=("no",)), costs=costs)
        trade = trades.iloc[0]
        assert (trade["direction"], trade["outcome_label"]) == ("no", "hot")
        assert trade["fill_price"] == pytest.approx(0.44)
        assert trade["best_edge"] == pytest.approx(0.46)
        assert trade["realized_pnl"] == pytest.approx(1.0 / 0.44 - 1.0)

    def test_the_stake_scales_pnl_and_the_flat_stake_reaches_the_cost_model(self) -> None:
        rows = [
            _row("m1", "2024-01-01", "hot", "cold", 0.60, 0.10),
            _row("m1", "2024-01-01", "cold", "cold", 0.40, 0.90),
        ]
        one = _run(rows, TradePolicy.live(min_edge=0.0, directions=("no",)), flat_stake=1.0)
        four = _run(rows, TradePolicy.live(min_edge=0.0, directions=("no",)), flat_stake=4.0)
        assert four.iloc[0]["realized_pnl"] == pytest.approx(4 * one.iloc[0]["realized_pnl"])


class TestTheDefaultStaysTheHistoricalBacktest:
    def _random_panel(self) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        rows = []
        for day in range(12):
            for market in range(3):
                labels = ["a", "b", "c"]
                winner = labels[int(rng.integers(0, 3))]
                prices = rng.dirichlet(np.ones(3))
                for label, price in zip(labels, prices, strict=True):
                    rows.append(
                        _row(
                            f"m{market}",
                            f"2024-02-{day + 1:02d}",
                            label,
                            winner,
                            float(price),
                            float(rng.random()),
                        )
                    )
        return _panel(rows)

    def test_no_policy_equals_the_explicit_historical_policy(self) -> None:
        panel = self._random_panel()
        implicit = rolling_origin_eval(panel, _FeatureProb(), min_train_rows=3, stride=1)
        explicit = rolling_origin_eval(
            panel, _FeatureProb(), min_train_rows=3, stride=1, policy=TradePolicy.top_bin_yes()
        )
        pd.testing.assert_frame_equal(implicit.trades, explicit.trades)
        assert (implicit.trades["direction"] == "yes").all()

    def test_the_historical_trades_are_the_ones_the_old_code_made(self) -> None:
        # Re-derive them independently of rolling_origin_eval: for each (market, date)
        # take the row with the highest forecast and settle a buy of its YES token.
        panel = self._random_panel()
        trades = rolling_origin_eval(panel, _FeatureProb(), min_train_rows=3, stride=1).trades
        assert len(trades) > 0
        for trade in trades.itertuples():
            block = panel[
                (panel["market_id"] == trade.market_id)
                & (panel["decision_date"] == trade.eval_date)
            ]
            top = block.loc[block["feature_p"].idxmax()]
            assert trade.outcome_label == top["outcome_label"]
            fee = 0.003
            price = float(top["market_price"])
            won = top["outcome_label"] == top["winning_label"]
            expected = (1.0 if won else 0.0) / price - 1.0 - fee
            assert trade.realized_pnl == pytest.approx(expected, abs=1e-12)

    def test_the_empty_result_keeps_every_column(self) -> None:
        panel = _panel([_row("m1", "2024-01-01", "a", "a", 0.5, 0.5)])
        result = rolling_origin_eval(
            panel, _FeatureProb(), min_train_rows=100, stride=1, policy=TradePolicy.live()
        )
        assert result.trades.empty
        for column in ("direction", "yes_edge", "no_edge", "best_edge", "fill_price", "segment"):
            assert column in result.trades.columns


class TestNoLookahead:
    """The decision at date t uses the price and forecast of date t and nothing later."""

    def _rows(self, price_later: float, winner_later: str) -> list[dict[str, Any]]:
        return [
            _row("m1", "2024-01-01", "hot", "cold", 0.60, 0.10),
            _row("m1", "2024-01-01", "cold", "cold", 0.40, 0.90),
            # Later dates of the same market: a different price and a different truth.
            _row("m1", "2024-01-02", "hot", winner_later, price_later, 0.10),
            _row("m1", "2024-01-02", "cold", winner_later, 1 - price_later, 0.95),
            _row("m1", "2024-01-03", "hot", winner_later, price_later, 0.10),
            _row("m1", "2024-01-03", "cold", winner_later, 1 - price_later, 0.95),
        ]

    def test_changing_the_future_does_not_change_an_earlier_decision(self) -> None:
        a = _run(self._rows(0.60, "cold"), TradePolicy.live(min_edge=0.0))
        b = _run(self._rows(0.05, "hot"), TradePolicy.live(min_edge=0.0))
        first_a = a[a["eval_date"] == "2024-01-01"]
        first_b = b[b["eval_date"] == "2024-01-01"]
        assert len(first_a) == 1
        pd.testing.assert_frame_equal(first_a, first_b)

    def test_the_price_used_is_the_one_on_the_decision_date(self) -> None:
        trades = _run(self._rows(0.20, "cold"), TradePolicy.live(min_edge=0.0))
        by_date = trades.set_index("eval_date")
        assert by_date.loc["2024-01-01", "market_price"] == pytest.approx(0.60)
        # On the 2nd "hot" is quoted 0.20 and "cold" 0.80. The NO on "hot" now has an edge
        # of only 0.90 - 0.80 = 0.10, and YES on "cold" has 0.95 - 0.80 = 0.15.
        assert by_date.loc["2024-01-02", "outcome_label"] == "cold"
        assert by_date.loc["2024-01-02", "direction"] == "yes"
        assert by_date.loc["2024-01-02", "market_price"] == pytest.approx(0.80)

    def test_the_training_rows_stay_strictly_before_the_decision_date(self) -> None:
        panel = _panel(self._rows(0.30, "cold"))
        result = rolling_origin_eval(
            panel,
            _FeatureProb(),
            min_train_rows=3,
            stride=1,
            policy=TradePolicy.live(min_edge=0.0),
            costs=_FRICTIONLESS,
        )
        for step in result.steps:
            assert step["train_rows"] == int((panel["decision_date"] < step["eval_date"]).sum())
