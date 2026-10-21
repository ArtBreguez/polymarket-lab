"""The backtest expectation per direction: YES and NO are different strategies.

A NO trade is a buy of the token the market under-prices least, picked where the model
is bearish; its per-stake PnL and forecast error are not draws from the YES
distribution. A live NO trade can only be held against a NO expectation, so the
champion records mean, spread and size (and Brier) for each side, plus the policy the
numbers were produced under.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.expectation import BacktestExpectation, GroupExpectation
from pmlab.core.policy import TradePolicy


def _trades(n_yes: int, n_no: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_yes + n_no
    return pd.DataFrame(
        {
            "direction": ["yes"] * n_yes + ["no"] * n_no,
            "realized_pnl": np.concatenate(
                [rng.normal(0.30, 1.0, n_yes), rng.normal(-0.10, 0.6, n_no)]
            ),
            "predicted_prob": rng.uniform(0.05, 0.95, n),
            "label_occurred": rng.random(n) < 0.5,
        }
    )


class TestPerDirection:
    def test_each_side_gets_its_own_mean_spread_and_size(self) -> None:
        trades = _trades(80, 50)
        exp = BacktestExpectation.from_trades(trades, flat_stake=2.0)
        assert exp.by_direction is not None
        assert set(exp.by_direction) == {"yes", "no"}
        for side in ("yes", "no"):
            rows = trades[trades["direction"] == side]
            per_stake = rows["realized_pnl"].to_numpy() / 2.0
            got = exp.by_direction[side]
            assert got.n_trades == len(rows)
            assert got.mean_pnl_per_stake == pytest.approx(per_stake.mean())
            assert got.std_pnl_per_stake == pytest.approx(per_stake.std(ddof=1))

    def test_each_side_gets_its_own_brier(self) -> None:
        trades = _trades(60, 60, seed=4)
        exp = BacktestExpectation.from_trades(trades)
        assert exp.by_direction is not None
        for side in ("yes", "no"):
            rows = trades[trades["direction"] == side]
            terms = (rows["predicted_prob"] - rows["label_occurred"].astype(float)) ** 2
            got = exp.by_direction[side]
            assert got.brier_score == pytest.approx(terms.mean())
            assert got.brier_std == pytest.approx(terms.std(ddof=1))

    def test_the_forecast_is_scored_against_the_label_not_the_side(self) -> None:
        # A NO trade whose label did NOT occur was a good bet and a good forecast:
        # p = 0.1, label absent -> squared error 0.01, not (1 - 0.1) ** 2.
        trades = pd.DataFrame(
            {
                "direction": ["no", "no"] * 20,
                "realized_pnl": [1.0, -1.0] * 20,
                "predicted_prob": [0.1] * 40,
                "label_occurred": [False] * 40,
            }
        )
        exp = BacktestExpectation.from_trades(trades)
        assert exp.by_direction is not None
        assert exp.by_direction["no"].brier_score == pytest.approx(0.01)

    def test_the_top_level_numbers_pool_both_sides_as_before(self) -> None:
        trades = _trades(80, 50)
        with_direction = BacktestExpectation.from_trades(trades)
        without = BacktestExpectation.from_trades(trades.drop(columns=["direction"]))
        assert with_direction.n_trades == without.n_trades == 130
        assert with_direction.mean_pnl_per_stake == without.mean_pnl_per_stake
        assert with_direction.std_pnl_per_stake == without.std_pnl_per_stake
        assert with_direction.brier_score == without.brier_score

    def test_without_a_direction_column_no_split_is_recorded(self) -> None:
        # Old frames carry no direction. By_direction None means "not recorded": the
        # pooled numbers describe the historical, YES-only backtest.
        exp = BacktestExpectation.from_trades(_trades(40, 40).drop(columns=["direction"]))
        assert exp.by_direction is None

    def test_a_yes_only_backtest_records_only_yes(self) -> None:
        exp = BacktestExpectation.from_trades(_trades(60, 0))
        assert exp.by_direction is not None
        assert set(exp.by_direction) == {"yes"}

    def test_a_side_with_too_few_trades_has_no_expectation(self) -> None:
        # 12 NO trades is not a promise a live NO book can be judged against.
        exp = BacktestExpectation.from_trades(_trades(80, 12))
        assert exp.by_direction is not None
        assert set(exp.by_direction) == {"yes"}

    def test_the_only_side_traded_is_kept_however_few_trades_it_has(self) -> None:
        # With no other side, it IS the pooled expectation, which has always been usable
        # from two trades up. Dropping it would leave a YES-only backtest with no promise.
        exp = BacktestExpectation.from_trades(_trades(12, 0))
        assert exp.by_direction is not None
        assert set(exp.by_direction) == {"yes"}
        assert exp.by_direction["yes"].n_trades == 12
        assert exp.by_direction["yes"].mean_pnl_per_stake == exp.mean_pnl_per_stake

    def test_when_both_sides_are_thin_neither_is_recorded(self) -> None:
        exp = BacktestExpectation.from_trades(_trades(12, 12))
        assert exp.by_direction == {}

    def test_the_minimum_is_configurable(self) -> None:
        exp = BacktestExpectation.from_trades(_trades(80, 12), min_group_trades=10)
        assert exp.by_direction is not None
        assert set(exp.by_direction) == {"yes", "no"}

    def test_a_side_at_exactly_the_minimum_is_kept(self) -> None:
        exp = BacktestExpectation.from_trades(_trades(80, 30))
        assert exp.by_direction is not None
        assert "no" in exp.by_direction

    def test_the_minimum_cannot_be_below_two(self) -> None:
        with pytest.raises(ValueError, match="min_group_trades"):
            BacktestExpectation.from_trades(_trades(10, 10), min_group_trades=1)

    def test_brier_is_absent_for_a_side_when_the_frame_has_no_forecast(self) -> None:
        trades = _trades(50, 50).drop(columns=["predicted_prob", "label_occurred"])
        exp = BacktestExpectation.from_trades(trades)
        assert exp.by_direction is not None
        assert exp.by_direction["no"].brier_score is None

    def test_an_unknown_direction_value_is_refused(self) -> None:
        trades = _trades(40, 40)
        trades.loc[0, "direction"] = "sideways"
        with pytest.raises(ValueError, match="direction"):
            BacktestExpectation.from_trades(trades)

    def test_the_policy_is_recorded(self) -> None:
        policy = TradePolicy.live(min_edge=0.07)
        exp = BacktestExpectation.from_trades(_trades(40, 40), policy=policy)
        assert exp.policy == policy
        assert BacktestExpectation.from_trades(_trades(40, 40)).policy is None


class TestSerialisation:
    def test_round_trips_through_json(self) -> None:
        exp = BacktestExpectation.from_trades(
            _trades(80, 50), flat_stake=2.0, policy=TradePolicy.live(min_edge=0.07)
        )
        restored = BacktestExpectation.from_dict(json.loads(json.dumps(exp.to_dict())))
        assert restored == exp

    def test_an_old_record_has_no_split_and_no_policy(self) -> None:
        old = {
            "n_trades": 120,
            "flat_stake": 1.0,
            "mean_pnl_per_stake": 0.05,
            "std_pnl_per_stake": 0.9,
            "brier_score": 0.21,
            "brier_std": 0.17,
        }
        exp = BacktestExpectation.from_dict(old)
        assert exp.by_direction is None
        assert exp.policy is None

    def test_the_old_keys_keep_their_meaning_and_name(self) -> None:
        data = BacktestExpectation.from_trades(_trades(50, 50)).to_dict()
        for key in (
            "n_trades",
            "flat_stake",
            "mean_pnl_per_stake",
            "std_pnl_per_stake",
            "brier_score",
            "brier_std",
        ):
            assert key in data

    def test_a_null_split_round_trips(self) -> None:
        exp = BacktestExpectation(100, 1.0, 0.1, 0.9, 0.2, 0.1)
        assert exp.to_dict()["by_direction"] is None
        assert BacktestExpectation.from_dict(exp.to_dict()) == exp


class TestAnUnusableSplitIsRefused:
    """Present but unusable must raise: a quiet None would weaken the watchdog."""

    def _base(self) -> dict[str, object]:
        return BacktestExpectation.from_trades(_trades(50, 50)).to_dict()

    def _with_split(self, side: str, **changes: object) -> dict[str, object]:
        data = self._base()
        split = data["by_direction"]
        assert isinstance(split, dict)
        split[side] = {**split[side], **changes}
        return data

    @pytest.mark.parametrize(
        "changes",
        [
            {"mean_pnl_per_stake": math.nan},
            {"std_pnl_per_stake": -0.1},
            {"std_pnl_per_stake": math.inf},
            {"n_trades": 1},
            {"n_trades": 10_000},  # more than the whole backtest
            {"brier_score": 1.5},
            {"brier_std": -1.0},
        ],
    )
    def test_a_bad_number_in_a_side_is_refused(self, changes: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            BacktestExpectation.from_dict(self._with_split("no", **changes))

    def test_an_unknown_side_is_refused(self) -> None:
        data = self._base()
        split = data["by_direction"]
        assert isinstance(split, dict)
        split["sideways"] = split["yes"]
        with pytest.raises(ValueError, match="direction"):
            BacktestExpectation.from_dict(data)

    def test_a_split_that_is_not_an_object_is_refused(self) -> None:
        data = self._base()
        data["by_direction"] = [1, 2]
        with pytest.raises(ValueError, match="by_direction"):
            BacktestExpectation.from_dict(data)

    def test_a_bad_policy_is_refused(self) -> None:
        data = self._base()
        data["policy"] = {"selection": "best_edge", "directions": [], "min_edge": 0.1}
        with pytest.raises(ValueError):
            BacktestExpectation.from_dict(data)

    def test_a_direction_expectation_validates_on_construction(self) -> None:
        with pytest.raises(ValueError, match="mean_pnl_per_stake"):
            GroupExpectation(30, math.nan, 1.0)


class TestTheExpectationRefusesWhatItCannotUse:
    def test_a_policy_that_is_not_a_policy(self) -> None:
        with pytest.raises(ValueError, match="TradePolicy"):
            BacktestExpectation(100, 1.0, 0.1, 0.9, policy="live")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["30", 30.0, True, 1, 0, -5])
    def test_min_group_trades_must_be_an_integer_of_at_least_two(self, bad: object) -> None:
        with pytest.raises(ValueError, match="min_group_trades"):
            BacktestExpectation.from_trades(_trades(40, 40), min_group_trades=bad)  # type: ignore[arg-type]

    def test_a_split_that_is_not_a_mapping(self) -> None:
        with pytest.raises(ValueError, match="by_direction"):
            BacktestExpectation(100, 1.0, 0.1, 0.9, by_direction=[1])  # type: ignore[arg-type]

    def test_a_side_that_is_not_a_group_expectation(self) -> None:
        with pytest.raises(ValueError, match="GroupExpectation"):
            BacktestExpectation(100, 1.0, 0.1, 0.9, by_direction={"yes": {"n_trades": 5}})  # type: ignore[dict-item]
