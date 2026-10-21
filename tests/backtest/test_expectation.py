"""Tests for BacktestExpectation: what a champion promises, recorded at promotion.

The gate records totals (`aggregate_pnl`, `aggregate_trades`) and nothing about
spread, so a live run has nothing to be surprised *against*: "the live mean is
below the backtest mean" is meaningless without knowing how noisy the mean is.
The expectation keeps the three numbers a significance test needs (mean, standard
deviation, count), on a per-unit-stake basis so a live stake that differs from the
backtest's does not read as a PnL shortfall.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.expectation import BacktestExpectation
from pmlab.backtest.rolling_origin import rolling_origin_eval


def _trades(pnl: list[float], prob: list[float] | None = None, occurred: list[bool] | None = None):
    data: dict[str, list] = {"realized_pnl": pnl}
    if prob is not None:
        data["predicted_prob"] = prob
    if occurred is not None:
        data["label_occurred"] = occurred
    return pd.DataFrame(data)


class TestFromTrades:
    def test_mean_and_std_are_per_unit_stake(self) -> None:
        pnl = [5.0, -10.0, 20.0, -10.0, 15.0]
        exp = BacktestExpectation.from_trades(_trades(pnl), flat_stake=10.0)

        per_stake = np.array(pnl) / 10.0
        assert exp.n_trades == 5
        assert exp.flat_stake == 10.0
        assert exp.mean_pnl_per_stake == pytest.approx(per_stake.mean())
        assert exp.std_pnl_per_stake == pytest.approx(per_stake.std(ddof=1))

    def test_stake_does_not_change_the_per_stake_numbers(self) -> None:
        """The same strategy at 10x the stake is the same expectation."""
        small = BacktestExpectation.from_trades(_trades([0.5, -1.0, 2.0]), flat_stake=1.0)
        large = BacktestExpectation.from_trades(_trades([5.0, -10.0, 20.0]), flat_stake=10.0)
        assert large.mean_pnl_per_stake == pytest.approx(small.mean_pnl_per_stake)
        assert large.std_pnl_per_stake == pytest.approx(small.std_pnl_per_stake)

    def test_brier_is_scored_on_the_forecast_against_the_label(self) -> None:
        prob = [0.9, 0.2, 0.7, 0.4]
        occurred = [True, False, False, True]
        exp = BacktestExpectation.from_trades(_trades([1.0, -1.0, -1.0, 1.0], prob, occurred))

        terms = np.array([(0.9 - 1) ** 2, (0.2 - 0) ** 2, (0.7 - 0) ** 2, (0.4 - 1) ** 2])
        assert exp.brier_score == pytest.approx(terms.mean())
        assert exp.brier_std == pytest.approx(terms.std(ddof=1))

    def test_brier_is_none_without_the_forecast_columns(self) -> None:
        exp = BacktestExpectation.from_trades(_trades([1.0, -1.0, 2.0]))
        assert exp.brier_score is None
        assert exp.brier_std is None

    def test_brier_is_none_when_only_one_forecast_column_is_present(self) -> None:
        exp = BacktestExpectation.from_trades(_trades([1.0, -1.0], prob=[0.5, 0.5]))
        assert exp.brier_score is None

    def test_probabilities_outside_unit_interval_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="probabilit"):
            BacktestExpectation.from_trades(_trades([1.0, -1.0], [1.4, 0.2], [True, False]))

    @pytest.mark.parametrize("pnl", [[], [1.0]])
    def test_fewer_than_two_trades_is_rejected(self, pnl: list[float]) -> None:
        """One trade has no spread; the expectation would be a fake certainty."""
        with pytest.raises(ValueError, match="at least 2"):
            BacktestExpectation.from_trades(_trades(pnl))

    def test_non_finite_pnl_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            BacktestExpectation.from_trades(_trades([1.0, math.nan, 2.0]))

    @pytest.mark.parametrize("stake", [0.0, -1.0, math.nan, math.inf])
    def test_bad_stake_is_rejected(self, stake: float) -> None:
        with pytest.raises(ValueError, match="flat_stake"):
            BacktestExpectation.from_trades(_trades([1.0, 2.0]), flat_stake=stake)

    def test_missing_pnl_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="realized_pnl"):
            BacktestExpectation.from_trades(pd.DataFrame({"x": [1, 2]}))


class TestSerialisation:
    def test_round_trip_is_exact(self) -> None:
        exp = BacktestExpectation.from_trades(
            _trades([1.0, -1.0, 2.0, 0.5], [0.8, 0.3, 0.6, 0.7], [True, False, True, True]),
            flat_stake=2.0,
        )
        assert BacktestExpectation.from_dict(exp.to_dict()) == exp

    def test_to_dict_is_strict_json(self) -> None:
        exp = BacktestExpectation.from_trades(_trades([1.0, -1.0, 2.0]))
        json.dumps(exp.to_dict(), allow_nan=False)

    def test_from_dict_tolerates_a_missing_brier(self) -> None:
        exp = BacktestExpectation.from_dict(
            {"n_trades": 10, "flat_stake": 1.0, "mean_pnl_per_stake": 0.1, "std_pnl_per_stake": 0.9}
        )
        assert exp.brier_score is None and exp.brier_std is None

    def test_from_dict_rejects_a_malformed_record(self) -> None:
        with pytest.raises((KeyError, ValueError)):
            BacktestExpectation.from_dict({"n_trades": 10})


def _panel(n_dates: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_dates):
        for m in range(3):
            winner = rng.choice(["YES", "NO"])
            for label in ("YES", "NO"):
                rows.append(
                    {
                        "market_id": f"m{d}-{m}",
                        "decision_date": f"2026-01-{d + 1:02d}"
                        if d < 31
                        else f"2026-02-{d - 30:02d}",
                        "outcome_label": label,
                        "winning_label": winner,
                        "market_price": 0.45,
                        "feature_x": float(rng.normal()),
                        "segment": "politics",
                    }
                )
    return pd.DataFrame(rows)


class _Const:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


class TestFromTheBacktest:
    def test_rolling_origin_records_whether_the_label_occurred(self) -> None:
        """Brier needs the label, and `outcome` is a profit flag, not the label."""
        result = rolling_origin_eval(_panel(), _Const(), min_train_rows=10, stride=10)
        assert "label_occurred" in result.trades.columns
        assert result.trades["label_occurred"].dtype == bool

    def test_expectation_from_a_real_backtest_carries_a_brier(self) -> None:
        result = rolling_origin_eval(_panel(), _Const(), min_train_rows=10, stride=2)
        exp = BacktestExpectation.from_trades(result.trades)
        assert exp.n_trades == len(result.trades) >= 2
        assert exp.brier_score is not None
        # Constant 0.7 forecast against a ~50% base rate: bounded and sensible.
        assert 0.0 < exp.brier_score < 1.0


class _Always:
    def __init__(self, p: float) -> None:
        self._p = p

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 1 - self._p), np.full(n, self._p)])


def test_label_occurred_is_the_label_and_not_whether_the_trade_made_money() -> None:
    """A win at 0.999 pays 0.001 a share and costs 0.003 in fees: the label occurred,
    the trade lost money. `outcome` says "lost"; calibration must still see a hit."""
    rows = []
    for d in range(30):
        for label in ("YES", "NO"):
            rows.append(
                {
                    "market_id": f"m{d}",
                    "decision_date": f"2026-01-{d + 1:02d}",
                    "outcome_label": label,
                    "winning_label": "YES",
                    "market_price": 0.999,
                    "feature_x": 1.0 if label == "YES" else 0.0,
                    "segment": "A",
                }
            )
    result = rolling_origin_eval(pd.DataFrame(rows), _Always(0.9), min_train_rows=4, stride=2)

    trades = result.trades
    assert len(trades) > 0
    assert trades["label_occurred"].all()
    assert (trades["realized_pnl"] < 0).all()
    assert (trades["outcome"] == "lost").all()
