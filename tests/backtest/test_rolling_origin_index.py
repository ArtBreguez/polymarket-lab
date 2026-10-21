"""`rolling_origin_eval` must not depend on the panel's index.

`_top_prob_decisions` reads the chosen rows back with `eval_df.loc[labels]`. A panel
assembled market by market (`pd.concat` of one frame per market, each indexed 0..K-1)
repeats every label K times per date, and `.loc` then returns the rows of *every* market
carrying that label: a trade per bin of every market instead of one per market.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.policy import TradePolicy

LABELS = ("a", "b", "c")


class _FeatureProb:
    """A forecaster whose predicted probability is the `feature_p` column."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["feature_p"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


def _market(i: int, date: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "market_id": f"m{i}",
            "decision_date": date,
            "outcome_label": label,
            "winning_label": LABELS[i % len(LABELS)],
            "market_price": 0.2 + 0.1 * k,
            "feature_p": 0.2 + 0.2 * ((i + k) % 3),
        }
        for k, label in enumerate(LABELS)
    ]
    return pd.DataFrame(rows)  # indexed 0..K-1, like every other market's frame


def _stacked() -> pd.DataFrame:
    dates = [f"2026-01-{d:02d}" for d in range(1, 9)]
    return pd.concat([_market(10 * d + m, date) for d, date in enumerate(dates) for m in range(4)])


@pytest.mark.parametrize(
    "policy", [TradePolicy.top_bin_yes(), TradePolicy.live(min_edge=0.0)], ids=["top", "live"]
)
def test_a_panel_stacked_market_by_market_trades_like_one_with_a_clean_index(
    policy: TradePolicy,
) -> None:
    stacked = _stacked()
    assert not stacked.index.is_unique
    clean = stacked.reset_index(drop=True)
    kwargs: dict[str, Any] = {
        "min_train_rows": 12,
        "stride": 1,
        "costs": CostModel.frictionless(),
        "policy": policy,
    }
    a = rolling_origin_eval(stacked, _FeatureProb(), **kwargs).trades
    b = rolling_origin_eval(clean, _FeatureProb(), **kwargs).trades
    cols = ["market_id", "eval_date", "outcome_label", "direction", "realized_pnl"]
    assert len(b) > 0
    assert a[cols].equals(b[cols])
    assert a.groupby(["market_id", "eval_date"]).size().max() == 1


def test_the_callers_panel_is_not_modified() -> None:
    stacked = _stacked()
    before = stacked.copy()
    rolling_origin_eval(
        stacked,
        _FeatureProb(),
        min_train_rows=12,
        stride=1,
        policy=TradePolicy.top_bin_yes(),
    )
    pd.testing.assert_frame_equal(stacked, before)
