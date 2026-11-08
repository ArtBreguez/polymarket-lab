"""A crafted panel where the historical rule and the live policy disagree about two configs.

Two forecasters read their probabilities straight from a feature column, so a "config" is
just which column to use and every number below is a by-hand consequence of the panel:

* **A** (``feature_p_a``) is a modest, well-aimed model: on the 3-bin ``n*`` markets it backs
  the bin that wins 60% of the time (priced 0.50), but never believes it beats a price by
  more than 0.10. Forced to trade every market (``top_prob``) it makes money; held to a
  threshold of 0.20 it never trades at all.
* **B** (``feature_p_b``) has no opinion on the ``n*`` markets (0.33/0.34/0.33: its top bin
  is a bin that wins 20% of the time, priced 0.30), and one strong view on the 2-bin ``z*``
  market: the 0.10 bin is worth 0.45. Forced to trade every market it bleeds on the ``n*``
  markets; held to 0.20 it only takes the ``z*`` bet, which pays.

So the historical top-prob rule ranks A over B, ``TradePolicy.live(min_edge=0.20)`` ranks B
over A, and ``TradePolicy.live(min_edge=0.05)`` ranks A over B again: the threshold alone
changes which hyper-parameters win.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

COL_A = "feature_p_a"
COL_B = "feature_p_b"

# (label, YES price, p_a, p_b) of the three-bin "n" markets and the two-bin "z" market.
_N_BINS = (("x", 0.5, 0.6, 0.33), ("y", 0.3, 0.2, 0.34), ("z", 0.2, 0.2, 0.33))
_Z_BINS = (("x", 0.9, 0.9, 0.55), ("y", 0.1, 0.1, 0.45))


class PickFeature:
    """A forecaster whose probability *is* the feature ``col`` (fit is a no-op)."""

    def __init__(self, col: str) -> None:
        self.col = col

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[self.col].to_numpy(float)
        return np.column_stack([1.0 - p, p])


def build_pick(params: dict[str, Any]) -> PickFeature:
    return PickFeature(params["col"])


def pick_space(trial: Any) -> dict[str, Any]:
    return {"col": trial.suggest_categorical("col", [COL_A, COL_B])}


def crafted_panel(n_dates: int = 48) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = pd.date_range("2026-01-01", periods=n_dates).strftime("%Y-%m-%d")
    for d, date in enumerate(dates):
        for j in range(4):
            slot = (d + j) % 5
            winner = "x" if slot in (0, 1, 2) else ("y" if slot == 3 else "z")
            for label, price, p_a, p_b in _N_BINS:
                rows.append(_row(f"n{d}-{j}", date, label, winner, price, p_a, p_b))
        winner = "y" if d % 6 == 0 else "x"
        for label, price, p_a, p_b in _Z_BINS:
            rows.append(_row(f"z{d}", date, label, winner, price, p_a, p_b))
    return pd.DataFrame(rows)


def _row(
    market: str, date: str, label: str, winner: str, price: float, p_a: float, p_b: float
) -> dict[str, Any]:
    return {
        "market_id": market,
        "decision_date": date,
        "outcome_label": label,
        "winning_label": winner,
        "market_price": price,
        "feature_p_a": p_a,
        "feature_p_b": p_b,
    }
