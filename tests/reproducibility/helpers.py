"""Shared builders for the reproducibility tests: a small panel and a stochastic LightGBM config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Bagging and feature sub-sampling make LightGBM consume its seed: with these, two different
# seeds give two different models, so "same metrics" means the seed really was re-applied.
STOCHASTIC_PARAMS: dict[str, Any] = {
    "n_estimators": 25,
    "num_leaves": 8,
    "min_child_samples": 5,
    "bagging_fraction": 0.6,
    "bagging_freq": 1,
    "feature_fraction": 0.6,
}
# The same params as `pmlab backtest --lgbm-param` flags.
STOCHASTIC_FLAGS: list[str] = [
    arg for k, v in STOCHASTIC_PARAMS.items() for arg in ("--lgbm-param", f"{k}={v}")
]


def make_panel(n_dates: int = 80, markets_per_date: int = 3, seed: int = 0) -> pd.DataFrame:
    """A walk-forward panel: per market a YES and a NO row, four noisy features."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        day = f"2026-{1 + d // 28:02d}-{d % 28 + 1:02d}"
        for k in range(markets_per_date):
            winner = "YES" if rng.random() < 0.5 else "NO"
            for label in ("YES", "NO"):
                signal = 1.0 if label == winner else -1.0
                rows.append(
                    {
                        "market_id": f"m{d}_{k}",
                        "decision_date": day,
                        "outcome_label": label,
                        "winning_label": winner,
                        "market_price": 0.5,
                        "feature_a": signal * 0.4 + rng.normal(),
                        "feature_b": rng.normal(),
                        "feature_c": rng.normal(),
                        "feature_d": rng.normal(),
                        "segment": "A",
                    }
                )
    return pd.DataFrame(rows)


def write_panel(path: Path, **kwargs: Any) -> Path:
    make_panel(**kwargs).to_parquet(path)
    return path
