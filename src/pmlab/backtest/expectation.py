"""What a champion promises: the backtest numbers a live run is held against.

The holdout gate records totals — `aggregate_pnl` over `aggregate_trades` — and
nothing about spread. A live run compared with a bare mean has nothing to be
surprised *against*: whether a live average of 0.02 per trade is a failure or
noise depends on how noisy the per-trade PnL is. So the expectation keeps the
three numbers a significance test needs (mean, standard deviation, count), and
records them at promotion, in `champion.json`, where the live run can read them.

Everything is **per unit stake** (`realized_pnl / flat_stake`). PnL per trade
scales with the stake, so comparing a backtest run at 1.0 with a live run at 10.0
would read as a ten-fold outperformance. Per-stake returns make the comparison
independent of the size traded (a depth-sensitive cost model aside — see
`docs/watchdog.md`).

The Brier terms are scored on the *traded* rows, not the whole panel: live trades
are a selected subset (those with an edge), and a calibration comparison is only
fair between like subsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["BacktestExpectation"]


@dataclass(frozen=True)
class BacktestExpectation:
    """Mean, spread and size of the backtest trades a champion was promoted on.

    `brier_score` / `brier_std` are the mean and standard deviation of the
    per-trade squared forecast error. They are `None` when the backtest trades did
    not carry both `predicted_prob` and `label_occurred`.

    An expectation is only worth recording if a live run can be measured against it,
    so the constructor refuses one that cannot: every number finite, a spread that is
    not negative, a stake above zero, a Brier score in [0, 1] and at least two
    trades (one trade has no spread, none divides by zero). NaN compares false with
    everything, so it would not fail a test, it would silently switch the check off.
    The mean may be zero or negative; that is a finding the watchdog reports, not a
    malformed record.
    """

    n_trades: int
    flat_stake: float
    mean_pnl_per_stake: float
    std_pnl_per_stake: float
    brier_score: float | None = None
    brier_std: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.n_trades, bool) or not isinstance(self.n_trades, int):
            raise ValueError(f"n_trades must be an integer, got {self.n_trades!r}")
        if self.n_trades < 2:
            raise ValueError(
                f"n_trades must be at least 2 (one trade has no spread), got {self.n_trades}"
            )
        _require_finite("flat_stake", self.flat_stake)
        if self.flat_stake <= 0:
            raise ValueError(f"flat_stake must be positive, got {self.flat_stake!r}")
        _require_finite("mean_pnl_per_stake", self.mean_pnl_per_stake)
        _require_finite("std_pnl_per_stake", self.std_pnl_per_stake)
        if self.std_pnl_per_stake < 0:
            raise ValueError(f"std_pnl_per_stake must be >= 0, got {self.std_pnl_per_stake!r}")
        if self.brier_score is not None:
            _require_finite("brier_score", self.brier_score)
            if not 0.0 <= self.brier_score <= 1.0:
                raise ValueError(f"brier_score must be within [0, 1], got {self.brier_score!r}")
        if self.brier_std is not None:
            _require_finite("brier_std", self.brier_std)
            if self.brier_std < 0:
                raise ValueError(f"brier_std must be >= 0, got {self.brier_std!r}")

    @classmethod
    def from_trades(cls, trades: pd.DataFrame, *, flat_stake: float = 1.0) -> BacktestExpectation:
        """Summarise a `rolling_origin_eval(...).trades` frame.

        Args:
            trades: Needs `realized_pnl`. If it also has `predicted_prob` and
                `label_occurred`, the Brier expectation is filled in too.
            flat_stake: The notional each backtest trade risked — the
                `flat_stake` the backtest was run with.

        Raises:
            ValueError: On fewer than two trades (no spread to measure), a
                non-finite PnL, a non-positive stake, or forecasts outside [0, 1].
        """
        if not math.isfinite(flat_stake) or flat_stake <= 0:
            raise ValueError(f"flat_stake must be a positive finite number, got {flat_stake!r}")
        if "realized_pnl" not in trades.columns:
            raise ValueError("trades has no realized_pnl column")

        pnl = trades["realized_pnl"].to_numpy(dtype=float)
        if len(pnl) < 2:
            raise ValueError(
                f"an expectation needs at least 2 trades to have a spread, got {len(pnl)}"
            )
        if not np.isfinite(pnl).all():
            raise ValueError("realized_pnl must be finite for every trade")

        per_stake = pnl / flat_stake
        brier_score: float | None = None
        brier_std: float | None = None
        if "predicted_prob" in trades.columns and "label_occurred" in trades.columns:
            prob = trades["predicted_prob"].to_numpy(dtype=float)
            if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
                raise ValueError("predicted_prob must be finite probabilities in [0, 1]")
            occurred = trades["label_occurred"].to_numpy(dtype=bool).astype(float)
            terms = (prob - occurred) ** 2
            brier_score = float(terms.mean())
            brier_std = float(terms.std(ddof=1))

        return cls(
            n_trades=len(pnl),
            flat_stake=float(flat_stake),
            mean_pnl_per_stake=float(per_stake.mean()),
            std_pnl_per_stake=float(per_stake.std(ddof=1)),
            brier_score=brier_score,
            brier_std=brier_std,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "flat_stake": self.flat_stake,
            "mean_pnl_per_stake": self.mean_pnl_per_stake,
            "std_pnl_per_stake": self.std_pnl_per_stake,
            "brier_score": self.brier_score,
            "brier_std": self.brier_std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BacktestExpectation:
        """Restore a record; raises ValueError (KeyError if a key is absent) if unusable."""
        brier_score = data.get("brier_score")
        brier_std = data.get("brier_std")
        return cls(
            n_trades=_as_int("n_trades", data["n_trades"]),
            flat_stake=_as_float("flat_stake", data["flat_stake"]),
            mean_pnl_per_stake=_as_float("mean_pnl_per_stake", data["mean_pnl_per_stake"]),
            std_pnl_per_stake=_as_float("std_pnl_per_stake", data["std_pnl_per_stake"]),
            brier_score=None if brier_score is None else _as_float("brier_score", brier_score),
            brier_std=None if brier_std is None else _as_float("brier_std", brier_std),
        )


def _require_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _as_float(name: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def _as_int(name: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
