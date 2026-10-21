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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pmlab.core.policy import TradePolicy

__all__ = ["MIN_GROUP_TRADES", "BacktestExpectation", "GroupExpectation"]

# A side or a segment is given an expectation of its own only if the backtest traded it at
# least this often; a promise made on fewer trades is not one a live book can be held to.
# It matches the watchdog's default `min_trades`.
MIN_GROUP_TRADES = 30

_DIRECTIONS = ("yes", "no")


@dataclass(frozen=True)
class GroupExpectation:
    """Mean, spread and size of one group of backtest trades: a side, or a segment.

    The same numbers `BacktestExpectation` keeps for the whole backtest, per unit
    stake. A group is held to the same rules as the whole: every number finite, a
    spread that is not negative, at least two trades, a Brier score in [0, 1].
    """

    n_trades: int
    mean_pnl_per_stake: float
    std_pnl_per_stake: float
    brier_score: float | None = None
    brier_std: float | None = None

    def __post_init__(self) -> None:
        _check_trade_count(self.n_trades)
        _check_statistics(
            self.mean_pnl_per_stake, self.std_pnl_per_stake, self.brier_score, self.brier_std
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "mean_pnl_per_stake": self.mean_pnl_per_stake,
            "std_pnl_per_stake": self.std_pnl_per_stake,
            "brier_score": self.brier_score,
            "brier_std": self.brier_std,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroupExpectation:
        """Restore a record; raises ValueError (KeyError if a key is absent) if unusable."""
        brier_score = data.get("brier_score")
        brier_std = data.get("brier_std")
        return cls(
            n_trades=_as_int("n_trades", data["n_trades"]),
            mean_pnl_per_stake=_as_float("mean_pnl_per_stake", data["mean_pnl_per_stake"]),
            std_pnl_per_stake=_as_float("std_pnl_per_stake", data["std_pnl_per_stake"]),
            brier_score=None if brier_score is None else _as_float("brier_score", brier_score),
            brier_std=None if brier_std is None else _as_float("brier_std", brier_std),
        )


@dataclass(frozen=True)
class BacktestExpectation:
    """Mean, spread and size of the backtest trades a champion was promoted on.

    `brier_score` / `brier_std` are the mean and standard deviation of the
    per-trade squared forecast error. They are `None` when the backtest trades did
    not carry both `predicted_prob` and `label_occurred`.

    `by_direction` holds the same numbers for the YES trades and for the NO trades on
    their own: a NO trade is a buy of a different token, picked where the model is
    bearish, and its PnL and forecast error are not draws from the YES distribution.
    ``None`` means the backtest did not record a split (a champion published before
    0.9.1, or a frame without a `direction` column): the pooled numbers then describe
    the historical backtest, which only bought YES. A side the backtest traded fewer
    than `MIN_GROUP_TRADES` times is absent from the mapping, unless it traded nothing
    else (then it is the pooled expectation); an empty mapping means neither side traded
    enough to be promised.

    `policy` is the decision rule the backtest ran (`TradePolicy`), recorded so live
    trading can use the same threshold. ``None`` when not recorded.

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
    by_direction: Mapping[str, GroupExpectation] | None = None
    policy: TradePolicy | None = None

    def __post_init__(self) -> None:
        _check_trade_count(self.n_trades)
        _require_finite("flat_stake", self.flat_stake)
        if self.flat_stake <= 0:
            raise ValueError(f"flat_stake must be positive, got {self.flat_stake!r}")
        _check_statistics(
            self.mean_pnl_per_stake, self.std_pnl_per_stake, self.brier_score, self.brier_std
        )
        if self.by_direction is not None:
            object.__setattr__(
                self,
                "by_direction",
                _checked_split(self.by_direction, _DIRECTIONS, "direction", self.n_trades),
            )
        if self.policy is not None and not isinstance(self.policy, TradePolicy):
            raise ValueError(f"policy must be a TradePolicy, got {self.policy!r}")

    @classmethod
    def from_trades(
        cls,
        trades: pd.DataFrame,
        *,
        flat_stake: float = 1.0,
        policy: TradePolicy | None = None,
        min_group_trades: int = MIN_GROUP_TRADES,
    ) -> BacktestExpectation:
        """Summarise a `rolling_origin_eval(...).trades` frame.

        Args:
            trades: Needs `realized_pnl`. If it also has `predicted_prob` and
                `label_occurred`, the Brier expectation is filled in too. If it has a
                `direction` column ("yes" / "no"), each side with at least
                `min_group_trades` trades gets an expectation of its own.
            flat_stake: The notional each backtest trade risked — the
                `flat_stake` the backtest was run with.
            policy: The trade policy the backtest ran, to record with the numbers.
            min_group_trades: Fewest trades a side needs to be recorded (at least 2).

        Raises:
            ValueError: On fewer than two trades (no spread to measure), a
                non-finite PnL, a non-positive stake, forecasts outside [0, 1] or a
                `direction` other than yes/no.
        """
        if not math.isfinite(flat_stake) or flat_stake <= 0:
            raise ValueError(f"flat_stake must be a positive finite number, got {flat_stake!r}")
        if isinstance(min_group_trades, bool) or not isinstance(min_group_trades, int):
            raise ValueError(f"min_group_trades must be an integer >= 2, got {min_group_trades!r}")
        if min_group_trades < 2:
            raise ValueError(f"min_group_trades must be an integer >= 2, got {min_group_trades!r}")
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
        prob: np.ndarray | None = None
        occurred: np.ndarray | None = None
        if "predicted_prob" in trades.columns and "label_occurred" in trades.columns:
            prob = trades["predicted_prob"].to_numpy(dtype=float)
            if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
                raise ValueError("predicted_prob must be finite probabilities in [0, 1]")
            occurred = trades["label_occurred"].to_numpy(dtype=bool).astype(float)

        by_direction: dict[str, GroupExpectation] | None = None
        if "direction" in trades.columns:
            sides = trades["direction"].astype(str).to_numpy()
            unknown = sorted(set(sides) - set(_DIRECTIONS))
            if unknown:
                raise ValueError(
                    f"trades has unknown direction value(s) {unknown}; valid: {list(_DIRECTIONS)}"
                )
            by_direction = {}
            for side in _DIRECTIONS:
                mask = sides == side
                n_side = int(mask.sum())
                # A side that holds every trade is the pooled expectation, which has
                # always been usable from two trades up; only a side that shares the
                # book with the other one needs enough trades to stand on its own.
                if n_side >= min_group_trades or n_side == len(sides):
                    by_direction[side] = _summarise(
                        per_stake[mask],
                        None if prob is None else prob[mask],
                        None if occurred is None else occurred[mask],
                    )

        whole = _summarise(per_stake, prob, occurred)
        return cls(
            n_trades=whole.n_trades,
            flat_stake=float(flat_stake),
            mean_pnl_per_stake=whole.mean_pnl_per_stake,
            std_pnl_per_stake=whole.std_pnl_per_stake,
            brier_score=whole.brier_score,
            brier_std=whole.brier_std,
            by_direction=by_direction,
            policy=policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "flat_stake": self.flat_stake,
            "mean_pnl_per_stake": self.mean_pnl_per_stake,
            "std_pnl_per_stake": self.std_pnl_per_stake,
            "brier_score": self.brier_score,
            "brier_std": self.brier_std,
            "by_direction": (
                None
                if self.by_direction is None
                else {side: stats.to_dict() for side, stats in self.by_direction.items()}
            ),
            "policy": None if self.policy is None else self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BacktestExpectation:
        """Restore a record; raises ValueError (KeyError if a key is absent) if unusable.

        A record written before 0.9.1 has neither `by_direction` nor `policy`; both load
        as ``None``. One that has them but cannot be used raises: it must never read as
        "not recorded" and quietly weaken the watchdog.
        """
        brier_score = data.get("brier_score")
        brier_std = data.get("brier_std")
        raw_split = data.get("by_direction")
        by_direction: dict[str, GroupExpectation] | None = None
        if raw_split is not None:
            if not isinstance(raw_split, Mapping):
                raise ValueError(f"by_direction must be an object, got {raw_split!r}")
            by_direction = {
                str(side): GroupExpectation.from_dict(stats) for side, stats in raw_split.items()
            }
        raw_policy = data.get("policy")
        return cls(
            n_trades=_as_int("n_trades", data["n_trades"]),
            flat_stake=_as_float("flat_stake", data["flat_stake"]),
            mean_pnl_per_stake=_as_float("mean_pnl_per_stake", data["mean_pnl_per_stake"]),
            std_pnl_per_stake=_as_float("std_pnl_per_stake", data["std_pnl_per_stake"]),
            brier_score=None if brier_score is None else _as_float("brier_score", brier_score),
            brier_std=None if brier_std is None else _as_float("brier_std", brier_std),
            by_direction=by_direction,
            policy=None if raw_policy is None else TradePolicy.from_dict(raw_policy),
        )


def _summarise(
    per_stake: np.ndarray, prob: np.ndarray | None, occurred: np.ndarray | None
) -> GroupExpectation:
    """Mean, spread and Brier of one group of trades (per-stake PnL, forecast terms)."""
    brier_score: float | None = None
    brier_std: float | None = None
    if prob is not None and occurred is not None:
        terms = (prob - occurred) ** 2
        brier_score = float(terms.mean())
        brier_std = float(terms.std(ddof=1))
    return GroupExpectation(
        n_trades=len(per_stake),
        mean_pnl_per_stake=float(per_stake.mean()),
        std_pnl_per_stake=float(per_stake.std(ddof=1)),
        brier_score=brier_score,
        brier_std=brier_std,
    )


def _check_trade_count(n_trades: object) -> None:
    if isinstance(n_trades, bool) or not isinstance(n_trades, int):
        raise ValueError(f"n_trades must be an integer, got {n_trades!r}")
    if n_trades < 2:
        raise ValueError(f"n_trades must be at least 2 (one trade has no spread), got {n_trades}")


def _check_statistics(
    mean: float, std: float, brier_score: float | None, brier_std: float | None
) -> None:
    _require_finite("mean_pnl_per_stake", mean)
    _require_finite("std_pnl_per_stake", std)
    if std < 0:
        raise ValueError(f"std_pnl_per_stake must be >= 0, got {std!r}")
    if brier_score is not None:
        _require_finite("brier_score", brier_score)
        if not 0.0 <= brier_score <= 1.0:
            raise ValueError(f"brier_score must be within [0, 1], got {brier_score!r}")
    if brier_std is not None:
        _require_finite("brier_std", brier_std)
        if brier_std < 0:
            raise ValueError(f"brier_std must be >= 0, got {brier_std!r}")


def _checked_split(
    split: Mapping[str, GroupExpectation],
    valid: tuple[str, ...],
    noun: str,
    n_trades: int,
) -> dict[str, GroupExpectation]:
    """A copy of `split` in the canonical order, or ValueError if it cannot be a split."""
    if not isinstance(split, Mapping):
        raise ValueError(f"by_{noun} must be an object, got {split!r}")
    unknown = sorted(set(split) - set(valid))
    if unknown:
        raise ValueError(f"unknown {noun}(s) {unknown} in the expectation; valid: {list(valid)}")
    checked: dict[str, GroupExpectation] = {}
    for key in valid:
        if key not in split:
            continue
        stats = split[key]
        if not isinstance(stats, GroupExpectation):
            raise ValueError(f"by_{noun}[{key!r}] must be a GroupExpectation, got {stats!r}")
        if stats.n_trades > n_trades:
            raise ValueError(
                f"{noun} {key!r} has {stats.n_trades} trades, more than the whole backtest's "
                f"{n_trades}"
            )
        checked[key] = stats
    return checked


def _require_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _as_float(name: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def _as_int(name: str, value: Any) -> int:
    """A whole number from JSON: an int, a whole float (50.0) or a numeric string."""
    try:
        number = int(value)
        if isinstance(value, float) and number != value:
            raise ValueError(value)  # int(2.7) would silently give 2
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    return number
