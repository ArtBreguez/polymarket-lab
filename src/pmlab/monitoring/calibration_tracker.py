"""Live calibration tracking on realized trades.

The holdout gate says a model was calibrated on historical data. This says
whether it still is on money that was actually at risk.

Two things worth knowing before reading the code:

**`outcome` is about the label, not the profit.** `SettlementEngine` writes
``outcome = "won" if trade["outcome_label"] == winning_label else "lost"``, with
no reference to `direction`. So a `direction="no"` trade whose label *does*
occur is recorded as ``"won"`` while its `realized_pnl` is negative. For
calibration that is exactly right — we are scoring a forecast about the label,
not a trade's profitability — but reading ``"won"`` as "we made money" would
invert the target. Hence `_label_occurred`, named for what it means.

**Calibration is not PnL.** A model can be perfectly calibrated and lose money
(fees, adverse selection), or badly calibrated and win (a lucky streak). Both
numbers are needed; neither substitutes for the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pmlab.modeling.diagnostics import brier_decomposition, reliability_data

__all__ = ["CalibrationWindow", "CalibrationTracker"]


@dataclass(frozen=True)
class CalibrationWindow:
    """Calibration over one set of realized trades.

    `reliability` is the Murphy reliability term: the probability-weighted mean
    squared gap between what was forecast and what occurred. Lower is better,
    and unlike `brier_score` it does not improve just because the base rate is
    extreme.
    """

    n_trades: int
    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    skill_score: float
    base_rate: float
    mean_forecast: float
    window_start: str
    window_end: str
    reliability_curve: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "brier_score": self.brier_score,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "skill_score": self.skill_score,
            "base_rate": self.base_rate,
            "mean_forecast": self.mean_forecast,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "reliability_curve": [dict(p) for p in self.reliability_curve],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationWindow:
        return cls(
            n_trades=int(data["n_trades"]),
            brier_score=float(data["brier_score"]),
            reliability=float(data["reliability"]),
            resolution=float(data["resolution"]),
            uncertainty=float(data["uncertainty"]),
            skill_score=float(data["skill_score"]),
            base_rate=float(data["base_rate"]),
            mean_forecast=float(data["mean_forecast"]),
            window_start=str(data["window_start"]),
            window_end=str(data["window_end"]),
            reliability_curve=tuple(dict(p) for p in data["reliability_curve"]),
        )


def _label_occurred(trade: dict[str, Any]) -> float:
    """1.0 if the forecast label occurred, else 0.0.

    Deliberately not "did this trade profit" — see the module docstring.
    """
    return 1.0 if trade.get("outcome") == "won" else 0.0


def _is_scoreable(trade: dict[str, Any]) -> bool:
    """A trade can be scored only if it has both a forecast and an outcome."""
    if trade.get("outcome") not in ("won", "lost"):
        return False  # still open: no truth yet, and no lookahead allowed
    prob = trade.get("model_prob")
    return isinstance(prob, (int, float)) and not isinstance(prob, bool)


def _window_from(trades: list[dict[str, Any]], n_bins: int) -> CalibrationWindow:
    y_true = np.array([_label_occurred(t) for t in trades], dtype=float)
    y_prob = np.array([float(t["model_prob"]) for t in trades], dtype=float)
    decomp = brier_decomposition(y_true, y_prob, n_bins=n_bins)

    centers, mean_pred, frac_pos = reliability_data(y_true, y_prob, n_bins=n_bins)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    curve: list[dict[str, float]] = []
    for center, mp, fp in zip(centers, mean_pred, frac_pos, strict=True):
        k = int(np.argmin(np.abs((bins[:-1] + bins[1:]) / 2.0 - center)))
        curve.append(
            {
                "bin_center": float(center),
                "mean_predicted": float(mp),
                "fraction_occurred": float(fp),
                "n": float(int((idx == k).sum())),
            }
        )

    dates = [str(t.get("target_date", "")) for t in trades]
    return CalibrationWindow(
        n_trades=len(trades),
        brier_score=decomp.brier_score,
        reliability=decomp.reliability,
        resolution=decomp.resolution,
        uncertainty=decomp.uncertainty,
        skill_score=decomp.skill_score,
        base_rate=float(y_true.mean()),
        mean_forecast=float(y_prob.mean()),
        window_start=min(dates) if dates else "",
        window_end=max(dates) if dates else "",
        reliability_curve=tuple(curve),
    )


class CalibrationTracker:
    """Scores realized trades from a paper/live trade log.

    Only settled rows that carry a `model_prob` are scored. Trades written before
    the prediction was persisted are counted in `n_skipped` rather than silently
    treated as zero — an unscoreable trade is missing information, not evidence
    of a bad forecast.
    """

    def __init__(self, trades: list[dict[str, Any]], *, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        scoreable = [t for t in trades if _is_scoreable(t)]
        # Truth arrives on target_date, so that is the order calibration evolves in.
        self._trades = sorted(scoreable, key=lambda t: str(t.get("target_date", "")))
        self.n_realized = len(self._trades)
        self.n_skipped = len(trades) - self.n_realized

    @classmethod
    def from_path(cls, path: Path | str, *, n_bins: int = 10) -> CalibrationTracker:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"trade log not found: {p}")
        payload = json.loads(p.read_text())
        return cls(list(payload.get("trades", [])), n_bins=n_bins)

    def overall(self) -> CalibrationWindow | None:
        """Calibration over every realized trade, or None if none are scoreable."""
        if not self._trades:
            return None
        return _window_from(self._trades, self.n_bins)

    def rolling(self, window: int) -> list[CalibrationWindow]:
        """One window per trade once `window` trades have accumulated."""
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if len(self._trades) < window:
            return []
        return [
            _window_from(self._trades[i - window : i], self.n_bins)
            for i in range(window, len(self._trades) + 1)
        ]

    def by_segment(self) -> dict[str, CalibrationWindow]:
        """Per-segment calibration.

        The gate promotes per segment, so an aggregate that hides one broken
        segment is the failure mode worth guarding against.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        for trade in self._trades:
            groups.setdefault(str(trade.get("city_or_segment", "unknown")), []).append(trade)
        return {seg: _window_from(rows, self.n_bins) for seg, rows in sorted(groups.items())}
