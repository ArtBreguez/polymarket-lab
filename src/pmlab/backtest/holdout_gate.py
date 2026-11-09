"""Holdout gate for go/no-go publish decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SegmentGateResult:
    segment: str
    num_trades: int
    total_pnl: float
    passes: bool
    reason: str


def _grade(pnl: pd.Series, min_trades: int, min_pnl: float) -> tuple[int, float, bool, str]:
    """(trades, total PnL, passes, reason) of one set of trades.

    A trade without a finite ``realized_pnl`` (still open, or its cost was NaN) is no evidence:
    ``Series.sum`` would skip it, so it would count towards ``min_trades`` and add nothing to the
    PnL, and sixty open trades would be a GO (#51). Such a set fails instead, saying why.
    """
    num = len(pnl)
    values = pd.to_numeric(pnl, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        return num, float(np.nansum(values)), False, "unsettled_trades"
    if np.isinf(values).any():
        return num, float(values.sum()), False, "non_finite_pnl"
    total_pnl = float(values.sum())
    if num < min_trades:
        return num, total_pnl, False, "insufficient_trades"
    if total_pnl < min_pnl:
        return num, total_pnl, False, "negative_pnl"
    return num, total_pnl, True, "ok"


@dataclass
class HoldoutGateResult:
    decision: str
    segment_results: list[SegmentGateResult]
    aggregate_pnl: float
    aggregate_trades: int

    @classmethod
    def evaluate(
        cls,
        trades: pd.DataFrame,
        required_segments: list[str] | None = None,
        min_trades_per_segment: int = 40,
        min_pnl_per_segment: float = 0.0,
    ) -> HoldoutGateResult:
        """Evaluate the holdout gate.

        Args:
            trades: Backtest trade log (as produced by ``rolling_origin_eval``).
            required_segments: Segments that must each pass. When ``None`` or an
                empty list, the gate is evaluated on a single aggregate ``"all"``
                segment — so the output of ``rolling_origin_eval`` can be graded
                directly without splitting into segments first.
            min_trades_per_segment: Minimum trades a segment needs to pass.
            min_pnl_per_segment: Minimum total PnL a segment needs to pass.
        """
        # Aggregate mode: no segmentation requested (None or empty list) — grade
        # the whole trade log. An empty list must NOT auto-pass: falling through
        # to the segment loop would yield zero results and all([]) == True → GO,
        # which would let a losing model be published (champion hard-gate breach).
        if not required_segments:
            pnl_all = trades["realized_pnl"] if not trades.empty else pd.Series([], dtype=float)
            num, total_pnl, passes, reason = _grade(
                pnl_all, min_trades_per_segment, min_pnl_per_segment
            )
            seg_result = SegmentGateResult(
                segment="all",
                num_trades=num,
                total_pnl=total_pnl,
                passes=passes,
                reason=reason,
            )
            return cls(
                decision="GO" if passes else "NO_GO",
                segment_results=[seg_result],
                aggregate_pnl=total_pnl,
                aggregate_trades=num,
            )

        segment_results: list[SegmentGateResult] = []

        for seg in required_segments:
            seg_trades = trades[trades["segment"] == seg] if not trades.empty else pd.DataFrame()

            if seg_trades.empty and seg not in (
                trades["segment"].unique() if not trades.empty else []
            ):
                segment_results.append(
                    SegmentGateResult(
                        segment=seg,
                        num_trades=0,
                        total_pnl=0.0,
                        passes=False,
                        reason="missing",
                    )
                )
                continue

            num, total_pnl, passes, reason = _grade(
                seg_trades["realized_pnl"], min_trades_per_segment, min_pnl_per_segment
            )

            segment_results.append(
                SegmentGateResult(
                    segment=seg,
                    num_trades=num,
                    total_pnl=total_pnl,
                    passes=passes,
                    reason=reason,
                )
            )

        all_pass = all(r.passes for r in segment_results)
        decision = "GO" if all_pass else "NO_GO"
        agg_pnl = (
            float(np.nansum(pd.to_numeric(trades["realized_pnl"], errors="coerce")))
            if not trades.empty
            else 0.0
        )
        agg_trades = len(trades)

        return cls(
            decision=decision,
            segment_results=segment_results,
            aggregate_pnl=agg_pnl,
            aggregate_trades=agg_trades,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "aggregate_pnl": self.aggregate_pnl,
            "aggregate_trades": self.aggregate_trades,
            "segment_results": [
                {
                    "segment": r.segment,
                    "num_trades": r.num_trades,
                    "total_pnl": r.total_pnl,
                    "passes": r.passes,
                    "reason": r.reason,
                }
                for r in self.segment_results
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HoldoutGateResult:
        segment_results = [
            SegmentGateResult(
                segment=r["segment"],
                num_trades=r["num_trades"],
                total_pnl=r["total_pnl"],
                passes=r["passes"],
                reason=r["reason"],
            )
            for r in d["segment_results"]
        ]
        return cls(
            decision=d["decision"],
            segment_results=segment_results,
            aggregate_pnl=d["aggregate_pnl"],
            aggregate_trades=d["aggregate_trades"],
        )
