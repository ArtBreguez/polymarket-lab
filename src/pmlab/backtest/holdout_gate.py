"""Holdout gate for go/no-go publish decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SegmentGateResult:
    segment: str
    num_trades: int
    total_pnl: float
    passes: bool
    reason: str


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
        required_segments: list[str],
        min_trades_per_segment: int = 40,
        min_pnl_per_segment: float = 0.0,
    ) -> HoldoutGateResult:
        """Evaluate holdout gate across required segments."""
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

            num = len(seg_trades)
            total_pnl = float(seg_trades["realized_pnl"].sum())

            if num < min_trades_per_segment:
                reason = "insufficient_trades"
                passes = False
            elif total_pnl < min_pnl_per_segment:
                reason = "negative_pnl"
                passes = False
            else:
                reason = "ok"
                passes = True

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
        agg_pnl = float(trades["realized_pnl"].sum()) if not trades.empty else 0.0
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
