"""Walk-forward (rolling origin) evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from pmlab.core.costs import CostModel
from pmlab.core.edge import edge_clears, quote_edges
from pmlab.core.pnl import Position, settle_position
from pmlab.core.policy import TradePolicy


class _ModelProtocol(Protocol):
    """Minimal protocol for forecasters used in rolling_origin_eval."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class RollingOriginResult:
    trades: pd.DataFrame
    steps: list[dict[str, Any]] = field(default_factory=list)


def rolling_origin_eval(
    panel: pd.DataFrame,
    model: _ModelProtocol,
    min_train_rows: int = 20,
    stride: int = 10,
    flat_stake: float = 1.0,
    taker_bps: float | None = None,
    costs: CostModel | None = None,
    policy: TradePolicy | None = None,
) -> RollingOriginResult:
    """Walk-forward evaluation on a panel dataset.

    panel columns required:
        market_id, decision_date (str YYYY-MM-DD),
        outcome_label, winning_label, market_price,
        + feature_* columns

    For each eval step:
        - Train on rows with decision_date < eval_date
        - Predict on rows at eval_date
        - Decide the trade of each (market_id, decision_date) under ``policy``
        - Compute PnL using settle_position

    ``market_price`` is the label's YES price **at the row's decision_date** (the plugin
    contract, see ``MarketPlugin.build_training_row``). A decision uses only its own row's
    price and forecast, so nothing later than the decision date reaches it.

    Args:
        panel: Training/eval panel (see column contract above).
        model: Anything with sklearn-style ``fit`` / ``predict_proba``.
        min_train_rows: Skip an eval date until this many training rows exist.
        stride: Evaluate every Nth distinct decision_date.
        flat_stake: Notional USDC risked per trade.
        taker_bps: Legacy fee-only knob. Mutually exclusive with ``costs``.
        costs: Full :class:`~pmlab.core.costs.CostModel` (fee + slippage + depth).
            Omit both arguments for the historical default (30bps, fill at quote).
        policy: The decision rule. Omit it for the historical backtest: YES on the bin
            with the highest predicted probability, every market, no threshold (the
            numbers published before policies existed reproduce). Pass
            ``TradePolicy.live(min_edge=...)`` to trade what ``record-trades`` trades:
            per market the (bin, side) with the best after-fee edge, YES or NO, taken if
            that edge is at least ``min_edge``. Both price the trade the way the broker
            and settlement do: a NO trade is a buy of the NO token at ``1 - market_price``.

    Returns:
        RollingOriginResult with all trade records and step metadata.

    Raises:
        ValueError: If both ``taker_bps`` and ``costs`` are supplied, which would
            give the fee two conflicting sources of truth.
    """
    if taker_bps is not None and costs is not None:
        raise ValueError(
            "pass either taker_bps or costs, not both — costs.taker_bps already carries the fee"
        )
    if costs is None:
        costs = CostModel() if taker_bps is None else CostModel(taker_bps=taker_bps)

    panel = panel.copy()
    panel["decision_date"] = panel["decision_date"].astype(str)

    feature_cols = [c for c in panel.columns if c.startswith("feature_")]

    sorted_dates = sorted(panel["decision_date"].unique())
    n_dates = len(sorted_dates)

    all_trades: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    for i in range(0, n_dates, stride):
        eval_date = sorted_dates[i]

        train_mask = panel["decision_date"] < eval_date
        train_df = panel[train_mask]

        if len(train_df) < min_train_rows:
            continue

        eval_df = panel[panel["decision_date"] == eval_date]
        if eval_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = (train_df["outcome_label"] == train_df["winning_label"]).astype(int)

        X_eval = eval_df[feature_cols]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_eval)

        if proba.ndim == 2 and proba.shape[1] >= 2:
            prob_positive = proba[:, 1]
        elif proba.ndim == 2:
            prob_positive = proba[:, 0]
        else:
            prob_positive = proba

        eval_df = eval_df.copy()
        eval_df["_predicted_prob"] = prob_positive

        if policy is None or policy.selection == "top_prob":
            decisions = _top_prob_decisions(eval_df)
        else:
            decisions = _best_edge_decisions(eval_df, policy, costs, flat_stake)

        for row, direction in decisions:
            price = float(row["market_price"])
            prob = float(row["_predicted_prob"])
            # The token this trade buys: the label's YES token, or its NO token, quoted
            # at the complement. Priced exactly as `PaperBroker` records it.
            token_price = price if direction == "yes" else 1.0 - price
            token_prob = prob if direction == "yes" else 1.0 - prob
            # Fill at the costed price, not the quote: edge measured against the
            # quote overstates every trade by whatever the spread and depth cost.
            fill = costs.fill_price(token_price, stake=flat_stake)
            edge = token_prob - fill
            fee_paid = costs.fee(flat_stake)

            pos = Position(
                outcome_label=str(row["outcome_label"]),
                price=fill,
                size=flat_stake / fill if fill > 0 else 0.0,
                side="buy" if direction == "yes" else "buy_no",
            )
            pnl = settle_position(pos, str(row["winning_label"]), fee_paid=fee_paid)
            yes_edge, no_edge = _informational_edges(prob, price, costs, flat_stake)

            all_trades.append(
                {
                    "market_id": row["market_id"],
                    "eval_date": eval_date,
                    "outcome_label": row["outcome_label"],
                    "direction": direction,
                    "predicted_prob": prob,
                    "market_price": price,
                    "fill_price": fill,
                    "realized_pnl": pnl,
                    "edge": edge,
                    # The after-fee edges of each side and of the one traded: the fields
                    # of the `EdgeSignal` a live scan would have produced for this row.
                    "yes_edge": yes_edge,
                    "no_edge": no_edge,
                    "best_edge": yes_edge if direction == "yes" else no_edge,
                    # Derived/passthrough columns so the trade log composes directly
                    # with compute_metrics ("outcome") and HoldoutGateResult.evaluate
                    # ("segment") — no manual glue required downstream.
                    "outcome": "won" if pnl > 0 else "lost",
                    "segment": str(row["segment"]) if "segment" in row else "all",
                    # Whether the label we forecast actually occurred. `outcome`
                    # above is a *profit* flag; calibration needs the label itself.
                    "label_occurred": bool(row["outcome_label"] == row["winning_label"]),
                }
            )

        steps.append(
            {
                "eval_date": eval_date,
                "train_rows": len(train_df),
                "eval_rows": len(eval_df),
                "trades": len(decisions),
            }
        )

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
    else:
        trades_df = pd.DataFrame(
            columns=[
                "market_id",
                "eval_date",
                "outcome_label",
                "direction",
                "predicted_prob",
                "market_price",
                "fill_price",
                "realized_pnl",
                "edge",
                "yes_edge",
                "no_edge",
                "best_edge",
                "outcome",
                "segment",
                "label_occurred",
            ]
        )

    return RollingOriginResult(trades=trades_df, steps=steps)


Decision = tuple["pd.Series[Any]", str]


def _top_prob_decisions(eval_df: pd.DataFrame) -> list[Decision]:
    """The historical rule: YES on the bin with the highest predicted probability."""
    best_idx = eval_df.groupby(["market_id", "decision_date"])["_predicted_prob"].idxmax()
    best_rows = eval_df.loc[best_idx]
    return [(row, "yes") for _, row in best_rows.iterrows()]


def _best_edge_decisions(
    eval_df: pd.DataFrame, policy: TradePolicy, costs: CostModel, stake: float
) -> list[Decision]:
    """The live rule: per market the (bin, side) with the best after-fee edge, if it clears.

    `quote_edges` is the computation a live `EdgeSignal` uses. A side that cannot be
    bought (priced at 0, or filled at certainty) has an edge of -inf and is never taken.
    Among equal edges the first bin in the panel wins, as the first of equal signals
    would be the one `PaperBroker` records.
    """
    assert policy.min_edge is not None  # best_edge policies always carry a threshold
    decisions: list[Decision] = []
    for _, block in eval_df.groupby(["market_id", "decision_date"], sort=True):
        best: tuple[pd.Series[Any], str, float] | None = None
        for _, row in block.iterrows():
            quote = quote_edges(
                float(row["_predicted_prob"]),
                float(row["market_price"]),
                costs,
                stake,
                policy.directions,
            )
            if best is None or quote.best_edge > best[2]:
                best = (row, quote.direction, quote.best_edge)
        if best is not None and edge_clears(best[2], policy.min_edge):
            decisions.append((best[0], best[1]))
    return decisions


def _informational_edges(
    prob: float, price: float, costs: CostModel, stake: float
) -> tuple[float, float]:
    """(yes_edge, no_edge) of a row, for the trade log; NaN when the forecast is not a number.

    The historical policy never validated the forecast (a NaN just gave a NaN edge), and
    reproducing old backtests means it still must not raise here.
    """
    if not math.isfinite(prob):
        return math.nan, math.nan
    quote = quote_edges(min(max(prob, 0.0), 1.0), price, costs, stake)
    return quote.yes_edge, quote.no_edge
