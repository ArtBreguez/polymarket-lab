"""Walk-forward (rolling origin) evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pmlab.core.pnl import Position, settle_position


@dataclass
class RollingOriginResult:
    trades: pd.DataFrame  # columns: market_id, eval_date, outcome_label, predicted_prob, market_price, realized_pnl, edge
    steps: list[dict] = field(default_factory=list)


def rolling_origin_eval(
    panel: pd.DataFrame,
    model,  # has fit(X, y) and predict_proba(X) -> ndarray
    min_train_rows: int = 20,
    stride: int = 10,
    flat_stake: float = 1.0,
    taker_bps: float = 30.0,
) -> RollingOriginResult:
    """Walk-forward evaluation on a panel dataset.

    panel columns required:
        market_id, decision_date (str YYYY-MM-DD),
        outcome_label, winning_label, market_price,
        + feature_* columns

    For each eval step:
        - Train on rows with decision_date < eval_date
        - Predict on rows at eval_date
        - Select best bin per (market_id, decision_date) by max predicted_prob
        - Compute PnL using settle_position

    Returns:
        RollingOriginResult with all trade records and step metadata.
    """
    panel = panel.copy()
    panel["decision_date"] = panel["decision_date"].astype(str)

    feature_cols = [c for c in panel.columns if c.startswith("feature_")]

    sorted_dates = sorted(panel["decision_date"].unique())
    n_dates = len(sorted_dates)

    all_trades: list[dict] = []
    steps: list[dict] = []

    # Walk-forward: iterate in stride steps starting after min_train_rows worth of dates
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

        # Handle both binary (shape N,2) and single-column output
        if proba.ndim == 2 and proba.shape[1] >= 2:
            prob_positive = proba[:, 1]
        elif proba.ndim == 2:
            prob_positive = proba[:, 0]
        else:
            prob_positive = proba

        eval_df = eval_df.copy()
        eval_df["_predicted_prob"] = prob_positive

        # Select best bin per (market_id, decision_date) by max prob
        best_idx = eval_df.groupby(["market_id", "decision_date"])["_predicted_prob"].idxmax()
        best_rows = eval_df.loc[best_idx]

        fee_rate = taker_bps / 10_000.0

        for _, row in best_rows.iterrows():
            price = float(row["market_price"])
            prob = float(row["_predicted_prob"])
            edge = prob - price
            fee_paid = flat_stake * fee_rate

            pos = Position(
                outcome_label=str(row["outcome_label"]),
                price=price,
                size=flat_stake / price if price > 0 else 0.0,
                side="buy",
            )
            pnl = settle_position(pos, str(row["winning_label"]), fee_paid=fee_paid)

            all_trades.append(
                {
                    "market_id": row["market_id"],
                    "eval_date": eval_date,
                    "outcome_label": row["outcome_label"],
                    "predicted_prob": prob,
                    "market_price": price,
                    "realized_pnl": pnl,
                    "edge": edge,
                }
            )

        steps.append(
            {
                "eval_date": eval_date,
                "train_rows": len(train_df),
                "eval_rows": len(eval_df),
                "trades": len(best_rows),
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
                "predicted_prob",
                "market_price",
                "realized_pnl",
                "edge",
            ]
        )

    return RollingOriginResult(trades=trades_df, steps=steps)
