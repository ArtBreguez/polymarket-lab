"""Walk-forward (rolling origin) evaluation."""

from __future__ import annotations

import math
import numbers
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
    *,
    embargo_days: int | None = None,
) -> RollingOriginResult:
    """Walk-forward evaluation on a panel dataset.

    panel columns required:
        market_id, decision_date (str YYYY-MM-DD),
        outcome_label, winning_label, market_price,
        + feature_* columns

    For each eval step:
        - Train on the rows decided before ``eval_date`` **whose label was already known when
          the eval decision was taken** (see below)
        - Predict on rows at eval_date
        - Decide the trade of each (market_id, decision_date) under ``policy``
        - Compute PnL using settle_position

    ``market_price`` is the label's YES price **at the row's decision_date** (the plugin
    contract, see ``MarketPlugin.build_training_row``). A decision uses only its own row's
    price and forecast, so nothing later than the decision date reaches it.

    **Labels are known later than decisions.** A decision on day D can ask about D+1 (the
    ``weather_tmax`` ``previous_evening`` horizon), so the row decided on the day before an
    eval date carries the eval day's own outcome. Training on it is lookahead. A row therefore
    trains a fold only when its label was known at that fold's decision:

    - ``market_close_ts`` (the plugin writes it on every row: the instant the market ends, which
      is when the day it asks about is over) must be at or before the fold's decision time, the
      earliest ``decision_ts`` of the fold's rows (the start of ``eval_date`` UTC when the panel
      has no ``decision_ts``);
    - a row with no usable ``market_close_ts``, or a panel without the column, is assumed to
      resolve the day after its decision, so it trains a fold only from two days later
      (``decision_date + 1 day < eval_date``). If your labels arrive later than that, put the
      time the label became known in ``market_close_ts``.

    The rule trusts ``market_close_ts`` to be no earlier than the moment the label can be computed.
    The weather plugin takes the earlier of the market's own close and the end of its target day,
    so a market that closes at noon on its target day counts as known from noon although the
    day's maximum is final only at midnight: that matters to a fold that decides later that same
    day (never to one that decides in the morning, as the sample's does).

    A fold therefore starts one decision day later. Backtests run before this rule (release
    1.0.2 and earlier) carried a day of lookahead.

    **A longer hold-back, on request.** ``embargo_days=N`` adds a second condition to the rule
    above: a row trains a fold only if its ``decision_date`` is at least ``N`` days before the
    fold's. Use it when you want a margin the labels do not give you (a model that must not see
    the last week, say). The two conditions are AND-ed: a row must have its label known *and* be at
    least ``N`` days old, so the embargo can only take rows out of a fold, never put one in. It
    cannot make a label that is not yet known trainable, whatever ``N``, and ``0`` or ``1`` change
    nothing (a fold already trains only on rows decided on an earlier day). ``None``, the default,
    is the rule above exactly: the results of release 1.0.3 are reproduced to the last bit.

    Args:
        panel: Training/eval panel (see column contract above).
        model: Anything with sklearn-style ``fit`` / ``predict_proba``.
        min_train_rows: Skip an eval date until this many training rows exist.
        stride: Evaluate every Nth distinct decision_date.
        flat_stake: Notional USDC risked per trade.
        taker_bps: Legacy fee-only knob. Mutually exclusive with ``costs``.
        costs: Full `CostModel` (fee + slippage + depth).
            Omit both arguments for the historical default (30bps, fill at quote).
        policy: The decision rule. Omit it for the historical backtest: YES on the bin
            with the highest predicted probability, every market, no threshold (the
            numbers published before policies existed reproduce). Pass
            ``TradePolicy.live(min_edge=...)`` to trade what ``record-trades`` trades:
            per market the (bin, side) with the best after-fee edge, YES or NO, taken if
            that edge is at least ``min_edge``. Both price the trade the way the broker
            and settlement do: a NO trade is a buy of the NO token at ``1 - market_price``.
        embargo_days: Keyword-only. ``None`` (default) trains on every row the label rule above
            allows. An int ``N >= 0`` also requires the row's ``decision_date`` to be at least
            ``N`` days before the fold's, so folds can only get smaller (``min_train_rows`` counts
            what is left, so the first fold may come later). A negative ``N`` raises
            ``ValueError``; a value that is not an int (a float, a string, a bool) raises
            ``TypeError``.

    Returns:
        RollingOriginResult with all trade records and step metadata.

    Raises:
        ValueError: If both ``taker_bps`` and ``costs`` are supplied, which would
            give the fee two conflicting sources of truth. Also if a ``decision_date`` is not a
            real ``YYYY-MM-DD`` date (e.g. ``2026-01-32``), since which labels were known
            when a fold decided cannot be told without it. Also if ``embargo_days`` is negative.
        TypeError: If ``embargo_days`` is neither ``None`` nor an int (``True`` is not one).
    """
    embargo = _validate_embargo_days(embargo_days)
    if taker_bps is not None and costs is not None:
        raise ValueError(
            "pass either taker_bps or costs, not both — costs.taker_bps already carries the fee"
        )
    if costs is None:
        costs = CostModel() if taker_bps is None else CostModel(taker_bps=taker_bps)

    # A clean index: `_top_prob_decisions` reads rows back by label, and a panel stacked
    # market by market repeats labels (K rows each starting at 0) and would match them all.
    panel = panel.reset_index(drop=True)
    panel["decision_date"] = panel["decision_date"].astype(str)

    feature_cols = [c for c in panel.columns if c.startswith("feature_")]

    sorted_dates = sorted(panel["decision_date"].unique())
    n_dates = len(sorted_dates)

    day_start = _day_starts(panel["decision_date"])
    label_known_at = _label_known_at(panel, day_start)
    decision_cutoffs = _decision_cutoffs(panel, day_start)
    fold_day_start = day_start.groupby(panel["decision_date"]).first()

    all_trades: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    for i in range(0, n_dates, stride):
        eval_date = sorted_dates[i]

        # Only rows whose label existed when this fold's decision was taken may train it.
        train_mask = (panel["decision_date"] < eval_date) & (
            label_known_at <= decision_cutoffs[eval_date]
        )
        if embargo is not None:
            # On top of the label rule, never instead of it: the masks are AND-ed, so a row whose
            # label is unknown stays out whatever the embargo, and the embargo only removes rows.
            age_days = (fold_day_start[eval_date] - day_start).dt.days
            train_mask &= age_days >= min(embargo, _MAX_EMBARGO_DAYS)
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


#: Decision days a label is assumed to lag its decision when the panel does not say: the next
#: day's value, as for every shipped plugin. A row decided on D then trains from D + 2 days.
_DEFAULT_LABEL_LAG_DAYS = 1


#: No decision date is this many days before another (pandas' timestamps span about 2.6e5 days),
#: so a larger ``embargo_days`` means "train on nothing" all the same; clamped for the comparison.
_MAX_EMBARGO_DAYS = 10**9


def _validate_embargo_days(embargo_days: object) -> int | None:
    """``None`` or a non-negative whole number of days; ``TypeError`` / ``ValueError`` otherwise.

    A bool is refused although Python counts it as an int: ``embargo_days=True`` is far more likely
    a flag mistaken for a number of days than a one-day embargo. A float is refused too, even
    ``2.0``: days are whole, and silently truncating ``2.5`` would hold back less than asked.
    """
    if embargo_days is None:
        return None
    if isinstance(embargo_days, bool) or not isinstance(embargo_days, numbers.Integral):
        raise TypeError(
            f"embargo_days must be a whole number of days (an int) or None; got {embargo_days!r}"
        )
    days = int(embargo_days)
    if days < 0:
        raise ValueError(f"embargo_days must be >= 0 (or None for no embargo); got {days}")
    return days


def _utc_times(values: pd.Series) -> pd.Series:
    """ISO-8601 text (or datetimes) as UTC timestamps; anything unreadable or missing is NaT."""
    return pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")


def _day_starts(decision_date: pd.Series) -> pd.Series:
    """Each ``decision_date`` as the start of that UTC day; a value that is no date is an error.

    The hold-back is measured in time, so a decision date must be a real calendar date. Silently
    skipping the rows whose date cannot be read would shrink the training set without a word.
    """
    days = pd.to_datetime(decision_date, utc=True, errors="coerce", format="ISO8601")
    unreadable = decision_date[days.isna()]
    if not unreadable.empty:
        raise ValueError(
            "decision_date must be a 'YYYY-MM-DD' calendar date, because a fold may train only "
            f"on labels known when it decides; got e.g. {unreadable.iloc[0]!r}."
        )
    return days


def _label_known_at(panel: pd.DataFrame, day_start: pd.Series) -> pd.Series:
    """When each row's label was known: its ``market_close_ts``, else the end of the next day.

    ``market_close_ts`` is the instant the market ends, which for a market on a day's value is
    the end of that day. A row without a readable one is assumed to resolve the day after its
    decision (`_DEFAULT_LABEL_LAG_DAYS`), whose end is the midnight two days after that date.
    """
    fallback = day_start + pd.Timedelta(days=_DEFAULT_LABEL_LAG_DAYS + 1)
    if "market_close_ts" not in panel.columns:
        return fallback
    return _utc_times(panel["market_close_ts"]).fillna(fallback)


def _decision_cutoffs(panel: pd.DataFrame, day_start: pd.Series) -> pd.Series:
    """Per ``decision_date``: the instant its decision was taken, the latest a label may be known.

    The earliest ``decision_ts`` of the day's rows (several horizons may decide on one day).
    A day with none readable is cut at its start, the safe side: only labels known at the
    start of the day count.
    """
    cutoffs = day_start.groupby(panel["decision_date"]).first()
    if "decision_ts" in panel.columns:
        decided = _utc_times(panel["decision_ts"]).groupby(panel["decision_date"]).min()
        cutoffs = decided.reindex(cutoffs.index).fillna(cutoffs)
    return cutoffs


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
    # -inf (a side that cannot be bought) would not survive strict JSON; as on an EdgeSignal,
    # it reads as losing the whole stake.
    return max(quote.yes_edge, -1.0), max(quote.no_edge, -1.0)
