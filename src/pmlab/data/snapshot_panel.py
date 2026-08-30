"""No-lookahead panel assembly from point-in-time snapshots + realized truth.

Closes the training loop: given a :class:`FeatureSnapshotStore` of open-market
feature snapshots captured over time, and a table of market resolutions
(``market_id``, ``decision_date``, ``winning_label``), build a training panel
where each row uses the latest snapshot captured *strictly before* that market's
decision date. No feature can originate after the decision — no lookahead by
construction.
"""

from __future__ import annotations

import pandas as pd

from pmlab.data.snapshot_store import FeatureSnapshotStore

__all__ = ["build_panel_from_snapshots"]

_RES_REQUIRED = ["market_id", "decision_date", "winning_label"]


def build_panel_from_snapshots(
    store: FeatureSnapshotStore,
    family: str,
    resolutions: pd.DataFrame,
    outcome_label: str = "Yes",
    price_col: str | None = None,
) -> pd.DataFrame:
    """Assemble a no-lookahead training panel from snapshots + resolutions.

    Args:
        store: The snapshot store to read point-in-time features from.
        family: Market family to read.
        resolutions: DataFrame with ``market_id``, ``decision_date`` (YYYY-MM-DD)
            and ``winning_label`` — one row per resolved market.
        outcome_label: Value for the panel's ``outcome_label`` column (the bin the
            row's features/price describe). Defaults to ``"Yes"``.
        price_col: Feature column to copy into ``market_price``. If ``None`` (the
            default), the first ``feature_*`` column is used; ``market_price`` must
            be numeric so the panel is backtestable.

    Returns:
        A training panel (one row per resolvable market) with ``market_id``,
        ``decision_date``, ``outcome_label``, ``winning_label``, ``market_price``
        and the snapshot's ``feature_*`` columns. Markets with no snapshot before
        their decision date, or absent from the store, are dropped.

    Raises:
        ValueError: if ``resolutions`` is missing a required column.
    """
    missing = [c for c in _RES_REQUIRED if c not in resolutions.columns]
    if missing:
        raise ValueError(f"resolutions is missing required column(s): {missing}")

    if resolutions.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for res in resolutions.itertuples(index=False):
        market_id = res.market_id
        decision_date = str(res.decision_date)
        winning_label = res.winning_label

        # as_of gives the latest snapshot per market strictly before the cutoff.
        as_of = store.as_of(family, decision_date)
        if as_of.empty:
            continue
        match = as_of[as_of["market_id"] == market_id]
        if match.empty:
            continue

        snap: dict[str, object] = {str(k): v for k, v in match.iloc[0].to_dict().items()}
        feature_cols = [c for c in snap if c.startswith("feature_")]
        if not feature_cols:
            continue

        chosen_price = price_col or feature_cols[0]
        row: dict[str, object] = {
            "market_id": market_id,
            "decision_date": decision_date,
            "outcome_label": outcome_label,
            "winning_label": winning_label,
            "market_price": float(snap[chosen_price]),  # type: ignore[arg-type]
        }
        for c in feature_cols:
            row[c] = snap[c]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)
