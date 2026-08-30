"""Typed panel builder.

Assembles a validated training panel from plugin rows, enforcing the schema
``rolling_origin_eval`` requires (``market_id``, ``decision_date`` as
YYYY-MM-DD, ``outcome_label``, ``winning_label``, ``market_price`` and at least
one ``feature_*`` column) and coercing feature columns to float. Turns the
KeyError-deep-in-the-backtest failure mode into a clear error at build time, and
can optionally run the leakage guard before returning.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from pmlab.data.leakage import check_no_leakage

__all__ = ["PanelSchemaError", "build_panel"]

_REQUIRED = ["market_id", "decision_date", "outcome_label", "winning_label", "market_price"]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class PanelSchemaError(Exception):
    """Raised when plugin rows do not form a valid training panel."""


def build_panel(
    rows: Iterable[dict[str, Any] | None],
    check_leakage: bool = False,
    max_auc: float = 0.97,
) -> pd.DataFrame:
    """Build a validated training panel from plugin rows.

    Args:
        rows: Iterable of row dicts (as returned by ``MarketPlugin.build_training_row``).
            ``None`` entries are dropped, so callers can pass unfiltered output.
        check_leakage: If True, run :func:`check_no_leakage` on the assembled
            panel and raise ``LeakageError`` if any feature leaks the label.
        max_auc: Threshold forwarded to the leakage check.

    Returns:
        A schema-validated panel DataFrame with float feature columns.

    Raises:
        PanelSchemaError: on empty input, missing required columns, no
            ``feature_*`` column, malformed ``decision_date``, or a feature that
            cannot be coerced to numeric.
        LeakageError: if ``check_leakage`` is True and leakage is detected.
    """
    materialized = [r for r in rows if r is not None]
    if not materialized:
        raise PanelSchemaError("No rows to build a panel from (empty after dropping None).")

    df = pd.DataFrame(materialized)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise PanelSchemaError(
            f"Panel is missing required column(s): {missing}. "
            f"Required: {_REQUIRED} + at least one 'feature_*' column."
        )

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        raise PanelSchemaError(
            "Panel has no 'feature_*' columns — a model has nothing to learn from."
        )

    bad_dates = df.loc[~df["decision_date"].astype(str).str.match(_DATE_RE), "decision_date"]
    if not bad_dates.empty:
        raise PanelSchemaError(
            f"decision_date must be 'YYYY-MM-DD'; got e.g. {bad_dates.iloc[0]!r}."
        )

    for col in feature_cols:
        original = df[col]
        coerced = pd.to_numeric(original, errors="coerce")
        newly_nan = coerced.isna() & original.notna()
        if bool(newly_nan.any()):
            offending = original[newly_nan].iloc[0]
            raise PanelSchemaError(
                f"Feature column '{col}' is not numeric; got e.g. {offending!r}."
            )
        df[col] = coerced.astype(float)

    df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce").astype(float)

    if check_leakage:
        check_no_leakage(df, max_auc=max_auc, raise_on_leak=True)

    return df.reset_index(drop=True)
