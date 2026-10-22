"""Typed panel builder.

Assembles a validated training panel from plugin rows, enforcing the schema
``rolling_origin_eval`` requires (``market_id``, ``decision_date`` as
YYYY-MM-DD, ``outcome_label``, ``winning_label``, ``market_price`` and at least
one ``feature_*`` column) and coercing feature columns to float. Turns the
KeyError-deep-in-the-backtest failure mode into a clear error at build time, and
can optionally run the leakage guard before returning.

A market is described by one row **per outcome bin** (the same ``market_id`` and
``decision_date``, a different ``outcome_label`` and ``market_price``): the backtest
chooses between a market's bins, and a panel that shows it only the bin that won cannot
tell a good decision from a lucky one. ``MarketPlugin.build_training_rows`` returns
exactly those rows, and :func:`build_panel` takes them as they come.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from pmlab.data.leakage import check_no_leakage, check_price_timing

__all__ = ["PanelSchemaError", "build_panel"]

_logger = logging.getLogger(__name__)

_REQUIRED = ["market_id", "decision_date", "outcome_label", "winning_label", "market_price"]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class PanelSchemaError(Exception):
    """Raised when plugin rows do not form a valid training panel."""


def _flatten(rows: Iterable[dict[str, Any] | list[dict[str, Any]] | None]) -> list[dict[str, Any]]:
    """Rows in order: a dict is one row, a list is a market's rows, ``None`` is nothing."""
    flat: list[dict[str, Any]] = []
    for r in rows:
        if r is None:
            continue
        if isinstance(r, dict):
            flat.append(r)
        else:
            flat.extend(x for x in r if x is not None)
    return flat


def build_panel(
    rows: Iterable[dict[str, Any] | list[dict[str, Any]] | None],
    check_leakage: bool = False,
    max_auc: float = 0.97,
) -> pd.DataFrame:
    """Build a validated training panel from plugin rows.

    Args:
        rows: Iterable of row dicts, or of lists of row dicts, so the output of either
            ``MarketPlugin.build_training_row`` (one row or ``None``) or
            ``MarketPlugin.build_training_rows`` (a market's rows) can be passed per market
            unchanged. ``None`` entries and empty lists are dropped.
        check_leakage: If True, run :func:`check_no_leakage` on the assembled
            panel and raise ``LeakageError`` if any feature leaks the label.
        max_auc: Threshold forwarded to the leakage check.

    A panel whose rows carry neither ``price_ts`` nor ``decision_ts`` is built, but logs a
    single WARNING (logger ``pmlab.data.panel_builder``, once per call): its ``market_price``
    cannot be checked against the decision it informs. That is the case for every plugin
    written before the check existed, for the ``sports_f1`` plugin, and for
    :func:`build_panel_from_snapshots`. Silence it by stamping the rows, or filter the logger.

    Returns:
        A schema-validated panel DataFrame with float feature columns.

    Raises:
        PanelSchemaError: on empty input, missing required columns, no
            ``feature_*`` column, malformed ``decision_date``, or a feature that
            cannot be coerced to numeric.
        LeakageError: if ``check_leakage`` is True and leakage is detected, or, always, if
            the rows carry ``price_ts`` and ``decision_ts`` and a price was observed at or
            after its decision, if they carry ``market_close_ts`` and a decision was taken
            at or after it, or if they carry only one of ``price_ts`` and ``decision_ts``
            (see :func:`check_price_timing`).
    """
    materialized = _flatten(rows)
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

    price_raw = df["market_price"]
    price = pd.to_numeric(price_raw, errors="coerce")
    newly_nan_price = price.isna() & price_raw.notna()
    if bool(newly_nan_price.any()):
        offending = price_raw[newly_nan_price].iloc[0]
        raise PanelSchemaError(f"Column 'market_price' is not numeric; got e.g. {offending!r}.")
    if bool(price.isna().all()):
        raise PanelSchemaError(
            "Column 'market_price' is entirely missing/NaN — it drives PnL and edge; "
            "a panel cannot be backtested without it."
        )
    df["market_price"] = price.astype(float)

    # Not behind `check_leakage`: it costs nothing, fires only on plugins that stamp their
    # prices, and a post-decision price is a defect rather than a judgement call.
    timing = check_price_timing(df)
    if timing.n_checked == 0:
        # `market_price` drives every trade the backtest takes; with no price_ts/decision_ts
        # nothing says it was known by then. One warning per panel, through logging: a CLI
        # that never builds a panel (pmlab monitor) cannot reach it, and stdout stays clean.
        _logger.warning(
            "build_panel: %d rows carry a market_price but no price_ts/decision_ts, so "
            "nothing checks that each price was observed before the decision it informs: "
            "lookahead in the prices would go undetected. Stamp every row with price_ts and "
            "decision_ts (MarketPlugin.build_training_rows does). Panels from "
            "build_panel_from_snapshots and the sports_f1 plugin are untimed.",
            len(df),
        )

    if check_leakage:
        check_no_leakage(df, max_auc=max_auc, raise_on_leak=True)

    return df.reset_index(drop=True)
