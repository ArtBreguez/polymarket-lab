"""TDD (RED first): typed panel builder.

Assembles the training panel from plugin rows with schema + dtype validation,
so the KeyError-at-backtest-time friction (missing required columns discovered
only deep in rolling_origin_eval) becomes a clear error at build time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.data import PanelSchemaError, build_panel


def _rows(n: int = 60, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        price = float(rng.uniform(0.05, 0.95))
        rows.append(
            {
                "market_id": f"m{i}",
                "decision_date": f"2025-01-{(i % 28) + 1:02d}",
                "outcome_label": "Yes",
                "winning_label": "Yes" if rng.uniform() > 0.5 else "No",
                "market_price": price,
                "feature_spread": float(rng.uniform(0, 0.1)),
                "feature_vol": float(rng.uniform(1000, 1e6)),
            }
        )
    return rows


class TestBuildPanelHappyPath:
    def test_builds_dataframe_from_rows(self):
        panel = build_panel(_rows())
        assert isinstance(panel, pd.DataFrame)
        assert len(panel) == 60

    def test_feature_columns_are_float(self):
        panel = build_panel(_rows())
        assert panel["feature_spread"].dtype == np.float64
        assert panel["feature_vol"].dtype == np.float64

    def test_drops_none_rows(self):
        rows = _rows(10) + [None, None]  # plugins return None for unbuildable rows
        panel = build_panel(rows)
        assert len(panel) == 10

    def test_accepts_iterable_of_rows(self):
        panel = build_panel(iter(_rows(5)))
        assert len(panel) == 5


class TestBuildPanelValidation:
    def test_missing_required_column_raises(self):
        rows = _rows(5)
        for r in rows:
            del r["market_price"]
        with pytest.raises(PanelSchemaError) as exc:
            build_panel(rows)
        assert "market_price" in str(exc.value)

    def test_no_feature_columns_raises(self):
        rows = [
            {
                "market_id": "m1",
                "decision_date": "2025-01-01",
                "outcome_label": "Yes",
                "winning_label": "Yes",
                "market_price": 0.4,
            }
        ]
        with pytest.raises(PanelSchemaError):
            build_panel(rows)

    def test_empty_rows_raises(self):
        with pytest.raises(PanelSchemaError):
            build_panel([])

    def test_bad_decision_date_format_raises(self):
        rows = _rows(5)
        rows[0]["decision_date"] = "01/01/2025"  # wrong format
        with pytest.raises(PanelSchemaError):
            build_panel(rows)

    def test_non_numeric_feature_raises(self):
        rows = _rows(5)
        rows[0]["feature_spread"] = "wide"
        with pytest.raises(PanelSchemaError):
            build_panel(rows)

    def test_all_nan_market_price_raises(self):
        # market_price is required and drives PnL/edge; an all-NaN column must
        # fail loud at build time, not produce NaN trades deep in the backtest.
        rows = _rows(5)
        for r in rows:
            r["market_price"] = None
        with pytest.raises(PanelSchemaError) as exc:
            build_panel(rows)
        assert "market_price" in str(exc.value)

    def test_non_numeric_market_price_raises(self):
        rows = _rows(5)
        rows[0]["market_price"] = "cheap"  # coerces to NaN silently otherwise
        with pytest.raises(PanelSchemaError) as exc:
            build_panel(rows)
        assert "market_price" in str(exc.value)


class TestBuildPanelLeakageIntegration:
    def test_leakage_check_runs_when_requested(self):
        rows = _rows(200)
        for r in rows:  # inject a leaking feature
            r["feature_leak"] = 1.0 if r["winning_label"] == "Yes" else 0.0
        from pmlab.data import LeakageError

        with pytest.raises(LeakageError):
            build_panel(rows, check_leakage=True)

    def test_leakage_check_off_by_default(self):
        rows = _rows(200)
        for r in rows:
            r["feature_leak"] = 1.0 if r["winning_label"] == "Yes" else 0.0
        panel = build_panel(rows)  # default: no leakage check, must not raise
        assert "feature_leak" in panel.columns
