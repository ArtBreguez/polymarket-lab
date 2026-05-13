"""Tests for backtest.rolling_origin."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from pmlab.backtest.rolling_origin import RollingOriginResult, rolling_origin_eval


def _make_panel(n_dates: int = 50, n_markets_per_date: int = 5) -> pd.DataFrame:
    """Create synthetic panel data for testing."""
    from datetime import date, timedelta

    base_date = date(2024, 1, 1)
    rows = []
    rng = np.random.default_rng(42)

    for i in range(n_dates):
        d = (base_date + timedelta(days=i)).isoformat()
        for m in range(n_markets_per_date):
            market_id = f"market_{m}"
            winning_label = "YES"
            rows.append(
                {
                    "market_id": market_id,
                    "decision_date": d,
                    "outcome_label": "YES",
                    "winning_label": winning_label,
                    "market_price": 0.5,
                    "feature_x1": rng.random(),
                    "feature_x2": rng.random(),
                }
            )

    return pd.DataFrame(rows)


def test_produces_trades():
    panel = _make_panel(n_dates=50)
    model = DummyClassifier(strategy="most_frequent")
    result = rolling_origin_eval(panel, model, min_train_rows=20, stride=10)

    assert isinstance(result, RollingOriginResult)
    assert not result.trades.empty
    expected_cols = {
        "market_id",
        "eval_date",
        "outcome_label",
        "predicted_prob",
        "market_price",
        "realized_pnl",
        "edge",
    }
    assert expected_cols.issubset(set(result.trades.columns))


def test_no_lookahead():
    """Ensure training data never includes rows from the eval date or future."""
    panel = _make_panel(n_dates=50)

    class LookaheadDetector:
        """Records the max training date seen at each step."""

        def __init__(self):
            self.violations = []
            self._dates_seen = []

        def fit(self, X, y):
            # X doesn't contain dates; we track via a side channel
            pass

        def predict_proba(self, X):
            # Return uniform probabilities
            return np.full((len(X), 2), 0.5)

    detector = LookaheadDetector()
    # We verify by checking result.steps: train_rows < total rows at eval_date
    result = rolling_origin_eval(panel, detector, min_train_rows=20, stride=10)
    # Each step should have train_rows that come strictly before eval_date
    for step in result.steps:
        eval_date = step["eval_date"]
        n_before = (panel["decision_date"] < eval_date).sum()
        assert step["train_rows"] == n_before, (
            f"Lookahead detected at {eval_date}: "
            f"train_rows={step['train_rows']} vs expected={n_before}"
        )


def test_respects_min_train():
    """Steps with fewer than min_train_rows training samples must be skipped."""
    panel = _make_panel(n_dates=50)
    model = DummyClassifier(strategy="most_frequent")

    # With min_train_rows=200, no step should ever run (panel has 250 total rows
    # but early dates have few training rows)
    result_high = rolling_origin_eval(panel, model, min_train_rows=200, stride=5)
    for step in result_high.steps:
        assert step["train_rows"] >= 200


def test_stride_controls_steps():
    """Larger stride means fewer evaluation steps."""
    panel = _make_panel(n_dates=50)
    model = DummyClassifier(strategy="most_frequent")

    result_stride5 = rolling_origin_eval(panel, model, min_train_rows=20, stride=5)
    result_stride10 = rolling_origin_eval(panel, model, min_train_rows=20, stride=10)
    result_stride20 = rolling_origin_eval(panel, model, min_train_rows=20, stride=20)

    assert len(result_stride5.steps) >= len(result_stride10.steps)
    assert len(result_stride10.steps) >= len(result_stride20.steps)


def test_empty_result_when_no_steps_qualify():
    """Line 130: all_trades empty → empty DataFrame returned with correct columns."""
    # Panel with only 5 rows — min_train_rows=100 means no step ever qualifies
    panel = _make_panel(n_dates=3, n_markets_per_date=2)
    model = DummyClassifier(strategy="most_frequent")
    result = rolling_origin_eval(panel, model, min_train_rows=100, stride=1)
    assert result.trades.empty
    # Must still have required columns even when empty
    required = {
        "market_id",
        "eval_date",
        "outcome_label",
        "predicted_prob",
        "market_price",
        "realized_pnl",
        "edge",
    }
    assert required.issubset(set(result.trades.columns))
    assert result.steps == []


def test_single_class_proba_branch():
    """Line 81: proba shape (N,1) — model returns only one class probability."""
    from sklearn.dummy import DummyClassifier

    panel = _make_panel(n_dates=30)
    # prior= forces DummyClassifier to output shape (N,1) when only one class seen
    model = DummyClassifier(strategy="prior")
    # Should not crash even with unusual proba shape
    result = rolling_origin_eval(panel, model, min_train_rows=10, stride=5)
    assert isinstance(result.trades, pd.DataFrame)
