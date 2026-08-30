"""TDD (RED first): leakage guards.

The real-data dogfood reported AUC 1.000 / Brier 0.000 without a single warning,
because a feature column was derived from the label. These guards make that
class of mistake loud instead of silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.data import LeakageError, check_no_leakage


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = rng.uniform(0.05, 0.95, n)
    return pd.DataFrame(
        {
            "market_id": [f"m{i}" for i in range(n)],
            "decision_date": ["2025-01-01"] * n,
            "outcome_label": ["Yes"] * n,
            "winning_label": np.where(price > 0.5, "Yes", "No"),
            "market_price": price,
            "feature_price": price,
            "feature_noise": rng.normal(0, 1, n),
        }
    )


class TestLeakageDetection:
    def test_feature_perfectly_predicting_label_raises(self):
        # feature_price → winning_label is deterministic (price>0.5). That is the
        # exact trap from the dogfood; it must be flagged.
        df = _frame()
        with pytest.raises(LeakageError) as exc:
            check_no_leakage(df)
        assert "feature_price" in str(exc.value)

    def test_clean_features_pass(self):
        df = _frame()
        clean = df.drop(columns=["feature_price"])
        # feature_noise is independent of the label → no leakage.
        check_no_leakage(clean)  # must not raise

    def test_returns_report_when_not_raising(self):
        df = _frame().drop(columns=["feature_price"])
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_noise" in report.scores
        assert report.leaked == []

    def test_report_lists_leaked_columns(self):
        df = _frame()
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_price" in report.leaked

    def test_threshold_is_configurable(self):
        # feature_noise separates the label only weakly. A strict default might
        # theoretically flag borderline cases; a lax threshold never does.
        df = _frame().drop(columns=["feature_price"])
        report = check_no_leakage(df, raise_on_leak=False, max_auc=0.999)
        assert report.leaked == []
        # and the perfect separator is always caught, even at the lax threshold
        leaky = _frame()
        report2 = check_no_leakage(leaky, raise_on_leak=False, max_auc=0.999)
        assert "feature_price" in report2.leaked


class TestLeakageEdgeCases:
    def test_no_feature_columns_is_noop(self):
        df = pd.DataFrame({"winning_label": ["Yes", "No"], "market_price": [0.4, 0.6]})
        check_no_leakage(df)  # nothing to check → passes

    def test_single_class_label_is_noop(self):
        # AUC undefined with one class; must not crash.
        df = _frame()
        df["winning_label"] = "Yes"
        check_no_leakage(df.drop(columns=["feature_price"]))

    def test_missing_label_column_raises_valueerror(self):
        df = pd.DataFrame({"feature_x": [1.0, 2.0]})
        with pytest.raises(ValueError):
            check_no_leakage(df)

    def test_constant_feature_is_skipped(self):
        # a feature with <2 distinct usable values can't be AUC-scored; skip it.
        df = _frame().drop(columns=["feature_price"])
        df["feature_const"] = 1.0
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_const" not in report.scores
