"""Tests for feature engineering transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.features.transforms import (
    add_lags,
    add_rolling_stats,
    clip_outliers,
    encode_cyclical,
    encode_onehot,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "city": ["A", "A", "A", "B", "B", "B"],
            "temp": [20.0, 22.0, 24.0, 15.0, 17.0, 19.0],
            "humidity": [50.0, 55.0, 60.0, 40.0, 45.0, 50.0],
        }
    )


class TestAddLags:
    def test_basic_lag(self, sample_df):
        out = add_lags(sample_df, ["temp"], [1])
        assert "temp_lag1" in out.columns
        assert out["temp_lag1"].iloc[1] == pytest.approx(20.0)

    def test_multiple_lags(self, sample_df):
        out = add_lags(sample_df, ["temp"], [1, 2])
        assert "temp_lag1" in out.columns and "temp_lag2" in out.columns

    def test_grouped_lag(self, sample_df):
        out = add_lags(sample_df, ["temp"], [1], group_by="city")
        assert np.isnan(out["temp_lag1"].iloc[0])
        assert np.isnan(out["temp_lag1"].iloc[3])
        assert out["temp_lag1"].iloc[1] == pytest.approx(20.0)

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError, match="nonexistent"):
            add_lags(sample_df, ["nonexistent"], [1])

    def test_no_mutation(self, sample_df):
        original_cols = list(sample_df.columns)
        add_lags(sample_df, ["temp"], [1])
        assert list(sample_df.columns) == original_cols


class TestAddRollingStats:
    def test_rolling_mean(self, sample_df):
        out = add_rolling_stats(sample_df, ["temp"], [2], stats=["mean"])
        assert "temp_roll2_mean" in out.columns

    def test_default_stats(self, sample_df):
        out = add_rolling_stats(sample_df, ["temp"], [2])
        assert "temp_roll2_mean" in out.columns and "temp_roll2_std" in out.columns

    def test_grouped_rolling(self, sample_df):
        out = add_rolling_stats(sample_df, ["temp"], [2], stats=["mean"], group_by="city")
        assert "temp_roll2_mean" in out.columns

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            add_rolling_stats(sample_df, ["bad_col"], [2])


class TestEncodeCyclical:
    def test_hour_encoding(self):
        hours = pd.Series(range(24), name="hour")
        sin_s, cos_s = encode_cyclical(hours, period=24)
        assert sin_s.name == "hour_sin" and cos_s.name == "hour_cos"
        assert len(sin_s) == 24
        assert sin_s.iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert cos_s.iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_unnamed_series(self):
        s = pd.Series([0, 6, 12, 18])
        sin_s, _ = encode_cyclical(s, 24)
        assert sin_s.name == "x_sin"


class TestEncodeOnehot:
    def test_basic(self, sample_df):
        out = encode_onehot(sample_df, ["city"])
        assert "city" not in out.columns
        dummy_cols = [c for c in out.columns if c.startswith("city_")]
        assert len(dummy_cols) == 1

    def test_no_drop_first(self, sample_df):
        out = encode_onehot(sample_df, ["city"], drop_first=False)
        dummy_cols = [c for c in out.columns if c.startswith("city_")]
        assert len(dummy_cols) == 2

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            encode_onehot(sample_df, ["nonexistent"])


class TestClipOutliers:
    def test_iqr_clips(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0, -100.0]})
        out = clip_outliers(df, ["x"], method="iqr", iqr_factor=1.5)
        assert out["x"].max() < 100.0
        assert out["x"].min() > -100.0

    def test_zscore_clips(self):
        df = pd.DataFrame({"x": [0.0] * 10 + [1000.0]})
        out = clip_outliers(df, ["x"], method="zscore", zscore_threshold=2.0)
        assert out["x"].max() < 1000.0

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(KeyError):
            clip_outliers(df, ["b"])

    def test_no_mutation(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 1000.0]})
        clip_outliers(df, ["x"])
        assert df["x"].max() == 1000.0
