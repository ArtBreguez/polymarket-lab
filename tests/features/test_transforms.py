"""Tests for feature engineering transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.features.transforms import (
    add_interactions,
    add_lags,
    add_missing_indicators,
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

    def test_grouped_rolling_values_when_groups_are_interleaved(self):
        # Rows are NOT sorted by the group key: a/b/a/b/a/b. The rolling window of each row
        # must be computed over its own group, and the value must land on that same row.
        df = pd.DataFrame(
            {
                "city": ["a", "b", "a", "b", "a", "b"],
                "temp": [1.0, 100.0, 3.0, 300.0, 5.0, 500.0],
            }
        )
        out = add_rolling_stats(df, ["temp"], [2], stats=["mean"], group_by="city")
        expected = [1.0, 100.0, 2.0, 200.0, 4.0, 400.0]
        assert out["temp_roll2_mean"].tolist() == pytest.approx(expected)

    def test_grouped_rolling_is_independent_of_row_order_of_groups(self):
        # Sorting by group key first must give the same per-row values as the interleaved frame.
        df = pd.DataFrame(
            {
                "city": ["b", "a", "c", "a", "b", "c", "a", "c", "b"],
                "temp": [10.0, 1.0, 7.0, 3.0, 20.0, 9.0, 5.0, 11.0, 40.0],
            },
            index=[10, 11, 12, 13, 14, 15, 16, 17, 18],
        )
        out = add_rolling_stats(df, ["temp"], [2, 3], stats=["mean", "max"], group_by="city")
        ref = add_rolling_stats(
            df.sort_values("city", kind="stable"),
            ["temp"],
            [2, 3],
            stats=["mean", "max"],
            group_by="city",
        ).loc[df.index]
        for col in ("temp_roll2_mean", "temp_roll2_max", "temp_roll3_mean", "temp_roll3_max"):
            assert out[col].tolist() == pytest.approx(ref[col].tolist())

    def test_grouped_rolling_with_duplicate_index_labels(self):
        df = pd.DataFrame(
            {"city": ["a", "b", "a", "b"], "temp": [1.0, 10.0, 3.0, 30.0]},
            index=[0, 0, 1, 1],
        )
        out = add_rolling_stats(df, ["temp"], [2], stats=["mean"], group_by="city")
        assert out["temp_roll2_mean"].tolist() == pytest.approx([1.0, 10.0, 2.0, 20.0])
        assert out.index.tolist() == [0, 0, 1, 1]

    def test_grouped_rolling_does_not_mutate_input(self):
        df = pd.DataFrame({"city": ["a", "b", "a", "b"], "temp": [1.0, 10.0, 3.0, 30.0]})
        before = df.copy()
        add_rolling_stats(df, ["temp"], [2], group_by="city")
        pd.testing.assert_frame_equal(df, before)


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


class TestAddMissingIndicators:
    def test_flags_missing_values(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"], "c": [1, 2, 3]})
        out = add_missing_indicators(df, ["a", "b"])
        assert out["a_isna"].tolist() == [0, 1, 0]
        assert out["b_isna"].tolist() == [0, 1, 0]
        assert "c_isna" not in out.columns

    def test_dtype_is_int8(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        assert add_missing_indicators(df, ["a"])["a_isna"].dtype == np.dtype("int8")

    def test_custom_suffix(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        out = add_missing_indicators(df, ["a"], suffix="_missing")
        assert out["a_missing"].tolist() == [0, 1]

    def test_complete_column_gets_all_zero_indicator(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        assert add_missing_indicators(df, ["a"])["a_isna"].tolist() == [0, 0]

    def test_keeps_original_column_and_index(self):
        df = pd.DataFrame({"a": [1.0, np.nan]}, index=[7, 3])
        out = add_missing_indicators(df, ["a"])
        assert out["a"].isna().tolist() == [False, True]
        assert out.index.tolist() == [7, 3]

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="nope"):
            add_missing_indicators(pd.DataFrame({"a": [1.0]}), ["a", "nope"])

    def test_no_mutation(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        before = df.copy()
        out = add_missing_indicators(df, ["a"])
        pd.testing.assert_frame_equal(df, before)
        assert out is not df
        assert "a_isna" not in df.columns


class TestAddInteractions:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"a": [2.0, 3.0, 4.0], "b": [5.0, 0.0, -2.0], "c": [1.0, 1.0, 1.0]})

    def test_mul_default_and_name(self, df):
        out = add_interactions(df, [("a", "b")])
        assert out["a_x_b"].tolist() == pytest.approx([10.0, 0.0, -8.0])

    @pytest.mark.parametrize(
        ("op", "name", "expected"),
        [
            ("mul", "a_x_b", [10.0, 0.0, -8.0]),
            ("add", "a_plus_b", [7.0, 3.0, 2.0]),
            ("sub", "a_minus_b", [-3.0, 3.0, 6.0]),
        ],
    )
    def test_ops(self, df, op, name, expected):
        out = add_interactions(df, [("a", "b")], op=op)
        assert out[name].tolist() == pytest.approx(expected)

    def test_div_by_zero_is_nan_not_inf(self, df):
        out = add_interactions(df, [("a", "b")], op="div")
        vals = out["a_div_b"]
        assert vals.iloc[0] == pytest.approx(0.4)
        assert np.isnan(vals.iloc[1])
        assert vals.iloc[2] == pytest.approx(-2.0)
        assert not np.isinf(vals.to_numpy(dtype=float)).any()

    def test_div_zero_over_zero_is_nan(self):
        df = pd.DataFrame({"a": [0.0, -1.0], "b": [0.0, 0.0]})
        out = add_interactions(df, [("a", "b")], op="div")
        assert out["a_div_b"].isna().all()

    def test_div_integer_columns_give_float(self):
        df = pd.DataFrame({"a": [1, 2], "b": [2, 0]})
        out = add_interactions(df, [("a", "b")], op="div")
        assert out["a_div_b"].iloc[0] == pytest.approx(0.5)
        assert np.isnan(out["a_div_b"].iloc[1])

    def test_nan_propagates(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 2.0]})
        assert np.isnan(add_interactions(df, [("a", "b")])["a_x_b"].iloc[1])

    def test_multiple_pairs_in_given_order(self, df):
        out = add_interactions(df, [("a", "b"), ("b", "c"), ("a", "a")])
        assert list(out.columns) == ["a", "b", "c", "a_x_b", "b_x_c", "a_x_a"]

    def test_names_are_deterministic_and_order_sensitive(self, df):
        first = add_interactions(df, [("a", "b")], op="div")
        second = add_interactions(df, [("a", "b")], op="div")
        assert list(first.columns) == list(second.columns)
        assert "b_div_a" not in first.columns  # (a, b) is not symmetrised for div

    def test_unknown_column_raises(self, df):
        with pytest.raises(KeyError, match="zzz"):
            add_interactions(df, [("a", "zzz")])
        with pytest.raises(KeyError, match="zzz"):
            add_interactions(df, [("zzz", "a")])

    def test_unknown_op_raises(self, df):
        with pytest.raises(ValueError, match="op"):
            add_interactions(df, [("a", "b")], op="pow")  # type: ignore[arg-type]

    def test_no_mutation(self, df):
        before = df.copy()
        out = add_interactions(df, [("a", "b")], op="div")
        pd.testing.assert_frame_equal(df, before)
        assert out is not df
