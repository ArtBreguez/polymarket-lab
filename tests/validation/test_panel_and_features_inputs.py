"""build_panel refuses a bad date or price when the panel is built; features refuse a lookahead (#60, #62)."""

from __future__ import annotations

import pandas as pd
import pytest

from pmlab import PanelSchemaError, add_lags, add_rolling_stats, build_panel, clip_outliers


def _row(**over):
    row = {
        "market_id": "m1",
        "decision_horizon": "h",
        "decision_date": "2025-01-10",
        "outcome_label": "Yes",
        "winning_label": "Yes",
        "market_price": 0.4,
        "feature_a": 1.0,
    }
    row.update(over)
    return row


class TestBuildPanel:
    def test_good_rows_build(self):
        assert (
            len(
                build_panel(
                    [
                        _row(),
                        _row(market_id="m2", market_price=1.0),
                        _row(market_id="m3", market_price=0.0),
                    ]
                )
            )
            == 3
        )

    @pytest.mark.parametrize("date", ["2025-13-45", "2025-02-30", "2025-00-10", "2025-01-32"])
    def test_an_impossible_date_is_refused(self, date):
        with pytest.raises(PanelSchemaError, match="decision_date"):
            build_panel([_row(decision_date=date)])

    @pytest.mark.parametrize("price", [1.5, -0.1, float("inf")])
    def test_a_price_outside_the_unit_interval_is_refused(self, price):
        with pytest.raises(PanelSchemaError, match="market_price"):
            build_panel([_row(), _row(market_id="m2", market_price=price)])

    def test_partial_nan_prices_are_still_the_plugins_to_drop(self):
        panel = build_panel([_row(), _row(market_id="m2", market_price=float("nan"))])
        assert len(panel) == 2


class TestFeatures:
    DF = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6]})

    @pytest.mark.parametrize("lag", [-1, -3])
    def test_negative_lags_are_a_lookahead(self, lag):
        with pytest.raises(ValueError, match="lag"):
            add_lags(self.DF, ["x"], [lag])

    def test_positive_lags_unchanged(self):
        out = add_lags(self.DF, ["x"], [1, 2])
        assert out["x_lag1"].tolist()[1:3] == [1.0, 2.0]

    def test_unknown_clip_method(self):
        with pytest.raises(ValueError, match="method"):
            clip_outliers(self.DF, ["x"], method="bogus")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kw", [{"iqr_factor": -1.0}, {"zscore_threshold": -2.0}, {"iqr_factor": float("nan")}]
    )
    def test_clip_factors(self, kw):
        with pytest.raises(ValueError):
            clip_outliers(self.DF, ["x"], **kw)

    def test_rolling_stats_arguments(self):
        with pytest.raises(ValueError, match="stat"):
            add_rolling_stats(self.DF, ["x"], [2], stats=["bogus"])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="window"):
            add_rolling_stats(self.DF, ["x"], [0])
        with pytest.raises(ValueError, match="window"):
            add_rolling_stats(self.DF, ["x"], [-2])
