"""TDD (RED first): typed panel builder.

Assembles the training panel from plugin rows with schema + dtype validation,
so the KeyError-at-backtest-time friction (missing required columns discovered
only deep in rolling_origin_eval) becomes a clear error at build time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pmlab.data import LeakageError, PanelSchemaError, build_panel


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


# ---------------------------------------------------------------------------
# Per-bin rows (issue #28): build_training_rows returns a list per market
# ---------------------------------------------------------------------------


def _bin_rows(market: str, date: str = "2025-01-05", winner: str = "b") -> list[dict]:
    prices = {"a": 0.2, "b": 0.5, "c": 0.3}
    return [
        {
            "market_id": market,
            "decision_date": date,
            "outcome_label": label,
            "winning_label": winner,
            "market_price": price,
            "feature_x": 1.0,
        }
        for label, price in prices.items()
    ]


class TestBuildPanelPerBinRows:
    def test_lists_of_rows_are_flattened_in_order(self):
        panel = build_panel([_bin_rows("m1"), _bin_rows("m2")])
        assert panel["market_id"].tolist() == ["m1"] * 3 + ["m2"] * 3
        assert panel["outcome_label"].tolist() == ["a", "b", "c"] * 2
        assert panel.index.tolist() == list(range(6))

    def test_dicts_lists_and_nones_can_be_mixed(self):
        single = _bin_rows("m3")[0]
        panel = build_panel([None, _bin_rows("m1"), [], single, None])
        assert len(panel) == 4

    def test_a_none_inside_a_list_is_dropped(self):
        a, b = _bin_rows("m1")[:2]
        assert len(build_panel([[a, None, b]])) == 2

    def test_a_generator_of_lists_is_accepted(self):
        panel = build_panel(_bin_rows(f"m{i}") for i in range(4))
        assert len(panel) == 12

    def test_only_empty_lists_is_an_empty_panel_error(self):
        with pytest.raises(PanelSchemaError, match="No rows"):
            build_panel([[], []])

    def test_a_price_after_the_decision_is_rejected_even_without_check_leakage(self):
        rows = _bin_rows("m1")
        for r in rows:
            r["decision_ts"] = "2025-01-05T08:00:00+00:00"
            r["price_ts"] = "2025-01-05T07:00:00+00:00"
        rows[1]["price_ts"] = "2025-01-05T08:00:01+00:00"
        with pytest.raises(LeakageError, match="after the decision"):
            build_panel([rows])

    def test_timed_in_time_prices_build(self):
        rows = _bin_rows("m1")
        for r in rows:
            r["decision_ts"] = "2025-01-05T08:00:00+00:00"
            r["price_ts"] = "2025-01-05T07:00:00+00:00"
        assert len(build_panel([rows])) == 3

    def test_a_panel_without_timing_columns_is_unchanged(self):
        assert len(build_panel([_bin_rows("m1")])) == 3


class TestBuildPanelMarketClose:
    def _timed_rows(self, close: str) -> list[dict]:
        rows = _bin_rows("m1")
        for r in rows:
            r["decision_ts"] = "2025-01-05T08:00:00+00:00"
            r["price_ts"] = "2025-01-05T07:00:00+00:00"
            r["market_close_ts"] = close
        return rows

    def test_a_decision_after_the_market_closed_is_rejected(self):
        with pytest.raises(LeakageError, match="close"):
            build_panel([self._timed_rows("2025-01-05T07:30:00+00:00")])

    def test_a_decision_before_the_close_builds(self):
        assert len(build_panel([self._timed_rows("2025-01-05T20:00:00+00:00")])) == 3


class TestBuildPanelPartialTimingColumns:
    """Half a timing schema must not be waved through as an untimed panel (review, M2)."""

    def _rows(self, drop: str) -> list[dict]:
        rows = _bin_rows("m1")
        for r in rows:
            r["decision_ts"] = "2025-01-05T08:00:00+00:00"
            r["price_ts"] = "2025-01-05T07:00:00+00:00"
            del r[drop]
        return rows

    def test_a_decision_time_without_a_price_time_is_refused(self):
        with pytest.raises(LeakageError, match="has 'decision_ts' but no 'price_ts'"):
            build_panel([self._rows("price_ts")])

    def test_a_price_time_without_a_decision_time_is_refused(self):
        with pytest.raises(LeakageError, match="has 'price_ts' but no 'decision_ts'"):
            build_panel([self._rows("decision_ts")])


class TestBuildPanelUntimedWarning:
    """A panel whose prices carry no timing cannot be checked for lookahead; say so, once (review, M3)."""

    LOGGER = "pmlab.data.panel_builder"

    def _timed(self, n: int = 3) -> list[dict]:
        rows = _bin_rows("m1")[:n]
        for r in rows:
            r["decision_ts"] = "2025-01-05T08:00:00+00:00"
            r["price_ts"] = "2025-01-05T07:00:00+00:00"
        return rows

    def _warnings(self, caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.name == self.LOGGER]

    def test_untimed_prices_log_exactly_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            build_panel(_rows(60))
        (record,) = self._warnings(caplog)
        assert record.levelno == logging.WARNING

    def test_the_warning_names_the_risk_and_the_remedy(self, caplog):
        with caplog.at_level(logging.WARNING):
            build_panel(_rows(10))
        text = self._warnings(caplog)[0].getMessage()
        for needle in ("market_price", "price_ts", "decision_ts", "lookahead", "10 rows"):
            assert needle in text, needle

    def test_it_is_one_per_call_not_one_per_market(self, caplog):
        with caplog.at_level(logging.WARNING):
            build_panel([_bin_rows(f"m{i}") for i in range(20)])
        assert len(self._warnings(caplog)) == 1

    def test_each_call_warns_once(self, caplog):
        with caplog.at_level(logging.WARNING):
            build_panel(_rows(5))
            build_panel(_rows(5))
        assert len(self._warnings(caplog)) == 2

    def test_it_goes_to_logging_and_never_to_stdout_or_stderr(self, caplog, capsys):
        with caplog.at_level(logging.WARNING):
            build_panel(_rows(5))
        out = capsys.readouterr()
        assert (out.out, out.err) == ("", "")
        assert len(self._warnings(caplog)) == 1

    def test_a_timed_panel_says_nothing(self, caplog):
        with caplog.at_level(logging.DEBUG):
            build_panel([self._timed()])
        assert self._warnings(caplog) == []

    def test_a_close_column_alone_does_not_make_a_panel_timed(self, caplog):
        rows = _bin_rows("m1")
        for r in rows:
            r["market_close_ts"] = "2025-01-05T20:00:00+00:00"
        with caplog.at_level(logging.WARNING):
            build_panel([rows])
        assert len(self._warnings(caplog)) == 1

    def test_half_a_timing_schema_is_an_error_not_a_warning(self, caplog):
        rows = self._timed()
        for r in rows:
            del r["price_ts"]
        with caplog.at_level(logging.WARNING), pytest.raises(LeakageError):
            build_panel([rows])
        assert self._warnings(caplog) == []

    def test_a_panel_that_fails_to_build_does_not_warn_first(self, caplog):
        with caplog.at_level(logging.WARNING), pytest.raises(PanelSchemaError):
            build_panel([])
        assert self._warnings(caplog) == []


# ---------------------------------------------------------------------------
# Duplicated rows (#68): a (market, bin, decision date, horizon) is one row
# ---------------------------------------------------------------------------


class TestBuildPanelDuplicates:
    """The panel is one row per outcome bin of a market and decision (docs/data.md).

    A repeated row double-weights its market in training and reaches ``min_train_rows`` early, so
    the backtest runs a different number of folds. Which of two rows with the same key is the
    right one cannot be told from the rows, so the panel is refused, not quietly thinned.
    """

    def test_an_exact_duplicate_is_refused(self):
        rows = _bin_rows("m1") + _bin_rows("m2")
        rows.append(dict(rows[0]))
        with pytest.raises(PanelSchemaError, match="duplicate"):
            build_panel([rows])

    def test_the_same_key_with_other_features_is_refused_too(self):
        # a re-fetched snapshot of the same market and day: which one is right is not ours to say
        first = _bin_rows("m1")
        again = _bin_rows("m1")
        for r in again:
            r["feature_x"] = 2.0
        with pytest.raises(PanelSchemaError, match="duplicate"):
            build_panel([first, again])

    def test_the_error_counts_the_rows_and_names_a_key(self):
        rows = _bin_rows("m1") + _bin_rows("m2") + [dict(_bin_rows("m2")[1])]
        with pytest.raises(PanelSchemaError) as err:
            build_panel(rows)
        message = str(err.value)
        assert "2 rows" in message  # both copies of the repeated key, not the one extra
        assert "market_id='m2'" in message and "outcome_label='b'" in message
        assert "decision_date='2025-01-05'" in message

    def test_a_dict_row_repeated_across_the_input_is_caught(self):
        row = _bin_rows("m1")[0]
        with pytest.raises(PanelSchemaError, match="duplicate"):
            build_panel([row, [row]])

    def test_other_bins_other_dates_and_other_markets_are_not_duplicates(self):
        panel = build_panel([_bin_rows("m1"), _bin_rows("m1", date="2025-01-06"), _bin_rows("m2")])
        assert len(panel) == 9

    def test_a_decision_horizon_is_part_of_the_key(self):
        rows = []
        for horizon in ("market_open", "previous_evening"):
            for r in _bin_rows("m1"):
                rows.append({**r, "decision_horizon": horizon})
        assert len(build_panel([rows])) == 6  # two horizons on one date are two decisions
        rows.append(dict(rows[0]))
        with pytest.raises(PanelSchemaError, match="decision_horizon"):
            build_panel([rows])

    def test_a_horizon_column_is_part_of_the_key_as_well(self):
        rows = [{**r, "horizon": h} for h in ("market_open", "morning_of") for r in _bin_rows("m1")]
        assert len(build_panel([rows])) == 6

    def test_the_check_comes_before_the_untimed_price_warning(self, caplog):
        rows = _bin_rows("m1") + _bin_rows("m1")
        with caplog.at_level(logging.WARNING), pytest.raises(PanelSchemaError):
            build_panel([rows])
        assert [r for r in caplog.records if r.name == "pmlab.data.panel_builder"] == []

    def test_the_check_runs_without_check_leakage_and_with_it(self):
        rows = _bin_rows("m1") * 2
        for check_leakage in (False, True):
            with pytest.raises(PanelSchemaError, match="duplicate"):
                build_panel([rows], check_leakage=check_leakage)
