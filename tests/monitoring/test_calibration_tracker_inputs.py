"""What `CalibrationTracker` accepts (#68).

* A settled trade whose ``model_prob`` is NaN, infinite or outside [0, 1] is not a forecast that
  turned out wrong, it is a broken record. The watchdog already leaves such trades out and counts
  them as bad data; the tracker counted them as *realized* and then failed in ``overall()`` with a
  ``ValueError`` about the probabilities, far from the cause. It now leaves them out, like a trade
  with no forecast, and counts them in ``n_skipped``.
* ``n_bins`` is checked when the tracker is built, not when the first window is scored.
* ``from_path`` on a file that is not a trade log says so, instead of ``AttributeError``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pmlab.monitoring.calibration_tracker import CalibrationTracker
from tests.monitoring.test_calibration_tracker import _trade


def _with_prob(prob, *, occurred: bool = True) -> dict:
    trade = _trade(0.5, occurred)
    trade["model_prob"] = prob
    return trade


class TestAProbabilityThatIsNotOne:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1.0000001, 1.5, -0.01, -1])
    def test_it_is_left_out_and_counted_not_scored(self, bad) -> None:
        good = [_trade(0.7, True), _trade(0.3, False), _trade(0.6, True)]
        tracker = CalibrationTracker([*good, _with_prob(bad)])
        assert tracker.n_realized == 3
        assert tracker.n_skipped == 1
        window = tracker.overall()  # used to raise ValueError about y_prob once a bad one was in
        assert window is not None and window.n_trades == 3
        assert math.isfinite(window.brier_score)

    def test_when_every_forecast_is_bad_there_is_nothing_to_score(self) -> None:
        tracker = CalibrationTracker([_with_prob(math.nan), _with_prob(2.0)])
        assert (tracker.n_realized, tracker.n_skipped) == (0, 2)
        assert tracker.overall() is None
        assert tracker.rolling(1) == []
        assert tracker.by_segment() == {}
        assert tracker.squared_errors() == []

    @pytest.mark.parametrize("edge", [0.0, 1.0, 0, 1, 0.5])
    def test_the_ends_of_the_interval_are_forecasts(self, edge) -> None:
        tracker = CalibrationTracker([_with_prob(edge, occurred=True)])
        assert tracker.n_realized == 1 and tracker.n_skipped == 0

    @pytest.mark.parametrize("bad", [True, False, None, "0.5", [0.5]])
    def test_what_was_never_a_number_is_still_skipped_as_before(self, bad) -> None:
        tracker = CalibrationTracker([_with_prob(bad)])
        assert (tracker.n_realized, tracker.n_skipped) == (0, 1)

    def test_an_open_trade_with_a_bad_forecast_is_just_open(self) -> None:
        open_trade = _with_prob(math.nan)
        open_trade["outcome"] = None
        tracker = CalibrationTracker([open_trade])
        assert (tracker.n_realized, tracker.n_skipped) == (0, 1)

    def test_segments_and_rolling_windows_leave_it_out_too(self) -> None:
        trades = [_trade(0.7, True, segment="a"), _trade(0.3, False, segment="b")]
        trades.append(_with_prob(math.nan))
        tracker = CalibrationTracker(trades)
        assert sorted(tracker.by_segment()) == ["a", "b"]
        assert len(tracker.rolling(2)) == 1


class TestNBins:
    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_below_one_is_a_value_error_at_construction(self, bad) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            CalibrationTracker([_trade(0.6, True)], n_bins=bad)

    @pytest.mark.parametrize("bad", [None, 2.5, True, "10", 10.0])
    def test_not_an_int_is_a_type_error_at_construction(self, bad) -> None:
        with pytest.raises(TypeError, match="n_bins"):
            CalibrationTracker([_trade(0.6, True)], n_bins=bad)

    def test_it_is_checked_even_when_there_is_nothing_to_score(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            CalibrationTracker([], n_bins=0)

    @pytest.mark.parametrize("n_bins", [1, 5, 10, 50, np.int64(7)])
    def test_a_positive_int_works(self, n_bins) -> None:
        trades = [_trade(0.7, True), _trade(0.3, False), _trade(0.5, True)]
        window = CalibrationTracker(trades, n_bins=n_bins).overall()
        assert window is not None and len(window.reliability_curve) <= int(n_bins)

    def test_the_default_is_unchanged(self) -> None:
        assert CalibrationTracker([]).n_bins == 10


class TestFromPath:
    def _write(self, tmp_path: Path, payload) -> Path:
        path = tmp_path / "trades.json"
        path.write_text(json.dumps(payload))
        return path

    @pytest.mark.parametrize(
        "payload",
        [[{"outcome": "won"}], [], {}, {"other": 1}, {"trades": None}, {"trades": "x"},
         {"trades": [1, 2]}, {"trades": {"a": 1}}, "text", 7, None],
    )  # fmt: skip
    def test_what_is_not_a_trade_log_is_a_value_error_that_says_so(self, tmp_path, payload) -> None:
        with pytest.raises(ValueError, match="'trades' list") as err:
            CalibrationTracker.from_path(self._write(tmp_path, payload))
        assert "trades.json" in str(err.value)

    def test_a_trade_log_still_loads(self, tmp_path) -> None:
        path = self._write(tmp_path, {"trades": [_trade(0.7, True), _trade(0.3, False)]})
        assert CalibrationTracker.from_path(path).n_realized == 2

    def test_an_empty_trade_log_is_a_tracker_with_nothing_in_it(self, tmp_path) -> None:
        tracker = CalibrationTracker.from_path(self._write(tmp_path, {"trades": []}))
        assert (tracker.n_realized, tracker.n_skipped) == (0, 0)

    def test_invalid_json_is_still_a_value_error(self, tmp_path) -> None:
        path = tmp_path / "trades.json"
        path.write_text("{not json")
        with pytest.raises(ValueError):
            CalibrationTracker.from_path(path)

    def test_a_missing_file_is_still_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            CalibrationTracker.from_path(tmp_path / "nope.json")

    def test_n_bins_is_checked_on_this_path_too(self, tmp_path) -> None:
        path = self._write(tmp_path, {"trades": []})
        with pytest.raises(ValueError, match="n_bins"):
            CalibrationTracker.from_path(path, n_bins=0)
