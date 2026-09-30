"""Tests for live calibration tracking on realized trades.

Closes the loop the gate opens: the gate says a model was calibrated on
historical data, this says whether it still is on money actually at risk.

Two traps this suite pins deliberately:

1. `model_prob` was not persisted in the trade log. Without the prediction there
   is nothing to compare an outcome against, so calibration on realized trades
   was impossible before this change.
2. The `outcome` field means "outcome_label occurred", NOT "this trade made
   money". A `direction="no"` trade whose label occurs is recorded as `"won"`
   with a NEGATIVE realized_pnl. Calibration wants the former; anyone reading
   "won" as profit gets the sign backwards.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pmlab.core.costs import CostModel
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker
from pmlab.monitoring.calibration_tracker import (
    CalibrationTracker,
    CalibrationWindow,
)

_NOW = datetime(2026, 6, 15, 5, 0, tzinfo=UTC)


def _signal(price: float = 0.50, prob: float = 0.56, direction: str = "yes") -> EdgeSignal:
    return EdgeSignal(
        market_id="m1",
        city_or_segment="politics",
        target_date="2026-06-15",
        horizon="morning_of",
        outcome_label="YES",
        direction=direction,
        gamma_price=price,
        model_prob=prob,
        best_edge=prob - price,
        yes_edge=prob - price,
        no_edge=price - prob,
    )


def _trade(
    prob: float,
    occurred: bool,
    *,
    segment: str = "politics",
    date: str = "2026-06-15",
    pnl: float = 1.0,
) -> dict:
    """A settled trade row, in the shape SettlementEngine leaves behind."""
    return {
        "recorded_at": f"{date}T05:00:00+00:00",
        "city_or_segment": segment,
        "target_date": date,
        "outcome_label": "YES",
        "direction": "yes",
        "gamma_price": 0.5,
        "fill_price": 0.5,
        "model_prob": prob,
        "edge_after_fee": prob - 0.5,
        "horizon": "morning_of",
        "flat_stake": 10.0,
        "size": 20.0,
        "fee_paid": 0.03,
        "outcome": "won" if occurred else "lost",
        "realized_pnl": pnl,
    }


class TestTheTradeLogCarriesThePrediction:
    """Prerequisite: you cannot score a forecast you did not write down."""

    def test_paper_broker_records_model_prob(self, tmp_path: Path) -> None:
        broker = PaperBroker(trades_path=tmp_path / "t.json", flat_stake=10.0)
        trades = broker.record(
            [_signal(prob=0.62)], city_timezones={"politics": "UTC"}, now_utc=_NOW
        )
        assert trades[0]["model_prob"] == 0.62

    def test_model_prob_survives_the_json_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        broker = PaperBroker(trades_path=path, flat_stake=10.0)
        broker.record([_signal(prob=0.62)], city_timezones={"politics": "UTC"}, now_utc=_NOW)
        on_disk = json.loads(path.read_text())["trades"][0]
        assert on_disk["model_prob"] == 0.62

    def test_model_prob_is_recorded_for_a_no_direction_too(self, tmp_path: Path) -> None:
        """The forecast is about the label, not about the side we took."""
        broker = PaperBroker(trades_path=tmp_path / "t.json", flat_stake=10.0)
        trades = broker.record(
            [_signal(price=0.62, prob=0.40, direction="no")],
            city_timezones={"politics": "UTC"},
            now_utc=_NOW,
        )
        assert trades[0]["model_prob"] == 0.40

    def test_recording_costs_does_not_disturb_model_prob(self, tmp_path: Path) -> None:
        broker = PaperBroker(
            trades_path=tmp_path / "t.json",
            flat_stake=10.0,
            costs=CostModel(taker_bps=30.0, slippage_bps=80.0),
        )
        trades = broker.record(
            [_signal(prob=0.62)], city_timezones={"politics": "UTC"}, now_utc=_NOW
        )
        t = trades[0]
        assert t["model_prob"] == 0.62
        assert t["fill_price"] > t["gamma_price"], "sanity: costs still applied"


class TestOnlyRealizedTradesEnter:
    """No-lookahead: an open trade has no outcome, so it cannot be scored."""

    def test_unsettled_trades_are_excluded(self) -> None:
        settled = _trade(0.6, True)
        open_trade = {**_trade(0.9, True), "outcome": None, "realized_pnl": None}
        tracker = CalibrationTracker([settled, open_trade])
        assert tracker.n_realized == 1

    def test_a_trade_without_model_prob_is_excluded(self) -> None:
        """Pre-0.8.5 rows have no prediction; they are unscoreable, not zero."""
        legacy = _trade(0.6, True)
        del legacy["model_prob"]
        tracker = CalibrationTracker([legacy, _trade(0.7, True)])
        assert tracker.n_realized == 1
        assert tracker.n_skipped == 1

    def test_all_unscoreable_is_an_empty_tracker_not_a_crash(self) -> None:
        open_trade = {**_trade(0.9, True), "outcome": None}
        tracker = CalibrationTracker([open_trade])
        assert tracker.n_realized == 0
        assert tracker.overall() is None


class TestOutcomeIsAboutTheLabelNotTheProfit:
    """The trap: `outcome="won"` can carry a negative realized_pnl."""

    def test_a_won_label_on_a_losing_no_trade_counts_as_occurred(self) -> None:
        """direction=no, label occurred: outcome="won", pnl<0, y_true must be 1."""
        losing_but_occurred = {
            **_trade(0.40, True, pnl=-6.70),
            "direction": "no",
        }
        tracker = CalibrationTracker([losing_but_occurred])
        window = tracker.overall()
        assert window is not None
        # y_true = 1 (label occurred), y_prob = 0.40 -> squared error 0.36
        assert window.brier_score == pytest.approx(0.36)

    def test_calibration_ignores_pnl_entirely(self) -> None:
        """Same forecasts and outcomes, different money: identical calibration."""
        cheap = [_trade(0.6, True, pnl=0.01), _trade(0.4, False, pnl=0.01)]
        rich = [_trade(0.6, True, pnl=999.0), _trade(0.4, False, pnl=-999.0)]
        a = CalibrationTracker(cheap).overall()
        b = CalibrationTracker(rich).overall()
        assert a is not None and b is not None
        assert a.brier_score == pytest.approx(b.brier_score)


class TestBrierAndReliability:
    def test_a_perfect_forecaster_scores_zero(self) -> None:
        trades = [_trade(1.0, True), _trade(0.0, False)] * 5
        window = CalibrationTracker(trades).overall()
        assert window is not None
        assert window.brier_score == pytest.approx(0.0)

    def test_a_maximally_wrong_forecaster_scores_one(self) -> None:
        trades = [_trade(0.0, True), _trade(1.0, False)] * 5
        window = CalibrationTracker(trades).overall()
        assert window is not None
        assert window.brier_score == pytest.approx(1.0)

    def test_brier_matches_the_manual_mean_squared_error(self) -> None:
        trades = [_trade(0.7, True), _trade(0.6, False), _trade(0.3, True)]
        expected = ((0.7 - 1) ** 2 + (0.6 - 0) ** 2 + (0.3 - 1) ** 2) / 3
        window = CalibrationTracker(trades).overall()
        assert window is not None
        assert window.brier_score == pytest.approx(expected)

    def test_reliability_curve_is_reported(self) -> None:
        trades = [_trade(0.2, False) for _ in range(8)] + [_trade(0.8, True) for _ in range(8)]
        window = CalibrationTracker(trades).overall()
        assert window is not None
        assert len(window.reliability_curve) >= 2
        for point in window.reliability_curve:
            assert set(point) == {"bin_center", "mean_predicted", "fraction_occurred", "n"}
        # This data is deliberately overconfident at both ends: forecast 0.2 and
        # nothing occurred, forecast 0.8 and everything did. The curve must show
        # that gap rather than smooth it away.
        low, high = window.reliability_curve[0], window.reliability_curve[-1]
        assert low["mean_predicted"] == pytest.approx(0.2)
        assert low["fraction_occurred"] == pytest.approx(0.0)
        assert high["mean_predicted"] == pytest.approx(0.8)
        assert high["fraction_occurred"] == pytest.approx(1.0)
        assert sum(p["n"] for p in window.reliability_curve) == 16

    def test_a_calibrated_curve_tracks_the_diagonal(self) -> None:
        """Forecast 0.25 with 25% occurring, 0.75 with 75%: gaps stay small."""
        trades = [_trade(0.25, i < 1) for i in range(4)]  # 1 of 4 occurred
        trades += [_trade(0.75, i < 3) for i in range(4)]  # 3 of 4 occurred
        window = CalibrationTracker(trades).overall()
        assert window is not None
        for point in window.reliability_curve:
            assert abs(point["mean_predicted"] - point["fraction_occurred"]) < 1e-9
        assert window.reliability == pytest.approx(0.0)

    def test_overconfidence_shows_up_as_reliability_error(self) -> None:
        """Predict 0.9 but only half occur: the curve must expose the gap."""
        trades = [_trade(0.9, i % 2 == 0) for i in range(20)]
        window = CalibrationTracker(trades).overall()
        assert window is not None
        assert window.reliability > 0.1, "an 0.9 forecast hitting 50% is miscalibrated"
        point = window.reliability_curve[0]
        assert point["mean_predicted"] == pytest.approx(0.9)
        assert point["fraction_occurred"] == pytest.approx(0.5)


class TestRollingWindows:
    def test_rolling_returns_one_window_per_step(self) -> None:
        trades = [_trade(0.6, True, date=f"2026-06-{d:02d}") for d in range(1, 11)]
        windows = CalibrationTracker(trades).rolling(window=5)
        assert len(windows) == 6  # 10 trades, window 5 -> positions 5..10
        assert all(w.n_trades == 5 for w in windows)

    def test_a_window_larger_than_the_history_yields_nothing(self) -> None:
        trades = [_trade(0.6, True) for _ in range(3)]
        assert CalibrationTracker(trades).rolling(window=10) == []

    def test_rolling_is_ordered_by_target_date_not_insertion(self) -> None:
        """Truth arrives on target_date; scoring must follow that order."""
        late = _trade(0.9, True, date="2026-06-20")
        early = _trade(0.1, False, date="2026-06-01")
        windows = CalibrationTracker([late, early]).rolling(window=1)
        assert windows[0].window_end == "2026-06-01"
        assert windows[1].window_end == "2026-06-20"

    def test_ordering_follows_target_date_not_recorded_at(self) -> None:
        """A backfilled trade log has the two dates disagreeing.

        Mutation testing caught this: sorting by `recorded_at` passed every other
        test, because the fixtures move both dates together. A trade recorded
        late about an early event belongs early in the calibration history — the
        forecast is scored when its truth lands, not when someone wrote the row.
        """
        early_event_recorded_late = {
            **_trade(0.1, False, date="2026-06-01"),
            "recorded_at": "2026-12-31T23:00:00+00:00",
        }
        late_event_recorded_early = {
            **_trade(0.9, True, date="2026-06-20"),
            "recorded_at": "2026-01-01T00:00:00+00:00",
        }
        windows = CalibrationTracker(
            [late_event_recorded_early, early_event_recorded_late]
        ).rolling(window=1)
        assert [w.window_end for w in windows] == ["2026-06-01", "2026-06-20"]

    def test_degradation_is_visible_across_windows(self) -> None:
        """Good forecasts then bad ones: Brier must rise."""
        good = [_trade(0.95, True, date=f"2026-06-{d:02d}") for d in range(1, 11)]
        bad = [_trade(0.95, False, date=f"2026-07-{d:02d}") for d in range(1, 11)]
        windows = CalibrationTracker(good + bad).rolling(window=10)
        assert windows[0].brier_score < 0.05, "first window should look good"
        assert windows[-1].brier_score > 0.8, "last window should look broken"
        assert windows[-1].brier_score > windows[0].brier_score

    def test_window_bounds_are_reported(self) -> None:
        trades = [_trade(0.6, True, date=f"2026-06-{d:02d}") for d in range(1, 6)]
        window = CalibrationTracker(trades).rolling(window=3)[0]
        assert window.window_start == "2026-06-01"
        assert window.window_end == "2026-06-03"

    def test_a_non_positive_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="window"):
            CalibrationTracker([_trade(0.6, True)]).rolling(window=0)


class TestSegmentBreakdown:
    """The gate is per-segment, so calibration has to be too."""

    def test_segments_are_scored_separately(self) -> None:
        trades = [
            _trade(0.95, True, segment="politics"),
            _trade(0.95, True, segment="politics"),
            _trade(0.95, False, segment="sports"),
            _trade(0.95, False, segment="sports"),
        ]
        by_segment = CalibrationTracker(trades).by_segment()
        assert set(by_segment) == {"politics", "sports"}
        assert by_segment["politics"].brier_score < 0.05
        assert by_segment["sports"].brier_score > 0.8

    def test_an_aggregate_can_hide_one_broken_segment(self) -> None:
        """The reason per-segment exists: the mean looks survivable."""
        trades = [_trade(0.95, True, segment="politics") for _ in range(18)]
        trades += [_trade(0.95, False, segment="sports") for _ in range(2)]
        tracker = CalibrationTracker(trades)
        overall = tracker.overall()
        assert overall is not None
        assert overall.brier_score < 0.15, "aggregate looks fine"
        assert tracker.by_segment()["sports"].brier_score > 0.8, "sports is broken"


class TestSerialisation:
    def test_to_dict_round_trips_through_json(self) -> None:
        trades = [_trade(0.7, True), _trade(0.3, False)]
        window = CalibrationTracker(trades).overall()
        assert window is not None
        restored = CalibrationWindow.from_dict(json.loads(json.dumps(window.to_dict())))
        assert restored == window

    def test_to_dict_contains_no_numpy_types(self) -> None:
        """A report that cannot be json.dumps'd is not a report."""
        window = CalibrationTracker([_trade(0.7, True), _trade(0.3, False)]).overall()
        assert window is not None
        text = json.dumps(window.to_dict())  # would raise on np.float64
        assert "model_prob" not in text  # it is a summary, not the raw rows

    def test_two_runs_on_the_same_trades_agree(self) -> None:
        trades = [_trade(0.6, i % 3 == 0) for i in range(30)]
        a = CalibrationTracker(trades).overall()
        b = CalibrationTracker(trades).overall()
        assert a is not None and b is not None
        assert a.to_dict() == b.to_dict()


class TestReadingFromDisk:
    def test_tracker_loads_the_broker_trade_log(self, tmp_path: Path) -> None:
        path = tmp_path / "trades.json"
        path.write_text(json.dumps({"trades": [_trade(0.7, True), _trade(0.3, False)]}))
        tracker = CalibrationTracker.from_path(path)
        assert tracker.n_realized == 2

    def test_a_missing_file_is_an_explicit_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CalibrationTracker.from_path(tmp_path / "nope.json")
