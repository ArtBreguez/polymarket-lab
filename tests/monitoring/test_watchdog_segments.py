"""The watchdog judges each segment as well as the book, and corrects for doing so.

A segment can bleed while the others hide it in the aggregate. So when the champion
recorded a per-segment expectation, calibration and PnL are also judged per segment:

- a segment that is clearly worse than its own backtest is named;
- the segments are a family of tests, and Holm's correction keeps the chance of a false
  alarm somewhere in the family at the level a single test would have, however many
  segments there are;
- a segment with too few live trades (or none promised in the backtest) is *not
  assessed*: listed, never counted as healthy, and never allowed to turn an otherwise
  conclusive check insufficient.

Fixtures use two-point samples (`mean +/- sd`, alternating) so the statistics a test
relies on are known in advance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats as sp

from pmlab.backtest.expectation import BacktestExpectation, GroupExpectation, SegmentExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring.watchdog import HealthReport, health_report

_PUBLISHED = "2026-01-01T00:00:00+00:00"
_AS_OF = "2026-03-01T00:00:00+00:00"


def _seg(
    n: int = 400,
    mean: float = 0.20,
    std: float = 0.60,
    brier: float = 0.20,
    brier_std: float = 0.10,
    by_direction: dict[str, GroupExpectation] | None = None,
) -> SegmentExpectation:
    return SegmentExpectation(n, mean, std, brier, brier_std, by_direction=by_direction)


def _expectation(
    segments: dict[str, SegmentExpectation | None] | None,
    *,
    by_direction: dict[str, GroupExpectation] | None = None,
) -> BacktestExpectation:
    n = 400 * max(1, len(segments or {}))
    return BacktestExpectation(
        n_trades=n,
        flat_stake=1.0,
        mean_pnl_per_stake=0.20,
        std_pnl_per_stake=0.60,
        brier_score=0.20,
        brier_std=0.10,
        by_direction=by_direction,
        by_segment=segments,
    )


def _champion(expectation: BacktestExpectation, passing: tuple[str, ...]) -> ChampionManifest:
    gate = HoldoutGateResult(
        decision="GO",
        segment_results=[SegmentGateResult(s, 100, 10.0, True, "ok") for s in passing],
        aggregate_pnl=10.0 * len(passing),
        aggregate_trades=100 * len(passing),
    )
    return ChampionManifest(
        model_name="champ",
        model_path=Path("champion.pkl"),
        calibrator_path=None,
        gate=gate,
        published_at=_PUBLISHED,
        plugin_family="politics",
        expectation=expectation,
    )


def _two_point(n: int, mean: float, sd: float) -> list[float]:
    return [mean + sd * (1 if i % 2 == 0 else -1) for i in range(n)]


def _trades(
    segment: str,
    n: int,
    *,
    pnl: float = 0.20,
    pnl_sd: float = 0.60,
    brier: float = 0.20,
    brier_sd: float = 0.10,
    side: str = "yes",
    **extra: Any,
) -> list[dict[str, Any]]:
    rows = []
    for term, ret in zip(_two_point(n, brier, brier_sd), _two_point(n, pnl, pnl_sd), strict=True):
        row = {
            "recorded_at": "2026-02-01T10:00:00+00:00",
            "city_or_segment": segment,
            "target_date": "2026-02-05",
            "outcome_label": "YES",
            "direction": side,
            "gamma_price": 0.40,
            "model_prob": math.sqrt(term),
            "flat_stake": 1.0,
            "outcome": "lost",
            "realized_pnl": ret,
            "settlement_version": 1,
            "settled_basis": "quote",
        }
        row.update(extra)
        rows.append(row)
    return rows


def _report(
    trades: list[dict[str, Any]],
    expectation: BacktestExpectation,
    passing: tuple[str, ...],
    **kwargs: Any,
) -> HealthReport:
    return health_report(
        champion=_champion(expectation, passing),
        trades=trades,
        drift=None,
        as_of=_AS_OF,
        checks=("calibration", "pnl"),
        **kwargs,
    )


def _check(report: HealthReport, name: str):
    return next(c for c in report.checks if c.name == name)


_ABC = ("A", "B", "C")


def _three() -> BacktestExpectation:
    return _expectation({s: _seg() for s in _ABC})


class TestAHealthyBookIsQuiet:
    def test_every_segment_as_promised(self) -> None:
        trades = [t for s in _ABC for t in _trades(s, 60)]
        report = _report(trades, _three(), _ABC)
        assert report.status == "ok"
        assert report.render() == ""
        pnl = _check(report, "pnl")
        assert set(pnl.metrics["segments"]) == set(_ABC)
        assert all(v["status"] == "ok" for v in pnl.metrics["segments"].values())
        assert pnl.metrics["segments"]["A"]["n_trades"] == 60

    def test_each_segment_is_measured_against_its_own_numbers(self) -> None:
        expectation = _expectation(
            {"A": _seg(mean=0.30), "B": _seg(mean=0.10), "C": _seg(mean=0.20)}
        )
        trades = _trades("A", 60, pnl=0.30) + _trades("B", 60, pnl=0.10) + _trades("C", 60)
        pnl = _check(_report(trades, expectation, _ABC), "pnl")
        assert pnl.metrics["segments"]["A"]["mean_backtest"] == pytest.approx(0.30)
        assert pnl.metrics["segments"]["B"]["mean_backtest"] == pytest.approx(0.10)
        assert pnl.status == "ok"

    def test_calibration_is_judged_per_segment_too(self) -> None:
        trades = [t for s in _ABC for t in _trades(s, 60)]
        calibration = _check(_report(trades, _three(), _ABC), "calibration")
        assert set(calibration.metrics["segments"]) == set(_ABC)
        assert calibration.metrics["segments"]["B"]["brier_backtest"] == pytest.approx(0.20)


class TestASegmentThatBleedsIsNamed:
    """Segment B is bleeding; 400 healthy trades in A hide it in the aggregate."""

    def _trades(self) -> list[dict[str, Any]]:
        return _trades("A", 400) + _trades("B", 40, pnl=-0.40)

    def _report(self) -> HealthReport:
        return _report(self._trades(), _expectation({"A": _seg(), "B": _seg()}), ("A", "B"))

    def test_the_aggregate_alone_is_quiet(self) -> None:
        pnl = _check(self._report(), "pnl")
        assert pnl.metrics["shortfall_rel"] < 0.5

    def test_the_check_alerts_and_names_the_segment(self) -> None:
        report = self._report()
        pnl = _check(report, "pnl")
        assert pnl.status == "alert"
        assert pnl.reasons[0].startswith("segment B:")
        assert "shortfall" in pnl.reasons[0]
        assert report.exit_code == 20
        assert "segment B" in report.render()
        assert "segment A" not in report.render().split("segment B")[0]

    def test_the_healthy_segment_is_reported_ok(self) -> None:
        pnl = _check(self._report(), "pnl")
        assert pnl.metrics["segments"]["A"]["status"] == "ok"
        assert pnl.metrics["segments"]["B"]["status"] == "alert"

    def test_the_segment_numbers_are_exact(self) -> None:
        b = _check(self._report(), "pnl").metrics["segments"]["B"]
        live = np.array(_two_point(40, -0.40, 0.60))
        z = (0.20 - live.mean()) / math.sqrt(live.var(ddof=1) / 40 + 0.60**2 / 400)
        assert b["mean_live"] == pytest.approx(-0.40)
        assert b["mean_backtest"] == pytest.approx(0.20)
        assert b["shortfall_rel"] == pytest.approx(3.0)
        assert b["z"] == pytest.approx(z)
        assert b["p_value"] == pytest.approx(sp.norm.sf(z))

    def test_the_overall_line_stays_as_context(self) -> None:
        pnl = _check(self._report(), "pnl")
        assert any(r.startswith("overall: ") for r in pnl.reasons)

    def test_a_miscalibrated_segment_is_named_by_the_calibration_check(self) -> None:
        trades = _trades("A", 400) + _trades("B", 60, brier=0.38)
        report = _report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B"))
        calibration = _check(report, "calibration")
        assert calibration.status == "alert"
        assert calibration.reasons[0].startswith("segment B:")
        assert calibration.metrics["segments"]["B"]["brier_live"] == pytest.approx(0.38)
        assert calibration.metrics["segments"]["A"]["status"] == "ok"

    def test_a_degraded_segment_beats_an_overall_warn_in_the_order_of_lines(self) -> None:
        # The segment is as bad as the book: the segment leads, because it names the culprit.
        trades = _trades("A", 60, pnl=-0.5) + _trades("B", 60, pnl=-0.5)
        pnl = _check(_report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B")), "pnl")
        assert pnl.reasons[0].startswith("segment ")


def _ten() -> BacktestExpectation:
    return _expectation({f"S{i}": _seg() for i in range(10)})


_TEN = tuple(f"S{i}" for i in range(10))
# Live PnL for S0 that sits at z = 2.3 against a 0.20 promise on 50 trades: a warn on its own.
_BORDERLINE = -0.0089


class TestHolmKeepsMoreSegmentsFromMeaningMoreFalseAlarms:
    def test_the_borderline_segment_warns_when_it_is_the_only_test(self) -> None:
        pnl = _check(_report(_trades("S0", 50, pnl=_BORDERLINE), _ten(), _TEN), "pnl")
        s0 = pnl.metrics["segments"]["S0"]
        assert 2.0 < s0["z"] < 2.6
        assert s0["status"] == "warn"
        assert s0["p_adjusted"] == pytest.approx(s0["p_value"])  # a family of one

    def test_the_same_segment_is_not_flagged_among_ten(self) -> None:
        trades = _trades("S0", 50, pnl=_BORDERLINE) + [
            t for i in range(1, 10) for t in _trades(f"S{i}", 50)
        ]
        pnl = _check(_report(trades, _ten(), _TEN), "pnl")
        s0 = pnl.metrics["segments"]["S0"]
        assert 2.0 < s0["z"] < 2.6  # the raw evidence is unchanged ...
        assert s0["p_adjusted"] > 9 * s0["p_value"]  # ... but ten tests were run,
        assert s0["z_adjusted"] < 2.0
        assert s0["status"] == "ok"  # so it is not enough
        assert pnl.status == "ok"

    def test_the_adjusted_p_value_is_holms(self) -> None:
        trades = _trades("S0", 50, pnl=_BORDERLINE) + [
            t for i in range(1, 10) for t in _trades(f"S{i}", 50)
        ]
        segments = _check(_report(trades, _ten(), _TEN), "pnl").metrics["segments"]
        raw = [segments[f"S{i}"]["p_value"] for i in range(10)]
        order = sorted(raw)
        expected_min = min(1.0, 10 * order[0])
        assert segments["S0"]["p_adjusted"] == pytest.approx(expected_min)
        assert segments["S0"]["z_adjusted"] == pytest.approx(sp.norm.isf(expected_min))

    def test_the_line_shows_both_z_values_inside_a_family(self) -> None:
        trades = _trades("S0", 50, pnl=_BORDERLINE) + _trades("S1", 50, pnl=-0.6)
        pnl = _check(_report(trades, _ten(), _TEN), "pnl")
        line = next(r for r in pnl.reasons if r.startswith("segment S1:"))
        assert "Holm z=" in line

    def test_a_real_problem_survives_the_correction(self) -> None:
        trades = _trades("S0", 50, pnl=-0.60) + [
            t for i in range(1, 10) for t in _trades(f"S{i}", 50)
        ]
        pnl = _check(_report(trades, _ten(), _TEN), "pnl")
        assert pnl.metrics["segments"]["S0"]["status"] == "alert"
        assert pnl.reasons[0].startswith("segment S0:")

    def test_calibration_segments_are_a_family_too(self) -> None:
        # Brier 0.30 against 0.20 on 50 trades: z about 2.1, a warn alone.
        trades = _trades("S0", 50, brier=0.30, brier_sd=0.1) + [
            t for i in range(1, 10) for t in _trades(f"S{i}", 50)
        ]
        single = _check(
            _report(_trades("S0", 50, brier=0.30, brier_sd=0.1), _ten(), _TEN), "calibration"
        )
        many = _check(_report(trades, _ten(), _TEN), "calibration")
        assert single.metrics["segments"]["S0"]["p_adjusted"] == pytest.approx(
            single.metrics["segments"]["S0"]["p_value"]
        )
        s0 = many.metrics["segments"]["S0"]
        assert s0["p_adjusted"] >= 9 * s0["p_value"] - 1e-12
        assert s0["z_adjusted"] < s0["z"]


class TestASegmentThatCannotBeAssessed:
    @pytest.mark.parametrize("check", ["calibration", "pnl"])
    def test_exactly_min_trades_is_enough_and_one_fewer_is_not(self, check: str) -> None:
        enough = _report(
            _trades("A", 400) + _trades("B", 30),
            _expectation({"A": _seg(), "B": _seg()}),
            ("A", "B"),
        )
        short = _report(
            _trades("A", 400) + _trades("B", 29),
            _expectation({"A": _seg(), "B": _seg()}),
            ("A", "B"),
        )
        assert _check(enough, check).metrics["segments"]["B"]["status"] == "ok"
        assert _check(short, check).metrics["segments"]["B"]["status"] == "insufficient"

    def test_a_promise_of_nothing_in_the_book_does_not_silence_the_segments_that_made_one(
        self,
    ) -> None:
        # The book as a whole earned -0.01 in the backtest, but segment A earned +0.20: the
        # check can run on A, and it must (a segment B that promised nothing is not assessed).
        expectation = BacktestExpectation(
            800, 1.0, -0.01, 0.6, 0.2, 0.1,
            by_segment={"A": _seg(), "B": _seg(mean=-0.05)},
        )  # fmt: skip
        trades = _trades("A", 60) + _trades("B", 60, pnl=-0.05)
        pnl = _check(_report(trades, expectation, ("A", "B")), "pnl")
        assert pnl.cause != "invalid_expectation"
        assert pnl.status == "ok"
        assert pnl.metrics["segments"]["A"]["status"] == "ok"
        assert pnl.metrics["segments"]["B"]["status"] == "insufficient"

    def test_too_few_live_trades_is_listed_not_judged(self) -> None:
        trades = _trades("A", 400) + _trades("B", 10, pnl=-1.0)
        report = _report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B"))
        pnl = _check(report, "pnl")
        b = pnl.metrics["segments"]["B"]
        assert b["status"] == "insufficient"
        assert b["cause"] == "too_few_samples"
        assert pnl.status == "ok"  # A is conclusive; B does not make the check insufficient
        assert any(r.startswith("not assessed:") and "segment B" in r for r in pnl.reasons)
        assert report.render() == ""  # and a blind spot alone does not make a run noisy

    def test_verbose_shows_the_blind_spot(self) -> None:
        trades = _trades("A", 60) + _trades("B", 10)
        report = _report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B"))
        assert "not assessed: segment B" in report.render(verbose=True)

    def test_a_segment_with_no_backtest_expectation_is_listed(self) -> None:
        trades = _trades("A", 60) + _trades("B", 60, pnl=-5.0)
        report = _report(trades, _expectation({"A": _seg(), "B": None}), ("A", "B"))
        pnl = _check(report, "pnl")
        assert pnl.metrics["segments"]["B"]["status"] == "insufficient"
        assert pnl.metrics["segments"]["B"]["cause"] == "no_expectation"
        assert "no backtest expectation" in " ".join(pnl.reasons)

    def test_a_segment_missing_from_the_record_is_the_same_as_an_absent_one(self) -> None:
        trades = _trades("A", 60) + _trades("B", 60)
        report = _report(trades, _expectation({"A": _seg()}), ("A", "B"))
        assert _check(report, "pnl").metrics["segments"]["B"]["cause"] == "no_expectation"

    def test_nothing_conclusive_anywhere_is_insufficient(self) -> None:
        trades = _trades("A", 10) + _trades("B", 10)
        report = _report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B"))
        pnl = _check(report, "pnl")
        assert pnl.status == "insufficient"
        assert pnl.cause == "too_few_samples"

    def test_a_thin_segment_does_not_hide_a_bleeding_one(self) -> None:
        trades = _trades("A", 10) + _trades("B", 60, pnl=-0.6)
        report = _report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B"))
        assert _check(report, "pnl").status == "alert"

    def test_many_blind_spots_are_summarised_in_the_text_and_complete_in_the_json(self) -> None:
        names = tuple(f"T{i:02d}" for i in range(12))
        trades = _trades("A", 60) + [t for n in names for t in _trades(n, 5)]
        expectation = _expectation({s: _seg() for s in ("A", *names)})
        pnl = _check(_report(trades, expectation, ("A", *names)), "pnl")
        note = next(r for r in pnl.reasons if r.startswith("not assessed:"))
        assert "and 7 more" in note
        assert len(pnl.metrics["segments"]) == 13
        assert note.count("segment T") == 5

    def test_the_thin_segments_trades_still_count_in_the_aggregate(self) -> None:
        trades = _trades("A", 60) + _trades("B", 10)
        pnl = _check(_report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B")), "pnl")
        assert pnl.metrics["n_trades"] == 70

    def test_a_segment_with_no_live_trades_is_not_listed(self) -> None:
        trades = _trades("A", 60)
        pnl = _check(_report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B")), "pnl")
        assert set(pnl.metrics["segments"]) == {"A"}


class TestWhenSegmentsAreNotJudged:
    def test_a_champion_with_no_per_segment_record_is_judged_in_aggregate_only(self) -> None:
        trades = _trades("A", 60) + _trades("B", 60, pnl=-0.6)
        report = _report(trades, _expectation(None), ("A", "B"))
        pnl = _check(report, "pnl")
        assert "segments" not in pnl.metrics
        assert pnl.metrics["segments_recorded"] is False

    def test_a_recorded_champion_says_so(self) -> None:
        pnl = _check(
            _report(_trades("A", 60), _expectation({"A": _seg(), "B": _seg()}), "AB"), "pnl"
        )
        assert pnl.metrics["segments_recorded"] is True

    def test_one_allowed_segment_is_the_whole_book_and_is_not_repeated(self) -> None:
        pnl = _check(_report(_trades("A", 60), _expectation({"A": _seg()}), ("A",)), "pnl")
        assert "segments" not in pnl.metrics

    def test_a_trade_outside_the_allowed_segments_is_still_excluded(self) -> None:
        trades = _trades("A", 60) + _trades("Z", 60, pnl=-9.0)
        pnl = _check(_report(trades, _expectation({"A": _seg(), "B": _seg()}), ("A", "B")), "pnl")
        assert pnl.metrics["n_out_of_segment"] == 60
        assert "Z" not in pnl.metrics.get("segments", {})


class TestASegmentIsHeldToItsOwnMixOfSides:
    def _expectation(self) -> BacktestExpectation:
        a = _seg(
            n=700,
            mean=0.157,
            by_direction={
                "yes": GroupExpectation(400, 0.40, 0.60, 0.20, 0.10),
                "no": GroupExpectation(300, 0.05, 0.50, 0.15, 0.08),
            },
        )
        return _expectation(
            {"A": a, "B": _seg()},
            by_direction={
                "yes": GroupExpectation(500, 0.30, 0.60, 0.20, 0.10),
                "no": GroupExpectation(300, 0.05, 0.50, 0.15, 0.08),
            },
        )

    def test_a_different_mix_than_the_segments_backtest_is_not_a_shortfall(self) -> None:
        # 20 YES and 160 NO, each exactly as the segment promised for that side.
        trades = _trades("A", 20, pnl=0.40, side="yes") + _trades(
            "A", 160, pnl=0.05, pnl_sd=0.50, side="no", brier=0.15, brier_sd=0.08
        )
        trades += _trades("B", 60)
        pnl = _check(_report(trades, self._expectation(), ("A", "B")), "pnl")
        a = pnl.metrics["segments"]["A"]
        assert a["status"] == "ok"
        assert a["mean_backtest"] == pytest.approx((20 * 0.40 + 160 * 0.05) / 180)

    def test_a_side_the_segment_never_promised_is_left_out_of_that_segment(self) -> None:
        a = _seg(n=400, by_direction={"yes": GroupExpectation(400, 0.20, 0.60, 0.20, 0.10)})
        expectation = _expectation(
            {"A": a, "B": _seg()},
            by_direction={
                "yes": GroupExpectation(500, 0.20, 0.60, 0.20, 0.10),
                "no": GroupExpectation(300, 0.05, 0.50, 0.15, 0.08),
            },
        )
        trades = _trades("A", 60) + _trades("A", 40, pnl=-5.0, side="no") + _trades("B", 60)
        pnl = _check(_report(trades, expectation, ("A", "B")), "pnl")
        assert pnl.metrics["segments"]["A"]["n_trades"] == 60


class TestAnUnusableSegmentExpectationIsNeverRead_AsFine:
    def test_a_non_finite_number_in_a_segment_makes_the_check_insufficient(self) -> None:
        expectation = _expectation({"A": _seg(), "B": _seg()})
        assert expectation.by_segment is not None
        b = expectation.by_segment["B"]
        assert b is not None
        object.__setattr__(b, "std_pnl_per_stake", math.nan)
        trades = _trades("A", 60) + _trades("B", 60)
        pnl = _check(_report(trades, expectation, ("A", "B")), "pnl")
        assert pnl.status == "insufficient"
        assert pnl.cause == "invalid_expectation"
        assert "segment B" in pnl.reasons[0]


class TestThePolicyIsWrittenDownAndSurvivesTheJson:
    def test_the_report_round_trips_with_segments(self) -> None:
        trades = [t for s in _ABC for t in _trades(s, 60)] + _trades("B", 5)
        report = _report(trades, _three(), _ABC)
        data = json.loads(json.dumps(report.to_dict(), allow_nan=False))
        assert HealthReport.from_dict(data) == report

    def test_segment_names_are_text_keys(self) -> None:
        trades = _trades("7", 60) + _trades("8", 60)
        expectation = _expectation({"7": _seg(), "8": _seg()})
        pnl = _check(_report(trades, expectation, ("7", "8")), "pnl")
        assert set(pnl.metrics["segments"]) == {"7", "8"}


def _aggregate_champion(expectation: BacktestExpectation) -> ChampionManifest:
    """Promoted on `HoldoutGateResult.evaluate(trades)`: one aggregate segment, "all"."""
    gate = HoldoutGateResult(
        decision="GO",
        segment_results=[SegmentGateResult("all", 300, 30.0, True, "ok")],
        aggregate_pnl=30.0,
        aggregate_trades=300,
    )
    return ChampionManifest(
        model_name="champ",
        model_path=Path("champion.pkl"),
        calibrator_path=None,
        gate=gate,
        published_at=_PUBLISHED,
        plugin_family="politics",
        expectation=expectation,
    )


def _aggregate_report(
    trades: list[dict[str, Any]], expectation: BacktestExpectation
) -> HealthReport:
    return health_report(
        champion=_aggregate_champion(expectation),
        trades=trades,
        drift=None,
        as_of=_AS_OF,
        checks=("calibration", "pnl"),
    )


class TestAChampionPromotedOnAnAggregateGate:
    """allowed_segments == {"all"} means every segment, here as in the broker."""

    def test_trades_in_every_segment_are_evidence(self) -> None:
        trades = _trades("Chicago", 40) + _trades("Dallas", 40) + _trades("Miami", 40)
        pnl = _check(_aggregate_report(trades, _expectation(None)), "pnl")
        assert pnl.metrics["n_out_of_segment"] == 0
        assert pnl.metrics["n_trades"] == 120
        assert pnl.metrics["allowed_segments"] == ["all"]
        assert pnl.status == "ok"

    def test_calibration_sees_them_too(self) -> None:
        trades = _trades("Chicago", 40) + _trades("Dallas", 40)
        calibration = _check(_aggregate_report(trades, _expectation(None)), "calibration")
        assert calibration.metrics["n_out_of_segment"] == 0
        assert calibration.metrics["n_trades"] == 80

    def test_a_segment_named_all_beside_others_does_not_make_the_champion_an_aggregate_one(
        self,
    ) -> None:
        trades = _trades("A", 40) + _trades("all", 40) + _trades("Chicago", 40)
        pnl = _check(
            _report(trades, _expectation({"A": _seg(), "all": _seg()}), ("A", "all")), "pnl"
        )
        assert pnl.metrics["n_out_of_segment"] == 40  # Chicago was graded in neither
        assert pnl.metrics["n_trades"] == 80

    def test_a_named_segment_champion_still_leaves_other_segments_out(self) -> None:
        trades = _trades("A", 40) + _trades("Chicago", 40)
        pnl = _check(_report(trades, _expectation(None), ("A",)), "pnl")
        assert pnl.metrics["n_out_of_segment"] == 40

    def test_segments_are_judged_when_the_backtest_recorded_them(self) -> None:
        expectation = _expectation({"Chicago": _seg(), "Dallas": _seg()})
        trades = _trades("Chicago", 400) + _trades("Dallas", 40, pnl=-0.40)
        pnl = _check(_aggregate_report(trades, expectation), "pnl")
        assert pnl.status == "alert"
        assert pnl.reasons[0].startswith("segment Dallas:")

    def test_a_backtest_that_only_knew_the_aggregate_is_not_judged_per_segment(self) -> None:
        expectation = _expectation({"all": _seg()})
        trades = _trades("Chicago", 60) + _trades("Dallas", 60)
        pnl = _check(_aggregate_report(trades, expectation), "pnl")
        assert "segments" not in pnl.metrics
        assert pnl.metrics["segments_recorded"] is False
        assert pnl.status == "ok"

    def test_the_wildcard_does_not_swallow_an_empty_book(self) -> None:
        pnl = _check(_aggregate_report([], _expectation(None)), "pnl")
        assert pnl.status == "insufficient"
        assert pnl.metrics["n_total"] == 0


class TestWhatASegmentDoesNotCover:
    """A segment is held to its own record. Trades on a side the segment never promised are
    not part of its comparison (the book judges them against the book's record for that side),
    and the segment says how many there were."""

    def test_the_segment_counts_the_trades_it_could_not_hold_to_its_own_record(self) -> None:
        a = _seg(n=400, by_direction={"yes": GroupExpectation(400, 0.20, 0.60, 0.20, 0.10)})
        expectation = _expectation(
            {"A": a, "B": _seg()},
            by_direction={
                "yes": GroupExpectation(500, 0.20, 0.60, 0.20, 0.10),
                "no": GroupExpectation(300, 0.05, 0.50, 0.15, 0.08),
            },
        )
        trades = _trades("A", 60) + _trades("A", 40, pnl=0.05, side="no") + _trades("B", 60)
        for name in ("pnl", "calibration"):
            segments = _check(_report(trades, expectation, ("A", "B")), name).metrics["segments"]
            assert segments["A"]["n_trades"] == 60
            assert segments["A"]["n_not_referenced"] == 40
            assert segments["B"]["n_not_referenced"] == 0
