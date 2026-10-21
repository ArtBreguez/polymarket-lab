"""The watchdog holds live YES and live NO trades to the backtest's YES and NO.

A champion promoted on a live-policy backtest records an expectation per side. The
PnL and calibration checks then:

- compare the live book with the backtest's per-side means *weighted by the live mix*,
  so a book that trades more NO than the backtest did is not mistaken for a shortfall;
- judge each side on its own as well, so a side that bleeds while the other side hides
  it in the aggregate is named;
- fall back to the old behaviour for a champion with no per-side record: its NO trades
  are left out of the PnL comparison and counted, never silently trusted.

Fixtures use two-point samples (`mean +/- sd`, alternating), so the statistic a test
relies on is known in advance rather than hoped for.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats as sp

from pmlab.backtest.expectation import BacktestExpectation, GroupExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring.watchdog import (
    CalibrationThresholds,
    HealthReport,
    PnLThresholds,
    WatchdogConfig,
    health_report,
)

_PUBLISHED = "2026-01-01T00:00:00+00:00"
_AS_OF = "2026-03-01T00:00:00+00:00"


def _group(n: int, mean: float, std: float, brier: float, brier_std: float) -> GroupExpectation:
    return GroupExpectation(n, mean, std, brier, brier_std)


def _expectation(
    yes: GroupExpectation | None = None, no: GroupExpectation | None = None, *, split: bool = True
) -> BacktestExpectation:
    yes = yes or _group(400, 0.20, 0.60, 0.20, 0.10)
    no = no or _group(300, 0.10, 0.50, 0.15, 0.08)
    n = yes.n_trades + no.n_trades
    mean = (yes.n_trades * yes.mean_pnl_per_stake + no.n_trades * no.mean_pnl_per_stake) / n
    brier = ((yes.brier_score or 0) * yes.n_trades + (no.brier_score or 0) * no.n_trades) / n
    return BacktestExpectation(
        n_trades=n,
        flat_stake=1.0,
        mean_pnl_per_stake=mean,
        std_pnl_per_stake=0.6,
        brier_score=brier,
        brier_std=0.1,
        by_direction={"yes": yes, "no": no} if split else None,
    )


def _champion(expectation: BacktestExpectation) -> ChampionManifest:
    gate = HoldoutGateResult(
        decision="GO",
        segment_results=[SegmentGateResult("A", 100, 10.0, True, "ok")],
        aggregate_pnl=10.0,
        aggregate_trades=100,
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


def _side(
    direction: str,
    n: int,
    *,
    pnl: float,
    pnl_sd: float,
    brier: float = 0.15,
    brier_sd: float = 0.05,
    **extra: Any,
) -> list[dict[str, Any]]:
    """`n` settled trades of one side with the given mean/spread of PnL and Brier term."""
    rows = []
    for term, ret in zip(_two_point(n, brier, brier_sd), _two_point(n, pnl, pnl_sd), strict=True):
        row = {
            "recorded_at": "2026-02-01T10:00:00+00:00",
            "city_or_segment": "A",
            "target_date": "2026-02-05",
            "outcome_label": "YES",
            "direction": direction,
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
    trades: list[dict[str, Any]], expectation: BacktestExpectation | None = None, **kwargs: Any
) -> HealthReport:
    return health_report(
        champion=_champion(expectation or _expectation()),
        trades=trades,
        drift=None,
        as_of=_AS_OF,
        checks=kwargs.pop("checks", ("calibration", "pnl")),
        **kwargs,
    )


def _check(report: HealthReport, name: str):
    return next(c for c in report.checks if c.name == name)


def _z(live: list[float], ref_mean: float, ref_sd: float, ref_n: int, *, worse_high: bool) -> float:
    arr = np.array(live)
    se = math.sqrt(arr.var(ddof=1) / len(arr) + ref_sd**2 / ref_n)
    diff = arr.mean() - ref_mean
    return float((diff if worse_high else -diff) / se)


# A book exactly as the backtest promised, side by side.
def _as_promised(n_yes: int, n_no: int) -> list[dict[str, Any]]:
    return _side("yes", n_yes, pnl=0.20, pnl_sd=0.60, brier=0.20, brier_sd=0.10) + _side(
        "no", n_no, pnl=0.10, pnl_sd=0.50, brier=0.15, brier_sd=0.08
    )


class TestEachSideIsHeldToItsOwnExpectation:
    def test_a_book_as_promised_on_both_sides_is_healthy(self) -> None:
        report = _report(_as_promised(80, 60))
        assert report.status == "ok"
        pnl = _check(report, "pnl")
        assert pnl.metrics["n_trades"] == 140
        assert pnl.metrics["n_by_direction"] == {"yes": 80, "no": 60}
        assert pnl.metrics["n_excluded_no_side"] == 0

    def test_no_trades_now_count_toward_the_pnl_evidence(self) -> None:
        pnl = _check(_report(_side("no", 60, pnl=0.10, pnl_sd=0.50)), "pnl")
        assert pnl.status == "ok"
        assert pnl.metrics["n_trades"] == 60
        assert pnl.metrics["n_excluded_no_side"] == 0

    def test_the_reference_is_the_per_side_means_weighted_by_the_live_mix(self) -> None:
        # 3 YES : 1 NO live. Reference = 0.75 * 0.20 + 0.25 * 0.10 = 0.175.
        report = _report(_as_promised(75, 25))
        pnl = _check(report, "pnl")
        assert pnl.metrics["mean_backtest"] == pytest.approx(0.175)
        live = [t["realized_pnl"] for t in _as_promised(75, 25)]
        assert pnl.metrics["mean_live"] == pytest.approx(float(np.mean(live)))

    def test_the_z_uses_the_mixture_variance_of_the_reference(self) -> None:
        trades = _as_promised(75, 25)
        live = [t["realized_pnl"] for t in trades]
        ref_var = 0.75**2 * 0.60**2 / 400 + 0.25**2 * 0.50**2 / 300
        arr = np.array(live)
        expected = (0.175 - arr.mean()) / math.sqrt(arr.var(ddof=1) / len(arr) + ref_var)
        assert _check(_report(trades), "pnl").metrics["z"] == pytest.approx(expected)

    def test_a_different_mix_than_the_backtest_is_not_a_shortfall(self) -> None:
        """YES earns 0.40 and NO 0.05. A book of 20 YES and 160 NO that earns exactly that
        has a mean of 0.089; the pooled backtest mean (0.25) would call it a 65% shortfall at
        z > 3, an alert. Each side is as promised, so it must be quiet."""
        expectation = _expectation(
            yes=_group(400, 0.40, 0.60, 0.20, 0.10), no=_group(300, 0.05, 0.50, 0.15, 0.08)
        )
        trades = _side("yes", 20, pnl=0.40, pnl_sd=0.60) + _side("no", 160, pnl=0.05, pnl_sd=0.50)
        pooled = BacktestExpectation(
            n_trades=700,
            flat_stake=1.0,
            mean_pnl_per_stake=expectation.mean_pnl_per_stake,
            std_pnl_per_stake=0.6,
        )
        live = np.array([t["realized_pnl"] for t in trades])
        pooled_z = _z(live.tolist(), pooled.mean_pnl_per_stake, 0.6, 700, worse_high=False)
        assert pooled_z > 3.0  # the false alarm this guards against, were the mix ignored
        report = _report(trades, expectation)
        assert _check(report, "pnl").status == "ok"

    def test_the_old_comparison_would_have_missed_a_real_shortfall_in_the_mix(self) -> None:
        # The mirror image: mostly YES live, each side exactly as promised. Vs the backtest's
        # pooled mean (0.157) the book looks better than promised; vs its own mix it is right.
        report = _report(_as_promised(300, 20))
        assert _check(report, "pnl").metrics["shortfall_rel"] == pytest.approx(0.0, abs=0.02)


class TestASideThatBleedsIsNamed:
    """The aggregate looks fine; one side is not."""

    def _bleeding_no(self) -> list[dict[str, Any]]:
        # NO is promised at +0.10 and earns -0.25 on 40 trades; 300 healthy YES trades hide it
        # in the aggregate (shortfall about 22%).
        return _side("yes", 300, pnl=0.20, pnl_sd=0.60) + _side("no", 40, pnl=-0.25, pnl_sd=0.50)

    def test_the_aggregate_alone_is_quiet(self) -> None:
        pnl = _check(_report(self._bleeding_no()), "pnl")
        overall = pnl.metrics["shortfall_rel"]
        assert overall < 0.5
        assert pnl.metrics["z"] < 2.0

    def test_the_check_still_alerts_and_names_the_no_side(self) -> None:
        report = _report(self._bleeding_no())
        pnl = _check(report, "pnl")
        assert pnl.status == "alert"
        assert pnl.reasons[0].startswith("NO side:")
        assert "shortfall" in pnl.reasons[0]
        assert report.exit_code == 20
        assert "NO side" in report.render()

    def test_the_other_side_is_reported_healthy(self) -> None:
        pnl = _check(_report(self._bleeding_no()), "pnl")
        assert pnl.metrics["directions"]["yes"]["status"] == "ok"
        assert pnl.metrics["directions"]["no"]["status"] == "alert"

    def test_the_yes_side_can_be_the_one_that_bleeds(self) -> None:
        trades = _side("yes", 80, pnl=-0.40, pnl_sd=0.60) + _side("no", 300, pnl=0.10, pnl_sd=0.50)
        pnl = _check(_report(trades), "pnl")
        assert pnl.status == "alert"
        assert pnl.reasons[0].startswith("YES side:")

    def test_per_side_metrics_are_complete_and_json_safe(self) -> None:
        pnl = _check(_report(self._bleeding_no()), "pnl")
        no = pnl.metrics["directions"]["no"]
        assert no["n_trades"] == 40
        assert no["mean_live"] == pytest.approx(-0.25)
        assert no["mean_backtest"] == pytest.approx(0.10)
        assert no["shortfall_rel"] == pytest.approx(3.5)
        expected_z = _z(
            [t["realized_pnl"] for t in self._bleeding_no() if t["direction"] == "no"],
            0.10,
            0.50,
            300,
            worse_high=False,
        )
        assert no["z"] == pytest.approx(expected_z)
        assert no["p_value"] == pytest.approx(sp.norm.sf(expected_z))
        json.dumps(pnl.metrics, allow_nan=False)

    def test_the_two_sides_are_a_family_and_holm_adjusts_their_p_values(self) -> None:
        pnl = _check(_report(self._bleeding_no()), "pnl")
        yes, no = pnl.metrics["directions"]["yes"], pnl.metrics["directions"]["no"]
        # Smallest p gets x2, the other x1 (never below the first's adjusted value).
        assert no["p_adjusted"] == pytest.approx(min(1.0, 2 * no["p_value"]))
        assert yes["p_adjusted"] == pytest.approx(max(yes["p_value"], no["p_adjusted"]))
        assert no["z_adjusted"] == pytest.approx(sp.norm.isf(no["p_adjusted"]))

    def test_a_single_side_is_not_a_family_and_is_not_reported_twice(self) -> None:
        pnl = _check(_report(_side("no", 60, pnl=0.10, pnl_sd=0.5)), "pnl")
        assert "directions" not in pnl.metrics

    def test_a_warn_level_shortfall_on_one_side_is_a_warn(self) -> None:
        # NO at -0.10 against +0.10 (shortfall 200%) on 40 trades: z = 2.37, adjusted ~2.1.
        trades = _side("yes", 300, pnl=0.20, pnl_sd=0.60) + _side("no", 40, pnl=-0.10, pnl_sd=0.50)
        pnl = _check(_report(trades), "pnl")
        assert pnl.status == "warn"
        assert pnl.reasons[0].startswith("NO side:")


class TestASideWithTooFewTradesIsNotAssessedButDoesNotBlock:
    def test_the_side_is_insufficient_and_the_check_is_not(self) -> None:
        trades = _as_promised(80, 10)
        pnl = _check(_report(trades), "pnl")
        assert pnl.status == "ok"
        assert pnl.metrics["directions"]["no"]["status"] == "insufficient"
        assert pnl.metrics["directions"]["no"]["cause"] == "too_few_samples"
        assert any("not assessed" in r and "NO side" in r for r in pnl.reasons)

    def test_those_trades_still_count_in_the_aggregate(self) -> None:
        pnl = _check(_report(_as_promised(80, 10)), "pnl")
        assert pnl.metrics["n_trades"] == 90

    def test_a_degraded_side_with_enough_trades_alerts_beside_a_thin_side(self) -> None:
        trades = _side("yes", 80, pnl=-0.40, pnl_sd=0.60) + _side("no", 10, pnl=0.10, pnl_sd=0.5)
        pnl = _check(_report(trades), "pnl")
        assert pnl.status == "alert"
        assert pnl.reasons[0].startswith("YES side:")


class TestWithoutAPerSideExpectation:
    """A champion published before the split: the old, YES-only contract."""

    def test_no_trades_are_left_out_and_counted(self) -> None:
        old = _expectation(split=False)
        trades = _side("yes", 60, pnl=0.157, pnl_sd=0.6) + _side("no", 40, pnl=-1.0, pnl_sd=0.0)
        pnl = _check(_report(trades, old), "pnl")
        assert pnl.metrics["n_trades"] == 60
        assert pnl.metrics["n_excluded_no_side"] == 40
        assert pnl.status == "ok"  # the losing NO book cannot page it, and cannot hide in it

    def test_the_reason_says_the_backtest_only_buys_yes(self) -> None:
        pnl = _check(
            _report(_side("no", 60, pnl=0.1, pnl_sd=0.5), _expectation(split=False)), "pnl"
        )
        assert pnl.status == "insufficient"
        assert "only buys YES" in " ".join(pnl.reasons)

    def test_calibration_keeps_pooling_every_trade_as_before(self) -> None:
        old = _expectation(split=False)
        trades = _side("yes", 40, pnl=0.1, pnl_sd=0.5) + _side("no", 40, pnl=0.1, pnl_sd=0.5)
        cal = _check(_report(trades, old), "calibration")
        assert cal.metrics["n_trades"] == 80
        assert "n_excluded_no_side" not in cal.metrics


class TestAYesOnlyPolicyChampion:
    """A split that records YES only: NO was not part of the promise."""

    def _yes_only(self) -> BacktestExpectation:
        yes = _group(400, 0.20, 0.60, 0.20, 0.10)
        return BacktestExpectation(400, 1.0, 0.20, 0.60, 0.20, 0.10, by_direction={"yes": yes})

    def test_live_no_trades_are_left_out_of_pnl_and_calibration_and_counted(self) -> None:
        trades = _side("yes", 60, pnl=0.20, pnl_sd=0.60, brier=0.20, brier_sd=0.10) + _side(
            "no", 40, pnl=-1.0, pnl_sd=0.0
        )
        report = _report(trades, self._yes_only())
        pnl, cal = _check(report, "pnl"), _check(report, "calibration")
        assert pnl.metrics["n_trades"] == cal.metrics["n_trades"] == 60
        assert pnl.metrics["n_excluded_no_side"] == cal.metrics["n_excluded_no_side"] == 40
        assert report.status == "ok"

    def test_yes_trades_are_judged_as_before(self) -> None:
        trades = _side("yes", 60, pnl=-0.20, pnl_sd=0.60)
        assert _check(_report(trades, self._yes_only()), "pnl").status == "alert"


class TestANoOnlyPolicyChampion:
    def test_live_yes_trades_are_counted_out(self) -> None:
        no = _group(300, 0.10, 0.50, 0.15, 0.08)
        expectation = BacktestExpectation(300, 1.0, 0.10, 0.50, 0.15, 0.08, by_direction={"no": no})
        trades = _side("no", 60, pnl=0.10, pnl_sd=0.5) + _side("yes", 20, pnl=-1.0, pnl_sd=0.0)
        pnl = _check(_report(trades, expectation), "pnl")
        assert pnl.metrics["n_trades"] == 60
        assert pnl.metrics["n_excluded_yes_side"] == 20
        assert pnl.status == "ok"


class TestNoTradesSettledByAnOlderReleaseAreNotTrusted:
    """Before 0.9.1 a NO trade was settled as a short YES at the NO price: its realized_pnl
    is wrong, and a perfectly good NO book would read as a shortfall (or a lucky one as a
    beat). `settlement_version` says which arithmetic booked it."""

    def test_a_no_trade_without_a_settlement_version_is_left_out_and_counted(self) -> None:
        trades = _side("yes", 60, pnl=0.20, pnl_sd=0.60) + [
            {k: v for k, v in t.items() if k != "settlement_version"}
            for t in _side("no", 40, pnl=-1.0, pnl_sd=0.0)
        ]
        pnl = _check(_report(trades), "pnl")
        assert pnl.metrics["n_trades"] == 60
        assert pnl.metrics["n_excluded_unfixed_no"] == 40
        assert pnl.status == "ok"
        assert "pmlab resettle" in " ".join(pnl.reasons)

    def test_calibration_does_not_care_how_the_pnl_was_settled(self) -> None:
        trades = [
            {k: v for k, v in t.items() if k != "settlement_version"}
            for t in _side("no", 60, pnl=-1.0, pnl_sd=0.0)
        ]
        assert _check(_report(trades), "calibration").metrics["n_trades"] == 60

    def test_a_yes_trade_never_needed_the_version(self) -> None:
        trades = [
            {k: v for k, v in t.items() if k != "settlement_version"}
            for t in _side("yes", 60, pnl=0.20, pnl_sd=0.60)
        ]
        assert _check(_report(trades), "pnl").metrics["n_trades"] == 60

    def test_a_current_version_is_trusted(self) -> None:
        pnl = _check(_report(_side("no", 60, pnl=0.10, pnl_sd=0.5, settlement_version=1)), "pnl")
        assert pnl.metrics["n_excluded_unfixed_no"] == 0
        assert pnl.metrics["n_trades"] == 60

    @pytest.mark.parametrize("version", [0, None, "1", True])
    def test_a_version_that_is_not_a_number_at_least_one_is_not_trusted(
        self, version: object
    ) -> None:
        pnl = _check(
            _report(_side("no", 60, pnl=0.10, pnl_sd=0.5, settlement_version=version)), "pnl"
        )
        assert pnl.metrics["n_excluded_unfixed_no"] == 60


class TestTheNoSideSettledBasisAndCosts:
    def test_a_no_trade_is_compared_to_the_no_quote_not_the_yes_quote(self) -> None:
        """Settled at the quote with a fill that equals the NO quote (1 - gamma): consistent."""
        trades = _side("no", 60, pnl=0.10, pnl_sd=0.5, gamma_price=0.30, fill_price=0.70)
        for t in trades:
            del t["settled_basis"]  # a log from before basis was written
        pnl = _check(_report(trades), "pnl")
        assert pnl.metrics["n_excluded_legacy_settlement"] == 0
        assert pnl.metrics["n_trades"] == 60

    def test_a_no_trade_with_a_slipped_fill_and_no_basis_is_excluded_as_legacy(self) -> None:
        trades = _side("no", 60, pnl=0.10, pnl_sd=0.5, gamma_price=0.30, fill_price=0.72)
        for t in trades:
            del t["settled_basis"]
        assert _check(_report(trades), "pnl").metrics["n_excluded_legacy_settlement"] == 60


class TestCalibrationPerSide:
    def test_a_book_as_promised_is_healthy(self) -> None:
        cal = _check(_report(_as_promised(80, 60)), "calibration")
        assert cal.status == "ok"
        assert cal.metrics["brier_backtest"] == pytest.approx((80 * 0.20 + 60 * 0.15) / 140)

    def test_a_different_mix_is_not_a_calibration_gap(self) -> None:
        """Live is YES only at exactly the backtest's YES Brier (0.20). The pooled backtest
        Brier is 0.179: a 0.021 gap, a warn on abs (0.02) and rel (12%) at large n."""
        trades = _side("yes", 600, pnl=0.20, pnl_sd=0.60, brier=0.20, brier_sd=0.10)
        cal = _check(_report(trades), "calibration")
        assert cal.status == "ok"
        assert cal.metrics["brier_backtest"] == pytest.approx(0.20)

    def test_the_side_that_is_miscalibrated_is_named(self) -> None:
        trades = _side("yes", 300, pnl=0.20, pnl_sd=0.60, brier=0.20, brier_sd=0.10) + _side(
            "no", 60, pnl=0.10, pnl_sd=0.50, brier=0.35, brier_sd=0.08
        )
        cal = _check(_report(trades), "calibration")
        assert cal.status == "alert"
        assert cal.reasons[0].startswith("NO side:")
        assert cal.metrics["directions"]["no"]["status"] == "alert"
        assert cal.metrics["directions"]["yes"]["status"] == "ok"

    def test_per_side_brier_metrics(self) -> None:
        trades = _side("yes", 300, pnl=0.2, pnl_sd=0.6, brier=0.20, brier_sd=0.10) + _side(
            "no", 60, pnl=0.1, pnl_sd=0.5, brier=0.35, brier_sd=0.08
        )
        no = _check(_report(trades), "calibration").metrics["directions"]["no"]
        terms = np.array(_two_point(60, 0.35, 0.08))
        expected_z = (terms.mean() - 0.15) / math.sqrt(terms.var(ddof=1) / 60 + 0.08**2 / 300)
        assert no["brier_live"] == pytest.approx(0.35)
        assert no["brier_backtest"] == pytest.approx(0.15)
        assert no["degradation_abs"] == pytest.approx(0.20)
        assert no["z"] == pytest.approx(expected_z)

    def test_the_forecast_is_scored_against_the_label_whichever_side_was_bought(self) -> None:
        """A NO trade on a label that did not occur, forecast 0.1: a good forecast. `outcome`
        is "lost" here because the label did not occur; the term is (0.1 - 0)**2."""
        trades = [
            {
                "recorded_at": "2026-02-01T10:00:00+00:00",
                "city_or_segment": "A",
                "target_date": "2026-02-05",
                "direction": "no",
                "model_prob": 0.1,
                "flat_stake": 1.0,
                "outcome": "lost",  # the label did not occur
                "realized_pnl": 1.0,  # and the NO bet won
                "settlement_version": 1,
            }
            for _ in range(40)
        ]
        cal = _check(_report(trades), "calibration")
        assert cal.metrics["brier_live"] == pytest.approx(0.01)

    def test_and_a_no_trade_on_a_label_that_occurred_is_a_miss(self) -> None:
        trades = [
            {
                "recorded_at": "2026-02-01T10:00:00+00:00",
                "city_or_segment": "A",
                "target_date": "2026-02-05",
                "direction": "no",
                "model_prob": 0.1,
                "flat_stake": 1.0,
                "outcome": "won",  # the label occurred
                "realized_pnl": -1.0,  # and the NO bet lost
                "settlement_version": 1,
            }
            for _ in range(40)
        ]
        assert _check(_report(trades), "calibration").metrics["brier_live"] == pytest.approx(0.81)

    def test_a_thin_side_does_not_block_calibration(self) -> None:
        cal = _check(_report(_as_promised(80, 10)), "calibration")
        assert cal.status == "ok"
        assert cal.metrics["directions"]["no"]["status"] == "insufficient"


class TestTheReportRoundTripsAndStaysQuiet:
    def test_the_json_round_trips(self) -> None:
        report = _report(_as_promised(80, 60))
        again = HealthReport.from_dict(json.loads(json.dumps(report.to_dict(), allow_nan=False)))
        assert again == report

    def test_a_healthy_two_sided_book_prints_nothing(self) -> None:
        assert _report(_as_promised(80, 60)).render() == ""

    def test_verbose_shows_the_blind_spot(self) -> None:
        text = _report(_as_promised(80, 10)).render(verbose=True)
        assert "not assessed" in text
        assert "NO side" in text

    def test_a_blind_spot_alone_does_not_make_the_report_non_ok(self) -> None:
        assert _report(_as_promised(80, 10)).status == "ok"


class TestThresholdsApplyPerSide:
    def test_a_side_is_judged_against_the_configured_min_trades(self) -> None:
        config = WatchdogConfig(
            pnl=PnLThresholds(min_trades=5), calibration=CalibrationThresholds(min_trades=5)
        )
        trades = _side("yes", 80, pnl=0.20, pnl_sd=0.60) + _side("no", 10, pnl=-0.5, pnl_sd=0.5)
        pnl = _check(_report(trades, config=config), "pnl")
        assert pnl.metrics["directions"]["no"]["status"] != "insufficient"
        assert pnl.status in {"warn", "alert"}


class TestWhatTheSplitCannotSupport:
    def test_an_empty_split_means_no_per_side_expectation(self) -> None:
        # Both sides were too thin in the backtest to be promised anything.
        expectation = BacktestExpectation(24, 1.0, 0.1, 0.5, 0.2, 0.1, by_direction={})
        report = _report(_as_promised(40, 40), expectation)
        for name in ("pnl", "calibration"):
            check = _check(report, name)
            assert check.status == "insufficient"
            assert check.cause == "no_expectation"
            assert "per-side" in check.reasons[0]

    def test_a_mix_that_averages_to_nothing_is_not_measured_as_a_shortfall(self) -> None:
        # YES promises +0.20, NO promises -0.30: a book of NO alone has no positive promise.
        expectation = _expectation(
            yes=_group(400, 0.20, 0.60, 0.20, 0.10), no=_group(300, -0.30, 0.50, 0.15, 0.08)
        )
        pnl = _check(_report(_side("no", 60, pnl=-0.30, pnl_sd=0.5), expectation), "pnl")
        assert pnl.status == "insufficient"
        assert pnl.cause == "invalid_expectation"
        assert "this mix of sides" in pnl.reasons[0]

    def test_a_side_without_a_positive_promise_is_not_assessed_but_the_book_is(self) -> None:
        expectation = _expectation(
            yes=_group(400, 0.20, 0.60, 0.20, 0.10), no=_group(300, -0.05, 0.50, 0.15, 0.08)
        )
        trades = _side("yes", 80, pnl=0.20, pnl_sd=0.60) + _side("no", 40, pnl=-0.05, pnl_sd=0.5)
        pnl = _check(_report(trades, expectation), "pnl")
        assert pnl.status == "ok"
        assert pnl.metrics["directions"]["no"]["status"] == "insufficient"
        assert pnl.metrics["directions"]["no"]["cause"] == "invalid_expectation"
        assert pnl.metrics["directions"]["yes"]["status"] == "ok"

    def test_a_non_finite_number_in_one_side_is_refused_never_read_as_ok(self) -> None:
        expectation = _expectation()
        assert expectation.by_direction is not None
        object.__setattr__(expectation.by_direction["no"], "std_pnl_per_stake", math.nan)
        for name in ("pnl", "calibration"):
            if name == "calibration":
                object.__setattr__(expectation.by_direction["no"], "brier_std", math.nan)
            check = _check(_report(_as_promised(40, 40), expectation), name)
            assert check.status == "insufficient"
            assert check.cause == "invalid_expectation"
            assert "no." in check.reasons[0]

    def test_a_thin_book_is_insufficient_with_the_usual_reason(self) -> None:
        pnl = _check(_report(_as_promised(10, 10)), "pnl")
        assert pnl.status == "insufficient"
        assert pnl.cause == "too_few_samples"
