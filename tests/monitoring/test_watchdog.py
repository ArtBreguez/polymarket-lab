"""Tests for the watchdog core: `health_report`.

The watchdog consumes drift monitoring, live calibration and the champion's
backtest expectation; it does not recompute them. What it owns is the policy:
which numbers are bad enough to act on, what "we could not check" counts as, and
what a cron job should be told. So the suite is about decisions.

Fixtures are built from exact-moment, two-point samples (`mean +/- sd`, alternating)
so the statistic a test relies on is known in advance rather than hoped for.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from pmlab.backtest.expectation import BacktestExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.core.costs import CostModel
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring.drift import DriftConfig, DriftReport, DriftThresholds, drift_report
from pmlab.monitoring.watchdog import (
    CHECKS,
    EXIT_ALERT,
    EXIT_INSUFFICIENT,
    EXIT_INTERNAL,
    EXIT_OK,
    EXIT_USAGE,
    EXIT_WARN,
    CalibrationThresholds,
    HealthReport,
    PnLThresholds,
    WatchdogConfig,
    WatchdogInputError,
    _cost_basis,
    exit_code_for,
    health_report,
)

_PUBLISHED = "2026-01-01T00:00:00+00:00"
_AS_OF = "2026-03-01T00:00:00+00:00"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _expectation(**overrides: object) -> BacktestExpectation:
    base: dict[str, object] = {
        "n_trades": 2000,
        "flat_stake": 1.0,
        "mean_pnl_per_stake": 0.10,
        "std_pnl_per_stake": 0.30,
        "brier_score": 0.20,
        "brier_std": 0.10,
    }
    base.update(overrides)
    return BacktestExpectation(**base)  # type: ignore[arg-type]


def _champion(
    expectation: BacktestExpectation | None = None,
    *,
    costs: CostModel | None = None,
    published_at: str = _PUBLISHED,
    passing: tuple[str, ...] = ("A", "B"),
    failing: tuple[str, ...] = ("C",),
) -> ChampionManifest:
    segments = [SegmentGateResult(s, 100, 10.0, True, "ok") for s in passing] + [
        SegmentGateResult(s, 100, -10.0, False, "negative_pnl") for s in failing
    ]
    gate = HoldoutGateResult(
        decision="GO",
        segment_results=segments,
        aggregate_pnl=10.0 * len(passing),
        aggregate_trades=100 * len(passing),
    )
    return ChampionManifest(
        model_name="champ",
        model_path=Path("champion.pkl"),
        calibrator_path=None,
        gate=gate,
        published_at=published_at,
        plugin_family="politics",
        costs=costs,
        expectation=_expectation() if expectation is None else expectation,
    )


def _two_point(n: int, mean: float, sd: float) -> list[float]:
    """`mean +/- sd`, alternating: a known mean with a known spread."""
    return [mean + sd * (1 if i % 2 == 0 else -1) for i in range(n)]


def _trade(
    term: float,
    pnl: float,
    *,
    segment: str = "A",
    stake: float = 1.0,
    recorded_at: str = "2026-02-01T10:00:00+00:00",
    target_date: str = "2026-02-05",
    open_: bool = False,
    **extra: object,
) -> dict:
    """A settled trade whose squared forecast error is exactly `term`.

    The label did not occur and the forecast was sqrt(term), so (p - 0)^2 = term.
    """
    trade = {
        "recorded_at": recorded_at,
        "city_or_segment": segment,
        "target_date": target_date,
        "outcome_label": "YES",
        "direction": "yes",
        "gamma_price": 0.40,
        "model_prob": math.sqrt(term),
        "flat_stake": stake,
        "outcome": None if open_ else "lost",
        "realized_pnl": None if open_ else pnl * stake,
    }
    trade.update(extra)
    return trade


def _trades(
    n: int = 100,
    *,
    brier: float = 0.20,
    pnl: float = 0.10,
    brier_sd: float = 0.10,
    pnl_sd: float = 0.30,
    **kwargs: object,
) -> list[dict]:
    terms = _two_point(n, brier, brier_sd)
    pnls = _two_point(n, pnl, pnl_sd)
    return [_trade(t, p, **kwargs) for t, p in zip(terms, pnls, strict=True)]


def _drift(shift: float = 0.0, *, n: int = 2000, **kwargs: object) -> DriftReport:
    """`shift` 0.0 is ok, 0.03 warn, 0.08 alert on `feature_price`."""
    rng = np.random.default_rng(0)
    reference = {
        "feature_price": rng.normal(0.5, 0.1, n),
        "feature_volume": rng.lognormal(3, 0.5, n),
    }
    current = {
        "feature_price": rng.normal(0.5 + shift, 0.1, n),
        "feature_volume": rng.lognormal(3, 0.5, n),
    }
    return drift_report(reference, current, **kwargs)  # type: ignore[arg-type]


def _report(
    *,
    trades: list[dict] | None = None,
    drift: DriftReport | None = None,
    champion: ChampionManifest | None = None,
    config: WatchdogConfig | None = None,
    **kwargs: object,
) -> HealthReport:
    return health_report(
        champion=champion or _champion(),
        trades=_trades() if trades is None else trades,
        drift=_drift() if drift is None else drift,
        as_of=_AS_OF,
        config=config,
        **kwargs,  # type: ignore[arg-type]
    )


def _check(report: HealthReport, name: str):
    return next(c for c in report.checks if c.name == name)


def _welch_z(
    live: list[float], ref_mean: float, ref_sd: float, ref_n: int, *, higher_is_worse: bool
):
    arr = np.array(live)
    se = math.sqrt(arr.var(ddof=1) / len(arr) + ref_sd**2 / ref_n)
    diff = arr.mean() - ref_mean
    return (diff if higher_is_worse else -diff) / se


# --------------------------------------------------------------------------
# Overall status
# --------------------------------------------------------------------------


class TestOverall:
    def test_everything_within_bounds_is_healthy(self) -> None:
        report = _report()
        assert report.status == "ok"
        assert report.exit_code == EXIT_OK
        assert [c.status for c in report.checks] == ["ok", "ok", "ok"]
        assert report.problems == ()

    def test_checks_come_in_a_fixed_order(self) -> None:
        assert [c.name for c in _report().checks] == list(CHECKS) == ["drift", "calibration", "pnl"]

    @pytest.mark.parametrize(
        ("drift_shift", "calib", "pnl", "expected"),
        [
            (0.08, 0.20, 0.10, "alert"),  # drift alone
            (0.0, 0.27, 0.10, "alert"),  # calibration alone
            (0.0, 0.20, -0.02, "alert"),  # pnl alone
            (0.03, 0.20, 0.10, "warn"),  # drift warn alone
            (0.03, 0.27, 0.10, "alert"),  # warn + alert -> alert, not the average or the first
            (0.08, 0.23, 0.10, "alert"),  # alert + warn
        ],
    )
    def test_overall_is_the_worst_check(
        self, drift_shift: float, calib: float, pnl: float, expected: str
    ) -> None:
        report = _report(trades=_trades(brier=calib, pnl=pnl), drift=_drift(drift_shift))
        assert report.status == expected

    def test_one_healthy_check_does_not_outvote_a_failing_one(self) -> None:
        report = _report(drift=_drift(0.08))
        assert [c.status for c in report.checks] == ["alert", "ok", "ok"]
        assert report.status == "alert"
        assert report.exit_code == EXIT_ALERT

    def test_the_report_is_frozen(self) -> None:
        report = _report()
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.as_of = "x"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.checks[0].status = "alert"  # type: ignore[misc]

    def test_inputs_are_not_mutated(self) -> None:
        trades = _trades()
        before = copy.deepcopy(trades)
        _report(trades=trades)
        assert trades == before


# --------------------------------------------------------------------------
# Drift check
# --------------------------------------------------------------------------


class TestDriftCheck:
    def test_a_clean_report_is_ok(self) -> None:
        check = _check(_report(drift=_drift(0.0)), "drift")
        assert check.status == "ok"
        assert check.cause is None

    def test_an_alert_names_the_offending_feature_and_the_number(self) -> None:
        check = _check(_report(drift=_drift(0.08)), "drift")
        assert check.status == "alert"
        text = " ".join(check.reasons)
        assert "feature_price" in text
        assert "psi" in text
        assert "feature_volume" not in text, "a healthy feature must not be blamed"
        assert check.metrics["offending"] == ["feature_price"]
        assert check.metrics["max_psi"] > 0.25

    def test_a_warn_is_a_warn(self) -> None:
        check = _check(_report(drift=_drift(0.03)), "drift")
        assert check.status == "warn"
        assert check.metrics["offending"] == ["feature_price"]

    def test_the_ok_summary_counts_features_and_names_the_prediction_separately(self) -> None:
        """A prediction is not a feature: 2 features plus a prediction must not read as 3."""
        rng = np.random.default_rng(1)
        pred = rng.beta(5, 5, 2000)
        without = _check(_report(drift=_drift(0.0)), "drift")
        with_pred = _check(
            _report(drift=_drift(0.0, reference_predictions=pred, current_predictions=pred.copy())),
            "drift",
        )
        assert without.reasons[0].startswith("2 feature(s) within thresholds")
        assert with_pred.reasons[0].startswith("2 feature(s) and the prediction within thresholds")

    def test_the_status_is_the_drift_reports_own_classification(self) -> None:
        """Consume the classification; do not re-derive it from the numbers."""
        lenient = DriftConfig(
            thresholds=DriftThresholds(psi_warn=5, psi_alert=9, kl_warn=5, kl_alert=9)
        )
        report = _drift(0.08, config=lenient)
        assert report.status == "ok"
        assert _check(_report(drift=report), "drift").status == "ok"

    def test_prediction_drift_counts_and_is_named(self) -> None:
        rng = np.random.default_rng(1)
        ref_pred = rng.beta(5, 5, 2000)
        cur_pred = rng.beta(2, 8, 2000)
        report = _drift(0.0, reference_predictions=ref_pred, current_predictions=cur_pred)
        assert report.status == "alert" and report.prediction is not None
        check = _check(_report(drift=report), "drift")
        assert check.status == "alert"
        assert "prediction" in " ".join(check.reasons)
        assert "prediction" in check.metrics["offending"]

    def test_too_little_live_data_is_insufficient_not_ok(self) -> None:
        tiny = drift_report(
            {"feature_price": np.linspace(0, 1, 2000)}, {"feature_price": np.linspace(0, 1, 10)}
        )
        check = _check(_report(drift=tiny), "drift")
        assert check.status == "insufficient"
        assert check.cause == "too_few_samples"
        assert "feature_price" in " ".join(check.reasons)

    def test_a_measured_alert_is_not_hidden_by_an_unmeasurable_feature(self) -> None:
        rng = np.random.default_rng(0)
        ref = {"feature_a": rng.normal(0, 1, 2000), "feature_b": rng.normal(0, 1, 2000)}
        sparse = np.concatenate([rng.normal(0, 1, 10), np.full(1990, np.nan)])
        ref["feature_b"] = sparse  # 10 usable values on each side: unmeasurable, not drifting
        cur = {"feature_a": rng.normal(3, 1, 2000), "feature_b": sparse.copy()}
        report = drift_report(ref, cur, config=DriftConfig(min_samples=300))
        by_name = {f.name: f.status for f in report.features}
        assert by_name == {"feature_a": "alert", "feature_b": "insufficient"}
        check = _check(_report(drift=report), "drift")
        assert check.status == "alert"
        assert check.metrics["offending"] == ["feature_a"]
        assert check.metrics["n_insufficient"] == 1

    def test_no_drift_report_is_unavailable_not_ok(self) -> None:
        check = _check(_no_drift(), "drift")
        assert check.status == "insufficient"
        assert check.cause == "not_evaluated"
        assert "reference" in " ".join(check.reasons)


def _no_drift() -> HealthReport:
    return health_report(champion=_champion(), trades=_trades(), drift=None, as_of=_AS_OF)


# --------------------------------------------------------------------------
# Calibration check
# --------------------------------------------------------------------------


class TestCalibrationCheck:
    def test_matching_the_backtest_is_ok(self) -> None:
        check = _check(_report(trades=_trades(brier=0.205)), "calibration")
        assert check.status == "ok"
        assert check.metrics["brier_live"] == pytest.approx(0.205)
        assert check.metrics["brier_backtest"] == pytest.approx(0.20)

    def test_a_better_brier_than_the_backtest_is_never_a_problem(self) -> None:
        assert _check(_report(trades=_trades(brier=0.10)), "calibration").status == "ok"

    def test_moderate_degradation_warns(self) -> None:
        check = _check(_report(trades=_trades(brier=0.23)), "calibration")
        assert check.status == "warn"
        assert check.metrics["degradation_abs"] == pytest.approx(0.03)
        assert check.metrics["degradation_rel"] == pytest.approx(0.15)

    def test_severe_degradation_alerts(self) -> None:
        check = _check(_report(trades=_trades(brier=0.27)), "calibration")
        assert check.status == "alert"
        assert "0.270" in " ".join(check.reasons) and "0.200" in " ".join(check.reasons)

    def test_the_z_score_is_the_two_sample_welch_statistic(self) -> None:
        trades = _trades(brier=0.27)
        live = [t["model_prob"] ** 2 for t in trades]
        check = _check(_report(trades=trades), "calibration")
        expected = _welch_z(live, 0.20, 0.10, 2000, higher_is_worse=True)
        assert check.metrics["z"] == pytest.approx(expected)
        assert check.metrics["p_value"] == pytest.approx(0.5 * math.erfc(expected / math.sqrt(2)))

    def test_both_an_absolute_and_a_relative_degradation_are_required(self) -> None:
        """+0.03 on a 0.60 Brier is 5%: material in absolute terms, not in relative ones."""
        champ = _champion(_expectation(brier_score=0.60))
        report = _report(trades=_trades(brier=0.63, brier_sd=0.05), champion=champ)
        assert _check(report, "calibration").status == "ok"

    def test_a_big_gap_inside_the_noise_is_not_a_finding(self) -> None:
        """A degradation that clears the effect thresholds but not the noise stays ok.

        Same live distribution twice: at 30 trades the gap is within sampling error;
        at 120 it is not. The verdict has to follow the evidence, not the point estimate.
        """
        noisy = {"brier": 0.28, "brier_sd": 0.28}
        champ = _champion(_expectation(brier_score=0.20, brier_std=0.10))
        few = _report(trades=_trades(30, **noisy, pnl=0.10), champion=champ)
        many = _report(trades=_trades(120, **noisy, pnl=0.10), champion=champ)
        assert _check(few, "calibration").status == "ok"
        assert _check(few, "calibration").metrics["z"] < 2.0
        assert _check(many, "calibration").status in ("warn", "alert")

    def test_fewer_than_min_trades_is_insufficient_even_if_awful(self) -> None:
        check = _check(_report(trades=_trades(29, brier=0.60)), "calibration")
        assert check.status == "insufficient"
        assert check.cause == "too_few_samples"
        assert check.metrics["n_trades"] == 29
        assert "30" in " ".join(check.reasons)

    def test_exactly_min_trades_is_evaluated(self) -> None:
        check = _check(_report(trades=_trades(30, brier=0.60)), "calibration")
        assert check.status == "alert"

    def test_min_trades_is_configurable(self) -> None:
        cfg = WatchdogConfig(calibration=CalibrationThresholds(min_trades=10))
        assert _check(
            _report(trades=_trades(12, brier=0.60), config=cfg), "calibration"
        ).status == ("alert")

    def test_no_trades_at_all_is_insufficient(self) -> None:
        check = _check(_report(trades=[]), "calibration")
        assert check.status == "insufficient"
        assert check.metrics["n_trades"] == 0

    def test_an_old_champion_without_a_brier_is_unavailable_not_green(self) -> None:
        champ = _champion(_expectation(brier_score=None, brier_std=None))
        check = _check(_report(trades=_trades(brier=0.60), champion=champ), "calibration")
        assert check.status == "insufficient"
        assert check.cause == "no_expectation"

    def test_no_expectation_at_all_is_unavailable_for_calibration(self) -> None:
        champ = _champion()
        champ = dataclasses.replace(champ, expectation=None)
        check = _check(_report(trades=_trades(brier=0.60), champion=champ), "calibration")
        assert check.status == "insufficient"
        assert check.cause == "no_expectation"
        assert "expectation" in " ".join(check.reasons)

    def test_trades_without_a_forecast_are_not_scored_and_are_counted(self) -> None:
        good = _trades(40, brier=0.20)
        blind = [{k: v for k, v in t.items() if k != "model_prob"} for t in _trades(40, brier=0.9)]
        check = _check(_report(trades=good + blind), "calibration")
        assert check.status == "ok"
        assert check.metrics["n_trades"] == 40
        assert check.metrics["n_skipped_no_forecast"] == 40

    def test_reliability_is_reported_for_context(self) -> None:
        metrics = _check(_report(), "calibration").metrics
        assert "reliability_live" in metrics


# --------------------------------------------------------------------------
# PnL check
# --------------------------------------------------------------------------


class TestPnlCheck:
    def test_matching_the_backtest_is_ok(self) -> None:
        check = _check(_report(trades=_trades(pnl=0.10)), "pnl")
        assert check.status == "ok"
        assert check.metrics["mean_live"] == pytest.approx(0.10)
        assert check.metrics["mean_backtest"] == pytest.approx(0.10)

    def test_outperforming_the_backtest_is_never_a_problem(self) -> None:
        """One-sided: a watchdog that pages on good luck gets muted."""
        assert _check(_report(trades=_trades(pnl=0.50)), "pnl").status == "ok"

    def test_a_significant_shortfall_warns(self) -> None:
        check = _check(_report(trades=_trades(pnl=0.03)), "pnl")
        assert check.status == "warn"
        assert check.metrics["shortfall_rel"] == pytest.approx(0.7)

    def test_losing_money_against_a_profitable_backtest_alerts(self) -> None:
        check = _check(_report(trades=_trades(pnl=-0.02)), "pnl")
        assert check.status == "alert"
        assert "-0.020" in " ".join(check.reasons) and "0.100" in " ".join(check.reasons)

    def test_the_z_score_is_the_two_sample_welch_statistic(self) -> None:
        trades = _trades(pnl=0.03)
        live = [t["realized_pnl"] for t in trades]
        check = _check(_report(trades=trades), "pnl")
        expected = _welch_z(live, 0.10, 0.30, 2000, higher_is_worse=False)
        assert check.metrics["z"] == pytest.approx(expected)
        assert check.metrics["p_value"] == pytest.approx(0.5 * math.erfc(expected / math.sqrt(2)))

    def test_the_backtests_own_uncertainty_is_part_of_the_noise(self) -> None:
        """A backtest of 20 trades is a weak promise; falling short of it means less."""
        weak = _champion(_expectation(n_trades=20, std_pnl_per_stake=1.5))
        strong = _champion(_expectation(n_trades=20000))
        trades = _trades(pnl=0.03)
        assert _check(_report(trades=trades, champion=weak), "pnl").status == "ok"
        assert _check(_report(trades=trades, champion=strong), "pnl").status == "warn"

    def test_a_shortfall_inside_the_noise_is_not_a_finding(self) -> None:
        few = _report(trades=_trades(50, pnl=-0.02, pnl_sd=1.0))
        assert _check(few, "pnl").status == "ok"
        assert _check(few, "pnl").metrics["z"] < 2.0

    def test_fewer_than_min_trades_is_insufficient_even_if_losing(self) -> None:
        cfg = WatchdogConfig(pnl=PnLThresholds(min_trades=50))
        check = _check(_report(trades=_trades(49, pnl=-1.0), config=cfg), "pnl")
        assert check.status == "insufficient"
        assert check.cause == "too_few_samples"
        assert "50" in " ".join(check.reasons)

    def test_exactly_min_trades_is_evaluated(self) -> None:
        cfg = WatchdogConfig(pnl=PnLThresholds(min_trades=50))
        assert _check(_report(trades=_trades(50, pnl=-1.0), config=cfg), "pnl").status == "alert"

    def test_returns_are_per_unit_stake(self) -> None:
        """Trading 0.1 per trade must not read as a 10x shortfall against a 1.0 backtest."""
        small = _trades(pnl=0.10, stake=0.1)
        assert _check(_report(trades=small), "pnl").status == "ok"
        assert _check(_report(trades=small), "pnl").metrics["mean_live"] == pytest.approx(0.10)

    def test_no_expectation_is_unavailable_not_ok(self) -> None:
        champ = dataclasses.replace(_champion(), expectation=None)
        check = _check(_report(trades=_trades(pnl=-1.0), champion=champ), "pnl")
        assert check.status == "insufficient"
        assert check.cause == "no_expectation"

    def test_a_non_positive_backtest_mean_cannot_anchor_a_shortfall(self) -> None:
        champ = _champion(_expectation(mean_pnl_per_stake=0.0))
        check = _check(_report(trades=_trades(pnl=-0.5), champion=champ), "pnl")
        assert check.status == "insufficient"
        assert check.cause == "invalid_expectation"

    def test_trades_with_a_zero_or_missing_stake_are_unscoreable(self) -> None:
        bad = _trades(40, pnl=-1.0, stake=0.0)
        check = _check(_report(trades=_trades(60) + bad), "pnl")
        assert check.status == "ok"
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_unscoreable"] == 40


class TestCostBasis:
    """Live PnL is only comparable with the backtest if both paid the same friction."""

    _COSTS = CostModel(taker_bps=30.0, slippage_bps=50.0)

    def _stamped(self, n: int, costs: CostModel, **kw: object) -> list[dict]:
        fill = round(costs.fill_price(0.40, stake=1.0), 8)
        fee = round(costs.fee(1.0), 6)
        # As the current SettlementEngine leaves them: settled on the recorded fill.
        return _trades(n, fill_price=fill, fee_paid=fee, settled_basis="fill_price", **kw)

    def test_trades_under_the_same_costs_are_compared(self) -> None:
        champ = _champion(costs=self._COSTS)
        check = _check(_report(trades=self._stamped(60, self._COSTS), champion=champ), "pnl")
        assert check.status == "ok"
        assert check.metrics["cost_basis"] == "match"
        assert check.metrics["n_excluded_cost_basis"] == 0

    def test_trades_filled_under_other_costs_are_excluded_and_counted(self) -> None:
        champ = _champion(costs=self._COSTS)
        frictionless = self._stamped(60, CostModel(taker_bps=30.0))  # no slippage: PnL looks better
        good = self._stamped(60, self._COSTS)
        check = _check(_report(trades=good + frictionless, champion=champ), "pnl")
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_excluded_cost_basis"] == 60

    def test_all_trades_on_a_different_cost_basis_is_insufficient(self) -> None:
        champ = _champion(costs=self._COSTS)
        check = _check(
            _report(trades=self._stamped(60, CostModel(taker_bps=80.0)), champion=champ), "pnl"
        )
        assert check.status == "insufficient"
        assert check.metrics["n_excluded_cost_basis"] == 60
        assert "cost" in " ".join(check.reasons)

    def test_unknown_champion_costs_are_reported_not_assumed(self) -> None:
        check = _check(_report(trades=self._stamped(60, self._COSTS)), "pnl")
        assert check.status == "ok"
        assert check.metrics["cost_basis"] == "unknown"

    def test_a_trade_with_no_cost_fields_is_kept_and_marked_unverified(self) -> None:
        champ = _champion(costs=self._COSTS)
        check = _check(_report(trades=_trades(60), champion=champ), "pnl")
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_cost_unverified"] == 60

    def test_the_no_direction_quote_is_the_complement(self) -> None:
        """NO trades are left out of the PnL check before the cost basis is read, so
        the complement arithmetic is pinned on the function that owns it."""
        fee = round(self._COSTS.fee(1.0), 6)
        fill = round(self._COSTS.fill_price(0.60, stake=1.0), 8)  # 1 - 0.40
        wrong = round(self._COSTS.fill_price(0.40, stake=1.0), 8)  # the YES quote
        base = {"direction": "no", "gamma_price": 0.40, "fee_paid": fee}
        assert _cost_basis({**base, "fill_price": fill}, self._COSTS, 1.0) == "match"
        assert _cost_basis({**base, "fill_price": wrong}, self._COSTS, 1.0) == "mismatch"


# --------------------------------------------------------------------------
# Which trades count
# --------------------------------------------------------------------------


class TestTradeSelection:
    def test_trades_outside_the_champions_allowed_segments_do_not_count(self) -> None:
        """Segment C failed the gate. Its losses are not this champion's evidence."""
        good = _trades(60, brier=0.20, pnl=0.10, segment="A")
        stray = _trades(60, brier=0.70, pnl=-1.0, segment="C")
        report = _report(trades=good + stray)
        assert report.status == "ok"
        assert _check(report, "pnl").metrics["n_out_of_segment"] == 60
        assert _check(report, "calibration").metrics["n_out_of_segment"] == 60

    def test_allowed_segments_come_from_the_champion_not_the_trades(self) -> None:
        champ = _champion(passing=("X",), failing=("A",))
        report = _report(trades=_trades(60, segment="A"), champion=champ)
        assert _check(report, "pnl").status == "insufficient"
        assert _check(report, "pnl").metrics["n_trades"] == 0

    def test_every_allowed_segment_counts(self) -> None:
        trades = _trades(40, segment="A") + _trades(40, segment="B")
        assert _check(_report(trades=trades), "pnl").metrics["n_trades"] == 80

    def test_trades_from_before_this_champion_are_not_its_evidence(self) -> None:
        """A re-promotion must not be judged on the previous champion's trades."""
        old = _trades(60, brier=0.70, pnl=-1.0, recorded_at="2025-12-15T10:00:00+00:00")
        new = _trades(60)
        report = _report(trades=old + new)
        assert report.status == "ok"
        assert _check(report, "pnl").metrics["n_before_champion"] == 60

    def test_the_boundary_is_inclusive_of_the_publish_instant(self) -> None:
        trades = _trades(60, recorded_at=_PUBLISHED)
        assert _check(_report(trades=trades), "pnl").metrics["n_trades"] == 60

    def test_trades_that_cannot_be_dated_are_excluded_not_assumed_current(self) -> None:
        undated = [{k: v for k, v in t.items() if k != "recorded_at"} for t in _trades(60)]
        check = _check(_report(trades=undated), "pnl")
        assert check.metrics["n_trades"] == 0
        assert check.metrics["n_unattributed"] == 60

    def test_naive_timestamps_are_read_as_utc(self) -> None:
        trades = _trades(60, recorded_at="2026-02-01T10:00:00")
        assert _check(_report(trades=trades), "pnl").metrics["n_trades"] == 60

    def test_open_trades_are_not_evidence(self) -> None:
        trades = _trades(60) + _trades(40, open_=True)
        check = _check(_report(trades=trades), "pnl")
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_open"] == 40

    def test_nothing_after_the_cutoff_leaks_in(self) -> None:
        """Replaying an old as_of must not see truth that had not arrived yet."""
        past = _trades(60)
        future_target = _trades(40, brier=0.9, pnl=-1.0, target_date="2026-03-02")
        future_record = _trades(40, brier=0.9, pnl=-1.0, recorded_at="2026-03-02T00:00:00+00:00")
        report = _report(trades=past + future_target + future_record)
        assert report.status == "ok"
        assert _check(report, "pnl").metrics["n_after_as_of"] == 80

    def test_the_trade_order_in_the_file_does_not_matter(self) -> None:
        trades = _trades(60) + _trades(60, brier=0.25, pnl=0.05, segment="B")
        shuffled = trades[:]
        random.Random(0).shuffle(shuffled)
        assert _report(trades=trades).to_dict() == _report(trades=shuffled).to_dict()

    def test_a_repeated_row_is_not_collapsed_by_the_watchdog(self) -> None:
        """De-duplication is the broker's job; silently dropping rows here would hide it."""
        trades = _trades(30)
        doubled = trades + trades
        assert _check(_report(trades=doubled), "pnl").metrics["n_trades"] == 60

    def test_per_segment_counts_are_reported(self) -> None:
        trades = _trades(40, segment="A") + _trades(30, segment="B")
        assert _check(_report(trades=trades), "pnl").metrics["n_by_segment"] == {"A": 40, "B": 30}


# --------------------------------------------------------------------------
# Insufficient is not healthy
# --------------------------------------------------------------------------


class TestInsufficientPolicy:
    def test_insufficient_alone_is_not_healthy(self) -> None:
        report = _report(trades=_trades(10))  # drift fine, trades too few
        assert [c.status for c in report.checks] == ["ok", "insufficient", "insufficient"]
        assert report.status == "insufficient"
        assert report.exit_code == EXIT_INSUFFICIENT != EXIT_OK

    def test_it_outranks_ok_and_loses_to_a_measured_problem(self) -> None:
        warn = _report(trades=_trades(10), drift=_drift(0.03))
        assert warn.status == "warn"
        alert = _report(trades=_trades(10), drift=_drift(0.08))
        assert alert.status == "alert"

    def test_warn_policy_promotes_insufficient_to_a_warn(self) -> None:
        cfg = WatchdogConfig(insufficient="warn")
        report = _report(trades=_trades(10), config=cfg)
        assert report.status == "warn"
        assert report.exit_code == EXIT_WARN
        # The checks themselves still say what they said.
        assert [c.status for c in report.checks] == ["ok", "insufficient", "insufficient"]

    def test_warn_policy_does_not_downgrade_an_alert(self) -> None:
        cfg = WatchdogConfig(insufficient="warn")
        assert _report(trades=_trades(10), drift=_drift(0.08), config=cfg).status == "alert"

    def test_a_missing_drift_input_is_not_a_silent_pass(self) -> None:
        report = _no_drift()
        assert report.status == "insufficient"
        assert report.exit_code != EXIT_OK

    def test_every_check_unavailable_is_never_healthy(self) -> None:
        champ = dataclasses.replace(_champion(), expectation=None)
        report = health_report(champion=champ, trades=[], drift=None, as_of=_AS_OF)
        assert report.status == "insufficient"


# --------------------------------------------------------------------------
# Skipping checks
# --------------------------------------------------------------------------


class TestSkippedChecks:
    def test_a_skipped_check_is_omitted_and_listed(self) -> None:
        report = _report(checks=("calibration", "pnl"))
        assert [c.name for c in report.checks] == ["calibration", "pnl"]
        assert report.skipped == ("drift",)

    def test_a_skipped_check_cannot_fail_the_report(self) -> None:
        report = _report(drift=_drift(0.08), checks=("calibration", "pnl"))
        assert report.status == "ok"

    def test_skipped_checks_are_recorded_in_the_serialised_report(self) -> None:
        assert _report(checks=("pnl",)).to_dict()["skipped"] == ["drift", "calibration"]

    def test_skipping_everything_is_an_error(self) -> None:
        with pytest.raises(WatchdogInputError, match="at least one"):
            _report(checks=())

    def test_an_unknown_check_name_is_an_error(self) -> None:
        with pytest.raises(WatchdogInputError, match="unknown check"):
            _report(checks=("drift", "unknwon"))


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


class TestConfig:
    def test_defaults_are_valid_and_round_trip(self) -> None:
        cfg = WatchdogConfig()
        assert WatchdogConfig.from_dict(cfg.to_dict()) == cfg

    def test_a_partial_dict_fills_in_defaults(self) -> None:
        cfg = WatchdogConfig.from_dict({"pnl": {"min_trades": 80}, "drift": {"min_samples": 100}})
        assert cfg.pnl.min_trades == 80
        assert cfg.pnl.shortfall_warn == PnLThresholds().shortfall_warn
        assert cfg.drift.min_samples == 100
        assert cfg.drift.thresholds == DriftThresholds()

    def test_partial_drift_thresholds_are_merged_with_the_defaults(self) -> None:
        cfg = WatchdogConfig.from_dict({"drift": {"thresholds": {"psi_alert": 0.5}}})
        assert cfg.drift.thresholds.psi_alert == 0.5
        assert cfg.drift.thresholds.psi_warn == DriftThresholds().psi_warn

    @pytest.mark.parametrize(
        "bad",
        [
            {"pnl": {"min_trade": 5}},  # typo
            {"calibration": {"nope": 1}},
            {"mystery": {}},
            {"drift": {"thresholds": {"psi_alrt": 1}}},
        ],
    )
    def test_an_unknown_key_is_rejected_rather_than_ignored(self, bad: dict) -> None:
        """A typo'd threshold would otherwise leave the default silently in force."""
        with pytest.raises(WatchdogInputError, match="unknown"):
            WatchdogConfig.from_dict(bad)

    @pytest.mark.parametrize(
        "bad",
        [
            {"calibration": {"min_trades": 1}},
            {"pnl": {"min_trades": 0}},
            {"calibration": {"brier_warn_abs": 0.1, "brier_alert_abs": 0.05}},
            {"calibration": {"brier_warn_rel": 0.5, "brier_alert_rel": 0.1}},
            {"calibration": {"z_warn": 4, "z_alert": 3}},
            {"pnl": {"shortfall_warn": 2.0, "shortfall_alert": 1.0}},
            {"pnl": {"z_warn": -1}},
            {"pnl": {"shortfall_warn": float("nan")}},
            {"insufficient": "ignore"},
        ],
    )
    def test_invalid_thresholds_are_rejected(self, bad: dict) -> None:
        with pytest.raises(WatchdogInputError):
            WatchdogConfig.from_dict(bad)

    def test_thresholds_change_the_verdict(self) -> None:
        trades = _trades(brier=0.23)
        assert _check(_report(trades=trades), "calibration").status == "warn"
        strict = WatchdogConfig(
            calibration=CalibrationThresholds(
                brier_warn_abs=0.01,
                brier_alert_abs=0.02,
                brier_warn_rel=0.05,
                brier_alert_rel=0.10,
                z_warn=1.0,
                z_alert=2.0,
            )
        )
        assert _check(_report(trades=trades, config=strict), "calibration").status == "alert"
        lax = WatchdogConfig(
            calibration=CalibrationThresholds(brier_warn_abs=0.2, brier_alert_abs=0.4)
        )
        assert _check(_report(trades=trades, config=lax), "calibration").status == "ok"

    def test_the_config_is_recorded_in_the_report(self) -> None:
        cfg = WatchdogConfig(pnl=PnLThresholds(min_trades=77))
        assert _report(config=cfg).to_dict()["config"]["pnl"]["min_trades"] == 77


# --------------------------------------------------------------------------
# Serialisation and determinism
# --------------------------------------------------------------------------


class TestSerialisation:
    @pytest.fixture(params=["healthy", "alert", "insufficient"])
    def report(self, request: pytest.FixtureRequest) -> HealthReport:
        return {
            "healthy": lambda: _report(),
            "alert": lambda: _report(trades=_trades(brier=0.27, pnl=-0.02), drift=_drift(0.08)),
            "insufficient": lambda: _report(trades=_trades(5), drift=None),
        }[request.param]()

    def test_round_trip_is_exact(self, report: HealthReport) -> None:
        assert HealthReport.from_dict(report.to_dict()) == report

    def test_round_trip_through_json_text(self, report: HealthReport) -> None:
        text = json.dumps(report.to_dict(), allow_nan=False)
        assert HealthReport.from_dict(json.loads(text)) == report

    def test_to_dict_is_strict_json(self, report: HealthReport) -> None:
        json.dumps(report.to_dict(), allow_nan=False)

    def test_the_serialised_status_and_exit_code_agree_with_the_object(
        self, report: HealthReport
    ) -> None:
        d = report.to_dict()
        assert d["status"] == report.status
        assert d["exit_code"] == report.exit_code

    def test_same_inputs_give_byte_identical_output(self, report: HealthReport) -> None:
        a = json.dumps(report.to_dict(), sort_keys=False)
        b = json.dumps(HealthReport.from_dict(report.to_dict()).to_dict(), sort_keys=False)
        assert a == b

    def test_two_runs_agree(self) -> None:
        a = json.dumps(_report(trades=_trades(brier=0.25)).to_dict())
        b = json.dumps(_report(trades=_trades(brier=0.25)).to_dict())
        assert a == b

    def test_the_champion_is_identified_in_the_report(self) -> None:
        d = _report().to_dict()["champion"]
        assert d["model_name"] == "champ"
        assert d["plugin_family"] == "politics"
        assert d["published_at"] == _PUBLISHED
        assert d["allowed_segments"] == ["A", "B"]
        assert d["has_expectation"] is True

    def test_non_finite_numbers_never_reach_the_json(self) -> None:
        zero_spread = [_trade(0.04, 0.1) for _ in range(40)]  # every trade identical
        report = _report(trades=zero_spread)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_as_of_is_recorded_verbatim(self) -> None:
        assert _report().to_dict()["as_of"] == _AS_OF


# --------------------------------------------------------------------------
# Exit codes
# --------------------------------------------------------------------------


class TestExitCodes:
    def test_codes_are_stable_and_distinct(self) -> None:
        codes = {
            "ok": EXIT_OK,
            "warn": EXIT_WARN,
            "alert": EXIT_ALERT,
            "insufficient": EXIT_INSUFFICIENT,
            "usage": EXIT_USAGE,
            "internal": EXIT_INTERNAL,
        }
        assert codes == {
            "ok": 0,
            "warn": 10,
            "alert": 20,
            "insufficient": 30,
            "usage": 64,
            "internal": 70,
        }
        assert len(set(codes.values())) == len(codes)

    def test_no_health_code_collides_with_pythons_crash_or_clicks_usage_error(self) -> None:
        """A traceback exits 1 and a bad flag exits 2; neither may read as a verdict."""
        verdicts = {EXIT_OK, EXIT_WARN, EXIT_ALERT, EXIT_INSUFFICIENT}
        assert 1 not in verdicts and 2 not in verdicts

    @pytest.mark.parametrize(
        ("status", "code"),
        [("ok", 0), ("warn", 10), ("alert", 20), ("insufficient", 30)],
    )
    def test_status_maps_to_its_code(self, status: str, code: int) -> None:
        assert exit_code_for(status) == code  # type: ignore[arg-type]

    def test_an_unknown_status_is_an_error_not_a_zero(self) -> None:
        with pytest.raises(ValueError):
            exit_code_for("fine")  # type: ignore[arg-type]


class TestInputErrors:
    def test_an_unparseable_as_of_is_an_input_error(self) -> None:
        with pytest.raises(WatchdogInputError, match="as_of"):
            health_report(champion=_champion(), trades=[], drift=None, as_of="yesterday-ish")

    def test_an_unparseable_publish_time_is_an_input_error(self) -> None:
        with pytest.raises(WatchdogInputError, match="published_at"):
            health_report(
                champion=_champion(published_at="whenever"), trades=[], drift=None, as_of=_AS_OF
            )

    def test_naive_as_of_is_read_as_utc(self) -> None:
        a = health_report(
            champion=_champion(), trades=_trades(), drift=_drift(), as_of="2026-03-01"
        )
        assert a.status == "ok"

    def test_as_of_may_be_a_datetime_string_with_z(self) -> None:
        r = health_report(
            champion=_champion(), trades=_trades(), drift=_drift(), as_of="2026-03-01T00:00:00Z"
        )
        assert r.status == "ok"

    def test_there_is_no_default_clock(self) -> None:
        """`as_of` is required: the core never reads the wall clock, so replays are exact."""
        with pytest.raises(TypeError):
            health_report(champion=_champion(), trades=[], drift=None)  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# Edge cases the statistics must survive
# --------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_spread_on_both_sides_has_no_z_and_decides_on_the_gap(self) -> None:
        """No noise at all: a z-score is undefined, so any shortfall is certain."""
        exact = _champion(_expectation(std_pnl_per_stake=0.0, brier_std=0.0))
        flat_bad = [_trade(0.4, -0.5) for _ in range(40)]
        report = _report(trades=flat_bad, champion=exact)
        pnl = _check(report, "pnl")
        assert pnl.metrics["z"] is None and pnl.metrics["p_value"] is None
        assert pnl.status == "alert"
        assert _check(report, "calibration").status == "alert"
        json.dumps(report.to_dict(), allow_nan=False)

    def test_zero_spread_with_no_gap_is_ok(self) -> None:
        exact = _champion(_expectation(std_pnl_per_stake=0.0, brier_std=0.0))
        same = [_trade(0.2, 0.10) for _ in range(40)]
        report = _report(trades=same, champion=exact)
        assert _check(report, "pnl").status == "ok"
        assert _check(report, "calibration").status == "ok"

    def test_a_perfect_backtest_brier_does_not_leak_infinity_into_the_json(self) -> None:
        """Brier 0 makes the relative degradation infinite; the report stays strict JSON."""
        perfect = _champion(_expectation(brier_score=0.0, brier_std=0.0))
        report = _report(trades=_trades(brier=0.20), champion=perfect)
        check = _check(report, "calibration")
        assert check.status == "alert"
        assert check.metrics["degradation_rel"] is None
        json.dumps(report.to_dict(), allow_nan=False)
        assert HealthReport.from_dict(report.to_dict()) == report

    def test_a_missing_rate_driver_is_named_with_its_threshold(self) -> None:
        rng = np.random.default_rng(0)
        ref = {"feature_a": rng.normal(0, 1, 2000)}
        cur_values = rng.normal(0, 1, 2000)
        cur_values[:600] = np.nan  # 0% -> 30% missing
        report = drift_report(ref, {"feature_a": cur_values})
        check = _check(_report(drift=report), "drift")
        assert check.status == "alert"
        assert "missing-rate change +0.300 >= 0.2" in " ".join(check.reasons)

    def test_several_offenders_each_get_a_line_worst_first(self) -> None:
        rng = np.random.default_rng(0)
        ref = {n: rng.normal(0, 1, 2000) for n in ("feature_a", "feature_b", "feature_c")}
        cur = {
            "feature_a": rng.normal(0.3, 1, 2000),  # warn
            "feature_b": rng.normal(3.0, 1, 2000),  # alert
            "feature_c": rng.normal(0, 1, 2000),  # fine
        }
        report = drift_report(ref, cur)
        statuses = {f.name: f.status for f in report.features}
        assert statuses == {"feature_a": "warn", "feature_b": "alert", "feature_c": "ok"}
        check = _check(_report(drift=report), "drift")
        assert [r.split(":")[0] for r in check.reasons] == ["feature_b", "feature_a"]

    def test_a_row_that_is_not_an_object_is_an_input_error(self) -> None:
        with pytest.raises(WatchdogInputError, match="trade #1"):
            _report(trades=[_trade(0.1, 0.1), "not a trade"])  # type: ignore[list-item]

    def test_a_row_with_a_non_finite_pnl_is_unscoreable_not_a_crash(self) -> None:
        bad = [_trade(0.04, float("nan")) for _ in range(5)]
        check = _check(_report(trades=_trades(60) + bad), "pnl")
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_unscoreable"] == 5

    def test_a_non_numeric_model_prob_is_skipped_by_calibration_not_a_crash(self) -> None:
        odd = [{**_trade(0.04, 0.1), "model_prob": "high"} for _ in range(5)]
        check = _check(_report(trades=_trades(60) + odd), "calibration")
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_skipped_no_forecast"] == 5


class TestConfigValidationMessages:
    @pytest.mark.parametrize("bad", [True, "abc", None, [1]])
    def test_a_threshold_that_is_not_a_number(self, bad: object) -> None:
        with pytest.raises(WatchdogInputError, match="must be a number"):
            PnLThresholds(z_warn=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [1.5, True, "5", None])
    def test_min_trades_must_be_an_integer(self, bad: object) -> None:
        with pytest.raises(WatchdogInputError, match="integer"):
            PnLThresholds(min_trades=bad)  # type: ignore[arg-type]

    def test_an_unknown_drift_setting_is_rejected(self) -> None:
        with pytest.raises(WatchdogInputError, match="unknown drift setting"):
            WatchdogConfig.from_dict({"drift": {"min_sample": 10}})

    def test_an_invalid_drift_setting_is_an_input_error_not_a_bare_value_error(self) -> None:
        with pytest.raises(WatchdogInputError, match="invalid drift settings"):
            WatchdogConfig.from_dict({"drift": {"n_bins": 1}})

    def test_the_drift_thresholds_are_validated_by_the_drift_module(self) -> None:
        with pytest.raises(WatchdogInputError, match="invalid drift settings"):
            WatchdogConfig.from_dict({"drift": {"thresholds": {"psi_warn": 9, "psi_alert": 1}}})

    def test_a_check_with_an_unknown_status_cannot_be_deserialised(self) -> None:
        data = _report().to_dict()
        data["checks"][0]["status"] = "fine"
        with pytest.raises(ValueError, match="unknown check status"):
            HealthReport.from_dict(data)


class TestRender:
    def test_a_z_score_that_rounds_to_zero_never_prints_a_minus_sign(self) -> None:
        """`-0.0` in a report reads as a bug, and is one in a log that gets grepped."""
        for pnl in (0.10 + 1e-9, 0.10 - 1e-9):
            text = " ".join(_check(_report(trades=_trades(pnl=pnl)), "pnl").reasons)
            assert "z=-0.0" not in text and "z=0.0" in text, text

    def test_a_healthy_report_renders_to_nothing(self) -> None:
        assert _report().render() == ""

    def test_verbose_renders_every_check_when_healthy(self) -> None:
        text = _report().render(verbose=True)
        assert text.startswith("pmlab monitor: OK")
        assert all(name in text for name in CHECKS)

    def test_a_problem_names_the_check_and_only_the_failing_ones(self) -> None:
        text = _report(trades=_trades(brier=0.27)).render()
        assert text.startswith("pmlab monitor: ALERT")
        assert "calibration" in text and "drift" not in text and "pnl" not in text

    def test_skipped_checks_are_mentioned(self) -> None:
        assert "skipped: drift" in _report(checks=("calibration", "pnl")).render(verbose=True)

    def test_the_text_is_plain_ascii(self) -> None:
        text = _report(trades=_trades(brier=0.27, pnl=-0.02), drift=_drift(0.08)).render(
            verbose=True
        )
        assert text.isascii() and "\x1b" not in text

    def test_the_insufficient_policy_is_visible_in_the_headline(self) -> None:
        cfg = WatchdogConfig(insufficient="warn")
        assert _report(trades=_trades(5), config=cfg).render().startswith("pmlab monitor: WARN")


class TestEffectSizeAndNoiseAreBothRequired:
    """Material AND distinguishable from noise: each guard catches what the other misses."""

    def test_a_certain_but_immaterial_pnl_shortfall_is_ok(self) -> None:
        """With enough trades and no spread, 10% short is certain; it is still not a page."""
        champ = _champion(_expectation(std_pnl_per_stake=0.001))
        check = _check(_report(trades=_trades(400, pnl=0.09, pnl_sd=0.001), champion=champ), "pnl")
        assert check.metrics["z"] > 10
        assert check.metrics["shortfall_rel"] == pytest.approx(0.10)
        assert check.status == "ok"

    def test_a_certain_but_immaterial_brier_gap_is_ok(self) -> None:
        champ = _champion(_expectation(brier_std=0.001))
        trades = _trades(400, brier=0.205, brier_sd=0.001)
        check = _check(_report(trades=trades, champion=champ), "calibration")
        assert check.metrics["z"] > 10
        assert check.status == "ok"

    def test_a_large_relative_change_on_a_tiny_brier_is_not_material(self) -> None:
        """+0.01 on a 0.05 Brier is +20%: large relatively, small absolutely."""
        champ = _champion(_expectation(brier_score=0.05, brier_std=0.001))
        trades = _trades(100, brier=0.06, brier_sd=0.001)
        check = _check(_report(trades=trades, champion=champ), "calibration")
        assert check.metrics["degradation_rel"] == pytest.approx(0.20)
        assert check.status == "ok"

    def test_a_certain_but_partial_pnl_shortfall_is_a_warn_not_an_alert(self) -> None:
        check = _check(_report(trades=_trades(400, pnl=0.03)), "pnl")
        assert check.metrics["z"] > 3.0, "certain enough for an alert on noise alone"
        assert check.metrics["shortfall_rel"] == pytest.approx(0.7)
        assert check.status == "warn", "but it is not a loss of the whole edge"

    def test_a_certain_but_moderate_brier_gap_is_a_warn_not_an_alert(self) -> None:
        check = _check(_report(trades=_trades(400, brier=0.23)), "calibration")
        assert check.metrics["z"] > 3.0
        assert check.metrics["degradation_abs"] == pytest.approx(0.03)
        assert check.status == "warn"

    def test_a_large_absolute_but_small_relative_gap_is_not_an_alert(self) -> None:
        """+0.06 on a 0.60 Brier clears the absolute alert bar but is only +10% relative."""
        champ = _champion(_expectation(brier_score=0.60, brier_std=0.001))
        trades = _trades(400, brier=0.66, brier_sd=0.001)
        check = _check(_report(trades=trades, champion=champ), "calibration")
        assert check.metrics["degradation_abs"] == pytest.approx(0.06)
        assert check.status == "warn"


class TestThresholdsAreInclusive:
    def test_float_round_off_at_a_threshold_does_not_flip_the_verdict(self) -> None:
        """0.25 - 0.20 is 0.04999999999999999 in floats: on the line, not under it.

        Read the measured gap back, then put the threshold a hair above it (inside
        the 1e-9 slack: still a hit) and a clear step above it (outside: a miss).
        """
        champ = _champion(_expectation(brier_std=0.001))
        trades = _trades(100, brier=0.23, brier_sd=0.001)
        gap = _check(_report(trades=trades, champion=champ), "calibration").metrics[
            "degradation_abs"
        ]

        def verdict(threshold: float) -> str:
            cfg = WatchdogConfig(
                calibration=CalibrationThresholds(
                    brier_warn_abs=threshold,
                    brier_alert_abs=1.0,
                    brier_warn_rel=0.0,
                    brier_alert_rel=1.0,
                )
            )
            return _check(_report(trades=trades, champion=champ, config=cfg), "calibration").status

        assert verdict(gap + 5e-10) == "warn"
        assert verdict(gap + 5e-8) == "ok"

    def test_a_brier_on_the_warn_line_with_no_noise_is_a_warn(self) -> None:
        champ = _champion(_expectation(brier_std=0.001))
        check = _check(
            _report(trades=_trades(100, brier=0.22, brier_sd=0.001), champion=champ), "calibration"
        )
        assert check.metrics["degradation_abs"] == pytest.approx(0.02)
        assert check.status == "warn"

    def test_a_shortfall_exactly_equal_to_the_warn_threshold_is_a_warn(self) -> None:
        """Inclusive: 'at or above' the line is in the band, not 'strictly above'."""
        trades = _trades(pnl=0.05)
        free = WatchdogConfig(pnl=PnLThresholds(z_warn=0.0, z_alert=0.0))
        measured = _check(_report(trades=trades, config=free), "pnl").metrics["shortfall_rel"]

        def verdict(threshold: float) -> str:
            cfg = WatchdogConfig(
                pnl=PnLThresholds(
                    shortfall_warn=threshold, shortfall_alert=2.0, z_warn=0.0, z_alert=0.0
                )
            )
            return _check(_report(trades=trades, config=cfg), "pnl").status

        assert verdict(measured) == "warn"
        assert verdict(measured + 1e-6) == "ok"

    def test_a_brier_gap_equal_to_the_warn_threshold_is_a_warn(self) -> None:
        champ = _champion(_expectation(brier_std=0.001))
        trades = _trades(100, brier=0.23, brier_sd=0.001)
        gap = _check(_report(trades=trades, champion=champ), "calibration").metrics[
            "degradation_abs"
        ]
        cfg = WatchdogConfig(
            calibration=CalibrationThresholds(
                brier_warn_abs=gap, brier_alert_abs=1.0, brier_warn_rel=0.0, brier_alert_rel=1.0
            )
        )
        assert _check(_report(trades=trades, champion=champ, config=cfg), "calibration").status == (
            "warn"
        )

    def test_a_large_relative_but_small_absolute_gap_is_a_warn_not_an_alert(self) -> None:
        """+0.03 on a 0.10 Brier is +30%: past the relative alert bar, short of the absolute one."""
        champ = _champion(_expectation(brier_score=0.10, brier_std=0.001))
        check = _check(
            _report(trades=_trades(100, brier=0.13, brier_sd=0.001), champion=champ), "calibration"
        )
        assert check.metrics["degradation_rel"] == pytest.approx(0.30)
        assert check.metrics["z"] > 3.0
        assert check.status == "warn"


# --------------------------------------------------------------------------
# NO-side trades are not the strategy the backtest measured
# --------------------------------------------------------------------------


class TestNoSideTradesAreLeftOutOfThePnlComparison:
    """The backtest only ever buys YES, at the model's top-ranked bin.

    A NO trade's per-stake PnL is the same arithmetic now that NO settles as a buy
    of the NO token, but it is not a draw from the distribution the backtest's mean
    and spread describe: the backtest never ran that strategy, so the z test's null
    hypothesis does not cover it. They are counted, not silently dropped.
    """

    def test_no_trades_do_not_count_toward_the_pnl_evidence(self) -> None:
        check = _check(_report(trades=_trades(60, direction="no", settlement_version=1)), "pnl")
        assert check.status == "insufficient"
        assert check.cause == "too_few_samples"
        assert check.metrics["n_trades"] == 0
        assert check.metrics["n_excluded_no_side"] == 60

    def test_the_reason_says_why_they_were_left_out(self) -> None:
        check = _check(_report(trades=_trades(60, direction="no", settlement_version=1)), "pnl")
        text = " ".join(check.reasons)
        assert "60" in text
        assert "NO side" in text
        assert "only buys YES" in text

    def test_losing_no_trades_cannot_page_a_healthy_yes_book(self) -> None:
        yes = _trades(60, pnl=0.10)
        losing_no = _trades(40, pnl=-1.0, pnl_sd=0.0, direction="no", settlement_version=1)
        check = _check(_report(trades=yes + losing_no), "pnl")
        assert check.status == "ok"
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_excluded_no_side"] == 40

    def test_yes_trades_are_still_judged_beside_no_trades(self) -> None:
        yes = _trades(60, pnl=-0.20)
        no = _trades(40, pnl=0.50, direction="no", settlement_version=1)
        check = _check(_report(trades=yes + no), "pnl")
        assert check.status == "alert"
        assert check.metrics["n_trades"] == 60

    def test_no_trades_still_count_for_calibration(self) -> None:
        """The forecast is about the label, whichever side was taken."""
        trades = _trades(60, direction="no", settlement_version=1)
        check = _check(_report(trades=trades), "calibration")
        assert check.status == "ok"
        assert check.metrics["n_trades"] == 60
        assert "n_excluded_no_side" not in check.metrics

    def test_a_trade_with_no_direction_is_a_yes_trade(self) -> None:
        trades = [{k: v for k, v in t.items() if k != "direction"} for t in _trades(60)]
        check = _check(_report(trades=trades), "pnl")
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_excluded_no_side"] == 0

    def test_the_exclusion_is_listed_in_the_rendered_summary(self) -> None:
        report = _report(trades=_trades(10, direction="no", settlement_version=1))
        assert "10 on the NO side" in report.render()


# --------------------------------------------------------------------------
# A corrupt expectation or statistic can never read as healthy
# --------------------------------------------------------------------------


def _unvalidated(**overrides: object) -> BacktestExpectation:
    """An expectation that skipped validation, as a mutated or duck-typed one would."""
    expectation = _expectation()
    for name, value in overrides.items():
        object.__setattr__(expectation, name, value)
    return expectation


# Live data bad enough to alert under any sane expectation.
_TERRIBLE = {"brier": 0.60, "pnl": -1.0}


class TestADefenceAgainstNonFiniteNumbers:
    """Defence in depth. `BacktestExpectation` refuses these values, so this is what
    happens if one gets past it: the check says it could not measure, never "ok"."""

    @pytest.mark.parametrize(
        "field", ["std_pnl_per_stake", "mean_pnl_per_stake", "brier_score", "brier_std"]
    )
    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_a_non_finite_expectation_is_insufficient_never_ok(
        self, field: str, bad: float
    ) -> None:
        champ = _champion(_unvalidated(**{field: bad}))
        report = _report(trades=_trades(60, **_TERRIBLE), champion=champ)
        for name in ("calibration", "pnl"):
            check = _check(report, name)
            assert check.status != "ok", (name, check.reasons)
        affected = "pnl" if field in ("std_pnl_per_stake", "mean_pnl_per_stake") else "calibration"
        check = _check(report, affected)
        assert check.status == "insufficient"
        assert check.cause == "invalid_expectation"
        assert field in " ".join(check.reasons)

    def test_a_nan_pnl_spread_with_terrible_live_data_is_not_silent(self) -> None:
        champ = _champion(_unvalidated(std_pnl_per_stake=math.nan))
        report = _report(trades=_trades(60, pnl=-1.0), champion=champ)
        assert report.status != "ok"
        assert report.exit_code == EXIT_INSUFFICIENT

    def test_a_nan_brier_score_is_not_silent(self) -> None:
        champ = _champion(_unvalidated(brier_score=math.nan))
        report = _report(trades=_trades(60, brier=0.60), champion=champ)
        assert _check(report, "calibration").status == "insufficient"

    @pytest.mark.parametrize("n", [0, 1, -5])
    def test_too_few_backtest_trades_is_insufficient_and_never_a_crash(self, n: int) -> None:
        champ = _champion(_unvalidated(n_trades=n))
        report = _report(trades=_trades(60), champion=champ)
        assert _check(report, "calibration").cause == "invalid_expectation"
        assert _check(report, "pnl").cause == "invalid_expectation"

    def test_a_negative_spread_is_insufficient(self) -> None:
        champ = _champion(_unvalidated(std_pnl_per_stake=-0.3, brier_std=-0.1))
        report = _report(trades=_trades(60), champion=champ)
        assert _check(report, "pnl").cause == "invalid_expectation"
        assert _check(report, "calibration").cause == "invalid_expectation"

    def test_the_insufficient_policy_still_applies(self) -> None:
        champ = _champion(_unvalidated(std_pnl_per_stake=math.nan))
        config = WatchdogConfig(insufficient="warn")
        assert _report(champion=champ, config=config).exit_code == EXIT_WARN

    def test_a_live_statistic_that_overflows_is_insufficient_not_ok(self) -> None:
        """Finite inputs can still give an infinite variance."""
        trades = _trades(60, pnl=0.0, pnl_sd=1e308)
        check = _check(_report(trades=trades), "pnl")
        assert check.status == "insufficient"
        assert check.cause == "invalid_statistic"
        assert "std_live" in " ".join(check.reasons) or "variance" in " ".join(check.reasons)

    def test_a_healthy_report_is_unchanged_by_the_guards(self) -> None:
        assert _report().status == "ok"


# --------------------------------------------------------------------------
# Trades settled at the quote that paid a different price
# --------------------------------------------------------------------------


class TestTradesSettledBeforeThePnlCarriedTheFill:
    """Up to 0.9.0 a trade was settled at the quote even if it recorded a fill_price.

    Its realized_pnl carries none of the slippage its fill_price says it paid, so a
    cost-basis check that compares fill_price with the champion's costs would call it
    "matched" while its PnL is the frictionless one: live looks better than the
    backtest by exactly the friction. Such a trade is left out, and counted.
    """

    _COSTS = CostModel(taker_bps=30.0, slippage_bps=100.0)
    _FILL = round(_COSTS.fill_price(0.40, stake=1.0), 8)  # a YES quote of 0.40, plus 1% slippage
    _FEE = round(_COSTS.fee(1.0), 6)

    def _legacy(self, n: int = 60, **extra: object) -> list[dict]:
        """Settled by an older release: no settled_basis, a fill that differs from the quote."""
        return _trades(n, fill_price=self._FILL, fee_paid=self._FEE, **extra)

    def _current(self, n: int = 60, **extra: object) -> list[dict]:
        return _trades(
            n, fill_price=self._FILL, fee_paid=self._FEE, settled_basis="fill_price", **extra
        )

    def test_a_quote_settled_trade_with_a_different_fill_is_left_out(self) -> None:
        champ = _champion(costs=self._COSTS)
        check = _check(_report(trades=self._legacy(), champion=champ), "pnl")
        assert check.metrics["n_excluded_legacy_settlement"] == 60
        assert check.metrics["n_trades"] == 0
        assert check.status == "insufficient"
        assert "settled at the quote" in " ".join(check.reasons)

    def test_it_is_left_out_even_when_the_champion_records_no_costs(self) -> None:
        check = _check(_report(trades=self._legacy()), "pnl")
        assert check.metrics["n_excluded_legacy_settlement"] == 60

    def test_a_trade_settled_on_its_fill_is_kept(self) -> None:
        champ = _champion(costs=self._COSTS)
        check = _check(_report(trades=self._current(), champion=champ), "pnl")
        assert check.metrics["n_excluded_legacy_settlement"] == 0
        assert check.metrics["n_trades"] == 60
        assert check.metrics["cost_basis"] == "match"

    def test_legacy_trades_cannot_dilute_a_real_shortfall(self) -> None:
        """Frictionless-looking legacy PnL beside current trades that really lost."""
        current = self._current(60, pnl=-0.20)
        legacy = self._legacy(60, pnl=0.50)
        check = _check(_report(trades=current + legacy), "pnl")
        assert check.status == "alert"
        assert check.metrics["n_trades"] == 60
        assert check.metrics["n_excluded_legacy_settlement"] == 60

    def test_a_legacy_trade_with_no_fill_recorded_is_exact_and_kept(self) -> None:
        """Nothing was recorded that the old arithmetic ignored: same number."""
        check = _check(_report(trades=_trades(60)), "pnl")
        assert check.metrics["n_excluded_legacy_settlement"] == 0
        assert check.metrics["n_trades"] == 60

    def test_a_legacy_trade_that_filled_at_the_quote_is_exact_and_kept(self) -> None:
        trades = _trades(60, fill_price=0.40)  # gamma_price is 0.40
        check = _check(_report(trades=trades), "pnl")
        assert check.metrics["n_excluded_legacy_settlement"] == 0
        assert check.metrics["n_trades"] == 60

    def test_a_legacy_fill_that_differs_by_even_a_little_is_not_exact(self) -> None:
        trades = _trades(60, fill_price=0.4001)
        check = _check(_report(trades=trades), "pnl")
        assert check.metrics["n_excluded_legacy_settlement"] == 60

    def test_a_fill_with_no_quote_to_compare_to_is_not_exact(self) -> None:
        trades = [{k: v for k, v in t.items() if k != "gamma_price"} for t in self._legacy()]
        check = _check(_report(trades=trades), "pnl")
        assert check.metrics["n_excluded_legacy_settlement"] == 60

    def test_a_trade_that_says_it_settled_at_the_quote_but_carries_another_fill(self) -> None:
        trades = _trades(60, fill_price=self._FILL, settled_basis="quote")
        check = _check(_report(trades=trades), "pnl")
        assert check.metrics["n_excluded_legacy_settlement"] == 60

    def test_a_no_trade_is_counted_once_as_no_side_not_as_legacy(self) -> None:
        trades = _trades(60, direction="no", fill_price=0.70)
        check = _check(_report(trades=trades), "pnl")
        assert check.metrics["n_excluded_no_side"] == 60
        assert check.metrics["n_excluded_legacy_settlement"] == 0

    def test_calibration_does_not_care_how_the_pnl_was_settled(self) -> None:
        check = _check(_report(trades=self._legacy()), "calibration")
        assert check.metrics["n_trades"] == 60
        assert check.status == "ok"


# --------------------------------------------------------------------------
# as_of is point-in-time for settlement
# --------------------------------------------------------------------------

_MIDNIGHT = "2026-03-01T00:00:00+00:00"


class TestAsOfIsPointInTimeForSettlement:
    """A trade counts only if its truth had arrived by `as_of`.

    `target_date` is a *date*: a trade for 2026-03-01 is eligible at a midnight
    as-of on 2026-03-01 by the date rule, yet cannot have settled before the day
    played out. `settled_at` is the timestamp that says when it did.
    """

    def _on(self, day: str, settled_at: str | None, n: int = 60) -> list[dict]:
        extra: dict = {} if settled_at is None else {"settled_at": settled_at}
        return _trades(n, target_date=day, recorded_at="2026-02-20T10:00:00+00:00", **extra)

    def _pnl(self, trades: list[dict], as_of: str = _MIDNIGHT):
        report = health_report(
            champion=_champion(), trades=trades, drift=None, as_of=as_of, checks=("pnl",)
        )
        return report.checks[0]

    def test_settled_after_as_of_on_the_as_of_day_is_excluded(self) -> None:
        """The reviewer's case: target_date == the as-of day, as-of at 00:00."""
        trades = self._on("2026-03-01", "2026-03-01T06:00:00+00:00")
        check = self._pnl(trades)
        assert check.metrics["n_after_as_of"] == 60
        assert check.metrics["n_in_scope"] == 0
        assert check.metrics["n_trades"] == 0

    def test_the_same_trades_count_once_as_of_is_after_they_settled(self) -> None:
        trades = self._on("2026-03-01", "2026-03-01T06:00:00+00:00")
        check = self._pnl(trades, as_of="2026-03-01T06:00:00+00:00")
        assert check.metrics["n_after_as_of"] == 0
        assert check.metrics["n_trades"] == 60

    def test_settled_exactly_at_as_of_counts(self) -> None:
        trades = self._on("2026-02-28", _MIDNIGHT)
        assert self._pnl(trades).metrics["n_trades"] == 60

    def test_a_second_after_as_of_does_not(self) -> None:
        trades = self._on("2026-02-28", "2026-03-01T00:00:01+00:00")
        assert self._pnl(trades).metrics["n_after_as_of"] == 60

    def test_settled_at_decides_when_it_is_present(self) -> None:
        """Even where the date rule would have excluded it."""
        trades = self._on("2026-03-05", "2026-02-28T12:00:00+00:00")
        assert self._pnl(trades).metrics["n_trades"] == 60

    def test_calibration_uses_the_same_rule(self) -> None:
        trades = self._on("2026-03-01", "2026-03-01T06:00:00+00:00")
        report = health_report(
            champion=_champion(),
            trades=trades,
            drift=None,
            as_of=_MIDNIGHT,
            checks=("calibration",),
        )
        assert report.checks[0].metrics["n_after_as_of"] == 60

    def test_an_unparseable_settled_at_is_unattributed_not_assumed_current(self) -> None:
        trades = self._on("2026-02-28", "yesterday-ish")
        check = self._pnl(trades)
        assert check.metrics["n_unattributed"] == 60
        assert check.metrics["n_trades"] == 0

    def test_a_trade_without_settled_at_falls_back_to_the_target_date_rule(self) -> None:
        late = self._on("2026-03-02", None, n=10)
        early = self._on("2026-02-28", None, n=60)
        check = self._pnl(early + late)
        assert check.metrics["n_after_as_of"] == 10
        assert check.metrics["n_trades"] == 60

    def test_trades_without_settled_at_are_counted_and_the_report_says_so(self) -> None:
        check = self._pnl(self._on("2026-02-28", None))
        assert check.metrics["n_no_settled_at"] == 60
        text = " ".join(check.reasons)
        assert "60" in text and "settled_at" in text and "target_date" in text

    def test_a_trade_with_settled_at_is_not_counted_as_lacking_it(self) -> None:
        check = self._pnl(self._on("2026-02-28", "2026-03-01T00:00:00+00:00"))
        assert check.metrics["n_no_settled_at"] == 0
        assert "settled_at" not in " ".join(check.reasons)

    def test_the_note_is_in_a_calibration_reason_too(self) -> None:
        report = health_report(
            champion=_champion(),
            trades=self._on("2026-02-28", None),
            drift=None,
            as_of=_MIDNIGHT,
            checks=("calibration",),
        )
        assert "settled_at" in " ".join(report.checks[0].reasons)

    def test_the_note_survives_when_the_check_is_insufficient(self) -> None:
        check = self._pnl(self._on("2026-02-28", None, n=5))
        assert check.status == "insufficient"
        assert check.metrics["n_no_settled_at"] == 5
        assert any("settled_at" in r for r in check.reasons)

    def test_the_note_does_not_make_a_healthy_run_noisy(self) -> None:
        report = health_report(
            champion=_champion(),
            trades=self._on("2026-02-28", None),
            drift=None,
            as_of=_MIDNIGHT,
            checks=("pnl",),
        )
        assert report.status == "ok" and report.render() == ""
