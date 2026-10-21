"""Sequential detection and the minimum detectable shortfall in the PnL check.

A fixed-sample z test at 100 trades notices a vanished edge about one run in five: per
trade PnL is far noisier than its mean. Two things can be done about it, and both are
here:

- use the evidence as it accumulates: a one-sided CUSUM on each trade's standardized
  shortfall, in the order the trades settled, signals a run of shortfalls that the
  pooled mean dilutes with the healthy trades before it;
- say what a quiet run is worth: every PnL result reports the smallest shortfall it
  could have detected at its size.

Fixtures use two-point samples (`mean +/- sd`, alternating), so the arithmetic is exact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pmlab.backtest.expectation import BacktestExpectation, GroupExpectation, SegmentExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring.stats import normal_isf, normal_sf
from pmlab.monitoring.watchdog import (
    HealthReport,
    PnLThresholds,
    WatchdogConfig,
    WatchdogInputError,
    health_report,
)

_PUBLISHED = "2026-01-01T00:00:00+00:00"
_AS_OF = "2026-09-01T00:00:00+00:00"
_MEAN, _SD = 0.20, 0.60


def _expectation(**overrides: Any) -> BacktestExpectation:
    base: dict[str, Any] = {
        "n_trades": 400,
        "flat_stake": 1.0,
        "mean_pnl_per_stake": _MEAN,
        "std_pnl_per_stake": _SD,
        "brier_score": 0.20,
        "brier_std": 0.10,
    }
    base.update(overrides)
    return BacktestExpectation(**base)


def _champion(
    expectation: BacktestExpectation, passing: tuple[str, ...] = ("A",)
) -> ChampionManifest:
    gate = HoldoutGateResult(
        decision="GO",
        segment_results=[SegmentGateResult(s, 100, 10.0, True, "ok") for s in passing],
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


def _trades(
    n: int,
    *,
    pnl: float = _MEAN,
    pnl_sd: float = _SD,
    day0: int = 0,
    side: str = "yes",
    segment: str = "A",
    settled: bool = True,
) -> list[dict[str, Any]]:
    """`n` trades settled one per hour from `day0` days after Feb 1: the order they arrive in."""
    rows = []
    for i, ret in enumerate(_two_point(n, pnl, pnl_sd)):
        hour = day0 * 24 + i
        rows.append(
            {
                "recorded_at": "2026-02-01T00:00:00+00:00",
                "city_or_segment": segment,
                "target_date": f"2026-0{3 + hour // (24 * 30)}-{(hour // 24) % 28 + 1:02d}",
                "horizon": "morning_of",
                "outcome_label": "YES",
                "direction": side,
                "gamma_price": 0.40,
                "model_prob": 0.4,
                "flat_stake": 1.0,
                "outcome": "lost",
                "realized_pnl": ret,
                "settlement_version": 1,
                "settled_basis": "quote",
                **(
                    {"settled_at": f"2026-03-{1 + hour // 24 % 28:02d}T{hour % 24:02d}:00:00+00:00"}
                    if settled
                    else {}
                ),
            }
        )
    return rows


def _settled_at(hours_after: int) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime(2026, 3, 1, tzinfo=UTC) + timedelta(hours=hours_after)).isoformat()


def _in_order(*batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate batches and stamp `settled_at` hour by hour, in that order."""
    rows = [dict(t) for batch in batches for t in batch]
    for hour, row in enumerate(rows):
        row["settled_at"] = _settled_at(hour)
        row["target_date"] = f"2026-03-{1 + hour // 24 % 28:02d}"
    return rows


def _report(
    trades: list[dict[str, Any]],
    expectation: BacktestExpectation | None = None,
    *,
    config: WatchdogConfig | None = None,
    passing: tuple[str, ...] = ("A",),
) -> HealthReport:
    return health_report(
        champion=_champion(expectation or _expectation(), passing),
        trades=trades,
        drift=None,
        as_of=_AS_OF,
        config=config,
        checks=("pnl",),
    )


def _pnl(report: HealthReport):
    return report.checks[0]


class TestTheSettings:
    def test_the_defaults(self) -> None:
        cfg = PnLThresholds()
        assert cfg.sequential is True
        assert cfg.cusum_allowance == 0.5
        assert cfg.cusum_threshold == 40.0
        assert cfg.cusum_min_backtest_trades == 100
        assert cfg.mde_power == 0.8

    def test_they_round_trip_and_partial_configs_keep_the_rest(self) -> None:
        cfg = WatchdogConfig.from_dict({"pnl": {"cusum_threshold": 25, "sequential": False}})
        assert cfg.pnl.cusum_threshold == 25
        assert cfg.pnl.sequential is False
        assert cfg.pnl.cusum_allowance == 0.5
        assert WatchdogConfig.from_dict(cfg.to_dict()) == cfg

    @pytest.mark.parametrize(
        "bad",
        [
            {"cusum_threshold": 0},
            {"cusum_threshold": -5},
            {"cusum_threshold": math.inf},
            {"cusum_threshold": math.nan},
            {"cusum_allowance": -0.1},
            {"cusum_allowance": math.nan},
            {"cusum_min_backtest_trades": 1},
            {"cusum_min_backtest_trades": 2.5},
            {"cusum_min_backtest_trades": True},
            {"mde_power": 0.0},
            {"mde_power": 1.0},
            {"mde_power": 1.5},
            {"sequential": "yes"},
            {"sequential": 1},
        ],
    )
    def test_nonsense_is_refused(self, bad: dict[str, Any]) -> None:
        with pytest.raises(WatchdogInputError):
            PnLThresholds(**bad)

    def test_a_typo_is_still_refused(self) -> None:
        with pytest.raises(WatchdogInputError, match="cusum_treshold"):
            WatchdogConfig.from_dict({"pnl": {"cusum_treshold": 10}})

    def test_a_report_written_before_these_settings_still_loads(self) -> None:
        data = _report(_trades(60)).to_dict()
        for key in ("sequential", "cusum_allowance", "cusum_threshold", "mde_power"):
            del data["config"]["pnl"][key]
        again = HealthReport.from_dict(data)
        assert again.config.pnl.cusum_threshold == 40.0


class TestTheSequentialTest:
    def test_a_constant_stream_has_a_known_statistic(self) -> None:
        # Every trade returns -0.4 against a promise of +0.2 +/- 0.6.
        #   standardized shortfall x = (0.2 + 0.4) / 0.6 = 1.0
        #   allowance k = 0.5 * 0.2 / 0.6 = 1/6        -> each step adds 5/6
        trades = _in_order(_trades(60, pnl=-0.4, pnl_sd=0.0))
        seq = _pnl(_report(trades)).metrics["sequential"]
        assert seq["n"] == 60
        assert seq["statistic"] == pytest.approx(60 * 5 / 6)
        assert seq["peak"] == pytest.approx(60 * 5 / 6)
        assert seq["allowance"] == pytest.approx(1 / 6)
        assert seq["threshold"] == 40.0
        assert seq["crossed_at"] == 48  # ceil(40 / (5/6))
        assert seq["signalled"] is True

    def test_a_healthy_book_does_not_signal(self) -> None:
        seq = _pnl(_report(_in_order(_trades(500)))).metrics["sequential"]
        assert seq["signalled"] is False
        assert seq["crossed_at"] is None
        assert seq["statistic"] < 40.0

    def test_a_slow_decay_is_caught_only_by_the_sequential_test(self) -> None:
        """400 trades as promised, then the edge is gone. The pooled mean over 670 trades
        is 40% short of the promise: not a finding. The run of shortfalls at the end is.

        After the change each trade adds (0.2 - 0) / 0.6 - 0.5 * 0.2 / 0.6 = 1/6 to the
        statistic, so it reaches 40 after 240 trades."""
        trades = _in_order(_trades(400), _trades(270, pnl=0.0))
        check = _pnl(_report(trades))
        assert check.metrics["shortfall_rel"] == pytest.approx(0.403, abs=0.01)
        assert check.metrics["shortfall_rel"] < 0.5  # the fixed test says: fine
        assert check.status == "alert"
        assert check.reasons[0].startswith("sequential:")
        seq = check.metrics["sequential"]
        assert seq["signalled"] is True
        assert seq["crossed_at"] == pytest.approx(400 + 240, abs=4)

    def test_the_line_says_where_it_signalled(self) -> None:
        trades = _in_order(_trades(400), _trades(270, pnl=0.0))
        seq = _pnl(_report(trades)).metrics["sequential"]
        line = _pnl(_report(trades)).reasons[0]
        assert f"trade {seq['crossed_at']} of 670" in line
        assert "CUSUM" in line

    def test_a_signal_is_remembered_after_the_book_recovers(self) -> None:
        trades = _in_order(_trades(100, pnl=-0.6), _trades(300))
        check = _pnl(_report(trades))
        seq = check.metrics["sequential"]
        assert seq["signalled"] is True
        assert seq["statistic"] < seq["peak"]  # it has come back down since
        assert check.status == "alert"

    def test_the_fixed_test_and_the_sequential_test_can_both_fire(self) -> None:
        check = _pnl(_report(_in_order(_trades(200, pnl=-0.6))))
        assert check.status == "alert"
        assert check.metrics["sequential"]["signalled"] is True
        assert len([r for r in check.reasons if "shortfall" in r]) >= 1

    def test_it_can_be_switched_off(self) -> None:
        config = WatchdogConfig(pnl=PnLThresholds(sequential=False))
        trades = _in_order(_trades(400), _trades(270, pnl=0.0))
        check = _pnl(_report(trades, config=config))
        assert "sequential" not in check.metrics
        assert check.status == "ok"

    def test_a_lower_threshold_signals_earlier(self) -> None:
        trades = _in_order(_trades(60, pnl=-0.4, pnl_sd=0.0))
        config = WatchdogConfig(pnl=PnLThresholds(cusum_threshold=10.0))
        assert _pnl(_report(trades, config=config)).metrics["sequential"]["crossed_at"] == 12

    def test_the_threshold_is_inclusive_to_the_last_digit(self) -> None:
        trades = _in_order(_trades(300), _trades(120, pnl=0.0))
        huge = WatchdogConfig(pnl=PnLThresholds(cusum_threshold=1e9))
        peak = _pnl(_report(trades, config=huge)).metrics["sequential"]["peak"]
        at = WatchdogConfig(pnl=PnLThresholds(cusum_threshold=peak))
        above = WatchdogConfig(pnl=PnLThresholds(cusum_threshold=peak + 1e-6))
        assert _pnl(_report(trades, config=at)).metrics["sequential"]["signalled"] is True
        assert _pnl(_report(trades, config=above)).metrics["sequential"]["signalled"] is False

    def test_the_allowance_is_a_fraction_of_the_promised_edge(self) -> None:
        trades = _in_order(_trades(60, pnl=-0.4, pnl_sd=0.0))
        none = WatchdogConfig(pnl=PnLThresholds(cusum_allowance=0.0, cusum_threshold=1e9))
        full = WatchdogConfig(pnl=PnLThresholds(cusum_allowance=1.0, cusum_threshold=1e9))
        # k = a * 0.2 / 0.6; x = 1.0
        assert _pnl(_report(trades, config=none)).metrics["sequential"][
            "statistic"
        ] == pytest.approx(60.0)
        assert _pnl(_report(trades, config=full)).metrics["sequential"][
            "statistic"
        ] == pytest.approx(60 * (1 - 1 / 3))


class TestTheOrderTheTradesSettledIn:
    def _cfg(self) -> WatchdogConfig:
        return WatchdogConfig(pnl=PnLThresholds(cusum_threshold=1e9))

    def test_the_file_order_does_not_matter(self) -> None:
        trades = _in_order(_trades(100), _trades(100, pnl=0.0))
        rng = np.random.default_rng(1)
        shuffled = [trades[i] for i in rng.permutation(len(trades))]
        a = _pnl(_report(trades, config=self._cfg())).metrics["sequential"]
        b = _pnl(_report(shuffled, config=self._cfg())).metrics["sequential"]
        assert a == b

    def test_it_is_the_settlement_order_that_counts(self) -> None:
        bad, good = _trades(40, pnl=-0.4, pnl_sd=0.0), _trades(40, pnl=0.8, pnl_sd=0.0)
        losses_last = _in_order(good, bad)
        losses_first = _in_order(bad, good)
        last = _pnl(_report(losses_last, config=self._cfg())).metrics["sequential"]
        first = _pnl(_report(losses_first, config=self._cfg())).metrics["sequential"]
        assert last["statistic"] == pytest.approx(40 * 5 / 6)
        assert first["statistic"] == 0.0  # the later wins wiped the earlier losses
        assert first["peak"] == pytest.approx(40 * 5 / 6)

    def test_trades_settled_in_the_same_hour_are_ordered_by_their_identity(self) -> None:
        a = _trades(60, pnl=-0.4, pnl_sd=0.0, segment="A")
        b = _trades(60, pnl=0.8, pnl_sd=0.0, segment="B")
        for row in a + b:
            row["settled_at"] = "2026-03-01T00:00:00+00:00"
            row["target_date"] = "2026-03-01"
        cfg = self._cfg()
        expectation = _expectation()
        one = _pnl(_report(a + b, expectation, config=cfg, passing=("A", "B"))).metrics[
            "sequential"
        ]
        two = _pnl(_report(b + a, expectation, config=cfg, passing=("A", "B"))).metrics[
            "sequential"
        ]
        assert one == two

    def test_a_trade_from_an_older_release_is_ordered_by_its_target_date(self) -> None:
        old = _in_order(_trades(40, pnl=-0.4, pnl_sd=0.0))
        for row in old:
            del row["settled_at"]
        seq = _pnl(_report(old, config=self._cfg())).metrics["sequential"]
        assert seq["n"] == 40


class TestEachSideIsStandardizedByItsOwnPromise:
    def _two_sided(self) -> BacktestExpectation:
        return _expectation(
            by_direction={
                "yes": GroupExpectation(250, 0.20, 0.60, 0.20, 0.10),
                "no": GroupExpectation(150, 0.10, 0.40, 0.15, 0.08),
            }
        )

    def test_a_book_as_promised_on_both_sides_is_quiet(self) -> None:
        trades = _in_order(
            [
                t
                for pair in zip(
                    _trades(150, pnl=0.20, pnl_sd=0.60),
                    _trades(150, pnl=0.10, pnl_sd=0.40, side="no"),
                    strict=True,
                )
                for t in pair
            ]
        )
        seq = _pnl(_report(trades, self._two_sided())).metrics["sequential"]
        assert seq["signalled"] is False
        assert seq["n"] == 300

    def test_no_trades_are_judged_against_the_no_promise(self) -> None:
        # NO earns 0.10 +/- 0.40 as promised: x = +/-1 around 0, never a shortfall that
        # accumulates. Against the YES promise (0.20 +/- 0.60) the same trades would have
        # looked like a steady shortfall of 0.1/0.6.
        trades = _in_order(_trades(300, pnl=0.10, pnl_sd=0.40, side="no"))
        cfg = WatchdogConfig(pnl=PnLThresholds(cusum_threshold=1e9))
        seq = _pnl(_report(trades, self._two_sided(), config=cfg)).metrics["sequential"]
        assert seq["statistic"] < 5.0


class TestAPromiseMadeOnFewTradesCannotAnchorTheSequentialTest:
    """The backtest mean is an estimate. The fixed test counts its uncertainty in the
    standard error; the CUSUM cannot, and a mean estimated from 30 trades is wrong by
    far more than the allowance, so it would drift to a false alarm. Simulated: about 1.4%
    of champions signal falsely within 400 trades when the backtest had 400 trades, 3.8%
    at 100, and 10% at 30. Below the floor the test does not run."""

    def test_a_short_backtest_switches_the_test_off_and_says_why(self) -> None:
        expectation = _expectation(n_trades=60)
        seq = _pnl(_report(_in_order(_trades(100)), expectation)).metrics["sequential"]
        assert seq["status"] == "not_applicable"
        assert seq["signalled"] is False
        assert "fewer than 100" in seq["reason"]
        assert seq["n_skipped"] == 100

    def test_a_losing_streak_on_a_short_backtest_is_left_to_the_fixed_test(self) -> None:
        expectation = _expectation(n_trades=60)
        check = _pnl(_report(_in_order(_trades(40, pnl=-0.4, pnl_sd=0.0)), expectation))
        assert check.metrics["sequential"]["signalled"] is False
        assert check.status == "alert"  # the fixed test still sees it
        assert not check.reasons[0].startswith("sequential")

    def test_exactly_the_floor_is_enough(self) -> None:
        seq = _pnl(_report(_in_order(_trades(60)), _expectation(n_trades=100))).metrics[
            "sequential"
        ]
        assert seq["status"] == "ok"

    def test_the_floor_is_configurable(self) -> None:
        config = WatchdogConfig(pnl=PnLThresholds(cusum_min_backtest_trades=50))
        seq = _pnl(_report(_in_order(_trades(60)), _expectation(n_trades=60), config=config))
        assert seq.metrics["sequential"]["status"] == "ok"

    def test_each_side_is_held_to_the_floor_on_its_own(self) -> None:
        expectation = _expectation(
            n_trades=400,
            by_direction={
                "yes": GroupExpectation(300, 0.20, 0.60, 0.20, 0.10),
                "no": GroupExpectation(40, 0.10, 0.40, 0.15, 0.08),
            },
        )
        trades = _in_order(_trades(60) + _trades(60, pnl=0.10, pnl_sd=0.40, side="no"))
        seq = _pnl(_report(trades, expectation)).metrics["sequential"]
        assert seq["n"] == 60
        assert seq["n_skipped"] == 60


class TestWhenTheSequentialTestCannotRun:
    def test_a_side_with_no_spread_cannot_be_standardized(self) -> None:
        expectation = _expectation(std_pnl_per_stake=0.0)
        check = _pnl(_report(_in_order(_trades(60, pnl=0.2, pnl_sd=0.0)), expectation))
        seq = check.metrics["sequential"]
        assert seq["status"] == "not_applicable"
        assert seq["signalled"] is False
        assert "spread" in seq["reason"]

    def test_a_side_without_a_positive_promise_is_skipped_and_counted(self) -> None:
        expectation = _expectation(
            by_direction={
                "yes": GroupExpectation(250, 0.20, 0.60, 0.20, 0.10),
                "no": GroupExpectation(150, -0.05, 0.40, 0.15, 0.08),
            }
        )
        trades = _in_order(_trades(100) + _trades(50, pnl=-0.05, pnl_sd=0.4, side="no"))
        seq = _pnl(_report(trades, expectation)).metrics["sequential"]
        assert seq["n"] == 100
        assert seq["n_skipped"] == 50

    def test_below_min_trades_the_check_is_insufficient_and_says_nothing_sequential(self) -> None:
        check = _pnl(_report(_in_order(_trades(20, pnl=-5.0, pnl_sd=0.0))))
        assert check.status == "insufficient"
        assert "sequential" not in check.metrics


class TestTheSequentialTestIsReportedLikeTheRest:
    def test_the_json_round_trips(self) -> None:
        trades = _in_order(_trades(400), _trades(270, pnl=0.0))
        report = _report(trades)
        again = HealthReport.from_dict(json.loads(json.dumps(report.to_dict(), allow_nan=False)))
        assert again == report

    def test_a_healthy_run_prints_nothing_but_verbose_shows_the_statistic(self) -> None:
        report = _report(_in_order(_trades(100)))
        assert report.render() == ""
        assert "CUSUM" in report.render(verbose=True)

    def test_segments_and_sides_do_not_each_get_a_cusum(self) -> None:
        expectation = _expectation(
            by_segment={
                "A": SegmentExpectation(200, 0.20, 0.60),
                "B": SegmentExpectation(200, 0.20, 0.60),
            }
        )
        trades = _in_order(_trades(60, segment="A"), _trades(60, segment="B"))
        check = _pnl(_report(trades, expectation, passing=("A", "B")))
        assert "sequential" in check.metrics
        assert all("sequential" not in v for v in check.metrics["segments"].values())


class TestTheMinimumDetectableShortfall:
    def _report(self, n: int = 100, **kwargs: Any) -> Any:
        return _pnl(_report(_in_order(_trades(n)), **kwargs))

    def _expected(self, n: int, z_alpha: float = 2.0, power: float = 0.8) -> float:
        live = np.array(_two_point(n, _MEAN, _SD))
        se = math.sqrt(live.var(ddof=1) / n + _SD**2 / 400)
        gap = (z_alpha + normal_isf(1 - power)) * se
        return gap / _MEAN

    def test_it_is_the_gap_a_80_percent_powered_test_could_see(self) -> None:
        check = self._report(100)
        assert check.metrics["mde_shortfall"] == pytest.approx(self._expected(100))
        assert check.metrics["mde_shortfall"] > 0.9  # about 96%: weak, and it says so

    def test_it_shrinks_as_trades_accumulate(self) -> None:
        small, large = self._report(60), self._report(400)
        assert large.metrics["mde_shortfall"] < small.metrics["mde_shortfall"]
        assert large.metrics["mde_shortfall"] == pytest.approx(self._expected(400))

    def test_the_effective_figure_is_never_below_the_effect_threshold(self) -> None:
        # With plenty of trades the noise would allow 26%, but the check needs 50% anyway.
        check = self._report(2000, expectation=_expectation(n_trades=20000))
        assert check.metrics["mde_shortfall"] < 0.5
        assert check.metrics["mde_shortfall_effective"] == 0.5
        assert "50%" in check.reasons[0]

    def test_a_quiet_run_says_what_it_could_see(self) -> None:
        check = self._report(100)
        assert check.status == "ok"
        assert (
            "could only detect a shortfall >= 96% of the backtest edge at n=100" in check.reasons[0]
        )

    def test_a_flagged_run_says_it_too(self) -> None:
        trades = _in_order(_trades(100, pnl=-0.4, pnl_sd=0.0))
        check = _pnl(_report(trades))
        assert check.status == "alert"
        assert "could only detect" in " ".join(check.reasons)

    def test_it_follows_the_configured_alpha_and_power(self) -> None:
        loose = self._report(100, config=WatchdogConfig(pnl=PnLThresholds(z_warn=1.5, z_alert=3.0)))
        strict = self._report(
            100, config=WatchdogConfig(pnl=PnLThresholds(z_warn=2.5, z_alert=3.0))
        )
        assert loose.metrics["mde_shortfall"] == pytest.approx(self._expected(100, z_alpha=1.5))
        assert strict.metrics["mde_shortfall"] == pytest.approx(self._expected(100, z_alpha=2.5))
        high = self._report(100, config=WatchdogConfig(pnl=PnLThresholds(mde_power=0.95)))
        assert high.metrics["mde_shortfall"] == pytest.approx(self._expected(100, power=0.95))

    def test_a_side_inside_a_family_pays_for_the_correction(self) -> None:
        expectation = _expectation(
            by_direction={
                "yes": GroupExpectation(250, 0.20, 0.60, 0.20, 0.10),
                "no": GroupExpectation(150, 0.20, 0.60, 0.20, 0.10),
            }
        )
        trades = _in_order(_trades(100), _trades(100, side="no"))
        check = _pnl(_report(trades, expectation))
        yes = check.metrics["directions"]["yes"]["mde_shortfall"]
        # Two sides: the level each must meet is alpha / 2, not alpha.
        z_family = normal_isf(normal_sf(2.0) / 2)
        live = np.array(_two_point(100, _MEAN, _SD))
        se = math.sqrt(live.var(ddof=1) / 100 + _SD**2 / 250)
        assert yes == pytest.approx((z_family + normal_isf(0.2)) * se / _MEAN)
        alone = (2.0 + normal_isf(0.2)) * se / _MEAN
        assert yes > alone  # and is not what it would have been as a lone test

    def test_a_segment_reports_its_own(self) -> None:
        expectation = _expectation(
            by_segment={
                "A": SegmentExpectation(200, 0.20, 0.60),
                "B": SegmentExpectation(200, 0.20, 0.60),
            }
        )
        trades = _in_order(_trades(60, segment="A"), _trades(60, segment="B"))
        check = _pnl(_report(trades, expectation, passing=("A", "B")))
        # 60 trades against 120 in total: a segment is a coarser instrument than the book.
        assert check.metrics["segments"]["A"]["mde_shortfall"] > check.metrics["mde_shortfall"]
        assert "could only detect a shortfall" in check.reasons[-1] or any(
            "segment A" in r or "could only detect" in r for r in check.reasons
        )

    def test_the_formula_really_is_80_percent_power(self) -> None:
        """Simulate the check's own z test at a true shortfall equal to the reported minimum,
        with the standard error known: it clears z_warn 80% of the time."""
        n = 100
        mde = self._report(n).metrics["mde_shortfall"]
        rng = np.random.default_rng(3)
        se = math.sqrt(_SD**2 / n + _SD**2 / 400)
        truth = _MEAN * (1 - mde)
        # The live mean is N(truth, _SD**2 / n); the backtest mean is N(_MEAN, _SD**2 / 400).
        live_mean = rng.normal(truth, _SD / math.sqrt(n), 200_000)
        backtest_mean = rng.normal(_MEAN, _SD / math.sqrt(400), 200_000)
        rate = float(np.mean((backtest_mean - live_mean) / se >= 2.0))
        assert rate == pytest.approx(0.8, abs=0.008)

    def test_it_holds_up_when_the_variance_has_to_be_estimated_too(self) -> None:
        """The same simulation with the live variance estimated from each sample, as the
        check does: a little more than 80%, because the estimate is noisy."""
        n = 100
        mde = self._report(n).metrics["mde_shortfall"]
        rng = np.random.default_rng(4)
        truth = _MEAN * (1 - mde)
        hits = 0
        reps = 8000
        for _ in range(reps):
            live = rng.normal(truth, _SD, n)
            backtest_mean = rng.normal(_MEAN, _SD / math.sqrt(400))
            z = (backtest_mean - live.mean()) / math.sqrt(live.var(ddof=1) / n + _SD**2 / 400)
            hits += z >= 2.0
        assert 0.78 <= hits / reps <= 0.86
