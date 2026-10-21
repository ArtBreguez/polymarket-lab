"""The power and false-alarm claims in docs/watchdog.md, checked by simulation.

`scripts/watchdog_power.py` produces every number in the guide's tables. These tests do
two things with a much smaller budget:

- hold its fast, vectorised copies of the checks to the real `health_report`, so a table
  cannot drift away from the code it describes;
- re-assert the claims the guide makes, with tolerances wide enough for the sample sizes
  here: the sequential test's false-alarm rate, what it adds over the fixed test, that
  calibration is not the earlier signal, and that Holm's correction keeps more segments
  from meaning more false alarms.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pmlab.monitoring.stats import cusum
from pmlab.monitoring.watchdog import PnLThresholds


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "watchdog_power.py"
    spec = importlib.util.spec_from_file_location("watchdog_power", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["watchdog_power"] = module
    spec.loader.exec_module(module)
    return module


wp = _load()


class TestTheFastCopiesAgreeWithTheRealCheck:
    @pytest.mark.parametrize("shift", [0.0, -0.06, -0.12])
    def test_the_vectorised_fixed_test_is_health_report(self, shift: float) -> None:
        rng = np.random.default_rng(11)
        for _ in range(12):
            backtest = wp.draw(rng, 400, 0.0)
            champ = wp.champion({"s": backtest})
            live = wp.draw(rng, 300, shift)
            rows = wp.trade_rows(live)
            mu, sd = float(backtest.ret.mean()), float(backtest.ret.std(ddof=1))
            warn = wp.fixed_test_fires(live.ret[None, :], np.array([mu]), np.array([sd]), 400)
            alert = wp.fixed_test_fires(
                live.ret[None, :], np.array([mu]), np.array([sd]), 400, shortfall=1.0, z_min=3.0
            )
            for n in (30, 31, 50, 77, 120, 200, 300):
                status = wp.report(champ, rows[:n], sequential=False).checks[0].status
                assert wp.fires(status) == bool(warn[0, n - 1]), (shift, n, status)
                assert (status == "alert") == bool(alert[0, n - 1]), (shift, n, status)

    @pytest.mark.parametrize("shift", [0.0, -0.06])
    def test_the_vectorised_cusum_is_the_production_cusum_and_health_reports(
        self, shift: float
    ) -> None:
        rng = np.random.default_rng(12)
        defaults = PnLThresholds()
        for _ in range(10):
            backtest = wp.draw(rng, 400, 0.0)
            champ = wp.champion({"s": backtest})
            live = wp.draw(rng, 500, shift)
            mu, sd = float(backtest.ret.mean()), float(backtest.ret.std(ddof=1))
            k = defaults.cusum_allowance * mu / sd + defaults.cusum_backtest_se / np.sqrt(400)
            expected = cusum(((mu - live.ret) / sd).tolist(), [k] * 500, defaults.cusum_threshold)
            first, at = wp.cusum_scan(
                live.ret[None, :],
                np.array([mu]),
                np.array([sd]),
                np.array([k]),
                defaults.cusum_threshold,
                checkpoints=(500,),
            )
            seq = wp.report(champ, wp.trade_rows(live)).checks[0].metrics["sequential"]
            assert seq["crossed_at"] == expected.crossed_at
            assert seq["statistic"] == pytest.approx(expected.statistic)
            assert seq["peak"] == pytest.approx(expected.peak)
            assert at[500][0] == pytest.approx(expected.statistic)
            if expected.crossed_at is None:
                assert np.isinf(first[0])
            else:
                assert first[0] == expected.crossed_at
            assert seq["signalled"] == (expected.statistic >= defaults.cusum_threshold)


class TestTheFastCopyCapsWhatOneTradeMayAddLikeTheRealCheck:
    """A near-certain cell: a 100-trade backtest at +0.0279 +/- 0.002, then 60 normal trades
    and one loss of the whole stake. The real check caps the loss at 5; so must the copy."""

    def _paths(self) -> tuple[Any, Any, float, float, float]:
        ret = np.array([0.0259, 0.0299] * 50)
        backtest = wp.Path_(np.full(100, 0.97), np.full(100, 0.97), ret > 0, ret)
        live_ret = np.array([0.0279] * 60 + [-1.003])
        live = wp.Path_(np.full(61, 0.97), np.full(61, 0.97), live_ret > 0, live_ret)
        mu, sd = float(ret.mean()), float(ret.std(ddof=1))
        k = 0.5 * mu / sd + 0.5 / np.sqrt(100)
        return backtest, live, mu, sd, k

    def test_the_capped_copy_is_the_production_statistic(self) -> None:
        backtest, live, mu, sd, k = self._paths()
        seq = (
            wp.report(wp.champion({"s": backtest}), wp.trade_rows(live))
            .checks[0]
            .metrics["sequential"]
        )
        _, at = wp.cusum_scan(
            live.ret[None, :], np.array([mu]), np.array([sd]), np.array([k]), 30.0, (61,)
        )
        assert seq["statistic"] == pytest.approx(5.0)
        assert at[61][0] == pytest.approx(seq["statistic"])
        assert seq["signalled"] is False

    def test_without_the_cap_the_copy_shows_what_one_trade_used_to_do(self) -> None:
        _, live, mu, sd, k = self._paths()
        _, at = wp.cusum_scan(
            live.ret[None, :],
            np.array([mu]),
            np.array([sd]),
            np.array([k]),
            30.0,
            (61,),
            cap=np.inf,
        )
        assert at[61][0] > 500.0


def _simulate(reps: int = 4000, shift: float = 0.0, **kwargs: Any) -> tuple[Any, Any]:
    defaults = PnLThresholds()
    settings: dict[str, Any] = {
        "allowance": defaults.cusum_allowance,
        "backtest_se": defaults.cusum_backtest_se,
        "h": defaults.cusum_threshold,
        "seed": 1,
        **kwargs,
    }
    return wp.simulate_cusum(reps, 3000, shift, **settings)


class TestTheSequentialTestsFalseAlarmRate:
    """Documented: with a 400-trade backtest, about 3% of champions have crossed the threshold
    at least once within 400 trades and 7% within 800, but the alert is on at only about 1-2%
    of runs at any moment, however long the book, because it clears when a run ends."""

    def test_events_within_400_and_800_trades(self) -> None:
        first, _ = _simulate()
        assert 0.012 <= float(np.mean(first <= 400)) <= 0.045
        assert 0.04 <= float(np.mean(first <= 800)) <= 0.10
        assert float(np.mean(first <= 100)) <= 0.004

    def test_the_alert_on_at_any_moment_stays_low_however_long_the_book(self) -> None:
        _, at = _simulate()
        threshold = PnLThresholds().cusum_threshold
        for n in (400, 800, 1500, 3000):
            assert float(np.mean(at[n] >= threshold)) <= 0.03, n

    def test_a_short_backtest_makes_it_less_trustworthy_which_is_why_there_is_a_floor(self) -> None:
        threshold = PnLThresholds().cusum_threshold
        _, short = _simulate(2500, seed=2, n_bt=30)
        _, long = _simulate(2500, seed=2, n_bt=400)
        assert float(np.mean(short[3000] >= threshold)) > 2 * float(
            np.mean(long[3000] >= threshold)
        )

    def test_the_backtests_uncertainty_in_the_allowance_is_what_bounds_it(self) -> None:
        """Without it the champions whose backtest flattered them drift to a standing alert."""
        threshold = PnLThresholds().cusum_threshold
        _, with_se = _simulate(3000, seed=3, backtest_se=0.5)
        _, without = _simulate(3000, seed=3, backtest_se=0.0)
        assert float(np.mean(without[3000] >= threshold)) > 2 * float(
            np.mean(with_se[3000] >= threshold)
        )


class TestWhatTheSequentialTestAdds:
    def test_it_finds_a_vanished_edge_that_the_fixed_test_misses(self) -> None:
        reps, n = 800, 800
        rng = np.random.default_rng(3)
        q = rng.uniform(0.30, 0.60, (reps, 400))
        backtest = np.where(rng.random((reps, 400)) < q + 0.06, 1.0 / q - 1.0, -1.0) - wp.COST
        mu, sd = backtest.mean(1), backtest.std(1, ddof=1)
        q = rng.uniform(0.30, 0.60, (reps, n))
        live = np.where(rng.random((reps, n)) < q + 0.0, 1.0 / q - 1.0, -1.0) - wp.COST
        defaults = PnLThresholds()
        allowance = defaults.cusum_allowance * mu / sd + defaults.cusum_backtest_se / np.sqrt(400)
        _, at = wp.cusum_scan(live, mu, sd, allowance, defaults.cusum_threshold, (n,))
        fixed = wp.fixed_test_fires(live, mu, sd, 400)[:, n - 1]  # one look, at 800 trades
        either = fixed | (at[n] >= defaults.cusum_threshold)
        assert float(either.mean()) > float(fixed.mean()) + 0.10

    def test_it_catches_a_decay_that_starts_late_which_the_pooled_mean_dilutes(self) -> None:
        _, at = wp.simulate_cusum(
            1500, 3000, -0.06, seed=4, allowance=0.5, backtest_se=0.5, h=30.0, change_at=300,
            checkpoints=(800,),
        )  # fmt: skip
        assert float(np.mean(at[800] >= 30.0)) > 0.20

    def test_in_a_healthy_book_it_says_nothing_before_a_hundred_trades(self) -> None:
        first, _ = _simulate(1500, seed=5)
        assert float(np.mean(first <= 100)) <= 0.002


class TestCalibrationIsNotTheEarlierSignal:
    """The guide used to call calibration an earlier warning than PnL. For a vanishing edge
    it is not: the edge is a few points of probability, and a squared error barely moves
    for a few points. A test of the mean forecast error has the power PnL has, because it is
    the same information; it is not more."""

    def _rates(self, n: int, sims: int = 160) -> tuple[float, float]:
        rng = np.random.default_rng(6)
        pnl = brier = 0
        for _ in range(sims):
            backtest = wp.draw(rng, 400, 0.0)
            champ = wp.champion({"s": backtest})
            rows = wp.trade_rows(wp.draw(rng, n, -0.06))
            checks = {
                c.name: c.status
                for c in wp.report(
                    champ, rows, sequential=False, checks=("calibration", "pnl")
                ).checks
            }
            pnl += wp.fires(checks["pnl"])
            brier += wp.fires(checks["calibration"])
        return pnl / sims, brier / sims

    def test_the_brier_check_is_far_behind_the_pnl_check_at_the_same_size(self) -> None:
        pnl, brier = self._rates(200)
        assert pnl > 0.20
        assert brier < 0.06
        assert pnl > 4 * brier


class TestMoreSegmentsDoNotMeanMoreFalseAlarms:
    def test_holm_holds_the_rate_down_where_independent_tests_do_not(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        wp.table_segments(100, counts=(10,))
        row = capsys.readouterr().out.strip().splitlines()[-1].split()
        alone, holm = (float(x.rstrip("%")) / 100 for x in row[1:3])
        assert alone > 0.15  # ten segments judged alone: a false alarm in about 3 runs of 10
        assert holm < 0.10
        assert alone > 3 * holm


class TestTheScriptRuns:
    @pytest.mark.parametrize(
        "table",
        [
            "firing",
            "cusum",
            "backtest-size",
            "near-certain",
            "continuous",
            "calibration",
            "segments",
            "whole",
        ],
    )
    def test_every_table_prints(self, table: str, capsys: pytest.CaptureFixture[str]) -> None:
        wp.main([table, "--sims", "3", "--reps", "60"])
        out = capsys.readouterr().out
        assert out.startswith(
            "settings: cusum_allowance=0.5, cusum_backtest_se=0.5, cusum_threshold=30"
        )
        assert "%" in out
