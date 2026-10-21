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

import numpy as np
import pytest

from pmlab.monitoring.stats import cusum


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
    def test_the_scripts_cusum_is_health_reports_cusum(self, shift: float) -> None:
        rng = np.random.default_rng(12)
        for _ in range(10):
            backtest = wp.draw(rng, 400, 0.0)
            champ = wp.champion({"s": backtest})
            live = wp.draw(rng, 500, shift)
            mu, sd = float(backtest.ret.mean()), float(backtest.ret.std(ddof=1))
            expected = cusum(((mu - live.ret) / sd).tolist(), [0.5 * mu / sd] * 500, threshold=40.0)
            seq = wp.report(champ, wp.trade_rows(live)).checks[0].metrics["sequential"]
            assert seq["crossed_at"] == expected.crossed_at
            assert seq["statistic"] == pytest.approx(expected.statistic)
            assert seq["peak"] == pytest.approx(expected.peak)


class TestTheSequentialTestsFalseAlarmRate:
    """Documented: about 1.2% of champions signal falsely within their first 400 trades (a
    backtest of 400), next to nothing within 100, and about 5% within 800."""

    def test_within_400_trades_it_is_about_a_percent(self) -> None:
        first = wp.cusum_paths(2500, 400, 0.0, seed=1, allowance=0.5, h=40.0)
        assert 0.004 <= float(np.mean(first <= 400)) <= 0.025
        assert float(np.mean(first <= 100)) <= 0.004

    def test_a_short_backtest_makes_it_less_trustworthy_which_is_why_there_is_a_floor(self) -> None:
        short = wp.cusum_paths(1500, 400, 0.0, seed=2, allowance=0.5, h=40.0, n_bt=30)
        long = wp.cusum_paths(1500, 400, 0.0, seed=2, allowance=0.5, h=40.0, n_bt=400)
        assert float(np.mean(short <= 400)) > 3 * float(np.mean(long <= 400))


class TestWhatTheSequentialTestAdds:
    def test_it_finds_a_vanished_edge_that_the_fixed_test_misses(self) -> None:
        reps = 800
        rng = np.random.default_rng(3)
        n = 800
        live = np.empty((reps, n))
        mus = np.empty(reps)
        sds = np.empty(reps)
        first = np.full(reps, np.inf)
        for r in range(reps):
            bt = wp.draw(rng, 400, 0.0).ret
            mus[r], sds[r] = bt.mean(), bt.std(ddof=1)
            live[r] = wp.draw(rng, n, -0.06).ret
            crossed = cusum(
                ((mus[r] - live[r]) / sds[r]).tolist(), [0.5 * mus[r] / sds[r]] * n, threshold=40.0
            ).crossed_at
            if crossed is not None:
                first[r] = crossed
        fixed = wp.fixed_test_fires(live, mus, sds, 400)[:, n - 1]  # one look, at 800 trades
        either = fixed | (first <= n)
        assert float(either.mean()) > float(fixed.mean()) + 0.15

    def test_it_catches_a_decay_that_starts_late_which_the_pooled_mean_dilutes(self) -> None:
        late = wp.cusum_paths(800, 600, -0.06, seed=4, allowance=0.5, h=40.0, change_at=150)
        assert float(np.mean(late <= 600)) > 0.25

    def test_in_a_healthy_book_it_says_nothing_before_a_hundred_trades(self) -> None:
        early = wp.cusum_paths(500, 100, 0.0, seed=5, allowance=0.5, h=40.0)
        assert float(np.mean(early <= 100)) == 0.0


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
        "table", ["firing", "cusum", "backtest-size", "continuous", "calibration", "segments"]
    )
    def test_every_table_prints(self, table: str, capsys: pytest.CaptureFixture[str]) -> None:
        wp.main([table, "--sims", "3", "--reps", "60"])
        out = capsys.readouterr().out
        assert out.startswith("settings: cusum_allowance=0.5, cusum_threshold=40")
        assert "%" in out
