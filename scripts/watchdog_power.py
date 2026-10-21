#!/usr/bin/env python3
"""How good is the watchdog at noticing a champion that stopped working?

Every power and false-alarm number in docs/watchdog.md comes from this script, so they
can be regenerated and checked::

    uv run python scripts/watchdog_power.py            # everything (a few minutes)
    uv run python scripts/watchdog_power.py firing     # one table
    uv run python scripts/watchdog_power.py --quick    # a fast, noisier run

The world is the one the guide has always used: every live trade buys a label at a price
between 0.30 and 0.60, the model forecasts the price plus 6 points, and the backtest
promised exactly that. "Edge halved" means the label occurs 3 points less often than the
forecast says, "vanished" 6 points less (the market now prices it correctly), "reversed"
12 points less. Costs are 0.3% of the stake.

Most tables call `health_report`, the real thing. The few that follow a book trade by
trade (continuous monitoring, thousands of paths of thousands of trades) use a vectorised
copy of the fixed z test and of the CUSUM recursion; `tests/monitoring/test_power.py` checks
that both agree with `health_report` and with the production `cusum`.

Two ways of counting a CUSUM alarm appear, because they answer different questions. The
watchdog alerts while the statistic is above the threshold, so "alert at a run after N
trades" is what a cron job sees. "Crossed at least once within N trades" is the chance that
*some* run along the way saw an alert: it only grows with the length of the book.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmlab import (
    BacktestExpectation,
    ChampionManifest,
    HoldoutGateResult,
    PnLThresholds,
    SegmentGateResult,
    WatchdogConfig,
    health_report,
)

COST = 0.003
EDGES = {"as promised": 0.0, "halved": -0.03, "vanished": -0.06, "reversed": -0.12}
_AS_OF = "2030-01-01T00:00:00+00:00"
_PUBLISHED = "2026-01-01T00:00:00+00:00"
_T0 = datetime(2026, 2, 1, tzinfo=UTC)


# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Path_:
    """A stretch of live (or backtest) trades: price, forecast, outcome, return per stake."""

    q: np.ndarray
    p: np.ndarray
    y: np.ndarray
    ret: np.ndarray


def draw(
    rng: np.random.Generator,
    n: int,
    shift: float,
    lo: float = 0.30,
    hi: float = 0.60,
    change_at: int = 0,
) -> Path_:
    """`n` trades. The label occurs with probability forecast + `shift` from trade `change_at` on."""
    q = rng.uniform(lo, hi, n)
    p = q + 0.06
    effective = np.where(np.arange(n) < change_at, 0.0, shift)
    y = rng.random(n) < p + effective
    ret = np.where(y, 1.0 / q - 1.0, -1.0) - COST
    return Path_(q, p, y, ret)


def trade_rows(path: Path_, segment: str = "s", start: int = 0) -> list[dict[str, Any]]:
    """The trade-log rows `PaperBroker` and `SettlementEngine` would leave for this path."""
    rows = []
    for i in range(len(path.ret)):
        settled = _T0 + timedelta(hours=start + i)
        rows.append(
            {
                "recorded_at": "2026-01-15T00:00:00+00:00",
                "city_or_segment": segment,
                "target_date": settled.date().isoformat(),
                "settled_at": settled.isoformat(),
                "horizon": "morning_of",
                "outcome_label": f"m{start + i}",
                "direction": "yes",
                "flat_stake": 1.0,
                "model_prob": float(path.p[i]),
                "outcome": "won" if path.y[i] else "lost",
                "realized_pnl": float(path.ret[i]),
            }
        )
    return rows


def champion(segments: dict[str, Path_], published_at: str = _PUBLISHED) -> ChampionManifest:
    """A champion promoted on these backtest paths, one per segment."""
    frames = [
        pd.DataFrame(
            {
                "segment": name,
                "direction": "yes",
                "realized_pnl": path.ret,
                "predicted_prob": path.p,
                "label_occurred": path.y,
            }
        )
        for name, path in segments.items()
    ]
    expectation = BacktestExpectation.from_trades(pd.concat(frames, ignore_index=True))
    gate = HoldoutGateResult(
        "GO",
        [
            SegmentGateResult(name, len(path.ret), 1.0, True, "ok")
            for name, path in segments.items()
        ],
        float(len(segments)),
        sum(len(path.ret) for path in segments.values()),
    )
    return ChampionManifest(
        "sim",
        Path("x"),
        None,
        gate,
        published_at,
        "sim",
        expectation=expectation,
    )


def report(
    champ: ChampionManifest,
    rows: list[dict[str, Any]],
    *,
    sequential: bool = True,
    checks: tuple[str, ...] = ("pnl",),
    **thresholds: Any,
) -> Any:
    config = WatchdogConfig(pnl=PnLThresholds(min_trades=30, sequential=sequential, **thresholds))
    return health_report(
        champion=champ, trades=rows, drift=None, as_of=_AS_OF, config=config, checks=checks
    )


def fires(status: str) -> bool:
    return status in ("warn", "alert")


def _pct(x: float) -> str:
    return f"{x:6.1%}"


# --------------------------------------------------------------------------
# 1. How often does the PnL check fire?
# --------------------------------------------------------------------------


def table_firing(sims: int, sizes: tuple[int, ...] = (30, 50, 100, 200, 400, 800)) -> None:
    """The PnL check run once on a book of n trades, `sims` champions per cell."""
    print(f"How often the PnL check warns or alerts ({sims} simulated champions per cell)")
    print("                   fixed test only                    fixed test or sequential test")
    header = "live trades   " + "".join(f"{name:>13}" for name in EDGES)
    print(header + "   |" + "".join(f"{name:>13}" for name in EDGES))
    results: dict[tuple[int, str, bool], int] = {}
    for edge_name, edge in EDGES.items():
        rng = np.random.default_rng(1000 + int(round(edge * 1000)))
        for _ in range(sims):
            champ = champion({"s": draw(rng, 400, 0.0)})
            rows = trade_rows(draw(rng, max(sizes), edge))
            for n in sizes:
                for sequential in (False, True):
                    status = report(champ, rows[:n], sequential=sequential).checks[0].status
                    results[(n, edge_name, sequential)] = results.get(
                        (n, edge_name, sequential), 0
                    ) + fires(status)
    for n in sizes:
        left = "".join(f"{_pct(results[(n, e, False)] / sims):>13}" for e in EDGES)
        right = "".join(f"{_pct(results[(n, e, True)] / sims):>13}" for e in EDGES)
        print(f"{n:>11}   {left}   |{right}")
    print()


# --------------------------------------------------------------------------
# 2. The sequential test on its own
# --------------------------------------------------------------------------

LONG_BOOKS = (100, 200, 400, 800, 1500, 3000)


def cusum_scan(
    live: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    allowance: np.ndarray,
    h: float,
    checkpoints: tuple[int, ...] = LONG_BOOKS,
    cap: float = PnLThresholds().cusum_increment_cap,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """A vectorised copy of `cusum` over many paths: (first crossing, statistic at checkpoints).

    `live` is (champions, trades) of returns per stake; `mu`/`sd` are each champion's backtest
    mean and spread, `allowance` its per-trade allowance, `cap` the most one trade may add
    (`np.inf` switches the cap off). Checked against the production `cusum` in
    tests/monitoring/test_power.py.
    """
    reps, n = live.shape
    statistic = np.zeros(reps)
    first = np.full(reps, np.inf)
    at: dict[int, np.ndarray] = {}
    for i in range(n):
        statistic = np.maximum(0.0, statistic + np.minimum((mu - live[:, i]) / sd - allowance, cap))
        first[(statistic >= h) & np.isinf(first)] = i + 1
        if i + 1 in checkpoints:
            at[i + 1] = statistic.copy()
    return first, at


def simulate_cusum(
    reps: int,
    n: int,
    shift: float,
    *,
    seed: int,
    allowance: float,
    backtest_se: float,
    h: float,
    n_bt: int = 400,
    change_at: int = 0,
    checkpoints: tuple[int, ...] = LONG_BOOKS,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """First crossing per simulated champion (inf: never) and the statistic at each checkpoint."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.30, 0.60, (reps, n_bt))
    backtest = np.where(rng.random((reps, n_bt)) < q + 0.06, 1.0 / q - 1.0, -1.0) - COST
    mu, sd = backtest.mean(1), backtest.std(1, ddof=1)
    k = allowance * mu / sd + backtest_se / np.sqrt(n_bt)
    q = rng.uniform(0.30, 0.60, (reps, n))
    effect = np.where(np.arange(n) < change_at, 0.0, shift)
    live = np.where(rng.random((reps, n)) < q + 0.06 + effect, 1.0 / q - 1.0, -1.0) - COST
    return cusum_scan(live, mu, sd, k, h, checkpoints)


def table_backtest_size(reps: int, allowance: float, backtest_se: float, h: float) -> None:
    print("The sequential test when the backtest is short (the promised mean is an estimate)")
    print(f"{reps} simulated champions per row")
    print()
    print(
        f"{'backtest trades':<18}{'crossed within 400':>20}{'alert at 3000':>15}"
        f"{'vanished: alert at 400':>24}{'at 800':>9}"
    )
    for n_bt in (30, 50, 100, 200, 400, 1000):
        first, at = simulate_cusum(
            reps, 3000, 0.0, seed=70, allowance=allowance, backtest_se=backtest_se, h=h, n_bt=n_bt
        )
        _, gone = simulate_cusum(
            reps, 800, -0.06, seed=71, allowance=allowance, backtest_se=backtest_se, h=h, n_bt=n_bt
        )
        print(
            f"{n_bt:<18}{_pct(float(np.mean(first <= 400))):>20}"
            f"{_pct(float(np.mean(at[3000] >= h))):>15}"
            f"{_pct(float(np.mean(gone[400] >= h))):>24}{_pct(float(np.mean(gone[800] >= h))):>9}"
        )
    print()


def table_cusum(reps: int, allowance: float, backtest_se: float, h: float) -> None:
    print(
        f"The sequential test alone (allowance {allowance:g} x the promised edge + "
        f"{backtest_se:g} standard errors of the promise, threshold {h:g}, a 400-trade backtest,"
        f" {reps} simulated champions per row)"
    )
    scenarios: list[tuple[str, float, int]] = [
        ("as promised", 0.0, 0),
        ("halved", -0.03, 0),
        ("vanished", -0.06, 0),
        ("reversed", -0.12, 0),
        ("halved after 150", -0.03, 150),
        ("vanished after 150", -0.06, 150),
    ]
    results = [
        (
            name,
            *simulate_cusum(
                reps,
                max(LONG_BOOKS),
                shift,
                seed=50 + i,
                allowance=allowance,
                backtest_se=backtest_se,
                h=h,
                change_at=change_at,
            ),
        )
        for i, (name, shift, change_at) in enumerate(scenarios)
    ]
    header = f"{'live edge':<22}" + "".join(f"{'N=' + str(n):>8}" for n in LONG_BOOKS)
    print()
    print("share of champions whose alert is on at the run after N live trades")
    print(header)
    for name, _, at in results:
        print(f"{name:<22}" + "".join(f"{_pct(float(np.mean(at[n] >= h))):>8}" for n in LONG_BOOKS))
    print()
    print("share that crossed the threshold at least once within the first N live trades")
    print(header)
    for name, first, _ in results:
        print(f"{name:<22}" + "".join(f"{_pct(float(np.mean(first <= n))):>8}" for n in LONG_BOOKS))
    print()
    print("median trade at which the threshold was first crossed, among those that crossed")
    for name, first, _ in results:
        seen = first[np.isfinite(first)]
        print(f"  {name:<22}{np.median(seen):>6.0f}" if len(seen) else f"  {name:<22}{'-':>6}")
    print()


# --------------------------------------------------------------------------
# 3. Continuous monitoring: the false-alarm rate of looking every run
# --------------------------------------------------------------------------


def fixed_test_fires(
    live: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    n_bt: int,
    *,
    shortfall: float = 0.5,
    z_min: float = 2.0,
    min_trades: int = 30,
) -> np.ndarray:
    """A vectorised copy of the fixed PnL test at every trade count: shape (reps, trades).

    Fires when the shortfall is at least `shortfall` of the promised mean and the one-sided
    Welch z is at least `z_min`, exactly as `_pnl_check`: warn is (0.5, 2), alert (1.0, 3).
    Checked against `health_report` in tests/monitoring/test_power.py.
    """
    reps, n = live.shape
    counts = np.arange(1, n + 1)
    csum = np.cumsum(live, axis=1)
    csq = np.cumsum(live**2, axis=1)
    mean = csum / counts
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (csq - counts * mean**2) / (counts - 1)
    se = np.sqrt(var / counts + (sd**2 / n_bt)[:, None])
    z = (mu[:, None] - mean) / se
    missing = (mu[:, None] - mean) / mu[:, None]
    fire = (z >= z_min - 1e-9) & (missing >= shortfall - 1e-9)
    fire[:, : min_trades - 1] = False
    return fire


def table_nearcertain(reps: int, allowance: float, backtest_se: float, h: float) -> None:
    """Cells whose backtest (almost) never lost: a tiny spread, and what the cap does there."""
    n_bt, p_promised = 100, 0.995
    print(f"Near-certain cells ({reps} simulated champions per row)")
    print(
        f"tokens priced 0.96-0.98, a {n_bt}-trade backtest that promised a {1 - p_promised:.1%} chance"
    )
    print("of losing the stake. Most backtests saw no loss at all, so their spread is tiny and")
    print("one loss of the stake is hundreds of standard deviations. Share of champions whose")
    print("alert is on at the run after N live trades (the sequential test with and without the")
    print("cap on what one trade may add, the fixed test, and the check as shipped)")
    print()
    print(
        f"{'live book':<28}{'N':>5}{'fixed test':>13}{'sequential, no cap':>21}"
        f"{'sequential, capped':>21}{'fixed or capped':>18}"
    )
    worlds = {
        "healthy (loses 0.5%)": p_promised,
        "edge gone (loses 3%)": 0.97,
        "collapsed (loses 10%)": 0.90,
    }
    for label, p_live in worlds.items():
        rng = np.random.default_rng(900 + int(p_live * 1000))
        q = rng.uniform(0.96, 0.98, (reps, n_bt))
        backtest = np.where(rng.random((reps, n_bt)) < p_promised, 1.0 / q - 1.0, -1.0) - COST
        mu, sd = backtest.mean(1), backtest.std(1, ddof=1)
        judged = mu > 0
        mu, sd = mu[judged], sd[judged]
        k = allowance * mu / sd + backtest_se / np.sqrt(n_bt)
        n_live = 400
        q = rng.uniform(0.96, 0.98, (len(mu), n_live))
        live = np.where(rng.random((len(mu), n_live)) < p_live, 1.0 / q - 1.0, -1.0) - COST
        points = (60, 400)
        _, capped = cusum_scan(live, mu, sd, k, h, points)
        _, loose = cusum_scan(live, mu, sd, k, h, points, cap=np.inf)
        fixed = fixed_test_fires(live, mu, sd, n_bt)
        for n in points:
            fixed_n = fixed[:, n - 1]
            capped_n = capped[n] >= h
            print(
                f"{label:<28}{n:>5}{_pct(fixed_n.mean()):>13}{_pct((loose[n] >= h).mean()):>21}"
                f"{_pct(capped_n.mean()):>21}{_pct((fixed_n | capped_n).mean()):>18}"
            )
    print()


def table_continuous(reps: int, allowance: float, backtest_se: float, h: float) -> None:
    print(
        f"False alarms when the check is run after every settled trade ({reps} simulated champions)"
    )
    print("nothing has changed. For each test: the share of champions that raised an alarm at")
    print("some run within the first N trades (events), and the share whose alarm is on at the")
    print("run after trade N (what a cron job sees at that moment)")
    rng = np.random.default_rng(7)
    n_max = max(LONG_BOOKS)
    q = rng.uniform(0.30, 0.60, (reps, 400))
    backtest = np.where(rng.random((reps, 400)) < q + 0.06, 1.0 / q - 1.0, -1.0) - COST
    mu, sd = backtest.mean(1), backtest.std(1, ddof=1)
    q = rng.uniform(0.30, 0.60, (reps, n_max))
    live = np.where(rng.random((reps, n_max)) < q + 0.06, 1.0 / q - 1.0, -1.0) - COST
    first, at = cusum_scan(live, mu, sd, allowance * mu / sd + backtest_se / np.sqrt(400), h)
    sequential_on = {n: at[n] >= h for n in LONG_BOOKS}
    levels = {
        "warn or alert (exit 10 or 20)": fixed_test_fires(live, mu, sd, 400),
        "alert only (exit 20)": fixed_test_fires(live, mu, sd, 400, shortfall=1.0, z_min=3.0),
    }
    for title, fires_at in levels.items():
        print()
        print(title)
        print(
            f"{'live trades':<13}{'fixed: events':>15}{'at N':>8}{'sequential: events':>21}{'at N':>8}"
            f"{'either: events':>17}"
        )
        ever = np.maximum.accumulate(fires_at, axis=1)
        for n in LONG_BOOKS:
            fixed_ev, fixed_now = ever[:, n - 1], fires_at[:, n - 1]
            seq_ev = first <= n
            print(
                f"{n:<13}{_pct(float(fixed_ev.mean())):>15}{_pct(float(fixed_now.mean())):>8}"
                f"{_pct(float(seq_ev.mean())):>21}{_pct(float(sequential_on[n].mean())):>8}"
                f"{_pct(float((fixed_ev | seq_ev).mean())):>17}"
            )
    print()


# --------------------------------------------------------------------------
# 4. Is calibration the earlier signal?
# --------------------------------------------------------------------------


def table_calibration(sims: int, sizes: tuple[int, ...] = (50, 100, 200, 400)) -> None:
    print(
        f"PnL check against calibration check on the same trades ({sims} simulated champions per cell)"
    )
    print("share of runs that warn or alert, at a live edge 6 points below the forecast (vanished)")
    print("and 3 points below it (halved); a mean forecast error test is shown for comparison")
    print()
    for label, (lo, hi) in {
        "prices 0.30-0.60": (0.30, 0.60),
        "cheap tokens 0.05-0.25": (0.05, 0.25),
    }.items():
        print(label)
        print(f"{'':<22}" + "".join(f"{'n=' + str(n):>8}" for n in sizes))
        for edge_name in ("as promised", "halved", "vanished"):
            shift = EDGES[edge_name]
            rng = np.random.default_rng(2000 + int(round(shift * 1000)) + int(lo * 100))
            hits = {
                "pnl": [0] * len(sizes),
                "calibration": [0] * len(sizes),
                "bias": [0] * len(sizes),
            }
            for _ in range(sims):
                backtest = draw(rng, 400, 0.0, lo, hi)
                champ = champion({"s": backtest})
                live = draw(rng, max(sizes), shift, lo, hi)
                rows = trade_rows(live)
                for j, n in enumerate(sizes):
                    rep = report(champ, rows[:n], sequential=False, checks=("calibration", "pnl"))
                    status = {c.name: c.status for c in rep.checks}
                    hits["pnl"][j] += fires(status["pnl"])
                    hits["calibration"][j] += fires(status["calibration"])
                    bias_bt = backtest.p - backtest.y
                    bias_live = live.p[:n] - live.y[:n]
                    se = np.sqrt(bias_live.var(ddof=1) / n + bias_bt.var(ddof=1) / len(bias_bt))
                    z = (bias_live.mean() - bias_bt.mean()) / se
                    hits["bias"][j] += bool(z >= 2.0 and bias_live.mean() - bias_bt.mean() >= 0.04)
            for key, name in (("pnl", "PnL"), ("calibration", "Brier"), ("bias", "mean error")):
                row = "".join(f"{_pct(h / sims):>8}" for h in hits[key])
                print(f"{edge_name + ', ' + name:<22}{row}")
        print()


# --------------------------------------------------------------------------
# 5. Segments: more of them must not mean more false alarms
# --------------------------------------------------------------------------


def table_segments(sims: int, counts: tuple[int, ...] = (2, 5, 10, 20), n_live: int = 60) -> None:
    print(
        f"False alarms from the per-segment checks when nothing has changed ({sims} simulated champions per row)"
    )
    print(
        f"each segment has {n_live} live trades; share of champions with at least one segment warned or alerted"
    )
    print()
    print(f"{'segments':<10}{'each judged alone':>20}{'with Holm':>12}")
    for count in counts:
        rng = np.random.default_rng(300 + count)
        names = [f"S{i:02d}" for i in range(count)]
        alone = holm = 0
        for _ in range(sims):
            champ = champion({name: draw(rng, 400, 0.0) for name in names})
            rows = [
                row
                for i, name in enumerate(names)
                for row in trade_rows(draw(rng, n_live, 0.0), name, i * n_live)
            ]
            check = report(champ, rows, sequential=False).checks[0]
            segments = check.metrics.get("segments", {})
            judged = [v for v in segments.values() if v["status"] != "insufficient"]
            holm += any(fires(v["status"]) for v in judged)
            # What the same tests would say with no correction: z and shortfall as reported.
            alone += any(
                v["z"] is not None and v["z"] >= 2.0 and v["shortfall_rel"] >= 0.5 for v in judged
            )
        print(f"{count:<10}{_pct(alone / sims):>20}{_pct(holm / sims):>12}")
    print()


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 6. All the parts together
# --------------------------------------------------------------------------


def table_whole(sims: int, counts: tuple[int, ...] = (1, 3, 5, 10), n_live: int = 100) -> None:
    print(
        f"False alarms from the whole PnL check, every part together ({sims} simulated champions per row)"
    )
    print(f"nothing has changed; each segment has {n_live} live trades, judged once at the end")
    print("(book, both sides, every segment, the sequential test)")
    print()
    print(
        f"{'segments':<10}{'warn or alert':>15}{'alert only':>13}{'alerts from the sequential test':>34}"
    )
    for count in counts:
        rng = np.random.default_rng(500 + count)
        names = [f"S{i:02d}" for i in range(count)]
        warn = alert = sequential = 0
        for _ in range(sims):
            champ = champion({name: draw(rng, 400, 0.0) for name in names})
            rows = [
                row
                for i, name in enumerate(names)
                for row in trade_rows(draw(rng, n_live, 0.0), name, i * n_live)
            ]
            check = report(champ, rows).checks[0]
            warn += fires(check.status)
            alert += check.status == "alert"
            sequential += bool(check.metrics.get("sequential", {}).get("signalled"))
        print(
            f"{count:<10}{_pct(warn / sims):>15}{_pct(alert / sims):>13}{_pct(sequential / sims):>34}"
        )
    print()


TABLES: dict[str, Callable[[argparse.Namespace], None]] = {
    "firing": lambda a: table_firing(a.sims),
    "cusum": lambda a: table_cusum(a.reps, a.allowance, a.backtest_se, a.threshold),
    "backtest-size": lambda a: table_backtest_size(
        a.reps // 2, a.allowance, a.backtest_se, a.threshold
    ),
    "near-certain": lambda a: table_nearcertain(
        a.reps // 2, a.allowance, a.backtest_se, a.threshold
    ),
    "continuous": lambda a: table_continuous(
        a.reps_continuous, a.allowance, a.backtest_se, a.threshold
    ),
    "calibration": lambda a: table_calibration(a.sims),
    "segments": lambda a: table_segments(a.sims),
    "whole": lambda a: table_whole(a.sims),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "tables", nargs="*", choices=[*TABLES, []], help="which tables (default: all)"
    )
    parser.add_argument("--quick", action="store_true", help="a fast, noisier run")
    parser.add_argument(
        "--sims", type=int, default=None, help="champions per cell for health_report tables"
    )
    parser.add_argument(
        "--reps", type=int, default=None, help="champions for the sequential-test table"
    )
    args = parser.parse_args(argv)
    defaults = PnLThresholds()
    args.allowance = defaults.cusum_allowance
    args.backtest_se = defaults.cusum_backtest_se
    args.threshold = defaults.cusum_threshold
    args.sims = args.sims or (80 if args.quick else 400)
    args.reps = args.reps or (1500 if args.quick else 10000)
    args.reps_continuous = 600 if args.quick else 4000
    print(
        f"settings: cusum_allowance={args.allowance:g}, cusum_backtest_se={args.backtest_se:g}, "
        f"cusum_threshold={args.threshold:g}, z_warn={defaults.z_warn:g}, "
        f"shortfall_warn={defaults.shortfall_warn:g}, min_trades={defaults.min_trades}, "
        f"cusum_increment_cap={defaults.cusum_increment_cap:g}"
    )
    print()
    for name in args.tables or list(TABLES):
        TABLES[name](args)


if __name__ == "__main__":
    main(sys.argv[1:])
