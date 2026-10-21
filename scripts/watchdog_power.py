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
trade (continuous monitoring, tens of thousands of paths) use a vectorised copy of the
fixed z test or the production `cusum`; `tests/monitoring/test_power.py` checks that both
agree with `health_report`.
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
from pmlab.monitoring.stats import cusum

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


def cusum_paths(
    reps: int,
    n: int,
    shift: float,
    *,
    change_at: int = 0,
    seed: int,
    allowance: float,
    h: float,
    n_bt: int = 400,
) -> np.ndarray:
    """First trade at which the production `cusum` signals, per simulated champion (inf: never)."""
    rng = np.random.default_rng(seed)
    first = np.full(reps, np.inf)
    for r in range(reps):
        bt = draw(rng, n_bt, 0.0).ret
        mu, sd = float(bt.mean()), float(bt.std(ddof=1))
        live = draw(rng, n, shift, change_at=change_at).ret
        result = cusum(((mu - live) / sd).tolist(), [allowance * mu / sd] * n, threshold=h)
        if result.crossed_at is not None:
            first[r] = result.crossed_at
    return first


def table_backtest_size(reps: int, allowance: float, h: float) -> None:
    print("The sequential test when the backtest is short (the promised mean is an estimate)")
    print(f"{reps} simulated champions per row; shares within the first 400 live trades")
    print()
    print(f"{'backtest trades':<18}{'false alarms':>14}{'edge vanished':>15}")
    for n_bt in (30, 50, 100, 200, 400, 1000):
        null = cusum_paths(reps, 400, 0.0, seed=70, allowance=allowance, h=h, n_bt=n_bt)
        gone = cusum_paths(reps, 400, -0.06, seed=71, allowance=allowance, h=h, n_bt=n_bt)
        print(
            f"{n_bt:<18}{_pct(float(np.mean(null <= 400))):>14}{_pct(float(np.mean(gone <= 400))):>15}"
        )
    print()


def table_cusum(reps: int, allowance: float, h: float) -> None:
    print(
        f"The sequential test alone (allowance {allowance:g} x the promised edge, threshold {h:g},"
        f" {reps} simulated champions per row)"
    )
    print("share of champions it has signalled on within the first N live trades;")
    print("median = the trade at which it signalled, among those it signalled on")
    print()
    print(f"{'live edge':<24}{'N=100':>8}{'N=200':>8}{'N=400':>8}{'N=800':>8}{'median':>9}")
    scenarios: list[tuple[str, float, int]] = [
        ("as promised", 0.0, 0),
        ("halved", -0.03, 0),
        ("vanished", -0.06, 0),
        ("reversed", -0.12, 0),
        ("halved after 150", -0.03, 150),
        ("vanished after 150", -0.06, 150),
    ]
    for i, (name, shift, change_at) in enumerate(scenarios):
        first = cusum_paths(
            reps, 800, shift, change_at=change_at, seed=50 + i, allowance=allowance, h=h
        )
        cells = "".join(f"{_pct(float(np.mean(first <= n))):>8}" for n in (100, 200, 400, 800))
        seen = first[np.isfinite(first)]
        median = f"{np.median(seen):>9.0f}" if len(seen) else f"{'-':>9}"
        print(f"{name:<24}{cells}{median}")
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


def table_continuous(reps: int, allowance: float, h: float) -> None:
    print(
        f"False alarms when the check is run after every settled trade ({reps} simulated champions)"
    )
    print("share of champions that raise an alarm at some run within the first N trades,")
    print("although nothing has changed")
    rng = np.random.default_rng(7)
    n_max = 800
    seq_first = np.full(reps, np.inf)
    mus = np.empty(reps)
    sds = np.empty(reps)
    live = np.empty((reps, n_max))
    for r in range(reps):
        bt = draw(rng, 400, 0.0).ret
        mus[r], sds[r] = bt.mean(), bt.std(ddof=1)
        live[r] = draw(rng, n_max, 0.0).ret
        crossed = cusum(
            ((mus[r] - live[r]) / sds[r]).tolist(),
            [allowance * mus[r] / sds[r]] * n_max,
            threshold=h,
        ).crossed_at
        if crossed is not None:
            seq_first[r] = crossed
    levels = {
        "warn or alert (exit 10 or 20)": fixed_test_fires(live, mus, sds, 400),
        "alert only (exit 20)": fixed_test_fires(live, mus, sds, 400, shortfall=1.0, z_min=3.0),
    }
    for title, fires_at in levels.items():
        print()
        print(title)
        print(f"{'live trades':<14}{'fixed test':>12}{'sequential':>12}{'either':>10}")
        ever = np.maximum.accumulate(fires_at, axis=1)
        for n in (100, 200, 400, 800):
            fixed = ever[:, n - 1]
            seq = seq_first <= n
            print(
                f"{n:<14}{_pct(float(fixed.mean())):>12}{_pct(float(seq.mean())):>12}"
                f"{_pct(float((fixed | seq).mean())):>10}"
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


TABLES: dict[str, Callable[[argparse.Namespace], None]] = {
    "firing": lambda a: table_firing(a.sims),
    "cusum": lambda a: table_cusum(a.reps, a.allowance, a.threshold),
    "backtest-size": lambda a: table_backtest_size(a.reps // 3, a.allowance, a.threshold),
    "continuous": lambda a: table_continuous(a.reps_continuous, a.allowance, a.threshold),
    "calibration": lambda a: table_calibration(a.sims),
    "segments": lambda a: table_segments(a.sims),
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
    args.threshold = defaults.cusum_threshold
    args.sims = args.sims or (80 if args.quick else 400)
    args.reps = args.reps or (1500 if args.quick else 10000)
    args.reps_continuous = 600 if args.quick else 4000
    print(
        f"settings: cusum_allowance={args.allowance:g}, cusum_threshold={args.threshold:g}, "
        f"z_warn={defaults.z_warn:g}, shortfall_warn={defaults.shortfall_warn:g}, min_trades={defaults.min_trades}"
    )
    print()
    for name in args.tables or list(TABLES):
        TABLES[name](args)


if __name__ == "__main__":
    main(sys.argv[1:])
