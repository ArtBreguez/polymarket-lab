"""Performance budget of pmlab's hot paths: the workloads, the timing harness and the budgets.

Single source of truth for ``docs/performance.md``, for the CI guard (``pytest -m perf``) and for
anyone who wants to know how fast pmlab is on *their* machine. Only needs an installed ``pmlab``
(plus its dependencies), nothing from the repository checkout::

    python examples/perf_budget.py           # measure every workload, print the table
    python examples/perf_budget.py --check   # same, exit 1 if a budget or a scaling limit is broken

The workloads are fixed, seeded and built in-process (no network, no files). Time is reported in
**calibration units**: seconds divided by the time of a small fixed pandas + Python loop
(:func:`calibration_seconds`: the hot paths' own pandas work in miniature) measured in the same run, so a slow CI runner and a fast laptop read
alike. A budget is a number of units, generous enough to absorb runner noise and tight enough to
catch a ~3x slowdown. See ``docs/performance.md`` for how to change one on purpose.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.policy import TradePolicy
from pmlab.data import build_panel

SEED = 20260508
N_FEATURES = 6
SCALING_RATIO_LIMIT = 3.0  # time at 2x the rows must stay under 3x the time at 1x (linear is ~2x)


# ── Synthetic workloads ───────────────────────────────────────────────────────────────────────


def make_panel(
    n_dates: int, n_markets: int, n_bins: int, seed: int = SEED, n_features: int = N_FEATURES
) -> pd.DataFrame:
    """A ``rolling_origin_eval`` panel: ``n_markets`` markets x ``n_dates`` dates x ``n_bins`` bins.

    Stacked market by market (as plugins emit it). Per market-date the bin prices are a
    distribution, exactly one bin wins, and ``feature_0`` carries a little signal about it.
    """
    rng = np.random.default_rng(seed)
    winner = rng.integers(0, n_bins, size=n_markets)  # a market resolves once
    won = np.zeros((n_markets, n_dates, n_bins), dtype=bool)
    won[np.arange(n_markets), :, winner] = True
    logits = rng.normal(size=won.shape) + 0.8 * won
    price = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)
    feats = rng.normal(size=(*won.shape, n_features))
    feats[..., 0] += 0.5 * won

    m, d, b = np.meshgrid(
        np.arange(n_markets), np.arange(n_dates), np.arange(n_bins), indexing="ij"
    )
    dates = pd.date_range("2024-01-01", periods=n_dates).strftime("%Y-%m-%d").to_numpy()
    bins = np.array([f"bin_{i}" for i in range(n_bins)])
    markets = np.array([f"m{i:05d}" for i in range(n_markets)])
    panel = pd.DataFrame(
        {
            "market_id": markets[m.ravel()],
            "decision_date": dates[d.ravel()],
            "outcome_label": bins[b.ravel()],
            "winning_label": bins[winner[m.ravel()]],
            "market_price": price.ravel(),
        }
    )
    for k in range(n_features):
        panel[f"feature_{k}"] = feats[..., k].ravel()
    return panel


def make_rows(
    n_markets: int, n_dates: int, n_bins: int, seed: int = SEED
) -> list[list[dict[str, Any]]]:
    """What ``MarketPlugin.build_training_rows`` yields, one list per market, timing-stamped."""
    panel = make_panel(n_dates, n_markets, n_bins, seed)
    day = pd.to_datetime(panel["decision_date"])
    panel["decision_ts"] = (day + pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    panel["price_ts"] = (day + pd.Timedelta(hours=11)).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    panel["market_close_ts"] = "2100-01-01T00:00:00+00:00"
    records = panel.to_dict("records")
    per_market = n_dates * n_bins
    return [records[i * per_market : (i + 1) * per_market] for i in range(n_markets)]


class LinearScoreModel:
    """A cheap, deterministic, single-threaded stand-in for a forecaster.

    ``fit`` is one pass over the training rows, ``predict_proba`` one matrix product: the cost of
    the model is small and linear, so the timing tracks pmlab's own code (masks, group-bys, the
    per-trade bookkeeping), not LightGBM's thread scheduling on a shared CI runner.
    """

    def __init__(self) -> None:
        self._mu = np.zeros(0)
        self._w = np.zeros(0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        x = X.to_numpy(dtype=float)
        yy = y.to_numpy(dtype=bool)
        self._mu = x.mean(axis=0)
        pos = x[yy].mean(axis=0) if yy.any() else self._mu
        neg = x[~yy].mean(axis=0) if (~yy).any() else self._mu
        self._w = pos - neg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = (X.to_numpy(dtype=float) - self._mu) @ self._w - 1.6
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


@dataclass(frozen=True)
class Workload:
    """One timed hot path. ``prepare(scale)`` builds its input at ``scale`` x the rows."""

    name: str
    describes: str
    prepare: Callable[[int], Any]
    run: Callable[[Any], Any]
    reference_units: float  # measured on the reference machine (docs/performance.md)
    budget_units: float  # what CI asserts: reference x headroom


# The sizes are the "realistic" ones: a season of daily decisions over a few hundred markets.
RO_DATES, RO_BINS = 120, 6
RO_MARKETS = 100  # 72,000 rows: the historical backtest
RO_LIVE_MARKETS = 40  # 28,800 rows: the live policy walks every bin row in Python
RO_STRIDE, RO_MIN_TRAIN = 4, 2000  # 29 refits (27 on the smaller live panel)
BP_MARKETS, BP_DATES, BP_BINS = 2000, 10, 6  # 120,000 rows


def _prepare_ro(scale: int) -> pd.DataFrame:
    return make_panel(RO_DATES, RO_MARKETS * scale, RO_BINS)


def _prepare_ro_live(scale: int) -> pd.DataFrame:
    return make_panel(RO_DATES, RO_LIVE_MARKETS * scale, RO_BINS)


def _run_ro(panel: pd.DataFrame) -> Any:
    return rolling_origin_eval(
        panel, LinearScoreModel(), min_train_rows=RO_MIN_TRAIN, stride=RO_STRIDE
    )


def _run_ro_live(panel: pd.DataFrame) -> Any:
    return rolling_origin_eval(
        panel,
        LinearScoreModel(),
        min_train_rows=RO_MIN_TRAIN,
        stride=RO_STRIDE,
        policy=TradePolicy.live(min_edge=0.02),
    )


def _prepare_bp(scale: int) -> list[list[dict[str, Any]]]:
    return make_rows(BP_MARKETS * scale, BP_DATES, BP_BINS)


def _run_bp(rows: list[list[dict[str, Any]]]) -> Any:
    return build_panel(rows)


WORKLOADS: tuple[Workload, ...] = (
    Workload(
        "rolling_origin_top_prob",
        "rolling_origin_eval, historical top-bin rule, 72k rows, 29 refits",
        _prepare_ro,
        _run_ro,
        reference_units=6.4,
        budget_units=15.0,
    ),
    Workload(
        "rolling_origin_live_policy",
        "rolling_origin_eval, best-edge live policy, 28.8k rows, 27 refits",
        _prepare_ro_live,
        _run_ro_live,
        reference_units=10.8,
        budget_units=25.0,
    ),
    Workload(
        "build_panel",
        "build_panel on 2,000 timing-stamped markets, 120k rows",
        _prepare_bp,
        _run_bp,
        reference_units=4.3,
        budget_units=10.0,
    ),
)
SCALING = tuple(w.name for w in WORKLOADS)  # all three are linear in the number of markets


# ── Timing harness ────────────────────────────────────────────────────────────────────────────


def best_of(
    fn: Callable[[], object],
    repeats: int = 5,
    warmup: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Minimum wall time of ``repeats`` runs after ``warmup`` untimed ones (as ``timeit`` advises).

    The minimum is the least noisy estimate of the cost of the code itself: a shared runner can
    only ever add time, never remove it. The garbage collector is paused while timing.
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = clock()
            fn()
            best = min(best, clock() - start)
    finally:
        if was_enabled:
            gc.enable()
    return best


@functools.cache
def _calibration_inputs() -> pd.DataFrame:
    return make_panel(n_dates=40, n_markets=60, n_bins=6)  # 14,400 rows, object + float columns


def _calibration_unit() -> None:
    """The hot paths' own pandas/Python work in miniature, without a line of pmlab.

    String-key masks and boolean takes on object columns, label comparison, a group-by on two
    string keys, ``iterrows`` over the picked rows, dict records -> DataFrame, and a bare
    interpreter loop: the mix that decides how fast a machine runs ``rolling_origin_eval`` and
    ``build_panel``. A pure-Python loop alone does not: a CI runner with a fast interpreter and
    slow memory reads very differently from a laptop with the opposite balance.
    """
    frame = _calibration_inputs()
    features = [c for c in frame.columns if c.startswith("feature_")]
    for day in ("2024-01-10", "2024-01-20", "2024-01-30"):
        train = frame[frame["decision_date"] < day]
        np.asarray((train["outcome_label"] == train["winning_label"]).astype(int))
        train[features].to_numpy(dtype=float).mean(axis=0)
        ev = frame[frame["decision_date"] == day].copy()
        ev["_p"] = ev["market_price"] * 0.5
        best = ev.groupby(["market_id", "decision_date"])["_p"].idxmax()
        for _, row in ev.loc[best].iterrows():
            float(row["market_price"])
    pd.DataFrame(frame.head(6_000).to_dict("records"))
    acc = 0.0
    seen: dict[int, float] = {}
    for i in range(40_000):
        acc += (i % 7) * 0.5
        seen[i % 997] = acc


def calibration_seconds(repeats: int = 5) -> float:
    """Best-of-``repeats`` time of the calibration loop: the 'speed of this machine, right now'."""
    return best_of(_calibration_unit, repeats=repeats, warmup=1)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    units: float
    budget_units: float
    rounds: int
    seconds: float = 0.0
    calibration: float = 0.0

    def describe(self, name: str) -> str:
        state = "ok" if self.ok else "OVER BUDGET"
        return (
            f"{name}: {self.units:.1f} units vs budget {self.budget_units:.1f} "
            f"({state}, best of {self.rounds} round(s))"
        )


def check_budget(
    measure: Callable[[], tuple[float, float]], budget_units: float, max_rounds: int = 3
) -> Verdict:
    """Fail only if *every* round is over budget.

    ``measure()`` returns ``(workload_seconds, calibration_seconds)`` taken back to back. A noisy
    neighbour can spoil one round, never all of them; a real regression is slow every time.
    """
    best, best_seconds, best_calibration, rounds = float("inf"), 0.0, 0.0, 0
    for rounds in range(1, max_rounds + 1):  # noqa: B007 - `rounds` is read after the loop
        seconds, calibration = measure()
        if seconds / calibration < best:
            best, best_seconds, best_calibration = seconds / calibration, seconds, calibration
        if best <= budget_units:
            break
    return Verdict(best <= budget_units, best, budget_units, rounds, best_seconds, best_calibration)


def measure_workload(w: Workload, scale: int = 1, payload: Any = None) -> tuple[float, float]:
    payload = w.prepare(scale) if payload is None else payload
    calibration = calibration_seconds()
    return best_of(lambda: w.run(payload)), calibration


def scaling_ratio(w: Workload, max_rounds: int = 3) -> float:
    """Best (lowest) of up to ``max_rounds`` ratios time(2x rows) / time(1x rows)."""
    one, two = w.prepare(1), w.prepare(2)
    best = float("inf")
    for _ in range(max_rounds):
        t1 = best_of(lambda: w.run(one))
        t2 = best_of(lambda: w.run(two))
        best = min(best, t2 / t1)
        if best < SCALING_RATIO_LIMIT:
            break
    return best


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="exit 1 if a budget/scaling is broken")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"python": sys.version.split()[0], "workloads": {}}
    failures: list[str] = []
    for w in WORKLOADS:
        payload = w.prepare(1)
        verdict = check_budget(lambda w=w, p=payload: measure_workload(w, 1, p), w.budget_units)
        ratio = scaling_ratio(w)
        report["workloads"][w.name] = {
            "seconds": round(verdict.seconds, 4),
            "calibration_seconds": round(verdict.calibration, 4),
            "units": round(verdict.units, 2),
            "reference_units": w.reference_units,
            "budget_units": w.budget_units,
            "scaling_ratio_2x": round(ratio, 2),
            "ok": verdict.ok and ratio < SCALING_RATIO_LIMIT,
        }
        if not verdict.ok:
            failures.append(verdict.describe(w.name))
        if ratio >= SCALING_RATIO_LIMIT:
            failures.append(f"{w.name}: 2x rows cost {ratio:.2f}x (limit {SCALING_RATIO_LIMIT:g}x)")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"python {report['python']}; time in calibration units (seconds / calibration loop)")
        print(
            f"{'workload':30} {'seconds':>8} {'units':>7} {'ref':>6} {'budget':>7} {'2x rows':>8}"
        )
        for name, r in report["workloads"].items():
            print(
                f"{name:30} {r['seconds']:8.3f} {r['units']:7.1f} {r['reference_units']:6g} "
                f"{r['budget_units']:7g} {r['scaling_ratio_2x']:7.2f}x"
            )
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if (failures and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
