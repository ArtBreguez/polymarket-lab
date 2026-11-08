"""Compute the numbers the documentation home page shows, from the bundled synthetic sample.

    uv run python scripts/make_site_data.py            # rewrite docs/assets/data/sample-run.json
    uv run python scripts/make_site_data.py --check    # fail if the committed file is out of date

Offline and deterministic, public API only (the same steps as ``notebooks/tutorial.ipynb``): the
panel, a tuned LightGBM model, a walk-forward backtest, the per-city gate, the refusal of
``ChampionManifest.publish`` and ``pmlab reproduce``. Nothing on the home page is typed by hand:
the rolling-origin timeline, the gate readout and the signal ticker are drawn from this file, and
``--check`` (run by the e2e job against the built wheel) keeps it honest as the library changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmlab import (
    BacktestConfig,
    ChampionManifest,
    CostModel,
    EdgeSignal,
    HoldoutGateResult,
    LGBMForecaster,
    TradePolicy,
    TunedForecaster,
    brier_decomposition,
    build_panel,
    capture_backtest,
    reproduce,
    resolve_seeds,
    run_backtest,
)
from pmlab.datasets import load_sample
from pmlab.reproducibility import build_forecaster, seeded

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "data" / "sample-run.json"
API_SNAPSHOT = ROOT / "tests" / "api" / "public_api.json"
SCHEMA = 1
TOLERANCE = 1e-3  # --check: LightGBM may differ in the last digits between platforms


def _bin_centre(bin_: Any) -> float:
    if bin_.lower is not None and bin_.upper is not None:
        return float((bin_.lower + bin_.upper) / 2)
    return float(bin_.upper - 1.0 if bin_.upper is not None else bin_.lower + 1.0)


def compute() -> dict[str, Any]:
    """Run the tutorial's pipeline on the bundled sample and return the home-page data."""
    warnings.filterwarnings("ignore", message="IProgress not found")
    sample = load_sample()
    resolver = sample.plugin()
    discovered = [
        (day, spec) for day in sample.days for spec in sample.plugin(day).discover_markets()
    ]
    specs = [spec for _, spec in discovered]
    by_market = {spec.market_id: spec for _, spec in discovered}
    rows = [
        resolver.build_training_rows(
            spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00", exclusions={}
        )
        for day, spec in discovered
    ]
    panel = build_panel(rows, check_leakage=True)

    meta = {(s.market_id, b.label): (s.metadata["city"], b) for s in specs for b in s.outcome_bins}
    pairs = list(zip(panel["market_id"], panel["outcome_label"], strict=True))
    centre = pd.Series([_bin_centre(meta[k][1]) for k in pairs], index=panel.index)
    panel["segment"] = [meta[k][0] for k in pairs]
    panel["feature_bin_dist"] = centre - panel["feature_fcst_tmax"]
    panel["feature_bin_z"] = panel["feature_bin_dist"] / panel["feature_ens_spread"]
    panel["feature_bin_dist_climo"] = centre - panel["feature_climo_tmax"]
    panel["feature_is_tail"] = panel["outcome_label"].str.contains("<=|>=").astype(float)
    panel["feature_market_price"] = panel["market_price"]
    features = [c for c in panel.columns if c.startswith("feature_")]

    # Tune on the first 60 days, backtest on the next 60, keep the last 30 as "live" days.
    dates = sorted(panel["decision_date"].unique())
    tune_end, backtest_end = dates[60], dates[120]
    tune_panel = panel[panel["decision_date"] < tune_end]
    backtest_panel = panel[panel["decision_date"] < backtest_end].reset_index(drop=True)
    live_days = dates[120:]

    def build(params: dict[str, Any]) -> LGBMForecaster:
        return LGBMForecaster(random_state=7, n_jobs=1, **params)

    def space(trial: Any) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 60, 200, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 24),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        }

    policy = TradePolicy.live(min_edge=0.05)
    costs = CostModel(taker_bps=30.0, slippage_bps=50.0)
    tuner = TunedForecaster(
        build, space, n_trials=8, metric="total_pnl", random_state=7, stride=10,
        min_train_rows=500, policy=policy, costs=costs,
    ).tune(tune_panel)  # fmt: skip
    best = tuner.best_params_
    config = BacktestConfig.resolve(
        plugin="weather_tmax", model_params=best, stride=1, min_train_rows=2700,
        costs=costs, policy=policy, tuning=tuner.tuning_config(),
    )  # fmt: skip
    seeds = resolve_seeds(7, optuna=tuner.random_state)
    outcome = run_backtest(backtest_panel, config, seeds)
    record = capture_backtest(backtest_panel, config, seeds, outcome.metrics, source="site data")
    trades = outcome.trades

    # The backtest keeps only the traded (bin, side): refit as rolling_origin_eval does, one fit
    # per fold, to get every bin's probability and so each fold's Brier score.
    bt_dates = sorted(backtest_panel["decision_date"].unique())
    model = build_forecaster(config, seeds)
    folds: list[dict[str, Any]] = []
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    with seeded(seeds):
        for step in outcome.steps:
            day = step["eval_date"]
            train = backtest_panel[backtest_panel["decision_date"] < day]
            test = backtest_panel[backtest_panel["decision_date"] == day]
            model.fit(
                train[features], (train["outcome_label"] == train["winning_label"]).astype(int)
            )
            p = model.predict_proba(test[features])[:, 1]
            y = (test["outcome_label"] == test["winning_label"]).astype(int).to_numpy(float)
            all_y.append(y)
            all_p.append(p)
            folds.append(
                {
                    "test": day,
                    "train_end": bt_dates[bt_dates.index(day) - 1],
                    "train_rows": int(step["train_rows"]),
                    "brier": round(float(np.mean((p - y) ** 2)), 4),
                }
            )
    pooled = brier_decomposition(np.concatenate(all_y), np.concatenate(all_p), n_bins=10)

    # The gate, over every city: one city loses money, so the verdict is NO_GO and publish raises.
    gate_all = HoldoutGateResult.evaluate(
        trades, required_segments=sample.cities, min_trades_per_segment=30
    )
    champion = build(best)
    won = backtest_panel["outcome_label"] == backtest_panel["winning_label"]
    champion.fit(backtest_panel[features], won.astype(int))
    with tempfile.TemporaryDirectory() as tmp:
        champion_dir = Path(tmp) / "champion"
        refusal = None
        try:
            ChampionManifest.publish(champion, gate_all, champion_dir, plugin_family="weather_tmax")
        except ValueError as err:
            refusal = str(err)
        passing = [r.segment for r in gate_all.segment_results if r.passes]
        gate_scoped = HoldoutGateResult.evaluate(
            trades, required_segments=passing, min_trades_per_segment=30
        )
        ChampionManifest.publish(
            champion, gate_scoped, champion_dir, plugin_family="weather_tmax",
            costs=costs, reproducibility=record,
        )  # fmt: skip
        loaded = ChampionManifest.load(champion_dir / "champion.json")
        repro = reproduce(loaded.reproducibility, panel=backtest_panel)
        model_live = loaded.load_model()
    allowed = set(loaded.get_allowed_segments())

    # Edge signals on the live days: the best bin of each market, as `record-trades` would pick it.
    live = panel[panel["decision_date"].isin(live_days)].copy()
    live["prob"] = model_live.predict_proba(live[features])[:, 1]
    signals: list[dict[str, Any]] = []
    for day in live_days:
        for market_id, group in live[live["decision_date"] == day].groupby("market_id"):
            spec = by_market[market_id]
            candidates = [
                EdgeSignal.from_quote(
                    market_id=market_id, city_or_segment=spec.metadata["city"],
                    target_date=spec.metadata["target_date"], horizon="previous_evening",
                    outcome_label=r.outcome_label, yes_price=r.market_price, model_prob=r.prob,
                    costs=costs, stake=1.0,
                )
                for r in group.itertuples()
            ]  # fmt: skip
            best_signal = max((c for c in candidates if c), key=lambda c: c.best_edge)
            if best_signal.best_edge >= 0.05:
                signals.append(
                    {
                        "city": best_signal.city_or_segment,
                        "target_date": best_signal.target_date,
                        "bin": best_signal.outcome_label,
                        "market_price": round(float(best_signal.gamma_price), 3),
                        "model_prob": round(float(best_signal.model_prob), 3),
                        "edge": round(float(best_signal.best_edge), 3),
                        "side": best_signal.direction,
                        "allowed": best_signal.city_or_segment in allowed,
                    }
                )
        if len(signals) >= 16:
            break

    pnl = round(float(trades["realized_pnl"].sum()), 2)
    snapshot = json.loads(API_SNAPSHOT.read_text())
    return {
        "schema": SCHEMA,
        "source": "scripts/make_site_data.py on the bundled synthetic sample (pmlab.datasets)",
        "sample": {
            "markets": len(specs),
            "days": len(sample.days),
            "cities": len(sample.cities),
            "outcome_bins": int(sum(len(s.outcome_bins) for s in specs)),
            "first_day": sample.days[0],
            "last_day": sample.days[-1],
        },
        "backtest": {
            "train_start": bt_dates[0],
            "folds": len(folds),
            "trades": int(len(trades)),
            "pnl": pnl,
            "brier": round(float(pooled.brier_score), 4),
            "brier_skill": round(float(pooled.skill_score), 4),
            "min_train_rows": 2700,
            "stride": 1,
        },
        "folds": folds,
        "gate": {
            "decision": gate_all.decision,
            "min_trades_per_segment": 30,
            "segments": [
                {
                    "segment": r.segment,
                    "trades": int(r.num_trades),
                    "pnl": round(float(r.total_pnl), 2),
                    "passes": bool(r.passes),
                    "reason": r.reason,
                }
                for r in gate_all.segment_results
            ],
            "publish_refused": refusal,
            "scoped": {"decision": gate_scoped.decision, "segments": passing},
        },
        "reproduce": {
            "status": str(repro.status),
            "data_match": bool(repro.data_match),
            "metrics_compared": len(repro.comparisons),
        },
        "signals": signals[:12],
        "api": {
            "names": len({n for entries in snapshot["modules"].values() for n in entries}),
            "modules": len(snapshot["modules"]),
            "commands": len(snapshot["cli"]["commands"]),
        },
    }


def differences(fresh: Any, committed: Any, path: str = "") -> list[str]:
    """Where two data trees differ: exact for text and counts, within TOLERANCE for floats."""
    if isinstance(fresh, dict) and isinstance(committed, dict):
        if fresh.keys() != committed.keys():
            return [f"{path}: keys {sorted(fresh)} != {sorted(committed)}"]
        return [d for k in fresh for d in differences(fresh[k], committed[k], f"{path}.{k}")]
    if isinstance(fresh, list) and isinstance(committed, list):
        if len(fresh) != len(committed):
            return [f"{path}: {len(fresh)} items != {len(committed)}"]
        return [
            d for i, (a, b) in enumerate(zip(fresh, committed, strict=True))
            for d in differences(a, b, f"{path}[{i}]")
        ]  # fmt: skip
    if isinstance(fresh, float) or isinstance(committed, float):
        if not isinstance(fresh, int | float) or not isinstance(committed, int | float):
            return [f"{path}: {fresh!r} != {committed!r}"]
        return (
            []
            if math.isclose(fresh, committed, abs_tol=TOLERANCE)
            else [f"{path}: {fresh} != {committed}"]
        )
    return [] if fresh == committed else [f"{path}: {fresh!r} != {committed!r}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    fresh = compute()
    text = json.dumps(fresh, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        committed = json.loads(args.output.read_text())
        found = differences(json.loads(text), committed)
        if found:
            print(f"{args.output} is out of date (run scripts/make_site_data.py):", file=sys.stderr)
            print("\n".join(found[:20]), file=sys.stderr)
            return 1
        print(f"{args.output} matches the library ({datetime.now(UTC):%Y-%m-%d})")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote {args.output} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
