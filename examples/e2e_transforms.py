"""End-to-end, as a user: new feature transforms -> LGBMForecaster -> rolling_origin_eval.

Standalone (only needs an installed ``pmlab``; nothing is imported from the repo checkout):

    python examples/e2e_transforms.py

What it does, on synthetic but realistic data (6 cities x 120 days, 4 temperature bins per market,
NaNs in the features, a binary target):

1. ``add_missing_indicators`` and ``add_interactions`` on the raw features;
2. ``TargetEncoder`` on a high-cardinality category: out-of-fold (time-ordered) on the training
   period, the fitted mapping on a later "live" period that contains an unseen category and a
   NaN category;
3. ``LGBMForecaster`` + ``rolling_origin_eval`` on the panel built with the new features, and the
   same model scored on the live rows;
4. asserts the user-visible guarantees (inputs unchanged, no inf, no NaN where there must not
   be one, live rows never use the OOF path, unseen category -> prior, finite metrics).

Exits non-zero (AssertionError) if any guarantee is broken, and prints a short summary.
``--panel-out FILE.parquet`` also writes the feature panel, so it can be fed to the CLI:
``pmlab backtest --plugin weather_tmax --panel FILE.parquet --stride 10 --min-train-rows 200``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab import (
    LGBMForecaster,
    TargetEncoder,
    add_interactions,
    add_missing_indicators,
    compute_metrics,
    rolling_origin_eval,
)

SEED = 7
CITIES = ["NYC", "LA", "CHI", "MIA", "SEA", "DEN"]
CITY_MEAN = {"NYC": -1.0, "LA": 3.0, "CHI": -3.5, "MIA": 4.0, "SEA": 0.5, "DEN": -2.0, "ANC": -5.0}
BIN_CENTERS = np.array([-6.0, -2.0, 2.0, 6.0])
N_DAYS = 120
CUTOFF_DAY = 84  # days [0, 84) are the training period, [84, 120) the "live" period
SMOOTHING = 20.0
N_FOLDS = 6


def make_panel(seed: int = SEED) -> pd.DataFrame:
    """One row per (market, bin): the layout ``rolling_origin_eval`` expects."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    rows: list[dict[str, object]] = []
    for day in range(N_DAYS):
        date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        for city in CITIES:
            live_period = day >= CUTOFF_DAY
            true_city = city
            if live_period and city == "DEN" and day % 9 == 0:
                true_city = "ANC"  # a city never seen while training
            actual = CITY_MEAN[true_city] + rng.normal(0.0, 4.0)
            win = int(np.argmin(np.abs(BIN_CENTERS - actual)))
            forecast: float = actual + rng.normal(0.0, 2.0)
            if rng.random() < 0.10:
                forecast = float("nan")
            elif rng.random() < 0.02:
                forecast = 0.0  # exact zero: the humidity / forecast ratio must be NaN, not inf
            humidity = 55.0 + 4.0 * rng.normal()
            if rng.random() < (0.40 if city == "DEN" else 0.05):  # informative missingness
                humidity = float("nan")
            center = CITY_MEAN[true_city] if np.isnan(forecast) else forecast
            logits = -((BIN_CENTERS - center) ** 2) / (2 * 3.0**2)
            probs = np.exp(logits) / np.exp(logits).sum()
            for j in range(len(BIN_CENTERS)):
                # the city is sometimes missing on a row, more often on the extreme bins, so the
                # NaN category carries signal of its own (its win rate differs from the prior)
                missing_city = rng.random() < (0.10 if j in (0, len(BIN_CENTERS) - 1) else 0.01)
                city_value = None if missing_city else true_city
                price = float(np.clip(probs[j] + rng.normal(0.0, 0.04), 0.02, 0.98))
                rows.append(
                    {
                        "market_id": f"{city}-{date}",
                        "decision_date": date,
                        "city": city_value,
                        "city_bin": None if city_value is None else f"{city_value}|b{j}",
                        "outcome_label": f"b{j}",
                        "winning_label": f"b{win}",
                        "market_price": price,
                        "feature_bin_center": float(BIN_CENTERS[j]),
                        "feature_temp_forecast": forecast,
                        "feature_humidity": humidity,
                    }
                )
    panel = pd.DataFrame(rows)
    panel["label"] = (panel["outcome_label"] == panel["winning_label"]).astype(int)
    return panel


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feature_")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel-out", type=Path, default=None, help="write the panel (parquet)")
    args = parser.parse_args(argv)

    raw = make_panel()
    snapshot = raw.copy(deep=True)
    dates = sorted(raw["decision_date"].unique())
    cutoff = dates[CUTOFF_DAY]

    # ---- 1. missing indicators + interactions (row-local, copy-on-write) --------------------
    feats = add_missing_indicators(raw, ["feature_temp_forecast", "feature_humidity"])
    feats = add_interactions(feats, [("feature_temp_forecast", "feature_bin_center")], op="sub")
    feats = add_interactions(feats, [("feature_humidity", "feature_temp_forecast")], op="div")
    pd.testing.assert_frame_equal(raw, snapshot)  # the input frame was not touched

    for col in ("feature_temp_forecast_isna", "feature_humidity_isna"):
        assert feats[col].dtype == np.dtype("int8"), col
        src = col.removesuffix("_isna")
        assert (feats[col].to_numpy() == raw[src].isna().to_numpy().astype(int)).all(), col
    ratio = feats["feature_humidity_div_feature_temp_forecast"]
    zero_den = raw["feature_temp_forecast"] == 0.0
    assert zero_den.any(), "the synthetic data should contain exact-zero forecasts"
    assert ratio[zero_den].isna().all(), "division by zero must be NaN"
    assert not np.isinf(feats.select_dtypes("number").to_numpy(dtype=float)).any()
    diff = feats["feature_temp_forecast_minus_feature_bin_center"]
    assert diff.isna().to_numpy().tolist() == raw["feature_temp_forecast"].isna().tolist()

    train = feats[feats["decision_date"] < cutoff].reset_index(drop=True)
    live = feats[feats["decision_date"] >= cutoff].reset_index(drop=True)
    train_snapshot = train.copy(deep=True)

    # ---- 2. target encoding: OOF on the training period, fitted mapping on live rows ---------
    enc = TargetEncoder(
        col="city_bin",
        target="label",
        time_col="decision_date",  # time-ordered folds: a row sees strictly earlier days only
        n_folds=N_FOLDS,
        smoothing=SMOOTHING,
        seed=SEED,
        output_col="feature_city_bin_te",  # the panel convention: feature_* are model inputs
    )
    train_te = enc.fit_transform(train)
    pd.testing.assert_frame_equal(train, train_snapshot)  # copy-on-write
    te_col = "feature_city_bin_te"
    folds = enc.assign_folds(train)
    assert np.isnan(train_te.loc[folds == 0, te_col]).all(), "first fold has no history -> NaN"
    assert train_te.loc[folds > 0, te_col].notna().all()
    assert np.isfinite(train_te.loc[folds > 0, te_col]).all()

    # no lookahead: rewriting the targets of the last 30% of the training days leaves the
    # encoding of every earlier row bit-for-bit identical
    late_days = sorted(train["decision_date"].unique())[int(0.7 * CUTOFF_DAY) :]
    tampered = train.copy()
    tampered.loc[tampered["decision_date"].isin(late_days), "label"] = 1 - tampered["label"]
    early = ~train["decision_date"].isin(late_days)
    tampered_te = enc.fit_transform(tampered)
    np.testing.assert_array_equal(
        tampered_te.loc[early, te_col].to_numpy(), train_te.loc[early, te_col].to_numpy()
    )
    # no self-leak: flipping a row's own target never changes that row's encoding
    for i in np.random.default_rng(SEED).choice(len(train), size=20, replace=False):
        flipped = train.copy()
        flipped.loc[i, "label"] = 1 - flipped.loc[i, "label"]
        a = enc.fit_transform(flipped).loc[i, te_col]
        b = train_te.loc[i, te_col]
        assert (np.isnan(a) and np.isnan(b)) or a == b, f"row {i} sees its own target"

    # live rows: no target column at all, full-train statistics (never the OOF path)
    fitted = enc.fit(train)
    live_in = live.drop(columns=["outcome_label", "winning_label", "label"])
    live_in_snapshot = live_in.copy(deep=True)
    live_te = fitted.transform(live_in)
    pd.testing.assert_frame_equal(live_in, live_in_snapshot)
    assert te_col in live_te.columns and list(live_te.columns[:-1]) == list(live_in.columns)

    known = live_te["city_bin"].isin(fitted.mapping)
    expect = live_te.loc[known, "city_bin"].map(fitted.mapping).astype(float)
    np.testing.assert_array_equal(live_te.loc[known, te_col].to_numpy(), expect.to_numpy())
    unseen = live_te["city_bin"].notna() & ~known
    assert unseen.any() and live_te["city_bin"][unseen].str.startswith("ANC").all()
    assert (live_te.loc[unseen, te_col] == fitted.prior).all(), "unseen category -> the prior"
    nan_cat = live_te["city_bin"].isna()
    assert nan_cat.any(), "the synthetic live period should contain a NaN category"
    assert abs(fitted.nan_value - fitted.prior) > 1e-3, "NaN category must have its own estimate"
    assert (live_te.loc[nan_cat, te_col] == fitted.nan_value).all()
    assert not (live_te.loc[nan_cat, te_col] == fitted.prior).any()
    assert np.isfinite(live_te[te_col]).all(), "live rows never get a NaN/inf encoding"
    # fitted path = one constant per category; OOF path varies across folds for the same category
    assert live_te.groupby("city_bin")[te_col].nunique().max() == 1
    assert train_te.groupby("city_bin")[te_col].nunique().max() > 1
    # and it is exactly the full-train smoothed statistic, recomputed independently
    prior = train["label"].mean()
    stats = train.dropna(subset=["city_bin"]).groupby("city_bin")["label"].agg(["sum", "count"])
    smoothed = (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)
    assert fitted.prior == prior
    for cat, value in smoothed.items():
        assert fitted.mapping[cat] == value
    # the fitted encoder ships with the model: pickle round trip gives the same live features
    clone = pickle.loads(pickle.dumps(fitted))
    pd.testing.assert_frame_equal(clone.transform(live_in), live_te)

    # ---- 3. model + rolling-origin backtest on the panel built with the new features --------
    panel = train_te.drop(columns=["label", "city", "city_bin"])
    cols = feature_columns(panel)
    assert panel["market_price"].notna().all()
    for col in cols:
        values = panel[col].to_numpy(dtype=float)
        assert not np.isinf(values).any(), col
    for col in ("feature_temp_forecast_isna", "feature_humidity_isna", "feature_bin_center"):
        assert panel[col].notna().all(), col

    def model() -> LGBMForecaster:
        return LGBMForecaster(n_estimators=60, learning_rate=0.05, num_leaves=15, n_jobs=1)

    result = rolling_origin_eval(panel, model(), min_train_rows=200, stride=7)
    base_cols = [
        "feature_bin_center",
        "feature_temp_forecast",
        "feature_humidity",
    ]
    base_panel = panel.drop(columns=[c for c in cols if c not in base_cols])
    baseline = rolling_origin_eval(base_panel, model(), min_train_rows=200, stride=7)
    m_new = compute_metrics(result.trades)
    m_base = compute_metrics(baseline.trades)
    for m in (m_new, m_base):
        assert m.num_trades > 0
        assert np.isfinite([m.total_pnl, m.hit_rate, m.avg_pnl_per_trade, m.avg_edge]).all()
    assert np.isfinite(result.trades["predicted_prob"]).all()
    assert result.trades["predicted_prob"].between(0.0, 1.0).all()
    assert result.trades["eval_date"].max() < cutoff, "the backtest never touched live days"

    # serve the same features to the live rows with the model fitted on the whole train panel
    final = model()
    final.fit(panel[cols], (panel["outcome_label"] == panel["winning_label"]).astype(int))
    live_X = live_te[cols]
    assert list(live_X.columns) == list(panel[cols].columns)
    proba = final.predict_proba(live_X)[:, 1]
    assert np.isfinite(proba).all() and ((proba > 0) & (proba < 1)).all()

    pd.testing.assert_frame_equal(raw, snapshot)  # still untouched after the whole workflow
    if args.panel_out is not None:
        panel.to_parquet(args.panel_out)

    # ---- 4. summary --------------------------------------------------------------------------
    print("pmlab e2e (as a user): OK")
    print(
        f"  data      {len(raw):,} rows = {raw['market_id'].nunique()} markets x "
        f"{len(BIN_CENTERS)} bins, train < {cutoff}: {len(train):,} rows, live: {len(live):,} rows"
    )
    print(
        f"  features  {len(cols)} model inputs: {', '.join(c.removeprefix('feature_') for c in cols)}"
    )
    print(
        f"  indicators/interactions  NaN forecast {int(raw['feature_temp_forecast'].isna().sum())}"
        f", exact-zero forecast {int(zero_den.sum())} -> ratio NaN, no inf"
    )
    print(
        f"  target enc  {len(fitted.mapping)} categories, prior {fitted.prior:.3f}, "
        f"NaN-category value {fitted.nan_value:.3f}; OOF first fold NaN: "
        f"{int((folds == 0).sum())} rows; live rows {len(live_te)} "
        f"(unseen->prior: {int(unseen.sum())}, NaN category: {int(nan_cat.sum())})"
    )
    print(
        f"  backtest (with new features)    trades {m_new.num_trades:>4}  "
        f"PnL {m_new.total_pnl:+8.2f}  hit {m_new.hit_rate:.3f}  avg edge {m_new.avg_edge:+.3f}"
    )
    print(
        f"  backtest (raw features only)    trades {m_base.num_trades:>4}  "
        f"PnL {m_base.total_pnl:+8.2f}  hit {m_base.hit_rate:.3f}  avg edge {m_base.avg_edge:+.3f}"
    )
    print(f"  live scoring  {len(proba)} rows, p in [{proba.min():.3f}, {proba.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
