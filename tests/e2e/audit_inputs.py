"""Boundary arguments, used from the installed wheel (audit 1.1.3): refused early, clearly.

Standalone (public names only):  python audit_inputs.py   ->   ``audit inputs: OK``
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from pmlab import (
    AsyncClobClient,
    CostModel,
    IsotonicCalibrator,
    LGBMForecaster,
    PanelSchemaError,
    PmlabSettings,
    add_lags,
    brier_decomposition,
    build_panel,
    estimate_fee,
    kelly_stake_size,
    rolling_origin_eval,
)


def refused(exc_type, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"{fn.__name__}{args}{kwargs} was accepted")


# calibration (#58)
y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
p = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
brier_decomposition(y, p)
refused(ValueError, brier_decomposition, y * 2, p)
refused(ValueError, brier_decomposition, y, p * 2)
refused(ValueError, brier_decomposition, y, p, n_bins=0)
refused(ValueError, IsotonicCalibrator().fit, y, p)  # the swapped call

# costs, fees, sizing (#59)
refused(ValueError, CostModel, taker_bps=float("nan"))
refused(ValueError, estimate_fee, -5, 30)
refused(ValueError, kelly_stake_size, 0.6, 0.5, -100)

# the backtest's own arguments, before the first fold (#59)
rows = [
    {"market_id": f"m{d}", "decision_date": f"2026-01-{d + 1:02d}", "outcome_label": lab,
     "winning_label": "a", "market_price": 0.4 if lab == "a" else 0.6, "feature_x": float(d)}
    for d in range(28) for lab in ("a", "b")
]  # fmt: skip
panel = pd.DataFrame(rows)
model = LGBMForecaster(n_estimators=3, min_child_samples=2, n_jobs=1)
for bad in (
    dict(stride=0),
    dict(stride=-1),
    dict(flat_stake=float("nan")),
    dict(min_train_rows=-1),
):
    refused(ValueError, rolling_origin_eval, panel, model, **bad)
refused(TypeError, rolling_origin_eval, panel, model, stride=None)

# the panel (#60) and the features (#62)
refused(
    PanelSchemaError, build_panel, [{**rows[0], "decision_date": "2025-13-45", "feature_a": 1.0}]
)
refused(PanelSchemaError, build_panel, [{**rows[0], "market_price": 1.5, "feature_a": 1.0}])
refused(ValueError, add_lags, pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["x"], [-1])

# the CLI (#61): a bad value is one line and exit 1
work = pathlib.Path(tempfile.mkdtemp(prefix="pmlab audit "))
panel.to_parquet(work / "panel.parquet")
exe = pathlib.Path(sys.executable).parent / "pmlab"
for extra in (["--flat-stake", "0"], ["--seed", "-1"], ["--lgbm-param", "n_estimators=abc"]):
    r = subprocess.run(
        [exe, "backtest", "--plugin", "weather_tmax", "--panel", "panel.parquet", "--stride", "10", "--min-train-rows", "4", *extra],
        cwd=work, capture_output=True, text=True, env={**os.environ, "NO_COLOR": "1", "COLUMNS": "200"},
    )  # fmt: skip
    assert r.returncode == 1 and "Traceback" not in r.stdout + r.stderr, (
        extra,
        r.returncode,
        r.stdout[-300:],
        r.stderr[-300:],
    )

# credentials stay out of the repr (#67)
shown = repr(
    PmlabSettings(
        POLY_API_KEY="key-123", POLY_API_SECRET="sekrit-456", POLY_API_PASSPHRASE="pass-789"
    )
)
assert not any(v in shown for v in ("key-123", "sekrit-456", "pass-789")), shown

# the clients (#63): a hang and a silent truncation are refused up front
refused(ValueError, AsyncClobClient, concurrency=0)


async def midpoint_check() -> None:
    client = AsyncClobClient(base_url="http://127.0.0.1:9", timeout=0.2, concurrency=1)
    assert await asyncio.wait_for(client.fetch_prices(["a"]), 5) == {}
    await client.aclose()


asyncio.run(midpoint_check())
print("audit inputs: OK")
