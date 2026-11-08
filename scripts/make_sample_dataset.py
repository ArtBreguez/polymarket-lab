#!/usr/bin/env python
"""Generate the bundled sample dataset ``src/pmlab/datasets/data/weather_tmax_sample.json.gz``.

The data is SYNTHETIC. It exists so the tutorial and the docs can run the whole pmlab
workflow offline, and it is written to behave like the real thing where it matters:

* one "Highest temperature in <city> on <date>?" market per city per day, nine bins
  (a single degree each, plus a tail at either end) centred where the book thinks the
  day will land;
* a forecast (the model's features) that is better than what the book priced in, so a
  model can find an edge in four cities, and a fifth city (Lisbon) whose book is
  efficient, so a per-city gate has something real to refuse;
* prices with an overround and quote noise, observed at 07:30 UTC the day before;
* integer-degree truths, final once the target day has ended.

Run it from the repo root to regenerate (the output is byte-for-byte deterministic for a
given numpy; the checksum next to it is what the loader verifies, not this script)::

    uv run python scripts/make_sample_dataset.py

Regenerating changes the data every test and the tutorial notebook was written against:
re-run both afterwards.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SEED = 5  # chosen so the per-city gate has something real to separate (see the docstring)
START = date(2025, 4, 2)  # first target day
N_DAYS = 150
HALF_WIDTH = 3  # single-degree bins at centre-3 .. centre+3, plus a tail each side: 9 bins

# name, timezone, mean April-to-August daily max (C), seasonal amplitude (C), book skill (the
# share of the forecast anomaly the book prices in; 1.0 prices all of it), book forecast error
# (C). Lisbon's book is sharper than our forecast (0.2 against 0.8): trading against it loses.
CITIES = (
    ("Madrid", "Europe/Madrid", 21.0, 8.0, 0.60, 1.5),
    ("Rome", "Europe/Rome", 21.0, 7.0, 0.60, 1.5),
    ("Athens", "Europe/Athens", 22.0, 7.5, 0.60, 1.5),
    ("Istanbul", "Europe/Istanbul", 19.0, 8.0, 0.60, 1.5),
    ("Lisbon", "Europe/Lisbon", 21.0, 5.0, 1.00, 0.2),
)
FORECAST_NOISE = 0.8  # sd of our forecast error (C)
BOOK_SIGMA = 2.6  # the book's spread, wider than the truth's (a book is cautious)
OVERROUND = 1.05

OUT = Path(__file__).resolve().parents[1] / "src" / "pmlab" / "datasets" / "data"
DATA = "weather_tmax_sample.json.gz"
SUM = "weather_tmax_sample.sha256"


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _masses(cdf: list[float]) -> list[float]:
    """Tail below, the single degrees between, tail above."""
    lows = [cdf[0]]
    mids = [b - a for a, b in zip(cdf, cdf[1:], strict=False)]
    return [*lows, *mids, 1.0 - cdf[-1]]


def generate(seed: int = SEED) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    days = [START + timedelta(days=i) for i in range(N_DAYS)]
    markets: list[dict[str, object]] = []
    per_city: dict[str, list[dict[str, object]]] = {}
    for name, _tz, base, amp, skill, book_noise in CITIES:
        anomaly = 0.0
        rows: list[dict[str, object]] = []
        for d in days:
            doy = d.timetuple().tm_yday
            climo = base + amp * math.sin(2 * math.pi * (doy - 105) / 365)
            anomaly = 0.65 * anomaly + float(rng.normal(0.0, 3.0 * math.sqrt(1 - 0.65**2)))
            mean = climo + anomaly
            sigma = 0.8 + 0.9 * float(rng.gamma(4.0, 0.25))
            truth = int(round(mean + sigma * float(rng.normal())))
            fcst = mean + float(rng.normal(0.0, FORECAST_NOISE))
            book_fcst = mean + float(rng.normal(0.0, book_noise))
            book_mean = climo + skill * (book_fcst - climo)
            book_sigma = BOOK_SIGMA if skill < 1 else sigma * 1.1
            centre = int(round(book_mean))
            edges = [centre + k + 0.5 for k in range(-HALF_WIDTH - 1, HALF_WIDTH + 1)]
            cdf = [_cdf((e - book_mean) / book_sigma) for e in edges]
            masses = _masses(cdf)
            quote_noise = np.exp(rng.normal(0.0, 0.08, size=len(masses)))
            noisy = np.array(masses) * quote_noise
            book = noisy / noisy.sum() * OVERROUND  # a stable overround, noise in the shape
            prices = [round(float(min(0.97, max(0.005, p))), 3) for p in book]
            humidity = float(np.clip(55 - 2.0 * anomaly + rng.normal(0, 8), 15, 95))
            features = {
                "feature_fcst_tmax": fcst,
                "feature_fcst_tmin": fcst - 8.5 + float(rng.normal(0, 1.2)),
                "feature_climo_tmax": climo,
                "feature_ens_spread": sigma * (1 + float(rng.normal(0, 0.1))),
                "feature_humidity": humidity,
                "feature_dew_point": fcst - 14 + 0.15 * (humidity - 55) + float(rng.normal(0, 1.5)),
                "feature_cloud_cover": float(
                    np.clip(45 - 5.0 * anomaly + rng.normal(0, 15), 0, 100)
                ),
                "feature_wind_speed": float(rng.gamma(3.0, 1.6)),
            }
            rows.append(
                {
                    "id": f"tmax-{name.lower()}-{d.isoformat()}",
                    "city": name,
                    "target_date": d.isoformat(),
                    "center": centre,
                    "prices": prices,
                    "truth_c": truth,
                    "features": {k: round(float(v), 2) for k, v in features.items()},
                }
            )
        per_city[name] = rows
    for d in days:  # interleave by day, as discovery sees them
        for name, *_ in CITIES:
            markets.append(next(m for m in per_city[name] if m["target_date"] == d.isoformat()))
    return {
        "schema_version": 1,
        "meta": {
            "synthetic": True,
            "family": "weather_tmax",
            "unit": "C",
            "horizon": "previous_evening",
            "seed": seed,
            "generator": "scripts/make_sample_dataset.py",
            "description": (
                "Synthetic 'Highest temperature in <city> on <date>?' markets: one per city per "
                f"day for {N_DAYS} days from {START.isoformat()}, nine bins each. Prices are the "
                "book as seen at 07:30 UTC the day before; features are the forecast at that time; "
                "truth_c is the day's official maximum, final once the day has ended. Not real "
                "Polymarket data."
            ),
            "cities": {name: tz for name, tz, *_ in CITIES},
            "half_width": HALF_WIDTH,
            "price_observed_utc": "07:30",
        },
        "markets": markets,
    }


def write(payload: dict[str, object], out: Path = OUT) -> tuple[Path, str]:
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    blob = gzip.compress(body, compresslevel=9, mtime=0)  # mtime 0: the bytes do not depend on when
    (out / DATA).write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    (out / SUM).write_text(digest + "\n")
    return out / DATA, digest


if __name__ == "__main__":
    path, digest = write(generate())
    print(f"{path} {path.stat().st_size} bytes sha256 {digest}")
    sys.exit(0)
