"""MkDocs hooks for the pmlab site: hand build-time data to the home-page template.

Registered in ``mkdocs.yml`` (``hooks:``). The home page shows numbers computed from the bundled
sample (``scripts/make_site_data.py`` writes ``docs/assets/data/sample-run.json``); this hook reads
that file, checks its schema, and turns it into what the template draws: percentages for the fold
timeline, bar heights for the Brier sparkline, the diverging per-city gate bars. The project version
comes from ``pyproject.toml``, so the page never states a stale one.

It does not import MkDocs, so ``tests/test_docs_theme.py`` can load it from a plain dev checkout.
"""

from __future__ import annotations

import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = 1
SHOWN_FOLDS = (0, 11, 23, 35, 47, 59)  # fold 1, 12, 24, 36, 48, 60 of 60
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _day(text: str) -> date:
    return date.fromisoformat(text)


def _pct(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def home_view(data: dict[str, Any]) -> dict[str, Any]:
    """The view-model of the home page: everything the template draws, as plain numbers."""
    folds = data["folds"]
    start = _day(data["backtest"]["train_start"])
    end = _day(folds[-1]["test"])
    span = (end - start).days + 1

    def x(day: str) -> float:
        return _pct((_day(day) - start).days / span * 100)

    rows = [
        {
            "n": i + 1,
            "train": x(folds[i]["train_end"]) + _pct(100 / span),  # the train day itself counts
            "test": x(folds[i]["test"]),
            "train_end": folds[i]["train_end"],
            "test_day": folds[i]["test"],
            "rows": folds[i]["train_rows"],
            "brier": folds[i]["brier"],
        }
        for i in SHOWN_FOLDS
        if i < len(folds)
    ]
    ticks = []
    month = date(start.year, start.month, 1)
    while month <= end:
        if month >= start:
            ticks.append({"at": x(month.isoformat()), "label": MONTHS[month.month - 1]})
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    briers = [f["brier"] for f in folds]
    low, high = min(briers), max(briers)
    spark = [round(18 + 82 * (b - low) / ((high - low) or 1), 1) for b in briers]

    segments = data["gate"]["segments"]
    pnls = [s["pnl"] for s in segments]
    lowest, highest = min(0.0, *pnls), max(0.0, *pnls)
    total = (highest - lowest) or 1.0
    zero = _pct(-lowest / total * 100)
    cities = [
        {
            **s,
            "negative": s["pnl"] < 0,
            "left": _pct((min(s["pnl"], 0.0) - lowest) / total * 100),
            "width": _pct(abs(s["pnl"]) / total * 100),
        }
        for s in segments
    ]
    return {
        "rows": rows,
        "ticks": ticks,
        "spark": spark,
        "spark_range": [low, high],
        "cities": cities,
        "zero": zero,
        "last_fold": folds[-1],
        "first_fold": folds[0],
    }


def on_config(config: Any) -> Any:
    docs = Path(config.docs_dir)
    data: dict[str, Any] = json.loads((docs / "assets" / "data" / "sample-run.json").read_text())
    if data.get("schema") != SCHEMA:
        raise RuntimeError(
            f"docs/assets/data/sample-run.json has schema {data.get('schema')!r}, the site "
            f"expects {SCHEMA}: rerun scripts/make_site_data.py"
        )
    config["extra"]["sample"] = data
    config["extra"]["home"] = home_view(data)
    pyproject = docs.parent / "pyproject.toml"
    if pyproject.exists():
        config["extra"]["pmlab_version"] = tomllib.loads(pyproject.read_text())["project"][
            "version"
        ]
    return config
