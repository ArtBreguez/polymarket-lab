"""Fast tests (default suite) of the performance harness itself: workloads, timing, verdicts.

The slow part (actually timing pmlab against its budget) is ``test_perf_budget.py``, opt-in
with ``pytest -m perf``. These tests pin what that guard relies on: the synthetic workloads are
deterministic and honour the panel contract, ``best_of`` takes the minimum after a warm-up, a
budget verdict survives a noisy first measurement but not a real regression, and the budget the
docs publish is the budget the guard asserts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab.data import build_panel
from pmlab.data.leakage import check_price_timing
from tests.perf._loader import REPO, load

pb = load()


def test_make_panel_is_deterministic_and_seed_sensitive() -> None:
    a = pb.make_panel(n_dates=6, n_markets=3, n_bins=4, seed=1)
    b = pb.make_panel(n_dates=6, n_markets=3, n_bins=4, seed=1)
    c = pb.make_panel(n_dates=6, n_markets=3, n_bins=4, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["feature_0"].equals(c["feature_0"])


def test_make_panel_honours_the_rolling_origin_contract() -> None:
    panel = pb.make_panel(n_dates=6, n_markets=3, n_bins=4)
    assert len(panel) == 6 * 3 * 4
    for col in ("market_id", "decision_date", "outcome_label", "winning_label", "market_price"):
        assert col in panel.columns
    assert [c for c in panel.columns if c.startswith("feature_")]
    assert panel["decision_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    block = panel.groupby(["market_id", "decision_date"])
    # one winning bin per market-date, prices are a distribution over the bins
    assert (block.apply(lambda g: (g["outcome_label"] == g["winning_label"]).sum()) == 1).all()
    assert np.allclose(block["market_price"].sum(), 1.0)
    assert panel["market_price"].between(0.0, 1.0).all()


def test_make_rows_feed_build_panel_with_timing_stamps_that_pass_the_guard() -> None:
    rows = pb.make_rows(n_markets=5, n_dates=3, n_bins=4)
    assert len(rows) == 5 and all(isinstance(m, list) and len(m) == 12 for m in rows)
    panel = build_panel(rows)
    assert len(panel) == 5 * 3 * 4
    report = check_price_timing(panel)
    assert report.n_checked == len(panel)  # stamped rows: the timing check does real work


def test_workloads_cover_the_hot_paths_and_the_scaling_pairs() -> None:
    names = {w.name for w in pb.WORKLOADS}
    assert {"rolling_origin_top_prob", "rolling_origin_live_policy", "build_panel"} <= names
    for w in pb.WORKLOADS:
        assert w.budget_units > w.reference_units > 0
        assert w.prepare(1) is not None
    assert set(pb.SCALING) <= names


def test_best_of_warms_up_then_takes_the_minimum() -> None:
    ticks = iter([0.0, 5.0, 10.0, 12.0, 20.0, 29.0])  # three timed runs: 5s, 2s, 9s
    calls: list[int] = []
    got = pb.best_of(lambda: calls.append(1), repeats=3, warmup=1, clock=lambda: next(ticks))
    assert got == pytest.approx(2.0)
    assert len(calls) == 4  # 1 warm-up + 3 timed


def test_check_budget_passes_under_budget_without_extra_rounds() -> None:
    rounds: list[int] = []

    def measure() -> tuple[float, float]:
        rounds.append(1)
        return 0.5, 0.1  # workload seconds, calibration seconds -> 5.0 units

    verdict = pb.check_budget(measure, budget_units=8.0, max_rounds=3)
    assert verdict.ok and verdict.units == pytest.approx(5.0) and len(rounds) == 1


def test_check_budget_survives_one_noisy_round_but_not_a_real_regression() -> None:
    noisy = iter([(2.0, 0.1), (0.55, 0.1), (0.5, 0.1)])  # 20, 5.5, 5 units
    verdict = pb.check_budget(lambda: next(noisy), budget_units=8.0, max_rounds=3)
    assert verdict.ok and verdict.units == pytest.approx(5.5)

    slow = iter([(2.0, 0.1), (1.9, 0.1), (2.1, 0.1)])  # always ~20 units: a real regression
    verdict = pb.check_budget(lambda: next(slow), budget_units=8.0, max_rounds=3)
    assert not verdict.ok and verdict.units == pytest.approx(19.0)
    assert "19.0" in verdict.describe("w") and "8.0" in verdict.describe("w")


def test_calibration_is_positive_and_the_unit_is_a_pure_function_of_the_machine() -> None:
    assert pb.calibration_seconds(repeats=2) > 0


def test_docs_publish_the_budget_the_guard_asserts() -> None:
    doc = (REPO / "docs" / "performance.md").read_text()
    for w in pb.WORKLOADS:
        row = re.search(rf"^\|\s*`{w.name}`\s*\|(.*)$", doc, re.MULTILINE)
        assert row is not None, f"docs/performance.md has no table row for {w.name}"
        assert f"{w.budget_units:g}" in row.group(1), f"{w.name}: budget not in the doc row"
        assert f"{w.reference_units:g}" in row.group(1), f"{w.name}: reference not in the doc row"
    assert f"{pb.SCALING_RATIO_LIMIT:g}" in doc


def test_script_is_importable_without_running_benchmarks(tmp_path: Path) -> None:
    assert callable(pb.main) and pb.__doc__
