"""The performance regression guard: pmlab's hot paths against their documented budget.

Opt-in (``pytest -m perf``; deselected by the default ``addopts``) because it times real work for
about a minute and must run on a quiet process: no coverage tracing, no other heavy job beside it.
Wall time is read in *calibration units* (seconds / a fixed pure-Python + pandas loop measured in
the same run), so the budgets hold across machines; ``docs/performance.md`` has the numbers and
what to do when a slowdown is intended. The scaling assertions are machine-independent.
"""

from __future__ import annotations

import sys

import pytest

from tests.perf._loader import load

pb = load()

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.gettrace() is not None,
        reason="a tracer (coverage, a debugger) is active: wall times would be meaningless",
    ),
]


@pytest.mark.parametrize("workload", pb.WORKLOADS, ids=lambda w: w.name)
def test_hot_path_stays_within_its_budget(workload: pb.Workload) -> None:
    payload = workload.prepare(1)
    verdict = pb.check_budget(
        lambda: pb.measure_workload(workload, 1, payload), workload.budget_units
    )
    print(
        f"\n{verdict.describe(workload.name)}; {verdict.seconds:.3f}s, "
        f"calibration {verdict.calibration * 1000:.1f}ms, reference {workload.reference_units:g}"
    )
    assert verdict.ok, (
        f"{verdict.describe(workload.name)}. If the slowdown is intended, update the budget "
        "(see docs/performance.md); if not, profile the hot path."
    )


@pytest.mark.parametrize("name", pb.SCALING)
def test_cost_grows_linearly_with_the_number_of_rows(name: str) -> None:
    workload = next(w for w in pb.WORKLOADS if w.name == name)
    ratio = pb.scaling_ratio(workload)
    print(f"\n{name}: 2x the rows cost {ratio:.2f}x")
    assert ratio < pb.SCALING_RATIO_LIMIT, (
        f"{name}: doubling the rows multiplied the time by {ratio:.2f} (limit "
        f"{pb.SCALING_RATIO_LIMIT:g}, linear is ~2): the cost has become super-linear"
    )


def test_calibration_is_stable_enough_to_normalise_by() -> None:
    """The unit must not itself wander, or every budget would: spread of 7 readings < 25%."""
    readings = sorted(pb.calibration_seconds(repeats=3) for _ in range(7))
    assert readings[-1] / readings[0] < 1.25, readings
