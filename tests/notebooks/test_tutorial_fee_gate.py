"""The bundled sample's gate, recomputed under each fee model, and written down (#72).

The tutorial walks the sample through tune -> backtest -> gate. Charging a Polymarket taker fee
instead of the flat 0.30% does not change the logic of `HoldoutGateResult`, only the PnL it reads, so
the question is whether the sample's decision survives: it does, under the default schedule and under
the weather category's. The same notebook, with only its cost line swapped, is run for each model
and the decision is held to what ``docs/costs.md`` says.

Runs the real notebook offline (about 12 s each), so it is a ``notebook`` test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.notebooks.test_tutorial_notebook import REPO, _nb, _run_notebook

FLAT_LINE = 'costs = CostModel(fee_model="flat-bps", slippage_bps=50.0)'
VARIANTS = {
    # what the shipped tutorial charges: the fee of releases before 1.5.0
    "flat-bps": FLAT_LINE,
    # the default model: the Polymarket fee at the "other" category's 0.05 (the highest published)
    "polymarket": "costs = CostModel(slippage_bps=50.0)",
    # the same fee model at the rate the weather category publishes (0.01)
    "polymarket-weather": (
        "from pmlab import FeeSchedule\n"
        'costs = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="user", '
        "slippage_bps=50.0)"
    ),
}
PASSING = {"Athens", "Istanbul", "Madrid", "Rome"}
ROW = re.compile(r"^(\w+)\s+(\d+)\s+(-?\d+\.\d+)\s+(True|False)\s+(\w+)\s*$", re.MULTILINE)


def _gate(tmp_path: Path, line: str) -> tuple[str, dict[str, tuple[float, bool, str]]]:
    """The decision and the per-city (pnl, passes, reason) of the notebook run with ``line``."""
    nb = _nb()
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and FLAT_LINE in src:
            cell["source"] = src.replace(FLAT_LINE, line)
            break
    else:  # pragma: no cover - the cost line moved; this test must move with it
        pytest.fail("could not find the notebook's cost line")
    patched = tmp_path / "variant.ipynb"
    patched.write_text(json.dumps(nb))
    executed = tmp_path / "executed.ipynb"
    proc = _run_notebook(tmp_path, "--out", str(executed), notebook=patched)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = ""
    for cell in json.loads(executed.read_text())["cells"]:
        if cell["cell_type"] != "code" or "gate_all = HoldoutGateResult.evaluate" not in "".join(
            cell["source"]
        ):
            continue
        for out in cell["outputs"]:
            text += "".join(out.get("text", "")) + "".join(
                out.get("data", {}).get("text/plain", "")
            )
    decision = re.search(r"decision: (\w+)", text)
    assert decision, text
    rows = {m[1]: (float(m[3]), m[4] == "True", m[5]) for m in ROW.finditer(text)}
    return decision[1], rows


@pytest.fixture(scope="module")
def gates(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[str, dict]]:
    return {name: _gate(tmp_path_factory.mktemp(name), line) for name, line in VARIANTS.items()}


@pytest.mark.notebook
class TestTheSamplesGateUnderEachFeeModel:
    @pytest.mark.parametrize("name", VARIANTS)
    def test_the_decision_is_no_go_and_lisbon_is_what_refuses(
        self, gates: dict[str, tuple[str, dict]], name: str
    ) -> None:
        decision, rows = gates[name]
        assert decision == "NO_GO"
        assert {city for city, (_, passes, _) in rows.items() if passes} == PASSING
        assert rows["Lisbon"][1:] == (False, "negative_pnl")

    def test_the_curve_fee_costs_every_passing_city_pnl_at_the_default_schedule(
        self, gates: dict[str, tuple[str, dict]]
    ) -> None:
        flat, polymarket = gates["flat-bps"][1], gates["polymarket"][1]
        for city in PASSING:
            assert polymarket[city][0] < flat[city][0], city

    def test_the_weather_rate_sits_between_the_two(
        self, gates: dict[str, tuple[str, dict]]
    ) -> None:
        flat = sum(gates["flat-bps"][1][c][0] for c in PASSING)
        weather = sum(gates["polymarket-weather"][1][c][0] for c in PASSING)
        default = sum(gates["polymarket"][1][c][0] for c in PASSING)
        assert default < weather < flat


class TestItIsDocumented:
    """The page says what the runs above show, so it cannot drift from them unnoticed."""

    def test_the_costs_page_records_the_samples_gate_under_each_model(self) -> None:
        text = (REPO / "docs" / "costs.md").read_text()
        section = text.split("## The bundled sample", 1)[1].split("\n## ", 1)[0]
        assert "NO_GO" in section and "Lisbon" in section and "negative_pnl" in section
        for city in sorted(PASSING):
            assert city in section, city
        for model in ("flat-bps", "polymarket"):
            assert model in section, model
        assert "0.05" in section and "0.01" in section
