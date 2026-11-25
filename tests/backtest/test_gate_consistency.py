"""A gate file is data someone can edit: a GO has to allow something (#68).

``HoldoutGateResult.evaluate`` only returns GO when a segment passed. ``from_dict`` used to copy
the ``decision`` string as it found it, so a file edited to ``"decision": "GO"`` over failing
segments published a champion with ``Allowed segments: set()``. A GO that still has a failed
segment next to a passing one is a documented state (docs/watchdog.md) and stays valid.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.cli.main import app
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster


def _trades(**pnl_by_segment: float) -> pd.DataFrame:
    rows = []
    for segment, pnl in pnl_by_segment.items():
        rows += [{"segment": segment, "realized_pnl": pnl / 50} for _ in range(50)]
    return pd.DataFrame(rows)


def _evaluated(**pnl_by_segment: float) -> dict[str, Any]:
    gate = HoldoutGateResult.evaluate(
        _trades(**pnl_by_segment), required_segments=list(pnl_by_segment)
    )
    return gate.to_dict()


def _go_file() -> dict[str, Any]:
    d = _evaluated(nyc=5.0, london=3.0)
    assert d["decision"] == "GO"
    return d


class TestGenuineGatesStillLoad:
    def test_a_go_round_trips(self) -> None:
        d = _go_file()
        assert HoldoutGateResult.from_dict(d).to_dict() == d

    def test_a_no_go_round_trips(self) -> None:
        d = _evaluated(nyc=5.0, london=-3.0)
        assert d["decision"] == "NO_GO"
        assert HoldoutGateResult.from_dict(d).to_dict() == d

    def test_every_reason_evaluate_can_give_loads(self) -> None:
        trades = _trades(a=5.0, b=-1.0, c=1.0)
        trades.loc[trades["segment"] == "c", "realized_pnl"] = np.nan  # unsettled
        d = HoldoutGateResult.evaluate(
            trades, required_segments=["a", "b", "c", "absent"], min_trades_per_segment=60
        ).to_dict()
        assert {r["reason"] for r in d["segment_results"]} >= {"insufficient_trades", "missing"}
        assert HoldoutGateResult.from_dict(d).to_dict() == d

    def test_the_aggregate_gate_loads(self) -> None:
        d = HoldoutGateResult.evaluate(_trades(all=4.0)).to_dict()
        assert d["decision"] == "GO" and d["segment_results"][0]["segment"] == "all"
        assert HoldoutGateResult.from_dict(d).decision == "GO"

    def test_a_no_go_over_nothing_loads(self) -> None:
        d = {
            "decision": "NO_GO",
            "segment_results": [],
            "aggregate_pnl": 0.0,
            "aggregate_trades": 0,
        }
        assert HoldoutGateResult.from_dict(d).decision == "NO_GO"

    def test_a_go_with_a_failed_segment_is_a_documented_state_and_still_loads(self) -> None:
        """docs/watchdog.md: a champion trades the segments that passed; the failed one's trades
        are no evidence. Rejecting it would change documented behaviour (it is left on purpose)."""
        d = {
            "decision": "GO",
            "aggregate_pnl": 6.0,
            "aggregate_trades": 240,
            "segment_results": [
                {
                    "segment": "politics",
                    "num_trades": 120,
                    "total_pnl": 9.0,
                    "passes": True,
                    "reason": "ok",
                },
                {
                    "segment": "sports",
                    "num_trades": 120,
                    "total_pnl": -3.0,
                    "passes": False,
                    "reason": "negative_pnl",
                },
            ],
        }
        gate = HoldoutGateResult.from_dict(d)
        assert {r.segment for r in gate.segment_results if r.passes} == {"politics"}

    def test_a_passing_segment_with_any_reason_text_loads(self) -> None:
        # scripts/api_surface.py builds one with the reason "reason": the text is not validated
        d = _go_file()
        d["segment_results"][0]["reason"] = "reason"
        assert HoldoutGateResult.from_dict(d).decision == "GO"


class TestAForgedGoIsRefused:
    def test_go_over_one_failing_segment(self) -> None:
        d = _evaluated(london=-3.0)
        assert d["decision"] == "NO_GO"
        d["decision"] = "GO"  # the edit
        with pytest.raises(ValueError, match="no segment passed"):
            HoldoutGateResult.from_dict(d)

    def test_go_over_every_segment_failing(self) -> None:
        d = _evaluated(nyc=-5.0, london=-3.0)
        d["decision"] = "GO"
        with pytest.raises(ValueError, match="2 segment result"):
            HoldoutGateResult.from_dict(d)

    def test_go_over_no_segments(self) -> None:
        d = {"decision": "GO", "segment_results": [], "aggregate_pnl": 9.0, "aggregate_trades": 99}
        with pytest.raises(ValueError, match="no segment passed"):
            HoldoutGateResult.from_dict(d)

    def test_go_over_only_missing_segments(self) -> None:
        d = HoldoutGateResult.evaluate(_trades(nyc=5.0), required_segments=["oslo"]).to_dict()
        d["decision"] = "GO"
        with pytest.raises(ValueError, match="no segment passed"):
            HoldoutGateResult.from_dict(d)

    @pytest.mark.parametrize("decision", ["go", "Go", "MAYBE", "", None, True, 1, ["GO"]])
    def test_a_decision_that_is_not_go_or_no_go(self, decision: Any) -> None:
        d = _go_file()
        d["decision"] = decision
        with pytest.raises(ValueError, match="decision"):
            HoldoutGateResult.from_dict(d)

    @pytest.mark.parametrize("passes", ["false", "true", "", 1, 0, None, [True]])
    def test_passes_must_be_a_real_bool(self, passes: Any) -> None:
        # the string "false" is truthy: `get_allowed_segments` would have allowed the segment
        d = _go_file()
        d["segment_results"][0]["passes"] = passes
        with pytest.raises(ValueError, match="passes"):
            HoldoutGateResult.from_dict(d)

    def test_a_missing_key_is_still_a_key_error(self) -> None:
        d = _go_file()
        del d["segment_results"][0]["reason"]
        with pytest.raises(KeyError):
            HoldoutGateResult.from_dict(d)

    def test_the_input_is_not_modified(self) -> None:
        d = _go_file()
        before = copy.deepcopy(d)
        HoldoutGateResult.from_dict(d)
        assert d == before


def _fitted() -> LGBMForecaster:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.random(50), "f2": rng.random(50)})
    model = LGBMForecaster(objective="binary")
    model.fit(X, pd.Series((X["f1"] > 0.5).astype(int)))
    return model


class TestPublishChecksTheObjectToo:
    """A gate built in Python, not read from a file, can carry the same lie."""

    def _forged(self) -> HoldoutGateResult:
        return HoldoutGateResult(
            decision="GO",
            segment_results=[
                SegmentGateResult("nyc", 50, -5.0, False, "negative_pnl"),
                SegmentGateResult("london", 50, -3.0, False, "negative_pnl"),
            ],
            aggregate_pnl=-8.0,
            aggregate_trades=100,
        )

    def test_publish_refuses_a_go_that_allows_nothing(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="no segment passed"):
            ChampionManifest.publish(
                model=_fitted(),
                gate=self._forged(),
                output_dir=tmp_path / "out",
                plugin_family="weather_tmax",
            )
        assert not (tmp_path / "out").exists()

    def test_publish_still_refuses_a_no_go_with_the_old_message(self, tmp_path) -> None:
        gate = HoldoutGateResult.from_dict(_evaluated(nyc=-5.0))
        with pytest.raises(ValueError, match="NO_GO"):
            ChampionManifest.publish(
                model=_fitted(), gate=gate, output_dir=tmp_path / "out", plugin_family="x"
            )

    def test_publish_accepts_a_genuine_go(self, tmp_path) -> None:
        gate = HoldoutGateResult.from_dict(_go_file())
        manifest = ChampionManifest.publish(
            model=_fitted(), gate=gate, output_dir=tmp_path / "out", plugin_family="weather_tmax"
        )
        assert manifest.get_allowed_segments() == {"nyc", "london"}

    def test_publish_accepts_a_go_with_a_failed_segment_and_allows_the_rest(self, tmp_path) -> None:
        gate = HoldoutGateResult(
            "GO",
            [
                SegmentGateResult("politics", 120, 9.0, True, "ok"),
                SegmentGateResult("sports", 120, -3.0, False, "negative_pnl"),
            ],
            6.0,
            240,
        )
        manifest = ChampionManifest.publish(
            model=_fitted(), gate=gate, output_dir=tmp_path / "out", plugin_family="politics"
        )
        assert manifest.get_allowed_segments() == {"politics"}


class TestPromoteChampionRefusesAForgedFile:
    def test_a_go_file_over_a_failing_segment_exits_1_and_publishes_nothing(self, tmp_path) -> None:
        d = _evaluated(london=-3.0)
        d["decision"] = "GO"
        gate = tmp_path / "gate.json"
        gate.write_text(json.dumps(d))
        model = tmp_path / "model.pkl"
        model.write_bytes(b"x")
        result = CliRunner().invoke(
            app,
            [
                "promote-champion",
                str(model),
                "--gate-path",
                str(gate),
                "--plugin",
                "weather_tmax",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1, result.output
        assert "Not a valid gate file" in result.output and "no segment passed" in result.output
        assert "Traceback" not in result.output and not (tmp_path / "out").exists()
