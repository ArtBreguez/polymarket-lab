"""Tests for `pmlab monitor`: the watchdog as a cron job would run it.

The contract a scheduler depends on, in order of importance:

1. a healthy champion produces **no output** and exits 0;
2. every problem exits with a code that says what kind, and prints why;
3. a crash, a bad path or a corrupt file can never look like a healthy run.

These tests drive the real command through typer's CliRunner against real files
(champion.json, a trade log, a parquet reference panel, a snapshot store). The core
decisions are pinned in tests/monitoring/test_watchdog.py; here the subject is the
loading, the wiring, the streams and the exit codes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pmlab.backtest.expectation import BacktestExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.cli.main import app
from pmlab.data import FeatureSnapshotStore
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.monitoring.watchdog import HealthReport

runner = CliRunner()

_AS_OF = "2026-03-01T00:00:00+00:00"
_PUBLISHED = "2026-01-01T00:00:00+00:00"
_FAMILY = "politics"

EXIT_OK, EXIT_WARN, EXIT_ALERT, EXIT_INSUFFICIENT, EXIT_USAGE, EXIT_INTERNAL = 0, 10, 20, 30, 64, 70


# --------------------------------------------------------------------------
# A small, real workspace
# --------------------------------------------------------------------------


def _expectation(**overrides: Any) -> BacktestExpectation:
    base = {
        "n_trades": 2000,
        "flat_stake": 1.0,
        "mean_pnl_per_stake": 0.10,
        "std_pnl_per_stake": 0.30,
        "brier_score": 0.20,
        "brier_std": 0.10,
    }
    base.update(overrides)
    return BacktestExpectation(**base)


def _two_point(n: int, mean: float, sd: float) -> list[float]:
    return [mean + sd * (1 if i % 2 == 0 else -1) for i in range(n)]


def _trades(
    n: int = 60, *, brier: float = 0.20, pnl: float = 0.10, segment: str = "A", **extra: Any
) -> list[dict[str, Any]]:
    """Settled trades with an exact mean Brier term and mean return per stake."""
    out = []
    for term, ret in zip(_two_point(n, brier, 0.10), _two_point(n, pnl, 0.30), strict=True):
        out.append(
            {
                "recorded_at": "2026-02-01T10:00:00+00:00",
                "city_or_segment": segment,
                "target_date": "2026-02-05",
                "outcome_label": "YES",
                "direction": "yes",
                "gamma_price": 0.40,
                "model_prob": math.sqrt(term),
                "flat_stake": 1.0,
                "outcome": "lost",
                "realized_pnl": ret,
                **extra,
            }
        )
    return out


@pytest.fixture(scope="module")
def forecaster() -> LGBMForecaster:
    """A low-capacity model, so in-sample and live predictions are comparable."""
    rng = np.random.default_rng(7)
    n = 600
    X = pd.DataFrame(
        {"feature_price": rng.normal(0.5, 0.1, n), "feature_volume": rng.lognormal(3, 0.5, n)}
    )
    y = pd.Series((X["feature_price"] + rng.normal(0, 0.1, n) > 0.5).astype(int))
    model = LGBMForecaster(n_estimators=15, num_leaves=4, min_child_samples=100)
    model.fit(X, y)
    return model


@dataclass
class World:
    root: Path
    champion: Path
    trades: Path
    reference: Path
    snapshots: Path
    forecaster: LGBMForecaster

    def args(self, *extra: str, drift: bool = True, as_of: str | None = _AS_OF) -> list[str]:
        args = ["monitor", "--champion", str(self.champion), "--trades", str(self.trades)]
        if drift:
            args += ["--reference", str(self.reference), "--snapshots", str(self.snapshots)]
        if as_of is not None:
            args += ["--as-of", as_of]
        return [*args, *extra]

    def write_trades(self, trades: list[dict[str, Any]]) -> None:
        self.trades.write_text(json.dumps({"trades": trades}))

    def patch_champion(self, **changes: Any) -> None:
        data = json.loads(self.champion.read_text())
        for key, value in changes.items():
            if value is _DELETE:
                data.pop(key, None)
            else:
                data[key] = value
        self.champion.write_text(json.dumps(data))


_DELETE = object()


def _make_world(
    tmp_path: Path,
    forecaster: LGBMForecaster,
    *,
    trades: list[dict[str, Any]] | None = None,
    expectation: BacktestExpectation | None = None,
    shift: float = 0.0,
) -> World:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_ref, n_live = 2000, 400

    gate = HoldoutGateResult(
        decision="GO",
        segment_results=[
            SegmentGateResult("A", 100, 10.0, True, "ok"),
            SegmentGateResult("B", 100, 10.0, True, "ok"),
            SegmentGateResult("C", 100, -10.0, False, "negative_pnl"),
        ],
        aggregate_pnl=20.0,
        aggregate_trades=200,
    )
    out_dir = tmp_path / "public_models"
    ChampionManifest.publish(
        model=forecaster,
        gate=gate,
        output_dir=out_dir,
        plugin_family=_FAMILY,
        expectation=expectation if expectation is not None else _expectation(),
    )
    world = World(
        root=tmp_path,
        champion=out_dir / "champion.json",
        trades=tmp_path / "forward_paper_trades.json",
        reference=tmp_path / "reference.parquet",
        snapshots=tmp_path / "snaps",
        forecaster=forecaster,
    )
    # Publish stamps "now"; pin it so the fixture trades (February) follow it.
    world.patch_champion(published_at=_PUBLISHED)
    world.write_trades(_trades() if trades is None else trades)

    pd.DataFrame(
        {
            "market_id": [f"train{i}" for i in range(n_ref)],
            "decision_date": "2026-01-10",
            "winning_label": "Yes",
            "feature_price": rng.normal(0.5, 0.1, n_ref),
            "feature_volume": rng.lognormal(3, 0.5, n_ref),
        }
    ).to_parquet(world.reference)

    store = FeatureSnapshotStore(world.snapshots)
    store.append(
        _FAMILY,
        [
            {
                "market_id": f"live{i}",
                "captured_at": "2026-02-25T12:00:00Z",
                "feature_price": float(rng.normal(0.5 + shift, 0.1)),
                "feature_volume": float(rng.lognormal(3, 0.5)),
            }
            for i in range(n_live)
        ],
    )
    return world


@pytest.fixture
def world(tmp_path: Path, forecaster: LGBMForecaster) -> World:
    return _make_world(tmp_path, forecaster)


# --------------------------------------------------------------------------
# Quiet on green
# --------------------------------------------------------------------------


class TestHealthy:
    def test_a_healthy_champion_exits_zero_and_prints_nothing(self, world: World) -> None:
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_OK, result.stderr
        assert result.stdout == ""
        assert result.stderr == ""

    def test_verbose_prints_the_full_report_even_when_green(self, world: World) -> None:
        result = runner.invoke(app, world.args("--verbose"))
        assert result.exit_code == EXIT_OK
        assert "OK" in result.stdout
        for name in ("drift", "calibration", "pnl"):
            assert name in result.stdout
        assert result.stderr == ""

    def test_json_is_written_even_when_healthy_and_the_terminal_stays_quiet(
        self, world: World
    ) -> None:
        out = world.root / "health.json"
        result = runner.invoke(app, world.args("--json", str(out)))
        assert result.exit_code == EXIT_OK
        assert result.stdout == "" and result.stderr == ""
        report = HealthReport.from_dict(json.loads(out.read_text()))
        assert report.status == "ok"
        assert [c.name for c in report.checks] == ["drift", "calibration", "pnl"]

    def test_the_json_round_trips_exactly(self, world: World) -> None:
        out = world.root / "health.json"
        runner.invoke(app, world.args("--json", str(out)))
        data = json.loads(out.read_text())
        assert HealthReport.from_dict(data).to_dict() == data

    def test_two_runs_with_the_same_as_of_write_identical_json(self, world: World) -> None:
        a, b = world.root / "a.json", world.root / "b.json"
        runner.invoke(app, world.args("--json", str(a)))
        runner.invoke(app, world.args("--json", str(b)))
        assert a.read_text() == b.read_text()

    def test_json_dash_writes_to_stdout_and_nothing_else_when_healthy(self, world: World) -> None:
        result = runner.invoke(app, world.args("--json", "-"))
        assert result.exit_code == EXIT_OK
        assert HealthReport.from_dict(json.loads(result.stdout)).status == "ok"
        assert result.stderr == ""

    def test_prediction_drift_is_part_of_the_default_run(self, world: World) -> None:
        out = world.root / "h.json"
        runner.invoke(app, world.args("--json", str(out)))
        metrics = json.loads(out.read_text())["checks"][0]["metrics"]
        assert metrics["prediction_psi"] is not None

    def test_no_prediction_skips_loading_the_model(self, world: World) -> None:
        (world.champion.parent / "champion.pkl").unlink()
        out = world.root / "h.json"
        result = runner.invoke(app, world.args("--no-prediction", "--json", str(out)))
        assert result.exit_code == EXIT_OK, result.stderr
        assert json.loads(out.read_text())["checks"][0]["metrics"]["prediction_psi"] is None


# --------------------------------------------------------------------------
# Each check drives the exit code on its own
# --------------------------------------------------------------------------


class TestDrift:
    def test_a_drifted_feature_alerts_and_is_named(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        world = _make_world(tmp_path, forecaster, shift=0.08)
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_ALERT
        assert "ALERT" in result.stdout
        assert "feature_price" in result.stdout
        assert "psi" in result.stdout
        assert "feature_volume" not in result.stdout
        assert result.stderr == ""

    def test_a_moderate_drift_warns(self, tmp_path: Path, forecaster: LGBMForecaster) -> None:
        world = _make_world(tmp_path, forecaster, shift=0.03)
        result = runner.invoke(app, world.args("--no-prediction"))
        assert result.exit_code == EXIT_WARN
        assert "feature_price" in result.stdout

    def test_drift_thresholds_come_from_the_config_file(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        world = _make_world(tmp_path, forecaster, shift=0.08)
        cfg = tmp_path / "wd.toml"
        cfg.write_text(
            "[drift.thresholds]\npsi_warn = 5.0\npsi_alert = 9.0\nkl_warn = 5.0\nkl_alert = 9.0\n"
        )
        result = runner.invoke(app, world.args("--no-prediction", "--config", str(cfg)))
        assert result.exit_code == EXIT_OK, result.stdout

    def test_snapshots_captured_after_as_of_are_invisible(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        """Point-in-time: a replay at an earlier as-of must not see later captures."""
        world = _make_world(tmp_path, forecaster)
        late = np.random.default_rng(5)
        FeatureSnapshotStore(world.snapshots).append(
            _FAMILY,
            [
                {
                    "market_id": f"live{i}",
                    "captured_at": "2026-03-02T12:00:00Z",
                    "feature_price": float(late.normal(0.95, 0.01)),
                    "feature_volume": 20.0,
                }
                for i in range(400)
            ],
        )
        before = runner.invoke(app, world.args("--no-prediction"))
        assert before.exit_code == EXIT_OK
        after = runner.invoke(app, world.args("--no-prediction", as_of="2026-03-03T00:00:00+00:00"))
        assert after.exit_code == EXIT_ALERT

    def test_too_few_live_rows_is_insufficient_not_ok(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        world = _make_world(tmp_path, forecaster)
        result = runner.invoke(
            app, world.args("--no-prediction", as_of="2026-02-20T00:00:00+00:00")
        )
        assert result.exit_code == EXIT_INSUFFICIENT
        assert "drift" in result.stdout and "insufficient" in result.stdout

    def test_without_drift_inputs_the_check_is_unavailable_not_green(self, world: World) -> None:
        result = runner.invoke(app, world.args(drift=False))
        assert result.exit_code == EXIT_INSUFFICIENT
        assert "drift" in result.stdout
        assert "--reference" in result.stdout or "reference" in result.stdout

    def test_skipping_drift_runs_the_other_two(self, world: World) -> None:
        result = runner.invoke(
            app,
            world.args(
                "--skip",
                "drift",
                drift=False,
            ),
        )
        assert result.exit_code == EXIT_OK, result.stdout
        assert result.stdout == ""

    def test_window_flags_are_honoured(self, world: World) -> None:
        narrow = runner.invoke(app, world.args("--no-prediction", "--window-days", "1"))
        assert narrow.exit_code == EXIT_INSUFFICIENT  # the 02-25 capture is outside a 1-day window
        explicit = runner.invoke(
            app, world.args("--no-prediction", "--window-start", "2026-02-20T00:00:00+00:00")
        )
        assert explicit.exit_code == EXIT_OK

    def test_the_family_defaults_to_the_champions_and_can_be_overridden(self, world: World) -> None:
        wrong = runner.invoke(app, world.args("--no-prediction", "--family", "sports"))
        assert wrong.exit_code == EXIT_USAGE  # the store has nothing for it: input error
        assert "sports" in wrong.stderr
        right = runner.invoke(app, world.args("--no-prediction", "--family", _FAMILY))
        assert right.exit_code == EXIT_OK

    def test_features_can_be_chosen_explicitly(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        world = _make_world(tmp_path, forecaster, shift=0.08)
        only_volume = runner.invoke(
            app, world.args("--no-prediction", "--feature", "feature_volume")
        )
        assert only_volume.exit_code == EXIT_OK, "feature_price was not asked for"

    def test_reference_and_snapshots_must_come_together(self, world: World) -> None:
        result = runner.invoke(app, world.args(drift=False) + ["--reference", str(world.reference)])
        assert result.exit_code == EXIT_USAGE
        assert "--snapshots" in result.stderr


class TestCalibration:
    def test_moderate_degradation_warns(self, world: World) -> None:
        world.write_trades(_trades(brier=0.23))
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_WARN
        assert "calibration" in result.stdout and "Brier" in result.stdout
        assert "0.230" in result.stdout and "0.200" in result.stdout

    def test_severe_degradation_alerts(self, world: World) -> None:
        world.write_trades(_trades(brier=0.27))
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_ALERT
        assert "ALERT" in result.stdout

    def test_only_the_failing_check_is_in_the_message(self, world: World) -> None:
        world.write_trades(_trades(brier=0.27))
        result = runner.invoke(app, world.args())
        assert "calibration" in result.stdout
        assert "drift" not in result.stdout and "pnl" not in result.stdout


class TestPnl:
    def test_a_shortfall_warns(self, world: World) -> None:
        world.write_trades(_trades(100, pnl=0.03))
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_WARN
        assert "pnl" in result.stdout and "backtest" in result.stdout

    def test_losing_money_alerts(self, world: World) -> None:
        world.write_trades(_trades(100, pnl=-0.02))
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_ALERT
        assert "-0.020" in result.stdout

    def test_the_worst_check_sets_the_exit_code(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        world = _make_world(tmp_path, forecaster, shift=0.03, trades=_trades(pnl=-0.02))
        result = runner.invoke(app, world.args("--no-prediction"))
        assert result.exit_code == EXIT_ALERT
        assert "drift" in result.stdout and "pnl" in result.stdout


# --------------------------------------------------------------------------
# Champion.json is the authority
# --------------------------------------------------------------------------


class TestChampionIsAuthoritative:
    def test_trades_outside_the_allowed_segments_are_ignored(self, world: World) -> None:
        """Segment C failed the gate; its losses are not this champion's evidence."""
        world.write_trades(_trades(60) + _trades(60, brier=0.70, pnl=-1.0, segment="C"))
        result = runner.invoke(app, world.args("--json", str(world.root / "h.json")))
        assert result.exit_code == EXIT_OK
        pnl = json.loads((world.root / "h.json").read_text())["checks"][2]["metrics"]
        assert pnl["n_out_of_segment"] == 60 and pnl["n_trades"] == 60

    def test_the_allowlist_is_read_from_champion_json_not_from_the_trades(
        self, world: World
    ) -> None:
        gate = json.loads(world.champion.read_text())["publish_gate"]
        for seg in gate["segment_results"]:
            seg["passes"] = seg["segment"] == "B"  # only B allowed now
        world.patch_champion(publish_gate=gate)
        result = runner.invoke(app, world.args("--skip", "drift", drift=False))
        assert result.exit_code == EXIT_INSUFFICIENT  # all 60 trades are in A: not evidence

    def test_the_champions_own_expectation_is_what_trades_are_held_against(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        # Identical trades; only champion.json differs.
        modest = _make_world(
            tmp_path / "m", forecaster, expectation=_expectation(mean_pnl_per_stake=0.10)
        )
        ambitious = _make_world(
            tmp_path / "a", forecaster, expectation=_expectation(mean_pnl_per_stake=0.40)
        )
        assert runner.invoke(app, modest.args("--no-prediction")).exit_code == EXIT_OK
        assert runner.invoke(app, ambitious.args("--no-prediction")).exit_code != EXIT_OK

    def test_trades_from_before_the_champion_was_published_do_not_count(self, world: World) -> None:
        world.write_trades(_trades(60, brier=0.70, pnl=-1.0, recorded_at="2025-12-01T00:00:00Z"))
        result = runner.invoke(app, world.args("--skip", "drift", drift=False))
        assert result.exit_code == EXIT_INSUFFICIENT
        assert "before this champion was published" in result.stdout

    def test_an_old_champion_without_an_expectation_is_unavailable_not_green(
        self, world: World
    ) -> None:
        world.patch_champion(backtest_expectation=_DELETE)
        result = runner.invoke(app, world.args("--no-prediction"))
        assert result.exit_code == EXIT_INSUFFICIENT
        assert "expectation" in result.stdout
        assert "Traceback" not in result.stdout + result.stderr

    def test_a_champion_with_a_null_expectation_is_the_same(self, world: World) -> None:
        world.patch_champion(backtest_expectation=None)
        assert runner.invoke(app, world.args("--no-prediction")).exit_code == EXIT_INSUFFICIENT

    def test_the_report_names_the_champion_it_judged(self, world: World) -> None:
        result = runner.invoke(app, world.args("--verbose", "--no-prediction"))
        assert "champion" in result.stdout


# --------------------------------------------------------------------------
# Insufficient is not healthy
# --------------------------------------------------------------------------


class TestInsufficient:
    def test_too_few_trades_has_its_own_exit_code(self, world: World) -> None:
        world.write_trades(_trades(10))
        result = runner.invoke(app, world.args("--no-prediction"))
        assert result.exit_code == EXIT_INSUFFICIENT
        assert "INSUFFICIENT" in result.stdout
        assert "need >= 30" in result.stdout

    def test_no_trades_file_content_is_insufficient(self, world: World) -> None:
        world.write_trades([])
        assert runner.invoke(app, world.args("--no-prediction")).exit_code == EXIT_INSUFFICIENT

    def test_the_warn_policy_turns_it_into_a_warn(self, world: World) -> None:
        world.write_trades(_trades(10))
        result = runner.invoke(app, world.args("--no-prediction", "--insufficient", "warn"))
        assert result.exit_code == EXIT_WARN

    def test_min_trades_can_be_lowered_on_the_command_line(self, world: World) -> None:
        world.write_trades(_trades(10))
        result = runner.invoke(app, world.args("--no-prediction", "--min-trades", "10"))
        assert result.exit_code == EXIT_OK, result.stdout

    def test_it_never_masks_a_measured_alert(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        world = _make_world(tmp_path, forecaster, shift=0.08, trades=_trades(10))
        for policy in ("separate", "warn"):
            result = runner.invoke(app, world.args("--no-prediction", "--insufficient", policy))
            assert result.exit_code == EXIT_ALERT, policy


# --------------------------------------------------------------------------
# Output streams
# --------------------------------------------------------------------------


class TestOutput:
    def test_problems_go_to_stdout_by_default(self, world: World) -> None:
        world.write_trades(_trades(brier=0.27))
        result = runner.invoke(app, world.args())
        assert result.stdout and result.stderr == ""

    def test_with_json_dash_stdout_is_pure_json_and_the_text_moves_to_stderr(
        self, world: World
    ) -> None:
        world.write_trades(_trades(brier=0.27))
        result = runner.invoke(app, world.args("--json", "-"))
        assert result.exit_code == EXIT_ALERT
        assert HealthReport.from_dict(json.loads(result.stdout)).status == "alert"
        assert "ALERT" in result.stderr and "calibration" in result.stderr

    def test_json_to_a_file_also_prints_the_problem(self, world: World) -> None:
        world.write_trades(_trades(brier=0.27))
        out = world.root / "h.json"
        result = runner.invoke(app, world.args("--json", str(out)))
        assert result.exit_code == EXIT_ALERT and "ALERT" in result.stdout
        assert json.loads(out.read_text())["exit_code"] == EXIT_ALERT

    def test_the_message_is_short(self, world: World) -> None:
        world.write_trades(_trades(brier=0.27, pnl=-0.02))
        result = runner.invoke(app, world.args())
        assert len(result.stdout.splitlines()) <= 6

    def test_output_is_plain_text_without_control_codes(self, world: World) -> None:
        """It lands in cron mail and log files."""
        world.write_trades(_trades(brier=0.27))
        text = runner.invoke(app, world.args()).stdout
        assert "\x1b" not in text
        assert text.isascii()

    def test_an_unwritable_json_path_is_an_error_not_a_green_run(self, world: World) -> None:
        result = runner.invoke(
            app, world.args("--json", str(world.root / "no" / "such" / "h.json"))
        )
        assert result.exit_code == EXIT_USAGE
        assert result.stderr


# --------------------------------------------------------------------------
# Inputs that are wrong must never look healthy
# --------------------------------------------------------------------------


class TestUsageErrors:
    def test_a_missing_champion(self, world: World) -> None:
        world.champion.unlink()
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_USAGE
        assert "champion not found" in result.stderr and str(world.champion) in result.stderr
        assert result.stdout == ""

    def test_a_corrupt_champion(self, world: World) -> None:
        world.champion.write_text("{not json")
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_USAGE
        assert "champion" in result.stderr

    def test_a_champion_missing_required_keys(self, world: World) -> None:
        world.champion.write_text(json.dumps({"model_name": "x"}))
        assert runner.invoke(app, world.args()).exit_code == EXIT_USAGE

    def test_a_champion_whose_gate_is_no_go_is_refused(self, world: World) -> None:
        """AGENTS.md rule 1: a NO_GO champion must not exist. Watching one is a red flag."""
        gate = json.loads(world.champion.read_text())["publish_gate"]
        gate["decision"] = "NO_GO"
        world.patch_champion(publish_gate=gate)
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_USAGE
        assert "NO_GO" in result.stderr

    def test_a_malformed_expectation_is_an_input_error_not_a_silent_skip(
        self, world: World
    ) -> None:
        world.patch_champion(backtest_expectation={"n_trades": 5})
        assert runner.invoke(app, world.args()).exit_code == EXIT_USAGE

    @pytest.mark.parametrize(
        ("broken", "named", "damage"),
        [
            # Live data is terrible on exactly the axis whose expectation is corrupt,
            # so a check that quietly reads "ok" is the only thing that can exit 0.
            ({"std_pnl_per_stake": float("nan")}, "std_pnl_per_stake", {"pnl": -1.0}),
            ({"mean_pnl_per_stake": float("inf")}, "mean_pnl_per_stake", {"pnl": -1.0}),
            ({"std_pnl_per_stake": -0.1}, "std_pnl_per_stake", {"pnl": -1.0}),
            ({"brier_std": float("nan")}, "brier_std", {"brier": 0.60}),
            ({"brier_score": float("nan")}, "brier_score", {"brier": 0.60}),
            ({"n_trades": 0}, "n_trades", {}),
        ],
    )
    def test_an_unusable_expectation_is_exit_64_never_a_silent_green(
        self,
        tmp_path: Path,
        forecaster: LGBMForecaster,
        broken: dict[str, Any],
        named: str,
        damage: dict[str, float],
    ) -> None:
        """NaN in champion.json used to disable the check it fed and exit 0 (or crash 70)."""
        world = _make_world(tmp_path, forecaster, trades=_trades(60, **damage))
        data = json.loads(world.champion.read_text())
        data["backtest_expectation"] = {**data["backtest_expectation"], **broken}
        world.champion.write_text(json.dumps(data))

        result = runner.invoke(app, world.args("--no-prediction"))

        assert result.exit_code == EXIT_USAGE, (result.exit_code, result.stdout, result.stderr)
        assert named in result.stderr
        assert result.stdout == ""

    def test_missing_trades(self, world: World) -> None:
        world.trades.unlink()
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_USAGE
        assert str(world.trades) in result.stderr

    def test_corrupt_trades(self, world: World) -> None:
        world.trades.write_text("[[[")
        assert runner.invoke(app, world.args()).exit_code == EXIT_USAGE

    @pytest.mark.parametrize(
        "payload", ['{"rows": []}', "[]", '{"trades": "no"}', '{"trades": [1]}']
    )
    def test_a_trade_log_of_the_wrong_shape(self, world: World, payload: str) -> None:
        world.trades.write_text(payload)
        assert (
            runner.invoke(app, world.args("--skip", "drift", drift=False)).exit_code == EXIT_USAGE
        )

    def test_a_bad_as_of(self, world: World) -> None:
        result = runner.invoke(app, world.args(as_of="last tuesday"))
        assert result.exit_code == EXIT_USAGE
        assert "as-of" in result.stderr or "as_of" in result.stderr

    @pytest.mark.parametrize("naive", ["2026-03-01", "2026-03-01T00:00:00"])
    def test_a_naive_as_of_is_refused_rather_than_read_as_utc(
        self, world: World, naive: str
    ) -> None:
        """A cron box in Paris must not have its midnight silently read as UTC."""
        result = runner.invoke(app, world.args("--no-prediction", as_of=naive))
        assert result.exit_code == EXIT_USAGE
        assert "--as-of" in result.stderr
        assert "UTC offset" in result.stderr and "Z" in result.stderr
        assert result.stdout == ""

    @pytest.mark.parametrize("given", ["2026-03-01T00:00:00Z", "2026-03-01T01:00:00+01:00"])
    def test_an_explicit_offset_is_accepted_and_means_the_same_instant(
        self, world: World, given: str
    ) -> None:
        out = world.root / "h.json"
        result = runner.invoke(app, world.args("--no-prediction", "--json", str(out), as_of=given))
        assert result.exit_code == EXIT_OK, result.stderr
        assert json.loads(out.read_text())["as_of"] == given

    def test_a_naive_window_start_is_refused_too(self, world: World) -> None:
        result = runner.invoke(
            app, world.args("--no-prediction", "--window-start", "2026-02-20T00:00:00")
        )
        assert result.exit_code == EXIT_USAGE
        assert "--window-start" in result.stderr and "UTC offset" in result.stderr

    def test_a_missing_reference_panel(self, world: World) -> None:
        world.reference.unlink()
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_USAGE
        assert "reference" in result.stderr

    def test_a_snapshot_directory_that_does_not_exist_is_not_created(self, world: World) -> None:
        missing = world.root / "typo"
        args = world.args(drift=False) + [
            "--reference",
            str(world.reference),
            "--snapshots",
            str(missing),
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
        assert not missing.exists(), "a typo must not create an empty store"

    def test_a_reference_without_market_ids(self, world: World) -> None:
        pd.read_parquet(world.reference).drop(columns="market_id").to_parquet(world.reference)
        result = runner.invoke(app, world.args("--no-prediction"))
        assert result.exit_code == EXIT_USAGE
        assert "market_id" in result.stderr

    def test_the_champion_model_missing_names_the_escape_hatch(self, world: World) -> None:
        (world.champion.parent / "champion.pkl").unlink()
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_USAGE
        assert "--no-prediction" in result.stderr

    def test_model_features_absent_from_the_reference(self, world: World) -> None:
        pd.read_parquet(world.reference).drop(columns="feature_volume").to_parquet(world.reference)
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_USAGE
        assert "feature_volume" in result.stderr

    @pytest.mark.parametrize(
        "args",
        [
            ["--insufficient", "ignore"],
            ["--skip", "everything"],
            ["--skip", "drift", "--skip", "calibration", "--skip", "pnl"],
            ["--min-trades", "1"],
            ["--window-days", "0"],
        ],
    )
    def test_bad_option_values(self, world: World, args: list[str]) -> None:
        assert runner.invoke(app, world.args(*args)).exit_code == EXIT_USAGE

    @pytest.mark.parametrize(
        "content",
        [
            "[pnl]\nmin_trade = 5\n",
            "[mystery]\nx = 1\n",
            "this is = = not toml",
            "[pnl]\nz_warn = -1\n",
        ],
    )
    def test_a_bad_config_file(self, world: World, tmp_path: Path, content: str) -> None:
        cfg = tmp_path / "wd.toml"
        cfg.write_text(content)
        result = runner.invoke(app, world.args("--config", str(cfg)))
        assert result.exit_code == EXIT_USAGE
        assert result.stderr

    def test_a_missing_config_file(self, world: World, tmp_path: Path) -> None:
        assert runner.invoke(app, world.args("--config", str(tmp_path / "no.toml"))).exit_code == (
            EXIT_USAGE
        )

    def test_a_json_config_file_is_accepted(self, world: World, tmp_path: Path) -> None:
        cfg = tmp_path / "wd.json"
        cfg.write_text(json.dumps({"calibration": {"min_trades": 5}}))
        world.write_trades(_trades(10))
        result = runner.invoke(
            app, world.args("--no-prediction", "--config", str(cfg), "--min-trades", "10")
        )
        assert result.exit_code == EXIT_OK, result.stdout

    def test_a_config_file_of_an_unsupported_type(self, world: World, tmp_path: Path) -> None:
        cfg = tmp_path / "wd.yaml"
        cfg.write_text("pnl: {}")
        result = runner.invoke(app, world.args("--config", str(cfg)))
        assert result.exit_code == EXIT_USAGE and ".toml or .json" in result.stderr

    def test_a_json_config_that_is_not_an_object(self, world: World, tmp_path: Path) -> None:
        cfg = tmp_path / "wd.json"
        cfg.write_text("[1, 2]")
        assert runner.invoke(app, world.args("--config", str(cfg))).exit_code == EXIT_USAGE

    def test_a_csv_reference_panel_works(self, world: World) -> None:
        csv = world.root / "reference.csv"
        pd.read_parquet(world.reference).to_csv(csv, index=False)
        result = runner.invoke(app, world.args("--no-prediction") + ["--reference", str(csv)])
        assert result.exit_code == EXIT_OK, result.stderr

    def test_an_unsupported_reference_format(self, world: World) -> None:
        odd = world.root / "reference.xlsx"
        odd.write_text("x")
        result = runner.invoke(app, world.args("--no-prediction") + ["--reference", str(odd)])
        assert result.exit_code == EXIT_USAGE and ".parquet or .csv" in result.stderr

    def test_a_corrupt_reference_panel(self, world: World) -> None:
        world.reference.write_text("not parquet")
        assert runner.invoke(app, world.args("--no-prediction")).exit_code == EXIT_USAGE

    def test_a_corrupt_champion_model(self, world: World) -> None:
        (world.champion.parent / "champion.pkl").write_bytes(b"garbage")
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_USAGE
        assert "--no-prediction" in result.stderr

    def test_the_model_is_found_next_to_champion_json_when_the_recorded_path_is_stale(
        self, world: World
    ) -> None:
        """champion.json stores the publisher's absolute path; a moved directory must still work."""
        world.patch_champion(model_path="/nonexistent/elsewhere/champion.pkl")
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_OK, result.stderr

    def test_live_snapshots_missing_a_model_feature(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        world = _make_world(tmp_path, forecaster)
        store = FeatureSnapshotStore(tmp_path / "bare")
        store.append(
            _FAMILY,
            [
                {"market_id": f"x{i}", "captured_at": "2026-02-25T12:00:00Z", "feature_price": 0.5}
                for i in range(400)
            ],
        )
        args = world.args(drift=False) + [
            "--reference",
            str(world.reference),
            "--snapshots",
            str(tmp_path / "bare"),
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == EXIT_USAGE
        assert "feature_volume" in result.stderr

    def test_a_bad_window_start(self, world: World) -> None:
        result = runner.invoke(app, world.args("--window-start", "sometime"))
        assert result.exit_code == EXIT_USAGE and "--window-start" in result.stderr

    def test_usage_errors_say_what_is_wrong_in_one_line(self, world: World) -> None:
        world.champion.unlink()
        assert len(runner.invoke(app, world.args()).stderr.strip().splitlines()) == 1


class TestCrashesAreNotHealthy:
    def test_an_unexpected_exception_exits_70_not_0_and_not_pythons_1(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**_: Any) -> None:
            raise RuntimeError("kaboom")

        monkeypatch.setattr("pmlab.cli.monitor.health_report", boom)
        result = runner.invoke(app, world.args())
        assert result.exit_code == EXIT_INTERNAL
        assert "kaboom" in result.stderr
        assert result.stdout == ""

    def test_an_internal_error_is_not_confused_with_an_input_error(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pmlab.cli.monitor.health_report", lambda **_: 1 / 0)
        assert runner.invoke(app, world.args()).exit_code == EXIT_INTERNAL

    def test_click_usage_errors_keep_clicks_code_which_no_verdict_uses(self) -> None:
        result = runner.invoke(app, ["monitor", "--no-such-flag"])
        assert result.exit_code == 2
        assert result.exit_code not in (EXIT_OK, EXIT_WARN, EXIT_ALERT, EXIT_INSUFFICIENT)


# --------------------------------------------------------------------------
# Where the files come from
# --------------------------------------------------------------------------


class TestPathResolution:
    def test_paths_resolve_from_the_artifacts_dir_and_workspace_like_status(
        self, tmp_path: Path, forecaster: LGBMForecaster
    ) -> None:
        artifacts = tmp_path / "artifacts"
        world = _make_world(tmp_path / "w", forecaster)
        (artifacts / "public_models").mkdir(parents=True)
        for f in world.champion.parent.iterdir():
            (artifacts / "public_models" / f.name).write_bytes(f.read_bytes())
        trades = artifacts / "workspaces" / "paper" / "signals" / "v2" / "forward_paper_trades.json"
        trades.parent.mkdir(parents=True)
        trades.write_bytes(world.trades.read_bytes())

        result = runner.invoke(
            app,
            [
                "monitor",
                "--artifacts-dir",
                str(artifacts),
                "--workspace",
                "paper",
                "--skip",
                "drift",
                "--as-of",
                _AS_OF,
            ],
        )
        assert result.exit_code == EXIT_OK, result.stderr
        assert result.stdout == ""

    def test_the_environment_is_honoured(
        self, tmp_path: Path, forecaster: LGBMForecaster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifacts = tmp_path / "artifacts"
        world = _make_world(tmp_path / "w", forecaster)
        (artifacts / "public_models").mkdir(parents=True)
        for f in world.champion.parent.iterdir():
            (artifacts / "public_models" / f.name).write_bytes(f.read_bytes())
        trades = artifacts / "workspaces" / "paper" / "signals" / "v2" / "forward_paper_trades.json"
        trades.parent.mkdir(parents=True)
        trades.write_bytes(world.trades.read_bytes())
        monkeypatch.setenv("PMLAB_ARTIFACTS_DIR", str(artifacts))
        monkeypatch.setenv("PMLAB_WORKSPACE", "paper")

        result = runner.invoke(app, ["monitor", "--skip", "drift", "--as-of", _AS_OF])
        assert result.exit_code == EXIT_OK, result.stderr

    def test_a_default_run_with_nothing_there_is_a_usage_error_not_a_pass(
        self, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["monitor", "--artifacts-dir", str(tmp_path / "none")])
        assert result.exit_code == EXIT_USAGE
        assert "champion" in result.stderr

    def test_the_default_as_of_is_now(self, world: World) -> None:
        """Without --as-of the report is stamped with the current UTC time."""
        out = world.root / "h.json"
        runner.invoke(app, world.args("--json", str(out), as_of=None))
        stamped = json.loads(out.read_text())["as_of"]
        assert stamped.endswith("+00:00")
        assert stamped[:4] >= "2026"


# --------------------------------------------------------------------------
# The existing commands still work
# --------------------------------------------------------------------------


def test_monitor_is_listed_in_the_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert "monitor" in result.stdout


def test_promote_champion_can_record_an_expectation(
    tmp_path: Path, forecaster: LGBMForecaster
) -> None:
    model_path = tmp_path / "model.pkl"
    forecaster.save(model_path)
    gate = HoldoutGateResult(
        "GO",
        [SegmentGateResult("A", 100, 10.0, True, "ok")],
        aggregate_pnl=10.0,
        aggregate_trades=100,
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate.to_dict()))
    expectation_path = tmp_path / "exp.json"
    expectation_path.write_text(json.dumps({"expectation": _expectation().to_dict()}))

    result = runner.invoke(
        app,
        [
            "promote-champion",
            str(model_path),
            "--gate-path",
            str(gate_path),
            "--plugin",
            _FAMILY,
            "--output-dir",
            str(tmp_path / "out"),
            "--expectation",
            str(expectation_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert ChampionManifest.load(tmp_path / "out" / "champion.json").expectation == _expectation()


def test_promote_champion_without_an_expectation_still_works(
    tmp_path: Path, forecaster: LGBMForecaster
) -> None:
    model_path = tmp_path / "model.pkl"
    forecaster.save(model_path)
    gate = HoldoutGateResult(
        "GO",
        [SegmentGateResult("A", 100, 10.0, True, "ok")],
        aggregate_pnl=10.0,
        aggregate_trades=100,
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate.to_dict()))
    result = runner.invoke(
        app,
        [
            "promote-champion",
            str(model_path),
            "--gate-path",
            str(gate_path),
            "--plugin",
            _FAMILY,
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0
    assert ChampionManifest.load(tmp_path / "out" / "champion.json").expectation is None


def test_promote_champion_rejects_a_bad_expectation_file(
    tmp_path: Path, forecaster: LGBMForecaster
) -> None:
    model_path = tmp_path / "model.pkl"
    forecaster.save(model_path)
    gate = HoldoutGateResult(
        "GO",
        [SegmentGateResult("A", 100, 10.0, True, "ok")],
        aggregate_pnl=10.0,
        aggregate_trades=100,
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate.to_dict()))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"expectation": {"n_trades": 3}}))
    result = runner.invoke(
        app,
        [
            "promote-champion",
            str(model_path),
            "--gate-path",
            str(gate_path),
            "--plugin",
            _FAMILY,
            "--output-dir",
            str(tmp_path / "out"),
            "--expectation",
            str(bad),
        ],
    )
    assert result.exit_code == 1
    assert not (tmp_path / "out" / "champion.json").exists()


def test_backtest_output_includes_an_expectation_ready_for_promotion(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for d in range(60):
        winner = rng.choice(["YES", "NO"])
        for label in ("YES", "NO"):
            rows.append(
                {
                    "market_id": f"m{d}",
                    "decision_date": f"2026-01-{(d % 28) + 1:02d}"
                    if d < 28
                    else f"2026-02-{(d % 28) + 1:02d}",
                    "outcome_label": label,
                    "winning_label": winner,
                    "market_price": 0.45,
                    "feature_x": float(rng.normal()),
                    "segment": "A",
                }
            )
    panel = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel, index=False)
    out = tmp_path / "metrics.json"
    result = runner.invoke(
        app,
        [
            "backtest",
            "--plugin",
            "x",
            "--panel",
            str(panel),
            "--stride",
            "10",
            "--min-train-rows",
            "10",
            "--flat-stake",
            "2.0",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    metrics = json.loads(out.read_text())
    exp = BacktestExpectation.from_dict(metrics["expectation"])
    assert exp.flat_stake == 2.0
    assert exp.n_trades == metrics["total_trades"]
