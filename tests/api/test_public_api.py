"""The public API is frozen: a snapshot, checked in, that a change must update on purpose.

What is public is defined in ``docs/api-stability.md``: the names in ``pmlab.__all__`` and in the
``__all__`` of the documented subpackages, the ``pmlab`` command line (commands, options), and the
file formats it writes. ``scripts/api_surface.py`` reads that surface off the installed package;
``tests/api/public_api.json`` is what it read when the surface was last changed on purpose.

To change the surface deliberately::

    uv run python scripts/api_surface.py --write

then commit the snapshot with the change and a CHANGELOG entry (a removal or a changed signature
is a breaking change: see the deprecation policy).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import pmlab

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "api_surface.py"
SNAPSHOT = REPO / "tests" / "api" / "public_api.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("api_surface", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["api_surface"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def api_surface() -> ModuleType:
    return _load_script()


@pytest.fixture(scope="module")
def current(api_surface: ModuleType) -> dict[str, Any]:
    surface: dict[str, Any] = api_surface.surface()
    return surface


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SNAPSHOT.read_text())
    return data


class TestSnapshotGuardsTheSurface:
    def test_the_surface_matches_the_checked_in_snapshot(
        self, api_surface: ModuleType, current: dict[str, Any], snapshot: dict[str, Any]
    ) -> None:
        changes = api_surface.diff(snapshot, current)
        assert not changes, (
            "the public API differs from tests/api/public_api.json:\n  "
            + "\n  ".join(changes)
            + "\nIf this is deliberate, run `uv run python scripts/api_surface.py --write`, "
            "commit the snapshot with a CHANGELOG entry, and follow docs/deprecation.md for "
            "anything removed or changed."
        )

    def test_the_snapshot_is_what_write_would_produce(
        self, api_surface: ModuleType, current: dict[str, Any]
    ) -> None:
        # Byte for byte: the file is generated, never edited by hand, and its diff in a PR
        # shows exactly which names changed.
        assert SNAPSHOT.read_text() == api_surface.render(current)

    def test_diff_reports_a_removed_a_changed_and_an_added_name(
        self, api_surface: ModuleType
    ) -> None:
        before = {"m": {"gone": {"kind": "function", "signature": "()"}, "same": {"k": 1}}}
        after = {"m": {"same": {"k": 2}, "new": {"kind": "function", "signature": "(x)"}}}
        changes = api_surface.diff(before, after)
        assert any("removed" in c and "m.gone" in c for c in changes)
        assert any("changed" in c and "m.same.k" in c and "1" in c and "2" in c for c in changes)
        assert any("added" in c and "m.new" in c for c in changes)
        assert api_surface.diff(before, before) == []

    def test_diff_names_a_list_entry_by_its_name_not_by_dumping_the_list(
        self, api_surface: ModuleType
    ) -> None:
        before = {"params": [{"name": "a", "type": "int"}, {"name": "b", "type": "str"}]}
        after = {"params": [{"name": "a", "type": "float"}, {"name": "b", "type": "str"}]}
        assert api_surface.diff(before, after) == ["changed: params[a].type: 'int' -> 'float'"]
        gone = {"params": [{"name": "b", "type": "str"}]}
        assert api_surface.diff(before, gone) == ["removed: params[a]"]
        assert api_surface.diff(gone, before) == ["added:   params[a]"]
        # Positional arguments depend on order: swapping two is a change.
        swapped = {"params": [before["params"][1], before["params"][0]]}
        assert any("order" in line for line in api_surface.diff(before, swapped))

    def test_check_exits_non_zero_when_the_snapshot_is_stale(
        self, api_surface: ModuleType, tmp_path: Path, snapshot: dict[str, Any]
    ) -> None:
        stale = json.loads(json.dumps(snapshot))
        stale["modules"]["pmlab"].pop("kelly_fraction")
        path = tmp_path / "stale.json"
        path.write_text(api_surface.render(stale))
        assert api_surface.main(["--check", "--snapshot", str(path)]) == 1
        assert api_surface.main(["--check", "--snapshot", str(SNAPSHOT)]) == 0

    def test_write_then_check_round_trips(self, api_surface: ModuleType, tmp_path: Path) -> None:
        path = tmp_path / "fresh.json"
        assert api_surface.main(["--write", "--snapshot", str(path)]) == 0
        assert api_surface.main(["--check", "--snapshot", str(path)]) == 0
        assert path.read_text() == SNAPSHOT.read_text()


class TestWhatIsPublic:
    def test_the_documented_modules_are_the_top_level_and_the_subpackages_with_an_all(
        self, api_surface: ModuleType
    ) -> None:
        assert api_surface.PUBLIC_MODULES == (
            "pmlab",
            "pmlab.data",
            "pmlab.datasets",
            "pmlab.features",
            "pmlab.monitoring",
            "pmlab.reports",
            "pmlab.reproducibility",
            "pmlab.tracking",
        )

    def test_every_subpackage_that_declares_an_all_is_in_the_list(
        self, api_surface: ModuleType
    ) -> None:
        """A new subpackage with an ``__all__`` is a public module the snapshot must cover."""
        declared = []
        for init in sorted((REPO / "src" / "pmlab").glob("*/__init__.py")):
            if "__all__" in init.read_text():
                declared.append(f"pmlab.{init.parent.name}")
        assert sorted(declared) == sorted(set(api_surface.PUBLIC_MODULES) - {"pmlab"})

    @pytest.mark.parametrize("name", sorted(set(pmlab.__all__)))
    def test_every_name_in_pmlab_all_resolves(self, name: str) -> None:
        assert hasattr(pmlab, name), f"pmlab.__all__ lists {name!r} but pmlab has no such name"

    def test_star_import_works_for_every_public_module(self, api_surface: ModuleType) -> None:
        for module_name in api_surface.PUBLIC_MODULES:
            scope: dict[str, Any] = {}
            exec(f"from {module_name} import *", scope)  # noqa: S102 - our own module names
            module = importlib.import_module(module_name)
            assert set(module.__all__) <= set(scope), module_name

    def test_no_public_module_lists_a_name_twice_or_a_private_one(
        self, api_surface: ModuleType
    ) -> None:
        for module_name in api_surface.PUBLIC_MODULES:
            names = importlib.import_module(module_name).__all__
            assert len(names) == len(set(names)), f"{module_name}.__all__ has duplicates"
            private = [n for n in names if n.startswith("_") and n != "__version__"]
            assert not private, f"{module_name}.__all__ exports private names {private}"

    def test_the_snapshot_covers_exactly_the_names_in_each_all(
        self, snapshot: dict[str, Any]
    ) -> None:
        for module_name, entries in snapshot["modules"].items():
            module = importlib.import_module(module_name)
            assert sorted(entries) == sorted(module.__all__), module_name

    def test_nothing_internal_leaks_into_the_snapshot(self, snapshot: dict[str, Any]) -> None:
        """No underscore module path and no memory address: the snapshot must be reproducible."""
        text = SNAPSHOT.read_text()
        assert not re.search(r"0x[0-9a-fA-F]{6,}", text)
        assert "pmlab._" not in text
        assert "/home/" not in text and "/tmp/" not in text


class TestSignaturesAreCaptured:
    def test_a_function_keeps_names_defaults_and_kinds(self, snapshot: dict[str, Any]) -> None:
        entry = snapshot["modules"]["pmlab"]["kelly_stake_size"]
        assert entry["kind"] == "function"
        signature = entry["signature"]
        assert signature.startswith("(") and "->" in signature

    def test_a_class_records_its_init_methods_and_abstract_methods(
        self, snapshot: dict[str, Any]
    ) -> None:
        plugin = snapshot["modules"]["pmlab"]["MarketPlugin"]
        assert plugin["kind"] == "class"
        # An abstract method added to a base class breaks every subclass: it is recorded.
        for name in ("discover_markets", "fetch_features", "fetch_truth", "build_training_row"):
            assert plugin["members"][name]["abstract"] is True
        assert plugin["members"]["build_training_rows"].get("abstract") is not True

    def test_private_parameters_are_not_part_of_a_signature(self, snapshot: dict[str, Any]) -> None:
        # PmlabSettings inherits dozens of `_env_*` keyword arguments from pydantic-settings:
        # not pmlab's API, and they change with that library's version.
        settings = snapshot["modules"]["pmlab"]["PmlabSettings"]["signature"]
        assert "_env_file" not in settings
        assert "artifacts_dir" in settings

    def test_constants_carry_their_value_but_the_version_does_not(
        self, snapshot: dict[str, Any]
    ) -> None:
        assert snapshot["modules"]["pmlab"]["DEFAULT_MIN_EDGE"]["kind"] == "constant"
        assert "value" in snapshot["modules"]["pmlab"]["DEFAULT_MIN_EDGE"]
        version = snapshot["modules"]["pmlab"]["__version__"]
        assert version == {"kind": "constant", "type": "str"}  # changes every release

    def test_deprecated_names_keep_their_original_signature(self, snapshot: dict[str, Any]) -> None:
        plugin = snapshot["modules"]["pmlab"]["WeatherTmaxPlugin"]
        assert "spec" in plugin["members"]["build_training_row"]["signature"]


class TestTheCommandLine:
    def test_every_command_and_option_is_recorded(self, snapshot: dict[str, Any]) -> None:
        cli = snapshot["cli"]
        assert cli["name"] == "pmlab"
        assert set(cli["commands"]) >= {
            "version",
            "status",
            "scan-markets",
            "record-trades",
            "resettle",
            "settle-trades",
            "backtest",
            "promote-champion",
            "reproduce",
            "monitor",
            "report",
        }
        backtest = {p["name"]: p for p in cli["commands"]["backtest"]["params"]}
        assert backtest["plugin"]["flags"] == ["--plugin", "-p"]
        assert backtest["plugin"]["required"] is True
        assert backtest["stride"]["default"] == "30"
        # Type names come from the click class, not from click's own label for it: that label
        # (`text`, `integer`) changed between click releases, and a user's fresh install must
        # not see the API "change" because a dependency was upgraded.
        assert backtest["stride"]["type"] == "int"
        assert backtest["plugin"]["type"] == "str"
        assert backtest["flat_stake"]["type"] == "float"
        types = {p["type"] for c in cli["commands"].values() for p in c["params"]}
        assert {"str", "int", "float", "bool"} <= types
        assert not types & {"text", "integer", "boolean", "integer range", "float range"}

    def test_environment_variables_that_configure_the_cli_are_recorded(
        self, snapshot: dict[str, Any]
    ) -> None:
        status = {p["name"]: p for p in snapshot["cli"]["commands"]["status"]["params"]}
        assert status["workspace"]["envvar"] == ["PMLAB_WORKSPACE", "PMLAB_WORKSPACE_NAME"]
        assert "PMLAB_ARTIFACTS_DIR" in snapshot["formats"]["environment_variables"]

    def test_typers_own_completion_options_are_not_pmlabs_surface(
        self, snapshot: dict[str, Any]
    ) -> None:
        assert "install_completion" not in json.dumps(snapshot["cli"])


class TestFileFormats:
    def test_champion_json_keys_are_frozen(self, snapshot: dict[str, Any]) -> None:
        keys = snapshot["formats"]["champion.json"]
        assert set(keys) == {
            "model_name",
            "model_path",
            "calibrator_path",
            "published_at",
            "plugin_family",
            "publish_gate",
            "costs",
            "backtest_expectation",
            "reproducibility",
        }
        assert set(keys["publish_gate"]) == {
            "decision",
            "segment_results",
            "aggregate_pnl",
            "aggregate_trades",
        }

    def test_the_written_champion_json_has_the_snapshotted_keys(
        self, tmp_path: Path, snapshot: dict[str, Any]
    ) -> None:
        """The snapshot is derived from ``to_dict``; the file ``publish`` writes must agree."""
        from pmlab import ChampionManifest, CostModel, HoldoutGateResult, SegmentGateResult

        class _StubModel:
            def save(self, path: Path) -> None:
                Path(path).write_bytes(b"stub")

        gate = HoldoutGateResult(
            decision="GO",
            segment_results=[SegmentGateResult("seg", 10, 1.0, True, "ok")],
            aggregate_pnl=1.0,
            aggregate_trades=10,
        )
        ChampionManifest.publish(
            _StubModel(),  # type: ignore[arg-type]
            gate,
            tmp_path,
            plugin_family="weather_tmax",
            costs=CostModel(),
        )
        on_disk = json.loads((tmp_path / "champion.json").read_text())
        assert set(on_disk) == set(snapshot["formats"]["champion.json"])
        assert set(on_disk["publish_gate"]) == set(
            snapshot["formats"]["champion.json"]["publish_gate"]
        )

    def test_the_run_record_and_the_reproducibility_record_are_frozen(
        self, snapshot: dict[str, Any]
    ) -> None:
        formats = snapshot["formats"]
        assert set(formats["run_record.json"]) == {
            "run_id",
            "experiment",
            "run_name",
            "params",
            "metrics",
            "gate",
            "reproducibility",
            "status",
            "started_at",
            "ended_at",
            "error",
        }
        repro = formats["reproducibility_record"]
        assert repro["schema_version"] == 1  # the version is part of the contract
        assert set(repro["keys"]) == {
            "schema_version",
            "kind",
            "created_at",
            "seeds",
            "environment",
            "config",
            "data",
            "metrics",
        }

    def test_the_plugin_entry_point_group_is_frozen(self, snapshot: dict[str, Any]) -> None:
        assert snapshot["formats"]["plugin_entry_point_group"] == "pmlab.plugins"
