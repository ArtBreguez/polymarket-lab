"""A champion survives the registry intact: publish -> record -> rollback loses nothing (#33).

`ModelRegistry.record()` dropped `backtest_expectation` and `rollback()` dropped `costs` and
`backtest_expectation`, so a rolled-back champion silently lost the promise the watchdog holds
live trading against. The tests are generic: they walk the manifest's fields and the keys of the
file on disk, so a field added later is covered without editing them.
"""

from __future__ import annotations

import json
from pathlib import Path

from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.registry import ModelRegistry
from tests.modeling.manifest_helpers import (
    assert_every_field_set,
    assert_manifest_survived,
    assert_same_champion_json,
    full_manifest,
)


def test_the_fixture_sets_every_manifest_field(tmp_path: Path) -> None:
    assert_every_field_set(full_manifest(tmp_path / "live"))


def test_record_archives_every_field(tmp_path: Path) -> None:
    original = full_manifest(tmp_path / "live")
    registry = ModelRegistry(tmp_path / "registry")

    version = registry.record(original)

    assert_manifest_survived(original, registry.get(version))
    assert_same_champion_json(
        tmp_path / "live" / "champion.json", tmp_path / "registry" / version / "champion.json"
    )


def test_rollback_restores_every_field(tmp_path: Path) -> None:
    original = full_manifest(tmp_path / "live")
    registry = ModelRegistry(tmp_path / "registry")
    version = registry.record(original)

    restored = registry.rollback(version, tmp_path / "restored")

    assert_manifest_survived(original, restored)
    assert_manifest_survived(
        original, ChampionManifest.load(tmp_path / "restored" / "champion.json")
    )
    assert_same_champion_json(
        tmp_path / "live" / "champion.json", tmp_path / "restored" / "champion.json"
    )


def test_a_rolled_back_champion_can_be_recorded_and_rolled_back_again(tmp_path: Path) -> None:
    original = full_manifest(tmp_path / "live")
    registry = ModelRegistry(tmp_path / "registry")
    first = registry.rollback(registry.record(original), tmp_path / "once")

    second = registry.rollback(registry.record(first), tmp_path / "twice")

    assert_manifest_survived(original, second)


def test_registry_entries_written_by_older_versions_still_load_and_roll_back(
    tmp_path: Path,
) -> None:
    """0.10.0 archived `costs` and `reproducibility` but not `backtest_expectation`; earlier
    versions archived none of the three. Those entries carry what they have and no more."""
    original = full_manifest(tmp_path / "live")
    registry = ModelRegistry(tmp_path / "registry")
    version = registry.record(original)
    archived = tmp_path / "registry" / version / "champion.json"
    full = json.loads(archived.read_text())

    for dropped in (
        {"backtest_expectation"},  # what 0.10.0 wrote
        {"backtest_expectation", "reproducibility"},  # 0.9.x
        {"backtest_expectation", "reproducibility", "costs"},  # before costs were archived
    ):
        archived.write_text(json.dumps({k: v for k, v in full.items() if k not in dropped}))
        manifest = registry.get(version)
        assert manifest.expectation is None and manifest.model_name == original.model_name
        assert (manifest.costs is None) == ("costs" in dropped)
        assert (manifest.reproducibility is None) == ("reproducibility" in dropped)

        restored = registry.rollback(version, tmp_path / f"restored_{len(dropped)}")
        assert restored.expectation is None
        assert restored.gate == original.gate and restored.published_at == original.published_at
        assert (restored.costs is None) == ("costs" in dropped)
        assert (restored.reproducibility is None) == ("reproducibility" in dropped)
