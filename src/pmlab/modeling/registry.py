"""Versioned champion registry.

`ChampionManifest` publishes a single `champion.json` that each retrain
overwrites — so the history of what was promoted, and the ability to roll back,
is lost. `ModelRegistry` archives every promotion as an immutable, versioned
record (model + calibrator + manifest) and adds list / diff / rollback on top,
without changing the existing publish flow.

Design:
  registry_dir/
    index.json                      # ordered list of version metadata
    <version_id>/
      champion.json                 # the archived manifest
      champion.pkl                  # the archived model
      calibrator.pkl                # optional, when the champion had one

`version_id` is derived from the manifest's `published_at` (ISO-8601, sortable)
plus the model name, so it is stable, human-readable, and collision-safe within
a run. The hard gate is re-asserted on `record()`: a NO_GO can never enter the
registry even if a caller hands over a tampered manifest.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pmlab.modeling.champion import ChampionManifest

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


class _IndexEntry(TypedDict):
    """One row of the registry index.json (typed for mypy --strict)."""

    version_id: str
    model_name: str
    plugin_family: str
    published_at: str
    gate_decision: str
    aggregate_pnl: float
    aggregate_trades: int


def _slug(text: str) -> str:
    """Filesystem-safe slug (also blocks path traversal via separators/..)."""
    cleaned = _SLUG_RE.sub("-", text).strip("-")
    return cleaned or "unnamed"


@dataclass(frozen=True)
class RegistryEntry:
    """Lightweight metadata for one archived champion (what `list()` returns)."""

    version_id: str
    model_name: str
    plugin_family: str
    published_at: str
    gate_decision: str
    aggregate_pnl: float
    aggregate_trades: int


class ModelRegistry:
    """Append-only, versioned store of promoted champions."""

    def __init__(self, registry_dir: Path | str) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "index.json"

    # ── internals ────────────────────────────────────────────────────────
    def _read_index(self) -> list[_IndexEntry]:
        if not self._index_path.exists():
            return []
        with open(self._index_path) as f:
            data: list[_IndexEntry] = json.load(f)
            return data

    def _write_index(self, entries: list[_IndexEntry]) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=2)
        tmp.replace(self._index_path)  # atomic

    @staticmethod
    def _version_id(manifest: ChampionManifest) -> str:
        # published_at is ISO-8601 → lexicographically sortable. Slug both parts
        # so the id is a safe single path segment.
        return f"{_slug(manifest.published_at)}__{_slug(manifest.model_name)}"

    # ── public API ───────────────────────────────────────────────────────
    def record(self, manifest: ChampionManifest) -> str:
        """Archive a published champion as an immutable version.

        Re-asserts the hard gate: refuses to record anything whose gate is not
        GO, independent of whatever the caller claims. Returns the version_id.
        """
        if manifest.gate.decision != "GO":
            raise ValueError(
                f"Cannot record a champion with NO_GO gate. Gate decision: {manifest.gate.decision}"
            )

        version_id = self._version_id(manifest)
        vdir = self.registry_dir / version_id
        vdir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts into the version dir so the archive is self-contained
        # even if the working output_dir is later overwritten by a new publish.
        archived_model = vdir / "champion.pkl"
        shutil.copy2(manifest.model_path, archived_model)

        archived_calibrator: Path | None = None
        if manifest.calibrator_path is not None:
            archived_calibrator = vdir / "calibrator.pkl"
            shutil.copy2(manifest.calibrator_path, archived_calibrator)

        json_data = {
            "model_name": manifest.model_name,
            "model_path": str(archived_model),
            "calibrator_path": str(archived_calibrator) if archived_calibrator else None,
            "published_at": manifest.published_at,
            "plugin_family": manifest.plugin_family,
            "publish_gate": manifest.gate.to_dict(),
        }
        with open(vdir / "champion.json", "w") as f:
            json.dump(json_data, f, indent=2)

        entry: _IndexEntry = {
            "version_id": version_id,
            "model_name": manifest.model_name,
            "plugin_family": manifest.plugin_family,
            "published_at": manifest.published_at,
            "gate_decision": manifest.gate.decision,
            "aggregate_pnl": manifest.gate.aggregate_pnl,
            "aggregate_trades": manifest.gate.aggregate_trades,
        }
        index = [e for e in self._read_index() if e["version_id"] != version_id]
        index.append(entry)
        self._write_index(index)
        return version_id

    def list(self) -> list[RegistryEntry]:
        """All archived versions, newest first (by published_at, then id)."""
        entries = self._read_index()
        entries.sort(key=lambda e: (e["published_at"], e["version_id"]), reverse=True)
        return [
            RegistryEntry(
                version_id=e["version_id"],
                model_name=e["model_name"],
                plugin_family=e["plugin_family"],
                published_at=e["published_at"],
                gate_decision=e["gate_decision"],
                aggregate_pnl=e["aggregate_pnl"],
                aggregate_trades=e["aggregate_trades"],
            )
            for e in entries
        ]

    def get(self, version_id: str) -> ChampionManifest:
        """Load the archived manifest for a version_id."""
        json_path = self.registry_dir / version_id / "champion.json"
        if not json_path.exists():
            raise KeyError(f"Unknown registry version: {version_id!r}")
        return ChampionManifest.load(json_path)

    def diff(self, version_a: str, version_b: str) -> dict[str, object]:
        """Compare two versions field by field.

        Each differing field maps to a (a_value, b_value) tuple; the gate
        summary (decision, pnl, trades, allowed segments) is always included so
        a reviewer sees the promotion-time evidence side by side.
        """
        a = self.get(version_a)
        b = self.get(version_b)
        return {
            "model_name": (a.model_name, b.model_name),
            "plugin_family": (a.plugin_family, b.plugin_family),
            "published_at": (a.published_at, b.published_at),
            "gate_decision": (a.gate.decision, b.gate.decision),
            "aggregate_pnl": (a.gate.aggregate_pnl, b.gate.aggregate_pnl),
            "aggregate_trades": (a.gate.aggregate_trades, b.gate.aggregate_trades),
            "allowed_segments": (
                sorted(a.get_allowed_segments()),
                sorted(b.get_allowed_segments()),
            ),
        }

    def rollback(self, version_id: str, output_dir: Path | str) -> ChampionManifest:
        """Restore an archived version as the live champion in output_dir.

        Rewrites champion.json / champion.pkl (and calibrator.pkl if present) in
        output_dir to point at the archived artifacts, so the standard
        ChampionManifest.load(output_dir / "champion.json") flow serves the
        rolled-back model. Does not delete newer versions — history is preserved.
        """
        archived = self.get(version_id)  # raises KeyError if unknown
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        live_model = output_dir / "champion.pkl"
        shutil.copy2(archived.model_path, live_model)

        live_calibrator: Path | None = None
        if archived.calibrator_path is not None:
            live_calibrator = output_dir / "calibrator.pkl"
            shutil.copy2(archived.calibrator_path, live_calibrator)

        json_data = {
            "model_name": archived.model_name,
            "model_path": str(live_model),
            "calibrator_path": str(live_calibrator) if live_calibrator else None,
            "published_at": archived.published_at,
            "plugin_family": archived.plugin_family,
            "publish_gate": archived.gate.to_dict(),
        }
        with open(output_dir / "champion.json", "w") as f:
            json.dump(json_data, f, indent=2)

        return ChampionManifest.load(output_dir / "champion.json")
