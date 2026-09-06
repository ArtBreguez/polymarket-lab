"""Experiment tracking (v0.8.0).

Every backtest run can log its params, metrics, and gate decision so results are
auditable and reproducible — the roadmap's "no result you can't re-run" principle.

Two backends share one :class:`ExperimentTracker` protocol:

* :class:`LocalJSONTracker` — the zero-dependency default. Writes one append-only
  JSON record per run under ``<root>/<experiment>/<run_id>.json``. Point-in-time
  and immutable: a new run never overwrites an old one (unique ``run_id``), so the
  run history is stable evidence, mirroring the FeatureSnapshotStore philosophy.
* ``MLflowTracker`` — optional, behind the ``track`` extra. Imported lazily so the
  core install never pulls MLflow. See :mod:`pmlab.tracking.mlflow_tracker`.

Metrics/gate objects log directly: ``log_metrics`` accepts a plain dict or a
:class:`~pmlab.backtest.metrics.BacktestMetrics`; ``log_gate`` accepts a
:class:`~pmlab.backtest.holdout_gate.HoldoutGateResult`. No manual glue.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ExperimentTracker",
    "ActiveRun",
    "LocalJSONTracker",
    "RunRecord",
]


def _json_safe(obj: Any) -> Any:
    """Recursively coerce a value into something ``json.dump`` accepts *and* that
    round-trips as spec-valid JSON.

    Handles the value types that actually flow out of numpy/pandas in a backtest:
    numpy scalars (``np.int64``, ``np.float64`` — the latter is a ``float`` but
    ``np.int64`` is NOT an ``int`` subclass, so ``json`` rejects it), numpy arrays,
    and non-finite floats (``NaN``/``Inf`` from empty-backtest divisions), which
    are encoded as ``None`` so the file stays valid JSON for any parser.
    """
    # Order matters: bool before int (bool is an int subclass); check finite floats.
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    # numpy scalars expose .item(); this normalizes np.int64 -> int, np.float64 -> float.
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "ndim", None) == 0:
        return _json_safe(item())
    # numpy arrays / other sequences with tolist().
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj  # leave anything else to json.dump (may raise — intentional)


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(name: str) -> str:
    """Turn an experiment name into a single safe path component.

    Prevents path traversal (``../``) and accidental nesting from slashes by
    collapsing any run of unsafe characters to ``_``. Stripping leading dots also
    blocks ``..`` and hidden-dir surprises.
    """
    cleaned = _SAFE_NAME.sub("_", name).strip("._")
    return cleaned or "_"


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Coerce a metrics/gate payload into a JSON-safe dict.

    Accepts a plain mapping, a dataclass (e.g. ``BacktestMetrics``), or any object
    exposing ``to_dict()`` (e.g. ``HoldoutGateResult``).
    """
    if isinstance(obj, dict):
        return dict(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if not isinstance(result, dict):  # pragma: no cover - defensive
            raise TypeError(f"{type(obj).__name__}.to_dict() did not return a dict")
        return result
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(
        f"Cannot log object of type {type(obj).__name__}: expected a dict, a "
        "dataclass, or an object with a to_dict() method."
    )


@dataclass
class RunRecord:
    """An immutable record of one tracked run."""

    run_id: str
    experiment: str
    run_name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    started_at: str = ""
    ended_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        return cls(
            run_id=d["run_id"],
            experiment=d["experiment"],
            run_name=d["run_name"],
            params=d.get("params", {}),
            metrics=d.get("metrics", {}),
            gate=d.get("gate", {}),
            status=d.get("status", "running"),
            started_at=d.get("started_at", ""),
            ended_at=d.get("ended_at"),
            error=d.get("error"),
        )


class ActiveRun:
    """A live run handle. Accumulates state and asks the tracker to persist.

    Obtained from :meth:`ExperimentTracker.run` and used as a context manager so
    the run is flushed (with a ``finished``/``failed`` status) on exit even when
    the body raises.
    """

    def __init__(self, tracker: _Persistable, record: RunRecord) -> None:
        self._tracker = tracker
        self._record = record

    @property
    def run_id(self) -> str:
        return self._record.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        self._record.params.update(params)
        self._tracker._persist(self._record)

    def log_metrics(self, metrics: Any) -> None:
        self._record.metrics.update(_to_plain_dict(metrics))
        self._tracker._persist(self._record)

    def log_gate(self, gate: Any) -> None:
        self._record.gate = _to_plain_dict(gate)
        self._tracker._persist(self._record)

    def __enter__(self) -> ActiveRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._record.ended_at = datetime.now(UTC).isoformat()
        if exc is not None:
            self._record.status = "failed"
            self._record.error = f"{type(exc).__name__}: {exc}"
        else:
            self._record.status = "finished"
        self._tracker._persist(self._record)
        # Do not suppress exceptions.


@runtime_checkable
class ExperimentTracker(Protocol):
    """The tracking contract shared by every backend."""

    def run(self, experiment: str, run_name: str) -> ActiveRun:
        """Open a new run under ``experiment`` and return a context-manager handle."""
        ...

    def list_runs(self, experiment: str) -> list[RunRecord]:
        """Return all runs for ``experiment``, newest first."""
        ...

    def get_run(self, experiment: str, run_id: str) -> RunRecord | None:
        """Return a single run by id, or ``None`` if absent."""
        ...


class _Persistable(Protocol):
    def _persist(self, record: RunRecord) -> None: ...


class LocalJSONTracker:
    """Zero-dependency tracker: one JSON file per run under ``<root>/<experiment>/``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def run(self, experiment: str, run_name: str) -> ActiveRun:
        record = RunRecord(
            run_id=uuid.uuid4().hex,
            experiment=experiment,
            run_name=run_name,
            started_at=datetime.now(UTC).isoformat(),
            status="running",
        )
        self._persist(record)
        return ActiveRun(self, record)

    def _exp_dir(self, experiment: str) -> Path:
        """Resolve an experiment's directory, sanitizing the name to a single
        safe path component (no traversal, no accidental nesting)."""
        return self.root / _safe_component(experiment)

    def list_runs(self, experiment: str) -> list[RunRecord]:
        exp_dir = self._exp_dir(experiment)
        if not exp_dir.is_dir():
            return []
        records = [RunRecord.from_dict(json.loads(p.read_text())) for p in exp_dir.glob("*.json")]
        # Newest first; started_at is an ISO string so lexical sort == chronological.
        records.sort(key=lambda r: r.started_at, reverse=True)
        return records

    def get_run(self, experiment: str, run_id: str) -> RunRecord | None:
        path = self._exp_dir(experiment) / f"{run_id}.json"
        if not path.is_file():
            return None
        return RunRecord.from_dict(json.loads(path.read_text()))

    def _persist(self, record: RunRecord) -> None:
        exp_dir = self._exp_dir(record.experiment)
        exp_dir.mkdir(parents=True, exist_ok=True)
        path = exp_dir / f"{record.run_id}.json"
        tmp = path.with_suffix(".json.tmp")
        # allow_nan=False guarantees a hard failure rather than emitting NaN/Infinity;
        # _json_safe has already mapped non-finite floats to None, so this holds.
        tmp.write_text(json.dumps(_json_safe(record.to_dict()), indent=2, allow_nan=False))
        tmp.replace(path)  # atomic on POSIX
