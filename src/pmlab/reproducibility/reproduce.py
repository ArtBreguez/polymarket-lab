"""``pmlab reproduce``: find a recorded run, re-execute it, and say whether it came back.

The verdict is about the *metrics*. A re-run reproduces when every recorded metric comes back
(counts exactly; floats within a documented tolerance) on the same data. Everything else is
evidence for the reader:

* **different data** (content hash) — refused, :class:`DataMismatchError`, unless the caller
  allows it; a run on other data can never be called a reproduction.
* **different library versions** — never a verdict by itself. They are listed (``report.drift``)
  next to the outcome: a run that reproduces despite drift is reproduced; one that does not,
  with drift listed, tells you where to look first.
* **nothing to re-run** (no record, unknown kind, missing panel) — :class:`ReproduceError`,
  with a message that says what to do about it.

Tolerance: on the machine and versions of the original run the result is bit-for-bit
identical, so the default is tight (``rtol=1e-9``, ``atol=1e-12``): it absorbs floating-point
summation order across CPUs and library builds, and nothing a different seed would produce.
Integer metrics (trade counts, steps) are compared exactly whatever the tolerance.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from pmlab.reproducibility.capture import (
    DataFingerprint,
    EnvDrift,
    ReproducibilityRecord,
    environment_drift,
    fingerprint_panel,
)
from pmlab.reproducibility.runner import BacktestConfig, load_panel, run_backtest

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "DataMismatchError",
    "MetricComparison",
    "ReproduceError",
    "ReproduceReport",
    "compare_metrics",
    "load_record",
    "reproduce",
]

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12
_RUN_ID = re.compile(r"[A-Za-z0-9._-]+")
_NO_RECORD = (
    "{origin} has no reproducibility record: it was written by pmlab before 0.10.0, or by "
    "something other than `pmlab backtest`. Re-run the backtest with pmlab >= 0.10.0 "
    "(`pmlab backtest ... --output metrics.json`) to capture one."
)


class ReproduceError(Exception):
    """The run cannot be re-executed. The message says why and what to do."""


class DataMismatchError(ReproduceError):
    """The panel is not the data the run was recorded on."""

    def __init__(self, recorded: DataFingerprint, current: DataFingerprint) -> None:
        self.recorded = recorded
        self.current = current
        parts = [
            f"content hash {recorded.content_sha256[:12]} (recorded) vs "
            f"{current.content_sha256[:12]} (now)",
            f"{recorded.n_rows} rows (recorded) vs {current.n_rows} rows (now)",
        ]
        only_then = [c for c in recorded.columns if c not in current.columns]
        only_now = [c for c in current.columns if c not in recorded.columns]
        if only_then:
            parts.append(f"columns only in the record: {only_then}")
        if only_now:
            parts.append(f"columns only in the data now: {only_now}")
        super().__init__(
            "The data differs from the data this run was recorded on ("
            + "; ".join(parts)
            + "). Refusing to compare metrics across different data; "
            "pass --allow-data-mismatch to run anyway."
        )


@dataclass(frozen=True)
class MetricComparison:
    name: str
    recorded: Any
    reproduced: Any
    abs_diff: float | None
    ok: bool


@dataclass
class ReproduceReport:
    """The outcome of one ``reproduce``: the verdict and the evidence behind it."""

    record: ReproducibilityRecord
    status: Literal["reproduced", "not_reproduced"]
    comparisons: list[MetricComparison]
    drift: list[EnvDrift]
    data_match: bool
    rtol: float
    atol: float
    notes: list[str] = field(default_factory=list)

    @property
    def reproduced(self) -> bool:
        return self.status == "reproduced"


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def compare_metrics(
    recorded: Mapping[str, Any],
    reproduced: Mapping[str, Any],
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> list[MetricComparison]:
    """Compare each recorded metric with its re-run value.

    Integers must be equal; floats must satisfy ``|a-b| <= atol + rtol*|a|``; ``None`` (a
    non-finite metric, stored as null) must come back ``None``. A metric the re-run no longer
    produces fails.
    """
    out: list[MetricComparison] = []
    for name, want in recorded.items():
        got = reproduced.get(name)
        if want is None or got is None:
            ok = want is None and got is None
            diff = None
        elif _is_int(want) and _is_int(got):
            ok, diff = want == got, abs(want - got)
        elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
            diff = abs(float(want) - float(got))
            ok = math.isclose(float(want), float(got), rel_tol=rtol, abs_tol=atol)
        else:
            continue  # not a number (nested payloads are not what a re-run is judged on)
        out.append(MetricComparison(name, want, got, diff, ok))
    return out


# ── finding the record ───────────────────────────────────────────────────
def _from_payload(data: Any, origin: str) -> ReproducibilityRecord:
    """The record inside a parsed JSON document: a run file, champion.json, a metrics file,
    or the bare record itself."""
    if isinstance(data, dict) and "reproducibility" in data:
        data = data["reproducibility"]
    elif isinstance(data, dict) and {"seeds", "environment", "config"} <= data.keys():
        pass  # the bare record
    else:
        raise ReproduceError(_NO_RECORD.format(origin=origin))
    if not data:
        raise ReproduceError(_NO_RECORD.format(origin=origin))
    try:
        return ReproducibilityRecord.from_dict(data)
    except (ValueError, TypeError, AttributeError) as e:
        raise ReproduceError(f"{origin}: {e}") from e


def _read(path: Path) -> ReproducibilityRecord:
    try:
        data = json.loads(path.read_text())
    except OSError as e:
        raise ReproduceError(f"{path} cannot be read: {e}") from e
    except ValueError as e:
        raise ReproduceError(f"{path} is not valid JSON: {e}") from e
    return _from_payload(data, str(path))


def load_record(
    ref: str,
    runs_dir: str | Path | None = None,
    registry_dir: str | Path | None = None,
) -> ReproducibilityRecord:
    """Resolve ``ref`` to a record. ``ref`` may be:

    * a path to a JSON file — a tracker run file, ``champion.json``, a registry version's
      ``champion.json``, the file ``pmlab backtest --output`` wrote, or a bare record;
    * a path to a directory holding ``champion.json``;
    * a tracker run id (or unique prefix) under ``runs_dir`` (default ``runs``);
    * a model-registry version id under ``registry_dir``.

    Raises :class:`ReproduceError` when nothing matches, or the match holds no record.
    """
    path = Path(ref)
    if path.is_dir():
        champion = path / "champion.json"
        if not champion.is_file():
            raise ReproduceError(f"{path} is a directory without a champion.json")
        return _read(champion)
    if path.is_file():
        return _read(path)

    looked: list[str] = []
    if registry_dir is not None and _RUN_ID.fullmatch(ref):
        archived = Path(registry_dir) / ref / "champion.json"
        looked.append(f"the registry {registry_dir}")
        if archived.is_file():
            return _read(archived)
    if _RUN_ID.fullmatch(ref):
        root = Path(runs_dir) if runs_dir is not None else Path("runs")
        looked.append(f"run ids under {root}")
        matches = sorted(root.glob(f"*/{ref}*.json")) if root.is_dir() else []
        if len(matches) == 1:
            return _read(matches[0])
        if len(matches) > 1:
            ids = ", ".join(m.stem for m in matches)
            raise ReproduceError(f"Run id {ref!r} is ambiguous: it matches {ids}. Give more of it.")
    where = "; ".join(looked) if looked else "paths"
    raise ReproduceError(
        f"No run {ref!r} found: it is not an existing file or directory (also looked in {where}). "
        "Pass the path of the record, or a run id with --runs-dir."
    )


# ── re-execution ─────────────────────────────────────────────────────────
def reproduce(
    record: ReproducibilityRecord,
    *,
    panel: pd.DataFrame | None = None,
    panel_path: str | Path | None = None,
    allow_data_mismatch: bool = False,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ReproduceReport:
    """Re-execute the recorded run and compare its metrics with the recorded ones.

    The panel is, in order of preference: ``panel`` (in memory), ``panel_path``, the path the
    record was made from. Its fingerprint must match the record's, else
    :class:`DataMismatchError` — unless ``allow_data_mismatch``, which runs anyway and can then
    only ever report ``not_reproduced``.
    """
    if record.kind != "backtest":
        raise ReproduceError(
            f"Cannot reproduce a run of kind {record.kind!r}: only 'backtest' runs can be "
            "re-executed from their record."
        )
    try:
        config = BacktestConfig.from_dict(record.config)
    except (ValueError, TypeError) as e:
        raise ReproduceError(f"The recorded config is not usable: {e}") from e

    if panel is None:
        source = panel_path if panel_path is not None else record.data.source
        if source is None:
            raise ReproduceError(
                "The record does not say where its data came from; pass the panel with --panel."
            )
        if not Path(source).is_file():
            raise ReproduceError(
                f"Panel not found: {source}. If the data moved, pass its new location with --panel."
            )
        try:
            panel = load_panel(source)
        except ValueError as e:
            raise ReproduceError(str(e)) from e

    current = fingerprint_panel(panel)
    data_match = current.content_sha256 == record.data.content_sha256
    notes: list[str] = []
    if not data_match:
        if not allow_data_mismatch:
            raise DataMismatchError(record.data, current)
        notes.append(
            "Ran on different data than the record (--allow-data-mismatch): the metrics are "
            "not comparable, so this can not count as a reproduction."
        )

    outcome = run_backtest(panel, config, record.seeds)
    comparisons = compare_metrics(record.metrics, outcome.metrics, rtol=rtol, atol=atol)
    reproduced = data_match and all(c.ok for c in comparisons)
    return ReproduceReport(
        record=record,
        status="reproduced" if reproduced else "not_reproduced",
        comparisons=comparisons,
        drift=environment_drift(record.environment),
        data_match=data_match,
        rtol=rtol,
        atol=atol,
        notes=notes,
    )
