"""What a run records about itself: seeds, library versions, resolved config, a data fingerprint.

A result you cannot re-run is not a result. ``ReproducibilityRecord`` is the one object that
every record store holds (the experiment tracker's run file, ``champion.json``, the model
registry's archived versions, the metrics file ``pmlab backtest --output`` writes), so
``pmlab reproduce`` reads them all the same way.

Four things are captured, each answering "why might a re-run differ?":

* **seeds** — every random generator the run draws from, by name (``python_random``, ``numpy``,
  ``lightgbm``, ...), with the value actually used. ``None`` means "not seeded by us".
* **environment** — pmlab, Python, numpy, pandas, lightgbm, scikit-learn and optuna versions
  plus the OS / CPU architecture. A different version can change a result without changing a
  line of the code.
* **config** — the *resolved* parameters (every default spelled out), so a later change of a
  default does not silently change what an old record means.
* **data** — a content hash of the panel, its row count and its columns: enough to tell that the
  data under a path is no longer the data the run was made on.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from pmlab.tracking import _json_safe

__all__ = [
    "SCHEMA_VERSION",
    "DataFingerprint",
    "EnvDrift",
    "ReproducibilityRecord",
    "capture_environment",
    "environment_drift",
    "fingerprint_panel",
    "seed_everything",
    "seeded",
]

# Bumped when a field of the record changes meaning. A reader refuses a *newer* schema (it
# cannot know what the new fields mean) and reads every older one.
SCHEMA_VERSION = 1

# Environment keys whose change can change a result. `platform` (the kernel string) and any
# other key are informational: a kernel upgrade is not a reason to distrust a number.
_COMPARED = (
    "pmlab",
    "python",
    "numpy",
    "pandas",
    "lightgbm",
    "scikit-learn",
    "optuna",
    "system",
    "machine",
)
_LIBRARIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "lightgbm": "lightgbm",
    "scikit-learn": "sklearn",
    "optuna": "optuna",
}


def _library_version(module: str) -> str | None:
    try:
        return str(importlib.import_module(module).__version__)
    except ImportError:  # an optional library that is not installed
        return None


def capture_environment() -> dict[str, str | None]:
    """Versions of pmlab and the libraries a result depends on, plus the OS / architecture.

    A library that is not installed (optuna, behind the ``tune`` extra) is recorded as ``None``
    rather than left out, so "was not there" and "was not recorded" are different statements.
    """
    import pmlab

    env: dict[str, str | None] = {
        "pmlab": pmlab.__version__,
        "python": platform.python_version(),
    }
    for name, module in _LIBRARIES.items():
        env[name] = _library_version(module)
    env["system"] = platform.system()
    env["machine"] = platform.machine()
    env["platform"] = platform.platform()
    env["cpu_count"] = str(os.cpu_count())  # informational, like `platform`: never compared
    return env


@dataclass(frozen=True)
class EnvDrift:
    """One library (or platform field) whose version differs from the one in the record."""

    name: str
    recorded: str | None
    current: str | None


def environment_drift(
    recorded: Mapping[str, str | None], current: Mapping[str, str | None] | None = None
) -> list[EnvDrift]:
    """What differs between the recorded environment and this one (default: the live one).

    Only keys the record holds are compared, so a record that predates a newly tracked library
    is not accused of drifting on it.
    """
    now = capture_environment() if current is None else current
    return [
        EnvDrift(name, recorded[name], now.get(name))
        for name in _COMPARED
        if name in recorded and recorded[name] != now.get(name)
    ]


# ── data fingerprint ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataFingerprint:
    """Enough of a dataset to tell, later, that it is not the same one.

    ``content_sha256`` decides; ``n_rows`` and ``columns`` are there so a mismatch can say *how*
    the data differs. ``source`` is where the panel was read from, if it came from a file.
    """

    content_sha256: str
    n_rows: int
    columns: list[str]
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "n_rows": self.n_rows,
            "columns": list(self.columns),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DataFingerprint:
        return cls(
            content_sha256=str(d["content_sha256"]),
            n_rows=int(d["n_rows"]),
            columns=[str(c) for c in d["columns"]],
            source=d.get("source"),
        )


def _column_digest(series: pd.Series) -> tuple[str, bytes]:
    """A (kind, digest) pair for one column that survives a dtype-width or file-format change.

    Values are normalised before hashing (any int -> int64, any float -> float64, timestamps ->
    UTC nanoseconds, strings -> object) so the same data read back from parquet, CSV or another
    pandas version hashes the same, while any changed value, or its position, does not.
    """
    kind = "text"
    values: pd.Series = series
    try:
        if ptypes.is_bool_dtype(series) and not ptypes.is_extension_array_dtype(series):
            kind, values = "bool", series
        elif ptypes.is_integer_dtype(series) and not ptypes.is_extension_array_dtype(series):
            kind, values = "int", series.astype("int64")
        elif ptypes.is_float_dtype(series):
            kind, values = "float", series.astype("float64")
        elif ptypes.is_datetime64_any_dtype(series):
            tz_naive = series.dt.tz_convert("UTC").dt.tz_localize(None) if series.dt.tz else series
            kind, values = "datetime", tz_naive.astype("datetime64[ns]")
        else:
            values = series.astype("object")
    except (TypeError, ValueError):  # e.g. a nullable integer holding <NA>: hash it as text
        values = series.astype("object")
    try:
        hashed = pd.util.hash_pandas_object(values, index=False)
    except TypeError:  # cells that cannot be hashed (lists, dicts, arrays): hash their text
        hashed = pd.util.hash_pandas_object(values.astype(str), index=False)
    return kind, hashed.to_numpy(dtype="uint64").tobytes()


def fingerprint_panel(df: pd.DataFrame, source: str | None = None) -> DataFingerprint:
    """Fingerprint a panel: content hash (values, column names, row order), rows, columns.

    The index is ignored (a panel re-read from disk has a fresh one); row order is not, because
    a walk-forward backtest can depend on it.
    """
    h = hashlib.sha256(b"pmlab-panel-fingerprint-v1\n")
    h.update(f"{len(df)}\n".encode())
    for name in df.columns:
        kind, digest = _column_digest(df[name])
        h.update(str(name).encode() + b"\0" + kind.encode() + b"\0" + digest)
    return DataFingerprint(
        content_sha256=h.hexdigest(),
        n_rows=len(df),
        columns=[str(c) for c in df.columns],
        source=source,
    )


# ── seeds ────────────────────────────────────────────────────────────────
def seed_everything(seeds: Mapping[str, int | None]) -> None:
    """Pin the process-wide generators named in ``seeds`` (``python_random``, ``numpy``).

    Library-specific seeds (``lightgbm``, ``optuna``, ...) are not global: they are handed to
    the estimator that uses them. A ``None`` seed is left alone — the generator keeps whatever
    state it has, and the record says so.
    """
    py = seeds.get("python_random")
    if py is not None:
        random.seed(py)
    npy = seeds.get("numpy")
    if npy is not None:
        np.random.seed(npy)


@contextmanager
def seeded(seeds: Mapping[str, int | None]) -> Iterator[None]:
    """Seed the process-wide generators for the block, then give them back as they were.

    A library call (``run_backtest``, ``reproduce``) must not reseed the caller's own numpy or
    ``random`` stream as a side effect, so it uses this rather than :func:`seed_everything`.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    seed_everything(seeds)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


# ── the record ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ReproducibilityRecord:
    """Everything needed to re-run a run and to judge whether it came back the same.

    ``kind`` names the recipe (``"backtest"``); ``config`` is its resolved parameters;
    ``metrics`` is what the original run produced, the target a re-run is compared against.
    """

    kind: str
    seeds: dict[str, int | None]
    environment: dict[str, str | None]
    config: dict[str, Any]
    data: DataFingerprint
    metrics: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def capture(
        cls,
        kind: str,
        seeds: Mapping[str, int | None],
        config: Mapping[str, Any],
        data: DataFingerprint,
        metrics: Mapping[str, Any],
    ) -> ReproducibilityRecord:
        """Record a run now: the live environment, plus what the caller says it used."""
        return cls(
            kind=kind,
            seeds=dict(seeds),
            environment=capture_environment(),
            config=_json_safe(dict(config)),
            data=data,
            metrics=_json_safe(dict(metrics)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "created_at": self.created_at,
            "seeds": dict(self.seeds),
            "environment": dict(self.environment),
            "config": _json_safe(self.config),
            "data": self.data.to_dict(),
            "metrics": _json_safe(self.metrics),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReproducibilityRecord:
        """Read a record back; ``ValueError`` names what is wrong with a malformed one."""
        if not isinstance(d, Mapping):
            raise ValueError(f"reproducibility record must be an object, got {type(d).__name__}")
        version = d.get("schema_version", SCHEMA_VERSION)
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            raise ValueError(
                f"reproducibility record has schema_version {version!r}, written by a newer "
                f"pmlab than this one (reads up to {SCHEMA_VERSION}); upgrade pmlab to read it"
            )
        for name in ("kind", "seeds", "environment", "config", "data", "metrics"):
            if name not in d:
                raise ValueError(f"reproducibility record is missing {name!r}")
        try:
            data = DataFingerprint.from_dict(d["data"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"reproducibility record has a malformed 'data' entry: {e!r}") from e
        return cls(
            kind=str(d["kind"]),
            seeds=dict(d["seeds"]),
            environment=dict(d["environment"]),
            config=dict(d["config"]),
            data=data,
            metrics=dict(d["metrics"]),
            schema_version=version,
            created_at=str(d.get("created_at", "")),
        )
