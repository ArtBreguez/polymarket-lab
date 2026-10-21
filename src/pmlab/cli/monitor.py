"""`pmlab monitor`: the watchdog's loader and printer.

`pmlab.monitoring.watchdog.health_report` decides; this module only gets the
inputs to it and the verdict out. It loads champion.json, the trade log and (for
drift) a reference panel and a snapshot store, hands them over, and prints.

The contract is the scheduler's, not the human's:

* a healthy run prints **nothing** and exits 0;
* a problem prints a short, plain-ASCII message and exits with a code that says
  what kind (see `pmlab.monitoring.watchdog` for the table);
* nothing that goes wrong, a missing file, a corrupt champion, an exception, can
  exit 0. Bad input is 64, a crash is 70; both print one line to stderr.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer

from pmlab.monitoring.watchdog import (
    CHECKS,
    EXIT_INTERNAL,
    EXIT_USAGE,
    HealthReport,
    WatchdogConfig,
    WatchdogInputError,
    health_report,
    parse_timestamp,
)


class UsageError(Exception):
    """Something the operator can fix: a path, a flag, a file. Exits 64."""


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _fail(message: str, code: int) -> int:
    typer.echo(f"pmlab monitor: error: {_one_line(message)}", err=True)
    return code


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _load_champion(path: Path) -> Any:
    from pmlab.modeling.champion import ChampionManifest

    if not path.is_file():
        raise UsageError(f"champion not found: {path}")
    try:
        manifest = ChampionManifest.load(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # JSONDecodeError is a ValueError; a missing key is a KeyError. All of them
        # mean "champion.json is not a usable champion", never "healthy".
        raise UsageError(
            f"champion.json is unreadable ({path}): {type(exc).__name__}: {exc}"
        ) from exc
    if manifest.gate.decision != "GO":
        raise UsageError(
            f"champion.json records a {manifest.gate.decision} gate ({path}); "
            "a champion can only be published with GO, so this file cannot be trusted"
        )
    return manifest


def _load_trades(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise UsageError(f"trade log not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise UsageError(f"trade log is unreadable ({path}): {type(exc).__name__}: {exc}") from exc
    rows = payload.get("trades") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise UsageError(f"trade log {path} must be a JSON object with a 'trades' list of objects")
    return rows


def _load_config(
    path: Path | None, *, min_trades: int | None, insufficient: str | None
) -> WatchdogConfig:
    data: dict[str, Any] = {}
    if path is not None:
        if not path.is_file():
            raise UsageError(f"config file not found: {path}")
        try:
            if path.suffix == ".toml":
                data = tomllib.loads(path.read_text())
            elif path.suffix == ".json":
                data = json.loads(path.read_text())
            else:
                raise UsageError(f"config file {path} must be .toml or .json")
        except (OSError, ValueError) as exc:  # TOMLDecodeError and JSONDecodeError
            raise UsageError(f"config file is unreadable ({path}): {exc}") from exc
        if not isinstance(data, dict):
            raise UsageError(f"config file {path} must hold a table/object at the top level")
    config = WatchdogConfig.from_dict(data)
    if min_trades is not None:
        config = replace(
            config,
            calibration=replace(config.calibration, min_trades=min_trades),
            pnl=replace(config.pnl, min_trades=min_trades),
        )
    if insufficient is not None:
        if insufficient not in ("separate", "warn"):
            raise UsageError(f"--insufficient must be 'separate' or 'warn', got {insufficient!r}")
        config = replace(config, insufficient=insufficient)  # type: ignore[arg-type]
    return config


def _load_reference(path: Path) -> Any:
    import pandas as pd

    if not path.is_file():
        raise UsageError(f"reference panel not found: {path}")
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    except (OSError, ValueError, ImportError) as exc:
        raise UsageError(f"reference panel is unreadable ({path}): {exc}") from exc
    raise UsageError(f"reference panel {path} must be .parquet or .csv")


def _load_model(manifest: Any, champion_path: Path) -> Any:
    from pmlab.modeling.lgbm_baseline import LGBMForecaster

    # champion.json records an absolute path from the machine that published it;
    # next to champion.json is where it lives after the directory is moved.
    candidates = [manifest.model_path, champion_path.parent / Path(manifest.model_path).name]
    for candidate in candidates:
        if Path(candidate).is_file():
            try:
                return LGBMForecaster.load(Path(candidate))
            except Exception as exc:
                raise UsageError(
                    f"cannot load the champion model ({candidate}): {exc}; "
                    "pass --no-prediction to skip prediction drift"
                ) from exc
    raise UsageError(
        f"champion model not found (looked at {', '.join(str(c) for c in candidates)}); "
        "pass --no-prediction to skip prediction drift"
    )


def _drift_report(
    *,
    manifest: Any,
    champion_path: Path,
    reference: Path,
    snapshots: Path,
    family: str | None,
    window_start: str,
    cutoff: str,
    features: Sequence[str] | None,
    use_prediction: bool,
    config: WatchdogConfig,
) -> Any:
    import numpy as np

    from pmlab.data import FeatureSnapshotStore
    from pmlab.monitoring.drift import drift_report_from_store

    panel = _load_reference(reference)
    if not snapshots.is_dir():
        # The store's constructor creates its root; a typo must not leave an empty
        # store behind that then reads as "nothing live yet".
        raise UsageError(f"snapshot directory not found: {snapshots}")

    columns = list(features) if features else None
    predict_fn: Callable[[Any], Any] | None = None
    if use_prediction:
        model = _load_model(manifest, champion_path)
        columns = columns or model.feature_names() or None
        if columns is None:
            columns = [c for c in panel.columns if c.startswith("feature_")]
        missing = [c for c in columns if c not in panel.columns]
        if missing:
            raise UsageError(f"the reference panel lacks feature(s) the model uses: {missing}")
        used = list(columns)

        def predict_fn(frame: Any) -> Any:
            lacking = [c for c in used if c not in frame.columns]
            if lacking:
                raise ValueError(f"the live snapshots lack feature(s) the model uses: {lacking}")
            proba = np.asarray(model.predict_proba(frame[used]))
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)

    try:
        return drift_report_from_store(
            FeatureSnapshotStore(snapshots),
            family or manifest.plugin_family,
            panel,
            window_start=window_start,
            cutoff=cutoff,
            features=columns,
            predict_fn=predict_fn,
            config=config.drift,
        )
    except ValueError as exc:
        raise UsageError(f"drift could not be computed: {exc}") from exc


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def _emit(report: HealthReport, *, verbose: bool, json_path: str | None) -> bool:
    """Print the report; returns False if the requested JSON could not be written."""
    to_stderr = json_path == "-"
    text = report.render(verbose=verbose)
    if text:
        typer.echo(text, err=to_stderr)
    if json_path is None:
        return True
    payload = json.dumps(report.to_dict(), indent=2)
    if json_path == "-":
        typer.echo(payload)
        return True
    try:
        Path(json_path).write_text(payload + "\n")
    except OSError as exc:
        typer.echo(f"pmlab monitor: error: cannot write {json_path}: {_one_line(exc)}", err=True)
        return False
    return True


def run_monitor(
    *,
    champion_path: Path,
    trades_path: Path,
    reference: Path | None,
    snapshots: Path | None,
    family: str | None,
    window_days: int,
    window_start: str | None,
    features: Sequence[str] | None,
    use_prediction: bool,
    as_of: str | None,
    config_path: Path | None,
    min_trades: int | None,
    insufficient: str | None,
    skip: Sequence[str],
    json_path: str | None,
    verbose: bool,
) -> int:
    """Run the watchdog once; returns the process exit code. Never raises."""
    try:
        return _run(
            champion_path=champion_path,
            trades_path=trades_path,
            reference=reference,
            snapshots=snapshots,
            family=family,
            window_days=window_days,
            window_start=window_start,
            features=features,
            use_prediction=use_prediction,
            as_of=as_of,
            config_path=config_path,
            min_trades=min_trades,
            insufficient=insufficient,
            skip=skip,
            json_path=json_path,
            verbose=verbose,
        )
    except (UsageError, WatchdogInputError) as exc:
        return _fail(str(exc), EXIT_USAGE)
    except Exception as exc:
        # Anything else is a bug or an environment fault. It must exit non-zero and
        # outside the verdict range: a traceback would exit 1, which reads as "warn".
        return _fail(f"internal error: {type(exc).__name__}: {exc}", EXIT_INTERNAL)


def _run(
    *,
    champion_path: Path,
    trades_path: Path,
    reference: Path | None,
    snapshots: Path | None,
    family: str | None,
    window_days: int,
    window_start: str | None,
    features: Sequence[str] | None,
    use_prediction: bool,
    as_of: str | None,
    config_path: Path | None,
    min_trades: int | None,
    insufficient: str | None,
    skip: Sequence[str],
    json_path: str | None,
    verbose: bool,
) -> int:
    # The one place the clock is read. Everything downstream is a function of it.
    as_of = as_of if as_of is not None else datetime.now(UTC).isoformat(timespec="seconds")
    cutoff = parse_timestamp(as_of)
    if cutoff is None:
        raise UsageError(f"--as-of {as_of!r} is not an ISO-8601 timestamp")
    if window_days < 1:
        raise UsageError(f"--window-days must be at least 1, got {window_days}")
    unknown = sorted(set(skip) - set(CHECKS))
    if unknown:
        raise UsageError(f"unknown check(s) to skip {unknown}; valid checks: {list(CHECKS)}")
    checks = [name for name in CHECKS if name not in skip]
    if not checks:
        raise UsageError("--skip removed every check; run at least one")

    config = _load_config(config_path, min_trades=min_trades, insufficient=insufficient)
    manifest = _load_champion(champion_path)
    trades = _load_trades(trades_path) if {"calibration", "pnl"} & set(checks) else []

    drift = None
    if "drift" in checks:
        if (reference is None) != (snapshots is None):
            raise UsageError("--reference and --snapshots must be given together")
        if reference is not None and snapshots is not None:
            start = window_start or (cutoff - timedelta(days=window_days)).isoformat()
            if parse_timestamp(start) is None:
                raise UsageError(f"--window-start {start!r} is not an ISO-8601 timestamp")
            drift = _drift_report(
                manifest=manifest,
                champion_path=champion_path,
                reference=reference,
                snapshots=snapshots,
                family=family,
                window_start=start,
                cutoff=cutoff.isoformat(),
                features=features,
                use_prediction=use_prediction,
                config=config,
            )

    report = health_report(
        champion=manifest,
        trades=trades,
        drift=drift,
        as_of=as_of,
        config=config,
        checks=checks,
    )
    if not _emit(report, verbose=verbose, json_path=json_path):
        return EXIT_USAGE
    return report.exit_code
