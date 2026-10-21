"""Re-settle a trade log whose realized PnL was booked by an older release.

Before 0.9.1 a NO trade was settled as a short YES at the NO price: a NO bet that lost its
stake booked a fraction of it, and one that won booked a stake-sized win. Settlement is
idempotent, so those numbers stayed in the log, and the watchdog refuses to judge a NO
book by them (a NO trade without `settlement_version` is left out and counted).

`resettle_trades` recomputes the PnL of every settled trade that lacks the current
`settlement_version`, from the fields the trade recorded: its side, its fill (or quote),
its stake and fee, and whether its label occurred (`outcome` is about the label, so it says).
It never asks a plugin for truth, never touches the network, leaves an open trade exactly as
it is, leaves a trade it cannot recompute exactly as it is (and says so), and a second pass
changes nothing. The arithmetic is `book_trade`, the settlement engine's own.

`resettle_file` wraps it for the CLI: a timestamped backup of the original first, then an
atomic replace; a dry run changes nothing.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmlab.execution.settlement import SETTLEMENT_VERSION, book_trade

__all__ = ["Change", "ResettleResult", "resettle_file", "resettle_trades"]


@dataclass(frozen=True)
class Change:
    """One trade whose realized PnL is different after re-settling."""

    description: str
    old: float | None
    new: float


@dataclass(frozen=True)
class ResettleResult:
    """The re-settled log, and an account of what was and was not done to it."""

    trades: list[dict[str, Any]]
    changes: list[Change] = field(default_factory=list)
    n_resettled: int = 0  # settled trades recomputed (their value may not have moved)
    n_changed: int = 0  # of those, how many realized_pnl values differ
    n_open: int = 0  # unsettled: never touched
    n_current: int = 0  # already at (or past) the current settlement version: not touched
    n_unrecomputable: int = 0  # settled, but a recorded field is unusable: not touched
    unrecomputable: list[str] = field(default_factory=list)
    total_before: float = 0.0  # realized PnL of the resettled trades, as it was
    total_after: float = 0.0  # and as it is now


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _describe(trade: Mapping[str, Any]) -> str:
    side = "yes" if trade.get("direction", "yes") == "yes" else "no"
    return (
        f"{trade.get('city_or_segment', '?')} {trade.get('target_date', '?')} "
        f"{trade.get('horizon', '?')} {side}"
    )


def _unusable(trade: Mapping[str, Any]) -> str | None:
    """Why this settled trade cannot be recomputed from its own fields, or None if it can."""
    if trade.get("outcome") not in ("won", "lost"):
        return f"outcome is {trade.get('outcome')!r}, not won or lost"
    label = trade.get("outcome_label")
    if not isinstance(label, str) or not label:
        return "no outcome_label"
    gamma = _number(trade.get("gamma_price"))
    if gamma is None or not 0.0 <= gamma <= 1.0:
        return f"gamma_price {trade.get('gamma_price')!r} is not a price in [0, 1]"
    stake = trade.get("flat_stake", 1.0)
    if _number(stake) is None or float(stake) <= 0:
        return f"flat_stake {stake!r} is not a positive number"
    fee = trade.get("fee_paid", 0.003)
    if _number(fee) is None or float(fee) < 0:
        return f"fee_paid {fee!r} is not a number >= 0"
    fill = trade.get("fill_price")
    entry = gamma if trade.get("direction", "yes") == "yes" else 1.0 - gamma
    if fill is not None:
        entry_value = _number(fill)
        if entry_value is None:
            return f"fill_price {fill!r} is not a number"
        entry = entry_value
    if not 0.0 < entry <= 1.0:
        return f"the entry price {entry!r} is not a price in (0, 1]"
    return None


def resettle_trades(trades: Sequence[Mapping[str, Any]], *, now_utc: datetime) -> ResettleResult:
    """Recompute the PnL of settled trades booked before the current settlement version.

    Args:
        trades: The log's trades. Not modified.
        now_utc: When the re-settlement happens, stamped as `resettled_at` on each trade it
            recomputes. Must be timezone-aware.

    Returns:
        The log with those trades recomputed and stamped with `settlement_version`, and the
        counts and per-trade changes that say what happened.
    """
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    stamp = now_utc.astimezone(UTC).isoformat()
    out: list[dict[str, Any]] = []
    changes: list[Change] = []
    unrecomputable: list[str] = []
    n_open = n_current = n_resettled = 0
    before = after = 0.0
    for raw in trades:
        trade = dict(raw)
        if trade.get("outcome") is None:
            n_open += 1
            out.append(trade)
            continue
        version = _number(trade.get("settlement_version"))
        if version is not None and version >= SETTLEMENT_VERSION:
            n_current += 1
            out.append(trade)
            continue
        reason = _unusable(trade)
        if reason is not None:
            unrecomputable.append(f"{_describe(trade)}: {reason}")
            out.append(trade)
            continue
        booked = book_trade(trade, label_occurred=trade["outcome"] == "won")
        old = _number(trade.get("realized_pnl"))
        n_resettled += 1
        before += old or 0.0
        after += booked.realized_pnl
        if old is None or abs(booked.realized_pnl - old) > 1e-9:
            changes.append(Change(_describe(trade), old, booked.realized_pnl))
        out.append(
            {
                **trade,
                "realized_pnl": booked.realized_pnl,
                "settled_basis": booked.basis,
                "settlement_version": SETTLEMENT_VERSION,
                "resettled_at": stamp,
            }
        )
    return ResettleResult(
        trades=out,
        changes=changes,
        n_resettled=n_resettled,
        n_changed=len(changes),
        n_open=n_open,
        n_current=n_current,
        n_unrecomputable=len(unrecomputable),
        unrecomputable=unrecomputable,
        total_before=before,
        total_after=after,
    )


def _backup_path(path: Path, now_utc: datetime) -> Path:
    stem = f"{path.name}.bak-{now_utc.astimezone(UTC):%Y%m%dT%H%M%SZ}"
    candidate = path.with_name(stem)
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{stem}-{counter}")
        counter += 1
    return candidate


def resettle_file(
    path: Path, *, dry_run: bool, now_utc: datetime
) -> tuple[ResettleResult, Path | None]:
    """Re-settle the trade log at `path`.

    A dry run reads and reports. Otherwise, if anything needs re-settling, the original is
    copied to ``<name>.bak-<UTC timestamp>`` next to it first, and the log is replaced
    atomically; if nothing does, no backup is made and the file is not rewritten.

    Returns:
        The result, and the backup's path (None for a dry run or when nothing was written).

    Raises:
        FileNotFoundError: `path` does not exist.
        ValueError: `path` is not a trade log (``{"trades": [...]}``).
    """
    if not path.is_file():
        raise FileNotFoundError(f"trade log not found: {path}")
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("trades"), list):
        raise ValueError(f'{path} is not a trade log: expected an object with a "trades" list')
    if not all(isinstance(t, dict) for t in data["trades"]):
        raise ValueError(f"{path} has a trade that is not an object")
    result = resettle_trades(data["trades"], now_utc=now_utc)
    if dry_run or result.n_resettled == 0:
        return result, None
    backup = _backup_path(path, now_utc)
    shutil.copy2(path, backup)
    payload = json.dumps({**data, "trades": result.trades}, indent=2)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(payload)
        shutil.copymode(path, temp_name)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return result, backup
