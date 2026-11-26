"""The fee schedule a market carries, where it is read from, and when it was seen (#71).

Polymarket publishes a market's fee in two places, and they do not say the same thing:

* **Gamma** (``feesEnabled``, ``feeType`` and ``feeSchedule {rate, exponent, takerOnly,
  rebateRate}`` on the market object) is the schedule: the fee of a fill is ``shares x rate x
  (price x (1 - price)) ** exponent`` (see ``pmlab.core.fee_schedule``).
* **CLOB** ``GET /fee-rate?token_id=`` answers ``{"base_fee": 1000}``. That number is **not the
  rate**. It is the ``feeRateBps`` of a v1 order, in basis points, and it is 1000 on every market
  that charges a fee (``rate`` 0.01 or 0.05 alike) and 0 on one that does not. Reading it as
  ``1000 / 10_000 = 0.10`` would make a 0.05 market twice and a 0.01 market ten times too dear.

What was read, 2026-11-26:

* ``Polymarket/py-clob-client`` (archived) b076b04d61135657e25dccc1bbd6866a96bd8c6e,
  ``py_clob_client/client.py`` lines 450-458 (``get_fee_rate_bps`` returns ``result.get("base_fee")
  or 0``) and 476-490 (it must equal the ``fee_rate_bps`` the user signs), and
  ``order_builder/builder.py`` lines 139 and 178 (``feeRateBps=str(order_args.fee_rate_bps)``);
  ``Polymarket/clob-client-v2`` 2a3f0b02b46de77afee2364881e0256eab4d75cb ``src/client.ts`` lines
  505-518 (``getFeeRateBps`` reads ``base_fee``) and ``src/types/ordersV1.ts`` lines 86-88 ("Fee
  rate, in basis points, charged to the order maker, charged on proceeds").
* ``Polymarket/py-clob-client-v2`` 9c7c9e89cbbc11528daae90dc66ef65c9380cbdf asks for it only for a
  v1 order (``py_clob_client_v2/client.py`` lines 773-774 and 842-843: ``... if version == 1 else
  None``) and takes the fee of a v2 order from the market's ``fd`` rate and exponent
  (``fees.py`` line 31). The v2 exchange (``Polymarket/ctf-exchange-v2`` ccc0596074f4dfd62c944fbca4de252893b82b4b,
  ``src/exchange/mixins/Fees.sol`` lines 17 and 36-51, ``Trading.sol`` lines 563-575) has no
  fee-rate field in an order: the operator passes the fee amount and the contract only checks it is
  at most ``maxFeeRateBps`` (500 by default) of the trade's cash value.
* The v1 contract (``Polymarket/ctf-exchange`` ed5c7708b7be3aa98bf5f0c6602b57cc498e2ef4,
  ``src/exchange/libraries/CalculatorHelper.sol`` lines 27-48) charged ``feeRateBps x min(price, 1 -
  price)``: a different curve from Gamma's, which is why the two numbers cannot be compared.
* Live Gamma and CLOB: the 222 fee-charging markets of one sample (six ``feeType`` values, rates
  0.01 and 0.05) all carried ``takerBaseFee`` and ``makerBaseFee`` 1000, and CLOB's ``base_fee`` was
  1000 for one market of each ``feeType``; the 8 fee-free markets had ``base_fee`` 0 (and no
  ``feeSchedule`` key).

So ``base_fee`` is a flag: it agrees with Gamma when both say "a fee is charged" or both say "none".
When they do not, Gamma's schedule wins and a warning names the market. The rate itself only ever
comes from Gamma, or from the category default of ``pmlab.core.fee_schedule`` when Gamma says
nothing, and the snapshot records which.

The snapshots are kept with the moment they were seen (`FeeScheduleStore`), because Polymarket
changes its rates without notice: a backtest at date T must price a trade with the schedule that was
known before T, not with today's.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pmlab.core.fee_schedule import FeeSchedule, category_fee_schedule
from pmlab.markets.tmax_bins import _as_list

if TYPE_CHECKING:
    from pmlab.data.snapshot_store import FeatureSnapshotStore

__all__ = ["FeeScheduleStore", "FeeSnapshot", "parse_fee_rate", "resolve_fee_snapshot"]

FeeSource = Literal["gamma", "clob", "category_default"]
"""Where a snapshot's schedule came from: Gamma's ``feeSchedule`` (or its ``feesEnabled: false``),
CLOB's ``base_fee`` of 0 when Gamma said nothing, or the category default (a fallback)."""

# The Gamma keys a fee is read from; the cache keeps these and nothing else of a market.
_FEE_MARKET_KEYS = (
    "id",
    "conditionId",
    "slug",
    "feesEnabled",
    "feeSchedule",
    "feeType",
    "clobTokenIds",
)
_SCHEDULE_KEYS = (
    ("exponent", "exponent"),
    ("takerOnly", "taker_only"),
    ("rebateRate", "rebate_rate"),
)
_FAMILY = "fee_schedules"


def _utc(value: datetime | date | str, what: str) -> datetime:
    """An aware UTC datetime from an aware datetime, a date (its midnight, UTC) or an ISO string.

    A datetime or timestamp string without a timezone is refused: read as local time it would be a
    lookahead (or a delay) of the machine's offset. A date, or a date-only string, is a day."""
    parsed: object = value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(f"{what} is not an ISO date or timestamp: {value!r}") from None
    if isinstance(parsed, datetime):
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{what} needs a timezone (UTC is fine), got {value!r}")
        return parsed.astimezone(UTC)
    if isinstance(parsed, date):
        return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
    raise TypeError(f"{what} must be a datetime, a date or an ISO string, got {value!r}")


@dataclass(frozen=True)
class FeeSnapshot:
    """The fee schedule of one market as it was seen at one moment, and where it came from.

    ``resolve_fee_snapshot`` and the clients' ``fetch_fee_snapshot`` make them, `FeeScheduleStore`
    keeps them. ``fallback`` says the schedule is the category default and not something the market
    itself published: such a number is a guess, and a backtest should know.

    Attributes:
        market_id: The market's Gamma ``id`` (its ``conditionId``, or its slug, when it has none).
        seen_at: When it was seen, timezone-aware (stored in UTC).
        schedule: The fee schedule to price a trade with.
        source: ``"gamma"`` when it is the market's own ``feeSchedule`` (or its
            ``feesEnabled: false``), ``"clob"`` when Gamma said nothing and CLOB's ``base_fee`` was
            0 (fee-free), ``"category_default"`` when it is the fallback.
        slug: The market's slug, when it has one.
        fee_type: Gamma's ``feeType`` (``"sports_fees_v3"``), when it has one.
        base_fee: CLOB's ``base_fee``, in basis points of a v1 order, when it was fetched. It is a
            flag (see the ``pmlab.markets.fees`` module docs) and never the rate.
        category: The category whose default was used; only set when ``source`` is
            ``"category_default"``.
    """

    market_id: str
    seen_at: datetime
    schedule: FeeSchedule
    source: FeeSource
    slug: str | None = None
    fee_type: str | None = None
    base_fee: int | None = None
    category: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seen_at", _utc(self.seen_at, "seen_at"))

    @property
    def fallback(self) -> bool:
        """True when the schedule is the category default, not one the market published."""
        return self.source == "category_default"

    @property
    def seen_on(self) -> date:
        """The UTC date it was seen."""
        return self.seen_at.date()

    def to_dict(self) -> dict[str, Any]:
        """A flat, JSON-serialisable mapping (the schedule's fields next to the snapshot's), which
        ``FeeSnapshot.from_dict`` reads back."""
        return {
            "market_id": self.market_id,
            "slug": self.slug,
            "seen_at": self.seen_at.isoformat(),
            "source": self.source,
            "fee_type": self.fee_type,
            "base_fee": self.base_fee,
            "category": self.category,
            **self.schedule.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeeSnapshot:
        """The snapshot ``to_dict`` wrote.

        Args:
            data: A mapping with at least ``market_id``, ``seen_at``, ``source`` and ``rate``.

        Returns:
            The snapshot.

        Raises:
            KeyError: If a required key is missing.
            ValueError: If a value is out of range.
        """
        schedule = FeeSchedule(
            **{f.name: data[f.name] for f in fields(FeeSchedule) if f.name in data}
        )
        return cls(
            market_id=data["market_id"],
            seen_at=data["seen_at"],
            schedule=schedule,
            source=data["source"],
            slug=data.get("slug"),
            fee_type=data.get("fee_type"),
            base_fee=data.get("base_fee"),
            category=data.get("category"),
        )


def parse_fee_rate(data: Any) -> int | None:
    """The ``base_fee`` (basis points) of a CLOB ``/fee-rate`` reply, or ``None`` if it has none.

    A reply that is not an object, or whose ``base_fee`` is missing, null, negative, fractional or
    not a number is *unknown*, not 0: the official clients read a missing ``base_fee`` as 0
    (``result.get("base_fee") or 0``), which would make a reply that says nothing look fee-free. A
    whole number sent as a float or as a string is read.
    """
    if not isinstance(data, Mapping):
        return None
    value = data.get("base_fee")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return None
    elif isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            return None
        value = int(value)
    elif not isinstance(value, int):
        return None
    return value if value >= 0 else None


def first_token_id(market: Mapping[str, Any]) -> str | None:
    """The first CLOB token id of a Gamma market (``clobTokenIds`` is a list or a JSON string of
    one; the first is the Yes token of a Yes/No market), or ``None`` if it has none."""
    tokens = _as_list(market.get("clobTokenIds"))
    token = tokens[0] if tokens else None
    if isinstance(token, bool) or not isinstance(token, str | int):
        return None
    return str(token).strip() or None


def fee_fields(market: Mapping[str, Any]) -> dict[str, Any]:
    """The keys of a Gamma market a fee is read from (what the clients cache)."""
    return {k: market[k] for k in _FEE_MARKET_KEYS if k in market}


def _text(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    return str(value).strip() or None


def _identity(market: Mapping[str, Any]) -> tuple[str, str | None]:
    slug = _text(market.get("slug"))
    for key in ("id", "conditionId"):
        found = _text(market.get(key))
        if found:
            return found, slug
    if slug:
        return slug, slug
    raise ValueError("a market with no id, conditionId or slug cannot be named, so not snapshotted")


def _warn(message: str, *args: object) -> None:
    from pmlab.logging import get_logger

    get_logger(__name__).warning(message, *args)


def _gamma_schedule(
    market: Mapping[str, Any], enabled: bool | None, name: str
) -> FeeSchedule | None:
    """Gamma's ``feeSchedule`` as a ``FeeSchedule``, or ``None`` (with a warning when Gamma says
    fees are on, or the schedule is there and cannot be read)."""
    raw = market.get("feeSchedule")
    if raw is None:
        if enabled is True:
            _warn("%s: feesEnabled is true but Gamma carries no feeSchedule", name)
        return None
    try:
        if not isinstance(raw, Mapping):
            raise ValueError(f"feeSchedule is a {type(raw).__name__}, not an object")
        kwargs = {dst: raw[src] for src, dst in _SCHEDULE_KEYS if raw.get(src) is not None}
        return FeeSchedule(rate=raw["rate"], enabled=enabled is not False, **kwargs)
    except (KeyError, TypeError, ValueError) as exc:
        _warn("%s: Gamma's feeSchedule %r cannot be read (%s)", name, raw, exc)
        return None


def resolve_fee_snapshot(
    market: Mapping[str, Any],
    *,
    base_fee: int | None = None,
    category: str | None = None,
    seen_at: datetime | date | str | None = None,
) -> FeeSnapshot:
    """The fee schedule of a Gamma market object, as a `FeeSnapshot`; no network, no cache.

    Gamma's ``feeSchedule`` is the schedule. ``feesEnabled: false`` is an answer too: the market is
    fee-free, whatever its category would charge. Only when Gamma says nothing usable (no
    ``feesEnabled``, or fees on with no readable schedule) is the category default used, and the
    snapshot says so (``fallback``); a ``base_fee`` of 0 on its own, with Gamma silent, reads as
    fee-free (``source="clob"``).

    ``base_fee`` is CLOB's ``/fee-rate`` answer (`parse_fee_rate`). It is a flag, 1000 on every
    market that charges a fee and 0 on one that does not, not the rate: ``rate`` is never
    ``base_fee / 10_000``. When it and Gamma disagree on whether a fee is charged, Gamma's schedule
    wins and a warning names the market; a category default that it contradicts (``feesEnabled``
    true with no readable schedule, CLOB 0) is kept, and warned about the same way. The sources are
    listed in "Fee schedules" of the market data clients page of the docs.

    Args:
        market: A Gamma market object (``id``, ``slug``, ``feesEnabled``, ``feeType``,
            ``feeSchedule``; only the keys that exist are read).
        base_fee: CLOB's ``base_fee`` for one of the market's tokens, if it was fetched.
        category: The category whose default is the fallback, a key of
            ``POLYMARKET_CATEGORY_FEE_RATES`` in ``pmlab.core.fee_schedule`` (``"weather"``,
            ``"sports"``). Required only if Gamma gives no schedule.
        seen_at: When the market was read, timezone-aware; default now (UTC).

    Returns:
        The snapshot.

    Raises:
        TypeError: If ``market`` is not a mapping.
        ValueError: If the market has no ``id``, ``conditionId`` or ``slug``, ``base_fee`` is not
            ``None`` or a non-negative int, ``seen_at`` has no timezone, or a fallback is needed and
            ``category`` is missing or unknown.
    """
    if not isinstance(market, Mapping):
        raise TypeError(f"market must be a Gamma market object (a mapping), got {market!r}")
    if base_fee is not None and (
        isinstance(base_fee, bool) or not isinstance(base_fee, int) or base_fee < 0
    ):
        raise ValueError(f"base_fee must be None or a non-negative int (bps), got {base_fee!r}")
    market_id, slug = _identity(market)
    name = f"market {slug or market_id!r} (id {market_id})"
    seen = datetime.now(UTC) if seen_at is None else _utc(seen_at, "seen_at")
    fee_type = _text(market.get("feeType"))
    flag = market.get("feesEnabled")
    enabled = flag if isinstance(flag, bool) else None

    source: FeeSource = "gamma"
    used_category: str | None = None
    schedule = _gamma_schedule(market, enabled, name)
    if schedule is None and enabled is False:
        schedule = FeeSchedule(rate=0.0, enabled=False)
    elif schedule is None and enabled is None and base_fee == 0:
        schedule, source = FeeSchedule(rate=0.0, enabled=False), "clob"
    elif schedule is None:
        if category is None:
            raise ValueError(
                f"{name} has no usable fee schedule on Gamma and no category was given for the "
                "fallback (category='weather', 'sports', ...)"
            )
        schedule, source, used_category = (
            category_fee_schedule(category),
            "category_default",
            category,
        )

    charges = schedule.enabled and schedule.rate > 0.0
    if base_fee is not None and charges != (base_fee > 0):
        says = f"a fee is charged (rate {schedule.rate})" if charges else "no fee is charged"
        if source == "gamma":
            _warn(
                "%s: Gamma says %s but CLOB /fee-rate says base_fee %d; using Gamma's schedule",
                name,
                says,
                base_fee,
            )
        else:  # the category default: a guess that CLOB's flag contradicts, and which stays
            _warn(
                "%s: Gamma gave no usable schedule, so the %s category default is used, and it "
                "says %s but CLOB /fee-rate says base_fee %d",
                name,
                used_category,
                says,
                base_fee,
            )
    return FeeSnapshot(
        market_id=market_id,
        seen_at=seen,
        schedule=schedule,
        source=source,
        slug=slug,
        fee_type=fee_type,
        base_fee=base_fee,
        category=used_category,
    )


def _plain(value: Any) -> Any:
    """A parquet cell as a plain Python value: NaN and null are ``None``, a numpy scalar its item,
    a pandas Timestamp a datetime."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value.item() if hasattr(value, "item") else value


def _snapshot_of(row: Mapping[Any, Any]) -> FeeSnapshot:
    data = {k: _plain(v) for k, v in row.items()}
    data["seen_at"] = data.pop("captured_at")
    if data.get("base_fee") is not None:
        data["base_fee"] = int(data["base_fee"])
    return FeeSnapshot.from_dict(data)


class FeeScheduleStore:
    """Fee schedules kept with the moment they were seen: what a backtest could have known.

    One parquet file (the ``fee_schedules`` family of a `FeatureSnapshotStore` in ``root``), so it
    shares that store's point-in-time rule: a read at a cutoff gets, for each market, the latest
    snapshot seen **strictly before** it, and a market first seen at or after it has none (never the
    later one). Recording the same sighting twice changes nothing; a changed fee is a new snapshot
    (with its own ``seen_at``), and `history` lists them all. Every sighting with a new ``seen_at``
    is a row, even when nothing changed, so a daily pass over many markets grows the file by one row
    per market per pass, and each `record` rewrites it: record in batches.

    Args:
        root: The directory of the store (created if missing).
    """

    def __init__(self, root: Path | str) -> None:
        # imported here: pmlab.data pulls pandas and the whole data package in, and pmlab.markets
        # (whose clients import this module) should stay light and free of import cycles
        from pmlab.data.snapshot_store import FeatureSnapshotStore

        self._store: FeatureSnapshotStore = FeatureSnapshotStore(root)

    @property
    def root(self) -> Path:
        """The directory of the store."""
        return self._store.root

    def record(self, snapshots: Iterable[FeeSnapshot]) -> int:
        """Add snapshots, in one write.

        A snapshot is identified by its market and the moment it was seen. One whose market and
        moment are stored already (or earlier in the same call) is ignored and the first stays: that
        is what makes recording the same sighting twice harmless. If the one that is ignored has a
        *different* schedule or source, a warning names the market and both, because a fee that
        changed between two sightings with the same ``seen_at`` (a ``date``, say, which is a
        midnight) would otherwise be lost without a trace: give each sighting its own moment.

        Args:
            snapshots: What ``resolve_fee_snapshot`` or ``fetch_fee_snapshot`` returned.

        Returns:
            The number of snapshots in the store after the write.

        Raises:
            TypeError: If an item is not a ``FeeSnapshot``.
        """
        snaps = list(snapshots)
        for snap in snaps:
            if not isinstance(snap, FeeSnapshot):
                raise TypeError(f"record takes FeeSnapshot objects, got {snap!r}")
        self._warn_if_a_sighting_changed(snaps)
        rows: list[dict[str, Any]] = []
        for snap in snaps:
            row = snap.to_dict()
            row["captured_at"] = row.pop("seen_at")
            rows.append(row)
        return self._store.append(_FAMILY, rows)

    def _warn_if_a_sighting_changed(self, snaps: list[FeeSnapshot]) -> None:
        """Warn for each snapshot the store is about to drop because its market and moment are taken
        by a first one with another schedule or source."""
        if not snaps:
            return
        import pandas as pd

        kept: dict[tuple[str, datetime], FeeSnapshot] = {}
        stored = self._store.load(_FAMILY)
        if not stored.empty:
            same_place = stored["market_id"].isin({s.market_id for s in snaps}) & stored[
                "captured_at"
            ].isin(pd.to_datetime([s.seen_at for s in snaps], utc=True))
            for row in stored[same_place].to_dict("records"):
                first = _snapshot_of(row)
                kept[(first.market_id, first.seen_at)] = first
        for snap in snaps:
            first = kept.setdefault((snap.market_id, snap.seen_at), snap)
            if first is not snap and (first.schedule, first.source) != (snap.schedule, snap.source):
                _warn(
                    "fee schedule of market %s: one is already recorded for %s (%s, rate %s, %s); "
                    "the new one (%s, rate %s, %s) is ignored. A snapshot is its market and the "
                    "moment it was seen: give each sighting its own seen_at",
                    snap.slug or snap.market_id,
                    snap.seen_at.isoformat(),
                    first.source,
                    first.schedule.rate,
                    "charges" if first.schedule.enabled else "no fee",
                    snap.source,
                    snap.schedule.rate,
                    "charges" if snap.schedule.enabled else "no fee",
                )

    def as_of(
        self, cutoff: datetime | date | str, market_ids: Iterable[str] | None = None
    ) -> dict[str, FeeSnapshot]:
        """The schedule each market had, as far as was known **strictly before** ``cutoff``.

        Args:
            cutoff: An aware datetime, a date (its midnight, UTC, so a schedule seen during that day
                is not yet known) or an ISO string.
            market_ids: Only these markets; default all.

        Returns:
            ``{market_id: snapshot}``; a market with no snapshot before ``cutoff`` is absent.

        Raises:
            ValueError: If ``cutoff`` is a datetime or timestamp without a timezone.
        """
        frame = self._store.as_of(_FAMILY, _utc(cutoff, "cutoff").isoformat())
        if frame.empty:
            return {}
        if market_ids is not None:
            frame = frame[frame["market_id"].isin({str(m) for m in market_ids})]
        return {str(r["market_id"]): _snapshot_of(r) for r in frame.to_dict("records")}

    def history(self, market_id: str) -> list[FeeSnapshot]:
        """Every snapshot of a market, oldest first: a change of fee shows as a change of schedule."""
        frame = self._store.load(_FAMILY)
        if frame.empty:
            return []
        frame = frame[frame["market_id"] == str(market_id)].sort_values("captured_at")
        return [_snapshot_of(r) for r in frame.to_dict("records")]
