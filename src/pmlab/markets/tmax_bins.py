"""The per-bin Yes/No temperature markets Polymarket lists, read into one entry per city and day.

Live Gamma has no market whose outcomes are temperature bins. It lists one Yes/No market per bin,
``Will the highest temperature in Warsaw be -4°C or below on November 26?``, and ties the eleven of
a city and day together as an *event*. This module reads the question of each market into a bin and
groups the bins of one city and day, so a caller sees what it sees for a multi-outcome market: one
entry, a list of bins, each with the token to trade it and its price.

It is pure (no I/O, standard library only) and is what ``GammaClient.discover_tmax_markets``,
``AsyncGammaClient.discover_tmax_markets`` and ``WeatherTmaxPlugin.discover_markets`` all use, so
that they cannot disagree about which markets are temperature markets.

**What a bin means.** The questions come in four forms (checked against 1,654 live markets on
2026-11-25): ``N°C`` (Celsius cities, one degree per bin), ``between A-B°F`` (Fahrenheit cities, two
degrees per bin), ``N°C or higher`` and ``N°C or below`` (the open ends). Gamma resolves them on
whole degrees (the source reports whole degrees), so a bin holds the readings that round into it:
``33°C`` is ``[32.5, 33.5)``, ``between 88-89°F`` is ``[87.5, 89.5)``, ``40°C or higher`` is
``[39.5, inf)`` and ``-4°C or below`` is ``(-inf, -3.5)``. Every bin is half-open (the lower end
is in, the upper end is out: `LOWER_INCLUSIVE`, `UPPER_INCLUSIVE`), so the bins of a full event
tile the line, with no gap and no overlap, and a reading of 33.5 is in the 34 bin, once. Only
whole degrees are read; a question with a fractional degree is refused, not rounded.

**What is refused**, with one warning per call (`PerBinGrouper.warn_refused`) and never a guess: a
question that is this family's wording but whose bin or date cannot be read; a market whose outcomes
are not Yes/No, or that has no Yes token id (Gamma lists a few stale ones: no ``clobTokenIds``, no
prices); one whose year cannot be told; and every market of an event whose bins disagree (they mix
°C and °F, name two cities or days, or repeat a label). A market whose price Gamma did not give
keeps its bin, with no price.

*Lowest* temperature markets are another family and are not read: `is_per_bin_question` is False
for them.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

# Gamma's `Daily Temperature` tag (the parent `Weather` tag is 84). `/markets/keyset` takes it as
# `tag_id`: on 2026-11-25 it listed 2,633 markets in 27 pages and the walk ended by itself, where
# the unfiltered universe is over 100,000 markets. `tag` (by name) is ignored by live Gamma.
DAILY_TEMPERATURE_TAG_ID = 103040

# Every bin is half-open: the lower end is in it, the upper end is not (see the module docstring).
LOWER_INCLUSIVE = True
UPPER_INCLUSIVE = False

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_NUMBER = {name.lower(): i for i, name in enumerate(_MONTHS, start=1)}
_MONTH_NUMBER.update({name[:3].lower(): i for i, name in enumerate(_MONTHS, start=1)})
_MONTH_NUMBER["sept"] = 9

# This family's wording, loosely: a question that starts like this is ours to read or to refuse, it
# is never the multi-outcome form (whose question reads "Highest temperature in X on <date>?").
_CANDIDATE = re.compile(r"^\s*will the highest temperature in\b", re.IGNORECASE)
_QUESTION = re.compile(
    r"^\s*Will the highest temperature in (?P<city>.+?) be (?P<bin>.+?) on "
    r"(?P<month>[A-Za-z]+)\.? (?P<day>\d{1,2})(?:, (?P<year>\d{4}))?\?\s*$",
    re.IGNORECASE,
)
_N = r"-?\d+"
_UNIT = r"\s*°\s*(?P<unit>[CF])"
_EXACT = re.compile(rf"^(?P<n>{_N}){_UNIT}$", re.IGNORECASE)
_RANGE = re.compile(
    rf"^between\s+(?P<a>{_N})\s*(?:-|–|—|to|and)\s*(?P<b>{_N}){_UNIT}$", re.IGNORECASE
)
_OPEN = re.compile(
    rf"^(?P<n>{_N}){_UNIT}\s+or\s+(?P<direction>higher|above|lower|below)$", re.IGNORECASE
)
_YEAR_IN_SLUG = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# Why a market was refused. Few and fixed, so that one warning can count them.
REASON_QUESTION = "cannot read the bin or the date from the question"
REASON_OUTCOMES = "outcomes are not Yes/No"
REASON_TOKEN = "no Yes token id"
REASON_ID = "no market id"
REASON_YEAR = "cannot tell the year of the date"
REASON_UNITS = "the bins of one event mix °C and °F"
REASON_DAYS = "the bins of one event name different cities or days"
REASON_LABEL = "two bins of one event have the same label"


@dataclass(frozen=True)
class BinQuestion:
    """What one per-bin question says."""

    city: str
    month: int
    day: int
    year: int | None  # only when the question gives one
    unit: str  # "C" or "F"
    label: str  # "33°C", "88-89°F", "40°C or higher", "-4°C or below"
    lower: float | None  # None: open at the bottom
    upper: float | None  # None: open at the top


@dataclass(frozen=True)
class PerBinBin:
    """One bin of a group: its own market, Yes token and Yes price."""

    label: str
    lower: float | None
    upper: float | None
    market_id: str
    token_id: str
    price: float | None  # the Yes price; None when Gamma gave none (or not one in [0, 1])
    active: bool


@dataclass(frozen=True)
class PerBinGroup:
    """The bins of one city and day, in order from the lowest to the highest."""

    market_id: str  # the event's id on Gamma; the slug when the markets name no event
    slug: str
    question: str  # "Highest temperature in <city> on <Month> <day>?"
    city: str
    target_date: str  # ISO, YYYY-MM-DD
    unit: str
    close_time: str  # the earliest close of its bins, as Gamma wrote it ("" when none did)
    bins: tuple[PerBinBin, ...]
    complete: bool  # an open bottom, an open top and no gap between: every bin is here
    active: bool  # every bin's market is
    event_id: str | None


def is_per_bin_question(question: Any) -> bool:
    """True for a question in this family's wording (``Will the highest temperature in ...``).

    That says it is ours to read, not that it can be read: see `parse_bin_question`. Lowest
    temperature questions, the multi-outcome wording and anything that is not text are False.
    """
    return isinstance(question, str) and _CANDIDATE.match(question) is not None


def parse_bin_question(question: str) -> BinQuestion:
    """Read ``Will the highest temperature in <city> be <bin> on <date>?``.

    ``<bin>`` is ``N°C``, ``between A-B°F`` (``A-B`` may be negative: ``-5--4``; ``to`` and
    ``and`` do for the dash), or ``N°C or higher`` / ``or below`` (``or above`` / ``or lower``
    too); whole degrees only.

    Raises:
        ValueError: The question is not in that form, the bin has a fractional degree, mixes
            units, or has its low end above its high end, the month is not one, or the day is not.
    """
    text = " ".join(question.replace("−", "-").split())
    m = _QUESTION.match(text)
    if m is None:
        raise ValueError(f"cannot read the question as <city>, <bin> and <date>: {question!r}")
    month = _MONTH_NUMBER.get(m.group("month").lower())
    day = int(m.group("day"))
    if month is None or not 1 <= day <= 31:
        raise ValueError(f"cannot read the date of the question: {question!r}")
    year = int(m.group("year")) if m.group("year") else None

    bin_text = m.group("bin")
    lower: float | None
    upper: float | None
    if (hit := _EXACT.match(bin_text)) is not None:
        n = int(hit.group("n"))
        unit = hit.group("unit").upper()
        label, lower, upper = f"{n}°{unit}", n - 0.5, n + 0.5
    elif (hit := _RANGE.match(bin_text)) is not None:
        a, b = int(hit.group("a")), int(hit.group("b"))
        if a > b:
            raise ValueError(f"the bin runs from {a} down to {b}: {question!r}")
        unit = hit.group("unit").upper()
        label, lower, upper = f"{a}-{b}°{unit}", a - 0.5, b + 0.5
    elif (hit := _OPEN.match(bin_text)) is not None:
        n = int(hit.group("n"))
        unit = hit.group("unit").upper()
        if hit.group("direction").lower() in ("higher", "above"):
            label, lower, upper = f"{n}°{unit} or higher", n - 0.5, None
        else:
            label, lower, upper = f"{n}°{unit} or below", None, n + 0.5
    else:
        raise ValueError(f"cannot read the bin {bin_text!r} of the question: {question!r}")

    city = m.group("city").strip()
    return BinQuestion(city, month, day, year, unit, label, lower, upper)


def _as_list(raw: Any) -> list[Any]:
    """A list from a JSON string or a list; anything else is an empty list."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return raw if isinstance(raw, list) else []


def _yes_index(raw: dict[str, Any]) -> int | None:
    """Where "Yes" is in the outcomes, which must be exactly Yes and No (either order)."""
    names = [str(o).strip().lower() for o in _as_list(raw.get("outcomes"))]
    if len(names) != 2 or sorted(names) != ["no", "yes"]:
        return None
    return names.index("yes")


def _yes_token(raw: dict[str, Any], index: int) -> str | None:
    tokens = _as_list(raw.get("clobTokenIds"))
    if (
        len(tokens) != 2
        or not isinstance(tokens[index], (str, int))
        or isinstance(tokens[index], bool)
    ):
        return None
    token = str(tokens[index]).strip()
    return token or None


def _yes_price(raw: dict[str, Any], index: int) -> float | None:
    prices = _as_list(raw.get("outcomePrices"))
    if len(prices) != 2:
        return None
    try:
        price = float(prices[index])
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and 0.0 <= price <= 1.0 else None


def _market_id(raw: dict[str, Any]) -> str | None:
    key = raw.get("id") or raw.get("conditionId")
    return str(key) if key else None


def _end_text(raw: dict[str, Any]) -> str:
    value = raw.get("endDate") or raw.get("endDateIso")
    return value.strip() if isinstance(value, str) else ""


_FAR = datetime.max.replace(tzinfo=UTC)


def _instant(text: str) -> datetime | None:
    """*text* as an aware datetime (a date or a time without a zone is UTC), or None."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


def _day(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _target_date(q: BinQuestion, end: str, slug: Any) -> date | None:
    """The day the question asks about; its year is the question's, else the one that puts the
    day nearest the market's close (a market closes within a day of the day it asks about, and a
    day near New Year can fall either side of it), else the slug's. None when none is a real day."""
    if q.year is not None:
        return _day(q.year, q.month, q.day)
    closes = _instant(end) if end else None
    if closes is not None:
        candidates = [
            _day(y, q.month, q.day) for y in (closes.year - 1, closes.year, closes.year + 1)
        ]
        real = [d for d in candidates if d is not None]
        if real:
            return min(real, key=lambda d: abs((d - closes.date()).days))
        return None
    if isinstance(slug, str) and (hit := _YEAR_IN_SLUG.search(slug)) is not None:
        return _day(int(hit.group(1)), q.month, q.day)
    return None


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class _Draft:
    """The bins of one event (or one city and day, when markets name no event) as they arrive."""

    def __init__(
        self, event_id: str | None, event_slug: str | None, q: BinQuestion, day: date
    ) -> None:
        self.event_id = event_id
        self.event_slug = event_slug
        self.city = q.city
        self.day = day
        self.unit = q.unit
        self.bins: list[PerBinBin] = []
        self.ends: list[str] = []
        self.conflict: str | None = None

    def add(self, q: BinQuestion, day: date, bin_: PerBinBin, end: str) -> None:
        if self.conflict is None:
            if q.unit != self.unit:
                self.conflict = REASON_UNITS
            elif (q.city.casefold(), day) != (self.city.casefold(), self.day):
                self.conflict = REASON_DAYS
            elif any(b.label == bin_.label for b in self.bins):
                self.conflict = REASON_LABEL
        self.bins.append(bin_)
        if end and _instant(end) is not None:  # an end that cannot be read is not a close
            self.ends.append(end)

    def build(self) -> PerBinGroup:
        bins = tuple(sorted(self.bins, key=lambda b: (b.lower is not None, b.lower or 0.0)))
        month_name = _MONTHS[self.day.month - 1]
        slug = self.event_slug or (
            f"highest-temperature-in-{_slugify(self.city)}-on-{month_name.lower()}-"
            f"{self.day.day}-{self.day.year}"
        )
        close = min(self.ends, key=lambda e: _instant(e) or _FAR) if self.ends else ""
        complete = (
            bins[0].lower is None
            and bins[-1].upper is None
            and all(a.upper == b.lower for a, b in zip(bins, bins[1:], strict=False))
        )
        return PerBinGroup(
            market_id=self.event_id or slug,
            slug=slug,
            question=f"Highest temperature in {self.city} on {month_name} {self.day.day}?",
            city=self.city,
            target_date=self.day.isoformat(),
            unit=self.unit,
            close_time=close,
            bins=bins,
            complete=complete,
            active=all(b.active for b in bins),
            event_id=self.event_id,
        )


class PerBinGrouper:
    """Takes raw Gamma markets one at a time and groups the per-bin temperature ones.

    ``offer(raw)`` is True for a market in this family's wording, whether it was taken or refused,
    and False for any other, which is the caller's to handle (the multi-outcome form, other
    families). ``groups()`` are the events with at least one bin read, in the order their first
    bin arrived, unless the event's bins disagree; ``refused`` lists ``(market id, reason)`` for
    the rest, and `warn_refused` says so once.
    """

    def __init__(self) -> None:
        self._drafts: dict[tuple[Any, ...], _Draft] = {}
        self._refused: list[tuple[str, str]] = []

    def offer(self, raw: Any) -> bool:
        if not isinstance(raw, dict) or not is_per_bin_question(raw.get("question")):
            return False
        market_id = _market_id(raw)
        label = market_id or "?"
        try:
            q = parse_bin_question(raw["question"])
        except ValueError:
            self._refused.append((label, REASON_QUESTION))
            return True
        yes = _yes_index(raw)
        if yes is None:
            self._refused.append((label, REASON_OUTCOMES))
            return True
        token = _yes_token(raw, yes)
        if token is None:
            self._refused.append((label, REASON_TOKEN))
            return True
        if market_id is None:
            self._refused.append((label, REASON_ID))
            return True
        end = _end_text(raw)
        day = _target_date(q, end, raw.get("slug"))
        if day is None:
            self._refused.append((label, REASON_YEAR))
            return True

        event_id, event_slug = _event_of(raw)
        key: tuple[Any, ...] = (
            ("event", event_id) if event_id else ("day", q.city.casefold(), day.isoformat())
        )
        draft = self._drafts.get(key)
        if draft is None:
            draft = self._drafts[key] = _Draft(event_id, event_slug, q, day)
        draft.add(
            q,
            day,
            PerBinBin(
                label=q.label,
                lower=q.lower,
                upper=q.upper,
                market_id=market_id,
                token_id=token,
                price=_yes_price(raw, yes),
                active=bool(raw.get("active", True)),
            ),
            end,
        )
        return True

    def groups(self) -> list[PerBinGroup]:
        return [d.build() for d in self._drafts.values() if d.conflict is None]

    @property
    def refused(self) -> list[tuple[str, str]]:
        """``(market id, reason)`` for every market of this family that is not in a group."""
        out = list(self._refused)
        for draft in self._drafts.values():
            if draft.conflict is not None:
                out.extend((b.market_id, draft.conflict) for b in draft.bins)
        return out

    def warn_refused(self, where: str) -> None:
        """Log one WARNING counting what was refused and why, with a few market ids of each kind."""
        refused = self.refused
        if not refused:
            return
        from pmlab.logging import get_logger

        by_reason: dict[str, list[str]] = {}
        for market_id, reason in refused:
            by_reason.setdefault(reason, []).append(market_id)
        parts = [
            f"{reason} ({len(ids)}: {', '.join(ids[:3])}{', ...' if len(ids) > 3 else ''})"
            for reason, ids in by_reason.items()
        ]
        n = len(refused)
        get_logger(__name__).warning(
            "%s: %d per-bin temperature market%s not read: %s",
            where,
            n,
            " was" if n == 1 else "s were",
            "; ".join(parts),
        )


def _event_of(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    """The id and slug of the event a market belongs to, when it names one."""
    events = raw.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and event.get("id"):
                slug = event.get("slug")
                return str(event["id"]), slug if isinstance(slug, str) and slug else None
    return None, None
