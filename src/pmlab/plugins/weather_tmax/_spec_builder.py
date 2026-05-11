"""Convert raw Gamma API market dicts to MarketSpec for temperature markets."""

from __future__ import annotations

import re
from typing import Any

from pmlab.core.market_spec import MarketSpec, OutcomeBin

# Regex to parse "Highest temperature in {City} on {Month} {Day}?"
_TMAX_PATTERN = re.compile(
    r"highest temperature in (?P<city>[\w\s]+) on (?P<date>.+?)\??$",
    re.IGNORECASE,
)


def build_tmax_spec(raw: dict[str, Any]) -> MarketSpec:
    """Build a MarketSpec from a raw Gamma API response dict."""
    question: str = raw.get("question", "")
    market_id: str = raw.get("id", raw.get("market_id", ""))
    slug: str = raw.get("slug", market_id)
    close_time: str = raw.get("endDate", raw.get("close_time", ""))

    # Parse city from question
    m = _TMAX_PATTERN.match(question)
    city = m.group("city").strip() if m else raw.get("city", "")

    # Build outcome bins from the tokens in the API response
    tokens: list[dict[str, Any]] = raw.get("tokens", []) or raw.get("outcomes", [])
    bins = _build_bins_from_tokens(tokens)

    return MarketSpec(
        market_id=market_id,
        slug=slug,
        question=question,
        outcome_bins=bins,
        close_time=close_time,
        market_family="range",
        tags=["weather", "temperature"],
        metadata={
            "city": city,
            "target_date": raw.get("target_date", ""),
            "timezone": raw.get("timezone", "UTC"),
        },
    )


def _build_bins_from_tokens(tokens: list[dict[str, Any]]) -> list[OutcomeBin]:
    """Build OutcomeBin list from raw token/outcome dicts."""
    bins = []
    for t in tokens:
        label: str = str(t.get("outcome", t.get("label", t.get("name", ""))))
        # Try to parse numeric bounds from label like "30°C" or ">35°C" or "28-30°C"
        lower, upper = _parse_bounds(label)
        bins.append(OutcomeBin(label=label, lower=lower, upper=upper))
    return bins


def _parse_bounds(label: str) -> tuple[float | None, float | None]:
    """Parse numeric temperature bounds from a bin label. Best-effort."""
    label_clean = label.replace("°C", "").replace("°F", "").strip()
    # Range: "28-30" or "28 to 30"
    range_m = re.match(r"([\d.]+)\s*[-–to]+\s*([\d.]+)", label_clean)
    if range_m:
        return float(range_m.group(1)), float(range_m.group(2))
    # Greater/less than: ">35" or ">=35" or "<20"
    gt_m = re.match(r">=?\s*([\d.]+)", label_clean)
    if gt_m:
        return float(gt_m.group(1)), None
    lt_m = re.match(r"<=?\s*([\d.]+)", label_clean)
    if lt_m:
        return None, float(lt_m.group(1))
    # Single value: "30"
    single_m = re.match(r"^([\d.]+)$", label_clean)
    if single_m:
        v = float(single_m.group(1))
        return v - 0.5, v + 0.5
    return None, None
