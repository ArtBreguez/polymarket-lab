"""What "allowed segments" means, in one place.

A champion promoted on a per-segment gate allows the segments that passed. One promoted on
an aggregate gate (`HoldoutGateResult.evaluate(trades)` with no `required_segments`, which
grades the whole trade log as a single segment called ``"all"``) was never graded per
segment, so it makes no per-segment claim: ``"all"`` means *every segment*. The broker, the
champion and the watchdog all read the allowlist through this one rule.
"""

from __future__ import annotations

import pytest

from pmlab.core.segments import ALL_SEGMENTS, allows_every_segment, segment_allowed


def test_the_wildcard_is_the_name_the_aggregate_gate_uses() -> None:
    import pandas as pd

    from pmlab.backtest.holdout_gate import HoldoutGateResult

    trades = pd.DataFrame({"realized_pnl": [1.0] * 50, "outcome": ["won"] * 50})
    gate = HoldoutGateResult.evaluate(trades)
    assert [r.segment for r in gate.segment_results] == [ALL_SEGMENTS] == ["all"]


@pytest.mark.parametrize("segment", ["Chicago", "politics", "", "all", "ALL"])
def test_all_allows_every_segment(segment: str) -> None:
    assert segment_allowed(segment, {"all"})


def test_all_beside_named_segments_is_a_segment_called_all_not_the_wildcard() -> None:
    # A per-segment gate that happens to contain a segment named "all" graded that segment,
    # not the whole log: it must not widen trading to every segment.
    allowed = {"all", "Dallas"}
    assert not segment_allowed("Chicago", allowed)
    assert segment_allowed("Dallas", allowed)
    assert segment_allowed("all", allowed)


@pytest.mark.parametrize("allowed", [{"all"}, ["all"], ("all", "all")])
def test_all_on_its_own_is_the_wildcard(allowed: object) -> None:
    assert allows_every_segment(allowed)  # type: ignore[arg-type]


@pytest.mark.parametrize("allowed", [{"all", "Dallas"}, {"Dallas"}, set(), ["all", "ALL"]])
def test_nothing_else_is_the_wildcard(allowed: object) -> None:
    assert not allows_every_segment(allowed)  # type: ignore[arg-type]


def test_named_segments_allow_only_themselves() -> None:
    assert segment_allowed("Dallas", {"Dallas", "Miami"})
    assert not segment_allowed("Chicago", {"Dallas", "Miami"})


def test_the_comparison_is_exact() -> None:
    assert not segment_allowed("dallas", {"Dallas"})
    assert not segment_allowed("Dallas ", {"Dallas"})
    assert not segment_allowed("ALL", {"Dallas"})  # only the literal "all" is the wildcard


def test_an_empty_allowlist_allows_nothing_and_none_allows_everything() -> None:
    assert not segment_allowed("Chicago", set())
    assert segment_allowed("Chicago", None)


class TestAStringIsNotACollectionOfSegments:
    """``segment in "Buenos Aires"`` is a substring test: ``"Aires"`` and ``"Bu"`` would pass (#68)."""

    @pytest.mark.parametrize("allowed", ["Buenos Aires", "xyz", "all", "", b"Buenos Aires"])
    def test_segment_allowed_refuses_a_str(self, allowed) -> None:
        with pytest.raises(TypeError, match="not a str"):
            segment_allowed("Aires", allowed)

    @pytest.mark.parametrize("allowed", ["all", "Buenos Aires", b"all"])
    def test_allows_every_segment_refuses_a_str(self, allowed) -> None:
        with pytest.raises(TypeError, match="collection"):
            allows_every_segment(allowed)

    def test_the_message_says_what_to_pass(self) -> None:
        with pytest.raises(TypeError) as err:
            segment_allowed("Aires", "Buenos Aires")
        assert "{'Buenos Aires'}" in str(err.value)

    @pytest.mark.parametrize(
        "allowed",
        [{"Buenos Aires"}, ["Buenos Aires"], ("Buenos Aires",), frozenset({"Buenos Aires"}),
         {"Buenos Aires": 1}.keys()],
    )  # fmt: skip
    def test_every_real_collection_still_works(self, allowed) -> None:
        assert segment_allowed("Buenos Aires", allowed) is True
        assert segment_allowed("Aires", allowed) is False
        assert segment_allowed("Buenos", allowed) is False

    def test_none_and_the_empty_collection_keep_their_meaning(self) -> None:
        assert segment_allowed("x", None) is True
        assert segment_allowed("x", set()) is False
        assert segment_allowed("x", []) is False
