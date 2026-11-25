"""`compare_metrics` judges every recorded metric, not only the numeric ones (#68).

A metric that was a string, a flag-like value or a nested payload used to be skipped when both
sides had one, so a changed string read REPRODUCED. They are compared by value now, with the same
tolerance for any number inside a nested payload.
"""

from __future__ import annotations

import math

import pytest

from pmlab.reproducibility import compare_metrics


def _one(recorded, reproduced, **kwargs):
    (result,) = compare_metrics({"m": recorded}, {"m": reproduced}, **kwargs)
    return result


class TestScalarsThatAreNotNumbers:
    def test_the_same_string_matches(self) -> None:
        got = _one("hist", "hist")
        assert (
            got.ok and got.abs_diff is None and (got.recorded, got.reproduced) == ("hist", "hist")
        )

    def test_a_changed_string_does_not(self) -> None:
        assert not _one("lightgbm", "sklearn").ok

    def test_case_and_whitespace_are_part_of_a_string(self) -> None:
        assert not _one("GO", "go").ok and not _one("a", "a ").ok

    @pytest.mark.parametrize(
        ("recorded", "reproduced"), [(1.5, "1.5"), ("1.5", 1.5), (3, "3"), ("x", 3)]
    )
    def test_a_number_that_became_text_or_back_does_not(self, recorded, reproduced) -> None:
        assert not _one(recorded, reproduced).ok

    def test_a_flag_that_flipped_does_not(self) -> None:
        assert not _one(True, False).ok
        assert _one(True, True).ok

    def test_a_missing_or_null_string_still_fails_as_before(self) -> None:
        assert not _one("text", None).ok
        results = compare_metrics({"m": "text"}, {})
        assert [r.ok for r in results] == [False]


class TestNestedPayloads:
    def test_equal_payloads_match(self) -> None:
        assert _one({"a": [1, 2], "b": "x"}, {"a": [1, 2], "b": "x"}).ok

    def test_a_changed_value_deep_inside_does_not(self) -> None:
        assert not _one({"a": {"b": [1, 2, 3]}}, {"a": {"b": [1, 2, 4]}}).ok

    def test_a_different_key_set_does_not(self) -> None:
        assert not _one({"a": 1}, {"a": 1, "b": 2}).ok
        assert not _one({"a": 1, "b": 2}, {"a": 1}).ok

    def test_a_different_length_does_not(self) -> None:
        assert not _one([1, 2], [1, 2, 3]).ok and not _one([1, 2, 3], [1, 2]).ok

    def test_a_dict_is_not_a_list(self) -> None:
        assert not _one({"a": 1}, [("a", 1)]).ok

    def test_numbers_inside_use_the_tolerance(self) -> None:
        recorded = {"by_segment": {"a": 0.5, "b": [1.0, 2.0]}}
        near = {"by_segment": {"a": 0.5 + 1e-12, "b": [1.0 + 1e-12, 2.0]}}
        far = {"by_segment": {"a": 0.5 + 1e-3, "b": [1.0, 2.0]}}
        assert _one(recorded, near).ok
        assert not _one(recorded, far).ok
        assert _one(recorded, far, rtol=1e-2).ok  # the caller's tolerance reaches them

    def test_integers_inside_must_be_equal(self) -> None:
        assert not _one({"n": 100}, {"n": 101}, rtol=0.5).ok

    def test_a_null_inside_must_come_back_null(self) -> None:
        assert _one({"x": None}, {"x": None}).ok
        assert not _one({"x": None}, {"x": 0.5}).ok

    def test_a_nan_inside_is_not_a_match(self) -> None:
        assert not _one([math.nan], [math.nan]).ok

    def test_tuples_and_lists_are_both_sequences(self) -> None:
        assert _one([1, 2], (1, 2)).ok


class TestWhatWasAlreadyJudgedIsUnchanged:
    def test_floats_ints_and_nulls(self) -> None:
        got = {
            c.name: c
            for c in compare_metrics(
                {"f": 1.0, "i": 5, "n": None, "gone": 2.0, "off": 1.0},
                {"f": 1.0 + 1e-12, "i": 5.0, "n": None, "off": 1.1},
            )
        }
        assert got["f"].ok and got["i"].ok and got["n"].ok
        assert not got["gone"].ok and not got["off"].ok
        assert got["off"].abs_diff == pytest.approx(0.1)

    def test_every_recorded_metric_is_listed_once_and_in_order(self) -> None:
        recorded = {"z": "s", "a": 1.0, "m": {"k": 1}, "b": None}
        assert [c.name for c in compare_metrics(recorded, dict(recorded))] == ["z", "a", "m", "b"]
