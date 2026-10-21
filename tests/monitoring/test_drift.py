"""Tests for drift monitoring: PSI / KL on feature and prediction distributions.

The champion was gated on a historical panel. This says whether the world it is
being fed today still looks like the world it was trained on.

Traps this suite pins deliberately:

1. **Bins come from the reference only.** Bin each sample on its *own* quantiles
   and both come out ~uniform, so a distribution shifted wholesale scores ~0 —
   drift hides exactly when it is largest. Take the shared edges from the
   *current* sample and the statistic depends on the sample being judged.
2. **Empty bins must not blow up.** A value range the reference never saw is the
   most important drift there is, and it is also the case where a naive
   ``p * log(p / 0)`` returns ``inf``. It must come back large *and finite*.
3. **KL is directional.** ``kl`` is KL(current || reference), asserted against a
   hand computation, not just "differs when swapped".
4. **Missing is drift.** A feature going 0% -> 40% missing is a change in the
   world even though every *remaining* value looks fine, so dropping NaNs
   silently would report "no drift".
5. **Unseen categories are drift**, not rows to discard.
6. **Too little data is "insufficient", not a number.**
7. **No lookahead.** The current window may only contain snapshots captured
   strictly before the cutoff; later rows must not change the report at all.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmlab.data import FeatureSnapshotStore, build_panel_from_snapshots
from pmlab.monitoring.drift import (
    DriftConfig,
    DriftReport,
    DriftThresholds,
    FeatureDrift,
    drift_report,
    drift_report_from_store,
)
from pmlab.tracking import LocalJSONTracker

N = 2000
# Isolates one signal: the other two can never fire.
_UNREACHABLE = 1e9


def _normal(n: int = N, loc: float = 0.0, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc, scale, n)


def _one(report: DriftReport, name: str) -> FeatureDrift:
    (match,) = [f for f in report.features if f.name == name]
    return match


def _psi_only(warn: float = 0.10, alert: float = 0.25) -> DriftConfig:
    """Config where only PSI can classify a feature."""
    return DriftConfig(
        thresholds=DriftThresholds(
            psi_warn=warn,
            psi_alert=alert,
            kl_warn=_UNREACHABLE,
            kl_alert=_UNREACHABLE,
            missing_warn=_UNREACHABLE,
            missing_alert=_UNREACHABLE,
        )
    )


def _walk(obj: Any) -> list[Any]:
    """Every leaf of a nested to_dict() payload."""
    if isinstance(obj, dict):
        return [leaf for v in obj.values() for leaf in _walk(v)]
    if isinstance(obj, (list, tuple)):
        return [leaf for v in obj for leaf in _walk(v)]
    return [obj]


# --------------------------------------------------------------------------- #
# Statistic properties
# --------------------------------------------------------------------------- #


class TestStatisticProperties:
    def test_psi_of_a_sample_against_itself_is_exactly_zero(self):
        x = _normal()
        fd = _one(drift_report({"x": x}, {"x": x}), "x")
        assert fd.psi == 0.0
        assert fd.kl == 0.0
        assert fd.status == "ok"

    def test_categorical_psi_against_itself_is_exactly_zero(self):
        cats = ["a"] * 300 + ["b"] * 200 + ["c"] * 100
        fd = _one(drift_report({"c": cats}, {"c": cats}), "c")
        assert fd.psi == 0.0
        assert fd.kl == 0.0

    @pytest.mark.parametrize("seed", range(12))
    def test_psi_and_kl_are_never_negative(self, seed):
        rng = np.random.default_rng(seed)
        ref = rng.normal(0, 1, 800)
        cur = rng.normal(rng.uniform(-2, 2), rng.uniform(0.3, 3), 800)
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi is not None and fd.psi >= 0.0
        assert fd.kl is not None and fd.kl >= 0.0

    def test_fresh_sample_from_same_distribution_is_ok(self):
        ref, cur = _normal(5000, seed=1), _normal(5000, seed=2)
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi is not None and fd.psi < 0.10
        assert fd.status == "ok"
        assert fd.drivers == ()

    def test_one_sd_mean_shift_is_in_the_alert_band(self):
        ref, cur = _normal(seed=1), _normal(loc=1.0, seed=2)
        fd = _one(drift_report({"x": ref}, {"x": cur}, config=_psi_only()), "x")
        assert fd.psi is not None and fd.psi > 0.25
        assert fd.status == "alert"
        assert fd.drivers == ("psi",)

    def test_small_mean_shift_is_in_the_warn_band(self):
        ref, cur = _normal(5000, seed=1), _normal(5000, loc=0.35, seed=2)
        fd = _one(drift_report({"x": ref}, {"x": cur}, config=_psi_only()), "x")
        assert fd.psi is not None and 0.10 <= fd.psi < 0.25
        assert fd.status == "warn"

    def test_unseen_rate_is_the_share_of_current_outside_the_reference_range(self):
        ref = _normal(seed=1)
        cur = _normal(seed=1)
        cur[:500] = ref.max() + 1.0  # 25% above anything seen in training
        cur[500:600] = ref.min() - 1.0  # 5% below it
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.unseen_rate == pytest.approx(0.30)
        assert fd.n_unseen_categories == 0 and fd.unseen_categories == ()

    def test_values_exactly_at_the_reference_extremes_are_not_unseen(self):
        ref = _normal(seed=1)
        assert _one(drift_report({"x": ref}, {"x": ref}), "x").unseen_rate == 0.0

    def test_scale_change_with_same_mean_is_detected(self):
        ref, cur = _normal(seed=1), _normal(scale=3.0, seed=2)
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi is not None and fd.psi > 0.25

    def test_larger_shift_scores_higher(self):
        ref = _normal(seed=1)
        scores = [
            _one(drift_report({"x": ref}, {"x": _normal(loc=s, seed=2)}), "x").psi
            for s in (0.0, 0.5, 1.0, 2.0)
        ]
        assert scores == sorted(scores)
        assert scores[0] < scores[-1]


# --------------------------------------------------------------------------- #
# Binning is defined on the reference only
# --------------------------------------------------------------------------- #


class TestReferenceOnlyBinning:
    def test_wholesale_shift_is_not_re_binned_onto_itself(self):
        # Same shape, shifted by 3 sd. Bin each sample on its own quantiles and both
        # come out ~uniform, so PSI is ~0 — the trap. On the reference's edges
        # nearly all the current mass lands in one bin.
        ref = _normal(seed=1)
        cur = _normal(loc=3.0, seed=2)

        def own_quantile_shares(x: np.ndarray) -> np.ndarray:
            edges = np.quantile(x, np.linspace(0, 1, 11)[1:-1])
            return np.bincount(np.searchsorted(edges, x), minlength=10) / len(x)

        r, c = own_quantile_shares(ref), own_quantile_shares(cur)
        assert float(np.sum((c - r) * np.log(c / r))) < 0.01  # what the trap reports

        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi is not None and fd.psi > 2.0

    def test_matches_an_independent_oracle_built_from_reference_quantiles(self):
        # Bins: below the reference minimum | ten-ish quantile bins inside its range |
        # above its maximum. The oracle builds them with np.histogram, then applies
        # the documented floor-and-renormalise by hand.
        ref, cur = _normal(seed=1), _normal(loc=0.4, scale=1.2, seed=2)
        assert (cur > ref.max()).any()  # the outer bin is exercised
        interior = np.quantile(ref, np.linspace(0, 1, 11)[1:-1])
        full = np.concatenate(
            [[-np.inf, ref.min()], interior, [np.nextafter(ref.max(), np.inf)], [np.inf]]
        )
        r = np.histogram(ref, bins=full)[0] / len(ref)
        c = np.histogram(cur, bins=full)[0] / len(cur)
        keep = (r + c) > 0
        r, c = np.maximum(r[keep], 1e-4), np.maximum(c[keep], 1e-4)
        r, c = r / r.sum(), c / c.sum()
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi == pytest.approx(float(np.sum((c - r) * np.log(c / r))), rel=1e-6)
        assert fd.kl == pytest.approx(float(np.sum(c * np.log(c / r))), rel=1e-6)

    def test_inside_the_reference_range_it_is_plain_quantile_binning(self):
        # With nothing outside the reference range the outer bins are empty on both
        # sides and drop out, leaving exactly the textbook quantile-bin PSI.
        ref = _normal(seed=1)
        cur = np.clip(_normal(loc=0.4, scale=1.2, seed=2), ref.min(), ref.max())
        edges = np.quantile(ref, np.linspace(0, 1, 11)[1:-1])
        full = np.concatenate([[-np.inf], edges, [np.inf]])
        r = np.histogram(ref, bins=full)[0] / len(ref)
        c = np.histogram(cur, bins=full)[0] / len(cur)
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi == pytest.approx(float(np.sum((c - r) * np.log(c / r))), rel=1e-6)
        assert fd.kl == pytest.approx(float(np.sum(c * np.log(c / r))), rel=1e-6)

    def test_n_bins_is_configurable(self):
        ref, cur = _normal(seed=1), _normal(loc=0.5, seed=2)
        ten = _one(drift_report({"x": ref}, {"x": cur}), "x").psi
        five = _one(drift_report({"x": ref}, {"x": cur}, config=DriftConfig(n_bins=5)), "x").psi
        assert ten != five

    def test_heavily_tied_reference_does_not_break_binning(self):
        # 70% zeros: the lower quantile edges all collapse onto 0.
        ref = np.concatenate([np.zeros(700), _normal(300, loc=5, seed=1)])
        fd = _one(drift_report({"x": ref}, {"x": ref}), "x")
        assert fd.psi == 0.0
        shifted = np.concatenate([np.zeros(300), _normal(700, loc=5, seed=1)])
        assert _one(drift_report({"x": ref}, {"x": shifted}), "x").psi > 0.25

    def test_constant_reference_flags_a_value_it_never_contained(self):
        ref = np.full(500, 3.0)
        fd = _one(drift_report({"x": ref}, {"x": np.full(500, 9.0)}), "x")
        assert fd.psi is not None and math.isfinite(fd.psi) and fd.psi > 1.0


# --------------------------------------------------------------------------- #
# Mass outside the reference range always gets its own bin
# --------------------------------------------------------------------------- #


# Quantile edges of a constant or heavily tied reference collapse. If the
# first/last bins were open-ended, everything below the reference minimum would
# share a bin with the reference's own mass and score 0.
def _psi_alert(fd: FeatureDrift) -> bool:
    return fd.status == "alert" and "psi" in fd.drivers


class TestOutOfRangeMass:
    def test_constant_reference_flags_mass_below_it(self):
        fd = _one(drift_report({"x": [1.0] * 500}, {"x": [1.0] * 250 + [0.0] * 250}), "x")
        assert _psi_alert(fd)
        assert fd.psi is not None and fd.psi > 1.0
        assert fd.unseen_rate == pytest.approx(0.5)

    def test_constant_reference_flags_mass_above_it(self):
        fd = _one(drift_report({"x": [1.0] * 500}, {"x": [1.0] * 250 + [2.0] * 250}), "x")
        assert _psi_alert(fd)

    def test_constant_zero_reference_against_negative_values(self):
        fd = _one(drift_report({"x": [0.0] * 500}, {"x": [0.0] * 250 + [-5.0] * 250}), "x")
        assert _psi_alert(fd)

    def test_below_and_above_score_the_same_for_a_constant_reference(self):
        below = _one(drift_report({"x": [1.0] * 500}, {"x": [1.0] * 250 + [0.0] * 250}), "x")
        above = _one(drift_report({"x": [1.0] * 500}, {"x": [1.0] * 250 + [2.0] * 250}), "x")
        assert below.psi == pytest.approx(above.psi)
        assert below.kl == pytest.approx(above.kl)

    def test_tied_reference_flags_a_current_sample_entirely_below_it(self):
        # 95% zeros + 5% normal; everything now far below the reference minimum.
        ref = np.concatenate([np.zeros(950), _normal(50, seed=1)])
        fd = _one(drift_report({"x": ref}, {"x": np.full(500, -5.0)}), "x")
        assert _psi_alert(fd)
        assert fd.psi is not None and fd.psi > 5.0  # was 0.255: the zeros' bin absorbed it

    def test_mirrored_tied_reference_flags_a_current_sample_entirely_above_it(self):
        ref = -np.concatenate([np.zeros(950), _normal(50, seed=1)])
        fd = _one(drift_report({"x": ref}, {"x": np.full(500, 5.0)}), "x")
        assert _psi_alert(fd)
        assert fd.psi is not None and fd.psi > 5.0

    def test_predictions_constant_reference_flags_a_move_in_either_direction(self):
        for ref_value, cur_value in ((0.7, 0.3), (0.3, 0.7)):
            report = drift_report(
                {},
                {},
                reference_predictions=[ref_value] * 500,
                current_predictions=[cur_value] * 500,
            )
            assert report.prediction is not None
            assert _psi_alert(report.prediction), (ref_value, cur_value)
            assert report.status == "alert"

    def test_lower_and_upper_tails_are_symmetric_for_continuous_data(self):
        ref = _normal(seed=1)
        cur = _normal(seed=2)
        cur[:200] = ref.min() - 1.0
        up = _one(drift_report({"x": ref}, {"x": cur}), "x")
        down = _one(drift_report({"x": -ref}, {"x": -cur}), "x")
        # mirrored: the low tail became the high tail
        assert up.psi == pytest.approx(down.psi, rel=1e-9)
        assert up.kl == pytest.approx(down.kl, rel=1e-9)

    def test_mass_just_outside_the_range_is_not_merged_with_the_nearest_bin(self):
        ref = _normal(seed=1)
        inside, outside = _normal(seed=2), _normal(seed=2)
        inside[:100] = ref.min()  # at the minimum: in the first bin
        outside[:100] = ref.min() - 1e-9  # a hair below it: its own bin
        a = _one(drift_report({"x": ref}, {"x": inside}), "x")
        b = _one(drift_report({"x": ref}, {"x": outside}), "x")
        assert a.unseen_rate == 0.0 and b.unseen_rate == pytest.approx(0.05)
        assert b.psi is not None and a.psi is not None and b.psi > a.psi + 0.1

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda: np.full(500, 3.0), id="constant"),
            pytest.param(
                lambda: np.concatenate([np.zeros(700), _normal(300, loc=5, seed=1)]), id="tied"
            ),
            pytest.param(lambda: np.random.default_rng(0).integers(0, 4, 800), id="discrete"),
            pytest.param(lambda: _normal(seed=1), id="continuous"),
        ],
    )
    def test_a_sample_against_itself_is_exactly_zero(self, make):
        x = make()
        fd = _one(drift_report({"x": x}, {"x": x}), "x")
        assert fd.psi == 0.0 and fd.kl == 0.0
        assert fd.status == "ok" and fd.unseen_rate == 0.0

    def test_a_discrete_reference_flags_a_value_beyond_its_largest(self):
        ref = np.random.default_rng(0).integers(0, 4, 800)
        cur = np.random.default_rng(1).integers(0, 4, 800)
        cur[:400] = 9
        assert _psi_alert(_one(drift_report({"x": ref}, {"x": cur}), "x"))

    def test_out_of_range_mass_alone_decides_the_status_at_the_default_thresholds(self):
        # Why `unseen_rate` needs no thresholds of its own: the outer bins are empty
        # in the reference, so the out-of-range share already moves PSI by itself.
        ref = _normal(5000, seed=1)

        def status_for(share: float) -> str:
            cur = _normal(5000, seed=1).copy()  # same draw: in-range part is identical
            cur[: int(share * len(cur))] = ref.max() + 1.0
            return _one(drift_report({"x": ref}, {"x": cur}), "x").status

        assert status_for(0.001) == "ok"
        assert status_for(0.02) == "warn"
        assert status_for(0.05) == "alert"
        assert status_for(0.20) == "alert"

    def test_a_coarse_epsilon_can_hide_out_of_range_mass_so_unseen_rate_still_reports_it(self):
        # The flip side, and why `unseen_rate` stays on the report: a large epsilon
        # makes the empty outer bin cheap, so 5% out of range no longer moves the status.
        ref = _normal(5000, seed=1)
        cur = _normal(5000, seed=1).copy()
        cur[:250] = ref.max() + 1.0
        fd = _one(drift_report({"x": ref}, {"x": cur}, config=DriftConfig(epsilon=1e-2)), "x")
        assert fd.status == "ok"
        assert fd.unseen_rate == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# Empty bins: large but finite
# --------------------------------------------------------------------------- #


class TestSmoothing:
    def test_range_the_reference_never_saw_is_large_but_finite(self):
        ref = _normal(seed=1)
        cur = _normal(seed=2) + 100.0  # entirely outside anything seen in training
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi is not None and fd.kl is not None
        assert math.isfinite(fd.psi) and math.isfinite(fd.kl)
        assert fd.psi > 5.0
        assert fd.status == "alert"

    def test_unseen_range_report_is_strict_json(self):
        report = drift_report({"x": _normal(seed=1)}, {"x": _normal(seed=2) + 100.0})
        json.dumps(report.to_dict(), allow_nan=False)  # raises on inf / NaN

    def test_epsilon_controls_the_magnitude_of_an_unseen_bin(self):
        ref, cur = _normal(seed=1), _normal(seed=2) + 100.0
        loose = _one(drift_report({"x": ref}, {"x": cur}, config=DriftConfig(epsilon=1e-2)), "x")
        tight = _one(drift_report({"x": ref}, {"x": cur}, config=DriftConfig(epsilon=1e-6)), "x")
        assert loose.psi is not None and tight.psi is not None
        assert loose.psi < tight.psi
        assert math.isfinite(tight.psi)

    def test_smoothing_is_floor_then_renormalise(self):
        # The documented convention, computed by hand: proportions are floored at
        # epsilon and the vector rescaled to sum to one. epsilon is large here so
        # that skipping the rescale would visibly move both numbers.
        eps = 1e-2
        ref = ["a"] * 50 + ["b"] * 50
        cur = ["a"] * 50 + ["b"] * 30 + ["new"] * 20
        r = np.array([0.5, 0.5, 0.0])  # a, b, <unseen>
        r = np.maximum(r, eps)
        r = r / r.sum()
        c = np.array([0.5, 0.3, 0.2])
        c = np.maximum(c, eps)
        c = c / c.sum()
        fd = _one(
            drift_report({"f": ref}, {"f": cur}, config=DriftConfig(epsilon=eps, min_samples=100)),
            "f",
        )
        assert fd.psi == pytest.approx(float(np.sum((c - r) * np.log(c / r))), rel=1e-9)
        assert fd.kl == pytest.approx(float(np.sum(c * np.log(c / r))), rel=1e-9)

    @pytest.mark.parametrize("bad", [0.0, -1e-4, 1.0, 2.0, float("nan")])
    def test_epsilon_must_be_a_proper_fraction(self, bad):
        with pytest.raises(ValueError, match="epsilon"):
            DriftConfig(epsilon=bad)


# --------------------------------------------------------------------------- #
# KL direction
# --------------------------------------------------------------------------- #


class TestKLDirection:
    _REF = ["a"] * 500 + ["b"] * 500
    _CUR = ["a"] * 900 + ["b"] * 100

    def test_kl_is_current_relative_to_reference(self):
        c, r = np.array([0.9, 0.1]), np.array([0.5, 0.5])
        kl_cur_ref = float(np.sum(c * np.log(c / r)))  # 0.3681...
        kl_ref_cur = float(np.sum(r * np.log(r / c)))  # 0.5108...
        assert kl_cur_ref != pytest.approx(kl_ref_cur, abs=0.05)

        fd = _one(drift_report({"f": self._REF}, {"f": self._CUR}), "f")
        assert fd.kl == pytest.approx(kl_cur_ref, rel=1e-9)
        assert fd.kl != pytest.approx(kl_ref_cur, abs=0.05)

    def test_swapping_the_arguments_changes_kl_but_not_psi(self):
        fwd = _one(drift_report({"f": self._REF}, {"f": self._CUR}), "f")
        rev = _one(drift_report({"f": self._CUR}, {"f": self._REF}), "f")
        assert fwd.kl != pytest.approx(rev.kl, abs=0.05)
        assert fwd.psi == pytest.approx(rev.psi, rel=1e-9)

    def test_psi_is_the_sum_of_both_kl_directions(self):
        fwd = _one(drift_report({"f": self._REF}, {"f": self._CUR}), "f")
        rev = _one(drift_report({"f": self._CUR}, {"f": self._REF}), "f")
        assert fwd.kl is not None and rev.kl is not None
        assert fwd.psi == pytest.approx(fwd.kl + rev.kl, rel=1e-9)

    def test_kl_punishes_current_mass_where_the_reference_has_none(self):
        # New category carries 20% of live traffic. KL(cur || ref) is large;
        # the reverse direction would barely notice.
        ref = ["a"] * 500 + ["b"] * 500
        cur = ["a"] * 400 + ["b"] * 400 + ["new"] * 200
        fwd = _one(drift_report({"f": ref}, {"f": cur}), "f")
        rev = _one(drift_report({"f": cur}, {"f": ref}), "f")
        assert fwd.kl is not None and rev.kl is not None
        assert fwd.kl > 1.0
        assert rev.kl < fwd.kl


# --------------------------------------------------------------------------- #
# Categorical features
# --------------------------------------------------------------------------- #


class TestCategorical:
    def test_unseen_category_is_drift_not_dropped(self):
        ref = ["a"] * 500 + ["b"] * 300 + ["c"] * 200
        cur = ["a"] * 350 + ["b"] * 210 + ["c"] * 140 + ["zzz"] * 300
        fd = _one(drift_report({"f": ref}, {"f": cur}), "f")
        assert fd.kind == "categorical"
        assert fd.unseen_categories == ("zzz",)
        assert fd.n_unseen_categories == 1
        assert fd.unseen_rate == pytest.approx(0.3)
        assert fd.psi is not None and fd.psi > 0.25
        assert fd.status == "alert"

    def test_a_small_share_of_unseen_categories_still_registers(self):
        ref = ["a"] * 500 + ["b"] * 500
        cur = ["a"] * 490 + ["b"] * 490 + ["new"] * 20
        fd = _one(drift_report({"f": ref}, {"f": cur}), "f")
        assert fd.unseen_rate == pytest.approx(0.02)
        assert fd.status in ("warn", "alert")

    def test_unseen_categories_are_pooled_and_all_counted(self):
        ref = ["a"] * 500 + ["b"] * 500
        cur = ["a"] * 400 + ["b"] * 400 + ["x"] * 40 + ["y"] * 100 + ["z"] * 60
        fd = _one(drift_report({"f": ref}, {"f": cur}), "f")
        assert fd.n_unseen_categories == 3
        assert fd.unseen_categories == ("y", "z", "x")  # by current frequency, not by name
        assert fd.unseen_rate == pytest.approx(0.2)

    def test_unseen_categories_with_equal_counts_are_listed_alphabetically(self):
        ref = ["a"] * 500 + ["b"] * 500
        cur = ["a"] * 400 + ["b"] * 400 + ["q"] * 100 + ["p"] * 100
        fd = _one(drift_report({"f": ref}, {"f": cur}), "f")
        assert fd.unseen_categories == ("p", "q")

    def test_listing_of_unseen_categories_is_capped_but_counted_in_full(self):
        ref = ["a"] * 500 + ["b"] * 500
        cur = ["a"] * 500 + [f"n{i:03d}" for i in range(100)]
        fd = _one(drift_report({"f": ref}, {"f": cur}), "f")
        assert fd.n_unseen_categories == 100
        assert len(fd.unseen_categories) < 100

    def test_category_that_vanished_from_current_is_drift_too(self):
        ref = ["a"] * 400 + ["b"] * 400 + ["c"] * 200
        cur = ["a"] * 500 + ["b"] * 500
        fd = _one(drift_report({"f": ref}, {"f": cur}), "f")
        assert fd.unseen_categories == ()
        assert fd.psi is not None and fd.psi > 0.10

    def test_int_and_float_spellings_of_one_category_are_the_same_category(self):
        # An int column becomes float64 the moment a NaN appears; 1 vs 1.0 must
        # not be reported as a brand-new category.
        ref = pd.Series([1, 2, 3] * 200)
        cur = pd.Series([1.0, 2.0, 3.0, np.nan] * 150)
        cfg = DriftConfig(categorical=("f",))
        fd = _one(drift_report({"f": ref}, {"f": cur}, config=cfg), "f")
        assert fd.kind == "categorical"
        assert fd.unseen_categories == ()
        assert fd.psi is not None and fd.psi < 0.01

    def test_one_and_one_point_zero_and_true_are_one_category(self):
        # Python hashes 1, 1.0 and True alike, so the category counter merges them.
        # That is intended: an int column turns float when a NaN appears, and a
        # bool column round-trips through ints. Anything else is a different category.
        ref = pd.Series([1, 2] * 300, dtype=object)
        cur = pd.Series([True, 2.0] * 200 + [1.0, 2] * 100, dtype=object)
        cfg = DriftConfig(categorical=("f",))
        fd = _one(drift_report({"f": ref}, {"f": cur}, config=cfg), "f")
        assert fd.unseen_categories == () and fd.n_unseen_categories == 0
        assert fd.psi == 0.0
        # ...but the string "1" is not the number 1.
        as_text = _one(drift_report({"f": ref}, {"f": ["1", "2"] * 300}, config=cfg), "f")
        assert as_text.n_unseen_categories == 2

    def test_missing_categorical_values_are_counted_not_binned(self):
        ref = ["a"] * 500 + ["b"] * 500
        cur = ["a"] * 250 + ["b"] * 250 + [None] * 250 + [np.nan] * 250
        fd = _one(drift_report({"f": ref}, {"f": cur}), "f")
        assert fd.n_current_valid == 500
        assert fd.current_missing_rate == pytest.approx(0.5)
        assert fd.psi is not None and fd.psi < 0.01


# --------------------------------------------------------------------------- #
# Numeric vs categorical detection
# --------------------------------------------------------------------------- #


class TestKindDetection:
    def _kinds(self, **cols: Any) -> dict[str, str]:
        report = drift_report(cols, cols)
        return {f.name: f.kind for f in report.features}

    def test_dtypes_are_classified_from_the_reference(self):
        kinds = self._kinds(
            f=_normal(200),
            i=np.arange(200),
            s=["a", "b"] * 100,
            b=[True, False] * 100,
            o=pd.Series([1.5, None, 2.5, 3.5] * 50, dtype=object),
        )
        assert kinds == {
            "b": "categorical",
            "f": "numeric",
            "i": "numeric",
            "o": "numeric",
            "s": "categorical",
        }

    def test_categorical_override_for_integer_codes(self):
        codes = [0, 1, 2, 3] * 200
        assert self._kinds(c=codes)["c"] == "numeric"
        report = drift_report({"c": codes}, {"c": codes}, config=DriftConfig(categorical=("c",)))
        assert _one(report, "c").kind == "categorical"

    def test_numeric_override_parses_numeric_strings(self):
        ref = [str(v) for v in _normal(300, seed=1)]
        cur = [str(v) for v in _normal(300, loc=3.0, seed=2)]
        report = drift_report({"x": ref}, {"x": cur}, config=DriftConfig(numeric=("x",)))
        fd = _one(report, "x")
        assert fd.kind == "numeric"
        assert fd.psi is not None and fd.psi > 1.0

    def test_numeric_override_on_non_numeric_data_names_the_feature(self):
        with pytest.raises(ValueError, match="'x'"):
            drift_report(
                {"x": ["a", "b"] * 100}, {"x": ["a", "b"] * 100}, config=DriftConfig(numeric=("x",))
            )

    def test_feature_cannot_be_declared_both_ways(self):
        with pytest.raises(ValueError, match="both"):
            DriftConfig(numeric=("x",), categorical=("x",))

    def test_infinities_count_as_missing_in_a_numeric_feature(self):
        ref = _normal(seed=1)
        cur = np.concatenate([_normal(1500, seed=2), np.full(500, np.inf)])
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.current_missing_rate == pytest.approx(0.25)
        assert fd.psi is not None and math.isfinite(fd.psi)


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #


class TestMissingness:
    def test_missing_rate_rising_from_zero_to_forty_percent_is_an_alert(self):
        ref = _normal(seed=1)
        cur = _normal(seed=2)
        cur[: int(0.4 * N)] = np.nan
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.reference_missing_rate == 0.0
        assert fd.current_missing_rate == pytest.approx(0.4)
        assert fd.missing_rate_change == pytest.approx(0.4)
        assert fd.status == "alert"
        assert "missing_rate" in fd.drivers

    def test_values_that_remain_look_fine_so_only_missingness_flags_it(self):
        ref = _normal(seed=1)
        cur = _normal(seed=1)  # identical values...
        cur[::2] = np.nan  # ...with half of them blanked out
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi is not None and fd.psi < 0.10
        assert fd.status == "alert"
        assert fd.drivers == ("missing_rate",)

    def test_missing_rate_falling_is_also_a_change(self):
        ref = _normal(seed=1)
        ref[: int(0.4 * N)] = np.nan
        fd = _one(drift_report({"x": ref}, {"x": _normal(seed=2)}), "x")
        assert fd.missing_rate_change == pytest.approx(-0.4)
        assert fd.status == "alert"

    def test_nan_never_leaks_into_the_statistic(self):
        ref, cur = _normal(seed=1), _normal(seed=2)
        ref[:100] = np.nan
        cur[:300] = np.nan
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.psi is not None and math.isfinite(fd.psi)
        assert fd.n_reference_valid == N - 100
        assert fd.n_current_valid == N - 300

    def test_a_column_that_went_entirely_missing_is_an_alert_not_a_crash(self):
        fd = _one(drift_report({"x": _normal(seed=1)}, {"x": np.full(N, np.nan)}), "x")
        assert fd.psi is None and fd.kl is None
        assert fd.current_missing_rate == 1.0
        assert fd.status == "alert"

    def test_a_feed_that_dies_into_all_none_is_still_judged_as_numeric(self):
        # An all-None object column infers as "empty"; the kind must come from the
        # reference, or this would be mis-read as categorical and mis-reported.
        fd = _one(drift_report({"x": _normal(seed=1)}, {"x": [None] * N}), "x")
        assert fd.kind == "numeric"
        assert fd.psi is None
        assert fd.current_missing_rate == 1.0
        assert fd.status == "alert"

    def test_small_missing_rate_change_stays_below_the_default_band(self):
        ref = _normal(seed=1)
        cur = _normal(seed=1)
        cur[: int(0.02 * N)] = np.nan
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.status == "ok"


# --------------------------------------------------------------------------- #
# Too-small samples
# --------------------------------------------------------------------------- #


class TestInsufficientData:
    def test_small_current_sample_is_insufficient_not_a_number(self):
        fd = _one(drift_report({"x": _normal(seed=1)}, {"x": _normal(20, seed=2)}), "x")
        assert fd.status == "insufficient"
        assert fd.psi is None and fd.kl is None
        assert fd.unseen_rate is None
        assert fd.n_current == 20  # the evidence is still shown

    def test_small_reference_sample_is_insufficient_too(self):
        fd = _one(drift_report({"x": _normal(20, seed=1)}, {"x": _normal(seed=2)}), "x")
        assert fd.status == "insufficient"
        assert fd.psi is None

    def test_min_samples_is_configurable(self):
        cfg = DriftConfig(min_samples=10)
        fd = _one(
            drift_report({"x": _normal(20, seed=1)}, {"x": _normal(20, seed=2)}, config=cfg), "x"
        )
        assert fd.psi is not None
        assert fd.status != "insufficient"

    def test_boundary_is_inclusive(self):
        cfg = DriftConfig(min_samples=50)
        at = _one(
            drift_report({"x": _normal(50, seed=1)}, {"x": _normal(50, seed=2)}, config=cfg), "x"
        )
        below = _one(
            drift_report({"x": _normal(49, seed=1)}, {"x": _normal(50, seed=2)}, config=cfg), "x"
        )
        assert at.psi is not None
        assert below.psi is None

    def test_only_valid_values_count_towards_the_minimum(self):
        # 500 rows, but only 20 are usable.
        cur = np.full(500, np.nan)
        cur[:20] = _normal(20, seed=2)
        fd = _one(drift_report({"x": _normal(seed=1)}, {"x": cur}), "x")
        assert fd.psi is None
        assert fd.n_current == 500 and fd.n_current_valid == 20

    def test_insufficient_values_does_not_hide_a_missingness_alert(self):
        cur = np.full(500, np.nan)
        cur[:20] = _normal(20, seed=2)
        fd = _one(drift_report({"x": _normal(seed=1)}, {"x": cur}), "x")
        assert fd.status == "alert"  # 96% missing is known even if the shape is not
        assert fd.drivers == ("missing_rate",)

    def test_insufficient_never_reads_as_ok(self):
        fd = _one(drift_report({"x": _normal(seed=1)}, {"x": _normal(20, seed=2)}), "x")
        assert fd.status != "ok"
        assert fd.drivers == ("psi", "kl", "missing_rate")

    def test_min_samples_must_be_positive(self):
        with pytest.raises(ValueError, match="min_samples"):
            DriftConfig(min_samples=0)


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


class TestThresholds:
    def test_defaults_are_the_documented_conventions(self):
        t = DriftThresholds()
        assert (t.psi_warn, t.psi_alert) == (0.10, 0.25)
        assert (t.kl_warn, t.kl_alert) == (0.05, 0.125)
        assert (t.missing_warn, t.missing_alert) == (0.05, 0.20)

    def test_default_config_values(self):
        cfg = DriftConfig()
        assert (cfg.n_bins, cfg.min_samples, cfg.epsilon) == (10, 300, 1e-4)
        assert cfg.thresholds == DriftThresholds()

    def test_default_min_samples_keeps_an_unchanged_feature_quiet(self):
        # PSI of an unchanged feature is not 0: its noise floor is ~ (n_bins - 1) / n.
        # The default minimum is the smallest window where that stays well under the
        # 0.10 warn line; at 100 values a large share of unchanged windows would warn.
        def share_warning(n_current: int) -> float:
            cfg = DriftConfig(min_samples=n_current)
            hits = 0
            for seed in range(200):
                rng = np.random.default_rng(seed)
                ref, cur = rng.normal(0, 1, 2000), rng.normal(0, 1, n_current)
                hits += _one(drift_report({"x": ref}, {"x": cur}, config=cfg), "x").status != "ok"
            return hits / 200

        assert share_warning(DriftConfig().min_samples) <= 0.02
        assert share_warning(100) >= 0.20

    def test_thresholds_are_configurable(self):
        ref, cur = _normal(seed=1), _normal(loc=0.35, seed=2)
        loose = _one(drift_report({"x": ref}, {"x": cur}, config=_psi_only(5.0, 9.0)), "x")
        strict = _one(drift_report({"x": ref}, {"x": cur}, config=_psi_only(0.001, 0.002)), "x")
        assert loose.status == "ok"
        assert strict.status == "alert"

    def test_band_edges_are_inclusive(self):
        ref, cur = _normal(seed=1), _normal(loc=0.5, seed=2)
        psi = _one(drift_report({"x": ref}, {"x": cur}), "x").psi
        assert psi is not None
        at_warn = _one(drift_report({"x": ref}, {"x": cur}, config=_psi_only(psi, psi + 1)), "x")
        at_alert = _one(drift_report({"x": ref}, {"x": cur}, config=_psi_only(0.0, psi)), "x")
        assert at_warn.status == "warn"
        assert at_alert.status == "alert"

    def test_kl_can_drive_the_status_on_its_own(self):
        ref, cur = _normal(seed=1), _normal(loc=1.0, seed=2)
        cfg = DriftConfig(
            thresholds=DriftThresholds(
                psi_warn=_UNREACHABLE,
                psi_alert=_UNREACHABLE,
                kl_warn=0.01,
                kl_alert=0.02,
                missing_warn=_UNREACHABLE,
                missing_alert=_UNREACHABLE,
            )
        )
        fd = _one(drift_report({"x": ref}, {"x": cur}, config=cfg), "x")
        assert fd.status == "alert"
        assert fd.drivers == ("kl",)

    def test_missing_rate_thresholds_apply_to_the_absolute_change(self):
        ref = _normal(seed=1)
        cur = _normal(seed=2)
        cur[: int(0.10 * N)] = np.nan
        warn = DriftConfig(thresholds=DriftThresholds(missing_warn=0.05, missing_alert=0.50))
        fd = _one(drift_report({"x": ref}, {"x": cur}, config=warn), "x")
        assert fd.status == "warn"
        assert "missing_rate" in fd.drivers

    @pytest.mark.parametrize(
        ("ref_missing", "cur_missing", "expected"),
        [
            (0.10, 0.30, "alert"),  # 0.30 - 0.10 == 0.19999999999999998 in floats
            (0.30, 0.10, "alert"),  # the same step downwards
            (0.10, 0.15, "warn"),  # 0.15 - 0.10 == 0.04999999999999999
            (0.15, 0.10, "warn"),
            (0.00, 0.05, "warn"),
            (0.00, 0.20, "alert"),
            (0.10, 0.299, "warn"),  # a real shortfall is not rounded up
            (0.10, 0.149, "ok"),
        ],
    )
    def test_missing_rate_change_exactly_at_a_threshold_is_in_the_band(
        self, ref_missing, cur_missing, expected
    ):
        n = 1000
        ref, cur = _normal(n, seed=1), _normal(n, seed=1)
        ref[: round(ref_missing * n)] = np.nan
        cur[: round(cur_missing * n)] = np.nan
        fd = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert fd.status == expected
        if expected != "ok":
            assert fd.drivers == ("missing_rate",)

    def test_the_float_trap_is_real(self):
        # Why the tolerance exists: the documented line is >= 0.20.
        assert 0.30 - 0.10 < 0.20
        ref, cur = _normal(1000, seed=1), _normal(1000, seed=1)
        ref[:100] = np.nan
        cur[:300] = np.nan
        change = _one(drift_report({"x": ref}, {"x": cur}), "x").missing_rate_change
        assert change is not None and change < 0.20 and change == pytest.approx(0.20)

    def test_psi_and_kl_within_round_off_of_a_threshold_are_in_the_band(self):
        ref, cur = _normal(seed=1), _normal(loc=0.5, seed=2)
        base = _one(drift_report({"x": ref}, {"x": cur}), "x")
        assert base.psi is not None and base.kl is not None
        round_off = 1e-12  # what a threshold computed another way would differ by

        psi_cfg = _psi_only(base.psi + round_off, base.psi + 1.0)
        assert _one(drift_report({"x": ref}, {"x": cur}, config=psi_cfg), "x").status == "warn"
        psi_cfg = _psi_only(0.0, base.psi + round_off)
        assert _one(drift_report({"x": ref}, {"x": cur}, config=psi_cfg), "x").status == "alert"

        kl_cfg = DriftConfig(
            thresholds=DriftThresholds(
                psi_warn=_UNREACHABLE,
                psi_alert=_UNREACHABLE,
                kl_warn=base.kl + round_off,
                kl_alert=base.kl + 1.0,
                missing_warn=_UNREACHABLE,
                missing_alert=_UNREACHABLE,
            )
        )
        fd = _one(drift_report({"x": ref}, {"x": cur}, config=kl_cfg), "x")
        assert fd.status == "warn" and fd.drivers == ("kl",)

    def test_a_threshold_a_real_margin_above_the_value_is_not_reached(self):
        ref, cur = _normal(seed=1), _normal(loc=0.5, seed=2)
        psi = _one(drift_report({"x": ref}, {"x": cur}), "x").psi
        assert psi is not None
        cfg = _psi_only(psi + 1e-6, psi + 1.0)
        assert _one(drift_report({"x": ref}, {"x": cur}, config=cfg), "x").status == "ok"

    def test_a_single_cutoff_is_allowed(self):
        # warn == alert means "no warn band": a legitimate policy, not an error.
        t = DriftThresholds(psi_warn=0.2, psi_alert=0.2)
        assert t.psi_warn == t.psi_alert == 0.2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"psi_warn": 0.3, "psi_alert": 0.2},
            {"kl_warn": 0.3, "kl_alert": 0.2},
            {"missing_warn": 0.3, "missing_alert": 0.2},
            {"psi_warn": -0.1},
            {"psi_alert": float("nan")},
            {"kl_alert": float("inf")},
        ],
    )
    def test_invalid_thresholds_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            DriftThresholds(**kwargs)

    def test_n_bins_must_allow_a_comparison(self):
        with pytest.raises(ValueError, match="n_bins"):
            DriftConfig(n_bins=1)


# --------------------------------------------------------------------------- #
# Report shape, ordering, determinism
# --------------------------------------------------------------------------- #


class TestReport:
    def _panel(self, seed: int, shift: float = 0.0) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        return {
            "z": rng.normal(shift, 1, 800),
            "a": rng.normal(0, 1, 800),
            "m": rng.choice(["u", "v", "w"], 800),
        }

    def test_same_inputs_give_an_identical_report(self):
        ref, cur = self._panel(1), self._panel(2, shift=0.7)
        first = drift_report(ref, cur)
        second = drift_report(ref, cur)
        assert first == second
        assert json.dumps(first.to_dict()) == json.dumps(second.to_dict())

    def test_features_are_ordered_by_name_whatever_the_column_order(self):
        ref, cur = self._panel(1), self._panel(2)
        report = drift_report(ref, cur)
        flipped = drift_report(dict(reversed(list(ref.items()))), dict(reversed(list(cur.items()))))
        assert [f.name for f in report.features] == ["a", "m", "z"]
        assert report == flipped

    def test_dataframe_and_mapping_inputs_are_equivalent(self):
        ref, cur = self._panel(1), self._panel(2, shift=0.7)
        assert drift_report(ref, cur) == drift_report(pd.DataFrame(ref), pd.DataFrame(cur))

    def test_index_of_a_dataframe_is_irrelevant(self):
        ref, cur = pd.DataFrame(self._panel(1)), pd.DataFrame(self._panel(2))
        shuffled = cur.sample(frac=1.0, random_state=0)
        assert drift_report(ref, cur) == drift_report(ref, shuffled)

    def test_inputs_are_not_mutated(self):
        ref, cur = pd.DataFrame(self._panel(1)), pd.DataFrame(self._panel(2))
        ref_before, cur_before = ref.copy(), cur.copy()
        drift_report(ref, cur)
        pd.testing.assert_frame_equal(ref, ref_before)
        pd.testing.assert_frame_equal(cur, cur_before)

    def test_features_argument_restricts_what_is_assessed(self):
        ref = pd.DataFrame({**self._panel(1), "market_id": [f"m{i}" for i in range(800)]})
        cur = pd.DataFrame({**self._panel(2), "market_id": [f"n{i}" for i in range(800)]})
        report = drift_report(ref, cur, features=["z", "a"])
        assert [f.name for f in report.features] == ["a", "z"]  # sorted, no market_id

    def test_unknown_feature_is_an_error_not_a_silent_skip(self):
        with pytest.raises(ValueError, match="nope"):
            drift_report(self._panel(1), self._panel(2), features=["a", "nope"])

    def test_feature_absent_from_current_is_a_total_loss_of_signal(self):
        ref, cur = self._panel(1), self._panel(2)
        del cur["z"]
        report = drift_report(ref, cur)
        assert report.absent_from_current == ("z",)
        z = _one(report, "z")
        assert z.current_missing_rate == 1.0
        assert z.status == "alert"
        assert report.status == "alert"

    def test_columns_only_in_current_are_reported_not_assessed(self):
        ref, cur = self._panel(1), self._panel(2)
        cur["brand_new"] = np.arange(800)
        report = drift_report(ref, cur)
        assert report.new_in_current == ("brand_new",)
        assert "brand_new" not in [f.name for f in report.features]

    def test_columns_of_unequal_length_are_rejected(self):
        with pytest.raises(ValueError, match="length"):
            drift_report({"a": [1, 2, 3], "b": [1, 2]}, {"a": [1], "b": [1]})

    def test_duplicate_dataframe_columns_are_rejected(self):
        dup = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])
        with pytest.raises(ValueError, match="duplicate"):
            drift_report(dup, dup)

    def test_declaring_the_kind_of_an_unknown_feature_is_an_error(self):
        # A typo here would otherwise silently leave the real feature on auto-detect.
        with pytest.raises(ValueError, match="typo_col"):
            drift_report(
                self._panel(1), self._panel(2), config=DriftConfig(categorical=("typo_col",))
            )

    def test_overall_status_is_the_worst_feature(self):
        ref, cur = self._panel(1), self._panel(2)
        cur["z"] = cur["z"] + 5.0
        report = drift_report(ref, cur)
        assert _one(report, "a").status == "ok"
        assert _one(report, "z").status == "alert"
        assert report.status == "alert"
        assert (report.n_ok, report.n_warn, report.n_alert, report.n_insufficient) == (2, 0, 1, 0)

    def test_alert_outranks_insufficient_outranks_ok(self):
        def sparse(seed: int) -> np.ndarray:
            """Same missing rate on both sides, but only 50 usable values."""
            values = np.full(N, np.nan)
            values[:50] = _normal(50, seed=seed)
            return values

        ref = {"a": _normal(seed=1), "b": sparse(2)}
        quiet = drift_report(ref, {"a": _normal(seed=3), "b": sparse(4)})
        assert _one(quiet, "a").status == "ok"
        assert _one(quiet, "b").status == "insufficient"
        assert quiet.status == "insufficient"  # unknown outranks fine

        loud = drift_report(ref, {"a": _normal(seed=3) + 5, "b": sparse(4)})
        assert _one(loud, "a").status == "alert"
        assert _one(loud, "b").status == "insufficient"
        assert loud.status == "alert"  # a measured problem is never masked

    def test_nothing_to_assess_is_not_ok(self):
        report = drift_report({}, {})
        assert report.features == ()
        assert report.prediction is None
        assert report.status == "insufficient"
        assert report.max_psi is None and report.max_kl is None

    def test_headline_numbers_are_none_not_zero_when_nothing_was_computed(self):
        report = drift_report({"x": _normal(seed=1)}, {"x": _normal(20, seed=2)})
        assert report.max_psi is None
        assert report.max_kl is None

    def test_headline_numbers_are_the_worst_feature(self):
        ref, cur = self._panel(1), self._panel(2, shift=1.0)
        report = drift_report(ref, cur)
        assert report.max_psi == max(f.psi for f in report.features if f.psi is not None)
        assert report.max_kl == max(f.kl for f in report.features if f.kl is not None)

    def test_row_counts_describe_the_feature_tables(self):
        report = drift_report(self._panel(1), {k: v[:300] for k, v in self._panel(2).items()})
        assert (report.n_reference, report.n_current) == (800, 300)

    def test_config_travels_with_the_report(self):
        cfg = DriftConfig(n_bins=8, min_samples=40, thresholds=DriftThresholds(psi_alert=0.4))
        report = drift_report(self._panel(1), self._panel(2), config=cfg)
        assert report.config == cfg

    def test_value_objects_are_frozen(self):
        report = drift_report(self._panel(1), self._panel(2))
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.status = "ok"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.features[0].psi = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Prediction drift
# --------------------------------------------------------------------------- #


class TestPredictionDrift:
    def _preds(self, a: float, b: float, seed: int, n: int = 1500) -> np.ndarray:
        return np.random.default_rng(seed).beta(a, b, n)

    def test_prediction_drift_is_reported_separately_from_features(self):
        feats = {"x": _normal(seed=1)}
        report = drift_report(
            feats,
            {"x": _normal(seed=2)},
            reference_predictions=self._preds(2, 5, 1),
            current_predictions=self._preds(5, 2, 2),
        )
        assert _one(report, "x").status == "ok"  # inputs look the same...
        assert report.prediction is not None
        assert report.prediction.name == "prediction"
        assert report.prediction.kind == "numeric"
        assert report.prediction.psi is not None and report.prediction.psi > 0.25
        assert report.prediction.status == "alert"  # ...but the model's output moved
        assert report.status == "alert"
        assert "prediction" not in [f.name for f in report.features]

    def test_stable_predictions_are_ok(self):
        report = drift_report(
            {"x": _normal(seed=1)},
            {"x": _normal(seed=2)},
            reference_predictions=self._preds(2, 5, 1),
            current_predictions=self._preds(2, 5, 2),
        )
        assert report.prediction is not None and report.prediction.status == "ok"
        assert report.status == "ok"

    def test_no_predictions_means_no_prediction_section(self):
        assert drift_report({"x": _normal(seed=1)}, {"x": _normal(seed=2)}).prediction is None

    def test_predictions_must_be_supplied_as_a_pair(self):
        with pytest.raises(ValueError, match="both"):
            drift_report({"x": [1.0]}, {"x": [1.0]}, reference_predictions=[0.5])
        with pytest.raises(ValueError, match="both"):
            drift_report({"x": [1.0]}, {"x": [1.0]}, current_predictions=[0.5])

    def test_prediction_drift_works_without_any_features(self):
        report = drift_report(
            {},
            {},
            reference_predictions=self._preds(2, 5, 1),
            current_predictions=self._preds(5, 2, 2),
        )
        assert report.features == ()
        assert report.status == "alert"

    def test_nan_predictions_are_missingness_not_a_crash(self):
        cur = self._preds(2, 5, 2)
        cur[:600] = np.nan
        report = drift_report(
            {},
            {},
            reference_predictions=self._preds(2, 5, 1),
            current_predictions=cur,
        )
        assert report.prediction is not None
        assert report.prediction.current_missing_rate == pytest.approx(0.4)
        assert report.prediction.status == "alert"

    def test_too_few_live_predictions_are_insufficient(self):
        report = drift_report(
            {},
            {},
            reference_predictions=self._preds(2, 5, 1),
            current_predictions=self._preds(2, 5, 2, n=15),
        )
        assert report.prediction is not None
        assert report.prediction.status == "insufficient"
        assert report.prediction.psi is None

    @pytest.mark.parametrize("bad", [-0.1, 1.5, 7.0, -3.0, float("inf"), float("-inf")])
    @pytest.mark.parametrize("side", ["reference", "current"])
    def test_predictions_outside_zero_one_are_rejected(self, side, bad):
        good = self._preds(2, 5, 1)
        poisoned = good.copy()
        poisoned[3] = bad
        ref, cur = (poisoned, good) if side == "reference" else (good, poisoned)
        with pytest.raises(ValueError, match=rf"{side} predictions.*\[0, 1\]"):
            drift_report({}, {}, reference_predictions=ref, current_predictions=cur)

    def test_the_error_says_how_many_are_out_of_range(self):
        cur = self._preds(2, 5, 2, n=100)
        cur[:7] = 7.0
        with pytest.raises(ValueError, match=r"7 of 100"):
            drift_report(
                {}, {}, reference_predictions=self._preds(2, 5, 1), current_predictions=cur
            )

    def test_zero_and_one_are_valid_probabilities(self):
        edge = np.array([0.0, 1.0] * 250)
        report = drift_report({}, {}, reference_predictions=edge, current_predictions=edge)
        assert report.prediction is not None and report.prediction.status == "ok"

    def test_non_numeric_predictions_are_rejected_with_a_clear_message(self):
        with pytest.raises(ValueError, match="current predictions"):
            drift_report(
                {},
                {},
                reference_predictions=self._preds(2, 5, 1),
                current_predictions=["high", "low"] * 200,
            )

    def test_reference_and_current_predictions_may_differ_in_length(self):
        report = drift_report(
            {},
            {},
            reference_predictions=self._preds(2, 5, 1, n=1500),
            current_predictions=self._preds(2, 5, 2, n=400),
        )
        assert report.prediction is not None
        assert report.prediction.n_reference == 1500 and report.prediction.n_current == 400
        assert report.prediction.status == "ok"

    def test_prediction_counts_towards_the_summary(self):
        report = drift_report(
            {"x": _normal(seed=1)},
            {"x": _normal(seed=2)},
            reference_predictions=self._preds(2, 5, 1),
            current_predictions=self._preds(5, 2, 2),
        )
        assert report.n_alert == 1 and report.n_ok == 1


# --------------------------------------------------------------------------- #
# Serialisation / ExperimentTracker
# --------------------------------------------------------------------------- #


def _awkward_report() -> DriftReport:
    """Numeric + categorical + unseen category + NaNs + insufficient + prediction."""
    rng = np.random.default_rng(3)
    cur_x = rng.normal(0.5, 1, 1500)
    cur_x[:200] = np.nan
    return drift_report(
        {
            "x": rng.normal(0, 1, 1500),
            "c": ["a"] * 900 + ["b"] * 600,
            "tiny": rng.normal(0, 1, 1500),
        },
        {
            "x": cur_x,
            "c": ["a"] * 600 + ["b"] * 400 + ["zz"] * 500,
            "tiny": np.concatenate([rng.normal(0, 1, 10), np.full(1490, np.nan)]),
        },
        reference_predictions=rng.beta(2, 5, 1500),
        current_predictions=rng.beta(4, 3, 1500),
        config=DriftConfig(numeric=("tiny",)),
    )


class TestSerialisation:
    def test_to_dict_is_strict_json(self):
        payload = _awkward_report().to_dict()
        json.dumps(payload, allow_nan=False)

    def test_to_dict_contains_only_plain_python_types(self):
        leaves = _walk(_awkward_report().to_dict())
        assert leaves  # the walk found something
        for leaf in leaves:
            assert leaf is None or type(leaf) in (int, float, str, bool), type(leaf)
            if isinstance(leaf, float):
                assert math.isfinite(leaf)

    def test_to_dict_round_trips(self):
        report = _awkward_report()
        assert DriftReport.from_dict(report.to_dict()) == report

    def test_round_trip_survives_json(self):
        report = _awkward_report()
        assert DriftReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report

    def test_feature_round_trips(self):
        fd = _awkward_report().features[0]
        assert FeatureDrift.from_dict(fd.to_dict()) == fd

    def test_a_status_that_does_not_exist_is_rejected_on_load(self):
        payload = _awkward_report().features[0].to_dict()
        payload["status"] = "fine"
        with pytest.raises(ValueError, match="fine"):
            FeatureDrift.from_dict(payload)

    def test_overlap_count_is_serialised_and_round_trips(self):
        report = dataclasses.replace(_awkward_report(), n_current_excluded_overlap=42)
        payload = report.to_dict()
        assert payload["n_current_excluded_overlap"] == 42
        assert isinstance(payload["n_current_excluded_overlap"], int)
        assert DriftReport.from_dict(json.loads(json.dumps(payload))) == report

    def test_the_pure_function_never_excludes_anything(self):
        assert _awkward_report().n_current_excluded_overlap == 0

    def test_config_round_trips(self):
        cfg = DriftConfig(
            n_bins=7,
            min_samples=12,
            epsilon=1e-3,
            thresholds=DriftThresholds(psi_warn=0.2, psi_alert=0.5),
            numeric=("b", "a"),
            categorical=("z",),
        )
        assert DriftConfig.from_dict(cfg.to_dict()) == cfg

    def test_top_level_carries_flat_numeric_summaries_for_the_tracker(self):
        payload = _awkward_report().to_dict()
        for key in ("n_ok", "n_warn", "n_alert", "n_insufficient", "n_reference", "n_current"):
            assert isinstance(payload[key], int)
        assert isinstance(payload["max_psi"], float)
        assert isinstance(payload["max_kl"], float)
        assert payload["status"] in ("ok", "warn", "alert", "insufficient")

    def test_unseen_categories_and_drivers_serialise_as_lists(self):
        payload = _awkward_report().to_dict()
        c = next(f for f in payload["features"] if f["name"] == "c")
        assert c["unseen_categories"] == ["zz"]
        assert isinstance(c["drivers"], list)

    def test_report_can_be_logged_to_an_experiment_tracker(self, tmp_path):
        tracker = LocalJSONTracker(tmp_path)
        report = _awkward_report()
        with tracker.run("drift", "nightly") as run:
            run.log_metrics(report)
        (record,) = tracker.list_runs("drift")
        assert record.metrics["status"] == report.status
        assert record.metrics["n_alert"] == report.n_alert
        assert record.metrics["max_psi"] == report.max_psi
        assert [f["name"] for f in record.metrics["features"]] == [f.name for f in report.features]


# --------------------------------------------------------------------------- #
# Point-in-time: store-backed construction
# --------------------------------------------------------------------------- #

_FAMILY = "politics"


def _snap(market_id: str, ts: str, **features: float) -> dict[str, Any]:
    return {"market_id": market_id, "captured_at": ts, **features}


def _reference_panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "market_id": [f"train{i}" for i in range(n)],
            "decision_date": "2025-01-10",
            "winning_label": "Yes",
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(10, 2, n),
        }
    )


def _live_snaps(
    n: int = 400, seed: int = 1, shift: float = 0.0, day: str = "2025-03-05"
) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        _snap(
            f"live{i}",
            f"{day}T12:00:00Z",
            feature_a=float(rng.normal(shift, 1)),
            feature_b=float(rng.normal(10, 2)),
        )
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path):
    return FeatureSnapshotStore(tmp_path / "snaps")


_WINDOW = {"window_start": "2025-03-01T00:00:00Z", "cutoff": "2025-03-10T00:00:00Z"}


class TestFromStore:
    def test_reports_on_the_window_it_was_asked_about(self, store):
        store.append(_FAMILY, _live_snaps(shift=1.5))
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report.n_current == 400
        assert _one(report, "feature_a").status == "alert"
        assert _one(report, "feature_b").status == "ok"

    def test_snapshots_after_the_cutoff_cannot_change_the_report(self, store):
        store.append(_FAMILY, _live_snaps(shift=0.0))
        before = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)

        # Wildly different values captured after the cutoff, for new and old markets.
        future = _live_snaps(n=400, seed=9, shift=50.0, day="2025-03-20")
        future += [
            _snap(f"live{i}", "2025-03-25T00:00:00Z", feature_a=999.0, feature_b=-999.0)
            for i in range(400)
        ]
        store.append(_FAMILY, future)
        after = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)

        assert after == before
        assert after.n_current == 400
        assert _one(after, "feature_a").status == "ok"

    def test_a_snapshot_exactly_at_the_cutoff_is_not_visible(self, store):
        # Same strict-before rule as FeatureSnapshotStore.as_of.
        store.append(_FAMILY, _live_snaps(n=300, shift=0.0))
        baseline = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        store.append(
            _FAMILY,
            [
                _snap(f"edge{i}", _WINDOW["cutoff"], feature_a=500.0, feature_b=500.0)
                for i in range(300)
            ],
        )
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report == baseline
        assert report.n_current == 300

    def test_a_later_snapshot_of_a_market_does_not_replace_its_earlier_one(self, store):
        # live0 is first seen sane, then re-captured after the cutoff with garbage.
        store.append(_FAMILY, _live_snaps(n=300))
        store.append(
            _FAMILY, [_snap("live0", "2025-04-01T00:00:00Z", feature_a=1e6, feature_b=1e6)]
        )
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report.n_current == 300
        assert _one(report, "feature_a").status == "ok"

    def test_window_start_excludes_older_snapshots(self, store):
        store.append(_FAMILY, _live_snaps(n=300, day="2025-03-05"))
        stale = [
            _snap(f"old{i}", "2025-02-15T00:00:00Z", feature_a=30.0, feature_b=10.0)
            for i in range(300)
        ]
        store.append(_FAMILY, stale)
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report.n_current == 300
        assert _one(report, "feature_a").status == "ok"

    def test_each_market_counts_once_using_its_latest_snapshot_in_the_window(self, store):
        store.append(_FAMILY, _live_snaps(n=300, day="2025-03-02"))
        store.append(_FAMILY, _live_snaps(n=300, day="2025-03-08"))  # re-captures, same ids
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report.n_current == 300

    def test_naive_timestamps_are_read_as_utc(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        aware = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        naive = drift_report_from_store(
            store,
            _FAMILY,
            _reference_panel(),
            window_start="2025-03-01T00:00:00",
            cutoff="2025-03-10T00:00:00",
        )
        assert naive == aware

    def test_defaults_to_the_feature_columns_of_the_reference(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert [f.name for f in report.features] == ["feature_a", "feature_b"]  # no market_id etc.

    def test_features_can_be_given_explicitly(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        report = drift_report_from_store(
            store, _FAMILY, _reference_panel(), features=["feature_b"], **_WINDOW
        )
        assert [f.name for f in report.features] == ["feature_b"]

    def test_an_empty_window_is_insufficient_not_ok(self, store):
        store.append(_FAMILY, _live_snaps(n=300, day="2025-01-02"))  # all before window_start
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report.n_current == 0
        assert report.status == "insufficient"
        assert report.absent_from_current == ()

    def test_an_unknown_family_is_an_error_not_a_quiet_insufficient_report(self, store):
        # A typo in the family name must not look like "no live data yet".
        store.append(_FAMILY, _live_snaps(n=300))
        with pytest.raises(ValueError, match="never-captured"):
            drift_report_from_store(store, "never-captured", _reference_panel(), **_WINDOW)

    def test_a_known_family_with_nothing_before_the_cutoff_is_just_insufficient(self, store):
        store.append(_FAMILY, _live_snaps(n=300, day="2025-06-01"))  # all after the cutoff
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report.n_current == 0
        assert report.status == "insufficient"

    def test_window_start_after_cutoff_is_an_error(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        with pytest.raises(ValueError, match="window_start"):
            drift_report_from_store(
                store,
                _FAMILY,
                _reference_panel(),
                window_start="2025-03-10T00:00:00Z",
                cutoff="2025-03-01T00:00:00Z",
            )

    def test_window_start_is_compared_after_normalising_time_zones(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        with pytest.raises(ValueError, match="window_start"):
            drift_report_from_store(
                store,
                _FAMILY,
                _reference_panel(),
                window_start="2025-03-10T02:00:00+02:00",  # 00:00Z, == cutoff below + 1h
                cutoff="2025-03-09T23:00:00Z",
            )

    def test_window_start_equal_to_cutoff_is_an_empty_window_not_an_error(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        report = drift_report_from_store(
            store,
            _FAMILY,
            _reference_panel(),
            window_start="2025-03-10T00:00:00Z",
            cutoff="2025-03-10T00:00:00Z",
        )
        assert report.n_current == 0 and report.status == "insufficient"

    def test_an_empty_features_list_is_an_error(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        with pytest.raises(ValueError, match="features"):
            drift_report_from_store(store, _FAMILY, _reference_panel(), features=[], **_WINDOW)

    def test_a_reference_with_no_feature_columns_and_no_features_given_is_an_error(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        bare = _reference_panel()[["market_id", "decision_date", "winning_label"]]
        with pytest.raises(ValueError, match="features"):
            drift_report_from_store(store, _FAMILY, bare, **_WINDOW)

    def test_predict_fn_is_not_called_on_an_empty_window(self, store):
        calls: list[int] = []

        def predict(frame: pd.DataFrame) -> np.ndarray:
            calls.append(len(frame))
            if frame.empty:
                raise AssertionError("predict_fn must not see an empty frame")
            return np.full(len(frame), 0.5)

        store.append(_FAMILY, _live_snaps(n=300, day="2025-01-02"))  # all before window_start
        report = drift_report_from_store(
            store, _FAMILY, _reference_panel(), predict_fn=predict, **_WINDOW
        )
        assert calls == [400]  # the reference only
        assert report.prediction is not None
        assert report.prediction.status == "insufficient"

    def test_predictions_come_from_a_callable_applied_to_both_sides(self, store):
        store.append(_FAMILY, _live_snaps(n=400, shift=1.5))
        seen: list[int] = []

        def predict(frame: pd.DataFrame) -> np.ndarray:
            seen.append(len(frame))
            return 1.0 / (1.0 + np.exp(-frame["feature_a"].to_numpy(dtype=float)))

        report = drift_report_from_store(
            store, _FAMILY, _reference_panel(), predict_fn=predict, **_WINDOW
        )
        assert sorted(seen) == [400, 400]
        assert report.prediction is not None
        assert report.prediction.status == "alert"

    # ---- the live window must not contain the markets the model was trained on

    def _with_training_markets_recaptured(self, store, n_live: int = 300, n_trained: int = 100):
        store.append(_FAMILY, _live_snaps(n=n_live))
        # Training markets, re-captured inside the window with wild values.
        store.append(
            _FAMILY,
            [
                _snap(f"train{i}", "2025-03-05T12:00:00Z", feature_a=50.0, feature_b=10.0)
                for i in range(n_trained)
            ],
        )

    def test_markets_in_the_reference_panel_are_excluded_from_the_live_window(self, store):
        self._with_training_markets_recaptured(store)
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report.n_current == 300
        assert report.n_current_excluded_overlap == 100
        assert _one(report, "feature_a").status == "ok"  # the wild values were not evidence

    def test_the_overlap_can_be_kept_explicitly(self, store):
        self._with_training_markets_recaptured(store)
        report = drift_report_from_store(
            store, _FAMILY, _reference_panel(), exclude_reference_markets=False, **_WINDOW
        )
        assert report.n_current == 400
        assert report.n_current_excluded_overlap == 0
        assert _one(report, "feature_a").status == "alert"  # and it shows how much it matters

    def test_with_no_overlap_nothing_is_excluded(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        report = drift_report_from_store(store, _FAMILY, _reference_panel(), **_WINDOW)
        assert report.n_current == 300 and report.n_current_excluded_overlap == 0

    @pytest.mark.parametrize(
        ("store_ids", "panel_ids"),
        [
            pytest.param(lambda i: str(i), lambda i: i, id="store-text-panel-int"),
            pytest.param(lambda i: i, lambda i: str(i), id="store-int-panel-text"),
            pytest.param(lambda i: i, lambda i: i, id="both-int"),
        ],
    )
    def test_market_ids_are_compared_as_text(self, store, store_ids, panel_ids):
        store.append(
            _FAMILY,
            [
                _snap(store_ids(i), "2025-03-05T12:00:00Z", feature_a=0.0, feature_b=10.0)
                for i in range(50)
            ],
        )
        panel = _reference_panel().assign(market_id=[panel_ids(i) for i in range(400)])
        report = drift_report_from_store(store, _FAMILY, panel, **_WINDOW)
        assert report.n_current_excluded_overlap == 50 and report.n_current == 0

    def test_a_window_made_only_of_training_markets_is_insufficient_not_ok(self, store):
        store.append(
            _FAMILY,
            [
                _snap(f"train{i}", "2025-03-05T12:00:00Z", feature_a=0.0, feature_b=10.0)
                for i in range(300)
            ],
        )

        def predict(frame: pd.DataFrame) -> np.ndarray:
            if len(frame) != 400:
                raise AssertionError("predict_fn must not see the emptied live window")
            return np.full(len(frame), 0.5)

        report = drift_report_from_store(
            store, _FAMILY, _reference_panel(), predict_fn=predict, **_WINDOW
        )
        assert report.n_current == 0 and report.n_current_excluded_overlap == 300
        assert report.status == "insufficient"

    def test_predict_fn_sees_the_window_without_the_training_markets(self, store):
        self._with_training_markets_recaptured(store)
        seen: list[int] = []

        def predict(frame: pd.DataFrame) -> np.ndarray:
            seen.append(len(frame))
            return np.full(len(frame), 0.5)

        drift_report_from_store(store, _FAMILY, _reference_panel(), predict_fn=predict, **_WINDOW)
        assert sorted(seen) == [300, 400]

    def test_a_reference_without_market_ids_cannot_be_checked_for_overlap(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        anonymous = _reference_panel().drop(columns=["market_id"])
        with pytest.raises(ValueError, match="market_id.*exclude_reference_markets"):
            drift_report_from_store(store, _FAMILY, anonymous, **_WINDOW)
        report = drift_report_from_store(
            store, _FAMILY, anonymous, exclude_reference_markets=False, **_WINDOW
        )
        assert report.n_current == 300

    def test_a_training_market_recaptured_in_the_window_is_excluded_for_a_real_panel(self, store):
        rng = np.random.default_rng(0)
        store.append(
            _FAMILY,
            [
                _snap(
                    f"t{i}",
                    "2025-01-05T00:00:00Z",
                    feature_a=float(rng.normal()),
                    feature_b=float(rng.normal(10, 2)),
                )
                for i in range(300)
            ],
        )
        resolutions = pd.DataFrame(
            {
                "market_id": [f"t{i}" for i in range(300)],
                "decision_date": "2025-01-10",
                "winning_label": ["Yes", "No"] * 150,
            }
        )
        panel = build_panel_from_snapshots(store, _FAMILY, resolutions)
        store.append(_FAMILY, _live_snaps(n=300))
        store.append(
            _FAMILY,
            [
                _snap(f"t{i}", "2025-03-06T00:00:00Z", feature_a=40.0, feature_b=10.0)
                for i in range(60)
            ],
        )
        report = drift_report_from_store(store, _FAMILY, panel, **_WINDOW)
        assert report.n_current == 300 and report.n_current_excluded_overlap == 60

    @pytest.mark.parametrize(
        "bad_output",
        [
            pytest.param(lambda n: np.full(n - 1, 0.5), id="one-short"),
            pytest.param(lambda n: np.full(n + 1, 0.5), id="one-over"),
            pytest.param(lambda n: np.full((n, 2), 0.5), id="predict_proba-both-columns"),
            pytest.param(lambda n: np.full((n, 1), 0.5), id="column-vector"),
            pytest.param(lambda n: 0.5, id="scalar"),
            pytest.param(lambda n: [], id="empty"),
        ],
    )
    @pytest.mark.parametrize("fails_on", ["reference", "current"])
    def test_predict_fn_must_return_one_value_per_row(self, store, fails_on, bad_output):
        store.append(_FAMILY, _live_snaps(n=300))
        panel = _reference_panel()

        def predict(frame: pd.DataFrame) -> Any:
            is_reference = len(frame) == len(panel)
            if is_reference == (fails_on == "reference"):
                return bad_output(len(frame))
            return np.full(len(frame), 0.5)

        label = "reference panel" if fails_on == "reference" else "live window"
        with pytest.raises(ValueError, match=rf"predict_fn.*one.*per row.*{label}"):
            drift_report_from_store(store, _FAMILY, panel, predict_fn=predict, **_WINDOW)

    def test_predict_fn_returning_non_numbers_is_a_clear_error(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        with pytest.raises(ValueError, match="predict_fn.*not numeric"):
            drift_report_from_store(
                store,
                _FAMILY,
                _reference_panel(),
                predict_fn=lambda frame: ["yes"] * len(frame),
                **_WINDOW,
            )

    def test_predict_fn_out_of_range_probabilities_are_rejected(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        with pytest.raises(ValueError, match=r"reference predictions.*\[0, 1\]"):
            drift_report_from_store(
                store,
                _FAMILY,
                _reference_panel(),
                predict_fn=lambda frame: np.full(len(frame), 7.0),
                **_WINDOW,
            )

    def test_predict_fn_may_return_a_list_or_a_series(self, store):
        store.append(_FAMILY, _live_snaps(n=300))
        for convert in (list, pd.Series):
            report = drift_report_from_store(
                store,
                _FAMILY,
                _reference_panel(),
                predict_fn=lambda frame, convert=convert: convert([0.5] * len(frame)),
                **_WINDOW,
            )
            assert report.prediction is not None

    def test_works_on_a_panel_built_by_the_real_panel_builder(self, store):
        rng = np.random.default_rng(0)
        train = [
            _snap(
                f"t{i}",
                "2025-01-05T00:00:00Z",
                feature_a=float(rng.normal()),
                feature_b=float(rng.normal(10, 2)),
            )
            for i in range(300)
        ]
        store.append(_FAMILY, train)
        resolutions = pd.DataFrame(
            {
                "market_id": [f"t{i}" for i in range(300)],
                "decision_date": "2025-01-10",
                "winning_label": ["Yes", "No"] * 150,
            }
        )
        panel = build_panel_from_snapshots(store, _FAMILY, resolutions)
        store.append(_FAMILY, _live_snaps(n=300, shift=1.5))
        report = drift_report_from_store(store, _FAMILY, panel, **_WINDOW)
        assert [f.name for f in report.features] == ["feature_a", "feature_b"]
        assert _one(report, "feature_a").status == "alert"
        assert report.n_reference == 300


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


class TestExports:
    def test_exported_from_the_monitoring_package(self):
        import pmlab.monitoring as monitoring

        for name in (
            "DriftConfig",
            "DriftReport",
            "DriftThresholds",
            "FeatureDrift",
            "drift_report",
            "drift_report_from_store",
        ):
            assert hasattr(monitoring, name), name
            assert name in monitoring.__all__

    def test_exported_from_the_top_level_package(self):
        import pmlab

        for name in (
            "DriftConfig",
            "DriftReport",
            "DriftThresholds",
            "FeatureDrift",
            "drift_report",
            "drift_report_from_store",
        ):
            assert hasattr(pmlab, name), name
            assert name in pmlab.__all__

    def test_version_is_bumped(self):
        import pmlab

        assert pmlab.__version__ == "0.9.1"
