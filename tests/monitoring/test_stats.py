"""The watchdog's statistics, checked against independent implementations.

`pmlab.monitoring.stats` holds the pure arithmetic under the health checks. Each
function is held to something that does not share its code: scipy's normal
distribution, a hand-worked example, a brute-force version of the textbook
procedure, or a simulation of the thing it claims to describe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sp

from pmlab.monitoring.stats import (
    holm_adjust,
    mixture_reference,
    normal_isf,
    normal_sf,
)


class TestNormalTails:
    @pytest.mark.parametrize("z", [-6.0, -3.0, -1.0, 0.0, 0.5, 1.645, 2.0, 3.0, 5.0, 8.0])
    def test_survival_function_matches_scipy(self, z: float) -> None:
        assert normal_sf(z) == pytest.approx(sp.norm.sf(z), rel=1e-12, abs=1e-300)

    @pytest.mark.parametrize("p", [1e-12, 1e-6, 0.00135, 0.0228, 0.05, 0.5, 0.9, 0.999])
    def test_inverse_survival_function_matches_scipy(self, p: float) -> None:
        assert normal_isf(p) == pytest.approx(sp.norm.isf(p), rel=1e-9, abs=1e-9)

    def test_the_two_are_inverses(self) -> None:
        for z in np.linspace(-4, 6, 41):
            assert normal_isf(normal_sf(float(z))) == pytest.approx(float(z), abs=1e-6)

    def test_the_endpoints_are_infinite(self) -> None:
        assert normal_isf(0.0) == math.inf
        assert normal_isf(1.0) == -math.inf

    @pytest.mark.parametrize("p", [-0.1, 1.1, math.nan])
    def test_a_probability_outside_0_1_is_refused(self, p: float) -> None:
        with pytest.raises(ValueError, match="probability"):
            normal_isf(p)


class TestHolm:
    def test_a_hand_worked_example(self) -> None:
        # Sorted: 0.005, 0.01, 0.03, 0.04 with multipliers 4, 3, 2, 1:
        #   0.02, 0.03, 0.06, 0.04  -> running maximum -> 0.02, 0.03, 0.06, 0.06.
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])
        assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_a_single_test_is_not_adjusted(self) -> None:
        assert holm_adjust([0.037]) == [0.037]

    def test_nothing_to_adjust(self) -> None:
        assert holm_adjust([]) == []

    def test_adjusted_values_are_capped_at_one(self) -> None:
        assert holm_adjust([0.5, 0.6]) == [1.0, 1.0]

    def test_ties_share_the_larger_adjustment(self) -> None:
        assert holm_adjust([0.01, 0.01]) == pytest.approx([0.02, 0.02])

    def test_the_order_of_the_input_is_kept(self) -> None:
        p = [0.2, 0.001, 0.04]
        adjusted = holm_adjust(p)
        # Smallest first gets x3, then x2, then x1, then the running maximum.
        assert adjusted == pytest.approx([0.2, 0.003, 0.08])

    def test_it_is_never_looser_than_bonferroni_and_never_below_the_raw_value(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = rng.random(int(rng.integers(1, 12))).tolist()
            adjusted = holm_adjust(p)
            for raw, adj in zip(p, adjusted, strict=True):
                assert raw <= adj + 1e-15
                assert adj <= min(1.0, len(p) * raw) + 1e-12

    def test_it_agrees_with_the_textbook_step_down_procedure(self) -> None:
        """Holm's procedure at level alpha, written out step by step, rejects exactly
        the hypotheses whose adjusted p-value is at most alpha."""

        def step_down(p: list[float], alpha: float) -> set[int]:
            rejected: set[int] = set()
            m = len(p)
            for rank, index in enumerate(sorted(range(m), key=lambda i: p[i])):
                if p[index] <= alpha / (m - rank):
                    rejected.add(index)
                else:
                    break  # the first failure stops the procedure
            return rejected

        rng = np.random.default_rng(1)
        for _ in range(500):
            m = int(rng.integers(1, 9))
            p = (rng.random(m) ** 3).tolist()  # skewed small, so there is something to reject
            adjusted = holm_adjust(p)
            for alpha in (0.01, 0.05, 0.2):
                from_adjusted = {i for i, a in enumerate(adjusted) if a <= alpha}
                assert from_adjusted == step_down(p, alpha)

    def test_it_controls_the_family_wise_error_rate_that_raw_p_values_do_not(self) -> None:
        """Ten true nulls: some raw p-value falls below 0.05 about 40% of the time, which
        is the false-alarm inflation of testing every segment on its own. After Holm the
        chance that any survives is at most 5%."""
        rng = np.random.default_rng(2)
        m, reps, alpha = 10, 20_000, 0.05
        raw_hits = holm_hits = 0
        for _ in range(reps):
            p = rng.random(m).tolist()
            raw_hits += min(p) <= alpha
            holm_hits += min(holm_adjust(p)) <= alpha
        raw_rate, holm_rate = raw_hits / reps, holm_hits / reps
        assert raw_rate == pytest.approx(1 - (1 - alpha) ** m, abs=0.02)  # ~0.40
        assert holm_rate <= alpha + 0.008
        assert holm_rate >= alpha - 0.012  # and it is not needlessly conservative

    @pytest.mark.parametrize("bad", [[-0.1], [1.2], [math.nan], [0.5, math.inf]])
    def test_a_p_value_outside_0_1_is_refused(self, bad: list[float]) -> None:
        with pytest.raises(ValueError, match="p-value"):
            holm_adjust(bad)


class TestMixtureReference:
    def test_a_hand_worked_example(self) -> None:
        # 3 YES : 1 NO. Weights 0.75 and 0.25.
        #   mean     = 0.75 * 0.20 + 0.25 * (-0.10)            = 0.125
        #   variance = 0.75**2 * 1.0**2/100 + 0.25**2 * 2.0**2/50 = 0.005625 + 0.005
        mean, variance = mixture_reference(
            {"yes": 3, "no": 1}, {"yes": (0.20, 1.0, 100), "no": (-0.10, 2.0, 50)}
        )
        assert mean == pytest.approx(0.125)
        assert variance == pytest.approx(0.010625)

    def test_one_group_is_the_ordinary_standard_error_of_its_mean(self) -> None:
        mean, variance = mixture_reference({"yes": 40}, {"yes": (0.15, 0.9, 400)})
        assert mean == 0.15
        assert variance == pytest.approx(0.9**2 / 400)

    def test_a_group_with_no_live_trades_has_no_weight(self) -> None:
        both = {"yes": (0.2, 1.0, 100), "no": (-5.0, 9.0, 3)}
        mean, variance = mixture_reference({"yes": 10, "no": 0}, both)
        assert (mean, variance) == pytest.approx((0.2, 1.0 / 100))

    def test_the_weights_follow_the_live_mix_not_the_backtest_mix(self) -> None:
        groups = {"yes": (0.30, 1.0, 100), "no": (0.00, 1.0, 100)}
        mostly_yes, _ = mixture_reference({"yes": 9, "no": 1}, groups)
        mostly_no, _ = mixture_reference({"yes": 1, "no": 9}, groups)
        assert mostly_yes == pytest.approx(0.27)
        assert mostly_no == pytest.approx(0.03)

    def test_the_variance_is_that_of_a_mix_weighted_sum_of_independent_estimates(self) -> None:
        """Independent check by simulation: draw each side's backtest mean from its own
        sampling distribution, weight them by the live mix, and compare the spread."""
        rng = np.random.default_rng(3)
        groups = {"yes": (0.20, 1.0, 100), "no": (-0.10, 2.0, 50)}
        counts = {"yes": 30, "no": 10}
        _, variance = mixture_reference(counts, groups)
        yes = rng.normal(0.20, 1.0 / math.sqrt(100), 400_000)
        no = rng.normal(-0.10, 2.0 / math.sqrt(50), 400_000)
        mixed = 0.75 * yes + 0.25 * no
        assert float(mixed.var()) == pytest.approx(variance, rel=0.02)

    @pytest.mark.parametrize(
        ("counts", "groups"),
        [
            ({}, {"yes": (0.1, 1.0, 10)}),
            ({"yes": 0}, {"yes": (0.1, 1.0, 10)}),
            ({"yes": 5, "no": 5}, {"yes": (0.1, 1.0, 10)}),  # NO has live trades, no reference
            ({"yes": -1}, {"yes": (0.1, 1.0, 10)}),
        ],
    )
    def test_a_mix_that_cannot_be_referenced_is_refused(
        self, counts: dict[str, int], groups: dict[str, tuple[float, float, int]]
    ) -> None:
        with pytest.raises(ValueError):
            mixture_reference(counts, groups)
