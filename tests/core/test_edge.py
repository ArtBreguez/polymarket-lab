"""Tests for edge calculation."""

from __future__ import annotations

from pmlab.core.edge import compute_edge


class TestComputeEdge:
    def test_positive_edge(self) -> None:
        # fair_prob=0.6, price=0.3, fee=0.003 → edge = 0.6 - 0.3 - 0.003 = 0.297
        assert abs(compute_edge(0.6, 0.3, 0.003) - 0.297) < 1e-9

    def test_negative_edge(self) -> None:
        assert compute_edge(0.2, 0.5, 0.003) < 0

    def test_zero_fee(self) -> None:
        assert abs(compute_edge(0.6, 0.4) - 0.2) < 1e-9

    def test_with_slippage(self) -> None:
        edge = compute_edge(0.6, 0.3, fee_estimate=0.003, slippage_estimate=0.01)
        assert abs(edge - (0.6 - 0.3 - 0.003 - 0.01)) < 1e-9

    def test_exactly_zero_edge(self) -> None:
        # fair == price + fee → edge=0
        fee = 0.003
        assert abs(compute_edge(0.303, 0.3, fee)) < 1e-9
