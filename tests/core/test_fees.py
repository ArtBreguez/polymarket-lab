"""Tests for fee estimation."""

from __future__ import annotations

from pmlab.core.fees import estimate_fee


class TestEstimateFee:
    def test_30bps(self) -> None:
        fee = estimate_fee(1.0, taker_bps=30)
        assert abs(fee - 0.003) < 1e-9

    def test_scales_with_stake(self) -> None:
        # fee is on flat_stake, not price * size
        fee_10 = estimate_fee(10.0, taker_bps=30)
        assert abs(fee_10 - 0.03) < 1e-9

    def test_zero_fee(self) -> None:
        assert estimate_fee(1.0, taker_bps=0) == 0.0

    def test_default_bps(self) -> None:
        # default should be 30bps on Polymarket CLOB
        fee = estimate_fee(1.0)
        assert fee == estimate_fee(1.0, taker_bps=30)
