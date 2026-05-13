"""Tests for kelly sizing utilities."""

from __future__ import annotations

import pytest

from pmlab.core.sizing import flat_stake_size, kelly_fraction, kelly_stake_size


class TestFlatStake:
    def test_basic(self):
        assert flat_stake_size(10.0, 0.5) == pytest.approx(20.0)

    def test_low_price(self):
        result = flat_stake_size(10.0, 1e-12)
        assert result > 0


class TestKellyFraction:
    def test_positive_edge(self):
        f = kelly_fraction(0.6, 0.5, fraction=1.0)
        assert f > 0
        assert f < 1

    def test_zero_edge(self):
        f = kelly_fraction(0.5, 0.5)
        assert f == pytest.approx(0.0, abs=1e-9)

    def test_negative_edge_clamped(self):
        f = kelly_fraction(0.3, 0.5)
        assert f == 0.0

    def test_fractional_kelly_scales(self):
        full = kelly_fraction(0.6, 0.5, fraction=1.0)
        quarter = kelly_fraction(0.6, 0.5, fraction=0.25)
        assert quarter == pytest.approx(full * 0.25, rel=1e-6)

    def test_invalid_price_zero(self):
        assert kelly_fraction(0.6, 0.0) == 0.0

    def test_invalid_price_one(self):
        assert kelly_fraction(0.6, 1.0) == 0.0


class TestKellyStakeSize:
    def test_basic(self):
        stake = kelly_stake_size(0.6, 0.5, bankroll=1000.0, fraction=0.25)
        assert stake > 0
        assert stake <= 50.0

    def test_max_exposure_cap(self):
        stake = kelly_stake_size(0.99, 0.01, bankroll=1000.0, fraction=1.0, max_exposure=0.05)
        assert stake == pytest.approx(50.0, rel=1e-6)

    def test_zero_edge(self):
        assert kelly_stake_size(0.5, 0.5, bankroll=1000.0) == 0.0

    def test_negative_edge(self):
        assert kelly_stake_size(0.3, 0.6, bankroll=1000.0) == 0.0
