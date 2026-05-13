from pmlab.core.sizing import flat_stake_size


def test_flat_stake_size_basic():
    # stake=1.0, price=0.25 → size=4.0 shares
    assert abs(flat_stake_size(1.0, 0.25) - 4.0) < 1e-9


def test_flat_stake_size_half():
    assert abs(flat_stake_size(1.0, 0.5) - 2.0) < 1e-9


def test_flat_stake_size_near_zero_price():
    # Should not divide by zero
    result = flat_stake_size(1.0, 0.0)
    assert result > 0  # clamps to 1e-9


def test_flat_stake_size_larger_stake():
    # stake=5.0, price=0.5 → 10.0 shares
    assert abs(flat_stake_size(5.0, 0.5) - 10.0) < 1e-9


def test_flat_stake_loss_at_zero():
    # At price p, if loses: loss = p * size = p * stake/p = stake
    stake = 2.0
    price = 0.3
    size = flat_stake_size(stake, price)
    max_loss = price * size  # should equal stake
    assert abs(max_loss - stake) < 1e-9
