# Position sizing

Three functions turn a probability and a price into a stake. They are pure arithmetic, with no state
and no I/O, and every broker and the backtest use the same ones.

```python
from pmlab import flat_stake_size, kelly_fraction, kelly_stake_size

# Flat stake: always risk 5 USDC, whatever the edge. The result is the number of shares.
shares = flat_stake_size(flat_stake=5.0, entry_price=0.35)
print(round(shares, 2), "shares")

# Kelly fraction: what share of the bankroll to wager (quarter-Kelly here).
f = kelly_fraction(win_prob=0.65, entry_price=0.35, fraction=0.25)
print(round(f, 4), "of the bankroll")

# Kelly stake: that fraction of a bankroll, in USDC, capped at max_exposure of the bankroll.
stake = kelly_stake_size(
    win_prob=0.65,
    entry_price=0.35,
    bankroll=500.0,
    fraction=0.25,
    max_exposure=0.05,  # never more than 5% of the bankroll on one trade
)
print(stake, "USDC")
```

`flat_stake_size` buys `flat_stake / entry_price` shares, so the most you can lose is the stake and a
win pays `flat_stake * (1 - entry_price) / entry_price`.

`kelly_fraction` is the binary-outcome Kelly formula `f* = (b·p − q) / b`, with net odds
`b = (1 − entry_price) / entry_price`, `p = win_prob` and `q = 1 − p`, scaled by `fraction` and
floored at 0: a bet with no edge sizes to nothing. In the example the quarter-Kelly fraction is
11.5% of the bankroll, so the 5% cap is what sets the stake, 25 USDC on a 500 USDC bankroll.

Quarter-Kelly with a 5% cap is the conservative default. Full Kelly (`fraction=1.0`) maximises
long-run growth if the probabilities are exactly right and produces deep drawdowns when they are not;
the settings recommended before real capital are in [Live trading](live-trading.md#kelly-sizing-reference).
