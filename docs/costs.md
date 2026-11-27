# Backtest Costs — `pmlab.core.costs`

A backtest that fills at the quoted price and charges the wrong fee reports an edge the broker
cannot reproduce. `CostModel` makes execution friction explicit: a **fee**, a **spread** and a
**depth** term. Since 1.5.0 the fee is the one Polymarket charges a taker, which depends on the
price; the flat 0.30% of the stake that earlier releases charged is still there, by name, so an old
run can be repeated.

!!! warning "The default fee changed in 1.5.0"
    `CostModel()`, `rolling_origin_eval`, `TunedForecaster`, `PaperBroker` and the CLI used to charge
    a flat 30 bps of the stake. A Polymarket taker pays `shares x rate x price x (1 - price)` per
    fill, which for a 0.05-rate market is **3.5% of the stake at 0.30 and 2.5% at 0.50**. On a
    mid-priced taker trade the old model understated the fee roughly tenfold, so a backtest that
    cleared the gate could be a NO_GO under the real fee. The numbers of every default run moved;
    [what is unchanged](#compatibility-and-migration) is listed below.

---

## Why this is not a detail

On Polymarket's mutually-exclusive (negRisk) events the gap between the mid and
the executable price is structural, not incidental. Read at the mid, a field of
outcomes sums to ~1.0 and looks coherent; read at the ask, it sums above 1.0, and
in large fields half the outcomes have no book at all. Any edge measured against a
mid price is a spread you have not paid yet.

The same applies inside a single market: `market_price` in the training panel is a
quote, and your fill is worse than a quote by the half-spread plus whatever the
book charges for your size, plus the fee.

---

## Basic Usage

```python
from pmlab import CostModel, FeeSchedule, rolling_origin_eval

# Default: the Polymarket taker fee, fill at the quote.
result = rolling_origin_eval(panel, model)

# Realistic: also cross the half-spread and walk the book on size.
costs = CostModel(
    slippage_bps=50.0,         # half-spread, as a fraction of price
    depth_bps_per_unit=20.0,   # extra bps per multiple of the reference stake
    depth_reference_stake=10.0,  # USDC that fills without moving the book
)
result = rolling_origin_eval(panel, model, flat_stake=25.0, costs=costs)

# The fee a market actually carries: its schedule, not the default.
weather = CostModel(fee_schedule=FeeSchedule(rate=0.01), slippage_bps=50.0)

# The model of 1.4.0 and before, to repeat an old run.
old = CostModel(fee_model="flat-bps", taker_bps=30.0)

# Upper bound: what the strategy would earn in a frictionless world.
best_case = rolling_origin_eval(panel, model, costs=CostModel.frictionless())
```

Comparing the runs is the point: the distance between them is how much of your edge is paid to the
market.

---

## The fee models

`fee_model` is `"polymarket"` (the default) or `"flat-bps"`.

### `"polymarket"`: the taker fee of a fill

```
fee = shares * rate * (price * (1 - price)) ** exponent        # exponent is 1 on every market today
    = stake * rate * (1 - price)                                # for exponent 1, shares = stake / price
```

rounded to 5 decimals, and a fee below 0.00001 is not charged (the exact rules, with the sources
they were checked against, are in `pmlab.core.fee_schedule`). Makers pay nothing; this model fills
takers, so `liquidity` is always `"taker"`. The fee is taken **at the price the fill happens at**
(`fill_price`, slippage included) and is paid on top of the stake: it does not reduce the shares
bought.

As a share of the stake, for a 0.05-rate market against the flat 0.30%:

| price | 0.05 | 0.10 | 0.30 | 0.50 | 0.70 | 0.90 | 0.99 |
|---|---|---|---|---|---|---|---|
| Polymarket, rate 0.05 | 4.75% | 4.50% | 3.50% | 2.50% | 1.50% | 0.50% | 0.05% |
| flat-bps | 0.30% | 0.30% | 0.30% | 0.30% | 0.30% | 0.30% | 0.30% |

A cheap token pays most of its stake in fee, which is why the fee moves a longshot-heavy strategy the
most.

**Which rate.** The fee schedule is Polymarket's, per market, and it changes. A `CostModel` takes it
as `fee_schedule=FeeSchedule(...)`, read from the market itself if you can
(`GammaClient.fetch_fee_snapshot`, `resolve_fee_snapshot`: the schedule the market published, and
when it was seen). Without one the model charges the `"other"` category's **0.05, the highest rate
Polymarket publishes**, so an unknown market is costed too high rather than too low, and every trade
says so: `fee_source="category_default"`. Weather markets publish 0.01 (2026-11-25), so for the
`weather_tmax` plugin pass `FeeSchedule(rate=0.01)` or the schedule the market carries; otherwise a
weather backtest is charged five times the fee.

`fee_source` says where the schedule came from: `"gamma"`, `"clob"` or `"category_default"` as a
`FeeSnapshot` says it, `"user"` for a `fee_schedule` you gave without a source (name one with
`fee_source=`), `"flat_bps"` for the flat model. A label has to describe the fee it sits on: a
polymarket model refuses `"flat_bps"`, and `"category_default"` for a schedule that is no
category's default, so `dataclasses.replace(costs, fee_schedule=...)`, which keeps the old label
unless it is given a new one, cannot leave a 0.01 fee labelled as the 0.05 fallback. To take a
snapshot's schedule, source and moment together, use
[`with_fee_snapshot`](#a-fee-is-priced-from-when-it-was-known).

`taker_bps` is read only by the flat model. Giving one to a polymarket model (`CostModel(fee_model=
"polymarket", taker_bps=50)`, or `dataclasses.replace(CostModel(), taker_bps=50)`) prices nothing, so
it warns instead of doing nothing in silence.

### `"flat-bps"`: the model of 1.4.0 and before

```
fee = stake * taker_bps / 10_000        # the same 0.30% at every price, taker_bps defaults to 30
```

`taker_bps` is read only by this model; the polymarket model ignores it. A `taker_bps` that is given
without a `fee_model` still selects this model, with a deprecation warning, so existing code runs
unchanged (see [below](#compatibility-and-migration)).

---

## The arithmetic

```
depth_penalty_bps = depth_bps_per_unit * max(0, stake - ref) / ref
total_bps         = slippage_bps + depth_penalty_bps
fill              = price * (1 + total_bps / 10_000) + slippage_fixed
fill              = min(fill, 1.0)
fee               = costs.fee(stake, fill)          # polymarket: stake * rate * (1 - fill)
```

| Parameter | Unit | Models |
|---|---|---|
| `fee_model` | `"polymarket"` or `"flat-bps"` | which fee arithmetic (default `"polymarket"`) |
| `fee_schedule` | `FeeSchedule` | the rate and rounding the polymarket model charges (default: the `"other"` category, 0.05) |
| `fee_source` | text | where the schedule came from, recorded on every trade |
| `fee_seen_at` | timestamp | when the schedule was seen (`with_fee_snapshot` sets it); a backtest refuses a fold that decided before it. Optional, written by `to_dict` only when set |
| `taker_bps` | bps of notional stake | the flat model's fee (default 30; ignored by polymarket) |
| `slippage_bps` | bps of price | half-spread crossed on entry |
| `slippage_fixed` | probability units | a floor cost (0.01 = one cent) |
| `depth_bps_per_unit` | bps per multiple of reference | walking the book on size |
| `depth_reference_stake` | USDC | size that fills at top of book |

`slippage_bps` is proportional to price, so a $0.10 outcome slips less in absolute
terms than a $0.90 one — which matches how a percentage spread behaves. Use
`slippage_fixed` when the cost is a floor rather than a percentage.

The fill is clamped at 1.0. A probability price above certainty would create a
position paying more than it can settle for. A token at 0 buys no shares and pays no fee.

`costs.fee(stake)` without a price is the flat model's call. The polymarket fee depends on the price
paid, so `fee(stake, price)` needs the fill price and raises `ValueError` without one rather than
guess it.

---

## What the depth term is, and is not

It is a **documented linear approximation**, not an order-book walk. pmlab's
training panel carries `market_price`, not L2 depth, so the model cannot know
where the book thins out. The penalty grows linearly in *excess* stake above a
reference you choose.

If you have depth data, measure it and set `depth_bps_per_unit` from the
measurement rather than guessing. If you don't, prefer a pessimistic value: a
backtest that overstates cost fails safe, one that understates it promotes a
champion that loses money live.

---

## What a trade records

Every trade of a backtest (`rolling_origin_eval`'s trade log) and of the paper broker
(`forward_paper_trades.json`) carries what charged its fee:

| Field | Holds |
|---|---|
| `fee_paid` | the fee, in USDC |
| `fee_rate` | the rate of the model that charged it: the schedule's `rate` (0 for a fee-free market) for polymarket, `taker_bps / 10_000` for flat-bps |
| `liquidity` | `"taker"`: this model fills at the book; maker fills (no fee, a rebate) are not modelled yet |
| `fee_source` | where the schedule came from (see above) |
| `fee_model` | `"polymarket"` or `"flat-bps"`: which formula `fee_rate` is the rate of. The two are not the same kind of number (`rate x (1 - price)` of the stake against `taker_bps / 10_000` of it), and `fee_source` alone tells them apart only for flat-bps |

`SettlementEngine` books `fee_paid` as the broker recorded it (so the PnL of a settled trade is its
gross result less the fee charged at the fill, and the five fields stay on the settled trade). A
trade with no `fee_paid` on record, from before it was written, is booked with the flat 0.003 it
always was.

`generate_report` and the CLI show what the fees cost: the HTML report has **Total Fees** and
**Fees % of Stake** cards (over the settled trades that recorded a fee, like the PnL and ROI beside
them, and `n/a` when none did) and a Fee column; `pmlab backtest`, `pmlab report` and `pmlab status`
print the total and its share of the stake. A fee of 2.5% of the stake is a different strategy from
one of 0.3%, and the PnL alone does not say which produced it.

---

## Where costs propagate

Costs are not cosmetic — they flow into the promotion decision, and into the live path too:

1. `rolling_origin_eval` fills at `fill_price` instead of `market_price`, charges the fee at that
   price, and records both in the trade log.
2. `edge` is computed as `predicted_prob - fill_price`. Measuring it against the
   quote would overstate every trade by the friction. Under the live policy
   (`TradePolicy.live`) the same arithmetic prices the NO side: a NO trade is a buy of
   the NO token, filled at `costs.fill_price(1 - market_price)`, worth `1 - predicted_prob`,
   and `best_edge` (the larger of the two sides, **after the fee**) is what `--min-edge` is
   held against: `p - fill - fee_per_stake(fill)`, where the fee per unit of stake is the one the
   trade will pay (`rate x (1 - fill)` for polymarket). `pmlab.core.edge.quote_edges` is the one
   place that is computed; a live `EdgeSignal.from_quote` and the backtest both call it.
   The fee is per unit of *stake* and the gap `p - fill` per *share*, as `compute_edge` has always
   subtracted them (a flat 30 bps hid the difference); per share the fee is `fill x rate x
   (1 - fill)`, so at a fill of 0.30 the edge deducts 3.5 points where the trade pays 1.05 cents a
   share. For a cheap token the edge is therefore **conservative**: it clears `--min-edge` later than
   the exact break-even `p > fill x (1 + rate x (1 - fill))`, the one Kelly sizing uses. The
   default `--min-edge` of 0.20 is large next to that gap (2.5 points at a fill of 0.30).
3. `kelly_fraction` and `kelly_stake_size` take `costs=` and size on the post-fee payoff (see
   [below](#sizing-on-the-post-fee-payoff)).
4. `compute_metrics` averages `edge` into `avg_edge`.
5. `HoldoutGateResult` reads those metrics. **A book whose edge exists only at the
   mid, or only below the fee, can no longer pass the gate.** The gate's logic did not change in
   1.5.0; the PnL it reads did.
6. `stability_report` bootstraps the post-cost PnL, so the confidence interval is
   around what you would actually have earned.
7. `TunedForecaster` scores every trial under its `costs`, so a tuner optimises the fee it will pay.
8. `ChampionManifest.publish(costs=...)` records the model in `champion.json`, and
   `ModelRegistry.record()` carries it into the archived version — so a rollback
   restores a champion whose cost assumptions are known.
9. `PaperBroker(costs=...)` fills at the same price the backtest would, and records
   `fill_price` next to `gamma_price`, with the fee fields above.
10. `SettlementEngine` settles each trade at its recorded `fill_price` (a NO trade as a
    buy of the NO token at its own price) and books its recorded `fee_paid`, so `realized_pnl`
    carries the friction. It writes `settled_basis` (`"fill_price"`, or `"quote"` when none was
    recorded) and `settlement_version` on the trade. Releases up to 0.9.0 settled at the quote even
    when a `fill_price` was recorded and wrote neither key; `pmlab monitor` leaves such
    trades out of its PnL comparison (see [the watchdog](watchdog.md#which-trades-count)).
11. `pmlab monitor` checks each trade's `fee_paid` against the champion's cost model **at the price
    the trade was filled at**, so trades that paid a different fee than the backtest are left out
    of the PnL comparison and counted.

---

## Sizing on the post-fee payoff

`kelly_fraction(win_prob, entry_price, costs=costs)` computes the Kelly fraction of a payoff that is
net of the fee: a share-worth of payoff costs `entry_price x (1 + fee per unit of stake)`, not
`entry_price`. At 0.40 in a 0.05-rate market that is `0.40 x 1.03 = 0.412`, so a forecast of 0.60
sizes to a quarter-Kelly of 8.0% of the bankroll instead of 8.3%, and a forecast of 0.41 (an edge
the fee eats) sizes to 0, not to a positive fraction.

`kelly_stake_size(..., costs=costs)` returns the stake whose **stake plus fee** is the capped
fraction, because the exposure cap is on what leaves the bankroll: `stake + fee(stake) <=
max_exposure x bankroll`. Without `costs` both functions are the gross payoff, as before 1.5.0; they
ignore the fee, which oversizes. Slippage is not applied by them: pass the price you expect to pay
as `entry_price`.

---

## Using one cost model for backtest and live

This is the whole point of steps 8 and 9. If the backtest models 50bps of slippage
and the paper broker fills at the quote, then live-vs-backtest PnL differ for two
reasons at once — the market, and your own arithmetic — and a drift alarm cannot
separate them.

```python
costs = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="user", slippage_bps=50.0)

# 1. validate under that friction
result = rolling_origin_eval(panel, model, flat_stake=25.0, costs=costs)
gate = HoldoutGateResult.evaluate(result.trades, required_segments=[...])

# 2. record it with the champion, so the assumption travels with the model
manifest = ChampionManifest.publish(
    model=model, gate=gate, output_dir=out, plugin_family="weather_tmax", costs=costs,
)

# 3. trade under the same friction — read it back rather than retyping it
broker = PaperBroker(
    trades_path=trades, flat_stake=25.0,
    allowed_segments=manifest.get_allowed_segments(),
    costs=manifest.costs,
)
```

Step 3 is the part worth insisting on: take `costs` from the manifest instead of
constructing a second `CostModel` by hand. Two hand-written models drift the moment
someone edits one of them.

For the same quote and stake, both sides produce the same fill and the same fee — asserted across a
grid of prices and stakes in `tests/execution/test_cost_model_unified.py` and
`tests/execution/test_backtest_live_parity.py`.

`manifest.costs` is `None` for champions published before v0.8.4. Treat that as
"cost assumptions unknown" rather than "no costs".

One `CostModel` carries one schedule. A strategy over markets that charge different rates builds one
model per market (`costs.with_fee_snapshot(snapshot)`) and calls `quote_edges` / the broker with the
one that belongs to the market; there is **no per-market resolver in 1.5.0**: a backtest prices every
trade with the one schedule of the `costs` it was given.

### A fee is priced from when it was known

Polymarket changes its rates without notice, so a backtest at date T must price a trade with the
schedule that was known before T, not with today's (`FeeScheduleStore` keeps every schedule with the
moment it was seen for exactly this). `CostModel` carries that moment:

```python
snapshots = store.as_of("2026-01-05", market_ids=["m0"])        # the schedules seen before that day
costs = CostModel(slippage_bps=50.0).with_fee_snapshot(snapshots["m0"])
costs.fee_seen_at                                               # when it was seen, in UTC
```

`with_fee_snapshot` sets the schedule, `fee_source` and `fee_seen_at` together. `rolling_origin_eval`
then refuses a fold that decided at or before `fee_seen_at` (`ValueError`, before any model is
fitted), the way a label unknown at the decision cannot train the fold: a schedule read in November
would otherwise price a January trade, and the log would call every one of them `fee_source="gamma"`.
So a backtest over a long history is one run per schedule, each over the dates that schedule was
known for, not one run with today's snapshot.

A schedule given **without** a moment (`CostModel(fee_schedule=snapshot.schedule, fee_source="user")`,
or `dataclasses.replace(costs, fee_schedule=...)`) claims none and is not checked. That is the
explicit approximation, one fee for the whole history, and it is yours to defend: its trades say
`fee_source="user"`, and a rate that has changed since the history began prices the part before the
change wrongly. The live broker prices a trade with the schedule it holds now, which is the schedule
known at that moment, so it needs no such check.

### From the command line

`backtest`, `promote-champion` and `record-trades` each take `--costs PATH`, a JSON file
of the same fields (`CostModel.to_dict()`), and `--fee-model {polymarket,flat-bps}`:

```json
{"fee_model": "polymarket", "slippage_bps": 50.0, "slippage_fixed": 0.0,
 "depth_bps_per_unit": 20.0, "depth_reference_stake": 10.0,
 "fee_schedule": {"rate": 0.01}, "fee_source": "user"}
```

```bash
pmlab backtest --plugin my_family --panel panel.parquet --costs costs.json --output metrics.json
pmlab backtest --plugin my_family --panel panel.parquet --fee-model flat-bps   # repeat a 1.4.0 run
pmlab promote-champion model.pkl --gate-path gate.json --plugin my_family --expectation metrics.json
pmlab record-trades --plugin my_family            # fills under the champion's costs
```

`--fee-model` defaults to `polymarket`. It picks the model of a costs file that does not name one
(a file of only `slippage_bps`, say), and contradicts a file that does: naming `flat-bps` against a
file with a `fee_schedule` exits 1 and says which two disagree. A file with a `taker_bps` and no
`fee_model` is what every release before 1.5.0 wrote; it is the flat model, and the CLI prints a
deprecation line asking you to add `"fee_model": "flat-bps"` (it still works).

A file with **no** `fee_model`, `fee_schedule` or `taker_bps` (only slippage, or nothing) is read as
the polymarket model. 1.4.0 read it as flat 30 bps plus that slippage, so the CLI prints a
`warning:` line saying which it did and how to ask for the other (`--fee-model flat-bps`, or
`"fee_model"` in the file). Name the model, in the file or with the flag, and the line goes.

The model is written once and then carried, not retyped:

- `backtest --output` records the costs its numbers are net of (the default model if
  `--costs` and `--fee-model` were omitted) under a `costs` key, plus `total_fees` and
  `fees_pct_of_stake`.
- `promote-champion --expectation metrics.json` publishes the champion with those costs.
  `--costs` / `--fee-model` there are only needed without an expectation file, and they must agree
  with the file's if both exist: a champion cannot be promoted on a backtest it did not run.
- `record-trades` fills under `champion.json`'s costs by default. `--costs` / `--fee-model` may name
  a model when the champion records none, but not a different one: trades filled under other costs
  are left out of the watchdog's PnL comparison, so recording them would waste the evidence.
  `--fee-model` alone names only the model, so naming the champion's own is the champion's costs
  (its schedule and slippage, not the model's defaults) and naming the other exits 1 and says which
  two models disagree; `--costs` is compared whole. `promote-champion --expectation` reads
  `--fee-model` the same way against the file's costs. A
  champion that records no costs is traded under the default model (polymarket, the `"other"` rate),
  and `record-trades` says so; pass `--fee-model flat-bps` to fill it under the flat fee of 1.4.0 and before.

A champion promoted with no costs anywhere still records `costs: null`, "unknown", and
`pmlab monitor` then reports `cost_basis: "unknown"` instead of checking it.

---

## The bundled sample

The tutorial ([tutorial.md](tutorial.md)) walks the synthetic sample through tune, backtest and
gate. The gate's logic is the same under every fee model; only the PnL it reads changes. Re-running
the same notebook with nothing but its cost line swapped (`slippage_bps=50` in all three):

| fee model | Athens | Istanbul | Lisbon | Madrid | Rome | backtest total | paper total (live window) |
|---|---|---|---|---|---|---|---|
| `flat-bps`, 30 bps (what the tutorial charges) | +17.38 | +22.99 | **−16.28** | +51.31 | +59.83 | +151.5 (+0.64 per trade) | +106.6 |
| `polymarket`, rate 0.01 (the weather category) | +14.69 | +25.03 | **−15.98** | +47.06 | +54.63 | +141.4 (+0.60 per trade) | +95.3 |
| `polymarket`, rate 0.05 (the default) | +11.98 | +18.87 | **−16.81** | +11.83 | +17.63 | +60.3 (+0.26 per trade) | +50.5 |

(The columns are the gate's per-city PnL on the backtest window, the whole backtest, and the paper
trades of the live window. The tuner finds different hyper-parameters under each fee, so the trade
sets differ.)

**The decision does not change:** the whole-sample gate is **NO_GO** under all three, because Lisbon
fails with `negative_pnl`, and the gate scoped to Athens, Istanbul, Madrid and Rome is a **GO**, as
before. What moves is the margin: under the default schedule the four cities earn less than half of
what the flat fee showed, Madrid's gate PnL falls from +51 to +12, and in the live window Rome's
paper trades end at −0.35. A GO that is thin is the difference the old fee hid. The notebook test
`tests/notebooks/test_tutorial_fee_gate.py` recomputes this on every notebook run and fails if the
decision moves.

The tutorial itself still charges `CostModel(fee_model="flat-bps", slippage_bps=50.0)`, so the
numbers on its page stay as written; the tutorial and the site's home page are re-run under the
default model with the rest of the migration (#74).

---

## Compatibility and migration

What did **not** move, and why:

- **An old record, champion or tuner replays with the model it ran under.** A cost dict with no
  `fee_model` (every `champion.json`, reproducibility record, tuner and `--costs` file written by
  1.4.0 and before) is the flat-bps model: `CostModel.from_dict` reads it that way, without a
  warning. So `pmlab reproduce` of a 1.4.0 record gives the numbers it gave, not the new default's,
  and `record-trades` trades a champion promoted under the flat fee under the flat fee.
- **The signature.** `CostModel(taker_bps=30.0, slippage_bps=0.0, ...)` keeps its fields, their order
  and their defaults; `fee_model`, `fee_schedule`, `fee_source` and `fee_seen_at` are appended.
  `taker_bps` still reads 30.0 on a default model, though it is unused there.
- **A `CostModel` pickled by 1.4.0** loads as the flat-bps model it was (pmlab itself never pickles
  one; a notebook or a cache might).
- `CostModel.frictionless()`, `estimate_fee`, `fill_price`, slippage and depth.
- `PaperBroker.taker_bps` and `TunedForecaster.taker_bps` still exist and read through to
  `costs.taker_bps`. That is the flat model's rate and means something only for `fee_model="flat-bps"`:
  under the polymarket model they read the unused 30.0 and say nothing of the fee charged, so read
  `costs` (`costs.fee_schedule`, `costs.fee(stake, price)`) to display it.

What moved:

- **The default fee**, as above: the default model is polymarket at 0.05. This is a result that was
  wrong, corrected in a minor release on the ground [the stability page](api-stability.md) gives: the
  old default charged a taker 0.30% of the stake where Polymarket charges 2.5% at a price of 0.50
  (0.05-rate market). Everything recorded before keeps the fee it ran under, and the old model is one
  argument away.
- A `--costs` file with no model, schedule or `taker_bps` is read as polymarket (1.4.0 read it as
  flat-bps); the CLI says so (see above).
- `CostModel.fee(stake)` needs the price under the polymarket model (see above).
- `CostModel.to_dict()` names the model and carries the schedule, so it is a mapping of mixed
  values (`dict[str, Any]`, not `dict[str, float]`), as are `BacktestConfig.costs` and the costs in a
  watchdog report. It still round-trips: `CostModel.from_dict(costs.to_dict()) == costs`.
- A `fee_paid` is checked by the watchdog against the champion's model at the trade's fill price.

**Deprecated since 1.5.0** (removed in 2.0.0; see [the policy](deprecation.md)): the bare
`taker_bps` as the way to pick a fee model, in four places:

| You wrote | Write instead |
|---|---|
| `CostModel(taker_bps=50.0)` | `CostModel(fee_model="flat-bps", taker_bps=50.0)`, or the default model |
| `rolling_origin_eval(..., taker_bps=50.0)` | `costs=CostModel(fee_model="flat-bps", taker_bps=50.0)` |
| `TunedForecaster(..., taker_bps=50.0)`, `PaperBroker(..., taker_bps=50.0)` | the same `costs=` |
| a `--costs` file with `taker_bps` and no `fee_model` | add `"fee_model": "flat-bps"`, or drop `taker_bps` |

Passing both `taker_bps=` and `costs=` still raises `ValueError`: two sources of truth for the fee
would silently double-charge it. To find every call that needs updating, turn the warning into an
error (`pytest -W error::pmlab.PmlabDeprecationWarning`).

## Reproducibility

`CostModel.to_dict()` returns a mapping that round-trips through `CostModel.from_dict` (and
`CostModel(**payload)`), so log it with the run:

```python
with tracker.run(experiment="weather_tmax", run_name="with_costs") as run:
    run.log_params(costs.to_dict())
```

The reproducibility record carries it as `config.costs`: the fee model, the schedule and where it
came from. That is what lets a run be told from one priced under another model, and what
`pmlab reproduce` replays. A PnL number without its cost model is not reproducible — the same panel
and the same model give a different answer under different friction.

## What is not modelled

- **Maker fills.** Every simulated fill is a taker fill; makers pay no fee and earn a rebate (#73).
- **Other venues' fees** (Kalshi, a negotiated rate) as a pluggable model (#75).
- **A walk of the real order book.** Fills still use the quote plus the linear depth penalty; a
  point-in-time book would replace it (#76).
