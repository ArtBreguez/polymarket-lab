# HTML Reports — pmlab.reports

`generate_report` produces a self-contained dark-themed HTML file from your paper trading results. No external JS or CSS dependencies — everything is inline.

---

## Basic Usage

```python
from pmlab import generate_report, PaperBroker

broker = PaperBroker(trades_path="artifacts/ops/trades.json", ...)
trades = broker.load_trades()

output = generate_report(
    trades,
    output_path="reports/may_2026.html",
    title="May 2026 — Buenos Aires Paper Trading",
)
print(f"Report saved to: {output}")
```

Open the file in any browser — no server required.

---

## With Brier Score

```python
from pmlab import generate_report, brier_decomposition

diag = brier_decomposition(y_true, y_prob)
generate_report(
    trades,
    output_path="reports/session.html",
    brier_score=diag.brier_score,
)
```

The Brier score appears as an additional summary card.

---

## Report Sections

### Summary Cards

| Card | Description |
|---|---|
| Total PnL | Sum of `realized_pnl` for all settled trades (green/red) |
| ROI | `total_pnl / total_staked * 100` |
| Hit Rate | % of settled trades with `realized_pnl > 0` |
| Avg Edge | Mean `edge_after_fee` across settled trades |
| Settled | Count of settled trades |
| Brier Score | Optional — shown if passed to `generate_report` |

### Equity Curve

SVG polyline of cumulative PnL over settled trades, ordered by `recorded_at`. A dashed zero-line is drawn if the curve crosses zero. Final PnL is labeled at the endpoint.

### Per-Segment Breakdown

Table of trade count, total PnL, and hit rate for each `city_or_segment`. Sorted alphabetically.

### Trade Log

Full table of all trades (settled and open), with columns: Date, Segment, Direction, Outcome, Price, Edge, PnL (`open` for unsettled).

---

## Trade Dict Format

`generate_report` works with the output of `PaperBroker.load_trades()`:

```python
{
    "recorded_at": "2026-05-01T10:00:00+00:00",  # ISO timestamp
    "city_or_segment": "Buenos Aires",
    "target_date": "2026-05-02",
    "outcome_label": "above_30",
    "direction": "yes",
    "gamma_price": 0.35,
    "edge_after_fee": 0.08,
    "flat_stake": 5.0,
    "realized_pnl": 9.28,   # None for open trades
}
```

Fields not present default to empty string or 0 — the report will not crash on partial data.

---

## Automating Weekly Reports

```python
from pathlib import Path
from datetime import date
from pmlab import generate_report, PaperBroker

def generate_weekly_report(trades_path: Path, reports_dir: Path) -> Path:
    broker = PaperBroker(trades_path=trades_path, flat_stake=5.0)
    trades = broker.load_trades()
    week = date.today().strftime("%Y-W%U")
    output = reports_dir / f"report_{week}.html"
    return generate_report(trades, output_path=output, title=f"pmlab Weekly — {week}")
```
