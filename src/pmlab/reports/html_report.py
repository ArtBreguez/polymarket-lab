"""HTML report generator for paper trading sessions."""
from __future__ import annotations
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

__all__ = ["generate_report"]

_STYLE = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
body { background: #0f1117; color: #e2e8f0; padding: 32px; }
h1 { font-size: 1.6rem; margin-bottom: 4px; color: #f8fafc; }
.subtitle { color: #94a3b8; font-size: 0.85rem; margin-bottom: 32px; }
h2 { font-size: 1.1rem; margin: 28px 0 12px; color: #cbd5e1; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th { background: #1e293b; color: #94a3b8; padding: 8px 12px; text-align: left; font-weight: 600; }
td { padding: 7px 12px; border-bottom: 1px solid #1e293b; color: #cbd5e1; }
tr:hover td { background: #1e2a3a; }
.pos { color: #4ade80; }
.neg { color: #f87171; }
.card { background: #1e293b; border-radius: 8px; padding: 20px 24px; display: inline-block; margin: 0 12px 12px 0; min-width: 160px; }
.card-label { color: #64748b; font-size: 0.75rem; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.card-value { font-size: 1.5rem; font-weight: 700; color: #f8fafc; }
.cards { margin-bottom: 24px; }
svg text { font-family: inherit; }
.chart-wrap { background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
</style>
"""


def generate_report(
    trades: list[dict[str, Any]],
    output_path: Path | str = Path("pmlab_report.html"),
    title: str = "pmlab Paper Trading Report",
    brier_score: float | None = None,
) -> Path:
    output_path = Path(output_path)
    settled = [t for t in trades if t.get("realized_pnl") is not None]
    open_trades = [t for t in trades if t.get("realized_pnl") is None]

    total_pnl = sum(t["realized_pnl"] for t in settled)
    hit_rate = (
        sum(1 for t in settled if (t.get("realized_pnl") or 0) > 0) / len(settled)
        if settled else 0.0
    )
    total_staked = sum(t.get("flat_stake", 0) for t in settled)
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0
    avg_edge = (
        sum(t.get("edge_after_fee", 0) for t in settled) / len(settled) if settled else 0.0
    )

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    html_parts = [
        "<!DOCTYPE html><html lang='en'><head>",
        f"<meta charset='UTF-8'><title>{title}</title>",
        _STYLE,
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p class='subtitle'>Generated {generated_at} &mdash; "
        f"{len(trades)} total trades ({len(settled)} settled, {len(open_trades)} open)</p>",
        "<div class='cards'>",
        _card("Total PnL", f"${total_pnl:+.2f}", "pos" if total_pnl >= 0 else "neg"),
        _card("ROI", f"{roi:+.1f}%", "pos" if roi >= 0 else "neg"),
        _card("Hit Rate", f"{hit_rate*100:.1f}%"),
        _card("Avg Edge", f"{avg_edge*100:.2f}%"),
        _card("Settled", str(len(settled))),
    ]
    if brier_score is not None:
        html_parts.append(_card("Brier Score", f"{brier_score:.4f}"))
    html_parts.append("</div>")

    if settled:
        html_parts.append("<h2>Equity Curve</h2>")
        html_parts.append("<div class='chart-wrap'>")
        html_parts.append(_equity_curve_svg(settled))
        html_parts.append("</div>")

    html_parts.append("<h2>Per-Segment Breakdown</h2>")
    html_parts.append(_segment_table(settled))
    html_parts.append("<h2>Trade Log</h2>")
    html_parts.append(_trade_table(trades))
    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path


def _card(label: str, value: str, extra_class: str = "") -> str:
    value_class = f"card-value {extra_class}".strip()
    return f"<div class='card'><div class='card-label'>{label}</div><div class='{value_class}'>{value}</div></div>"


def _equity_curve_svg(settled: list[dict[str, Any]]) -> str:
    sorted_trades = sorted(settled, key=lambda t: t.get("recorded_at", ""))
    pnl_values = [t.get("realized_pnl", 0) for t in sorted_trades]
    cumulative: list[float] = []
    running = 0.0
    for v in pnl_values:
        running += v
        cumulative.append(running)
    if not cumulative:
        return "<p style='color:#64748b'>No data.</p>"
    W, H, pad = 800, 200, 40
    min_v, max_v = min(cumulative), max(cumulative)
    if min_v == max_v:
        min_v -= 1; max_v += 1
    x_scale = (W - 2 * pad) / max(len(cumulative) - 1, 1)
    y_scale = (H - 2 * pad) / (max_v - min_v)
    def to_xy(i: int, v: float) -> tuple[float, float]:
        return pad + i * x_scale, H - pad - (v - min_v) * y_scale
    points = [to_xy(i, v) for i, v in enumerate(cumulative)]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    final_color = "#4ade80" if cumulative[-1] >= 0 else "#f87171"
    zero_y = H - pad - (0 - min_v) * y_scale
    zero_line = f"<line x1='{pad}' y1='{zero_y:.1f}' x2='{W-pad}' y2='{zero_y:.1f}' stroke='#334155' stroke-width='1' stroke-dasharray='4'/>" if pad <= zero_y <= H - pad else ""
    fx, fy = points[-1]
    return (
        f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg'>"
        f"{zero_line}"
        f"<polyline points='{polyline}' fill='none' stroke='{final_color}' stroke-width='2'/>"
        f"<text x='{fx+4:.1f}' y='{fy:.1f}' fill='{final_color}' font-size='11'>${cumulative[-1]:+.2f}</text>"
        f"<text x='{pad}' y='{H-4}' fill='#475569' font-size='10'>Trade #1</text>"
        f"<text x='{W-pad}' y='{H-4}' fill='#475569' font-size='10' text-anchor='end'>Trade #{len(cumulative)}</text>"
        f"</svg>"
    )


def _segment_table(settled: list[dict[str, Any]]) -> str:
    from collections import defaultdict
    seg_data: dict[str, dict[str, Any]] = defaultdict(lambda: {"pnl": 0.0, "count": 0, "wins": 0})
    for t in settled:
        seg = t.get("city_or_segment", "unknown")
        pnl = t.get("realized_pnl", 0) or 0.0
        seg_data[seg]["pnl"] += pnl
        seg_data[seg]["count"] += 1
        if pnl > 0:
            seg_data[seg]["wins"] += 1
    rows = []
    for seg, d in sorted(seg_data.items()):
        hit = d["wins"] / d["count"] * 100 if d["count"] else 0
        pnl_class = "pos" if d["pnl"] >= 0 else "neg"
        rows.append(f"<tr><td>{seg}</td><td>{d['count']}</td><td class='{pnl_class}'>${d['pnl']:+.2f}</td><td>{hit:.0f}%</td></tr>")
    return "<table><thead><tr><th>Segment</th><th>Trades</th><th>PnL</th><th>Hit Rate</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _trade_table(trades: list[dict[str, Any]]) -> str:
    rows = []
    for t in trades:
        pnl = t.get("realized_pnl")
        pnl_str = f"${pnl:+.2f}" if pnl is not None else "open"
        pnl_class = "" if pnl is None else ("pos" if pnl >= 0 else "neg")
        price = t.get("gamma_price", "")
        price_str = f"{price:.3f}" if isinstance(price, float) else str(price)
        rows.append(
            f"<tr><td>{t.get('target_date','')}</td><td>{t.get('city_or_segment','')}</td>"
            f"<td>{t.get('direction','')}</td><td>{t.get('outcome_label','')}</td>"
            f"<td>{price_str}</td><td>{t.get('edge_after_fee',0)*100:.2f}%</td>"
            f"<td class='{pnl_class}'>{pnl_str}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Date</th><th>Segment</th><th>Dir</th><th>Outcome</th>"
        "<th>Price</th><th>Edge</th><th>PnL</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
