"""pmlab CLI — real implementation using WorkspaceContext and ChampionManifest."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="pmlab",
    help="Generic ML framework for Polymarket prediction markets.",
    no_args_is_help=True,
)

console = Console()


@app.command("version")
def version_cmd() -> None:
    """Print the installed pmlab version."""
    from pmlab import __version__

    rprint(f"[bold cyan]pmlab[/bold cyan] [green]{__version__}[/green]")


@app.command("status")
def status_cmd(
    artifacts_dir: str = typer.Option(
        "artifacts",
        "--artifacts-dir",
        "-a",
        envvar="PMLAB_ARTIFACTS_DIR",
        help="Root artifacts directory",
    ),
    workspace: str = typer.Option(
        "ops_daily",
        "--workspace",
        "-w",
        envvar="PMLAB_WORKSPACE",
        help="Active workspace name",
    ),
) -> None:
    """Show champion status, open trades, and cumulative PnL."""
    from pmlab.config import PmlabSettings

    settings = PmlabSettings()
    settings.artifacts_dir = Path(artifacts_dir)
    settings.workspace = workspace

    champion_path = settings.champion_json_path
    trades_path = settings.trades_path

    # Champion
    if champion_path.exists():
        data = json.loads(champion_path.read_text())
        console.print("\n[bold]Champion[/bold]")
        table = Table(show_header=False, box=None)
        table.add_row("Model", data.get("model_name", "?"))
        table.add_row("Plugin", data.get("plugin_family", "?"))
        table.add_row("Published", data.get("published_at", "?")[:19])
        gate = data.get("publish_gate", {})
        table.add_row("Gate", gate.get("decision", "?"))
        table.add_row("Agg PnL", f"${gate.get('aggregate_pnl', 0):.2f}")
        table.add_row("Agg Trades", str(gate.get("aggregate_trades", 0)))
        segs = [
            f"[green]{r['segment']}[/green]" if r["passes"] else f"[red]{r['segment']}[/red]"
            for r in gate.get("segment_results", [])
        ]
        table.add_row("Segments", ", ".join(segs) if segs else "none")
        console.print(table)
    else:
        rprint("[yellow]No champion published yet.[/yellow]")

    # Trades
    if trades_path.exists():
        raw = json.loads(trades_path.read_text())
        trades = raw.get("trades", [])
        open_t = [t for t in trades if t.get("realized_pnl") is None]
        settled_t = [t for t in trades if t.get("realized_pnl") is not None]
        total_pnl = sum(t["realized_pnl"] for t in settled_t)
        pnl_color = "green" if total_pnl >= 0 else "red"
        console.print(f"\n[bold]Paper Trades[/bold] (workspace: {workspace})")
        table2 = Table(show_header=False, box=None)
        table2.add_row("Total trades", str(len(trades)))
        table2.add_row("Open", str(len(open_t)))
        table2.add_row("Settled", str(len(settled_t)))
        table2.add_row("Total PnL", f"[{pnl_color}]${total_pnl:+.2f}[/{pnl_color}]")
        if settled_t:
            wins = sum(1 for t in settled_t if t["realized_pnl"] > 0)
            hit = wins / len(settled_t) * 100
            table2.add_row("Hit Rate", f"{hit:.1f}%")
        console.print(table2)
    else:
        rprint(f"[yellow]No trades file found at {trades_path}[/yellow]")


@app.command("scan-markets")
def scan_markets_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p", help="Plugin family name"),
    tag: str | None = typer.Option(None, "--tag", help="Gamma API tag filter"),
    keyword: str | None = typer.Option(None, "--keyword", help="Keyword filter"),
    limit: int = typer.Option(100, "--limit", help="Max markets to fetch"),
) -> None:
    """Discover open markets from Polymarket Gamma API."""
    from pmlab.config import get_settings
    from pmlab.markets.gamma_client import GammaClient

    settings = get_settings()
    rprint(
        f"[bold]Scanning markets[/bold] (plugin={plugin}, tag={tag or 'none'}, keyword={keyword or 'none'})"
    )

    with GammaClient(base_url=settings.gamma_api_base) as client:
        try:
            markets = client.fetch_markets(tag=tag, keyword=keyword, limit=limit)
        except Exception as e:
            rprint(f"[red]Error fetching markets: {e}[/red]")
            raise typer.Exit(1) from e

    if not markets:
        rprint("[yellow]No markets found.[/yellow]")
        return

    table = Table(title=f"{len(markets)} markets found")
    table.add_column("Question", max_width=60)
    table.add_column("Condition ID", max_width=20)
    table.add_column("Active")
    for m in markets[:50]:  # show first 50
        cid = m.get("conditionId", m.get("condition_id", ""))
        q = m.get("question", "?")[:60]
        active = "[green]yes[/green]" if m.get("active") else "[red]no[/red]"
        table.add_row(q, str(cid)[:18], active)
    console.print(table)
    if len(markets) > 50:
        rprint(f"[dim]... and {len(markets) - 50} more[/dim]")


@app.command("record-trades")
def record_trades_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p"),
    workspace: str = typer.Option("ops_daily", "--workspace", "-w", envvar="PMLAB_WORKSPACE"),
    artifacts_dir: str = typer.Option(
        "artifacts", "--artifacts-dir", "-a", envvar="PMLAB_ARTIFACTS_DIR"
    ),
    min_edge: float = typer.Option(0.20, "--min-edge"),
    flat_stake: float = typer.Option(1.0, "--flat-stake"),
) -> None:
    """Record paper trades from latest scan-edge signals."""
    from pmlab.config import PmlabSettings
    from pmlab.execution.paper_broker import PaperBroker
    from pmlab.modeling.champion import ChampionManifest

    settings = PmlabSettings()
    settings.artifacts_dir = Path(artifacts_dir)

    if not settings.champion_json_path.exists():
        rprint("[red]No champion found. Run 'pmlab promote-champion' first.[/red]")
        raise typer.Exit(1)

    manifest = ChampionManifest.load(settings.champion_json_path)
    allowed = manifest.get_allowed_segments()
    rprint(f"Champion: [green]{manifest.model_name}[/green], allowed segments: {allowed}")

    # Look for signals file in workspace
    signals_path = (
        Path(artifacts_dir) / "workspaces" / workspace / "signals" / "v2" / "scan_edge_signals.json"
    )
    if not signals_path.exists():
        rprint(f"[yellow]No signals file found at {signals_path}.[/yellow]")
        rprint("[dim]Signals are generated by your plugin's scan-edge step.[/dim]")
        raise typer.Exit(0)

    raw = json.loads(signals_path.read_text())
    all_signals = raw.get("signals", [])

    from pmlab.execution.edge_signal import EdgeSignal

    signals = [EdgeSignal(**s) for s in all_signals if float(s.get("best_edge", 0)) >= min_edge]
    rprint(f"Found [cyan]{len(signals)}[/cyan] signals with edge >= {min_edge}")

    trades_path = (
        Path(artifacts_dir)
        / "workspaces"
        / workspace
        / "signals"
        / "v2"
        / "forward_paper_trades.json"
    )
    broker = PaperBroker(
        trades_path=trades_path,
        allowed_segments=allowed,
        flat_stake=flat_stake,
    )
    new_trades = broker.record(signals)
    rprint(f"[green]Recorded {len(new_trades)} new trades.[/green]")
    for t in new_trades:
        rprint(
            f"  + {t['city_or_segment']} {t['target_date']} {t['outcome_label']} @ {t['gamma_price']:.3f} (edge {t['edge_after_fee']:.3f})"
        )


@app.command("settle-trades")
def settle_trades_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p"),
    workspace: str = typer.Option("ops_daily", "--workspace", "-w", envvar="PMLAB_WORKSPACE"),
    artifacts_dir: str = typer.Option(
        "artifacts", "--artifacts-dir", "-a", envvar="PMLAB_ARTIFACTS_DIR"
    ),
) -> None:
    """Settle open paper trades against resolved market truth."""
    rprint(f"[bold]settle-trades[/bold] plugin={plugin} workspace={workspace}")
    rprint("[dim]Settlement requires a plugin instance with real data connections.[/dim]")
    rprint("[dim]Use your plugin's workspace script or call SettlementEngine directly.[/dim]")
    rprint(
        '[dim]Example: scripts/pmlab-workspace ops_daily python -c "from pmlab import SettlementEngine; ..."[/dim]'
    )


@app.command("backtest")
def backtest_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p"),
    panel_path: str = typer.Option(..., "--panel", help="Path to historical panel .parquet file"),
    model: str = typer.Option("lgbm_baseline", "--model", "-m"),
    stride: int = typer.Option(30, "--stride"),
    min_train_rows: int = typer.Option(20, "--min-train-rows"),
    flat_stake: float = typer.Option(1.0, "--flat-stake"),
    output: str | None = typer.Option(None, "--output", "-o", help="Save metrics JSON to path"),
) -> None:
    """Run walk-forward backtest on a historical panel."""
    if stride < 10:
        rprint(f"[red]stride={stride} is too low — minimum 10.[/red]")
        raise typer.Exit(1)

    import pandas as pd

    from pmlab.backtest.rolling_origin import rolling_origin_eval
    from pmlab.modeling.lgbm_baseline import LGBMForecaster

    panel_file = Path(panel_path)
    if not panel_file.exists():
        rprint(f"[red]Panel file not found: {panel_path}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold]Backtest[/bold] plugin={plugin} model={model} stride={stride}")
    rprint(f"Loading panel from [cyan]{panel_path}[/cyan]...")

    if panel_path.endswith(".parquet"):
        panel = pd.read_parquet(panel_file)
    elif panel_path.endswith(".csv"):
        panel = pd.read_csv(panel_file)
    else:
        rprint("[red]Panel must be .parquet or .csv[/red]")
        raise typer.Exit(1)

    rprint(
        f"Panel: {len(panel)} rows, {panel['decision_date'].nunique()} dates, {panel['market_id'].nunique()} markets"
    )

    forecaster = LGBMForecaster()
    with console.status("Running walk-forward backtest..."):
        result = rolling_origin_eval(
            panel, forecaster, min_train_rows=min_train_rows, stride=stride, flat_stake=flat_stake
        )

    trades = result.trades
    if trades.empty:
        rprint("[yellow]No trades produced. Try reducing stride or min_train_rows.[/yellow]")
        raise typer.Exit(0)

    total_pnl = trades["realized_pnl"].sum()
    hit_rate = (trades["realized_pnl"] > 0).mean()
    avg_edge = trades["edge"].mean()
    pnl_color = "green" if total_pnl >= 0 else "red"

    table = Table(title="Backtest Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total trades", str(len(trades)))
    table.add_row("Steps evaluated", str(len(result.steps)))
    table.add_row("Total PnL", f"[{pnl_color}]${total_pnl:+.2f}[/{pnl_color}]")
    table.add_row("Hit rate", f"{hit_rate:.1%}")
    table.add_row("Avg edge", f"{avg_edge:.4f}")
    console.print(table)

    if output:
        metrics = {
            "total_trades": len(trades),
            "steps": len(result.steps),
            "total_pnl": float(total_pnl),
            "hit_rate": float(hit_rate),
            "avg_edge": float(avg_edge),
        }
        Path(output).write_text(json.dumps(metrics, indent=2))
        rprint(f"[dim]Metrics saved to {output}[/dim]")


@app.command("promote-champion")
def promote_champion_cmd(
    model_path: str = typer.Argument(..., help="Path to trained model .pkl"),
    gate_path: str = typer.Option(..., "--gate-path", help="Path to holdout gate JSON"),
    plugin: str = typer.Option(..., "--plugin", "-p"),
    output_dir: str = typer.Option("artifacts/public_models", "--output-dir"),
    model_name: str = typer.Option("champion", "--name"),
) -> None:
    """Promote a model to champion if gate decision is GO."""
    from pmlab.backtest.holdout_gate import HoldoutGateResult
    from pmlab.modeling.champion import ChampionManifest
    from pmlab.modeling.lgbm_baseline import LGBMForecaster

    model_file = Path(model_path)
    gate_file = Path(gate_path)

    if not model_file.exists():
        rprint(f"[red]Model file not found: {model_path}[/red]")
        raise typer.Exit(1)
    if not gate_file.exists():
        rprint(f"[red]Gate file not found: {gate_path}[/red]")
        raise typer.Exit(1)

    gate_data = json.loads(gate_file.read_text())
    gate = HoldoutGateResult.from_dict(gate_data)

    rprint(
        f"Gate decision: [{'green' if gate.decision == 'GO' else 'red'}]{gate.decision}[/{'green' if gate.decision == 'GO' else 'red'}]"
    )
    rprint(f"Aggregate PnL: ${gate.aggregate_pnl:+.2f} over {gate.aggregate_trades} trades")

    if gate.decision != "GO":
        rprint("[red]Cannot promote — gate is NO_GO. Investigate failing segments.[/red]")
        for r in gate.segment_results:
            status = "[green]PASS[/green]" if r.passes else "[red]FAIL[/red]"
            rprint(
                f"  {status} {r.segment}: {r.num_trades} trades, ${r.total_pnl:+.2f} ({r.reason})"
            )
        raise typer.Exit(1)

    model = LGBMForecaster.load(model_file)
    try:
        manifest = ChampionManifest.publish(
            model=model,
            gate=gate,
            output_dir=Path(output_dir),
            plugin_family=plugin,
            model_name=model_name,
        )
        rprint("[green]Champion published![/green]")
        rprint(f"  Model: {manifest.model_path}")
        rprint(f"  Manifest: {output_dir}/champion.json")
        rprint(f"  Allowed segments: {manifest.get_allowed_segments()}")
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command("report")
def report_cmd(
    trades: str = typer.Option(..., "--trades", "-t", help="Path to forward_paper_trades.json"),
    output: str = typer.Option("pmlab_report.html", "--output", "-o", help="Output HTML path"),
    title: str = typer.Option("pmlab Paper Trading Report", "--title", help="Report title"),
    brier_score: float | None = typer.Option(
        None, "--brier", help="Optional Brier score to display"
    ),
) -> None:
    """Generate a self-contained HTML report from a paper trades JSON file."""
    import json as _json

    from pmlab.reports.html_report import generate_report

    trades_path = Path(trades)
    if not trades_path.exists():
        rprint(f"[red]Trades file not found: {trades}[/red]")
        raise typer.Exit(1)

    raw = _json.loads(trades_path.read_text())
    all_trades = raw.get("trades", [])

    if not all_trades:
        rprint("[yellow]No trades found in file.[/yellow]")

    output_path = Path(output)
    result = generate_report(
        trades=all_trades,
        output_path=output_path,
        title=title,
        brier_score=brier_score,
    )

    settled = [t for t in all_trades if t.get("realized_pnl") is not None]
    open_t = [t for t in all_trades if t.get("realized_pnl") is None]
    total_pnl = sum(t["realized_pnl"] for t in settled)
    pnl_color = "green" if total_pnl >= 0 else "red"

    rprint(f"[bold]Report generated:[/bold] {result}")
    rprint(f"  Trades: {len(all_trades)} total ({len(settled)} settled, {len(open_t)} open)")
    rprint(f"  Total PnL: [{pnl_color}]${total_pnl:+.2f}[/{pnl_color}]")


if __name__ == "__main__":
    app()
