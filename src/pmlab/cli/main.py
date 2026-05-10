"""CLI entry point — thin typer shell (filled in Phase 8)."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="pmlab",
    help="Generic ML framework for Polymarket prediction markets.",
    no_args_is_help=True,
)


@app.command("version")
def version_cmd() -> None:
    """Print the installed version."""
    from pmlab import __version__

    typer.echo(f"pmlab {__version__}")


@app.command("status")
def status_cmd() -> None:
    """Show current champion, open trades count, and cumulative PnL."""
    typer.echo("pmlab status: no champion published yet.")


@app.command("scan-markets")
def scan_markets_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p", help="Plugin family name (e.g. weather_tmax)"),
    workspace: str = typer.Option("ops_daily", "--workspace", "-w"),
) -> None:
    """Discover open markets for a plugin family."""
    typer.echo(f"[scan-markets] plugin={plugin} workspace={workspace}")
    typer.echo("(Configure plugin clients via environment variables)")


@app.command("record-trades")
def record_trades_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p"),
    workspace: str = typer.Option("ops_daily", "--workspace", "-w"),
    min_edge: float = typer.Option(0.20, "--min-edge"),
) -> None:
    """Record paper trades from latest scan-edge signals."""
    typer.echo(f"[record-trades] plugin={plugin} workspace={workspace} min_edge={min_edge}")


@app.command("settle-trades")
def settle_trades_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p"),
    workspace: str = typer.Option("ops_daily", "--workspace", "-w"),
) -> None:
    """Settle open paper trades against resolved market truth."""
    typer.echo(f"[settle-trades] plugin={plugin} workspace={workspace}")


@app.command("backtest")
def backtest_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p"),
    model: str = typer.Option("lgbm_baseline", "--model", "-m"),
    stride: int = typer.Option(30, "--stride", help="Walk-forward stride (min 10, never 1 in production)"),
    workspace: str = typer.Option("historical_real", "--workspace", "-w"),
) -> None:
    """Run walk-forward backtest for a plugin+model combination."""
    if stride < 10:
        typer.echo(f"[error] stride={stride} is too low — minimum 10. Default stride=1 takes hours.", err=True)
        raise typer.Exit(1)
    typer.echo(f"[backtest] plugin={plugin} model={model} stride={stride} workspace={workspace}")


@app.command("promote-champion")
def promote_champion_cmd(
    model_path: str = typer.Argument(..., help="Path to trained model .pkl"),
    gate_path: str = typer.Option(..., "--gate-path", help="Path to holdout gate JSON"),
    plugin: str = typer.Option(..., "--plugin", "-p"),
    output_dir: str = typer.Option("artifacts/public_models", "--output-dir"),
) -> None:
    """Promote a model to champion if gate decision is GO."""
    typer.echo(f"[promote-champion] model={model_path} gate={gate_path} plugin={plugin}")


if __name__ == "__main__":
    app()
