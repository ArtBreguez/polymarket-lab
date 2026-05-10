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


if __name__ == "__main__":
    app()
