"""pmlab CLI — real implementation using WorkspaceContext and ChampionManifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from pmlab.core.costs import CostModel
    from pmlab.core.policy import TradePolicy

app = typer.Typer(
    name="pmlab",
    help="Generic ML framework for Polymarket prediction markets.",
    no_args_is_help=True,
)

console = Console()

_COSTS_HELP = (
    "JSON file of CostModel fields (taker_bps, slippage_bps, slippage_fixed, "
    "depth_bps_per_unit, depth_reference_stake), or any JSON with them under a "
    "'costs' key such as the file `pmlab backtest --output` writes. "
)


def _costs_from(raw: object, source: str) -> CostModel:
    """A CostModel from parsed JSON, or a clean exit 1 naming `source`."""
    from pmlab.core.costs import CostModel

    fields = raw.get("costs", raw) if isinstance(raw, dict) else raw
    try:
        if not isinstance(fields, dict):
            raise TypeError("expected a JSON object of cost fields")
        return CostModel(**fields)
    except (TypeError, ValueError) as e:
        rprint(f"[red]Not a valid cost model ({source}): {e}[/red]")
        raise typer.Exit(1) from e


def _load_costs(path: str | None) -> CostModel | None:
    """Read a `--costs` file; None when the option was not given."""
    if path is None:
        return None
    file = Path(path)
    if not file.is_file():
        rprint(f"[red]Costs file not found: {path}[/red]")
        raise typer.Exit(1)
    try:
        raw = json.loads(file.read_text())
    except (OSError, ValueError) as e:
        rprint(f"[red]Not a valid cost model ({path}): costs file is unreadable: {e}[/red]")
        raise typer.Exit(1) from e
    return _costs_from(raw, path)


def _describe_segments(allowed: set[str]) -> str:
    """The allowed segments for the console; ``"all"`` is what an aggregate gate allows."""
    from pmlab.core.segments import ALL_SEGMENTS

    if ALL_SEGMENTS in allowed:
        return "every segment (the gate was graded in aggregate)"
    return str(allowed)


def _describe_policy(policy: TradePolicy) -> str:
    """One line for the console: what the backtest traded."""
    if policy.selection == "top_prob":
        return "top-bin-yes (YES on the most probable bin, every market, no threshold)"
    sides = "+".join(d.upper() for d in policy.directions)
    return f"live (best edge >= {policy.min_edge:g}, {sides})"


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
    from pmlab.plugins.registry import available_families, build_plugin

    settings = get_settings()
    rprint(
        f"[bold]Scanning markets[/bold] (plugin={plugin}, tag={tag or 'none'}, keyword={keyword or 'none'})"
    )

    with GammaClient(base_url=settings.gamma_api_base) as client:
        try:
            market_plugin = build_plugin(plugin, gamma_client=client)
        except KeyError as e:
            rprint(f"[red]{e}[/red]")
            rprint(f"[dim]Available plugins: {available_families()}[/dim]")
            raise typer.Exit(1) from e

        # Delegate discovery to the plugin so results are filtered to its market
        # family (e.g. weather_tmax returns only temperature markets, not
        # whatever the raw Gamma feed happens to surface).
        discover_kwargs: dict[str, object] = {"limit": limit}
        if tag is not None:
            discover_kwargs["tag"] = tag
        if keyword is not None:
            discover_kwargs["keyword"] = keyword
        try:
            specs = market_plugin.discover_markets(**discover_kwargs)
        except Exception as e:
            rprint(f"[red]Error discovering markets: {e}[/red]")
            raise typer.Exit(1) from e

    if not specs:
        rprint(f"[yellow]No open '{plugin}' markets found right now.[/yellow]")
        return

    table = Table(title=f"{len(specs)} '{plugin}' markets found")
    table.add_column("Question", max_width=60)
    table.add_column("Market ID", max_width=20)
    table.add_column("Family")
    for s in specs[:50]:  # show first 50
        table.add_row(s.question[:60], str(s.market_id)[:18], s.market_family)
    console.print(table)
    if len(specs) > 50:
        rprint(f"[dim]... and {len(specs) - 50} more[/dim]")


@app.command("record-trades")
def record_trades_cmd(
    plugin: str = typer.Option(..., "--plugin", "-p"),
    workspace: str = typer.Option("ops_daily", "--workspace", "-w", envvar="PMLAB_WORKSPACE"),
    artifacts_dir: str = typer.Option(
        "artifacts", "--artifacts-dir", "-a", envvar="PMLAB_ARTIFACTS_DIR"
    ),
    min_edge: float | None = typer.Option(
        None,
        "--min-edge",
        help=(
            "Smallest after-fee edge worth a trade. Default: the threshold the champion's "
            "backtest ran with (recorded in champion.json), else 0.20."
        ),
    ),
    flat_stake: float = typer.Option(1.0, "--flat-stake"),
    costs_path: str | None = typer.Option(
        None,
        "--costs",
        help=(
            _COSTS_HELP + "Default: the costs recorded in champion.json, so paper fills match "
            "the backtest. Must agree with them if both exist."
        ),
    ),
) -> None:
    """Record paper trades from latest scan-edge signals."""
    from pmlab.config import PmlabSettings
    from pmlab.core.edge import DEFAULT_MIN_EDGE, edge_clears
    from pmlab.execution.paper_broker import PaperBroker
    from pmlab.modeling.champion import ChampionManifest

    settings = PmlabSettings()
    settings.artifacts_dir = Path(artifacts_dir)

    if not settings.champion_json_path.exists():
        rprint("[red]No champion found. Run 'pmlab promote-champion' first.[/red]")
        raise typer.Exit(1)

    manifest = ChampionManifest.load(settings.champion_json_path)
    allowed = manifest.get_allowed_segments()
    rprint(
        f"Champion: [green]{manifest.model_name}[/green], allowed segments: "
        f"{_describe_segments(allowed)}"
    )

    # Fill under the cost model the champion was promoted under. A flag may name one
    # when the champion records none, but not a different one: trades filled under
    # other costs are left out of the watchdog's PnL comparison, so recording them
    # would only waste the evidence.
    costs = _load_costs(costs_path)
    if costs is None:
        costs = manifest.costs
    elif manifest.costs is not None and costs != manifest.costs:
        rprint(
            "[red]--costs differs from the costs recorded in champion.json; paper trades "
            "must be filled under the champion's cost model.[/red]"
        )
        rprint(f"  champion.json: {manifest.costs.to_dict()}")
        rprint(f"  --costs:       {costs.to_dict()}")
        raise typer.Exit(1)

    # Trade the threshold the champion was backtested with. A different one is allowed (it
    # is the operator's call) but called out: the backtest says nothing about that strategy.
    recorded_policy = manifest.expectation.policy if manifest.expectation is not None else None
    recorded_edge = recorded_policy.min_edge if recorded_policy is not None else None
    if min_edge is None:
        min_edge = recorded_edge if recorded_edge is not None else DEFAULT_MIN_EDGE
    elif recorded_edge is not None and abs(min_edge - recorded_edge) > 1e-9:
        rprint(
            f"[yellow]--min-edge {min_edge:g} differs from the champion's policy "
            f"({recorded_edge:g}): the backtest it was promoted on did not simulate this "
            "threshold.[/yellow]"
        )

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

    signals = [
        EdgeSignal(**s) for s in all_signals if edge_clears(float(s.get("best_edge", 0)), min_edge)
    ]
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
        costs=costs,
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
    costs_path: str | None = typer.Option(
        None,
        "--costs",
        help=_COSTS_HELP + "Default: 30bps taker fee, fill at the quote.",
    ),
    policy_name: str = typer.Option(
        "live",
        "--policy",
        help=(
            "Which trades the backtest takes. 'live' (default) is what record-trades trades: "
            "per market the best-edge bin and side, YES or NO, at or above --min-edge. "
            "'top-bin-yes' is the historical backtest: YES on the most probable bin of every "
            "market, no threshold."
        ),
    ),
    min_edge: float | None = typer.Option(
        None,
        "--min-edge",
        help=(
            "Smallest after-fee edge worth a trade under --policy live (default 0.20, the "
            "same default as record-trades). Recorded in champion.json, which record-trades "
            "then uses as its own default."
        ),
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Save metrics JSON to path"),
) -> None:
    """Run walk-forward backtest on a historical panel."""
    if stride < 10:
        rprint(f"[red]stride={stride} is too low — minimum 10.[/red]")
        raise typer.Exit(1)
    costs = _load_costs(costs_path)

    from pmlab.core.edge import DEFAULT_MIN_EDGE
    from pmlab.core.policy import TradePolicy

    if policy_name == "live":
        try:
            trade_policy = TradePolicy.live(
                min_edge=DEFAULT_MIN_EDGE if min_edge is None else min_edge
            )
        except ValueError as e:
            rprint(f"[red]Not a valid --min-edge: {e}[/red]")
            raise typer.Exit(1) from e
    elif policy_name == "top-bin-yes":
        if min_edge is not None:
            rprint(
                "[red]--min-edge has no meaning under --policy top-bin-yes, which trades every "
                "market with no threshold; use --policy live to set one.[/red]"
            )
            raise typer.Exit(1)
        trade_policy = TradePolicy.top_bin_yes()
    else:
        rprint(f"[red]Unknown --policy {policy_name!r}: choose 'live' or 'top-bin-yes'.[/red]")
        raise typer.Exit(1)

    import pandas as pd

    from pmlab.backtest.rolling_origin import rolling_origin_eval
    from pmlab.core.costs import CostModel
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

    rprint(f"Policy: {_describe_policy(trade_policy)}")

    forecaster = LGBMForecaster()
    with console.status("Running walk-forward backtest..."):
        result = rolling_origin_eval(
            panel,
            forecaster,
            min_train_rows=min_train_rows,
            stride=stride,
            flat_stake=flat_stake,
            costs=costs,
            policy=trade_policy,
        )

    trades = result.trades
    if trades.empty:
        hint = " or lowering --min-edge" if trade_policy.selection == "best_edge" else ""
        rprint(f"[yellow]No trades produced. Try reducing stride or min_train_rows{hint}.[/yellow]")
        raise typer.Exit(0)

    total_pnl = trades["realized_pnl"].sum()
    hit_rate = (trades["realized_pnl"] > 0).mean()
    avg_edge = trades["edge"].mean()
    pnl_color = "green" if total_pnl >= 0 else "red"

    table = Table(title="Backtest Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total trades", str(len(trades)))
    sides = trades["direction"].value_counts()
    table.add_row("YES / NO trades", f"{int(sides.get('yes', 0))} / {int(sides.get('no', 0))}")
    table.add_row("Steps evaluated", str(len(result.steps)))
    table.add_row("Total PnL", f"[{pnl_color}]${total_pnl:+.2f}[/{pnl_color}]")
    table.add_row("Hit rate", f"{hit_rate:.1%}")
    table.add_row("Avg edge", f"{avg_edge:.4f}")
    console.print(table)

    if output:
        metrics: dict[str, object] = {
            "total_trades": len(trades),
            "steps": len(result.steps),
            "total_pnl": float(total_pnl),
            "hit_rate": float(hit_rate),
            "avg_edge": float(avg_edge),
            # The friction these numbers are net of, so `promote-champion` can record
            # the same model rather than trust the operator to retype it.
            "costs": (costs or CostModel()).to_dict(),
            # The decision rule these numbers were produced under.
            "policy": trade_policy.to_dict(),
        }
        if len(trades) >= 2:
            # What `promote-champion --expectation` records in champion.json, so the
            # watchdog can hold live PnL and calibration against this backtest.
            from pmlab.backtest.expectation import BacktestExpectation

            metrics["expectation"] = BacktestExpectation.from_trades(
                trades, flat_stake=flat_stake, policy=trade_policy
            ).to_dict()
        Path(output).write_text(json.dumps(metrics, indent=2))
        rprint(f"[dim]Metrics saved to {output}[/dim]")


@app.command("promote-champion")
def promote_champion_cmd(
    model_path: str = typer.Argument(..., help="Path to trained model .pkl"),
    gate_path: str = typer.Option(..., "--gate-path", help="Path to holdout gate JSON"),
    plugin: str = typer.Option(..., "--plugin", "-p"),
    output_dir: str = typer.Option("artifacts/public_models", "--output-dir"),
    model_name: str = typer.Option("champion", "--name"),
    expectation_path: str | None = typer.Option(
        None,
        "--expectation",
        help=(
            "JSON with the backtest expectation to record in champion.json: the "
            "'expectation' key written by `pmlab backtest --output`, or a bare "
            "expectation object. Without it `pmlab monitor` cannot check live PnL "
            "or calibration against the backtest."
        ),
    ),
    costs_path: str | None = typer.Option(
        None,
        "--costs",
        help=(
            _COSTS_HELP + "Recorded in champion.json as the friction the gate was computed "
            "under. Default: the costs the --expectation file was computed under, if it has "
            "them. Must agree with them if both exist."
        ),
    ),
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

    costs = _load_costs(costs_path)
    expectation = None
    if expectation_path is not None:
        from pmlab.backtest.expectation import BacktestExpectation

        expectation_file = Path(expectation_path)
        if not expectation_file.exists():
            rprint(f"[red]Expectation file not found: {expectation_path}[/red]")
            raise typer.Exit(1)
        try:
            raw = json.loads(expectation_file.read_text())
            expectation = BacktestExpectation.from_dict(raw.get("expectation", raw))
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            rprint(f"[red]Not a valid backtest expectation ({expectation_path}): {e}[/red]")
            raise typer.Exit(1) from e
        # `backtest --output` records the costs its numbers are net of. The champion
        # must be published under those same costs, not whatever is remembered later.
        if isinstance(raw, dict) and raw.get("costs") is not None:
            backtest_costs = _costs_from(raw["costs"], expectation_path)
            if costs is None:
                costs = backtest_costs
            elif costs != backtest_costs:
                rprint(
                    "[red]--costs differs from the costs the expectation was computed under "
                    f"({expectation_path}); the champion would be promoted on a backtest it "
                    "did not run.[/red]"
                )
                rprint(f"  expectation: {backtest_costs.to_dict()}")
                rprint(f"  --costs:     {costs.to_dict()}")
                raise typer.Exit(1)

    model = LGBMForecaster.load(model_file)
    try:
        manifest = ChampionManifest.publish(
            model=model,
            gate=gate,
            output_dir=Path(output_dir),
            plugin_family=plugin,
            model_name=model_name,
            costs=costs,
            expectation=expectation,
        )
        rprint("[green]Champion published![/green]")
        rprint(f"  Model: {manifest.model_path}")
        rprint(f"  Manifest: {output_dir}/champion.json")
        rprint(f"  Allowed segments: {_describe_segments(manifest.get_allowed_segments())}")
        rprint(f"  Costs: {costs.to_dict() if costs is not None else 'not recorded'}")
        policy = expectation.policy if expectation is not None else None
        rprint(f"  Policy: {_describe_policy(policy) if policy is not None else 'not recorded'}")
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command("monitor")
def monitor_cmd(
    champion: str | None = typer.Option(
        None,
        "--champion",
        help="Path to champion.json (default: <artifacts-dir>/public_models/champion.json)",
    ),
    trades: str | None = typer.Option(
        None,
        "--trades",
        "-t",
        help="Path to forward_paper_trades.json (default: from --artifacts-dir and --workspace)",
    ),
    artifacts_dir: str = typer.Option(
        "artifacts", "--artifacts-dir", "-a", envvar="PMLAB_ARTIFACTS_DIR", help="Artifacts root"
    ),
    workspace: str = typer.Option(
        "ops_daily", "--workspace", "-w", envvar="PMLAB_WORKSPACE", help="Active workspace name"
    ),
    reference: str | None = typer.Option(
        None, "--reference", help="Training panel (.parquet/.csv) the drift check compares against"
    ),
    snapshots: str | None = typer.Option(
        None, "--snapshots", help="FeatureSnapshotStore directory holding the live window"
    ),
    family: str | None = typer.Option(
        None, "--family", help="Snapshot family (default: the champion's plugin family)"
    ),
    window_days: int = typer.Option(
        7, "--window-days", help="Live drift window, ending at --as-of"
    ),
    window_start: str | None = typer.Option(
        None,
        "--window-start",
        help="ISO start of the live drift window, with a UTC offset (overrides --window-days)",
    ),
    feature: Annotated[
        list[str] | None,
        typer.Option(
            "--feature",
            help="Drift feature to assess, repeatable (default: the model's own features)",
        ),
    ] = None,
    no_prediction: bool = typer.Option(
        False, "--no-prediction", help="Skip prediction drift (does not load the champion model)"
    ),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="ISO-8601 cutoff with a UTC offset, e.g. 2026-03-01T00:00:00Z (default: now, UTC)",
    ),
    config: str | None = typer.Option(
        None, "--config", help="TOML/JSON file overriding drift, calibration and PnL thresholds"
    ),
    min_trades: int | None = typer.Option(
        None, "--min-trades", help="Settled trades needed before calibration and PnL are judged"
    ),
    insufficient: str | None = typer.Option(
        None,
        "--insufficient",
        help="'separate' (exit 30, default) or 'warn' (exit 10) for a check that cannot be measured",
    ),
    skip: Annotated[
        list[str] | None,
        typer.Option("--skip", help="Check to leave out: drift, calibration or pnl (repeatable)"),
    ] = None,
    json_path: str | None = typer.Option(
        None, "--json", help="Always write the health report as JSON to PATH ('-' for stdout)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print the full report even when healthy"
    ),
) -> None:
    """Watchdog: drift, live calibration and PnL against the champion's backtest.

    Silent when healthy, so it can run from cron: `pmlab monitor ... || notify`.
    A problem prints a short message and sets the exit code:

    0 healthy | 10 warn | 20 alert | 30 insufficient data or a check unavailable |
    64 bad input (missing/corrupt files, bad options) | 70 internal error.
    """
    from pmlab.cli.monitor import run_monitor
    from pmlab.config import PmlabSettings

    settings = PmlabSettings()
    settings.artifacts_dir = Path(artifacts_dir)
    settings.workspace = workspace

    code = run_monitor(
        champion_path=Path(champion) if champion else settings.champion_json_path,
        trades_path=Path(trades) if trades else settings.trades_path,
        reference=Path(reference) if reference else None,
        snapshots=Path(snapshots) if snapshots else None,
        family=family,
        window_days=window_days,
        window_start=window_start,
        features=feature or None,
        use_prediction=not no_prediction,
        as_of=as_of,
        config_path=Path(config) if config else None,
        min_trades=min_trades,
        insufficient=insufficient,
        skip=skip or [],
        json_path=json_path,
        verbose=verbose,
    )
    raise typer.Exit(code)


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
