# Command line

`pip install pmlab` puts a `pmlab` command on your path. `pmlab --help` lists the commands and
`pmlab <command> --help` the options of one. The commands, their options and their environment
variables are part of the [frozen public API](api-stability.md#what-is-public); the text they print
is for people and may change in any release.

## The commands

| Command | What it does | Needs |
|---|---|---|
| `version` | Prints the installed version. | |
| `backtest` | Walk-forward backtest of a panel (`.parquet`). By default it simulates what `record-trades` trades: per market the best-edge bin and side, YES or NO, taken when the after-fee edge clears `--min-edge` (default 0.20). `--policy top-bin-yes` is the historical YES-only rule. Records seeds, library versions, resolved config and a data fingerprint with the run. | a panel |
| `reproduce` | Re-executes a recorded run and verifies the metrics come back. It refuses a changed dataset and warns on a changed library version. See [Reproducibility](reproducibility.md). | a recorded run |
| `promote-champion` | Publishes a model as champion **only if the gate is GO**; a NO_GO gate is refused. Writes `champion.json`. See [the registry](registry.md). | a model and a gate file |
| `scan-markets` | Lists the open markets of a plugin's family from the Gamma API. | network |
| `record-trades` | Paper-trades the signals in the workspace's `signals/v2/scan_edge_signals.json` whose after-fee edge clears `--min-edge`, restricted to the champion's segments (read from `champion.json`) and to the sides its backtest traded. Fills use the champion's recorded costs. | signals written by your plugin's scan step |
| `status` | Champion, open trades and cumulative PnL of a workspace. | trades recorded |
| `resettle` | Recomputes the PnL of settled trades that an older release booked wrongly (NO trades). `--dry-run` shows what it would change. | a trades file |
| `report` | Self-contained HTML report from a paper-trades file. | a trades file |
| `monitor` | The watchdog: drift, live calibration and live PnL against the champion's own backtest. **Silent and exit 0 when healthy**; otherwise it says what is wrong and exits 10 (warn), 20 (alert) or 30 (could not check); 64 is bad input and 70 an internal error. Built for cron. See [Watchdog](watchdog.md). | a champion and trades |
| `settle-trades` | **A placeholder that refuses.** It settles nothing and **exits 1**, saying on stderr what it cannot do and printing the `SettlementEngine` call that does settle, with this workspace's trade log in it. Settlement needs a plugin wired to real data, which a bare command cannot construct. Before 1.1.0 it exited 0, so a cron line `record-trades && settle-trades` looked healthy while every trade stayed open. | |

Anything that has to know about a market family (`--plugin weather_tmax`) builds that family's plugin
by name. `weather_tmax` ships ready to use and `sports_f1` is a skeleton; your own family is found
through the `pmlab.plugins` entry-point group. See [Plugin authoring](plugin-authoring.md#registering-your-plugin).

## A first session, offline

The [README quickstart](https://github.com/ArtBreguez/polymarket-lab#quickstart) builds a panel from the
bundled sample and writes it to `panel.parquet`. Everything below runs on it, with no network.

```bash
pmlab version
pmlab backtest --plugin weather_tmax --panel panel.parquet \
    --stride 10 --min-train-rows 2700 --min-edge 0.05 --output metrics.json
pmlab reproduce metrics.json
```

`--min-train-rows` counts panel **rows**. A panel has one row per market bin, nine bins per market in
the sample, so 2700 rows is 300 markets of history before the first prediction.

`--embargo-days N` holds the walk-forward back further: a fold then trains only on rows decided at least
`N` days before it, in addition to (never instead of) the rule that a row's label must already be known.
Without it the backtest is what 1.0.3 ran. See [the leakage page](leakage.md#holding-back-more-embargo_days).

The output file carries the record of the run (seeds, versions, config, a fingerprint of the data):
`pmlab reproduce` re-executes it and prints one line per metric, ending in `REPRODUCED` when they
match.

## Workspaces

`ops_daily` is live forward evidence and `historical_real` is training data; mixing them
contaminates a model, so every command that reads or writes trades takes a workspace
(`--workspace`, default `ops_daily`) and keeps its files under
`<artifacts-dir>/workspaces/<workspace>/`. The same choices are available as the environment
variables `PMLAB_WORKSPACE` (or `PMLAB_WORKSPACE_NAME`) and `PMLAB_ARTIFACTS_DIR`. A workspace name is a single
path component: an empty name, `.`, `..` or one with a `/` is refused (exit 2), so a name can never reach another
workspace's files.

In a checkout of the repository, `scripts/pmlab-workspace <name> <command>` sets all of the
workspace variables (data, parquet, DuckDB and artifacts directories) for one command:

```shell
scripts/pmlab-workspace ops_daily pmlab status
scripts/pmlab-workspace historical_real pmlab backtest --plugin weather_tmax --panel panel.parquet
```

The script is not part of the installed package; with a pip install, pass `--workspace` and
`--artifacts-dir` or export the variables yourself.

## Costs

`backtest`, `promote-champion` and `record-trades` accept `--costs costs.json`: one cost model end to
end, so the backtest, the champion and the paper fills charge the same friction. See
[Backtest costs](costs.md).
