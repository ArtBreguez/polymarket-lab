"""Reports, workspace names and the tracker, used from the installed wheel (audit 1.1.2).

Standalone (public names only):  python audit_report_names.py
Prints ``audit report names: OK``.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from datetime import date

import pandas as pd

from pmlab import FeatureSnapshotStore, LocalJSONTracker, WorkspaceContext, generate_report

work = pathlib.Path(tempfile.mkdtemp(prefix="pmlab audit "))
os.chdir(work)

# 1. The documented example writes into a folder that does not exist yet, and nothing is markup.
trade = {
    "recorded_at": "2026-05-01T10:00:00+00:00", "city_or_segment": "<img src=x onerror=alert(1)>",
    "target_date": "2026-05-02", "outcome_label": "<b>x</b>", "direction": "yes", "gamma_price": 0.35,
    "edge_after_fee": 0.08, "flat_stake": 5.0, "realized_pnl": 9.28,
}  # fmt: skip
out = generate_report(
    [trade], output_path="reports/may_2026.html", title="<script>alert(1)</script>"
)
html = out.read_text()
assert out == pathlib.Path("reports/may_2026.html") and "<script>alert(1)</script>" not in html
assert "<img src=x" not in html and "&lt;img src=x onerror=alert(1)&gt;" in html

# 2. The CLI too: a folder to create, and a workspace that cannot leave its place.
(work / "trades.json").write_text(json.dumps({"trades": [trade]}))
exe = pathlib.Path(sys.executable).parent / "pmlab"
ok = subprocess.run(
    [exe, "report", "--trades", "trades.json", "--output", "out dir/rep.html"],
    capture_output=True,
    text=True,
)
assert ok.returncode == 0 and (work / "out dir" / "rep.html").exists(), ok.stderr
for bad in ("..", "../x", "a/b"):
    r = subprocess.run(
        [exe, "status", "--workspace", bad],
        capture_output=True,
        text=True,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert r.returncode == 2 and "Traceback" not in r.stdout + r.stderr, (
        bad,
        r.returncode,
        r.stdout,
        r.stderr,
    )
for bad in ("..", "", "/abs"):
    try:
        WorkspaceContext.from_name(bad, work)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
try:
    FeatureSnapshotStore(work / "store").append(
        "../../esc", [{"market_id": "m", "captured_at": "2025-01-09T00:00:00Z"}]
    )
except ValueError:
    pass
else:
    raise AssertionError("family traversal")
assert not list(work.glob("esc*")) and not (work.parent / "esc.parquet").exists()

# 3. The tracker takes what an experiment logs, and a dead run is not "running".
tracker = LocalJSONTracker("runs")
with tracker.run(experiment="e", run_name="r") as run:
    run.log_params(
        {
            "panel": pathlib.Path("/data/panel.parquet"),
            "start": date(2026, 1, 1),
            "ts": pd.Timestamp("2026-01-01", tz="UTC"),
        }
    )
rec = tracker.list_runs("e")[0]
assert (
    rec.status == "finished"
    and rec.params["panel"] == "/data/panel.parquet"
    and rec.params["start"] == "2026-01-01"
)

print("audit report names: OK")
