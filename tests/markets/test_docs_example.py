"""The code block of ``docs/reference/markets.md`` runs as written and prints what the page shows.

A block on a docs page that stopped running is a documentation bug, and this page's example is the
one place a user sees ``resolve_fee_snapshot`` and ``FeeScheduleStore`` used together. It is offline
and takes about a second; ``pytest -m e2e`` runs the same block again against the built wheel.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "reference" / "markets.md"
WALKTHROUGH = REPO / "tests" / "e2e" / "run_doc_walkthrough.py"
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def test_the_fee_schedule_example_runs_and_prints_what_the_page_shows(tmp_path: Path) -> None:
    kinds = [kind for kind, _ in FENCE.findall(PAGE.read_text())]
    shown = [code for kind, code in FENCE.findall(PAGE.read_text()) if kind == "text"]
    assert kinds.count("python") == 1 and len(shown) == 1, kinds

    proc = subprocess.run(
        [sys.executable, "-I", str(WALKTHROUGH), str(PAGE), "--cwd", str(tmp_path / "run"),
         "--session"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )  # fmt: skip
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    for line in shown[0].splitlines():
        assert line in proc.stdout.splitlines(), (line, proc.stdout)
