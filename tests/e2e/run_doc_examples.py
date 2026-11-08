"""Run the fenced ``python`` blocks of a markdown file, unmodified, against an installed pmlab.

Standalone (no pytest, no repo imports), so it can run with the python of a fresh venv:

    python run_doc_examples.py docs/features.md
    python run_doc_examples.py README.md --contains add_missing_indicators

Each block runs in its own namespace, seeded with synthetic frames the prose assumes (``df``,
``train``, ``live``) and preceded by the file's first block (the imports the page sets up at the
top). ``pandas.read_parquet`` returns the synthetic frame so the "load your panel" lines run.
The code of the blocks is never edited: a block that does not run is a documentation bug.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def synthetic_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """A panel with every column the docs mention, NaNs, zeros and interleaved groups."""
    rng = np.random.default_rng(seed)
    city = np.array(["NYC", "LA", "SF"])[np.arange(n) % 3]  # interleaved, NOT sorted by city
    df = pd.DataFrame(
        {
            "city": city,
            "season": np.array(["winter", "spring", "summer", "fall"])[(np.arange(n) // 7) % 4],
            "date": np.arange(n) // 3,  # three rows share each day
            "temp": rng.normal(15, 8, n),
            "humidity": np.where(rng.random(n) < 0.15, np.nan, rng.uniform(30, 90, n)),
            "wind_speed": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(0, 20, n)),
            "pressure": np.where(rng.random(n) < 0.1, np.nan, rng.normal(1013, 8, n)),
            "hour": np.arange(n) % 24,
            "day_of_year": 1 + (np.arange(n) % 365),
            "month": 1 + (np.arange(n) // 20) % 12,
            "price": rng.uniform(0.05, 0.95, n),
            "liquidity": np.where(rng.random(n) < 0.1, 0.0, rng.uniform(100, 5000, n)),
            "wind": rng.uniform(0, 15, n),
        }
    )
    df["outcome"] = (rng.random(n) < 0.3 + 0.2 * (df["city"] == "NYC")).astype(int)
    return df


def namespace() -> dict[str, object]:
    df = synthetic_frame()
    return {
        "df": df.copy(),
        "train": df.copy(),
        "live": pd.DataFrame({"city": ["NYC", "Atlantis", None]}),
        "__name__": "__docs__",
    }


def blocks(path: Path) -> list[str]:
    return [m.group(1) for m in FENCE.finditer(path.read_text())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--contains", default=None, help="only run blocks containing this text")
    args = ap.parse_args()

    pd.read_parquet = lambda *a, **k: synthetic_frame()  # type: ignore[assignment]
    all_blocks = blocks(args.file)
    if not all_blocks:
        print(f"no python blocks found in {args.file}")
        return 2
    preamble = all_blocks[0]
    todo = [(i, b) for i, b in enumerate(all_blocks) if args.contains is None or args.contains in b]
    if not todo:
        print(f"no block in {args.file} contains {args.contains!r}")
        return 2
    failed = 0
    for i, code in todo:
        label = next((ln for ln in code.splitlines() if ln.strip()), "")[:70]
        ns = namespace()
        try:
            exec(compile(preamble, f"{args.file}#0", "exec"), ns)
            exec(compile(code, f"{args.file}#{i}", "exec"), ns)
        except Exception:
            failed += 1
            print(f"FAIL block {i}: {label}")
            traceback.print_exc()
        else:
            print(f"ok   block {i}: {label}")
    print(f"{len(todo) - failed}/{len(todo)} blocks ran from {args.file}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
