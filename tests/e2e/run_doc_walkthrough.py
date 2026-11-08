"""Run a markdown page's ``python`` and ``bash`` blocks in order, exactly as written.

Standalone (no pytest, no repo imports), so it runs with the python of a fresh venv:

    python run_doc_walkthrough.py docs/reproducibility.md --cwd /some/empty/dir

A walkthrough is a story told in commands: one block creates a file, the next uses it. So the
blocks share a working directory and run in document order, each in its own process
(``python -I`` for python blocks, ``bash -euo pipefail`` for bash ones, with the directory of
this python first on ``PATH`` so ``pmlab`` is the installed one). Blocks of any other kind
(``text`` for sample output, ``console``, ...) are documentation and are not run. The code is
never edited: a block that does not run is a documentation bug.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

FENCE = re.compile(r"^```(python|bash)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--cwd", type=Path, required=True, help="empty directory the blocks run in")
    args = ap.parse_args()

    blocks = [(m.group(1), m.group(2)) for m in FENCE.finditer(args.file.read_text())]
    if not blocks:
        print(f"no python/bash blocks found in {args.file}")
        return 2
    args.cwd.mkdir(parents=True, exist_ok=True)
    bindir = str(Path(sys.executable).parent)
    env = {
        **{k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV"}},
        "PATH": bindir + os.pathsep + os.environ.get("PATH", ""),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    ran = 0
    for i, (kind, code) in enumerate(blocks, start=1):
        cmd = (
            [sys.executable, "-I", "-c", code]
            if kind == "python"
            else ["bash", "-euo", "pipefail", "-c", code]
        )
        proc = subprocess.run(
            cmd, cwd=args.cwd, env=env, capture_output=True, text=True, timeout=900, check=False
        )
        print(f"--- block {i} ({kind}) exit {proc.returncode}")
        print(proc.stdout, end="")
        if proc.returncode != 0:
            print(f"FAIL block {i} ({kind}):\n{code}\n{proc.stderr}")
            return 1
        ran += 1
    print(f"{ran}/{len(blocks)} blocks ran from {args.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
