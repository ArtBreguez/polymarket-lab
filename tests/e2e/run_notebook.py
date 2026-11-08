"""Execute a notebook top to bottom, in a fresh kernel, and fail on the first erroring cell.

Used by the `notebook` CI job (tests/notebooks) and by the e2e test, which runs it with the python
of a fresh venv that has only the installed wheel (plus nbclient/ipykernel) from outside the repo.
Stdlib + nbformat + nbclient only; it never imports pmlab itself.

    python run_notebook.py NOTEBOOK --cwd DIR [--timeout SECONDS] [--prepend FILE] [--out FILE]

``--prepend FILE`` runs the code in FILE as a first cell (the tests use it to break the bundled
dataset on purpose). ``--out`` saves the executed copy; the input is never modified.
Exit status: 0 ran clean, 1 a cell raised, 2 usage or the notebook could not be read.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--cwd", type=Path, required=True, help="kernel working directory")
    ap.add_argument("--timeout", type=int, default=300, help="seconds allowed per cell")
    ap.add_argument("--prepend", type=Path, help="python source to run as a first cell")
    ap.add_argument("--out", type=Path, help="write the executed notebook here")
    args = ap.parse_args(argv)

    try:
        nb = nbformat.read(args.notebook, as_version=4)
    except (OSError, nbformat.reader.NotJSONError) as exc:
        print(f"cannot read {args.notebook}: {exc}", file=sys.stderr)
        return 2
    if args.prepend:
        nb.cells.insert(0, nbformat.v4.new_code_cell(args.prepend.read_text()))
    args.cwd.mkdir(parents=True, exist_ok=True)

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(args.cwd)}},
    )
    started = time.perf_counter()
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError) as exc:
        print(f"NOTEBOOK FAILED after {time.perf_counter() - started:.1f}s\n{exc}")
        return 1
    finally:
        if args.out:
            nbformat.write(nb, args.out)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"NOTEBOOK OK: {n_code} code cells in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
