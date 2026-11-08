"""Load ``examples/perf_budget.py`` (the single source of the workloads and the budgets).

The script lives in ``examples/`` so a user with a checkout can run it as
``python examples/perf_budget.py``; the tests import the very same file, so the numbers the
docs quote, the numbers CI asserts and the numbers a user measures cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "examples" / "perf_budget.py"


def load() -> ModuleType:
    if "perf_budget" in sys.modules:
        return sys.modules["perf_budget"]
    spec = importlib.util.spec_from_file_location("perf_budget", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["perf_budget"] = module
    spec.loader.exec_module(module)
    return module
