"""End-to-end, as a user: build the wheel, install it in a FRESH venv, run the public API there.

Opt-in (``pytest -m e2e``; deselected by the default ``addopts``) because it needs ``uv`` and the
network and takes a couple of minutes. Nothing here imports ``pmlab`` from the checkout: every
check runs with the python of the fresh venv, from a directory outside the repository, in isolated
mode (``-I``), so only the installed, public API is exercised.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "e2e_transforms.py"
PERF_SCRIPT = REPO / "examples" / "perf_budget.py"
DOC_RUNNER = REPO / "tests" / "e2e" / "run_doc_examples.py"
WALKTHROUGH = REPO / "tests" / "e2e" / "run_doc_walkthrough.py"
NOTEBOOK_RUNNER = REPO / "tests" / "e2e" / "run_notebook.py"
TUTORIAL = REPO / "notebooks" / "tutorial.ipynb"
OK_LINE = "pmlab e2e (as a user): OK"


def _clean_env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"}
    }
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1", "TERM": "dumb", **extra})
    return env


def _run(cmd: list[str | Path], cwd: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=_clean_env(**extra_env),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


class Installed:
    def __init__(self, venv: Path, workdir: Path, wheel: Path) -> None:
        self.venv = venv
        self.wheel = wheel
        self.python = venv / "bin" / "python"
        self.pmlab = venv / "bin" / "pmlab"
        self.workdir = workdir

    def py(self, *args: str | Path) -> subprocess.CompletedProcess[str]:
        return _run([self.python, "-I", *args], cwd=self.workdir)


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> Installed:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build and install the wheel")
    root = tmp_path_factory.mktemp("pmlab-e2e")
    dist, venv, workdir = root / "dist", root / ".venv", root / "work"
    workdir.mkdir()
    _ok(_run([uv, "build", "--wheel", "--out-dir", dist], cwd=REPO))
    wheel = next(dist.glob("pmlab-*.whl"))
    _ok(_run([uv, "venv", "-q", venv, "--python", sys.executable], cwd=workdir))
    _ok(_run([uv, "pip", "install", "-q", "--python", venv / "bin" / "python", wheel], cwd=workdir))
    yield Installed(venv, workdir, wheel)
    shutil.rmtree(venv, ignore_errors=True)  # a venv with lightgbm/scipy is hundreds of MB


def test_the_wheel_is_installed_for_real_not_the_checkout(installed: Installed) -> None:
    out = _ok(installed.py("-c", "import pmlab; print(pmlab.__version__); print(pmlab.__file__)"))
    version, location = out.split("\n")[:2]
    assert "site-packages" in location
    assert REPO not in Path(location).parents
    pyproject = (REPO / "pyproject.toml").read_text()
    assert f'version = "{version}"' in pyproject


def test_user_workflow_script(installed: Installed) -> None:
    panel = installed.workdir / "panel.parquet"
    out = _ok(installed.py(EXAMPLE, "--panel-out", panel))
    assert OK_LINE in out
    assert "backtest (with new features)" in out and panel.exists()


def test_docs_examples_run_unmodified(installed: Installed) -> None:
    out = _ok(installed.py(DOC_RUNNER, REPO / "docs" / "features.md"))
    assert "FAIL" not in out and "blocks ran from" in out
    n_ok = int(out.strip().splitlines()[-1].split("/")[0])
    assert n_ok >= 10
    readme = _ok(
        installed.py(DOC_RUNNER, REPO / "README.md", "--contains", "add_missing_indicators")
    )
    assert "1/1 blocks ran" in readme


def test_cli_backtests_a_panel_built_with_the_new_features(installed: Installed) -> None:
    panel = installed.workdir / "panel_cli.parquet"
    _ok(installed.py(EXAMPLE, "--panel-out", panel))
    assert (
        _ok(_run([installed.pmlab, "version"], cwd=installed.workdir)).strip().startswith("pmlab")
    )
    metrics = installed.workdir / "metrics.json"
    out = _ok(
        _run(
            [
                installed.pmlab,
                "backtest",
                "--plugin",
                "weather_tmax",
                "--panel",
                panel,
                "--stride",
                "10",
                "--min-train-rows",
                "200",
                "--policy",
                "top-bin-yes",
                "--output",
                metrics,
            ],  # fmt: skip
            cwd=installed.workdir,
            PMLAB_ARTIFACTS_DIR=str(installed.workdir / "artifacts"),
        )
    )
    assert "Backtest Results" in out
    data = json.loads(metrics.read_text())
    assert data["total_trades"] > 0
    for key in ("total_pnl", "hit_rate", "avg_edge"):
        assert data[key] == data[key] and abs(data[key]) != float("inf"), key


PERF_SMOKE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("perf_budget", sys.argv[1])
pb = importlib.util.module_from_spec(spec)
sys.modules["perf_budget"] = pb
spec.loader.exec_module(pb)
import pmlab
assert "site-packages" in pmlab.__file__, pmlab.__file__
for w in pb.WORKLOADS:
    out = w.run(w.prepare(1))  # one real run of each hot path, from the installed wheel
    if w.name.startswith("rolling_origin"):
        assert len(out.trades) > 500 and len(out.steps) >= 27, (w.name, len(out.trades))
    else:
        assert len(out) == 120_000 and out["market_price"].between(0, 1).all()
    print("ran", w.name)
print("calibration_ms", round(pb.calibration_seconds(repeats=2) * 1000, 1))
print("perf workloads (as a user): OK")
"""


def test_perf_workloads_and_the_performance_doc_run_from_the_installed_wheel(
    installed: Installed,
) -> None:
    """The public API the benchmarks use works from the wheel, and docs/performance.md runs as is.

    Times are deliberately NOT asserted here (a fresh-venv CI job is no place for a budget):
    the guard is ``pytest -m perf``, a dev-only test that is not shipped in the wheel. What a
    user can run is ``examples/perf_budget.py`` (from a checkout) and the doc's snippet.
    """
    out = _ok(installed.py("-c", PERF_SMOKE, PERF_SCRIPT))
    assert "perf workloads (as a user): OK" in out
    assert out.count("ran ") == 3
    doc = _ok(installed.py(DOC_RUNNER, REPO / "docs" / "performance.md"))
    assert "FAIL" not in doc and "2/2 blocks ran" in doc
    assert "build_panel:" in doc and "rolling_origin_eval:" in doc


def test_reproducibility_walkthrough_runs_as_written_and_reproduces(installed: Installed) -> None:
    """docs/reproducibility.md, block by block, against the wheel: run -> reproduce -> failure modes.

    The blocks (python and bash) share a directory and run in order, unmodified: the panel is
    written, the backtest records its seeds/versions/config/data hash, `pmlab reproduce` brings
    the metrics back, a changed dataset is refused (exit 2), a changed library version is
    warned about, a missing run is a clear error, and a champion carries the record.
    """
    work = installed.workdir / "repro_doc"
    out = _ok(installed.py(WALKTHROUGH, REPO / "docs" / "reproducibility.md", "--cwd", work))
    assert "FAIL" not in out
    done, total = out.strip().splitlines()[-1].split(" blocks ran")[0].split("/")
    assert done == total and int(done) >= 12
    assert out.count("REPRODUCED: 5 metrics match") == 4  # file, run id, older lightgbm, champion
    assert out.count("NOT REPRODUCED") == 1  # --allow-data-mismatch, on edited data
    assert out.count("exit status: 2") == 2  # the changed dataset, the unknown run id
    assert out.count("exit status: 1") == 1 and out.count("exit status: 0") == 1
    assert "Version drift" in out and "3.3.0" in out and "--allow-data-mismatch" in out
    assert "gate: GO" in out

    # What was recorded is what the installed wheel is: version and the library set.
    record = json.loads((work / "metrics.json").read_text())["reproducibility"]
    assert record["environment"]["pmlab"] == _pyproject_version()
    assert {"python", "numpy", "pandas", "lightgbm", "scikit-learn", "optuna"} <= set(
        record["environment"]
    )
    assert record["seeds"] == {"python_random": 123, "numpy": 123, "lightgbm": 123}
    assert len(list((work / "runs" / "demo").glob("*.json"))) >= 1
    champion = json.loads((work / "champion" / "champion.json").read_text())
    assert champion["reproducibility"]["data"] == record["data"]

    # From another directory, in another process: the recorded source path is absolute.
    again = _run([installed.pmlab, "reproduce", work / "metrics.json"], cwd=installed.workdir)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "REPRODUCED" in again.stdout


def test_registry_page_runs_as_written_and_a_rollback_loses_nothing(installed: Installed) -> None:
    """docs/registry.md against the wheel (issue #33): backtest -> promote-champion -> registry
    record -> rollback, with non-default costs, an expectation and a reproducibility record, then
    the archived version re-runs. Then an entry written by 0.10.0 (no expectation) still rolls
    back."""
    work = installed.workdir / "registry_doc"
    out = _ok(installed.py(WALKTHROUGH, REPO / "docs" / "registry.md", "--cwd", work))
    assert "FAIL" not in out and "LOST" not in out and " kept" in out
    done, total = out.strip().splitlines()[-1].split(" blocks ran")[0].split("/")
    assert done == total and int(done) >= 6
    assert "REPRODUCED: 5 metrics match" in out and "NOT REPRODUCED" not in out

    # What is on disk, read from this process: the promoted, the archived and the restored
    # champion.json hold the same keys and values, bar the artifact paths.
    version = (work / "version.txt").read_text()
    files = {
        "promoted": work / "champion" / "champion.json",
        "archived": work / "registry" / version / "champion.json",
        "restored": work / "restored" / "champion.json",
    }
    docs = {
        name: {
            k: v
            for k, v in json.loads(p.read_text()).items()
            if k not in {"model_path", "calibrator_path"}
        }
        for name, p in files.items()
    }
    assert docs["archived"] == docs["promoted"] and docs["restored"] == docs["promoted"]
    restored = docs["restored"]
    assert restored["costs"]["taker_bps"] == 45.0 and restored["costs"]["slippage_bps"] == 20.0
    assert restored["backtest_expectation"]["n_trades"] > 1
    assert restored["reproducibility"]["seeds"]["lightgbm"] == 7
    assert (work / "restored" / "champion.pkl").is_file()

    # An entry archived by 0.10.0 had no backtest_expectation: it loads and rolls back, and the
    # champion comes back without an expectation rather than with a made-up one.
    archived = files["archived"]
    full = json.loads(archived.read_text())
    del full["backtest_expectation"]
    archived.write_text(json.dumps(full))
    legacy = (
        "import sys\n"
        "from pmlab.modeling.registry import ModelRegistry\n"
        "m = ModelRegistry(sys.argv[1]).rollback(sys.argv[2], sys.argv[3])\n"
        "assert m.expectation is None and m.costs is not None and m.reproducibility is not None\n"
        "print('legacy rollback OK')\n"
    )
    proc = installed.py("-c", legacy, work / "registry", version, work / "legacy_restored")
    assert "legacy rollback OK" in _ok(proc)


def test_leakage_check_accepts_a_ranged_panel_and_still_catches_a_leak(
    installed: Installed,
) -> None:
    """docs/leakage.md against the wheel (issue #35), then the same claims probed directly:
    the bundled per-bin sample passes ``build_panel(check_leakage=True)``, a planted copy of the
    outcome is refused on that shape and on a one-row-per-market one, and an explicit label is
    used as given."""
    work = installed.workdir / "leakage_doc"
    out = _ok(installed.py(WALKTHROUGH, REPO / "docs" / "leakage.md", "--cwd", work))
    assert "FAIL" not in out
    done, total = out.strip().splitlines()[-1].split(" blocks ran")[0].split("/")
    assert done == total and int(done) == 4
    assert "2700 rows, 300 markets, 9 bins each" in out
    assert "scored against: outcome_label == winning_label" in out and "leaked: none" in out
    assert "feature_oops (score=1.000)" in out
    assert "won: none" in out and "'feature_fcst_tmax'" in out  # the old target still on request

    probe = (
        "import pmlab\n"
        "from pmlab import LeakageError, build_panel, check_no_leakage\n"
        "from pmlab.datasets import load_sample\n"
        "s = load_sample(); r = s.plugin()\n"
        "rows = [r.build_training_rows(sp, 'previous_evening', decision_time=f'{d}T12:00:00+00:00')\n"
        "        for d in s.days for sp in s.plugin(d).discover_markets()]\n"
        "panel = build_panel(rows, check_leakage=True)  # the call that raised before 0.11.2\n"
        "assert len(panel) == 750 * 9 and 'site-packages' in pmlab.__file__\n"
        "assert check_no_leakage(panel).target == 'outcome_label == winning_label'\n"
        "won = (panel['outcome_label'] == panel['winning_label']).astype(float)\n"
        "def refused(frame, **kw):\n"
        "    try:\n"
        "        check_no_leakage(frame, **kw)\n"
        "    except LeakageError as e:\n"
        "        return str(e)\n"
        "    return None\n"
        "bin_leak = refused(panel.assign(feature_copy=won))\n"
        "assert bin_leak and 'feature_copy' in bin_leak, bin_leak\n"
        "one = panel.drop_duplicates('market_id')  # one row per market: winning_label, as before\n"
        "assert check_no_leakage(one, raise_on_leak=False).target == 'winning_label'\n"
        "codes = {k: float(i) for i, k in enumerate(sorted(one['winning_label'].unique()))}\n"
        "rarest = one['winning_label'].value_counts().index[-1]\n"
        "# a feature that is 1.0 exactly on the rarest winning label, 0.0 elsewhere\n"
        "leak = (one['winning_label'] == rarest).astype(float)\n"
        "assert refused(one.assign(feature_copy=leak)), 'one-row-per-market leak not caught'\n"
        "assert refused(panel.assign(won=won), label_col='won') is None\n"
        "print('leakage probe OK', pmlab.__version__)\n"
    )
    ok = _ok(installed.py("-c", probe))
    assert f"leakage probe OK {_pyproject_version()}" in ok


def test_tuning_page_runs_as_written_and_a_tuned_run_records_its_objective(
    installed: Installed,
) -> None:
    """docs/tuning.md against the wheel (issue #36): the page's crafted panel ranks two models
    differently under the historical and the live rule, ``TunedForecaster(policy=...)`` follows
    the rule it is given, and the tuned backtest's record carries the objective and the sampler
    seed and reproduces with ``pmlab reproduce`` from another directory."""
    uv = shutil.which("uv")
    assert uv is not None
    _ok(  # Optuna is the `tune` extra: the page says to install it, so the test does
        _run(
            [uv, "pip", "install", "-q", "--python", installed.python]
            + [f"pmlab[tune] @ {installed.wheel.as_uri()}"],
            cwd=installed.workdir,
        )
    )
    work = installed.workdir / "tuning_doc"
    out = _ok(installed.py(WALKTHROUGH, REPO / "docs" / "tuning.md", "--cwd", work))
    assert "FAIL" not in out
    done, total = out.strip().splitlines()[-1].split(" blocks ran")[0].split("/")
    assert done == total and int(done) == 4
    # The mismatch, from the backtest itself: the two rules rank A and B in opposite orders.
    assert "historical (top-prob)    A: +32.6   B: -64.0" in out
    assert "live, min_edge=0.20      A: no trades   B: +23.9" in out
    assert "live, min_edge=0.05      A: +25.4   B: -12.7" in out
    # ...and the tuner follows the rule it is given.
    assert "historical : {'col': 'feature_p_a'}" in out
    assert "live  0.20 : {'col': 'feature_p_b'}" in out
    assert "live  0.05 : {'col': 'feature_p_a'}" in out
    assert "REPRODUCED: 5 metrics match" in out

    record = json.loads((work / "metrics.json").read_text())["reproducibility"]
    assert record["environment"]["pmlab"] == _pyproject_version()
    assert record["environment"]["optuna"] is not None
    assert record["seeds"] == {"python_random": 11, "numpy": 11, "lightgbm": 11, "optuna": 3}
    tuning = record["config"]["tuning"]
    assert tuning["policy"] == {
        "selection": "best_edge",
        "directions": ["yes", "no"],
        "min_edge": 0.05,
    }
    assert tuning["costs"]["slippage_bps"] == 50.0 and tuning["metric"] == "total_pnl"
    assert record["config"]["policy"] == tuning["policy"]

    # From another directory, in another process: the recorded panel path is absolute.
    again = _run([installed.pmlab, "reproduce", work / "metrics.json"], cwd=installed.workdir)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "REPRODUCED" in again.stdout and "optuna" in again.stdout

    probe = (
        "import pmlab\n"
        "from pmlab import CostModel, TradePolicy, TunedForecaster\n"
        "assert 'site-packages' in pmlab.__file__\n"
        "build, space = (lambda p: None), (lambda t: {})\n"
        "t = TunedForecaster(build, space)  # nothing passed: the historical objective\n"
        "assert t.policy is None and t.taker_bps == 30.0 and t.costs == CostModel()\n"
        "assert t.tuning_config()['policy'] == TradePolicy.top_bin_yes().to_dict()\n"
        "for bad in ({'policy': 'live'}, {'costs': {'taker_bps': 30.0}}):\n"
        "    try:\n"
        "        TunedForecaster(build, space, **bad)\n"
        "    except TypeError:\n"
        "        continue\n"
        "    raise AssertionError(bad)\n"
        "try:\n"
        "    TunedForecaster(build, space, costs=CostModel(), taker_bps=30.0)\n"
        "except ValueError as e:\n"
        "    assert 'taker_bps or costs' in str(e)\n"
        "else:\n"
        "    raise AssertionError('two sources of the fee')\n"
        "print('tuning probe OK', pmlab.__version__)\n"
    )
    assert f"tuning probe OK {_pyproject_version()}" in _ok(installed.py("-c", probe))


def test_a_record_from_before_0_10_0_loads_and_reproduce_says_there_is_nothing_to_run(
    installed: Installed,
) -> None:
    legacy = installed.workdir / "legacy_champion.json"
    legacy.write_text(json.dumps({"model_name": "champion", "plugin_family": "demo"}))
    proc = _run([installed.pmlab, "reproduce", legacy], cwd=installed.workdir)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "no reproducibility record" in proc.stdout and "0.10.0" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr


def test_bundled_sample_and_tutorial_notebook_run_from_the_installed_wheel(
    installed: Installed,
) -> None:
    """Issue #15, as a user: the sample ships in the wheel, loads, discovers offline, and the
    tutorial notebook executes top to bottom in the fresh venv, from outside the repo."""
    # 1. The data is inside the wheel and the wheel stays lean.
    with zipfile.ZipFile(installed.wheel) as zf:
        sizes = {i.filename: i.file_size for i in zf.infolist()}
    data = [n for n in sizes if n.startswith("pmlab/datasets/data/")]
    assert sorted(data) == [
        "pmlab/datasets/data/weather_tmax_sample.json.gz",
        "pmlab/datasets/data/weather_tmax_sample.sha256",
    ]
    assert sum(sizes[n] for n in data) < 200_000
    assert installed.wheel.stat().st_size < 3_000_000

    # 2. The documented quick start, as written, then the loader from the installed package.
    doc = _ok(
        installed.py(WALKTHROUGH, REPO / "docs" / "tutorial.md", "--cwd", installed.workdir / "doc")
    )
    assert "FAIL" not in doc and "SampleDataset(750 markets, 5 cities" in doc
    assert "Highest temperature in Madrid on 2025-04-02?" in doc
    probe = _ok(
        installed.py(
            "-c",
            "import pmlab.datasets as d; s = d.load_sample(); print(d.__file__); "
            "print(len(s.plugin(s.days[3]).discover_markets()), s.meta['synthetic'])",
        )
    )
    location, counts = probe.strip().splitlines()
    assert "site-packages" in location and REPO not in Path(location).parents
    assert counts == "5 True"

    # 3. The notebook, from the installed wheel (+ the `tune` extra and the notebook executor).
    uv = shutil.which("uv")
    assert uv is not None
    _ok(
        _run(
            [
                uv,
                "pip",
                "install",
                "-q",
                "--python",
                installed.python,
                f"pmlab[tune] @ {installed.wheel.as_uri()}",
                "nbclient",
                "nbformat",
                "ipykernel",
            ],
            cwd=installed.workdir,
        )
    )
    executed = installed.workdir / "tutorial.executed.ipynb"
    out = _ok(
        installed.py(
            NOTEBOOK_RUNNER, TUTORIAL, "--cwd", installed.workdir / "kernel", "--out", executed
        )
    )
    assert "NOTEBOOK OK: 17 code cells" in out

    cells = json.loads(executed.read_text())["cells"]
    shown = "\n".join(
        "".join(o.get("text", "")) if o["output_type"] == "stream" else ""
        for c in cells
        if c["cell_type"] == "code"
        for o in c["outputs"]
    )
    assert f"pmlab {_pyproject_version()}" in shown  # the installed version, not a checkout's
    assert (
        "decision: NO_GO" in shown and "refused: Cannot publish champion with NO_GO gate" in shown
    )
    assert "GO on [" in shown and "Lisbon" not in shown.split("recorded:")[1].split("\n")[0]
    assert "reproduced |" in shown and "'pending': 0" in shown


def _pyproject_version() -> str:
    for line in (REPO / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    raise AssertionError("no version in pyproject.toml")
