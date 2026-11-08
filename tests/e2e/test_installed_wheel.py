"""End-to-end, as a user: build the wheel, install it in a FRESH venv, run the public API there.

Opt-in (``pytest -m e2e``; deselected by the default ``addopts``) because it needs ``uv`` and the
network and takes a couple of minutes. Nothing here imports ``pmlab`` from the checkout: every
check runs with the python of the fresh venv, from a directory outside the repository, in isolated
mode (``-I``), so only the installed, public API is exercised.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator
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


def test_the_readme_runs_as_written_from_the_installed_wheel(installed: Installed) -> None:
    """The first thing a user copies: the README quickstart, then its command-line block, then the
    command-line page, which reads the panel the quickstart wrote. Nothing is edited."""
    work = installed.workdir / "readme"
    out = _ok(installed.py(WALKTHROUGH, REPO / "README.md", "--cwd", work, "--session"))
    assert "FAIL" not in out and "3/3 blocks ran" in out
    assert re.search(r"\d+ trades \| PnL -?[\d.]+ \| (GO|NO_GO)", out)
    assert "refused: Cannot publish champion with NO_GO gate" in out
    assert "REPRODUCED: 5 metrics match" in out
    assert (work / "panel.parquet").exists() and not (work / "champion").exists()

    cli_page = _ok(installed.py(WALKTHROUGH, REPO / "docs" / "cli.md", "--cwd", work, "--session"))
    assert "FAIL" not in cli_page and "REPRODUCED: 5 metrics match" in cli_page
    assert "pmlab " + _pyproject_version() in cli_page


def test_the_guide_pages_run_as_written_from_the_installed_wheel(installed: Installed) -> None:
    """Modeling, data layer and sizing: the examples that used to live in the README."""
    modeling = _ok(installed.py(DOC_RUNNER, REPO / "docs" / "modeling.md"))
    assert "FAIL" not in modeling and "7/7 blocks ran" in modeling

    data = _ok(
        installed.py(
            WALKTHROUGH, REPO / "docs" / "data.md", "--cwd", installed.workdir / "data", "--session"
        )
    )
    assert "leaked: none" in data and "rows (one per market and bin)" in data

    sizing = _ok(
        installed.py(
            WALKTHROUGH, REPO / "docs" / "sizing.md", "--cwd", installed.workdir / "sz", "--session"
        )
    )
    assert "14.29 shares" in sizing and "0.1154 of the bankroll" in sizing and "25.0 USDC" in sizing


def test_the_package_description_is_the_readme_and_renders_on_pypi(installed: Installed) -> None:
    """PyPI shows the wheel's long description. It must be the README, and `twine check --strict`
    (which renders it with the library PyPI uses) must pass."""
    with zipfile.ZipFile(installed.wheel) as wheel:
        name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = wheel.read(name).decode()
    description = metadata.split("\n\n", 1)[1]
    assert description.strip() == (REPO / "README.md").read_text().strip()
    assert "Classifier: Development Status :: 5 - Production/Stable" in metadata
    assert "Classifier: Typing :: Typed" in metadata

    uv = shutil.which("uv")
    assert uv is not None
    checked = _run(
        [uv, "tool", "run", "twine", "check", "--strict", installed.wheel], cwd=installed.workdir
    )
    assert checked.returncode == 0 and "PASSED" in checked.stdout, checked.stdout + checked.stderr


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
    """docs/leakage.md against the wheel (issues #35 and #39), then the same claims probed
    directly: the bundled per-bin sample passes ``build_panel(check_leakage=True)``, a planted
    copy of the outcome is refused on that shape and on a one-row-per-market one, a market-level
    copy of the realised value is refused too (whichever way the open-ended bins are worded),
    and an explicit label is used as given."""
    work = installed.workdir / "leakage_doc"
    out = _ok(installed.py(WALKTHROUGH, REPO / "docs" / "leakage.md", "--cwd", work))
    assert "FAIL" not in out
    done, total = out.strip().splitlines()[-1].split(" blocks ran")[0].split("/")
    assert done == total and int(done) == 8
    assert "2700 rows, 300 markets, 9 bins each" in out
    # 1.1.0: the page's embargo_days demonstration (rows trained per fold, by embargo)
    assert "embargo_days=None  04-11: 405  04-21: 855  05-01: 1305  05-11: 1755" in out
    assert "embargo_days=2     04-11: 405  04-21: 855  05-01: 1305  05-11: 1755" in out
    assert "embargo_days=7     04-21: 630  05-01: 1080  05-11: 1530" in out
    assert "embargo_days=14    04-21: 315  05-01: 765  05-11: 1215" in out
    # issue #47: the page's own demonstration of which labels are known when a model decides
    assert "810 of 900 rows are labelled by 2025-04-20 12:00 UTC" in out
    assert (
        "latest decision day with a label: 2025-04-18 | this decision's own day: 2025-04-20" in out
    )
    assert "scored against: outcome_label == winning_label" in out and "leaked: none" in out
    assert "feature_oops (score=1.000)" in out
    assert "won: none" in out and "'feature_fcst_tmax'" in out  # the old target still on request
    # the market-level copy (#39): caught, named, and the noise sweep the page prints
    assert "feature_realised_tmax (score=1.000, by its distance to each bin)" in out
    assert "feature_fcst_tmax      0.698  (distance to the bins)" in out
    assert "feature_market_price   0.646  (as is)" in out
    assert "copy + noise of 0.0 degrees: 1.000" in out
    assert (
        "copy + noise of 0.3 degrees: 0.995" in out and "copy + noise of 1.0 degrees: 0.879" in out
    )

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
        "# 0.12.1 (#39): a market-level copy of the realised value, on the whole sample\n"
        "realised = panel['winning_label'].str.extract(r'(-?\\d+)', expand=False).astype(float)\n"
        "msg = refused(panel.assign(feature_realised_tmax=realised))\n"
        "assert msg and 'feature_realised_tmax (score=1.000, by its distance to each bin)' in msg, msg\n"
        "honest = check_no_leakage(panel)\n"
        "assert honest.leaked == [] and len(honest.bin_relative) == 8, honest\n"
        "assert max(honest.scores.values()) < 0.8, honest.scores\n"
        "# the same panel with the tails worded the other way: still read, still refused\n"
        "import re\n"
        "def reword(label):\n"
        "    m = re.fullmatch(r'<=(-?\\d+)°C', label) or None\n"
        "    n = re.fullmatch(r'>=(-?\\d+)°C', label) or None\n"
        "    return f'{m[1]}°C or below' if m else f'{n[1]}°C or above' if n else label\n"
        "worded = panel.assign(outcome_label=panel['outcome_label'].map(reword),\n"
        "                      winning_label=panel['winning_label'].map(reword))\n"
        "assert (worded['outcome_label'] != panel['outcome_label']).any()\n"
        "assert refused(worded.assign(feature_realised_tmax=realised)), 'worded tails not read'\n"
        "assert check_no_leakage(worded).leaked == [], 'worded tails: honest sample flagged'\n"
        "# one row per market and an explicit label: no distance score at all\n"
        "assert check_no_leakage(one, raise_on_leak=False).bin_relative == []\n"
        "assert check_no_leakage(panel.assign(won=won), label_col='won').bin_relative == []\n"
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
    assert "historical (top-prob)    A: +32.5   B: -63.5" in out
    assert "live, min_edge=0.20      A: no trades   B: +24.9" in out
    assert "live, min_edge=0.05      A: +25.2   B: -11.7" in out
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


# ── API stability (v1.0.0, #17): the frozen surface, from the installed wheel ─────────────────────
API_SURFACE = REPO / "scripts" / "api_surface.py"
API_SNAPSHOT = REPO / "tests" / "api" / "public_api.json"


def _snapshot() -> dict:
    return json.loads(API_SNAPSHOT.read_text())


def test_the_installed_wheel_matches_the_public_api_snapshot(installed: Installed) -> None:
    """The same script the default suite runs, with the python of a venv holding only the wheel."""
    snapshot = _snapshot()
    total = sum(len(v) for v in snapshot["modules"].values())
    out = _ok(installed.py(API_SURFACE, "--check", "--snapshot", API_SNAPSHOT))
    assert f"{total} public names imported" in out
    assert f"{len(snapshot['cli']['commands'])} commands" in out


def test_every_name_in_the_snapshot_imports_from_the_installed_wheel(installed: Installed) -> None:
    """Independent of the script: plain ``importlib`` and ``getattr`` over every snapshot entry."""
    code = (
        "import importlib, json, sys\n"
        "snap = json.load(open(sys.argv[1]))\n"
        "n = 0\n"
        "for module, names in snap['modules'].items():\n"
        "    m = importlib.import_module(module)\n"
        "    for name in names:\n"
        "        getattr(m, name)\n"
        "        n += 1\n"
        "    assert sorted(m.__all__) == sorted(names), module\n"
        "print('imported', n)\n"
    )
    out = _ok(installed.py("-c", code, API_SNAPSHOT))
    assert out.strip() == f"imported {sum(len(v) for v in _snapshot()['modules'].values())}"


def test_every_command_and_option_in_the_snapshot_is_in_the_installed_cli(
    installed: Installed,
) -> None:
    for command, spec in _snapshot()["cli"]["commands"].items():
        out = _ok(_run([installed.pmlab, command, "--help"], cwd=installed.workdir, COLUMNS="200"))
        for param in spec["params"]:
            for flag in param["flags"]:
                assert flag in out, f"`pmlab {command} --help` lost {flag}"


def test_a_deprecated_path_warns_the_caller_from_the_installed_wheel(installed: Installed) -> None:
    script = installed.workdir / "uses_deprecated.py"
    script.write_text(
        "import warnings\n"
        "import pmlab\n"
        "from pmlab._deprecation import deprecated\n"
        "\n"
        "plugin = pmlab.WeatherTmaxPlugin()\n"
        "try:\n"
        "    plugin.build_training_row(None, 'd1')  # line 7: the deprecated call\n"
        "except AttributeError:\n"
        "    pass  # past the warning, the body ran (no market was given)\n"
        "\n"
        "@deprecated(since='1.2.0', removed_in='2.0.0', alternative='new_thing')\n"
        "def old():\n"
        "    return 1\n"
        "\n"
        "assert old() == 1  # line 15\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('error', pmlab.PmlabDeprecationWarning)\n"
        "    try:\n"
        "        old()\n"
        "    except pmlab.PmlabDeprecationWarning as e:\n"
        "        print('as an error:', e)\n"
    )
    proc = installed.py(script)
    assert proc.returncode == 0, proc.stderr
    # Default filters, no -W: a script sees the warning, on the line of its own call.
    assert "uses_deprecated.py:7: PmlabDeprecationWarning" in proc.stderr
    assert "WeatherTmaxPlugin.build_training_row is deprecated since 0.9.2" in proc.stderr
    assert "removed in 2.0.0; use build_training_rows" in proc.stderr
    assert "uses_deprecated.py:15: PmlabDeprecationWarning" in proc.stderr
    assert "use new_thing instead" in proc.stderr
    assert "as an error: old is deprecated since 1.2.0" in proc.stdout


def test_the_tutorial_and_readme_imports_resolve_in_the_installed_wheel(
    installed: Installed,
) -> None:
    """The documented import lines, the first thing a new user copies, run against the wheel."""
    imports = _ok(
        installed.py(
            "-c",
            "import pmlab; from pmlab import ModelRegistry, RegistryEntry, PmlabDeprecationWarning;"
            "from pmlab.tracking import MLflowTracker; import sys;"
            "print('mlflow' in sys.modules)",
        )
    )
    assert imports.strip() == "False"  # the class is importable; MLflow itself is not pulled in


# ── the docs site, built from the sdist in a clean venv with only the `docs` extra ─────────────
DocsBuild = tuple[dict[str, Path], subprocess.CompletedProcess[str]]


@pytest.fixture(scope="module")
def docs_build(
    tmp_path_factory: pytest.TempPathFactory, installed: Installed
) -> Iterator[DocsBuild]:
    import tarfile

    uv = shutil.which("uv")
    assert uv is not None
    root = tmp_path_factory.mktemp("pmlab-docs-e2e")
    dist, venv, work = root / "dist", root / ".docs-venv", root / "work"
    work.mkdir()
    _ok(_run([uv, "build", "--sdist", "--out-dir", dist], cwd=REPO))
    sdist = next(dist.glob("pmlab-*.tar.gz"))
    unpacked = root / "sdist"
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    (project,) = list(unpacked.iterdir())
    _ok(_run([uv, "venv", "-q", venv, "--python", sys.executable], cwd=work))
    # The installed wheel plus ONLY the docs extra: what a contributor gets from `pmlab[docs]`.
    spec = f"pmlab[docs] @ {installed.wheel.as_uri()}"
    _ok(_run([uv, "pip", "install", "-q", "--python", venv / "bin" / "python", spec], cwd=work))
    site = root / "site"
    build = _run(
        [venv / "bin" / "mkdocs", "build", "--strict", "-f", project / "mkdocs.yml", "-d", site],
        cwd=work,  # outside the repository and outside the sdist
        NO_MKDOCS_2_WARNING="1",
    )
    yield {"project": project, "site": site, "venv": venv, "work": work, "root": root}, build
    shutil.rmtree(venv, ignore_errors=True)


def test_the_sdist_carries_everything_the_docs_build_needs(docs_build: DocsBuild) -> None:
    paths, _ = docs_build
    project = paths["project"]
    for needed in (
        "mkdocs.yml",
        "docs/index.md",
        "docs/api-stability.md",
        "docs/deprecation.md",
        "docs/reference/index.md",
        "scripts/griffe_deprecated.py",
        "scripts/mkdocs_hooks.py",
        "scripts/make_site_data.py",
        "pyproject.toml",
        "overrides/main.html",
        "overrides/home.html",
        "overrides/404.html",
        "overrides/partials/source.html",
        "overrides/partials/logo.html",
        "docs/assets/stylesheets/pmlab.css",
        "docs/assets/javascripts/pmlab.js",
        "docs/assets/data/sample-run.json",
        "docs/assets/fonts/manrope-latin-wght-normal.woff2",
        "src/pmlab/__init__.py",
    ):
        assert (project / needed).exists(), f"the sdist is missing {needed}"


def test_mkdocs_build_strict_passes_from_the_sdist(docs_build: DocsBuild) -> None:
    paths, build = docs_build
    assert build.returncode == 0, f"mkdocs build --strict failed:\n{build.stdout}\n{build.stderr}"
    site = paths["site"]
    assert (site / "index.html").exists()
    for page in ("api-stability", "deprecation", "reference/core", "reference/modeling"):
        assert (site / page / "index.html").exists(), page


def test_the_built_reference_documents_the_public_api_and_the_deprecation(
    docs_build: DocsBuild,
) -> None:
    paths, _ = docs_build
    site = paths["site"]
    core = (site / "reference" / "core" / "index.html").read_text()
    assert 'id="pmlab.CostModel"' in core and 'id="pmlab.kelly_stake_size"' in core
    plugins = (site / "reference" / "plugins" / "index.html").read_text()
    assert "Deprecated since 0.9.2 and will be removed in 2.0.0" in plugins
    # Only public names: nothing under an underscore or an internal module path.
    every_id = "".join(p.read_text() for p in (site / "reference").rglob("index.html"))
    assert 'id="pmlab._' not in every_id and 'id="pmlab.core.' not in every_id


def test_strict_mode_really_fails_on_a_broken_link(docs_build: DocsBuild) -> None:
    """The negative control: without it, a green build could just mean nothing is checked."""
    paths, _ = docs_build
    project, work = paths["project"], paths["work"]
    page = project / "docs" / "index.md"
    original = page.read_text()
    page.write_text(original + "\n[a link that goes nowhere](no-such-page.md)\n")
    try:
        broken = _run(
            [
                paths["venv"] / "bin" / "mkdocs",
                "build",
                "--strict",
                "-f",
                project / "mkdocs.yml",
                "-d",
                paths["root"] / "site-broken",
            ],
            cwd=work,
            NO_MKDOCS_2_WARNING="1",
        )
    finally:
        page.write_text(original)
    assert broken.returncode != 0
    assert "no-such-page.md" in broken.stderr + broken.stdout


# ── the redesigned site, as a visitor gets it: the built pages, not the templates ──────────────
SITE_URL = "https://pmlab.hiddenterps.si/"
SITE_DATA = REPO / "scripts" / "make_site_data.py"


def _page(docs_build: DocsBuild, name: str) -> str:
    site = docs_build[0]["site"]
    path = site / name if name.endswith(".html") else site / name / "index.html"
    return path.read_text()


def test_the_built_home_page_shows_the_real_numbers_and_the_installed_version(
    docs_build: DocsBuild,
) -> None:
    paths, _ = docs_build
    html = _page(docs_build, "index.html")
    data = json.loads((paths["project"] / "docs/assets/data/sample-run.json").read_text())
    assert f"pmlab {_pyproject_version()}" in html
    assert "pip install pmlab" in html and "uv add pmlab" in html
    assert data["gate"]["publish_refused"] in html  # the refusal is quoted, not retyped
    assert f"<dd>{data['backtest']['trades']}</dd>" in html
    assert html.count("<h1") == 1
    assert f'<link rel="canonical" href="{SITE_URL}"' in html
    text = re.sub(r"<[^>]+>", "", html)  # the code is syntax-highlighted: look at its text
    assert "from pmlab import" in text and "rolling_origin_eval(" in text  # quickstart rendered
    # the ticker is two copies of the signals, the second hidden from assistive technology
    assert html.count('<li aria-hidden="true">') == len(data["signals"])


def test_the_built_site_loads_nothing_from_a_third_party(docs_build: DocsBuild) -> None:
    site = docs_build[0]["site"]
    for page in site.rglob("*.html"):
        text = page.read_text()
        assert "fonts.googleapis.com" not in text and "fonts.gstatic.com" not in text, page
        assert "api.github.com" not in text, page  # Material's live repository facts are off
        for tag in re.findall(r"<(?:script|link)\b[^>]*>", text):
            for url in re.findall(r'(?:src|href)="(https?://[^"]+)"', tag):
                assert url.startswith(SITE_URL), f"{page.relative_to(site)}: {tag[:120]}"
    for sheet in site.rglob("*.css"):
        text = sheet.read_text()
        assert "fonts.googleapis.com" not in text and "@import url(http" not in text, sheet


def test_every_asset_a_built_page_points_at_is_in_the_site(docs_build: DocsBuild) -> None:
    site = docs_build[0]["site"]
    for name in (
        "index.html",
        "404.html",
        "reproducibility/index.html",
        "reference/core/index.html",
    ):
        page = site / name
        base = page.parent
        text = page.read_text()
        refs = re.findall(
            r'(?:src|href)="((?:\.\./|\./|/|assets/)[^"#?]+\.(?:css|js|svg|woff2|json|png))"', text
        )
        assert refs, name
        for ref in refs:
            # the 404 page is served from any depth, so Material roots its links at the site path
            target = site / ref.lstrip("/") if ref.startswith("/") else (base / ref).resolve()
            assert target.exists(), f"{name} -> {ref}"
    fonts = {p.name for p in (site / "assets" / "fonts").glob("*.woff2")}
    assert fonts == {
        "manrope-latin-wght-normal.woff2",
        "barlow-condensed-latin-800-normal.woff2",
        "barlow-condensed-latin-600-normal.woff2",
        "ibm-plex-mono-latin-400-normal.woff2",
        "ibm-plex-mono-latin-500-normal.woff2",
    }


def test_the_404_page_is_the_themed_one(docs_build: DocsBuild) -> None:
    html = _page(docs_build, "404.html")
    assert "No route." in html and "Search the docs" in html
    assert "pm-404" in html and "assets/stylesheets/pmlab.css" in html


def test_the_site_is_served_from_the_new_domain(docs_build: DocsBuild) -> None:
    site = docs_build[0]["site"]
    sitemap = (site / "sitemap.xml").read_text()
    assert f"<loc>{SITE_URL}</loc>" in sitemap and "artbreguez.github.io" not in sitemap
    index = json.loads((site / "search" / "search_index.json").read_text())
    assert len(index["docs"]) > 100, "the search index is nearly empty"
    assert "artbreguez.github.io" not in _page(docs_build, "reproducibility")


def test_the_home_page_numbers_are_what_the_installed_wheel_produces(installed: Installed) -> None:
    """scripts/make_site_data.py --check: recompute the sample run with the wheel, compare the file."""
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
            ],
            cwd=installed.workdir,
        )
    )
    out = _ok(installed.py(SITE_DATA, "--check"))
    assert "matches the library" in out


# ── issue #47: a fold never trains on a label that was not yet known ──────────────────────────────

_NO_LOOKAHEAD_PROBE = '''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pmlab
from pmlab import build_panel, rolling_origin_eval
from pmlab.datasets import load_sample

work = Path(sys.argv[1])
sample = load_sample()
resolver = sample.plugin()
rows = [
    resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
    for day in sample.days
    for spec in sample.plugin(day).discover_markets()
]
panel = build_panel(rows, check_leakage=True).reset_index(drop=True)
panel.to_parquet(work / "panel.parquet")


class Recorder:
    """Predicts 0.5 and remembers which rows each fold trained and tested on."""

    def __init__(self):
        self.folds, self.train = [], None

    def fit(self, X, y):
        self.train = X.index

    def predict_proba(self, X):
        self.folds.append((self.train, X.index))
        return np.full((len(X), 2), 0.5)


rec = Recorder()
result = rolling_origin_eval(panel, rec, min_train_rows=500, stride=5)
assert len(result.steps) >= 20 and len(rec.folds) == len(result.steps)
for train, test in rec.folds:
    decides = pd.to_datetime(panel.loc[test, "decision_ts"], utc=True).min()
    known = pd.to_datetime(panel.loc[train, "market_close_ts"], utc=True)
    assert (known <= decides).all(), (decides, known.max())  # no label from the future
    eval_day = pd.Timestamp(panel.loc[test, "decision_date"].iloc[0])
    last_train = pd.Timestamp(panel.loc[train, "decision_date"].max())
    assert last_train == eval_day - pd.Timedelta(days=2)  # the eve of the eval day is held back
print("no lookahead probe OK", pmlab.__version__, len(result.steps), "folds")
'''


def test_a_backtest_never_trains_on_a_label_from_the_future(installed: Installed) -> None:
    """The installed wheel, on the bundled sample: every fold of ``rolling_origin_eval`` trains only
    on labels known at its decision, and the CLI backtest of that panel is recorded and reproduces."""
    work = installed.workdir / "no_lookahead"
    work.mkdir()
    probe = work / "probe.py"
    probe.write_text(_NO_LOOKAHEAD_PROBE)
    out = _ok(installed.py(probe, work))
    assert f"no lookahead probe OK {_pyproject_version()}" in out

    metrics = work / "metrics.json"
    backtest = _run(
        [
            installed.pmlab, "backtest", "--plugin", "weather_tmax", "--panel", work / "panel.parquet",
            "--stride", "20", "--min-train-rows", "1350", "--min-edge", "0.05", "--output", metrics,
        ],
        cwd=work,
    )  # fmt: skip
    assert "Steps evaluated" in _ok(backtest)
    reproduced = _ok(_run([installed.pmlab, "reproduce", metrics], cwd=work))
    assert "REPRODUCED" in reproduced and "NOT REPRODUCED" not in reproduced


_SETTLE_PROBE = """
import json, sys
from pathlib import Path
from typing import Any

from pmlab import MarketPlugin, MarketSpec, OutcomeBin

message, log = Path(sys.argv[1]).read_text(), Path(sys.argv[2])
lines = message.splitlines()
start = next(i for i, ln in enumerate(lines) if ln.strip() == "from pathlib import Path")
block = []
for ln in lines[start:]:
    if ln.strip() and not ln.startswith("    "):
        break
    block.append(ln)
code = "\\n".join(ln[4:] for ln in block)  # the message's own call, as printed, dedented


class Truth(MarketPlugin):
    family = "fake"
    final = True

    def discover_markets(self, **kw: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kw: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, spec: MarketSpec, **kw: Any) -> str | None:
        return "YES"

    def build_training_row(self, spec: MarketSpec, horizon: str, **kw: Any) -> dict | None:
        return None

    def is_truth_final(self, spec: MarketSpec, **kw: Any) -> bool:
        return self.final


spec = MarketSpec(
    market_id="m1", slug="s", question="q",
    outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
    close_time="2024-01-15T20:00:00Z", market_family="binary",
    metadata={"city": "Chicago", "target_date": "2024-01-15"},
)
trade = {
    "recorded_at": "2024-01-14T10:00:00+00:00", "city_or_segment": "Chicago",
    "target_date": "2024-01-15", "outcome_label": "YES", "direction": "yes",
    "gamma_price": 0.6, "edge_after_fee": 0.12, "horizon": "morning_of",
    "flat_stake": 1.0, "size": 1.666667, "fee_paid": 0.003, "outcome": None, "realized_pnl": None,
}

log.parent.mkdir(parents=True, exist_ok=True)
log.write_text(json.dumps({"trades": [trade]}))
not_final = Truth()
not_final.final = False
exec(code, {"plugin": not_final, "specs": [spec]})
assert json.loads(log.read_text())["trades"][0]["realized_pnl"] is None, "settled before final"
exec(code, {"plugin": Truth(), "specs": [spec]})
settled = json.loads(log.read_text())["trades"][0]
assert settled["outcome"] == "won" and settled["realized_pnl"] is not None, settled
print("settle-trades call OK, pnl", round(settled["realized_pnl"], 4))
"""


def test_settle_trades_refuses_and_the_call_it_prints_settles(installed: Installed) -> None:
    """Issue #43, from the installed wheel: the command exits 1 on stderr with nothing on stdout,
    leaves the workspace untouched, and the Python it prints (run as printed) settles a trade."""
    work = installed.workdir / "settle_trades"
    work.mkdir()
    artifacts = work / "artifacts"
    proc = _run(
        [installed.pmlab, "settle-trades", "--plugin", "weather_tmax", "-a", artifacts, "-w", "ws"],
        cwd=work,
    )
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout == ""
    assert "cannot settle trades" in proc.stderr and "nothing was settled" in proc.stderr
    assert "SettlementEngine(plugin," in proc.stderr and ".settle_all(specs)" in proc.stderr
    log = artifacts / "workspaces" / "ws" / "signals" / "v2" / "forward_paper_trades.json"
    assert str(log) in proc.stderr
    assert not artifacts.exists()  # the command writes nothing, not even the directory

    # a pipeline that chains it now stops at it
    chained = _run(
        ["bash", "-c", f"{installed.pmlab} settle-trades -p weather_tmax -a {artifacts} && echo went-on"],
        cwd=work,
    )  # fmt: skip
    assert chained.returncode == 1 and "went-on" not in chained.stdout

    message = work / "message.txt"
    message.write_text(proc.stderr)
    probe = work / "probe.py"
    probe.write_text(_SETTLE_PROBE)
    out = _ok(installed.py(probe, message, log))
    assert "settle-trades call OK" in out


_EMBARGO_PROBE = """
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

import pmlab
from pmlab import BacktestConfig, build_panel, rolling_origin_eval
from pmlab.datasets import load_sample

assert "site-packages" in pmlab.__file__
work = Path(sys.argv[1])
s = load_sample()
r = s.plugin()
rows = [r.build_training_rows(sp, "previous_evening", decision_time=f"{d}T12:00:00+00:00")
        for d in s.days[:60] for sp in s.plugin(d).discover_markets()]
panel = build_panel(rows).reset_index(drop=True)
panel.to_parquet(work / "panel.parquet")


class Rec:
    def __init__(self):
        self.folds = []
    def fit(self, X, y):
        self.train = X.index
    def predict_proba(self, X):
        self.folds.append((self.train, X.index))
        return np.full((len(X), 2), 0.5)


def folds(**kw):
    m = Rec()
    rolling_origin_eval(panel, m, min_train_rows=200, stride=5, **kw)
    return {panel.loc[e, "decision_date"].iloc[0]: set(t) for t, e in m.folds}


base, none = folds(), folds(embargo_days=None)
assert base == none and len(base) >= 8, len(base)
assert folds(embargo_days=0) == base and folds(embargo_days=2) == base
held = folds(embargo_days=7)
assert held.keys() <= base.keys() and any(held[d] < base[d] for d in held)
for day, train in held.items():
    ages = [(pd.Timestamp(day) - pd.Timestamp(panel.loc[i, "decision_date"])).days for i in train]
    assert min(ages) >= 7, (day, min(ages))
    known = pd.to_datetime(panel.loc[list(train), "market_close_ts"], utc=True)
    assert (known <= pd.Timestamp(panel.loc[panel["decision_date"] == day, "decision_ts"].min())).all()

for bad, exc in ((-1, ValueError), (2.5, TypeError), (True, TypeError), ("3", TypeError)):
    try:
        rolling_origin_eval(panel, Rec(), embargo_days=bad)
    except exc:
        pass
    else:
        raise AssertionError(f"embargo_days={bad!r} was accepted")
assert "embargo_days" not in BacktestConfig.resolve(plugin="weather_tmax").to_dict()
assert BacktestConfig.resolve(plugin="weather_tmax", embargo_days=3).to_dict()["embargo_days"] == 3
print("embargo probe OK", pmlab.__version__, len(base), "folds", len(held), "with a 7-day embargo")
"""


def test_embargo_days_from_the_installed_wheel(installed: Installed) -> None:
    """1.1.0, from the installed wheel: ``embargo_days`` only ever removes training rows, never
    pulls in an unknown label; the CLI option is recorded, replayed and refused when invalid."""
    work = installed.workdir / "embargo_days"
    work.mkdir()
    probe = work / "probe.py"
    probe.write_text(_EMBARGO_PROBE)
    out = _ok(installed.py(probe, work))
    assert f"embargo probe OK {_pyproject_version()}" in out

    def backtest(out_file: Path, *extra: str | Path) -> subprocess.CompletedProcess[str]:
        return _run(
            [
                installed.pmlab, "backtest", "--plugin", "weather_tmax", "--panel", work / "panel.parquet",
                "--stride", "10", "--min-train-rows", "400", "--min-edge", "0.05",
                "--output", out_file, *extra,
            ],
            cwd=work,
        )  # fmt: skip

    plain, held = work / "plain.json", work / "held.json"
    _ok(backtest(plain))
    shown = _ok(backtest(held, "--embargo-days", "7"))
    assert "Embargo: a fold trains on rows decided at least 7 days before it" in shown
    plain_rec = json.loads(plain.read_text())["reproducibility"]
    held_rec = json.loads(held.read_text())["reproducibility"]
    assert "embargo_days" not in plain_rec["config"]  # a run without it records what 1.0.3 did
    assert held_rec["config"]["embargo_days"] == 7
    assert held_rec["metrics"]["steps"] < plain_rec["metrics"]["steps"]
    for record in (plain, held):
        replay = _ok(_run([installed.pmlab, "reproduce", record], cwd=work))
        assert "REPRODUCED" in replay and "NOT REPRODUCED" not in replay

    refused = backtest(work / "bad.json", "--embargo-days", "-1")
    assert refused.returncode == 1 and "embargo_days" in refused.stdout + refused.stderr
    assert backtest(work / "bad.json", "--embargo-days", "2.5").returncode == 2
