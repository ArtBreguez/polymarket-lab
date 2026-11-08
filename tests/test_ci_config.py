"""Smoke tests ensuring CI configuration is sane."""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_ci() -> dict:
    ci_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(ci_path.read_text())


def test_ci_has_test_job():
    ci = _load_ci()
    assert "test" in ci["jobs"]


def test_ci_has_build_job():
    ci = _load_ci()
    assert "build" in ci["jobs"]


def test_ci_test_job_runs_mypy():
    ci = _load_ci()
    steps = ci["jobs"]["test"]["steps"]
    step_names = [s.get("name", "").lower() for s in steps]
    assert any("mypy" in n for n in step_names), f"No mypy step found. Steps: {step_names}"


def test_ci_test_job_runs_pytest_with_coverage():
    ci = _load_ci()
    steps = ci["jobs"]["test"]["steps"]
    cmds = [s.get("run", "") for s in steps]
    assert any("--cov" in c for c in cmds), "No coverage flag found in pytest step"


def test_ci_coverage_report_xml():
    ci = _load_ci()
    steps = ci["jobs"]["test"]["steps"]
    cmds = [s.get("run", "") for s in steps]
    assert any("xml" in c for c in cmds), "No XML coverage report generated"


def test_ci_has_upload_coverage_step():
    ci = _load_ci()
    steps = ci["jobs"]["test"]["steps"]
    step_names = [s.get("name", "").lower() for s in steps]
    uses = [s.get("uses", "").lower() for s in steps]
    has_upload = any("coverage" in n for n in step_names) or any(
        "codecov" in u or "coverage" in u for u in uses
    )
    assert has_upload, f"No coverage upload step found. Steps: {step_names}"


def test_ci_triggers_on_pr():
    ci = _load_ci()
    assert "pull_request" in ci["on"]


def test_ci_has_e2e_job_running_the_installed_wheel_tests():
    """The wheel-install end-to-end tests run in their own job, not in the default pytest."""
    ci = _load_ci()
    assert "e2e" in ci["jobs"]
    cmds = [s.get("run", "") for s in ci["jobs"]["e2e"]["steps"]]
    assert any("pytest" in c and "-m e2e" in c for c in cmds), cmds


def test_default_pytest_deselects_e2e():
    import tomllib

    root = Path(__file__).parent.parent
    cfg = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert "not e2e" in cfg["addopts"]
    assert any(m.startswith("e2e") for m in cfg["markers"])


def _pyproject() -> dict:
    import tomllib

    return tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())


def test_coverage_gate_is_at_least_90_and_is_the_single_source_of_truth():
    """The gate lives in pyproject; CI must not carry a second, lower (or any) copy of it."""
    import re

    gate = _pyproject()["tool"]["coverage"]["report"]["fail_under"]
    assert gate >= 90
    ci = _load_ci()
    cmds = [s.get("run", "") for s in ci["jobs"]["test"]["steps"]]
    pytest_cmd = next(c for c in cmds if "--cov" in c)
    overrides = [int(x) for x in re.findall(r"--cov-fail-under=(\d+)", pytest_cmd)]
    assert all(o >= gate for o in overrides), (overrides, gate)
    assert "--cov-fail-under=70" not in pytest_cmd


def test_ci_coverage_step_fails_the_job_when_the_gate_is_missed():
    """No `|| true`/continue-on-error on the step that runs pytest with coverage."""
    steps = _load_ci()["jobs"]["test"]["steps"]
    step = next(s for s in steps if "--cov" in s.get("run", ""))
    assert not step.get("continue-on-error")
    assert "|| true" not in step["run"] and "--no-cov" not in step["run"]


def test_ci_has_perf_job_running_the_budget_guard_without_coverage_tracing():
    ci = _load_ci()
    assert "perf" in ci["jobs"]
    cmds = [s.get("run", "") for s in ci["jobs"]["perf"]["steps"]]
    perf_cmd = next(c for c in cmds if "pytest" in c and "-m perf" in c)
    assert "--cov" not in perf_cmd, "tracing would make every wall time meaningless"
    assert not ci["jobs"]["perf"]["steps"][-1].get("continue-on-error")


def test_default_pytest_deselects_perf():
    cfg = _pyproject()["tool"]["pytest"]["ini_options"]
    assert "not perf" in cfg["addopts"]
    assert any(m.startswith("perf") for m in cfg["markers"])
