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
    has_upload = any("coverage" in n for n in step_names) or any("codecov" in u or "coverage" in u for u in uses)
    assert has_upload, f"No coverage upload step found. Steps: {step_names}"


def test_ci_triggers_on_pr():
    ci = _load_ci()
    assert "pull_request" in ci["on"]
