"""Optional MLflow backend for experiment tracking.

Behind the ``track`` extra: ``uv sync --extra track`` (or ``pip install pmlab[track]``).
MLflow is imported lazily inside ``run()`` so importing this module — or the core
``pmlab.tracking`` package — never requires MLflow to be installed. If it isn't,
constructing an ``MLflowTracker`` raises a clear, actionable ImportError.

It maps pmlab's run semantics onto MLflow: ``experiment`` → MLflow experiment,
params → ``log_params``, metrics → ``log_metrics``, the gate decision →
``log_params({"gate_decision": ...})`` plus the full gate dict as a JSON artifact.
It also mirrors every run into a :class:`~pmlab.tracking.RunRecord` so
``list_runs``/``get_run`` return the same typed objects as the local backend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pmlab.tracking import ActiveRun, RunRecord, _json_safe, _to_plain_dict

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["MLflowTracker"]


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "MLflowTracker requires the 'track' extra. Install it with "
            "`uv sync --extra track` or `pip install pmlab[track]`."
        ) from e
    return mlflow


class MLflowTracker:
    """Experiment tracker backed by MLflow. Conforms to ``ExperimentTracker``."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._mlflow = _require_mlflow()
        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)

    def run(self, experiment: str, run_name: str) -> ActiveRun:
        self._mlflow.set_experiment(experiment)
        active = self._mlflow.start_run(run_name=run_name)
        record = RunRecord(
            run_id=active.info.run_id,
            experiment=experiment,
            run_name=run_name,
            started_at=datetime.now(UTC).isoformat(),
            status="running",
        )
        return _MLflowActiveRun(self, record)

    def list_runs(self, experiment: str) -> list[RunRecord]:
        mlflow = self._mlflow
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            return []
        df = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
        records: list[RunRecord] = []
        for _, row in df.iterrows():
            records.append(self._row_to_record(experiment, row))
        return records

    def get_run(self, experiment: str, run_id: str) -> RunRecord | None:
        mlflow = self._mlflow
        try:
            run = mlflow.get_run(run_id)
        except Exception:
            return None
        params = dict(run.data.params)
        metrics = dict(run.data.metrics)
        gate = {"decision": params.get("gate_decision")} if "gate_decision" in params else {}
        return RunRecord(
            run_id=run_id,
            experiment=experiment,
            run_name=run.data.tags.get("mlflow.runName", ""),
            params=params,
            metrics=metrics,
            gate=gate,
            status=run.info.status.lower(),
            started_at=str(run.info.start_time),
        )

    def _row_to_record(self, experiment: str, row: Any) -> RunRecord:
        params = {k.split("params.", 1)[1]: v for k, v in row.items() if k.startswith("params.")}
        metrics = {k.split("metrics.", 1)[1]: v for k, v in row.items() if k.startswith("metrics.")}
        return RunRecord(
            run_id=row.get("run_id", ""),
            experiment=experiment,
            run_name=row.get("tags.mlflow.runName", ""),
            params=params,
            metrics=metrics,
            gate={"decision": params["gate_decision"]} if "gate_decision" in params else {},
            status=str(row.get("status", "")).lower(),
            started_at=str(row.get("start_time", "")),
        )


class _MLflowActiveRun(ActiveRun):
    """ActiveRun that forwards logs to MLflow instead of a local JSON file."""

    def __init__(self, tracker: MLflowTracker, record: RunRecord) -> None:
        super().__init__(tracker, record)  # type: ignore[arg-type]
        self._mlflow = tracker._mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        safe = _json_safe(params)
        self._record.params.update(safe)
        self._mlflow.log_params(safe)

    def log_metrics(self, metrics: Any) -> None:
        plain = _json_safe(_to_plain_dict(metrics))
        self._record.metrics.update(plain)
        # np.int64 is normalized to int by _json_safe, so it's no longer silently
        # dropped by this numeric filter; non-finite floats became None -> skipped.
        numeric = {
            k: v
            for k, v in plain.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if numeric:
            self._mlflow.log_metrics(numeric)

    def log_gate(self, gate: Any) -> None:
        plain = _json_safe(_to_plain_dict(gate))
        self._record.gate = plain
        decision = plain.get("decision")
        if decision is not None:
            self._mlflow.log_param("gate_decision", decision)
        self._mlflow.log_dict(plain, "gate.json")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        status = "FAILED" if exc is not None else "FINISHED"
        if exc is not None:
            self._record.status = "failed"
            self._record.error = f"{type(exc).__name__}: {exc}"
        else:
            self._record.status = "finished"
        self._mlflow.end_run(status=status)
