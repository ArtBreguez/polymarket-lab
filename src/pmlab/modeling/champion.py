"""Champion model manifest with hard gate enforcement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.core.costs import CostModel
from pmlab.modeling.base import MarketForecaster

if TYPE_CHECKING:
    from pmlab.modeling.calibration import IsotonicCalibrator


@dataclass
class ChampionManifest:
    model_name: str
    model_path: Path
    calibrator_path: Path | None
    gate: HoldoutGateResult
    published_at: str
    plugin_family: str
    # The friction the gate was computed under. None for champions published
    # before costs were recorded — a live run cannot then verify it is trading
    # under the same assumptions the promotion was based on.
    costs: CostModel | None = None

    @classmethod
    def publish(
        cls,
        model: MarketForecaster,
        gate: HoldoutGateResult,
        output_dir: Path,
        plugin_family: str,
        model_name: str = "champion",
        calibrator: IsotonicCalibrator | None = None,
        costs: CostModel | None = None,
    ) -> ChampionManifest:
        """Publish a champion model.

        HARD GATE: raises ValueError if gate.decision != "GO".

        Args:
            costs: The cost model the gate was computed under. Recording it lets a
                live run assert it is trading with the same friction the promotion
                assumed; without it, live and backtest PnL can diverge for reasons
                that have nothing to do with the market.
        """
        if gate.decision != "GO":
            raise ValueError(
                f"Cannot publish champion with NO_GO gate. Gate decision: {gate.decision}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "champion.pkl"
        model.save(model_path)

        calibrator_path: Path | None = None
        if calibrator is not None:
            calibrator_path = output_dir / "calibrator.pkl"
            calibrator.save(calibrator_path)

        published_at = datetime.now(UTC).isoformat()

        manifest = cls(
            model_name=model_name,
            model_path=model_path,
            calibrator_path=calibrator_path,
            gate=gate,
            published_at=published_at,
            plugin_family=plugin_family,
            costs=costs,
        )

        json_data = {
            "model_name": model_name,
            "model_path": str(model_path),
            "calibrator_path": str(calibrator_path) if calibrator_path else None,
            "published_at": published_at,
            "plugin_family": plugin_family,
            "publish_gate": gate.to_dict(),
            "costs": costs.to_dict() if costs is not None else None,
        }
        json_path = output_dir / "champion.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        return manifest

    @classmethod
    def load(cls, json_path: Path) -> ChampionManifest:
        """Load a ChampionManifest from champion.json."""
        json_path = Path(json_path)
        with open(json_path) as f:
            data = json.load(f)

        gate = HoldoutGateResult.from_dict(data["publish_gate"])
        calibrator_path = Path(data["calibrator_path"]) if data.get("calibrator_path") else None
        # .get() rather than [], so a manifest written before costs were recorded
        # still loads instead of raising KeyError.
        raw_costs = data.get("costs")
        costs = CostModel(**raw_costs) if raw_costs else None

        return cls(
            model_name=data["model_name"],
            model_path=Path(data["model_path"]),
            calibrator_path=calibrator_path,
            gate=gate,
            published_at=data["published_at"],
            plugin_family=data["plugin_family"],
            costs=costs,
        )

    def get_allowed_segments(self) -> set[str]:
        """Return segments where the gate result passes."""
        return {r.segment for r in self.gate.segment_results if r.passes}

    def load_model(self) -> MarketForecaster:
        """Load the champion model from disk."""
        from pmlab.modeling.lgbm_baseline import LGBMForecaster

        return LGBMForecaster.load(self.model_path)
