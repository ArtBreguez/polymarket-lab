"""pmlab — Generic ML framework for Polymarket prediction markets."""
from __future__ import annotations

__version__ = "0.1.0"

from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.backtest.metrics import BacktestMetrics, compute_metrics
from pmlab.core.edge import compute_edge
from pmlab.core.fees import estimate_fee
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.core.pnl import Position, settle_position
from pmlab.core.sizing import flat_stake_size
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker
from pmlab.execution.settlement import SettlementEngine
from pmlab.modeling.base import MarketForecaster
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.plugins.base import MarketPlugin
from pmlab.plugins.registry import PluginRegistry
from pmlab.workspace.context import WorkspaceContext

__all__ = [
    "__version__",
    "MarketSpec", "OutcomeBin",
    "Position", "settle_position",
    "compute_edge", "estimate_fee", "flat_stake_size",
    "MarketPlugin", "PluginRegistry",
    "MarketForecaster", "ChampionManifest", "LGBMForecaster",
    "HoldoutGateResult", "SegmentGateResult",
    "BacktestMetrics", "compute_metrics",
    "EdgeSignal", "PaperBroker", "SettlementEngine",
    "WorkspaceContext",
]
