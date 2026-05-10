"""pmlab — Generic ML framework for Polymarket prediction markets."""
from __future__ import annotations

__version__ = "0.1.0"

from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.backtest.metrics import BacktestMetrics, compute_metrics
from pmlab.core.edge import compute_edge
from pmlab.core.fees import estimate_fee
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.core.pnl import Position, settle_position
from pmlab.core.sizing import flat_stake_size, kelly_fraction, kelly_stake_size
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.live_broker import LiveBroker, LiveBrokerError, OrderReceipt
from pmlab.execution.paper_broker import PaperBroker
from pmlab.execution.settlement import SettlementEngine
from pmlab.features.transforms import (
    add_lags,
    add_rolling_stats,
    clip_outliers,
    encode_cyclical,
    encode_onehot,
)
from pmlab.markets.async_clob_client import AsyncClobClient
from pmlab.markets.async_gamma_client import AsyncGammaClient
from pmlab.markets.cache import DiskCache
from pmlab.markets.clob_client import ClobClient
from pmlab.markets.gamma_client import GammaClient
from pmlab.modeling.base import MarketForecaster
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.diagnostics import BrierDecomposition, brier_decomposition, reliability_data
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.plugins.base import MarketPlugin
from pmlab.plugins.registry import PluginRegistry
from pmlab.reports.html_report import generate_report
from pmlab.workspace.context import WorkspaceContext

__all__ = [
    "__version__",
    # Core
    "MarketSpec", "OutcomeBin",
    "Position", "settle_position",
    "compute_edge", "estimate_fee",
    "flat_stake_size", "kelly_fraction", "kelly_stake_size",
    # Execution
    "EdgeSignal", "PaperBroker", "SettlementEngine",
    "LiveBroker", "LiveBrokerError", "OrderReceipt",
    # Features
    "add_lags", "add_rolling_stats", "encode_cyclical", "encode_onehot", "clip_outliers",
    # Markets
    "GammaClient", "ClobClient", "AsyncGammaClient", "AsyncClobClient", "DiskCache",
    # Modeling
    "MarketForecaster", "ChampionManifest", "LGBMForecaster",
    "BrierDecomposition", "brier_decomposition", "reliability_data",
    # Backtest
    "HoldoutGateResult", "SegmentGateResult",
    "BacktestMetrics", "compute_metrics",
    # Reports
    "generate_report",
    # Plugins / Workspace
    "MarketPlugin", "PluginRegistry",
    "WorkspaceContext",
]
