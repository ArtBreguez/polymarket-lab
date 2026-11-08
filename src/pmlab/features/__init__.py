"""pmlab.features - Feature engineering utilities."""

from __future__ import annotations

from pmlab.features.target_encoding import FittedTargetEncoder, TargetEncoder
from pmlab.features.transforms import (
    add_interactions,
    add_lags,
    add_missing_indicators,
    add_rolling_stats,
    clip_outliers,
    encode_cyclical,
    encode_onehot,
)

__all__ = [
    "FittedTargetEncoder",
    "TargetEncoder",
    "add_interactions",
    "add_lags",
    "add_missing_indicators",
    "add_rolling_stats",
    "clip_outliers",
    "encode_cyclical",
    "encode_onehot",
]
