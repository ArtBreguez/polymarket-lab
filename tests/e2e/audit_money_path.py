"""The money-path guards, used the way a user would, from the installed wheel (audit 1.1.1).

Standalone (public names only, no repo imports):  python audit_money_path.py
Prints ``audit money path: OK`` and exits 0, or raises on the first guard that does not hold.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab import (
    HoldoutGateResult,
    LGBMForecaster,
    MarketPlugin,
    MarketSpec,
    OutcomeBin,
    SettlementEngine,
    compute_metrics,
)

# 1. The gate: trades that never settled are not evidence (#51).
open_log = pd.DataFrame([{"realized_pnl": None, "outcome": None, "segment": "all"}] * 60)
for required in (None, ["all"]):
    gate = HoldoutGateResult.evaluate(open_log, required_segments=required)
    assert gate.decision == "NO_GO", (required, gate)
    assert gate.segment_results[0].reason == "unsettled_trades"
assert compute_metrics(open_log).num_trades == 0
settled = pd.DataFrame([{"realized_pnl": 0.1, "outcome": "won", "segment": "all"}] * 60)
assert HoldoutGateResult.evaluate(settled).decision == "GO"


# 2 and 3. Settlement: a NaN truth settles nothing; trades no spec matches are reported (#52, #53).
class Plugin(MarketPlugin):
    family = "audit"

    def __init__(self, truth: float) -> None:
        self.truth = truth

    def discover_markets(self, **kwargs):  # type: ignore[no-untyped-def]
        return []

    def fetch_features(self, spec, horizon, **kwargs):  # type: ignore[no-untyped-def]
        return {}

    def fetch_truth(self, spec, **kwargs):  # type: ignore[no-untyped-def]
        return self.truth

    def build_training_row(self, spec, horizon, **kwargs):  # type: ignore[no-untyped-def]
        return None


def spec(metadata: dict[str, str]) -> MarketSpec:
    return MarketSpec(
        market_id="m",
        slug="s",
        question="q",
        outcome_bins=[OutcomeBin(label="lo", upper=27.0), OutcomeBin(label="hi", lower=31.0)],
        close_time="2026-01-01T00:00:00Z",
        market_family="audit",
        metadata=metadata,
    )


trade = {
    "recorded_at": "2026-01-01T00:00:00+00:00", "city_or_segment": "A", "target_date": "2026-01-02",
    "outcome_label": "lo", "direction": "yes", "gamma_price": 0.4, "fill_price": 0.4, "model_prob": 0.6,
    "edge_after_fee": 0.2, "horizon": "morning_of", "flat_stake": 1.0, "size": 2.5, "fee_paid": 0.003,
    "outcome": None, "realized_pnl": None,
}  # fmt: skip
work = Path(tempfile.mkdtemp(prefix="pmlab audit "))
log = work / "trades.json"
log.write_text(json.dumps({"trades": [trade]}))
engine = SettlementEngine(Plugin(math.nan), log)
summary = engine.settle_all(
    [spec({"city": "A", "target_date": "2026-01-02"})], today_str="2026-02-01"
)
assert summary == {"settled": 0, "pending": 1, "total_pnl": 0.0}, summary
assert json.loads(log.read_text())["trades"][0]["outcome"] is None
summary = SettlementEngine(Plugin(20.0), log).settle_all(
    [spec({"city": "A", "target_date": "2026-01-02"})], today_str="2026-02-01"
)
assert summary["settled"] == 1 and summary["total_pnl"] > 0, summary


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


cap = Capture()
logging.getLogger("pmlab").addHandler(cap)
log.write_text(json.dumps({"trades": [trade]}))
SettlementEngine(Plugin(20.0), log).settle_all(
    [spec({"segment": "A", "target_date": "2026-01-02"})], today_str="2026-02-01"
)
assert any("metadata['city']" in m for m in cap.messages), cap.messages

# 4. A model fed the same columns in another order predicts the same (#54).
rng = np.random.default_rng(0)
X = pd.DataFrame({"feature_a": rng.normal(size=200), "feature_b": rng.normal(size=200)})
model = LGBMForecaster(n_estimators=10, n_jobs=1)
model.fit(X, (X["feature_a"] > 0).astype(int))
assert np.allclose(model.predict_proba(X[["feature_b", "feature_a"]]), model.predict_proba(X))
model.save(work / "m.pkl")
loaded = LGBMForecaster.load(work / "m.pkl")
assert np.allclose(loaded.predict_proba(X[["feature_b", "feature_a"]]), model.predict_proba(X))

print("audit money path: OK")
