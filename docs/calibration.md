# Calibration Diagnostics

`pmlab.modeling.diagnostics` provides tools to evaluate how well your model's predicted probabilities match realized outcomes.

---

## Why Calibration Matters for Prediction Markets

A model with good calibration means: when it says 70% probability, the outcome occurs ~70% of the time.

Poor calibration leads to:
- **Overconfident model:** entry_price looks cheap, but actual win_prob is lower → negative edge
- **Underconfident model:** avoids valid bets → missed EV

Calibration is especially important in prediction markets because **edge = model_prob - market_price**. A miscalibrated model produces systematically wrong edge estimates.

---

## Brier Score

The Brier score is the mean squared error between predicted probabilities and binary outcomes:

```
BS = (1/N) * Σ (f_i - o_i)²
```

Where `f_i` is the forecast probability and `o_i` is 0 or 1.

- **BS = 0**: perfect calibration
- **BS = 0.25**: climatological forecast (always predicting base rate)
- **BS > 0.25**: worse than climatology

---

## Murphy (1973) Decomposition

```python
from pmlab import brier_decomposition, BrierDecomposition

result = brier_decomposition(y_true, y_prob, n_bins=10)
print(result)
```

The decomposition: `BS = reliability - resolution + uncertainty`

| Component | Meaning | Goal |
|---|---|---|
| `uncertainty` | Variance of the base rate — property of the data, not the model | N/A |
| `reliability` | Mean squared deviation between forecast and conditional outcome | → 0 (low = well calibrated) |
| `resolution` | How much forecasts deviate from the base rate | → high (high = informative) |
| `brier_score` | Overall BS = reliability - resolution + uncertainty | → 0 |
| `skill_score` | 1 - BS / BS_climatology | → 1 (positive = better than climatology) |

```python
result = brier_decomposition(y_true, y_prob)
print(f"Brier score:  {result.brier_score:.4f}")
print(f"Skill score:  {result.skill_score:.4f}")
print(f"Reliability:  {result.reliability:.4f}")
print(f"Resolution:   {result.resolution:.4f}")
print(f"Uncertainty:  {result.uncertainty:.4f}")
print(f"N samples:    {result.n_samples}")
```

**Interpretation:**
- High reliability + low resolution → model is overconfident and not informative
- Low reliability + high resolution → well-calibrated and informative (ideal)
- skill_score > 0 → model beats climatology

---

## Reliability Diagram Data

A reliability diagram plots mean predicted probability vs fraction of positives per bin. A perfectly calibrated model follows the diagonal.

```python
from pmlab import reliability_data
import matplotlib.pyplot as plt  # or any plotting lib

centers, mean_pred, frac_pos = reliability_data(y_true, y_prob, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
plt.plot(mean_pred, frac_pos, "o-", label="Model")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability Diagram")
plt.legend()
plt.show()
```

Returns only non-empty bins. All three arrays have the same length.

---

## Using with IsotonicCalibrator

If your reliability diagram shows a systematic curve (overconfident or underconfident), apply `IsotonicCalibrator`:

```python
from pmlab.modeling.calibration import IsotonicCalibrator

cal = IsotonicCalibrator()
cal.fit(y_true_val, raw_probs_val)
calibrated_probs = cal.predict(raw_probs_test)

# Compare before/after
before = brier_decomposition(y_true_test, raw_probs_test)
after = brier_decomposition(y_true_test, calibrated_probs)
print(f"Before: BS={before.brier_score:.4f}, reliability={before.reliability:.4f}")
print(f"After:  BS={after.brier_score:.4f}, reliability={after.reliability:.4f}")
```

Isotonic calibration is non-parametric and well-suited to binary prediction markets. Fit it on a held-out validation set, not the training set.

---

## Recommended Workflow

1. Run `rolling_origin_eval` → get `result.trades` with model probabilities
2. Compute `brier_decomposition` on the backtest probabilities
3. Plot `reliability_data` — check for systematic bias
4. If reliability > 0.01, fit `IsotonicCalibrator` on a validation fold
5. Re-evaluate after calibration
6. Include `brier_score` in `generate_report` for tracking over time
