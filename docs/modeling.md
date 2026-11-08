# Modeling: ensembles, calibration, conformal sets, time-aware CV

Every forecaster in pmlab implements the same `MarketForecaster` interface (`fit`, `predict_proba`,
`save`, `load`), so they compose freely and drop into the backtest, the champion gate and the
brokers unchanged. This page shows the building blocks that wrap or combine forecasters. Tuning has
[its own page](tuning.md); the diagnostics that tell you whether a probability can be trusted are in
[Calibration diagnostics](calibration.md).

Every example below is executed against the installed wheel by the end-to-end test, as written. The
first block builds the small synthetic training data the others use.

```python
import numpy as np
import pandas as pd

from pmlab import LGBMForecaster, SklearnForecaster

n = 2000
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"feature_{i}" for i in range(4)])
signal = X["feature_0"] + 0.5 * X["feature_1"] + rng.normal(scale=0.7, size=n)
y = (signal > 0).astype(int)                            # a binary market: did the outcome occur?
y_multi = pd.Series(np.digitize(signal, [-0.7, 0.7]))   # three outcomes, for the multiclass example
split = 1500                                            # train on the first rows, test on the rest
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
```

## Ensembles

Blend several models to cut variance: a weighted average of the members' probabilities.

```python
from pmlab import EnsembleForecaster, LGBMForecaster, SklearnForecaster

ens = EnsembleForecaster(
    forecasters=[
        LGBMForecaster(),
        SklearnForecaster(estimator="random_forest", n_estimators=200),
        SklearnForecaster(estimator="logistic_regression"),
    ],
    weights=[2.0, 1.0, 1.0],  # optional; defaults to equal weight
)
ens.fit(X_train, y_train)
proba = ens.predict_proba(X_test)
print(proba.shape)
```

## Probability calibration

Edge is computed from probabilities, so calibration matters. Wrap any forecaster to calibrate its
positive-class output on a held-out part of the training data:

```python
from pmlab import CalibratedForecaster, LGBMForecaster

clf = CalibratedForecaster(LGBMForecaster(), method="isotonic")  # or "sigmoid" (Platt)
clf.fit(X_train, y_train)
calibrated = clf.predict_proba(X_test)  # calibrated (n, 2) probabilities
print(calibrated[:3].round(3))
```

Or calibrate raw scores directly:

```python
from pmlab import IsotonicCalibrator, SigmoidCalibrator

model = LGBMForecaster()
model.fit(X_train, y_train)
raw_scores = model.predict_proba(X_test)[:, 1]
cal = SigmoidCalibrator()          # Platt scaling: sample-efficient
cal.fit(raw_scores, y_test)        # in real use: fit on a split the model never trained on
p = cal.transform(raw_scores)
print(p[:3].round(3))
```

`IsotonicCalibrator` is the non-parametric alternative (monotone, needs more data).

## Conformal prediction sets

Turn a point probability into a **prediction set** with a distribution-free coverage guarantee: on
exchangeable data, the set contains the true label with probability at least 1 − alpha, however
well-specified the base model is. A singleton set is a confident call; a full set is the model
saying "toss-up".

```python
from pmlab import ConformalForecaster, LGBMForecaster

clf = ConformalForecaster(LGBMForecaster(), alpha=0.1)  # target 90% coverage
clf.fit(X_train, y_train)

sets = clf.predict_set(X_test)     # list[set[int]]: e.g. {1}, {0, 1}, {0}
proba = clf.predict_proba(X_test)  # still a drop-in forecaster
print(sets[:5])
```

Exchangeable is the operative word: the guarantee holds when calibration and test rows come from the
same distribution. A market that drifts breaks it, which is what [drift monitoring](drift-monitoring.md)
is for.

## Multiclass markets

Calibration extends to multi-outcome markets (an F1 winner, a multi-candidate race) one-vs-rest:
pass a target with three or more classes.

```python
from pmlab import CalibratedForecaster, SklearnForecaster, multiclass_brier

clf = CalibratedForecaster(SklearnForecaster("random_forest"), method="isotonic")
clf.fit(X_train, y_multi.iloc[:split])           # three classes
proba = clf.predict_proba(X_test)              # calibrated (n, n_classes), rows sum to 1

score = multiclass_brier(y_multi.iloc[split:], proba)   # Brier + climatology skill score
print(round(score.brier_score, 3), round(score.skill_score, 3))
```

## Time-aware cross-validation

For model selection outside the backtest, use splitters that respect time instead of
`sklearn.model_selection.KFold`, which would shuffle the future into the past:

```python
import numpy as np

from pmlab import embargoed_split, purged_kfold

idx = np.arange(n)                   # row positions, sorted by decision_date
for train, test in purged_kfold(idx, n_splits=5, embargo=3):
    ...   # test is a contiguous block; train purges an embargo band around it

for train, test in embargoed_split(idx, n_splits=5, embargo=3):
    ...   # expanding-origin walk-forward: train is always strictly earlier
```

On a panel with one row per bin, sort by `decision_date` and keep the bins of one market on the same
side of every split (see [One row per bin](plugin-authoring.md#one-row-per-bin)).

## Hyper-parameter search

`TunedForecaster` searches with Optuna but scores every trial through `rolling_origin_eval`, so model
selection is walk-forward, and it can score under the trade policy and costs you will actually trade.
It needs the `tune` extra (`pip install "pmlab[tune]"`). See [Tuning](tuning.md).
