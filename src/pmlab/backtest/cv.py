"""Leakage-aware cross-validation splits for time-series / panel data.

Standard k-fold shuffles rows, which leaks the future into the training fold on
any time-ordered dataset — fatal for prediction-market models. These splitters
respect time:

- ``purged_kfold``: contiguous test blocks with an *embargo* that purges training
  rows adjacent to each test block (López de Prado style), so information either
  side of the test window can't bleed in.
- ``embargoed_split``: strict walk-forward (expanding origin) — train is always
  earlier than test, separated by an embargo gap. Mirrors production, where you
  only ever have the past.

Both operate on positional indices ordered by time; pass indices already sorted
by ``decision_date``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def purged_kfold(
    samples: NDArray[np.int_] | np.ndarray,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Yield (train_idx, test_idx) for purged k-fold CV.

    The samples are split into ``n_splits`` contiguous test blocks (preserving
    time order). For each, the training set is every other sample minus an
    ``embargo`` band immediately before and after the test block.

    Args:
        samples: 1-D array whose length defines the sample count (values unused;
            pass ``np.arange(n)`` or the time-ordered index).
        n_splits: Number of folds (>= 2).
        embargo: Number of samples to purge on each side of the test block.

    Yields:
        (train_idx, test_idx) as positional-index arrays into ``samples``.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits}.")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0; got {embargo}.")

    n = len(samples)
    all_idx = np.arange(n)
    fold_bounds = np.linspace(0, n, n_splits + 1, dtype=int)

    for f in range(n_splits):
        start, stop = fold_bounds[f], fold_bounds[f + 1]
        test_idx = all_idx[start:stop]
        if test_idx.size == 0:
            continue
        # Purge an embargo band on both sides of the contiguous test block.
        lo = max(0, start - embargo)
        hi = min(n, stop + embargo)
        purged = np.zeros(n, dtype=bool)
        purged[lo:hi] = True
        train_idx = all_idx[~purged]
        yield train_idx, test_idx


def embargoed_split(
    samples: NDArray[np.int_] | np.ndarray,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Yield (train_idx, test_idx) for walk-forward CV with an embargo gap.

    Expanding-origin: fold ``f``'s training set is everything up to the start of
    its test block minus an ``embargo`` gap, and the test block is the next
    contiguous slice. Train is always strictly earlier than test.

    Args:
        samples: 1-D array whose length defines the sample count.
        n_splits: Number of forward test blocks (>= 2).
        embargo: Gap (in samples) between the end of train and the start of test.

    Yields:
        (train_idx, test_idx) positional-index arrays. Early folds may yield an
        empty train_idx when the embargo consumes the available history.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits}.")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0; got {embargo}.")

    n = len(samples)
    all_idx = np.arange(n)
    # Reserve the first block as initial training history; test on the rest.
    fold_bounds = np.linspace(0, n, n_splits + 2, dtype=int)

    for f in range(1, n_splits + 1):
        start, stop = fold_bounds[f], fold_bounds[f + 1]
        test_idx = all_idx[start:stop]
        if test_idx.size == 0:
            continue
        train_end = max(0, start - embargo)
        train_idx = all_idx[:train_end]
        yield train_idx, test_idx
