"""TDD tests for purged / embargoed time-series CV splits — RED first."""

from __future__ import annotations

import numpy as np
import pytest

from pmlab.backtest.cv import embargoed_split, purged_kfold


class TestPurgedKFold:
    def test_returns_n_splits(self) -> None:
        dates = np.arange(100)
        splits = list(purged_kfold(dates, n_splits=5))
        assert len(splits) == 5

    def test_train_and_test_disjoint(self) -> None:
        dates = np.arange(100)
        for train_idx, test_idx in purged_kfold(dates, n_splits=5, embargo=2):
            assert set(train_idx).isdisjoint(set(test_idx))

    def test_embargo_removes_neighbors(self) -> None:
        # With an embargo, train indices adjacent to the test block are dropped.
        dates = np.arange(50)
        embargo = 3
        for train_idx, test_idx in purged_kfold(dates, n_splits=5, embargo=embargo):
            lo, hi = test_idx.min(), test_idx.max()
            # No train index within `embargo` of the test block boundaries.
            for ti in train_idx:
                assert not (lo - embargo <= ti < lo) or ti < lo - embargo
                assert not (hi < ti <= hi + embargo)

    def test_every_sample_tested_once(self) -> None:
        dates = np.arange(60)
        tested: list[int] = []
        for _, test_idx in purged_kfold(dates, n_splits=6):
            tested.extend(test_idx.tolist())
        assert sorted(tested) == list(range(60))

    def test_invalid_n_splits_raises(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            list(purged_kfold(np.arange(10), n_splits=1))

    def test_negative_embargo_raises(self) -> None:
        with pytest.raises(ValueError, match="embargo"):
            list(purged_kfold(np.arange(10), n_splits=3, embargo=-1))


class TestEmbargoedSplit:
    def test_train_before_test_only(self) -> None:
        # Walk-forward: train is strictly earlier than test, with an embargo gap.
        dates = np.arange(100)
        for train_idx, test_idx in embargoed_split(dates, n_splits=4, embargo=5):
            if len(train_idx):
                assert train_idx.max() < test_idx.min()
                # embargo gap enforced
                assert test_idx.min() - train_idx.max() > 5

    def test_expanding_train_window(self) -> None:
        dates = np.arange(100)
        sizes = [len(tr) for tr, _ in embargoed_split(dates, n_splits=4, embargo=2)]
        # Expanding-origin: each successive train fold is at least as large.
        assert sizes == sorted(sizes)

    def test_no_lookahead(self) -> None:
        dates = np.arange(80)
        for train_idx, test_idx in embargoed_split(dates, n_splits=4, embargo=3):
            assert set(train_idx).isdisjoint(set(test_idx))
            if len(train_idx):
                assert max(train_idx) < min(test_idx)
