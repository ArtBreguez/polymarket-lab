"""Tests for modeling.calibration."""

import numpy as np

from pmlab.modeling.calibration import IsotonicCalibrator


def test_transform_clips_01():
    cal = IsotonicCalibrator()
    probs = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 1, 1])
    cal.fit(probs, labels)

    # Even extreme inputs should be clipped
    result = cal.transform(np.array([-0.5, 1.5]))
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_save_load_roundtrip(tmp_path):
    cal = IsotonicCalibrator()
    probs = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    cal.fit(probs, labels)

    path = tmp_path / "cal.pkl"
    cal.save(path)

    loaded = IsotonicCalibrator.load(path)
    test_probs = np.array([0.2, 0.5, 0.8])
    orig_out = cal.transform(test_probs)
    loaded_out = loaded.transform(test_probs)
    np.testing.assert_allclose(orig_out, loaded_out, atol=1e-8)
