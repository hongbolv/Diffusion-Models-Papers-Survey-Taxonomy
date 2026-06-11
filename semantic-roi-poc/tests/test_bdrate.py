import numpy as np
import pytest

from semantic_roi_poc import bdrate


def test_bd_rate_zero_for_identical_curves():
    rate = np.array([100.0, 200.0, 400.0, 800.0])
    quality = np.array([30.0, 34.0, 38.0, 42.0])
    assert bdrate.bd_rate(rate, quality, rate, quality) == pytest.approx(0.0, abs=1e-6)


def test_bd_rate_negative_when_test_cheaper():
    # Test needs 0.8x the bitrate of ref at every quality -> ~-20% BD-rate.
    rate_ref = np.array([100.0, 200.0, 400.0, 800.0])
    quality = np.array([30.0, 34.0, 38.0, 42.0])
    rate_test = rate_ref * 0.8
    bd = bdrate.bd_rate(rate_ref, quality, rate_test, quality)
    assert bd == pytest.approx(-20.0, abs=0.5)


def test_bd_rate_positive_when_test_costlier():
    rate_ref = np.array([100.0, 200.0, 400.0, 800.0])
    quality = np.array([30.0, 34.0, 38.0, 42.0])
    rate_test = rate_ref * 1.25
    assert bdrate.bd_rate(rate_ref, quality, rate_test, quality) > 0


def test_bd_quality_positive_when_test_better():
    rate = np.array([100.0, 200.0, 400.0, 800.0])
    quality_ref = np.array([30.0, 34.0, 38.0, 42.0])
    quality_test = quality_ref + 1.5
    assert bdrate.bd_quality(rate, quality_ref, rate, quality_test) == pytest.approx(1.5, abs=0.1)


def test_non_overlapping_quality_raises():
    rate = np.array([100.0, 200.0, 400.0, 800.0])
    q_ref = np.array([30.0, 31.0, 32.0, 33.0])
    q_test = np.array([40.0, 41.0, 42.0, 43.0])
    with pytest.raises(ValueError):
        bdrate.bd_rate(rate, q_ref, rate, q_test)
