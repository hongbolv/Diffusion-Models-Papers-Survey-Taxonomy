import numpy as np
import pytest

from semantic_roi_poc import metrics


def _imgs():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    noisy = np.clip(
        ref.astype(np.int16) + rng.integers(-10, 10, ref.shape), 0, 255
    ).astype(np.uint8)
    return ref, noisy


def test_psnr_identical_is_inf():
    ref, _ = _imgs()
    assert metrics.psnr(ref, ref) == float("inf")


def test_psnr_decreases_with_noise():
    ref, noisy = _imgs()
    rng = np.random.default_rng(1)
    noisier = np.clip(
        ref.astype(np.int16) + rng.integers(-40, 40, ref.shape), 0, 255
    ).astype(np.uint8)
    assert metrics.psnr(ref, noisy) > metrics.psnr(ref, noisier)


def test_ssim_identical_is_one():
    ref, _ = _imgs()
    assert metrics.ssim(ref, ref) == pytest.approx(1.0, abs=1e-6)


def test_ssim_in_range():
    ref, noisy = _imgs()
    val = metrics.ssim(ref, noisy)
    assert -1.0 <= val <= 1.0


def test_masked_psnr_matches_full_for_full_mask():
    ref, noisy = _imgs()
    full_mask = np.ones(ref.shape[:2], dtype=bool)
    assert metrics.masked_psnr(ref, noisy, full_mask) == pytest.approx(
        metrics.psnr(ref, noisy), rel=1e-9
    )


def test_masked_psnr_focuses_on_region():
    ref, _ = _imgs()
    test = ref.copy()
    # Corrupt only the top-left quadrant.
    test[:32, :32] = 0
    mask_corrupt = np.zeros(ref.shape[:2], dtype=bool)
    mask_corrupt[:32, :32] = True
    mask_clean = ~mask_corrupt
    assert metrics.masked_psnr(ref, test, mask_clean) == float("inf")
    assert metrics.masked_psnr(ref, test, mask_corrupt) < float("inf")


def test_mask_shape_mismatch_raises():
    ref, noisy = _imgs()
    with pytest.raises(ValueError):
        metrics.masked_psnr(ref, noisy, np.ones((10, 10), dtype=bool))


def test_empty_mask_raises():
    ref, noisy = _imgs()
    with pytest.raises(ValueError):
        metrics.masked_psnr(ref, noisy, np.zeros(ref.shape[:2], dtype=bool))
