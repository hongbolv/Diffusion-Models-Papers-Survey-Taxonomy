"""Stage D (part 1): Bjontegaard-Delta rate / PSNR computation.

BD-rate summarises the average bitrate difference between two encoders over a
shared quality range (negative = the test codec saves bitrate at equal
quality). BD-PSNR is the dual (average quality gain at equal bitrate).

This is the standard Bjontegaard metric using a piecewise-cubic fit of
quality vs. log10(bitrate) and integration over the overlapping quality
interval. Pure NumPy, fully unit testable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bd_rate", "bd_quality"]


def _check(rate: np.ndarray, quality: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    rate = np.asarray(rate, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if rate.shape != quality.shape:
        raise ValueError(f"{name}: rate and quality must have equal shape")
    if rate.size < 4:
        raise ValueError(f"{name}: need at least 4 rate-distortion points, got {rate.size}")
    if np.any(rate <= 0):
        raise ValueError(f"{name}: bitrates must be positive")
    order = np.argsort(quality)
    return rate[order], quality[order]


def _cubic_poly_int(coeffs: np.ndarray, lo: float, hi: float) -> float:
    """Definite integral of a cubic given numpy polyfit coefficients."""
    p_int = np.polyint(coeffs)
    return float(np.polyval(p_int, hi) - np.polyval(p_int, lo))


def bd_rate(
    rate_ref: np.ndarray,
    quality_ref: np.ndarray,
    rate_test: np.ndarray,
    quality_test: np.ndarray,
) -> float:
    """Average bitrate difference (%) of *test* vs *ref* over equal quality.

    A negative value means the test configuration needs *less* bitrate than the
    reference at the same quality (i.e. the test is better).
    """
    r_ref, q_ref = _check(rate_ref, quality_ref, "ref")
    r_test, q_test = _check(rate_test, quality_test, "test")

    lr_ref = np.log10(r_ref)
    lr_test = np.log10(r_test)

    # Fit log10(rate) as a cubic function of quality.
    p_ref = np.polyfit(q_ref, lr_ref, 3)
    p_test = np.polyfit(q_test, lr_test, 3)

    lo = max(q_ref.min(), q_test.min())
    hi = min(q_ref.max(), q_test.max())
    if hi <= lo:
        raise ValueError("ref and test quality ranges do not overlap")

    int_ref = _cubic_poly_int(p_ref, lo, hi)
    int_test = _cubic_poly_int(p_test, lo, hi)

    avg_diff = (int_test - int_ref) / (hi - lo)
    return float((10.0 ** avg_diff - 1.0) * 100.0)


def bd_quality(
    rate_ref: np.ndarray,
    quality_ref: np.ndarray,
    rate_test: np.ndarray,
    quality_test: np.ndarray,
) -> float:
    """Average quality difference (test - ref) over the shared bitrate range.

    Positive means the test configuration yields higher quality at equal
    bitrate.
    """
    r_ref, q_ref = _check(rate_ref, quality_ref, "ref")
    r_test, q_test = _check(rate_test, quality_test, "test")

    lr_ref = np.log10(r_ref)
    lr_test = np.log10(r_test)

    p_ref = np.polyfit(lr_ref, q_ref, 3)
    p_test = np.polyfit(lr_test, q_test, 3)

    lo = max(lr_ref.min(), lr_test.min())
    hi = min(lr_ref.max(), lr_test.max())
    if hi <= lo:
        raise ValueError("ref and test bitrate ranges do not overlap")

    int_ref = _cubic_poly_int(p_ref, lo, hi)
    int_test = _cubic_poly_int(p_test, lo, hi)
    return float((int_test - int_ref) / (hi - lo))
