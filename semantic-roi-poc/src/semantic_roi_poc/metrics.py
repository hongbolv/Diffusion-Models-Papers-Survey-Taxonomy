"""Stage D (part 2): image-quality metrics, including ROI-masked variants.

The most important number in this PoC is the quality of the *region of
interest* at a given total bitrate, so every metric supports an optional
boolean/float mask that restricts the computation to the ROI.

PSNR and SSIM are implemented in pure NumPy (no SciPy/OpenCV dependency).
VMAF is optional: it is only used when the local ffmpeg build exposes the
``libvmaf`` filter (this environment ships ``vmafmotion`` only, so the code
gracefully reports VMAF as unavailable).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["psnr", "ssim", "masked_psnr", "masked_ssim", "vmaf_available", "vmaf"]

_MAX = 255.0


def _as_float_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        # Rec.601 luma.
        arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    return arr


def _resolve_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask, dtype=np.float64)
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape} != image shape {shape}")
    return m > 0.5


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB) on the luma channel."""
    a, b = _as_float_gray(ref), _as_float_gray(test)
    if a.shape != b.shape:
        raise ValueError("ref and test must have the same shape")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(_MAX * _MAX / mse))


def masked_psnr(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> float:
    """PSNR computed only over pixels where ``mask`` is truthy."""
    a, b = _as_float_gray(ref), _as_float_gray(test)
    m = _resolve_mask(mask, a.shape)
    if m is None or not m.any():
        raise ValueError("mask selects no pixels")
    mse = float(np.mean((a[m] - b[m]) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(_MAX * _MAX / mse))


def _ssim_map(a: np.ndarray, b: np.ndarray, win: int = 7) -> np.ndarray:
    """Per-pixel SSIM using a uniform sliding window (valid region)."""
    c1 = (0.01 * _MAX) ** 2
    c2 = (0.03 * _MAX) ** 2

    def box(x: np.ndarray) -> np.ndarray:
        # Separable uniform filter via cumulative sums (valid output).
        csum = np.cumsum(np.cumsum(x, axis=0), axis=1)
        csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
        s = (
            csum[win:, win:]
            - csum[:-win, win:]
            - csum[win:, :-win]
            + csum[:-win, :-win]
        )
        return s / (win * win)

    mu_a = box(a)
    mu_b = box(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a = box(a * a) - mu_a2
    sigma_b = box(b * b) - mu_b2
    sigma_ab = box(a * b) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a + sigma_b + c2)
    return num / den


def ssim(ref: np.ndarray, test: np.ndarray, win: int = 7) -> float:
    """Mean structural similarity over the whole image."""
    a, b = _as_float_gray(ref), _as_float_gray(test)
    if a.shape != b.shape:
        raise ValueError("ref and test must have the same shape")
    if min(a.shape) < win:
        raise ValueError("image smaller than SSIM window")
    return float(_ssim_map(a, b, win).mean())


def masked_ssim(ref: np.ndarray, test: np.ndarray, mask: np.ndarray, win: int = 7) -> float:
    """Mean SSIM restricted to the ROI.

    The SSIM map is ``win-1`` pixels smaller per axis (valid windows); the mask
    is cropped to match by trimming the border.
    """
    a, b = _as_float_gray(ref), _as_float_gray(test)
    m = _resolve_mask(mask, a.shape)
    if m is None or not m.any():
        raise ValueError("mask selects no pixels")
    smap = _ssim_map(a, b, win)
    off = win // 2
    m_crop = m[off : off + smap.shape[0], off : off + smap.shape[1]]
    if not m_crop.any():
        raise ValueError("mask selects no pixels within the valid SSIM region")
    return float(smap[m_crop].mean())


def vmaf_available(ffmpeg: str = "ffmpeg") -> bool:
    """Return True iff the ffmpeg build exposes the full ``libvmaf`` filter."""
    exe = shutil.which(ffmpeg)
    if exe is None:
        return False
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return False
    return any(line.split()[1] == "libvmaf" for line in out.splitlines() if len(line.split()) > 1)


def vmaf(reference: Path, distorted: Path, ffmpeg: str = "ffmpeg") -> float:
    """Compute VMAF of ``distorted`` against ``reference`` (raw or encoded).

    Requires an ffmpeg build with ``libvmaf``. Raises ``RuntimeError`` if VMAF
    is not available so callers can fall back to PSNR/SSIM.
    """
    if not vmaf_available(ffmpeg):
        raise RuntimeError("ffmpeg build has no libvmaf filter; use PSNR/SSIM instead")
    exe = shutil.which(ffmpeg)
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vmaf.json"
        cmd = [
            exe, "-hide_banner", "-i", str(distorted), "-i", str(reference),
            "-lavfi", f"libvmaf=log_fmt=json:log_path={log}",
            "-f", "null", "-",
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        data = json.loads(log.read_text())
    return float(data["pooled_metrics"]["vmaf"]["mean"])
