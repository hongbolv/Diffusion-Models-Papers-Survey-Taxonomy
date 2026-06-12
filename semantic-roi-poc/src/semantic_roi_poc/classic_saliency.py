"""Stage C comparison group: a classic, training-free saliency baseline.

We use the Spectral Residual method (Hou & Zhang, CVPR 2007), implemented in
pure NumPy. It is the fair "traditional saliency" comparison group required by
the PoC: it represents the conventional way an ROI map is obtained *without*
any generation-side information, so we can show that the cross-attention
scheme's advantage is "free + from generation intent" rather than a raw
bitrate win.
"""

from __future__ import annotations

import numpy as np

__all__ = ["spectral_residual_saliency"]


def _to_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    return arr


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur with reflect padding."""
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="reflect")
    csum = np.cumsum(xp, axis=0)
    xp = (csum[k - 1 :, :] - np.vstack([np.zeros((1, xp.shape[1])), csum[:-k, :]])) / k
    csum = np.cumsum(xp, axis=1)
    xp = (csum[:, k - 1 :] - np.hstack([np.zeros((xp.shape[0], 1)), csum[:, :-k]])) / k
    return xp[: x.shape[0], : x.shape[1]]


def spectral_residual_saliency(img: np.ndarray, smooth: int = 3) -> np.ndarray:
    """Compute a spectral-residual saliency map normalised to ``[0, 1]``.

    Parameters
    ----------
    img:
        ``(H, W)`` or ``(H, W, 3)`` image.
    smooth:
        Side length of the averaging filter applied to the log spectrum and to
        the final saliency map.
    """
    gray = _to_gray(img)
    f = np.fft.fft2(gray)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    avg = _box_blur(log_amp, smooth)
    spectral_residual = log_amp - avg
    recon = np.fft.ifft2(np.exp(spectral_residual + 1j * phase))
    sal = np.abs(recon) ** 2
    sal = _box_blur(sal, max(smooth, 3))
    lo, hi = float(sal.min()), float(sal.max())
    if hi - lo < 1e-12:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - lo) / (hi - lo)).astype(np.float32)
