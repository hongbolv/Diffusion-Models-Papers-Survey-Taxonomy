"""Stage B: convert a saliency map into a zero-mean per-block QP-offset map.

The key fairness requirement of the PoC is that the ROI scheme must *not* spend
extra bitrate overall: it should only *move* bits from the background to the
subject. We achieve this by making the per-block QP offsets approximately
zero-mean (weighted by block area), so the average quantiser is unchanged and
only its spatial distribution follows the saliency map.

All functions here are pure NumPy and free of any I/O, so they are fully unit
testable without a GPU or an encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["QPMap", "pool_to_blocks", "saliency_to_qp_offsets", "offsets_to_qoffset"]


@dataclass
class QPMap:
    """A per-block QP-offset map.

    Attributes
    ----------
    offsets:
        ``(rows, cols)`` array of integer QP offsets per block. Negative means
        *more* quality (lower QP), positive means *less* quality (higher QP).
    block_size:
        Side length in pixels of each (square) block, e.g. the codec CTU size.
    height, width:
        The pixel dimensions of the image the map refers to.
    """

    offsets: np.ndarray
    block_size: int
    height: int
    width: int

    @property
    def rows(self) -> int:
        return int(self.offsets.shape[0])

    @property
    def cols(self) -> int:
        return int(self.offsets.shape[1])

    def area_weighted_mean(self) -> float:
        """Mean QP offset weighted by the pixel area each block covers.

        Edge blocks may be smaller than ``block_size`` when the image is not an
        exact multiple of it; weighting by real covered area gives the honest
        average quantiser shift.
        """
        weights = self._block_areas()
        return float(np.sum(self.offsets * weights) / np.sum(weights))

    def _block_areas(self) -> np.ndarray:
        row_px = _segment_sizes(self.height, self.block_size, self.rows)
        col_px = _segment_sizes(self.width, self.block_size, self.cols)
        return np.outer(row_px, col_px).astype(np.float64)


def _segment_sizes(total: int, block: int, count: int) -> np.ndarray:
    """Pixel extent covered by each block index along one axis."""
    sizes = np.full(count, block, dtype=np.float64)
    covered = block * count
    if covered > total:
        sizes[-1] = total - block * (count - 1)
    return sizes


def pool_to_blocks(saliency: np.ndarray, block_size: int, *, reduce: str = "mean") -> np.ndarray:
    """Pool a pixel-resolution saliency map down to a per-block grid.

    Parameters
    ----------
    saliency:
        ``(H, W)`` float array. Values need not be normalised.
    block_size:
        Side length of each pooling block in pixels.
    reduce:
        ``"mean"`` (default) or ``"max"`` pooling within each block.
    """
    if saliency.ndim != 2:
        raise ValueError(f"saliency must be 2-D, got shape {saliency.shape}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    h, w = saliency.shape
    rows = (h + block_size - 1) // block_size
    cols = (w + block_size - 1) // block_size
    out = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        r0, r1 = r * block_size, min((r + 1) * block_size, h)
        for c in range(cols):
            c0, c1 = c * block_size, min((c + 1) * block_size, w)
            patch = saliency[r0:r1, c0:c1]
            if patch.size == 0:
                continue
            out[r, c] = patch.max() if reduce == "max" else patch.mean()
    return out


def saliency_to_qp_offsets(
    saliency: np.ndarray,
    block_size: int,
    *,
    strength: float = 6.0,
    max_offset: int = 8,
    reduce: str = "mean",
    zero_mean: bool = True,
) -> QPMap:
    """Map a saliency map to integer per-block QP offsets.

    The mapping is deliberately simple (linear), per the PoC plan: salient
    blocks receive a *negative* offset (more bits / higher quality) and dull
    blocks a *positive* offset. The result is then re-centred so its
    area-weighted mean is ~0, keeping the average quantiser -- and therefore
    the bitrate -- comparable to the uniform baseline.

    Parameters
    ----------
    saliency:
        ``(H, W)`` saliency map.
    block_size:
        Block side length in pixels (use the codec CTU size, e.g. 64).
    strength:
        Linear gain (in QP units) applied to centred, normalised saliency.
    max_offset:
        Clamp on the absolute QP offset per block.
    reduce:
        Pooling mode forwarded to :func:`pool_to_blocks`.
    zero_mean:
        If True (default) re-centre offsets to be area-weighted zero-mean so
        the comparison against the uniform baseline is bitrate-fair.
    """
    if max_offset < 0:
        raise ValueError("max_offset must be non-negative")

    h, w = saliency.shape
    blocks = pool_to_blocks(saliency, block_size, reduce=reduce)

    # Normalise the pooled saliency to [0, 1] robustly.
    lo, hi = float(blocks.min()), float(blocks.max())
    if hi - lo < 1e-12:
        norm = np.zeros_like(blocks)
    else:
        norm = (blocks - lo) / (hi - lo)

    # Centre on 0.5 so the mean salient block is neutral; salient -> negative QP.
    offsets = -strength * (norm - 0.5)

    qmap = QPMap(
        offsets=np.zeros_like(offsets, dtype=np.int64),
        block_size=block_size,
        height=h,
        width=w,
    )

    if zero_mean:
        weights = qmap._block_areas()
        weighted_mean = np.sum(offsets * weights) / np.sum(weights)
        offsets = offsets - weighted_mean

    offsets = np.clip(np.rint(offsets), -max_offset, max_offset).astype(np.int64)
    qmap.offsets = offsets
    return qmap


def offsets_to_qoffset(qp_offsets: np.ndarray, qp_range: int = 51) -> np.ndarray:
    """Convert integer QP offsets to ffmpeg ``addroi`` normalised qoffsets.

    The ``addroi`` filter takes a quantisation offset in ``[-1, 1]`` that the
    encoder scales by the codec QP range, using the *same* sign convention as a
    QP delta: a **negative** qoffset lowers the local QP and therefore raises
    quality. Our QP offsets follow the same convention (salient block ->
    negative offset -> higher quality), so the mapping is a simple scaling by
    the QP range with **no sign flip**.
    """
    return np.clip(qp_offsets / float(qp_range), -1.0, 1.0)
