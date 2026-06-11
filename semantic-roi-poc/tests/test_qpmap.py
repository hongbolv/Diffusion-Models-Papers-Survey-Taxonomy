import numpy as np
import pytest

from semantic_roi_poc import qpmap


def _blob(h=256, w=256, cy=128, cx=128, sy=40, sx=40):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))


def test_pool_to_blocks_shape_and_values():
    sal = np.arange(64, dtype=float).reshape(8, 8)
    pooled = qpmap.pool_to_blocks(sal, block_size=4)
    assert pooled.shape == (2, 2)
    # top-left block is mean of the 4x4 sub-grid
    assert pooled[0, 0] == pytest.approx(sal[0:4, 0:4].mean())


def test_pool_handles_non_multiple_dimensions():
    sal = np.ones((10, 7))
    pooled = qpmap.pool_to_blocks(sal, block_size=4)
    assert pooled.shape == (3, 2)  # ceil(10/4)=3, ceil(7/4)=2
    assert np.allclose(pooled, 1.0)


def test_offsets_are_zero_mean_area_weighted():
    sal = _blob()
    qm = qpmap.saliency_to_qp_offsets(sal, block_size=64, strength=8, max_offset=8)
    # After rounding/clamping the area-weighted mean should be near zero.
    assert abs(qm.area_weighted_mean()) <= 1.0


def test_salient_region_gets_negative_offset():
    sal = _blob()
    qm = qpmap.saliency_to_qp_offsets(sal, block_size=64, strength=8, max_offset=8)
    # Centre block (most salient) must be higher quality (negative offset)
    # than a corner block (background, positive offset).
    centre = qm.offsets[qm.rows // 2, qm.cols // 2]
    corner = qm.offsets[0, 0]
    assert centre < 0
    assert corner > 0
    assert centre < corner


def test_offsets_respect_max_offset():
    sal = _blob()
    qm = qpmap.saliency_to_qp_offsets(sal, block_size=32, strength=100, max_offset=5)
    assert qm.offsets.min() >= -5
    assert qm.offsets.max() <= 5


def test_flat_saliency_yields_zero_offsets():
    sal = np.full((128, 128), 0.5)
    qm = qpmap.saliency_to_qp_offsets(sal, block_size=64, strength=8)
    assert np.all(qm.offsets == 0)


def test_qoffset_sign_matches_qp_convention():
    # Negative QP offset (higher quality) -> negative addroi qoffset.
    offs = np.array([[-8, 0, 8]])
    qo = qpmap.offsets_to_qoffset(offs, qp_range=51)
    assert qo[0, 0] < 0 < qo[0, 2]
    assert qo[0, 1] == pytest.approx(0.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        qpmap.pool_to_blocks(np.zeros((4, 4, 3)), 2)
    with pytest.raises(ValueError):
        qpmap.pool_to_blocks(np.zeros((4, 4)), 0)
    with pytest.raises(ValueError):
        qpmap.saliency_to_qp_offsets(np.zeros((8, 8)), 4, max_offset=-1)
