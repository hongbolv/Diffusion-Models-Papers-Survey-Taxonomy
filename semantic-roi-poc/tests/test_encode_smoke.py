import numpy as np
import pytest

from semantic_roi_poc import encode, qpmap
from semantic_roi_poc.saliency import SyntheticSaliency

pytestmark = pytest.mark.skipif(
    not encode.have_encoder("ffmpeg", "libx265"),
    reason="ffmpeg with libx265 not available",
)


def test_encode_decode_roundtrip(tmp_path):
    img = SyntheticSaliency(128, 128, seed=0).generate().image
    res = encode.encode_image(img, crf=28, workdir=tmp_path)
    assert res.bits > 0
    decoded = encode.decode_to_array(res.path)
    assert decoded.shape == img.shape


def test_lower_crf_costs_more_bits(tmp_path):
    img = SyntheticSaliency(128, 128, seed=1).generate().image
    hi_q = encode.encode_image(img, crf=20, workdir=tmp_path / "a")
    lo_q = encode.encode_image(img, crf=36, workdir=tmp_path / "b")
    assert hi_q.bits > lo_q.bits


def test_roi_offsets_change_bitstream(tmp_path):
    res = SyntheticSaliency(128, 128, seed=2).generate()
    qm = qpmap.saliency_to_qp_offsets(res.saliency, block_size=32, strength=10, max_offset=10)
    base = encode.encode_image(res.image, crf=28, workdir=tmp_path / "base")
    roi = encode.encode_image(res.image, crf=28, qmap=qm, workdir=tmp_path / "roi")
    # addroi in CRF mode must actually alter the encode.
    assert roi.bits != base.bits
