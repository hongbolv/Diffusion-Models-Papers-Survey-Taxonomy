import pytest

from semantic_roi_poc import encode, pipeline
from semantic_roi_poc.saliency import SyntheticSaliency

pytestmark = pytest.mark.skipif(
    not encode.have_encoder("ffmpeg", "libx265"),
    reason="ffmpeg with libx265 not available",
)


def test_run_image_produces_rd_curves(tmp_path):
    result = SyntheticSaliency(192, 192, seed=2).generate("synthetic subject")
    rep = pipeline.run_image(
        result,
        crfs=[24, 30, 36],
        block_size=32,
        workdir=tmp_path,
    )
    assert set(rep.schemes) == {"baseline", "attention", "classic"}
    for scheme in rep.schemes.values():
        assert len(scheme.points) == 3
        # Monotonic: higher CRF -> fewer bits.
        bits = [p.bits for p in scheme.points]
        assert bits == sorted(bits, reverse=True)


def test_attention_roi_improves_bd_rate(tmp_path):
    result = SyntheticSaliency(192, 192, seed=2).generate("synthetic subject")
    rep = pipeline.run_image(
        result,
        crfs=[24, 28, 32, 36],
        block_size=32,
        workdir=tmp_path,
    )
    # On the synthetic fixture the attention map equals the ground-truth ROI,
    # so ROI BD-rate vs uniform baseline must be negative (a saving).
    assert rep.bd_rate_roi("attention") < 0
