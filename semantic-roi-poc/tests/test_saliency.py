"""Torch-free tests for Stage A helpers and model-family defaults.

These never load a real pipeline, so they run on any machine. The actual
Stable Diffusion / SDXL generation path needs torch + diffusers + a GPU and is
exercised separately.
"""

import numpy as np

from semantic_roi_poc.saliency import (
    DiffusersAttentionExtractor,
    SyntheticSaliency,
)


def test_synthetic_saliency_shapes_and_range():
    res = SyntheticSaliency(64, 80, seed=1).generate("subject")
    assert res.image.shape == (64, 80, 3)
    assert res.saliency.shape == (64, 80)
    assert res.saliency.min() >= 0.0 and res.saliency.max() <= 1.0
    mask = res.roi_mask(0.7)
    assert mask.shape == (64, 80)
    assert mask.any()  # some ROI pixels selected


def test_resolve_defaults_sd():
    ext = DiffusersAttentionExtractor()
    ext.is_sdxl = False
    ext._resolve_defaults()
    assert ext.height == 512 and ext.width == 512
    assert ext.layer_resolutions == (16, 32)


def test_resolve_defaults_sdxl():
    ext = DiffusersAttentionExtractor()
    ext.is_sdxl = True
    ext._resolve_defaults()
    assert ext.height == 1024 and ext.width == 1024
    assert ext.layer_resolutions == (32, 64)


def test_resolve_defaults_keeps_explicit_values():
    ext = DiffusersAttentionExtractor(
        height=768, width=768, layer_resolutions=(16,)
    )
    ext.is_sdxl = True
    ext._resolve_defaults()
    # Explicit values must not be overwritten by family defaults.
    assert ext.height == 768 and ext.width == 768
    assert ext.layer_resolutions == (16,)


def test_aggregate_selects_subject_tokens():
    # Two square attention maps (res=2 -> hw=4) over 5 tokens.
    res = 2
    p = np.zeros((res * res, 5), dtype=np.float32)
    # Subject token 1 lights up the top-left spatial cell only.
    p[0, 1] = 1.0
    agg = DiffusersAttentionExtractor._aggregate([(res, p)], token_idx=[1])
    assert agg.shape == (res, res)
    assert agg[0, 0] == agg.max() and agg[0, 0] > 0


def test_resize_preserves_corners():
    arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    out = DiffusersAttentionExtractor._resize(arr, 4, 4)
    assert out.shape == (4, 4)
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[-1, -1], 3.0)
