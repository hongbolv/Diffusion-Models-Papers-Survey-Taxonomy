import numpy as np

from semantic_roi_poc import classic_saliency


def test_output_shape_and_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    sal = classic_saliency.spectral_residual_saliency(img)
    assert sal.shape == (96, 96)
    assert sal.dtype == np.float32
    assert sal.min() >= 0.0
    assert sal.max() <= 1.0


def test_grayscale_input_supported():
    img = np.zeros((48, 48), dtype=np.uint8)
    sal = classic_saliency.spectral_residual_saliency(img)
    assert sal.shape == (48, 48)
    assert sal.dtype == np.float32
    assert sal.min() >= 0.0 and sal.max() <= 1.0


def test_highlights_an_isolated_object():
    img = np.full((96, 96), 40, dtype=np.uint8)
    img[40:56, 40:56] = 220  # a bright square on a flat field
    sal = classic_saliency.spectral_residual_saliency(img)
    # Spectral residual responds to the object's edges: the region enclosing
    # the square (including its border) is more salient than a far flat corner.
    obj_region = sal[34:62, 34:62].mean()
    corner = sal[:16, :16].mean()
    assert obj_region > corner
