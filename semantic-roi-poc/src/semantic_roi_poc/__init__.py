"""Semantic ROI Coding PoC.

Validate the hypothesis that a diffusion model's cross-attention saliency map
can drive a per-block QP allocation so that, at an equal total bitrate, the
subject (region of interest) is encoded at higher quality than uniform
encoding -- and at least on par with a classic saliency baseline, while the
saliency map itself is a *free* by-product of generation.

The package is organised by the PoC stages:

* Stage A -- :mod:`semantic_roi_poc.saliency`         (generate + extract attention saliency)
* Stage B -- :mod:`semantic_roi_poc.qpmap`            (saliency -> zero-mean per-block QP offsets)
* Stage C -- :mod:`semantic_roi_poc.encode`           (ROI-aware HEVC encoding via ffmpeg/libx265)
              :mod:`semantic_roi_poc.classic_saliency` (spectral-residual control baseline)
* Stage D -- :mod:`semantic_roi_poc.metrics`          (PSNR/SSIM, ROI-masked, optional VMAF)
              :mod:`semantic_roi_poc.bdrate`           (Bjøntegaard BD-rate)

The orchestration lives in :mod:`semantic_roi_poc.pipeline` and the command
line entry point in :mod:`semantic_roi_poc.cli`.
"""

__all__ = [
    "qpmap",
    "bdrate",
    "metrics",
    "saliency",
    "classic_saliency",
    "encode",
    "pipeline",
]

__version__ = "0.1.0"
