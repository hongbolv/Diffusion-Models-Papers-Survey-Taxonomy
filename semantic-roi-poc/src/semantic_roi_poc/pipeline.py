"""Pipeline: orchestrate stages A -> B -> C -> D for one image.

For a given image + saliency map we encode three comparison groups over a CRF
sweep, decode each, and measure ROI-masked and full-frame quality. We then
summarise with BD-rate on the ROI quality (the metric that matters): a negative
BD-rate means the scheme reaches the same ROI quality at a lower bitrate than
the uniform baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bdrate, classic_saliency, encode, metrics
from .qpmap import saliency_to_qp_offsets
from .saliency import SaliencyResult

__all__ = ["RDPoint", "SchemeResult", "PipelineReport", "run_image"]


@dataclass
class RDPoint:
    crf: int
    bits: int
    roi_psnr: float
    full_psnr: float
    roi_ssim: float
    full_ssim: float
    vmaf: float | None = None


@dataclass
class SchemeResult:
    name: str
    points: list[RDPoint] = field(default_factory=list)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bits = np.array([p.bits for p in self.points], dtype=np.float64)
        roi = np.array([p.roi_psnr for p in self.points], dtype=np.float64)
        full = np.array([p.full_psnr for p in self.points], dtype=np.float64)
        return bits, roi, full


@dataclass
class PipelineReport:
    prompt: str
    schemes: dict[str, SchemeResult]
    roi_coverage: float

    def bd_rate_roi(self, test: str, ref: str = "baseline") -> float:
        rb, rroi, _ = self.schemes[ref].arrays()
        tb, troi, _ = self.schemes[test].arrays()
        return bdrate.bd_rate(rb, rroi, tb, troi)

    def bd_rate_full(self, test: str, ref: str = "baseline") -> float:
        rb, _, rfull = self.schemes[ref].arrays()
        tb, _, tfull = self.schemes[test].arrays()
        return bdrate.bd_rate(rb, rfull, tb, tfull)


def _measure(
    ref_img: np.ndarray,
    decoded: np.ndarray,
    mask: np.ndarray,
    ref_path: Path | None,
    dec_path: Path | None,
    use_vmaf: bool,
) -> tuple[float, float, float, float, float | None]:
    roi_psnr = metrics.masked_psnr(ref_img, decoded, mask)
    full_psnr = metrics.psnr(ref_img, decoded)
    roi_ssim = metrics.masked_ssim(ref_img, decoded, mask)
    full_ssim = metrics.ssim(ref_img, decoded)
    vmaf_val = None
    if use_vmaf and ref_path is not None and dec_path is not None:
        try:
            vmaf_val = metrics.vmaf(ref_path, dec_path)
        except (RuntimeError, OSError):
            vmaf_val = None
    return roi_psnr, full_psnr, roi_ssim, full_ssim, vmaf_val


def run_image(
    result: SaliencyResult,
    *,
    crfs: list[int] | None = None,
    block_size: int = 64,
    strength: float = 8.0,
    max_offset: int = 8,
    roi_quantile: float = 0.7,
    workdir: Path | None = None,
    encoder: str = "libx265",
    include_classic: bool = True,
) -> PipelineReport:
    """Run the full comparison for one image and return a report.

    Schemes: ``baseline`` (uniform), ``attention`` (saliency-driven ROI) and,
    optionally, ``classic`` (spectral-residual saliency ROI).
    """
    crfs = crfs or [22, 26, 30, 34]
    image = result.image
    mask = result.roi_mask(roi_quantile)
    use_vmaf = metrics.vmaf_available()

    attn_qmap = saliency_to_qp_offsets(
        result.saliency, block_size, strength=strength, max_offset=max_offset
    )

    schemes: dict[str, SchemeResult] = {
        "baseline": SchemeResult("baseline"),
        "attention": SchemeResult("attention"),
    }
    qmaps = {"baseline": None, "attention": attn_qmap}

    if include_classic:
        classic_sal = classic_saliency.spectral_residual_saliency(image)
        schemes["classic"] = SchemeResult("classic")
        qmaps["classic"] = saliency_to_qp_offsets(
            classic_sal, block_size, strength=strength, max_offset=max_offset
        )

    base = Path(workdir) if workdir else None
    for name, qmap in qmaps.items():
        sub = (base / name) if base else None
        for crf in crfs:
            enc = encode.encode_image(image, crf, qmap=qmap, workdir=sub, encoder=encoder)
            decoded = encode.decode_to_array(enc.path)
            roi_psnr, full_psnr, roi_ssim, full_ssim, vmaf_val = _measure(
                image, decoded, mask, None, None, use_vmaf
            )
            schemes[name].points.append(
                RDPoint(
                    crf=crf,
                    bits=enc.bits,
                    roi_psnr=roi_psnr,
                    full_psnr=full_psnr,
                    roi_ssim=roi_ssim,
                    full_ssim=full_ssim,
                    vmaf=vmaf_val,
                )
            )

    return PipelineReport(
        prompt=result.prompt,
        schemes=schemes,
        roi_coverage=float(mask.mean()),
    )
