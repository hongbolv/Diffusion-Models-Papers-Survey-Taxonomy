"""Stage C: ROI-aware HEVC encoding through ffmpeg + libx265.

We encode a still image (treated as a 1-frame video) at a chosen base QP and,
optionally, apply a per-block QP-offset map via a chain of ``addroi`` filters
(one rectangle per block). ``addroi`` attaches a normalised quantiser offset to
each region that libx265 honours, giving us a spatial QP map purely through the
ffmpeg CLI -- no codec-IP modification and no private API.

Three encoding modes implement the PoC's comparison groups:

* ``"baseline"``  -- uniform QP, no ROI.
* a :class:`~semantic_roi_poc.qpmap.QPMap` -- attention- or classic-driven ROI.

The returned :class:`EncodeResult` records the byte size so callers can build
rate-distortion curves.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .qpmap import QPMap, offsets_to_qoffset

__all__ = ["EncodeResult", "encode_image", "decode_to_array", "have_encoder"]


@dataclass
class EncodeResult:
    """Outcome of one encode: the bitstream path, its size and the CRF used."""

    path: Path
    size_bytes: int
    crf: int

    @property
    def bits(self) -> int:
        return self.size_bytes * 8


def have_encoder(ffmpeg: str = "ffmpeg", encoder: str = "libx265") -> bool:
    exe = shutil.which(ffmpeg)
    if exe is None:
        return False
    out = subprocess.run(
        [exe, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    ).stdout
    return any(encoder in line for line in out.splitlines())


def _write_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    if image.ndim == 2:
        Image.fromarray(image.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def _addroi_chain(qmap: QPMap) -> str:
    """Build an ``addroi`` filtergraph applying one ROI per block."""
    qoff = offsets_to_qoffset(qmap.offsets)
    bs = qmap.block_size
    parts = []
    for r in range(qmap.rows):
        y = r * bs
        h = min(bs, qmap.height - y)
        for c in range(qmap.cols):
            x = c * bs
            w = min(bs, qmap.width - x)
            q = float(qoff[r, c])
            if abs(q) < 1e-6:
                continue
            # qoffset is a rational; format with enough precision.
            parts.append(f"addroi=x={x}:y={y}:w={w}:h={h}:qoffset={q:.6f}")
    return ",".join(parts)


def encode_image(
    image: np.ndarray,
    crf: int,
    *,
    qmap: QPMap | None = None,
    workdir: Path | None = None,
    ffmpeg: str = "ffmpeg",
    encoder: str = "libx265",
) -> EncodeResult:
    """Encode ``image`` at the given ``crf``; apply ``qmap`` ROI offsets if given.

    CRF (constant rate factor) mode is used deliberately: libx265/libx264 only
    honour ``addroi`` quantiser offsets in rate-controlled modes, **not** in
    constant-QP mode. Sweeping ``crf`` produces the rate-distortion points used
    to build BD-rate curves.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` uint8 RGB (or ``(H, W)`` grayscale) image.
    crf:
        Constant rate factor (lower = higher quality / more bits).
    qmap:
        Optional per-block QP-offset map (Stage B output). ``None`` => baseline.
    workdir:
        Directory for intermediate files; a temp dir is used if omitted.
    """
    if not have_encoder(ffmpeg, encoder):
        raise RuntimeError(f"{encoder} not available in {ffmpeg}")

    workdir = Path(workdir or tempfile.mkdtemp(prefix="roi_enc_"))
    workdir.mkdir(parents=True, exist_ok=True)

    src = workdir / "src.png"
    _write_png(image, src)
    out = workdir / f"crf{crf}{'_roi' if qmap is not None else ''}.hevc"

    vf = "format=yuv420p"
    if qmap is not None:
        chain = _addroi_chain(qmap)
        if chain:
            vf = f"{chain},{vf}"

    exe = shutil.which(ffmpeg)
    container = "hevc" if encoder == "libx265" else "h264"
    params_flag = "-x265-params" if encoder == "libx265" else "-x264-params"
    cmd = [
        exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-c:v", encoder,
        "-crf", str(crf),
        params_flag, "log-level=none",
        "-frames:v", "1",
        "-f", container,
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return EncodeResult(path=out, size_bytes=out.stat().st_size, crf=crf)


def decode_to_array(bitstream: Path, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Decode an encoded bitstream back to an ``(H, W, 3)`` uint8 RGB array."""
    exe = shutil.which(ffmpeg)
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "dec.png"
        cmd = [
            exe, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(bitstream), "-frames:v", "1", str(png),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        from PIL import Image

        return np.asarray(Image.open(png).convert("RGB"))
