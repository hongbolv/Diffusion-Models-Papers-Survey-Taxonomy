# Semantic ROI Coding — Minimal PoC

A minimal, reproducible proof-of-concept for **generation-aware encoding**: use a
diffusion model's **cross-attention saliency** to drive **per-block QP allocation**
in a video/image codec, so the *subject* (ROI) keeps higher quality at the **same
bitrate** as uniform encoding — at **zero extra compute**, because the saliency map
is a by-product of generation.

The deliverable is a single number per image and a curve: **BD-rate of ROI quality
vs. a uniform-QP baseline**, plus a control against a classic (spectral-residual)
saliency detector.

> This PoC deliberately stays in the **image / software-encoder** scope to validate
> the *direction* (attention → QP → ROI quality gain). Video temporal consistency,
> hardware codec IP, GOP/I-frame control and generation acceleration are explicitly
> **out of scope** (see §"Scope").

## Hypothesis under test

> *Allocating QP from a diffusion model's cross-attention saliency map yields better
> ROI quality at equal bitrate than uniform encoding, on par with classic saliency
> ROI — but for free (a generation by-product) and aligned with generative intent.*

**GO** if, at equal bitrate, ROI BD-rate vs. baseline is a clear saving (e.g. 5–15%)
with no severe full-frame degradation. **NO-GO / revisit** if the attention map is
inaccurate or the gain is ~0.

## Pipeline stages

| Stage | Module | What it does |
|-------|--------|--------------|
| A | `saliency.py` | Generate image + saliency map. `SyntheticSaliency` (no deps, a known-ROI fixture) or `DiffusersAttentionExtractor` (real Stable Diffusion / SDXL cross-attention; model family auto-detected). |
| B | `qpmap.py` | Pool saliency to codec blocks → **zero-mean** per-block QP offsets (move bits, don't add them). |
| C | `encode.py`, `classic_saliency.py` | Encode baseline / attention-ROI / classic-ROI with ffmpeg `addroi` in **CRF mode**; spectral-residual control. |
| D | `metrics.py`, `bdrate.py` | ROI-masked & full PSNR/SSIM (VMAF if `libvmaf` present), Bjøntegaard BD-rate. |
| — | `pipeline.py`, `cli.py` | Orchestrate A→D, sweep CRF, emit a JSON report. |

## Install

```bash
cd semantic-roi-poc
pip install -e .                 # core: numpy, pillow
# Stage A with a real model (GPU recommended):
pip install -e ".[diffusion]"    # torch, diffusers, transformers, accelerate
```

External tool: **ffmpeg built with libx265** (or libx264). On Debian/Ubuntu:
`apt-get install ffmpeg x265`. Full VMAF additionally needs an ffmpeg with the
`libvmaf` filter; otherwise the PoC falls back to PSNR/SSIM automatically.

## Run

Synthetic smoke run (no model/GPU needed — validates the encode/measure plumbing):

```bash
python -m semantic_roi_poc.cli --synthetic --num-synthetic 3 \
    --crfs 24 28 32 36 --out roi_report.json
```

Real Stable Diffusion attention:

```bash
python -m semantic_roi_poc.cli --prompts examples/prompts.jsonl \
    --model runwayml/stable-diffusion-v1-5 --device cuda \
    --crfs 22 26 30 34 --out roi_report.json
```

SDXL (the mainstream, higher-quality model — more convincing evidence):

```bash
python -m semantic_roi_poc.cli --prompts examples/prompts.jsonl \
    --model stabilityai/stable-diffusion-xl-base-1.0 --device cuda \
    --crfs 22 26 30 34 --out roi_report.json
```

The model family (classic Stable Diffusion 1.5/2.1 vs SDXL) is auto-detected, and
generation size and the aggregated cross-attention resolutions default
accordingly (512 / `(16,32)` for SD, 1024 / `(32,64)` for SDXL). Override with
`--height/--width` and `--layer-resolutions` if needed. SDXL needs more VRAM
(≈12–16 GB recommended).

The report's key field is `bd_rate_roi.attention` (negative = bitrate saving at
equal ROI quality).

> **Note on the synthetic fixture:** in `--synthetic` mode the "attention" map *is*
> the ground-truth ROI, so it beats baseline (and the classic control) by
> construction. Synthetic mode validates the *plumbing*, not real-world superiority —
> use real prompts for evidence.

## Test

```bash
pip install -e ".[test]"
pytest
```

Pure-logic tests (qpmap, bdrate, metrics, classic saliency) always run; encoder and
pipeline tests auto-skip when ffmpeg/libx265 is absent.

## Scope

In scope: single images, software encoder, "does attention-driven QP help ROI at
equal rate?". Out of scope (separate PoCs): video temporal smoothing, hardware codec
IP interfaces, prompt/latent-change-driven GOP/I-frame placement, generation
acceleration, real-time throughput.
