"""Command-line entry point for the semantic ROI coding PoC.

Examples
--------
Smoke-test the whole encode/measure pipeline with the synthetic generator
(no GPU, no model weights needed)::

    python -m semantic_roi_poc.cli --synthetic --out report.json

Run with real Stable Diffusion cross-attention (needs torch+diffusers+GPU)::

    python -m semantic_roi_poc.cli \
        --prompts examples/prompts.jsonl \
        --model runwayml/stable-diffusion-v1-5 \
        --device cuda --out report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def _load_prompts(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _report_to_dict(rep) -> dict:
    return {
        "prompt": rep.prompt,
        "roi_coverage": rep.roi_coverage,
        "schemes": {
            name: [asdict(p) for p in s.points] for name, s in rep.schemes.items()
        },
        "bd_rate_roi": {
            name: rep.bd_rate_roi(name)
            for name in rep.schemes
            if name != "baseline"
        },
        "bd_rate_full": {
            name: rep.bd_rate_full(name)
            for name in rep.schemes
            if name != "baseline"
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic ROI coding PoC")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use the synthetic fixture generator")
    src.add_argument("--prompts", type=Path, help="JSONL file with prompt rows")

    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    p.add_argument("--crfs", type=int, nargs="+", default=[22, 26, 30, 34])
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--strength", type=float, default=8.0)
    p.add_argument("--max-offset", type=int, default=8)
    p.add_argument("--roi-quantile", type=float, default=0.7)
    p.add_argument("--encoder", default="libx265", choices=["libx265", "libx264"])
    p.add_argument("--no-classic", action="store_true", help="skip the classic-saliency control")
    p.add_argument("--num-synthetic", type=int, default=3, help="images for --synthetic mode")

    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("roi_report.json"))
    return p


def _iter_results(args):
    from .saliency import DiffusersAttentionExtractor, SyntheticSaliency

    if args.synthetic:
        for i in range(args.num_synthetic):
            yield SyntheticSaliency(args.height, args.width, seed=i).generate(
                f"synthetic-{i}"
            )
    else:
        extractor = DiffusersAttentionExtractor(
            model_id=args.model,
            device=args.device,
            steps=args.steps,
            height=args.height,
            width=args.width,
        )
        for row in _load_prompts(args.prompts):
            prompt = row["prompt"] if isinstance(row, dict) else str(row)
            subject = row.get("subject") if isinstance(row, dict) else None
            yield extractor.generate(prompt, subject=subject)


def main(argv: list[str] | None = None) -> int:
    from . import pipeline

    args = build_parser().parse_args(argv)
    reports = []
    for idx, result in enumerate(_iter_results(args)):
        wd = (args.workdir / f"img{idx}") if args.workdir else None
        rep = pipeline.run_image(
            result,
            crfs=args.crfs,
            block_size=args.block_size,
            strength=args.strength,
            max_offset=args.max_offset,
            roi_quantile=args.roi_quantile,
            workdir=wd,
            encoder=args.encoder,
            include_classic=not args.no_classic,
        )
        d = _report_to_dict(rep)
        reports.append(d)
        roi_bd = d["bd_rate_roi"].get("attention")
        print(f"[{idx}] prompt={rep.prompt!r} ROI BD-rate(attention vs baseline)={roi_bd:.2f}%")

    summary = {
        "num_images": len(reports),
        "mean_bd_rate_roi_attention": (
            sum(r["bd_rate_roi"]["attention"] for r in reports) / len(reports)
            if reports
            else None
        ),
        "reports": reports,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}  (mean ROI BD-rate attention = "
          f"{summary['mean_bd_rate_roi_attention']:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
