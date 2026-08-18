"""
psifx_eval/run_sam3_baseline.py
==================================
Standalone runner for JUST the real-psifx SAM3 baseline config (the
`b) sam3_baseline` step of `run_four_way_comparison.py`, extracted here
so it can be run alone -- without also paying for the 3 SAM3.1 configs
`run_four_way_comparison.py` always runs alongside it -- when the SAM3.1
side of the comparison isn't needed (2026-08, Michele: full-session test
using only SAM3 + `merge_fragments` post-processing).

Same call into real psifx's `Sam3TrackingTool.infer()` as
`run_four_way_comparison._run_psifx_sam3_and_write_maskdir` (kept in
sync by hand -- this is intentionally a thin, disposable wrapper, not a
new abstraction to import from elsewhere), same defaults
(`chunk_size=400`, `iou_threshold=0.15` -- psifx's own production
config per CHUV's `config.yaml`, already confirmed the right baseline
in `run_four_way_comparison.py`'s docstring).

Not runnable in this project's sandbox (needs the real `psifx` package
and a GPU) -- verify on Michele's machine, same as the rest of
`psifx_eval`.

Usage:
    python -m psifx_eval.run_sam3_baseline \\
        --video ../../camera_a_color.mp4 \\
        --mask-dir psifx_eval_out/naomi_full_session/sam3_baseline/MaskDir \\
        --device cuda
"""

from __future__ import annotations

import argparse


def run_sam3_baseline(
    *,
    video_path: str,
    mask_dir: str,
    chunk_size: int = 400,
    text_prompt: str = "person",
    iou_threshold: float = 0.15,
    max_objects: int | None = None,
    model_path: str | None = None,
    device: str = "cuda",
    overwrite: bool = False,
) -> None:
    """Runs real psifx's `Sam3TrackingTool.infer()` and writes the
    MaskDir to `mask_dir` -- see module docstring."""
    from psifx.video.tracking.sam3.tool import Sam3TrackingTool

    kwargs = dict(model_path=model_path) if model_path else {}
    tool = Sam3TrackingTool(
        device=device, max_num_objects=max_objects, overwrite=overwrite, verbose=True, **kwargs,
    )
    tool.infer(
        video_path=video_path, mask_dir=mask_dir, text_prompt=text_prompt,
        chunk_size=chunk_size, iou_threshold=iou_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs ONLY the real-psifx SAM3 baseline tracking config (chunk_size=400, "
                     "psifx's own production stitching) -- see module docstring.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--mask-dir", required=True, help="Output MaskDir path")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument("--iou-threshold", type=float, default=0.15,
                         help="psifx's own cross-chunk stitching threshold -- default matches CHUV's config.yaml")
    parser.add_argument("--max-objects", type=int, default=None, help="Uncapped by default")
    parser.add_argument("--model-path", default=None, help="Overrides psifx's default SAM3_PATH")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_sam3_baseline(
        video_path=args.video, mask_dir=args.mask_dir, chunk_size=args.chunk_size,
        text_prompt=args.text_prompt, iou_threshold=args.iou_threshold,
        max_objects=args.max_objects, model_path=args.model_path,
        device=args.device, overwrite=args.overwrite,
    )
    print(f"\nDone -- MaskDir written to {args.mask_dir}")


if __name__ == "__main__":
    main()
