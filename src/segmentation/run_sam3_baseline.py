"""
segmentation/run_sam3_baseline.py
==================================
Thin wrapper around real psifx's `Sam3TrackingTool`: runs SAM3
text-prompted segmentation + tracking on a video and writes the result
as a MaskDir (one `<id>.mp4` per tracked identity). Called by
`run_pipeline.py` as step 2, or standalone for just this step.

Usage:
    python -m segmentation.run_sam3_baseline \\
        --video processed/camera_a.mp4 --mask-dir masks/camera_a --device cuda
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
