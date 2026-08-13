"""
psifx_eval/run_four_way_comparison.py
========================================
Unified comparison script, REWRITTEN 2026-08-13 per Michele's second
pivot on this design. The SAM3-continuous "oracle" (one pass, no
chunking at all) is GONE: Michele visually inspected it frame-by-frame
and confirmed it inherits SAM3's own native tracking errors on hard
occlusions/crossings (two confirmed relabeling events, ~14s and
~38-39s at 15fps) -- it was never actually ground truth, just another
imperfect run. Trusting it as the reference meant scoring everything
else against a flawed target.

The new reference is `sam31_overlap_osnet_cap4`: this repo's own
`Sam31Tracker`, SAM 3.1's natural chunk_size/overlap (600/50), OSNet
appearance-fallback ON, capped at the known headcount (max_people=4).
Michele ran this exact config from the GUI and confirmed it visually:
ids stay stable on the right person throughout, only occasional
single-pixel mask edge slop, no real identity errors ("config 4
perfetto"). It becomes the new "oracolo": the 3 other configs are
scored AGAINST it now, not the other way around.

  a. sam31_overlap_osnet_cap4  -- THE REFERENCE. chunk_size=600,
                                   overlap=50, appearance_fallback=True,
                                   max_people=4 (the known headcount --
                                   this is what makes it trustworthy:
                                   no config's raw id count is inflated
                                   by spurious detections). Run FIRST,
                                   per Michele's explicit instruction,
                                   so its MaskDir exists before any
                                   metrics are computed against it.
  b. sam3_baseline              -- real psifx, SAM3, chunk_size=400,
                                    psifx's own single-frame greedy
                                    stitching. Uncapped (no max_objects)
                                    -- only the reference gets the cap.
  c. sam31_min_overlap          -- native SAM 3.1, SAME chunk_size as
                                    the reference (600) but overlap
                                    forced to the minimum the code
                                    allows (1 -- `ChunkedVideoPredictorBackend`
                                    hard-validates overlap >= 1, see
                                    segmentation/sam_backend.py), no
                                    OSNet, uncapped. Isolates "does
                                    overlap alone help, even without
                                    appearance recovery".
  d. sam31_overlap              -- native SAM 3.1, chunk_size=600,
                                    overlap=50 (SAME as the reference),
                                    no OSNet, uncapped. Isolates "does
                                    OSNet + the headcount cap matter,
                                    holding chunk_size/overlap fixed" --
                                    this is the single config that
                                    differs from the reference by
                                    exactly those two knobs.

Configs c and d share the reference's chunk_size/overlap on purpose --
b is the one apples-to-apples ablation against real psifx (does native
SAM3.1 + Hungarian reconciliation alone beat psifx's greedy stitching),
c/d isolate what OSNet + the cap add on top of SAM3.1's own best
chunking.

IMPORTANT -- chunk boundary math still differs between psifx and
Sam31Tracker (unchanged from the previous version of this script):
psifx's real chunks never overlap, so boundaries fall at exact
multiples of `chunk_size`. Sam31Tracker's chunks DO overlap
(`segmentation/chunking.py::iter_chunk_ranges`): chunk N+1 starts
`chunk_size - overlap` frames after chunk N started, so its real
re-seeding boundaries fall at multiples of that stride, not
`chunk_size` itself. `id_metrics.compute_metrics()`'s `chunk_size`
argument is used directly as that period -- so for every Sam31Tracker
config this script passes `chunk_size - overlap` (the real stride),
never the raw chunk_size.

Not runnable in this project's sandbox (needs the real `psifx` package
for config b, the real `sam3` package + CUDA + gated checkpoint access
for configs a/c/d) -- verify on Michele's machine.

Usage:
    python -m psifx_eval.run_four_way_comparison \\
        --video ../../test_video.mp4 \\
        --out-dir psifx_eval_out/dancetrack/four_way \\
        --sam3-chunk-size 400 \\
        --sam31-chunk-size 600 --overlap 50 --min-overlap 1 \\
        --text-prompt person --max-people-cap 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _run_psifx_sam3_and_write_maskdir(
    *, video_path: str, mask_dir: str, chunk_size: int, text_prompt: str,
    iou_threshold: float, model_path: str | None, max_objects: int | None,
    device: str, overwrite: bool,
) -> None:
    """Thin call into real psifx's `Sam3TrackingTool.infer()` -- the one
    place this logic lives now that `run_baseline_vs_oracle.py` is
    retired."""
    from psifx.video.tracking.sam3.tool import Sam3TrackingTool

    kwargs = dict(model_path=model_path) if model_path else {}
    tool = Sam3TrackingTool(
        device=device, max_num_objects=max_objects, overwrite=overwrite, verbose=True, **kwargs,
    )
    tool.infer(
        video_path=video_path, mask_dir=mask_dir, text_prompt=text_prompt,
        chunk_size=chunk_size, iou_threshold=iou_threshold,
    )


def run_four_way_comparison(
    *,
    video_path: str,
    out_dir: str,
    sam3_chunk_size: int,
    sam31_chunk_size: int,
    overlap: int,
    min_overlap: int,
    max_people_cap: int | None,
    text_prompt: str,
    psifx_iou_threshold: float,
    sam31_iou_threshold: float,
    eval_iou_threshold: float,
    model_path: str | None,
    device: str,
    overwrite: bool,
) -> dict:
    from psifx_eval.id_metrics import compute_metrics
    from psifx_eval.mask_io import load_mask_dir
    from psifx_eval.run_sam31_native import run_sam31_and_write_maskdir

    out_dir_path = Path(out_dir)
    reference_mask_dir = out_dir_path / "reference_sam31_overlap_osnet_cap4" / "MaskDir"
    sam3_baseline_mask_dir = out_dir_path / "sam3_baseline" / "MaskDir"
    sam31_min_overlap_mask_dir = out_dir_path / "sam31_min_overlap" / "MaskDir"
    sam31_overlap_mask_dir = out_dir_path / "sam31_overlap" / "MaskDir"

    # --- a) THE REFERENCE, run first per Michele's instruction, so its
    # MaskDir is on disk before anything is scored against it. ---
    print(f"\n{'=' * 70}\n1/4  reference: sam31_overlap_osnet_cap4 "
          f"(chunk_size={sam31_chunk_size}, overlap={overlap}, "
          f"appearance_fallback=True, max_people={max_people_cap})\n{'=' * 70}")
    run_sam31_and_write_maskdir(
        video_path=video_path, mask_dir=str(reference_mask_dir), chunk_size=sam31_chunk_size,
        overlap=overlap, text_prompt=text_prompt, iou_threshold=sam31_iou_threshold,
        max_people=max_people_cap, device=device, appearance_fallback=True,
    )

    # --- b) sam3_baseline: real psifx, uncapped ---
    print(f"\n{'=' * 70}\n2/4  sam3_baseline (real psifx SAM3, chunk_size={sam3_chunk_size}, uncapped)\n{'=' * 70}")
    _run_psifx_sam3_and_write_maskdir(
        video_path=video_path, mask_dir=str(sam3_baseline_mask_dir), chunk_size=sam3_chunk_size,
        text_prompt=text_prompt, iou_threshold=psifx_iou_threshold, model_path=model_path,
        max_objects=None, device=device, overwrite=overwrite,
    )

    # --- c) sam31_min_overlap: same chunk_size as the reference, overlap forced to the minimum, no OSNet, uncapped ---
    print(f"\n{'=' * 70}\n3/4  sam31_min_overlap (native SAM 3.1, chunk_size={sam31_chunk_size}, "
          f"overlap={min_overlap}, appearance_fallback=False, uncapped)\n{'=' * 70}")
    run_sam31_and_write_maskdir(
        video_path=video_path, mask_dir=str(sam31_min_overlap_mask_dir), chunk_size=sam31_chunk_size,
        overlap=min_overlap, text_prompt=text_prompt, iou_threshold=sam31_iou_threshold,
        max_people=None, device=device, appearance_fallback=False,
    )

    # --- d) sam31_overlap: same chunk_size/overlap as the reference, no OSNet, uncapped ---
    print(f"\n{'=' * 70}\n4/4  sam31_overlap (native SAM 3.1, chunk_size={sam31_chunk_size}, "
          f"overlap={overlap}, appearance_fallback=False, uncapped)\n{'=' * 70}")
    run_sam31_and_write_maskdir(
        video_path=video_path, mask_dir=str(sam31_overlap_mask_dir), chunk_size=sam31_chunk_size,
        overlap=overlap, text_prompt=text_prompt, iou_threshold=sam31_iou_threshold,
        max_people=None, device=device, appearance_fallback=False,
    )

    print(f"\n{'=' * 70}\nscoring the other 3 configs against the reference\n{'=' * 70}")
    reference_masks = load_mask_dir(reference_mask_dir)
    # (mask_dir, metrics_chunk_size) -- metrics_chunk_size is the REAL
    # boundary period for each tracker (see module docstring): psifx
    # never overlaps (period = chunk_size), Sam31Tracker does (period =
    # chunk_size - overlap, its real re-seeding stride).
    configs = {
        "sam3_baseline": (sam3_baseline_mask_dir, sam3_chunk_size),
        "sam31_min_overlap": (sam31_min_overlap_mask_dir, sam31_chunk_size - min_overlap),
        "sam31_overlap": (sam31_overlap_mask_dir, sam31_chunk_size - overlap),
    }

    results: dict[str, dict] = {}
    for label, (mask_dir, metrics_chunk_size) in configs.items():
        masks = load_mask_dir(mask_dir)
        report = compute_metrics(
            oracle=reference_masks, baseline=masks, chunk_size=metrics_chunk_size,
            iou_threshold=eval_iou_threshold,
        )
        results[label] = report.to_dict()
        print(f"\n--- {label} (boundary period used for classification: {metrics_chunk_size}) ---")
        print(report.summary())

    print(f"\n{'=' * 70}\nsummary (all vs sam31_overlap_osnet_cap4)\n{'=' * 70}")
    header = f"{'config':18s} {'fragmentation':>13s} {'swaps':>6s} {'boundary_acc':>13s}"
    print(header)
    for label, report in results.items():
        acc = report["boundary_accuracy"]
        acc_str = f"{acc:.1%}" if acc is not None else "n/a"
        print(f"{label:18s} {report['fragmentation_count']:>13} {report['swap_count']:>6} {acc_str:>13s}")

    comparison_path = out_dir_path / "four_way_comparison_report.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison written to {comparison_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs the new reference config (native SAM3.1, chunk=600/overlap=50, OSNet on, "
                     "capped at max_people_cap) FIRST, then real psifx SAM3 (chunked), native SAM3.1 with "
                     "minimal overlap, and native SAM3.1 with overlap=50 (no OSNet, uncapped) -- and scores "
                     "the last 3 against the reference with the same id_metrics, writing one combined report.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--out-dir", required=True, help="Output directory for all 4 MaskDirs + the combined report")
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument("--sam3-chunk-size", type=int, default=400,
                         help="chunk_size for sam3_baseline (real psifx) only")
    parser.add_argument("--sam31-chunk-size", type=int, default=600,
                         help="Shared chunk_size for the reference, sam31_min_overlap, and sam31_overlap "
                              "-- SAM 3.1's natural/recommended value (segmentation/sam_backend.py's own default)")
    parser.add_argument("--overlap", type=int, default=50,
                         help="Shared overlap for the reference AND sam31_overlap -- kept equal on purpose so "
                              "the only difference between them is OSNet + the max_people cap")
    parser.add_argument("--min-overlap", type=int, default=1,
                         help="sam31_min_overlap's overlap -- forced to the minimum the code allows "
                              "(ChunkedVideoPredictorBackend hard-validates overlap >= 1)")
    parser.add_argument("--max-people-cap", type=int, default=4,
                         help="max_people for the REFERENCE config only (the known headcount for this test "
                              "video) -- sam3_baseline/sam31_min_overlap/sam31_overlap are all left uncapped")
    parser.add_argument("--psifx-iou-threshold", type=float, default=0.15,
                         help="psifx's OWN cross-chunk stitching threshold (sam3_baseline) -- default matches CHUV's config.yaml")
    parser.add_argument("--sam31-iou-threshold", type=float, default=0.3,
                         help="Sam31Tracker's OWN cross-chunk reconciliation threshold (all 3 native SAM3.1 configs)")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.1,
                         help="THIS script's reference-to-run correspondence threshold (id_metrics.compute_metrics), "
                              "used identically for all 3 comparisons")
    parser.add_argument("--model-path", default=None, help="Overrides psifx's default SAM3_PATH (sam3_baseline only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_four_way_comparison(
        video_path=args.video, out_dir=args.out_dir, sam3_chunk_size=args.sam3_chunk_size,
        sam31_chunk_size=args.sam31_chunk_size, overlap=args.overlap, min_overlap=args.min_overlap,
        max_people_cap=args.max_people_cap, text_prompt=args.text_prompt,
        psifx_iou_threshold=args.psifx_iou_threshold, sam31_iou_threshold=args.sam31_iou_threshold,
        eval_iou_threshold=args.eval_iou_threshold, model_path=args.model_path,
        device=args.device, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
