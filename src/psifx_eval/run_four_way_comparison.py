"""
psifx_eval/run_four_way_comparison.py
========================================
The unified comparison Michele asked for (2026-08), replacing the old
`run_baseline_vs_oracle.py` (retired): runs FOUR configurations on the
SAME video and scores three of them against the fourth (the oracle)
with the SAME `id_metrics.compute_metrics()`, so all the numbers land
in one report instead of three separately-run scripts you'd have to
compare by hand.

  1. oracle              -- real psifx, SAM3, ONE continuous chunk (no
                             stitching at all) -- the reference point
                             every other config is scored against.
  2. sam3_baseline        -- real psifx, SAM3, chunked (chunk_size),
                             psifx's own single-frame greedy stitching.
  3. sam31_no_osnet       -- this repo's own Sam31Tracker (native SAM
                             3.1, NOT going through psifx -- see
                             run_sam31_native.py's docstring for why
                             that's fine here), SAME chunk_size as config
                             2 (fair, apples-to-apples ablation against
                             the psifx baseline), geometric/Hungarian
                             reconciliation only, appearance fallback OFF.
  4. sam31_osnet          -- Michele's "best serious candidate" (2026-08):
                             SAM 3.1's OWN natural chunk_size/overlap
                             (600/50, this repo's own default -- see
                             segmentation/sam_backend.py, NOT forced to
                             match config 2/3's 400), appearance fallback
                             ON (OSNet embedding gallery recovers people
                             lost across a chunk boundary with no
                             geometric match, see segmentation/
                             identity_gallery.py -- makes no difference
                             on THIS test video, per Michele, but matters
                             on footage with real exit/re-entry, e.g.
                             someone leaving the room).

Configs 2 and 3 use the SAME chunk_size (fair head-to-head) -- config 4
deliberately does NOT, since it's meant to be SAM3.1 given its best
realistic shot, not another point on the same ablation. All 4 configs
cap detections at `max_people`/`max_objects` = 4 by default (the known
headcount for this test video) so no config's raw id count is inflated
by spurious detections the others don't have.

IMPORTANT -- chunk boundary math differs between psifx and Sam31Tracker:
psifx's real chunks never overlap (chunk N+1 starts exactly where N
ended), so its boundaries fall at exact multiples of `chunk_size`. This
repo's own `Sam31Tracker` chunks DO overlap (`segmentation/chunking.py
::iter_chunk_ranges`): chunk N+1 starts at `chunk_size - overlap` frames
after chunk N started, not at `chunk_size` -- so its real re-seeding
boundaries fall at multiples of `chunk_size - overlap` (the "stride"),
NOT `chunk_size` itself. `id_metrics.compute_metrics()`'s `chunk_size`
argument is used directly as that period for both boundary positions and
cross/intra-chunk classification -- so for configs 3/4 this script
passes `chunk_size - overlap` (the real stride), not the raw chunk_size,
or every boundary check would silently test the WRONG frames.

Not runnable in this project's sandbox (needs the real `psifx` package
for configs 1/2, the real `sam3` package + CUDA + gated checkpoint
access for configs 3/4) -- verify on Michele's machine.

Usage:
    python -m psifx_eval.run_four_way_comparison \\
        --video ../../test_video.mp4 \\
        --out-dir psifx_eval_out/dancetrack/four_way \\
        --chunk-size 400 --overlap 50 \\
        --sam31-osnet-chunk-size 600 --sam31-osnet-overlap 50 \\
        --text-prompt person --max-objects 4 --max-people 4
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
    """Thin call into real psifx's `Sam3TrackingTool.infer()` -- same
    call `run_baseline_vs_oracle.py` used to make for both its "oracle"
    and "baseline" passes (that script is retired; this is the one
    place that logic now lives, since it's only ever needed alongside
    the other 3 configs, not standalone anymore)."""
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
    chunk_size: int,
    overlap: int,
    sam31_osnet_chunk_size: int,
    sam31_osnet_overlap: int,
    text_prompt: str,
    psifx_iou_threshold: float,
    sam31_iou_threshold: float,
    eval_iou_threshold: float,
    max_objects: int | None,
    max_people: int | None,
    model_path: str | None,
    oracle_chunk_size: int | None,
    device: str,
    overwrite: bool,
) -> dict:
    from psifx_eval.id_metrics import compute_metrics
    from psifx_eval.mask_io import load_mask_dir
    from psifx_eval.run_sam31_native import run_sam31_and_write_maskdir
    from psifx_eval.video_probe import probe_total_frames

    out_dir_path = Path(out_dir)
    oracle_mask_dir = out_dir_path / "oracle_continuous" / "MaskDir"
    sam3_baseline_mask_dir = out_dir_path / "sam3_baseline_chunked" / "MaskDir"
    sam31_no_osnet_mask_dir = out_dir_path / "sam31_no_osnet" / "MaskDir"
    sam31_osnet_mask_dir = out_dir_path / "sam31_osnet" / "MaskDir"

    if oracle_chunk_size is None:
        total_frames = probe_total_frames(video_path)
        oracle_chunk_size = total_frames + 1
        print(f"[run_four_way_comparison] auto-detected {total_frames} frames -> "
              f"oracle chunk_size={oracle_chunk_size}")

    print(f"\n{'=' * 70}\n1/4  oracle (real psifx SAM3, continuous, chunk_size={oracle_chunk_size})\n{'=' * 70}")
    _run_psifx_sam3_and_write_maskdir(
        video_path=video_path, mask_dir=str(oracle_mask_dir), chunk_size=oracle_chunk_size,
        text_prompt=text_prompt, iou_threshold=psifx_iou_threshold, model_path=model_path,
        max_objects=max_objects, device=device, overwrite=overwrite,
    )

    print(f"\n{'=' * 70}\n2/4  sam3_baseline (real psifx SAM3, chunk_size={chunk_size})\n{'=' * 70}")
    _run_psifx_sam3_and_write_maskdir(
        video_path=video_path, mask_dir=str(sam3_baseline_mask_dir), chunk_size=chunk_size,
        text_prompt=text_prompt, iou_threshold=psifx_iou_threshold, model_path=model_path,
        max_objects=max_objects, device=device, overwrite=overwrite,
    )

    print(f"\n{'=' * 70}\n3/4  sam31_no_osnet (native SAM 3.1, chunk_size={chunk_size}, "
          f"appearance_fallback=False)\n{'=' * 70}")
    run_sam31_and_write_maskdir(
        video_path=video_path, mask_dir=str(sam31_no_osnet_mask_dir), chunk_size=chunk_size,
        overlap=overlap, text_prompt=text_prompt, iou_threshold=sam31_iou_threshold,
        max_people=max_people, device=device, appearance_fallback=False,
    )

    print(f"\n{'=' * 70}\n4/4  sam31_osnet (native SAM 3.1, chunk_size={sam31_osnet_chunk_size}, "
          f"overlap={sam31_osnet_overlap}, appearance_fallback=True)\n{'=' * 70}")
    run_sam31_and_write_maskdir(
        video_path=video_path, mask_dir=str(sam31_osnet_mask_dir), chunk_size=sam31_osnet_chunk_size,
        overlap=sam31_osnet_overlap, text_prompt=text_prompt, iou_threshold=sam31_iou_threshold,
        max_people=max_people, device=device, appearance_fallback=True,
    )

    print(f"\n{'=' * 70}\ncomputing cross-chunk ID persistence metrics vs the oracle, for all 3\n{'=' * 70}")
    oracle_masks = load_mask_dir(oracle_mask_dir)
    # (mask_dir, metrics_chunk_size) -- metrics_chunk_size is the REAL
    # boundary period for each tracker (see module docstring): psifx
    # never overlaps (period = chunk_size), Sam31Tracker does (period =
    # chunk_size - overlap, its real re-seeding stride).
    configs = {
        "sam3_baseline": (sam3_baseline_mask_dir, chunk_size),
        "sam31_no_osnet": (sam31_no_osnet_mask_dir, chunk_size - overlap),
        "sam31_osnet": (sam31_osnet_mask_dir, sam31_osnet_chunk_size - sam31_osnet_overlap),
    }

    results: dict[str, dict] = {}
    for label, (mask_dir, metrics_chunk_size) in configs.items():
        masks = load_mask_dir(mask_dir)
        report = compute_metrics(
            oracle=oracle_masks, baseline=masks, chunk_size=metrics_chunk_size,
            iou_threshold=eval_iou_threshold,
        )
        results[label] = report.to_dict()
        print(f"\n--- {label} (boundary period used for classification: {metrics_chunk_size}) ---")
        print(report.summary())

    print(f"\n{'=' * 70}\nsummary (all vs the same oracle)\n{'=' * 70}")
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
        description="Runs SAM3 oracle, SAM3 chunked (psifx baseline), SAM3.1 chunked (no OSNet), "
                     "and SAM3.1 chunked (+OSNet) on the same video, scores the last 3 against the "
                     "oracle with the same id_metrics, and writes one combined report.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--out-dir", required=True, help="Output directory for all 4 MaskDirs + the combined report")
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument("--chunk-size", type=int, default=400,
                         help="Shared chunk_size for configs 2/3 (sam3_baseline, sam31_no_osnet) -- fair "
                              "apples-to-apples ablation. Config 4 (sam31_osnet) uses its own "
                              "--sam31-osnet-chunk-size instead. Only the oracle ignores this (single chunk)")
    parser.add_argument("--overlap", type=int, default=50, help="Sam31Tracker's own overlap for config 3 only")
    parser.add_argument("--sam31-osnet-chunk-size", type=int, default=600,
                         help="Config 4's OWN chunk_size -- SAM 3.1's natural/recommended value "
                              "(segmentation/sam_backend.py's own default), not forced to match configs 2/3")
    parser.add_argument("--sam31-osnet-overlap", type=int, default=50, help="Config 4's own overlap")
    parser.add_argument("--psifx-iou-threshold", type=float, default=0.15,
                         help="psifx's OWN cross-chunk stitching threshold (configs 1/2) -- default matches CHUV's config.yaml")
    parser.add_argument("--sam31-iou-threshold", type=float, default=0.3,
                         help="Sam31Tracker's OWN cross-chunk reconciliation threshold (configs 3/4)")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.1,
                         help="THIS script's oracle-to-run correspondence threshold (id_metrics.compute_metrics), "
                              "used identically for all 3 comparisons")
    parser.add_argument("--max-objects", type=int, default=4,
                         help="psifx's max_num_objects (configs 1/2) -- default 4, the known headcount for "
                              "this test video, so no config's raw id count is inflated by spurious detections")
    parser.add_argument("--max-people", type=int, default=4, help="Sam31Tracker's max_people (configs 3/4), same reasoning")
    parser.add_argument("--model-path", default=None, help="Overrides psifx's default SAM3_PATH (configs 1/2 only)")
    parser.add_argument("--oracle-chunk-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_four_way_comparison(
        video_path=args.video, out_dir=args.out_dir, chunk_size=args.chunk_size, overlap=args.overlap,
        sam31_osnet_chunk_size=args.sam31_osnet_chunk_size, sam31_osnet_overlap=args.sam31_osnet_overlap,
        text_prompt=args.text_prompt, psifx_iou_threshold=args.psifx_iou_threshold,
        sam31_iou_threshold=args.sam31_iou_threshold, eval_iou_threshold=args.eval_iou_threshold,
        max_objects=args.max_objects, max_people=args.max_people, model_path=args.model_path,
        oracle_chunk_size=args.oracle_chunk_size, device=args.device, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
