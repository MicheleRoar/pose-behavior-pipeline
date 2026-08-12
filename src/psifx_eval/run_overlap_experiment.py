"""
psifx_eval/run_overlap_experiment.py
=======================================
Experiment 2 (items 2/3 of Michele/Loic's brief -- "possibilmente fare
overlap di chunks per evitare che l'id venga perso"): compares vanilla
psifx's chunked baseline against the "overlapping chunks + mask IoU"
strategy (`overlap_tracking.OverlappingChunkSam3TrackingTool`), BOTH
measured against the SAME continuous oracle run of the same video --
so the two `IdPersistenceReport`s are directly comparable (same video,
same oracle, same `chunk_size` for boundary classification, only the
cross-chunk linking differs).

One loaded model, three passes (`Sam3TrackingTool.infer` is inherited
unchanged by `OverlappingChunkSam3TrackingTool`, so the oracle and
vanilla-baseline passes here are byte-for-byte the same psifx code as
`run_baseline_vs_oracle.py` uses -- only the third pass,
`infer_with_overlap`, is new):

  1. oracle       -- chunk_size forced to cover the whole video (see
                      run_baseline_vs_oracle.py's docstring for the
                      OOM-splitting caveat, applies here too).
  2. vanilla       -- real chunk_size, real psifx single-frame greedy
                      stitching (Sam3TrackingTool.infer, unmodified).
  3. overlap       -- same chunk_size, PLUS `overlap` shared frames
                      between consecutive chunks, multi-frame Hungarian
                      IoU stitching (see overlap_strategy.py).

Not runnable in this project's sandbox (needs the real `psifx` package,
CUDA GPU, gated SAM3 checkpoint access) -- verify on Michele's real
machine. The strategy's own algorithmic core IS fully unit-tested
without any of that, see `tests/overlap_strategy_check.py`.

Usage:
    python -m psifx_eval.run_overlap_experiment \\
        --video /path/to/session.mp4 --out-dir psifx_eval_out/session_01 \\
        --chunk-size 400 --overlap 75 --iou-threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from psifx_eval.run_baseline_vs_oracle import _probe_total_frames


def run_overlap_experiment(
    *,
    video_path: str,
    out_dir: str,
    text_prompt: str,
    chunk_size: int,
    overlap: int,
    iou_threshold: float,
    max_objects: int | None,
    eval_iou_threshold: float,
    model_path: str | None,
    oracle_chunk_size: int | None,
    device: str,
    overwrite: bool,
) -> dict:
    from psifx_eval.id_metrics import compute_metrics
    from psifx_eval.mask_io import load_mask_dir
    from psifx_eval.overlap_tracking import OverlappingChunkSam3TrackingTool

    out_dir_path = Path(out_dir)
    oracle_mask_dir = out_dir_path / "oracle_continuous" / "MaskDir"
    vanilla_mask_dir = out_dir_path / "vanilla_baseline" / "MaskDir"
    overlap_mask_dir = out_dir_path / "overlap_strategy" / "MaskDir"

    if oracle_chunk_size is None:
        total_frames = _probe_total_frames(video_path)
        oracle_chunk_size = total_frames + 1
        print(f"[run_overlap_experiment] auto-detected {total_frames} frames -> "
              f"oracle chunk_size={oracle_chunk_size}")

    kwargs = dict(model_path=model_path) if model_path else {}
    tool = OverlappingChunkSam3TrackingTool(
        device=device, max_num_objects=max_objects, overwrite=overwrite, verbose=True, **kwargs,
    )

    print(f"\n=== oracle (continuous, chunk_size={oracle_chunk_size} -> single chunk) ===")
    tool.infer(
        video_path=video_path, mask_dir=oracle_mask_dir, text_prompt=text_prompt,
        chunk_size=oracle_chunk_size, iou_threshold=iou_threshold,
    )

    print(f"\n=== vanilla psifx baseline (chunk_size={chunk_size}, iou_threshold={iou_threshold}) ===")
    tool.infer(
        video_path=video_path, mask_dir=vanilla_mask_dir, text_prompt=text_prompt,
        chunk_size=chunk_size, iou_threshold=iou_threshold,
    )

    print(f"\n=== overlap strategy (chunk_size={chunk_size}, overlap={overlap}, "
          f"iou_threshold={iou_threshold}) ===")
    tool.infer_with_overlap(
        video_path=video_path, mask_dir=overlap_mask_dir, text_prompt=text_prompt,
        chunk_size=chunk_size, overlap=overlap, iou_threshold=iou_threshold,
    )

    print("\n=== computing cross-chunk ID persistence metrics for both runs ===")
    oracle_masks = load_mask_dir(oracle_mask_dir)
    vanilla_masks = load_mask_dir(vanilla_mask_dir)
    overlap_masks = load_mask_dir(overlap_mask_dir)

    vanilla_report = compute_metrics(
        oracle=oracle_masks, baseline=vanilla_masks,
        chunk_size=chunk_size, iou_threshold=eval_iou_threshold,
    )
    overlap_report = compute_metrics(
        oracle=oracle_masks, baseline=overlap_masks,
        chunk_size=chunk_size, iou_threshold=eval_iou_threshold,
    )

    print("\n--- vanilla psifx baseline ---")
    print(vanilla_report.summary())
    print("\n--- overlap strategy ---")
    print(overlap_report.summary())

    comparison = {
        "vanilla_baseline": vanilla_report.to_dict(),
        "overlap_strategy": overlap_report.to_dict(),
    }
    report_path = out_dir_path / "overlap_vs_vanilla_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nFull comparison written to {report_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compares vanilla psifx chunked tracking against the overlapping-chunks + "
                     "multi-frame mask IoU cross-chunk strategy, both measured against the same "
                     "continuous oracle run -- see the module docstring.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--out-dir", required=True,
                         help="Output directory for all 3 MaskDirs and the JSON comparison")
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument("--chunk-size", type=int, default=400,
                         help="Real chunk size, used by BOTH the vanilla and overlap runs "
                              "(for a fair comparison) -- match CHUV's real deployed config")
    parser.add_argument("--overlap", type=int, default=75,
                         help="Frames shared between consecutive chunks in the overlap "
                              "strategy only (Michele/Loic brief: 50-100)")
    parser.add_argument("--iou-threshold", type=float, default=0.15,
                         help="Cross-chunk stitching threshold, used by BOTH vanilla psifx "
                              "and the overlap strategy (same knob, fair comparison) -- default "
                              "matches CHUV's config.yaml")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.1,
                         help="THIS script's own oracle-to-run correspondence threshold "
                              "(id_metrics.compute_metrics) -- see run_baseline_vs_oracle.py's "
                              "flag of the same name for why it's a separate knob")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--model-path", default=None,
                         help="Overrides psifx's default SAM3_PATH -- set to whatever "
                              "checkpoint CHUV's real deployment uses, see "
                              "run_baseline_vs_oracle.py's checkpoint-mismatch warning")
    parser.add_argument("--oracle-chunk-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_overlap_experiment(
        video_path=args.video, out_dir=args.out_dir, text_prompt=args.text_prompt,
        chunk_size=args.chunk_size, overlap=args.overlap, iou_threshold=args.iou_threshold,
        max_objects=args.max_objects, eval_iou_threshold=args.eval_iou_threshold,
        model_path=args.model_path, oracle_chunk_size=args.oracle_chunk_size,
        device=args.device, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
