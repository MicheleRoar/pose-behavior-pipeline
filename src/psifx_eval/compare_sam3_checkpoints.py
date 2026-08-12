"""
psifx_eval/compare_sam3_checkpoints.py
=========================================
Requested directly by Michele (2026-08): "dobbiamo testare se sam 3.1
overperforma rispetto a sam 3." Runs the SAME baseline-vs-oracle
experiment (`run_baseline_vs_oracle.run_experiment`, real psifx,
unmodified stitching algorithm) twice -- once per SAM3 checkpoint --
so any difference in the resulting `IdPersistenceReport` numbers is
attributable to the CHECKPOINT alone, not the chunking/stitching
algorithm, which is held identical between the two runs.

Why this matters beyond curiosity: psifx's own public-repo default
(`psifx/utils/constants.py::SAM3_PATH`) is `"facebook/sam3"`, NOT
`"facebook/sam3.1"` -- see `run_baseline_vs_oracle.py`'s docstring. If
CHUV's production deployment is still on the older checkpoint, simply
upgrading it (zero algorithm changes) might already fix a meaningful
share of the cross-chunk failures under investigation, independent of
whatever cross-chunk strategy Experiment 2 (`run_overlap_experiment.py`)
finds -- worth knowing before attributing every improvement to a new
stitching algorithm when part of it could just be a newer, more capable
tracker underneath.

Not runnable in this project's sandbox -- verify on Michele's real
machine, same as the other `psifx_eval` scripts.

Usage:
    python -m psifx_eval.compare_sam3_checkpoints \\
        --video /path/to/session.mp4 --out-dir psifx_eval_out/session_01 \\
        --chunk-size 400 --iou-threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# psifx's own default (facebook/sam3) vs the newer, faster release this
# project's own sam_backend.py already uses (facebook/sam3.1, see its
# docstring on SAM 3.1's "object multiplexing" speed improvements).
CHECKPOINTS = {
    "sam3": "facebook/sam3",
    "sam3.1": "facebook/sam3.1",
}


def compare_checkpoints(
    *,
    video_path: str,
    out_dir: str,
    text_prompt: str,
    chunk_size: int,
    iou_threshold: float,
    max_objects: int | None,
    eval_iou_threshold: float,
    oracle_chunk_size: int | None,
    device: str,
    overwrite: bool,
) -> dict:
    from psifx_eval.run_baseline_vs_oracle import run_experiment

    results: dict[str, dict] = {}
    for label, model_path in CHECKPOINTS.items():
        print(f"\n{'=' * 70}\n{label}  ({model_path})\n{'=' * 70}")
        results[label] = run_experiment(
            video_path=video_path,
            out_dir=os.path.join(out_dir, label),
            text_prompt=text_prompt,
            chunk_size=chunk_size,
            iou_threshold=iou_threshold,
            max_objects=max_objects,
            eval_iou_threshold=eval_iou_threshold,
            model_path=model_path,
            oracle_chunk_size=oracle_chunk_size,
            device=device,
            overwrite=overwrite,
        )

    print("\n\n=== SAM3 vs SAM3.1 summary (same video, same stitching algorithm) ===")
    for label, report in results.items():
        acc = report["boundary_accuracy"]
        acc_str = f"{acc:.1%}" if acc is not None else "n/a"
        print(f"  {label:6s}: fragmentation={report['fragmentation_count']:>3}  "
              f"swaps={report['swap_count']:>3}  boundary_accuracy={acc_str}")

    comparison_path = Path(out_dir) / "sam3_vs_sam31_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison written to {comparison_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs the same real-psifx baseline-vs-oracle experiment on both "
                     "facebook/sam3 and facebook/sam3.1, stitching algorithm held fixed, "
                     "so any difference is attributable to the checkpoint alone.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--iou-threshold", type=float, default=0.15)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.1)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--oracle-chunk-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    compare_checkpoints(
        video_path=args.video, out_dir=args.out_dir, text_prompt=args.text_prompt,
        chunk_size=args.chunk_size, iou_threshold=args.iou_threshold,
        max_objects=args.max_objects, eval_iou_threshold=args.eval_iou_threshold,
        oracle_chunk_size=args.oracle_chunk_size, device=args.device, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
