"""
psifx_eval/run_baseline_vs_oracle.py
======================================
Experiment 1 (Michele/Loic brief, 2026-08: "psifx implementation of
SAM3[.1] does not seem to have adequate re-identification, possibly due
to the chunking of video segments... jointly investigate"): reproduces
CHUV's REAL production psifx behaviour, unmodified, and quantifies
exactly where its cross-chunk identity linking breaks down, against a
continuous (non-chunked) "oracle" run of the same video as the
reference point -- see `id_metrics.py`'s module docstring for the full
methodology and why a continuous run is a defensible proxy for ground
truth here (isolates chunk-stitching failures from SAM3's own native
tracking failures) without requiring hand-annotated identity labels.

This calls the REAL `psifx` package directly (`Sam3TrackingTool`, see
psifx/video/tracking/sam3/tool.py in github.com/psifx/psifx) --
deliberately NOT a reimplementation: the whole point of this experiment
is to measure and eventually fix the actual system CHUV runs in
production (github.com/LiubovRev/Video-Annotation-System), so results
have to come from psifx's own code, not a look-alike. See
`Sam3TrackingTool.__init__`/`.infer()` for the exact API being called
here -- `chunk_size`/`iou_threshold` below are THEIR parameters,
unmodified.

IMPORTANT -- checkpoint mismatch risk: psifx's own default
(`psifx/utils/constants.py::SAM3_PATH`) is `"facebook/sam3"`. This
project's OWN SAM 3.1 integration (`segmentation/sam_backend.py`) uses
a DIFFERENT checkpoint (`facebook/sam3.1`) via a DIFFERENT code path
(the `facebookresearch/sam3` package directly, not HuggingFace
`transformers`'s `Sam3VideoModel` port that psifx uses). Before treating
any result from this script as representative of "what CHUV sees in
production," confirm which checkpoint their actual deployed psifx
`SAM3_PATH` points to (`--model-path` below lets you override it) --
otherwise this script would faithfully reproduce psifx's ALGORITHM but
on different weights than production, which defeats the purpose.

Two passes over the SAME video, same `Sam3TrackingTool` instance (model
loaded once, `.infer()` carries no state between calls -- see
`Sam3TrackingTool.infer`, everything mutated is local to that call):

  1. "baseline" -- `chunk_size` as configured (CHUV's own
     `Video-Annotation-System/src/config/config.yaml` /
     `project_config.yaml` default is 400, see `processing.py`;
     `--chunk-size` here defaults to that same 400 for comparability,
     override if CHUV's real deployed config differs).
  2. "oracle" -- `chunk_size` forced to (at least) the video's total
     frame count, so `_iter_video_chunks` yields exactly ONE chunk and
     `_map_chunk_object_ids`'s cross-chunk-linking branch never
     triggers at all (it requires a non-empty `prev_last_global_masks`,
     which only exists after a first chunk has already been
     processed). Caveat, worth knowing before trusting long-video
     results: psifx's own CUDA-OOM handling
     (`Sam3TrackingTool.infer`'s `pending_subchunks` retry loop) SPLITS
     an over-large chunk into two and reprocesses them AS SEPARATE
     CHUNKS if the whole video doesn't fit in memory at once -- which
     would silently reintroduce cross-chunk stitching into what's
     supposed to be the chunk-free oracle. Check the printed OOM
     warnings; if any fired during the oracle pass, its `chunk_size`
     wasn't actually "one chunk" and the results should be treated with
     that caveat (or re-run on a shorter clip that does fit).

Not runnable in this project's development sandbox (needs the real
`psifx` package installed, a gated HuggingFace SAM3 checkpoint, and a
CUDA GPU) -- verify on Michele's real machine. The comparison logic
itself (`mask_io.py`/`id_metrics.py`) IS fully unit-tested without any
of that, see `tests/psifx_eval_check.py`.

Usage:
    python -m psifx_eval.run_baseline_vs_oracle \\
        --video /path/to/session.mp4 \\
        --out-dir psifx_eval_out/session_01 \\
        --chunk-size 400 --iou-threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2


def _probe_total_frames(video_path: str) -> int:
    """Container frame count only (no decoding) -- just enough to pick
    an oracle `chunk_size` that's guaranteed to cover the whole video in
    one psifx chunk. Deliberately self-contained (not imported from
    webui/api.py's `probe_video_metadata`) so `psifx_eval` doesn't pull
    in the webui package as a dependency for an unrelated purpose."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(
                f"Container for {video_path!r} doesn't declare a usable frame count "
                f"({frame_count}) -- pass --oracle-chunk-size explicitly instead of "
                f"relying on auto-detection."
            )
        return frame_count
    finally:
        cap.release()


def run_experiment(
    *,
    video_path: str,
    out_dir: str,
    text_prompt: str,
    chunk_size: int,
    iou_threshold: float,
    max_objects: int | None,
    eval_iou_threshold: float,
    model_path: str | None,
    oracle_chunk_size: int | None,
    device: str,
    overwrite: bool,
) -> dict:
    # Delayed import: this module's pure-logic helpers (_probe_total_frames)
    # stay importable/testable without the real `psifx` package installed;
    # only calling this function actually requires it.
    from psifx.video.tracking.sam3.tool import Sam3TrackingTool

    from psifx_eval.id_metrics import compute_metrics
    from psifx_eval.mask_io import load_mask_dir

    out_dir_path = Path(out_dir)
    baseline_mask_dir = out_dir_path / "baseline_chunked" / "MaskDir"
    oracle_mask_dir = out_dir_path / "oracle_continuous" / "MaskDir"

    if oracle_chunk_size is None:
        total_frames = _probe_total_frames(video_path)
        oracle_chunk_size = total_frames + 1  # strictly > total_frames: guarantees one chunk
        print(f"[run_baseline_vs_oracle] auto-detected {total_frames} frames -> "
              f"oracle chunk_size={oracle_chunk_size}")

    kwargs = dict(model_path=model_path) if model_path else {}
    tool = Sam3TrackingTool(
        device=device,
        max_num_objects=max_objects,
        overwrite=overwrite,
        verbose=True,
        **kwargs,
    )

    print(f"\n=== baseline (chunked, chunk_size={chunk_size}, iou_threshold={iou_threshold}) ===")
    tool.infer(
        video_path=video_path,
        mask_dir=baseline_mask_dir,
        text_prompt=text_prompt,
        chunk_size=chunk_size,
        iou_threshold=iou_threshold,
    )

    print(f"\n=== oracle (continuous, chunk_size={oracle_chunk_size} -> single chunk) ===")
    tool.infer(
        video_path=video_path,
        mask_dir=oracle_mask_dir,
        text_prompt=text_prompt,
        chunk_size=oracle_chunk_size,
        # iou_threshold is irrelevant here: with everything in one chunk,
        # _map_chunk_object_ids's cross-chunk branch never runs (see
        # module docstring) -- passed through anyway for psifx API
        # completeness, not because it does anything in this pass.
        iou_threshold=iou_threshold,
    )

    print("\n=== computing cross-chunk ID persistence metrics ===")
    baseline_masks = load_mask_dir(baseline_mask_dir)
    oracle_masks = load_mask_dir(oracle_mask_dir)
    report = compute_metrics(
        oracle=oracle_masks,
        baseline=baseline_masks,
        chunk_size=chunk_size,
        iou_threshold=eval_iou_threshold,
    )

    print("\n" + report.summary())
    report_path = out_dir_path / "id_persistence_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nFull report written to {report_path}")

    return report.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduces real psifx+SAM3 chunked tracking and quantifies cross-chunk "
                     "identity persistence against a continuous (non-chunked) oracle run of "
                     "the same video -- see the module docstring for the full methodology.")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--out-dir", required=True,
                         help="Output directory for both MaskDirs and the JSON report")
    parser.add_argument("--text-prompt", default="person",
                         help="SAM3 text prompt (fixed to 'person' per the agreed experiment "
                              "design, default matches CHUV's own config)")
    parser.add_argument("--chunk-size", type=int, default=400,
                         help="Real (baseline) chunk size -- match CHUV's actual deployed "
                              "config if it differs from their repo's default of 400")
    parser.add_argument("--iou-threshold", type=float, default=0.15,
                         help="psifx's OWN cross-chunk stitching threshold (passed straight "
                              "to Sam3TrackingTool.infer) -- default matches CHUV's config.yaml, "
                              "NOT the same knob as --eval-iou-threshold below")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.1,
                         help="THIS script's own oracle-to-baseline correspondence threshold "
                              "(id_metrics.compute_metrics) -- deliberately separate from "
                              "--iou-threshold, which is psifx's internal stitching decision, "
                              "not ours")
    parser.add_argument("--max-objects", type=int, default=None,
                         help="Sam3TrackingTool(max_num_objects=...) -- CHUV's config.yaml "
                              "default is 3, passed directly here (NOT via the PSIFX_MAX_OBJECTS "
                              "env var their CLI wrapper uses, since we call the Python API "
                              "directly, see module docstring)")
    parser.add_argument("--model-path", default=None,
                         help="Overrides psifx's default SAM3_PATH ('facebook/sam3') -- set "
                              "this to whatever checkpoint CHUV's real deployment actually "
                              "uses, see the module docstring's checkpoint-mismatch warning")
    parser.add_argument("--oracle-chunk-size", type=int, default=None,
                         help="Overrides auto-detected (total_frames + 1) chunk size for the "
                              "oracle pass -- set explicitly if frame-count auto-detection is "
                              "unavailable/wrong for this file")
    parser.add_argument("--device", default="cuda",
                         help="Sam3TrackingTool requires a CUDA GPU for any real-sized video "
                              "(default 'cuda') -- 'cpu' will technically run but is not what "
                              "CHUV's production deployment uses, so results wouldn't be "
                              "representative")
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow writing into a non-empty MaskDir (passed to Sam3TrackingTool)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_experiment(
        video_path=args.video,
        out_dir=args.out_dir,
        text_prompt=args.text_prompt,
        chunk_size=args.chunk_size,
        iou_threshold=args.iou_threshold,
        max_objects=args.max_objects,
        eval_iou_threshold=args.eval_iou_threshold,
        model_path=args.model_path,
        oracle_chunk_size=args.oracle_chunk_size,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
