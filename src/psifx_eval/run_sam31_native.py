"""
psifx_eval/run_sam31_native.py
=================================
Tests whether SAM 3.1 (native `facebookresearch/sam3` package, via this
project's own `Sam31Tracker` in `segmentation/sam31_estimation.py`) beats
the psifx+SAM3 baseline/oracle already measured -- WITHOUT trying to make
SAM3.1 usable *inside* psifx itself (per Michele, 2026-08: "non dobbiamo
mettere sam3.1 in psifx, limitiamoci a testare se va meglio delle
baseline"). `sam3_model.py`/`compare_sam3_checkpoints.py` already
established that `facebook/sam3.1` can't be loaded through psifx's
`transformers`-based code path at all (it's published as a raw
`sam3.1_multiplex.pt` checkpoint, not a `transformers`-compatible
release) -- this script sidesteps that entirely by using the SEPARATE,
already-working native-package integration this repo already has for
SAM 3.1 (`Sam31Tracker`: SAM's own chunking + Hungarian multi-window
reconciliation + appearance-gallery fallback, NOT psifx's single-frame
greedy stitching).

Runs `Sam31Tracker` on the same video, writes its output into the SAME
MaskDir format psifx uses (`<id>.mp4`, back-filled to the full video
length -- see `Sam3TrackingTool._write_chunk_masks` in real psifx for
the convention being matched), then reuses the EXISTING
`id_metrics.compute_metrics()` unchanged, comparing against the
oracle_continuous/MaskDir a prior `run_baseline_vs_oracle.py` run
already produced (real psifx+SAM3, one continuous chunk -- still a
valid reference point here: it represents "how many real identities
there are and where", independent of which tracker produced the run
being scored against it).

Mask videos here are written with OpenCV's always-available 'mp4v'
fourcc (NOT `common/video_writer.py`'s VP9 writer): these files are only
ever read back by `mask_io.py` via `cv2.VideoCapture` for metrics, never
played in the GUI directly, so the "does this decode in QtWebEngine on
Linux" concern that motivated VP9 elsewhere in this project doesn't
apply here -- and 'mp4v' keeps the `<id>.mp4` filename `mask_io.py`
already expects. Run `visualize_masks.py` afterward (same as for the
psifx runs) to get a GUI-viewable overlay of THIS run's masks.

Not runnable in this project's sandbox (needs the real `sam3` package,
CUDA, gated `facebook/sam3.1` checkpoint) -- verify on Michele's machine.

Usage:
    python -m psifx_eval.run_sam31_native \\
        --video ../../test_video.mp4 \\
        --oracle-mask-dir psifx_eval_out/dancetrack/oracle_continuous/MaskDir \\
        --out-dir psifx_eval_out/dancetrack/sam31_native \\
        --chunk-size 400 --overlap 50 --text-prompt person
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _rasterize_polygon(poly: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """`(N,2)` polygon (as returned by `SegFrameResult.people`, see
    `segmentation/sam_backend.py::_mask_to_polygon`) -> `(H, W)` boolean
    mask, via `cv2.fillPoly` -- the same rasterization already relied on
    implicitly by `chunking.polygon_iou` elsewhere in this project.
    Empty polygon -> all-False mask (person not present this frame)."""
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if poly.shape[0] >= 3:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask.astype(bool)


def run_sam31_and_write_maskdir(
    *,
    video_path: str,
    mask_dir: str,
    chunk_size: int,
    overlap: int,
    text_prompt: str,
    iou_threshold: float,
    max_people: int | None,
    device: str,
    appearance_fallback: bool = True,
) -> str:
    """Runs `Sam31Tracker` end to end and writes its output as a psifx-
    shaped MaskDir at `mask_dir`. Returns `mask_dir` (as a str) once
    every tracked id's video has been back-filled to the full source
    video length, matching real psifx's own invariant (see module
    docstring) that `mask_io.load_mask_dir()` depends on.

    `appearance_fallback` (default True, matching `Sam31Tracker`'s own
    default): toggles the OSNet-embedding gallery used when a chunk
    boundary finds no geometric match (see `segmentation/
    identity_gallery.py`) -- set `False` to get "pure" SAM 3.1 +
    geometric/Hungarian reconciliation only, no appearance recovery.
    This is the ONE knob that distinguishes configs 3 and 4 of the
    four-way comparison (`run_four_way_comparison.py`): everything else
    about this function is identical between the two calls."""
    from segmentation.sam31_estimation import Sam31Tracker
    from psifx_eval.video_probe import probe_total_frames

    total_frames = probe_total_frames(video_path)

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    frame_shape = (height, width)

    mask_dir_path = Path(mask_dir)
    mask_dir_path.mkdir(parents=True, exist_ok=True)

    tracker = Sam31Tracker(
        device=device, chunk_size=chunk_size, overlap=overlap,
        text_prompt=text_prompt, iou_threshold=iou_threshold, max_people=max_people,
        appearance_fallback=appearance_fallback,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    empty_mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    writers: dict[int, cv2.VideoWriter] = {}
    written_frames: dict[int, int] = {}

    try:
        for result in tracker.run(video_path):
            frame_index = result.frame_index

            masks_by_id: dict[int, np.ndarray] = {}
            for obj_id, _box, poly, _conf in result.people:
                mask = _rasterize_polygon(poly, frame_shape)
                if mask.any():
                    masks_by_id[obj_id] = mask

            # Lazily open a writer for any id seen for the first time,
            # back-filling empty/black frames up to its first appearance
            # -- same convention as psifx's own _write_chunk_masks.
            for obj_id in masks_by_id:
                if obj_id in writers:
                    continue
                path = mask_dir_path / f"{obj_id}.mp4"
                writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open mask writer for id {obj_id} at {path}")
                writers[obj_id] = writer
                written_frames[obj_id] = 0
                for _ in range(frame_index):
                    writer.write(empty_mask_rgb)
                    written_frames[obj_id] += 1

            # Every currently-open writer gets a frame this round, black
            # if this id isn't present right now -- keeps every mask
            # video the same length, the invariant mask_io.py checks.
            for obj_id, writer in writers.items():
                mask = masks_by_id.get(obj_id)
                if mask is None:
                    writer.write(empty_mask_rgb)
                else:
                    mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., np.newaxis], 3, axis=-1)
                    writer.write(mask_rgb)
                written_frames[obj_id] += 1

        # Defensive pad: if Sam31Tracker stopped short of the full video
        # (e.g. it lost everyone near the end and emitted no further
        # frames for them), back-fill every writer up to total_frames so
        # mask_io.load_mask_dir()'s equal-length check still passes.
        for obj_id, writer in writers.items():
            while written_frames[obj_id] < total_frames:
                writer.write(empty_mask_rgb)
                written_frames[obj_id] += 1
    finally:
        for writer in writers.values():
            writer.release()

    if not writers:
        print("[run_sam31_native] warning: Sam31Tracker produced no tracked ids at all.")

    return str(mask_dir_path)


def run_experiment(
    *,
    video_path: str,
    oracle_mask_dir: str,
    out_dir: str,
    chunk_size: int,
    overlap: int,
    text_prompt: str,
    iou_threshold: float,
    eval_iou_threshold: float,
    max_people: int | None,
    device: str,
    appearance_fallback: bool = True,
) -> dict:
    from psifx_eval.id_metrics import compute_metrics
    from psifx_eval.mask_io import load_mask_dir

    out_dir_path = Path(out_dir)
    sam31_mask_dir = out_dir_path / "MaskDir"

    print(f"\n=== SAM 3.1 native (chunk_size={chunk_size}, overlap={overlap}, "
          f"text_prompt={text_prompt!r}, appearance_fallback={appearance_fallback}) ===")
    run_sam31_and_write_maskdir(
        video_path=video_path, mask_dir=str(sam31_mask_dir), chunk_size=chunk_size,
        overlap=overlap, text_prompt=text_prompt, iou_threshold=iou_threshold,
        max_people=max_people, device=device, appearance_fallback=appearance_fallback,
    )

    print("\n=== computing cross-chunk ID persistence metrics vs the existing oracle ===")
    oracle_masks = load_mask_dir(oracle_mask_dir)
    sam31_masks = load_mask_dir(sam31_mask_dir)
    report = compute_metrics(
        oracle=oracle_masks, baseline=sam31_masks,
        chunk_size=chunk_size, iou_threshold=eval_iou_threshold,
    )

    print("\n" + report.summary())
    report_path = out_dir_path / "sam31_native_vs_oracle_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nFull report written to {report_path}")

    return report.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs SAM 3.1 (native package, this repo's own Sam31Tracker) on a video "
                     "and scores its cross-chunk ID persistence against an EXISTING oracle "
                     "MaskDir (from a prior run_baseline_vs_oracle.py run) -- see module docstring.")
    parser.add_argument("--video", required=True, help="Path to the source video (SAME video used for the oracle)")
    parser.add_argument("--oracle-mask-dir", required=True,
                         help="Path to an existing oracle_continuous/MaskDir (from run_baseline_vs_oracle.py)")
    parser.add_argument("--out-dir", required=True, help="Output directory for SAM3.1's own MaskDir + report")
    parser.add_argument("--chunk-size", type=int, default=400,
                         help="Match the psifx baseline's chunk_size for a fair head-to-head "
                              "(also used to classify events as cross_chunk/intra_chunk)")
    parser.add_argument("--overlap", type=int, default=50, help="Sam31Tracker's own overlap (must be >= 1)")
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                         help="Sam31Tracker's OWN cross-chunk reconciliation threshold "
                              "(chunking.reconcile_ids_windowed) -- not the same knob as --eval-iou-threshold")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.1,
                         help="THIS script's oracle-to-run correspondence threshold (id_metrics.compute_metrics)")
    parser.add_argument("--max-people", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--appearance-fallback", dest="appearance_fallback", action="store_true", default=True,
                         help="OSNet-embedding recovery at chunk boundaries with no geometric match (default: on)")
    parser.add_argument("--no-appearance-fallback", dest="appearance_fallback", action="store_false",
                         help="Disable the OSNet fallback -- pure geometric/Hungarian reconciliation only")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_experiment(
        video_path=args.video, oracle_mask_dir=args.oracle_mask_dir, out_dir=args.out_dir,
        chunk_size=args.chunk_size, overlap=args.overlap, text_prompt=args.text_prompt,
        iou_threshold=args.iou_threshold, eval_iou_threshold=args.eval_iou_threshold,
        max_people=args.max_people, device=args.device, appearance_fallback=args.appearance_fallback,
    )


if __name__ == "__main__":
    main()
