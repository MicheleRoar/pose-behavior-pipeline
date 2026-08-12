"""
psifx_eval/visualize_masks.py
================================
Renders a colored, labeled mask overlay video from a psifx MaskDir (the
per-`<id>.mp4` output of `run_baseline_vs_oracle.py` /
`run_overlap_experiment.py` / `compare_sam3_checkpoints.py`) on top of
the original source video -- so a run's identity assignments can be
inspected visually, e.g. loaded side by side into this project's own
webui "Compare runs" window, instead of only read as JSON metrics.

Deliberately does NOT reuse psifx's own `TrackingTool.visualize()`
(inherited by `Sam3TrackingTool`): that writes H.264 (`-c:v libx264`),
which this project already found -- fixing the Compare runs window,
earlier in this repo's history -- does NOT play back reliably in the
GUI's QtWebEngine on Linux (stock Qt builds ship without H.264 decode
at all, patent licensing). Reuses `common/video_writer.py`'s VP9/WebM
writer instead, so anything this script produces is guaranteed to
actually play in the GUI. Also avoids instantiating `Sam3TrackingTool`
(which would load the full SAM3 model just to draw an overlay) --
this only needs `mask_io.py`'s already-built MaskDir reader plus plain
OpenCV drawing.

Usage (run once per MaskDir you want to inspect -- baseline, oracle,
overlap_strategy, ...):
    python -m psifx_eval.visualize_masks \\
        --video ../../test_video.mp4 \\
        --mask-dir psifx_eval_out/dancetrack/baseline_chunked/MaskDir \\
        --out psifx_eval_out/dancetrack/baseline_chunked/visualization

Then open this project's GUI (`cd src && python webui_app.py`) -> Compare
runs -> load the original video plus each `visualization.*` produced
here, side by side.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from psifx_eval.mask_io import load_mask_dir


def _make_colors(ids: list) -> Dict[int, Tuple[int, int, int]]:
    """Deterministic (seeded) per-id colors -- same id gets the same
    color across separate runs of this script, which matters when
    eyeballing baseline vs oracle vs overlap side by side in the GUI:
    id 3 should look the same color in every visualization video."""
    rng = random.Random(0)
    return {obj_id: tuple(rng.randint(50, 255) for _ in range(3)) for obj_id in ids}


def visualize(
    *,
    video_path: str,
    mask_dir: str,
    out_path: str,
    blackout: bool = False,
    labels: bool = True,
) -> str:
    """Overlays every `<id>.mp4` mask in `mask_dir` onto `video_path` in
    a distinct color (+ optional id label at the mask centroid), writes
    the result via `open_annotated_video_writer` (VP9/WebM, falls back
    to H.264/MPEG-4 only if this machine truly lacks a VP9 encoder --
    see that module's docstring), and returns the actual path written
    (may differ in extension from `out_path` if a fallback kicked in)."""
    from common.video_writer import open_annotated_video_writer

    masks = load_mask_dir(mask_dir)
    colors = _make_colors(sorted(masks.keys()))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer, actual_path, codec_label = open_annotated_video_writer(out_path, fps, width, height)
        print(f"Writing visualization as {codec_label} -> {actual_path}")

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                overlay = np.zeros_like(frame)
                label_positions: Dict[int, Tuple[int, int]] = {}
                for obj_id, mask_arr in masks.items():
                    if frame_idx >= mask_arr.shape[0]:
                        continue
                    mask = mask_arr[frame_idx]
                    if not mask.any():
                        continue
                    color = colors[obj_id]
                    for c in range(3):
                        overlay[..., c] = np.where(mask, color[c], overlay[..., c])
                    if labels:
                        ys, xs = np.where(mask)
                        label_positions[obj_id] = (int(xs.mean()), max(int(ys.mean()) - 10, 0))

                background = np.zeros_like(frame) if blackout else frame
                composited = cv2.addWeighted(background, 1.0, overlay, 0.5, 0)

                for obj_id, (x, y) in label_positions.items():
                    text = str(obj_id)
                    cv2.putText(composited, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(composited, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                writer.write(composited)
                frame_idx += 1
        finally:
            writer.release()
    finally:
        cap.release()

    return actual_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renders a colored/labeled mask overlay video from a psifx MaskDir, "
                     "as a VP9/WebM file playable in this project's Compare runs GUI window.")
    parser.add_argument("--video", required=True, help="Path to the original source video")
    parser.add_argument("--mask-dir", required=True, help="Path to a MaskDir (e.g. .../baseline_chunked/MaskDir)")
    parser.add_argument("--out", required=True, help="Output path (extension is just a hint, see open_annotated_video_writer)")
    parser.add_argument("--blackout", action="store_true", help="Black out everything except the tracked masks")
    parser.add_argument("--no-labels", action="store_true", help="Skip drawing id numbers on each mask")
    args = parser.parse_args()

    actual_path = visualize(
        video_path=args.video, mask_dir=args.mask_dir, out_path=args.out,
        blackout=args.blackout, labels=not args.no_labels,
    )
    print(f"\nDone -> {actual_path}")


if __name__ == "__main__":
    main()
