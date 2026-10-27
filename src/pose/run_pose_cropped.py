"""
pose/run_pose_cropped.py
==========================
Same chain as `pose.run_pose` + `pose.run_depth` (keypoints -> smoothing ->
3D -> movement metrics), but the pose is estimated on the BORDER-CROPPED
masks and video (`merged/<name>_cropped<margin>/` and its `_source/`
video, written by `segmentation.run_pipeline`'s crop step), so stray mask
pixels in the camera's lateral border artifact cannot stretch the crop box
MediaPipe sees.

The keypoints are shifted back to FULL-frame pixel coordinates before
smoothing, so `keypoints_3d.csv` reads the depth at the right pixel and
every output is directly comparable with the uncropped run. The offset is
derived from the two videos' sizes: ffmpeg's `crop` starts at an even
pixel, so a 25 px margin starts at 24 (see ADVANCE's config.yaml note) --
override with `--offset-px` if needed.

`--backend vitpose` uses `pose.extract_keypoints_vitpose` (masked person,
largest-blob box, batched per frame) instead of MediaPipe, and writes to
`pose/<name>_vitpose/`; its `in_own_mask` column makes `pose/depth.py`
skip the depth of keypoints outside the person's mask (hidden parts).

Writes to `pose/<name>_cropmasks/` (MediaPipe) or `pose/<name>_vitpose/`
(never touches `pose/<name>/`):
    keypoints_raw.csv, keypoints_smoothed.csv, keypoints_3d.csv,
    movement_summary.csv, movement_per_frame.csv

Usage:
    cd src
    python -m pose.run_pose_cropped --video ".../10_individual_12/camera_b.mkv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def _size(path: Path) -> tuple[int, int, float, int]:
    c = cv2.VideoCapture(str(path))
    out = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           c.get(cv2.CAP_PROP_FPS), int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
    c.release()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Pose + depth on the border-cropped masks -- see module docstring.")
    p.add_argument("--video", required=True, help="ORIGINAL .mkv (same as run_pose/run_depth)")
    p.add_argument("--ss", default=None)
    p.add_argument("--to", default=None)
    p.add_argument("--crop-margin", type=int, default=25, help="margin used by run_pipeline's crop step")
    p.add_argument("--offset-px", type=int, default=None, help="override the x/y shift back to full frame")
    p.add_argument("--backend", default="vitpose", choices=["vitpose", "mediapipe"])
    p.add_argument("--model-variant", default="lite", choices=["lite", "full", "heavy"], help="MediaPipe only")
    p.add_argument("--vitpose-model", default="usyd-community/vitpose-base-simple", help="ViTPose only")
    p.add_argument("--device", default=None, choices=["cpu", "gpu"],
                   help="default: gpu for vitpose, cpu for mediapipe")
    p.add_argument("--overlay", action="store_true", help="also render the skeleton overlay on the full-frame video")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--redo-depth", action="store_true",
                   help="recompute only keypoints_3d.csv + movement metrics (pose and smoothing are kept)")
    a = p.parse_args()

    from segmentation.run_pipeline import resolve_name, resolve_paths

    name = resolve_name(a.video, a.ss, a.to)
    paths = resolve_paths(a.video, a.ss, a.to)
    video_dir = paths["video_dir"]
    crop_dir = video_dir / "merged" / f"{name}_cropped"
    if not crop_dir.is_dir():
        crop_dir = video_dir / "merged" / f"{name}_cropped{a.crop_margin}"
    crop_video = crop_dir / "_source" / f"{name}_cropped.mp4"
    if not crop_video.exists():
        cands = sorted((crop_dir / "_source").glob("*.mp4"))
        if not cands:
            raise FileNotFoundError(f"No cropped source video in {crop_dir / '_source'}")
        crop_video = cands[0]
    full_video = paths["processed_clip"]

    W, H, fps, n_full = _size(Path(full_video))
    w, h, _, n_crop = _size(crop_video)
    if a.offset_px is not None:
        ox = oy = a.offset_px
    else:
        ox, oy = ((W - w) // 2) // 2 * 2, ((H - h) // 2) // 2 * 2
    print(f"[cropped] full {W}x{H} ({n_full} fr), cropped {w}x{h} ({n_crop} fr) -> shift +{ox},+{oy} px")
    if n_full != n_crop:
        print(f"[cropped] WARNING: frame counts differ ({n_full} vs {n_crop}) -- check before trusting the 3D step")

    device = a.device or ("gpu" if a.backend == "vitpose" else "cpu")
    out = video_dir / "pose" / (f"{name}_vitpose" if a.backend == "vitpose" else f"{name}_cropmasks")
    out.mkdir(parents=True, exist_ok=True)
    raw, smooth = out / "keypoints_raw.csv", out / "keypoints_smoothed.csv"
    k3d, summ, perf = out / "keypoints_3d.csv", out / "movement_summary.csv", out / "movement_per_frame.csv"

    if (not raw.exists() or a.overwrite) and a.backend == "vitpose":
        from pose.extract_keypoints_vitpose import extract_keypoints_vitpose
        extract_keypoints_vitpose(video_path=str(crop_video), merged_mask_dir=str(crop_dir), out_csv=str(raw),
                                  session_id=name, model_name=a.vitpose_model, device=device,
                                  x_offset=ox, y_offset=oy)
    elif not raw.exists() or a.overwrite:
        from pose.extract_keypoints import extract_keypoints
        tmp = out / "keypoints_raw_cropcoords.csv"
        extract_keypoints(video_path=crop_video, merged_mask_dir=str(crop_dir), out_csv=str(tmp),
                          session_id=name, model_variant=a.model_variant, device=device)
        df = pd.read_csv(tmp)
        df["x"] += ox
        df["y"] += oy
        df.to_csv(raw, index=False)
        tmp.unlink()
    if not smooth.exists() or a.overwrite:
        from pose.stabilization import smooth_keypoints
        smooth_keypoints(pd.read_csv(raw), fps=fps).to_csv(smooth, index=False)
    if not k3d.exists() or a.overwrite or a.redo_depth:
        from pose.depth import extract_depth
        extract_depth(mkv_path=a.video, keypoints_csv=str(smooth), out_csv=str(k3d),
                      session_id=name, ss=a.ss)
    if not summ.exists() or a.overwrite or a.redo_depth:
        from pose.movement_metrics import compute_movement_metrics
        per_frame, summary = compute_movement_metrics(pd.read_csv(k3d), fps=fps)
        summary.to_csv(summ, index=False)
        per_frame.to_csv(perf, index=False)
        print(summary.to_string(index=False))
    if a.overlay:
        from pose.visualize import render_pose_overlay
        path = render_pose_overlay(video_path=str(full_video), keypoints_csv=str(smooth),
                                   out_path=str(out / "keypoints_overlay"), conf_threshold=0.4)
        print(f"[cropped] overlay -> {path}")
    print(f"[cropped] wrote {out}")


if __name__ == "__main__":
    main()
