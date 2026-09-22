"""
pose/run_depth.py
===================
Single entry point for the depth step (Azure Kinect depth -> 3D
keypoints -> movement metrics), meant to run AFTER `pose.run_pose` has
already produced `pose/<name>/keypoints_smoothed.csv` for the same
video/range -- this step only consumes that, plus the ORIGINAL `.mkv`
(never `processed/<name>.mp4` -- see `pose/depth.py`'s module
docstring for why).

    <video_dir>/
        pose/<name>/
            keypoints_smoothed.csv   # from pose.run_pose -- this step's input
            keypoints_3d.csv         # depth.py: (X, Y, Z) mm per keypoint per frame
            movement_summary.csv     # movement_metrics.py: distance/velocity/variability per person
            movement_per_frame.csv   # movement_metrics.py: velocity time series

Same `--video`/`--ss`/`--to` convention as `pose.run_pose` and
`segmentation.run_pipeline` -- `--video` here is the ORIGINAL `.mkv`
(the same one given to those two), since this step reads its depth
stream directly, not the transcoded `processed/<name>.mp4`.

Requires `pyk4a` AND the real Azure Kinect Sensor SDK (`libk4a`)
installed system-wide -- see README.md's Setup section.

Usage:
    cd src
    python -m pose.run_depth --video ~/Bureau/The\\ Sense/Sessions/9_group_1_3/camera_a.mkv
"""

from __future__ import annotations

import argparse
from pathlib import Path


def resolve_depth_paths(video_path: str, ss: str | None, to: str | None) -> dict[str, Path]:
    """This step's own paths, on top of `pose.run_pose`'s (which are
    themselves on top of `segmentation.run_pipeline`'s) -- computed by
    importing rather than recomputing the `<name>` rule, same
    discipline as `pose.run_pose` itself."""
    from pose.run_pose import resolve_pose_paths

    paths = resolve_pose_paths(video_path, ss, to)
    paths["keypoints_3d"] = paths["pose_dir"] / "keypoints_3d.csv"
    paths["movement_summary"] = paths["pose_dir"] / "movement_summary.csv"
    paths["movement_per_frame"] = paths["pose_dir"] / "movement_per_frame.csv"
    return paths


def run_depth(
    *,
    video_path: str,
    ss: str | None = None,
    to: str | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Runs the depth step for one video (optionally trimmed to
    `[ss, to]`, matching an already-completed `pose.run_pose` run for
    the same range), resuming past any step whose output already
    exists unless `overwrite=True`. Returns the resolved output paths.

    `video_path` must be the ORIGINAL `.mkv`, not `processed/<name>.mp4`
    -- see `pose/depth.py`'s module docstring for why."""
    if (ss is None) != (to is None):
        raise ValueError("--ss and --to must be given together, or not at all")

    paths = resolve_depth_paths(video_path, ss, to)

    if not paths["keypoints_smoothed"].exists():
        raise FileNotFoundError(
            f"No {paths['keypoints_smoothed']} -- run pose.run_pose for this exact "
            f"--video/--ss/--to first (pose.run_depth only consumes its output, it "
            f"doesn't run pose extraction/stabilization itself)."
        )

    # Step 1: depth -> pose/<name>/keypoints_3d.csv
    if paths["keypoints_3d"].exists() and not overwrite:
        print(f"[run_depth] 3D keypoints already exist, skipping -> {paths['keypoints_3d']}")
    else:
        from pose.depth import extract_depth
        from segmentation.run_pipeline import resolve_name

        name = resolve_name(video_path, ss, to)
        print(f"[run_depth] extracting depth -> {paths['keypoints_3d']}")
        extract_depth(
            mkv_path=video_path, keypoints_csv=str(paths["keypoints_smoothed"]),
            out_csv=str(paths["keypoints_3d"]), session_id=name, ss=ss,
        )

    # Step 2: movement_metrics -> pose/<name>/movement_summary.csv (+ per-frame)
    if paths["movement_summary"].exists() and not overwrite:
        print(f"[run_depth] movement metrics already exist, skipping -> {paths['movement_summary']}")
    else:
        import cv2
        import pandas as pd
        from pose.movement_metrics import compute_movement_metrics

        cap = cv2.VideoCapture(str(paths["processed_clip"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or fps <= 0:
            raise ValueError(f"Could not read a valid fps from {paths['processed_clip']}")

        df_3d = pd.read_csv(paths["keypoints_3d"])
        print(f"[run_depth] computing movement metrics (fps={fps}) -> {paths['movement_summary']}")
        per_frame, summary = compute_movement_metrics(df_3d, fps=fps)
        summary.to_csv(paths["movement_summary"], index=False)
        per_frame.to_csv(paths["movement_per_frame"], index=False)
        print(summary.to_string(index=False))

    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs the depth step (Azure Kinect depth -> 3D keypoints -> movement metrics) "
                     "for one video/range already processed by pose.run_pose -- see module docstring.")
    parser.add_argument("--video", required=True, help="Path to the ORIGINAL .mkv (same one given to segmentation.run_pipeline/pose.run_pose)")
    parser.add_argument("--ss", default=None, help="Trim start -- must match the pose.run_pose run")
    parser.add_argument("--to", default=None, help="Trim end -- must match the pose.run_pose run")
    parser.add_argument("--overwrite", action="store_true", help="Force both steps to re-run")

    args = parser.parse_args()
    if (args.ss is None) != (args.to is None):
        parser.error("--ss and --to must be given together")

    result = run_depth(video_path=args.video, ss=args.ss, to=args.to, overwrite=args.overwrite)
    print("\nDone:")
    for key, path in result.items():
        if key not in ("video_dir", "keypoints_overlay_stem"):
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
