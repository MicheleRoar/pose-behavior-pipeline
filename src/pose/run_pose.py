"""
pose/run_pose.py
==================
Single entry point chaining the pose steps (keypoint extraction ->
stabilization -> optional overlay video) into one resumable run, meant to
be run AFTER `segmentation.run_pipeline` has produced a `merged/<name>/`
MaskDir for the same video/range -- this step does not run SAM3 or the
identity merge itself, it only consumes their output.

Everything lives next to the source video, in a new `pose/<name>/`
sibling of `masks/<name>/` and `merged/<name>/` (same "rooted at the
source video's own directory" convention as `segmentation.run_pipeline`
-- see its module docstring):

    <video_dir>/
        processed/<name>.mp4          # from segmentation.run_pipeline
        masks/<name>/                 # from segmentation.run_pipeline
        merged/<name>/                # from segmentation.run_pipeline -- this step's input
        pose/<name>/
            keypoints_raw.csv         # extract_keypoints.py: MediaPipe PoseLandmarker per person per frame
            keypoints_smoothed.csv    # stabilization.py: confidence gating + One Euro Filter
            keypoints_overlay.webm    # visualize.py: skeleton drawn on the video -- only with --overlay (see below)

`<name>` is computed the exact same way as `segmentation.run_pipeline`
(same video path + `--ss`/`--to` -> same name), by importing and calling
its own `resolve_name`/`resolve_paths` rather than recomputing the rule
-- if that rule ever changes, both steps change together instead of
silently drifting apart.

Each step is skipped if its output already exists (resumable, same
policy as `segmentation.run_pipeline`) unless `--overwrite`.

Usage:
    cd src
    python -m pose.run_pose --video ~/Bureau/The\\ Sense/Sessions/9_group_1_3/camera_a.mkv \\
        --ss 00:22:34 --to 00:27:40

(same `--video`/`--ss`/`--to` you passed to
`segmentation.run_pipeline` for this clip -- that's what makes the two
steps agree on `<name>` and therefore on `merged_dir`.)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def resolve_pose_paths(video_path: str, ss: str | None, to: str | None) -> dict[str, Path]:
    """This step's own paths, plus the `segmentation.run_pipeline` paths
    it reads from (`processed_clip`, `merged_dir`) -- computed by
    importing that module's own `resolve_paths` rather than
    reimplementing the `<name>` rule here (see module docstring)."""
    from segmentation.run_pipeline import resolve_paths as _segmentation_paths

    paths = _segmentation_paths(video_path, ss, to)
    from segmentation.run_pipeline import resolve_name
    name = resolve_name(video_path, ss, to)

    paths["pose_dir"] = paths["video_dir"] / "pose" / name
    paths["keypoints_raw"] = paths["pose_dir"] / "keypoints_raw.csv"
    paths["keypoints_smoothed"] = paths["pose_dir"] / "keypoints_smoothed.csv"
    # Extension decided at write time (VP9/.webm preferred, falls back to
    # .mp4 -- see pose/video_writer.py), so this is a stem, not a final path.
    paths["keypoints_overlay_stem"] = paths["pose_dir"] / "keypoints_overlay"
    return paths


def run_pose(
    *,
    video_path: str,
    ss: str | None = None,
    to: str | None = None,
    overwrite: bool = False,
    # extract_keypoints pass-through
    model_path: str | None = None,
    model_variant: str = "lite",
    min_pose_detection_confidence: float = 0.5,
    bbox_padding: float = 0.15,
    device: str = "cpu",
    # stabilization pass-through
    conf_threshold: float = 0.4,
    max_gap_frames: int = 5,
    min_cutoff: float = 1.0,
    beta: float = 0.0,
    # overlay (optional 3rd step)
    overlay: bool = False,
    overlay_source: str = "smoothed",
) -> dict[str, str]:
    """Runs the pose pipeline for one video (optionally trimmed to
    `[ss, to]`, matching an already-completed `segmentation.run_pipeline`
    run for the same range), resuming past any step whose output already
    exists unless `overwrite=True`. Returns the resolved output paths.

    IMPORTANT: `beta=0.0` here is the library default, not a
    recommendation -- see `pose/stabilization.py`'s `OneEuroFilter`
    docstring. Tune `min_cutoff`/`beta` on a real short clip (compare
    `keypoints_raw.csv` vs. `keypoints_smoothed.csv` for a fast-moving
    keypoint like a wrist) before trusting the defaults on a full
    session -- `overlay=True` renders exactly that comparison as a video
    (see `pose/visualize.py`): interpolated/filled-in segments are drawn
    in gray, real detections in the person's own color, so it's visually
    obvious whether the stabilization is helping or over-smoothing on
    this clip. `overlay_source` picks which CSV to render: `"smoothed"`
    (default, shows what the interpolation/filter actually did) or
    `"raw"`."""
    if (ss is None) != (to is None):
        raise ValueError("--ss and --to must be given together, or not at all")

    paths = resolve_pose_paths(video_path, ss, to)

    if not paths["merged_dir"].is_dir() or not any(paths["merged_dir"].glob("*.mp4")):
        raise FileNotFoundError(
            f"No merged MaskDir at {paths['merged_dir']} -- run "
            f"segmentation.run_pipeline for this exact --video/--ss/--to first "
            f"(pose/run_pose.py only consumes its output, it doesn't run SAM3/merge itself)."
        )

    from segmentation.run_pipeline import resolve_name
    name = resolve_name(video_path, ss, to)

    # Step 1: extract_keypoints -> pose/<name>/keypoints_raw.csv
    if paths["keypoints_raw"].exists() and not overwrite:
        print(f"[run_pose] raw keypoints already exist, skipping extraction -> {paths['keypoints_raw']}")
    else:
        from pose.extract_keypoints import extract_keypoints

        paths["pose_dir"].mkdir(parents=True, exist_ok=True)
        print(f"[run_pose] extracting keypoints -> {paths['keypoints_raw']}")
        extract_keypoints(
            video_path=paths["processed_clip"], merged_mask_dir=str(paths["merged_dir"]),
            out_csv=str(paths["keypoints_raw"]), session_id=name,
            model_path=model_path, model_variant=model_variant,
            min_pose_detection_confidence=min_pose_detection_confidence,
            bbox_padding=bbox_padding, device=device,
        )

    # Step 2: stabilization -> pose/<name>/keypoints_smoothed.csv
    if paths["keypoints_smoothed"].exists() and not overwrite:
        print(f"[run_pose] smoothed keypoints already exist, skipping -> {paths['keypoints_smoothed']}")
    else:
        import cv2
        import pandas as pd
        from pose.stabilization import smooth_keypoints

        cap = cv2.VideoCapture(str(paths["processed_clip"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or fps <= 0:
            raise ValueError(f"Could not read a valid fps from {paths['processed_clip']}")

        df = pd.read_csv(paths["keypoints_raw"])
        print(f"[run_pose] stabilizing keypoints (fps={fps}) -> {paths['keypoints_smoothed']}")
        out = smooth_keypoints(
            df, fps=fps, conf_threshold=conf_threshold, max_gap_frames=max_gap_frames,
            min_cutoff=min_cutoff, beta=beta,
        )
        out.to_csv(paths["keypoints_smoothed"], index=False)

    # Step 3 (optional): visualize -> pose/<name>/keypoints_overlay.<ext>
    if overlay:
        if overlay_source not in ("smoothed", "raw"):
            raise ValueError(f"overlay_source must be 'smoothed' or 'raw', got {overlay_source!r}")
        existing = sorted(paths["pose_dir"].glob("keypoints_overlay.*"))
        if existing and not overwrite:
            print(f"[run_pose] overlay video already exists, skipping -> {existing[0]}")
            paths["keypoints_overlay"] = existing[0]
        else:
            from pose.visualize import render_pose_overlay

            source_csv = paths["keypoints_smoothed"] if overlay_source == "smoothed" else paths["keypoints_raw"]
            print(f"[run_pose] rendering {overlay_source} overlay video from {source_csv}")
            actual_path = render_pose_overlay(
                video_path=str(paths["processed_clip"]), keypoints_csv=str(source_csv),
                out_path=str(paths["keypoints_overlay_stem"]), conf_threshold=conf_threshold,
            )
            paths["keypoints_overlay"] = Path(actual_path)

    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs the pose pipeline (keypoint extraction -> stabilization) for one "
                     "video/range already processed by segmentation.run_pipeline -- see module docstring.")
    parser.add_argument("--video", required=True, help="Path to the source video (same one given to segmentation.run_pipeline)")
    parser.add_argument("--ss", default=None, help="Trim start, e.g. 00:22:34 -- must match the segmentation.run_pipeline run")
    parser.add_argument("--to", default=None, help="Trim end -- must match the segmentation.run_pipeline run")
    parser.add_argument("--overwrite", action="store_true", help="Force both steps to re-run")

    extract = parser.add_argument_group("extract_keypoints")
    extract.add_argument("--model-path", default=None, help="Local .task file; auto-downloaded by variant if omitted")
    extract.add_argument("--model-variant", default="lite", choices=["lite", "full", "heavy"])
    extract.add_argument("--min-pose-detection-confidence", type=float, default=0.5)
    extract.add_argument("--bbox-padding", type=float, default=0.15)
    extract.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                          help="MediaPipe's own delegate -- unrelated to SAM3's --device; 'cpu' is the safe default (see extract_keypoints.py)")

    stab = parser.add_argument_group("stabilization")
    stab.add_argument("--conf-threshold", type=float, default=0.4)
    stab.add_argument("--max-gap-frames", type=int, default=5)
    stab.add_argument("--min-cutoff", type=float, default=1.0)
    stab.add_argument("--beta", type=float, default=0.0,
                       help="0.0 is the library default, NOT a recommendation -- see stabilization.py")

    vis = parser.add_argument_group("visualize (optional -- off by default)")
    vis.add_argument("--overlay", action="store_true",
                      help="Also render an annotated video with the skeleton drawn on top -- see pose/visualize.py")
    vis.add_argument("--overlay-source", default="smoothed", choices=["smoothed", "raw"],
                      help="Which CSV to render when --overlay is set (default: smoothed)")

    args = parser.parse_args()
    if (args.ss is None) != (args.to is None):
        parser.error("--ss and --to must be given together")

    result = run_pose(
        video_path=args.video, ss=args.ss, to=args.to, overwrite=args.overwrite,
        model_path=args.model_path, model_variant=args.model_variant,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        bbox_padding=args.bbox_padding, device=args.device,
        conf_threshold=args.conf_threshold, max_gap_frames=args.max_gap_frames,
        min_cutoff=args.min_cutoff, beta=args.beta,
        overlay=args.overlay, overlay_source=args.overlay_source,
    )
    print("\nDone:")
    for key, path in result.items():
        if key not in ("video_dir", "keypoints_overlay_stem"):
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
