"""
pose/visualize.py
===================
Draws the COCO-17 skeleton for every tracked identity onto the video,
frame by frame, from a `pose/extract_keypoints.py` / `pose/stabilization.py`
output CSV -- a QA/sanity-check tool ("does this actually look right on a
real clip") explicitly called for by this repo's README and by
`pose/stabilization.py`'s own docstring ("compare keypoints_raw.csv against
keypoints_smoothed.csv... before running on a full session").

Works on EITHER `keypoints_raw.csv` or `keypoints_smoothed.csv` (detected
from the CSV's own columns, not a flag the caller has to get right):

  - given `keypoints_smoothed.csv` (has `x_smooth`/`y_smooth`/`interpolated`),
    draws the SMOOTHED position, and renders any point/edge touched by
    `pose/stabilization.py`'s short-gap linear interpolation (no real
    detection that frame) in a muted gray instead of the person's own
    color -- so it's visually obvious which parts of the skeleton are
    filled-in vs. actually observed, same "don't hide that a value is
    invented" principle as the rest of this step.
  - given `keypoints_raw.csv` (only `x`/`y`/`confidence`), draws the raw
    position in the person's own color; a keypoint below
    `--conf-threshold` is skipped entirely rather than drawn (display-only
    gating -- doesn't modify the CSV).

Drawing utilities (`pose/viz.py`) and the output video writer
(`pose/video_writer.py`, VP9/WebM-first for reliable playback) are both
ported from `pose-behavior-pipeline_legacy` -- same reasoning as the rest
of this step (see README): reuse debugged conventions instead of
reinventing them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose.keypoints import KP, SKELETON_EDGES
from pose.video_writer import open_annotated_video_writer
from pose.viz import draw_person_label, get_track_color

_INTERP_COLOR = (140, 140, 140)  # muted gray (BGR) -- "not a real observation", see module docstring


def _draw_skeleton_with_interp(frame: np.ndarray, kxy: np.ndarray, is_real: np.ndarray,
                                color: tuple[int, int, int]) -> np.ndarray:
    """Like `pose.viz.draw_skeleton`, but a point/edge where the value came
    from interpolation rather than a real detection that frame (`is_real`
    False) is drawn in `_INTERP_COLOR` instead of the person's own color.
    For raw (ungated) input, `is_real` is all True, so this behaves
    identically to the plain single-color version."""
    for a_name, b_name in SKELETON_EDGES:
        a_idx, b_idx = KP[a_name], KP[b_name]
        a, b = kxy[a_idx], kxy[b_idx]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        edge_color = color if (is_real[a_idx] and is_real[b_idx]) else _INTERP_COLOR
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), edge_color, 2, cv2.LINE_AA)

    for idx in range(kxy.shape[0]):
        if np.isnan(kxy[idx]).any():
            continue
        point_color = (0, 165, 255) if is_real[idx] else _INTERP_COLOR
        cv2.circle(frame, tuple(kxy[idx].astype(int)), 3, point_color, -1, cv2.LINE_AA)
    return frame


def _person_label_position(kxy: np.ndarray) -> np.ndarray | None:
    """Topmost (smallest y) valid keypoint -- usually the head -- used as
    the anchor for the "ID N" label. `None` if the person has no valid
    keypoint at all this frame (nothing to anchor the label to)."""
    valid = kxy[~np.isnan(kxy).any(axis=1)]
    if len(valid) == 0:
        return None
    return valid[np.argmin(valid[:, 1])]


def _frame_to_person_arrays(rows: pd.DataFrame, use_smooth: bool,
                             conf_threshold: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """`rows`: all keypoint rows for ONE frame (any number of people /
    keypoints). Returns `{global_person_id: (kxy (17,2), is_real (17,))}` --
    `kxy` is `NaN` for any keypoint not drawn at all this frame (missing,
    or below `conf_threshold` in the raw case); `is_real` is False only
    for a keypoint whose value came from `stabilization.py`'s
    interpolation (smoothed input only -- see module docstring)."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, person_rows in rows.groupby("global_person_id", sort=False):
        kxy = np.full((17, 2), np.nan)
        is_real = np.zeros(17, dtype=bool)
        for row in person_rows.itertuples(index=False):
            name = getattr(row, "keypoint_name")
            if name not in KP:
                continue
            idx = KP[name]
            if use_smooth:
                x, y = getattr(row, "x_smooth"), getattr(row, "y_smooth")
                if pd.isna(x) or pd.isna(y):
                    continue
                kxy[idx] = [x, y]
                is_real[idx] = not bool(getattr(row, "interpolated"))
            else:
                x, y, conf = getattr(row, "x"), getattr(row, "y"), getattr(row, "confidence")
                if pd.isna(x) or pd.isna(y) or conf < conf_threshold:
                    continue
                kxy[idx] = [x, y]
                is_real[idx] = True
        out[int(pid)] = (kxy, is_real)
    return out


def render_pose_overlay(
    *,
    video_path: str,
    keypoints_csv: str,
    out_path: str,
    conf_threshold: float = 0.4,
) -> str:
    """Renders `video_path` with the COCO-17 skeleton for every identity in
    `keypoints_csv` drawn on top (see module docstring for the raw vs.
    smoothed behavior). Returns the ACTUAL output path written -- its
    extension can differ from `out_path`'s, see
    `pose.video_writer.open_annotated_video_writer`."""
    df = pd.read_csv(keypoints_csv)
    use_smooth = "x_smooth" in df.columns
    print(f"[visualize] rendering {'smoothed' if use_smooth else 'raw'} keypoints from {keypoints_csv}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise ValueError(f"Could not read a valid fps from {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, actual_path, codec_label = open_annotated_video_writer(out_path, fps, frame_w, frame_h)
    print(f"[visualize] writing {codec_label} -> {actual_path}")

    # One sub-DataFrame per frame index -- built once up front (not
    # re-filtered on every video frame) since the CSV can have millions of
    # rows for a full session.
    frames_by_idx = dict(tuple(df.groupby("frame", sort=False)))

    frame_idx = 0
    n_frames_with_pose = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rows = frames_by_idx.get(frame_idx)
            if rows is not None:
                n_frames_with_pose += 1
                for pid, (kxy, is_real) in _frame_to_person_arrays(rows, use_smooth, conf_threshold).items():
                    color = get_track_color(pid)
                    _draw_skeleton_with_interp(frame_bgr, kxy, is_real, color)
                    label_pos = _person_label_position(kxy)
                    if label_pos is not None:
                        draw_person_label(frame_bgr, label_pos, pid, color)
            writer.write(frame_bgr)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"[visualize] {n_frames_with_pose}/{frame_idx} frames had at least one tracked pose -> {actual_path}")
    return actual_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draws the COCO-17 skeleton for every tracked identity onto a video, from an "
                     "extract_keypoints.py/stabilization.py output CSV -- see module docstring.")
    parser.add_argument("--video", required=True, help="Source video (the processed clip extract_keypoints.py ran on)")
    parser.add_argument("--keypoints-csv", required=True, help="keypoints_raw.csv or keypoints_smoothed.csv")
    parser.add_argument("--out", required=True, help="Output video path (extension is only a hint -- see pose/video_writer.py)")
    parser.add_argument("--conf-threshold", type=float, default=0.4,
                         help="Only used when --keypoints-csv is the RAW (ungated) file")
    args = parser.parse_args()

    actual_path = render_pose_overlay(
        video_path=args.video, keypoints_csv=args.keypoints_csv,
        out_path=args.out, conf_threshold=args.conf_threshold,
    )
    print(f"Done: {actual_path}")


if __name__ == "__main__":
    main()
