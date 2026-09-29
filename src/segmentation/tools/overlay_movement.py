"""
segmentation/tools/overlay_movement.py
========================================
Renders a colored mask overlay (same soft-edge technique as
`overlay_subvideo.py`) on top of the ORIGINAL source video, annotated
with each person's current speed from `pose/movement_metrics.py`'s
per-frame CSV -- an "ad occhio" (by eye) sanity check: does the number
on screen match what's visibly happening in the room, frame by frame.

Built to validate the root-smoothing + deadzone fix (see
`pose/movement_metrics.py`'s module docstring): the on-screen number is
the same smoothed, deadzoned `velocity_mm_s` that ends up in
`movement_summary.csv`, so a seated person should stay at "0.0" and a
walking person should keep a nonzero number moving with them.

A frame flagged `is_outlier=True` by `movement_metrics.py` (rejected as
an unreliable depth reading) is marked with a thin red outline around
that person's label instead of a wall of extra text, so the outlier
rejection is still visible against the real footage without cluttering
the frame.

Unlike `overlay_subvideo.py`, person labels use the REAL
`global_person_id` (not a resequenced 1/2/3 display id) -- this tool is
for cross-checking against `movement_summary.csv`/`movement_per_frame.csv`
by id, so keeping the same id on screen as in those files matters more
than a friendlier sequential label. Colors come from `pose/viz.py`'s
`get_track_color`, the same palette `pose/visualize.py` uses for the
skeleton overlay, so a person keeps the same color across both QA
videos.

Output uses `pose/video_writer.py`'s VP9/WebM-first writer (not
`overlay_subvideo.py`'s plain `mp4v`) so the result reliably plays back
in a browser -- see that module's docstring for why `mp4v` alone is a
known problem here.

Usage:
    python -m segmentation.tools.overlay_movement \
        --video /path/to/session/processed/camera_a.mp4 \
        --mask-dir /path/to/session/merged/camera_a_veto \
        --movement-csv /path/to/session/movement_per_frame.csv \
        --out /path/to/session/movement_overlay.webm

    # prefer km/h over the default m/s for the on-screen speed:
    python -m segmentation.tools.overlay_movement \
        --video ... --mask-dir ... --movement-csv ... --out ... \
        --speed-unit kmh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose.movement_metrics import DEFAULT_STILL_DEADZONE_MM_S
from pose.video_writer import open_annotated_video_writer
from pose.viz import get_track_color
from segmentation.merging.mask_io import DEFAULT_MASK_THRESHOLD, _list_mask_files

# Same calibrated value movement_metrics.py itself now deadzones with (see its
# module docstring's "Suppressing the residual sub-threshold noise floor") --
# imported, not re-typed, so the two can't silently drift apart. Expressed
# here in km/h only because that's this tool's on-screen display unit.
DEFAULT_STILL_DEADZONE_KMH = DEFAULT_STILL_DEADZONE_MM_S * 0.0036


def _format_speed(v_mm_s: float, unit: str, deadzone_mm_s: float = 0.0) -> str:
    """`unit`: "kmh" (default -- what was actually asked for) or "ms"
    (often more legible at room/walking scale, where km/h tends to be a
    small number with decimals -- exposed as a flag rather than picking
    one for the user). NaN (gap, first frame, or rejected outlier frame
    -- see pose/movement_metrics.py) is shown as "--", never a
    made-up number.

    `deadzone_mm_s`: a speed whose magnitude is at or below this shows
    as zero ("0.0 km/h"/"0.00 m/s") instead of the small nonzero value.
    Since `movement_metrics.py` now applies this SAME deadzone (by
    default) when computing `velocity_mm_s` itself -- zeroing it there,
    not just at display time, see that module's docstring -- this is
    mostly a display-time no-op for a CSV computed with the default
    settings, and only does real work for a CSV computed with the
    deadzone off/overridden, or to preview a different threshold
    without recomputing the metrics."""
    if v_mm_s is None or np.isnan(v_mm_s):
        return "--"
    if abs(v_mm_s) <= deadzone_mm_s:
        v_mm_s = 0.0
    if unit == "kmh":
        return f"{v_mm_s * 0.0036:.1f} km/h"
    return f"{v_mm_s / 1000.0:.2f} m/s"


def _draw_speed_label(frame: np.ndarray, position: np.ndarray, person_id: int,
                       color: tuple[int, int, int], speed_text: str, is_outlier: bool) -> None:
    """One compact, semi-transparent pill: "ID <n>  <speed>" in the
    person's track color. An outlier frame (rejected by
    `movement_metrics.py`) gets a thin red outline instead of a second
    text line -- keeps the label small and legible instead of growing a
    stack of lines every frame."""
    x, y = int(position[0]), max(int(position[1]) - 14, 14)
    label = f"ID {person_id}  {speed_text}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    pad_x, pad_y = 8, 6
    x1, y1 = x - pad_x, y - th - pad_y
    x2, y2 = x + tw + pad_x, y + baseline + pad_y

    # Semi-transparent fill (rather than the old solid block) so the
    # video underneath the label stays partly visible -- reads as a
    # softer "pill" instead of an opaque sticker on the frame.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)

    border_color = (0, 0, 230) if is_outlier else (255, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2 if is_outlier else 1)
    cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray, speed_unit: str) -> None:
    """One compact translucent line in the top-left corner, instead of
    the old multi-line block -- enough to make the video
    self-explanatory without this script's docstring open next to it,
    without permanently covering a chunk of the frame."""
    unit_label = "km/h" if speed_unit == "kmh" else "m/s"
    text = f"speed ({unit_label}), smoothed  ·  0.0 = below noise floor  ·  red outline = rejected frame"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    x1, y1 = 8, 8
    x2, y2 = x1 + tw + 2 * pad, y1 + th + baseline + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)
    cv2.putText(frame, text, (x1 + pad, y1 + pad + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def render_movement_overlay(
    *,
    video_path: str,
    mask_dir: str,
    movement_csv: str,
    out_path: str,
    session_id: str | None = None,
    threshold: int = DEFAULT_MASK_THRESHOLD,
    alpha: float = 0.5,
    speed_unit: str = "ms",
    show_legend: bool = True,
    still_deadzone_kmh: float = DEFAULT_STILL_DEADZONE_KMH,
) -> str:
    """Returns the ACTUAL output path written (see
    `pose.video_writer.open_annotated_video_writer` -- its extension can
    differ from `out_path`'s).

    `still_deadzone_kmh`: the on-screen speed shows as zero when at or
    below this, instead of a small nonzero value from residual sensor
    jitter that survived root-smoothing -- see `_format_speed`'s
    `deadzone_mm_s` docstring, and `pose/movement_metrics.py`'s module
    docstring for why this defaults to the SAME calibrated value that
    module now deadzones with at the metrics level. 0 disables it.
    Applied as a physical (unit-independent) threshold, so it behaves
    the same whether `speed_unit` is "kmh" or "ms"."""
    still_deadzone_mm_s = still_deadzone_kmh / 0.0036
    mask_paths = _list_mask_files(mask_dir)
    ids = sorted(mask_paths.keys())
    if not ids:
        raise ValueError(f"No '<id>.mp4' mask files found in {mask_dir}")

    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    df = pd.read_csv(movement_csv)
    if session_id is None:
        distinct_sessions = df["session_id"].unique()
        if len(distinct_sessions) > 1:
            raise ValueError(
                f"{movement_csv} has more than one session_id ({list(distinct_sessions)}) -- "
                f"pass --session-id explicitly to pick one."
            )
        session_id = distinct_sessions[0] if len(distinct_sessions) else None
    df = df[df["session_id"] == session_id].copy()

    # {frame_idx: {global_person_id: row}} -- built once up front, same
    # reasoning as pose/visualize.py's frames_by_idx: a full-session CSV
    # can have a lot of rows, don't re-filter it on every video frame.
    cols = ["global_person_id", "velocity_mm_s", "is_outlier"]
    by_frame: dict[int, dict[int, pd.Series]] = {}
    for frame_idx, rows in df.groupby("frame", sort=False):
        by_frame[int(frame_idx)] = {int(r.global_person_id): r for r in rows[cols].itertuples(index=False)}

    mask_caps = {i: cv2.VideoCapture(str(mask_paths[i])) for i in ids}
    writer, actual_path, codec_label = open_annotated_video_writer(out_path, fps, w, h)
    print(f"[overlay_movement] ids: {ids}, speed unit: {speed_unit}, writing {codec_label} -> {actual_path}")

    frame_idx = 0
    for _ in range(n_frames):
        ok, frame = video_cap.read()
        if not ok:
            break
        frame_float = frame.astype(np.float32)
        rows_here = by_frame.get(frame_idx, {})

        for pid in ids:
            ok_m, mraw = mask_caps[pid].read()
            if not ok_m:
                continue
            gray = mraw[:, :, 0] if mraw.ndim == 3 else mraw
            hard_mask = gray > threshold
            if not hard_mask.any():
                continue

            color = get_track_color(pid)
            # soft edge instead of the raw pixel mask -- same technique as
            # overlay_subvideo.py, so the two overlay tools look consistent.
            soft = cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32) / 255.0
            soft = np.clip(soft, 0.0, 1.0) * alpha
            color_arr = np.array(color, dtype=np.float32)
            frame_float = frame_float * (1 - soft[..., None]) + color_arr[None, None, :] * soft[..., None]

            ys, xs = np.where(hard_mask)
            position = np.array([xs.min(), ys.min()])

            row = rows_here.get(pid)
            speed_text = _format_speed(row.velocity_mm_s if row is not None else np.nan, speed_unit,
                                        deadzone_mm_s=still_deadzone_mm_s)
            is_outlier = bool(row.is_outlier) if row is not None else False

            frame = frame_float.astype(np.uint8)
            _draw_speed_label(frame, position, pid, color, speed_text, is_outlier)
            frame_float = frame.astype(np.float32)

        frame = frame_float.astype(np.uint8)
        if show_legend:
            _draw_legend(frame, speed_unit)
        writer.write(frame)
        frame_idx += 1

    writer.release()
    video_cap.release()
    for c in mask_caps.values():
        c.release()
    print(f"[overlay_movement] {frame_idx} frames written -> {actual_path}")
    return actual_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlays masks + per-frame speed (from movement_metrics.py) on the source video, "
                     "for an ad-occhio sanity check of the root-smoothing fix -- see module docstring.")
    parser.add_argument("--video", required=True, help="Original source video the MaskDir was produced from.")
    parser.add_argument("--mask-dir", required=True, help="MaskDir with <id>.mp4 files (e.g. a merge_fragments.py output).")
    parser.add_argument("--movement-csv", required=True, help="movement_per_frame.csv from pose/movement_metrics.py "
                                                                "(needs --out-per-frame passed when it was generated).")
    parser.add_argument("--out", required=True, help="Output video path (extension is only a hint -- see pose/video_writer.py).")
    parser.add_argument("--session-id", default=None, help="Pick one session if --movement-csv covers more than one.")
    parser.add_argument("--threshold", type=int, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--speed-unit", choices=["kmh", "ms"], default="ms",
                         help="ms (default) or kmh -- m/s keeps the gap between 0 and the deadzone threshold "
                              "small in absolute terms (0.00 -> ~0.27), which reads less jarring on screen "
                              "than the equivalent km/h jump (0.0 -> ~1.0) even though it's the same threshold.")
    parser.add_argument("--no-legend", action="store_true", help="Skip the compact corner legend.")
    parser.add_argument("--still-deadzone-kmh", type=float, default=DEFAULT_STILL_DEADZONE_KMH,
                         help="The on-screen speed shows as 0.0 at or below this (km/h-equivalent, applied "
                              "regardless of --speed-unit) -- a seated person's residual sensor jitter that "
                              "survived root-smoothing, not real movement. 0 disables it.")
    args = parser.parse_args()

    actual_path = render_movement_overlay(
        video_path=args.video, mask_dir=args.mask_dir, movement_csv=args.movement_csv,
        out_path=args.out, session_id=args.session_id, threshold=args.threshold, alpha=args.alpha,
        speed_unit=args.speed_unit, show_legend=not args.no_legend, still_deadzone_kmh=args.still_deadzone_kmh,
    )
    print(f"Done: {actual_path}")


if __name__ == "__main__":
    main()
