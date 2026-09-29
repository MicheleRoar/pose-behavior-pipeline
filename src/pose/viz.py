"""
pose/viz.py
============
Drawing utilities for rendering the COCO-17 skeleton onto video frames.
Ported from `pose-behavior-pipeline_legacy/src/common/viz.py`, trimmed to
just what `pose/visualize.py` needs (skeleton + per-person label/color) --
the legacy module also has hand/face-landmark drawing helpers, not
applicable here since this repo doesn't have `pose/hands.py`/`gaze_head.py`.
"""

from __future__ import annotations

import cv2
import numpy as np

from pose.keypoints import KP, SKELETON_EDGES

# Palette of distinct colors (BGR, as OpenCV expects) assigned cyclically
# per global_person_id, so each tracked identity has a different,
# recognizable color at a glance (skeleton + ID label) instead of a fixed
# color for everyone.
TRACK_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (0, 220, 0),      # green
    (0, 140, 255),    # orange
    (255, 0, 255),    # magenta
    (255, 220, 0),    # cyan
    (0, 255, 255),    # yellow
    (255, 0, 0),      # blue
    (180, 105, 255),  # pink
    (0, 128, 128),    # olive/teal
]


def get_track_color(person_id: int) -> tuple[int, int, int]:
    """Stable, distinct color for a given `global_person_id`, cycling
    through the palette."""
    return TRACK_COLOR_PALETTE[person_id % len(TRACK_COLOR_PALETTE)]


def draw_person_label(frame: np.ndarray, position: np.ndarray, person_id: int,
                       color: tuple[int, int, int]) -> np.ndarray:
    """Draws a readable "ID N" label near `position` (typically the
    person's topmost visible keypoint), with a colored background
    matching their skeleton color."""
    text = f"ID {person_id}"
    x, y = int(position[0]), max(int(position[1]) - 20, 20)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x - 6, y - th - 10), (x + tw + 6, y + 6), color, -1)
    cv2.rectangle(frame, (x - 6, y - th - 10), (x + tw + 6, y + 6), (0, 0, 0), 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, conf: np.ndarray | None = None,
                   color: tuple[int, int, int] = (0, 220, 0), conf_threshold: float = 0.3) -> np.ndarray:
    """Draws keypoints and skeleton connections on a frame (in place).
    Plain, single-color version -- see `pose/visualize.py`'s own
    `_draw_skeleton_with_interp` for the two-color (real vs. interpolated)
    variant used there."""
    def ok(idx: int) -> bool:
        if conf is None:
            return True
        return conf[idx] >= conf_threshold

    for a_name, b_name in SKELETON_EDGES:
        a_idx, b_idx = KP[a_name], KP[b_name]
        if not (ok(a_idx) and ok(b_idx)):
            continue
        a, b = kpts[a_idx], kpts[b_idx]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, 2, cv2.LINE_AA)

    for idx in range(kpts.shape[0]):
        if not ok(idx) or np.isnan(kpts[idx]).any():
            continue
        cv2.circle(frame, tuple(kpts[idx].astype(int)), 3, (0, 165, 255), -1, cv2.LINE_AA)

    return frame
