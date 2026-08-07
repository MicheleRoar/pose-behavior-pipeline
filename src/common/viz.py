"""
viz.py
======
Drawing utilities shared between the live pipeline (`live_demo.py`, which
reads from Canon R8 / webcam / video file) and the verification scripts
runnable in environments without a camera (see
`tests/live_render_check.py`), so the same rendering logic is tested even
where a real video source isn't available.
"""

from __future__ import annotations

import cv2
import numpy as np

from pose.keypoints import KP, SKELETON_EDGES

# Palette of distinct colors (BGR, as OpenCV expects) assigned cyclically
# per track_id, so each person in the frame has a different, recognizable
# color at a glance (skeleton, metrics-box border, ID label) instead of
# the fixed green used for everyone before this function.
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


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Stable, distinct color for a given track_id, cycling through the
    palette. Used to visually differentiate multiple tracked people in
    the same frame (skeleton, metrics box, label)."""
    return TRACK_COLOR_PALETTE[track_id % len(TRACK_COLOR_PALETTE)]


def draw_person_label(frame: np.ndarray, position: np.ndarray, track_id: int,
                       color: tuple[int, int, int], is_target: bool = False) -> np.ndarray:
    """Draws a large, readable "ID N" label above the person's head, with
    a colored background (same color as their skeleton) — more prominent
    than just small text in the metrics box. If `is_target` is True
    (person selected with --target-track-id), adds an extra visual
    indicator.
    """
    text = f"ID {track_id}" + (" ★ TARGET" if is_target else "")
    x, y = int(position[0]), max(int(position[1]) - 20, 20)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x - 6, y - th - 10), (x + tw + 6, y + 6), color, -1)
    cv2.rectangle(frame, (x - 6, y - th - 10), (x + tw + 6, y + 6), (0, 0, 0), 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, conf: np.ndarray | None = None,
                   color: tuple[int, int, int] = (0, 220, 0), conf_threshold: float = 0.3) -> np.ndarray:
    """Draws keypoints and skeleton connections on a frame (in place)."""
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


def text_block_size(lines: list[str], font_scale: float = 0.5) -> tuple[int, int]:
    """Size (width, height) in pixels of the box that `draw_text_block`
    would draw for these lines -- used to stack multiple boxes (one per
    person) without overlapping them, without duplicating the sizing
    logic."""
    line_height = int(22 * font_scale / 0.5)
    box_h = line_height * len(lines) + 10
    box_w = max((len(l) for l in lines), default=0) * int(9 * font_scale / 0.5) + 16
    return box_w, box_h


def draw_text_block(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 10),
                     font_scale: float = 0.5, color: tuple[int, int, int] = (255, 255, 255),
                     border_color: tuple[int, int, int] | None = None) -> np.ndarray:
    """Draws a multi-line text block with a semi-transparent background for
    readability (used to show each person's live metrics). If
    `border_color` is specified, also draws a border of that color
    (typically the same color as the person's skeleton, see
    `get_track_color`) to visually link the box to the corresponding
    person when there's more than one in the frame.
    """
    x, y = origin
    line_height = int(22 * font_scale / 0.5)
    box_w, box_h = text_block_size(lines, font_scale)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if border_color is not None:
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), border_color, 2)

    for i, line in enumerate(lines):
        ty = y + 18 + i * line_height
        cv2.putText(frame, line, (x + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return frame


def draw_hand(frame: np.ndarray, hand_xy: np.ndarray, color: tuple[int, int, int] = (255, 200, 0)) -> np.ndarray:
    """Draws the 21 landmarks of a hand and their connections.
    HAND_CONNECTIONS import delayed to avoid a heavy import of
    `hands.py` (and therefore of mediapipe) when not needed.
    """
    from pose.hands import HAND_CONNECTIONS

    for a_idx, b_idx in HAND_CONNECTIONS:
        a, b = hand_xy[a_idx], hand_xy[b_idx]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, 2, cv2.LINE_AA)
    for pt in hand_xy:
        if np.isnan(pt).any():
            continue
        cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 100, 255), -1, cv2.LINE_AA)
    return frame


def draw_face_signals(frame: np.ndarray, mouth_pts: np.ndarray | None = None,
                       left_eye_pts: np.ndarray | None = None,
                       right_eye_pts: np.ndarray | None = None,
                       left_eyebrow_pts: np.ndarray | None = None,
                       right_eyebrow_pts: np.ndarray | None = None,
                       mouth_color: tuple[int, int, int] = (0, 220, 255),
                       eye_color: tuple[int, int, int] = (255, 220, 0),
                       eyebrow_color: tuple[int, int, int] = (0, 140, 255)) -> np.ndarray:
    """Draws a small overlay for mouth, eyes, and eyebrows (points +
    lines), from the MediaPipe landmarks already extracted in
    `gaze_head.py` (MOUTH_TOP/BOTTOM/LEFT/RIGHT, *_EYE_EAR_IDX,
    *_EYEBROW_IDX indices).

    `mouth_pts`: array (4, 2) in order [top, bottom, left, right].
    `left_eye_pts`/`right_eye_pts`: array (6, 2) in EAR order
    [left_corner, top1, top2, right_corner, bottom2, bottom1].
    `left_eyebrow_pts`/`right_eyebrow_pts`: array (5, 2), eyebrow outline
    (open polyline, not closed like the eyes).
    Without this overlay, mouth/eyes/eyebrows only appeared as a number
    in the text box, with no mark drawn on the face -- unlike head
    (arrow) and hands (skeleton).
    """
    if mouth_pts is not None and not np.isnan(mouth_pts).any():
        top, bottom, left, right = mouth_pts
        cv2.line(frame, tuple(top.astype(int)), tuple(bottom.astype(int)), mouth_color, 2, cv2.LINE_AA)
        cv2.line(frame, tuple(left.astype(int)), tuple(right.astype(int)), mouth_color, 1, cv2.LINE_AA)
        for pt in mouth_pts:
            cv2.circle(frame, tuple(pt.astype(int)), 2, mouth_color, -1, cv2.LINE_AA)

    for eye_pts in (left_eye_pts, right_eye_pts):
        if eye_pts is None or np.isnan(eye_pts).any():
            continue
        pts = eye_pts.astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)

    for brow_pts in (left_eyebrow_pts, right_eyebrow_pts):
        if brow_pts is None or np.isnan(brow_pts).any():
            continue
        pts = brow_pts.astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=False, color=eyebrow_color, thickness=2, lineType=cv2.LINE_AA)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1] - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
