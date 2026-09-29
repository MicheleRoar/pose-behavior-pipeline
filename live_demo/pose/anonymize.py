"""
anonymize.py
============
Face obfuscation based on head keypoints (nose, eyes, ears), to be applied
to frames BEFORE saving or sharing any video/clip involving minors.

Approach: since the head keypoints are already available from the pose
estimation model's output, no separate face detector is needed: a radius
is estimated from the shoulder-to-shoulder distance (a proxy for the
person's scale in the image) and a strong Gaussian blur is applied to the
corresponding region.

This does NOT replace a data handling policy approved by the ethics
committee; it's a technical defense-in-depth measure to apply regardless.
"""

from __future__ import annotations

import numpy as np
import cv2

from pose.keypoints import KP, HEAD_KEYPOINTS


def _valid_points(frame_kpts: np.ndarray, conf: np.ndarray | None, names: list[str],
                   conf_threshold: float = 0.3) -> np.ndarray:
    idxs = [KP[n] for n in names]
    pts = frame_kpts[idxs]
    if conf is not None:
        mask = conf[idxs] >= conf_threshold
        pts = pts[mask]
    return pts[~np.isnan(pts).any(axis=1)] if len(pts) else pts


def estimate_head_radius(frame_kpts: np.ndarray) -> float:
    """Estimates a reasonable radius for the face blur using the
    shoulder-to-shoulder distance as a scale reference (more robust than
    the eye-to-eye distance alone, which can be very small or absent in
    profile view)."""
    l_sh, r_sh = frame_kpts[KP["left_shoulder"]], frame_kpts[KP["right_shoulder"]]
    shoulder_width = np.linalg.norm(l_sh - r_sh)
    if not np.isfinite(shoulder_width) or shoulder_width < 1e-3:
        return 40.0  # pixel fallback, should be adapted to the video resolution
    return float(0.6 * shoulder_width)


def blur_face(frame: np.ndarray, frame_kpts: np.ndarray,
              conf: np.ndarray | None = None) -> np.ndarray:
    """Applies a strong Gaussian blur to the face region of ONE person,
    located from their head keypoints.

    Parameters
    ----------
    frame : BGR image (as returned by OpenCV)
    frame_kpts : (17, 2) array for the current person
    conf : optional (17,) array of per-keypoint confidences

    Returns
    -------
    The frame with the face region blurred (a copy, it doesn't modify the
    original array in place unless explicitly intended).
    """
    head_pts = _valid_points(frame_kpts, conf, HEAD_KEYPOINTS)
    if len(head_pts) == 0:
        return frame

    center = head_pts.mean(axis=0)
    radius = estimate_head_radius(frame_kpts)

    x, y = int(center[0]), int(center[1])
    r = max(int(radius), 10)

    h, w = frame.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)
    if x1 <= x0 or y1 <= y0:
        return frame

    out = frame.copy()
    roi = out[y0:y1, x0:x1]
    # odd kernel, proportional to the region
    k = max(3, (min(roi.shape[:2]) // 2) | 1)
    out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out
