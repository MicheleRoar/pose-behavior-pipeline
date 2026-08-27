"""
segmentation/tools/subvideo.py
============================
Cuts a time window out of an existing (video, MaskDir) pair -- writes a
short video clip plus a matching windowed MaskDir, so `merge_fragments`
can be run/tuned on a few minutes of footage instead of a full session.
Used by `run_osnet_window.py`; split into its own module 2026-08-27 so
that CLI is a thin orchestrator (extract window -> call
`merge_fragments`) instead of one script mixing both concerns.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_video_window(video_path: str, start_frame: int, end_frame: int, out_path: Path) -> float:
    """Writes frames `[start_frame, end_frame)` of `video_path` to
    `out_path`. Returns the source fps (needed by the caller to convert
    minutes to frame indices, and to write the mask window at the same
    rate)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()
    return fps


def extract_mask_window(
    mask_path: Path, start_frame: int, end_frame: int, out_path: Path, fps: float, threshold: int = 127,
) -> bool:
    """Same window cut as `extract_video_window`, for one `<id>.mp4`
    mask file. Returns `False` (and writes nothing) if the id has no
    non-empty frame anywhere in the window -- an id with no signal in
    this specific time slice isn't worth carrying into the windowed
    MaskDir at all."""
    cap = cv2.VideoCapture(str(mask_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames, has_content = [], False
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        gray = frame[:, :, 0] if frame.ndim == 3 else frame
        if bool((gray > threshold).any()):
            has_content = True
        frames.append(frame)
    cap.release()
    if not has_content:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True
