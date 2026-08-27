"""
segmentation/merging/mask_io.py
================================
Reads/writes psifx's MaskDir format: one lossless MP4 per tracked
identity (`<id>.mp4`), white-on-black binary masks, every file padded
to the same length as the source video. Pure I/O, no comparison logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

# psifx writes plain "<int>.mp4" filenames; anything else in the
# directory is ignored rather than raising.
_MASK_FILENAME_RE = re.compile(r"^(\d+)\.mp4$")

# white-on-black decode threshold shared by every mask-reading function
# in this module.
DEFAULT_MASK_THRESHOLD = 127


def read_mask_video(path: str | Path, *, threshold: int = DEFAULT_MASK_THRESHOLD) -> np.ndarray:
    """Reads one mask video into a `(T, H, W)` boolean array."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open mask video: {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = frame[:, :, 0] if frame.ndim == 3 else frame
            frames.append(gray > threshold)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Mask video has no frames: {path}")
    return np.stack(frames, axis=0)


def load_mask_dir(mask_dir: str | Path, *, threshold: int = DEFAULT_MASK_THRESHOLD) -> dict[int, np.ndarray]:
    """Reads every `<id>.mp4` in a MaskDir into `{id: (T, H, W) bool array}`,
    eagerly (whole dir in RAM -- for streaming use `_list_mask_files`/
    `_scan_track`/`_read_mask_frame` below instead). Raises if the
    directory has no mask files, or if the files disagree on frame count
    (psifx is expected to back-fill every id to the same length)."""
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {mask_dir}")

    masks: dict[int, np.ndarray] = {}
    for entry in sorted(mask_dir.iterdir()):
        match = _MASK_FILENAME_RE.match(entry.name)
        if not match:
            continue
        global_id = int(match.group(1))
        masks[global_id] = read_mask_video(entry, threshold=threshold)

    if not masks:
        raise ValueError(
            f"No '<id>.mp4' mask files found in {mask_dir} -- expected psifx's "
            f"MaskDir output format (e.g. '0.mp4', '1.mp4', ...)."
        )

    lengths = {global_id: arr.shape[0] for global_id, arr in masks.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Mask videos in {mask_dir} have inconsistent frame counts: {lengths} -- "
            f"expected all identities back-filled to the same total length."
        )

    return masks


# ---------------------------------------------------------------------------
# Streaming variants: read a MaskDir's <id>.mp4 files without decoding a
# whole track into memory (a full clinical session doesn't fit in RAM).
# ---------------------------------------------------------------------------


def _list_mask_files(mask_dir: str) -> dict[int, Path]:
    """`{id: path}` for every `<id>.mp4` in `mask_dir`, without decoding
    anything (unlike `load_mask_dir`)."""
    dir_path = Path(mask_dir)
    paths: dict[int, Path] = {}
    if not dir_path.is_dir():
        return paths
    for p in dir_path.iterdir():
        m = _MASK_FILENAME_RE.match(p.name)
        if m:
            paths[int(m.group(1))] = p
    return paths


def _scan_track(
    mask_path: Path,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> tuple[int, int, int, int, int, int] | None:
    """Streams one id's mask video once (O(1) memory) and returns
    `(first, last, real_frames, decoded_frame_count, height, width)`,
    or `None` if the track has no non-empty frame at all.
    `decoded_frame_count` is measured by actually walking the file
    rather than trusting `cv2.CAP_PROP_FRAME_COUNT` metadata, which has
    disagreed with the real count in practice."""
    cap = cv2.VideoCapture(str(mask_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open mask video: {mask_path}")

    first = last = None
    real_frames = 0
    decoded = 0
    height = width = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if height is None:
                height, width = frame.shape[0], frame.shape[1]
            gray = frame[:, :, 0] if frame.ndim == 3 else frame
            if bool((gray > threshold).any()):
                if first is None:
                    first = decoded
                last = decoded
                real_frames += 1
            decoded += 1
    finally:
        cap.release()

    if first is None:
        return None
    return first, last, real_frames, decoded, height, width


def _read_mask_frame(
    cap: cv2.VideoCapture,
    frame_idx: int,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> np.ndarray | None:
    """Seeks an already-open mask-file `VideoCapture` to `frame_idx` and
    returns the decoded boolean mask frame, or `None` on a failed
    seek/read (past end of file, corrupt frame)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, raw = cap.read()
    if not ok:
        return None
    gray = raw[:, :, 0] if raw.ndim == 3 else raw
    return gray > threshold
