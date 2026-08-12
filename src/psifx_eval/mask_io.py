"""
psifx_eval/mask_io.py
=======================
Reads psifx's own MaskDir output format -- one lossless MP4 per tracked
identity, named `<global_id>.mp4`, white-on-black binary masks, every
file the SAME length as the source video (psifx back-fills frames
before an identity first appears with empty/black frames -- see
`Sam3TrackingTool._write_chunk_masks` in the real psifx source,
psifx/video/tracking/sam3/tool.py). This is exactly the format already
used elsewhere in this ecosystem (see LiubovRev/Video-Annotation-System's
`MaskDir/`), so reading it here means `id_metrics.py` can be pointed at
ANY real psifx run's output directory, not just ones produced by this
project's own orchestration script.

Deliberately separate from `id_metrics.py`: this module is pure I/O
(disk -> in-memory boolean arrays), no comparison logic, so it can be
tested with tiny synthetic mask videos, and swapped out later for a
streaming/lower-memory reader without touching the metrics code.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

# psifx writes and expects plain "<int>.mp4" filenames (see
# Sam3TrackingTool._write_chunk_masks: `mask_dir / f"{global_obj_id}.mp4"`)
# -- anything else in the directory (a tracking visualization, a
# processed_video.mp4, etc.) is intentionally ignored here rather than
# raising, since MaskDir is sometimes a sibling of other project files.
_MASK_FILENAME_RE = re.compile(r"^(\d+)\.mp4$")


def read_mask_video(path: str | Path, *, threshold: int = 127) -> np.ndarray:
    """Reads one psifx mask video into a `(T, H, W)` boolean array.
    psifx writes masks as grayscale-equivalent white-on-black RGB
    (`mask.astype(uint8) * 255`, repeated across 3 channels) -- any
    reasonable threshold works since there's no real anti-aliasing, but
    a mid-range one (default 127) is robust to mild video-codec
    compression noise around the mask edges."""
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


def load_mask_dir(mask_dir: str | Path, *, threshold: int = 127) -> dict[int, np.ndarray]:
    """Reads every `<id>.mp4` in a psifx MaskDir into `{global_id: (T, H, W)
    bool array}`. Skips non-matching files (see `_MASK_FILENAME_RE`)
    rather than erroring, since a real MaskDir can contain other
    artifacts. Raises `ValueError` if the directory has zero mask
    files at all -- an empty MaskDir almost always means psifx found no
    detections for the whole video (a real, worth-surfacing failure,
    not something to silently return `{}` for)."""
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
        # psifx back-fills every identity's video to the SAME total
        # frame count (see module docstring) -- if that's not true here,
        # something upstream (a partial/interrupted run, a hand-edited
        # MaskDir) broke the invariant id_metrics.py's frame-by-frame
        # alignment depends on.
        raise ValueError(
            f"Mask videos in {mask_dir} have inconsistent frame counts: {lengths} -- "
            f"expected all identities back-filled to the same total length (see "
            f"psifx's Sam3TrackingTool._write_chunk_masks)."
        )

    return masks
