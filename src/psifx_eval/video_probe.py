"""
psifx_eval/video_probe.py
============================
Tiny, dependency-light video metadata helper shared across `psifx_eval`
scripts (`run_overlap_experiment.py`, `run_sam31_native.py`, and
previously `run_baseline_vs_oracle.py` before it was retired in favor of
a single unified 4-way comparison script). Split out on its own so it
isn't tied to any one experiment script's lifetime.

Deliberately self-contained (not imported from webui/api.py's
`probe_video_metadata`) so `psifx_eval` doesn't pull in the webui
package as a dependency for an unrelated purpose.
"""

from __future__ import annotations

import cv2


def probe_total_frames(video_path: str) -> int:
    """Container frame count only (no decoding) -- just enough to pick
    an oracle `chunk_size` that's guaranteed to cover the whole video in
    one psifx chunk, or to pad a MaskDir written by a non-psifx tracker
    (see `run_sam31_native.py`) to the full video length."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(
                f"Container for {video_path!r} doesn't declare a usable frame count "
                f"({frame_count}) -- pass an explicit chunk size instead of relying "
                f"on auto-detection."
            )
        return frame_count
    finally:
        cap.release()
