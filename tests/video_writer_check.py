"""
video_writer_check.py
=======================
Verifies `common/video_writer.py::open_annotated_video_writer` -- the
fix for a real bug (Michele, 2026-08): annotated videos exported by
`Api.export_video()` / `export_backend_comparisons.py` used
`cv2.VideoWriter_fourcc(*"mp4v")`, which produces MPEG-4 Part 2, not
H.264 -- undecodable by an HTML5 <video> element on Linux (see the
module docstring for the full story). This only checks that the
function opens SOME working writer and reports which codec it actually
used -- it does NOT assert which codec wins on any given machine (that
depends on the local OpenCV/FFmpeg build, e.g. this sandbox has no
libx264 and always falls back to 'mpeg4', see part1's assertion is
deliberately permissive about that).

Run with: python video_writer_check.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from common.video_writer import open_annotated_video_writer


def part1_returns_a_working_writer_and_a_known_codec_label():
    out_path = tempfile.mktemp(suffix=".mp4")
    try:
        writer, codec = open_annotated_video_writer(out_path, fps=10.0, width=32, height=32)
        assert codec in ("h264", "mpeg4"), f"unexpected codec label: {codec!r}"
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for _ in range(5):
            writer.write(frame)
        writer.release()
        assert os.path.exists(out_path) and os.path.getsize(out_path) > 0, \
            "expected a non-empty video file"

        cap = cv2.VideoCapture(out_path)
        try:
            assert cap.isOpened(), "the written file should itself be re-openable/decodable by OpenCV"
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert count == 5, f"expected 5 frames written, container reports {count}"
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    print(f"Part 1: open_annotated_video_writer() opens a working writer and reports a real "
          f"codec label (got {codec!r} on this machine's OpenCV/FFmpeg build) — OK")


def part2_unwritable_path_raises_a_clear_runtime_error():
    # A directory that doesn't exist -- no fourcc can open a file there,
    # this must surface as a RuntimeError, not a silent unusable writer.
    bad_path = os.path.join(tempfile.mktemp(), "nested", "does", "not", "exist.mp4")
    try:
        open_annotated_video_writer(bad_path, fps=10.0, width=32, height=32)
        raise AssertionError("expected a RuntimeError for an unwritable path")
    except RuntimeError as exc:
        assert "Could not open" in str(exc), str(exc)
    print("Part 2: an unwritable path raises a clear RuntimeError instead of returning a "
          "silently-broken writer — OK")


if __name__ == "__main__":
    part1_returns_a_working_writer_and_a_known_codec_label()
    part2_unwritable_path_raises_a_clear_runtime_error()
    print("\nVerification completed with no errors: open_annotated_video_writer() opens a "
          "real, working video writer and always reports which codec it actually used.")
