"""
video_writer_check.py
=======================
Verifies `common/video_writer.py::open_annotated_video_writer` -- the
fix for two real bugs found together (Michele, 2026-08):

1. Annotated videos exported by `Api.export_video()` /
   `export_backend_comparisons.py` used `cv2.VideoWriter_fourcc(*"mp4v")`,
   which produces MPEG-4 Part 2, not H.264 -- undecodable by pretty much
   any modern browser engine.
2. Switching to real H.264 wouldn't have fully fixed it anyway: this
   project's Linux backend (pywebview[qt] -> QtWebEngine) never includes
   H.264 in its stock pip build at all (patent licensing). VP9/.webm is
   royalty-free and IS included -- see the module docstring for the full
   story.

This checks that the function opens SOME working writer, reports which
codec it actually used, and returns the RIGHT extension for that codec
(not just whatever was passed in) -- it does NOT assert which codec wins
on any given machine (that depends on the local OpenCV/FFmpeg build).
This sandbox happens to have a working VP9 encoder (libvpx, bundled by
virtually every FFmpeg build) but no H.264 one at all -- part1 accepts
either "vp9", "h264", or "mpeg4" and just verifies internal consistency
(the extension always matches the reported codec) rather than assuming
one specific outcome.

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

_EXPECTED_EXT = {"vp9": ".webm", "h264": ".mp4", "mpeg4": ".mp4"}


def part1_returns_a_working_writer_with_a_matching_extension():
    # Passed in as ".mp4" -- if VP9 wins (as it does in this sandbox,
    # and as it should on the user's real machine, see module docstring),
    # the ACTUAL path must come back as ".webm", not ".mp4": a codec must
    # match its container, and callers are told to always use the
    # returned path for exactly this reason.
    base = tempfile.mktemp(suffix=".mp4")
    try:
        writer, actual_path, codec = open_annotated_video_writer(base, fps=10.0, width=32, height=32)
        assert codec in ("vp9", "h264", "mpeg4"), f"unexpected codec label: {codec!r}"
        assert actual_path.endswith(_EXPECTED_EXT[codec]), (
            f"codec {codec!r} should produce a {_EXPECTED_EXT[codec]!r} file, "
            f"got {actual_path!r}"
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for _ in range(5):
            writer.write(frame)
        writer.release()
        assert os.path.exists(actual_path) and os.path.getsize(actual_path) > 0, \
            "expected a non-empty video file"

        cap = cv2.VideoCapture(actual_path)
        try:
            assert cap.isOpened(), "the written file should itself be re-openable/decodable by OpenCV"
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert count == 5, f"expected 5 frames written, container reports {count}"
    finally:
        for p in (base, os.path.splitext(base)[0] + ".webm", os.path.splitext(base)[0] + ".mp4"):
            if os.path.exists(p):
                os.remove(p)
    print(f"Part 1: open_annotated_video_writer() opens a working writer whose actual path "
          f"extension always matches the reported codec (got {codec!r} -> {_EXPECTED_EXT[codec]!r} "
          f"on this machine's OpenCV/FFmpeg build) — OK")


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
    part1_returns_a_working_writer_with_a_matching_extension()
    part2_unwritable_path_raises_a_clear_runtime_error()
    print("\nVerification completed with no errors: open_annotated_video_writer() opens a "
          "real, working video writer, always reports which codec it actually used, and the "
          "returned path's extension always matches that codec's container.")
