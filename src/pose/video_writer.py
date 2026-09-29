"""
pose/video_writer.py
======================
Opens a `cv2.VideoWriter` for annotated-output videos, preferring VP9/
WebM over H.264/MP4 or the old MPEG-4 Part 2 'mp4v' default. Ported
verbatim from `pose-behavior-pipeline_legacy/src/common/video_writer.py`
so `pose/visualize.py`'s overlay video doesn't reintroduce a bug already
found and fixed there.

Why this exists (real bug, found in the legacy pipeline, Michele 2026-08):
a video written with `cv2.VideoWriter_fourcc(*"mp4v")` (the old, common
default) does NOT produce H.264 -- it produces `codec_name=mpeg4` (old
MPEG-4 Part 2 / DivX-era), which no mainstream browser engine's <video>
tag decodes. It played fine in some tools (e.g. a plain file:// open in
VLC) and silently failed (black box, no error) in others -- exactly the
kind of failure mode you don't want on a QA/sanity-check video meant to
be shared and watched.

Why VP9/WebM specifically and not just switch to real H.264: real H.264
encoders are often NOT bundled in a stock OpenCV/FFmpeg build (patent-
licensing reasons), unlike VP9 (libvpx), which is bundled in virtually
every FFmpeg build, this sandbox's included. VP9-in-WebM is royalty-free,
decodes in every modern browser and in Safari/WKWebView (VP9 support
since Safari 14.1), and needs no system package or custom build.
"""

from __future__ import annotations

import os

import cv2

# Tried in order; first one whose VideoWriter actually opens wins. Each
# entry is (fourcc, container extension, codec label) -- the extension
# MUST match the codec's container, so `open_annotated_video_writer` may
# return a different path than the one it was asked for (see its
# docstring: always use the returned `actual_path`).
_CANDIDATES = (
    ("VP90", ".webm", "vp9"),    # preferred: royalty-free, works everywhere this
                                  # project runs without any system package -- see
                                  # module docstring.
    ("avc1", ".mp4", "h264"),    # real H.264, if this machine happens to have an
                                  # encoder for it.
    ("mp4v", ".mp4", "mpeg4"),   # last resort -- opens on almost any FFmpeg build,
                                  # but NOT decodable by <video> pretty much anywhere
                                  # modern, see warning below.
)


def open_annotated_video_writer(path: str, fps: float, width: int, height: int):
    """Returns `(writer, actual_path, codec_label)`. `path`'s extension
    is only a hint for the preferred candidate (VP9/.webm) -- if a
    fallback is needed, the extension is swapped to match ITS container
    (a codec must match its container). ALWAYS use `actual_path`, not
    the `path` that was passed in, for anything shown to the user or
    reopened later -- it can differ if a fallback kicked in.
    `codec_label` is `"vp9"`, `"h264"`, or `"mpeg4"` (the last one
    printed as a loud warning, since it means this file likely won't
    play back in most browsers). Raises `RuntimeError` if no candidate
    opens at all (e.g. an unwritable path)."""
    stem = os.path.splitext(path)[0]
    for fourcc_str, ext, label in _CANDIDATES:
        actual_path = stem + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(actual_path, fourcc, fps, (width, height))
        if writer.isOpened():
            if label == "mpeg4":
                print(
                    f"[video_writer] WARNING: no VP9 or H.264 encoder available in "
                    f"this OpenCV/FFmpeg build -- falling back to 'mp4v' (MPEG-4 "
                    f"Part 2) for {actual_path!r}. This file will likely NOT play "
                    f"back in most browsers (see pose/video_writer.py's module "
                    f"docstring) -- install/build FFmpeg with libvpx (VP9) or "
                    f"libx264 (H.264) to fix this."
                )
            return writer, actual_path, label
        writer.release()
    raise RuntimeError(
        f"Could not open {path!r} for writing with a VP9, H.264, or MPEG-4 encoder "
        f"(unwritable path, or no working video encoder in this OpenCV/FFmpeg build "
        f"at all)."
    )
