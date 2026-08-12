"""
common/video_writer.py
========================
Opens a `cv2.VideoWriter` for annotated-output MP4s, preferring a real
H.264 encoder ('avc1' fourcc) over OpenCV's default 'mp4v'.

Why this exists (Michele, 2026-08): the Compare runs window's videos
loaded fine on Linux once compare.js was switched from file:// to a
local HTTP server (see local_media_server.py), but pressing Play did
nothing -- no error anywhere, just a black box. Root cause, confirmed
with `ffprobe` on a file exported by `Api.export_video()`:
`cv2.VideoWriter_fourcc(*"mp4v")` does NOT produce H.264 -- it produces
`codec_name=mpeg4` (old MPEG-4 Part 2 / DivX-era). No mainstream browser
engine's <video> tag decodes that codec; Chromium/WebKit only support
H.264/H.265/VP8/VP9/AV1. This silently "worked" on macOS because
WKWebView plays video through AVFoundation, which still has legacy
MPEG-4 Part 2 decode support built into the OS -- Linux's GStreamer-based
decoders (used by both WebKitGTK and QtWebEngine) generally don't, so the
file loaded (no permission error) but simply couldn't be decoded, and
`el.play()` failed silently (see compare.js's `.catch(() => {})`).

'avc1' is the standard fourcc for real H.264 in an MP4 container --
playable everywhere IF OpenCV's FFmpeg build actually has a usable H.264
ENCODER (needs libx264 or a driveable hardware encoder -- NOT guaranteed
on every machine/OpenCV build, confirmed in this project's own sandbox:
'avc1'/'H264'/'X264' all fail to open here, only 'mp4v' does). So this
tries 'avc1' first and only falls back to 'mp4v' -- loudly, with a
printed warning -- rather than silently producing another unplayable
file, which is exactly the failure mode this module exists to catch.
"""

from __future__ import annotations

import cv2

# Tried in order; first one whose VideoWriter actually opens wins. 'avc1'
# is real H.264 (what browsers need); 'mp4v' is the guaranteed-available
# fallback (works with any FFmpeg build) but produces browser-incompatible
# MPEG-4 Part 2 -- see module docstring.
_CANDIDATES = (("avc1", "h264"), ("mp4v", "mpeg4"))


def open_annotated_video_writer(path: str, fps: float, width: int, height: int):
    """Returns `(writer, codec_label)` where `codec_label` is `"h264"` or
    `"mpeg4"` (the latter meaning: opened, but likely won't play back in
    an HTML5 `<video>` element on Linux -- see module docstring; callers
    should surface this to the user, not just silently accept it).
    Raises `RuntimeError` if no candidate opens at all (e.g. an
    unwritable path)."""
    for fourcc_str, label in _CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            if label == "mpeg4":
                print(
                    f"[video_writer] WARNING: no H.264 encoder available in this "
                    f"OpenCV/FFmpeg build -- falling back to 'mp4v' (MPEG-4 Part 2) "
                    f"for {path!r}. This file will likely NOT play in an HTML5 "
                    f"<video> element on Linux (GStreamer-based decoders there "
                    f"typically lack MPEG-4 Part 2 support, even though the file "
                    f"itself is perfectly valid) -- install/build FFmpeg with "
                    f"libx264 to get real H.264 output instead."
                )
            return writer, label
        writer.release()
    raise RuntimeError(
        f"Could not open {path!r} for writing with either an H.264 or MPEG-4 "
        f"encoder (unwritable path, or no working video encoder in this "
        f"OpenCV/FFmpeg build at all)."
    )
