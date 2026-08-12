"""
common/video_writer.py
========================
Opens a `cv2.VideoWriter` for annotated-output videos, preferring VP9/
WebM over H.264/MP4 or the old MPEG-4 Part 2 'mp4v' default.

Why this exists (Michele, 2026-08): the Compare runs window's videos
loaded fine on Linux once compare.js was switched from file:// to a
local HTTP server (see local_media_server.py), but pressing Play did
nothing -- no error anywhere, just a black box. Root cause, confirmed
with `ffprobe` on a file exported by `Api.export_video()`:
`cv2.VideoWriter_fourcc(*"mp4v")` does NOT produce H.264 -- it produces
`codec_name=mpeg4` (old MPEG-4 Part 2 / DivX-era), which no mainstream
browser engine's <video> tag decodes.

Why VP9/WebM and not just switch to real H.264: this project pins
`pywebview[qt]` on Linux (see requirements.txt) -- the Compare window's
video engine there is QtWebEngine (PyQtWebEngine's bundled Chromium).
Confirmed (Qt's own docs/mailing list, 2026-08): the STOCK pip-installed
PyQtWebEngine build never includes H.264 decode at all, on any Linux
distro, regardless of what's installed system-wide -- it's excluded for
patent-licensing reasons and only re-appears if PyQtWebEngine itself is
rebuilt from source with `-webengine-proprietary-codecs` (a heavy,
fragile, machine-specific rebuild, not a `sudo apt install`). VP9 (in a
WebM container) is royalty-free and IS included in every stock
open-source Chromium/QtWebEngine build -- no system package, no
rebuild, works out of the box on any Linux machine running this
project's pinned pywebview[qt]. It also decodes fine in Safari/WKWebView
(macOS, VP9 support since Safari 14.1) so this doesn't regress the Mac
side either.

'avc1' (real H.264) and 'mp4v' (MPEG-4 Part 2) remain as fallbacks, in
that order, ONLY for the rare case this machine's OpenCV/FFmpeg build
has no usable VP9 ENCODER (unlikely: libvpx is bundled in virtually
every FFmpeg build, unlike the often-excluded libx264 -- confirmed by
testing in this project's own sandbox, which has no H.264 encoder at
all but DOES have a working VP9 one).
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
                                  # encoder for it (still useless for the Compare
                                  # window on this project's Linux/Qt setup, see
                                  # above, but harmless to prefer over mpeg4 for
                                  # anything else that opens the file).
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
    printed as a loud warning -- see below -- since it means this file
    will likely NOT play back in this project's Compare runs window on
    Linux). Raises `RuntimeError` if no candidate opens at all (e.g. an
    unwritable path)."""
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
                    f"in the Compare runs window on Linux (see common/video_writer.py "
                    f"module docstring) -- install/build FFmpeg with libvpx (VP9) or "
                    f"libx264 (H.264) to fix this."
                )
            return writer, actual_path, label
        writer.release()
    raise RuntimeError(
        f"Could not open {path!r} for writing with a VP9, H.264, or MPEG-4 encoder "
        f"(unwritable path, or no working video encoder in this OpenCV/FFmpeg build "
        f"at all)."
    )
