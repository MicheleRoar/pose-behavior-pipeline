"""
pose/depth.py
==============
Reads Azure Kinect depth from the ORIGINAL `.mkv` recording (never
`processed/<name>.mp4` -- that transcode keeps only the color stream,
see below) and attaches a Z to each already-extracted 2D pose keypoint,
producing a genuine 3D point `(X, Y, Z)` in millimeters, in the COLOR
camera's own coordinate frame, per (session, person, keypoint, frame) --
the "canonical/world coordinate" skeleton `pose/movement_metrics.py`
needs for distance-traveled/velocity/variability.

Confirmed on a real recording (`ffprobe`, 2026-09-22, `9_group_1_3/camera_a.mkv`):
4 streams -- COLOR (`mjpeg`, 1280x720), DEPTH (`rawvideo`/`gray16be`,
320x288), IR (same format/size as DEPTH), and an attachment (the
device's factory calibration, per Microsoft's own recording-format
docs). 320x288 is Azure Kinect's NFOV-2x2-binned depth mode -- a
narrower field of view (~75x65 degrees) than the color sensor's, which
is the whole reason this step can't just resize the depth image onto
the color one: the two don't see the same extent of the room.

Why this reads the original .mkv, not processed/<name>.mp4
-------------------------------------------------------------
`segmentation.run_pipeline`'s ffmpeg transcode step (see its module
docstring) re-encodes the color stream only -- no `-map` for the depth
(track index 1) or IR (index 2) streams, so ffmpeg's default stream
selection drops them entirely. This step opens the SOURCE `.mkv`
directly instead, via the real Azure Kinect SDK's own playback API
(`pyk4a`, a wrapper over `libk4a`) -- the depth track's raw 16-bit
codec isn't something a generic OpenCV/ffmpeg pipeline can align to
the color frame; the camera's own factory depth<->color calibration
(baked into the recording, see above) is required to do that, not a
resize or a naive coordinate rescale.

Frame alignment with the (already-computed) 2D keypoints
------------------------------------------------------------
`extract_keypoints.py`/`stabilization.py` operate on
`processed/<name>.mp4`, whose frame 0 is exactly `--ss` seconds into
the ORIGINAL `.mkv` (ffmpeg's `-ss`/`-to` there are given AFTER `-i` in
`segmentation/run_pipeline.py`'s `_transcode`, i.e. frame-accurate
trimming, not a fast keyframe-only seek). So this step seeks the
playback to `ss` (0 if no range was given) before reading captures, and
walks forward one capture per processed-clip frame, assuming the two
streams share the same fps -- true here, since the transcode never
passes ffmpeg `-r` and so preserves the source's own frame rate.

Why depth gets a NaN instead of a guess
--------------------------------------------
A keypoint detected in the color frame (e.g. near an edge, or an
extended arm) can fall OUTSIDE the narrower depth FOV entirely, or land
on a pixel with no valid depth return at all (out of range, low-angle
reflection -- ordinary limitations of this kind of sensor). Both cases
get `NaN`, never the nearest valid neighbor or an interpolated value --
same "no made-up signal" principle as the rest of this project
(`pose/stabilization.py`, `segmentation/merging/mask_utils.py`).

Sampled at the STABILIZED (x_smooth, y_smooth) position from
`keypoints_smoothed.csv` when available (falls back to raw x/y if
given `keypoints_raw.csv` instead) -- depth should be read at the
already-jitter-reduced 2D location, not the raw noisy one; Z itself is
NOT filtered by this step (it's a fresh, separate noise source: the
depth sensor's own per-pixel measurement noise/dropouts, not addressed
by `stabilization.py`'s OneEuroFilter, which today only smooths x/y --
see README's Known limitations).

Requires the real Azure Kinect Sensor SDK installed system-wide
(`libk4a`), not just `pip install pyk4a` -- pyk4a is a thin wrapper
around it and will fail to import without it. Delayed-imported, like
`mediapipe` in `extract_keypoints.py` -- the rest of the pose step
works without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read_person_keypoints(keypoints_csv: str) -> pd.DataFrame:
    """Reads `keypoints_smoothed.csv` (preferred, see module docstring)
    or `keypoints_raw.csv` as a fallback -- renamed to a common
    `x_smooth`/`y_smooth` shape either way, so the rest of this module
    doesn't need to care which one it got."""
    df = pd.read_csv(keypoints_csv)
    if "x_smooth" in df.columns:
        return df
    return df.rename(columns={"x": "x_smooth", "y": "y_smooth"})


def _timestamp_to_usec(ts: str) -> int:
    """ffmpeg-style 'HH:MM:SS[.ms]' -> microseconds -- must be the
    EXACT same string given to `segmentation.run_pipeline`'s `--ss`,
    so this step seeks to the same point in the original recording
    that `processed/<name>.mp4`'s frame 0 corresponds to."""
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected an 'HH:MM:SS[.ms]' timestamp, got {ts!r}")
    h, m, s = parts
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return int(round(total_seconds * 1_000_000))


def extract_depth(
    *,
    mkv_path: str,
    keypoints_csv: str,
    out_csv: str,
    session_id: str,
    ss: str | None = None,
) -> str:
    """Attaches a `(X_mm, Y_mm, Z_mm)` 3D point (color-camera frame) to
    every valid (frame, person, keypoint) row read from `keypoints_csv`,
    written long-format to `out_csv`. `ss` must be the exact `--ss`
    string given to `segmentation.run_pipeline` for this clip (or
    `None` for a whole-video run) -- see module docstring."""
    try:
        from pyk4a import PyK4APlayback, SeekOrigin
    except ImportError as exc:
        raise ImportError(
            "pose/depth.py requires 'pyk4a' AND the real Azure Kinect Sensor "
            "SDK (libk4a) installed system-wide -- 'pip install pyk4a' alone "
            "is not enough, it's a thin wrapper over the native SDK. See "
            "README.md's Setup section."
        ) from exc

    df = _read_person_keypoints(keypoints_csv)
    frames_by_idx = dict(tuple(df.groupby("frame", sort=False)))

    playback = PyK4APlayback(str(mkv_path))
    playback.open()
    if ss is not None:
        playback.seek(_timestamp_to_usec(ss), SeekOrigin.BEGIN)

    rows: list[dict] = []
    n_out_of_fov = 0
    n_no_return = 0
    n_no_capture = 0
    frame_idx = 0
    try:
        while True:
            try:
                capture = playback.get_next_capture()
            except EOFError:
                break

            kp_rows = frames_by_idx.get(frame_idx)
            if kp_rows is not None:
                point_cloud = capture.transformed_depth_point_cloud  # (H, W, 3) mm, color-camera space
                if point_cloud is None:
                    n_no_capture += 1
                ph, pw = point_cloud.shape[:2] if point_cloud is not None else (0, 0)

                for row in kp_rows.itertuples(index=False):
                    x, y = getattr(row, "x_smooth"), getattr(row, "y_smooth")
                    if pd.isna(x) or pd.isna(y):
                        continue  # no 2D detection to project in the first place

                    X = Y = Z = np.nan
                    px, py = int(round(x)), int(round(y))
                    if point_cloud is None:
                        pass  # whole frame had no depth capture -- already counted in n_no_capture above
                    elif not (0 <= px < pw and 0 <= py < ph):
                        n_out_of_fov += 1  # narrower depth FOV than color -- keypoint pixel isn't in it at all
                    else:
                        Xc, Yc, Zc = point_cloud[py, px]
                        if Zc > 0:
                            X, Y, Z = float(Xc), float(Yc), float(Zc)
                        else:
                            n_no_return += 1  # in-FOV pixel, but no valid depth (range/reflectivity)

                    rows.append({
                        "session_id": session_id,
                        "global_person_id": getattr(row, "global_person_id"),
                        "frame": frame_idx,
                        "keypoint_name": getattr(row, "keypoint_name"),
                        "x": x, "y": y,
                        "X_mm": X, "Y_mm": Y, "Z_mm": Z,
                    })
            frame_idx += 1
    finally:
        playback.close()

    if n_no_capture:
        print(f"[depth] {n_no_capture} frame(s) had no depth capture at all (sensor drop) -- "
              f"those frames' keypoints got NaN X/Y/Z")
    print(f"[depth] {n_out_of_fov} keypoint(s) fell outside the depth sensor's narrower field of "
          f"view, {n_no_return} more were inside it but had no valid depth return (out of range or "
          f"a low-angle reflection) -- all set to NaN, not guessed")

    out_df = pd.DataFrame(rows, columns=[
        "session_id", "global_person_id", "frame", "keypoint_name",
        "x", "y", "X_mm", "Y_mm", "Z_mm",
    ])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    n_valid = int(out_df["Z_mm"].notna().sum()) if len(out_df) else 0
    print(f"[depth] {n_valid}/{len(out_df)} keypoint rows have a valid 3D point -> {out_csv}")
    return str(out_csv)
