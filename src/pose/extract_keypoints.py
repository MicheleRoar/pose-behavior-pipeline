"""
pose/extract_keypoints.py
===========================
Runs MediaPipe's PoseLandmarker (Tasks API -- not the older, deprecated
`mp.solutions.pose`) on the merged MaskDir produced by
`segmentation/merging/merge_fragments.py`, one crop per known
`global_person_id` per frame, and writes a long-format CSV of 2D
keypoints -- the raw input `pose/stabilization.py` expects.

Design choice worth being explicit about: identity comes from the mask
(SAM3 + OSNet + the merge heuristics already validated in `segmentation/`),
never from MediaPipe's own multi-person detection. `num_poses=1` per
landmarker, one landmarker instance per `global_person_id`, run on that
person's own mask-bbox crop. The alternative -- one landmarker call on the
full frame with `num_poses=N`, then matching detections back to identities
by centroid-in-mask -- would need its own re-matching logic every frame
(MediaPipe's own multi-pose output isn't identity-persistent across
frames) to solve a problem the merge step already solved. Costs one
landmarker call per person per frame instead of one call per frame total;
negligible for BlazePose on a couple of people.

This design (crop-per-tracked-identity, single-pose mode) independently
matches `pose-behavior-pipeline_legacy/src/pose/mediapipe_pose.py`'s own
rationale -- discovered only after this module was first written. The
schema/conventions below (COCO-17 output, `<repo>/models/` caching, pinned
model URL, `min_pose_detection_confidence`, timestamp clamp, 0.15 padding)
were revised to match that legacy module exactly, so results and code stay
consistent with it rather than diverging for no reason. See `pose/keypoints.py`
and `pose/model_cache.py` (both ported from the legacy repo) and this
repo's README for the full rationale.

`mediapipe` is a real, non-optional dependency of this step only (not
listed in the top of requirements.txt, same delayed-import treatment as
`torch`/`torchreid` in `pose/appearance_embedding.py` -- the segmentation
half of the pipeline works without it installed).

Output schema (long format, one row per COCO-17 keypoint per frame per
person -- matches what `pose/stabilization.py` and the project's general
"group by session_id + global_person_id, never by label" convention
expect):

    session_id, global_person_id, frame, timestamp_ms, keypoint_name, x, y, confidence

`keypoint_name` is one of the 17 COCO names in `pose.keypoints.COCO17`
(the 16 BlazePose landmarks with no COCO equivalent -- inner/outer eyes,
mouth corners, fingers, heels, foot tips -- are discarded, same as the
legacy pipeline, so downstream code never needs to know which model
produced the keypoints). `x`/`y` are full-frame pixel coordinates (already
converted back from the crop-normalized coordinates PoseLandmarker
returns). `confidence` is MediaPipe's per-landmark `visibility`. A frame
with no detection at all for a person (occluded, empty mask, crop too
small to trust) writes no rows for that person/frame -- "no made-up
signal", same principle as `segmentation/merging/mask_utils.py` and
`pose/appearance_embedding.py`; `pose/stabilization.py`'s densification
step is what turns that absence into an explicit, interpolatable gap.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose.keypoints import KP
from pose.model_cache import resolve_model_path

# BlazePose landmark index (0-32, MediaPipe Pose Landmarker's fixed output
# order) -> COCO-17 name (pose/keypoints.py). Ported as-is from
# pose-behavior-pipeline_legacy/src/pose/mediapipe_pose.py's
# BLAZEPOSE_TO_COCO. Landmarks with no direct COCO equivalent (inner/outer
# eyes, mouth corners, fingers, heels, foot tips) don't appear here: they're
# discarded. See https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# for the full 33-landmark reference.
BLAZEPOSE_TO_COCO: dict[int, str] = {
    0: "nose",
    2: "left_eye", 5: "right_eye",
    7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}

# Pinned model version (float16/1/..., not .../latest/...): matches the
# legacy pipeline's convention -- a "latest" URL can silently start
# serving different weights later, which is a problem for a research
# pipeline that needs reproducible numbers across sessions/runs.
_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

# Same "no made-up signal" threshold philosophy as
# pose/appearance_embedding.py's MIN_CROP_W/MIN_CROP_H: below this, the
# crop is too small/squashed to trust a pose estimate from it.
MIN_CROP_W = 40
MIN_CROP_H = 60

_MASK_FILENAME_RE = re.compile(r"^(\d+)\.mp4$")
_MASK_THRESHOLD = 127  # matches segmentation/merging/mask_io.py's DEFAULT_MASK_THRESHOLD


def _resolve_model_path(model_path: str | None, model_variant: str) -> str:
    """Returns a local `.task` file path. If `model_path` isn't given,
    resolves the bare default name for `model_variant` via
    `pose.model_cache.resolve_model_path` -- cached inside `<repo>/models/`
    (independent of cwd), downloaded there on first use (public download,
    not gated like the SAM3 checkpoint -- see README)."""
    if model_variant not in _MODEL_URLS:
        raise ValueError(f"model_variant must be one of {list(_MODEL_URLS)}, got {model_variant!r}")
    download_url = _MODEL_URLS[model_variant]
    if model_path is None:
        model_path = download_url.rsplit("/", 1)[-1]  # bare default name
    return resolve_model_path(model_path, download_url=download_url)


def _create_landmarker(model_path: str, device: str, min_pose_detection_confidence: float):
    """Delayed import -- see module docstring on mediapipe being a
    real but non-hard-pinned dependency of this step only."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "pose/extract_keypoints.py requires 'mediapipe', not installed by "
            "default (this step's own dependency -- see requirements.txt). "
            "Install with: pip install mediapipe"
        ) from exc

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    delegate = BaseOptions.Delegate.GPU if device == "gpu" else BaseOptions.Delegate.CPU
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,  # identity already known from the mask crop -- see module docstring
        min_pose_detection_confidence=min_pose_detection_confidence,
    )
    return PoseLandmarker.create_from_options(options)


def _list_mask_files(mask_dir: Path) -> dict[int, Path]:
    """`{global_person_id: path}` for every `<id>.mp4` in `mask_dir`.
    Independent, small re-implementation of
    `segmentation/merging/mask_io.py`'s private `_list_mask_files` --
    pose/ only depends on segmentation's MaskDir *output format*
    (documented in segmentation's README), not on its internals."""
    paths: dict[int, Path] = {}
    for p in sorted(mask_dir.iterdir()):
        m = _MASK_FILENAME_RE.match(p.name)
        if m:
            paths[int(m.group(1))] = p
    if not paths:
        raise ValueError(f"No '<id>.mp4' mask files found in {mask_dir}")
    return paths


def _bbox_from_mask(mask_frame: np.ndarray, frame_w: int, frame_h: int,
                     padding: float) -> tuple[int, int, int, int] | None:
    """Tight bbox of the nonzero pixels in a decoded mask frame, padded
    by `padding` (fraction of the bbox's own width/height) on each side
    and clipped to the frame -- avoids cutting off extremities (raised
    hands, feet) when the mask is tight on the silhouette. `None` if the
    mask is empty. Same `padded_crop_box` logic as the legacy pipeline's
    `pose/mediapipe_pose.py` (default padding matches it too: 0.15)."""
    ys, xs = np.where(mask_frame[:, :, 0] > _MASK_THRESHOLD if mask_frame.ndim == 3
                       else mask_frame > _MASK_THRESHOLD)
    if xs.size == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    x1, x2 = int(np.clip(x1 - pad_x, 0, frame_w)), int(np.clip(x2 + pad_x, 0, frame_w))
    y1, y2 = int(np.clip(y1 - pad_y, 0, frame_h)), int(np.clip(y2 + pad_y, 0, frame_h))
    return x1, y1, x2, y2


def extract_keypoints(
    *,
    video_path: str,
    merged_mask_dir: str,
    out_csv: str,
    session_id: str,
    model_path: str | None = None,
    model_variant: str = "lite",
    min_pose_detection_confidence: float = 0.5,
    bbox_padding: float = 0.15,
    device: str = "cpu",
) -> str:
    """Runs PoseLandmarker over `video_path`, once per `global_person_id`
    found in `merged_mask_dir` (one `<id>.mp4` per identity, psifx
    MaskDir format), and writes the long-format COCO-17 keypoints CSV
    described in the module docstring to `out_csv`. Returns `out_csv`.

    `model_variant` defaults to "lite" (fastest; "full"/"heavy" trade
    speed for accuracy), matching the legacy pipeline's own default choice.

    `device='gpu'` uses MediaPipe's own GPU delegate (OpenGL/EGL-based --
    a different thing from SAM3's CUDA device, and can be finicky
    headless on a server without a display; 'cpu' is the safe default,
    BlazePose is light enough for this to be practical on CPU)."""
    import mediapipe as mp  # already validated importable by _create_landmarker below

    mask_dir = Path(merged_mask_dir)
    mask_files = _list_mask_files(mask_dir)
    resolved_model_path = _resolve_model_path(model_path, model_variant)

    src_cap = cv2.VideoCapture(str(video_path))
    if not src_cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = src_cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise ValueError(
            f"Could not read a valid fps from {video_path} (got {fps!r}) -- "
            f"needed to convert frame index to a timestamp for PoseLandmarker."
        )
    frame_w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_caps = {pid: cv2.VideoCapture(str(p)) for pid, p in mask_files.items()}
    for pid, cap in mask_caps.items():
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open mask video for id {pid}: {mask_files[pid]}")

    landmarkers = {
        pid: _create_landmarker(resolved_model_path, device, min_pose_detection_confidence)
        for pid in mask_files
    }
    # One landmarker instance is reused across the whole video for a given
    # pid (never shared between pids), so timestamps naturally increase
    # with frame_idx -- but `detect_for_video` (VIDEO mode) requires
    # STRICTLY increasing timestamps per instance, and low-fps/rounding
    # edge cases could in principle produce a duplicate ms value. This
    # defensive clamp (not a substitute for the one-instance-per-identity
    # design, just a safety net) is ported from the same pattern in
    # pose-behavior-pipeline_legacy/src/pose/mediapipe_pose.py.
    last_timestamp_ms: dict[int, int] = {}

    rows: list[dict] = []
    n_small_crop_skipped = 0
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = src_cap.read()
            if not ok:
                break
            timestamp_ms = int(round(frame_idx / fps * 1000))

            for pid, cap in mask_caps.items():
                ok_m, mask_frame = cap.read()
                if not ok_m:
                    continue  # this id's MaskDir file ended (shouldn't happen -- padded to source length)

                bbox = _bbox_from_mask(mask_frame, frame_w, frame_h, bbox_padding)
                if bbox is None:
                    continue  # person absent/occluded this frame -- no row, see module docstring
                x1, y1, x2, y2 = bbox
                if x2 - x1 < MIN_CROP_W or y2 - y1 < MIN_CROP_H:
                    n_small_crop_skipped += 1
                    continue

                pid_timestamp_ms = timestamp_ms
                prev = last_timestamp_ms.get(pid)
                if prev is not None and pid_timestamp_ms <= prev:
                    pid_timestamp_ms = prev + 1
                last_timestamp_ms[pid] = pid_timestamp_ms

                crop_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                result = landmarkers[pid].detect_for_video(mp_image, pid_timestamp_ms)
                if not result.pose_landmarks:
                    continue  # PoseLandmarker found no pose in this crop -- no row

                crop_w, crop_h = (x2 - x1), (y2 - y1)
                landmarks = result.pose_landmarks[0]  # num_poses=1: at most one pose
                for blaze_idx, coco_name in BLAZEPOSE_TO_COCO.items():
                    lm = landmarks[blaze_idx]
                    visibility = getattr(lm, "visibility", None)
                    rows.append({
                        "session_id": session_id,
                        "global_person_id": pid,
                        "frame": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "keypoint_name": coco_name,
                        "x": x1 + lm.x * crop_w,
                        "y": y1 + lm.y * crop_h,
                        "confidence": float(visibility) if visibility is not None else 1.0,
                    })
            frame_idx += 1
    finally:
        src_cap.release()
        for cap in mask_caps.values():
            cap.release()
        for landmarker in landmarkers.values():
            landmarker.close()

    if n_small_crop_skipped:
        print(f"[extract_keypoints] skipped {n_small_crop_skipped} frame/person crop(s) "
              f"smaller than {MIN_CROP_W}x{MIN_CROP_H}px (too small to trust a pose estimate)")

    df = pd.DataFrame(rows, columns=[
        "session_id", "global_person_id", "frame", "timestamp_ms",
        "keypoint_name", "x", "y", "confidence",
    ])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_frames_total = frame_idx
    print(f"[extract_keypoints] {len(df)} keypoint rows (COCO-17) over {n_frames_total} frames, "
          f"{len(mask_files)} identities -> {out_csv}")
    return str(out_csv)
