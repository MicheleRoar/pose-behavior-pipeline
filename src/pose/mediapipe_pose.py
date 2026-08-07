"""
mediapipe_pose.py
===================
Pose estimation (body keypoints) with MediaPipe Tasks PoseLandmarker,
applied INSIDE the crop (bbox) of a single person ALREADY tracked by the
segmentation pipeline (`segmentation/seg_estimation.py` +
`segmentation/seg_reid.py`) -- not a multi-person detector on the whole
frame.

Why "inside the crop" and not on the whole frame
------------------------------------------------------
MediaPipe Tasks PoseLandmarker supports multi-person detection
(`num_poses`), but does NOT provide any temporal tracking: no id
persistent across frames, unlike YOLO+ByteTrack. Building an equivalent
tracker from scratch on top of its raw detections would be a substantial
amount of work for uncertain gain (exactly the kind of tracking
instability this pipeline moved away from by switching to
segmentation_demo.py in the first place, see README).

The choice made here is simpler and reuses what already works: the
segmentation pipeline already tracks each person's identity stably
(silhouette + position/color/shape, see seg_reid.py). This module simply
applies MediaPipe in SINGLE-PERSON mode (the library's most mature and
reliable, no multi-person association problem to solve) inside the box
of each ALREADY-tracked person, frame by frame -- identity is borrowed
from segmentation, not reconstructed here. This was already the plan
described in `seg_estimation.py`'s docstring ("reconnect pose applied
inside the tracked silhouette").

Remapping to COCO-17
------------------------
The 33 BlazePose landmarks are remapped to the 17 COCO-17 names already
used throughout the pose pipeline (`pose/keypoints.py`, see
`BLAZEPOSE_TO_COCO` below), so the existing feature engineering functions
(`pose/features.py`, `common/viz.draw_skeleton`, etc.) work IDENTICALLY
regardless of which model produced the keypoints. The 16 BlazePose
landmarks with no direct COCO equivalent (inner/outer eyes, mouth
corners, fingers, heels, foot tips) are discarded: not needed by the
existing features, all written for the COCO-17 schema.

Honest limitations
-------------------
  - No tracking/sliding-window features (movement energy, repetitiveness,
    etc.) are wired in here yet -- only joint angles that can be computed
    instantaneously, frame by frame (see the wiring in
    segmentation_demo.py). Connecting the rest of pose/features.py would
    require managing per-person buffers in segmentation_demo.py too, not
    yet done.
  - A segmentation box tight on the silhouette can cut off hands raised
    above the head or feet near the edge: the padding in `estimate()`
    helps but doesn't eliminate the problem.
  - Per-joint confidence (MediaPipe's `visibility`) and YOLO detection
    confidence (`pose_estimation.py`) aren't necessarily on the same
    scale: treating them as interchangeable in quantitative analysis
    needs validation.

Required setup:

    pip install mediapipe

The Pose Landmarker model ("lite", the fastest -- "full"/"heavy" are more
accurate but slower, replace "lite" with "full"/"heavy" in `_MODEL_URL`
below) is downloaded AUTOMATICALLY on first run into a fixed cache inside
the project (`<repo>/models/`), no more manual `curl` needed -- see
`common/mediapipe_models.py` for details (logic shared with
`pose/hands.py`/`pose/gaze_head.py`, same bug, same fix).
"""

from __future__ import annotations

import numpy as np

from common.mediapipe_models import resolve_model_path
from pose.keypoints import KP

# Same "lite" variant already documented above -- for "full"/"heavy"
# replace "lite" with "full"/"heavy" in the URL.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# BlazePose landmark index (0-32, MediaPipe Pose Landmarker schema) ->
# COCO-17 name (pose/keypoints.py). BlazePose landmarks with no direct
# COCO equivalent (inner/outer eyes, mouth corners, fingers, heels, foot
# tips) don't appear here: they're discarded.
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


def _empty_pose() -> tuple[np.ndarray, np.ndarray]:
    """"Empty" (kxy, kconf): 17 NaN joints/zero confidence, same schema
    as a frame in which no pose was detected."""
    return np.full((17, 2), np.nan), np.zeros(17)


def blazepose_to_coco(landmarks, frame_offset_xy: tuple[float, float],
                       crop_size_wh: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Converts a list of 33 BlazePose landmarks (in NORMALIZED 0-1
    coordinates relative to the crop, as returned by MediaPipe) into
    (kxy, kconf) COCO-17 in FULL-FRAME PIXEL coordinates.

    `frame_offset_xy`: top-left corner of the crop in the full frame
    (x1, y1). `crop_size_wh`: crop dimensions in pixels. Isolated from
    the class that calls MediaPipe so it's testable without
    mediapipe/camera (see demo/mediapipe_pose_check.py).
    """
    x1, y1 = frame_offset_xy
    crop_w, crop_h = crop_size_wh
    kxy, kconf = _empty_pose()
    for blaze_idx, coco_name in BLAZEPOSE_TO_COCO.items():
        lm = landmarks[blaze_idx]
        coco_idx = KP[coco_name]
        kxy[coco_idx] = [x1 + lm.x * crop_w, y1 + lm.y * crop_h]
        visibility = getattr(lm, "visibility", None)
        kconf[coco_idx] = float(visibility) if visibility is not None else 1.0
    return kxy, kconf


def padded_crop_box(bbox: np.ndarray, frame_shape: tuple[int, int],
                     padding: float = 0.15) -> tuple[int, int, int, int]:
    """Crop box (x1,y1,x2,y2, integers, clamped to the frame's edges)
    starting from a segmentation bbox, expanded by `padding` (fraction of
    width/height) to avoid cutting off extremities (raised hands, feet)
    when the box is tight on the silhouette. Isolated so it's testable
    without mediapipe/camera."""
    h, w = frame_shape
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(w, int(x2 + bw * padding))
    y2 = min(h, int(y2 + bh * padding))
    return x1, y1, x2, y2


class MediaPipeCropPoseEstimator:
    """Wrapper over MediaPipe Tasks PoseLandmarker in SINGLE-person mode
    (`num_poses=1`), applied to a crop of the frame -- see the module
    docstring for why. mediapipe import delayed (like
    `pose_estimation.PoseTracker` / `gaze_head.HeadGazeEstimator`), so the
    rest of the pipeline remains usable/testable even without mediapipe
    installed.
    """

    def __init__(self, model_path: str = "pose_landmarker_lite.task",
                 min_pose_detection_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = resolve_model_path(model_path, download_url=_MODEL_URL)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms: int | None = None  # see the clamp in estimate()

    def estimate(self, frame_bgr: np.ndarray, bbox: np.ndarray, timestamp_ms: int,
                 padding: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        """Detects the pose INSIDE `bbox` (x1,y1,x2,y2, e.g. from the
        segmentation tracker). Returns (kxy, kconf) in FULL-FRAME PIXEL
        coordinates (not the crop's), same COCO-17 schema as
        `pose_estimation.py` -- NaN/0 for joints with no BlazePose
        equivalent or if no pose was detected in the crop.

        WARNING (see also `MediaPipePoseByTrack`'s docstring):
        `detect_for_video` (MediaPipe's VIDEO mode) requires STRICTLY
        increasing timestamps for the same instance -- calling this
        method with the same timestamp twice (e.g. for two different
        people in the same frame, on the SAME instance) raises
        `ValueError: Input timestamp must be monotonically increasing`.
        An instance must therefore be used for ONLY one person over time
        (see `MediaPipePoseByTrack`, which attaches one instance per
        track_id). As an additional safety net -- not a substitute for
        that design -- the actual timestamp is still forced to be
        greater than the last one used here, so a duplicate or
        out-of-order timestamp (e.g. from rounding at very low fps)
        doesn't crash but at most loses a millisecond of precision."""
        import cv2

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = padded_crop_box(bbox, (h, w), padding)
        if x2 <= x1 or y2 <= y1:
            return _empty_pose()

        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        crop = frame_bgr[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return _empty_pose()

        landmarks = result.pose_landmarks[0]  # num_poses=1: at most one pose
        crop_h, crop_w = crop.shape[:2]
        return blazepose_to_coco(landmarks, (x1, y1), (crop_w, crop_h))


class MediaPipePoseByTrack:
    """Pool of an INDEPENDENT `MediaPipeCropPoseEstimator` for each
    `track_id`, instead of a single instance shared across all people in
    the frame.

    Why a single instance isn't enough
    ------------------------------------
    `PoseLandmarker.detect_for_video` (VIDEO mode) is designed for ONE
    continuous stream per instance: it keeps internal state between calls
    (temporal filtering/smoothing of landmarks) and requires strictly
    increasing timestamps. `segmentation_demo.py`'s per-person loop,
    however, calls `.estimate(...)` ONCE PER PERSON in the same frame,
    all with the same timestamp (the frame's `now`) -- on a shared
    instance this (a) triggers `ValueError: Input timestamp must be
    monotonically increasing` for the second person in the frame, and (b)
    even working around the crash, would mix the temporal smoothing state
    of different people as if they were a single person teleporting from
    one body to another.

    The solution is for each track_id to get its own independent
    instance/"stream" -- created on the track's first appearance and
    reused for its whole lifetime, so each one sees a coherent timestamp
    sequence and a smoothing state that belongs only to it."""

    def __init__(self, model_path: str = "pose_landmarker_lite.task",
                 min_pose_detection_confidence: float = 0.5):
        # Resolved/downloaded ONCE here (not on every new track_id in
        # estimate()): resolve_model_path() is cheap to call, but there's
        # no point repeating the check/printing the download message for
        # every person who enters the scene.
        self._model_path = resolve_model_path(model_path, download_url=_MODEL_URL)
        self._min_conf = min_pose_detection_confidence
        self._estimators: dict[int, MediaPipeCropPoseEstimator] = {}

    def estimate(self, track_id: int, frame_bgr: np.ndarray, bbox: np.ndarray,
                 timestamp_ms: int, padding: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        estimator = self._estimators.get(track_id)
        if estimator is None:
            estimator = MediaPipeCropPoseEstimator(
                model_path=self._model_path, min_pose_detection_confidence=self._min_conf)
            self._estimators[track_id] = estimator
        return estimator.estimate(frame_bgr, bbox, timestamp_ms=timestamp_ms, padding=padding)

    def forget(self, track_id: int) -> None:
        """Removes the instance of a track that has left the scene (e.g.
        expired in seg_reid) -- avoids accumulating landmarkers for
        long-dead ids in long sessions with a lot of turnover. Not
        mandatory (a handful of extra instances isn't a practical
        problem), but cheap to call when a track is already known to be
        dead."""
        self._estimators.pop(track_id, None)
