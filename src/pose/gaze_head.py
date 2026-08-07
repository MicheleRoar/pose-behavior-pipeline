"""
gaze_head.py
============
Head-pose estimation and a "joint attention" proxy from MediaPipe Tasks
FaceLandmarker, to be used together with YOLO-pose's multi-person
tracking: YOLO locates and tracks people in the frame (child/caregiver),
FaceLandmarker is applied to the whole frame to obtain dense facial
landmarks + a facial transformation matrix (from which head orientation
is derived), and the two outputs are then matched by spatial proximity.

Why head/gaze before finger-level hands: the literature on behavioral
markers in child neurodevelopment (see README) explicitly cites gaze
coordination and head movement frequency as relevant indicators (e.g.
"head turn to disengage attention", child-caregiver joint attention).

Important methodological note: what's implemented here is a simplified
2D PROXY, not true calibrated 3D gaze tracking. With a single RGB camera,
without intrinsic calibration or depth estimation, it's not possible to
precisely reconstruct where a person is looking in 3D space. The
heuristic used (`joint_attention_score`) compares the estimated
head-yaw with the bearing towards the other person's head in the image,
assuming the two people are at a comparable distance from the camera --
a reasonable approximation for a single observation room, but to be
validated empirically before any interpretive use.

Required setup:

    pip install mediapipe

The Face Landmarker model is downloaded AUTOMATICALLY on first run into
a fixed cache inside the project (`<repo>/models/`), no more manual
`curl` needed -- see `common/mediapipe_models.py` for details (same
bug/fix as `pose/mediapipe_pose.py`/`pose/hands.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.mediapipe_models import resolve_model_path

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


# ---------------------------------------------------------------------------
# Geometry: rotation matrix -> Euler angles
# ---------------------------------------------------------------------------

def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Extracts (yaw, pitch, roll) in degrees from a 3x3 rotation matrix.

    Convention: yaw = rotation around the vertical axis (left/right,
    positive towards image-right), pitch = up/down, roll = lateral tilt.
    The exact sign depends on MediaPipe's axis convention: must be
    validated empirically on your own setup (e.g. moving the head to the
    right and checking that the estimated yaw increases); if it comes
    out inverted, just negate the value in `HeadGazeEstimator`.
    """
    assert R.shape == (3, 3), f"expected a 3x3 matrix, got {R.shape}"
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
        roll = np.arctan2(-R[1, 2], R[1, 1])

    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Inverse of `rotation_matrix_to_euler`, used only to generate
    synthetic test cases (no use in the live pipeline).
    """
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Joint attention proxy
# ---------------------------------------------------------------------------

def bearing_to_target(head_xy: np.ndarray, target_xy: np.ndarray,
                       frame_width: float, fov_deg: float = 60.0) -> float:
    """Approximate horizontal angle (degrees) towards `target_xy` in the
    image, as seen from `head_xy`, assuming a pinhole camera with
    horizontal field-of-view `fov_deg` and the two people at a comparable
    distance from the camera (no real depth estimation). Positive =
    target to the right.
    """
    dx = target_xy[0] - head_xy[0]
    return float((dx / (frame_width / 2.0)) * (fov_deg / 2.0))


def joint_attention_score(head_xy: np.ndarray, yaw_deg: float, target_xy: np.ndarray,
                           frame_width: float, fov_deg: float = 60.0,
                           tolerance_deg: float = 20.0) -> float:
    """0-1 score of how much the head appears oriented towards
    `target_xy` (e.g. the other tracked person's head), as a simplified
    joint-attention proxy. See the methodological note at the top of the
    module.
    """
    expected_yaw = bearing_to_target(head_xy, target_xy, frame_width, fov_deg)
    diff = abs(yaw_deg - expected_yaw)
    return float(max(0.0, 1.0 - diff / tolerance_deg))


# ---------------------------------------------------------------------------
# Mouth (mouth aspect ratio) and eyes (eye aspect ratio / blink)
# ---------------------------------------------------------------------------
#
# Face landmark indices (MediaPipe Face Landmarker's 468/478-point
# schema) used for mouth and eyes. These are the indices commonly used in
# the literature/community for computing mouth/eye aspect ratio (the same
# principle as the classic Eye Aspect Ratio from Soukupová & Čech, here
# applied to MediaPipe landmarks instead of the original dlib ones).

MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291

# Order [left_corner, upper_eyelid_1, upper_eyelid_2, right_corner, lower_eyelid_2, lower_eyelid_1]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

# Eyebrow outline (5 points each), same indices used by MediaPipe's
# FACEMESH_LEFT/RIGHT_EYEBROW connections.
RIGHT_EYEBROW_IDX = [70, 63, 105, 66, 107]
LEFT_EYEBROW_IDX = [336, 296, 334, 293, 300]


def mouth_aspect_ratio(face_xy: np.ndarray) -> float:
    """Ratio of vertical opening / horizontal width of the mouth. Low
    values = mouth closed, higher values = mouth open. Useful as a rough
    proxy for vocalization or repetitive mouth movements (e.g. mouthing),
    not for speech recognition.
    """
    top, bottom = face_xy[MOUTH_TOP], face_xy[MOUTH_BOTTOM]
    left, right = face_xy[MOUTH_LEFT], face_xy[MOUTH_RIGHT]
    horizontal = np.linalg.norm(left - right)
    if horizontal < 1e-6:
        return np.nan
    vertical = np.linalg.norm(top - bottom)
    return float(vertical / horizontal)


def eye_aspect_ratio(face_xy: np.ndarray, idx: list[int]) -> float:
    """Eye Aspect Ratio (EAR) for one eye: ratio between average vertical
    opening and horizontal width. Drops sharply during a blink. `idx`: 6
    indices [left_corner, top1, top2, right_corner, bottom2, bottom1].
    """
    p1, p2, p3, p4, p5, p6 = [face_xy[i] for i in idx]
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return np.nan
    vertical = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / 2.0
    return float(vertical / horizontal)


def mean_eye_aspect_ratio(face_xy: np.ndarray) -> float:
    """Average EAR over both eyes (more robust than a single eye in case
    of slight head rotation)."""
    left = eye_aspect_ratio(face_xy, LEFT_EYE_EAR_IDX)
    right = eye_aspect_ratio(face_xy, RIGHT_EYE_EAR_IDX)
    values = [v for v in (left, right) if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


# ---------------------------------------------------------------------------
# Eyebrows
# ---------------------------------------------------------------------------

def interocular_distance(face_xy: np.ndarray) -> float:
    """Distance between the outer corners of the two eyes: used as the
    face's scale unit (invariant to distance from the camera) to
    normalize eyebrow raise."""
    return float(np.linalg.norm(face_xy[RIGHT_EYE_EAR_IDX[0]] - face_xy[LEFT_EYE_EAR_IDX[0]]))


def eyebrow_raise_ratio(face_xy: np.ndarray, eyebrow_idx: list[int], eye_idx: list[int],
                         ref_distance: float) -> float:
    """Eyebrow raise score: vertical distance between the eyebrow and the
    corresponding upper eyelid, normalized by interocular distance (so
    it's comparable regardless of distance from the camera). Higher
    values = raised eyebrow (e.g. surprise); lower/negative values =
    lowered or furrowed eyebrow (e.g. concentration, displeasure).
    Interpretation thresholds not calibrated: to be tuned to the specific
    subject/context, as with MAR/EAR.
    """
    if ref_distance < 1e-6 or np.isnan(ref_distance):
        return np.nan
    eyebrow_y = float(np.mean([face_xy[i][1] for i in eyebrow_idx]))
    # upper eyelid = points top1, top2 of the EAR schema (indices 1, 2)
    eyelid_y = float(np.mean([face_xy[eye_idx[1]][1], face_xy[eye_idx[2]][1]]))
    return (eyelid_y - eyebrow_y) / ref_distance


def mean_eyebrow_raise(face_xy: np.ndarray) -> tuple[float, float]:
    """Left and right eyebrow raise (in this order)."""
    ref = interocular_distance(face_xy)
    left = eyebrow_raise_ratio(face_xy, LEFT_EYEBROW_IDX, LEFT_EYE_EAR_IDX, ref)
    right = eyebrow_raise_ratio(face_xy, RIGHT_EYEBROW_IDX, RIGHT_EYE_EAR_IDX, ref)
    return left, right


# ---------------------------------------------------------------------------
# Matching faces (FaceLandmarker) <-> tracked people (YOLO)
# ---------------------------------------------------------------------------

def match_faces_to_tracks(face_centers: list[np.ndarray], track_ids: list[int],
                           track_head_centers: list[np.ndarray],
                           max_distance: float = 80.0) -> dict[int, int]:
    """Matches each face detected by FaceLandmarker to the (YOLO) track_id
    with the nearest head-center, within `max_distance` pixels. Greedy
    nearest-neighbor: sufficient for 2-3 people in the scene (child +
    caregiver).

    Returns
    -------
    dict {face_index: track_id}
    """
    assignments: dict[int, int] = {}
    used_tracks: set[int] = set()

    order = sorted(
        range(len(face_centers)),
        key=lambda i: min(
            (np.linalg.norm(face_centers[i] - c) for c in track_head_centers),
            default=np.inf,
        ),
    )

    for face_idx in order:
        best_tid, best_dist = None, max_distance
        for tid, center in zip(track_ids, track_head_centers):
            if tid in used_tracks:
                continue
            d = float(np.linalg.norm(face_centers[face_idx] - center))
            if d < best_dist:
                best_tid, best_dist = tid, d
        if best_tid is not None:
            assignments[face_idx] = best_tid
            used_tracks.add(best_tid)

    return assignments


# ---------------------------------------------------------------------------
# Wrapper over MediaPipe Tasks FaceLandmarker (requires mediapipe + model)
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    landmarks_xy: np.ndarray       # (478, 2) in the frame's pixel coordinates
    yaw: float
    pitch: float
    roll: float
    mouth_ratio: float = float("nan")
    eye_ratio: float = float("nan")
    left_eyebrow_raise: float = float("nan")
    right_eyebrow_raise: float = float("nan")


class HeadGazeEstimator:
    """Wrapper over MediaPipe Tasks FaceLandmarker for multi-face
    head-pose.

    mediapipe import delayed (as in `pose_estimation.PoseTracker`) so the
    rest of the pipeline remains usable/testable even without mediapipe
    installed.
    """

    def __init__(self, model_path: str = "face_landmarker.task", num_faces: int = 3):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = resolve_model_path(model_path, download_url=_MODEL_URL)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[FaceResult]:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame_bgr.shape[:2]
        out = []
        for i, face_landmarks in enumerate(result.face_landmarks):
            xy = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])
            if result.facial_transformation_matrixes:
                R = np.array(result.facial_transformation_matrixes[i])[:3, :3]
                yaw, pitch, roll = rotation_matrix_to_euler(R)
            else:
                yaw, pitch, roll = np.nan, np.nan, np.nan
            mouth_ratio = mouth_aspect_ratio(xy)
            eye_ratio = mean_eye_aspect_ratio(xy)
            left_brow, right_brow = mean_eyebrow_raise(xy)
            out.append(FaceResult(landmarks_xy=xy, yaw=yaw, pitch=pitch, roll=roll,
                                   mouth_ratio=mouth_ratio, eye_ratio=eye_ratio,
                                   left_eyebrow_raise=left_brow, right_eyebrow_raise=right_brow))
        return out
