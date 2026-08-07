"""
hands.py
========
Finger-level hand tracking (21 landmarks per hand) via MediaPipe Tasks
HandLandmarker, to be combined with YOLO-pose's multi-person tracking:
YOLO tracks people (child/caregiver) and provides wrists as an anchor
point; HandLandmarker runs on the whole frame and detected hands are
matched to the nearest YOLO wrist.

Why finger-level and not just the wrist: the "repetitive movement" score
already present in `features.py` uses wrist speed as a proxy for manual
stereotypies -- it works, but doesn't distinguish, for example, hand
clapping (wrist nearly still, hands opening/closing) from true
hand-flapping (oscillating wrist). With the 21 landmarks per hand, an
index of hand openness/closedness over time can be added, complementary
to wrist kinematics alone.

MediaPipe Hands 21-landmark schema (indices):
    0 wrist
    1-4   thumb (CMC, MCP, IP, TIP)
    5-8   index (MCP, PIP, DIP, TIP)
    9-12  middle (MCP, PIP, DIP, TIP)
    13-16 ring (MCP, PIP, DIP, TIP)
    17-20 pinky (MCP, PIP, DIP, TIP)

Required setup:

    pip install mediapipe

The Hand Landmarker model is downloaded AUTOMATICALLY on first run into a
fixed cache inside the project (`<repo>/models/`), no more manual `curl`
needed -- see `common/mediapipe_models.py` for details (same bug/fix as
`pose/mediapipe_pose.py`: the bare default "hand_landmarker.task" was
resolved by MediaPipe relative to the cwd, breaking if launched from a
cwd different from the one used for the manual download).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.mediapipe_models import resolve_model_path
from pose.geometry import angle_at

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WRIST = 0
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

# Connections between the 21 landmarks, to draw the hand skeleton
# (approximation of the official MediaPipe HAND_CONNECTIONS set).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),             # knuckles (palm)
]

# Triplets (mcp, pip, tip) to estimate how "curled" each finger is: the
# angle at the PIP joint (or equivalent) approaches 180 degrees when the
# finger is extended, and decreases as the finger curls towards the palm.
FINGER_CURL_TRIPLETS = {
    "thumb": (2, 3, 4),      # MCP, IP, TIP (the thumb has no PIP)
    "index": (5, 6, 8),      # MCP, PIP, TIP
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


def compute_finger_curls(hand_xy: np.ndarray) -> dict[str, float]:
    """Flexion angle (degrees) for each finger: ~180 = extended, lower
    values = curled towards the palm. `hand_xy`: array (21, 2).
    """
    curls = {}
    for finger, (a_idx, b_idx, c_idx) in FINGER_CURL_TRIPLETS.items():
        curls[f"{finger}_curl"] = angle_at(hand_xy[a_idx], hand_xy[b_idx], hand_xy[c_idx])
    return curls


def hand_openness(hand_xy: np.ndarray) -> float:
    """Index 0 (closed fist) - 1 (open hand), based on the average
    wrist-to-tip distance of each finger, normalized by hand size (wrist
    to middle knuckle distance, robust to scale/distance from the
    camera).
    """
    palm_size = np.linalg.norm(hand_xy[WRIST] - hand_xy[9])  # middle knuckle
    if palm_size < 1e-6:
        return np.nan
    tip_distances = [np.linalg.norm(hand_xy[WRIST] - hand_xy[idx]) for idx in FINGER_TIPS.values()]
    avg_extension = np.mean(tip_distances) / palm_size
    # empirical normalization: closed fist ~1.0-1.3, open hand ~1.8-2.2
    return float(np.clip((avg_extension - 1.0) / 1.0, 0.0, 1.0))


def match_hands_to_wrists(hand_wrist_points: list[np.ndarray],
                           track_wrists: list[tuple[int, str, np.ndarray]],
                           max_distance: float = 60.0) -> dict[int, tuple[int, str]]:
    """Matches each detected hand (MediaPipe wrist point, index 0) to the
    nearest YOLO wrist among all tracked people.

    Parameters
    ----------
    hand_wrist_points : list of (x, y) points, one per detected hand
    track_wrists : list of (track_id, "left"|"right", YOLO wrist point)
    max_distance : threshold beyond which the match is discarded

    Returns
    -------
    dict {hand_index: (track_id, "left"|"right")}
    """
    assignments: dict[int, tuple[int, str]] = {}
    used: set[tuple[int, str]] = set()

    order = sorted(
        range(len(hand_wrist_points)),
        key=lambda i: min(
            (np.linalg.norm(hand_wrist_points[i] - w) for _, _, w in track_wrists),
            default=np.inf,
        ),
    )

    for hand_idx in order:
        best_key, best_dist = None, max_distance
        for tid, side, w in track_wrists:
            key = (tid, side)
            if key in used:
                continue
            d = float(np.linalg.norm(hand_wrist_points[hand_idx] - w))
            if d < best_dist:
                best_key, best_dist = key, d
        if best_key is not None:
            assignments[hand_idx] = best_key
            used.add(best_key)

    return assignments


# ---------------------------------------------------------------------------
# Wrapper over MediaPipe Tasks HandLandmarker (requires mediapipe + model)
# ---------------------------------------------------------------------------

@dataclass
class HandResult:
    landmarks_xy: np.ndarray   # (21, 2) in the frame's pixel coordinates
    handedness: str            # "Left" | "Right" (MediaPipe label, see note)


class HandTracker:
    """Wrapper over MediaPipe Tasks HandLandmarker.

    Note on handedness: MediaPipe's Left/Right label is computed from the
    point of view of the person in the image (not the camera), and can
    come out flipped depending on whether the feed is mirrored or not.
    For this reason, matching to a specific person/side in this pipeline
    is based on spatial proximity to the YOLO wrist
    (`match_hands_to_wrists`), not on MediaPipe's label.
    """

    def __init__(self, model_path: str = "hand_landmarker.task", num_hands: int = 4):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = resolve_model_path(model_path, download_url=_MODEL_URL)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandResult]:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame_bgr.shape[:2]
        out = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            xy = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
            label = handedness[0].category_name if handedness else "Unknown"
            out.append(HandResult(landmarks_xy=xy, handedness=label))
        return out
