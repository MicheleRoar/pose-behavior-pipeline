"""
chuv_features.py
=================
Real-time replica of the CHUV repository's feature engineering
(Video-Annotation-System, `src/models/train.py::normalize_keypoints` +
`add_derived_pose_features` + `add_temporal_features`), adapted from
BODY-25 (psifx/SAM3 + MediaPipe, OpenPose-style) to COCO-17 (YOLO-pose)
and from an offline/batch computation to a real-time frame-by-frame one.

Why here and not in the CHUV repository: same reason as `reid.py` -- that
pipeline requires SAM3 on a CUDA GPU, not available on a MacBook M1. The
feature engineering (angles, distances, symmetry, center of mass,
temporal derivatives) doesn't depend on SAM3/psifx: it's pure geometry on
2D keypoints, so it's portable and testable here with YOLO+ByteTrack on
unprotected data -- this module's goal is to see the SAME NUMBERS the
original repository would compute, but produced by a much lighter
tracker.

What is NOT replicated
-------------------------
- 5 columns of the CHUV repository's final feature set are raw
  toe-tip/heel/background coordinates (l_big_toe_y, l_heel_y,
  r_small_toe_x, r_small_toe_y, r_heel_x, background_y): they come from
  the BODY-25 (OpenPose) schema, which includes them; COCO-17
  (YOLO-pose) does NOT have them, so these columns aren't reproducible
  here.
- The trained model (model_xgboost.joblib) is NOT loaded: its classes
  are clinical annotation codes specific to a format (WAKEE) that
  require data labeled by a human observer following a protocol that
  doesn't exist here, and the file mapping the model's numeric indices
  to labels (the LabelEncoder) isn't saved by the original repository.
  This module stops at feature engineering.

Deliberate difference: temporal derivatives
-------------------------------------------
In the CHUV repository velocities/accelerations are computed with
`df.groupby(annotation_label).diff()` -- a side effect of the fact that
training is offline and grouped by class (the derivative resets to zero
at the boundaries between annotation classes). Here, in real time, "the
class" of the current frame doesn't exist while it's being acquired, so
the derivative is computed continuously, frame by frame, for each
person_id/track_id (see `ChuvFeatureTracker`) -- more physically correct,
but not a number directly comparable 1:1 with the original repository's
output on the same video.

Note on the center of mass (com_x, com_y): after normalization relative
to the pelvis (mid_hip always becomes the origin (0,0)), com_x/com_y
mathematically reduce to half the neck position -- not a true
multi-segment physical center of mass. This is a characteristic of the
CHUV repository's original computation (`com_x = (mid_hip_x + neck_x) /
2` on already-normalized coordinates), faithfully reproduced here, not
"corrected": this module's goal is fidelity to the repository, not
improving it.

Note on normalization: in the CHUV repository, a torso_length of zero is
replaced with the median computed over the entire offline dataset
(`normalize_keypoints`). Here, in real time, "the entire dataset" from
which to estimate a median in advance doesn't exist: if a frame's
torso_length is invalid (nan/too small), that frame's normalized
coordinates are NaN -- a deliberately honest choice rather than an
arbitrary fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pose.keypoints import KP

# ---------------------------------------------------------------------------
# BODY-25 "virtual" keypoints reconstructed from COCO-17
# ---------------------------------------------------------------------------

def _neck_xy(kxy: np.ndarray) -> np.ndarray:
    return (kxy[KP["left_shoulder"]] + kxy[KP["right_shoulder"]]) / 2.0


def _mid_hip_xy(kxy: np.ndarray) -> np.ndarray:
    return (kxy[KP["left_hip"]] + kxy[KP["right_hip"]]) / 2.0


RAW_POINTS = [
    "nose", "neck", "mid_hip",
    "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist",
    "r_hip", "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

_COCO_NAME = {  # CHUV name (r_/l_ prefix, BODY-25 style) -> COCO-17 name (KP)
    "r_shoulder": "right_shoulder", "l_shoulder": "left_shoulder",
    "r_elbow": "right_elbow", "l_elbow": "left_elbow",
    "r_wrist": "right_wrist", "l_wrist": "left_wrist",
    "r_hip": "right_hip", "l_hip": "left_hip",
    "r_knee": "right_knee", "l_knee": "left_knee",
    "r_ankle": "right_ankle", "l_ankle": "left_ankle",
    "r_eye": "right_eye", "l_eye": "left_eye",
    "r_ear": "right_ear", "l_ear": "left_ear",
}


def _raw_point(kxy: np.ndarray, name: str) -> np.ndarray:
    if name == "neck":
        return _neck_xy(kxy)
    if name == "mid_hip":
        return _mid_hip_xy(kxy)
    if name == "nose":
        return kxy[KP["nose"]]
    return kxy[KP[_COCO_NAME[name]]]


# ---------------------------------------------------------------------------
# Stage 1: normalization relative to the pelvis (identical to
# normalize_keypoints from the CHUV repository, except for the fallback
# note above)
# ---------------------------------------------------------------------------

def normalize_keypoints(kxy: np.ndarray) -> dict[str, np.ndarray]:
    """Coordinates normalized relative to the pelvis: (x - mid_hip_x) /
    torso_length, (y - mid_hip_y) / torso_length -- same formula as
    `train.py::normalize_keypoints` in the CHUV repository. Returns a
    dict name -> array (2,) [x, y]; NaN where the source keypoint(s) are
    missing or torso_length is invalid."""
    mid_hip = _mid_hip_xy(kxy)
    neck = _neck_xy(kxy)
    torso = float(np.linalg.norm(neck - mid_hip))
    valid_torso = np.isfinite(torso) and torso >= 1e-3

    out = {}
    for name in RAW_POINTS:
        p = _raw_point(kxy, name)
        out[name] = (p - mid_hip) / torso if valid_torso else np.full(2, np.nan)
    return out


# ---------------------------------------------------------------------------
# Stage 2: derived features (angles, distances, symmetry, COM, spread) --
# same logic as add_derived_pose_features from the CHUV repository, on
# already-normalized coordinates.
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


CHUV_ANGLE_TRIPLETS: dict[str, tuple[str, str, str]] = {
    "r_elbow_angle": ("r_shoulder", "r_elbow", "r_wrist"),
    "l_elbow_angle": ("l_shoulder", "l_elbow", "l_wrist"),
    "r_shoulder_angle": ("neck", "r_shoulder", "r_elbow"),
    "l_shoulder_angle": ("neck", "l_shoulder", "l_elbow"),
    "r_knee_angle": ("r_hip", "r_knee", "r_ankle"),
    "l_knee_angle": ("l_hip", "l_knee", "l_ankle"),
    "r_hip_angle": ("mid_hip", "r_hip", "r_knee"),
    "l_hip_angle": ("mid_hip", "l_hip", "l_knee"),
    "trunk_angle": ("nose", "neck", "mid_hip"),
}
# note: r_shoulder_angle/l_shoulder_angle here use the "neck" vertex (as
# in the CHUV repository) -- a different definition from the
# left_shoulder_angle/right_shoulder_angle already present in features.py
# (which use the "hip" vertex), so they are NOT duplicates: they are two
# shoulder angles computed with two different conventions, both kept.

DERIVED_COLS = [
    *CHUV_ANGLE_TRIPLETS.keys(),
    "eye_to_eye", "nose_to_neck", "r_wrist_to_hip", "l_wrist_to_hip",
    "r_wrist_to_nose", "l_wrist_to_nose", "nose_to_ankles", "hip_to_ankle",
    "shoulder_y_diff", "hip_y_diff", "elbow_angle_diff", "knee_angle_diff",
    "wrist_to_hip_diff", "shoulder_angle_diff",
    "com_x", "com_y", "body_spread_x", "body_spread_y",
]


def compute_derived_features(norm: dict[str, np.ndarray]) -> dict[str, float]:
    """Angles, distances, symmetry, COM and spread -- same formula as
    `add_derived_pose_features` in the CHUV repository, applied to the
    coordinates already normalized by `normalize_keypoints`."""
    out: dict[str, float] = {}

    for name, (a, b, c) in CHUV_ANGLE_TRIPLETS.items():
        out[name] = _angle_deg(norm[a], norm[b], norm[c])

    out["eye_to_eye"] = _dist(norm["l_eye"], norm["r_eye"])
    out["nose_to_neck"] = _dist(norm["nose"], norm["neck"])
    out["r_wrist_to_hip"] = _dist(norm["r_wrist"], norm["mid_hip"])
    out["l_wrist_to_hip"] = _dist(norm["l_wrist"], norm["mid_hip"])
    out["r_wrist_to_nose"] = _dist(norm["r_wrist"], norm["nose"])
    out["l_wrist_to_nose"] = _dist(norm["l_wrist"], norm["nose"])
    out["nose_to_ankles"] = (_dist(norm["nose"], norm["l_ankle"]) + _dist(norm["nose"], norm["r_ankle"])) / 2.0
    out["hip_to_ankle"] = (_dist(norm["mid_hip"], norm["l_ankle"]) + _dist(norm["mid_hip"], norm["r_ankle"])) / 2.0

    out["shoulder_y_diff"] = float(norm["l_shoulder"][1] - norm["r_shoulder"][1])
    out["hip_y_diff"] = float(norm["l_hip"][1] - norm["r_hip"][1])
    out["elbow_angle_diff"] = out["l_elbow_angle"] - out["r_elbow_angle"]
    out["knee_angle_diff"] = out["l_knee_angle"] - out["r_knee_angle"]
    out["wrist_to_hip_diff"] = out["l_wrist_to_hip"] - out["r_wrist_to_hip"]
    out["shoulder_angle_diff"] = out["l_shoulder_angle"] - out["r_shoulder_angle"]

    out["com_x"] = float((norm["mid_hip"][0] + norm["neck"][0]) / 2.0)
    out["com_y"] = float((norm["mid_hip"][1] + norm["neck"][1]) / 2.0)

    spread_x_pts = np.array([norm["l_wrist"][0], norm["r_wrist"][0], norm["l_ankle"][0], norm["r_ankle"][0]])
    spread_y_pts = np.array([norm["nose"][1], norm["l_ankle"][1], norm["r_ankle"][1]])
    out["body_spread_x"] = float(np.nanmax(spread_x_pts) - np.nanmin(spread_x_pts))
    out["body_spread_y"] = float(np.nanmax(spread_y_pts) - np.nanmin(spread_y_pts))

    return out


# ---------------------------------------------------------------------------
# Stage 3: temporal derivatives (velocity/acceleration) -- same keypoint
# selection as add_temporal_features in the CHUV repository, but computed
# frame-by-frame in real time (see "Deliberate difference" in the module
# docstring).
# ---------------------------------------------------------------------------

TEMPORAL_POINTS = ["com", "nose", "l_wrist", "r_wrist", "l_ankle", "r_ankle", "neck", "mid_hip"]
TEMPORAL_COLS = [f"{name}_{axis}_{kind}" for name in TEMPORAL_POINTS
                  for axis in ("x", "y") for kind in ("vel", "acc")]


@dataclass
class ChuvFeatureTracker:
    """Keeps, per person_id/track_id, the last normalized frame and the
    last computed velocity, to derive velocity/acceleration frame-by-frame
    without having to keep the entire session in memory (unlike the CHUV
    repository, which operates offline on a complete CSV).
    """
    _prev: dict[int, dict] = field(default_factory=dict)

    def update(self, track_id: int, norm: dict[str, np.ndarray], now: float) -> dict[str, float]:
        points = {
            "com": np.array([(norm["mid_hip"][0] + norm["neck"][0]) / 2.0,
                              (norm["mid_hip"][1] + norm["neck"][1]) / 2.0]),
            **{name: norm[name] for name in TEMPORAL_POINTS if name != "com"},
        }

        prev = self._prev.get(track_id)
        out: dict[str, float] = {}

        if prev is None:
            for col in TEMPORAL_COLS:
                out[col] = np.nan
            self._prev[track_id] = {
                "points": points,
                "vel": {name: np.full(2, np.nan) for name in TEMPORAL_POINTS},
                "t": now,
            }
            return out

        dt = max(now - prev["t"], 1e-3)
        vel = {}
        for name in TEMPORAL_POINTS:
            v = (points[name] - prev["points"][name]) / dt
            vel[name] = v
            out[f"{name}_x_vel"], out[f"{name}_y_vel"] = float(v[0]), float(v[1])
            a = (v - prev["vel"][name]) / dt
            out[f"{name}_x_acc"], out[f"{name}_y_acc"] = float(a[0]), float(a[1])

        self._prev[track_id] = {"points": points, "vel": vel, "t": now}
        return out

    def forget(self, track_id: int) -> None:
        """To be called when a track_id/person_id definitively leaves the
        frame, so as not to leave state attached to an id that won't
        reappear -- and to avoid a velocity close to zero (instead of
        NaN) if that same id is reassigned much later after a huge gap
        (e.g. from `reid.py` after a long re-entry)."""
        self._prev.pop(track_id, None)


# ---------------------------------------------------------------------------
# Single entry point for use in live_demo.py
# ---------------------------------------------------------------------------

def compute_chuv_features(kxy: np.ndarray, track_id: int, now: float,
                           tracker: ChuvFeatureTracker) -> dict[str, float]:
    """All "CHUV-style" features for a single frame of a person:
    normalized coordinates, derived features (angles/distances/symmetry/
    COM/spread) and temporal derivatives (velocity/acceleration).
    """
    norm = normalize_keypoints(kxy)
    out: dict[str, float] = {}
    for name, xy in norm.items():
        out[f"{name}_x"], out[f"{name}_y"] = float(xy[0]), float(xy[1])
    out.update(compute_derived_features(norm))
    out.update(tracker.update(track_id, norm, now))
    return out
