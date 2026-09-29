"""
keypoints.py
============
Shared constants and utilities for the COCO-17 keypoint schema, used both
by Ultralytics YOLO-pose and by many other pose estimation pipelines.

Having this as a separate module allows reusing the indices in
`features.py`, `anonymize.py`, and analysis scripts, avoiding "magic
numbers" scattered around the code.
"""

from __future__ import annotations

# COCO-17 schema: index -> keypoint name
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

KP = {name: idx for idx, name in enumerate(COCO17)}

# Head keypoints, used for anonymization (face blurring)
HEAD_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]

# Left/right pairs for computing symmetry indices
LR_PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

# Standard COCO-17 skeleton connections, used to draw the skeleton over the
# video frame (real-time overlay)
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]

# (a, b, c) triplets for computing joint angles at joint b
JOINT_ANGLE_TRIPLETS = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
}
