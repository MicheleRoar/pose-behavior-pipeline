"""
pose/keypoints.py
==================
Shared COCO-17 keypoint schema constants -- ported as-is from
`pose-behavior-pipeline_legacy/src/pose/keypoints.py` (the pre-2026-08
pipeline), so keypoint names stay consistent with that codebase's
`features.py`/`chuv_features.py`/`anonymize.py` if/when they get ported
here too. `extract_keypoints.py` remaps MediaPipe's 33 BlazePose
landmarks down to this same 17-name schema (see its `BLAZEPOSE_TO_COCO`)
specifically so that downstream code doesn't need to care which model
produced the keypoints.

Having this as a separate module allows reusing the indices/names
elsewhere (a future `features.py`/`anonymize.py` port, analysis scripts)
without scattering "magic strings" around the code.
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

# Head keypoints, used for anonymization (face blurring) -- see
# pose-behavior-pipeline_legacy/src/pose/anonymize.py if/when it's ported.
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

# Standard COCO-17 skeleton connections, for drawing a skeleton overlay
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

# MediaPipe HandLandmarker's fixed 21-point output order (0-20), names as
# documented at https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# -- used by `pose/extract_hand_keypoints.py`. Unlike COCO-17 for the body,
# this is MediaPipe's own canonical naming (no remapping needed/possible --
# there's no other widely-used hand keypoint schema to normalize onto).
HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Standard hand skeleton connections, for drawing a hand overlay (same
# convention as SKELETON_EDGES above).
HAND_SKELETON_EDGES = [
    ("wrist", "thumb_cmc"), ("thumb_cmc", "thumb_mcp"), ("thumb_mcp", "thumb_ip"), ("thumb_ip", "thumb_tip"),
    ("wrist", "index_finger_mcp"), ("index_finger_mcp", "index_finger_pip"),
    ("index_finger_pip", "index_finger_dip"), ("index_finger_dip", "index_finger_tip"),
    ("wrist", "middle_finger_mcp"), ("middle_finger_mcp", "middle_finger_pip"),
    ("middle_finger_pip", "middle_finger_dip"), ("middle_finger_dip", "middle_finger_tip"),
    ("wrist", "ring_finger_mcp"), ("ring_finger_mcp", "ring_finger_pip"),
    ("ring_finger_pip", "ring_finger_dip"), ("ring_finger_dip", "ring_finger_tip"),
    ("wrist", "pinky_mcp"), ("pinky_mcp", "pinky_pip"), ("pinky_pip", "pinky_dip"), ("pinky_dip", "pinky_tip"),
    ("index_finger_mcp", "middle_finger_mcp"), ("middle_finger_mcp", "ring_finger_mcp"),
    ("ring_finger_mcp", "pinky_mcp"),
]

# MediaPipe FaceLandmarker's 52 ARKit-style blendshape names, in the fixed
# order the model returns them (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
# -- "Face blendshapes model" section). Used by `pose/extract_face_keypoints.py`
# to label the wide-format blendshape CSV columns.
FACE_BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
