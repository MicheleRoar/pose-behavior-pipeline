"""
keypoints.py
============
Costanti e utility condivise per lo schema di keypoint COCO-17, usato sia da
Ultralytics YOLO-pose sia da molte altre pipeline di pose estimation.

Avere questo modulo separato permette di riutilizzare gli indici in
`features.py`, `anonymize.py` e negli script di analisi, evitando "numeri
magici" sparsi nel codice.
"""

from __future__ import annotations

# Schema COCO-17: indice -> nome del keypoint
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

KP = {name: idx for idx, name in enumerate(COCO17)}

# Keypoint della testa, usati per l'anonimizzazione (blur del volto)
HEAD_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]

# Coppie sinistra/destra per il calcolo di indici di simmetria
LR_PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

# Connessioni scheletriche standard COCO-17, usate per disegnare lo
# scheletro sopra il frame video (overlay in tempo reale)
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

# Terne (a, b, c) per il calcolo di angoli articolari all'articolazione b
JOINT_ANGLE_TRIPLETS = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
}
