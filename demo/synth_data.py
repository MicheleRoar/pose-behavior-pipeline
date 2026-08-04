"""
synth_data.py
=============
Generatori di sequenze di keypoint COCO-17 sintetiche (bambino con
movimento ripetitivo, caregiver con movimento naturale), condivisi tra
`synthetic_demo.py` (verifica batch di features.py) e `live_render_check.py`
(verifica del rendering live frame-by-frame). Nessun dato reale coinvolto.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.keypoints import KP


def base_skeleton(center_x: float, center_y: float, scale: float = 1.0) -> np.ndarray:
    """Scheletro COCO-17 di riposo, in proporzioni antropometriche
    approssimate, centrato su (center_x, center_y).
    """
    layout = {
        "nose": (0, -70), "left_eye": (-5, -74), "right_eye": (5, -74),
        "left_ear": (-10, -72), "right_ear": (10, -72),
        "left_shoulder": (-20, -50), "right_shoulder": (20, -50),
        "left_elbow": (-30, -20), "right_elbow": (30, -20),
        "left_wrist": (-35, 5), "right_wrist": (35, 5),
        "left_hip": (-15, 0), "right_hip": (15, 0),
        "left_knee": (-18, 45), "right_knee": (18, 45),
        "left_ankle": (-20, 90), "right_ankle": (20, 90),
    }
    kpts = np.array([layout[name] for name in KP], dtype=float) * scale
    kpts[:, 0] += center_x
    kpts[:, 1] += center_y
    return kpts


def make_child_sequence(n_frames: int, fps: float, seed: int | None = None) -> np.ndarray:
    """Bambino: corpo sostanzialmente fermo, braccio destro con movimento
    ripetitivo ad alta frequenza (~3 Hz) e piccola ampiezza — proxy
    tecnico/dimostrativo di una stereotipia motoria.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    seq = np.tile(base_skeleton(300, 300, scale=0.8), (n_frames, 1, 1))
    freq_hz, amplitude = 3.0, 25.0
    oscillation = amplitude * np.sin(2 * np.pi * freq_hz * t)
    seq[:, KP["right_wrist"], 1] += oscillation
    seq[:, KP["right_wrist"], 0] += 0.4 * oscillation
    seq += rng.normal(0, 0.6, seq.shape)
    return seq


def make_caregiver_sequence(n_frames: int, fps: float, seed: int | None = None) -> np.ndarray:
    """Caregiver: movimento di reaching lento e irregolare verso il
    bambino, nessuna periodicità dominante.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    seq = np.tile(base_skeleton(500, 300, scale=1.1), (n_frames, 1, 1))
    slow_drift = 15 * np.sin(2 * np.pi * 0.15 * t) + 5 * np.sin(2 * np.pi * 0.4 * t + 1.0)
    seq[:, KP["left_wrist"], 0] -= slow_drift
    seq[:, KP["left_wrist"], 1] += 0.3 * slow_drift
    seq[:, :, 0] -= (t / (n_frames / fps))[:, None] * 40
    seq += rng.normal(0, 0.8, seq.shape)
    return seq
