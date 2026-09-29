"""
geometry.py
===========
Utility geometriche minime condivise tra `features.py` (angoli su
scheletro COCO-17) e `hands.py` (angoli sulle dita, 21 landmark per mano).
Isolata qui per evitare di duplicare la stessa formula in due moduli.
"""

from __future__ import annotations

import numpy as np


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angolo (in gradi) al vertice b, formato dai segmenti b-a e b-c."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return np.nan
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
