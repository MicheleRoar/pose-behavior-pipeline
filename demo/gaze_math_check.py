"""
gaze_math_check.py
===================
Verifica della sola logica matematica di `gaze_head.py` (decomposizione
della matrice di rotazione in yaw/pitch/roll, euristica di joint attention)
SENZA MediaPipe né fotocamera: utile per validare la correttezza del codice
in qualunque ambiente, prima di testarlo con l'estimator reale sul Mac.

Esegui con: python gaze_math_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from gaze_head import rotation_matrix_to_euler, euler_to_rotation_matrix, joint_attention_score, bearing_to_target


def check_rotation_roundtrip():
    print("--- Round-trip yaw/pitch/roll -> matrice -> yaw/pitch/roll ---")
    cases = [(0, 0, 0), (30, 0, 0), (-45, 10, 5), (15, -20, 0), (60, -30, 15)]
    all_ok = True
    for yaw, pitch, roll in cases:
        R = euler_to_rotation_matrix(yaw, pitch, roll)
        y2, p2, r2 = rotation_matrix_to_euler(R)
        ok = abs(y2 - yaw) < 1e-3 and abs(p2 - pitch) < 1e-3 and abs(r2 - roll) < 1e-3
        all_ok &= ok
        print(f"  atteso=({yaw:6.1f},{pitch:6.1f},{roll:6.1f})  "
              f"ottenuto=({y2:6.2f},{p2:6.2f},{r2:6.2f})  OK={ok}")
    assert all_ok, "Round-trip fallito: controllare rotation_matrix_to_euler"
    print("  Tutti i casi OK.\n")


def check_joint_attention():
    print("--- Euristica di joint attention (proxy 2D) ---")
    frame_w = 640.0
    head = np.array([160.0, 200.0])
    target = np.array([480.0, 200.0])  # a destra, stessa altezza

    expected_yaw = bearing_to_target(head, target, frame_w)
    print(f"  bearing atteso verso il target: {expected_yaw:.1f} deg")

    s_aligned = joint_attention_score(head, expected_yaw, target, frame_w)
    s_straight = joint_attention_score(head, 0.0, target, frame_w)
    s_opposite = joint_attention_score(head, -expected_yaw, target, frame_w)

    print(f"  score con yaw allineato al target : {s_aligned:.2f}  (atteso ~1.0)")
    print(f"  score con yaw dritto in avanti     : {s_straight:.2f}  (atteso basso)")
    print(f"  score con yaw opposto al target    : {s_opposite:.2f}  (atteso 0.0)")

    assert s_aligned > 0.95, "Score atteso vicino a 1 quando yaw coincide col bearing"
    assert s_opposite == 0.0, "Score atteso 0 quando la testa guarda dalla parte opposta"
    assert s_straight < s_aligned, "Guardare dritto avanti deve avere score minore che guardare il target"
    print("  Tutti i controlli OK.\n")


if __name__ == "__main__":
    check_rotation_roundtrip()
    check_joint_attention()
    print("Verifica matematica completata senza errori.")
    print("Nota: HeadGazeEstimator (MediaPipe FaceLandmarker) richiede il modello")
    print("'face_landmarker.task' e una sorgente video reale — va testato sul Mac.")
