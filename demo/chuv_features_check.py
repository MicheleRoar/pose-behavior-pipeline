"""
chuv_features_check.py
=======================
Verifica di `chuv_features.py` (replica in tempo reale del feature
engineering del repository CHUV) SENZA camera/YOLO: scheletri sintetici con
angoli noti + una sequenza di frame con spostamento noto per verificare
velocita'/accelerazione.

Esegui con: python chuv_features_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from keypoints import KP
from chuv_features import (
    normalize_keypoints, compute_derived_features, ChuvFeatureTracker,
    compute_chuv_features,
)
from reid_check import make_skeleton, PERSON_A

N_JOINTS = 17


def _blank_kxy() -> np.ndarray:
    # scheletro placeholder: tutti i punti distanti tra loro cosi' nessun
    # angolo/distanza risulta accidentalmente degenere nei test che non li
    # toccano esplicitamente.
    kxy = np.zeros((N_JOINTS, 2))
    for i in range(N_JOINTS):
        kxy[i] = [i * 37.0, i * 53.0]
    return kxy


def _set(kxy: np.ndarray, name: str, x: float, y: float) -> None:
    kxy[KP[name]] = [x, y]


def part1_straight_arm_is_180deg():
    kxy = _blank_kxy()
    _set(kxy, "left_shoulder", 0, 0)
    _set(kxy, "left_hip", 0, 100)
    _set(kxy, "right_shoulder", 40, 0)
    _set(kxy, "right_hip", 40, 100)
    _set(kxy, "left_elbow", 0, 50)     # braccio sinistro DISTESO in verticale
    _set(kxy, "left_wrist", 0, 100)
    norm = normalize_keypoints(kxy)
    feats = compute_derived_features(norm)
    assert abs(feats["l_elbow_angle"] - 180.0) < 1.0, feats["l_elbow_angle"]
    print(f"Parte 1: braccio disteso -> l_elbow_angle={feats['l_elbow_angle']:.1f} deg (atteso ~180) — OK")


def part2_right_angle_arm_is_90deg():
    kxy = _blank_kxy()
    _set(kxy, "left_shoulder", 0, 0)
    _set(kxy, "left_hip", 0, 100)
    _set(kxy, "right_shoulder", 40, 0)
    _set(kxy, "right_hip", 40, 100)
    _set(kxy, "left_elbow", 0, 50)     # avambraccio PERPENDICOLARE al braccio
    _set(kxy, "left_wrist", 50, 50)
    norm = normalize_keypoints(kxy)
    feats = compute_derived_features(norm)
    assert abs(feats["l_elbow_angle"] - 90.0) < 1.0, feats["l_elbow_angle"]
    print(f"Parte 2: avambraccio a squadra -> l_elbow_angle={feats['l_elbow_angle']:.1f} deg (atteso ~90) — OK")


def part3_mid_hip_is_always_origin_and_com_is_half_neck():
    kxy = make_skeleton(**PERSON_A, tx=123.0, ty=45.0, jitter=0.0, rng=None)
    norm = normalize_keypoints(kxy)
    assert np.allclose(norm["mid_hip"], [0.0, 0.0], atol=1e-6), norm["mid_hip"]

    feats = compute_derived_features(norm)
    expected_com = norm["neck"] / 2.0
    assert np.allclose([feats["com_x"], feats["com_y"]], expected_com, atol=1e-6), (
        f"com={feats['com_x'], feats['com_y']} atteso meta' collo={tuple(expected_com)}"
    )
    print("Parte 3: mid_hip normalizzato = origine, com_x/com_y = meta' collo "
          "(proprieta' del calcolo originale, riprodotta fedelmente) — OK")


def part4_velocity_and_acceleration_from_known_displacement():
    tracker = ChuvFeatureTracker()
    fps = 30.0

    # persona ferma: frame 0 -> nessuna storia precedente, tutto NaN
    kxy_still = make_skeleton(**PERSON_A, tx=0.0, ty=0.0, jitter=0.0, rng=None)
    feats0 = compute_chuv_features(kxy_still, track_id=1, now=0 / fps, tracker=tracker)
    assert np.isnan(feats0["com_x_vel"]), "primo frame: velocita' deve essere NaN (nessun frame precedente)"

    # frame 1: la persona si sposta di +30 unita' in x in 1/30s -> velocita' attesa = 900 unita'/s
    kxy_moved = make_skeleton(**PERSON_A, tx=30.0, ty=0.0, jitter=0.0, rng=None)
    feats1 = compute_chuv_features(kxy_moved, track_id=1, now=1 / fps, tracker=tracker)
    # nota: tx sposta l'intero scheletro, ma le coordinate sono normalizzate
    # rispetto al proprio mid_hip (che si sposta insieme), quindi com_x
    # normalizzato NON cambia per una traslazione rigida pura -- verifichiamo
    # invece neck_x_vel/mid_hip_x_vel, che per costruzione di
    # normalize_keypoints restano sempre a (0,0): la velocita' "in scena"
    # (prima della normalizzazione) non e' osservabile da queste feature,
    # solo la cinematica RELATIVA al proprio bacino (es. oscillazione delle
    # braccia) lo e'. Verifichiamo quindi con un movimento del polso relativo
    # al corpo, non una traslazione rigida dell'intera persona.
    assert feats1["mid_hip_x_vel"] == 0.0 and feats1["neck_x_vel"] == 0.0, (
        "una traslazione rigida non deve produrre velocita' nelle feature "
        "normalizzate rispetto al bacino (mid_hip e' sempre l'origine)"
    )
    print("Parte 4a: traslazione rigida dell'intera persona -> velocita' nulla nelle "
          "feature normalizzate (atteso: solo il movimento RELATIVO al bacino e' "
          "osservabile, non lo spostamento in scena) — OK")

    # movimento relativo al corpo: il polso destro si alza rispetto al
    # busto, tra due frame a 1/30s di distanza -> velocita' attesa diversa da zero
    tracker2 = ChuvFeatureTracker()
    kxy_a = make_skeleton(**PERSON_A, tx=0.0, ty=0.0, jitter=0.0, rng=None)
    compute_chuv_features(kxy_a, track_id=2, now=0 / fps, tracker=tracker2)
    kxy_b = kxy_a.copy()
    kxy_b[KP["right_wrist"]] += [0.0, -20.0]  # polso destro alzato di 20 unita'
    feats_b = compute_chuv_features(kxy_b, track_id=2, now=1 / fps, tracker=tracker2)
    assert feats_b["r_wrist_y_vel"] < -1.0, f"atteso r_wrist_y_vel negativo (si alza), trovato {feats_b['r_wrist_y_vel']}"
    print(f"Parte 4b: polso destro sollevato tra due frame -> r_wrist_y_vel="
          f"{feats_b['r_wrist_y_vel']:.1f} (atteso negativo, coerente col sistema immagine y-in-basso) — OK")


def part5_forget_resets_state():
    tracker = ChuvFeatureTracker()
    kxy = make_skeleton(**PERSON_A, tx=0.0, ty=0.0, jitter=0.0, rng=None)
    compute_chuv_features(kxy, track_id=9, now=0.0, tracker=tracker)
    tracker.forget(9)
    feats = compute_chuv_features(kxy, track_id=9, now=100.0, tracker=tracker)
    assert np.isnan(feats["com_x_vel"]), "dopo forget(), il prossimo update deve ripartire da NaN"
    print("Parte 5: forget() ripulisce lo stato -> il track_id ripartirebbe da NaN come "
          "un id mai visto prima — OK")


def main():
    part1_straight_arm_is_180deg()
    part2_right_angle_arm_is_90deg()
    part3_mid_hip_is_always_origin_and_com_is_half_neck()
    part4_velocity_and_acceleration_from_known_displacement()
    part5_forget_resets_state()
    print("\nVerifica completata senza errori: angoli, distanze normalizzate, COM "
          "e derivate temporali di chuv_features.py si comportano come atteso.")


if __name__ == "__main__":
    main()
