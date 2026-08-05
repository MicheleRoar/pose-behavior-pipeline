"""
mediapipe_pose_check.py
=========================
Verifica della logica pura di `pose/mediapipe_pose.py` (rimappatura
BlazePose -> COCO-17, calcolo del box di ritaglio con padding) SENZA
mediapipe/camera installati: usa oggetti landmark sintetici (x, y,
visibility normalizzati, come restituiti da MediaPipe) invece di una vera
inferenza.

Esegui con: python mediapipe_pose_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.keypoints import KP
from pose.mediapipe_pose import BLAZEPOSE_TO_COCO, blazepose_to_coco, padded_crop_box, _empty_pose


class _FakeLandmark:
    """Sostituto minimo di un NormalizedLandmark di MediaPipe: solo i campi
    letti da blazepose_to_coco (x, y normalizzati 0-1, visibility)."""
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def make_landmarks(overrides: dict[int, tuple[float, float, float]]) -> list[_FakeLandmark]:
    """33 landmark BlazePose, di default al centro (0.5, 0.5) con
    visibility 0.0 (cosi' i test possono verificare che gli indici NON
    sovrascritti restino "vuoti" nell'output), con gli indici in
    `overrides` impostati esplicitamente a (x, y, visibility)."""
    landmarks = [_FakeLandmark(0.5, 0.5, 0.0) for _ in range(33)]
    for idx, (x, y, vis) in overrides.items():
        landmarks[idx] = _FakeLandmark(x, y, vis)
    return landmarks


def part1_all_coco_joints_get_mapped():
    """Ogni voce di BLAZEPOSE_TO_COCO deve tradursi nel giunto COCO giusto,
    con le coordinate normalizzate riportate correttamente in pixel del
    frame intero (offset + scala del ritaglio)."""
    overrides = {blaze_idx: (0.5, 0.5, 0.9) for blaze_idx in BLAZEPOSE_TO_COCO}
    landmarks = make_landmarks(overrides)
    # ritaglio di 100x200 px con angolo in alto a sinistra a (300, 50) nel
    # frame intero: il centro normalizzato (0.5, 0.5) deve finire a
    # (300 + 50, 50 + 100) = (350, 150) in pixel del frame intero.
    kxy, kconf = blazepose_to_coco(landmarks, frame_offset_xy=(300.0, 50.0), crop_size_wh=(100.0, 200.0))

    for coco_name in BLAZEPOSE_TO_COCO.values():
        idx = KP[coco_name]
        assert np.allclose(kxy[idx], [350.0, 150.0]), (
            f"{coco_name}: atteso (350, 150), trovato {kxy[idx]}"
        )
        assert abs(kconf[idx] - 0.9) < 1e-6, f"{coco_name}: attesa confidenza 0.9, trovata {kconf[idx]}"
    print(f"Parte 1: tutti i {len(BLAZEPOSE_TO_COCO)} giunti COCO mappati da BlazePose finiscono "
          "nelle coordinate pixel attese (offset + scala del ritaglio applicati correttamente) — OK")


def part2_per_joint_confidence_is_preserved_not_averaged():
    """Ogni giunto COCO deve riportare la visibility del SUO landmark
    BlazePose specifico, non un valore medio/condiviso -- un giunto
    incerto (bassa visibility) non deve "contaminare" la confidenza di un
    giunto ben rilevato nello stesso frame."""
    landmarks = make_landmarks({
        0: (0.5, 0.5, 0.95),   # nose: alta confidenza
        7: (0.5, 0.5, 0.10),   # left_ear: bassa confidenza
    })
    kxy, kconf = blazepose_to_coco(landmarks, frame_offset_xy=(0.0, 0.0), crop_size_wh=(100.0, 100.0))

    assert abs(kconf[KP["nose"]] - 0.95) < 1e-6, f"nose: attesa confidenza 0.95, trovata {kconf[KP['nose']]}"
    assert abs(kconf[KP["left_ear"]] - 0.10) < 1e-6, (
        f"left_ear: attesa confidenza 0.10, trovata {kconf[KP['left_ear']]}"
    )
    print("Parte 2: la confidenza per giunto riflette la visibility del SUO landmark BlazePose "
          "specifico (nose=0.95, left_ear=0.10 restano distinti, non mescolati) — OK")


def part2b_empty_pose_is_all_nan_zero_confidence():
    """_empty_pose() (usata quando nessuna posa e' rilevata nel ritaglio)
    deve avere la stessa forma (17, 2)/(17,) di un rilevamento vero, tutta
    NaN/confidenza 0 -- cosi' il resto della pipeline (che si aspetta
    sempre un array di quella forma) non deve gestire un caso speciale."""
    kxy, kconf = _empty_pose()
    assert kxy.shape == (17, 2) and kconf.shape == (17,), f"forma inattesa: {kxy.shape}, {kconf.shape}"
    assert np.isnan(kxy).all(), "kxy di _empty_pose() deve essere tutto NaN"
    assert (kconf == 0.0).all(), "kconf di _empty_pose() deve essere tutto zero"
    print("Parte 2b: _empty_pose() ha la forma corretta (17,2)/(17,), tutta NaN/confidenza 0 — OK")


def part3_padded_crop_box_clamped_to_frame():
    """Il box di ritaglio va allargato del padding richiesto ma MAI oltre i
    bordi del frame (altrimenti si tenterebbe di ritagliare pixel
    inesistenti)."""
    frame_shape = (480, 640)  # (h, w)

    # box centrale: il padding non deve toccare i bordi
    bbox = np.array([200.0, 150.0, 300.0, 350.0])
    x1, y1, x2, y2 = padded_crop_box(bbox, frame_shape, padding=0.1)
    bw, bh = 100.0, 200.0
    assert x1 == int(200 - bw * 0.1) and y1 == int(150 - bh * 0.1)
    assert x2 == int(300 + bw * 0.1) and y2 == int(350 + bh * 0.1)

    # box vicino al bordo in alto a sinistra: il padding sfonderebbe (x<0,
    # y<0) senza il clamping
    bbox_edge = np.array([5.0, 5.0, 60.0, 80.0])
    x1e, y1e, x2e, y2e = padded_crop_box(bbox_edge, frame_shape, padding=0.5)
    assert x1e == 0 and y1e == 0, f"atteso clamping a (0,0), trovato ({x1e},{y1e})"

    # box vicino al bordo in basso a destra: stesso discorso su x2/y2
    bbox_edge2 = np.array([600.0, 440.0, 638.0, 478.0])
    x1e2, y1e2, x2e2, y2e2 = padded_crop_box(bbox_edge2, frame_shape, padding=0.5)
    assert x2e2 == 640 and y2e2 == 480, f"atteso clamping a (640,480), trovato ({x2e2},{y2e2})"

    print("Parte 3: il box di ritaglio con padding resta sempre dentro i bordi del frame — OK")


def main():
    part1_all_coco_joints_get_mapped()
    part2_per_joint_confidence_is_preserved_not_averaged()
    part2b_empty_pose_is_all_nan_zero_confidence()
    part3_padded_crop_box_clamped_to_frame()
    print("\nVerifica completata senza errori: la rimappatura BlazePose -> COCO-17 e il calcolo "
          "del box di ritaglio in mediapipe_pose.py si comportano come atteso.")
    print("Nota: la classe MediaPipeCropPoseEstimator (che richiama mediapipe/una vera camera) "
          "non e' testabile in questo ambiente sandbox -- va verificata sul Mac.")


if __name__ == "__main__":
    main()
