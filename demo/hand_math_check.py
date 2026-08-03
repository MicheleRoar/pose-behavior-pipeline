"""
hand_math_check.py
===================
Verifica della sola logica matematica di `hands.py` (angoli di flessione
delle dita, indice di apertura della mano, matching mani<->polsi) SENZA
MediaPipe né fotocamera, usando due mani sintetiche a 21 landmark: una
completamente aperta e una a pugno chiuso.

Esegui con: python hand_math_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from hands import compute_finger_curls, hand_openness, match_hands_to_wrists, FINGER_CURL_TRIPLETS


def make_open_hand(origin: np.ndarray) -> np.ndarray:
    """Mano sintetica con tutte le dita completamente distese, radiali
    rispetto al polso (angoli di flessione attesi ~180 gradi)."""
    hand = np.zeros((21, 2))
    hand[0] = origin  # wrist
    finger_dirs = {
        "thumb": (-40, -60), "index": (-20, -100), "middle": (0, -110),
        "ring": (20, -100), "pinky": (40, -85),
    }
    joints_per_finger = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8],
                          "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
                          "pinky": [17, 18, 19, 20]}
    for finger, idxs in joints_per_finger.items():
        dx, dy = finger_dirs[finger]
        length = np.linalg.norm([dx, dy])
        direction = np.array([dx, dy]) / length
        for i, idx in enumerate(idxs, start=1):
            hand[idx] = origin + direction * (length * i / len(idxs))
    return hand


def make_fist_hand(origin: np.ndarray) -> np.ndarray:
    """Mano sintetica a pugno chiuso: le punte delle dita ripiegate verso
    il palmo (angoli di flessione attesi molto minori di 180 gradi)."""
    hand = np.zeros((21, 2))
    hand[0] = origin
    joints_per_finger = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8],
                          "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
                          "pinky": [17, 18, 19, 20]}
    mcp_offsets = {"thumb": (-30, -40), "index": (-15, -70), "middle": (0, -75),
                   "ring": (15, -70), "pinky": (30, -60)}
    for finger, idxs in joints_per_finger.items():
        mcp = origin + np.array(mcp_offsets[finger])
        # pip e dip/tip ripiegati verso il basso (verso il polso), non in linea con MCP
        fold_dir = (mcp - origin)
        fold_dir = fold_dir / np.linalg.norm(fold_dir)
        perp = np.array([-fold_dir[1], fold_dir[0]])
        hand[idxs[0]] = mcp
        hand[idxs[1]] = mcp + perp * 15 - fold_dir * 5    # pip: piega laterale
        hand[idxs[2]] = mcp + perp * 10 + fold_dir * 10   # dip: torna verso il polso
        hand[idxs[3]] = mcp + fold_dir * 15               # tip: vicino al polso
    return hand


def main():
    origin = np.array([300.0, 300.0])
    open_hand = make_open_hand(origin)
    fist_hand = make_fist_hand(origin)

    print("--- Angoli di flessione delle dita ---")
    open_curls = compute_finger_curls(open_hand)
    fist_curls = compute_finger_curls(fist_hand)
    for finger in FINGER_CURL_TRIPLETS:
        o, f = open_curls[f"{finger}_curl"], fist_curls[f"{finger}_curl"]
        print(f"  {finger:8s}  mano aperta={o:6.1f} deg   pugno={f:6.1f} deg   "
              f"OK={'si' if o > f else 'NO'}")
        assert o > f, f"attesa mano aperta ({o}) con angolo maggiore del pugno ({f}) per {finger}"

    print("\n--- Indice di apertura della mano ---")
    o_open = hand_openness(open_hand)
    o_fist = hand_openness(fist_hand)
    print(f"  mano aperta: {o_open:.2f}  (atteso alto, vicino a 1)")
    print(f"  pugno:       {o_fist:.2f}  (atteso basso, vicino a 0)")
    assert o_open > o_fist, "L'indice di apertura deve essere maggiore per la mano aperta"

    print("\n--- Matching mani -> polsi tracciati ---")
    # due persone (track 1 e 2), ciascuna con polso sinistro e destro
    track_wrists = [
        (1, "left", np.array([100.0, 400.0])), (1, "right", np.array([200.0, 400.0])),
        (2, "left", np.array([500.0, 400.0])), (2, "right", np.array([600.0, 400.0])),
    ]
    # tre mani rilevate, vicine rispettivamente a (1,right), (2,left), (2,right)
    hand_wrist_points = [np.array([205.0, 402.0]), np.array([498.0, 401.0]), np.array([603.0, 399.0])]
    assignment = match_hands_to_wrists(hand_wrist_points, track_wrists)
    print(" ", assignment)
    assert assignment[0] == (1, "right")
    assert assignment[1] == (2, "left")
    assert assignment[2] == (2, "right")
    print("  Matching corretto.")

    print("\nVerifica matematica completata senza errori.")
    print("Nota: HandTracker (MediaPipe HandLandmarker) richiede il modello")
    print("'hand_landmarker.task' e una sorgente video reale — va testato sul Mac.")


if __name__ == "__main__":
    main()
