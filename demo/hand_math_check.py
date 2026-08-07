"""
hand_math_check.py
===================
Verifies only the math logic of `hands.py` (finger flexion angles, hand
openness index, hand<->wrist matching) WITHOUT MediaPipe or a camera,
using two synthetic 21-landmark hands: one fully open and one closed in a
fist.

Run with: python hand_math_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.hands import compute_finger_curls, hand_openness, match_hands_to_wrists, FINGER_CURL_TRIPLETS


def make_open_hand(origin: np.ndarray) -> np.ndarray:
    """Synthetic hand with all fingers fully extended, radial relative to
    the wrist (expected flexion angles ~180 degrees)."""
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
    """Synthetic closed-fist hand: fingertips folded back towards the
    palm (expected flexion angles much smaller than 180 degrees)."""
    hand = np.zeros((21, 2))
    hand[0] = origin
    joints_per_finger = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8],
                          "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
                          "pinky": [17, 18, 19, 20]}
    mcp_offsets = {"thumb": (-30, -40), "index": (-15, -70), "middle": (0, -75),
                   "ring": (15, -70), "pinky": (30, -60)}
    for finger, idxs in joints_per_finger.items():
        mcp = origin + np.array(mcp_offsets[finger])
        # pip and dip/tip folded downward (towards the wrist), not aligned with MCP
        fold_dir = (mcp - origin)
        fold_dir = fold_dir / np.linalg.norm(fold_dir)
        perp = np.array([-fold_dir[1], fold_dir[0]])
        hand[idxs[0]] = mcp
        hand[idxs[1]] = mcp + perp * 15 - fold_dir * 5    # pip: sideways fold
        hand[idxs[2]] = mcp + perp * 10 + fold_dir * 10   # dip: bends back towards the wrist
        hand[idxs[3]] = mcp + fold_dir * 15               # tip: close to the wrist
    return hand


def main():
    origin = np.array([300.0, 300.0])
    open_hand = make_open_hand(origin)
    fist_hand = make_fist_hand(origin)

    print("--- Finger flexion angles ---")
    open_curls = compute_finger_curls(open_hand)
    fist_curls = compute_finger_curls(fist_hand)
    for finger in FINGER_CURL_TRIPLETS:
        o, f = open_curls[f"{finger}_curl"], fist_curls[f"{finger}_curl"]
        print(f"  {finger:8s}  open hand={o:6.1f} deg   fist={f:6.1f} deg   "
              f"OK={'yes' if o > f else 'NO'}")
        assert o > f, f"expected open hand ({o}) to have a larger angle than the fist ({f}) for {finger}"

    print("\n--- Hand openness index ---")
    o_open = hand_openness(open_hand)
    o_fist = hand_openness(fist_hand)
    print(f"  open hand: {o_open:.2f}  (expected high, close to 1)")
    print(f"  fist:      {o_fist:.2f}  (expected low, close to 0)")
    assert o_open > o_fist, "The openness index must be higher for the open hand"

    print("\n--- Matching hands -> tracked wrists ---")
    # two people (track 1 and 2), each with a left and right wrist
    track_wrists = [
        (1, "left", np.array([100.0, 400.0])), (1, "right", np.array([200.0, 400.0])),
        (2, "left", np.array([500.0, 400.0])), (2, "right", np.array([600.0, 400.0])),
    ]
    # three detected hands, close to (1,right), (2,left), (2,right) respectively
    hand_wrist_points = [np.array([205.0, 402.0]), np.array([498.0, 401.0]), np.array([603.0, 399.0])]
    assignment = match_hands_to_wrists(hand_wrist_points, track_wrists)
    print(" ", assignment)
    assert assignment[0] == (1, "right")
    assert assignment[1] == (2, "left")
    assert assignment[2] == (2, "right")
    print("  Correct matching.")

    print("\nMath verification completed with no errors.")
    print("Note: HandTracker (MediaPipe HandLandmarker) requires the")
    print("'hand_landmarker.task' model and a real video source -- must be tested on the Mac.")


if __name__ == "__main__":
    main()
