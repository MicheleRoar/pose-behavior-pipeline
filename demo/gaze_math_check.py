"""
gaze_math_check.py
===================
Verifies only the math logic of `gaze_head.py` (rotation matrix
decomposition into yaw/pitch/roll, joint attention heuristic) WITHOUT
MediaPipe or a camera: useful to validate the code's correctness in any
environment, before testing it with the real estimator on the Mac.

Run with: python gaze_math_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.gaze_head import rotation_matrix_to_euler, euler_to_rotation_matrix, joint_attention_score, bearing_to_target


def check_rotation_roundtrip():
    print("--- Round-trip yaw/pitch/roll -> matrix -> yaw/pitch/roll ---")
    cases = [(0, 0, 0), (30, 0, 0), (-45, 10, 5), (15, -20, 0), (60, -30, 15)]
    all_ok = True
    for yaw, pitch, roll in cases:
        R = euler_to_rotation_matrix(yaw, pitch, roll)
        y2, p2, r2 = rotation_matrix_to_euler(R)
        ok = abs(y2 - yaw) < 1e-3 and abs(p2 - pitch) < 1e-3 and abs(r2 - roll) < 1e-3
        all_ok &= ok
        print(f"  expected=({yaw:6.1f},{pitch:6.1f},{roll:6.1f})  "
              f"got=({y2:6.2f},{p2:6.2f},{r2:6.2f})  OK={ok}")
    assert all_ok, "Round-trip failed: check rotation_matrix_to_euler"
    print("  All cases OK.\n")


def check_joint_attention():
    print("--- Joint attention heuristic (2D proxy) ---")
    frame_w = 640.0
    head = np.array([160.0, 200.0])
    target = np.array([480.0, 200.0])  # to the right, same height

    expected_yaw = bearing_to_target(head, target, frame_w)
    print(f"  expected bearing towards target: {expected_yaw:.1f} deg")

    s_aligned = joint_attention_score(head, expected_yaw, target, frame_w)
    s_straight = joint_attention_score(head, 0.0, target, frame_w)
    s_opposite = joint_attention_score(head, -expected_yaw, target, frame_w)

    print(f"  score with yaw aligned to target : {s_aligned:.2f}  (expected ~1.0)")
    print(f"  score with yaw straight ahead     : {s_straight:.2f}  (expected low)")
    print(f"  score with yaw opposite to target : {s_opposite:.2f}  (expected 0.0)")

    assert s_aligned > 0.95, "Expected score close to 1 when yaw matches the bearing"
    assert s_opposite == 0.0, "Expected score 0 when the head looks the opposite way"
    assert s_straight < s_aligned, "Looking straight ahead must score lower than looking at the target"
    print("  All checks OK.\n")


if __name__ == "__main__":
    check_rotation_roundtrip()
    check_joint_attention()
    print("Math verification completed with no errors.")
    print("Note: HeadGazeEstimator (MediaPipe FaceLandmarker) requires the")
    print("'face_landmarker.task' model and a real video source -- must be tested on the Mac.")
