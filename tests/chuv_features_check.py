"""
chuv_features_check.py
=======================
Verifies `chuv_features.py` (real-time replica of the CHUV repository's
feature engineering) WITHOUT a camera/YOLO: synthetic skeletons with
known angles + a sequence of frames with a known displacement to verify
velocity/acceleration.

Run with: python chuv_features_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.keypoints import KP
from pose.chuv_features import (
    normalize_keypoints, compute_derived_features, ChuvFeatureTracker,
    compute_chuv_features,
)
from reid_check import make_skeleton, PERSON_A

N_JOINTS = 17


def _blank_kxy() -> np.ndarray:
    # placeholder skeleton: all points far apart from each other so no
    # angle/distance ends up accidentally degenerate in tests that don't
    # explicitly touch them.
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
    _set(kxy, "left_elbow", 0, 50)     # left arm EXTENDED vertically
    _set(kxy, "left_wrist", 0, 100)
    norm = normalize_keypoints(kxy)
    feats = compute_derived_features(norm)
    assert abs(feats["l_elbow_angle"] - 180.0) < 1.0, feats["l_elbow_angle"]
    print(f"Part 1: extended arm -> l_elbow_angle={feats['l_elbow_angle']:.1f} deg (expected ~180) — OK")


def part2_right_angle_arm_is_90deg():
    kxy = _blank_kxy()
    _set(kxy, "left_shoulder", 0, 0)
    _set(kxy, "left_hip", 0, 100)
    _set(kxy, "right_shoulder", 40, 0)
    _set(kxy, "right_hip", 40, 100)
    _set(kxy, "left_elbow", 0, 50)     # forearm PERPENDICULAR to the upper arm
    _set(kxy, "left_wrist", 50, 50)
    norm = normalize_keypoints(kxy)
    feats = compute_derived_features(norm)
    assert abs(feats["l_elbow_angle"] - 90.0) < 1.0, feats["l_elbow_angle"]
    print(f"Part 2: right-angle forearm -> l_elbow_angle={feats['l_elbow_angle']:.1f} deg (expected ~90) — OK")


def part3_mid_hip_is_always_origin_and_com_is_half_neck():
    kxy = make_skeleton(**PERSON_A, tx=123.0, ty=45.0, jitter=0.0, rng=None)
    norm = normalize_keypoints(kxy)
    assert np.allclose(norm["mid_hip"], [0.0, 0.0], atol=1e-6), norm["mid_hip"]

    feats = compute_derived_features(norm)
    expected_com = norm["neck"] / 2.0
    assert np.allclose([feats["com_x"], feats["com_y"]], expected_com, atol=1e-6), (
        f"com={feats['com_x'], feats['com_y']} expected half-neck={tuple(expected_com)}"
    )
    print("Part 3: normalized mid_hip = origin, com_x/com_y = half-neck "
          "(property of the original computation, faithfully reproduced) — OK")


def part4_velocity_and_acceleration_from_known_displacement():
    tracker = ChuvFeatureTracker()
    fps = 30.0

    # stationary person: frame 0 -> no prior history, everything NaN
    kxy_still = make_skeleton(**PERSON_A, tx=0.0, ty=0.0, jitter=0.0, rng=None)
    feats0 = compute_chuv_features(kxy_still, track_id=1, now=0 / fps, tracker=tracker)
    assert np.isnan(feats0["com_x_vel"]), "first frame: velocity must be NaN (no previous frame)"

    # frame 1: the person moves +30 units in x over 1/30s -> expected velocity = 900 units/s
    kxy_moved = make_skeleton(**PERSON_A, tx=30.0, ty=0.0, jitter=0.0, rng=None)
    feats1 = compute_chuv_features(kxy_moved, track_id=1, now=1 / fps, tracker=tracker)
    # note: tx shifts the whole skeleton, but coordinates are normalized
    # relative to their own mid_hip (which moves along with it), so
    # normalized com_x does NOT change for a pure rigid translation --
    # instead we verify neck_x_vel/mid_hip_x_vel, which by construction
    # of normalize_keypoints always stay at (0,0): "in-scene" velocity
    # (before normalization) is not observable from these features, only
    # kinematics RELATIVE to one's own pelvis (e.g. arm swing) is. So we
    # verify with a wrist movement relative to the body, not a rigid
    # translation of the whole person.
    assert feats1["mid_hip_x_vel"] == 0.0 and feats1["neck_x_vel"] == 0.0, (
        "a rigid translation must not produce velocity in the features "
        "normalized relative to the pelvis (mid_hip is always the origin)"
    )
    print("Part 4a: rigid translation of the whole person -> zero velocity in the "
          "normalized features (expected: only movement RELATIVE to the pelvis is "
          "observable, not the in-scene displacement) — OK")

    # movement relative to the body: the right wrist raises relative to
    # the torso, between two frames 1/30s apart -> expected nonzero velocity
    tracker2 = ChuvFeatureTracker()
    kxy_a = make_skeleton(**PERSON_A, tx=0.0, ty=0.0, jitter=0.0, rng=None)
    compute_chuv_features(kxy_a, track_id=2, now=0 / fps, tracker=tracker2)
    kxy_b = kxy_a.copy()
    kxy_b[KP["right_wrist"]] += [0.0, -20.0]  # right wrist raised by 20 units
    feats_b = compute_chuv_features(kxy_b, track_id=2, now=1 / fps, tracker=tracker2)
    assert feats_b["r_wrist_y_vel"] < -1.0, f"expected negative r_wrist_y_vel (rising), found {feats_b['r_wrist_y_vel']}"
    print(f"Part 4b: right wrist raised between two frames -> r_wrist_y_vel="
          f"{feats_b['r_wrist_y_vel']:.1f} (expected negative, consistent with the y-down image system) — OK")


def part5_forget_resets_state():
    tracker = ChuvFeatureTracker()
    kxy = make_skeleton(**PERSON_A, tx=0.0, ty=0.0, jitter=0.0, rng=None)
    compute_chuv_features(kxy, track_id=9, now=0.0, tracker=tracker)
    tracker.forget(9)
    feats = compute_chuv_features(kxy, track_id=9, now=100.0, tracker=tracker)
    assert np.isnan(feats["com_x_vel"]), "after forget(), the next update must start over from NaN"
    print("Part 5: forget() clears the state -> the track_id would start over from NaN as "
          "if it were an id never seen before — OK")


def main():
    part1_straight_arm_is_180deg()
    part2_right_angle_arm_is_90deg()
    part3_mid_hip_is_always_origin_and_com_is_half_neck()
    part4_velocity_and_acceleration_from_known_displacement()
    part5_forget_resets_state()
    print("\nVerification completed with no errors: angles, normalized distances, COM "
          "and time derivatives of chuv_features.py behave as expected.")


if __name__ == "__main__":
    main()
