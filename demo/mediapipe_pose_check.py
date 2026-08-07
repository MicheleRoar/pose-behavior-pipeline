"""
mediapipe_pose_check.py
=========================
Verifies the pure logic of `pose/mediapipe_pose.py` (BlazePose -> COCO-17
remapping, padded crop box computation) WITHOUT mediapipe/camera
installed: uses synthetic landmark objects (x, y, normalized visibility,
as returned by MediaPipe) instead of real inference.

Run with: python mediapipe_pose_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.keypoints import KP
from pose.mediapipe_pose import BLAZEPOSE_TO_COCO, blazepose_to_coco, padded_crop_box, _empty_pose


class _FakeLandmark:
    """Minimal substitute for a MediaPipe NormalizedLandmark: only the
    fields read by blazepose_to_coco (x, y normalized 0-1, visibility)."""
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def make_landmarks(overrides: dict[int, tuple[float, float, float]]) -> list[_FakeLandmark]:
    """33 BlazePose landmarks, defaulting to the center (0.5, 0.5) with
    visibility 0.0 (so tests can verify that indices NOT overridden stay
    "empty" in the output), with the indices in `overrides` explicitly
    set to (x, y, visibility)."""
    landmarks = [_FakeLandmark(0.5, 0.5, 0.0) for _ in range(33)]
    for idx, (x, y, vis) in overrides.items():
        landmarks[idx] = _FakeLandmark(x, y, vis)
    return landmarks


def part1_all_coco_joints_get_mapped():
    """Every entry of BLAZEPOSE_TO_COCO must translate to the right COCO
    joint, with normalized coordinates correctly reported in pixels of
    the full frame (crop offset + scale)."""
    overrides = {blaze_idx: (0.5, 0.5, 0.9) for blaze_idx in BLAZEPOSE_TO_COCO}
    landmarks = make_landmarks(overrides)
    # 100x200 px crop with top-left corner at (300, 50) in the full
    # frame: the normalized center (0.5, 0.5) must land at
    # (300 + 50, 50 + 100) = (350, 150) in full-frame pixels.
    kxy, kconf = blazepose_to_coco(landmarks, frame_offset_xy=(300.0, 50.0), crop_size_wh=(100.0, 200.0))

    for coco_name in BLAZEPOSE_TO_COCO.values():
        idx = KP[coco_name]
        assert np.allclose(kxy[idx], [350.0, 150.0]), (
            f"{coco_name}: expected (350, 150), found {kxy[idx]}"
        )
        assert abs(kconf[idx] - 0.9) < 1e-6, f"{coco_name}: expected confidence 0.9, found {kconf[idx]}"
    print(f"Part 1: all {len(BLAZEPOSE_TO_COCO)} COCO joints mapped from BlazePose land "
          "at the expected pixel coordinates (crop offset + scale correctly applied) — OK")


def part2_per_joint_confidence_is_preserved_not_averaged():
    """Every COCO joint must report the visibility of ITS specific
    BlazePose landmark, not a shared/averaged value -- an uncertain joint
    (low visibility) must not "contaminate" the confidence of a
    well-detected joint in the same frame."""
    landmarks = make_landmarks({
        0: (0.5, 0.5, 0.95),   # nose: high confidence
        7: (0.5, 0.5, 0.10),   # left_ear: low confidence
    })
    kxy, kconf = blazepose_to_coco(landmarks, frame_offset_xy=(0.0, 0.0), crop_size_wh=(100.0, 100.0))

    assert abs(kconf[KP["nose"]] - 0.95) < 1e-6, f"nose: expected confidence 0.95, found {kconf[KP['nose']]}"
    assert abs(kconf[KP["left_ear"]] - 0.10) < 1e-6, (
        f"left_ear: expected confidence 0.10, found {kconf[KP['left_ear']]}"
    )
    print("Part 2: per-joint confidence reflects the visibility of ITS specific BlazePose landmark "
          "(nose=0.95, left_ear=0.10 stay distinct, not mixed together) — OK")


def part2b_empty_pose_is_all_nan_zero_confidence():
    """_empty_pose() (used when no pose is detected in the crop) must
    have the same shape (17, 2)/(17,) as a real detection, all
    NaN/confidence 0 -- so the rest of the pipeline (which always expects
    an array of that shape) doesn't need to handle a special case."""
    kxy, kconf = _empty_pose()
    assert kxy.shape == (17, 2) and kconf.shape == (17,), f"unexpected shape: {kxy.shape}, {kconf.shape}"
    assert np.isnan(kxy).all(), "_empty_pose()'s kxy must be all NaN"
    assert (kconf == 0.0).all(), "_empty_pose()'s kconf must be all zero"
    print("Part 2b: _empty_pose() has the correct shape (17,2)/(17,), all NaN/confidence 0 — OK")


def part3_padded_crop_box_clamped_to_frame():
    """The crop box must be expanded by the requested padding but NEVER
    beyond the frame's edges (otherwise it would attempt to crop
    nonexistent pixels)."""
    frame_shape = (480, 640)  # (h, w)

    # central box: padding must not touch the edges
    bbox = np.array([200.0, 150.0, 300.0, 350.0])
    x1, y1, x2, y2 = padded_crop_box(bbox, frame_shape, padding=0.1)
    bw, bh = 100.0, 200.0
    assert x1 == int(200 - bw * 0.1) and y1 == int(150 - bh * 0.1)
    assert x2 == int(300 + bw * 0.1) and y2 == int(350 + bh * 0.1)

    # box near the top-left edge: padding would go out of bounds (x<0,
    # y<0) without clamping
    bbox_edge = np.array([5.0, 5.0, 60.0, 80.0])
    x1e, y1e, x2e, y2e = padded_crop_box(bbox_edge, frame_shape, padding=0.5)
    assert x1e == 0 and y1e == 0, f"expected clamping to (0,0), found ({x1e},{y1e})"

    # box near the bottom-right edge: same story for x2/y2
    bbox_edge2 = np.array([600.0, 440.0, 638.0, 478.0])
    x1e2, y1e2, x2e2, y2e2 = padded_crop_box(bbox_edge2, frame_shape, padding=0.5)
    assert x2e2 == 640 and y2e2 == 480, f"expected clamping to (640,480), found ({x2e2},{y2e2})"

    print("Part 3: the padded crop box always stays within the frame's edges — OK")


def _find_pose_model() -> str | None:
    """Path to the pose_landmarker_lite.task model if present in one of
    the known locations -- the project's fixed cache (models/ at the
    root, see common/mediapipe_models.py) or the old convention next to
    src/ (for compatibility with anyone who had already downloaded it by
    hand before the cwd-path bug fix). None if absent everywhere, so
    tests that require mediapipe/the real model skip themselves with a
    note instead of failing the whole suite on a machine without the
    model downloaded."""
    from common.mediapipe_models import MODELS_CACHE_DIR
    candidates = [
        MODELS_CACHE_DIR / "pose_landmarker_lite.task",
        Path(__file__).resolve().parent.parent / "src" / "pose_landmarker_lite.task",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def part4_defensive_clamp_absorbs_duplicate_or_out_of_order_timestamps():
    """`MediaPipeCropPoseEstimator.estimate()` includes a defensive
    clamp: even calling it twice with the SAME timestamp (the bug
    reported by the user: two people in the same frame on the same
    instance) or with a timestamp even LOWER than the previous one, it
    must not raise 'Input timestamp must be monotonically increasing' --
    the actual value is silently forced to increase. This is an
    additional safety net, NOT the main fix: the main fix is to never
    reuse the same instance for different people, see
    MediaPipePoseByTrack (part 5)."""
    import mediapipe  # noqa: F401 -- only for the availability check
    model_path = _find_pose_model()
    if model_path is None:
        print("Part 4: SKIPPED (pose_landmarker_lite.task not found next to src/ in this "
              "environment) — must be verified on the Mac where the model is downloaded.")
        return

    from pose.mediapipe_pose import MediaPipeCropPoseEstimator

    est = MediaPipeCropPoseEstimator(model_path=model_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = np.array([20.0, 20.0, 150.0, 200.0])
    est.estimate(frame, bbox, timestamp_ms=1000)
    est.estimate(frame, bbox, timestamp_ms=1000)  # same timestamp again: must not raise
    est.estimate(frame, bbox, timestamp_ms=500)   # timestamp even LOWER: must not raise
    print("Part 4: the defensive clamp in MediaPipeCropPoseEstimator.estimate() absorbs "
          "duplicate/out-of-order timestamps (the exact scenario of the reported bug) without "
          "raising 'monotonically increasing' — OK")


def part5_pool_per_track_gives_each_person_an_independent_stream():
    """MediaPipePoseByTrack (the main fix): two people in the same frame
    (same timestamp), passed with two different track_ids, each get
    their own independent `MediaPipeCropPoseEstimator` instance (distinct
    Python objects, not just "no crash" -- see the module docstring on
    why sharing one would both crash AND mix the temporal smoothing of
    different people), across multiple consecutive frames, and forget()
    removes only the indicated track."""
    import mediapipe  # noqa: F401
    model_path = _find_pose_model()
    if model_path is None:
        print("Part 5: SKIPPED (pose_landmarker_lite.task not found) — must be verified on the Mac.")
        return

    from pose.mediapipe_pose import MediaPipePoseByTrack

    pool = MediaPipePoseByTrack(model_path=model_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox_a = np.array([10.0, 10.0, 100.0, 150.0])
    bbox_b = np.array([150.0, 10.0, 260.0, 150.0])

    for frame_idx in range(3):  # a few "consecutive" frames for both people
        now_ms = int((frame_idx / 15.0) * 1000)
        kxy_a, kconf_a = pool.estimate(track_id=1, frame_bgr=frame, bbox=bbox_a, timestamp_ms=now_ms)
        kxy_b, kconf_b = pool.estimate(track_id=2, frame_bgr=frame, bbox=bbox_b, timestamp_ms=now_ms)
        assert kxy_a.shape == (17, 2) and kconf_a.shape == (17,)
        assert kxy_b.shape == (17, 2) and kconf_b.shape == (17,)

    assert set(pool._estimators.keys()) == {1, 2}, "expected one instance per track_id seen"
    assert pool._estimators[1] is not pool._estimators[2], (
        "the two people must have DISTINCT instances, not a shared one"
    )
    pool.forget(1)
    assert set(pool._estimators.keys()) == {2}, "forget() must remove only the indicated track's instance"
    print("Part 5: MediaPipePoseByTrack gives each track_id its own independent instance/"
          "'stream' (two distinct objects, not shared) across multiple consecutive frames, "
          "without crashing — OK (fix for the reported bug)")


def main():
    part1_all_coco_joints_get_mapped()
    part2_per_joint_confidence_is_preserved_not_averaged()
    part2b_empty_pose_is_all_nan_zero_confidence()
    part3_padded_crop_box_clamped_to_frame()
    part4_defensive_clamp_absorbs_duplicate_or_out_of_order_timestamps()
    part5_pool_per_track_gives_each_person_an_independent_stream()
    print("\nVerification completed with no errors: the BlazePose -> COCO-17 remapping, the "
          "crop box computation, and (where mediapipe/the model are available) the fix for the "
          "'monotonically increasing' bug in mediapipe_pose.py behave as expected.")


if __name__ == "__main__":
    main()
