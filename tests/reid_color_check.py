"""
reid_color_check.py
====================
Verifies the optional shirt/pants color signal added to `reid.py` (see
"Optional signal: shirt/pants color" in the module docstring), WITHOUT a
camera/YOLO: builds synthetic frames (colored rectangles) instead of a
real video.

Three parts:
  1. Unit verification of `compute_color_signature` (the color sampled
     from the shoulders/hips/knees polygon matches the drawn color).
  2. Unit verification of `color_similarity` (identical -> 1.0, opposite
     hue -> low).
  3. Value verification: a re-entry scenario with DISTORTED body
     proportions (noisy keypoints, as can happen at the edges of the
     frame) that:
       - FAILS with the anthropometric signature alone (frame=None) --
         the concrete reported problem ("doesn't work very well");
       - SUCCEEDS when the frame with the same clothing color is also
         passed (frame=...) -- the distance gets "discounted" enough to
         fall back under threshold;
       - a real clothing change (different color) does NOT prevent an
         otherwise valid match on proportions alone -- color helps, it
         doesn't replace or block the anthropometric signature.

Run with: python reid_color_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from pose.keypoints import KP
from pose.reid import (
    ReIdentifier, COLOR_SEGMENTS, compute_color_signature, color_similarity,
    hair_corners, MAX_POSITION_GAP_SECONDS,
)
from reid_check import make_skeleton, PERSON_A

FPS = 30.0
CONF = np.ones(17)
CANVAS = (400, 700, 3)  # (h, w, 3)


def draw_person_patches(kxy: np.ndarray, shirt_bgr: tuple, pants_bgr: tuple,
                         hair_bgr: tuple | None = None) -> np.ndarray:
    """Synthetic frame (gray background) with shirt/pants/hair colored in
    the same region sampled by `compute_color_signature`."""
    frame = np.full(CANVAS, 128, dtype=np.uint8)
    for region, color in (("shirt", shirt_bgr), ("pants", pants_bgr)):
        corners = COLOR_SEGMENTS[region]
        pts = np.round(kxy[[KP[c] for c in corners]]).astype(np.int32)
        cv2.fillPoly(frame, [pts], color)
    if hair_bgr is not None:
        pts = np.round(hair_corners(kxy)).astype(np.int32)
        cv2.fillPoly(frame, [pts], hair_bgr)
    return frame


def expected_hs(bgr: tuple) -> tuple[float, float]:
    patch = np.uint8([[bgr]])
    h, s, _ = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0]
    return float(h) / 180.0, float(s) / 255.0


RED = (0, 0, 220)      # BGR
BLUE = (200, 30, 0)
GREEN = (0, 180, 0)
BROWN = (19, 69, 139)  # hair color proxy


def part1_signature_matches_drawn_color():
    kxy = make_skeleton(**PERSON_A, tx=300, ty=150, jitter=0.0, rng=None)
    frame = draw_person_patches(kxy, RED, BLUE, hair_bgr=BROWN)
    sig = compute_color_signature(frame, kxy)

    exp_shirt = expected_hs(RED)
    exp_pants = expected_hs(BLUE)
    exp_hair = expected_hs(BROWN)
    assert np.allclose(sig[:2], exp_shirt, atol=0.02), f"shirt {sig[:2]} vs expected {exp_shirt}"
    assert np.allclose(sig[2:4], exp_pants, atol=0.02), f"pants {sig[2:4]} vs expected {exp_pants}"
    assert np.allclose(sig[4:], exp_hair, atol=0.02), f"hair {sig[4:]} vs expected {exp_hair}"
    print(f"Part 1: sampled color shirt={sig[:2]} (expected {exp_shirt}), "
          f"pants={sig[2:4]} (expected {exp_pants}), hair={sig[4:]} (expected {exp_hair}) — OK")


def part2_color_similarity_sane():
    a = np.array([*expected_hs(RED), *expected_hs(BLUE), *expected_hs(BROWN)])
    b = np.array([*expected_hs(RED), *expected_hs(BLUE), *expected_hs(BROWN)])
    same = color_similarity(a, b)
    assert same is not None and same > 0.98, f"identical color should give similarity ~1, found {same}"

    c = np.array([*expected_hs(GREEN), *expected_hs(BLUE), *expected_hs(BROWN)])
    diff = color_similarity(a, c)
    assert diff is not None and diff < same, "red vs green shirt must have lower similarity than red vs red"
    print(f"Part 2: same-color similarity={same:.3f}, red-vs-green shirt={diff:.3f} — OK")


def _run_reentry_scenario(*, reentry_scale: float, reentry_shirt: tuple, reentry_pants: tuple,
                           use_color: bool) -> tuple[int | None, int]:
    """Person present for 20 frames (normal proportions, red/blue
    clothes), exits long enough to zero out the positional signal too
    (beyond `MAX_POSITION_GAP_SECONDS`, to isolate this test to only the
    proportions+color contribution), then re-enters with a new raw
    track_id for 15 frames with the given proportions/clothes. Returns
    (original_person_id, person_id_at_reentry) -- if reid doesn't kick
    in, the second will be a new id (different from the first).
    """
    reid = ReIdentifier(max_lost_seconds=60.0, max_signature_dist=0.12,
                         min_signature_frames=15, color_bonus_weight=0.5)
    rng = np.random.default_rng(1)
    frame_t = 0
    person_id_initial = None

    for _ in range(20):
        now = frame_t / FPS
        kxy = make_skeleton(**PERSON_A, tx=300, ty=150, jitter=0.01, rng=rng)
        frame = draw_person_patches(kxy, RED, BLUE, hair_bgr=BROWN) if use_color else None
        resolved = reid.resolve([(1, kxy, CONF)], now, frame=frame)
        person_id_initial = resolved[0][0]
        frame_t += 1

    absence_frames = int(FPS * (MAX_POSITION_GAP_SECONDS + 5))
    for _ in range(absence_frames):
        # resolve() with an empty list on every frame, not just a
        # frame_t increment: needed because "lost_time" is recorded when
        # the system NOTICES the absence (the next resolve() without
        # that raw_id), not when the absence begins -- without these
        # calls lost_time would coincide with the re-entry, artificially
        # zeroing out the gap we want to simulate.
        reid.resolve([], frame_t / FPS)
        frame_t += 1  # absent from the frame (long enough to zero out the positional bonus)

    distorted = dict(PERSON_A)
    distorted["shoulder_w"] *= reentry_scale
    distorted["hip_w"] /= reentry_scale
    distorted["upper_arm"] *= reentry_scale
    distorted["thigh"] /= reentry_scale
    rng2 = np.random.default_rng(2)
    person_id_reentry = None
    for _ in range(15):
        now = frame_t / FPS
        kxy = make_skeleton(**distorted, tx=300, ty=150, jitter=0.01, rng=rng2)
        frame = draw_person_patches(kxy, reentry_shirt, reentry_pants, hair_bgr=BROWN) if use_color else None
        resolved = reid.resolve([(2, kxy, CONF)], now, frame=frame)
        person_id_reentry = resolved[0][0]
        frame_t += 1

    return person_id_initial, person_id_reentry


def part3_color_recovers_noisy_reentry():
    # (a) noisy keypoints on re-entry, WITHOUT color: must FAIL
    initial, reentry = _run_reentry_scenario(
        reentry_scale=1.6, reentry_shirt=RED, reentry_pants=BLUE, use_color=False)
    assert reentry != initial, (
        "expected reid based only on noisy proportions to fail "
        f"(person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Part 3a (without color, noisy proportions): initial person_id={initial}, "
          f"at reentry={reentry} -> NOT re-associated (expected, demonstrates the problem)")

    # (b) same noisy keypoints, WITH color (same clothes): must SUCCEED
    initial, reentry = _run_reentry_scenario(
        reentry_scale=1.6, reentry_shirt=RED, reentry_pants=BLUE, use_color=True)
    assert reentry == initial, (
        f"expected a match thanks to color (person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Part 3b (with color, same clothes): initial person_id={initial}, "
          f"at reentry={reentry} -> correctly re-associated")

    # (c) CHANGED clothes but correct proportions (not distorted): must
    # still succeed on proportions alone -- different color must NEVER
    # block an otherwise valid match.
    initial, reentry = _run_reentry_scenario(
        reentry_scale=1.0, reentry_shirt=GREEN, reentry_pants=GREEN, use_color=True)
    assert reentry == initial, (
        f"a clothing change must not prevent the match on proportions "
        f"(person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Part 3c (clothes changed, correct proportions): initial person_id={initial}, "
          f"at reentry={reentry} -> still re-associated (color didn't block the match)")


def main():
    part1_signature_matches_drawn_color()
    part2_color_similarity_sane()
    part3_color_recovers_noisy_reentry()
    print("\nVerification completed with no errors: the color signal helps "
          "recover re-entries with noisy proportions when the clothes "
          "stay the same, without compromising invariance to "
          "clothing when the clothes truly change.")


if __name__ == "__main__":
    main()
