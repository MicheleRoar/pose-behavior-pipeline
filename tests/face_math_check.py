"""
face_math_check.py
===================
Verifies the mouth/eye logic in `gaze_head.py` (mouth aspect ratio, eye
aspect ratio, blink counting) WITHOUT MediaPipe or a camera, using
synthetic face landmarks in "open" and "closed" configurations.

Run with: python face_math_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.gaze_head import (
    mouth_aspect_ratio, eye_aspect_ratio, mean_eye_aspect_ratio,
    mean_eyebrow_raise,
    MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT,
    RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX,
    RIGHT_EYEBROW_IDX, LEFT_EYEBROW_IDX,
)

N_LANDMARKS = 478  # FaceLandmarker schema (468 + 10 iris)


def make_face(mouth_open: bool, eyes_open: bool, eyebrow: str = "neutral") -> np.ndarray:
    face = np.full((N_LANDMARKS, 2), 500.0)  # neutral filler, unused by the tests

    # --- mouth ---
    face[MOUTH_LEFT] = [480, 601]
    face[MOUTH_RIGHT] = [520, 601]
    if mouth_open:
        face[MOUTH_TOP] = [500, 590]
        face[MOUTH_BOTTOM] = [500, 620]
    else:
        face[MOUTH_TOP] = [500, 600]
        face[MOUTH_BOTTOM] = [500, 602]

    # --- right eye (RIGHT_EYE_EAR_IDX schema: [left_corner, top1, top2, right_corner, bottom2, bottom1]) ---
    r1, r2, r3, r4, r5, r6 = RIGHT_EYE_EAR_IDX
    face[r1] = [400, 400]
    face[r4] = [430, 400]
    if eyes_open:
        face[r2] = [410, 390]; face[r6] = [410, 410]
        face[r3] = [420, 390]; face[r5] = [420, 410]
    else:
        face[r2] = [410, 400]; face[r6] = [410, 401]
        face[r3] = [420, 400]; face[r5] = [420, 401]

    # --- left eye (same logic, LEFT_EYE_EAR_IDX indices) ---
    l1, l2, l3, l4, l5, l6 = LEFT_EYE_EAR_IDX
    face[l1] = [600, 400]
    face[l4] = [630, 400]
    if eyes_open:
        face[l2] = [610, 390]; face[l6] = [610, 410]
        face[l3] = [620, 390]; face[l5] = [620, 410]
    else:
        face[l2] = [610, 400]; face[l6] = [610, 401]
        face[l3] = [620, 400]; face[l5] = [620, 401]

    # --- eyebrows (only used by the eyebrow_raise tests, require eyes_open
    # for a stable upper-eyelid reference at y=390) ---
    brow_y = {"raised": 365.0, "neutral": 380.0, "lowered": 392.0}[eyebrow]
    for i, idx in enumerate(RIGHT_EYEBROW_IDX):
        face[idx] = [400 + i * 8, brow_y]
    for i, idx in enumerate(LEFT_EYEBROW_IDX):
        face[idx] = [600 + i * 8, brow_y]

    return face


def count_blinks(ear_sequence: np.ndarray, threshold: float) -> int:
    """Same logic used in live_demo.py: counts open->closed transitions
    (a proxy for the number of blinks in the window)."""
    closed = ear_sequence < threshold
    return int(np.sum(np.diff(closed.astype(int)) == 1))


def main():
    print("--- Mouth aspect ratio ---")
    mar_open = mouth_aspect_ratio(make_face(mouth_open=True, eyes_open=True))
    mar_closed = mouth_aspect_ratio(make_face(mouth_open=False, eyes_open=True))
    print(f"  open mouth:   MAR={mar_open:.3f}")
    print(f"  closed mouth: MAR={mar_closed:.3f}")
    assert mar_open > mar_closed, "Expected a higher MAR with an open mouth"

    print("\n--- Eye aspect ratio ---")
    face_open = make_face(mouth_open=False, eyes_open=True)
    face_closed = make_face(mouth_open=False, eyes_open=False)
    ear_open_r = eye_aspect_ratio(face_open, RIGHT_EYE_EAR_IDX)
    ear_closed_r = eye_aspect_ratio(face_closed, RIGHT_EYE_EAR_IDX)
    ear_open_mean = mean_eye_aspect_ratio(face_open)
    ear_closed_mean = mean_eye_aspect_ratio(face_closed)
    print(f"  right eye open:   EAR={ear_open_r:.3f}")
    print(f"  right eye closed: EAR={ear_closed_r:.3f}")
    print(f"  mean of both eyes, open: {ear_open_mean:.3f}  closed: {ear_closed_mean:.3f}")
    assert ear_open_r > ear_closed_r, "Expected a higher EAR with an open eye"
    assert ear_open_mean > ear_closed_mean

    print("\n--- Blink count (threshold 0.2) ---")
    # synthetic sequence: open (0.6) for a while, then 3 blinks (drops below
    # threshold for one frame), then open again
    ear_seq = np.array([0.6, 0.6, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6, 0.6, 0.6, 0.1, 0.6, 0.6])
    n = count_blinks(ear_seq, threshold=0.2)
    print(f"  blinks detected: {n} (expected 3)")
    assert n == 3, f"Expected 3 blinks, detected {n}"

    print("\n--- Eyebrow raise ---")
    face_raised = make_face(mouth_open=False, eyes_open=True, eyebrow="raised")
    face_neutral = make_face(mouth_open=False, eyes_open=True, eyebrow="neutral")
    face_lowered = make_face(mouth_open=False, eyes_open=True, eyebrow="lowered")
    l_raised, r_raised = mean_eyebrow_raise(face_raised)
    l_neutral, r_neutral = mean_eyebrow_raise(face_neutral)
    l_lowered, r_lowered = mean_eyebrow_raise(face_lowered)
    print(f"  raised:  left={l_raised:.3f} right={r_raised:.3f}")
    print(f"  neutral: left={l_neutral:.3f} right={r_neutral:.3f}")
    print(f"  lowered: left={l_lowered:.3f} right={r_lowered:.3f}")
    assert l_raised > l_neutral > l_lowered, "Expected order raised > neutral > lowered (left)"
    assert r_raised > r_neutral > r_lowered, "Expected order raised > neutral > lowered (right)"

    print("\nMath verification completed with no errors.")


if __name__ == "__main__":
    main()
