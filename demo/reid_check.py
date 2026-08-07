"""
reid_check.py
=============
Verifies the logic of `reid.py` (real-time re-identification via
anthropometric signature) WITHOUT a camera/YOLO, simulating frame-by-frame
a session with: two people of different build present together, one of
them leaving the frame and re-entering after a time gap with a NEW
track_id (as ByteTrack would do), and a third "stranger" person appearing
during the same period, to verify they aren't confused with either of the
first two.

Run with: python reid_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.keypoints import KP
from pose.reid import (
    ReIdentifier, SIGNATURE_COLS, compute_signature_frame,
    MAX_POSITION_DIST_TORSOS,
)

N_JOINTS = 17  # COCO-17 schema


def make_skeleton(shoulder_w: float, hip_w: float, upper_arm: float, forearm: float,
                   thigh: float, shin: float, torso: float = 100.0,
                   tx: float = 0.0, ty: float = 0.0, jitter: float = 0.0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic COCO-17 skeleton consistent with the given proportions
    (as a fraction of torso length), centered on (tx, ty). Unlike
    `demo/synth_data.py` (which scales ALL joints uniformly, so it always
    produces the same normalized signature), here each segment has an
    independent ratio — necessary to simulate different people.
    """
    j = lambda v: (v + rng.normal(0, jitter * torso)) if (rng is not None and jitter) else v
    kxy = np.zeros((N_JOINTS, 2))

    def set_pt(name, x, y):
        kxy[KP[name]] = [j(x), j(y)]

    # shoulder-center at (tx, ty), hip-center at (tx, ty+torso): same
    # convention as features.torso_length (shoulder_center -> hip_center)
    set_pt("nose", tx, ty - 0.25 * torso)
    set_pt("left_eye", tx - 3, ty - 0.27 * torso)
    set_pt("right_eye", tx + 3, ty - 0.27 * torso)
    set_pt("left_ear", tx - 6, ty - 0.26 * torso)
    set_pt("right_ear", tx + 6, ty - 0.26 * torso)

    set_pt("left_shoulder", tx - shoulder_w * torso / 2, ty)
    set_pt("right_shoulder", tx + shoulder_w * torso / 2, ty)
    set_pt("left_hip", tx - hip_w * torso / 2, ty + torso)
    set_pt("right_hip", tx + hip_w * torso / 2, ty + torso)

    set_pt("left_elbow", tx - shoulder_w * torso / 2, ty + upper_arm * torso)
    set_pt("left_wrist", tx - shoulder_w * torso / 2, ty + (upper_arm + forearm) * torso)
    set_pt("right_elbow", tx + shoulder_w * torso / 2, ty + upper_arm * torso)
    set_pt("right_wrist", tx + shoulder_w * torso / 2, ty + (upper_arm + forearm) * torso)

    set_pt("left_knee", tx - hip_w * torso / 2, ty + torso + thigh * torso)
    set_pt("left_ankle", tx - hip_w * torso / 2, ty + torso + (thigh + shin) * torso)
    set_pt("right_knee", tx + hip_w * torso / 2, ty + torso + thigh * torso)
    set_pt("right_ankle", tx + hip_w * torso / 2, ty + torso + (thigh + shin) * torso)

    return kxy


PERSON_A = dict(shoulder_w=0.50, hip_w=0.40, upper_arm=0.35, forearm=0.30, thigh=0.45, shin=0.42)
PERSON_B = dict(shoulder_w=0.35, hip_w=0.32, upper_arm=0.28, forearm=0.25, thigh=0.38, shin=0.35)
PERSON_C = dict(shoulder_w=0.60, hip_w=0.52, upper_arm=0.42, forearm=0.38, thigh=0.50, shin=0.48)

FPS = 30.0
CONF = np.ones(N_JOINTS)  # full confidence on all joints, not the subject of this test


def head_segments_computed_correctly():
    """Unit check of the two head segments (eye_to_eye/ear_to_ear) added
    to the signature: `make_skeleton` places the eyes at +-3 and the ears
    at +-6 from the center, so 6/torso and 12/torso are expected."""
    kxy = make_skeleton(**PERSON_A, torso=100.0, tx=0, ty=0, jitter=0.0, rng=None)
    sig = compute_signature_frame(kxy)
    eye_val = sig[SIGNATURE_COLS.index("eye_to_eye")]
    ear_val = sig[SIGNATURE_COLS.index("ear_to_ear")]
    assert np.isclose(eye_val, 6.0 / 100.0, atol=1e-6), f"expected eye_to_eye 0.06, found {eye_val}"
    assert np.isclose(ear_val, 12.0 / 100.0, atol=1e-6), f"expected ear_to_ear 0.12, found {ear_val}"
    print(f"Head segments: eye_to_eye={eye_val:.3f}, ear_to_ear={ear_val:.3f} — OK")


def noisy_reentry_recovers_via_retry():
    """Verifies that the match is NOT attempted only once: a re-entry
    with the first frames noisy (person still at the edge of the frame,
    distorted proportions) must not stay "lost forever" -- once the
    sliding window clears up with correct frames, the match must still
    trigger, without needing a new track_id."""
    reid = ReIdentifier(max_lost_seconds=30.0, max_signature_dist=0.12, min_signature_frames=15)
    rng = np.random.default_rng(42)
    frame_t = 0

    for _ in range(50):
        now = frame_t / FPS
        kxy = make_skeleton(**PERSON_A, tx=0, ty=0, jitter=0.01, rng=rng)
        resolved = reid.resolve([(1, kxy, CONF)], now)
        person_id_initial = resolved[0][0]
        frame_t += 1

    for _ in range(100):
        frame_t += 1  # A outside the frame

    distorted = dict(PERSON_A)
    distorted["shoulder_w"] *= 1.45
    distorted["hip_w"] /= 1.45
    distorted["upper_arm"] *= 1.45
    distorted["thigh"] /= 1.45

    matched_at = None
    for i in range(30):
        now = frame_t / FPS
        # first 10 re-entry frames: noisy proportions (person still at
        # the edge of the frame); from frame 11 on: clean.
        params = distorted if i < 10 else PERSON_A
        kxy = make_skeleton(**params, tx=0, ty=0, jitter=0.01, rng=rng)
        resolved = reid.resolve([(5, kxy, CONF)], now)
        if matched_at is None and resolved[0][0] == person_id_initial:
            matched_at = i
        frame_t += 1

    assert matched_at is not None, "the re-entry was never re-associated, even after the frames cleared up"
    assert matched_at >= 10, (
        f"the match triggered at frame {matched_at}, before the noisy data (frames 0-9) "
        "could leave the window -- suggests the threshold is too permissive "
        "to be a valid test, not that the retry really works"
    )
    print(f"Noisy then clean re-entry: re-associated at relative frame {matched_at} "
          "(the initial attempt at 15 frames, with still-noisy data, fails; "
          "the retry on subsequent, cleaner frames recovers the match) — OK")


def _run_in_place_scenario(*, reentry_tx: float, absence_seconds: float) -> tuple[int, int]:
    """Person present for 20 frames at tx=300 (normal proportions),
    absent for `absence_seconds` (simulated by calling resolve([], now)
    on every frame, so lost_time reflects the true moment of
    disappearance, not that of re-entry), then re-enters with distorted
    proportions (too different for a match on signature alone) at
    `reentry_tx`. No color passed: if the re-entry still gets recovered,
    it's thanks to position alone."""
    reid = ReIdentifier(max_lost_seconds=60.0, max_signature_dist=0.12, min_signature_frames=15)
    rng = np.random.default_rng(7)
    frame_t = 0
    person_id_initial = None

    for _ in range(20):
        now = frame_t / FPS
        kxy = make_skeleton(**PERSON_A, tx=300, ty=150, jitter=0.01, rng=rng)
        resolved = reid.resolve([(1, kxy, CONF)], now)
        person_id_initial = resolved[0][0]
        frame_t += 1

    for _ in range(int(FPS * absence_seconds)):
        reid.resolve([], frame_t / FPS)
        frame_t += 1

    distorted = dict(PERSON_A)
    distorted["shoulder_w"] *= 1.6
    distorted["hip_w"] /= 1.6
    distorted["upper_arm"] *= 1.6
    distorted["thigh"] /= 1.6

    rng2 = np.random.default_rng(8)
    person_id_reentry = None
    for _ in range(15):
        now = frame_t / FPS
        kxy = make_skeleton(**distorted, tx=reentry_tx, ty=150, jitter=0.01, rng=rng2)
        resolved = reid.resolve([(2, kxy, CONF)], now)
        person_id_reentry = resolved[0][0]
        frame_t += 1

    return person_id_initial, person_id_reentry


def position_signal_recovers_in_place_reentry():
    """"Jacket change" scenario: the child doesn't leave the frame but
    stays occluded for ~10s (e.g. while being dressed), then reappears in
    THE SAME SPOT with proportions too distorted for a match on
    signature alone (no color passed). It must still be re-associated
    thanks to position alone. Negative control: same distorted
    proportions but re-entry FAR from the last known position -- must
    not trigger, proving that position only helps when it's truly close,
    it doesn't force a match regardless."""
    initial, reentry = _run_in_place_scenario(reentry_tx=300, absence_seconds=10.0)
    assert reentry == initial, (
        f"expected a match thanks to position alone (person didn't move, ~10s of absence) "
        f"(person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"In-place re-entry after occlusion (~10s, e.g. jacket change), without color: "
          f"initial person_id={initial}, at reentry={reentry} -> re-associated thanks to position")

    far_tx = 300 + (MAX_POSITION_DIST_TORSOS + 2) * 100  # well beyond the 4-torso-length radius (torso=100)
    initial, reentry = _run_in_place_scenario(reentry_tx=far_tx, absence_seconds=10.0)
    assert reentry != initial, (
        "negative control failed: a distant re-entry with distorted proportions must not "
        f"be re-associated just because the absence time is short "
        f"(person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Negative control (same gap, but re-entry far from the last known position): "
          f"initial person_id={initial}, at reentry={reentry} -> NOT re-associated (expected, "
          "position never forces a match)")


def max_people_forces_capacity_reentry():
    """1v1 session (max_people=2): A and B present together (roster at
    the cap), A leaves and re-enters with proportions too distorted AND
    far from the last known position -- no normal signal (signature,
    position, color not passed) would lead to a match. With max_people=2
    and B still active (the only "lost" candidate is A), the re-entry
    must still be forced onto A: it can't be a third person by
    definition. See the negative control in
    position_signal_recovers_in_place_reentry for the same scenario
    WITHOUT max_people, where the match doesn't trigger."""
    reid = ReIdentifier(max_lost_seconds=60.0, max_signature_dist=0.12,
                         min_signature_frames=15, max_people=2)
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(12)
    frame_t = 0

    # A and B present together: the roster reaches the cap of 2.
    by_raw = {}
    for _ in range(20):
        now = frame_t / FPS
        people = [
            (1, make_skeleton(**PERSON_A, tx=0, ty=0, jitter=0.01, rng=rng_a), CONF),
            (2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF),
        ]
        resolved = reid.resolve(people, now)
        by_raw = {1: resolved[0][0], 2: resolved[1][0]}
        frame_t += 1
    person_id_a_initial, person_id_b = by_raw[1], by_raw[2]

    # A leaves the frame, B remains alone.
    for _ in range(int(FPS * 5)):
        now = frame_t / FPS
        reid.resolve([(2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF)], now)
        frame_t += 1

    # A re-enters: proportions too distorted for the signature, position
    # too far for the positional signal, no color passed -- no "honest"
    # signal would lead to a match.
    distorted = dict(PERSON_A)
    distorted["shoulder_w"] *= 1.6
    distorted["hip_w"] /= 1.6
    distorted["upper_arm"] *= 1.6
    distorted["thigh"] /= 1.6
    far_tx = (MAX_POSITION_DIST_TORSOS + 2) * 100

    rng_a2 = np.random.default_rng(13)
    person_id_a_reentry = None
    for _ in range(int(FPS * 3)):  # beyond _PENDING_RETRY_SECONDS, so the forced fallback triggers
        now = frame_t / FPS
        people = [
            (2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF),
            (3, make_skeleton(**distorted, tx=far_tx, ty=0, jitter=0.01, rng=rng_a2), CONF),
        ]
        resolved = reid.resolve(people, now)
        by_raw = {raw_id: pid for (raw_id, *_rest), (pid, *_rest2) in zip(people, resolved)}
        person_id_a_reentry = by_raw[3]
        frame_t += 1

    assert person_id_a_reentry == person_id_a_initial, (
        f"with max_people=2 and the roster at the cap, the re-entry should have been forced onto A "
        f"(person_id_a_initial={person_id_a_initial}, person_id_a_reentry={person_id_a_reentry})"
    )
    forced_events = [e for e in reid.merge_log if e.forced]
    assert len(forced_events) == 1, f"expected exactly 1 forced event, found {len(forced_events)}"
    print(f"max_people=2, re-entry with no normal signals available: "
          f"initial person_id={person_id_a_initial}, at reentry={person_id_a_reentry} "
          "-> FORCED re-association (roster at the cap, only lost candidate)")


class _FakeEmbedder:
    """Test stub for the appearance embedding signal (see
    `pose/appearance_embedding.OSNetEmbedder`, deliberately NOT used
    here: doesn't require 'torch'/'torchreid' in the test suite, which
    must stay runnable everywhere). Same interface (`embed(frame, bbox)`)
    -- instead of a real OSNet embedding, reads a color marker fixed in
    the top-left corner of the synthetic frame (`make_marked_frame`) and
    maps it to a unit vector. The marker is fixed in the corner,
    INDEPENDENT of the person's position in the frame (which here
    deliberately varies) -- this demonstrates that the signal is a
    contribution independent of position/proportions, not just an
    indirect way of re-encoding them."""
    _MARKERS = {(255, 0, 0): np.array([1.0, 0.0]), (0, 0, 255): np.array([-1.0, 0.0])}

    def embed(self, frame_bgr, bbox_xyxy, poly=None):
        marker = tuple(int(v) for v in frame_bgr[0, 0])
        return self._MARKERS.get(marker)


def make_marked_frame(marker: tuple[int, int, int], size: int = 2000) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[0:5, 0:5] = marker
    return frame


def embedding_signal_recovers_reentry_when_signature_and_position_fail():
    """Scenario in which NEITHER the signature (proportions too
    distorted) NOR the position (re-entry far from the last known
    position) would be enough alone -- same pattern as the negative
    control in `position_signal_recovers_in_place_reentry`, here however
    with an appearance embedding (`_FakeEmbedder`) that INSTEAD recovers
    the match, because the "color" (the fixed marker in the frame) is
    consistent with the lost person. Negative control: same distortion,
    same distance, but WRONG marker (different person) -- no match must
    trigger, proving that the embedding discounts but never forces
    (same philosophy as color/position, see reid.py's docstring)."""
    RED, BLUE = (255, 0, 0), (0, 0, 255)

    def run(*, reentry_marker: tuple[int, int, int]) -> tuple[int, int]:
        reid = ReIdentifier(max_lost_seconds=60.0, max_signature_dist=0.12,
                             min_signature_frames=15, color_bonus_weight=0.0,
                             position_bonus_weight=0.5, embedder=_FakeEmbedder(),
                             embedding_bonus_weight=0.7)
        rng = np.random.default_rng(21)
        frame_t = 0
        frame_red = make_marked_frame(RED)

        person_id_initial = None
        for _ in range(20):
            now = frame_t / FPS
            kxy = make_skeleton(**PERSON_A, tx=0, ty=0, jitter=0.01, rng=rng)
            resolved = reid.resolve([(1, kxy, CONF)], now, frame=frame_red)
            person_id_initial = resolved[0][0]
            frame_t += 1

        for _ in range(int(FPS * 10)):
            frame_t += 1  # A outside the frame, no resolve() (like noisy_reentry_recovers_via_retry)

        distorted = dict(PERSON_A)
        distorted["shoulder_w"] *= 1.6
        distorted["hip_w"] /= 1.6
        distorted["upper_arm"] *= 1.6
        distorted["thigh"] /= 1.6
        far_tx = 300 + (MAX_POSITION_DIST_TORSOS + 2) * 100

        rng2 = np.random.default_rng(22)
        frame_reentry = make_marked_frame(reentry_marker)
        person_id_reentry = None
        for _ in range(15):
            now = frame_t / FPS
            kxy = make_skeleton(**distorted, tx=far_tx, ty=0, jitter=0.01, rng=rng2)
            resolved = reid.resolve([(2, kxy, CONF)], now, frame=frame_reentry)
            person_id_reentry = resolved[0][0]
            frame_t += 1

        return person_id_initial, person_id_reentry

    initial, reentry = run(reentry_marker=RED)
    assert reentry == initial, (
        f"correct marker (same person): expected a match thanks to the embedding alone "
        f"(signature too distorted, position too far) -- initial={initial}, reentry={reentry}"
    )
    print(f"Re-entry with insufficient signature+position, CORRECT appearance marker: "
          f"initial person_id={initial}, at reentry={reentry} -> re-associated thanks to the embedding alone")

    initial, reentry = run(reentry_marker=BLUE)
    assert reentry != initial, (
        f"negative control failed: WRONG marker (different person) must not produce "
        f"a match just because signature/position are ambiguous -- initial={initial}, reentry={reentry}"
    )
    print(f"Negative control (same distortion, WRONG appearance marker): "
          f"initial person_id={initial}, at reentry={reentry} -> NOT re-associated (expected, "
          "the embedding discounts but never forces)")


def main():
    reid = ReIdentifier(max_lost_seconds=30.0, max_signature_dist=0.12, min_signature_frames=15)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    rng_c = np.random.default_rng(3)

    person_id_a_initial = None
    person_id_b = None
    person_id_a_reentry = None
    person_id_c = None

    frame = 0

    # --- Phase 1: A and B present together, frames 0-49 (raw track 1=A, 2=B) ---
    for i in range(50):
        now = frame / FPS
        people_raw = [
            (1, make_skeleton(**PERSON_A, tx=0, ty=0, jitter=0.01, rng=rng_a), CONF),
            (2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF),
        ]
        resolved = reid.resolve(people_raw, now)
        by_raw = {1: resolved[0][0], 2: resolved[1][0]}
        frame += 1

    person_id_a_initial = by_raw[1]
    person_id_b = by_raw[2]
    print(f"Phase 1: A -> person_id={person_id_a_initial}, B -> person_id={person_id_b}")
    assert person_id_a_initial != person_id_b

    # --- Phase 2: A leaves the frame, B stays alone, frames 50-149 (100 frames = ~3.3s) ---
    for i in range(100):
        now = frame / FPS
        people_raw = [(2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF)]
        resolved = reid.resolve(people_raw, now)
        assert resolved[0][0] == person_id_b, "B must never change identity"
        frame += 1

    print(f"Phase 2: A has disappeared (held in memory as 'lost'), B continues with person_id={person_id_b}")
    assert person_id_a_initial in {lost_id for lost_id in reid.lost}, "A must be among the 'lost' people"

    # --- Phase 3: A re-enters with a NEW raw track_id (3), simulating a
    # clothing change/new SAM/ByteTrack track; C (a stranger) appears in
    # parallel with a different track_id (4) to verify no false matches ---
    for i in range(30):
        now = frame / FPS
        people_raw = [
            (2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF),
            (3, make_skeleton(**PERSON_A, tx=50, ty=0, jitter=0.01, rng=rng_a), CONF),   # A re-enters
            (4, make_skeleton(**PERSON_C, tx=350, ty=0, jitter=0.01, rng=rng_c), CONF),  # stranger
        ]
        resolved = reid.resolve(people_raw, now)
        by_raw = {raw_id: pid for (raw_id, kxy, kconf), (pid, _, _) in
                  zip(people_raw, resolved)}
        frame += 1

    person_id_a_reentry = by_raw[3]
    person_id_c = by_raw[4]
    print(f"Phase 3: A re-entered (raw track 3) -> person_id={person_id_a_reentry}, "
          f"C (stranger, raw track 4) -> person_id={person_id_c}")

    assert person_id_a_reentry == person_id_a_initial, (
        f"A must be re-associated with her original identity "
        f"({person_id_a_initial}), instead has person_id={person_id_a_reentry}"
    )
    assert person_id_c not in {person_id_a_initial, person_id_b}, (
        "C (stranger) must not be confused with A or B"
    )
    assert person_id_b == by_raw[2], "B must remain the same person for the whole session"

    assert len(reid.merge_log) == 1, f"expected exactly 1 merge event, found {len(reid.merge_log)}"
    event = reid.merge_log[0]
    print(f"Merge event recorded: raw_track={event.raw_track_id}, "
          f"provisional={event.provisional_person_id} -> "
          f"restored={event.matched_person_id}, distance={event.distance:.3f}")

    head_segments_computed_correctly()
    noisy_reentry_recovers_via_retry()
    position_signal_recovers_in_place_reentry()
    max_people_forces_capacity_reentry()
    embedding_signal_recovers_reentry_when_signature_and_position_fail()

    print("\nVerification completed with no errors: real-time re-identification "
          "works on a simulated exit/re-entry, without confusing a stranger.")


if __name__ == "__main__":
    main()
