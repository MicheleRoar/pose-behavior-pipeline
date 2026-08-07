"""
seg_reid_check.py
==================
Verifies the logic of `seg_reid.py` (re-identification for the
segmentation-only pipeline, hard cap on max_people) WITHOUT a
camera/YOLO, simulating frame-by-frame synthetic mask polygons
(hand-positioned squares/rectangles, not real silhouettes).

Run with: python seg_reid_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from segmentation.seg_reid import SegReIdentifier

FPS = 15.0


def make_person(cx: float, cy: float, w: float = 60.0, h: float = 140.0,
                 jitter: float = 0.0, rng: np.random.Generator | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic bbox + rectangular polygon centered on (cx, cy), with
    optional jitter on the vertices (to simulate segmentation noise)."""
    j = lambda v: v + (rng.normal(0, jitter) if (rng is not None and jitter) else 0.0)
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    bbox = np.array([x1, y1, x2, y2])
    poly = np.array([[j(x1), j(y1)], [j(x2), j(y1)], [j(x2), j(y2)], [j(x1), j(y2)]])
    return bbox, poly


def hard_cap_never_exceeded_under_heavy_churn():
    """1v1 session (max_people=2): after warm-up, simulates MANY new raw
    track_ids appearing/disappearing in rapid succession (heavy churn, as
    in the real case) near the two known positions. Verifies that the
    number of distinct person_ids NEVER exceeds 2, no matter how many raw
    track_ids are generated."""
    reid = SegReIdentifier(max_people=2)
    rng = np.random.default_rng(1)
    frame_t = 0
    next_raw_id = 1

    # warm-up: A and B, two well-separated positions
    a_bbox, a_poly = make_person(100, 300)
    b_bbox, b_poly = make_person(500, 300)
    resolved = reid.resolve([(1, a_bbox, a_poly, 0.9), (2, b_bbox, b_poly, 0.9)], frame_t / FPS)
    person_ids_seen = {pid for pid, *_ in resolved}
    assert len(person_ids_seen) == 2, f"expected 2 distinct identities after warm-up, found {person_ids_seen}"
    next_raw_id = 3
    frame_t += 1

    # 60 frames of heavy churn: every frame, A and B BOTH reappear with a
    # new raw_id (as if ByteTrack lost and recreated the track on every
    # frame) near their position, with a bit of jitter.
    all_person_ids: set[int] = set(person_ids_seen)
    for _ in range(60):
        now = frame_t / FPS
        a_bbox, a_poly = make_person(100 + rng.normal(0, 5), 300 + rng.normal(0, 5), jitter=2.0, rng=rng)
        b_bbox, b_poly = make_person(500 + rng.normal(0, 5), 300 + rng.normal(0, 5), jitter=2.0, rng=rng)
        people = [(next_raw_id, a_bbox, a_poly, 0.9), (next_raw_id + 1, b_bbox, b_poly, 0.9)]
        resolved = reid.resolve(people, now)
        all_person_ids.update(pid for pid, *_ in resolved)
        next_raw_id += 2
        frame_t += 1

    assert len(all_person_ids) <= 2, (
        f"the max_people=2 cap was exceeded: {len(all_person_ids)} distinct person_ids "
        f"({sorted(all_person_ids)}) despite {next_raw_id - 1} raw track_ids generated"
    )
    print(f"Heavy churn (raw track_ids generated: {next_raw_id - 1}) -> "
          f"distinct person_ids: {sorted(all_person_ids)} (cap respected) — OK")


def churned_track_relinks_to_nearest_position():
    """A exits (its raw track disappears), reappears with a NEW raw_id
    near the same position while B keeps a stable raw_id with no
    interruptions. A's new raw_id must bind to A's original person_id
    (nearby position), not to B's (which in the same frame is already
    claimed by a different raw_id and is far away anyway)."""
    reid = SegReIdentifier(max_people=2)
    frame_t = 0

    a_bbox, a_poly = make_person(100, 300)
    b_bbox, b_poly = make_person(500, 300)
    resolved = reid.resolve([(1, a_bbox, a_poly, 0.9), (2, b_bbox, b_poly, 0.9)], frame_t / FPS)
    by_raw = {1: resolved[0][0], 2: resolved[1][0]}
    person_id_a, person_id_b = by_raw[1], by_raw[2]
    frame_t += 1

    # A disappears for a few frames, B keeps the same raw_id
    for _ in range(5):
        now = frame_t / FPS
        b_bbox, b_poly = make_person(500, 300)
        reid.resolve([(2, b_bbox, b_poly, 0.9)], now)
        frame_t += 1

    # A re-enters with a NEW raw_id (3), near the old position; B
    # continues with the same raw_id 2, in the same frame
    now = frame_t / FPS
    a_bbox, a_poly = make_person(110, 305)  # slightly shifted, like a real re-entry
    b_bbox, b_poly = make_person(500, 300)
    resolved = reid.resolve([(2, b_bbox, b_poly, 0.9), (3, a_bbox, a_poly, 0.9)], now)
    by_raw = {2: resolved[0][0], 3: resolved[1][0]}

    assert by_raw[3] == person_id_a, (
        f"the new raw_id 3 (near A's position) should have bound to person_id={person_id_a}, "
        f"instead it has person_id={by_raw[3]}"
    )
    assert by_raw[2] == person_id_b, "B must never change identity"
    print(f"Re-entry with a new raw_id near the known position: "
          f"A's initial person_id={person_id_a}, after re-entry={by_raw[3]} -> re-associated — OK")


def soft_match_below_cap_relinks_instead_of_minting_new():
    """The bug reported on real footage: with `max_people` deliberately
    set high to leave margin (e.g. 5 for a group with at most 2-3
    children expected), a person who briefly disappears (occlusion,
    exiting the frame edge) and re-enters with a new raw_id MUST
    reconnect to their original identity even if the max_people cap is
    still far from being reached -- before the fix, no comparison was
    attempted below the cap and every disappearance always opened a new
    id (a temporary swap visible even with re-id active)."""
    reid = SegReIdentifier(max_people=5)  # generous cap, real headcount = 2
    frame_t = 0

    a_bbox, a_poly = make_person(100, 300)
    b_bbox, b_poly = make_person(500, 300)
    resolved = reid.resolve([(1, a_bbox, a_poly, 0.9), (2, b_bbox, b_poly, 0.9)], frame_t / FPS)
    by_raw = {1: resolved[0][0], 2: resolved[1][0]}
    person_id_a, person_id_b = by_raw[1], by_raw[2]
    assert len(reid.persons) == 2, "after warm-up there should be only 2 identities, not 5"
    frame_t += 1

    # A disappears for a few frames (B keeps the same raw_id)
    for _ in range(5):
        now = frame_t / FPS
        b_bbox, b_poly = make_person(500, 300)
        reid.resolve([(2, b_bbox, b_poly, 0.9)], now)
        frame_t += 1

    # A re-enters with a NEW raw_id (3), near the old position -- the cap
    # (5) is still far from being reached (only 2 identities exist so far).
    now = frame_t / FPS
    a_bbox, a_poly = make_person(110, 305)
    resolved = reid.resolve([(3, a_bbox, a_poly, 0.9)], now)
    person_id_a_reentry = resolved[0][0]

    assert person_id_a_reentry == person_id_a, (
        f"with cap=5 still far away (2 identities existing), A's re-entry should have reconnected "
        f"to person_id={person_id_a}, instead it opened person_id="
        f"{person_id_a_reentry} (bug: no comparison was attempted below the cap)"
    )
    assert len(reid.persons) == 2, (
        f"no extra identity should have been opened (soft match succeeded): "
        f"found {len(reid.persons)} identities instead of 2"
    )
    print(f"Generous cap (max_people=5, real headcount=2): A's re-entry below the cap "
          f"reconnects to person_id={person_id_a} instead of opening a new one — OK")


def single_person_session_always_id_one():
    """max_people=1: no matter how many raw track_id changes, the
    person_id must always stay the same -- no comparison necessary, by
    definition there's only one possible person."""
    reid = SegReIdentifier(max_people=1)
    frame_t = 0
    seen_person_ids = set()
    for raw_id in range(1, 21):  # 20 different raw track_ids in sequence
        now = frame_t / FPS
        bbox, poly = make_person(200 + raw_id * 3, 300)  # shifts slightly each time
        resolved = reid.resolve([(raw_id, bbox, poly, 0.9)], now)
        seen_person_ids.add(resolved[0][0])
        frame_t += 1

    assert seen_person_ids == {1}, f"with max_people=1 a single person_id is expected, found {seen_person_ids}"
    print(f"Single-person session, 20 different raw track_ids -> person_id always {seen_person_ids} — OK")


def pathological_same_frame_overflow_does_not_crash():
    """Pathological case: max_people=2, but 3 "new" raw_ids appear
    together in one frame (e.g. a spurious double detection). The cap
    must still be respected (never a third person_id) even if this means
    two different raw_ids share the same person_id for that frame."""
    reid = SegReIdentifier(max_people=2)
    now = 0.0
    people = [
        (1, *make_person(100, 300), 0.9),
        (2, *make_person(500, 300), 0.9),
        (3, *make_person(300, 300), 0.9),  # third spurious raw_id, same frame
    ]
    resolved = reid.resolve(people, now)
    person_ids = {pid for pid, *_ in resolved}
    assert len(person_ids) <= 2, f"same-frame overflow not handled correctly: {person_ids}"
    print(f"Same-frame overflow (3 raw_ids, max_people=2) -> "
          f"distinct person_ids: {sorted(person_ids)} (no crash, cap respected) — OK")


class _FakeEmbedder:
    """Test stub for the appearance embedding signal (see
    `pose/appearance_embedding.OSNetEmbedder`, deliberately NOT used
    here: no 'torch'/'torchreid' in the test suite). Same interface
    (`embed(frame, bbox, poly=None)`) -- reads a color marker fixed in
    the top-left corner of the synthetic frame (`make_marked_frame`)
    instead of a real OSNet embedding, and maps it to a unit vector."""
    _MARKERS = {(255, 0, 0): np.array([1.0, 0.0]), (0, 0, 255): np.array([-1.0, 0.0])}

    def embed(self, frame_bgr, bbox_xyxy, poly=None):
        marker = tuple(int(v) for v in frame_bgr[0, 0])
        return self._MARKERS.get(marker)


def make_marked_frame(marker: tuple[int, int, int], size: int = 2000) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[0:5, 0:5] = marker
    return frame


def embedding_signal_relinks_reentry_position_cannot_help():
    """Re-entry DELIBERATELY far from both known positions (A and B): the
    position signal doesn't favor either one (color/shape disabled with
    weight 0 to isolate the embedding). With an appearance marker
    (`_FakeEmbedder`) consistent with A, the re-entry must still
    reconnect to A thanks to the embedding alone. Negative control: same
    far position, WRONG marker -- no match (below `soft_match_threshold`),
    a new identity is minted instead of guessing, proving the embedding
    never forces a match out of nothing."""
    RED, BLUE = (255, 0, 0), (0, 0, 255)

    def run(*, reentry_marker: tuple[int, int, int]) -> tuple[int, int, set[int]]:
        reid = SegReIdentifier(max_people=5, position_weight=0.5, color_weight=0.0,
                                shape_weight=0.0, embedder=_FakeEmbedder(),
                                embedding_weight=0.7, soft_match_threshold=0.6)
        frame_t = 0
        frame_red = make_marked_frame(RED)

        a_bbox, a_poly = make_person(100, 300)
        b_bbox, b_poly = make_person(500, 300)
        resolved = reid.resolve([(1, a_bbox, a_poly, 0.9), (2, b_bbox, b_poly, 0.9)],
                                 frame_t / FPS, frame=frame_red)
        person_id_a, person_id_b = resolved[0][0], resolved[1][0]
        frame_t += 1

        for _ in range(5):
            now = frame_t / FPS
            b_bbox, b_poly = make_person(500, 300)
            reid.resolve([(2, b_bbox, b_poly, 0.9)], now, frame=frame_red)
            frame_t += 1

        now = frame_t / FPS
        far_bbox, far_poly = make_person(1500, 300)  # far from both A (100) and B (500)
        b_bbox, b_poly = make_person(500, 300)
        frame_reentry = make_marked_frame(reentry_marker)
        resolved = reid.resolve([(2, b_bbox, b_poly, 0.9), (3, far_bbox, far_poly, 0.9)],
                                 now, frame=frame_reentry)
        by_raw = {2: resolved[0][0], 3: resolved[1][0]}
        return person_id_a, by_raw[3], set(reid.persons.keys())

    person_id_a, reentry_id, roster = run(reentry_marker=RED)
    assert reentry_id == person_id_a, (
        f"correct marker: expected a match with A thanks to the embedding alone (position "
        f"far from both) -- person_id_a={person_id_a}, reentry={reentry_id}"
    )
    assert len(roster) == 2, f"no extra identity should have been opened, found {len(roster)}"
    print(f"Re-entry far from both A and B, CORRECT appearance marker (A): "
          f"person_id A={person_id_a}, at re-entry={reentry_id} -> re-associated thanks to the embedding alone")

    person_id_a, reentry_id, roster = run(reentry_marker=BLUE)
    assert reentry_id != person_id_a, (
        f"negative control failed: WRONG marker must not produce a match "
        f"(position far from both, no other signal) -- person_id_a={person_id_a}, "
        f"reentry={reentry_id}"
    )
    print(f"Negative control (same far position, WRONG appearance marker): "
          f"person_id A={person_id_a}, at re-entry={reentry_id} -> new identity opened (expected, "
          "the embedding discounts but never forces)")


def main():
    hard_cap_never_exceeded_under_heavy_churn()
    churned_track_relinks_to_nearest_position()
    soft_match_below_cap_relinks_instead_of_minting_new()
    single_person_session_always_id_one()
    pathological_same_frame_overflow_does_not_crash()
    embedding_signal_relinks_reentry_position_cannot_help()
    print("\nVerification completed with no errors: seg_reid.py always respects the "
          "max_people cap, even under heavy churn or same-frame overflow.")


if __name__ == "__main__":
    main()
