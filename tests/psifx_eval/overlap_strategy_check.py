"""
overlap_strategy_check.py
===========================
Verifies `psifx_eval/overlap_strategy.py` -- the pure algorithmic core
of the "overlapping chunks" + "overlap + mask IoU" cross-chunk
strategies (items 2/3 of Michele/Loic's brief). Plain numpy, no psifx,
no SAM3, no GPU, no real video -- see that module's docstring. The
headline scenario (part5) is the whole point of this strategy: a person
who vanilla psifx's single-frame comparison would lose (occluded/off-
mask on exactly that one frame) gets correctly re-linked because the
overlap window gives multiple chances.

Run with: python overlap_strategy_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np

from psifx_eval.overlap_strategy import (
    chunk_with_overlap,
    extract_overlap_window,
    local_frames_to_write,
    map_chunk_ids_via_overlap,
    map_first_chunk_ids,
    stash_overlap_tail,
)


def _square(h: int, w: int, box) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def _empty(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


# =============================================================================
# chunk_with_overlap
# =============================================================================

def part1_chunk_with_overlap_windows_and_start_frames_are_correct():
    frames = list(range(23))  # plain ints stand in for frame-like objects
    chunks = list(chunk_with_overlap(frames, chunk_size=10, overlap=3))

    # stride = 7: starts at 0, 7, 14, 21(partial)
    starts = [start for start, _ in chunks]
    assert starts == [0, 7, 14], starts

    first_start, first_chunk = chunks[0]
    assert first_chunk == list(range(0, 10))
    second_start, second_chunk = chunks[1]
    # last 3 of chunk 0 (frames 7,8,9) must be the first 3 of chunk 1
    assert second_chunk[:3] == [7, 8, 9] == first_chunk[-3:]
    assert second_chunk == list(range(7, 17))
    third_start, third_chunk = chunks[2]
    assert third_chunk == list(range(14, 23)), third_chunk  # final, shorter than chunk_size (9 < 10)
    print("Part 1: chunk_with_overlap() produces correctly overlapping windows with the right "
          "start_frame stride, including a shorter final chunk — OK")


def part2_chunk_with_overlap_rejects_bad_overlap_values():
    for bad_overlap in (-1, 10, 15):
        try:
            list(chunk_with_overlap(range(20), chunk_size=10, overlap=bad_overlap))
            raise AssertionError(f"expected ValueError for overlap={bad_overlap}")
        except ValueError:
            pass
    print("Part 2: chunk_with_overlap() rejects overlap outside [0, chunk_size) — OK")


def part3_chunk_with_overlap_zero_overlap_matches_plain_chunking():
    frames = list(range(25))
    chunks = list(chunk_with_overlap(frames, chunk_size=10, overlap=0))
    starts = [start for start, _ in chunks]
    assert starts == [0, 10, 20], starts
    assert [c for _, c in chunks] == [list(range(0, 10)), list(range(10, 20)), list(range(20, 25))]
    print("Part 3: overlap=0 reduces to plain non-overlapping chunking — OK")


# =============================================================================
# map_first_chunk_ids / extract_overlap_window
# =============================================================================

def part4_map_first_chunk_ids_assigns_fresh_ids_in_order_and_respects_cap():
    chunk_outputs = {
        0: {"object_ids": [5, 2], "masks": [None, None]},
        1: {"object_ids": [5, 2, 9], "masks": [None, None, None]},
    }
    id_mapping, next_id = map_first_chunk_ids(chunk_outputs, next_global_id=0, max_num_objects=None)
    assert id_mapping == {5: 0, 2: 1, 9: 2}, id_mapping
    assert next_id == 3

    capped_mapping, capped_next = map_first_chunk_ids(chunk_outputs, next_global_id=0, max_num_objects=2)
    assert capped_mapping == {5: 0, 2: 1}, capped_mapping  # id 9 dropped, over the cap
    assert capped_next == 2
    print("Part 4: map_first_chunk_ids() mints fresh ids in first-appearance order and "
          "respects max_num_objects — OK")


# =============================================================================
# map_chunk_ids_via_overlap -- the core of this whole strategy
# =============================================================================

def part5_recovers_identity_lost_by_a_single_bad_frame_using_the_rest_of_the_overlap():
    # THE scenario this strategy exists for: a person (local id 7 in the
    # new chunk) is badly occluded/clipped on exactly the FIRST shared
    # frame (would be vanilla psifx's ONLY comparison point) but clearly
    # visible and matching on the other 2 shared frames. Overlap=3.
    h, w = 20, 20
    box = (2, 10, 2, 10)
    tiny_sliver = (2, 3, 2, 3)  # near-zero-IoU overlap with `box` -- simulates occlusion/clipping

    prev_overlap_window = {
        0: {100: _square(h, w, box)},
        1: {100: _square(h, w, box)},
        2: {100: _square(h, w, box)},
    }
    chunk_outputs = {
        0: {"object_ids": [7], "masks": [_square(h, w, tiny_sliver)]},  # bad match on frame 0
        1: {"object_ids": [7], "masks": [_square(h, w, box)]},          # good match on frame 1
        2: {"object_ids": [7], "masks": [_square(h, w, box)]},          # good match on frame 2
    }

    id_mapping, next_id = map_chunk_ids_via_overlap(
        chunk_outputs, overlap=3, prev_overlap_window=prev_overlap_window,
        iou_threshold=0.3, next_global_id=200, max_num_objects=None,
    )
    assert id_mapping == {7: 100}, (
        f"expected local id 7 to be correctly linked to global id 100 using the "
        f"2 good frames despite 1 bad one, got {id_mapping}"
    )
    print("Part 5: mean-IoU-over-the-whole-overlap-window recovers an identity that a "
          "single-frame comparison would have lost — OK (this is the fix)")


def part6_single_frame_would_have_failed_this_same_scenario():
    # Sanity check that part5's scenario is a REAL improvement, not a
    # strawman: verify that if only the ONE bad frame were used (as
    # vanilla psifx does), the match would indeed fail below any
    # reasonable threshold.
    h, w = 20, 20
    box = (2, 10, 2, 10)
    tiny_sliver = (2, 3, 2, 3)
    single_frame_prev = {0: {100: _square(h, w, box)}}
    single_frame_curr = {"object_ids": [7], "masks": [_square(h, w, tiny_sliver)]}
    id_mapping, _ = map_chunk_ids_via_overlap(
        {0: single_frame_curr}, overlap=1, prev_overlap_window=single_frame_prev,
        iou_threshold=0.3, next_global_id=200, max_num_objects=None,
    )
    assert id_mapping == {7: 200}, id_mapping  # NOT linked to 100 -- a fresh id, i.e. a lost identity
    print("Part 6: confirmed the single-shared-frame version of this same scenario fails to "
          "link (mints a new id instead) — the overlap window is what fixes it — OK")


def part7_two_people_are_matched_optimally_not_greedily():
    # Two prev people (100 close to box_a, 101 close to box_b) and two
    # curr local ids -- verify the Hungarian assignment picks the
    # GLOBALLY best pairing, not whichever curr id happens to be
    # iterated first (which greedy matching, like vanilla psifx, would
    # be vulnerable to).
    h, w = 30, 30
    box_a = (0, 8, 0, 8)
    box_b = (20, 28, 20, 28)
    prev_overlap_window = {0: {100: _square(h, w, box_a), 101: _square(h, w, box_b)}}
    chunk_outputs = {0: {"object_ids": [9, 8], "masks": [_square(h, w, box_b), _square(h, w, box_a)]}}

    id_mapping, _ = map_chunk_ids_via_overlap(
        chunk_outputs, overlap=1, prev_overlap_window=prev_overlap_window,
        iou_threshold=0.3, next_global_id=200, max_num_objects=None,
    )
    assert id_mapping == {9: 101, 8: 100}, id_mapping
    print("Part 7: two simultaneous people are matched by actual mask overlap (Hungarian), "
          "not by iteration order — OK")


def part8_new_person_not_in_overlap_window_gets_a_fresh_id():
    prev_overlap_window = {0: {100: _square(20, 20, (0, 8, 0, 8))}}
    # local id 55 only appears at local frame 5, past the overlap window (range(0,1) here)
    chunk_outputs = {
        0: {"object_ids": [1], "masks": [_square(20, 20, (0, 8, 0, 8))]},
        5: {"object_ids": [1, 55], "masks": [_square(20, 20, (0, 8, 0, 8)), _square(20, 20, (10, 18, 10, 18))]},
    }
    id_mapping, next_id = map_chunk_ids_via_overlap(
        chunk_outputs, overlap=1, prev_overlap_window=prev_overlap_window,
        iou_threshold=0.3, next_global_id=200, max_num_objects=None,
    )
    assert id_mapping[1] == 100
    assert id_mapping[55] == 200  # genuinely new, correctly minted rather than dropped
    print("Part 8: a person appearing only later in the chunk (outside the overlap window) "
          "still gets a fresh id instead of being silently dropped — OK")


def part9_max_num_objects_cap_is_respected_for_new_ids():
    prev_overlap_window = {0: {100: _square(20, 20, (0, 8, 0, 8))}}
    chunk_outputs = {0: {
        "object_ids": [1, 2],
        "masks": [_square(20, 20, (0, 8, 0, 8)), _square(20, 20, (10, 18, 10, 18))],
    }}
    id_mapping, next_id = map_chunk_ids_via_overlap(
        chunk_outputs, overlap=1, prev_overlap_window=prev_overlap_window,
        iou_threshold=0.3, next_global_id=100, max_num_objects=100,  # already at cap
    )
    assert id_mapping == {1: 100}, id_mapping  # id 2 (new) dropped, id 1 (matched) kept
    print("Part 9: max_num_objects caps only NEW identities, matched ones are never dropped — OK")


# =============================================================================
# stash_overlap_tail / local_frames_to_write
# =============================================================================

def part10_stash_overlap_tail_extracts_and_relabels_to_global_ids():
    chunk_outputs = {
        7: {"object_ids": [1], "masks": [_square(10, 10, (0, 5, 0, 5))]},
        8: {"object_ids": [1], "masks": [_square(10, 10, (0, 5, 0, 5))]},
        9: {"object_ids": [1], "masks": [_square(10, 10, (0, 5, 0, 5))]},
    }
    id_mapping = {1: 42}
    tail = stash_overlap_tail(chunk_outputs, id_mapping, chunk_length=10, overlap=3)
    assert set(tail.keys()) == {0, 1, 2}, tail.keys()  # relabeled to 0-based within the tail
    assert all(list(frame.keys()) == [42] for frame in tail.values())
    print("Part 10: stash_overlap_tail() extracts the chunk's last `overlap` frames and "
          "relabels local ids to their resolved global ids — OK")


def part11_stash_overlap_tail_returns_empty_when_overlap_is_zero():
    assert stash_overlap_tail({0: {"object_ids": [], "masks": []}}, {}, chunk_length=10, overlap=0) == {}
    print("Part 11: stash_overlap_tail() with overlap=0 returns {} (caller should skip "
          "overlap-based linking entirely) — OK")


def part12_local_frames_to_write_skips_the_carried_over_tail():
    assert list(local_frames_to_write(chunk_length=10, skip_local_frames=0)) == list(range(0, 10))
    assert list(local_frames_to_write(chunk_length=10, skip_local_frames=3)) == list(range(3, 10))
    print("Part 12: local_frames_to_write() skips exactly the carried-over overlap frames, "
          "so they're never written to disk twice — OK")


if __name__ == "__main__":
    part1_chunk_with_overlap_windows_and_start_frames_are_correct()
    part2_chunk_with_overlap_rejects_bad_overlap_values()
    part3_chunk_with_overlap_zero_overlap_matches_plain_chunking()
    part4_map_first_chunk_ids_assigns_fresh_ids_in_order_and_respects_cap()
    part5_recovers_identity_lost_by_a_single_bad_frame_using_the_rest_of_the_overlap()
    part6_single_frame_would_have_failed_this_same_scenario()
    part7_two_people_are_matched_optimally_not_greedily()
    part8_new_person_not_in_overlap_window_gets_a_fresh_id()
    part9_max_num_objects_cap_is_respected_for_new_ids()
    part10_stash_overlap_tail_extracts_and_relabels_to_global_ids()
    part11_stash_overlap_tail_returns_empty_when_overlap_is_zero()
    part12_local_frames_to_write_skips_the_carried_over_tail()
    print("\nVerification completed with no errors: overlap_strategy.py's windowing, "
          "multi-frame Hungarian stitching, and write-skip logic all behave as expected.")
