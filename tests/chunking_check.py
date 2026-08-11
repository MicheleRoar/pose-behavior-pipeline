"""
chunking_check.py
===================
Verifies `segmentation/chunking.py` (overlapping chunk splitting, IoU on
rasterized polygons, greedy id reconciliation, global id allocator) with
synthetic data -- no dependency on SAM/SAM2/GPU, only numpy/cv2 already
required by the rest of the project.

Usage:
    python tests/chunking_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.chunking import (  # noqa: E402
    GlobalIdAllocator, iter_chunk_ranges, polygon_iou, reconcile_ids, reconcile_ids_windowed,
)


def _square(x: int, y: int, size: int = 40) -> np.ndarray:
    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=float)


def part1_chunk_ranges_cover_full_video_with_overlap():
    ranges = list(iter_chunk_ranges(total_frames=1000, chunk_size=300, overlap=50))
    assert ranges[0] == (0, 300), ranges[0]
    assert ranges[1] == (250, 550), ranges[1]
    assert ranges[2] == (500, 800), ranges[2]
    assert ranges[3] == (750, 1000), ranges[3]  # last chunk is shorter
    assert ranges[-1][1] == 1000, "must cover up to the last frame"
    # every chunk (except the first) starts exactly 'overlap' frames before
    # the end of the previous one
    for (prev_start, prev_end), (start, _end) in zip(ranges, ranges[1:]):
        assert start == prev_end - 50, (prev_end, start)
    print("PASS part1_chunk_ranges_cover_full_video_with_overlap")


def part2_chunk_ranges_exact_division_no_dangling_tiny_chunk():
    # exactly 900 frames / chunk 300 overlap 50 -> must stop cleanly
    ranges = list(iter_chunk_ranges(total_frames=900, chunk_size=300, overlap=50))
    assert ranges[-1][1] == 900
    assert all(end - start > 0 for start, end in ranges)
    print("PASS part2_chunk_ranges_exact_division_no_dangling_tiny_chunk")


def part3_chunk_size_must_exceed_overlap():
    try:
        list(iter_chunk_ranges(total_frames=100, chunk_size=50, overlap=50))
        raise AssertionError("expected ValueError with chunk_size <= overlap")
    except ValueError:
        pass
    print("PASS part3_chunk_size_must_exceed_overlap")


def part4_polygon_iou_known_values():
    shape = (200, 200)
    a = _square(10, 10, 40)
    identical = _square(10, 10, 40)
    disjoint = _square(150, 150, 40)
    half_overlap = _square(30, 10, 40)  # partial overlap on x

    assert abs(polygon_iou(a, identical, shape) - 1.0) < 1e-6
    assert polygon_iou(a, disjoint, shape) == 0.0
    iou_partial = polygon_iou(a, half_overlap, shape)
    assert 0.0 < iou_partial < 1.0, iou_partial
    # degenerate polygon (< 3 points) -> 0.0, not a crash
    assert polygon_iou(a, np.empty((0, 2)), shape) == 0.0
    print(f"PASS part4_polygon_iou_known_values (partial iou={iou_partial:.3f})")


def part5_reconcile_ids_matches_by_geometry():
    shape = (200, 200)
    # previous chunk: global ids 10 and 20, at two distinct positions
    prev = {10: _square(10, 10), 20: _square(100, 100)}
    # new chunk: SAM assigned local ids 0 and 1, at the same positions
    # (people stationary in the anchor frame) but enumerated in a different order
    new = {0: _square(100, 100), 1: _square(10, 10)}

    mapping = reconcile_ids(prev, new, shape, iou_threshold=0.3)
    assert mapping == {0: 20, 1: 10}, mapping
    print("PASS part5_reconcile_ids_matches_by_geometry")


def part6_reconcile_ids_unmatched_local_id_gets_no_mapping():
    shape = (200, 200)
    prev = {10: _square(10, 10)}
    # person 1 is the same as before (10,10); person 2 is NEW (entered
    # the frame during this chunk, no matching mask before)
    new = {0: _square(10, 10), 1: _square(150, 150)}

    mapping = reconcile_ids(prev, new, shape, iou_threshold=0.3)
    assert mapping == {0: 10}, mapping
    assert 1 not in mapping, "local id 1 (new person) must have no match"
    print("PASS part6_reconcile_ids_unmatched_local_id_gets_no_mapping")


def part7_reconcile_ids_never_double_assigns():
    # two new ids are both geometrically close to the same old id
    # (ambiguous/rare case): only one of them can inherit it, the other
    # stays without a mapping instead of duplicating the identity.
    shape = (200, 200)
    prev = {10: _square(10, 10)}
    new = {0: _square(12, 12), 1: _square(15, 15)}

    mapping = reconcile_ids(prev, new, shape, iou_threshold=0.3)
    assert len(mapping) == 1, mapping
    assert list(mapping.values()) == [10]
    print("PASS part7_reconcile_ids_never_double_assigns")


def part5b_hungarian_beats_greedy_on_close_by_people():
    # Reproduces the bug reported by Michele on the 'dancing tracks'
    # test set: two already-known people close together (X, Y); two new
    # detections (A, B) where A happens to overlap X's OLD polygon
    # AND Y's old polygon almost equally well (both 0.6), while B
    # overlaps ONLY X well (0.6) and barely touches Y (0.14, below
    # iou_threshold). The single highest-IoU pair in the whole matrix is
    # a 3-way tie at 0.6 between (X,A), (X,B) and (Y,A) -- a GREEDY
    # matcher iterating old ids in order (X before Y) picks (X,A) first
    # (tie broken by insertion order), which then blocks (Y,A) from ever
    # being tried (A already used) and leaves Y with only B at 0.14,
    # below threshold -- Y ends up with NO match at all, even though a
    # strictly better assignment exists (X-B, Y-A: BOTH matched, same
    # total IoU as the greedy pick plus Y's 0.6, i.e. 1.2 vs 0.6).
    # Hungarian (`reconcile_ids`, maximizing TOTAL iou) must find that
    # better assignment instead.
    shape = (200, 200)
    old_x, old_y = 100, 200
    prev = {old_x: _square(20, 0), old_y: _square(40, 0)}
    new = {0: _square(30, 0), 1: _square(10, 0)}  # 0='A', 1='B'

    # sanity-check the crafted geometry actually produces the intended
    # near-tie (fails loudly here, not deep inside the assertion below,
    # if _square's semantics ever change)
    assert abs(polygon_iou(prev[old_x], new[0], shape) - 0.6) < 0.01
    assert abs(polygon_iou(prev[old_x], new[1], shape) - 0.6) < 0.01
    assert abs(polygon_iou(prev[old_y], new[0], shape) - 0.6) < 0.01
    assert polygon_iou(prev[old_y], new[1], shape) < 0.3  # below threshold

    mapping = reconcile_ids(prev, new, shape, iou_threshold=0.3)
    assert mapping == {1: old_x, 0: old_y}, (
        f"expected the GLOBALLY optimal assignment (both X and Y matched), got {mapping} "
        f"-- a greedy matcher would have produced {{0: old_x}} only, orphaning Y"
    )
    print("PASS part5b_hungarian_beats_greedy_on_close_by_people")


def part5c_reconcile_ids_windowed_matches_single_frame_case():
    # Sanity/equivalence check: with exactly one frame on each side,
    # reconcile_ids_windowed must behave exactly like reconcile_ids.
    shape = (200, 200)
    prev = {10: _square(10, 10), 20: _square(100, 100)}
    new = {0: _square(100, 100), 1: _square(10, 10)}

    single = reconcile_ids(prev, new, shape, iou_threshold=0.3)
    windowed = reconcile_ids_windowed({0: prev}, {0: new}, shape, iou_threshold=0.3)
    assert windowed == single == {0: 20, 1: 10}, (single, windowed)
    print("PASS part5c_reconcile_ids_windowed_matches_single_frame_case")


def part5d_reconcile_ids_windowed_recovers_from_degenerate_anchor_frame():
    # The exact scenario reconcile_ids_windowed exists for: at the
    # PRECISE anchor frame, person 10's polygon is garbage (e.g. a
    # contour-finding artifact during a brief occlusion right at the
    # chunk boundary) -- but a couple of frames earlier they were
    # clearly at (10,10), matching the new chunk's single discovery
    # frame well. A plain single-anchor-frame reconcile_ids would miss
    # this entirely; reconcile_ids_windowed must not.
    shape = (200, 200)
    prev_by_frame = {
        -2: {10: _square(10, 10)},
        -1: {10: _square(10, 10)},
        0: {10: _square(150, 150)},  # degenerate/garbage at the exact anchor
    }
    new_by_frame = {0: {5: _square(12, 12)}}  # true continuation, close to (10,10)

    single_frame_only = reconcile_ids(prev_by_frame[0], new_by_frame[0], shape, iou_threshold=0.3)
    assert single_frame_only == {}, "sanity check: the degenerate anchor alone must NOT match"

    windowed = reconcile_ids_windowed(prev_by_frame, new_by_frame, shape, iou_threshold=0.3)
    assert windowed == {5: 10}, (
        f"expected the trailing-history frames to recover the match, got {windowed}"
    )
    print("PASS part5d_reconcile_ids_windowed_recovers_from_degenerate_anchor_frame")


def part8_global_id_allocator_never_repeats():
    allocator = GlobalIdAllocator()
    ids = [allocator.next_id() for _ in range(5)]
    assert ids == [1, 2, 3, 4, 5], ids
    print("PASS part8_global_id_allocator_never_repeats")


def main():
    part1_chunk_ranges_cover_full_video_with_overlap()
    part2_chunk_ranges_exact_division_no_dangling_tiny_chunk()
    part3_chunk_size_must_exceed_overlap()
    part4_polygon_iou_known_values()
    part5_reconcile_ids_matches_by_geometry()
    part5b_hungarian_beats_greedy_on_close_by_people()
    part5c_reconcile_ids_windowed_matches_single_frame_case()
    part5d_reconcile_ids_windowed_recovers_from_degenerate_anchor_frame()
    part6_reconcile_ids_unmatched_local_id_gets_no_mapping()
    part7_reconcile_ids_never_double_assigns()
    part8_global_id_allocator_never_repeats()
    print("\nAll chunking.py tests passed.")


if __name__ == "__main__":
    main()
