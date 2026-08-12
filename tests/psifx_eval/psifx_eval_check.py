"""
psifx_eval_check.py
=====================
Verifies `psifx_eval/mask_io.py` (reading psifx's real MaskDir output
format) and `psifx_eval/id_metrics.py` (the oracle-vs-baseline
correspondence and cross-chunk ID persistence metrics -- see that
module's docstring for the full methodology) with small synthetic data.
No psifx, no SAM3, no GPU required -- these two modules are pure
mask-in/report-out logic, deliberately separated from
`run_baseline_vs_oracle.py` (which DOES need the real psifx package
plus a CUDA GPU plus gated HF access, and can only be verified by hand
on Michele's real machine, same as this project's existing SAM 3.1
integration).

Run with: python psifx_eval_check.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import cv2
import numpy as np

from psifx_eval.id_metrics import (
    build_correspondence,
    compute_metrics,
    mask_iou,
    match_frame_by_iou,
)
from psifx_eval.mask_io import load_mask_dir, read_mask_video


# =============================================================================
# mask_io.py
# =============================================================================

def _write_synthetic_mask_video(path: Path, masks: np.ndarray, fps: float = 10.0) -> None:
    """Writes a (T, H, W) bool array as a white-on-black RGB video,
    mirroring psifx's own convention (`mask.astype(uint8) * 255`,
    repeated across 3 channels) closely enough for read_mask_video()
    round-tripping -- codec choice doesn't matter here, this is just a
    test fixture writer, not the pipeline's real export path (see
    common/video_writer.py for that)."""
    t, h, w = masks.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), f"test fixture writer failed to open for {path}"
    for frame_mask in masks:
        rgb = np.repeat((frame_mask.astype(np.uint8) * 255)[..., None], 3, axis=-1)
        writer.write(rgb)
    writer.release()


def part1_read_mask_video_round_trips_a_synthetic_mask():
    masks = np.zeros((6, 20, 20), dtype=bool)
    masks[:, 5:12, 5:12] = True  # a static "person" square present every frame
    masks[3, :, :] = False       # one frame with nobody visible (occlusion)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "0.mp4"
        _write_synthetic_mask_video(path, masks)
        read_back = read_mask_video(path)

    assert read_back.shape == masks.shape, (read_back.shape, masks.shape)
    # lossy codec (mp4v test fixture) -- allow a tiny mismatch fraction
    # instead of exact equality, which is what read_mask_video's
    # threshold is FOR.
    mismatch_fraction = (read_back != masks).mean()
    assert mismatch_fraction < 0.01, f"mask round-trip mismatch too high: {mismatch_fraction:.4f}"
    print("Part 1: read_mask_video() round-trips a synthetic white-on-black mask video — OK")


def part2_load_mask_dir_reads_every_id_and_skips_unrelated_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        masks_0 = np.zeros((4, 10, 10), dtype=bool)
        masks_0[:, :5, :5] = True
        masks_1 = np.zeros((4, 10, 10), dtype=bool)
        masks_1[:, 5:, 5:] = True
        _write_synthetic_mask_video(tmp_path / "0.mp4", masks_0)
        _write_synthetic_mask_video(tmp_path / "1.mp4", masks_1)
        # a file that should be IGNORED (not a bare "<int>.mp4")
        _write_synthetic_mask_video(tmp_path / "tracking_visualization.mp4", masks_0)

        loaded = load_mask_dir(tmp_path)

    assert set(loaded.keys()) == {0, 1}, loaded.keys()
    assert loaded[0].shape == (4, 10, 10)
    print("Part 2: load_mask_dir() reads every '<id>.mp4' and ignores unrelated files — OK")


def part3_load_mask_dir_raises_on_empty_or_inconsistent_dir():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_mask_dir(tmp)
            raise AssertionError("expected ValueError for an empty MaskDir")
        except ValueError as exc:
            assert "No" in str(exc)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_synthetic_mask_video(tmp_path / "0.mp4", np.zeros((5, 8, 8), dtype=bool))
        _write_synthetic_mask_video(tmp_path / "1.mp4", np.zeros((7, 8, 8), dtype=bool))
        try:
            load_mask_dir(tmp_path)
            raise AssertionError("expected ValueError for inconsistent frame counts")
        except ValueError as exc:
            assert "inconsistent" in str(exc)
    print("Part 3: load_mask_dir() rejects an empty MaskDir and mismatched-length id videos — OK")


# =============================================================================
# id_metrics.py
# =============================================================================

def _square_mask(h: int, w: int, box) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def part4_mask_iou_basic_cases():
    a = _square_mask(10, 10, (0, 5, 0, 5))
    b = _square_mask(10, 10, (0, 5, 0, 5))
    assert mask_iou(a, b) == 1.0
    c = _square_mask(10, 10, (5, 10, 5, 10))
    assert mask_iou(a, c) == 0.0
    empty = np.zeros((10, 10), dtype=bool)
    assert mask_iou(empty, empty) == 0.0  # union == 0 -> defined as 0, not NaN
    print("Part 4: mask_iou() matches on identical masks, zero on disjoint/empty — OK")


def part5_match_frame_by_iou_respects_threshold():
    oracle_frame = {10: _square_mask(20, 20, (0, 8, 0, 8))}
    baseline_frame = {
        99: _square_mask(20, 20, (0, 8, 0, 8)),       # perfect overlap
        100: _square_mask(20, 20, (15, 20, 15, 20)),  # irrelevant, no overlap
    }
    result = match_frame_by_iou(oracle_frame, baseline_frame, iou_threshold=0.5)
    assert result == {10: 99}, result

    # below threshold -> None, not a bad match
    oracle_frame2 = {10: _square_mask(20, 20, (0, 8, 0, 8))}
    baseline_frame2 = {99: _square_mask(20, 20, (6, 14, 6, 14))}  # partial, low IoU
    result2 = match_frame_by_iou(oracle_frame2, baseline_frame2, iou_threshold=0.9)
    assert result2 == {10: None}, result2
    print("Part 5: match_frame_by_iou() picks the best overlap and respects iou_threshold — OK")


def _tracks_to_masks(total_frames: int, h: int, w: int, id_to_box_by_frame: dict[int, dict[int, tuple]]) -> dict[int, np.ndarray]:
    """Builds `{id: (T, H, W) bool}` from `{id: {frame_idx: box_or_None}}`
    -- a compact way to specify test scenarios without hand-writing full
    arrays."""
    out: dict[int, np.ndarray] = {}
    for entity_id, per_frame in id_to_box_by_frame.items():
        arr = np.zeros((total_frames, h, w), dtype=bool)
        for frame_idx, box in per_frame.items():
            if box is not None:
                arr[frame_idx] = _square_mask(h, w, box)
        out[entity_id] = arr
    return out


def part6_perfect_correspondence_has_zero_events_and_full_boundary_accuracy():
    # One person, present the whole video, oracle and baseline agree on
    # every frame AND use the same id -- the "nothing went wrong" case.
    total, h, w, chunk_size = 10, 20, 20, 5
    box = (0, 8, 0, 8)
    oracle = _tracks_to_masks(total, h, w, {0: {t: box for t in range(total)}})
    baseline = _tracks_to_masks(total, h, w, {0: {t: box for t in range(total)}})

    report = compute_metrics(oracle, baseline, chunk_size=chunk_size)
    assert report.fragmentation_count == 0, report.fragmentation_events
    assert report.swap_count == 0, report.swap_events
    assert report.total_boundaries_checked == 1, report.boundary_checks  # frames 0-9, chunk_size=5 -> 1 boundary at frame 5
    assert report.correct_boundary_matches == 1
    assert report.boundary_accuracy == 1.0
    print("Part 6: identical oracle/baseline runs report zero fragmentation/swaps and 100% "
          "boundary accuracy — OK")


def part7_fragmentation_at_a_chunk_boundary_is_detected_and_marked_cross_chunk():
    # Real person (oracle id 0) present the whole video. Baseline correctly
    # tracks them as id 0 for chunk 0 (frames 0-4), but at the chunk 1
    # boundary (frame 5) psifx-style stitching fails and mints a NEW id (5)
    # instead of continuing id 0 -- exactly the failure mode under
    # investigation (Michele/Loic brief: chunking breaks re-identification).
    total, h, w, chunk_size = 10, 20, 20, 5
    box = (0, 8, 0, 8)
    oracle = _tracks_to_masks(total, h, w, {0: {t: box for t in range(total)}})
    baseline_map: dict[int, dict] = {0: {}, 5: {}}
    for t in range(5):
        baseline_map[0][t] = box
    for t in range(5, 10):
        baseline_map[5][t] = box
    baseline = _tracks_to_masks(total, h, w, baseline_map)

    report = compute_metrics(oracle, baseline, chunk_size=chunk_size)
    assert report.fragmentation_count == 1, report.fragmentation_events
    event = report.fragmentation_events[0]
    assert event.from_id == 0 and event.to_id == 5
    assert event.cross_chunk is True, event
    assert report.swap_count == 0, report.swap_events
    assert report.total_boundaries_checked == 1
    assert report.correct_boundary_matches == 0
    assert report.boundary_accuracy == 0.0
    print("Part 7: an id switch AT a chunk boundary is detected as fragmentation, marked "
          "cross_chunk, and correctly fails the boundary check — OK")


def part8_mid_chunk_switch_is_intra_chunk_and_not_a_boundary_failure():
    # Same kind of id switch as part7, but happening in the MIDDLE of a
    # single (large) chunk -- a native SAM3 tracking failure, not
    # something chunk-stitching caused, so it must be classified
    # differently and NOT counted against boundary accuracy (there IS no
    # boundary here at all: chunk_size=20 > total_frames=10).
    total, h, w, chunk_size = 10, 20, 20, 20
    box = (0, 8, 0, 8)
    oracle = _tracks_to_masks(total, h, w, {0: {t: box for t in range(total)}})
    baseline_map: dict[int, dict] = {0: {}, 7: {}}
    for t in range(5):
        baseline_map[0][t] = box
    for t in range(5, 10):
        baseline_map[7][t] = box
    baseline = _tracks_to_masks(total, h, w, baseline_map)

    report = compute_metrics(oracle, baseline, chunk_size=chunk_size)
    assert report.fragmentation_count == 1, report.fragmentation_events
    assert report.fragmentation_events[0].cross_chunk is False
    assert report.total_boundaries_checked == 0  # no chunk boundary exists in a 10-frame, chunk_size=20 run
    assert report.boundary_accuracy is None
    print("Part 8: a mid-chunk id switch is classified intra-chunk (native SAM3 failure), "
          "separate from chunk-boundary accuracy — OK")


def part9_two_real_people_merged_under_one_id_is_a_swap_not_a_fragmentation():
    # Two DIFFERENT real people (oracle ids 0 and 1), each present for
    # half the video, both end up matched to the SAME baseline id (0) --
    # a genuine identity swap/merge (the more serious error: two people
    # silently conflated), which must show up as a swap_event, NOT a
    # fragmentation_event (each real person, individually, only ever
    # wore ONE baseline id -- nothing fragmented from their side).
    total, h, w, chunk_size = 10, 20, 20, 5
    box_a = (0, 8, 0, 8)
    box_b = (10, 18, 10, 18)
    oracle = _tracks_to_masks(total, h, w, {
        0: {t: box_a for t in range(5)},
        1: {t: box_b for t in range(5, 10)},
    })
    baseline = _tracks_to_masks(total, h, w, {
        0: {**{t: box_a for t in range(5)}, **{t: box_b for t in range(5, 10)}},
    })

    report = compute_metrics(oracle, baseline, chunk_size=chunk_size)
    assert report.fragmentation_count == 0, report.fragmentation_events
    assert report.swap_count == 1, report.swap_events
    swap = report.swap_events[0]
    assert swap.entity_id == 0  # the baseline id that got reused
    assert swap.cross_chunk is True
    print("Part 9: two real people merged under one baseline id is detected as a swap "
          "(not fragmentation), correctly attributed to the reused id — OK")


def part10_build_correspondence_rejects_mismatched_video_lengths():
    oracle = {0: np.zeros((10, 8, 8), dtype=bool)}
    baseline = {0: np.zeros((12, 8, 8), dtype=bool)}
    try:
        build_correspondence(oracle, baseline)
        raise AssertionError("expected ValueError for mismatched frame counts")
    except ValueError as exc:
        assert "same" in str(exc) or "SAME" in str(exc)
    print("Part 10: build_correspondence() refuses to compare runs of different lengths — OK")


if __name__ == "__main__":
    part1_read_mask_video_round_trips_a_synthetic_mask()
    part2_load_mask_dir_reads_every_id_and_skips_unrelated_files()
    part3_load_mask_dir_raises_on_empty_or_inconsistent_dir()
    part4_mask_iou_basic_cases()
    part5_match_frame_by_iou_respects_threshold()
    part6_perfect_correspondence_has_zero_events_and_full_boundary_accuracy()
    part7_fragmentation_at_a_chunk_boundary_is_detected_and_marked_cross_chunk()
    part8_mid_chunk_switch_is_intra_chunk_and_not_a_boundary_failure()
    part9_two_real_people_merged_under_one_id_is_a_swap_not_a_fragmentation()
    part10_build_correspondence_rejects_mismatched_video_lengths()
    print("\nVerification completed with no errors: psifx_eval's mask I/O and cross-chunk "
          "ID persistence metrics behave as expected on synthetic data.")
