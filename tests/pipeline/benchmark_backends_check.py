"""
benchmark_backends_check.py
==============================
Verifies `benchmark_backends.py`: the aggregation part (track duration,
number of raw ids, "short" percentage, processing fps) with a FAKE
tracker injected in place of `build_tracker()` (no dependency on real
YOLO/SAM/SAM2, same philosophy as the other tests/*_check.py), and the
skip logic for the sam31/sam2 methods when the detected device is not
"cuda".

Usage:
    python tests/benchmark_backends_check.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import benchmark_backends as bb  # noqa: E402
from segmentation.seg_estimation import SegFrameResult  # noqa: E402


class _FakeTracker:
    """Fake tracker: returns a hand-fixed sequence of `SegFrameResult`, to
    verify that `run_one_method()` aggregates the right metrics from a
    known input (not a real video/model)."""

    def __init__(self, results):
        self._results = results

    def run(self, source, stream: bool = True):
        yield from self._results


def _fake_results():
    # id 1 present in all 3 frames, id 2 only in the middle frame (a
    # "shorter" track) -- what matters is the RELATIVE comparison between
    # the two durations, not the absolute value (both intentionally below
    # SHORT_LIVED_THRESHOLD_FRAMES=15: a 3-frame test video shouldn't
    # claim to have "long" tracks in an absolute sense).
    box = np.array([0.0, 0.0, 10.0, 10.0])
    poly = np.array([[0, 0], [10, 0], [10, 10]])
    return [
        SegFrameResult(frame_index=0, frame=np.zeros((2, 2, 3), dtype=np.uint8),
                        people=[(1, box, poly, 0.9)]),
        SegFrameResult(frame_index=1, frame=np.zeros((2, 2, 3), dtype=np.uint8),
                        people=[(1, box, poly, 0.9), (2, box, poly, 0.8)]),
        SegFrameResult(frame_index=2, frame=np.zeros((2, 2, 3), dtype=np.uint8),
                        people=[(1, box, poly, 0.9)]),
    ]


def part1_run_one_method_aggregates_lifespans_correctly():
    original_build_tracker = bb.build_tracker
    bb.build_tracker = lambda *a, **k: _FakeTracker(_fake_results())
    try:
        result = bb.run_one_method("yolo", source="unused.mp4", fps=15.0, device="cpu")
    finally:
        bb.build_tracker = original_build_tracker

    assert result is not None
    assert result["method"] == "yolo"
    assert result["n_frames"] == 3
    assert result["n_raw_ids"] == 2
    assert result["lifespan_min_frames"] == 1, "id 2 is present in only one frame"
    assert result["lifespan_max_frames"] == 3, "id 1 is present in all 3 frames"
    assert result["short_lived_ids_pct"] == 100.0, "both ids are below the threshold (15 frames)"
    assert result["lifespan_median_s"] == round(result["lifespan_median_frames"] / 15.0, 2), \
        "the conversion to seconds must use the passed fps"
    print("PASS part1_run_one_method_aggregates_lifespans_correctly")


def part2_sam_methods_skipped_without_cuda():
    result = bb.run_one_method("sam31", source="unused.mp4", fps=15.0, device="mps")
    assert result is None, "sam31 on device!='cuda' must be skipped (None), without building anything"
    print("PASS part2_sam_methods_skipped_without_cuda")


def part3_run_benchmark_skips_gracefully_and_keeps_valid_methods():
    original_build_tracker = bb.build_tracker
    bb.build_tracker = lambda *a, **k: _FakeTracker(_fake_results())
    try:
        df = bb.run_benchmark(["yolo", "sam31"], source="unused.mp4", fps=15.0, device="mps")
    finally:
        bb.build_tracker = original_build_tracker

    assert len(df) == 1, "sam31 must be skipped (mps device), only 'yolo' (fake) should remain"
    assert df.iloc[0]["method"] == "yolo"
    print("PASS part3_run_benchmark_skips_gracefully_and_keeps_valid_methods")


def part4_unknown_method_raises():
    try:
        bb.run_benchmark(["made-up-method"], source="unused.mp4", fps=15.0, device="cpu")
        raise AssertionError("expected ValueError for an unknown method")
    except ValueError:
        pass
    print("PASS part4_unknown_method_raises")


def part5_sweep_produces_cartesian_product_for_sam_only():
    # sam_chunk_size/overlap/redetect_every reported in the CSV: yolo
    # ignores them (a single run, None in the columns), sam31 runs the
    # cartesian product of all given combinations.
    original_build_tracker = bb.build_tracker
    bb.build_tracker = lambda *a, **k: _FakeTracker(_fake_results())
    try:
        df = bb.run_benchmark(
            ["yolo", "sam31"], source="unused.mp4", fps=15.0, device="cuda",
            sam_chunk_sizes=[300, 600], sam_overlaps=[30, 50], sam_redetect_everys=[100, None],
        )
    finally:
        bb.build_tracker = original_build_tracker

    yolo_rows = df[df["method"] == "yolo"]
    sam_rows = df[df["method"] == "sam31"]
    assert len(yolo_rows) == 1, \
        "yolo ignores the three sam_* parameters -- a single run, not repeated for every combination"
    # pandas converts None -> NaN in a numeric column mixed with the
    # integers from the sam31 rows -- pd.isna(), not "is None", to verify it.
    assert pd.isna(yolo_rows.iloc[0]["sam_chunk_size"]), \
        "yolo doesn't use sam_chunk_size -- NaN/None in the CSV, not a misleading value"
    assert yolo_rows.iloc[0]["run_label"] == "yolo", \
        "with a single run (yolo ignores the sweep), run_label == method, no suffix"
    assert len(sam_rows) == 8, "2 chunk_size x 2 overlap x 2 redetect_every = 8 combinations for sam31"
    assert set(sam_rows["sam_chunk_size"]) == {300, 600}
    assert set(sam_rows["sam_overlap"]) == {30, 50}
    # sam_redetect_every: pandas turns the integer column (including
    # yolo's None) into float64 -- None becomes NaN, not comparable with
    # "==" nor present in a set in the usual way (nan != nan).
    redetect_values = sam_rows["sam_redetect_every"].tolist()
    redetect_non_null = {v for v in redetect_values if pd.notna(v)}
    assert redetect_non_null == {100}, f"expected 100 among the non-null values, found {redetect_non_null}"
    assert any(pd.isna(v) for v in redetect_values), \
        "the redetect_every=None (disabled) case should also be present in the sweep"
    assert sam_rows["run_label"].str.contains(r"sam31\[cs=").all(), \
        "with more than one combination, run_label must distinguish them"
    print("PASS part5_sweep_produces_cartesian_product_for_sam_only")


def part6_sweep_skips_invalid_chunk_size_overlap_combo():
    original_build_tracker = bb.build_tracker
    bb.build_tracker = lambda *a, **k: _FakeTracker(_fake_results())
    try:
        df = bb.run_benchmark(
            ["sam31"], source="unused.mp4", fps=15.0, device="cuda",
            sam_chunk_sizes=[50, 300], sam_overlaps=[100], sam_redetect_everys=[None],
        )
    finally:
        bb.build_tracker = original_build_tracker
    # chunk_size=50 <= overlap=100 -- invalid combination, skipped; only chunk_size=300 remains
    assert len(df) == 1, "the invalid combination (50<=100) must be skipped, must not crash"
    assert df.iloc[0]["sam_chunk_size"] == 300
    print("PASS part6_sweep_skips_invalid_chunk_size_overlap_combo")


def part7_parse_int_list_handles_commas_and_none():
    assert bb._parse_int_list("600") == [600]
    assert bb._parse_int_list("300,600") == [300, 600]
    assert bb._parse_int_list("300, 600 ") == [300, 600], "spaces around the comma are tolerated"
    assert bb._parse_int_list("", allow_none=True) == [None], \
        "empty string with allow_none -- default 'disabled', a single run"
    assert bb._parse_int_list("100,", allow_none=True) == [100, None], \
        "empty element between commas -- also includes the 'disabled' case in the sweep"
    print("PASS part7_parse_int_list_handles_commas_and_none")


def main():
    part1_run_one_method_aggregates_lifespans_correctly()
    part2_sam_methods_skipped_without_cuda()
    part3_run_benchmark_skips_gracefully_and_keeps_valid_methods()
    part4_unknown_method_raises()
    part5_sweep_produces_cartesian_product_for_sam_only()
    part6_sweep_skips_invalid_chunk_size_overlap_combo()
    part7_parse_int_list_handles_commas_and_none()
    print("\nAll benchmark_backends.py tests passed.")


if __name__ == "__main__":
    main()
