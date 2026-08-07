"""
chunk_store_check.py
======================
Verifies `segmentation/chunk_store.py` (save/load round-trip to disk) with
synthetic `SegFrameResult` -- no SAM/GPU dependency.

Usage:
    python tests/chunk_store_check.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.chunk_store import load_chunk, save_chunk  # noqa: E402
from segmentation.seg_estimation import SegFrameResult  # noqa: E402


def _fake_results() -> list[SegFrameResult]:
    return [
        SegFrameResult(frame_index=0, frame=np.zeros((4, 4, 3), dtype=np.uint8), people=[
            (10, np.array([0.0, 0.0, 10.0, 10.0]), np.array([[0, 0], [10, 0], [10, 10]]), 0.9),
            (20, np.array([20.0, 20.0, 30.0, 30.0]), np.array([[20, 20], [30, 20], [30, 30]]), 0.8),
        ]),
        SegFrameResult(frame_index=1, frame=np.zeros((4, 4, 3), dtype=np.uint8), people=[
            (10, np.array([1.0, 1.0, 11.0, 11.0]), np.array([[1, 1], [11, 1], [11, 11]]), 0.85),
        ]),
        SegFrameResult(frame_index=2, frame=np.zeros((4, 4, 3), dtype=np.uint8), people=[]),  # frame with nobody
    ]


def part1_round_trip_preserves_all_fields():
    tmp_dir = tempfile.mkdtemp()
    try:
        results = _fake_results()
        path = save_chunk(results, tmp_dir, chunk_index=3)
        assert os.path.basename(path) == "chunk_0003.npz", path
        assert os.path.exists(path)

        records = load_chunk(path)
        assert len(records) == 3, "2 people in frame 0 + 1 in frame 1 + 0 in frame 2 = 3 rows"

        by_track = {(r.frame_index, r.track_id): r for r in records}
        assert (0, 10) in by_track and (0, 20) in by_track and (1, 10) in by_track

        r = by_track[(0, 10)]
        assert np.allclose(r.bbox, [0.0, 0.0, 10.0, 10.0])
        assert np.allclose(r.polygon, [[0, 0], [10, 0], [10, 10]])
        assert abs(r.conf - 0.9) < 1e-6
        print("PASS part1_round_trip_preserves_all_fields")
    finally:
        shutil.rmtree(tmp_dir)


def part2_empty_chunk_produces_valid_empty_file():
    tmp_dir = tempfile.mkdtemp()
    try:
        empty_results = [SegFrameResult(frame_index=0, frame=np.zeros((2, 2, 3), dtype=np.uint8), people=[])]
        path = save_chunk(empty_results, tmp_dir, chunk_index=0)
        records = load_chunk(path)
        assert records == [], "a chunk with nobody in it must yield an empty list, not an error"
        print("PASS part2_empty_chunk_produces_valid_empty_file")
    finally:
        shutil.rmtree(tmp_dir)


def part3_multiple_chunks_get_distinct_sortable_filenames():
    tmp_dir = tempfile.mkdtemp()
    try:
        for i in (0, 1, 10):
            save_chunk(_fake_results(), tmp_dir, chunk_index=i)
        files = sorted(os.listdir(tmp_dir))
        assert files == ["chunk_0000.npz", "chunk_0001.npz", "chunk_0010.npz"], files
        print("PASS part3_multiple_chunks_get_distinct_sortable_filenames")
    finally:
        shutil.rmtree(tmp_dir)


def main():
    part1_round_trip_preserves_all_fields()
    part2_empty_chunk_produces_valid_empty_file()
    part3_multiple_chunks_get_distinct_sortable_filenames()
    print("\nAll chunk_store.py tests passed.")


if __name__ == "__main__":
    main()
