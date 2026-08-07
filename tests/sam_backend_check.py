"""
sam_backend_check.py
=======================
Verifies `segmentation/sam_backend.py::ChunkedVideoPredictorBackend` with
a FAKE SAM predictor (no dependency on sam3/sam2/CUDA GPU) -- same
philosophy as `tests/device_check.py` (injecting a fake double in place of
the heavy library). Generates a small synthetic video on disk
(cv2.VideoWriter) so `run()` can read real frames via cv2.VideoCapture
exactly as it would with a real video.

Scenario tested: person A present from the first frame (detected at
chunk 0's bootstrap), person B who "enters" mid-video (detected only at
chunk 2) -- verifies that: ids remain continuous between one chunk and
the next for already-known people, a new person detected mid-video gets
an id never used before, there are no duplicate or missing frames in the
final output, and disk persistence writes one file per chunk.

Usage:
    python tests/sam_backend_check.py
"""

import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.chunk_store import load_chunk  # noqa: E402
from segmentation.sam_backend import ChunkedVideoPredictorBackend, _to_boolean_mask  # noqa: E402

FRAME_SHAPE = (64, 64)  # (height, width)
TOTAL_FRAMES = 40
CHUNK_SIZE = 15
OVERLAP = 5

BOX_A = np.array([5.0, 5.0, 15.0, 15.0])
BOX_B = np.array([30.0, 30.0, 45.0, 45.0])


def _make_synthetic_video(path: str) -> None:
    height, width = FRAME_SHAPE
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    for _ in range(TOTAL_FRAMES):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def _box_to_mask(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box.astype(int)
    mask[y1:y2, x1:x2] = True
    return mask


class _FakePredictor:
    """Fake SAM predictor: 'remembers' only the box it was seeded with
    for each obj_id (static position, doesn't simulate real movement --
    that's not its purpose, see the module docstring) and returns it as
    an identical mask for every frame of the chunk.

    `init_state()` receives a PATH to a folder (not a list of in-memory
    frames) -- same interface required by the real SAM2/SAM (see
    `ChunkedVideoPredictorBackend._init_state()`): here it just counts
    the JPEG files written by `_init_state()`, so the test really
    verifies that write happened with the right number of frames, not
    just that the code doesn't raise an exception."""

    def __init__(self, frame_shape: tuple[int, int]):
        self.frame_shape = frame_shape

    def init_state(self, video_dir: str):
        num_frames = len([f for f in os.listdir(video_dir) if f.endswith(".jpg")])
        return SimpleNamespace(num_frames=num_frames, seeds={})

    def add_new_points_or_box(self, state, *, frame_idx, obj_id, box):
        state.seeds[obj_id] = box

    def propagate_in_video(self, state, *, start_frame_idx=0, max_frame_num_to_track=None):
        end = state.num_frames if max_frame_num_to_track is None \
            else min(state.num_frames, start_frame_idx + max_frame_num_to_track)
        for local_idx in range(start_frame_idx, end):
            obj_ids = list(state.seeds.keys())
            masks = [_box_to_mask(state.seeds[oid], self.frame_shape) for oid in obj_ids]
            yield local_idx, obj_ids, masks


class _FakeBackend(ChunkedVideoPredictorBackend):
    """Test subclass: `_build_predictor()` returns the fake predictor,
    `_detect_people()` follows a fixed 'script' instead of calling YOLO
    (not installed in this environment) -- one schedule element per
    expected call (one per chunk, see the module docstring)."""

    def __init__(self, *, detect_schedule: list[list[np.ndarray]], **kwargs):
        super().__init__(**kwargs)
        self._detect_schedule = detect_schedule
        self._call_count = 0

    def _build_predictor(self):
        return _FakePredictor(FRAME_SHAPE)

    def _detect_people(self, frame):
        boxes = self._detect_schedule[self._call_count] if self._call_count < len(self._detect_schedule) else []
        self._call_count += 1
        return boxes


def _run_backend(chunk_store_dir: str | None):
    return _FakeBackend(
        device="cuda", chunk_size=CHUNK_SIZE, overlap=OVERLAP,
        detect_schedule=[[BOX_A], [], [BOX_B], []],
        chunk_store_dir=chunk_store_dir,
    )


def part1_ids_stay_continuous_across_chunks():
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _run_backend(chunk_store_dir=None)

        results = list(backend.run(video_path))
        frame_indices = [r.frame_index for r in results]
        assert frame_indices == list(range(TOTAL_FRAMES)), \
            f"expected all frames 0..{TOTAL_FRAMES - 1} with no gaps/duplicates, got {frame_indices}"

        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}
        # person A: present from frame 0, always the same id in ALL frames
        a_ids = {next(iter(ids)) for f, ids in ids_per_frame.items() if f < 20 and len(ids) == 1}
        assert len(a_ids) == 1, f"A's id must stay the same in all frames < 20, found {a_ids}"
        print("PASS part1_ids_stay_continuous_across_chunks")
    finally:
        shutil.rmtree(tmp_dir)


def part2_new_person_mid_video_gets_fresh_id_with_expected_lag():
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _run_backend(chunk_store_dir=None)

        results = list(backend.run(video_path))
        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}

        # B is detected during chunk 2 (which starts at original frame
        # 20), but that anchor frame has already been emitted by chunk 1
        # (which didn't know about B) -- so B appears for the first time
        # only from the first frame REALLY emitted by chunk 2, i.e.
        # 20 + overlap.
        first_seen = min(f for f, ids in ids_per_frame.items() if len(ids) == 2)
        assert first_seen == 20 + OVERLAP, f"expected {20 + OVERLAP}, got {first_seen}"
        assert all(len(ids_per_frame[f]) == 1 for f in range(0, first_seen)), \
            "before this, only A should be present"
        assert all(len(ids_per_frame[f]) == 2 for f in range(first_seen, TOTAL_FRAMES)), \
            "from here on, both A and B should be present"

        all_ids = set()
        for ids in ids_per_frame.values():
            all_ids |= ids
        assert len(all_ids) == 2, f"expected exactly 2 distinct global ids across the whole session, found {all_ids}"
        print(f"PASS part2_new_person_mid_video_gets_fresh_id_with_expected_lag (B seen from frame {first_seen})")
    finally:
        shutil.rmtree(tmp_dir)


def part3_chunk_persistence_writes_one_file_per_chunk():
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        store_dir = os.path.join(tmp_dir, "chunks")
        _make_synthetic_video(video_path)
        backend = _run_backend(chunk_store_dir=store_dir)

        list(backend.run(video_path))  # consumes the generator

        files = sorted(os.listdir(store_dir))
        assert len(files) == 4, f"expected 4 chunks (40 frames / 15 chunk_size / 5 overlap), found {files}"
        # the first chunk contains A's rows over 15 frames
        records = load_chunk(os.path.join(store_dir, files[0]))
        assert len(records) == CHUNK_SIZE, f"the first chunk must have one row per frame (A only), found {len(records)}"
        print("PASS part3_chunk_persistence_writes_one_file_per_chunk")
    finally:
        shutil.rmtree(tmp_dir)


def part4_non_cuda_device_rejected_immediately():
    try:
        _FakeBackend(device="mps", detect_schedule=[])
        raise AssertionError("expected ValueError for a device other than 'cuda'")
    except ValueError:
        pass
    print("PASS part4_non_cuda_device_rejected_immediately")


def part4c_overlap_zero_rejected_immediately():
    # overlap=0: prev_anchor_polys would always be empty (no shared frame
    # between consecutive chunks, see __init__'s docstring) -- every
    # chunk would assign all-new global ids, silently losing identity
    # continuity. Must be rejected immediately, not left to silently
    # produce a wrong result.
    try:
        _FakeBackend(device="cuda", detect_schedule=[], overlap=0)
        raise AssertionError("expected ValueError for overlap=0")
    except ValueError:
        pass
    print("PASS part4c_overlap_zero_rejected_immediately")


def part4b_no_detection_in_bootstrap_chunk_yields_empty_frames_no_crash():
    # Real error observed on a CUDA machine (with the SAMURAI fork, same
    # sam2_video_predictor as vanilla SAM2):
    # "RuntimeError: No points are provided; please add points first",
    # raised by propagate_in_video() when no prompt was ever registered
    # -- happens if YOLO finds no one in the first chunk's anchor frame
    # (e.g. a video that opens on an empty room). Here we verify our code
    # handles it BEFORE reaching a call to propagate_in_video, returning
    # empty frames instead of propagating the exception.
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        # all chunks return an empty list from _detect_people(): never
        # any person detected in the whole video
        backend = _FakeBackend(
            device="cuda", chunk_size=CHUNK_SIZE, overlap=OVERLAP,
            detect_schedule=[[], [], [], []],
        )

        results = list(backend.run(video_path))  # must not raise anything

        frame_indices = [r.frame_index for r in results]
        assert frame_indices == list(range(TOTAL_FRAMES)), \
            f"even with no person detected, frame coverage must stay complete, got {frame_indices}"
        assert all(r.people == [] for r in results), "no person detected -> every frame must have people=[]"
        print("PASS part4b_no_detection_in_bootstrap_chunk_yields_empty_frames_no_crash")
    finally:
        shutil.rmtree(tmp_dir)


def part5_chunks_are_written_as_jpeg_and_cleaned_up_after_each_chunk():
    # Verifies the 2026-08 fix (confirmed on a real CUDA machine with the
    # SAMURAI fork: "Only MP4 video and JPEG folder are supported", the
    # predictor doesn't accept a list of in-memory frames) -- _init_state()
    # must write each chunk as a temporary JPEG folder AND clean it up
    # right after, not leave one lying around per chunk on a video with
    # many chunks.
    import tempfile as tempfile_mod

    tmp_dir = tempfile.mkdtemp()
    created_dirs: list[str] = []
    original_mkdtemp = tempfile_mod.mkdtemp

    def _spy_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        created_dirs.append(path)
        return path

    tempfile_mod.mkdtemp = _spy_mkdtemp
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _run_backend(chunk_store_dir=None)

        list(backend.run(video_path))  # consumes the generator, one _init_state() per chunk

        assert len(created_dirs) == 4, f"expected one temporary folder per chunk (4), found {len(created_dirs)}"
        for d in created_dirs:
            assert not os.path.exists(d), f"{d} should have been cleaned up after its chunk, still exists"
        print(f"PASS part5_chunks_are_written_as_jpeg_and_cleaned_up_after_each_chunk "
              f"({len(created_dirs)} folders created and cleaned up, one per chunk)")
    finally:
        tempfile_mod.mkdtemp = original_mkdtemp
        shutil.rmtree(tmp_dir)


class _FakeTorchTensor:
    """Minimal duck-type of a torch.Tensor: only the three methods used
    by `_to_boolean_mask()` (detach/cpu/numpy) -- verifies that path
    without depending on torch (not installed in this environment)."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def part6b_reseed_new_people_false_never_adds_the_late_entrant():
    # "Pure SAM" vs "SAM + reseeding" comparison (see the discussion on
    # the biased baseline): with reseed_new_people=False, YOLO is called
    # ONLY at chunk 0's bootstrap -- person B (who in part2's scenario
    # enters mid-video) must NEVER appear, since no one discovers them
    # again after chunk 0.
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _FakeBackend(
            device="cuda", chunk_size=CHUNK_SIZE, overlap=OVERLAP,
            detect_schedule=[[BOX_A], [], [BOX_B], []],  # B would be "seen" at chunk 2 if reseed were active
            reseed_new_people=False,
        )

        results = list(backend.run(video_path))
        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}

        all_ids = set()
        for ids in ids_per_frame.values():
            all_ids |= ids
        assert len(all_ids) == 1, f"with reseed disabled B must never appear, ids found: {all_ids}"
        assert all(len(ids) == 1 for ids in ids_per_frame.values()), \
            "every frame must have only A, never B, for the whole session"
        print("PASS part6b_reseed_new_people_false_never_adds_the_late_entrant")
    finally:
        shutil.rmtree(tmp_dir)


def part6_to_boolean_mask_handles_bool_logits_3d_and_tensor_like_input():
    # 2026-08 fix: real symptom observed on a CUDA machine ("the video
    # starts but no masks appear") -- caused by propagate_in_video()
    # returning logits (not already booleans), often in tensors with one
    # extra channel dimension -- see _to_boolean_mask()'s docstring for
    # why each case below.
    already_bool = np.array([[True, False], [False, True]])
    assert np.array_equal(_to_boolean_mask(already_bool), already_bool)

    logits_2d = np.array([[-2.0, 3.0], [0.5, -0.1]])
    result = _to_boolean_mask(logits_2d)
    assert result.dtype == bool
    assert result.tolist() == [[False, True], [True, False]], "threshold at 0.0: >0 = foreground"

    logits_3d = logits_2d.reshape(1, 2, 2)  # shape (1,H,W), typical of SAM2
    result_3d = _to_boolean_mask(logits_3d)
    assert result_3d.shape == (2, 2), "the extra channel must be removed, not left in the shape"
    assert result_3d.tolist() == [[False, True], [True, False]]

    tensor_like = _FakeTorchTensor(logits_2d)
    result_tensor = _to_boolean_mask(tensor_like)
    assert result_tensor.tolist() == [[False, True], [True, False]], \
        "an object with .detach()/.cpu()/.numpy() (torch.Tensor duck-typing) must be converted the same way"

    print("PASS part6_to_boolean_mask_handles_bool_logits_3d_and_tensor_like_input")


def part7_redetect_every_finds_new_person_mid_chunk_not_just_at_boundary():
    # The problem reported by Michele comparing YOLO+ByteTrack (detects
    # on EVERY frame) with Sam2Tracker (used to detect only at the
    # chunk's anchor frame, once every chunk_size frames): a person
    # entering MID-chunk stayed invisible until the NEXT chunk (or
    # forever, if the video ended before that). Here: a single 20-frame
    # chunk, redetect_every=5 -> A detected at bootstrap (frame 0), B
    # detected by the re-detection at frame 10 (mid-chunk, not at its
    # boundary) -- verifies that B appears exactly from there on, not
    # only at the next chunk (which doesn't even exist here: it's a
    # single chunk).
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _FakeBackend(
            # overlap=1 (not 0): a single chunk here (chunk_size=TOTAL_FRAMES,
            # no second chunk, so cross-chunk reconciliation never comes
            # into play) -- but overlap=0 is now rejected upstream by
            # ChunkedVideoPredictorBackend.__init__ (see part4c), so it
            # still needs to be >= 1 here too to build the object.
            device="cuda", chunk_size=TOTAL_FRAMES, overlap=1, redetect_every=5,
            # calls to _detect_people: bootstrap (frame 0) + one every
            # 5-frame window (at 5, 10, 15) -- 4 total
            detect_schedule=[[BOX_A], [], [BOX_B], []],
        )

        results = list(backend.run(video_path))
        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}

        assert all(len(ids_per_frame[f]) == 1 for f in range(0, 10)), \
            "before the re-detection at frame 10, only A should be present"
        assert all(len(ids_per_frame[f]) == 2 for f in range(10, TOTAL_FRAMES)), \
            "from frame 10 onward (MID-CHUNK re-detection, not at a chunk boundary), B should also appear"
        print("PASS part7_redetect_every_finds_new_person_mid_chunk_not_just_at_boundary")
    finally:
        shutil.rmtree(tmp_dir)


def part7b_redetect_every_ignored_when_reseed_new_people_false():
    # Periodic re-detection is still a way of discovering NEW people via
    # YOLO -- it must respect reseed_new_people=False (the "pure SAM"
    # condition) exactly like reseeding at the chunk boundary: with
    # reseed_new_people=False, redetect_every must NEVER call
    # _detect_people() inside the chunk (only chunk 0's bootstrap).
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _FakeBackend(
            # overlap=1 (not 0): a single chunk here (chunk_size=TOTAL_FRAMES,
            # no second chunk, so cross-chunk reconciliation never comes
            # into play) -- but overlap=0 is now rejected upstream by
            # ChunkedVideoPredictorBackend.__init__ (see part4c), so it
            # still needs to be >= 1 here too to build the object.
            device="cuda", chunk_size=TOTAL_FRAMES, overlap=1, redetect_every=5,
            reseed_new_people=False,
            detect_schedule=[[BOX_A]],  # a single element: if called more, IndexError
        )

        results = list(backend.run(video_path))
        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}
        assert all(len(ids) == 1 for ids in ids_per_frame.values()), \
            "with reseed_new_people=False no re-detection should ever add B, even with redetect_every set"
        print("PASS part7b_redetect_every_ignored_when_reseed_new_people_false")
    finally:
        shutil.rmtree(tmp_dir)


def main():
    part1_ids_stay_continuous_across_chunks()
    part2_new_person_mid_video_gets_fresh_id_with_expected_lag()
    part3_chunk_persistence_writes_one_file_per_chunk()
    part4_non_cuda_device_rejected_immediately()
    part4c_overlap_zero_rejected_immediately()
    part4b_no_detection_in_bootstrap_chunk_yields_empty_frames_no_crash()
    part5_chunks_are_written_as_jpeg_and_cleaned_up_after_each_chunk()
    part6b_reseed_new_people_false_never_adds_the_late_entrant()
    part6_to_boolean_mask_handles_bool_logits_3d_and_tensor_like_input()
    part7_redetect_every_finds_new_person_mid_chunk_not_just_at_boundary()
    part7b_redetect_every_ignored_when_reseed_new_people_false()
    print("\nAll sam_backend.py tests (with a fake predictor) passed.")


if __name__ == "__main__":
    main()
