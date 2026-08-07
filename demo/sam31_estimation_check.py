"""
sam31_estimation_check.py
============================
Verifies `segmentation/sam31_estimation.py::Sam31Tracker` with a FAKE SAM
3 predictor that speaks the REAL `handle_request()` API (confirmed by the
official facebookresearch/sam3 README -- see the docstring of the module
under test), not the SAM2-style one used by `sam_backend_check.py` for
the base `Sam2Tracker`/`ChunkedVideoPredictorBackend`.

Two scenarios:
- Box mode (default, `text_prompt=None`): the same YOLO logic as always,
  but routed through `handle_request()` instead of SAM2-style calls --
  verifies that the translation doesn't break anything.
- Text prompt mode (`text_prompt="person"`): SAM 3 "discovers" instances
  on its own (ids LOCAL to the chunk, different each time) -- verifies
  that `chunking.reconcile_ids()` correctly reconciles them with the
  GLOBAL ids from the previous chunk by geometry, and that a person with
  a very different box (no overlap) gets a new id.

Usage:
    python demo/sam31_estimation_check.py
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.sam31_estimation import Sam31Tracker  # noqa: E402

FRAME_SHAPE = (64, 64)  # (height, width)
TOTAL_FRAMES = 20
CHUNK_SIZE = 15
OVERLAP = 5

BOX_A = np.array([5.0, 5.0, 15.0, 15.0])
BOX_B = np.array([40.0, 40.0, 55.0, 55.0])  # far from A, no overlap


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


class _FakeSam3Predictor:
    """Simulates SAM 3's `handle_request()`/`handle_stream_request()`
    (see the docstring of sam31_estimation.py): `start_session` opens a
    session (counts the JPEGs written by `_init_state()`, same check as
    `sam_backend_check.py`), `add_prompt` with `text=` "discovers"
    instances from a fixed script (`discover_schedule`, one element per
    expected call) and responds with the REAL shape confirmed on a CUDA
    machine (out_obj_ids/out_boxes_xywh/out_binary_masks, not
    boxes/object_ids as initially assumed), `add_prompt` with
    `box=`/`obj_id=` registers a known seed. `handle_request()` REJECTS
    "propagate_in_video" (like the real SAM 3 dispatcher, confirmed on a
    CUDA machine) -- propagation goes through `handle_stream_request()`,
    a GENERATOR that returns the static masks for the current obj_ids,
    one response per frame, in the requested window."""

    def __init__(self, frame_shape: tuple[int, int], discover_schedule: list[dict[int, np.ndarray]]):
        self.frame_shape = frame_shape
        self._discover_schedule = discover_schedule
        self._discover_call_count = 0
        self._sessions: dict[int, dict] = {}
        self._next_session = 0

    def handle_request(self, *, request: dict) -> dict:
        req_type = request["type"]

        if req_type == "start_session":
            sid = self._next_session
            self._next_session += 1
            num_frames = len([f for f in os.listdir(request["resource_path"]) if f.endswith(".jpg")])
            self._sessions[sid] = {"num_frames": num_frames, "seeds": {}}
            return {"session_id": sid}

        if req_type == "add_prompt":
            session = self._sessions[request["session_id"]]
            if "text" in request:
                discovered = self._discover_schedule[self._discover_call_count]
                self._discover_call_count += 1
                session["seeds"].update(discovered)
                obj_ids = list(discovered.keys())
                # REAL shape confirmed on a CUDA machine (see the
                # docstring of the module under test): out_boxes_xywh is
                # (x,y,w,h), not (x1,y1,x2,y2) -- here it's converted
                # from BOX_A/BOX_B (already x1y1x2y2, also used for
                # _box_to_mask) ONLY for the network payload, so the test
                # also verifies the conversion done by
                # _xywh_to_xyxy_pixels().
                boxes_xywh = [
                    np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]], dtype=float)
                    for b in discovered.values()
                ]
                return {
                    "frame_index": request["frame_index"],
                    "outputs": {
                        "out_obj_ids": obj_ids,
                        "out_boxes_xywh": boxes_xywh,
                        "out_binary_masks": [_box_to_mask(discovered[oid], self.frame_shape) for oid in obj_ids],
                    },
                }
            session["seeds"][request["obj_id"]] = request["box"]
            return {"outputs": {}}

        # CONFIRMED on a real run (see the docstring of the module under
        # test): handle_request() does NOT recognize
        # "propagate_in_video", only handle_stream_request() does -- here
        # the same rejection from the real SAM 3 dispatcher is replicated,
        # so the test fails loudly if the code under test were to
        # mistakenly call handle_request() for propagation.
        raise RuntimeError(f"invalid request type: {req_type}")

    def handle_stream_request(self, *, request: dict):
        """Simulates real VIDEO propagation: a GENERATOR, one response
        per frame (not a single dict with an "outputs" list), see the
        docstring of the module under test for the confirmation on a CUDA
        machine."""
        req_type = request["type"]
        if req_type != "propagate_in_video":
            raise RuntimeError(f"invalid stream request type: {req_type}")
        session = self._sessions[request["session_id"]]
        start = request["start_frame_index"]
        # Key OMITTED (not explicit None) when not set -- see
        # sam31_estimation.py::_propagate() -- .get() instead of []
        # to mirror exactly what the real code sends.
        max_n = request.get("max_frame_num_to_track")
        end = session["num_frames"] if max_n is None else min(session["num_frames"], start + max_n)
        for local_idx in range(start, end):
            obj_ids = list(session["seeds"].keys())
            masks = [_box_to_mask(session["seeds"][oid], self.frame_shape) for oid in obj_ids]
            yield {
                "frame_index": local_idx,
                "outputs": {"out_obj_ids": obj_ids, "out_binary_masks": masks},
            }


def _make_tracker(fake_predictor, **kwargs) -> Sam31Tracker:
    tracker = Sam31Tracker(device="cuda", chunk_size=CHUNK_SIZE, overlap=OVERLAP, **kwargs)
    tracker._build_predictor = lambda: fake_predictor  # bypass sam3 import (not installed here)
    return tracker


def part1_box_mode_via_handle_request_matches_original_behaviour():
    # text_prompt=None (default): the same YOLO/box logic as always, just
    # routed through handle_request() -- verifies that the API
    # translation didn't break any of the original behavior.
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        fake_predictor = _FakeSam3Predictor(FRAME_SHAPE, discover_schedule=[])
        tracker = _make_tracker(fake_predictor)
        tracker._detect_people = lambda frame: [BOX_A]  # always the same person, for simplicity

        results = list(tracker.run(video_path))
        frame_indices = [r.frame_index for r in results]
        assert frame_indices == list(range(TOTAL_FRAMES)), \
            f"expected all frames with no gaps/duplicates, got {frame_indices}"
        assert all(len(r.people) == 1 for r in results), "only one person (A) in every frame"
        all_ids = {p[0] for r in results for p in r.people}
        assert len(all_ids) == 1, f"same id across the whole session (box-mode = we choose obj_id), found {all_ids}"
        print("PASS part1_box_mode_via_handle_request_matches_original_behaviour")
    finally:
        shutil.rmtree(tmp_dir)


def part2_text_prompt_reconciles_continuing_person_and_assigns_new_id_to_stranger():
    # chunk 0: SAM 3 "discovers" a person with its local id 0 (box A).
    # chunk 1: SAM 3 rediscovers the SCENE from scratch (new text prompt)
    # and assigns DIFFERENT local ids (7 and 8, arbitrary: shows that no
    # numbering coincidence is relied upon) -- 7 is in the SAME position
    # as A (must reconcile to the same global id), 8 is in a position
    # never seen before (box B, must receive a new global id).
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        fake_predictor = _FakeSam3Predictor(FRAME_SHAPE, discover_schedule=[
            {0: BOX_A},
            {7: BOX_A, 8: BOX_B},
        ])
        tracker = _make_tracker(fake_predictor, text_prompt="person")

        results = list(tracker.run(video_path))
        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}

        # Chunk 1 discovers B at ITS frame 0 (global 10), but that frame
        # is in the overlap window already emitted by chunk 0 -- as with
        # YOLO reseeding (see part2 in sam_backend_check.py), B appears
        # in the output only from the first frame REALLY emitted by
        # chunk 1, i.e. 10 + OVERLAP.
        first_seen = 10 + OVERLAP
        assert all(len(ids_per_frame[f]) == 1 for f in range(0, first_seen)), \
            f"before {first_seen}, only person A should be present"
        assert all(len(ids_per_frame[f]) == 2 for f in range(first_seen, TOTAL_FRAMES)), \
            f"from {first_seen} onward, both A (continuing) and B (new) should be present"

        a_id_before = next(iter(ids_per_frame[0]))
        a_id_after = ids_per_frame[first_seen] & ids_per_frame[0]  # id shared between before and after
        assert a_id_after == {a_id_before}, (
            f"A's global id must remain the same despite SAM reassigning a "
            f"different local id (0 -> 7) -- expected {{{a_id_before}}}, found {a_id_after}"
        )
        new_id = ids_per_frame[first_seen] - {a_id_before}
        assert len(new_id) == 1, f"expected exactly one new id for B, found {new_id}"
        assert next(iter(new_id)) != a_id_before, "B must have a GLOBAL id different from A's"

        all_ids = set()
        for ids in ids_per_frame.values():
            all_ids |= ids
        assert len(all_ids) == 2, f"expected exactly 2 global ids across the whole session, found {all_ids}"
        print("PASS part2_text_prompt_reconciles_continuing_person_and_assigns_new_id_to_stranger")
    finally:
        shutil.rmtree(tmp_dir)


def main():
    part1_box_mode_via_handle_request_matches_original_behaviour()
    part2_text_prompt_reconciles_continuing_person_and_assigns_new_id_to_stranger()
    print("\nAll sam31_estimation.py tests (with a fake SAM 3 predictor) passed.")


if __name__ == "__main__":
    main()
