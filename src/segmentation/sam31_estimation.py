"""
sam31_estimation.py
=====================
`Sam31Tracker`: segmentation/tracking backend based on SAM 3.1
(facebookresearch/sam3, checkpoint `facebook/sam3.1` -- latest version
available as of 2026-08, released 2026-03-27 as the "Object Multiplex"
update to SAM 3, see https://github.com/facebookresearch/sam3). All the
chunking/persistence logic lives in
`segmentation/sam_backend.py::ChunkedVideoPredictorBackend` -- here is
the specific part: how to build the SAM 3.1 predictor and, new, how to
seed a chunk using SAM 3's TEXT prompt instead of YOLO box-seeding (see
`text_prompt` below).

Real API (confirmed by the official facebookresearch/sam3 README, "Basic
Usage" section -- NOT the SAM2-style one initially assumed by this
module/by `ChunkedVideoPredictorBackend`, completely different):

    from sam3.model_builder import build_sam3_video_predictor
    video_predictor = build_sam3_video_predictor()
    response = video_predictor.handle_request(request=dict(
        type="start_session", resource_path=video_path,  # JPEG folder or MP4 file
    ))
    response = video_predictor.handle_request(request=dict(
        type="add_prompt", session_id=response["session_id"],
        frame_index=0, text="<PROMPT>",
    ))
    output = response["outputs"]

A single function (`handle_request`, a "request" dict with a `type`
field) replaces both `init_state`/`add_new_points_or_box` and
(presumably) `propagate_in_video` from SAM2 -- here translated into the
three overridable methods of `ChunkedVideoPredictorBackend`
(`_init_state`, `_add_box_prompt`, `_propagate`) to reuse the
chunking/reconciliation/persistence logic without duplicating it.

Honesty about certainty level (updated after the first test on Michele's
real CUDA machine, 2026-08-06 -- see below for what changed):
- `start_session` and `add_prompt` with `text=`/`frame_index=` --
  CONFIRMED by the official README (snippet copied above) AND by the
  real run.
- REAL shape of the `response` for `add_prompt` with `text=` (CONFIRMED
  by the real run, different from what was initially assumed -- what was
  assumed was the shape of SAM 3's IMAGE API, `boxes`/`object_ids`/
  `scores`, not the VIDEO one):
      {
          "frame_index": ...,
          "outputs": {
              "out_obj_ids": [...],
              "out_boxes_xywh": [...],   # (x, y, width, height) per box
              "out_binary_masks": [...],
          },
      }
  `_add_text_prompt()`/`_propagate()` read these keys with a fallback to
  the old ones (`boxes`/`object_ids`/`masks`) for safety, and raise an
  explicit `RuntimeError` (with the found keys included in the message)
  if neither variant is found, instead of a silent `KeyError` -- see what
  happened the first time.
- `out_boxes_xywh` box format: (x, y, width, height), NOT
  (x1,y1,x2,y2) like the rest of the pipeline (`_polygon_to_box`,
  `_detect_people` via YOLO). It's also not clear from the value alone
  whether the coordinates are already in pixels or normalized [0,1]
  (common convention for detection APIs, not explicitly documented in
  the README) -- see `_xywh_to_xyxy_pixels()` below for the heuristic
  used (and its limitation).
- `add_prompt` with `box=`/`obj_id=` instead of `text=` (used by
  box-seeding via YOLO, kept for compatibility with this class's
  original use) -- STILL NOT confirmed on a real run (in the first test
  YOLO found no one on the anchor frame, so this path was never
  executed): remains an extrapolation by analogy with SAM2. First thing
  to check if a problem shows up here.
- `propagate_in_video` does NOT go through `handle_request()` --
  CONFIRMED on a real run (Michele, CUDA machine, 2026-08-06): the SAM 3
  dispatcher raised `RuntimeError("invalid request type: propagate_in_video")`.
  Multi-frame propagation uses a separate STREAMING method,
  `handle_stream_request()`, which returns a generator (one response per
  frame), see `_propagate()` below. `handle_request()` remains correct
  only for `start_session`/`add_prompt` (single response, confirmed).
- Exact field names of the streaming request (`start_frame_index`
  instead of `start_frame_idx`, `propagation_direction="forward"`) and
  the exact shape of each generator response -- NOT confirmed with the
  same certainty as the method change: inferred for consistency with the
  `frame_index` key already confirmed in `add_prompt`. The code first
  tries the real keys already confirmed for `add_prompt`
  (`out_obj_ids`/`out_binary_masks`, reused here by analogy), then the
  old ones as a fallback, and raises an explicit error with the found
  keys if neither matches -- instead of silently guessing again.

Text prompt mode (`text_prompt`, e.g. "person")
-----------------------------------------------------------
Born from a direct comparison with the CHUV production pipeline
(Video-Annotation-System, which uses `psifx video tracking sam3
inference --text_prompt "person"`, confirmed working in production): SAM
3 can discover ALL instances of a text concept in a frame ON ITS OWN,
without needing YOLO as a proposer. When `text_prompt` is set,
`_seed_new_chunk()` calls SAM 3 with that prompt on the anchor frame
instead of YOLO: SAM 3, HOWEVER, assigns its own LOCAL ids to the
discovered instances (we can't ask it to reuse one of our global ids,
unlike with the box-prompt) -- they are therefore reconciled by geometry
with the previous chunk's global ids using `chunking.reconcile_ids_windowed()`
(Hungarian assignment over a short multi-frame window, see chunking.py
for the 2026-08 revision -- this used to be a single-anchor-frame
greedy match, `chunking.reconcile_ids()`, which could steal an
already-known id for a nearby new detection; see the module docstring
of chunking.py for the concrete bug and the fix). A local id that finds
no geometric match anywhere in the window is tried against the
appearance gallery (`identity_gallery.py`) before a brand-new global id
is minted. If `text_prompt` is
`None` (default), behavior stays the original box-via-YOLO one,
unchanged -- `Sam31Tracker()` with no arguments behaves exactly as
before this change.

`redetect_every` (see sam_backend.py) stays compatible with
`text_prompt`: periodic re-detection inside the chunk still uses
YOLO/box (not a new text prompt, to avoid having to reconcile twice
mid-chunk), but the box-prompts pass an `obj_id` chosen by US (a global
id) -- being already a global id it needs no translation, `_propagate()`
passes it through unchanged (see `_local_to_global` below).
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from segmentation.chunking import GlobalIdAllocator, reconcile_ids_windowed
from segmentation.identity_gallery import SegmentationIdentityGallery
from segmentation.sam_backend import (
    ChunkedVideoPredictorBackend, _box_to_polygon, _resolve_new_person_id, _to_boolean_mask,
)

DEFAULT_CHECKPOINT = "facebook/sam3.1"


def _xywh_to_xyxy_pixels(box, frame_shape: tuple[int, int]) -> np.ndarray:
    """Converts a SAM 3.1 `out_boxes_xywh` box (x, y, width, height) into
    the (x1,y1,x2,y2) pixel format used by the rest of the pipeline
    (`_polygon_to_box`, `_detect_people` via YOLO). It's not documented
    whether the coordinates arrive already in pixels or normalized
    [0,1] -- normalized is assumed only if all 4 values are <= 1.5
    (margin above 1.0): a real person almost always occupies more than
    1.5 pixels, so the ambiguous case (pixels mistaken for normalized)
    is unlikely. If boxes appear tiny/clustered in a corner in the
    overlay, this is the first thing to check by hand."""
    x, y, w, h = (float(v) for v in box[:4])
    height, width = frame_shape
    if max(x, y, w, h) <= 1.5:
        x, y, w, h = x * width, y * height, w * width, h * height
    return np.array([x, y, x + w, y + h], dtype=float)


class Sam31Tracker(ChunkedVideoPredictorBackend):
    """See the module docstring. `text_prompt=None` (default): box-
    seeding via YOLO, original behavior unchanged. `text_prompt=
    "person"` (or another open-ended concept): SAM 3 discovers instances
    on its own, YOLO is never called for chunk seeding (it remains
    available only for `redetect_every`, see the module docstring)."""

    def __init__(self, *, checkpoint: str = DEFAULT_CHECKPOINT,
                 text_prompt: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.text_prompt = text_prompt
        # {SAM_local_id: global_id} for the CURRENT chunk, populated by
        # _seed_new_chunk() only in text_prompt mode -- see _propagate().
        # None/empty in box mode (the local ids ARE already the global
        # ids, having been chosen by us).
        self._local_to_global: dict[int, int] = {}

    def _build_predictor(self):
        # Delayed import: raises ImportError with a clear message if the
        # sam3 package isn't installed, instead of breaking the import of
        # the whole `segmentation` module for anyone not using this backend.
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as exc:
            raise ImportError(
                "Sam31Tracker requires the 'sam3' package (not installed). "
                "See https://github.com/facebookresearch/sam3 -- "
                "'git clone ... && pip install -e .', gated checkpoint on "
                "Hugging Face (facebook/sam3.1, requires approved access)."
            ) from exc
        return build_sam3_video_predictor()

    # ------------------------------------------------- handle_request API
    def _init_state(self, predictor, frames: list[np.ndarray]):
        """Like `ChunkedVideoPredictorBackend._init_state()` (writes the
        chunk as a temporary JPEG folder, same "MP4 or JPEG folder"
        constraint inherited from SAM2), but here the returned 'state' is
        a `session_id` (string) from `start_session`, not a library
        object -- see the module docstring for the real signature
        confirmed by the README."""
        self._current_chunk_tmpdir = tempfile.mkdtemp(prefix="sam31_tracker_")
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(self._current_chunk_tmpdir, f"{i:06d}.jpg"), frame)
        response = predictor.handle_request(request=dict(
            type="start_session", resource_path=self._current_chunk_tmpdir,
        ))
        return response["session_id"]

    def _add_box_prompt(self, predictor, state, *, frame_idx: int, obj_id: int, box: np.ndarray) -> None:
        # NOT confirmed by the README (which only shows the text example)
        # -- see the "Honesty" note in the module docstring.
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=state, frame_index=frame_idx,
            box=box, obj_id=obj_id,
        ))

    def _add_text_prompt(self, predictor, state, *, frame_idx: int, text: str,
                          frame_shape: tuple[int, int]) -> dict[int, np.ndarray]:
        """Calls SAM 3 with the text prompt on the given frame: REGISTERS
        the prompt AND returns the discovered instances in one shot
        (unlike the box-prompt, there's no way to separate 'discover'
        from 'track' here). Returns `{SAM_local_id: box_x1y1x2y2_pixel}`:
        these ids are LOCAL to this call/chunk, not persistent --
        reconciliation with global ids happens in `_seed_new_chunk()`,
        not here.

        `response` shape CONFIRMED on a real run (Michele, CUDA machine,
        2026-08-06) -- see the module docstring for details and what
        changed from the initial assumption (which was the shape of the
        IMAGE API, not the VIDEO one)."""
        response = predictor.handle_request(request=dict(
            type="add_prompt", session_id=state, frame_index=frame_idx, text=text,
        ))
        outputs = response.get("outputs", response)
        boxes_xywh = outputs.get("out_boxes_xywh", outputs.get("boxes"))
        if boxes_xywh is None:
            raise RuntimeError(
                f"SAM 3.1 returned no bounding boxes for the text prompt {text!r}. "
                f"Keys in response: {list(response.keys())}; "
                f"keys in outputs: {list(outputs.keys())}."
            )
        object_ids = outputs.get("out_obj_ids", outputs.get("object_ids", range(len(boxes_xywh))))
        return {
            int(oid): _xywh_to_xyxy_pixels(box, frame_shape)
            for oid, box in zip(object_ids, boxes_xywh)
        }

    def _propagate(self, predictor, state, *, start_frame_idx: int = 0,
                    max_frame_num_to_track: int | None = None):
        """See the base signature in `ChunkedVideoPredictorBackend`. In
        addition: translates the LOCAL ids discovered by the text prompt
        (see `_add_text_prompt`) into the corresponding GLOBAL ids, using
        the map built by `_seed_new_chunk()`. An id absent from the map
        (box-mode case, or a `redetect_every` box-prompt) passes through
        unchanged: it's already the global id, having been chosen by us
        at seeding time (see `_add_box_prompt`).

        CONFIRMED on a real run (Michele, CUDA machine, 2026-08-06):
        `handle_request()` does NOT recognize `type="propagate_in_video"`
        -- the SAM 3 dispatcher raises `RuntimeError("invalid request type")`.
        Multi-frame propagation goes through a separate STREAMING method,
        `handle_stream_request()`, which returns a GENERATOR (one
        response per frame) instead of a single dict with an "outputs"
        list -- `handle_request()` remains correct only for single-
        response requests (`start_session`, `add_prompt`, already
        confirmed). Exact field names of the streaming request
        (`start_frame_index` instead of `start_frame_idx`,
        `propagation_direction`) -- inferred for consistency with the
        `frame_index` key already confirmed in `add_prompt`, then
        independently corroborated (same names) by comparing with the
        official SAM 3 predictor: if the next error is again an "invalid
        request"/unknown parameter, this is the first thing to review.
        `max_frame_num_to_track` is OMITTED from the request if `None`
        instead of passing it explicitly (defensive: if the API validates
        types strictly, an explicit `None` might break where a missing
        key would fall back to the default)."""
        request = {
            "type": "propagate_in_video", "session_id": state,
            "propagation_direction": "forward", "start_frame_index": start_frame_idx,
        }
        if max_frame_num_to_track is not None:
            request["max_frame_num_to_track"] = max_frame_num_to_track
        local_to_global = self._local_to_global
        for response in predictor.handle_stream_request(request=request):
            outputs = response.get("outputs", response)
            frame_idx = response.get("frame_index", outputs.get("frame_index"))
            obj_ids = outputs.get("out_obj_ids", outputs.get("object_ids", outputs.get("obj_ids")))
            masks = outputs.get("out_binary_masks", outputs.get("masks"))
            if frame_idx is None or obj_ids is None or masks is None:
                raise RuntimeError(
                    f"SAM 3.1 handle_stream_request: unexpected frame response. "
                    f"Keys in response: {list(response.keys())}; "
                    f"keys in outputs: {list(outputs.keys()) if outputs is not response else '(not nested)'}."
                )
            remapped: dict[int, np.ndarray] = {}
            for oid, mask in zip(obj_ids, masks):
                oid = int(oid)
                global_id = local_to_global.get(oid, oid)
                remapped[global_id] = _to_boolean_mask(mask)
            yield int(frame_idx), remapped

    # ------------------------------------------------------------ seeding
    def _seed_new_chunk(self, predictor, state, *, chunk_frames: list[np.ndarray],
                         chunk_index: int, frame_shape: tuple[int, int],
                         prev_anchor_polys: dict[int, np.ndarray],
                         allocator: GlobalIdAllocator,
                         prev_anchor_polys_history: dict[int, dict[int, np.ndarray]] | None = None,
                         identity_gallery: SegmentationIdentityGallery | None = None,
                         ) -> dict[int, np.ndarray]:
        if not self.text_prompt:
            self._local_to_global = {}
            return super()._seed_new_chunk(
                predictor, state, chunk_frames=chunk_frames, chunk_index=chunk_index,
                frame_shape=frame_shape, prev_anchor_polys=prev_anchor_polys, allocator=allocator,
                prev_anchor_polys_history=prev_anchor_polys_history, identity_gallery=identity_gallery,
            )

        discovered = self._add_text_prompt(
            predictor, state, frame_idx=0, text=self.text_prompt, frame_shape=frame_shape,
        )

        local_to_global: dict[int, int] = {}
        seed_boxes: dict[int, np.ndarray] = {}
        if chunk_index == 0 or not prev_anchor_polys:
            for local_id, box in discovered.items():
                global_id = allocator.next_id()
                local_to_global[local_id] = global_id
                seed_boxes[global_id] = box
        else:
            discovered_polys = {local_id: _box_to_polygon(box) for local_id, box in discovered.items()}
            # Reconciles against a short trailing HISTORY of the
            # previous chunk (not just its single last anchor frame,
            # see chunking.reconcile_ids_windowed's docstring for why:
            # a degenerate polygon exactly at the boundary frame
            # shouldn't be the only chance a continuing person gets to
            # be recognized). Falls back to the single-frame dict if no
            # history was passed (e.g. an older caller/test).
            prev_by_frame = prev_anchor_polys_history or {0: prev_anchor_polys}
            mapping = reconcile_ids_windowed(prev_by_frame, {0: discovered_polys}, frame_shape,
                                              iou_threshold=self.iou_threshold)
            for local_id, box in discovered.items():
                global_id = mapping.get(local_id)
                if global_id is None:
                    # No geometric match anywhere in the window: try the
                    # appearance gallery before minting a brand-new id
                    # (see identity_gallery.py) -- this is the case of
                    # someone occluded across the WHOLE previous chunk,
                    # geometry alone can never recover them.
                    global_id = _resolve_new_person_id(identity_gallery, allocator, chunk_frames[0], box)
                local_to_global[local_id] = global_id
                seed_boxes[global_id] = box

        if self.max_people is not None and len(seed_boxes) > self.max_people:
            keep = set(list(seed_boxes.keys())[: self.max_people])
            seed_boxes = {gid: box for gid, box in seed_boxes.items() if gid in keep}
            local_to_global = {lid: gid for lid, gid in local_to_global.items() if gid in keep}

        self._local_to_global = local_to_global
        return seed_boxes
