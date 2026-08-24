"""
sam31_estimation.py
=====================
`Sam31Tracker`: segmentation/tracking backend based on SAM 3.1
(facebookresearch/sam3, checkpoint `facebook/sam3.1`, the "Object
Multiplex" update to SAM 3). All chunking/persistence logic lives in
`segmentation/sam_backend.py::ChunkedVideoPredictorBackend`; this module
supplies the SAM 3.1-specific parts: building the predictor, and seeding
a chunk with SAM 3's TEXT prompt instead of YOLO box-seeding (see
`text_prompt` below).

Real API (per the official facebookresearch/sam3 README -- a single
`handle_request(request=dict(type=..., ...))` call, NOT the SAM2-style
`init_state`/`add_new_points_or_box`/`propagate_in_video` initially
assumed), mapped onto `ChunkedVideoPredictorBackend`'s three overridable
methods (`_init_state`, `_add_box_prompt`, `_propagate`):

    video_predictor = build_sam3_video_predictor()
    r = video_predictor.handle_request(request=dict(type="start_session", resource_path=video_path))
    r = video_predictor.handle_request(request=dict(type="add_prompt", session_id=r["session_id"], frame_index=0, text="<PROMPT>"))
    output = r["outputs"]

Confirmed on a real CUDA run, notes for anyone touching this file again:
- `add_prompt` with `text=` returns `{"frame_index": ..., "outputs":
  {"out_obj_ids": [...], "out_boxes_xywh": [...], "out_binary_masks":
  [...]}}` -- the VIDEO api's shape, different from SAM 3's IMAGE api
  (`boxes`/`object_ids`/`scores`) initially assumed. Code reads these
  keys with a fallback to the old names, and raises an explicit
  `RuntimeError` (found keys included) rather than a silent `KeyError`
  if neither matches.
- `out_boxes_xywh` is (x, y, width, height), not (x1,y1,x2,y2) like the
  rest of the pipeline -- see `_xywh_to_xyxy_pixels()`. Whether the
  coordinates are pixels or normalized [0,1] isn't documented; that
  method's docstring has the heuristic used.
- `add_prompt` with `box=`/`obj_id=` (for YOLO box-seeding) does not
  exist as assumed by SAM2 analogy -- confirmed both by a real crash
  (`AssertionError: at least one type of prompt (text, boxes) must be
  provided`) and by reading `sam3/model/sam3_base_predictor.py` +
  `sam3_video_inference.py`: boxes are `bounding_boxes`/
  `bounding_box_labels` and route through the SEMANTIC prompt method,
  which resets the whole session and ignores a caller-chosen `obj_id`.
  Fixed in `_add_box_prompt()` by going through the POINT-based tracker
  api instead (`points=`/`point_labels=`/`obj_id=`, confirmed to respect
  `obj_id` and not reset the session) -- see that method's docstring.
- `propagate_in_video` is not a `handle_request()` type (confirmed:
  raises `RuntimeError("invalid request type: propagate_in_video")`).
  Multi-frame propagation is the separate streaming method
  `handle_stream_request()`, returning a generator (see `_propagate()`).
  Exact streaming request/response field names are inferred by analogy
  with `add_prompt`'s confirmed shape, not independently confirmed; code
  tries the confirmed keys first, then falls back, then raises with the
  found keys rather than guessing silently.

Text prompt mode (`text_prompt`, e.g. "person"): mirrors CHUV's
production pipeline (`psifx video tracking sam3 inference --text_prompt
"person"`). SAM 3 can discover every instance of a concept on its own,
no YOLO proposer needed. When set, `_seed_new_chunk()` calls SAM 3 with
the prompt on the anchor frame instead of YOLO; SAM 3 assigns its own
LOCAL ids to what it finds (can't be told to reuse a global id like with
box-prompting), so they're reconciled against the previous chunk's
global ids via `chunking.reconcile_ids_windowed()` (Hungarian over a
short multi-frame window -- see chunking.py for why this replaced a
single-anchor-frame greedy match). An id with no geometric match in the
window is tried against the appearance gallery (`identity_gallery.py`)
before minting a new global id. `text_prompt=None` (default) keeps the
original box-via-YOLO behavior unchanged.

`redetect_every` technically runs alongside `text_prompt` (still uses
YOLO/box mid-chunk, passing a global `obj_id` straight through, no
translation needed) but the combination isn't recommended: injecting a
person mid-chunk via `_add_box_prompt` perturbs SAM 3.1's session-wide
"masklet confirmation" bookkeeping (see `Sam31Tracker.__init__`'s
warning) and was observed to make already-tracked people flicker, not
just cleanly add the new one. Prefer a smaller `chunk_size` over
`redetect_every` when using `text_prompt`.
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

        if self.text_prompt and self.redetect_every:
            # 2026-08 finding (Michele, dancing-tracks test): with
            # text_prompt set, `redetect_every` injects a new tracked
            # object MID-PROPAGATION via the points-based tracker API
            # (`_add_box_prompt`, see its docstring). Reading the real
            # source (sam3_video_inference.py::propagate_in_video)
            # revealed SAM 3.1 runs its OWN internal "masklet
            # confirmation" state machine during propagation (an object
            # must be detected `masklet_confirmation_consecutive_det_thresh`
            # consecutive frames before being "confirmed" and shown;
            # `hotstart_delay` buffers frames while this resolves;
            # `suppressed_obj_ids`/`removed_obj_ids`/`unconfirmed_obj_ids`
            # hide objects that don't pass) -- this bookkeeping is
            # session-wide, not isolated per object. Observed symptom:
            # injecting a person mid-chunk made ALREADY-tracked people
            # flicker/disappear in alternation, not just the new one --
            # consistent with the injection perturbing that shared
            # confirmation state, not (only) a point-vs-box precision
            # issue. No known way to inject an object mid-propagation
            # without touching this state, so the two are NOT safely
            # combinable yet -- use a smaller `chunk_size` instead to
            # get more frequent (but always clean, full-session) text
            # re-discovery, which already goes through the tested
            # Hungarian + multi-frame + appearance-gallery reconciliation
            # in chunking.py/identity_gallery.py. Logged, not raised: the
            # combination doesn't crash and might be fine on a calmer
            # (less crowded/occluded) video -- this is a quality warning,
            # not a hard incompatibility.
            print(f"[{type(self).__name__}] warning: text_prompt + redetect_every are set "
                  f"together -- SAM 3.1's internal masklet-confirmation state during "
                  f"propagation is session-wide, so injecting a new person mid-chunk can "
                  f"make ALREADY-tracked people flicker/disappear too (observed on the "
                  f"dancing-tracks test, 2026-08). Recommended instead: leave redetect_every "
                  f"unset and use a smaller chunk_size so full text re-discovery (already "
                  f"reconciled via chunking.reconcile_ids_windowed + the appearance gallery) "
                  f"happens more often.")

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

    def _add_box_prompt(self, predictor, state, *, frame_idx: int, obj_id: int, box: np.ndarray,
                         frame_shape: tuple[int, int] | None = None) -> None:
        """Registers ONE new tracked object at a caller-chosen `obj_id`,
        WITHOUT disturbing any object already tracked in this session.

        CORRECTED 2026-08 -- real crash on a CUDA machine (Michele,
        `redetect_every` mid-chunk): `AssertionError: at least one type
        of prompt (text, boxes) must be provided`. The previous version
        sent `box=`/`obj_id=` in the request dict, assuming SAM 3.1 would
        behave like SAM2's `add_new_points_or_box` (per-object box
        prompt, caller-chosen id) -- the "Honesty" note above used to
        flag this path as literally never exercised on a real run.

        Read the real source (github.com/facebookresearch/sam3,
        confirmed against `sam3/model/sam3_base_predictor.py` and
        `sam3/model/sam3_video_inference.py`, both fetched 2026-08) to
        find the actual cause and the correct fix:
        - `Sam3BasePredictor.add_prompt()` (the `handle_request()`
          dispatch target for `type="add_prompt"`) has NO `box` request
          key at all -- boxes are named `bounding_boxes`/
          `bounding_box_labels`. Our old `box=` key was silently dropped,
          `text`/`bounding_boxes` both stayed `None`, hence the crash.
        - Even the CORRECT box key (`bounding_boxes`) would only route
          to `Sam3VideoInference.add_prompt()` (`sam3_video_inference.py:843`),
          the SEMANTIC (text/box discovery) method: it calls
          `self.reset_state(...)` UNCONDITIONALLY on every call -- wiping
          out every object already tracked in the session -- and its
          `obj_id` argument is silently dropped for the box branch: ids
          are assigned by SAM, not chosen by the caller. Fine for
          text-prompt discovery (we already reconcile SAM's own ids
          ourselves, see `_seed_new_chunk`/`chunking.reconcile_ids_windowed`),
          but wrong for "add exactly this one already-identified person
          back, keep everyone else": it would silently un-track everyone
          else in the session on every single call -- a worse bug than
          the crash, just quieter (see chunking.py's own admission of a
          previously-unknown-unknown here).
        - The primitive that DOES respect a caller-chosen `obj_id` and
          does NOT reset the session is the POINT-based tracker path
          (`Sam3VideoInferenceWithInstanceInteractivity.add_prompt()` ->
          `add_tracker_new_points()`, `sam3_video_inference.py:1364-1401`):
          reached by sending `points=`/`point_labels=` (NOT
          `bounding_boxes=`) together with `obj_id=`.

        Converts the YOLO/redetect box to its CENTROID as a single
        positive point, normalized to 0..1 (`rel_coordinates=True` is
        the real API's default for points -- same normalization already
        confirmed for `out_boxes_xywh`, see `_xywh_to_xyxy_pixels`).
        Coarser than a full box (SAM infers the object's extent from one
        click instead of being told its boundary) -- trades a possibly
        less precise FIRST-frame mask for correctness (right id, no
        session wipe); propagation over the following frames typically
        recovers precision. Worth watching in crowded/overlapping scenes
        (dancing-tracks test) -- if the initial mask consistently bleeds
        into a neighbor, a second (negative) point on the neighbor is the
        natural next refinement, now possible precisely because this
        path doesn't reset the session either.

        `frame_shape` is REQUIRED here (unlike the base class's version,
        which doesn't need it) to normalize the point -- raises
        `ValueError` instead of silently sending an unnormalized/wrong
        point if a caller forgets it."""
        if frame_shape is None:
            raise ValueError(
                "Sam31Tracker._add_box_prompt requires frame_shape (needed to "
                "normalize the point prompt to the 0..1 range the real SAM 3.1 "
                "API expects, see this method's docstring)"
            )
        height, width = frame_shape
        cx = float((box[0] + box[2]) / 2.0) / width
        cy = float((box[1] + box[3]) / 2.0) / height
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=state, frame_index=frame_idx,
            points=[[cx, cy]], point_labels=[1], obj_id=obj_id,
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
