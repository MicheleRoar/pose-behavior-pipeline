"""
sam_backend.py
================
Shared base for `Sam31Tracker` (sam31_estimation.py) and `Sam2Tracker`
(sam2_estimation.py): both libraries expose the SAME stateful video API
(official docs: facebookresearch/sam3 and facebookresearch/sam2) --

    state = predictor.init_state(frames)
    predictor.add_new_points_or_box(state, frame_idx=..., obj_id=..., box=...)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        ...

-- so all the chunking/prompting/reconciliation/persistence logic lives
here ONCE; the two subclasses implement only `_build_predictor()`
(which library to import/instantiate, delayed import because neither
sam3 nor sam2 are installable in this environment: they require Python
3.12+/CUDA 12.6+, see requirements.txt).

Why chunking (see also segmentation/chunking.py)
------------------------------------------------------------
`init_state()` loads the pixels of ALL passed frames into memory -- on a
video several minutes long it's not feasible to pass it whole. It's
processed in overlapping windows (`chunk_size` frames, `overlap` shared
between one chunk and the next).

ID prompting/continuity strategy
------------------------------------------------
- First chunk: no known person yet. YOLO is used (the same model already
  used by `SegTracker`, here only as a DETECTOR on a single frame, not
  as a tracker) to propose the initial boxes on the chunk's first frame.
  NOTE: this means the quality of the initial prompt still depends on
  YOLO -- SAM here replaces TRACKING/re-id over time, not necessarily
  the initial detection (one could switch to a text prompt "person" if
  the SAM 3.1 concept-prompting model supports it well enough; to be
  verified on the CUDA machine, see Sam31Tracker).
- Subsequent chunks: for each already-known person (global id) a box
  prompt is derived from their mask in the LAST frame of the previous
  chunk (which is also the FIRST frame -- "anchor frame" -- of the new
  chunk, being in the overlap window), and it's registered with
  `obj_id=<global id>`: SAM thus continues to directly use the same id,
  no need for after-the-fact matching in the common case. YOLO is ALSO
  run on the anchor frame to spot NEW people (who entered the frame
  during the previous chunk) not already covered by an existing prompt
  (low IoU with all already-seeded boxes): these are assigned a
  never-used global id.
- As a consistency check (not as a primary matching mechanism, see
  above), the mask SAM produces on the anchor frame is still compared
  against the seeded one (`chunking.polygon_iou`): if the IoU drops
  below `iou_threshold` only a warning is logged -- it may mean SAM lost
  the person or "swapped" identities in the overlap, useful to see in
  the logs but not handled automatically in this first version (see
  chunking.py for the known limitation).

Periodic re-detection within the chunk (`redetect_every`)
-------------------------------------------------------------
Observed problem (Michele, direct comparison YOLO+ByteTrack vs
Sam2Tracker on the same video): YOLO+ByteTrack detects on EVERY frame,
while in the scheme above YOLO runs only once per chunk (the anchor
frame) -- with the default `chunk_size` of 600, one detection every ~40s
at 15fps. Whoever isn't exactly in the anchor frame stays invisible for
the ENTIRE rest of the chunk, even if visible in 99% of the other
frames; on top of that SAM propagates by inertia (internal memory) and
if it loses a person halfway through a chunk (occlusion, fast movement)
it has no way to re-detect them before the next chunk boundary. Result:
many fewer masks compared to YOLO+ByteTrack, not due to a model quality
difference but due to re-detection frequency.

`redetect_every` (default `None` = unchanged behavior, a single window
as large as the chunk) splits each chunk into sub-windows of
`redetect_every` frames: one sub-window is propagated at a time
(`propagate_in_video(state, start_frame_idx=..., max_frame_num_to_track=...)`,
assumed available because documented in the official SAM2/SAM3 notebooks
for adding objects mid-video -- NOT yet verified on a real CUDA machine,
see the "Honesty" note below), then YOLO is called on the first frame of
the NEXT sub-window to propose any new people not yet seeded (same IoU
comparison as `reseed_new_people`, and indeed it respects the same flag:
with `reseed_new_people=False` no re-detection happens, neither at the
chunk boundary nor within the chunk -- it remains the "pure SAM"
condition). Already-known people keep propagating automatically (same
`state`, same SAM memory) -- no need to reseed them at every
sub-window, re-detection ONLY proposes any new entries.

Honesty about what's verified here
--------------------------------------
This module was initially written and tested ONLY with a fake predictor
injected in place of sam3/sam2 (see `tests/sam_backend_check.py`, no CUDA
GPU in this environment). `_init_state()` below was however CORRECTED
based on a real test on a CUDA machine (with the SAMURAI fork, which
vendors the same `sam2_video_predictor` as vanilla SAM2 -- same expected
behavior with `Sam2Tracker`): the predictor does NOT accept a list of
in-memory frames as was initially assumed -- it hit "Only MP4 video and
JPEG folder are supported". Each chunk is therefore written as a JPEG
sequence in a temporary folder (see below) before calling
`init_state()`. Not yet confirmed whether SAM 3.1 has the same
constraint (it inherits the same video code from SAM2, so likely) -- if
it turns out it also accepts frame lists, `_init_state()` remains
overridable per subclass in case it's worth differentiating in the
future.

Why Sam2Tracker and not SamuraiTracker
------------------------------------------
`SamuraiTracker` (removed) used the same predictor with SAMURAI's
motion-aware mode active. Verified on a real CUDA machine: that code
(`sam2_base.py::_forward_sam_heads`) assumes a single object per session
(written/validated on the LaSOT/GOT-10k/TrackingNet single-target visual
object tracking benchmarks) -- seeding multiple people in the same
session (the normal case here) crashes with `RuntimeError: Boolean value
of Tensor with more than one value is ambiguous`. Vanilla SAM2
(`Sam2Tracker`, sam2_estimation.py) uses the same predictor WITHOUT that
patch: it natively supports batched multi-object, at the cost of losing
motion-modeling through occlusions. See the docstring of
`segmentation/sam2_estimation.py` for the full detail.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Iterator

import cv2
import numpy as np

from common.yolo_models import resolve_yolo_weights
from segmentation.chunk_store import save_chunk
from segmentation.chunking import GlobalIdAllocator, iter_chunk_ranges, polygon_iou
from segmentation.identity_gallery import SegmentationIdentityGallery
from segmentation.seg_estimation import SegFrameResult

COCO_PERSON_CLASS_ID = 0
NEW_PERSON_IOU_THRESHOLD = 0.2  # below this threshold a YOLO detection on
# the anchor frame is considered a NEW person (not already seeded)

RECONCILE_HISTORY_FRAMES = 5  # how many trailing frames of the just-finished
# chunk's overlap window are kept as extra "recent evidence" for the NEXT
# chunk's reconciliation (see chunking.reconcile_ids_windowed, used by
# Sam31Tracker in text-prompt mode) -- not used by the base class's own
# box-mode seeding, which reuses global ids deterministically and never
# needs geometric reconciliation in the first place.


def _probe_video(source) -> tuple[int, tuple[int, int]]:
    """Total frame count and (height, width) of the source video --
    used to size `polygon_iou`'s rasterizations and to compute chunks.
    Same "metadata only, no inference" approach as
    `webui/api.py::probe_video_metadata`, reimplemented here to avoid
    creating a dependency of `segmentation/` on `webui/`."""
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise ValueError(f"Unable to open video source: {source!r}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return total_frames, (height, width)
    finally:
        cap.release()


def _read_frame_range(source, start: int, end: int) -> list[np.ndarray]:
    """Reads frames `[start, end)` into memory (BGR). Seeking via
    `CAP_PROP_POS_FRAMES` isn't guaranteed to be perfectly accurate on
    all codecs/containers (a note already present elsewhere in the
    project, see `probe_video_metadata`), but is sufficient for this
    use: any one- or two-frame drift doesn't compromise reconciliation,
    which works on an overlap window of dozens of frames anyway."""
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(end - start):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames
    finally:
        cap.release()


def _to_boolean_mask(mask) -> np.ndarray:
    """Converts the mask object returned by `propagate_in_video()` into
    a boolean numpy array (H,W), whatever the starting format --
    verified ONLY with the fake predictor in the tests so far (no CUDA
    GPU in this environment): SAM2 typically returns a PyTorch LOGIT
    tensor (real values, not already 0/1), on GPU, often with an extra
    channel dimension (e.g. shape (1,H,W) instead of (H,W)). Without
    this conversion `_mask_to_polygon()` received data in the wrong
    format and produced empty or nonsensical contours WITHOUT raising an
    exception -- hence the observed symptom ("the video plays but no
    masks appear"), not a crash.

    - If the object has a `.detach()` method (duck-typing for
      torch.Tensor, no `import torch` here: it must not be required to
      use only the YOLO backend), it's moved to CPU and converted to
      numpy.
    - If it remains a 3-dimensional array (extra channel like (1,H,W)),
      only the first channel is kept.
    - If it's already boolean, it's returned as-is (the fake predictor's
      case in the tests, and of a hypothetical predictor that already
      returns ready-made masks).
    - Otherwise it's ASSUMED to be logits and thresholded at 0.0
      (foreground if > 0), the convention SAM2 uses for its
      mask_logits. If on the CUDA machine the masks turn out to be
      obviously wrong (too small/large/empty) despite this fix, it's
      the first thing to check: they might already be probabilities in
      [0,1], in which case the right threshold would be 0.5, not 0.0."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.dtype == bool:
        return mask
    return mask > 0.0


def _mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    """Binary mask (H,W) -> polygon (N,2) of the largest outer contour
    (a person can produce multiple connected components due to a "hole"
    in the mask; only the main one is kept, the same pragmatic choice as
    `mask_area`/`mask_centroid` in seg_estimation.py, which treat a mask
    as a single polygon). Empty polygon (0,2) if the mask contains no
    positive pixels."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2))
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(float)


def _polygon_to_box(poly: np.ndarray) -> np.ndarray:
    """Bounding box (x1,y1,x2,y2) of the polygon, used as a prompt for
    `add_new_points_or_box()`. `[0,0,0,0]` if the polygon is empty."""
    if poly.shape[0] == 0:
        return np.zeros(4)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    return np.array([x1, y1, x2, y2], dtype=float)


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    """Inverse of `_polygon_to_box`: rectangular polygon (4,2) from a box
    (x1,y1,x2,y2) -- needed to reuse `chunking.polygon_iou`/`reconcile_ids`
    (which work on polygons) when the only information available is a
    box (e.g. a YOLO detection, or instances discovered by a SAM 3 text
    prompt, see `Sam31Tracker._seed_new_chunk()`)."""
    return np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])


class ChunkedVideoPredictorBackend:
    """See the module docstring. Subclasses must implement
    `_build_predictor()`; they can override `_init_state()` /
    `_add_box_prompt()` / `_propagate()` if the real library signature
    diverges from what's assumed here (see the "Honesty about what's
    verified" note above)."""

    def __init__(self, *, device: str = "cuda", chunk_size: int = 600,
                 overlap: int = 50, iou_threshold: float = 0.3,
                 prompt_model: str = "yolo26s-seg.pt",
                 conf_threshold: float = 0.1,
                 chunk_store_dir: str | None = None,
                 max_people: int | None = None,
                 reseed_new_people: bool = True,
                 redetect_every: int | None = None,
                 appearance_fallback: bool = True,
                 appearance_device: str = "cpu",
                 identity_similarity_threshold: float | None = None,
                 identity_max_lost_age_chunks: int | None = None):
        if device != "cuda":
            # SAM 3.1/SAM2 don't declare mps/cpu support (see the
            # research cited in the README) -- better to fail
            # immediately with a clear message than to let it try and
            # get an obscure error inside the library.
            raise ValueError(
                f"{type(self).__name__} requires device='cuda' (SAM 3.1/SAM2 "
                f"don't currently support mps/cpu) -- received device={device!r}"
            )
        if overlap < 1:
            # ID reconciliation between consecutive chunks
            # (reconcile_ids, see run() below) compares masks on the
            # SAME frame produced by both chunks (the "anchor frame",
            # see the module/chunking.py docstring) -- with overlap=0
            # the chunks are adjacent but share NO frame, so
            # `prev_anchor_polys` would always be empty and every new
            # chunk would assign ALL-new global ids, silently losing
            # identity continuity at every boundary -- a worse failure
            # than an explicit error here. The production CHUV pipeline
            # doesn't have an overlap parameter (see
            # Video-Annotation-System, chunk_size=400 with no overlap),
            # but reconciles differently (direct IoU between the last
            # tracked frame of chunk N and the first of chunk N+1, not a
            # comparison on the same frame) -- a different design from
            # this one, not replicable just by setting overlap=0 here.
            raise ValueError(
                f"overlap must be >= 1 (received {overlap}) -- id reconciliation "
                "between chunks requires at least one shared frame, see the "
                "docstring of ChunkedVideoPredictorBackend.__init__"
            )
        self.device = device
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold
        self.prompt_model = prompt_model
        self.conf_threshold = conf_threshold
        self.chunk_store_dir = chunk_store_dir
        self.max_people = max_people
        # If False: YOLO is called ONLY at the bootstrap of chunk 0
        # (there must be a first prompt from somewhere), never to
        # discover NEW people at subsequent chunk boundaries -- SAM/SAM2
        # remains free to handle (or not handle) someone entering
        # mid-video on its own. Needed to obtain a "pure SAM" condition
        # to compare against the default one (with reseeding), instead
        # of having a single hybrid method passed off as "SAM's
        # capability" -- see the discussion on the skewed baseline and
        # benchmark_backends.py.
        self.reseed_new_people = reseed_new_people
        # See "Periodic re-detection" in the module docstring. None = a
        # single window per chunk (original, unchanged behavior).
        self.redetect_every = redetect_every
        self._detector = None  # YOLO lazily loaded in _detect_people()
        self._current_chunk_tmpdir: str | None = None  # see _init_state()/run()

        # Appearance-based fallback (see segmentation/identity_gallery.py)
        # for detections that find no geometric match at a chunk boundary
        # or during periodic re-detection -- tries an OSNet embedding
        # match against RECENTLY LOST global ids before minting a brand
        # new one. Gracefully becomes a no-op (every lookup returns None)
        # if torch/torchreid aren't installed, same as everywhere else
        # OSNet is used in this project -- appearance_fallback=True
        # (default) never raises just because the optional dependency is
        # missing; set it to False explicitly to skip trying altogether.
        gallery_kwargs = {}
        if identity_similarity_threshold is not None:
            gallery_kwargs["similarity_threshold"] = identity_similarity_threshold
        if identity_max_lost_age_chunks is not None:
            gallery_kwargs["max_lost_age_chunks"] = identity_max_lost_age_chunks
        self._identity_gallery = SegmentationIdentityGallery(
            device=appearance_device, enabled=appearance_fallback, **gallery_kwargs,
        )

    # -------------------------------------------------- to implement
    def _build_predictor(self):
        raise NotImplementedError

    # ------------------------------------------------- optional override
    def _init_state(self, predictor, frames: list[np.ndarray]):
        """SAM2 (and presumably SAM 3.1, same originating video code)
        does NOT accept a list of in-memory frames: it wants a path to
        an MP4 video or a folder of sequential JPEGs (verified on a real
        CUDA machine, see the module docstring). The current chunk is
        therefore written as `000000.jpg`, `000001.jpg`, ... in a
        temporary folder -- which stays alive until the chunk has been
        fully propagated: `run()` deletes it right after (not here,
        because this method returns before `propagate_in_video()` has
        read the files)."""
        self._current_chunk_tmpdir = tempfile.mkdtemp(prefix="chunked_video_predictor_")
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(self._current_chunk_tmpdir, f"{i:06d}.jpg"), frame)
        return predictor.init_state(self._current_chunk_tmpdir)

    def _cleanup_chunk_tmpdir(self) -> None:
        if self._current_chunk_tmpdir is not None:
            shutil.rmtree(self._current_chunk_tmpdir, ignore_errors=True)
            self._current_chunk_tmpdir = None

    def _add_box_prompt(self, predictor, state, *, frame_idx: int, obj_id: int, box: np.ndarray,
                         frame_shape: tuple[int, int] | None = None) -> None:
        """`frame_shape` is accepted for signature parity with
        `Sam31Tracker`'s override (which needs it to normalize a point
        prompt to SAM 3.1's real API -- see there for why) but unused
        here: SAM2's `add_new_points_or_box` takes a pixel-space box
        directly, no normalization needed."""
        predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id, box=box)

    def _seed_new_chunk(self, predictor, state, *, chunk_frames: list[np.ndarray],
                         chunk_index: int, frame_shape: tuple[int, int],
                         prev_anchor_polys: dict[int, np.ndarray],
                         allocator: GlobalIdAllocator,
                         prev_anchor_polys_history: dict[int, dict[int, np.ndarray]] | None = None,
                         identity_gallery: SegmentationIdentityGallery | None = None,
                         ) -> dict[int, np.ndarray]:
        """Decides who to follow in this chunk, REGISTERS the prompts
        with the predictor (side effect, not left to the caller: see
        `Sam31Tracker._seed_new_chunk()` for why -- in text-prompt mode
        a single call discovers AND registers people together, the two
        steps can't be separated as in the box case), and returns
        `{global_id: box}` for whoever is currently known.

        Default (used by `Sam2Tracker` and by `Sam31Tracker` in box
        mode): YOLO proposes the boxes, see the module docstring.
        `prev_anchor_polys_history` is accepted for signature parity
        with `Sam31Tracker`'s override (which DOES use it, via
        `chunking.reconcile_ids_windowed`) but ignored here: box mode
        never runs geometric reconciliation in the first place -- an
        already-known person's global id is reused directly as the SAM
        `obj_id`, no ambiguity to resolve. `identity_gallery`, however,
        IS used here: a YOLO detection that doesn't overlap anyone
        already seeded is first tried against the appearance gallery
        (see `identity_gallery.py`) before minting a brand-new id --
        catches the case of someone occluded across an entire chunk who
        would otherwise resurface under a new identity."""
        seed_boxes: dict[int, np.ndarray] = {}
        if chunk_index == 0:
            for box in self._detect_people(chunk_frames[0]):
                seed_boxes[allocator.next_id()] = box
        else:
            for global_id, poly in prev_anchor_polys.items():
                seed_boxes[global_id] = _polygon_to_box(poly)
            if self.reseed_new_people:
                for box in self._detect_people(chunk_frames[0]):
                    if not _overlaps_any(box, seed_boxes.values(), frame_shape, NEW_PERSON_IOU_THRESHOLD):
                        global_id = _resolve_new_person_id(
                            identity_gallery, allocator, chunk_frames[0], box,
                        )
                        seed_boxes[global_id] = box  # new (or re-identified) person
            # if reseed_new_people is False: no call to YOLO here,
            # SAM/SAM2 continues ONLY the already-known tracks (or finds
            # none left if everyone has left the frame) -- this is the
            # "pure SAM" condition for the comparison in
            # benchmark_backends.py.

        if self.max_people is not None and len(seed_boxes) > self.max_people:
            # hard cap, same logic as cap_by_confidence elsewhere in the
            # project: here we have no confidence to sort by, so we
            # simply keep discovery order (already-known people before
            # new ones, being inserted first into the dict).
            seed_boxes = dict(list(seed_boxes.items())[: self.max_people])

        for global_id, box in seed_boxes.items():
            self._add_box_prompt(predictor, state, frame_idx=0, obj_id=global_id, box=box,
                                  frame_shape=frame_shape)
        return seed_boxes

    def _propagate(self, predictor, state, *, start_frame_idx: int = 0,
                    max_frame_num_to_track: int | None = None
                    ) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        """Must return, for every frame local to the chunk, `(frame_idx,
        {obj_id: boolean_mask})`. The conversion to a boolean mask (from
        a torch/logit tensor to numpy bool, see `_to_boolean_mask()`)
        happens HERE, not in the caller, so the rest of `run()` can
        always assume the same format regardless of what the real
        library returns.

        `start_frame_idx`/`max_frame_num_to_track`: propagate only a
        SUB-window of the chunk (used by `redetect_every`, see the
        module docstring) instead of the whole chunk at once -- a
        signature present in the official SAM2/SAM3 notebooks for
        resuming propagation after adding new objects mid-video, but
        NEVER called on a real CUDA machine in this project so far: if
        the real predictor rejects it or behaves differently, it's the
        first thing to fix (along with `_add_box_prompt()` for the
        non-zero frame_idx used by re-detection)."""
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            state, start_frame_idx=start_frame_idx, max_frame_num_to_track=max_frame_num_to_track,
        ):
            yield frame_idx, {obj_id: _to_boolean_mask(mask) for obj_id, mask in zip(obj_ids, masks)}

    # --------------------------------------------------------------- run
    def run(self, source, stream: bool = True) -> Iterator[SegFrameResult]:
        total_frames, frame_shape = _probe_video(source)
        predictor = self._build_predictor()
        allocator = GlobalIdAllocator()
        prev_anchor_polys: dict[int, np.ndarray] = {}
        # Short trailing history of {local_frame: {global_id: poly}} from
        # the END of the previous chunk (see RECONCILE_HISTORY_FRAMES) --
        # extra evidence for Sam31Tracker's text-prompt reconciliation
        # (chunking.reconcile_ids_windowed), unused by box mode. Kept
        # separate from prev_anchor_polys (still the single frame used
        # for the direct obj_id reuse / consistency check below).
        prev_anchor_polys_history: dict[int, dict[int, np.ndarray]] = {}

        for chunk_index, (start, end) in enumerate(iter_chunk_ranges(total_frames, self.chunk_size, self.overlap)):
            chunk_start_time = time.time()
            chunk_frames = _read_frame_range(source, start, end)
            if not chunk_frames:
                break
            state = self._init_state(predictor, chunk_frames)

            # Decides who to follow AND registers the prompts with the
            # predictor (side effect of _seed_new_chunk, see its
            # docstring) -- default YOLO/box, overridden by
            # Sam31Tracker for SAM 3's text prompt when `text_prompt` is
            # set.
            seed_boxes = self._seed_new_chunk(
                predictor, state, chunk_frames=chunk_frames, chunk_index=chunk_index,
                frame_shape=frame_shape, prev_anchor_polys=prev_anchor_polys, allocator=allocator,
                prev_anchor_polys_history=prev_anchor_polys_history, identity_gallery=self._identity_gallery,
            )

            chunk_results: list[SegFrameResult] = []
            polys_by_local_frame: dict[int, dict[int, np.ndarray]] = {}
            # Last known box for each id (updated after every propagated
            # sub-window) -- used by periodic re-detection to decide
            # whether a YOLO detection is an ALREADY-known person (high
            # IoU with their last position) or a NEW one. Deliberately
            # NOT the original seed_boxes (which would go stale after
            # someone moves): see "Periodic re-detection" in the module
            # docstring.
            known_boxes: dict[int, np.ndarray] = dict(seed_boxes)

            try:
                # A single window as large as the chunk if
                # redetect_every isn't set (original, unchanged
                # behavior).
                window_size = self.redetect_every or len(chunk_frames)
                local_idx = 0
                while local_idx < len(chunk_frames):
                    window_end = min(local_idx + window_size, len(chunk_frames))

                    if not known_boxes:
                        # No person to follow in this window: neither an
                        # ongoing track nor a new one detected by YOLO --
                        # can happen with a video that opens on an empty
                        # room, or if YOLO misses the detection on that
                        # specific frame (lighting, pose, confidence
                        # threshold). SAM/SAM2's `propagate_in_video()`
                        # RAISES an error ("No points are provided;
                        # please add points first") if called with no
                        # prompt registered -- here empty frames are
                        # emitted for THIS window instead of blowing up
                        # the whole pipeline, and re-detection is still
                        # retried at the start of the next window (if
                        # redetect_every is set). If this appears for
                        # EVERY window of EVERY chunk, the problem is
                        # almost certainly upstream: check that
                        # `_detect_people()` actually finds someone in
                        # the frame.
                        print(f"[{type(self).__name__}] warning: no person to follow "
                              f"in local window [{local_idx},{window_end}) of chunk "
                              f"{chunk_index} (original frames [{start + local_idx},"
                              f"{start + window_end})) -- empty frames for this window.")
                        chunk_results.extend(
                            SegFrameResult(frame_index=start + i, frame=chunk_frames[i], people=[])
                            for i in range(local_idx, window_end)
                        )
                    else:
                        for local_out_idx, masks_by_id in self._propagate(
                            predictor, state, start_frame_idx=local_idx,
                            max_frame_num_to_track=window_end - local_idx,
                        ):
                            people = []
                            polys_this_frame: dict[int, np.ndarray] = {}
                            for obj_id, mask in masks_by_id.items():
                                poly = _mask_to_polygon(mask)
                                polys_this_frame[obj_id] = poly
                                box = _polygon_to_box(poly)
                                # SAM doesn't produce a detection
                                # confidence comparable to YOLO's: 1.0
                                # as an explicit placeholder, NEVER used
                                # for the max_people cap here (already
                                # applied above on the seeds) -- see
                                # cap_by_confidence in tracking_common.py
                                # for the YOLO case, where the
                                # confidence is instead real.
                                people.append((obj_id, box, poly, 1.0))
                            polys_by_local_frame[local_out_idx] = polys_this_frame
                            chunk_results.append(SegFrameResult(
                                frame_index=start + local_out_idx, frame=chunk_frames[local_out_idx],
                                people=people,
                            ))
                        # most recent known position for each still-tracked
                        # id (an empty polygon = SAM lost it: it exits
                        # known_boxes, re-detectable as "new" later)
                        last_polys = polys_by_local_frame.get(window_end - 1, {})
                        known_boxes = {
                            obj_id: _polygon_to_box(poly)
                            for obj_id, poly in last_polys.items() if poly.shape[0] > 0
                        }

                    local_idx = window_end
                    if local_idx >= len(chunk_frames):
                        break

                    if self.redetect_every and self.reseed_new_people:
                        # Periodic re-detection: proposes ONLY new
                        # people (low IoU with all known positions) --
                        # whoever is already tracked keeps propagating
                        # on its own, no need to reseed them.
                        for box in self._detect_people(chunk_frames[local_idx]):
                            if not _overlaps_any(box, known_boxes.values(), frame_shape, NEW_PERSON_IOU_THRESHOLD):
                                new_id = _resolve_new_person_id(
                                    self._identity_gallery, allocator, chunk_frames[local_idx], box,
                                )
                                self._add_box_prompt(predictor, state, frame_idx=local_idx, obj_id=new_id, box=box,
                                                      frame_shape=frame_shape)
                                known_boxes[new_id] = box
            finally:
                # the chunk's temporary JPEG folder (see _init_state())
                # is no longer needed once propagate_in_video() has been
                # fully consumed (or was never called, see above) --
                # cleaned up even if inference raises an exception
                # mid-chunk, to avoid leaving orphaned temp folders on a
                # long video with many chunks.
                self._cleanup_chunk_tmpdir()

            # consistency check (log only, see module docstring)
            anchor_polys = polys_by_local_frame.get(0, {})
            for global_id, seeded_poly in prev_anchor_polys.items():
                produced_poly = anchor_polys.get(global_id)
                if produced_poly is None:
                    print(f"[{type(self).__name__}] warning: id {global_id} not found "
                          f"at the anchor frame of chunk {chunk_index}")
                    continue
                iou = polygon_iou(seeded_poly, produced_poly, frame_shape)
                if iou < self.iou_threshold:
                    print(f"[{type(self).__name__}] warning: id {global_id} low IoU "
                          f"({iou:.2f}) at the anchor frame of chunk {chunk_index} "
                          f"-- possible identity loss/swap")

            if self.chunk_store_dir:
                save_chunk(chunk_results, self.chunk_store_dir, chunk_index)

            # anchor frame for the NEXT chunk: the last frame of this
            # chunk that falls within the overlap window with the next one
            next_anchor_local = len(chunk_frames) - self.overlap
            new_prev_anchor_polys = polys_by_local_frame.get(next_anchor_local, {})

            # Appearance gallery bookkeeping (see identity_gallery.py):
            # refresh the embedding of everyone still active at the new
            # anchor, mark as "lost" anyone who WAS known coming into
            # this chunk but isn't anymore (didn't survive
            # reconciliation, or SAM/YOLO lost them mid-chunk), and
            # forget identities lost for too long. A no-op loop if
            # appearance_fallback=False or torchreid isn't installed
            # (SegmentationIdentityGallery.enabled is False in that
            # case -- observe()/mark_lost() themselves are no-ops).
            history_start = max(0, next_anchor_local - RECONCILE_HISTORY_FRAMES + 1)
            prev_anchor_polys_history = {
                local_idx: polys_by_local_frame[local_idx]
                for local_idx in range(history_start, next_anchor_local + 1)
                if local_idx in polys_by_local_frame
            }
            if self._identity_gallery.enabled and 0 <= next_anchor_local < len(chunk_frames):
                anchor_frame_img = chunk_frames[next_anchor_local]
                for global_id, poly in new_prev_anchor_polys.items():
                    self._identity_gallery.observe(global_id, anchor_frame_img, _polygon_to_box(poly), poly=poly)
                for global_id in prev_anchor_polys:
                    if global_id not in new_prev_anchor_polys:
                        self._identity_gallery.mark_lost(global_id, chunk_index)
                self._identity_gallery.forget_stale(chunk_index)

            prev_anchor_polys = new_prev_anchor_polys

            # avoids re-emitting the overlap window's frames twice
            # (already emitted by the previous chunk, except for the
            # first chunk)
            skip = 0 if chunk_index == 0 else self.overlap
            for r in chunk_results[skip:]:
                yield r

            n_ids = len({obj_id for polys in polys_by_local_frame.values() for obj_id in polys})
            elapsed = time.time() - chunk_start_time
            print(f"[{type(self).__name__}] chunk {chunk_index} done -- frames "
                  f"[{start},{end}) ({end - start} frames), {n_ids} id(s) tracked, "
                  f"{elapsed:.1f}s")

    # --------------------------------------------------------------- YOLO
    def _detect_people(self, frame: np.ndarray) -> list[np.ndarray]:
        """Boxes (x1,y1,x2,y2) of people detected by YOLO on a single
        frame -- used ONLY to propose initial prompts to SAM (not to
        track), see the module docstring for why. Delayed import: same
        reason as `SegTracker`.

        Always logs what it finds (or doesn't find): useful to tell
        "the frame really has no people" apart from "YOLO has a
        problem" -- see the SAMURAI debugging session where the
        'no person to follow' warning appeared even with people clearly
        visible in the frame (later resolved: the real problem was
        SAMURAI's multi-object crash, not YOLO, see the module
        docstring)."""
        if self._detector is None:
            from ultralytics import YOLO
            self._detector = YOLO(resolve_yolo_weights(self.prompt_model))
            print(f"[{type(self).__name__}] YOLO proposer loaded: model={self.prompt_model!r} "
                  f"device={self.device!r} conf_threshold={self.conf_threshold}")
        result = self._detector.predict(
            source=frame, device=self.device, conf=self.conf_threshold,
            classes=[COCO_PERSON_CLASS_ID], verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            # "result.boxes is None" and "len(...) == 0 but not None"
            # are both "zero detections", but distinguishing them in
            # the log helps whoever is debugging: if this line appears
            # even on a frame with people clearly visible, suspicion
            # shifts from "the frame is empty" to "this detector's
            # model/threshold/device has a problem" (e.g. compare with
            # --backend yolo on the same video: if YOLO finds the
            # people THERE, the problem is specific to this
            # .predict() call, not the model).
            print(f"[{type(self).__name__}] YOLO: 0 people detected on this frame "
                  f"(conf_threshold={self.conf_threshold})")
            return []
        confs = [round(c, 2) for c in result.boxes.conf.cpu().numpy().tolist()]
        print(f"[{type(self).__name__}] YOLO: {len(confs)} person/people detected on this frame "
              f"(confidences: {confs})")
        return [box for box in result.boxes.xyxy.cpu().numpy()]


def _overlaps_any(box: np.ndarray, existing_boxes, frame_shape: tuple[int, int], threshold: float) -> bool:
    """True if `box` has IoU >= `threshold` with at least one of the
    `existing_boxes` -- boxes, not polygons, so a rectangular polygon is
    built on the fly to reuse `polygon_iou`."""
    box_poly = _box_to_polygon(box)
    for other in existing_boxes:
        other_poly = _box_to_polygon(other)
        if polygon_iou(box_poly, other_poly, frame_shape) >= threshold:
            return True
    return False


def _resolve_new_person_id(identity_gallery: SegmentationIdentityGallery | None,
                            allocator: GlobalIdAllocator, frame_bgr: np.ndarray,
                            box: np.ndarray, poly: np.ndarray | None = None) -> int:
    """Single decision point shared by every place in this module (and
    in `sam31_estimation.py`) that's about to mint a brand-new global id
    for a detection with no geometric match: tries the appearance
    gallery first (see `identity_gallery.py`) -- a confident match means
    this is someone who was already known and got lost (occlusion,
    left and re-entered frame), not a genuinely new person. Falls back
    to `allocator.next_id()` whenever the gallery is disabled/absent or
    finds nothing (never raises: `SegmentationIdentityGallery` itself is
    a documented no-op when torch/torchreid aren't installed)."""
    if identity_gallery is not None:
        matched = identity_gallery.match_or_none(frame_bgr, box, poly=poly)
        if matched is not None:
            identity_gallery.revive(matched)
            print(f"[identity_gallery] re-identified global id {matched} by appearance "
                  f"(no geometric match this chunk) -- reused instead of a new id")
            return matched
    return allocator.next_id()
