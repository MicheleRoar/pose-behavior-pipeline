"""
chunking.py
============
Logic for splitting video into overlapping chunks (windows) and
reconciling IDs between one chunk and the next -- independent of
SAM/SAM2, used by `sam_backend.py`. No heavy dependency (only numpy/cv2,
already required by the rest of the project): testable with synthetic
masks, without needing a GPU or the SAM weights installed (see
tests/chunking_check.py).

Why chunking is necessary (not just an optimization)
------------------------------------------------------------------
The SAM 3.1 and SAM2 video API is stateful: `init_state(video)` loads
the pixels of ALL passed frames into memory, even before starting to
propagate masks. On a video several minutes long this isn't just slow,
it's a memory problem (VRAM/RAM) -- so the whole video is never passed
to `init_state()`, a window of `chunk_size` frames is passed at a time.
The price to pay is that each chunk starts "without memory" of the
previous chunk: the IDs SAM assigns within a chunk are local to that
chunk, there's no guarantee that id 1 of chunk 2 is the same person as
id 1 of chunk 1. Hence the overlap (`overlap` frames shared between one
chunk and the next) and the geometric reconciliation below: the masks
produced by the two chunks are compared on the SAME frames (the shared
ones) and a stable global id is reconstructed.

2026-08 revision (dancing-tracks test, reported by Michele): the
original version matched old/new ids GREEDILY (highest IoU pair first,
then the next-highest among what's left, ...). Greedy is only correct
when the single best pair is always part of SOME globally-optimal
assignment -- not guaranteed in general. Concrete failure observed:
two already-known people close together (A, B) near a chunk boundary;
a new detection happens to overlap A's OLD polygon slightly more than
B's own (moved) polygon does. Greedy grabs that pair first (highest
single IoU), "stealing" A's global id for what's actually B's
continuation -- B, now unmatched, gets a brand-new id, even though a
strictly better assignment existed (matching each new detection to
its TRUE previous id would have covered both people, with a higher
TOTAL iou). `reconcile_ids()` below now solves the assignment
GLOBALLY with the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`, maximizing the sum of IoU
over the whole set at once) instead of greedily -- see
`tests/chunking_check.py::part5b_hungarian_beats_greedy_on_close_by_people`
for a worked-out reproduction of the old bug and the fixed assignment.
This does NOT solve every ambiguous case (two people who genuinely
crossed paths, ending up with SAM's discovered geometry more similar
to swapped identities than to the true ones -- no purely-geometric
algorithm, greedy or optimal, can resolve that from IoU alone): it
only removes the additional, avoidable error of picking a
sum-suboptimal assignment. See `reconcile_ids_windowed()` and
`segmentation/identity_gallery.py` for the two further mitigations
(multi-frame evidence instead of trusting a single anchor frame, and
an appearance-based fallback for detections that still find no
geometric match) built on top of this.

Known limitation still present: reconciliation only compares polygons
across a SHORT window around the chunk boundary, not the person's
whole history -- someone who's been occluded for an entire chunk (no
polygon anywhere in the window) can't be recovered by geometry no
matter how good the matcher is; that's what the appearance gallery
(`identity_gallery.py`) is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def iter_chunk_ranges(total_frames: int, chunk_size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Generates (start, end) pairs [end exclusive] covering `[0, total_frames)`
    with an overlap of `overlap` frames between one chunk and the next.
    The last chunk may be shorter than `chunk_size` (never extends past
    `total_frames`). Raises `ValueError` if `chunk_size <= overlap`
    (otherwise it would never advance, infinite loop)."""
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")
    if total_frames <= 0:
        return
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        yield start, end
        if end >= total_frames:
            break
        start = end - overlap


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray, frame_shape: tuple[int, int]) -> float:
    """IoU (intersection over union) between two mask polygons, computed by
    rasterizing them onto a grid the size of the frame (`frame_shape`
    = (height, width)) and comparing the resulting binary masks --
    correct even for non-convex polygons, unlike an IoU computed on
    bounding boxes alone. Returns 0.0 if either polygon has fewer than
    3 points (degenerate/absent)."""
    if poly_a.shape[0] < 3 or poly_b.shape[0] < 3:
        return 0.0
    mask_a = np.zeros(frame_shape, dtype=np.uint8)
    mask_b = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillPoly(mask_a, [poly_a.astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [poly_b.astype(np.int32)], 1)
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union > 0 else 0.0


def reconcile_ids(
    prev_polys_at_anchor: dict[int, np.ndarray],
    new_polys_at_anchor: dict[int, np.ndarray],
    frame_shape: tuple[int, int],
    iou_threshold: float = 0.3,
) -> dict[int, int]:
    """Compares the masks of the previous chunk (`prev_polys_at_anchor`,
    keys = GLOBAL ids already assigned) with those of the new chunk
    (`new_polys_at_anchor`, keys = LOCAL ids assigned by SAM within this
    chunk) on the same anchor frame (a frame both chunks produced, within
    the overlap window).

    Returns a `{local_id: global_id}` dict only for the local ids that
    found a match above `iou_threshold`. A local id absent from the dict
    is a "new" person for the caller (entered the frame during the
    chunk, or the match was too uncertain to be sure) -- the caller will
    assign it a global id never used before (see `GlobalIdAllocator`),
    or try the appearance-based fallback in `identity_gallery.py`.

    Matching via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`),
    maximizing the TOTAL IoU across the whole assignment at once rather
    than greedily taking the single best pair first -- see the module
    docstring's 2026-08 revision note for the concrete bug this fixes.
    Each id (old or new) is used at most once, so two nearby people are
    never both matched to the same id."""
    old_ids = list(prev_polys_at_anchor.keys())
    new_ids = list(new_polys_at_anchor.keys())
    if not old_ids or not new_ids:
        return {}

    iou_matrix = np.zeros((len(old_ids), len(new_ids)))
    for i, old_id in enumerate(old_ids):
        old_poly = prev_polys_at_anchor[old_id]
        for j, new_id in enumerate(new_ids):
            iou_matrix[i, j] = polygon_iou(old_poly, new_polys_at_anchor[new_id], frame_shape)

    # linear_sum_assignment MINIMIZES total cost -- maximizing total IoU
    # is the same problem with cost = 1 - iou. Works fine on a
    # rectangular matrix too (more old ids than new, or vice versa):
    # returns min(len(old_ids), len(new_ids)) pairs, exactly the correct
    # "everyone who CAN be matched, is" semantics needed here.
    row_idx, col_idx = linear_sum_assignment(1.0 - iou_matrix)

    mapping: dict[int, int] = {}
    for r, c in zip(row_idx, col_idx):
        iou = iou_matrix[r, c]
        if iou >= iou_threshold:
            mapping[new_ids[c]] = old_ids[r]
    return mapping


def reconcile_ids_windowed(
    prev_polys_by_frame: dict[int, dict[int, np.ndarray]],
    new_polys_by_frame: dict[int, dict[int, np.ndarray]],
    frame_shape: tuple[int, int],
    iou_threshold: float = 0.3,
) -> dict[int, int]:
    """Like `reconcile_ids`, but instead of trusting a single anchor
    frame on each side, aggregates evidence across several frames
    (`prev_polys_by_frame`/`new_polys_by_frame`, each `{frame_key:
    {id: polygon}}` -- the frame keys are caller-defined and don't need
    to refer to the SAME real video frame on the two sides, see below)
    before running the Hungarian assignment once, on the aggregated
    matrix.

    Motivation (2026-08, dancing-tracks test): a single degenerate frame
    right at the chunk boundary (motion blur, partial occlusion, a
    contour-finding artifact on a barely-visible person) can make
    `reconcile_ids` see a spuriously low IoU for the person who is
    ACTUALLY still there, sending them to a brand-new id even though
    they were clearly identifiable a few frames earlier/later. Giving
    the matcher more than one frame to look at removes that single
    point of failure.

    Aggregation: for each (old_id, new_id) pair, the score is the MAX
    IoU observed over every (prev_frame, new_frame) combination where
    BOTH ids have a polygon -- deliberately MAX, not mean: the two
    sides are typically NOT the same real video frame (e.g. a short
    trailing history from the end of the previous chunk vs. a single
    discovery frame at the start of the new one, see
    `sam31_estimation.py`), so there's no reason to expect every
    combination to agree with each other -- only ONE frame pair with
    good geometric agreement is needed to trust the match. A pair with
    no frame in common where both ids appear naturally scores 0.0 (and
    is therefore excluded once compared against `iou_threshold`, same
    as `reconcile_ids`)."""
    old_ids: set[int] = set()
    new_ids: set[int] = set()
    for polys in prev_polys_by_frame.values():
        old_ids.update(polys.keys())
    for polys in new_polys_by_frame.values():
        new_ids.update(polys.keys())
    old_ids_list, new_ids_list = sorted(old_ids), sorted(new_ids)
    if not old_ids_list or not new_ids_list:
        return {}

    old_pos = {old_id: i for i, old_id in enumerate(old_ids_list)}
    new_pos = {new_id: j for j, new_id in enumerate(new_ids_list)}
    iou_matrix = np.zeros((len(old_ids_list), len(new_ids_list)))
    for prev_polys in prev_polys_by_frame.values():
        for new_polys in new_polys_by_frame.values():
            for old_id, old_poly in prev_polys.items():
                i = old_pos[old_id]
                for new_id, new_poly in new_polys.items():
                    j = new_pos[new_id]
                    iou = polygon_iou(old_poly, new_poly, frame_shape)
                    if iou > iou_matrix[i, j]:
                        iou_matrix[i, j] = iou

    row_idx, col_idx = linear_sum_assignment(1.0 - iou_matrix)
    mapping: dict[int, int] = {}
    for r, c in zip(row_idx, col_idx):
        iou = iou_matrix[r, c]
        if iou >= iou_threshold:
            mapping[new_ids_list[c]] = old_ids_list[r]
    return mapping


@dataclass
class GlobalIdAllocator:
    """Distributes progressive global ids, shared by all chunks of the same
    session. `next_id()` always returns an integer never returned before
    -- used for local ids that `reconcile_ids()` failed to match to any
    already-known id."""
    _next: int = field(default=1)

    def next_id(self) -> int:
        value = self._next
        self._next += 1
        return value
