"""
chunking.py
============
Logic for splitting video into overlapping chunks (windows) and
reconciling IDs between one chunk and the next -- independent of
SAM/SAM2, used by `sam_backend.py`. No heavy dependency (only numpy/cv2,
already required by the rest of the project): testable with synthetic
masks, without needing a GPU or the SAM weights installed (see
demo/chunking_check.py).

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

Known limitation of this first version: reconciliation only uses
geometric IoU on the anchor frame (the last overlap frame). Works well
if people haven't swapped positions within the overlap window; in
crowded scenes with occlusions right at the chunk boundary it can get
it wrong -- natural extension, if needed: add an appearance similarity
score (color/texture, as `segmentation/seg_reid.py` already does for
ByteTrack) alongside the IoU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np


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
    assign it a global id never used before (see `GlobalIdAllocator`).

    Greedy matching by decreasing IoU: each id (old or new) is used at
    most once, so two nearby people are never both matched to the same
    id."""
    candidates: list[tuple[float, int, int]] = []
    for old_id, old_poly in prev_polys_at_anchor.items():
        for new_id, new_poly in new_polys_at_anchor.items():
            iou = polygon_iou(old_poly, new_poly, frame_shape)
            if iou > 0.0:
                candidates.append((iou, old_id, new_id))
    candidates.sort(key=lambda c: c[0], reverse=True)

    mapping: dict[int, int] = {}
    used_old: set[int] = set()
    used_new: set[int] = set()
    for iou, old_id, new_id in candidates:
        if iou < iou_threshold:
            break  # sorted by decreasing iou: everything else is below threshold
        if old_id in used_old or new_id in used_new:
            continue
        mapping[new_id] = old_id
        used_old.add(old_id)
        used_new.add(new_id)
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
