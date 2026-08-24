"""
chunking.py
============
Logic for splitting video into overlapping chunks (windows) and
reconciling IDs between one chunk and the next -- independent of
SAM/SAM2, used by `sam_backend.py`. Testable with synthetic masks, no
GPU/SAM weights needed (see tests/chunking_check.py).

Why chunking is necessary: the SAM 3.1/SAM2 video API is stateful --
`init_state(video)` loads all passed frames into memory, a memory
problem on a multi-minute video -- so only a `chunk_size`-frame window
is passed at a time. Each chunk then starts "without memory" of the
previous one: SAM's within-chunk ids are local, not guaranteed stable
across chunks. Hence the overlap (`overlap` shared frames between
consecutive chunks) and the geometric reconciliation below: masks from
both chunks are compared on the shared frames to reconstruct a stable
global id.

`reconcile_ids()` matches old/new ids via the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`, maximizing total IoU across
the whole set), not greedily -- greedy (highest-IoU pair first) can
steal one person's id for another's continuation when two known people
are close together near a boundary, since the single best pair isn't
always part of the globally-best assignment. See
`tests/chunking_check.py::part5b_hungarian_beats_greedy_on_close_by_people`
for a worked example. This doesn't resolve every ambiguous case (two
people who genuinely swap positions can still fool any purely-geometric
matcher); it only removes the additional, avoidable error of a
sum-suboptimal assignment. `reconcile_ids_windowed()` (multi-frame
evidence instead of trusting a single anchor frame) and
`segmentation/identity_gallery.py` (appearance-based fallback when
geometry finds no match) build further mitigations on top.

Known limitation: reconciliation only compares polygons in a short
window around the chunk boundary, not a person's whole history --
someone occluded for an entire chunk can't be recovered by geometry
alone; that's what the appearance gallery is for.

Motion compensation (`estimate_velocities()`, used by
`reconcile_ids_windowed()`): per-id velocity is estimated from trailing
history (plain first-difference, no Kalman filter -- see that function's
docstring for why) and used to translate an older polygon to its
predicted position before computing IoU, always compared alongside the
un-shifted (static) IoU and keeping the max of the two -- so a
noisy/bad velocity estimate can only ever be a no-op, never worse than
static-only matching. Motivated by the DanceTrack benchmark's oracle
analysis: a constant-velocity model beats static-position IoU by a wide
margin once people move fast between frames. See
`tests/chunking_check.py::part5e_reconcile_ids_windowed_motion_compensation_recovers_fast_linear_motion`
for a case static IoU alone misses.
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


def _centroid(poly: np.ndarray) -> np.ndarray | None:
    """Mean of the polygon's points, or `None` if degenerate (< 3
    points) -- same "no made-up signal" convention as `polygon_iou`."""
    if poly.shape[0] < 3:
        return None
    return poly.mean(axis=0)


def estimate_velocities(polys_by_frame: dict[int, dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """Per-id centroid velocity (dx, dy PER FRAME STEP), estimated from
    `polys_by_frame` (`{frame_offset: {id: polygon}}`, same shape as
    `reconcile_ids_windowed`'s `prev_polys_by_frame`) -- used to predict
    where a person should be by the comparison frame instead of
    assuming they didn't move, see the module docstring's 2026-08
    motion-compensation note for why.

    Deliberately a plain first-difference between consecutive
    observations (median across steps if there are more than two), NOT
    a Kalman filter or anything fitted -- the DanceTrack oracle finding
    this is based on already showed a SIMPLE motion estimate closes
    most of the gap versus static IoU; a full filter is a natural next
    step if this turns out not to be enough in practice, not a
    prerequisite to try the idea at all.

    Ids seen in fewer than 2 frames (nothing to estimate a velocity
    from) are simply absent from the returned dict -- callers must
    treat a missing id as "no motion known" and fall back to comparing
    the raw (un-shifted) polygon, not as an error."""
    tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame_offset, polys in polys_by_frame.items():
        for obj_id, poly in polys.items():
            centroid = _centroid(poly)
            if centroid is not None:
                tracks.setdefault(obj_id, []).append((frame_offset, centroid))

    velocities: dict[int, np.ndarray] = {}
    for obj_id, observations in tracks.items():
        if len(observations) < 2:
            continue
        observations.sort(key=lambda o: o[0])
        steps = []
        for (f_prev, c_prev), (f_next, c_next) in zip(observations, observations[1:]):
            gap = f_next - f_prev
            if gap != 0:
                steps.append((c_next - c_prev) / gap)
        if steps:
            velocities[obj_id] = np.median(np.stack(steps), axis=0)
    return velocities


def _shift_polygon(poly: np.ndarray, velocity: np.ndarray, num_frames: float) -> np.ndarray:
    """Translates `poly` by `velocity * num_frames` -- constant-velocity
    extrapolation forward (or backward) in time."""
    return poly + velocity * num_frames


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
    as `reconcile_ids`).

    Motion compensation (2026-08 addition, see module docstring): if an
    old id has at least two observations in `prev_polys_by_frame`, its
    constant-velocity estimate (`estimate_velocities`) is used to
    predict where its polygon would be at each new frame offset before
    comparing IoU, and that shifted-IoU is combined with the static
    (un-shifted) IoU via `max()`. This can only ever RAISE a pair's
    score, never lower it -- a fast-moving person whose static overlap
    falls below `iou_threshold` still gets a chance to match via their
    predicted position, while a stationary person is unaffected
    (shifted polygon == original polygon when velocity is ~0)."""
    old_ids: set[int] = set()
    new_ids: set[int] = set()
    for polys in prev_polys_by_frame.values():
        old_ids.update(polys.keys())
    for polys in new_polys_by_frame.values():
        new_ids.update(polys.keys())
    old_ids_list, new_ids_list = sorted(old_ids), sorted(new_ids)
    if not old_ids_list or not new_ids_list:
        return {}

    velocities = estimate_velocities(prev_polys_by_frame)

    old_pos = {old_id: i for i, old_id in enumerate(old_ids_list)}
    new_pos = {new_id: j for j, new_id in enumerate(new_ids_list)}
    iou_matrix = np.zeros((len(old_ids_list), len(new_ids_list)))
    for prev_frame_offset, prev_polys in prev_polys_by_frame.items():
        for new_frame_offset, new_polys in new_polys_by_frame.items():
            for old_id, old_poly in prev_polys.items():
                i = old_pos[old_id]
                velocity = velocities.get(old_id)
                for new_id, new_poly in new_polys.items():
                    j = new_pos[new_id]
                    iou = polygon_iou(old_poly, new_poly, frame_shape)
                    if velocity is not None:
                        num_frames = new_frame_offset - prev_frame_offset
                        shifted = _shift_polygon(old_poly, velocity, num_frames)
                        iou = max(iou, polygon_iou(shifted, new_poly, frame_shape))
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
