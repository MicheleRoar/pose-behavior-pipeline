"""
psifx_eval/overlap_strategy.py
=================================
Pure algorithmic logic for cross-chunk strategies #2 ("Overlapping
chunks -- share 50-100 frames between consecutive chunks") and #3
("Overlap + mask IoU -- associate SAM3 IDs across chunks using masks
observed over multiple shared frames") from Michele/Loic's brief.
Deliberately separated from `overlap_tracking.py` (the thin class that
actually calls real SAM3 through psifx's `Sam3TrackingTool`) so this
module can be fully unit-tested with plain numpy arrays -- no psifx, no
GPU, no real video -- same split as `id_metrics.py` vs
`run_baseline_vs_oracle.py`.

Vanilla psifx (`Sam3TrackingTool._map_chunk_object_ids`, see
`run_baseline_vs_oracle.py`'s docstring) links chunk N+1 to chunk N
using exactly ONE frame from each side (the single last frame-with-
objects of chunk N vs the single first frame-with-objects of chunk
N+1), greedy assignment. That's fragile exactly where the brief says it
would be: a person near a border, or occluded for a moment, at THAT ONE
comparison frame, and their identity is lost even though they were
clearly visible one frame earlier or later.

The fix implemented here: chunks OVERLAP by `overlap` frames (chunk
N+1 starts `overlap` frames before chunk N ends, so both chunks
independently re-segment that shared span), and cross-chunk linking
uses the MEAN IoU across every shared frame where both a previous-chunk
identity and a current-chunk identity are present (not just one frame),
matched OPTIMALLY (Hungarian, not greedy). A person who's clipped or
occluded on the single frame vanilla psifx happens to compare can still
be correctly linked via any of the other `overlap - 1` shared frames.
"""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

import numpy as np
from scipy.optimize import linear_sum_assignment

T = TypeVar("T")

# Same dict shape psifx's own `Sam3TrackingTool._segment_chunk` produces:
# {local_frame_idx: {"object_ids": [int, ...], "masks": [np.ndarray, ...]}}
ChunkOutputs = dict


def chunk_with_overlap(frames: Iterable[T], chunk_size: int, overlap: int) -> Iterator[tuple[int, list[T]]]:
    """Generic sliding-window chunker: yields `(start_frame, chunk)`
    where consecutive chunks share their last/first `overlap` items.
    Works on ANY iterable of frame-like objects (PIL Images in the real
    pipeline, plain ints in tests) -- single pass, no seeking, carries
    the overlap tail forward in memory instead of re-reading the
    source (mirrors psifx's own `Sam3TrackingTool._iter_video_chunks`
    streaming style, see there).

    The final chunk is yielded even if shorter than `chunk_size`, as
    long as it contains at least one frame beyond the carried-over
    overlap (a chunk that's ONLY the overlap tail with nothing new
    would mean the previous chunk already covered the whole video)."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if not (0 <= overlap < chunk_size):
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap} (chunk_size={chunk_size}).")

    stride = chunk_size - overlap
    chunk: list[T] = []
    start_frame = 0
    for frame in frames:
        chunk.append(frame)
        if len(chunk) >= chunk_size:
            yield start_frame, chunk
            carry_over = chunk[-overlap:] if overlap > 0 else []
            start_frame += stride
            chunk = list(carry_over)

    if len(chunk) > overlap:
        yield start_frame, chunk


def map_first_chunk_ids(
    chunk_outputs: ChunkOutputs,
    next_global_id: int,
    max_num_objects: int | None,
) -> tuple[dict[int, int], int]:
    """The very first chunk has nothing to link to -- every local id
    becomes a fresh global id, in first-appearance order (same policy
    as vanilla psifx's `_map_chunk_object_ids` for a first chunk, since
    `prev_last_global_masks` starts empty there too)."""
    id_mapping: dict[int, int] = {}
    for frame_idx in sorted(chunk_outputs.keys()):
        for obj_id in chunk_outputs[frame_idx]["object_ids"]:
            if obj_id not in id_mapping:
                if max_num_objects is not None and next_global_id >= max_num_objects:
                    continue
                id_mapping[obj_id] = next_global_id
                next_global_id += 1
    return id_mapping, next_global_id


def _mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Same formula as psifx's own `Sam3TrackingTool._compute_mask_iou`."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def extract_overlap_window(
    chunk_outputs: ChunkOutputs, *, local_indices: range
) -> dict[int, dict[int, np.ndarray]]:
    """Reshapes `chunk_outputs` (frame-indexed) into a `{local_frame_idx
    (relative to `local_indices.start`): {local_id: mask}}` view of just
    the requested frame range -- used to pull out either the TAIL of the
    previous chunk (its last `overlap` frames) or the HEAD of the
    current chunk (its first `overlap` frames), which correspond to the
    exact same physical video frames."""
    window: dict[int, dict[int, np.ndarray]] = {}
    for local_idx in local_indices:
        frame_out = chunk_outputs.get(local_idx, {"object_ids": [], "masks": []})
        window[local_idx - local_indices.start] = dict(
            zip(frame_out["object_ids"], frame_out["masks"])
        )
    return window


def map_chunk_ids_via_overlap(
    chunk_outputs: ChunkOutputs,
    overlap: int,
    prev_overlap_window: dict[int, dict[int, np.ndarray]],
    iou_threshold: float,
    next_global_id: int,
    max_num_objects: int | None,
) -> tuple[dict[int, int], int]:
    """Links this chunk's local ids to the PREVIOUS chunk's global ids
    using every shared overlap frame (not just one), matched optimally
    by mean IoU -- see module docstring. `prev_overlap_window` is the
    previous chunk's tail, already mapped to GLOBAL ids (see
    `overlap_tracking.py`'s caller, which stashes this after writing
    each chunk); `chunk_outputs` is this (new) chunk's RAW output
    (local ids, not yet mapped to anything)."""
    curr_overlap_window = extract_overlap_window(chunk_outputs, local_indices=range(overlap))

    prev_ids = sorted({pid for frame in prev_overlap_window.values() for pid in frame.keys()})
    curr_ids = sorted({cid for frame in curr_overlap_window.values() for cid in frame.keys()})

    id_mapping: dict[int, int] = {}
    if prev_ids and curr_ids:
        iou_sum = np.zeros((len(prev_ids), len(curr_ids)), dtype=float)
        co_occurrence = np.zeros((len(prev_ids), len(curr_ids)), dtype=float)
        for local_idx in range(overlap):
            prev_frame = prev_overlap_window.get(local_idx, {})
            curr_frame = curr_overlap_window.get(local_idx, {})
            for i, pid in enumerate(prev_ids):
                if pid not in prev_frame:
                    continue
                for j, cid in enumerate(curr_ids):
                    if cid not in curr_frame:
                        continue
                    iou_sum[i, j] += _mask_iou(prev_frame[pid], curr_frame[cid])
                    co_occurrence[i, j] += 1

        mean_iou = np.divide(
            iou_sum, co_occurrence,
            out=np.zeros_like(iou_sum),
            where=co_occurrence > 0,
        )
        cost = 1.0 - mean_iou
        row_idx, col_idx = linear_sum_assignment(cost)
        for i, j in zip(row_idx, col_idx):
            if co_occurrence[i, j] == 0:
                continue  # this pair never actually co-occurred in any shared frame
            if mean_iou[i, j] >= iou_threshold:
                id_mapping[curr_ids[j]] = prev_ids[i]

    # Anything not linked above (genuinely new people, or people who
    # only appear later in this chunk, past the overlap window) get a
    # fresh global id -- same cap policy as vanilla psifx.
    for frame_idx in sorted(chunk_outputs.keys()):
        for obj_id in chunk_outputs[frame_idx]["object_ids"]:
            if obj_id not in id_mapping:
                if max_num_objects is not None and next_global_id >= max_num_objects:
                    continue
                id_mapping[obj_id] = next_global_id
                next_global_id += 1

    return id_mapping, next_global_id


def stash_overlap_tail(
    chunk_outputs: ChunkOutputs, id_mapping: dict[int, int], chunk_length: int, overlap: int
) -> dict[int, dict[int, np.ndarray]]:
    """After finishing a chunk, extracts its LAST `overlap` frames
    (already resolved to GLOBAL ids via `id_mapping`) so the NEXT
    chunk's `map_chunk_ids_via_overlap` call has something to compare
    against. Returns `{}` if `overlap == 0` (overlap disabled -- the
    caller should then fall back to vanilla single-frame linking, not
    call this function)."""
    if overlap == 0:
        return {}
    tail_start = chunk_length - overlap
    window = extract_overlap_window(chunk_outputs, local_indices=range(tail_start, chunk_length))
    return {
        local_idx: {id_mapping[local_id]: mask for local_id, mask in frame.items() if local_id in id_mapping}
        for local_idx, frame in window.items()
    }


def local_frames_to_write(chunk_length: int, skip_local_frames: int) -> range:
    """Which of this chunk's local frame indices are genuinely NEW and
    should be written to the output MaskDir -- the first
    `skip_local_frames` (the overlap carried over from the previous
    chunk) were already written as that chunk's tail, and must not be
    duplicated on disk. `skip_local_frames` should be 0 for the first
    chunk (nothing carried over yet) and `overlap` for every chunk
    after that."""
    return range(skip_local_frames, chunk_length)
