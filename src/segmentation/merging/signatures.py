"""
segmentation/merging/signatures.py
===================================
Builds an appearance signature (OSNet embedding + hue-color histogram)
for a track or a group of tracks, by sampling a handful of "clean"
frames (single connected component, not touching the frame border --
see `mask_utils`). Used by `reappearance_merge.py` to compare tracks.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from segmentation.merging.mask_io import DEFAULT_MASK_THRESHOLD, _read_mask_frame
from segmentation.merging.mask_utils import (
    _histogram_similarity, _mask_hue_histogram, _mask_to_polygon_single_component,
    _polygon_to_box, _touches_border,
)
from pose.appearance_embedding import OSNetEmbedder, embedding_similarity

# absolute cap on how far _search_nearby_signal looks on either side of
# an evenly-spaced target frame before giving up on that sampling slot.
MAX_POOLED_SEARCH_RADIUS = 200


def _frame_signal(
    cap: cv2.VideoCapture,
    mask_frame: np.ndarray,
    frame_idx: int,
    embedder: OSNetEmbedder | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """OSNet embedding + hue histogram for ONE frame, or `(None, None)`
    if the frame is empty or its mask isn't a single, non-border-touching
    component (see `mask_utils`)."""
    if not mask_frame.any():
        return None, None
    poly = _mask_to_polygon_single_component(mask_frame)
    if poly.shape[0] < 3:
        return None, None
    height, width = mask_frame.shape[:2]
    if _touches_border(poly, height, width):
        return None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None, None
    box = _polygon_to_box(poly)
    embedding = None
    if embedder is not None:
        embedding = embedder.embed(frame, box, poly=poly)
    histogram = _mask_hue_histogram(frame, poly)
    return embedding, histogram


def _finalize_signature(
    embeddings: list[np.ndarray],
    histograms: list[np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Averages collected per-frame embeddings/histograms into one
    signature (re-normalized after averaging). `None` for whichever
    signal had no usable frames at all."""
    embedding = None
    if embeddings:
        embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm if norm > 1e-9 else None

    histogram = None
    if histograms:
        histogram = np.mean(histograms, axis=0)
        total = histogram.sum()
        histogram = histogram / total if total > 1e-9 else None

    return embedding, histogram


def _sample_signature(
    cap: cv2.VideoCapture,
    mask_path: Path,
    candidate_frames: list[int],
    signature_samples: int,
    embedder: OSNetEmbedder | None,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
    """Averages OSNet embedding + hue histogram over up to
    `signature_samples` usable frames drawn from `candidate_frames`
    (see `_frame_signal`), stopping once that many are collected. Also
    returns the frame indices actually used, so a suspicious similarity
    score can be checked by opening those exact frames.

    `candidate_frames` must be ordered from the track's risky edge
    OUTWARD (backward from the last frame for an ending track, forward
    from the first for a starting one), so skipped ambiguous frames
    naturally extend the search into the track's more stable middle."""
    embeddings: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    used_frame_indices: list[int] = []

    mask_cap = cv2.VideoCapture(str(mask_path))
    if not mask_cap.isOpened():
        raise FileNotFoundError(f"Could not open mask video: {mask_path}")

    used_frames = 0
    try:
        for frame_idx in candidate_frames:
            if used_frames >= signature_samples:
                break
            if frame_idx < 0:
                continue
            mask_frame = _read_mask_frame(mask_cap, frame_idx, threshold)
            if mask_frame is None:
                continue  # past this track's own file bounds -- skip, don't guess
            emb, hist = _frame_signal(cap, mask_frame, frame_idx, embedder)
            if emb is None and hist is None:
                continue  # empty, ambiguous, or border-touching -- skip, don't guess
            if emb is not None:
                embeddings.append(emb)
            if hist is not None:
                histograms.append(hist)
            used_frames += 1
            used_frame_indices.append(frame_idx)
    finally:
        mask_cap.release()

    embedding, histogram = _finalize_signature(embeddings, histograms)
    return embedding, histogram, used_frame_indices


def _evenly_spaced_frames(first: int, last: int, n: int) -> list[int]:
    """`n` frame indices evenly spread across `[first, last]` -- used by
    `_pooled_group_signature` to sample pose/lighting diversity across a
    fragment's whole span, unlike `_sample_signature`'s edge-anchored
    order."""
    span = last - first + 1
    if span <= n:
        return list(range(first, last + 1))
    if n <= 1:
        return [first]
    return sorted({int(round(first + i * (span - 1) / (n - 1))) for i in range(n)})


def _search_nearby_signal(
    cap: cv2.VideoCapture,
    member_cap: cv2.VideoCapture,
    target: int,
    first: int,
    last: int,
    max_radius: int,
    exclude: set[int],
    embedder: OSNetEmbedder | None,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> tuple[int, np.ndarray | None, np.ndarray | None] | None:
    """Tries `target`, then alternately expands outward (+1, -1, +2, -2,
    ...) up to `max_radius`, clamped to `[first, last]` and skipping
    `exclude`, returning the first frame that passes `_frame_signal`'s
    checks -- so one bad frame at an evenly-spaced target position
    doesn't lose that sampling slot entirely."""
    offsets = [0]
    for r in range(1, max_radius + 1):
        offsets.append(r)
        offsets.append(-r)
    for offset in offsets:
        frame_idx = target + offset
        if frame_idx < first or frame_idx > last or frame_idx in exclude:
            continue
        mask_frame = _read_mask_frame(member_cap, frame_idx, threshold)
        if mask_frame is None:
            continue
        emb, hist = _frame_signal(cap, mask_frame, frame_idx, embedder)
        if emb is None and hist is None:
            continue
        return frame_idx, emb, hist
    return None


def _pooled_group_signature(
    cap: cv2.VideoCapture,
    mask_paths: dict[int, Path],
    member_ids: list[int],
    bounds: dict[int, tuple[int, int]],
    samples_per_member: int,
    embedder: OSNetEmbedder | None,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, list[int]]]:
    """Aggregated appearance signature for an already-confirmed GROUP of
    ids, pooling clean frames spread across ALL members (different
    times/poses/lighting) instead of just one fragment's edge frames --
    the pass-two fallback for an orphan start track pass one couldn't
    match. For each evenly-spaced target frame per member, searches a
    small neighborhood (`_search_nearby_signal`) instead of only the
    exact target, so one unlucky frame doesn't cost that member a whole
    sample. Also returns `{member_id: [frame indices used]}` for the
    report."""
    embeddings: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    used_frame_indices: dict[int, list[int]] = {}
    for member_id in member_ids:
        if member_id not in bounds or member_id not in mask_paths:
            continue  # e.g. a too-short fragment folded into this group
        first, last = bounds[member_id]
        targets = _evenly_spaced_frames(first, last, samples_per_member)
        # search radius per target: roughly half the average spacing
        # between targets, capped by MAX_POOLED_SEARCH_RADIUS.
        avg_gap = max(1, (last - first + 1) // max(1, samples_per_member))
        max_radius = min(max(1, avg_gap // 2), MAX_POOLED_SEARCH_RADIUS)

        member_cap = cv2.VideoCapture(str(mask_paths[member_id]))
        if not member_cap.isOpened():
            raise FileNotFoundError(f"Could not open mask video: {mask_paths[member_id]}")
        try:
            used: set[int] = set()
            for target in targets:
                found = _search_nearby_signal(
                    cap, member_cap, target, first, last, max_radius, used, embedder, threshold,
                )
                if found is None:
                    continue  # nothing usable near this slot -- some signal beats none
                frame_idx, emb, hist = found
                used.add(frame_idx)
                if emb is not None:
                    embeddings.append(emb)
                if hist is not None:
                    histograms.append(hist)
                used_frame_indices.setdefault(member_id, []).append(frame_idx)
        finally:
            member_cap.release()
    embedding, histogram = _finalize_signature(embeddings, histograms)
    return embedding, histogram, used_frame_indices


def _pair_similarity(
    end_sig: tuple[np.ndarray | None, np.ndarray | None],
    start_sig: tuple[np.ndarray | None, np.ndarray | None],
) -> float:
    """0..1 similarity between two signatures: the STRONGEST of (OSNet,
    color), never their sum/average."""
    end_emb, end_hist = end_sig
    start_emb, start_hist = start_sig
    scores: list[float] = []
    emb_sim = embedding_similarity(end_emb, start_emb)
    if emb_sim is not None:
        scores.append(emb_sim)
    if end_hist is not None and start_hist is not None:
        scores.append(_histogram_similarity(end_hist, start_hist))
    return max(scores) if scores else 0.0
