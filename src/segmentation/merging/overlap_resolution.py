"""
segmentation/merging/overlap_resolution.py
===========================================
Zeroth-pass same-time overlap resolution: for every pair of ids alive
at the same time (something `reappearance_merge.py` structurally can't
consider, since it only matches a track that ENDS against one that
STARTS later), decide whether they're the same body split into
simultaneous fragments (e.g. a coat detected apart from the person
wearing it) rather than two different people. Either two fixed
thresholds (real pixel-to-pixel mask distance + centroid distance,
calibrated on real CHUV footage -- see `merge_fragments.py`'s module
docstring for the calibration history) or a small classifier trained
by `classifier/train_overlap_classifier.py`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from segmentation.merging.mask_io import DEFAULT_MASK_THRESHOLD
from segmentation.merging.mask_utils import _centroid_distance, _mask_bbox_diagonal, _mask_min_pixel_distance


def _stream_overlap_stats(
    path_a: Path,
    path_b: Path,
    start_frame: int,
    end_frame: int,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> tuple[list[float], list[float], list[float]]:
    """Streams both mask files over `[start_frame, end_frame]` and, for
    every frame where BOTH have content, computes the real pixel-to-
    pixel minimum distance, the centroid distance, and a scale
    reference (mean bbox diagonal). Returns the three lists in frame
    order."""
    cap_a = cv2.VideoCapture(str(path_a))
    cap_b = cv2.VideoCapture(str(path_b))
    if start_frame > 0:
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pixel_dists: list[float] = []
    centroid_dists: list[float] = []
    scales: list[float] = []
    try:
        for _f in range(start_frame, end_frame + 1):
            ok_a, ra = cap_a.read()
            ok_b, rb = cap_b.read()
            if not ok_a or not ok_b:
                break
            ga = ra[:, :, 0] if ra.ndim == 3 else ra
            gb = rb[:, :, 0] if rb.ndim == 3 else rb
            ma = ga > threshold
            mb = gb > threshold
            if not ma.any() or not mb.any():
                continue
            pixel_dists.append(_mask_min_pixel_distance(ma, mb))
            centroid_dists.append(_centroid_distance(ma, mb))
            scales.append((_mask_bbox_diagonal(ma) + _mask_bbox_diagonal(mb)) / 2.0)
    finally:
        cap_a.release()
        cap_b.release()

    return pixel_dists, centroid_dists, scales


def _classifier_score(classifier: dict, feature_values: dict[str, float]) -> float:
    """`sigmoid(bias + sum(weight_i * feature_i))` for the learned
    overlap classifier. `classifier` is the JSON produced by
    `train_overlap_classifier.py`: `{"features": [...], "weights": [...],
    "bias": ...}`."""
    z = float(classifier["bias"])
    for name, w in zip(classifier["features"], classifier["weights"]):
        z += float(w) * feature_values[name]
    return float(1.0 / (1.0 + np.exp(-z)))


def _resolve_overlap_merges(
    mask_paths: dict[int, Path],
    bounds: dict[int, tuple[int, int]],
    pixel_gap_threshold: float,
    centroid_threshold: float,
    min_overlap_frames: int,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    classifier: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """For every pair of ids that overlap in time and both have
    trustworthy bounds: stream both mask files over their shared
    window, and if enough both-active frames exist, compute the MEDIAN
    pixel and centroid distance. Accept as "same body, split into
    simultaneous fragments" only if both medians clear their threshold
    (pixel distance alone can't tell a same-body split from a brief
    occlusion between two different people; centroid distance is what
    actually separates the two classes in the calibration data).

    If `classifier` is given, it REPLACES the two fixed thresholds: a
    pair qualifies if the model's probability >= 0.5, ranked by that
    probability instead of raw centroid distance.

    Returns `(accepted, rejected)`, recording both raw medians and the
    both-active frame count for every candidate considered, not just
    accepted ones.

    IMPORTANT: each id is accepted into at most one pair (its single
    closest match). A small static object mistakenly given its own SAM
    track (e.g. a jacket on a radiator) can clear both thresholds
    against TWO different real people who happen to sit near it at
    different times -- accepting both pairs would union those two
    people together by transitivity, even though comparing them
    directly correctly rejects the match. Restricting each id to its
    nearest match (greedy, closest pairs claimed first) prevents this
    without needing to know in advance which candidate is the false one."""
    ids = sorted(bounds.keys())
    rejected: list[dict] = []

    # first pass: collect every pair that clears both thresholds as a
    # QUALIFYING candidate (not yet accepted -- greedy matching below
    # decides who actually gets it).
    qualifying: list[dict] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            a_first, a_last = bounds[a]
            b_first, b_last = bounds[b]
            overlap_start = max(a_first, b_first)
            overlap_end = min(a_last, b_last)
            if overlap_start > overlap_end:
                continue  # no temporal overlap at all -- pass 1/2 already handle this pair

            pixel_dists, centroid_dists, scales = _stream_overlap_stats(
                mask_paths[a], mask_paths[b], overlap_start, overlap_end, mask_threshold,
            )
            n = len(pixel_dists)
            if n < min_overlap_frames:
                continue  # too few both-active frames to trust a median

            median_pixel = float(np.median(pixel_dists))
            median_centroid = float(np.median(centroid_dists))
            median_scale = float(np.median(scales))
            # normalized = centroid distance as a fraction of on-screen
            # scale right now -- portable across cameras/zoom levels.
            median_centroid_norm = median_centroid / median_scale if median_scale > 0 else float("inf")
            record = {
                "id_a": a,
                "id_b": b,
                "median_pixel_gap_px": round(median_pixel, 1),
                "median_centroid_dist_px": round(median_centroid, 1),
                "median_scale_px": round(median_scale, 1),
                "median_centroid_dist_norm": round(median_centroid_norm, 4),
                "overlap_frames": n,
            }
            if classifier is not None:
                score = _classifier_score(classifier, {
                    "centroid_dist_norm": median_centroid_norm,
                    "pixel_gap_px": median_pixel,
                })
                record["classifier_score"] = round(score, 4)
                if score >= 0.5:
                    qualifying.append(record)
                else:
                    rejected.append(record)
            elif median_pixel <= pixel_gap_threshold and median_centroid <= centroid_threshold:
                qualifying.append(record)
            else:
                rejected.append(record)

    # second pass: greedy nearest-first (or most-confident-first, with a
    # classifier) matching -- the strongest pairs claim their ids first,
    # so a fragment already claimed can't bridge a second, weaker match
    # (see docstring above). A qualifying pair that loses out this way
    # moves to `rejected` tagged `lost_to_closer_match`, kept visible
    # rather than silently dropped.
    if classifier is not None:
        qualifying.sort(key=lambda r: -r["classifier_score"])
    else:
        qualifying.sort(key=lambda r: r["median_centroid_dist_px"])
    claimed: set[int] = set()
    accepted: list[dict] = []
    for record in qualifying:
        a, b = record["id_a"], record["id_b"]
        if a in claimed or b in claimed:
            rejected.append({**record, "lost_to_closer_match": True})
            continue
        accepted.append(record)
        claimed.add(a)
        claimed.add(b)

    return accepted, rejected
