"""
segmentation/merging/reappearance_merge.py
===========================================
Pass-1/pass-2 whole-video reappearance merge: global (Hungarian, never
greedy) assignment between a track that ENDS and a later track that
STARTS, using the signatures from `signatures.py`. Handles a person
lost/occluded/off-screen who reappears as a brand-new id --
`overlap_resolution.py` handles the separate case of two ids alive at
the same time.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from segmentation.merging.signatures import _pair_similarity

# cost for a temporally-impossible pair (end doesn't precede start) in
# the Hungarian matrix -- worse than any real similarity score (cost =
# 1 - similarity, i.e. [0, 1]) but finite, so Hungarian never picks it
# ahead of a real candidate.
_IMPOSSIBLE_COST = 10.0


def _resolve_merges(
    end_ids: list[int],
    start_ids: list[int],
    bounds: dict[int, tuple[int, int]],
    end_sigs: dict[int, tuple],
    start_sigs: dict[int, tuple],
    merge_threshold: float,
) -> list[dict]:
    """Pass 1: global Hungarian assignment between `end_ids` (rows) and
    `start_ids` (columns). Returns every temporally-valid pair Hungarian
    assigned (end strictly before start), each tagged
    `"accepted": similarity >= merge_threshold` -- including near-misses,
    useful when tuning the threshold. A single global assignment (not
    pairwise greedy) so two simultaneous fragmentation events don't
    steal each other's correct match."""
    if not end_ids or not start_ids:
        return []

    cost = np.full((len(end_ids), len(start_ids)), _IMPOSSIBLE_COST)
    for i, e in enumerate(end_ids):
        for j, s in enumerate(start_ids):
            if e == s or bounds[e][1] >= bounds[s][0]:
                continue  # same track, or start doesn't come after end
            sim = _pair_similarity(end_sigs[e], start_sigs[s])
            cost[i, j] = 1.0 - sim

    row_idx, col_idx = linear_sum_assignment(cost)
    candidates = []
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] >= _IMPOSSIBLE_COST:
            continue  # not a real candidate -- see docstring
        similarity = float(1.0 - cost[r, c])
        candidates.append({
            "from_id": end_ids[r],
            "into_id": start_ids[c],
            "similarity": round(similarity, 3),
            "accepted": bool(similarity >= merge_threshold),
        })
    return candidates


def _resolve_group_merges(
    orphan_ids: list[int],
    groups: dict[int, list[int]],
    bounds: dict[int, tuple[int, int]],
    group_sigs: dict[int, tuple],
    orphan_sigs: dict[int, tuple],
    merge_threshold: float,
) -> list[dict]:
    """Pass 2: global Hungarian assignment between orphan start tracks
    (rows -- ones pass one left unmatched) and candidate GROUPS
    (columns, `{canonical_id: member_ids}` from pass one). A group only
    qualifies for an orphan if NO member overlaps the orphan in time (a
    real person can't be two simultaneous tracks) and at least one
    member genuinely ends before the orphan starts. Same global-
    assignment, "tag every real candidate" conventions as
    `_resolve_merges`."""
    if not orphan_ids or not groups:
        return []

    group_ids = list(groups.keys())
    cost = np.full((len(orphan_ids), len(group_ids)), _IMPOSSIBLE_COST)
    for i, o in enumerate(orphan_ids):
        o_first, o_last = bounds[o]
        for j, g in enumerate(group_ids):
            members = groups[g]
            if o in members:
                continue  # orphan is (trivially) already part of this group
            member_bounds = [bounds[m] for m in members if m in bounds]
            if not member_bounds:
                continue  # e.g. every member was too short to have bounds
            if any(m_first <= o_last and m_last >= o_first for m_first, m_last in member_bounds):
                continue  # a member of this group is active at the same time as the orphan -- can't be the same person
            if not any(m_last < o_first for _m_first, m_last in member_bounds):
                continue  # no member of this group actually ends before the orphan starts
            sim = _pair_similarity(group_sigs[g], orphan_sigs[o])
            cost[i, j] = 1.0 - sim

    row_idx, col_idx = linear_sum_assignment(cost)
    candidates = []
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] >= _IMPOSSIBLE_COST:
            continue  # not a real candidate -- see docstring
        similarity = float(1.0 - cost[r, c])
        candidates.append({
            "orphan_id": orphan_ids[r],
            "group_id": group_ids[c],
            "similarity": round(similarity, 3),
            "accepted": bool(similarity >= merge_threshold),
        })
    return candidates


def _group_chains(merges: list[tuple[int, int, float]], all_ids: list[int]) -> dict[int, int]:
    """Turns a list of accepted (end_id -> start_id) merges into
    `{original_id: canonical_id}` via union-find, following chains (A
    merges into B, B merges into C => all map to the same id). The
    canonical id is the group's SMALLEST original id -- a stable,
    deterministic output filename, no other meaning."""
    parent = {oid: oid for oid in all_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for end_id, start_id, _sim in merges:
        union(end_id, start_id)

    return {oid: find(oid) for oid in all_ids}
