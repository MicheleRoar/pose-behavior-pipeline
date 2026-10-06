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
    blacklist: set[tuple[int, int]] | None = None,
) -> list[dict]:
    """Pass 1: global Hungarian assignment between `end_ids` (rows) and
    `start_ids` (columns). Returns every temporally-valid pair Hungarian
    assigned (end strictly before start), each tagged
    `"accepted": similarity >= merge_threshold` -- including near-misses,
    useful when tuning the threshold. A single global assignment (not
    pairwise greedy) so two simultaneous fragmentation events don't
    steal each other's correct match.
 
    `blacklist` is a set of `(end_id, start_id)` pairs forced to
    `_IMPOSSIBLE_COST` -- used by the caller to retry the assignment
    after a temporal veto, so the vetoed edge can't win the same slot
    twice and the next-best (possibly temporally-sensible) candidate
    gets a chance instead. See `merge_fragments`'s veto-retry loop."""
    if not end_ids or not start_ids:
        return []
 
    blacklist = blacklist or set()
    cost = np.full((len(end_ids), len(start_ids)), _IMPOSSIBLE_COST)
    for i, e in enumerate(end_ids):
        for j, s in enumerate(start_ids):
            if e == s or bounds[e][1] >= bounds[s][0] or (e, s) in blacklist:
                continue  # same track, start doesn't come after end, or vetoed earlier
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
    orphan_groups: dict[int, list[int]] | None = None,
    blacklist: set[tuple[int, int]] | None = None,
) -> list[dict]:
    """Pass 2: global Hungarian assignment between orphan start tracks
    (rows -- ones pass one left unmatched) and candidate GROUPS
    (columns, `{canonical_id: member_ids}` from pass one). A group only
    qualifies for an orphan if NO member overlaps the orphan in time (a
    real person can't be two simultaneous tracks) and at least one
    member genuinely ends before the orphan starts. Same global-
    assignment, "tag every real candidate" conventions as
    `_resolve_merges`.
 
    `orphan_groups` (`{orphan_id: member_ids}`) is the orphan's OWN
    pass-one group: an orphan is unmatched on its START side but may
    already have been merged on its END side (x -> y), so accepting
    "orphan into group G" really merges its whole chain into G. The
    overlap check therefore runs on every member of the orphan's chain,
    not just the orphan -- otherwise a 30-frame fragment can bridge two
    people who coexist for the entire session.
 
    `blacklist` is a set of `(orphan_id, group_id)` pairs forced to
    `_IMPOSSIBLE_COST`, same veto-retry purpose as in `_resolve_merges`."""
    if not orphan_ids or not groups:
        return []
 
    blacklist = blacklist or set()
    group_ids = list(groups.keys())
    cost = np.full((len(orphan_ids), len(group_ids)), _IMPOSSIBLE_COST)
    for i, o in enumerate(orphan_ids):
        o_first, o_last = bounds[o]
        chain = (orphan_groups or {}).get(o, [o])
        chain_bounds = [bounds[m] for m in chain if m in bounds] or [bounds[o]]
        for j, g in enumerate(group_ids):
            members = groups[g]
            if o in members or (o, g) in blacklist:
                continue  # orphan is (trivially) already part of this group, or vetoed earlier
            member_bounds = [bounds[m] for m in members if m in bounds]
            if not member_bounds:
                continue  # e.g. every member was too short to have bounds
            if any(m_first <= c_last and m_last >= c_first
                   for m_first, m_last in member_bounds
                   for c_first, c_last in chain_bounds):
                continue  # a member of this group is active at the same time as the orphan's chain -- can't be the same person
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
 
 
def _overlap_frames(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Number of frames where both `(first, last)` spans are active
    (0 when they don't overlap)."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
 
 
def _group_chains_with_temporal_veto(
    merges: list[tuple[int, int, float]],
    all_ids: list[int],
    bounds: dict[int, tuple[int, int]],
    allowed_overlap_pairs: set[frozenset[int]],
    max_tolerated_overlap_frames: int = 0,
) -> tuple[dict[int, int], list[dict]]:
    """Same union-find as `_group_chains`, but every union is applied
    ONLY if the resulting group stays temporally consistent: a real
    person can't be two tracks alive at the same time, so a merge that
    would put two simultaneously-active ids into the same group is
    vetoed -- unless that exact pair was accepted by the zeroth pass
    (`allowed_overlap_pairs`), which is the one legitimate case of
    same-body simultaneous fragments (a garment being put on/taken
    off). Overlaps of at most `max_tolerated_overlap_frames` are
    ignored (tracker jitter at a fragment boundary, not real
    coexistence).
 
    This is the whole-video counterpart of the zeroth pass's
    one-match-per-fragment rule: pass 1/2 only ever check the two ids
    of a pair against each other, so a short ambiguous fragment can
    still bridge two different people by transitivity (A->x and x->B
    both look plausible even though A and B coexist for thousands of
    frames). Merges are applied in the order given -- put the most
    trusted ones first, they win any conflict.
 
    Returns `({original_id: canonical_id}, vetoed)` where `vetoed`
    lists every merge skipped, with the conflicting pair and its
    overlap length, for the report."""
    parent = {oid: oid for oid in all_ids}
    members: dict[int, list[int]] = {oid: [oid] for oid in all_ids}
 
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
 
    vetoed: list[dict] = []
    for end_id, start_id, sim in merges:
        ra, rb = find(end_id), find(start_id)
        if ra == rb:
            continue
        conflict = None
        for a in members[ra]:
            if a not in bounds:
                continue
            for b in members[rb]:
                if b not in bounds or frozenset((a, b)) in allowed_overlap_pairs:
                    continue
                ov = _overlap_frames(bounds[a], bounds[b])
                if ov > max_tolerated_overlap_frames:
                    conflict = (a, b, ov)
                    break
            if conflict:
                break
        if conflict:
            vetoed.append({
                "from_id": end_id, "into_id": start_id, "similarity": round(sim, 3),
                "conflict_ids": [conflict[0], conflict[1]], "conflict_overlap_frames": conflict[2],
            })
            continue
        keep, drop = min(ra, rb), max(ra, rb)
        parent[drop] = keep
        members[keep].extend(members.pop(drop))
 
    return {oid: find(oid) for oid in all_ids}, vetoed