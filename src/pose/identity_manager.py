"""
identity_manager.py
====================
Shared decision layer for "Identity & Re-identification", used both by the
pose pipeline (`reid.py`, anthropometric signature) and the segmentation
one (`seg_reid.py`, mask centroid/color/shape). This module does NOT
recompute any similarity signal -- those remain entirely in
`reid.py`/`seg_reid.py` (body proportions, color, position,
silhouette...), with all their causality rules and guardrails already
documented there. What it centralizes here:

  1. `IdentityMode` / `SessionMode`: the shared vocabulary exposed by the
     "Identity & Re-identification" section of the GUI (see webui/app.js),
     instead of two parameter surfaces that risk diverging.

  2. Batch, globally optimal assignment (Hungarian algorithm, via
     `scipy.optimize.linear_sum_assignment`): when MULTIPLE people
     "awaiting verification" become ready in the SAME frame, the previous
     version (sequential loop in `reid.py`/`seg_reid.py`) resolved them
     one at a time in iteration order, mutating the map of "lost" people
     as it went -- correct (it doesn't allow a double claim in the same
     frame) but not necessarily the overall best pairing when there are
     multiple candidates and multiple lost identities competing. Here a
     candidates x lost-identities cost matrix is built and solved in one
     shot.

  3. Three outcomes instead of a binary match/non-match: `matched` (score
     comfortably above threshold) / `new` (no plausible candidate, mints
     a new identity) / `uncertain` (gray zone between the two thresholds
     -- for the policy "better to fragment an identity than to silently
     swap two", an uncertain candidate is FLAGGED but NOT merged
     automatically).

Convention: this module always works in COST space (lower values = more
similar), consistent with the RMS distance already used by reid.py.
`seg_reid.py`, which internally uses a 0..1 similarity score (higher =
more similar), converts it with `cost = 1 - score` before calling
`resolve_batch()` -- see there for the wiring. Pairs that are impossible
per the causality rule (a person "lost" AFTER the candidate had already
appeared) must be passed as `inf`, never omitted: `resolve_batch()`
treats them as "that pair doesn't exist" without excluding the row/column
from the Hungarian assignment (which requires a complete matrix).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class IdentityMode(str, Enum):
    """What the pipeline should do with ids, from least to most
    persistent. Interpreted by the caller (pipeline_runner.py /
    webui/api.py): this module doesn't decide on its own whether to
    instantiate a ReIdentifier/SegReIdentifier, it just names the three
    options in a single place."""

    FRAME_BY_FRAME = "frame_by_frame"   # no id persistent across frames
    TRACKING_ONLY = "tracking_only"     # id kept as long as the tracker's track is continuous
    TRACKING_REID = "tracking_reid"     # + recovery after exit/occlusion/track loss


class SessionMode(str, Enum):
    """How many people are expected in the session. Doesn't add new
    mechanics: it translates into the parameters already existing in
    both engines (`max_people=1` for SINGLE on the reid.py/seg_reid.py
    side, a higher or absent cap for MULTIPLE) -- see
    `suggested_max_people_policy()` below."""

    SINGLE = "single"
    MULTIPLE = "multiple"


def wants_reid_engine(mode: IdentityMode) -> bool:
    """True only if the mode requires instantiating a ReIdentifier/
    SegReIdentifier (TRACKING_REID). FRAME_BY_FRAME and TRACKING_ONLY
    both use the underlying tracker's (ByteTrack/SAM) raw ids directly --
    the difference between the two is whether that id is also exposed as
    a "stable identity" in the CSV/overlay (TRACKING_ONLY) or whether
    every frame is treated as independent even with the same track_id
    (FRAME_BY_FRAME, useful only for comparisons "with no continuity
    assumption at all")."""
    return mode == IdentityMode.TRACKING_REID


@dataclass
class IdentityManagerConfig:
    """Shared configuration exposed by the "Identity & Re-identification"
    section of the GUI."""

    mode: IdentityMode = IdentityMode.TRACKING_REID
    session_mode: SessionMode = SessionMode.MULTIPLE
    max_people: int | None = None
    lost_identity_memory_s: float = 180.0
    # matching policy: always conservative by construction (see
    # `resolve_batch`) -- this flag ONLY controls whether a candidate in
    # the "gray zone" is flagged as `uncertain` (default) or treated as
    # `new` with no trace (previous, binary behavior).
    flag_uncertain: bool = True
    # Below (in cost, i.e. <=) = automatic merge. Above `reject_cost`
    # (exclusive) = new identity, no plausible candidate. In between =
    # gray zone ("uncertain").
    accept_cost: float = 0.12
    reject_cost: float = 0.20


@dataclass
class AssignmentOutcome:
    candidate_index: int              # row in the cost matrix passed to resolve_batch
    matched_lost_id: int | None       # claimed identity (person_id), if any
    cost: float | None
    status: str                       # "matched" | "new" | "uncertain"


def resolve_batch(cost_matrix: np.ndarray, lost_ids: list[int],
                   config: IdentityManagerConfig) -> list[AssignmentOutcome]:
    """Jointly resolves which candidate "awaiting verification" (rows)
    claims which lost identity (columns), given a cost matrix (lower
    values = more similar; `inf`/`nan` for pairs impossible due to
    causality or insufficient signal). One outcome per row, always --
    even with an empty `lost_ids` (all "new") or with more candidates
    than lost identities (the excess candidates, not assignable by the
    Hungarian algorithm on a rectangular matrix, remain "new").

    Solved with `scipy.optimize.linear_sum_assignment` (minimizes total
    cost): when multiple people re-enter the same frame and compete for
    the same lost identities, this finds the overall best pairing instead
    of whichever candidate processed first "grabs" one (previous
    sequential loop).
    """
    n_candidates = cost_matrix.shape[0]
    if n_candidates == 0:
        return []
    if not lost_ids:
        return [AssignmentOutcome(i, None, None, "new") for i in range(n_candidates)]

    from scipy.optimize import linear_sum_assignment

    # Impossible pairs (inf/nan) become a huge but finite cost:
    # linear_sum_assignment requires finite values. Large enough to never
    # be chosen unless it's the ONLY option left for that row/column (in
    # which case it's discarded right after anyway by the reject_cost
    # check, see below).
    safe_cost = np.where(np.isfinite(cost_matrix), cost_matrix, 1e6)
    row_ind, col_ind = linear_sum_assignment(safe_cost)
    assigned_col = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    outcomes: list[AssignmentOutcome] = []
    for i in range(n_candidates):
        col = assigned_col.get(i)
        cost = float(cost_matrix[i, col]) if col is not None else None
        if col is None or cost is None or not np.isfinite(cost) or cost > config.reject_cost:
            outcomes.append(AssignmentOutcome(i, None, cost, "new"))
        elif cost <= config.accept_cost:
            outcomes.append(AssignmentOutcome(i, int(lost_ids[col]), cost, "matched"))
        else:
            # gray zone: the Hungarian algorithm still proposes this pair
            # as the best one available, but not confident enough to
            # merge automatically -- see the module docstring.
            if config.flag_uncertain:
                outcomes.append(AssignmentOutcome(i, int(lost_ids[col]), cost, "uncertain"))
            else:
                outcomes.append(AssignmentOutcome(i, None, cost, "new"))
    return outcomes


def suggested_max_people_policy(session_mode: SessionMode, configured_max_people: int | None) -> int | None:
    """Translates `SessionMode` into the hard `max_people` cap already
    understood by ReIdentifier/SegReIdentifier. SINGLE forces 1 (only one
    person expected, identity recovery can be as permissive as possible:
    whoever re-enters must necessarily be that person -- see
    `_force_match_at_capacity` in reid.py / the absolute constraint in
    seg_reid.py). MULTIPLE respects the user-configured value (can remain
    `None`: no cap, never force a claim just because "there can't be one
    more person")."""
    if session_mode == SessionMode.SINGLE:
        return 1
    return configured_max_people
