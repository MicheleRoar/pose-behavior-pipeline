"""
seg_reid.py
============
Re-identification for the segmentation-only pipeline
(seg_estimation.py / segmentation_demo.py), analogous to reid.py but
WITHOUT keypoints: here the "signature" is built from position (mask
centroid), color sampled inside the mask, and shape of the silhouette
(box aspect ratio, mask/box fill ratio) -- not from body proportions,
which don't exist without keypoints.

Color is a hue HISTOGRAM (see `_mask_hue_histogram`), not an average
hue: a striped/two-tone garment averages out to a generic color
indistinguishable from many others (verified on a real session: a child
with a striped shirt wasn't being re-linked on re-entry), while the
histogram captures the distribution (two peaks for a two-tone garment)
-- compared by intersection, not by the distance of a single value.

Deliberate design difference from reid.py
----------------------------------------------------
When the number of session participants is known AND SMALL (1-2, the 1v1
case), here the cap is TRULY unbreakable -- not "in most cases" like
reid.py's optional `max_people` fallback (which can give up and still
mint a new person_id if it finds no eligible candidate), but an absolute
constraint: NEVER more than `max_people` identities in the whole
session, no exceptions.

This is defensible precisely because the number here is small and
certain: with only 1 person expected, every detection is by definition
that person, no comparison needed. With 2 (or more, up to about a dozen),
once the cap is already reached a new raw track_id can always be linked
to the closest one by position/color/shape, even without a strong
signal -- the cost of this guarantee is that in extremely rare
pathological cases (e.g. a spurious double detection in the same frame,
more "new" raw_ids than remaining free slots) two different raw_ids may
end up on the same person_id for that single frame, instead of one of
the two staying "without an identity" -- an explicit choice, consistent
with the requirement of NEVER having the possibility of an extra id.

Two levels of linking (important if `max_people` is set with margin,
e.g. 20 for a group of ~10 children "to be safe"): a new raw_id ALWAYS
first attempts a "soft" link (with `soft_match_threshold`) to a known
but currently-not-visible person, even if there are still free slots --
without this step, a wide `max_people` meant the cap was never really
reached, and every brief disappearance (occlusion, exiting the frame
edge) opened a new id instead of re-linking: a temporary id swap visible
even with re-id active. Only if the soft link fails (no candidate above
threshold) does the previous logic kick in: free slot -> new id, cap
reached -> forced link without threshold.

Unlike reid.py, here position/color/shape are not "bonuses" on a primary
signature (an anthropometric signature doesn't exist without keypoints):
they're combined with a weighted average (not "the strongest wins" like
in reid.py, which made sense there because they were secondary signals
on an already-significant primary signal -- here they're the only
available signals).

Optional signal: appearance embedding (OSNet)
--------------------------------------------------
Fourth term of the weighted average, same scheme as position/color/shape
above -- see `pose/appearance_embedding.py` for the module and the why
of a real embedding (OSNet) instead of another geometric/color
heuristic. It's typically the most reliable of the four (higher default
weight, see `embedding_weight`): unlike position/shape/color, it doesn't
depend on where the person is in the frame nor on a single color
channel, only on their overall appearance. As in reid.py, a known
person's embedding is updated on every frame in which they're visible
with an exponential moving average (`_update_person_state`, the
StrongSORT idea cited in `appearance_embedding.ema_update`) instead of
being replaced by only the current frame -- the estimate consolidates
the longer the person stays visible, exactly the requested behavior
("stay in memory to be easily linked on re-entry"). Requires an
`embedder` (an `OSNetEmbedder` instance) passed to the constructor; if
`None` (default), behavior is identical to before this addition.

No multi-frame buffer/wait: unlike reid.py (which needs ~15 frames to
stabilize an anthropometric signature), here position and color are
already usable from the very first frame in which a raw track_id
appears -- the decision is made immediately, with no sliding window or
retry.

Honest limitations
-------------------
  - No minimum similarity threshold on the "cap reached" path: the
    choice is always "the best available candidate", never "no
    candidate is good enough" -- that's the required guarantee (never
    an extra id), but it means a link can happen even with a low
    similarity score if there's nothing better.
  - No expiry for identity memory: unlike reid.py (`max_lost_seconds`),
    here known people's positions/colors remain valid for the whole
    session -- correct for the use case (short session, few fixed
    people), but would make the comparison progressively less reliable
    on a very long session with major scene changes.
  - Default weights and thresholds are not calibrated on real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from segmentation.seg_estimation import mask_area, mask_centroid
from pose.identity_manager import (
    SessionMode, IdentityManagerConfig, resolve_batch, suggested_max_people_policy,
)
from pose.appearance_embedding import OSNetEmbedder, embedding_similarity, ema_update


_HUE_HIST_BINS = 16  # 180deg/16 = 11.25deg/bin -- coarse enough to withstand
                      # lighting noise, fine enough to separate two distinct colors


def _mask_hue_histogram(frame: np.ndarray, poly: np.ndarray,
                         bins: int = _HUE_HIST_BINS) -> np.ndarray | None:
    """Hue histogram (OpenCV Hue, 0-179) of the pixels inside the mask
    polygon, weighted by saturation (nearly-desaturated pixels --
    shadows, neutral fabrics -- count less: their hue is noise, not
    signal) and normalized to sum 1.

    Replaces the AVERAGE hue used previously: a two-tone or striped
    garment (e.g. an olive/cream striped shirt) averages out to a
    generic intermediate color, indistinguishable from many other
    garments -- a real test session showed exactly this, a child with a
    striped shirt that wasn't being re-linked on re-entry into frame. A
    histogram instead captures the distribution (two distinct peaks for
    a two-tone garment), much closer to how a human recognizes a pattern
    at a glance.

    `None` if the polygon is empty/degenerate, covers a negligible area,
    or (after discarding overly desaturated pixels) not enough pixels
    remain to be trusted -- same "no signal available" treatment as the
    rest of the module, never a made-up histogram."""
    if poly.shape[0] < 3:
        return None
    h, w = frame.shape[:2]
    pts = np.round(poly).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    if cv2.countNonZero(mask) < 25:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    hue = hsv[ys, xs, 0].astype(np.float64)          # 0..179
    sat = hsv[ys, xs, 1].astype(np.float64) / 255.0  # 0..1, used as weight

    valid = sat > 0.15  # nearly gray pixels: their hue is sensor noise
    if valid.sum() < 25:
        return None
    hue, sat = hue[valid], sat[valid]

    hist, _ = np.histogram(hue, bins=bins, range=(0, 180), weights=sat)
    total = hist.sum()
    if total <= 0:
        return None
    return hist / total


def _shape_descriptor(bbox: np.ndarray, poly: np.ndarray) -> tuple[float, float]:
    """(box aspect ratio w/h, mask-area/box-area fill ratio) -- coarse
    shape descriptor, invariant to absolute position/scale, weak posture
    signal (standing/sitting/crouching). (nan, nan) if the polygon is
    empty."""
    x1, y1, x2, y2 = bbox
    box_w, box_h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    if poly.shape[0] < 3:
        return np.nan, np.nan
    aspect = box_w / box_h
    fill = mask_area(poly) / (box_w * box_h)
    return aspect, fill


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """0..1 similarity between two normalized (sum 1) hue histograms via
    INTERSECTION (sum(min(a, b))) -- standard metric for color
    histograms: 1.0 = identical distributions, 0.0 = no overlap. Captures
    a two-tone/striped pattern much better than a single distance
    between average hues (see `_mask_hue_histogram`'s docstring)."""
    return float(np.clip(np.minimum(a, b).sum(), 0.0, 1.0))


def _shape_similarity(a: tuple[float, float], b: tuple[float, float]) -> float:
    """0..1 similarity between two (aspect, fill) pairs."""
    d_aspect = abs(a[0] - b[0]) / max(a[0], b[0], 1e-6)
    d_fill = abs(a[1] - b[1])
    return float(np.clip(1.0 - (min(d_aspect, 1.0) + d_fill) / 2.0, 0.0, 1.0))


@dataclass
class _PersonState:
    position: np.ndarray                      # last known mask centroid
    scale: float                               # sqrt(area) or box diagonal
    color: np.ndarray | None                   # hue histogram (see _mask_hue_histogram)
    shape: tuple[float, float] | None
    embedding: np.ndarray | None                # OSNet, EMA average (see appearance_embedding.ema_update)
    last_seen: float


@dataclass
class _MergeEvent:
    raw_track_id: int
    matched_person_id: int
    frame_time: float
    score: float
    forced: bool = False  # True only if linked without threshold because the cap was already reached


@dataclass
class _UncertainEvent:
    """Gray-zone candidate (see identity_manager.resolve_batch): below
    `soft_match_threshold` but not far enough to entirely rule out a
    link to a known person -- NOT automatically linked (unless the
    `max_people` cap forces a link anyway, see `resolve()`), only
    flagged for review."""
    raw_track_id: int
    candidate_person_id: int
    frame_time: float
    score: float


class SegReIdentifier:
    """Maintains the track_id (ByteTrack) -> person_id correspondence for
    the segmentation pipeline, with a HARD cap of `max_people` identities
    -- see the module docstring for the why and the limitations.

    Usage in segmentation_demo.py (a single wiring point):

        seg_reidentifier = SegReIdentifier(max_people=2)
        ...
        for frame_result in tracker.run(...):
            people = seg_reidentifier.resolve(frame_result.people, now=...,
                                               frame=frame_result.frame)
            # 'people' has the same shape as frame_result.people
            # [(id, bbox, poly, conf), ...], but id is now a stable
            # person_id, never more than max_people distinct values in
            # the whole session.
    """

    def __init__(self, max_people: int, position_weight: float = 0.5,
                 color_weight: float = 0.3, shape_weight: float = 0.2,
                 max_position_dist_scales: float = 6.0,
                 max_position_gap_seconds: float = 30.0,
                 soft_match_threshold: float = 0.6,
                 session_mode: SessionMode = SessionMode.MULTIPLE,
                 flag_uncertain: bool = True,
                 uncertain_score_margin: float = 0.15,
                 embedder: OSNetEmbedder | None = None,
                 embedding_weight: float = 0.7,
                 embedding_ema_alpha: float = 0.9):
        """`session_mode` (new): SINGLE forces `max_people=1` (see
        `identity_manager.suggested_max_people_policy`), overriding
        `max_people` if in conflict -- consistent with `ReIdentifier`
        (reid.py).

        `flag_uncertain` / `uncertain_score_margin` (new): when the soft
        link of a new raw_id has a score below `soft_match_threshold`
        but above `soft_match_threshold - uncertain_score_margin`, with
        `flag_uncertain=True` (default) it is NOT linked but recorded in
        `self.uncertain_log`/`self.last_uncertain` for review (unless
        the max_people cap forces a link anyway, see resolve()); with
        `flag_uncertain=False` the behavior reverts to binary as before.

        `embedder` / `embedding_weight` / `embedding_ema_alpha` (new):
        see "Optional signal: appearance embedding (OSNet)" in the
        module docstring. The weight is ADDITIVE relative to the
        existing three (it doesn't rescale them): with `embedder=None`
        (default) the embedding term is always 0 for everyone, so
        `_pair_score` behaves exactly as before this addition
        (`position_weight`/`color_weight`/`shape_weight` still sum to
        1.0 by default, same meaning as before for
        `soft_match_threshold`) -- no regression for callers who don't
        pass an `embedder`. When the embedder is active, the maximum
        possible score rises up to 1.7 (0.7 more): a match with a very
        similar embedding can exceed the threshold even with mediocre
        position/color/shape, consistent with the fact that the
        embedding, alone, is the single most reliable signal."""
        max_people = suggested_max_people_policy(session_mode, max_people)
        if max_people < 1:
            raise ValueError("max_people must be at least 1")
        self.max_people = max_people
        self.session_mode = session_mode
        self.position_weight = position_weight
        self.color_weight = color_weight
        self.shape_weight = shape_weight
        self.max_position_dist_scales = max_position_dist_scales
        self.max_position_gap_seconds = max_position_gap_seconds
        # threshold below which a NON-forced link (cap not yet reached)
        # is rejected -- see resolve() for why it's needed even when a
        # free slot is still available.
        self.soft_match_threshold = soft_match_threshold
        self.flag_uncertain = flag_uncertain
        self.uncertain_score_margin = uncertain_score_margin
        self.embedder = embedder
        self.embedding_weight = embedding_weight
        self.embedding_ema_alpha = embedding_ema_alpha

        self.raw_to_person: dict[int, int] = {}
        self.persons: dict[int, _PersonState] = {}  # ALL identities ever created (<= max_people)
        self.merge_log: list[_MergeEvent] = []
        self.uncertain_log: list[_UncertainEvent] = []
        # raw_track_id -> (candidate_person_id, score) ONLY for the frame
        # of the last resolve() -- same pattern as ReIdentifier.last_uncertain
        # (reid.py), so a caller can add an "identity_uncertain" column
        # to the CSV without this module's return value shape changing.
        self.last_uncertain: dict[int, tuple[int, float]] = {}
        self._next_person_id = 1

    # -- public API -----------------------------------------------------

    def resolve(self, people: list[tuple[int, np.ndarray, np.ndarray, float]],
                now: float, frame: np.ndarray | None = None,
                ) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
        """Translates a frame's (raw_track_id, bbox, poly, conf) list
        into a (person_id, bbox, poly, conf) list, guaranteeing that
        person_id never exceeds `max_people` distinct values in the
        whole session. `frame` optional: if absent, color is not an
        available signal (only position and shape are used)."""
        current_raw_ids = {rid for rid, *_ in people}
        claimed_this_frame: set[int] = {
            self.raw_to_person[rid] for rid in current_raw_ids if rid in self.raw_to_person
        }
        self.last_uncertain = {}

        # -- descriptors for each person in this frame (computed once,
        # reused both for the possible batch match and to update the
        # state at the end of the function) --
        descriptors: dict[int, tuple[np.ndarray, float, tuple, tuple, tuple]] = {}
        for raw_id, bbox, poly, conf in people:
            centroid = mask_centroid(poly)
            area = mask_area(poly)
            scale = float(np.sqrt(area)) if area > 1.0 else float(np.linalg.norm(bbox[2:] - bbox[:2]))
            color = _mask_hue_histogram(frame, poly) if frame is not None else None
            shape = _shape_descriptor(bbox, poly)
            embedding = (self.embedder.embed(frame, bbox, poly=poly)
                         if frame is not None and self.embedder is not None else None)
            descriptors[raw_id] = (centroid, scale, color, shape, embedding)

        # -- batch (Hungarian) resolution of ALL raw_ids never seen
        # before, all at once in this frame, instead of the previous
        # sequential loop (one raw_id at a time, which would grab the
        # preferred slot without considering other new raw_ids in the
        # same frame) -- see identity_manager.resolve_batch(). --
        new_raw_ids = [rid for rid, *_ in people if rid not in self.raw_to_person]
        if new_raw_ids and self.persons:
            self._resolve_new_batch(new_raw_ids, descriptors, now, claimed_this_frame)

        # -- raw_ids still without a person_id after the batch match (no
        # known person in the roster, or no candidate above threshold):
        # free slot -> new id; cap reached -> forced link (the only
        # exception to the module's "never force" rule, see docstring).
        # `known_count` counts the roster AS IT WILL BE after this frame
        # (already-known people + those just minted below): `self.persons`
        # is not updated until resolve()'s final loop (the
        # centroid/color/shape of EVERY person in the frame is needed
        # first, not just the new ones), so without this local counter
        # multiple "new" raw_ids in the same frame would all still see
        # the roster as it was AT THE START of the frame and would
        # exceed the cap together (bug: the previous version updated
        # self.persons inside the same loop, here the two steps are
        # deliberately separate for the Hungarian batch above). --
        known_count = len(self.persons)
        for raw_id in new_raw_ids:
            if raw_id in self.raw_to_person:
                continue
            centroid, scale, color, shape, embedding = descriptors[raw_id]
            if known_count < self.max_people:
                person_id = self._next_person_id
                self._next_person_id += 1
                known_count += 1
            else:
                person_id, score = self._best_match(centroid, scale, color, shape, embedding, now,
                                                      exclude=claimed_this_frame)
                if person_id is None:
                    # pathological case: more "new" raw_ids in this frame
                    # than remaining free slots (e.g. a spurious double
                    # detection). The "never an extra id" constraint has
                    # absolute priority: reuse the best slot anyway, even
                    # if already claimed.
                    person_id, score = self._best_match(centroid, scale, color, shape, embedding,
                                                          now, exclude=set())
                self.merge_log.append(_MergeEvent(
                    raw_track_id=raw_id, matched_person_id=person_id,
                    frame_time=now, score=score, forced=True))
            self.raw_to_person[raw_id] = person_id
            claimed_this_frame.add(person_id)
            # Update the person's state right away (not only at the end,
            # see the `out` loop below): the "pathological case" fallback
            # just above (`_best_match`) and subsequent "new" raw_ids in
            # THIS SAME frame must be able to see people just
            # minted/linked here, not only those already known at the
            # start of the frame -- otherwise (a bug already seen once,
            # see `known_count` above) the fallback would find no
            # candidates and `_best_match` would return `None`.
            self._update_person_state(person_id, centroid, scale, color, shape, embedding, now)

        out = []
        for raw_id, bbox, poly, conf in people:
            centroid, scale, color, shape, embedding = descriptors[raw_id]
            person_id = self.raw_to_person[raw_id]
            self._update_person_state(person_id, centroid, scale, color, shape, embedding, now)
            out.append((person_id, bbox, poly, conf))

        gone_raw_ids = [rid for rid in self.raw_to_person if rid not in current_raw_ids]
        for rid in gone_raw_ids:
            del self.raw_to_person[rid]

        return out

    # -- internal ------------------------------------------------------

    def _update_person_state(self, person_id: int, centroid: np.ndarray, scale: float,
                              color: np.ndarray | None, shape: tuple[float, float],
                              embedding: np.ndarray | None, now: float) -> None:
        """Updates (or creates) `person_id`'s state with this frame's
        descriptor -- an absent color (`None`, see `_mask_hue_histogram`)
        or NaN shape does not overwrite a previously valid value (same
        behavior as before: a frame with an unsamplable color must not
        "forget" the last known good histogram). The embedding does NOT
        replace the previous one but refines it with an exponential
        moving average (`appearance_embedding.ema_update`, see "Optional
        signal: appearance embedding" in the module docstring) -- it
        consolidates the longer the person stays visible, instead of
        depending on a single frame."""
        prev = self.persons.get(person_id)
        self.persons[person_id] = _PersonState(
            position=centroid, scale=scale,
            color=color if color is not None else (prev.color if prev else None),
            shape=shape if not np.isnan(shape).any() else (prev.shape if prev else None),
            embedding=ema_update(prev.embedding if prev else None, embedding, self.embedding_ema_alpha),
            last_seen=now,
        )

    def _resolve_new_batch(self, new_raw_ids: list[int],
                            descriptors: dict[int, tuple[np.ndarray, float, tuple, tuple, tuple]],
                            now: float, claimed_this_frame: set[int]) -> None:
        """"Soft" link (with threshold), ALWAYS attempted first for
        EVERY new raw_id in this frame, even if the max_people cap
        hasn't been reached yet: a new ByteTrack raw_id is very often an
        already-seen person to whom the tracker has just reassigned a
        different id (brief occlusion, exiting/re-entering the frame
        edge), not a truly never-seen-before person (see the module
        docstring for why this is needed even with a cap set wide on
        purpose for margin). Resolved all at once (Hungarian) among all
        new raw_ids of this frame and all known people NOT already
        claimed by a raw_id already existing in this same frame."""
        person_ids = [pid for pid in self.persons if pid not in claimed_this_frame]
        if not person_ids:
            return
        cost_matrix = np.full((len(new_raw_ids), len(person_ids)), np.inf)
        for i, raw_id in enumerate(new_raw_ids):
            centroid, scale, color, shape, embedding = descriptors[raw_id]
            for j, person_id in enumerate(person_ids):
                score = self._pair_score(centroid, scale, color, shape, embedding,
                                          now, self.persons[person_id])
                cost_matrix[i, j] = 1.0 - score

        config = IdentityManagerConfig(
            max_people=self.max_people, flag_uncertain=self.flag_uncertain,
            accept_cost=1.0 - self.soft_match_threshold,
            reject_cost=1.0 - self.soft_match_threshold + self.uncertain_score_margin,
        )
        outcomes = resolve_batch(cost_matrix, person_ids, config)

        for outcome in outcomes:
            raw_id = new_raw_ids[outcome.candidate_index]
            if outcome.status == "matched":
                person_id, score = outcome.matched_lost_id, 1.0 - outcome.cost
                self.raw_to_person[raw_id] = person_id
                claimed_this_frame.add(person_id)
                self.merge_log.append(_MergeEvent(
                    raw_track_id=raw_id, matched_person_id=person_id,
                    frame_time=now, score=score, forced=False))
            elif outcome.status == "uncertain":
                self.uncertain_log.append(_UncertainEvent(
                    raw_track_id=raw_id, candidate_person_id=outcome.matched_lost_id,
                    frame_time=now, score=1.0 - outcome.cost))
                self.last_uncertain[raw_id] = (outcome.matched_lost_id, 1.0 - outcome.cost)
            # "new": no action here, handled by the caller (resolve())
            # by allocating a free slot or a forced link.

    def _pair_score(self, centroid: np.ndarray, scale: float,
                     color: np.ndarray | None, shape: tuple[float, float],
                     embedding: np.ndarray | None,
                     now: float, state: "_PersonState") -> float:
        """0..1 score of ONE new-raw_id/known-person pair (weighted
        average of position/color/shape/embedding) -- extracted from
        `_best_match` so it can also be called when building the batch
        cost matrix (`_resolve_new_batch`)."""
        pos_sim = 0.0
        if not np.isnan(centroid).any() and not np.isnan(state.position).any() and state.scale > 1e-6:
            dist_scales = float(np.linalg.norm(centroid - state.position)) / state.scale
            spatial = max(0.0, 1.0 - dist_scales / self.max_position_dist_scales)
            temporal = max(0.0, 1.0 - (now - state.last_seen) / self.max_position_gap_seconds)
            pos_sim = spatial * temporal

        color_sim = 0.0
        if state.color is not None and color is not None:
            color_sim = _histogram_similarity(color, state.color)

        shape_sim = 0.0
        if state.shape is not None and not np.isnan(shape).any():
            shape_sim = _shape_similarity(shape, state.shape)

        embedding_sim = 0.0
        if state.embedding is not None and embedding is not None:
            sim = embedding_similarity(embedding, state.embedding)
            if sim is not None:
                embedding_sim = sim

        return (self.position_weight * pos_sim + self.color_weight * color_sim
                + self.shape_weight * shape_sim + self.embedding_weight * embedding_sim)

    def _best_match(self, centroid: np.ndarray, scale: float,
                     color: np.ndarray | None, shape: tuple[float, float],
                     embedding: np.ndarray | None,
                     now: float, exclude: set[int]) -> tuple[int | None, float]:
        """Sequential search for the best candidate -- reused ONLY for
        the forced link (max_people cap already reached, see
        resolve()): there the batch pairing isn't needed, every raw_id
        still without a person_id at that point is linked without
        threshold anyway, one at a time, excluding slots already claimed
        in this frame."""
        best_pid, best_score = None, -1.0
        for pid, state in self.persons.items():
            if pid in exclude:
                continue
            score = self._pair_score(centroid, scale, color, shape, embedding, now, state)
            if score > best_score:
                best_pid, best_score = pid, score
        return best_pid, best_score
