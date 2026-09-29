"""
reid.py
=======
Real-time re-identification based on an anthropometric signature, to
recover a person's identity when ByteTrack assigns them a new track_id
after they've fully left the frame (or their appearance changes between
an exit and a re-entry). Same idea as `reid_signature.py` in the CHUV
repository (Video-Annotation-System), readapted for COCO-17 (not
BODY-25) and for real-time operation (decide at the instant a new
track_id appears, against a memory of recently-lost people, instead of
comparing already-finished tracks from a full CSV). Prototyped here
first because that pipeline needs SAM3/CUDA, not runnable on an M1 Mac
-- the signature itself is tracker-independent.

Design (in brief)
------------------
- A never-seen track_id gets a provisional `person_id` immediately (no
  perceptible overlay delay). In parallel, a sliding window of body
  proportions accumulates for it; once enough valid frames exist, a
  median signature is compared against recently-lost people (`lost`,
  expiring after `max_lost_seconds`). No match doesn't mean giving up:
  the window keeps updating and retrying each frame (so a re-entry with
  noisy first frames isn't lost forever, just delayed), bounded by two
  guardrails -- never compare against someone lost AFTER this track had
  already appeared (they were visible together, can't be the same
  person), and give up retrying after `_PENDING_RETRY_SECONDS`.
- On a match under threshold, all frames AFTER that point go to the
  previous `person_id`; already-emitted frames keep the provisional id
  (never rewritten). Every merge is logged (`self.merge_log`) for
  audit/traceability, since there's no human in the loop in real time.

Optional signals (each only ever DISCOUNTS the proportion distance,
never blocks or substitutes for it; when more than one applies, the
STRONGEST discount wins, never the sum of several mediocre ones):
- **Color** (`resolve()` given a frame): average hue+saturation over
  torso, thigh, and above-the-ears (hair) regions, sampled from the
  keypoint-defined polygons. Hair color specifically helps when someone
  re-enters in different clothes.
- **Position**: last known position (hip center) + scale (torso
  length) at loss, compared against a new track's first position within
  `MAX_POSITION_DIST_TORSOS` and `MAX_POSITION_GAP_SECONDS` (both must
  hold). Targets brief in-scene occlusions (e.g. someone dressing a
  child) where the person barely moved.
- **Appearance embedding (OSNet)**: unlike color/position (computed
  once at match time), an active person's embedding updates every frame
  via EMA (`self.embedding_ema`, see `appearance_embedding.ema_update`)
  so a later re-entry compares against a stabilized estimate. Needs an
  `embedder` passed to the constructor; `None` (default) simply omits
  the signal.

`max_people` (closed session): the one signal that FORCES a merge
rather than discounting. If the known headcount is already fully
confirmed and a new track matches no one under normal thresholds, it
can't be a new person -- it's forced onto the closest currently-"lost"
person (even above `max_signature_dist`). Guardrails: still respects
the causality rule (never force against someone visible together with
this track), and only forces against people currently lost, never
against someone else active right now. If the cap is reached with no
one lost to recover, forcing is abandoned and a new id is minted with a
warning printed (a signal the configured count or `max_people` filter
upstream may be wrong).

Honest limitations
-------------------
- Needs a minimum number of confident-keypoint frames; a very brief
  appearance never gets a reliable signature.
- Similar-build false positives are possible; the default threshold is
  conservative but not validated on real data.
- Color and position are both deliberate trade-offs, not hidden bugs:
  color helps only while clothes stay the same and can raise
  false-positive risk between similarly-dressed, similarly-built
  people; position can push an already-ambiguous match the wrong way in
  scenes with two people close together (e.g. child+caregiver).
- Merges apply automatically with no real-time human confirmation --
  hence the audit log.
- `max_people` forcing can misattribute frames if the configured count
  is wrong (e.g. an unexpected extra adult); use only when the
  participant count is genuinely fixed and known.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from pose.keypoints import KP
from pose.features import torso_length, hip_center
from pose.identity_manager import (
    IdentityMode, SessionMode, IdentityManagerConfig, resolve_batch,
    suggested_max_people_policy,
)
from pose.appearance_embedding import OSNetEmbedder, embedding_similarity, ema_update

# ---------------------------------------------------------------------------
# Anthropometric signature (adapted from reid_signature.py, COCO-17 schema)
# ---------------------------------------------------------------------------

SIGNATURE_SEGMENTS: dict[str, tuple[str, str]] = {
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "hip_width": ("left_hip", "right_hip"),
    "upper_arm_l": ("left_shoulder", "left_elbow"),
    "upper_arm_r": ("right_shoulder", "right_elbow"),
    "forearm_l": ("left_elbow", "left_wrist"),
    "forearm_r": ("right_elbow", "right_wrist"),
    "thigh_l": ("left_hip", "left_knee"),
    "thigh_r": ("right_hip", "right_knee"),
    "shin_l": ("left_knee", "left_ankle"),
    "shin_r": ("right_knee", "right_ankle"),
    # head geometry: as independent of clothing as body proportions,
    # same equal treatment as the other segments (no special weight) --
    # adds signal when body/arms are noisy.
    "eye_to_eye": ("left_eye", "right_eye"),
    "ear_to_ear": ("left_ear", "right_ear"),
}
SIGNATURE_COLS = list(SIGNATURE_SEGMENTS.keys())


def _segment_length(kxy: np.ndarray, a_name: str, b_name: str) -> float:
    a, b = kxy[KP[a_name]], kxy[KP[b_name]]
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    return float(np.linalg.norm(a - b))


def compute_signature_frame(kxy: np.ndarray) -> np.ndarray:
    """Body proportions normalized by torso for a single frame (array in
    the order of `SIGNATURE_COLS`, NaN where not computable). Reuses
    `features.torso_length` (already used for self-touch/vertical
    excursion) as the scale unit, so the signature is invariant to
    distance from the camera.
    """
    torso = torso_length(kxy)
    out = np.full(len(SIGNATURE_COLS), np.nan)
    if torso < 1e-6 or np.isnan(torso):
        return out
    for i, (a_name, b_name) in enumerate(SIGNATURE_SEGMENTS.values()):
        seg = _segment_length(kxy, a_name, b_name)
        out[i] = seg / torso if not np.isnan(seg) else np.nan
    return out


def signature_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """RMS distance between two signatures, only on the dimensions valid
    in both. `None` if too few dimensions remain to be trusted."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 3:
        return None
    return float(np.linalg.norm(a[valid] - b[valid]) / np.sqrt(valid.sum()))


# ---------------------------------------------------------------------------
# Shirt/pants color (optional signal, complementary to the anthropometric
# signature -- see "Optional signal" in the module docstring)
# ---------------------------------------------------------------------------

COLOR_SEGMENTS: dict[str, tuple[str, str, str, str]] = {
    "shirt": ("left_shoulder", "right_shoulder", "right_hip", "left_hip"),
    "pants": ("left_hip", "right_hip", "right_knee", "left_knee"),
}
COLOR_COLS = ["shirt_h", "shirt_s", "pants_h", "pants_s", "hair_h", "hair_s"]
_HUE_IDX = [0, 2, 4]   # circular indices (0..1) in COLOR_COLS
_SAT_IDX = [1, 3, 5]   # linear indices (0..1) in COLOR_COLS


def _region_mean_hs(frame: np.ndarray, pts: np.ndarray) -> tuple[float, float]:
    """Average hue/saturation (OpenCV HSV, then normalized 0-1) of the
    pixels inside the polygon defined by the 4 given (x, y) points.
    (nan, nan) if a point is missing (NaN) or the polygon is
    degenerate/too small."""
    if np.isnan(pts).any():
        return np.nan, np.nan
    h, w = frame.shape[:2]
    poly = np.round(pts).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if cv2.countNonZero(mask) < 9:  # polygon too small to be trusted
        return np.nan, np.nan
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_h, mean_s = cv2.mean(hsv, mask=mask)[:2]
    return float(mean_h) / 180.0, float(mean_s) / 255.0


def hair_corners(kxy: np.ndarray) -> np.ndarray:
    """Four approximate corners of the region above the ears (hair color
    proxy, independent of clothing): ear-to-ear width, extended upward
    by the neck-to-nose distance (proportional to head size, invariant
    to distance from the camera). NaN if a source keypoint is missing --
    same treatment as the other color regions, never a made-up
    fallback."""
    neck = (kxy[KP["left_shoulder"]] + kxy[KP["right_shoulder"]]) / 2.0
    nose, l_ear, r_ear = kxy[KP["nose"]], kxy[KP["left_ear"]], kxy[KP["right_ear"]]
    if np.isnan(np.array([neck, nose, l_ear, r_ear])).any():
        return np.full((4, 2), np.nan)
    up = nose - neck
    return np.array([r_ear, l_ear, l_ear + up, r_ear + up])


def compute_color_signature(frame: np.ndarray, kxy: np.ndarray) -> np.ndarray:
    """Average color (hue, saturation) of shirt, pants and hair for a
    single frame, in the order of `COLOR_COLS`, NaN where not
    samplable. Hue/Saturation only (not Value/brightness): more robust
    to exposure/lighting changes between an exit and a re-entry."""
    out = np.full(len(COLOR_COLS), np.nan)
    for i, corner_names in enumerate(COLOR_SEGMENTS.values()):
        pts = kxy[[KP[name] for name in corner_names]]
        out[2 * i], out[2 * i + 1] = _region_mean_hs(frame, pts)
    out[4], out[5] = _region_mean_hs(frame, hair_corners(kxy))
    return out


def color_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    """0..1 similarity (1 = identical color) between two color
    signatures, on the dimensions valid in both. Circular hue distance
    (the Hue scale "wraps around" at 1.0). `None` if too few valid
    dimensions remain."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 2:
        return None
    dist = np.empty(len(a))
    diff = np.abs(a - b)
    dist[_HUE_IDX] = np.minimum(diff[_HUE_IDX], 1.0 - diff[_HUE_IDX])
    dist[_SAT_IDX] = diff[_SAT_IDX]
    return float(np.clip(1.0 - dist[valid].mean(), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Position (optional signal, complementary to the anthropometric signature
# -- see "Optional signal: position" in the module docstring)
# ---------------------------------------------------------------------------

# Spatial radius (in torso lengths, so invariant to distance from the
# camera) beyond which position no longer counts as "the same spot". 4
# torso lengths covers both a re-entry through the same door (small
# displacement) and a child standing still while being put in a jacket
# (near-zero displacement).
MAX_POSITION_DIST_TORSOS = 4.0

# Time beyond which the positional signal drops to zero. A jacket
# change typically lasts about ten seconds: at 20s the weight is zero,
# at 10s it's still at half -- not at the edge of validity.
MAX_POSITION_GAP_SECONDS = 20.0


def position_similarity(pos_a: np.ndarray, pos_b: np.ndarray, torso_scale: float,
                         gap_seconds: float) -> float | None:
    """0..1 positional similarity (1 = same spot, just happened) between
    two positions (hip center), combining spatial proximity (in torso
    lengths, linear decay up to `MAX_POSITION_DIST_TORSOS`) and temporal
    proximity (linear decay up to `MAX_POSITION_GAP_SECONDS`). `None` if
    a position is missing or the scale isn't valid."""
    if np.isnan(pos_a).any() or np.isnan(pos_b).any():
        return None
    if torso_scale < 1e-6 or np.isnan(torso_scale):
        return None
    dist_torsos = float(np.linalg.norm(pos_a - pos_b)) / torso_scale
    spatial = max(0.0, 1.0 - dist_torsos / MAX_POSITION_DIST_TORSOS)
    temporal = max(0.0, 1.0 - gap_seconds / MAX_POSITION_GAP_SECONDS)
    return spatial * temporal


# ---------------------------------------------------------------------------
# Appearance embedding (optional signal, see "Optional signal: appearance
# embedding (OSNet)" in the module docstring)
# ---------------------------------------------------------------------------

# Fraction by which the keypoint-derived bbox is expanded: the COCO-17
# skeleton covers joints, not the outline of clothing (a loose shirt
# extends past the shoulders) -- without this margin, the crop passed
# to OSNet would systematically cut off the garment at the edges,
# exactly the signal it's meant to capture.
_EMBED_BBOX_PAD_FRAC = 0.25
_EMBED_MIN_VALID_JOINTS = 4


def _bbox_from_keypoints(kxy: np.ndarray, pad_frac: float = _EMBED_BBOX_PAD_FRAC) -> np.ndarray | None:
    """Bbox (x1, y1, x2, y2) enclosing the valid keypoints, expanded by
    `pad_frac` per side -- used only to crop the person to pass to
    `OSNetEmbedder.embed()`, for no other purpose (no "real" detection
    box is available in this pipeline, only keypoints). `None` if too
    few valid keypoints remain for a reliable bbox."""
    valid = ~np.isnan(kxy).any(axis=1)
    if valid.sum() < _EMBED_MIN_VALID_JOINTS:
        return None
    pts = kxy[valid]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w, h = x2 - x1, y2 - y1
    if w < 1e-6 or h < 1e-6:
        return None
    return np.array([x1 - pad_frac * w, y1 - pad_frac * h, x2 + pad_frac * w, y2 + pad_frac * h])


# ---------------------------------------------------------------------------
# Per-person/track state
# ---------------------------------------------------------------------------

@dataclass
class _LostPerson:
    signature: np.ndarray
    lost_time: float
    color: np.ndarray | None = None
    position: np.ndarray | None = None    # last known hip center
    last_torso: float | None = None       # scale to normalize position
    embedding: np.ndarray | None = None   # OSNet, frozen at loss time (EMA average up to that point)


@dataclass
class _MergeEvent:
    raw_track_id: int
    provisional_person_id: int
    matched_person_id: int
    distance: float
    frame_time: float
    color_used: bool = False
    position_used: bool = False
    embedding_used: bool = False
    forced: bool = False  # match forced by max_people (see module docstring)


@dataclass
class _UncertainEvent:
    """Candidate in a "gray zone" (see identity_manager.resolve_batch):
    the Hungarian algorithm proposed it as the best available pairing,
    but the distance exceeds `max_signature_dist` while still staying
    below `uncertain_signature_dist` -- NOT merged automatically, only
    flagged (policy "better to fragment an identity than to silently
    swap two"). `raw_track_id` keeps being retried on subsequent frames
    like any provisional in `pending_check`, until the retry expires or
    it finds a confirmed match."""
    raw_track_id: int
    provisional_person_id: int
    candidate_person_id: int
    distance: float
    frame_time: float


class ReIdentifier:
    """Maintains the track_id (ByteTrack) -> person_id (stable
    identity) correspondence, re-associating people who re-enter the
    frame based on their anthropometric signature.

    Usage in `live_demo.py` (a single wiring point):

        reidentifier = ReIdentifier()
        ...
        for frame_result in tracker.run(...):
            people = reidentifier.resolve(frame_result.people, now=time.time(),
                                           frame=frame_result.frame)  # frame optional
            # 'people' has the same shape as frame_result.people
            # [(id, kxy, kconf), ...], but id is now a stable person_id
    """

    # Maximum time (from the appearance of a never-before-seen track)
    # during which the match against "lost" people is retried: beyond
    # this threshold, the track is accepted as a new identity forever
    # instead of being re-checked every frame for the rest of the
    # session (see the note in the module docstring on why a limit is
    # needed, not just "retry until you find something").
    _PENDING_RETRY_SECONDS = 2.0

    def __init__(self, max_lost_seconds: float = 180.0,
                 max_signature_dist: float = 0.12,
                 min_signature_frames: int = 15,
                 color_bonus_weight: float = 0.5,
                 position_bonus_weight: float = 0.5,
                 max_people: int | None = None,
                 session_mode: SessionMode = SessionMode.MULTIPLE,
                 flag_uncertain: bool = True,
                 uncertain_signature_dist: float | None = None,
                 embedder: OSNetEmbedder | None = None,
                 embedding_bonus_weight: float = 0.7,
                 embedding_ema_alpha: float = 0.9):
        """`session_mode` (new): SINGLE forces `max_people=1` (see
        `identity_manager.suggested_max_people_policy` -- only one
        person expected, every re-entry is by construction that person,
        recovery can be as permissive as possible), overriding
        `max_people` if in conflict. MULTIPLE (default) respects
        `max_people` as passed (can remain `None`).

        `flag_uncertain` / `uncertain_signature_dist` (new): with
        `flag_uncertain=True` (default), a candidate with distance
        between `max_signature_dist` and `uncertain_signature_dist`
        (default: 1.6x `max_signature_dist` if unspecified) is not
        merged but recorded in `self.uncertain_log`/`self.last_uncertain`
        -- see `_UncertainEvent`. With `flag_uncertain=False` behavior
        reverts to binary as before this module (match/no-match at
        `max_signature_dist`).

        `embedder` (new, optional): an `OSNetEmbedder` instance (see
        `pose/appearance_embedding.py`) -- if passed, every frame also
        computes an appearance embedding for the active person,
        maintained with an exponential moving average
        (`embedding_ema_alpha`, the StrongSORT idea cited in the module
        docstring) and used as a third possible discount
        (`embedding_bonus_weight`, typically the most reliable of the
        three -- higher default weight than color/position) in the same
        "the strongest wins" scheme as `_pair_cost`. If `None`
        (default), no behavior different from before."""
        self.max_lost_seconds = max_lost_seconds
        self.max_signature_dist = max_signature_dist
        self.min_signature_frames = min_signature_frames
        self.color_bonus_weight = color_bonus_weight
        self.position_bonus_weight = position_bonus_weight
        self.session_mode = session_mode
        self.max_people = suggested_max_people_policy(session_mode, max_people)
        self.flag_uncertain = flag_uncertain
        self.uncertain_signature_dist = (
            uncertain_signature_dist if uncertain_signature_dist is not None
            else max_signature_dist * 1.6
        )
        self.embedder = embedder
        self.embedding_bonus_weight = embedding_bonus_weight
        self.embedding_ema_alpha = embedding_ema_alpha

        self.raw_to_person: dict[int, int] = {}
        self.buffers: dict[int, deque] = {}
        self.color_buffers: dict[int, deque] = {}
        self.pending_check: set[int] = set()  # raw_track_id still to be evaluated
        self.pending_since: dict[int, float] = {}  # raw_track_id -> now at first sighting
        self.last_position: dict[int, tuple[np.ndarray, float]] = {}  # person_id -> (hip_xy, torso)
        self.embedding_ema: dict[int, np.ndarray] = {}  # person_id -> OSNet embedding (EMA average)
        self.lost: dict[int, _LostPerson] = {}
        self.merge_log: list[_MergeEvent] = []
        self.uncertain_log: list[_UncertainEvent] = []
        # raw_track_id -> (candidate_person_id, distance) ONLY for the
        # frame of the last resolve(): a caller who wants to add columns
        # like "identity_uncertain"/"reid_uncertain_candidate_id" to the
        # CSV can read it right after resolve() without this module
        # needing to change the shape of the return value (stays
        # [(person_id, kxy, kconf), ...], unchanged for all existing
        # callers).
        self.last_uncertain: dict[int, tuple[int, float]] = {}
        self._next_person_id = 1

    # -- public API ---------------------------------------------------

    def resolve(self, people: list[tuple[int, np.ndarray, np.ndarray]],
                now: float, frame: np.ndarray | None = None) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Translates a frame's (raw_track_id, kxy, kconf) list into a
        (person_id, kxy, kconf) list, re-associating identities when
        possible. Must be called once per frame, with ALL people
        detected in that frame (even just to keep the internal state of
        who is still present up to date).

        `frame` is optional: if passed (the current BGR frame), the
        shirt/pants color is also used as a signal complementary to the
        anthropometric signature (see module docstring). If omitted,
        behavior is identical to the version based only on body
        proportions.
        """
        current_raw_ids = set()
        self.last_uncertain = {}

        # -- pass 1: per-track bookkeeping (identical to before) + list
        # of candidates "ready" for a match attempt in this frame --
        ready: list[int] = []  # raw_track_id
        for raw_id, kxy, kconf in people:
            current_raw_ids.add(raw_id)

            if raw_id not in self.raw_to_person:
                self._assign_provisional(raw_id, now)

            person_id = self.raw_to_person[raw_id]
            self.buffers[person_id].append(compute_signature_frame(kxy))
            if frame is not None:
                self.color_buffers[person_id].append(compute_color_signature(frame, kxy))
                if self.embedder is not None:
                    bbox = _bbox_from_keypoints(kxy)
                    new_embedding = self.embedder.embed(frame, bbox) if bbox is not None else None
                    self.embedding_ema[person_id] = ema_update(
                        self.embedding_ema.get(person_id), new_embedding, self.embedding_ema_alpha)

            hip, torso = hip_center(kxy), torso_length(kxy)
            if not np.isnan(hip).any() and not np.isnan(torso) and torso > 1e-6:
                self.last_position[person_id] = (hip, torso)

            if raw_id in self.pending_check and len(self.buffers[person_id]) >= self.min_signature_frames:
                ready.append(raw_id)

        # -- pass 2: batch (Hungarian, see identity_manager.py)
        # resolution of ALL ready candidates in this frame against ALL
        # lost identities at once -- instead of the previous sequential
        # loop (one candidate at a time, which would grab its own
        # preferred lost identity without considering the other ready
        # candidates in the same frame). --
        if ready:
            self._resolve_ready_batch(ready, now)

        # -- pass 3: builds the output with the ids (person_id possibly
        # already updated by the match in pass 2) --
        out = [(self.raw_to_person[raw_id], kxy, kconf) for raw_id, kxy, kconf in people]

        self._retire_disappeared_tracks(current_raw_ids, now)
        self._expire_old_lost_people(now)
        return out

    # -- internal ----------------------------------------------------------

    def _assign_provisional(self, raw_id: int, now: float) -> None:
        person_id = self._next_person_id
        self._next_person_id += 1
        self.raw_to_person[raw_id] = person_id
        self.buffers[person_id] = deque(maxlen=self.min_signature_frames)
        self.color_buffers[person_id] = deque(maxlen=self.min_signature_frames)
        self.pending_check.add(raw_id)
        self.pending_since[raw_id] = now

    def _pair_cost(self, median_sig: np.ndarray, median_color: np.ndarray | None,
                    current_position: tuple[np.ndarray, float] | None,
                    current_embedding: np.ndarray | None,
                    lost: "_LostPerson", now: float) -> tuple[float | None, bool, bool, bool]:
        """Cost (distance) of ONE candidate/lost-identity pair -- same
        discount logic as before (color, position, or appearance
        embedding, the STRONGEST of the three, never summed; none can
        EVER raise the distance, see the module docstring), extracted
        here to be callable both when building the batch cost matrix
        (`_resolve_ready_batch`) and, if needed, for a single
        comparison. Doesn't apply the causality rule (done by the
        caller, which knows `pending_since`)."""
        prop_dist = signature_distance(median_sig, lost.signature)
        if prop_dist is None:
            return None, False, False, False

        color_bonus, color_used = 0.0, False
        if median_color is not None and lost.color is not None:
            sim = color_similarity(median_color, lost.color)
            if sim is not None:
                color_bonus, color_used = self.color_bonus_weight * sim, True

        position_bonus, position_used = 0.0, False
        if current_position is not None and lost.position is not None:
            cur_xy, cur_torso = current_position
            scale = cur_torso if lost.last_torso is None else (cur_torso + lost.last_torso) / 2.0
            sim = position_similarity(cur_xy, lost.position, scale, now - lost.lost_time)
            if sim is not None:
                position_bonus, position_used = self.position_bonus_weight * sim, True

        embedding_bonus, embedding_used = 0.0, False
        if current_embedding is not None and lost.embedding is not None:
            sim = embedding_similarity(current_embedding, lost.embedding)
            if sim is not None:
                embedding_bonus, embedding_used = self.embedding_bonus_weight * sim, True

        best_bonus = max(color_bonus, position_bonus, embedding_bonus)
        if best_bonus == 0.0:
            return prop_dist, False, False, False
        if best_bonus == embedding_bonus:
            return prop_dist * (1.0 - embedding_bonus), False, False, embedding_used
        if best_bonus == color_bonus:
            return prop_dist * (1.0 - color_bonus), color_used, False, False
        return prop_dist * (1.0 - position_bonus), False, position_used, False

    def _resolve_ready_batch(self, ready: list[int], now: float) -> None:
        """Builds the ready-candidates x lost-identities cost matrix and
        solves it all at once with `identity_manager.resolve_batch`
        (Hungarian algorithm + accept/reject/gray-zone thresholds).
        Then applies the outcomes: `matched` -> merge (identical to
        before, see `_apply_merge`); `uncertain` -> flagged in
        `uncertain_log`/`last_uncertain`, NO merge, the candidate stays
        in `pending_check` and will be retried next frame; `new` ->
        same, stays pending until `_PENDING_RETRY_SECONDS` expires, at
        which point (if `max_people` is set) the last-resort forcing is
        attempted (`_force_match_at_capacity`, unchanged -- the module's
        only exception to "never force")."""
        lost_ids = list(self.lost.keys())
        cost_matrix = np.full((len(ready), max(len(lost_ids), 1)), np.inf)
        pair_flags: dict[tuple[int, int], tuple[bool, bool, bool]] = {}

        if lost_ids:
            for i, raw_id in enumerate(ready):
                person_id = self.raw_to_person[raw_id]
                median_sig = np.nanmedian(np.array(self.buffers[person_id]), axis=0)
                color_buffer = self.color_buffers.get(person_id)
                median_color = (np.nanmedian(np.array(color_buffer), axis=0)
                                 if color_buffer else None)
                current_position = self.last_position.get(person_id)
                current_embedding = self.embedding_ema.get(person_id)
                pending_since = self.pending_since[raw_id]
                for j, lost_person_id in enumerate(lost_ids):
                    lost = self.lost[lost_person_id]
                    if lost.lost_time > pending_since:
                        continue  # causality rule -- stays inf
                    cost, color_used, position_used, embedding_used = self._pair_cost(
                        median_sig, median_color, current_position, current_embedding, lost, now)
                    if cost is None:
                        continue
                    cost_matrix[i, j] = cost
                    pair_flags[(i, j)] = (color_used, position_used, embedding_used)

        config = IdentityManagerConfig(
            max_people=self.max_people, flag_uncertain=self.flag_uncertain,
            accept_cost=self.max_signature_dist, reject_cost=self.uncertain_signature_dist,
        )
        outcomes = resolve_batch(cost_matrix, lost_ids, config)

        still_pending: list[int] = []
        for i, outcome in enumerate(outcomes):
            raw_id = ready[i]
            person_id = self.raw_to_person[raw_id]
            if outcome.status == "matched":
                j = lost_ids.index(outcome.matched_lost_id)
                color_used, position_used, embedding_used = pair_flags.get((i, j), (False, False, False))
                self._apply_merge(raw_id, person_id, outcome.matched_lost_id, outcome.cost,
                                   now, color_used, position_used, embedding_used, forced=False)
            elif outcome.status == "uncertain":
                self.uncertain_log.append(_UncertainEvent(
                    raw_track_id=raw_id, provisional_person_id=person_id,
                    candidate_person_id=outcome.matched_lost_id, distance=outcome.cost, frame_time=now))
                self.last_uncertain[raw_id] = (outcome.matched_lost_id, outcome.cost)
                still_pending.append(raw_id)
            else:  # "new"
                still_pending.append(raw_id)

        for raw_id in still_pending:
            if now - self.pending_since[raw_id] <= self._PENDING_RETRY_SECONDS:
                continue  # still retried on subsequent frames, see above
            self.pending_check.discard(raw_id)
            if self.max_people is None:
                continue
            person_id = self.raw_to_person[raw_id]
            match = self._force_match_at_capacity(
                self.buffers[person_id], self.pending_since[raw_id], person_id)
            if match is not None:
                matched_person_id, dist, color_used, position_used, forced = match
                self._apply_merge(raw_id, person_id, matched_person_id, dist,
                                   now, color_used, position_used, False, forced=forced)

    def _apply_merge(self, raw_id: int, person_id: int, matched_person_id: int,
                      dist: float, now: float, color_used: bool, position_used: bool,
                      embedding_used: bool, forced: bool) -> None:
        self.pending_check.discard(raw_id)
        self.merge_log.append(_MergeEvent(
            raw_track_id=raw_id, provisional_person_id=person_id,
            matched_person_id=matched_person_id, distance=dist, frame_time=now,
            color_used=color_used, position_used=position_used,
            embedding_used=embedding_used, forced=forced,
        ))
        assist = ", ".join(s for s, used in (("color", color_used), ("position", position_used),
                                              ("embedding", embedding_used)) if used)
        tag = "FORCED (max_people reached)" if forced else (f'{assist}-assisted' if assist else None)
        print(f"[reid] track {raw_id}: provisional person_id {person_id} "
              f"re-matched to person_id {matched_person_id} "
              f"(distance={dist:.3f}{f', {tag}' if tag else ''})")
        del self.lost[matched_person_id]
        del self.buffers[person_id]
        self.color_buffers.pop(person_id, None)
        self.raw_to_person[raw_id] = matched_person_id
        # The EMA embedding keeps evolving on the restored identity: it
        # resumes from the just-merged provisional's (the most recent
        # estimate available), not from the one frozen at the moment of
        # loss -- consistent with "staying in memory and refining over
        # time" (see appearance_embedding.ema_update).
        provisional_embedding = self.embedding_ema.pop(person_id, None)
        if provisional_embedding is not None:
            self.embedding_ema[matched_person_id] = provisional_embedding
        # new buffer for the restored identity: the provisional one was
        # just deleted, but a buffer is needed to keep accumulating the
        # signature in case this person disappears again later.
        self.buffers[matched_person_id] = deque(maxlen=self.min_signature_frames)
        self.color_buffers[matched_person_id] = deque(maxlen=self.min_signature_frames)

    def _force_match_at_capacity(self, buffer: deque, pending_since: float,
                                  pending_person_id: int,
                                  ) -> tuple[int, float, bool, bool, bool] | None:
        """Last resort, only when `max_people` is set: if, WITHOUT
        COUNTING the provisional person being evaluated, the number of
        known identities (active + lost) has already reached the cap,
        then this can't be an extra person -- it must be one of those
        currently "lost". Picks the closest by signature even above
        `max_signature_dist` (see "Optional signal: maximum number of
        people" in the module docstring). Never forces against
        currently active people, only against lost ones, and respects
        the same causality rule as normal matches. No eligible
        candidate -> None (gives up, doesn't force emptily).

        The count deliberately excludes `pending_person_id`: if it
        didn't, even the very first appearance ever of a genuinely new
        person (e.g. the second person of a 1v1 session, never lost by
        anyone) would trigger as "at the cap" as soon as its own retry
        expires, risking a wrong forced merge with whoever happens to be
        lost at that moment -- the cap must refer to HOW MANY OTHER
        identities already exist, not count itself.
        """
        roster_size = len((set(self.raw_to_person.values()) | set(self.lost.keys()))
                           - {pending_person_id})
        if roster_size < self.max_people:
            return None
        if not self.lost:
            print(f"[reid] warning: max_people={self.max_people} reached but no "
                  f"'lost' person to recover -- minting a new person_id anyway "
                  f"(check the configured number, or a spurious detection got past "
                  f"pose_estimation.py's max_people filter)")
            return None

        median_sig = np.nanmedian(np.array(buffer), axis=0)
        best_person_id, best_dist = None, None
        for person_id, lost in self.lost.items():
            if lost.lost_time > pending_since:
                continue  # same causality rule as normal matches
            prop_dist = signature_distance(median_sig, lost.signature)
            if prop_dist is None:
                continue
            if best_dist is None or prop_dist < best_dist:
                best_person_id, best_dist = person_id, prop_dist

        if best_person_id is None:
            print(f"[reid] warning: max_people={self.max_people} reached but no "
                  f"eligible candidate (causality rule) -- minting a new "
                  f"person_id anyway")
            return None
        return (best_person_id, best_dist, False, False, True)

    def _retire_disappeared_tracks(self, current_raw_ids: set[int], now: float) -> None:
        gone_raw_ids = [rid for rid in self.raw_to_person if rid not in current_raw_ids]
        for raw_id in gone_raw_ids:
            person_id = self.raw_to_person.pop(raw_id)
            self.pending_check.discard(raw_id)
            self.pending_since.pop(raw_id, None)
            buffer = self.buffers.pop(person_id, None)
            color_buffer = self.color_buffers.pop(person_id, None)
            last_pos = self.last_position.pop(person_id, None)
            # The EMA embedding accumulated while the person was active
            # is "frozen" into _LostPerson.embedding exactly like
            # signature/color/position -- removed from
            # self.embedding_ema (no longer an active person to update
            # frame by frame).
            last_embedding = self.embedding_ema.pop(person_id, None)
            if buffer and len(buffer) > 0:
                median_sig = np.nanmedian(np.array(buffer), axis=0)
                if not np.isnan(median_sig).all():
                    median_color = None
                    if color_buffer and len(color_buffer) > 0:
                        cand = np.nanmedian(np.array(color_buffer), axis=0)
                        if not np.isnan(cand).all():
                            median_color = cand
                    position, last_torso = (last_pos if last_pos is not None else (None, None))
                    self.lost[person_id] = _LostPerson(
                        signature=median_sig, lost_time=now, color=median_color,
                        position=position, last_torso=last_torso, embedding=last_embedding)

    def _expire_old_lost_people(self, now: float) -> None:
        expired = [pid for pid, lost in self.lost.items()
                   if now - lost.lost_time > self.max_lost_seconds]
        for pid in expired:
            del self.lost[pid]
