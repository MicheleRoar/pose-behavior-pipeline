"""
reid.py
=======
Real-time re-identification based on an anthropometric signature, to
recover a person's identity when ByteTrack assigns them a new track_id
after they've fully left the frame (or when their appearance changes,
e.g. different clothes, between an exit and a re-entry in the same
session).

Same idea as `reid_signature.py` in the CHUV repository
(Video-Annotation-System, see that module for the full context), here
readapted along two axes:
  - COCO-17 schema (this pipeline) instead of BODY-25 (psifx/OpenPose);
  - REAL-TIME operation instead of batch: there, already-finished
    tracks are compared by reading a complete CSV; here we need to
    decide "is this a never-seen person or an already-seen one?" at the
    instant a new track_id appears, against a memory of people who
    recently disappeared from the frame.

Why prototype it here and not in the CHUV repository: that pipeline
requires SAM3 on a CUDA GPU, not runnable on an M1 Mac -- and that's a
hardware limitation, not one of the strategy. The anthropometric
signature as a re-identification signal is independent of the
underlying tracker (ByteTrack here, SAM3 there); it makes sense to
validate it where iteration is fast, on unprotected data, before
proposing it again for the CHUV repository.

Design (in brief)
------------------
- Every time ByteTrack presents a track_id NEVER seen before, it's
  immediately assigned a provisional `person_id` (no perceptible delay
  in the live overlay).
- In parallel, a small buffer (sliding window) of body proportions
  accumulates for that track_id. Once enough valid frames are
  collected, a median signature is computed and compared against
  recently disappeared people (`lost`, expiring after
  `max_lost_seconds`). If there's no match, the attempt does NOT stop
  immediately: at every subsequent frame the window updates (older
  frames drop out, new ones come in) and it retries -- a re-entry with
  the first frames noisy (person still at the edges of the frame) is
  thus not lost forever, only delayed until the data is clean enough.
  Two guardrails, however, keep the retry from becoming dangerous: (1)
  a track is never compared against a "lost" person AFTER the track
  itself had already appeared -- the two were visible together (under
  different raw_ids), so they can't be the same person (otherwise
  whoever is present from the very start of the session would risk
  being coincidentally linked to an identity lost much later); (2) the
  retry gives up anyway after `_PENDING_RETRY_SECONDS` from the track's
  appearance, so as not to keep re-checking it forever.
- If there's a match under threshold, ALL frames AFTER that point are
  attributed to the previous `person_id` instead of the new one -- the
  frames already emitted (CSV/overlay) with the provisional person_id
  are NOT rewritten. The merge event is explicitly logged
  (`self.merge_log`) for transparency/audit, so a later analysis can
  reconnect the two pieces if needed -- same "propose/make traceable,
  don't decide silently" philosophy as reid_signature.py, adapted to
  the fact that there's no human in the loop here in real time.

Optional signal: shirt/pants/hair color
----------------------------------------------------
The anthropometric signature alone, in practice, can be too weak when
keypoints are noisy (partial occlusions, person at the edges of the
frame during exit/re-entry): two slightly-off body proportions can
cause a real match to be missed. If the video frame is passed to
`resolve()`, an average color (hue + saturation, not brightness -- more
robust to exposure changes) is ALSO computed over three regions: torso
(shirt), thigh (pants) -- sampled from the pixels inside the polygon
defined by the shoulder/hip/knee keypoints -- and a region above the
ears (hair color proxy, estimated by extending the ear-to-ear width
upward by the neck-to-nose distance). Hair color is independent of
clothing: it specifically helps in the case where a person re-enters in
different clothes (e.g. without a shirt).

Color does NOT replace the proportions nor ever raise the rejection
threshold: if the color is very different (e.g. clothes changed between
exit and re-entry) the match is decided exactly as before, based on
proportions alone. If instead the color is similar (the most common
case: same clothes during the session), the distance between
proportions is "discounted" -- making it easier to recover a real match
even with somewhat noisy proportions, without ever worsening the robust
clothing-invariance that was the prototype's original goal.

Optional signal: position
------------------------------
Designed for the concrete case of an "in-scene" clothing change (e.g.
putting a jacket/apron on a child during an activity, typically about
ten seconds): the child never leaves the frame, but the partial
occlusion by whoever is dressing them can lose the track and generate a
new one at separation. In this case the child has barely moved, so the
last known position (hip center) before the loss and the new track's
first position are very close -- the same principle also applies to
exiting/re-entering through the same door. Every frame, the current
person's position and scale (torso length) are stored; on loss, that
position is frozen alongside the signature. A new track is compared on
spatial proximity (in torso lengths, `MAX_POSITION_DIST_TORSOS`) AND
temporal proximity (`MAX_POSITION_GAP_SECONDS`) -- both must count for
something, one alone isn't enough.

Same principle as color: position can only discount the distance,
NEVER block a match nor substitute for the signature. Color and
position don't add up -- the STRONGEST of the two available discounts
is taken, not the sum, to avoid two only-mediocre signals accumulating
into a confidence neither would justify alone.

Optional signal: appearance embedding (OSNet)
--------------------------------------------------
A third possible discount, same "the strongest wins, never the sum"
principle as color/position above -- see `pose/appearance_embedding.py`
for the module and why it's a real embedding (OSNet) and not just
another heuristic. Unlike color/position (computed once from the median
buffer at match time), an ACTIVE person's embedding is updated every
frame with an exponential moving average (`self.embedding_ema`, see
`appearance_embedding.ema_update`) -- the StrongSORT idea cited there:
the longer a person stays visible, the more their stored appearance
signature stabilizes, so a later re-entry can be compared against a
consolidated estimate instead of a single (often noisy: motion blur,
pose, partial occlusion) frame. Requires an `embedder` (an
`OSNetEmbedder` instance) passed to the constructor -- if `None`
(default), the signal is simply absent, no different behavior from the
rest of the module.

Optional signal: maximum number of people (closed session)
------------------------------------------------------------------
When the number of people who can appear in the session is known ahead
of time (e.g. 2 for a 1v1 child-caregiver session, up to about a dozen
for a group session), that number becomes a hard constraint, not just
another "discount": if `max_people` distinct identities have already
been confirmed and a new track finds no match with the normal
thresholds, it CANNOT still be an eleventh person -- it must
necessarily be one of the already-known people who is currently
"lost". In this case, and only this case, the merge is FORCED with the
lost person closest by signature (even above `max_signature_dist`),
instead of giving up and minting a new person_id.

Two guardrails keep the forcing safe: (1) the same causality rule as
the other matches applies -- a match can't be forced with someone who
was already visible together with this track under a different id; (2)
it forces ONLY against people currently "lost" (out of frame), never
against people active at that moment -- two people visible at the same
time always remain two distinct identities, the cap never merges them.
If the cap is reached but there's no one "lost" to recover, forcing is
still abandoned (no sensible candidate) and a new person_id is minted
while printing a warning -- a signal that the configured count might be
wrong or that a spurious detection got past `pose_estimation.py`'s
`max_people` filter.

Honest limitations
-------------------
  - The signature needs a minimum number of frames with sufficient
    confidence on the key joints; a very brief appearance in the frame
    will never produce a reliable signature and will remain its own
    person_id.
  - Two people of similar build can generate a false-positive merge --
    the default threshold is conservative but needs to be calibrated on
    your real data, it's not a validated value.
  - Color helps when clothes stay the same, but for the same reason it
    can increase the false-positive risk if two different people wear
    similarly-colored clothes AND have close body proportions -- this
    is an explicit trade-off, not a hidden problem.
  - Position is a double-edged sword in scenes with two people close
    together (e.g. child+caregiver): ONLY the discount (never the
    forcing) keeps the risk in check, but if two people's proportions
    are already ambiguous on their own, spatial proximity can push a
    borderline comparison past threshold in the wrong direction --
    a deliberate choice, not a hidden problem.
  - Once decided, the merge is applied automatically (there's no way to
    ask a human for confirmation in real time) -- that's why the event
    always stays in the log, instead of silently disappearing.
  - `max_people` is a real forcing (the sole exception to "never force"
    in the module): if the configured number is wrong (e.g. an extra
    adult briefly enters the scene, outside the expected count) the
    forced merge can attribute their frames to the wrong lost person --
    it should only be used when the number of participants is truly
    fixed and known.
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
