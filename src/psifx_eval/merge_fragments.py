"""
psifx_eval/merge_fragments.py
================================
Post-processing pass that re-links fragmented identities across an
ENTIRE already-produced MaskDir (any tracker: `sam3_baseline`, the
overlap-strategy SAM3 runs, native SAM3.1, ...), using appearance
(OSNet embedding + hue-histogram color, same two signals already used
by `identity_gallery.py`/`seg_reid.py`) instead of the geometric
chunk-boundary matching those other mechanisms rely on.

Why this is a separate, video-wide pass and not another chunk-boundary
hook (2026-08, real-footage test, Michele)
------------------------------------------------------------------------
Both `identity_gallery.py` (SAM3.1 native path) and the overlap
strategy's own reconciliation only ever run at a CHUNK BOUNDARY (the
shared anchor frame between chunk N and N+1) -- see `sam_backend.py`'s
per-chunk loop. On real clinical footage this turned out to miss a
whole class of id changes: a person (a child, in the motivating case)
disappearing and reappearing entirely WITHIN a single chunk -- SAM3's
own video-session tracking loses them mid-propagation and mints a new
local id on rediscovery, with no chunk boundary anywhere near the
event. Nothing in the existing chunk-boundary machinery is even called
in that case, so no amount of tuning it would have helped.

This module instead treats the WHOLE MaskDir as a bag of "tracks" (one
per id, each with a first and last non-empty frame) and asks, for
every track that STARTS after frame 0 (a "reappearance" candidate,
whatever caused the gap -- chunk boundary or mid-chunk loss, this pass
doesn't care which): is there an EARLIER track that ENDED before this
one started, whose appearance (OSNet + color) is close enough to be
the same real person? If so, merge them into one id. This is deliberately
agnostic to *why* a track fragmented -- it only looks at the geometry of
time (does track A end before track B starts) and appearance (do they
look like the same person), never chunk indices.

Matching philosophy, consistent with the rest of the project
------------------------------------------------------------------------
- GLOBAL assignment via Hungarian (`scipy.optimize.linear_sum_assignment`),
  not greedy -- same reasoning as `chunking.reconcile_ids`: taking the
  single best-looking pair first can steal a match that belongs to
  someone else, when multiple fragmentation events are being resolved
  at once.
- "Take the strongest signal, don't sum them" for combining OSNet with
  color -- same convention as `reid.py`/`seg_reid.py`'s `_pair_cost`/
  `_pair_score`: a very strong match on ONE reliable signal shouldn't be
  diluted by a weak/absent second one.
- Signatures are averaged over a few frames at the very end of a track
  (for the "ending" side) / very start of a track (for the "starting"
  side), not a single frame -- the same "don't trust one frame" lesson
  already learned the hard way for geometric matching in
  `chunking.reconcile_ids_windowed`.
- Fragments shorter than `min_fragment_frames` are excluded as merge
  CANDIDATES (too few frames for a trustworthy signature) but are still
  copied through to the output unchanged -- never silently dropped.

Single-component frames only, searched outward from the risky edge
(2026-08, real-footage bug, Michele)
------------------------------------------------------------------------
On real footage a track's mask can occasionally contain a SECOND,
unrelated blob in the same frame (a stray detection stuck to the frame
border was the observed case) alongside the real person. Naively taking
the largest connected component (the normal rule elsewhere in this
project, see `sam_backend._mask_to_polygon`) picks whichever one has
more pixels -- if the stray blob happens to be bigger than the real
person that frame, the signature ends up built from the WRONG region
entirely (observed: a similarity of 0.04 between two fragments that
were, by visual inspection, unambiguously the same child).

A first attempt tried to salvage these frames by predicting the
person's position from recent motion and picking whichever region was
closest to the prediction. That turned out to have its own failure
mode: the very first frame sampled has no prior motion to predict
from, so it still falls back to "largest region" -- and if THAT seed
frame is already ambiguous, the whole prediction chain anchors to the
wrong region from the start and never recovers (observed directly: the
last frame before the child hides IS itself a two-region frame, both
in and around the box).

The simpler, more robust rule adopted instead: don't try to guess which
region is right when a frame has more than one. Just don't trust that
frame for a signature at all -- `_sample_signature` only uses frames
whose mask is a SINGLE connected component (see
`_mask_to_polygon_single_component`), and searches OUTWARD from the
risky edge of the track (backward from the last frame, for an ending
track; forward from the first frame, for a starting one) until it has
collected enough clean frames, skipping ambiguous ones instead of
picking between their regions. A filter on mask SIZE was considered
and rejected too -- a crouching or partially-occluded person can
legitimately have a small mask, and shouldn't be penalized for it; this
rule doesn't touch size at all, only "is this frame unambiguous."

Second pass: pooled-group signature fallback (2026-08, real-footage bug,
Michele)
------------------------------------------------------------------------
Widening `signature_samples` to reach past a contaminated fragment
boundary (real-footage finding) fixed one rejected candidate but broke
a previously-accepted one: a bigger window on a SINGLE fragment dilutes
its signature with more varied-but-nearby frames, trading one failure
for another -- there's no one window size that's simultaneously tight
enough for easy adjacent matches and wide enough for hard ones.

So `merge_fragments` runs a SECOND pass, only for start tracks pass one
left unmatched (`orphan_start_ids`): instead of comparing the orphan
against one candidate fragment's own (tight, edge-anchored) signature,
it's compared against a POOLED signature built from clean frames spread
across EVERY member of an already-confirmed group from pass one (see
`_pooled_group_signature`) -- genuine diversity across multiple
confirmed sightings of the same person (different times, poses,
lighting), not just more frames from one narrow moment. Same Hungarian/
"tag every real candidate" conventions as pass one, just at group
granularity instead of fragment granularity (see `_resolve_group_merges`).

Streaming, not preloaded (2026-08, full-session OOM, Michele)
------------------------------------------------------------------------
The first working version loaded every `<id>.mp4` in the MaskDir fully
into RAM (via `mask_io.load_mask_dir`) as a dense `(T, H, W)` boolean
array per id -- fine for a short clip, but on a full clinical session
this exceeded available memory and the process was killed outright
(no traceback, just `Killed`, the classic Linux OOM-killer signature).

This is a pure I/O/implementation limitation of THIS post-processing
step, not a re-run of SAM3.1's chunking problem: SAM3.1's chunk
boundaries lose real information (the tracker's own attention/feature
propagation genuinely can't span the whole video, so identity can
actually be lost at a boundary), whereas here the full set of already-
computed masks exists on disk the whole time -- the only question is
*when* pixels are read into memory, not what the algorithm can see.
The matching logic never actually needed a whole track's pixel data at
once: only its `(first, last)` bounds (two integers) and a handful of
specific sampled frames per track. So every function that used to index
a preloaded array now takes a mask file `Path` instead and opens its
own `cv2.VideoCapture`, seeking to just the frames it needs (`_scan_
track`, `_sample_signature`, `_pooled_group_signature`), and the final
merged-MaskDir write reads every group member in lock-step alongside
the output writer instead of allocating a full merged array -- at no
point does this version hold more than a few frames' worth of pixels in
memory, and every decision (which frames count as signature candidates,
which pairs are temporally valid, the Hungarian assignment itself) is
identical to the preloaded version.

Border-touching frames also excluded from signatures (2026-08,
full-session finding, Michele)
------------------------------------------------------------------------
On the full-session video, `merge_fragments` merged the child's track
into the therapist's group with a suspiciously high similarity (0.897)
in the pooled-group fallback (pass two), well into a stretch where
manual inspection showed they were never actually confused by SAM3
itself -- the child's own raw, pre-merge track was clean (verified by
opening it directly, before `merge_fragments` ever touches it). What
that raw track DID contain, though, was a persistent stray artifact
along the left edge of frame (same kind of border-stuck blob already
described above) -- present, intermittently, in BOTH the child's and
the therapist's raw tracks.

The existing single-connected-component rule doesn't catch this case:
it was designed for frames with a stray blob ALONGSIDE the real person
(two components, rejected as ambiguous), but here some frames have
ONLY the border artifact as their single component -- e.g. a moment
where the real person isn't well segmented yet, but the artifact is.
Such a frame passes the single-component check (it genuinely is just
one region) even though it isn't the person at all, and if the same
artifact appears in signature-building frames on both sides of a
comparison, it can make two different people's signatures spuriously
resemble each other.

So `_frame_signal` (shared by `_sample_signature` and
`_pooled_group_signature`, i.e. every place a frame is considered for a
signature) now additionally rejects any single-component frame whose
region touches the frame's border (`_touches_border`) -- same "skip,
don't guess" philosophy as the multi-region case, just catching a
region that's suspicious for a different reason. A real person fully
in frame essentially never has their whole silhouette touch the very
edge, so this should cost very few genuine frames; a person leaving
frame does, but a partial, mid-exit view is poor signature material
anyway.

To make issues like this easier to catch (and confirm) without
guessing, `merge_fragments`'s report also now records the exact frame
indices used to build every track's and every pooled group's
signature (`signature_frames` in the report) -- so a suspiciously high
or low similarity can be checked by opening the specific frames that
produced it, the same way this particular bug was found.

Pooled signature: search nearby instead of giving up on an exact
target frame (2026-08, full-session finding, Michele)
------------------------------------------------------------------------
`signature_frames` from the very first full-session test with the
border filter above showed the fix working (frames were being
rejected) but not helping: the false merge's similarity barely moved.
The reason was a separate, pre-existing weakness in
`_pooled_group_signature` that the new, stricter filtering exposed.
`_evenly_spaced_frames` picks `samples_per_member` EXACT target frame
indices spread across a member's span, and the old code tried each one
exactly -- if a target frame failed any check (empty, ambiguous, or now
also border-touching), that sampling "slot" was simply lost, with no
attempt at a nearby alternative. On a member spanning tens of thousands
of frames, this could -- and did -- collapse 5 requested samples down
to a single surviving frame, leaving the pooled signature built from
far too little (and far too lucky-or-unlucky) material to be a
reliable average, undermining the whole point of pooling ("genuine
diversity across multiple confirmed sightings", see the "Second pass"
section above).

`_pooled_group_signature` now searches a small neighborhood around
each target frame (`_search_nearby_signal`, expanding outward one
frame at a time on both sides, capped in radius, and never revisiting
a frame already used by an earlier slot for the same member) before
giving up on that slot -- the same "keep looking nearby instead of
giving up immediately" principle `_sample_signature` already applies
via its own directional scan, just applied locally per target instead
of as one long scan. The search radius per target is capped at roughly
half the average spacing between targets (so one slot's search stays
in its own neighborhood instead of drifting into an adjacent slot's
territory, preserving the intended spread-out diversity) and at
`MAX_POOLED_SEARCH_RADIUS` frames absolutely (so a member with a huge,
mostly-unusable span can't make a single pooled-signature call scan
enormous numbers of frames).

Not runnable in this project's sandbox (needs `psifx`, a real MaskDir on
disk, and optionally `torch`/`torchreid` for the OSNet signal -- verify
on Michele's machine, same as the rest of `psifx_eval`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from psifx_eval.mask_io import _MASK_FILENAME_RE
from segmentation.sam_backend import _polygon_to_box
from segmentation.seg_reid import _mask_hue_histogram, _histogram_similarity
from pose.appearance_embedding import OSNetEmbedder, embedding_similarity

DEFAULT_MIN_FRAGMENT_FRAMES = 8
DEFAULT_MERGE_THRESHOLD = 0.6
DEFAULT_SIGNATURE_SAMPLES = 5
DEFAULT_POOLED_SAMPLES_PER_MEMBER = 5
# absolute cap on how far _search_nearby_signal will look on either side
# of an evenly-spaced target frame before giving up on that sampling
# slot -- bounds the worst case (a member with a huge, mostly-unusable
# span) to a fixed amount of work per pooled-signature call, see module
# docstring's "Pooled signature: search nearby" section.
MAX_POOLED_SEARCH_RADIUS = 200
# same white-on-black threshold convention as mask_io.read_mask_video --
# kept in sync by hand since this module deliberately doesn't preload
# through mask_io anymore (see module docstring's "Streaming" section).
DEFAULT_MASK_THRESHOLD = 127
# cost assigned to a temporally-impossible pair (end doesn't precede
# start) in the Hungarian matrix -- worse than any real similarity score
# (which lives in cost = 1 - similarity, i.e. [0, 1]), so these pairs are
# never picked ahead of a real candidate, but stay finite (Hungarian
# doesn't handle inf cleanly).
_IMPOSSIBLE_COST = 10.0


def _list_mask_files(mask_dir: str) -> dict[int, Path]:
    """`{id: path}` for every `<id>.mp4` in `mask_dir`, same filename
    convention as `mask_io.load_mask_dir` (`_MASK_FILENAME_RE`) -- but
    without decoding anything, unlike `load_mask_dir`, which eagerly
    loads every file fully into RAM (the whole reason for this
    module's streaming rewrite, see module docstring)."""
    dir_path = Path(mask_dir)
    paths: dict[int, Path] = {}
    if not dir_path.is_dir():
        return paths
    for p in dir_path.iterdir():
        m = _MASK_FILENAME_RE.match(p.name)
        if m:
            paths[int(m.group(1))] = p
    return paths


def _scan_track(
    mask_path: Path,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> tuple[int, int, int, int, int, int] | None:
    """Streams one id's mask video ONCE -- each frame is decoded, checked,
    and discarded, so memory stays O(1) regardless of video length --
    and returns `(first, last, real_frames, decoded_frame_count, height,
    width)`, or `None` if the track has no non-empty frame at all
    (defensive -- normal MaskDirs shouldn't contain such an id, since
    nothing would have written it). This replaces the old
    `_fragment_bounds`, which sliced a fully preloaded `(T, H, W)` array
    (see module docstring's "Streaming" section for why that no longer
    works on a full-session video).

    `real_frames` counts every non-empty frame in the track, which by
    construction all fall inside `[first, last]` -- same quantity the
    old version computed via `masks[obj_id][first:last+1].any(axis=(1,2)
    ).sum()`.

    `decoded_frame_count` -- the number of frames this call actually
    walked through -- is used as this track's contribution to
    `total_frames` in `merge_fragments`, rather than trusting
    `cv2.CAP_PROP_FRAME_COUNT` container metadata, since this project
    has already hit cases where the two disagree."""
    cap = cv2.VideoCapture(str(mask_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open mask video: {mask_path}")

    first = last = None
    real_frames = 0
    decoded = 0
    height = width = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if height is None:
                height, width = frame.shape[0], frame.shape[1]
            gray = frame[:, :, 0] if frame.ndim == 3 else frame
            if bool((gray > threshold).any()):
                if first is None:
                    first = decoded
                last = decoded
                real_frames += 1
            decoded += 1
    finally:
        cap.release()

    if first is None:
        return None
    return first, last, real_frames, decoded, height, width


def _read_mask_frame(
    cap: cv2.VideoCapture,
    frame_idx: int,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> np.ndarray | None:
    """Seeks `cap` (an already-open mask-file `VideoCapture`) to
    `frame_idx` and returns the decoded boolean mask frame, or `None` if
    the seek/read failed (past end of file, corrupt frame, ...) -- same
    `gray > threshold` decode convention as `mask_io.read_mask_video`,
    replicated here since this module streams instead of going through
    `mask_io.load_mask_dir` (see module docstring)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, raw = cap.read()
    if not ok:
        return None
    gray = raw[:, :, 0] if raw.ndim == 3 else raw
    return gray > threshold


def _mask_to_polygon_single_component(mask_frame: np.ndarray) -> np.ndarray:
    """Polygon of the mask frame's connected component, but ONLY if
    there is EXACTLY ONE (empty polygon `(0,2)` otherwise -- zero
    regions, same as an empty mask, or more than one, meaning the frame
    is ambiguous). Unlike `sam_backend._mask_to_polygon` (which always
    returns the largest region, even when there's more than one), this
    deliberately refuses to guess between multiple disconnected regions
    -- see module docstring for why a real person plus e.g. a stray
    detection stuck to the frame border made that the wrong call for
    building an appearance signature."""
    mask_u8 = mask_frame.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return np.empty((0, 2))
    return contours[0].reshape(-1, 2).astype(float)


def _touches_border(poly: np.ndarray, height: int, width: int, margin: int = 1) -> bool:
    """True if `poly`'s bounding box comes within `margin` pixels of any
    edge of a `(height, width)` frame. Used to reject single-component
    frames whose one region is actually a stray artifact stuck to the
    frame border (2026-08, full-session finding, Michele) -- see module
    docstring's "Border-touching frames" section for why the plain
    single-component check alone doesn't catch this: such a frame IS
    technically one connected region, just not the person."""
    if poly.shape[0] == 0:
        return False
    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    return bool(
        x_min <= margin or y_min <= margin
        or x_max >= width - 1 - margin or y_max >= height - 1 - margin
    )


def _frame_signal(
    cap: cv2.VideoCapture,
    mask_frame: np.ndarray,
    frame_idx: int,
    embedder: OSNetEmbedder | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """OSNet embedding + hue histogram for ONE frame -- `(None, None)`
    if the frame is empty, ambiguous (its mask isn't a SINGLE connected
    component, see `_mask_to_polygon_single_component`), OR its one
    component touches the frame border (see `_touches_border` and the
    module docstring's "Border-touching frames" section -- a stray
    artifact stuck to the edge can be the ONLY region in a frame,
    passing the single-component check while still not being the real
    person). Shared by `_sample_signature` (per-fragment, edge-anchored)
    and `_pooled_group_signature` (per-group, evenly spread) so both use
    the exact same "don't trust a suspicious frame" rules."""
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
    signature, re-normalized after averaging (an average of unit
    vectors isn't itself unit norm). `(None, None)` components for
    whichever signal had no usable frames at all -- same "no made-up
    signal" convention as the rest of this module."""
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
    """Average OSNet embedding + average hue histogram over up to
    `signature_samples` USABLE frames drawn from `candidate_frames`
    (see `_frame_signal` for what "usable" means), stopping as soon as
    that many have been collected. Also returns the list of frame
    indices actually used, in the order they were collected -- recorded
    in `merge_fragments`'s report (`signature_frames`) so a suspicious
    similarity can be checked by opening the exact frames that produced
    it (see module docstring's "Border-touching frames" section, added
    after that's exactly how a real bug was found).

    Streams `mask_path`'s frames on demand through its own
    `cv2.VideoCapture` (opened once per call, seeked per candidate
    frame) instead of indexing a preloaded array -- see module
    docstring's "Streaming" section. `cap` remains the SOURCE video
    capture, passed through to `_frame_signal` unchanged.

    `candidate_frames` MUST be ordered from the risky edge of the track
    OUTWARD (the end-of-track caller walks backward starting at the
    last frame; the start-of-track caller walks forward starting at the
    first frame -- see the two call sites in `merge_fragments`), so
    that when ambiguous/empty frames are skipped, the search naturally
    extends further into the track's more stable middle rather than
    giving up right at the risky boundary. A short and/or heavily
    ambiguous fragment may simply not have `signature_samples` clean
    frames available in `candidate_frames` at all -- whatever was found
    is still used, same "some signal beats none" convention as the rest
    of this module."""
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
    """`n` frame indices evenly spread across `[first, last]` inclusive
    (or every frame in the span, if it has fewer than `n`) -- used by
    `_pooled_group_signature` to sample pose/lighting DIVERSITY across
    a fragment's whole duration, unlike `_sample_signature`'s "closest
    to the risky edge first" ordering, which deliberately favors
    frames near the fragment's boundary instead."""
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
    """Tries `target` first, then alternately expands outward
    (+1, -1, +2, -2, ...) up to `max_radius` frames on either side --
    clamped to `[first, last]` and skipping anything already in
    `exclude` -- and returns the first `(frame_idx, embedding,
    histogram)` that passes `_frame_signal`'s checks, or `None` if
    nothing nearby works. Used by `_pooled_group_signature` so one bad
    frame at an evenly-spaced target position (empty, ambiguous, or
    border-touching -- see `_frame_signal`) doesn't lose that sampling
    slot entirely -- the same "keep looking, don't give up" principle
    `_sample_signature` already applies via its own directional scan,
    just applied locally per target instead of as one long scan (see
    module docstring's "Pooled signature: search nearby" section)."""
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
    ids (formed by pass-one merges in `merge_fragments`), pooling clean
    frames spread ACROSS ALL MEMBERS -- different times, poses, and
    lighting across the group's whole history -- instead of only the
    edge frames of one fragment. See the module docstring's "Second
    pass" section for why this, rather than just widening
    `signature_samples` on a single fragment, is the fallback used for
    a candidate pass one rejected.

    For each of the `samples_per_member` evenly-spaced target frames,
    searches a small neighborhood around it (`_search_nearby_signal`)
    instead of trying only the exact target -- see module docstring's
    "Pooled signature: search nearby" section for why: without this, a
    member spanning a huge span could end up represented by far fewer
    (sometimes just one) usable frame than requested, if a handful of
    unlucky exact target frames happened to fail.

    Also returns `{member_id: [frame indices actually used]}` -- see
    `_sample_signature`'s docstring for why this is recorded (same
    `signature_frames` report field, keyed by pooled group here).

    Streams each member's own mask file through its own
    `cv2.VideoCapture` (opened per member, released before moving to
    the next) -- see module docstring's "Streaming" section."""
    embeddings: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    used_frame_indices: dict[int, list[int]] = {}
    for member_id in member_ids:
        if member_id not in bounds or member_id not in mask_paths:
            continue  # e.g. a too-short fragment folded into this group
        first, last = bounds[member_id]
        targets = _evenly_spaced_frames(first, last, samples_per_member)
        # search radius per target: roughly half the average spacing
        # between targets, so a slot's search stays in its own
        # neighborhood instead of drifting into an adjacent slot's --
        # capped absolutely by MAX_POOLED_SEARCH_RADIUS regardless of
        # how large the member's own span is.
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
                    continue  # nothing usable near this slot -- same "some signal beats none"
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
    """0..1 similarity between an ending track's signature and a
    starting track's signature -- the strongest of (OSNet, color),
    never their sum/average (see module docstring)."""
    end_emb, end_hist = end_sig
    start_emb, start_hist = start_sig
    scores: list[float] = []
    emb_sim = embedding_similarity(end_emb, start_emb)
    if emb_sim is not None:
        scores.append(emb_sim)
    if end_hist is not None and start_hist is not None:
        scores.append(_histogram_similarity(end_hist, start_hist))
    return max(scores) if scores else 0.0


def _resolve_merges(
    end_ids: list[int],
    start_ids: list[int],
    bounds: dict[int, tuple[int, int]],
    end_sigs: dict[int, tuple],
    start_sigs: dict[int, tuple],
    merge_threshold: float,
) -> list[dict]:
    """Global (Hungarian) assignment between `end_ids` (rows) and
    `start_ids` (columns). Returns EVERY temporally-valid pair Hungarian
    actually assigned (end strictly before start), each tagged with
    `"accepted": similarity >= merge_threshold` -- not just the accepted
    ones. This is deliberate: a near-miss rejection (e.g. similarity
    0.58 against a 0.6 threshold) is exactly the kind of thing worth
    seeing when tuning `merge_threshold`, and silently dropping rejected
    candidates would hide it. Temporally-invalid pairs (same track, or
    start not after end -- cost still at `_IMPOSSIBLE_COST`) are left out
    entirely, since Hungarian only "assigned" them to fill out a
    rectangular matrix, not because they're real candidates.

    A single Hungarian call over the whole bipartite set instead of
    pairwise greedy matching, so two simultaneous fragmentation events
    (e.g. both people losing their id around the same time) don't steal
    each other's correct match -- same reasoning as
    `chunking.reconcile_ids`."""
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
    """Second-pass Hungarian assignment: orphan start tracks (rows,
    ones pass one's fragment-to-fragment comparison left unmatched)
    against candidate GROUPS (columns, `{canonical_id: member_ids}`
    from pass one) -- see the module docstring's "Second pass" section.
    A group is a temporally valid candidate for an orphan if ANY of its
    members ends before the orphan starts (mirrors `_resolve_merges`'s
    single-id check, just applied to every member). Same global
    assignment + "tag every real candidate" conventions as
    `_resolve_merges`."""
    if not orphan_ids or not groups:
        return []

    group_ids = list(groups.keys())
    cost = np.full((len(orphan_ids), len(group_ids)), _IMPOSSIBLE_COST)
    for i, o in enumerate(orphan_ids):
        for j, g in enumerate(group_ids):
            members = groups[g]
            if o in members:
                continue  # orphan is (trivially) already part of this group
            if not any(m in bounds and bounds[m][1] < bounds[o][0] for m in members):
                continue  # no member of this group ends before the orphan starts
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
    `{original_id: canonical_id}`, following chains (A merges into B,
    B separately merges into C => A, B, C all map to the same canonical
    id) via union-find. The canonical id for a group is its SMALLEST
    original id, purely for a stable, deterministic output filename --
    carries no other meaning."""
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


def merge_fragments(
    *,
    video_path: str,
    mask_dir: str,
    out_mask_dir: str,
    min_fragment_frames: int = DEFAULT_MIN_FRAGMENT_FRAMES,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    signature_samples: int = DEFAULT_SIGNATURE_SAMPLES,
    pooled_samples_per_member: int = DEFAULT_POOLED_SAMPLES_PER_MEMBER,
    device: str = "cpu",
    use_osnet: bool = True,
) -> dict:
    """Reads `mask_dir`, merges fragmented identities via appearance
    (see module docstring), writes a NEW MaskDir at `out_mask_dir`
    (never touches the original), and returns a JSON-able report of
    every merge decision made (accepted and -- for transparency --
    every candidate pair actually considered).

    Streams every mask video from disk instead of preloading the whole
    MaskDir into RAM -- see module docstring's "Streaming" section
    (added 2026-08 after a full-session run hit the Linux OOM killer on
    the old `load_mask_dir`-based version). At no point does this
    function hold more than a handful of individual frames in memory;
    the set of frames actually used for matching, and every matching
    decision itself, is unchanged."""
    mask_paths = _list_mask_files(mask_dir)
    if not mask_paths:
        raise ValueError(f"No <id>.mp4 mask files found in {mask_dir}")

    all_ids = sorted(mask_paths.keys())
    bounds: dict[int, tuple[int, int]] = {}
    excluded_short: list[int] = []
    total_frames: int | None = None
    total_frames_source_id: int | None = None
    height = width = None
    for obj_id in all_ids:
        scan = _scan_track(mask_paths[obj_id])
        if scan is None:
            continue
        first, last, real_frames, decoded_frames, h, w = scan
        if total_frames is None:
            total_frames = decoded_frames
            total_frames_source_id = obj_id
            height, width = h, w
        elif decoded_frames != total_frames:
            raise ValueError(
                f"id {obj_id} has {decoded_frames} frames but id {total_frames_source_id} "
                f"has {total_frames} -- every <id>.mp4 in a MaskDir is expected to be padded "
                f"to the same total frame count (same invariant mask_io.load_mask_dir checks)."
            )
        if real_frames < min_fragment_frames:
            excluded_short.append(obj_id)
            continue
        bounds[obj_id] = (first, last)

    if not bounds:
        raise ValueError(
            f"No id in {mask_dir} has at least {min_fragment_frames} non-empty "
            f"frames -- nothing to merge (every id was too short/noisy)."
        )

    embedder = None
    if use_osnet:
        try:
            embedder = OSNetEmbedder(device=device)
        except ImportError as exc:
            print(f"[merge_fragments] OSNet unavailable ({exc}) -- "
                  f"falling back to color-only matching.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    end_ids = list(bounds.keys())
    # a track starting at frame 0 is the video's first sighting of that
    # id, not a "reappearance" -- nothing plausible for it to resume, so
    # it's excluded from the START side (it can still be an END, i.e.
    # something else can resume INTO it later).
    start_ids = [oid for oid in bounds if bounds[oid][0] > 0]

    # per-signature frame indices actually used, for transparency (see
    # module docstring's "Border-touching frames" section) -- filled in
    # below, included in the report as `signature_frames`.
    signature_frames: dict[str, dict] = {"end": {}, "start": {}, "pooled_groups": {}}

    try:
        end_sigs = {}
        for obj_id in end_ids:
            first, last = bounds[obj_id]
            # search BACKWARD from `last` (the risky edge, right before
            # the track goes empty) toward `first` -- ambiguous/empty
            # frames near the edge are skipped, extending the search
            # further into the track's more stable middle instead of
            # forcing a fixed-size window (see _sample_signature).
            candidates = list(range(last, first - 1, -1))
            emb, hist, frames_used = _sample_signature(cap, mask_paths[obj_id], candidates, signature_samples, embedder)
            end_sigs[obj_id] = (emb, hist)
            signature_frames["end"][obj_id] = frames_used

        start_sigs = {}
        for obj_id in start_ids:
            first, last = bounds[obj_id]
            # search FORWARD from `first` (the risky reappearance edge)
            # toward `last`, same reasoning as above.
            candidates = list(range(first, last + 1))
            emb, hist, frames_used = _sample_signature(cap, mask_paths[obj_id], candidates, signature_samples, embedder)
            start_sigs[obj_id] = (emb, hist)
            signature_frames["start"][obj_id] = frames_used
    finally:
        cap.release()

    candidate_pairs = _resolve_merges(end_ids, start_ids, bounds, end_sigs, start_sigs, merge_threshold)
    merges = [
        (c["from_id"], c["into_id"], c["similarity"])
        for c in candidate_pairs if c["accepted"]
    ]
    canonical = _group_chains(merges, all_ids)

    # --- second pass: pooled-group fallback for orphan start tracks
    # pass one couldn't match against any single fragment (see module
    # docstring's "Second pass" section) ---
    merged_start_ids = {c["into_id"] for c in candidate_pairs if c["accepted"]}
    orphan_start_ids = [s for s in start_ids if s not in merged_start_ids]

    group_candidates: list[dict] = []
    if orphan_start_ids:
        pass1_groups: dict[int, list[int]] = {}
        for oid in all_ids:
            pass1_groups.setdefault(canonical[oid], []).append(oid)
        # a lone orphan's own (still-trivial) "group" isn't a useful
        # pooled candidate for itself -- excluded up front rather than
        # relying on the `o in members` check alone, since that check
        # only guards against matching an orphan to ITS OWN group, not
        # against wastefully computing a pooled signature for it.
        candidate_group_ids = [g for g in pass1_groups if g not in orphan_start_ids]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        try:
            group_sigs = {}
            for g in candidate_group_ids:
                emb, hist, member_frames_used = _pooled_group_signature(
                    cap, mask_paths, pass1_groups[g], bounds, pooled_samples_per_member, embedder,
                )
                group_sigs[g] = (emb, hist)
                signature_frames["pooled_groups"][g] = member_frames_used
        finally:
            cap.release()
        orphan_sigs = {o: start_sigs[o] for o in orphan_start_ids}

        group_candidates = _resolve_group_merges(
            orphan_start_ids,
            {g: pass1_groups[g] for g in candidate_group_ids},
            bounds, group_sigs, orphan_sigs, merge_threshold,
        )
        extra_merges = [
            (min(pass1_groups[c["group_id"]]), c["orphan_id"], c["similarity"])
            for c in group_candidates if c["accepted"]
        ]
        if extra_merges:
            canonical = _group_chains(merges + extra_merges, all_ids)

    # write the merged MaskDir: one file per DISTINCT canonical id,
    # union (logical OR) of every member's mask -- members shouldn't
    # overlap in time by construction (merges only ever go
    # earlier-end -> later-start), OR is just a safe way to combine
    # them without assuming that never breaks.
    #
    # Streamed lock-step: every member's mask file is already padded to
    # the SAME total_frames length (project-wide MaskDir format
    # invariant, checked above), so frame t of each member's file
    # already aligns with global frame t -- no seeking needed, every
    # member can just be read forward in step with the output writer,
    # OR-ing each frame together and writing it immediately. This never
    # allocates a full (total_frames, height, width) array, which was
    # the other big memory cost in the pre-streaming version (see
    # module docstring's "Streaming" section).
    out_dir_path = Path(out_mask_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[int]] = {}
    for oid in all_ids:
        groups.setdefault(canonical[oid], []).append(oid)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for canon_id, members in groups.items():
        member_caps = [cv2.VideoCapture(str(mask_paths[m])) for m in members]
        for member, mcap in zip(members, member_caps):
            if not mcap.isOpened():
                for c in member_caps:
                    c.release()
                raise FileNotFoundError(f"Could not open mask video: {mask_paths[member]}")

        out_path = out_dir_path / f"{canon_id}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            for c in member_caps:
                c.release()
            raise RuntimeError(f"Could not open mask writer at {out_path}")

        try:
            for _t in range(total_frames):
                merged_frame = np.zeros((height, width), dtype=bool)
                for mcap in member_caps:
                    ok, raw = mcap.read()
                    if not ok:
                        continue  # this member's file ended early (shouldn't
                                  # happen given the padding invariant, but
                                  # degrade gracefully rather than crash)
                    gray = raw[:, :, 0] if raw.ndim == 3 else raw
                    merged_frame |= gray > DEFAULT_MASK_THRESHOLD
                frame_rgb = np.repeat((merged_frame.astype(np.uint8) * 255)[..., np.newaxis], 3, axis=-1)
                writer.write(frame_rgb)
        finally:
            writer.release()
            for mcap in member_caps:
                mcap.release()

    report = {
        "source_mask_dir": str(mask_dir),
        "output_mask_dir": str(out_mask_dir),
        "total_frames": total_frames,
        "original_id_count": len(all_ids),
        "merged_id_count": len(groups),
        "excluded_as_too_short": excluded_short,
        "merge_threshold": merge_threshold,
        "min_fragment_frames": min_fragment_frames,
        "osnet_used": embedder is not None,
        "accepted_merges": [
            {"from_id": e, "into_id": s, "similarity": round(sim, 3)}
            for e, s, sim in merges
        ],
        "rejected_candidates": [
            {"from_id": c["from_id"], "into_id": c["into_id"], "similarity": c["similarity"]}
            for c in candidate_pairs if not c["accepted"]
        ],
        "pooled_group_samples_per_member": pooled_samples_per_member,
        "pooled_group_candidates": [
            {"orphan_id": c["orphan_id"], "group_id": c["group_id"],
             "similarity": c["similarity"], "accepted": c["accepted"]}
            for c in group_candidates
        ],
        "groups": {str(canon): members for canon, members in groups.items()},
        "signature_frames": {
            "end": {str(oid): frames for oid, frames in signature_frames["end"].items()},
            "start": {str(oid): frames for oid, frames in signature_frames["start"].items()},
            "pooled_groups": {
                str(gid): {str(mid): frames for mid, frames in member_frames.items()}
                for gid, member_frames in signature_frames["pooled_groups"].items()
            },
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merges fragmented identities across a whole MaskDir using appearance "
                     "(OSNet + color), independent of whether the fragmentation happened at "
                     "a chunk boundary or mid-chunk -- see module docstring.")
    parser.add_argument("--video", required=True, help="Path to the source video (same one the MaskDir was produced from)")
    parser.add_argument("--mask-dir", required=True, help="Existing MaskDir to read (any tracker's output)")
    parser.add_argument("--out-dir", required=True, help="Output directory for the merged MaskDir (never overwrites --mask-dir)")
    parser.add_argument("--min-fragment-frames", type=int, default=DEFAULT_MIN_FRAGMENT_FRAMES,
                         help="Fragments with fewer non-empty frames than this are excluded as merge "
                              "candidates (still copied through unchanged, never dropped)")
    parser.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD,
                         help="Minimum appearance similarity (0..1) to accept a merge")
    parser.add_argument("--signature-samples", type=int, default=DEFAULT_SIGNATURE_SAMPLES,
                         help="Number of frames averaged at each track's start/end to build its signature")
    parser.add_argument("--pooled-group-samples", type=int, default=DEFAULT_POOLED_SAMPLES_PER_MEMBER,
                         help="Pass-two fallback: frames sampled per group MEMBER (evenly spread across "
                              "each member's own span) when comparing an unmatched start track against an "
                              "already-confirmed group's pooled appearance, instead of one fragment's own "
                              "edge-anchored signature")
    parser.add_argument("--device", default="cpu", help="Device for the OSNet embedder")
    parser.add_argument("--no-osnet", dest="use_osnet", action="store_false", default=True,
                         help="Disable OSNet, use color (hue histogram) only")
    args = parser.parse_args()

    report = merge_fragments(
        video_path=args.video, mask_dir=args.mask_dir, out_mask_dir=args.out_dir,
        min_fragment_frames=args.min_fragment_frames, merge_threshold=args.merge_threshold,
        signature_samples=args.signature_samples, pooled_samples_per_member=args.pooled_group_samples,
        device=args.device, use_osnet=args.use_osnet,
    )

    print(f"\n{report['original_id_count']} original ids -> {report['merged_id_count']} after merging "
          f"({len(report['accepted_merges'])} pass-1 merge(s) accepted, "
          f"{len(report['excluded_as_too_short'])} id(s) excluded as too short to use as a signature)")
    for m in report["accepted_merges"]:
        print(f"  id {m['from_id']} -> id {m['into_id']}  (similarity {m['similarity']})")
    if report["rejected_candidates"]:
        print(f"\n{len(report['rejected_candidates'])} pass-1 candidate pair(s) considered but "
              f"below --merge-threshold ({report['merge_threshold']}):")
        for c in sorted(report["rejected_candidates"], key=lambda c: -c["similarity"]):
            print(f"  id {c['from_id']} -> id {c['into_id']}  (similarity {c['similarity']})")
    if report["pooled_group_candidates"]:
        accepted = [c for c in report["pooled_group_candidates"] if c["accepted"]]
        rejected = [c for c in report["pooled_group_candidates"] if not c["accepted"]]
        print(f"\npass 2 (pooled-group fallback for orphan start tracks): "
              f"{len(accepted)} accepted, {len(rejected)} rejected")
        for c in sorted(report["pooled_group_candidates"], key=lambda c: -c["similarity"]):
            tag = "accepted" if c["accepted"] else "rejected"
            print(f"  id {c['orphan_id']} -> group {c['group_id']}  (similarity {c['similarity']}, {tag})")

    report_path = Path(args.out_dir) / "merge_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
