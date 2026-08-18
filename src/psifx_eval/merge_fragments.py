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

Trajectory-aware contour selection (2026-08, real-footage bug, Michele)
------------------------------------------------------------------------
On real footage a track's mask can occasionally contain a SECOND,
unrelated blob in the same frame (a stray detection stuck to the frame
border was the observed case) alongside the real person. Naively taking
the largest connected component (the normal rule elsewhere in this
project, see `sam_backend._mask_to_polygon`) picks whichever one has
more pixels -- if the stray blob happens to be bigger than the real
person that frame, the signature ends up built from the WRONG region
entirely (observed: a similarity of 0.04 between two fragments that
were, by visual inspection, unambiguously the same child). `_sample_signature`
guards against this via `_select_best_contour`: it predicts where the
tracked person should be from the track's own recent motion (a plain
constant-velocity estimate, the same idea as `chunking.estimate_velocities`,
just applied frame-by-frame) and picks whichever disconnected region is
closest to that prediction, not whichever is largest. A filter on mask
SIZE was considered and rejected -- a crouching or partially-occluded
person can legitimately have a small mask, and a person legitimately
exiting through the edge of the frame can legitimately be near the
border; neither should be penalized. Motion continuity doesn't have
that problem: a real person's predicted position moves WITH them
(including toward an edge they're walking out of), while a fixed
background artifact doesn't track any predicted trajectory at all.

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

from psifx_eval.mask_io import load_mask_dir
from segmentation.sam_backend import _mask_to_polygon, _polygon_to_box
from segmentation.chunking import _centroid as _polygon_centroid
from segmentation.seg_reid import _mask_hue_histogram, _histogram_similarity
from pose.appearance_embedding import OSNetEmbedder, embedding_similarity

DEFAULT_MIN_FRAGMENT_FRAMES = 8
DEFAULT_MERGE_THRESHOLD = 0.6
DEFAULT_SIGNATURE_SAMPLES = 5
# cost assigned to a temporally-impossible pair (end doesn't precede
# start) in the Hungarian matrix -- worse than any real similarity score
# (which lives in cost = 1 - similarity, i.e. [0, 1]), so these pairs are
# never picked ahead of a real candidate, but stay finite (Hungarian
# doesn't handle inf cleanly).
_IMPOSSIBLE_COST = 10.0


def _fragment_bounds(mask_arr: np.ndarray) -> tuple[int, int] | None:
    """(first, last) non-empty-frame indices for one id's mask array, or
    `None` if the id has no non-empty frame at all (defensive -- normal
    MaskDirs shouldn't contain such an id, since nothing would have
    written it)."""
    nonempty = np.where(mask_arr.any(axis=(1, 2)))[0]
    if nonempty.size == 0:
        return None
    return int(nonempty[0]), int(nonempty[-1])


def _select_best_contour(
    mask_frame: np.ndarray,
    expected_centroid: np.ndarray | None,
) -> np.ndarray:
    """Like `_mask_to_polygon`, but when a mask frame has MULTIPLE
    disconnected regions (a real person plus e.g. a stray detection
    stuck to the frame border), picks the region whose centroid is
    closest to `expected_centroid` -- a position predicted from the
    track's OWN recent motion (see `_sample_signature`) -- instead of
    blindly the largest by area.

    Why not filter by size instead (2026-08, real-footage bug, Michele):
    a person can legitimately have a SMALL mask (crouching, partially
    behind an object) -- rejecting small regions would throw away good
    signal. A spurious region stuck to the frame edge, on the other
    hand, doesn't move the way a real person does: its position stays
    essentially fixed regardless of where the person the track is
    supposed to be following actually goes. Comparing against a
    motion-predicted expected position (not a fixed reference point)
    also still recognizes a person legitimately walking out through the
    edge of the frame, since the prediction moves with them.

    Falls back to `_mask_to_polygon`'s "largest contour" rule when
    `expected_centroid` is `None` (no motion estimate yet -- e.g. the
    very first sampled frame of a track) or when there's only one
    region to begin with -- same "no made-up signal" convention used
    throughout this module."""
    if expected_centroid is None:
        return _mask_to_polygon(mask_frame)
    mask_u8 = mask_frame.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2))
    if len(contours) == 1:
        return contours[0].reshape(-1, 2).astype(float)

    def _dist_to_expected(contour: np.ndarray) -> float:
        poly = contour.reshape(-1, 2).astype(float)
        centroid = poly.mean(axis=0)
        return float(np.linalg.norm(centroid - expected_centroid))

    best = min(contours, key=_dist_to_expected)
    return best.reshape(-1, 2).astype(float)


def _sample_signature(
    cap: cv2.VideoCapture,
    mask_arr: np.ndarray,
    frame_indices: list[int],
    embedder: OSNetEmbedder | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Average OSNet embedding + average hue histogram over whichever of
    `frame_indices` actually have a non-empty mask (a fragment can have
    internal gaps even between its own first/last frame). Re-normalized
    after averaging (an average of unit vectors isn't itself unit norm).
    `(None, None)` if no usable frame was found -- same "no made-up
    signal" convention as the underlying per-frame functions.

    `frame_indices` MUST be walked in the order the caller wants motion
    estimated in (the end-of-track window walks TOWARD the risky last
    frame; the start-of-track window walks AWAY from the risky first
    frame -- see the two call sites in `merge_fragments`). Each frame's
    contour is chosen via `_select_best_contour`, using a plain
    constant-velocity prediction (this frame's expected centroid =
    last confirmed centroid + last observed per-frame velocity * frame
    gap) built up incrementally from whichever earlier frames in this
    same window already had a confident, unambiguous contour -- the
    same "simple first-difference beats nothing" motion compensation
    already used in `chunking.estimate_velocities`, just applied frame
    by frame instead of chunk-boundary by chunk-boundary. The very
    first usable frame of the window has no prior observation to
    predict from, so it falls back to the old "largest contour" rule --
    a corrupted seed frame is a known limitation, not a claim this is
    bulletproof (see module docstring)."""
    embeddings: list[np.ndarray] = []
    histograms: list[np.ndarray] = []

    prev_centroid: np.ndarray | None = None
    prev_frame_idx: int | None = None
    velocity: np.ndarray | None = None

    for frame_idx in frame_indices:
        if frame_idx >= mask_arr.shape[0] or not mask_arr[frame_idx].any():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        expected_centroid = None
        if prev_centroid is not None and prev_frame_idx is not None:
            step = frame_idx - prev_frame_idx
            shift = velocity * step if velocity is not None else 0.0
            expected_centroid = prev_centroid + shift

        poly = _select_best_contour(mask_arr[frame_idx], expected_centroid)
        if poly.shape[0] < 3:
            continue

        centroid = _polygon_centroid(poly)
        if centroid is not None:
            if prev_centroid is not None and prev_frame_idx is not None and frame_idx != prev_frame_idx:
                velocity = (centroid - prev_centroid) / (frame_idx - prev_frame_idx)
            prev_centroid = centroid
            prev_frame_idx = frame_idx

        box = _polygon_to_box(poly)
        if embedder is not None:
            vec = embedder.embed(frame, box, poly=poly)
            if vec is not None:
                embeddings.append(vec)
        hist = _mask_hue_histogram(frame, poly)
        if hist is not None:
            histograms.append(hist)

    embedding = None
    if embeddings:
        embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(embedding)
        if norm > 1e-9:
            embedding = embedding / norm
        else:
            embedding = None

    histogram = None
    if histograms:
        histogram = np.mean(histograms, axis=0)
        total = histogram.sum()
        histogram = histogram / total if total > 1e-9 else None

    return embedding, histogram


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
        similarity = 1.0 - cost[r, c]
        candidates.append({
            "from_id": end_ids[r],
            "into_id": start_ids[c],
            "similarity": round(float(similarity), 3),
            "accepted": similarity >= merge_threshold,
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
    device: str = "cpu",
    use_osnet: bool = True,
) -> dict:
    """Reads `mask_dir`, merges fragmented identities via appearance
    (see module docstring), writes a NEW MaskDir at `out_mask_dir`
    (never touches the original), and returns a JSON-able report of
    every merge decision made (accepted and -- for transparency --
    every candidate pair actually considered)."""
    masks = load_mask_dir(mask_dir)
    total_frames = next(iter(masks.values())).shape[0]

    all_ids = sorted(masks.keys())
    bounds: dict[int, tuple[int, int]] = {}
    excluded_short: list[int] = []
    for obj_id in all_ids:
        fb = _fragment_bounds(masks[obj_id])
        if fb is None:
            continue
        first, last = fb
        real_frames = int(masks[obj_id][first:last + 1].any(axis=(1, 2)).sum())
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

    try:
        end_sigs = {}
        for obj_id in end_ids:
            first, last = bounds[obj_id]
            window = list(range(max(first, last - signature_samples + 1), last + 1))
            end_sigs[obj_id] = _sample_signature(cap, masks[obj_id], window, embedder)

        start_sigs = {}
        for obj_id in start_ids:
            first, last = bounds[obj_id]
            window = list(range(first, min(last, first + signature_samples - 1) + 1))
            start_sigs[obj_id] = _sample_signature(cap, masks[obj_id], window, embedder)
    finally:
        cap.release()

    candidate_pairs = _resolve_merges(end_ids, start_ids, bounds, end_sigs, start_sigs, merge_threshold)
    merges = [
        (c["from_id"], c["into_id"], c["similarity"])
        for c in candidate_pairs if c["accepted"]
    ]
    canonical = _group_chains(merges, all_ids)

    # write the merged MaskDir: one file per DISTINCT canonical id,
    # union (logical OR) of every member's mask -- members shouldn't
    # overlap in time by construction (merges only ever go
    # earlier-end -> later-start), OR is just a safe way to combine
    # them without assuming that never breaks.
    out_dir_path = Path(out_mask_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[int]] = {}
    for oid in all_ids:
        groups.setdefault(canonical[oid], []).append(oid)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    height, width = masks[all_ids[0]].shape[1:]
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for canon_id, members in groups.items():
        merged = np.zeros((total_frames, height, width), dtype=bool)
        for member in members:
            merged |= masks[member]
        path = out_dir_path / f"{canon_id}.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open mask writer at {path}")
        try:
            for t in range(total_frames):
                frame_rgb = np.repeat((merged[t].astype(np.uint8) * 255)[..., np.newaxis], 3, axis=-1)
                writer.write(frame_rgb)
        finally:
            writer.release()

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
        "groups": {str(canon): members for canon, members in groups.items()},
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
    parser.add_argument("--device", default="cpu", help="Device for the OSNet embedder")
    parser.add_argument("--no-osnet", dest="use_osnet", action="store_false", default=True,
                         help="Disable OSNet, use color (hue histogram) only")
    args = parser.parse_args()

    report = merge_fragments(
        video_path=args.video, mask_dir=args.mask_dir, out_mask_dir=args.out_dir,
        min_fragment_frames=args.min_fragment_frames, merge_threshold=args.merge_threshold,
        signature_samples=args.signature_samples, device=args.device, use_osnet=args.use_osnet,
    )

    print(f"\n{report['original_id_count']} original ids -> {report['merged_id_count']} after merging "
          f"({len(report['accepted_merges'])} merge(s) accepted, "
          f"{len(report['excluded_as_too_short'])} id(s) excluded as too short to use as a signature)")
    for m in report["accepted_merges"]:
        print(f"  id {m['from_id']} -> id {m['into_id']}  (similarity {m['similarity']})")
    if report["rejected_candidates"]:
        print(f"\n{len(report['rejected_candidates'])} candidate pair(s) considered but "
              f"below --merge-threshold ({report['merge_threshold']}):")
        for c in sorted(report["rejected_candidates"], key=lambda c: -c["similarity"]):
            print(f"  id {c['from_id']} -> id {c['into_id']}  (similarity {c['similarity']})")

    report_path = Path(args.out_dir) / "merge_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
