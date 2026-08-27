"""
segmentation/merging/merge_fragments.py
========================================
Re-links fragmented identities across an ENTIRE already-produced
MaskDir, using appearance (OSNet embedding + hue-color) instead of the
geometric chunk-boundary matching real psifx's own cross-chunk
stitching relies on. Called by
`run_pipeline.py` as step 3 of 4; also runnable standalone.

Orchestration only: `merge_fragments()` below
just calls, in order -- see each module's own docstring for the actual
algorithm:
  1. `mask_io._list_mask_files`/`_scan_track` -- discover every raw
     track's `(first, last)` bounds.
  2. `overlap_resolution._resolve_overlap_merges` -- zeroth pass: ids
     simultaneously alive that are actually one body split into parts.
  3. `signatures._sample_signature`/`_pooled_group_signature` -- build
     each track's/group's appearance signature.
  4. `reappearance_merge._resolve_merges`/`_resolve_group_merges`/
     `_group_chains` -- pass 1/2: match a track that ENDS against one
     that STARTS later (global Hungarian, never greedy).
  5. Write the merged MaskDir (one file per canonical id, union of its
     members) and return the full JSON-able report.

Streams every mask video from disk instead of preloading the whole
MaskDir into RAM (a full clinical session doesn't fit in memory) --
`signature_frames` in the report records exactly which frames built
each signature, so a suspicious similarity can be checked by hand.

Zeroth-pass overlap thresholds were calibrated on 4 real cases from
9_group_1_3/9_individual_58 (positives -- same body: centroids
12-158px; negatives -- real occlusion/different people: centroids
173-400px) via real pixel-to-pixel mask distance
(`cv2.distanceTransform`, not bounding-box gap -- a bbox gap alone
false-positived on adjacent-but-different seated people) plus centroid
distance as a second filter. `DEFAULT_OVERLAP_CENTROID_THRESHOLD`'s
comment below has the exact recalibration history. Each id is
restricted to its single closest overlap match (see
`overlap_resolution._resolve_overlap_merges`'s docstring) -- without
that, one small ambiguous fragment (e.g. a static jacket near two
different people at different times) can bridge two real people
together by transitivity even though comparing them directly correctly
rejects the match.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from segmentation.merging.mask_io import DEFAULT_MASK_THRESHOLD, _list_mask_files, _scan_track
from segmentation.merging.overlap_resolution import _resolve_overlap_merges
from segmentation.merging.signatures import _pooled_group_signature, _sample_signature
from segmentation.merging.reappearance_merge import _group_chains, _resolve_group_merges, _resolve_merges
from pose.appearance_embedding import OSNetEmbedder

DEFAULT_MIN_FRAGMENT_FRAMES = 8
DEFAULT_MERGE_THRESHOLD = 0.6
DEFAULT_SIGNATURE_SAMPLES = 5
DEFAULT_POOLED_SAMPLES_PER_MEMBER = 5

# --- zeroth pass: same-time overlap (see module docstring) ---
# real pixel-to-pixel minimum distance between two masks, in px --
# small = the two masks' silhouettes actually touch/nearly touch.
DEFAULT_OVERLAP_PIXEL_GAP_THRESHOLD = 15.0
# distance between the two masks' centroids, in px -- the discriminant
# that actually separates same-body splits from different-people
# occlusions. Original calibration (4 hand-checked pairs): positives
# under ~160px, negatives over ~170px. LOWERED to 115.0 after a full
# stress test on 9_group_1_3 surfaced a false positive at 130.3px
# (2 different real people) sitting below the old 160px threshold but
# above the highest known true positive (81.3px) -- 115.0 is the
# midpoint of that gap.
DEFAULT_OVERLAP_CENTROID_THRESHOLD = 115.0
# minimum number of frames where BOTH ids are simultaneously active,
# for a pair's median to be trusted at all -- same role as
# min_fragment_frames for pass 1/2.
DEFAULT_MIN_OVERLAP_FRAMES = 8


def merge_fragments(
    *,
    video_path: str,
    mask_dir: str,
    out_mask_dir: str,
    min_fragment_frames: int = DEFAULT_MIN_FRAGMENT_FRAMES,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    signature_samples: int = DEFAULT_SIGNATURE_SAMPLES,
    pooled_samples_per_member: int = DEFAULT_POOLED_SAMPLES_PER_MEMBER,
    overlap_pixel_gap_threshold: float = DEFAULT_OVERLAP_PIXEL_GAP_THRESHOLD,
    overlap_centroid_threshold: float = DEFAULT_OVERLAP_CENTROID_THRESHOLD,
    min_overlap_frames: int = DEFAULT_MIN_OVERLAP_FRAMES,
    overlap_classifier_path: str | None = None,
    device: str = "cpu",
    use_osnet: bool = True,
) -> dict:
    """Reads `mask_dir`, merges fragmented identities via appearance
    (see module docstring), writes a NEW MaskDir at `out_mask_dir`
    (never touches the original), and returns a JSON-able report of
    every merge decision made -- accepted and, for transparency, every
    candidate pair actually considered."""
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

    # --- zeroth pass: same-body fragments that coexist in time, see
    # module docstring's ATTENZIONE section. Computed before pass 1/2 so
    # its unions are already folded into `canonical` by the time pass 2
    # builds its candidate groups from it. If `overlap_classifier_path`
    # points at a JSON produced by `train_overlap_classifier.py`, it
    # REPLACES the two fixed thresholds (see `_resolve_overlap_merges`
    # docstring) -- falls back to the thresholds if not given, which is
    # the default until enough labeled examples exist to train one. ---
    classifier = None
    if overlap_classifier_path:
        with open(overlap_classifier_path) as f:
            classifier = json.load(f)
    overlap_accepted, overlap_rejected = _resolve_overlap_merges(
        mask_paths, bounds, overlap_pixel_gap_threshold, overlap_centroid_threshold, min_overlap_frames,
        classifier=classifier,
    )
    overlap_merge_tuples = [(r["id_a"], r["id_b"], 0.0) for r in overlap_accepted]

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
    canonical = _group_chains(merges + overlap_merge_tuples, all_ids)

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
        # EVERY pass-one group is a candidate here, including a trivial
        # one-member group formed from another orphan -- excluding those
        # would also hide them as candidates for every OTHER orphan, not
        # just for themselves. Self-matching is already excluded below
        # via `if o in members: continue`.
        candidate_group_ids = list(pass1_groups.keys())

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
            canonical = _group_chains(merges + overlap_merge_tuples + extra_merges, all_ids)

    # Write the merged MaskDir: one file per distinct canonical id,
    # union (logical OR) of every member's mask. Streamed lock-step --
    # every member's file is already padded to the same total_frames
    # length, so frame t aligns across members with no seeking needed,
    # and nothing allocates a full (total_frames, height, width) array.
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
        "overlap_pixel_gap_threshold_px": overlap_pixel_gap_threshold,
        "overlap_centroid_threshold_px": overlap_centroid_threshold,
        "min_overlap_frames": min_overlap_frames,
        "overlap_classifier_used": classifier is not None,
        "overlap_classifier_path": overlap_classifier_path,
        "accepted_overlap_merges": overlap_accepted,
        "rejected_overlap_candidates": overlap_rejected,
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
    parser.add_argument("--overlap-pixel-gap-threshold", type=float, default=DEFAULT_OVERLAP_PIXEL_GAP_THRESHOLD,
                         help="Zeroth pass: max median real pixel-to-pixel distance (px) between two "
                              "temporally-overlapping masks to consider them the same body split into parts")
    parser.add_argument("--overlap-centroid-threshold", type=float, default=DEFAULT_OVERLAP_CENTROID_THRESHOLD,
                         help="Zeroth pass: max median centroid distance (px) between two temporally-"
                              "overlapping masks to consider them the same body split into parts")
    parser.add_argument("--min-overlap-frames", type=int, default=DEFAULT_MIN_OVERLAP_FRAMES,
                         help="Zeroth pass: minimum number of both-active frames required to trust a pair's "
                              "median distance at all")
    parser.add_argument("--overlap-classifier", default=None,
                         help="Path to a weights JSON from train_overlap_classifier.py -- if given, REPLACES "
                              "--overlap-pixel-gap-threshold/--overlap-centroid-threshold for the zeroth pass "
                              "(see overlap_calibration.jsonl / train_overlap_classifier.py)")
    parser.add_argument("--device", default="cpu", help="Device for the OSNet embedder")
    parser.add_argument("--no-osnet", dest="use_osnet", action="store_false", default=True,
                         help="Disable OSNet, use color (hue histogram) only")
    args = parser.parse_args()

    report = merge_fragments(
        video_path=args.video, mask_dir=args.mask_dir, out_mask_dir=args.out_dir,
        min_fragment_frames=args.min_fragment_frames, merge_threshold=args.merge_threshold,
        signature_samples=args.signature_samples, pooled_samples_per_member=args.pooled_group_samples,
        overlap_pixel_gap_threshold=args.overlap_pixel_gap_threshold,
        overlap_centroid_threshold=args.overlap_centroid_threshold,
        min_overlap_frames=args.min_overlap_frames,
        overlap_classifier_path=args.overlap_classifier,
        device=args.device, use_osnet=args.use_osnet,
    )

    print(f"\n{report['original_id_count']} original ids -> {report['merged_id_count']} after merging "
          f"({len(report['accepted_merges'])} pass-1 merge(s), "
          f"{len(report['accepted_overlap_merges'])} zeroth-pass overlap merge(s) accepted, "
          f"{len(report['excluded_as_too_short'])} id(s) excluded as too short to use as a signature)")
    if report["accepted_overlap_merges"]:
        print("\nzeroth pass -- accepted overlap merges (same body, simultaneous fragments):")
        for r in report["accepted_overlap_merges"]:
            print(f"  id {r['id_a']} <-> id {r['id_b']}  "
                  f"(median pixel gap {r['median_pixel_gap_px']}px, "
                  f"median centroid dist {r['median_centroid_dist_px']}px, "
                  f"{r['overlap_frames']} both-active frames)")
    if report["rejected_overlap_candidates"]:
        print(f"\nzeroth pass -- {len(report['rejected_overlap_candidates'])} candidate pair(s) considered but rejected:")
        for r in sorted(report["rejected_overlap_candidates"], key=lambda r: r["median_centroid_dist_px"]):
            print(f"  id {r['id_a']} <-> id {r['id_b']}  "
                  f"(median pixel gap {r['median_pixel_gap_px']}px, "
                  f"median centroid dist {r['median_centroid_dist_px']}px, "
                  f"{r['overlap_frames']} both-active frames)")
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
