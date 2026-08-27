"""
segmentation/classifier/extract_overlap_candidates.py
=======================================================
Step 1 of building the overlap classifier: scans a MaskDir and dumps,
for every pair of raw ids that overlap in TIME, their zeroth-pass
geometric features to a CSV with an empty "label" column. Fill it in
by hand (1 = same body split into fragments, 0 = different people) --
watch each pair's frame range (e.g. via tools/overlay_subvideo.py) --
then feed the CSV to train_overlap_classifier.py.

Usage (whole session):
    python -m segmentation.classifier.extract_overlap_candidates \
        --session 9_group_1_3 \
        --mask-dir /home/michele/Bureau/9_group_1_3/masks/camera_a \
        --out overlap_candidates_9_group_1_3.csv

Usage (just a time window, e.g. minute 0-10) -- works directly on the
full session's raw mask dir, no separate windowed extraction needed
(--start-min/--end-min just restrict which frames get scanned):
    python -m segmentation.classifier.extract_overlap_candidates \
        --session 9_individual_58_0-10min \
        --mask-dir /home/michele/Bureau/9_individual_58/masks/camera_a \
        --start-min 0 --end-min 10 \
        --out overlap_candidates_9_individual_58_0-10min.csv

--start-min/--end-min assume 15fps -- pass --fps to override. Use
--start-frame/--end-frame instead for an exact frame range.

Run once per session/window with a distinct --session label (just a
CSV column, not a real path); append or pass multiple CSVs to
train_overlap_classifier.py, either works.

Columns written: session, id_a, id_b, overlap_start_frame,
overlap_end_frame, overlap_frames, median_pixel_gap_px,
median_centroid_dist_px, median_scale_px, median_centroid_dist_norm,
label, notes.

Not runnable in this project's sandbox (needs psifx and a real MaskDir
on disk).
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from segmentation.merging.mask_io import DEFAULT_MASK_THRESHOLD, _list_mask_files, _scan_track
from segmentation.merging.merge_fragments import DEFAULT_MIN_FRAGMENT_FRAMES, DEFAULT_MIN_OVERLAP_FRAMES
from segmentation.merging.overlap_resolution import _stream_overlap_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True,
                         help="Free-form label for this video/session, carried into the CSV "
                              "(e.g. '9_group_1_3' or '9_group_1_3_22-30min') -- doesn't need to "
                              "match any real path")
    parser.add_argument("--mask-dir", required=True,
                         help="Directory of raw <id>.mp4 mask files (the same kind of MaskDir "
                              "merge_fragments.py itself reads -- can be a full session's or a "
                              "windowed subset like run_osnet_window.py's _window_source/masks)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--min-fragment-frames", type=int, default=DEFAULT_MIN_FRAGMENT_FRAMES,
                         help="Ids with fewer non-empty frames than this are skipped entirely "
                              "(too short/noisy to trust, same rule merge_fragments.py itself uses)")
    parser.add_argument("--min-overlap-frames", type=int, default=DEFAULT_MIN_OVERLAP_FRAMES,
                         help="Minimum both-active frames required to trust a pair's median at all")
    parser.add_argument("--mask-threshold", type=int, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--start-frame", type=int, default=None,
                         help="Restrict extraction to frames >= this (inclusive). Default: from the "
                              "start of the mask files. Mutually exclusive with --start-min.")
    parser.add_argument("--end-frame", type=int, default=None,
                         help="Restrict extraction to frames <= this (inclusive). Default: to the "
                              "end of the mask files. Mutually exclusive with --end-min.")
    parser.add_argument("--start-min", type=float, default=None,
                         help="Same as --start-frame but in minutes of video time (converted via --fps)")
    parser.add_argument("--end-min", type=float, default=None,
                         help="Same as --end-frame but in minutes of video time (converted via --fps)")
    parser.add_argument("--fps", type=float, default=15.0,
                         help="Used only to convert --start-min/--end-min to frame numbers -- the "
                              "confirmed rate for these recordings (see merge_fragments.py's "
                              "ATTENZIONE section); override if a session's source differs")
    args = parser.parse_args()

    if args.start_min is not None and args.start_frame is not None:
        raise SystemExit("pass either --start-min or --start-frame, not both")
    if args.end_min is not None and args.end_frame is not None:
        raise SystemExit("pass either --end-min or --end-frame, not both")
    win_start = args.start_frame
    if args.start_min is not None:
        win_start = round(args.start_min * 60 * args.fps)
    win_end = args.end_frame
    if args.end_min is not None:
        win_end = round(args.end_min * 60 * args.fps) - 1  # inclusive, matching --end-frame
    if win_start is not None or win_end is not None:
        print(f"restricting to frame window [{win_start if win_start is not None else 0}, "
              f"{win_end if win_end is not None else 'end'}]"
              + (f"  ({args.start_min}-{args.end_min}min @ {args.fps}fps)"
                 if args.start_min is not None or args.end_min is not None else ""))

    mask_paths = _list_mask_files(args.mask_dir)
    if not mask_paths:
        raise SystemExit(f"No <id>.mp4 mask files found in {args.mask_dir}")

    bounds: dict[int, tuple[int, int]] = {}
    for obj_id, path in sorted(mask_paths.items()):
        scan = _scan_track(path)
        if scan is None:
            continue
        first, last, real_frames, _decoded, _h, _w = scan
        if real_frames < args.min_fragment_frames:
            print(f"id {obj_id}: skipped, only {real_frames} non-empty frames "
                  f"(< --min-fragment-frames {args.min_fragment_frames})")
            continue
        bounds[obj_id] = (first, last)

    ids = sorted(bounds.keys())
    print(f"{len(ids)} usable id(s): {ids}")

    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            a_first, a_last = bounds[a]
            b_first, b_last = bounds[b]
            overlap_start = max(a_first, b_first)
            overlap_end = min(a_last, b_last)
            if win_start is not None:
                overlap_start = max(overlap_start, win_start)
            if win_end is not None:
                overlap_end = min(overlap_end, win_end)
            if overlap_start > overlap_end:
                continue  # no temporal overlap within the requested window

            pixel_dists, centroid_dists, scales = _stream_overlap_stats(
                mask_paths[a], mask_paths[b], overlap_start, overlap_end, args.mask_threshold,
            )
            n = len(pixel_dists)
            if n < args.min_overlap_frames:
                continue

            median_pixel = float(np.median(pixel_dists))
            median_centroid = float(np.median(centroid_dists))
            median_scale = float(np.median(scales))
            median_norm = median_centroid / median_scale if median_scale > 0 else float("inf")
            rows.append({
                "session": args.session,
                "id_a": a,
                "id_b": b,
                "overlap_start_frame": overlap_start,
                "overlap_end_frame": overlap_end,
                "overlap_frames": n,
                "median_pixel_gap_px": round(median_pixel, 1),
                "median_centroid_dist_px": round(median_centroid, 1),
                "median_scale_px": round(median_scale, 1),
                "median_centroid_dist_norm": round(median_norm, 4),
                "label": "",
                "notes": "",
            })
            print(f"  id {a} <-> id {b}: frames [{overlap_start}, {overlap_end}] "
                  f"({n} both-active) -- centroid dist {median_centroid:.1f}px "
                  f"(norm {median_norm:.3f}), pixel gap {median_pixel:.1f}px")

    if not rows:
        print("\nNo overlapping candidate pairs found -- nothing to write.")
        return

    fieldnames = ["session", "id_a", "id_b", "overlap_start_frame", "overlap_end_frame",
                  "overlap_frames", "median_pixel_gap_px", "median_centroid_dist_px",
                  "median_scale_px", "median_centroid_dist_norm", "label", "notes"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} candidate pair(s) written to {args.out}")
    print("Fill in the 'label' column: 1 = same body split into simultaneous fragments "
          "(merge, like giacca/mantellina), 0 = different real people/objects (keep separate, "
          "e.g. a genuine occlusion). Leave blank if unsure -- unlabeled rows are skipped by "
          "train_overlap_classifier.py.")


if __name__ == "__main__":
    main()
