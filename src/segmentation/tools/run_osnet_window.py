"""
segmentation/tools/run_osnet_window.py
====================================
CLI: cuts a time window out of an existing (video, MaskDir) pair
(`subvideo.py`) and runs `merge_fragments` on just that window --
output in `<session>/merged/<out-name>/`. Useful to tune/inspect a
merge on a few minutes of footage instead of a full session.

Usage:
    python -m segmentation.tools.run_osnet_window \
        --video /path/to/session/processed/camera_a.mp4 \
        --mask-dir /path/to/session/masks/camera_a \
        --start-min 22 --end-min 27 --device cuda

Use --out-name to avoid overwriting a previously extracted window for
the same session (e.g. --out-name subvideo2 for a second window).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from segmentation.merging.mask_io import _list_mask_files
from segmentation.merging.merge_fragments import merge_fragments
from segmentation.tools.subvideo import extract_mask_window, extract_video_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--start-min", type=float, required=True)
    parser.add_argument("--end-min", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-name", default="subvideo",
                         help="Name of the output subfolder under <session>/merged/ -- change this "
                              "(e.g. 'subvideo2') when extracting a second window from the same "
                              "session, so it doesn't overwrite the first window's results")
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    session_root = mask_dir.parent.parent  # .../<session>/masks/<camera> -> <session>
    out_dir = session_root / "merged" / args.out_name
    window_dir = out_dir / "_window_source"

    probe = cv2.VideoCapture(args.video)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()
    start_frame = round(args.start_min * 60 * fps)
    end_frame = round(args.end_min * 60 * fps)
    print(f"fps={fps:.3f}  window frames [{start_frame}, {end_frame})")

    window_video = window_dir / "camera_a_window.mp4"
    extract_video_window(args.video, start_frame, end_frame, window_video)

    window_mask_dir = window_dir / "masks"
    mask_paths = _list_mask_files(str(mask_dir))
    kept = []
    for obj_id, path in sorted(mask_paths.items()):
        if extract_mask_window(path, start_frame, end_frame, window_mask_dir / f"{obj_id}.mp4", fps):
            kept.append(obj_id)
    print(f"ids with signal in this window: {kept}")

    report = merge_fragments(
        video_path=str(window_video),
        mask_dir=str(window_mask_dir),
        out_mask_dir=str(out_dir),
        device=args.device,
        use_osnet=True,
    )
    with open(out_dir / "merge_report_subvideo.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{report['original_id_count']} id -> {report['merged_id_count']} after merge")

    # Renumber the merged <canonical_id>.mp4 files to sequential ids
    # (1.mp4, 2.mp4, ...) -- merge_report_subvideo.json above keeps the
    # original ids for diagnostics; id_mapping.json below records the
    # old->new map for traceability.
    final_ids = sorted(int(k) for k in report["groups"].keys())
    id_map = {old: new for new, old in enumerate(final_ids, start=1)}
    tmp_suffix = ".renaming.mp4"
    for old_id in final_ids:
        (out_dir / f"{old_id}.mp4").rename(out_dir / f"{old_id}{tmp_suffix}")
    for old_id, new_id in id_map.items():
        (out_dir / f"{old_id}{tmp_suffix}").rename(out_dir / f"{new_id}.mp4")
    with open(out_dir / "id_mapping.json", "w") as f:
        json.dump(id_map, f, indent=2)
    print(f"files renamed: {id_map}")
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
