"""
segmentation/tools/check_overlap_iou.py
========================================
Frame-by-frame spatial IoU + bbox/centroid distance between two raw
mask ids, over the window where they overlap in time. Diagnostic for
inspecting a specific overlap episode by hand.

Basic usage (whole temporal overlap):
    python -m segmentation.tools.check_overlap_iou \
        --masks-dir .../masks --id-a 1 --id-b 11

With a restricted window (isolate one episode inside a longer overlap):
    python -m segmentation.tools.check_overlap_iou \
        --masks-dir .../masks --id-a 2 --id-b 4 \
        --start-frame 2025 --end-frame 2550
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def _bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def _bbox_gap(box_a, box_b):
    """Minimum pixel distance between two bounding boxes: 0 if they
    touch/overlap, positive otherwise."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    dx = max(0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0, max(ay0, by0) - min(ay1, by1))
    return float(np.hypot(dx, dy))


def _centroid(mask):
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _centroid_dist(mask_a, mask_b):
    ax, ay = _centroid(mask_a)
    bx, by = _centroid(mask_b)
    return float(np.hypot(ax - bx, ay - by))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--id-a", type=int, required=True)
    parser.add_argument("--id-b", type=int, required=True)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--start-frame", type=int, default=0,
                         help="first frame to read (inclusive)")
    parser.add_argument("--end-frame", type=int, default=None,
                         help="last frame to read (exclusive); default = end of file")
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    cap_a = cv2.VideoCapture(str(masks_dir / f"{args.id_a}.mp4"))
    cap_b = cv2.VideoCapture(str(masks_dir / f"{args.id_b}.mp4"))
    n_frames = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = args.end_frame if args.end_frame is not None else n_frames

    if args.start_frame > 0:
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    rows = []
    for f in range(args.start_frame, end_frame):
        ok_a, ra = cap_a.read()
        ok_b, rb = cap_b.read()
        if not ok_a or not ok_b:
            break
        ga = ra[:, :, 0] if ra.ndim == 3 else ra
        gb = rb[:, :, 0] if rb.ndim == 3 else rb
        ma = ga > args.threshold
        mb = gb > args.threshold
        if not ma.any() or not mb.any():
            continue  # only frames where BOTH have content = actual overlap
        iou = _iou(ma, mb)
        gap = _bbox_gap(_bbox(ma), _bbox(mb))
        cdist = _centroid_dist(ma, mb)
        rows.append((f, iou, gap, cdist))

    cap_a.release()
    cap_b.release()

    if not rows:
        print(f"no temporal overlap between id {args.id_a} and id {args.id_b} "
              f"in range [{args.start_frame}, {end_frame})")
        return

    print(f"overlapping frames: {len(rows)} (from {rows[0][0]} to {rows[-1][0]})")
    for f, iou, gap, cdist in rows:
        print(f"  frame {f}: IoU = {iou:.3f}  bbox gap = {gap:6.1f}px  centroid dist = {cdist:6.1f}px")

    ious = [r[1] for r in rows]
    gaps = [r[2] for r in rows]
    cdists = [r[3] for r in rows]
    print(f"\nIoU           mean: {np.mean(ious):.3f}  min: {min(ious):.3f}  max: {max(ious):.3f}")
    print(f"bbox gap      mean: {np.mean(gaps):6.1f}px  min: {min(gaps):6.1f}px  max: {max(gaps):6.1f}px")
    print(f"centroid dist mean: {np.mean(cdists):6.1f}px  min: {min(cdists):6.1f}px  max: {max(cdists):6.1f}px")


if __name__ == "__main__":
    main()
