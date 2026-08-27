"""
segmentation/tools/overlay_subvideo.py
=======================================
Renders a colored overlay of a merged MaskDir on top of its source
clip, for visual QA. Reads `<subvideo-dir>/_window_source/camera_a_window.mp4`
+ `<subvideo-dir>/<id>.mp4`, writes `<subvideo-dir>/overlay.mp4`.

Usage:
    python -m segmentation.tools.overlay_subvideo \
        --subvideo-dir /path/to/session/merged/subvideo
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np

from segmentation.merging.mask_io import _list_mask_files


def _distinct_colors(n):
    """`n` evenly-spaced HSV colors, high saturation/brightness so they
    stay visible and clean-looking on the overlay."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR for OpenCV
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subvideo-dir", required=True)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    subvideo_dir = Path(args.subvideo_dir)
    video_path = subvideo_dir / "_window_source" / "camera_a_window.mp4"
    out_path = subvideo_dir / "overlay.mp4"

    mask_paths = _list_mask_files(str(subvideo_dir))
    ids = sorted(mask_paths.keys())
    id_to_seq = {id_: idx + 1 for idx, id_ in enumerate(ids)}  # sequential ids for label/color
    colors = _distinct_colors(len(ids))
    print(f"overlay ids: {ids} -> sequential labels {list(id_to_seq.values())}")

    video_cap = cv2.VideoCapture(str(video_path))
    fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mask_caps = {i: cv2.VideoCapture(str(mask_paths[i])) for i in ids}
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for _ in range(n_frames):
        ok, frame = video_cap.read()
        if not ok:
            break
        frame = frame.astype(np.float32)
        for i in ids:
            ok_m, mraw = mask_caps[i].read()
            if not ok_m:
                continue
            gray = mraw[:, :, 0] if mraw.ndim == 3 else mraw
            hard_mask = gray > args.threshold
            if not hard_mask.any():
                continue
            # soft edge instead of the raw pixel mask: blur the binary
            # mask -> continuous alpha that fades at the edges instead
            # of a stair-stepped outline.
            soft = cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32) / 255.0
            soft = np.clip(soft, 0.0, 1.0) * args.alpha
            color = np.array(colors[id_to_seq[i] - 1], dtype=np.float32)
            frame = frame * (1 - soft[..., None]) + color[None, None, :] * soft[..., None]
            ys, xs = np.where(hard_mask)
            if len(xs):
                cv2.putText(frame, str(id_to_seq[i]), (int(xs.min()), int(ys.min()) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(float(c) for c in color), 2)
        writer.write(frame.astype(np.uint8))

    writer.release()
    video_cap.release()
    for c in mask_caps.values():
        c.release()
    print(f"overlay written to {out_path}")


if __name__ == "__main__":
    main()
