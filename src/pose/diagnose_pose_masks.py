"""
pose/diagnose_pose_masks.py
=============================
Shows, frame by frame, how each person's MASK, the CROP box derived from it
(`extract_keypoints._bbox_from_mask`, exactly as the pose step computes it)
and the resulting SKELETON relate -- to check whether skeleton jumps come
from the mask (stray pixels, e.g. the camera's lateral border artifact,
stretching the crop box) rather than from MediaPipe itself.

Two outputs:
- a video (`--out-video`) for a time window: mask tinted per person, the
  crop box (RED when it reaches the `--border-px` band at the frame edge,
  person colour otherwise), the skeleton, and a small label with the
  number of separate mask blobs;
- a per-(person, frame) CSV (`--out-csv`) over the WHOLE session (or the
  window, with `--stats-window-only`): crop box size, whether it touches
  the border band, number of mask blobs, area of the largest blob vs the
  whole mask, torso jump in pixels from the previous frame. At the end it
  prints how much more often large torso jumps happen when the box
  touches the border / when the mask is fragmented -- the number that
  says whether the hypothesis holds.

Usage:
    cd src
    python -m pose.diagnose_pose_masks \\
        --video .../processed/camera_b.mp4 \\
        --mask-dir .../merged/camera_b \\
        --keypoints-csv .../pose/camera_b/keypoints_smoothed.csv \\
        --start-s 120 --end-s 180 \\
        --out-video .../pose/camera_b/diag_masks_pose.mp4 \\
        --out-csv .../pose/camera_b/diag_masks_pose.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose.extract_keypoints import _MASK_THRESHOLD, _bbox_from_mask, _list_mask_files
from pose.keypoints import SKELETON_EDGES

COLORS = [(40, 220, 255), (120, 255, 120), (255, 160, 60), (200, 120, 255),
          (255, 120, 200), (80, 180, 255), (180, 220, 90), (220, 180, 90)]
TORSO = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _edge_names():
    from pose.keypoints import COCO17
    out = []
    for a, b in SKELETON_EDGES:
        out.append((COCO17[a] if isinstance(a, int) else a, COCO17[b] if isinstance(b, int) else b))
    return out


def mask_stats(mask: np.ndarray, w: int, h: int, padding: float, border_px: int) -> dict | None:
    m = (mask[:, :, 0] if mask.ndim == 3 else mask) > _MASK_THRESHOLD
    if not m.any():
        return None
    n, labels, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    x1, y1, x2, y2 = _bbox_from_mask(mask, w, h, padding)
    ys, xs = np.nonzero(m)
    near_border = bool((xs < border_px).any() or (xs >= w - border_px).any()
                       or (ys < border_px).any() or (ys >= h - border_px).any())
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "box_w": x2 - x1, "box_h": y2 - y1,
            "mask_px": int(m.sum()), "n_blobs": int(n - 1),
            "largest_blob_frac": float(areas.max() / areas.sum()),
            "mask_touches_border": near_border}


def torso_centers(kp: pd.DataFrame) -> pd.DataFrame:
    t = kp[kp["keypoint_name"].isin(TORSO)].dropna(subset=["x", "y"])
    return t.groupby(["global_person_id", "frame"])[["x", "y"]].mean()


def main() -> None:
    p = argparse.ArgumentParser(description="Masks + crop boxes + skeletons, to correlate skeleton jumps with "
                                            "mask artifacts -- see module docstring.")
    p.add_argument("--video", required=True, help="the video the pose step ran on (processed/<name>.mp4)")
    p.add_argument("--mask-dir", required=True, help="the merged MaskDir the pose step used")
    p.add_argument("--keypoints-csv", required=True, help="keypoints_smoothed.csv (or keypoints_raw.csv)")
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--end-s", type=float, default=60.0)
    p.add_argument("--out-video", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--stats-window-only", action="store_true", help="compute the CSV only on the video window")
    p.add_argument("--bbox-padding", type=float, default=0.15, help="must match the pose run (default 0.15)")
    p.add_argument("--border-px", type=int, default=30, help="width of the frame-edge band")
    p.add_argument("--jump-px", type=float, default=40.0, help="torso jump counted as 'large'")
    a = p.parse_args()

    kp = pd.read_csv(a.keypoints_csv)
    centers = torso_centers(kp)
    kp_idx = kp.dropna(subset=["x", "y"]).set_index(["global_person_id", "frame"])
    edges = _edge_names()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    masks = {pid: cv2.VideoCapture(str(pth)) for pid, pth in _list_mask_files(Path(a.mask_dir)).items()}
    f0, f1 = int(a.start_s * fps), int(a.end_s * fps)
    scan_from = f0 if a.stats_window_only else 0

    if scan_from > 0:
        for c in [cap, *masks.values()]:
            c.set(cv2.CAP_PROP_POS_FRAMES, scan_from)
    writer = cv2.VideoWriter(a.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows = []
    frame = scan_from
    while True:
        need_video = f0 <= frame < f1
        if a.stats_window_only and frame >= f1:
            break
        ok, img = cap.read() if need_video else (cap.grab(), None)
        if not ok:
            break
        canvas = img.copy() if need_video else None
        for pid, mc in masks.items():
            okm, m = mc.read()
            if not okm:
                continue
            s = mask_stats(m, w, h, a.bbox_padding, a.border_px)
            if s is None:
                continue
            s.update(global_person_id=pid, frame=frame)
            rows.append(s)
            if need_video:
                col = COLORS[pid % len(COLORS)]
                mm = (m[:, :, 0] if m.ndim == 3 else m) > _MASK_THRESHOLD
                canvas[mm] = (0.55 * canvas[mm] + 0.45 * np.array(col)).astype(np.uint8)
                box_touch = s["x1"] <= a.border_px or s["y1"] <= a.border_px or \
                    s["x2"] >= w - a.border_px or s["y2"] >= h - a.border_px
                cv2.rectangle(canvas, (s["x1"], s["y1"]), (s["x2"], s["y2"]),
                              (0, 0, 255) if box_touch else col, 3 if box_touch else 1)
                cv2.putText(canvas, f"id{pid} blobs:{s['n_blobs']}", (s["x1"] + 4, s["y1"] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if (pid, frame) in kp_idx.index:
                    pts = kp_idx.loc[(pid, frame)]
                    pts = pts.set_index("keypoint_name")[["x", "y"]] if isinstance(pts, pd.DataFrame) else None
                    if pts is not None:
                        for ka, kb in edges:
                            if ka in pts.index and kb in pts.index:
                                pa = tuple(int(v) for v in pts.loc[ka]); pb = tuple(int(v) for v in pts.loc[kb])
                                cv2.line(canvas, pa, pb, (255, 255, 255), 4)
                                cv2.line(canvas, pa, pb, col, 2)
        if need_video:
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 180), 1)
            cv2.putText(canvas, f"frame {frame}  t={frame / fps:7.2f}s", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(canvas)
        frame += 1
        if frame % 1500 == 0:
            print(f"[diag] frame {frame}")
    writer.release()

    st = pd.DataFrame(rows)
    st["box_touches_border"] = (st["x1"] <= a.border_px) | (st["y1"] <= a.border_px) | \
        (st["x2"] >= w - a.border_px) | (st["y2"] >= h - a.border_px)
    c = centers.reset_index().sort_values(["global_person_id", "frame"])
    c["torso_jump_px"] = np.where(c.groupby("global_person_id")["frame"].diff() == 1,
                                  np.hypot(c.groupby("global_person_id")["x"].diff(),
                                           c.groupby("global_person_id")["y"].diff()), np.nan)
    st = st.merge(c[["global_person_id", "frame", "torso_jump_px"]], on=["global_person_id", "frame"], how="left")
    st.to_csv(a.out_csv, index=False)

    j = st.dropna(subset=["torso_jump_px"])
    big = j["torso_jump_px"] > a.jump_px
    print(f"\n[diag] {len(j)} person-frames with a torso jump measured; {big.mean():.2%} are > {a.jump_px:.0f} px")
    for col, label in [("box_touches_border", "crop box reaches the border band"),
                       ("mask_touches_border", "mask has pixels in the border band")]:
        on = j[col]
        if on.any() and (~on).any():
            print(f"[diag] {label}: {on.mean():.1%} of frames | large jumps {big[on].mean():.2%} when yes "
                  f"vs {big[~on].mean():.2%} when no")
    frag = j["n_blobs"] > 1
    if frag.any() and (~frag).any():
        print(f"[diag] mask in >1 blob: {frag.mean():.1%} of frames | large jumps {big[frag].mean():.2%} when yes "
              f"vs {big[~frag].mean():.2%} when no")
    print(f"[diag] wrote {a.out_video} and {a.out_csv}")


if __name__ == "__main__":
    main()
