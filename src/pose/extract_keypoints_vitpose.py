"""
pose/extract_keypoints_vitpose.py
===================================
ViTPose alternative to `pose/extract_keypoints.py` (MediaPipe), chosen
after `pose.benchmark_pose` on 10_individual_12 camera_b: with the person
MASKED (everything outside its own dilated mask set to black) ViTPose put
the fewest keypoints on other people and on the background, and gave a pose
for every person in every frame.

Same identity design as `extract_keypoints.py`: who is who comes from the
merged MaskDir, never from the pose model. Per frame and person:
  1. box from the LARGEST connected blob of the mask only (stray fragments,
     e.g. the camera's lateral border artifact, cannot stretch it), padded;
  2. the frame with everything outside the person's dilated mask blacked
     out, so neighbours inside the box are invisible to the model;
  3. ViTPose (top-down, from Hugging Face `transformers`) on that box. All
     people of a frame go through the model in ONE batched call.

Same output schema as `extract_keypoints.py` (so `stabilization`, `depth`,
`movement_metrics` work unchanged), plus one column:

    session_id, global_person_id, frame, timestamp_ms, keypoint_name, x, y,
    confidence, in_own_mask

`confidence` is ViTPose's heatmap score. `in_own_mask` is False when the
keypoint lies outside its person's dilated mask: typically a hidden body
part the model completed (legs behind the table) -- fine in 2D, but the
depth at that pixel belongs to whatever hides it, so `pose/depth.py` does
not read depth there when the column is present.

`x_offset`/`y_offset` shift the output back to full-frame coordinates when
the input is the border-cropped video/masks (see `pose.run_pose_cropped`).

Requires `transformers>=4.48` (ViTPose support) and torch.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose.keypoints import COCO17

DEFAULT_MODEL = "usyd-community/vitpose-base-simple"
MASK_THRESHOLD = 127
MIN_CROP_W, MIN_CROP_H = 40, 60  # same as extract_keypoints.py


def _mask_files(mask_dir: Path) -> dict[int, Path]:
    files = {int(p.stem): p for p in mask_dir.glob("*.mp4") if p.stem.isdigit()}
    if not files:
        raise ValueError(f"No '<id>.mp4' mask files found in {mask_dir}")
    return dict(sorted(files.items()))


def _largest_blob(m: np.ndarray) -> np.ndarray:
    n, labels, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    return labels == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


def _box(m: np.ndarray, w: int, h: int, padding: float):
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return (int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py)))


def extract_keypoints_vitpose(
    *,
    video_path: str,
    merged_mask_dir: str,
    out_csv: str,
    session_id: str,
    model_name: str = DEFAULT_MODEL,
    device: str = "gpu",
    bbox_padding: float = 0.15,
    dilate_px: int = 12,
    x_offset: int = 0,
    y_offset: int = 0,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> str:
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    dev = "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    proc = AutoProcessor.from_pretrained(model_name)
    model = VitPoseForPoseEstimation.from_pretrained(model_name).to(dev).eval()
    print(f"[vitpose] {model_name} on {dev}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mcaps = {pid: cv2.VideoCapture(str(p)) for pid, p in _mask_files(Path(merged_mask_dir)).items()}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    end = n_total if end_frame is None else min(end_frame, n_total)

    rows: list[tuple] = []
    n_small = n_err = 0
    f = 0
    try:
        while f < end:
            reading = f >= start_frame
            ok, img = cap.read() if reading else (cap.grab(), None)
            masks = {pid: (mc.read() if reading else (mc.grab(), None)) for pid, mc in mcaps.items()}
            if not ok:
                break
            if not reading:
                f += 1
                continue

            pids, images, boxes, dils = [], [], [], []
            for pid, (okm, mf) in masks.items():
                if not okm:
                    continue
                m = (mf[:, :, 0] if mf.ndim == 3 else mf) > MASK_THRESHOLD
                if not m.any():
                    continue
                b = _box(_largest_blob(m), w, h, bbox_padding)
                if b is None or b[2] - b[0] < MIN_CROP_W or b[3] - b[1] < MIN_CROP_H:
                    n_small += 1
                    continue
                d = cv2.dilate(m.astype(np.uint8), kernel)
                images.append(cv2.cvtColor(img * d[:, :, None], cv2.COLOR_BGR2RGB))
                boxes.append([[b[0], b[1], b[2] - b[0], b[3] - b[1]]])
                pids.append(pid)
                dils.append(d.astype(bool))

            if pids:
                try:
                    inputs = proc(images, boxes=boxes, return_tensors="pt").to(dev)
                    kw = {}
                    if "plus" in model_name:  # ViTPose++: head 0 = COCO-17
                        kw["dataset_index"] = torch.zeros(inputs["pixel_values"].shape[0], dtype=torch.long,
                                                          device=dev)
                    with torch.no_grad():
                        out = model(**inputs, **kw)
                    res = proc.post_process_pose_estimation(out, boxes=boxes)
                except Exception as e:  # never lose a whole session to one odd frame
                    n_err += 1
                    if n_err <= 3:
                        print(f"[vitpose] frame {f}: {type(e).__name__}: {e} (frame skipped)")
                    res = None
                if res is not None:
                    ts = int(round(f / fps * 1000))
                    for pid, r, d in zip(pids, res, dils):
                        kp = r[0]["keypoints"].cpu().numpy()[:17]
                        sc = r[0]["scores"].cpu().numpy()[:17]
                        xi = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
                        yi = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
                        inside = d[yi, xi]
                        for k in range(17):
                            rows.append((session_id, pid, f, ts, COCO17[k], float(kp[k, 0]) + x_offset,
                                         float(kp[k, 1]) + y_offset, float(sc[k]), bool(inside[k])))
            f += 1
            if f % 1500 == 0:
                print(f"[vitpose] frame {f}/{end}")
    finally:
        cap.release()
        for mc in mcaps.values():
            mc.release()

    if n_small:
        print(f"[vitpose] skipped {n_small} person/frame crops smaller than {MIN_CROP_W}x{MIN_CROP_H}px")
    if n_err:
        print(f"[vitpose] {n_err} frame(s) skipped after a model error")
    df = pd.DataFrame(rows, columns=["session_id", "global_person_id", "frame", "timestamp_ms", "keypoint_name",
                                     "x", "y", "confidence", "in_own_mask"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[vitpose] {len(df)} keypoint rows over {f - start_frame} frames, {len(mcaps)} identities, "
          f"{(~df['in_own_mask']).mean():.1%} outside own mask -> {out_csv}")
    return str(out_csv)
