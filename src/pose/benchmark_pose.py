"""
pose/benchmark_pose.py
========================
Side-by-side comparison of pose-estimation methods on a short window of a
session, on the BORDER-CROPPED video + masks (`merged/<name>_cropped*/`),
so the lateral border artifact cannot influence the masks or the crops.

Each "config" = backend + how the person is cut out of the frame:
  mp_lite_box      MediaPipe PoseLandmarker lite, plain mask-bbox crop
                   (= what pose.run_pose does today)
  mp_heavy_box     MediaPipe heavy, plain bbox crop
  mp_heavy_masked  MediaPipe heavy, crop with everything OUTSIDE the
                   person's (dilated) mask blacked out
  rtmpose_masked   RTMPose-m (rtmlib, ONNX) on the masked frame + box
  vitpose_masked   ViTPose (HF transformers) on the masked frame + box

"masked" configs also take the box from the LARGEST connected blob of the
mask only, so stray fragments cannot stretch it.

Outputs (in --out-dir):
  <config>.csv            keypoints (COCO-17, crop-pixel coordinates; add the
                          crop offset, e.g. +24, to compare with full frames)
  metrics.csv             per config x person: coverage, jitter, keypoints
                          falling outside the person's mask
  benchmark.mp4           one panel per config, same frames, side by side:
                          mask tinted per person, crop box, skeleton; a
                          keypoint on ANOTHER person's mask = red cross
                          (latched onto someone else), on the background =
                          grey circle (hidden part completed by the model)
  benchmark_clean.mp4     same, without the keypoints outside their own mask
                          and the segments touching them
  --render-only           redraws both videos from the saved CSVs, no models
  metrics printed at the end.

Metrics (all on the RAW model output, no smoothing, so they measure the
model, not the filter):
  coverage        frames with a pose / frames where the person has a mask
  jitter_med/p95  frame-to-frame movement of the torso centre, as a
                  fraction of the person's box height (scale-free: a far
                  person is not penalised for being small)
  on_other_person        share of confident keypoints lying on ANOTHER
                         person's mask -- the wrong-identity signal
  torso_on_other_person  same, shoulders/hips only: the clearest sign of a
                         skeleton attached to the wrong person
  on_background          keypoints outside their mask on the scene: mostly
                         hidden parts the model completed (not an error in
                         2D, but no usable depth there)

Missing optional backends are skipped with the install command printed:
  python -m pip install rtmlib onnxruntime-gpu      (rtmpose)
  python -m pip install "transformers>=4.48"        (vitpose)

Usage:
    cd src
    python -m pose.benchmark_pose \\
        --crop-dir ".../10_individual_12/merged/camera_b_cropped" \\
        --start-s 60 --end-s 180 \\
        --out-dir ".../10_individual_12/pose/benchmark_camera_b"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose.keypoints import COCO17, SKELETON_EDGES

MASK_THRESHOLD = 127
COLORS = [(40, 220, 255), (120, 255, 120), (255, 160, 60), (200, 120, 255),
          (255, 120, 200), (80, 180, 255), (180, 220, 90), (220, 180, 90)]
TORSO = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
ALL_CONFIGS = ["mp_lite_box", "mp_heavy_box", "mp_heavy_masked", "rtmpose_masked", "vitpose_masked",
               "vitpose_plus_huge_masked", "sapiens_1b_masked"]

RTMPOSE_M_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")
VITPOSE_MODEL = "usyd-community/vitpose-base-simple"


# ----------------------------------------------------------------------------
# Masks and boxes
# ----------------------------------------------------------------------------

def binarize(mask_frame: np.ndarray) -> np.ndarray:
    m = mask_frame[:, :, 0] if mask_frame.ndim == 3 else mask_frame
    return m > MASK_THRESHOLD


def largest_blob(m: np.ndarray) -> np.ndarray:
    n, labels, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    return labels == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


MIN_BOX_W, MIN_BOX_H = 20, 30  # same spirit as extract_keypoints' 40x60 skip, before padding


def box_of(m: np.ndarray, w: int, h: int, padding: float = 0.15):
    """Padded box of the mask; None when empty or too thin to hold a body
    (a 1-pixel sliver gives a zero-width box, which ViTPose's affine warp
    cannot invert)."""
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    if xs.max() - xs.min() < MIN_BOX_W or ys.max() - ys.min() < MIN_BOX_H:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return (int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py)))


# ----------------------------------------------------------------------------
# Backends: each returns {pid: (17x2 xy, 17 scores)} for one frame
# ----------------------------------------------------------------------------

class MediaPipeBackend:
    # MediaPipe BlazePose 33 landmarks -> COCO-17 (same mapping as pose/extract_keypoints.py)
    BLAZE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(self, variant: str, device: str, masked: bool, fps: float):
        from pose.extract_keypoints import _create_landmarker, _resolve_model_path
        self._create = lambda: _create_landmarker(_resolve_model_path(None, variant), device, 0.5)
        self.masked, self.fps, self.lm, self.last_ts = masked, fps, {}, {}

    def __call__(self, frame_idx, img, masks, boxes):
        import mediapipe as mp
        out = {}
        for pid, box in boxes.items():
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if x2 - x1 < 20 or y2 - y1 < 30:
                continue
            src = img
            if self.masked:
                src = img * masks[pid][:, :, None]
            crop = np.ascontiguousarray(cv2.cvtColor(src[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            lm = self.lm.get(pid) or self.lm.setdefault(pid, self._create())
            ts = max(int(round(frame_idx / self.fps * 1000)), self.last_ts.get(pid, -1) + 1)
            self.last_ts[pid] = ts
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), ts)
            if not res.pose_landmarks:
                continue
            L = res.pose_landmarks[0]
            xy = np.array([[x1 + L[i].x * (x2 - x1), y1 + L[i].y * (y2 - y1)] for i in self.BLAZE_TO_COCO])
            sc = np.array([L[i].visibility for i in self.BLAZE_TO_COCO])
            out[pid] = (xy, sc)
        return out


class RTMPoseBackend:
    def __init__(self, device: str):
        from rtmlib import RTMPose
        self.model = RTMPose(onnx_model=RTMPOSE_M_URL, model_input_size=(192, 256),
                             backend="onnxruntime", device="cuda" if device == "gpu" else "cpu")

    def __call__(self, frame_idx, img, masks, boxes):
        out = {}
        for pid, box in boxes.items():
            if box is None:
                continue
            masked = img * masks[pid][:, :, None]
            kps, scores = self.model(masked, bboxes=[list(box)])
            out[pid] = (np.asarray(kps)[0][:17], np.asarray(scores)[0][:17])
        return out


class ViTPoseBackend:
    """Any ViTPose / ViTPose++ checkpoint from Hugging Face. All people of a
    frame go through the model in one batched call. ViTPose++ ("plus")
    checkpoints have one head per training dataset: dataset_index 0 = COCO
    (17 keypoints)."""

    def __init__(self, device: str, model_name: str = VITPOSE_MODEL):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.torch = torch
        self.dev = "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name).to(self.dev).eval()
        self.plus = "plus" in model_name

    def __call__(self, frame_idx, img, masks, boxes):
        pids, images, bxs = [], [], []
        for pid, box in boxes.items():
            if box is None:
                continue
            x1, y1, x2, y2 = box
            images.append(cv2.cvtColor(img * masks[pid][:, :, None], cv2.COLOR_BGR2RGB))
            bxs.append([[x1, y1, x2 - x1, y2 - y1]])
            pids.append(pid)
        if not pids:
            return {}
        inputs = self.proc(images, boxes=bxs, return_tensors="pt").to(self.dev)
        kw = {}
        if self.plus:
            kw["dataset_index"] = self.torch.zeros(inputs["pixel_values"].shape[0], dtype=self.torch.long,
                                                   device=self.dev)
        with self.torch.no_grad():
            o = self.model(**inputs, **kw)
        res = self.proc.post_process_pose_estimation(o, boxes=bxs)
        return {pid: (r[0]["keypoints"].cpu().numpy()[:17], r[0]["scores"].cpu().numpy()[:17])
                for pid, r in zip(pids, res)}


class SapiensBackend:
    """Meta Sapiens pose (TorchScript "lite" checkpoints from Hugging Face,
    facebook/sapiens-pose-<size>-torchscript), 308 Goliath keypoints, of
    which the first 17 are the COCO body keypoints in COCO order (same
    convention as normalize_sapiens_pose_predictions.py in the
    test-depth-anything repo). Top-down: masked person crop, padded to the
    model's 3:4 aspect, resized to 768x1024, heatmap argmax with a
    quarter-pixel refinement."""

    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)
    IN_W, IN_H = 768, 1024

    def __init__(self, device: str, size: str = "1b"):
        import torch
        from huggingface_hub import hf_hub_download, list_repo_files
        self.torch = torch
        self.dev = "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        repo = f"facebook/sapiens-pose-{size}-torchscript"
        files = [f for f in list_repo_files(repo) if f.endswith((".pt2", ".pt"))]
        if not files:
            raise FileNotFoundError(f"no TorchScript file in {repo}")
        path = hf_hub_download(repo, files[0])
        self.model = torch.jit.load(path, map_location=self.dev).eval()
        self.half = self.dev == "cuda"
        if self.half:
            self.model = self.model.half()

    def _crop(self, img, box):
        x1, y1, x2, y2 = box
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if bw / bh > self.IN_W / self.IN_H:
            bh = bw * self.IN_H / self.IN_W
        else:
            bw = bh * self.IN_W / self.IN_H
        sx, sy = self.IN_W / bw, self.IN_H / bh
        M = np.array([[sx, 0, -(cx - bw / 2) * sx], [0, sy, -(cy - bh / 2) * sy]], np.float32)
        crop = cv2.warpAffine(img, M, (self.IN_W, self.IN_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        return crop, (cx - bw / 2, cy - bh / 2, bw / self.IN_W, bh / self.IN_H)

    def __call__(self, frame_idx, img, masks, boxes):
        pids, batch, tfs = [], [], []
        for pid, box in boxes.items():
            if box is None:
                continue
            crop, tf = self._crop(img * masks[pid][:, :, None], box)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            batch.append(((rgb - self.MEAN) / self.STD).transpose(2, 0, 1))
            pids.append(pid)
            tfs.append(tf)
        if not pids:
            return {}
        t = self.torch.from_numpy(np.stack(batch)).to(self.dev)
        with self.torch.no_grad():
            hm = self.model(t.half() if self.half else t).float().cpu().numpy()[:, :17]  # (B,17,h,w)
        out = {}
        for pid, h, (ox, oy, kx, ky) in zip(pids, hm, tfs):
            K, H, W = h.shape
            flat = h.reshape(K, -1)
            idx = flat.argmax(1)
            py, px = (idx // W).astype(float), (idx % W).astype(float)
            for k in range(K):  # quarter-pixel shift toward the higher neighbour
                ix, iy = int(px[k]), int(py[k])
                if 0 < ix < W - 1:
                    px[k] += 0.25 * np.sign(h[k, iy, ix + 1] - h[k, iy, ix - 1])
                if 0 < iy < H - 1:
                    py[k] += 0.25 * np.sign(h[k, iy + 1, ix] - h[k, iy - 1, ix])
            sx, sy = self.IN_W / W, self.IN_H / H
            xy = np.c_[ox + (px + 0.5) * sx * kx, oy + (py + 0.5) * sy * ky]
            out[pid] = (xy, np.clip(flat.max(1), 0, 1))
        return out


VITPOSE_CONFIGS = {
    "vitpose_masked": VITPOSE_MODEL,
    "vitpose_plus_base_masked": "usyd-community/vitpose-plus-base",
    "vitpose_plus_large_masked": "usyd-community/vitpose-plus-large",
    "vitpose_plus_huge_masked": "usyd-community/vitpose-plus-huge",
}
SAPIENS_CONFIGS = {"sapiens_0.3b_masked": "0.3b", "sapiens_0.6b_masked": "0.6b",
                   "sapiens_1b_masked": "1b", "sapiens_2b_masked": "2b"}


def make_backend(cfg: str, device: str, fps: float):
    if cfg == "mp_lite_box":
        return MediaPipeBackend("lite", device, False, fps)
    if cfg == "mp_heavy_box":
        return MediaPipeBackend("heavy", device, False, fps)
    if cfg == "mp_heavy_masked":
        return MediaPipeBackend("heavy", device, True, fps)
    if cfg == "rtmpose_masked":
        return RTMPoseBackend(device)
    if cfg in VITPOSE_CONFIGS:
        return ViTPoseBackend(device, VITPOSE_CONFIGS[cfg])
    if cfg in SAPIENS_CONFIGS:
        return SapiensBackend(device, SAPIENS_CONFIGS[cfg])
    raise ValueError(f"unknown config {cfg}")


INSTALL_HINT = {"rtmpose_masked": 'python -m pip install rtmlib onnxruntime-gpu',
                **{c: 'python -m pip install "transformers>=4.48"' for c in VITPOSE_CONFIGS},
                **{c: 'python -m pip install huggingface_hub  (Sapiens TorchScript weights download)'
                   for c in SAPIENS_CONFIGS}}


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------

def classify_outside(xy, sc, pid, dil, raw_masks, w, h, score_thr):
    """Per keypoint: '' (inside its own dilated mask, or low confidence),
    'other' (on ANOTHER person's mask: likely latched onto someone else) or
    'background' (on the scene: usually a hidden body part the model
    completed, e.g. legs behind the table -- plausible in 2D, but the depth
    there is the occluder's, not the person's)."""
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    kind = np.full(17, "", dtype=object)
    out = (~dil[pid][yi, xi]) & (sc >= score_thr)
    on_other = np.zeros(17, bool)
    for q, m in raw_masks.items():
        if q != pid:
            on_other |= m[yi, xi]
    kind[out & on_other] = "other"
    kind[out & ~on_other] = "background"
    return kind


def draw_panel(img, masks, boxes, poses, label, kinds, score_thr, hide_outside: bool):
    """hide_outside=False: every keypoint drawn; 'other' = red cross,
    'background' = grey hollow circle. hide_outside=True: keypoints outside
    their own mask, and every segment touching them, are not drawn."""
    canvas = img.copy()
    for pid, m in masks.items():
        if m.any():
            col = np.array(COLORS[pid % len(COLORS)])
            canvas[m] = (0.6 * canvas[m] + 0.4 * col).astype(np.uint8)
    for pid, box in boxes.items():
        if box is not None:
            cv2.rectangle(canvas, box[:2], box[2:], COLORS[pid % len(COLORS)], 1)
    for pid, (xy, sc) in poses.items():
        col = COLORS[pid % len(COLORS)]
        kind = kinds.get(pid, np.full(17, "", dtype=object))
        ok = sc >= score_thr
        if hide_outside:
            ok = ok & (kind == "")
        for a, b in SKELETON_EDGES:
            ia, ib = COCO17.index(a), COCO17.index(b)
            if ok[ia] and ok[ib]:
                pa, pb = tuple(int(v) for v in xy[ia]), tuple(int(v) for v in xy[ib])
                cv2.line(canvas, pa, pb, (255, 255, 255), 4)
                cv2.line(canvas, pa, pb, col, 2)
        for k in range(17):
            if not ok[k]:
                continue
            p = tuple(int(v) for v in xy[k])
            if kind[k] == "other":
                cv2.drawMarker(canvas, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 3)
            elif kind[k] == "background":
                cv2.circle(canvas, p, 6, (170, 170, 170), 2)
            else:
                cv2.circle(canvas, p, 3, col, -1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def load_saved_poses(out: Path, names: list[str]) -> dict:
    """{cfg: {frame: {pid: (xy, score)}}} from the <cfg>.csv of a previous run."""
    saved = {}
    for cfg in names:
        df = pd.read_csv(out / f"{cfg}.csv")
        df["k"] = df["keypoint_name"].map(COCO17.index)
        d = {}
        for (f, pid), g in df.sort_values("k").groupby(["frame", "global_person_id"]):
            if len(g) == 17:
                d.setdefault(int(f), {})[int(pid)] = (g[["x", "y"]].to_numpy(), g["score"].to_numpy())
        saved[cfg] = d
    return saved


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Compare pose methods on a window -- see module docstring.")
    p.add_argument("--crop-dir", required=True, help="merged/<name>_cropped*/ (masks <id>.mp4 + _source/*.mp4)")
    p.add_argument("--video", default=None, help="override the video (default: the one in <crop-dir>/_source/)")
    p.add_argument("--start-s", type=float, default=60.0)
    p.add_argument("--end-s", type=float, default=180.0)
    p.add_argument("--configs", default=",".join(ALL_CONFIGS),
                   help="comma list; also vitpose_plus_{base,large}_masked, sapiens_{0.3b,0.6b,2b}_masked")
    p.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    p.add_argument("--dilate-px", type=int, default=12, help="mask dilation for masking and the outside test")
    p.add_argument("--score-thr", type=float, default=0.3, help="keypoint confidence to draw/count")
    p.add_argument("--panel-scale", type=float, default=0.5, help="panel size in the output video")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--render-only", action="store_true",
                   help="no model runs: re-draw the videos from the <config>.csv of a previous run in --out-dir")
    a = p.parse_args()

    crop_dir = Path(a.crop_dir)
    video = Path(a.video) if a.video else sorted((crop_dir / "_source").glob("*.mp4"))[0]
    mask_files = {int(f.stem): f for f in crop_dir.glob("*.mp4") if f.stem.isdigit()}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mcaps = {pid: cv2.VideoCapture(str(f)) for pid, f in sorted(mask_files.items())}
    f0, f1 = int(a.start_s * fps), int(a.end_s * fps)

    backends = {}
    wanted = [c.strip() for c in a.configs.split(",") if c.strip()]
    saved = None
    if a.render_only:
        wanted = [c for c in wanted if (out / f"{c}.csv").exists()]
        saved = load_saved_poses(out, wanted)
        backends = {c: None for c in wanted}
        print(f"[bench] render-only from saved CSVs: {wanted}")
    for cfg in ([] if a.render_only else wanted):
        try:
            backends[cfg] = make_backend(cfg, a.device, fps)
            print(f"[bench] {cfg}: ready")
        except Exception as e:  # missing package / weights: skip, say how to get it
            print(f"[bench] {cfg}: SKIPPED ({type(e).__name__}: {e})"
                  + (f"\n        install: {INSTALL_HINT[cfg]}" if cfg in INSTALL_HINT else ""))
    if not backends:
        raise SystemExit("No backend available")

    names = list(backends)
    ncol = 2 if len(names) > 1 else 1
    nrow = int(np.ceil(len(names) / ncol))
    pw, ph = int(w * a.panel_scale) // 2 * 2, int(h * a.panel_scale) // 2 * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out / "benchmark.mp4"), fourcc, fps, (pw * ncol, ph * nrow))
    writer_clean = cv2.VideoWriter(str(out / "benchmark_clean.mp4"), fourcc, fps, (pw * ncol, ph * nrow))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate_px + 1, 2 * a.dilate_px + 1))

    rows = {c: [] for c in names}
    timing = {c: 0.0 for c in names}
    errors = {c: 0 for c in names}
    print(f"[bench] {video.name}: frames {f0}-{f1} ({(f1 - f0) / fps:.0f} s), {len(mcaps)} ids, configs {names}")

    for f in range(f1):
        ok, img = (cap.read() if f >= f0 else (cap.grab(), None))
        mframes = {pid: (mc.read() if f >= f0 else (mc.grab(), None)) for pid, mc in mcaps.items()}
        if not ok:
            break
        if f < f0:
            continue
        masks_raw = {pid: binarize(m) if okm else np.zeros((h, w), bool) for pid, (okm, m) in mframes.items()}
        dil = {pid: cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for pid, m in masks_raw.items()}
        boxes_plain = {pid: box_of(m, w, h) for pid, m in masks_raw.items()}
        boxes_blob = {pid: box_of(largest_blob(m), w, h) if m.any() else None for pid, m in masks_raw.items()}
        masks_u8 = {pid: m.astype(np.uint8) for pid, m in dil.items()}

        panels, panels_clean = [], []
        for cfg in names:
            boxes = boxes_blob if cfg.endswith("masked") else boxes_plain
            t0 = time.perf_counter()
            if saved is not None:
                poses = saved[cfg].get(f, {})
            else:
                try:
                    poses = backends[cfg](f, img, masks_u8, boxes)
                except Exception as e:  # one bad frame must not kill a long run
                    errors[cfg] += 1
                    if errors[cfg] <= 3:
                        print(f"[bench] {cfg} frame {f}: {type(e).__name__}: {e} (frame skipped)")
                    poses = {}
            timing[cfg] += time.perf_counter() - t0
            kinds = {}
            for pid, (xy, sc) in poses.items():
                kinds[pid] = classify_outside(xy, sc, pid, dil, masks_raw, w, h, a.score_thr)
                bh = (boxes[pid][3] - boxes[pid][1]) if boxes.get(pid) else np.nan
                for k in range(17):
                    rows[cfg].append((pid, f, COCO17[k], xy[k, 0], xy[k, 1], sc[k], kinds[pid][k], bh))
            panels.append(cv2.resize(draw_panel(img, masks_raw, boxes, poses, cfg, kinds, a.score_thr, False), (pw, ph)))
            panels_clean.append(cv2.resize(draw_panel(img, masks_raw, boxes, poses, cfg + "  (only points inside own mask)",
                                                      kinds, a.score_thr, True), (pw, ph)))
        for pl, wr in ((panels, writer), (panels_clean, writer_clean)):
            while len(pl) < ncol * nrow:
                pl.append(np.zeros((ph, pw, 3), np.uint8))
            grid = np.vstack([np.hstack(pl[r * ncol:(r + 1) * ncol]) for r in range(nrow)])
            cv2.putText(grid, f"frame {f}  t={f / fps:.1f}s", (10, grid.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            wr.write(grid)
        if (f - f0) % 150 == 0:
            print(f"[bench] t={f / fps:.0f}s")
    writer.release()
    writer_clean.release()

    # ---- metrics
    metrics = []
    for cfg in names:
        df = pd.DataFrame(rows[cfg], columns=["global_person_id", "frame", "keypoint_name", "x", "y", "score",
                                              "outside_kind", "box_h"])
        df["outside_kind"] = df["outside_kind"].fillna("")
        if not a.render_only:
            df.to_csv(out / f"{cfg}.csv", index=False)
        for pid, g in df.groupby("global_person_id"):
            conf = g[g["score"] >= a.score_thr]
            tor = conf[conf["keypoint_name"].isin(TORSO)]
            c = tor.groupby("frame").agg(x=("x", "mean"), y=("y", "mean"), bh=("box_h", "first")).sort_index()
            consecutive = np.diff(c.index.to_numpy()) == 1
            step = np.hypot(np.diff(c["x"]), np.diff(c["y"]))[consecutive] / c["bh"].to_numpy()[1:][consecutive]
            metrics.append({
                "config": cfg, "global_person_id": pid,
                "frames_with_pose": g["frame"].nunique(),
                "jitter_med": float(np.nanmedian(step)) if step.size else np.nan,
                "jitter_p95": float(np.nanpercentile(step, 95)) if step.size else np.nan,
                "on_other_person": float((conf["outside_kind"] == "other").mean()) if len(conf) else np.nan,
                "torso_on_other_person": float((tor["outside_kind"] == "other").mean()) if len(tor) else np.nan,
                "on_background": float((conf["outside_kind"] == "background").mean()) if len(conf) else np.nan,
                "sec_per_frame": timing[cfg] / max(1, f1 - f0),
                "frames_skipped_by_error": errors[cfg],
            })
    m = pd.DataFrame(metrics)
    m.to_csv(out / "metrics.csv", index=False)
    summary = m.groupby("config").agg(frames_with_pose=("frames_with_pose", "sum"), jitter_med=("jitter_med", "mean"),
                                      jitter_p95=("jitter_p95", "mean"), on_other_person=("on_other_person", "mean"),
                                      torso_on_other_person=("torso_on_other_person", "mean"),
                                      on_background=("on_background", "mean"), sec_per_frame=("sec_per_frame", "first"))
    print("\n[bench] per config (mean over people; jitter = torso move per frame / box height):")
    print(summary.round(4).to_string())
    if a.render_only:
        print("[bench] (render-only: sec_per_frame is drawing time, not model time)")
    print(f"\n[bench] per person: {out / 'metrics.csv'}")
    print(f"[bench] videos: {out / 'benchmark.mp4'} (all points: red x = on another person, grey o = on background)")
    print(f"                {out / 'benchmark_clean.mp4'} (points outside their own mask removed)")


if __name__ == "__main__":
    main()
