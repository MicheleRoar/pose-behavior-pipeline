"""
live_render_check.py
=====================
Verifies, WITHOUT a camera or Ultralytics/torch, that `live_demo.py`'s
logic is correct: simulates a frame-by-frame loop "as if" keypoints were
coming from a live source (here: the synthetic sequences from
`synth_data.py`), applies exactly the same sliding-window + skeleton/
overlay drawing logic used in the real live script, and writes an
inspectable mp4 video.

Also measures the CPU time spent per frame in the feature extraction +
drawing part (excluding model inference, which must be measured
separately on the Mac): useful to understand how much frame-budget margin
is left for YOLO inference given a target FPS.

Run with: python live_render_check.py
Output: demo_outputs/live_render_check.mp4, demo_outputs/live_frame_timing.csv
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import cv2

from pose.keypoints import KP
from pose.features import compute_joint_angles, repetitive_motion_score
from common.viz import draw_skeleton, draw_text_block, draw_fps
from synth_data import make_child_sequence, make_caregiver_sequence

OUT_DIR = Path(__file__).resolve().parent / "demo_outputs"
OUT_DIR.mkdir(exist_ok=True)

FPS = 30.0
DURATION_S = 8.0
N_FRAMES = int(FPS * DURATION_S)
WINDOW_SECONDS = 2.0
WINDOW_LEN = max(8, int(WINDOW_SECONDS * FPS))
CANVAS_W, CANVAS_H = 800, 500


def format_metrics(track_id: int, angles: dict, energy: float, rep_score: dict) -> list[str]:
    lines = [f"ID {track_id}"]
    lines.append(f"movement energy: {energy:6.1f}" if not np.isnan(energy) else "movement energy: --")
    if not np.isnan(rep_score.get("peak_power_ratio", np.nan)):
        lines.append(f"repetitiveness: {rep_score['peak_power_ratio']:.2f} @ {rep_score['peak_freq_hz']:.1f}Hz")
    else:
        lines.append("repetitiveness: --")
    for name in ("left_elbow_angle", "right_elbow_angle"):
        v = angles.get(name, np.nan)
        lines.append(f"{name}: {v:5.0f} deg" if not np.isnan(v) else f"{name}: --")
    return lines


def main():
    child_seq = make_child_sequence(N_FRAMES, FPS, seed=42)
    caregiver_seq = make_caregiver_sequence(N_FRAMES, FPS, seed=43)
    people = {1: child_seq, 2: caregiver_seq}

    writer = cv2.VideoWriter(
        str(OUT_DIR / "live_render_check.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CANVAS_W, CANVAS_H),
    )

    buffers = {tid: deque(maxlen=WINDOW_LEN) for tid in people}
    timing_rows = []
    smoothed_fps = FPS

    prev_t = time.perf_counter()
    for frame_idx in range(N_FRAMES):
        t0 = time.perf_counter()
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 30, dtype=np.uint8)  # dark gray background

        for track_id, seq in people.items():
            kxy = seq[frame_idx]
            buffers[track_id].append(kxy)

            angles = compute_joint_angles(kxy)
            energy, rep_score = np.nan, {"peak_freq_hz": np.nan, "peak_power_ratio": np.nan}

            if len(buffers[track_id]) >= WINDOW_LEN:
                window = np.stack(buffers[track_id])
                diffs = np.diff(window, axis=0) * FPS
                speed = np.linalg.norm(diffs, axis=2)
                energy = float((speed ** 2).sum(axis=1).mean())
                wrist_speed = speed[:, [KP["left_wrist"], KP["right_wrist"]]].mean(axis=1)
                rep_score = repetitive_motion_score(wrist_speed, FPS)

            draw_skeleton(canvas, kxy)
            origin = (max(int(kxy[:, 0].min()) - 40, 0), max(int(kxy[:, 1].min()) - 95, 0))
            draw_text_block(canvas, format_metrics(track_id, angles, energy, rep_score), origin=origin)

        now = time.perf_counter()
        dt = max(now - prev_t, 1e-6)
        smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / dt)
        prev_t = now

        draw_fps(canvas, smoothed_fps)
        writer.write(canvas)

        timing_rows.append({"frame": frame_idx, "processing_ms": (time.perf_counter() - t0) * 1000})

    writer.release()

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(OUT_DIR / "live_frame_timing.csv", index=False)

    avg_ms = timing_df["processing_ms"].mean()
    p95_ms = timing_df["processing_ms"].quantile(0.95)
    print(f"Frames processed: {len(timing_df)}")
    print(f"Average time per frame (feature+drawing, EXCLUDING YOLO inference): {avg_ms:.2f} ms  (p95: {p95_ms:.2f} ms)")
    print(f"Theoretical remaining budget for inference at {FPS:.0f} FPS: {1000/FPS - avg_ms:.2f} ms/frame")
    print(f"Verification video: {OUT_DIR / 'live_render_check.mp4'}")


if __name__ == "__main__":
    main()
