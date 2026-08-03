"""
pipeline.py
===========
Orchestrazione end-to-end: video -> keypoint multi-persona (pose_estimation)
-> buffer per-persona -> feature comportamentali (features) -> tabella tidy
esportabile in CSV per analisi successive con pandas/scikit-learn.

Uso tipico (video pre-registrato, caso d'uso principale per l'analisi
clinica batch):

    python pipeline.py --source path/to/video.mp4 --fps 30 --out features.csv

Uso live (dimostrativo, es. con Canon R8 collegata via EOS Webcam Utility
come sorgente 0, o capture card HDMI):

    python pipeline.py --source 0 --fps 30 --out live_session.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from features import build_person_features, repetitive_motion_score, symmetry_index
from pose_estimation import PoseTracker


def run_pipeline(source, fps: float, model_name: str = "yolov8n-pose.pt",
                  device: str = "mps") -> pd.DataFrame:
    """Esegue l'intera pipeline e restituisce una tabella tidy con una riga
    per (frame, persona), pronta per l'analisi in pandas.
    """
    tracker = PoseTracker(model_name=model_name, device=device)

    # Buffer per accumulare la sequenza di keypoint di ciascun track_id
    buffers: dict[int, list[np.ndarray]] = defaultdict(list)

    for frame_result in tracker.run(source=source):
        for track_id, kpts_xy, kpts_conf in frame_result.people:
            buffers[track_id].append(kpts_xy)

    all_frames = []
    for track_id, seq in buffers.items():
        seq_arr = np.stack(seq)  # (n_frame, 17, 2)
        feature_table = build_person_features(seq_arr, track_id, fps)
        df = feature_table.to_dataframe()

        # Feature riassuntive a livello di persona (non per-frame): le
        # aggiungiamo come colonne costanti per semplicità di merge/analisi
        sym = symmetry_index(seq_arr, fps)
        for k, v in sym.items():
            df[k] = v

        rep = repetitive_motion_score(feature_table.speed[["left_wrist", "right_wrist"]].mean(axis=1).to_numpy(), fps)
        df["wrist_peak_freq_hz"] = rep["peak_freq_hz"]
        df["wrist_peak_power_ratio"] = rep["peak_power_ratio"]

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline pose estimation -> feature comportamentali")
    parser.add_argument("--source", required=True, help="Percorso video o indice webcam (es. 0)")
    parser.add_argument("--fps", type=float, required=True, help="Frame rate della sorgente")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Modello YOLO-pose Ultralytics")
    parser.add_argument("--device", default="mps", help="mps | cpu | cuda")
    parser.add_argument("--out", default="features.csv", help="Percorso CSV di output")
    args = parser.parse_args()

    # Permette di passare un intero (webcam) o una stringa (file/stream)
    source = int(args.source) if args.source.isdigit() else args.source

    df = run_pipeline(source, fps=args.fps, model_name=args.model, device=args.device)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
