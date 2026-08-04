"""
synthetic_demo.py
==================
Dimostrazione end-to-end del modulo `features.py` su dati SINTETICI (nessun
video reale, nessun dato di minori coinvolto): genera due scheletri COCO-17
plausibili — un "bambino" con un braccio che compie un movimento ripetitivo
(proxy di una stereotipia motoria) e un "caregiver" con un movimento di
reaching più irregolare/naturale — e calcola le feature comportamentali.

Obiettivo: verificare che la pipeline di feature extraction sia corretta e
produca output sensati, senza dipendere da Ultralytics/torch (pesanti da
installare in questo ambiente sandbox) né da dati video reali.

Esegui con:  python synthetic_demo.py
Output: demo_outputs/features.csv, demo_outputs/*.png
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pose.features import (
    build_person_features, symmetry_index, repetitive_motion_score,
    proximity_series, windowed_synchrony,
)
from synth_data import make_child_sequence, make_caregiver_sequence

OUT_DIR = Path(__file__).resolve().parent / "demo_outputs"
OUT_DIR.mkdir(exist_ok=True)

FPS = 30.0
DURATION_S = 12.0
N_FRAMES = int(FPS * DURATION_S)
T = np.arange(N_FRAMES) / FPS


def main():
    print("Genero sequenze sintetiche di keypoint (bambino, caregiver)...")
    child_seq = make_child_sequence(N_FRAMES, FPS, seed=0)
    caregiver_seq = make_caregiver_sequence(N_FRAMES, FPS, seed=1)

    print("Costruisco le feature per persona...")
    child_features = build_person_features(child_seq, track_id=1, fps=FPS)
    caregiver_features = build_person_features(caregiver_seq, track_id=2, fps=FPS)

    child_df = child_features.to_dataframe()
    caregiver_df = caregiver_features.to_dataframe()
    child_df["role"] = "child"
    caregiver_df["role"] = "caregiver"

    combined = pd.concat([child_df, caregiver_df], ignore_index=True)

    print("Calcolo simmetria, movimento ripetitivo, prossimità e sincronia...")
    child_sym = symmetry_index(child_seq, FPS)
    caregiver_sym = symmetry_index(caregiver_seq, FPS)

    child_wrist_speed = child_features.speed[["left_wrist", "right_wrist"]].mean(axis=1).to_numpy()
    caregiver_wrist_speed = caregiver_features.speed[["left_wrist", "right_wrist"]].mean(axis=1).to_numpy()

    child_repetitive = repetitive_motion_score(child_wrist_speed, FPS)
    caregiver_repetitive = repetitive_motion_score(caregiver_wrist_speed, FPS)

    proximity = proximity_series(child_seq, caregiver_seq)
    synchrony = windowed_synchrony(child_features.energy, caregiver_features.energy, window=30, step=10)

    # --- salvataggio output ---
    combined.to_csv(OUT_DIR / "features.csv", index=False)
    synchrony.to_csv(OUT_DIR / "synchrony.csv", index=False)

    summary = pd.DataFrame([
        {"role": "child", **child_sym.to_dict(), **{f"wrist_{k}": v for k, v in child_repetitive.items()}},
        {"role": "caregiver", **caregiver_sym.to_dict(), **{f"wrist_{k}": v for k, v in caregiver_repetitive.items()}},
    ])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # --- grafici ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False)

    axes[0].plot(T, child_wrist_speed, label="bambino (polso, media L/R)")
    axes[0].plot(T, caregiver_wrist_speed, label="caregiver (polso, media L/R)")
    axes[0].set_title("Velocità del polso nel tempo")
    axes[0].set_xlabel("tempo (s)")
    axes[0].set_ylabel("velocità (px/s)")
    axes[0].legend()

    axes[1].plot(T, proximity, color="darkgreen")
    axes[1].set_title("Prossimità bambino-caregiver (distanza centro-bacino)")
    axes[1].set_xlabel("tempo (s)")
    axes[1].set_ylabel("distanza (px)")

    valid = synchrony.dropna()
    axes[2].plot((valid["frame_start"] + valid["frame_end"]) / 2 / FPS, valid["synchrony"], color="purple")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_title("Sincronia motoria (correlazione finestrata dell'energia di movimento)")
    axes[2].set_xlabel("tempo (s)")
    axes[2].set_ylabel("correlazione")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "demo_plots.png", dpi=140)
    plt.close(fig)

    print("\n=== Riepilogo ===")
    print(summary.to_string(index=False))
    print(f"\nOutput salvati in: {OUT_DIR}")


if __name__ == "__main__":
    main()
