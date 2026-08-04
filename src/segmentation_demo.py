"""
segmentation_demo.py
=====================
Pipeline principale ATTUALE (temporanea, vedi seg_estimation.py e README):
tracking di sagome via YOLO26-seg + ByteTrack, overlay live con contorno
maschera + etichetta ID, CSV con una riga per (frame, persona). Nessun
keypoint/feature comportamentale per ora -- solo verifica visiva e
quantitativa della stabilita' del tracking, in attesa di ricollegare
pose/features/gaze/hands/reid (vedi seg_estimation.py per il contesto
completo e il piano).

Uso, su un video gia' registrato:

    python segmentation_demo.py --source video.mp4 --fps 15 \\
        --model yolo26s-seg.pt --tracker bytetrack_permissive.yaml \\
        --conf-threshold 0.1 --max-people 2 --out session_seg.csv

Con --no-window elabora senza aprire una finestra (piu' veloce, utile per
un batch senza bisogno di guardare l'overlay in diretta).
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from seg_estimation import SegTracker, mask_area, mask_centroid
from viz import draw_fps, draw_person_label, get_track_color


def run_segmentation(source, fps: float, model_name: str = "yolo26s-seg.pt",
                      device: str = "mps", conf_threshold: float = 0.1,
                      tracker_config: str = "bytetrack.yaml",
                      max_people: int | None = None,
                      out_csv: str = "segmentation_session.csv",
                      show_window: bool = True) -> pd.DataFrame:
    tracker = SegTracker(model_name=model_name, device=device,
                          conf_threshold=conf_threshold, tracker=tracker_config,
                          max_people=max_people)

    rows: list[dict] = []
    id_frame_count: dict[int, int] = defaultdict(int)
    win_name = "segmentation_demo"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    t_start = time.time()
    n_frames = 0
    for frame_result in tracker.run(source=source):
        n_frames = frame_result.frame_index + 1
        now = frame_result.frame_index / fps
        vis = frame_result.frame.copy() if show_window else None

        for track_id, bbox, poly, conf in frame_result.people:
            id_frame_count[track_id] += 1
            centroid = mask_centroid(poly)
            area = mask_area(poly)
            rows.append({
                "frame": frame_result.frame_index, "time_s": now, "track_id": track_id,
                "bbox_x1": float(bbox[0]), "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]), "bbox_y2": float(bbox[3]),
                "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
                "mask_area_px": area, "box_conf": conf,
            })

            if show_window:
                color = get_track_color(track_id)
                if poly.shape[0] >= 3:
                    pts = poly.astype(np.int32).reshape(-1, 1, 2)
                    overlay = vis.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                else:
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label_pos = centroid if not np.isnan(centroid).any() else bbox[:2]
                draw_person_label(vis, label_pos, track_id, color)

        if show_window:
            elapsed = time.time() - t_start
            draw_fps(vis, n_frames / elapsed if elapsed > 0 else 0.0)
            cv2.imshow(win_name, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show_window:
        cv2.destroyAllWindows()

    n_ids = len(id_frame_count)
    lifespans = sorted(id_frame_count.values())
    # 15 frame = min_signature_frames di default in reid.py: stesso
    # riferimento usato in track_stability_check.py, per confrontabilita'.
    short_lived = sum(1 for v in lifespans if v < 15)
    print(f"Frame processati: {n_frames}  |  Id distinti: {n_ids}")
    if lifespans:
        mediana = lifespans[len(lifespans) // 2]
        print(f"Durata id in frame: min={lifespans[0]}  mediana={mediana}  max={lifespans[-1]}")
        print(f"Id sotto 15 frame: {short_lived}/{n_ids} ({100 * short_lived / n_ids:.0f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Salvate {len(df)} righe in {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline di tracking basata su segmentation (YOLO26-seg + ByteTrack), "
                     "con overlay live e CSV. Nessun keypoint per ora -- vedi seg_estimation.py.")
    parser.add_argument("--source", required=True, help="Percorso video o indice webcam")
    parser.add_argument("--fps", type=float, required=True, help="Frame rate della sorgente")
    parser.add_argument("--model", default="yolo26s-seg.pt",
                         help="Modello YOLO26 di instance segmentation (yolo26n/s/m/l/x-seg.pt)")
    parser.add_argument("--device", default="mps", help="mps | cpu | cuda")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                         help="Tenere a o sotto track_low_thresh di ByteTrack (0.1 di default)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Config tracker Ultralytics, es. bytetrack_permissive.yaml")
    parser.add_argument("--max-people", type=int, default=None,
                         help="Numero noto di partecipanti alla sessione (2 per 1v1, fino a "
                              "una decina per un gruppo): tiene solo le N detection piu' "
                              "sicure per frame")
    parser.add_argument("--out", default="segmentation_session.csv", help="CSV di output")
    parser.add_argument("--no-window", action="store_true",
                         help="Esegui senza finestra video (solo log + CSV, piu' veloce)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_segmentation(source, fps=args.fps, model_name=args.model, device=args.device,
                      conf_threshold=args.conf_threshold, tracker_config=args.tracker,
                      max_people=args.max_people, out_csv=args.out,
                      show_window=not args.no_window)


if __name__ == "__main__":
    main()
