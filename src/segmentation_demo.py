"""
segmentation_demo.py
=====================
Pipeline principale ATTUALE (temporanea, vedi seg_estimation.py e README):
tracking di sagome via YOLO26-seg + ByteTrack, overlay live con contorno
maschera + etichetta ID, CSV con una riga per (frame, persona). Nessuna
feature comportamentale a finestra scorrevole per ora (energia di
movimento, repetitivita', gaze, mani -- vedi pipeline pose, on hold).

Opzionale, con `--with-mediapipe-pose`: applica MediaPipe Pose Landmarker
in modalita' SINGOLA persona DENTRO il ritaglio di ciascuna sagoma gia'
tracciata (non un rilevatore multi-persona sull'intero frame -- vedi
`pose/mediapipe_pose.py` per il perche' di questa scelta), disegna lo
scheletro sopra la maschera e aggiunge gli angoli articolari istantanei
(non a finestra scorrevole) al CSV.

Uso, su un video gia' registrato:

    python segmentation_demo.py --source video.mp4 --fps 15 \\
        --model yolo26s-seg.pt --tracker configs/bytetrack_permissive.yaml \\
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

from segmentation.seg_estimation import SegTracker, mask_area, mask_centroid
from segmentation.seg_reid import SegReIdentifier
from common.viz import draw_fps, draw_person_label, draw_skeleton, get_track_color
from common.device import detect_default_device
from pose.mediapipe_pose import MediaPipePoseByTrack
from pose.features import compute_joint_angles

BACKEND_KEYS = {"yolo", "sam31", "samurai"}


def build_tracker(backend: str, *, model_name: str, device: str, conf_threshold: float,
                    tracker_config: str, max_people: int | None,
                    sam_chunk_size: int, sam_overlap: int, sam_chunk_store_dir: str | None,
                    sam_reseed_new_people: bool = True):
    """Istanzia il tracker giusto in base a `backend` -- unico punto in cui
    la scelta YOLO/SAM 3.1/SAMURAI si traduce in una classe concreta.
    Tutti e tre rispettano lo stesso protocollo `SegmentationBackend`
    (vedi segmentation/backend.py), quindi il resto di questa funzione
    (sotto) non ha bisogno di sapere quale sia stato scelto. Non piu'
    "privata" (senza underscore): riusata anche da `benchmark_backends.py`
    per costruire lo stesso tracker senza passare da `iter_segmentation_
    frames()` (che disegna un overlay qui inutile).

    `sam_reseed_new_people` (solo sam31/samurai, ignorato con "yolo"):
    False da' la condizione "SAM puro" per il confronto tra metodi (vedi
    benchmark_backends.py) -- YOLO propone i box SOLO al primo frame del
    video, mai per scoprire persone nuove ai confini dei chunk successivi.
    Default True (comportamento gia' in uso finora, invariato)."""
    if backend == "yolo":
        return SegTracker(model_name=model_name, device=device,
                           conf_threshold=conf_threshold, tracker=tracker_config,
                           max_people=max_people)
    if backend == "sam31":
        from segmentation.sam31_estimation import Sam31Tracker
        return Sam31Tracker(device=device, conf_threshold=conf_threshold,
                             chunk_size=sam_chunk_size, overlap=sam_overlap,
                             chunk_store_dir=sam_chunk_store_dir, max_people=max_people,
                             reseed_new_people=sam_reseed_new_people)
    if backend == "samurai":
        from segmentation.samurai_estimation import SamuraiTracker
        return SamuraiTracker(device=device, conf_threshold=conf_threshold,
                               chunk_size=sam_chunk_size, overlap=sam_overlap,
                               chunk_store_dir=sam_chunk_store_dir, max_people=max_people,
                               reseed_new_people=sam_reseed_new_people)
    raise ValueError(f"backend sconosciuto: {backend!r} (atteso 'yolo'|'sam31'|'samurai')")


def iter_segmentation_frames(source, fps: float, model_name: str = "yolo26s-seg.pt",
                              device: str = "mps", conf_threshold: float = 0.1,
                              tracker_config: str = "bytetrack.yaml",
                              max_people: int | None = None,
                              seg_reidentifier: SegReIdentifier | None = None,
                              mediapipe_pose_estimator: MediaPipePoseByTrack | None = None,
                              backend: str = "yolo",
                              sam_chunk_size: int = 600, sam_overlap: int = 50,
                              sam_chunk_store_dir: str | None = None,
                              sam_reseed_new_people: bool = True):
    """Generatore che contiene TUTTA la logica per-frame della pipeline di
    segmentazione (tracking, re-id opzionale, pose opzionale per maschera,
    disegno overlay), condiviso da `run_segmentation()` (CLI, sotto) e da
    `pipeline_runner.py` (GUI) -- stessa scelta di `iter_live_frames()` in
    live_demo.py, vedi il suo docstring per il perche'. `seg_reidentifier`
    e `mediapipe_pose_estimator` vanno costruiti dal chiamante (istanze
    persistenti per tutta la sessione, non ricreabili qui frame per frame);
    passare `None` per disattivarli. `mediapipe_pose_estimator` e' un
    `MediaPipePoseByTrack` (un'istanza MediaPipe indipendente PER PERSONA,
    non condivisa) -- vedi il suo docstring per il perche' un'unica istanza
    condivisa fra le persone del frame manderebbe MediaPipe in crash
    ("Input timestamp must be monotonically increasing").

    `backend` sceglie il motore di tracking/segmentazione: "yolo" (default,
    YOLO26-seg + ByteTrack, invariato), "sam31" o "samurai" (vedi
    segmentation/sam_backend.py -- richiedono device="cuda" e le rispettive
    librerie installate, non disponibili su mps/cpu). `sam_chunk_size` /
    `sam_overlap` / `sam_chunk_store_dir` sono usati solo con questi ultimi
    due (ignorati con "yolo").

    Disegna SEMPRE l'overlay (maschera, contorno, etichetta ID, +
    scheletro pose se `mediapipe_pose_estimator` e' attivo) sul frame
    restituito.

    Yield per ogni frame processato: `(vis, rows, now, frame_index, raw_ids)`
    dove `rows` e' la lista di dict (una riga per persona in questo frame,
    stesso schema del CSV finale -- con in piu' gli angoli articolari
    `pose_*` se `mediapipe_pose_estimator` e' attivo) e `raw_ids` sono i
    track_id grezzi di ByteTrack PRIMA dell'eventuale re-id (utile solo per
    le statistiche di churn stampate da `run_segmentation()`).
    """
    tracker = build_tracker(
        backend, model_name=model_name, device=device, conf_threshold=conf_threshold,
        tracker_config=tracker_config, max_people=max_people,
        sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
        sam_chunk_store_dir=sam_chunk_store_dir, sam_reseed_new_people=sam_reseed_new_people,
    )

    for frame_result in tracker.run(source=source):
        now = frame_result.frame_index / fps
        vis = frame_result.frame.copy()
        raw_ids = [p[0] for p in frame_result.people]

        people = frame_result.people
        if seg_reidentifier is not None:
            people = seg_reidentifier.resolve(people, now=now, frame=frame_result.frame)

        rows_this_frame = []
        for track_id, bbox, poly, conf in people:
            centroid = mask_centroid(poly)
            area = mask_area(poly)
            row = {
                "frame": frame_result.frame_index, "time_s": now, "track_id": track_id,
                "bbox_x1": float(bbox[0]), "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]), "bbox_y2": float(bbox[3]),
                "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
                "mask_area_px": area, "box_conf": conf,
            }

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

            # -- pose DENTRO la maschera tracciata (opzionale): identita'
            # "presa in prestito" da seg_reid/ByteTrack, vedi
            # pose/mediapipe_pose.py per il perche' di questo design.
            if mediapipe_pose_estimator is not None:
                kxy, kconf = mediapipe_pose_estimator.estimate(
                    track_id, frame_result.frame, bbox, timestamp_ms=int(now * 1000))
                draw_skeleton(vis, kxy, kconf, color=color)
                angles = compute_joint_angles(kxy)
                row.update({f"pose_{k}": v for k, v in angles.items()})

            rows_this_frame.append(row)

        yield vis, rows_this_frame, now, frame_result.frame_index, raw_ids


def run_segmentation(source, fps: float, model_name: str = "yolo26s-seg.pt",
                      device: str = "mps", conf_threshold: float = 0.1,
                      tracker_config: str = "bytetrack.yaml",
                      max_people: int | None = None,
                      with_seg_reid: bool = False,
                      with_mediapipe_pose: bool = False,
                      pose_landmarker_model: str = "pose_landmarker_lite.task",
                      out_csv: str = "segmentation_session.csv",
                      show_window: bool = True,
                      backend: str = "yolo",
                      sam_chunk_size: int = 600, sam_overlap: int = 50,
                      sam_chunk_store_dir: str | None = None,
                      sam_reseed_new_people: bool = True) -> pd.DataFrame:
    """CLI: consuma `iter_segmentation_frames()` (unica fonte della logica
    per-frame, condivisa con la GUI), gestisce la finestra cv2 (se
    show_window) e stampa le statistiche finali di churn/re-id."""
    # -- re-identificazione (opzionale, richiede --max-people): sostituisce
    # subito frame_result.people con la versione a person_id stabile, con
    # tetto rigido su max_people -- vedi seg_reid.py per il perche' e i
    # limiti (unico punto di wiring, come in live_demo.py per reid.py).
    if with_seg_reid and max_people is None:
        raise ValueError("--with-seg-reid richiede --max-people (il tetto rigido "
                          "ha senso solo con un numero di persone noto)")
    seg_reidentifier = SegReIdentifier(max_people=max_people) if with_seg_reid else None
    mediapipe_pose_estimator = (
        MediaPipePoseByTrack(model_path=pose_landmarker_model) if with_mediapipe_pose else None
    )

    rows: list[dict] = []
    raw_id_frame_count: dict[int, int] = defaultdict(int)   # id grezzi assegnati da ByteTrack
    final_id_frame_count: dict[int, int] = defaultdict(int)  # id finali (= raw se seg_reid disattivo)
    win_name = "segmentation_demo"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    t_start = time.time()
    n_frames = 0
    for vis, rows_this_frame, now, frame_index, raw_ids in iter_segmentation_frames(
        source=source, fps=fps, model_name=model_name, device=device,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
        max_people=max_people, seg_reidentifier=seg_reidentifier,
        mediapipe_pose_estimator=mediapipe_pose_estimator,
        backend=backend, sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
        sam_chunk_store_dir=sam_chunk_store_dir, sam_reseed_new_people=sam_reseed_new_people,
    ):
        n_frames = frame_index + 1
        for raw_id in raw_ids:
            raw_id_frame_count[raw_id] += 1
        for row in rows_this_frame:
            final_id_frame_count[row["track_id"]] += 1
        rows.extend(rows_this_frame)

        if show_window:
            elapsed = time.time() - t_start
            draw_fps(vis, n_frames / elapsed if elapsed > 0 else 0.0)
            cv2.imshow(win_name, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show_window:
        cv2.destroyAllWindows()

    n_raw_ids = len(raw_id_frame_count)
    lifespans = sorted(raw_id_frame_count.values())
    # 15 frame = min_signature_frames di default in reid.py: stesso
    # riferimento usato in track_stability_check.py, per confrontabilita'.
    short_lived = sum(1 for v in lifespans if v < 15)
    print(f"Frame processati: {n_frames}  |  Id grezzi assegnati da ByteTrack: {n_raw_ids}")
    if lifespans:
        mediana = lifespans[len(lifespans) // 2]
        print(f"Durata id grezzi in frame: min={lifespans[0]}  mediana={mediana}  max={lifespans[-1]}")
        print(f"Id grezzi sotto 15 frame: {short_lived}/{n_raw_ids} ({100 * short_lived / n_raw_ids:.0f}%)")
    if seg_reidentifier is not None:
        n_final_ids = len(final_id_frame_count)
        print(f"seg_reid: {len(seg_reidentifier.merge_log)} raw track_id ri-associati -> "
              f"{n_final_ids} id finali (tetto max_people={max_people} rispettato per costruzione)")

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
    parser.add_argument("--device", default=None,
                         help="mps | cpu | cuda (default: auto-rilevato -- cuda se una GPU "
                              "NVIDIA e' disponibile, altrimenti mps su Apple Silicon, "
                              "altrimenti cpu, vedi common/device.py)")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                         help="Tenere a o sotto track_low_thresh di ByteTrack (0.1 di default)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Config tracker Ultralytics, es. configs/bytetrack_permissive.yaml")
    parser.add_argument("--max-people", type=int, default=None,
                         help="Numero noto di partecipanti alla sessione (2 per 1v1, fino a "
                              "una decina per un gruppo): tiene solo le N detection piu' "
                              "sicure per frame; con --with-seg-reid diventa anche un tetto "
                              "rigido sul numero totale di identita' della sessione")
    parser.add_argument("--with-seg-reid", action="store_true",
                         help="Ri-associa i raw track_id di ByteTrack a un numero fisso di "
                              "person_id stabili (posizione/colore/forma della sagoma, vedi "
                              "seg_reid.py), garantendo che non vengano MAI creati piu' di "
                              "--max-people id in tutta la sessione. Richiede --max-people.")
    parser.add_argument("--with-mediapipe-pose", action="store_true",
                         help="Applica MediaPipe Pose Landmarker (modalita' singola persona) "
                              "dentro il ritaglio di ciascuna sagoma tracciata: disegna lo "
                              "scheletro e aggiunge gli angoli articolari (pose_*) al CSV. "
                              "Richiede mediapipe + il modello --pose-landmarker-model, vedi "
                              "pose/mediapipe_pose.py.")
    parser.add_argument("--pose-landmarker-model", default="pose_landmarker_lite.task",
                         help="Modello MediaPipe Pose Landmarker (usato solo con "
                              "--with-mediapipe-pose)")
    parser.add_argument("--out", default="segmentation_session.csv", help="CSV di output")
    parser.add_argument("--no-window", action="store_true",
                         help="Esegui senza finestra video (solo log + CSV, piu' veloce)")
    parser.add_argument("--backend", default="yolo", choices=sorted(BACKEND_KEYS),
                         help="Motore di segmentazione/tracking: 'yolo' (default, YOLO26-seg + "
                              "ByteTrack) | 'sam31' | 'samurai' (vedi segmentation/sam_backend.py "
                              "-- richiedono device=cuda e le rispettive librerie installate)")
    parser.add_argument("--sam-chunk-size", type=int, default=600,
                         help="Solo con --backend sam31/samurai: numero di frame per chunk")
    parser.add_argument("--sam-overlap", type=int, default=50,
                         help="Solo con --backend sam31/samurai: frame in comune tra un chunk "
                              "e il successivo, usati per la riconciliazione degli id")
    parser.add_argument("--sam-chunk-store-dir", default=None,
                         help="Solo con --backend sam31/samurai: cartella dove salvare "
                              "incrementalmente i risultati di ogni chunk (opzionale)")
    parser.add_argument("--sam-no-reseed-new-people", action="store_true",
                         help="Solo con --backend sam31/samurai: disattiva la scoperta di "
                              "persone NUOVE ai confini dei chunk (YOLO propone i box SOLO "
                              "al primo frame del video). Da' la condizione 'SAM puro' per "
                              "confrontare con la versione di default (con reseeding) -- "
                              "vedi benchmark_backends.py e segmentation/sam_backend.py.")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = args.device or detect_default_device()
    run_segmentation(source, fps=args.fps, model_name=args.model, device=device,
                      conf_threshold=args.conf_threshold, tracker_config=args.tracker,
                      max_people=args.max_people, with_seg_reid=args.with_seg_reid,
                      with_mediapipe_pose=args.with_mediapipe_pose,
                      pose_landmarker_model=args.pose_landmarker_model,
                      out_csv=args.out, show_window=not args.no_window,
                      backend=args.backend, sam_chunk_size=args.sam_chunk_size,
                      sam_overlap=args.sam_overlap, sam_chunk_store_dir=args.sam_chunk_store_dir,
                      sam_reseed_new_people=not args.sam_no_reseed_new_people)


if __name__ == "__main__":
    main()
