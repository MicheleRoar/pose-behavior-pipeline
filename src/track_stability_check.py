"""
track_stability_check.py
=========================
Confronto diagnostico: quanti track_id distinti produce ByteTrack usando un
modello di SEGMENTATION (yolo26-seg) invece del modello di POSE
(yolo26-pose) attualmente usato in pose_estimation.py, sulla stessa
sorgente e con la stessa configurazione di tracker/soglie.

Motivazione: sul video reale (visione dall'alto, movimento rapido,
illuminazione artificiale) ByteTrack + YOLO-pose produce moltissimi id
diversi (50+ in pochi minuti anche con conf-threshold/tracker/max-people
gia' tarati). Ipotesi da verificare: un modello di segmentation, che deve
solo delimitare la sagoma (non stimare 17 keypoint precisi), potrebbe
mantenere una confidenza di detection piu' stabile su una persona
parzialmente visibile o in movimento rapido, e quindi un tracking piu'
continuo -- indipendentemente dal fatto che poi si riesca o no a estrarne
keypoint utilizzabili.

Questo script NON estrae keypoint (i modelli -seg di Ultralytics non ne
hanno, solo maschera + box): serve solo a contare id distinti e la loro
durata, per decidere se vale la pena investire nel percorso "segmentation +
pose sulla sagoma" (di cui si e' discusso ma che NON e' ancora
implementato -- vedi il modulo pose_estimation.py per l'approccio attuale
a modello singolo YOLO-pose).

Confronto suggerito: lanciare questo script E live_demo.py/pipeline.py
sullo STESSO video con la STESSA configurazione di --tracker/--conf-
threshold/--max-people, e confrontare il numero di id distinti riportato
qui con quello osservato nel CSV/log della pipeline normale.

Uso:
    python track_stability_check.py --source video.mp4 --fps 15 \
        --model yolo26s-seg.pt --tracker bytetrack_permissive.yaml \
        --conf-threshold 0.1 --max-people 2
"""

from __future__ import annotations

import argparse
from collections import defaultdict


def run(source, fps: float, model_name: str, device: str, conf_threshold: float,
        tracker_config: str, max_people: int | None) -> None:
    # Import ritardato, stessa ragione di pose_estimation.py: il resto del
    # pacchetto resta testabile senza ultralytics/torch installati.
    from ultralytics import YOLO

    from tracking_common import cap_by_confidence

    model = YOLO(model_name)
    id_frame_count: dict[int, int] = defaultdict(int)
    id_first_frame: dict[int, int] = {}
    id_last_frame: dict[int, int] = {}

    results = model.track(
        source=source,
        device=device,
        conf=conf_threshold,
        tracker=tracker_config,
        stream=True,
        verbose=False,
    )

    n_frames = 0
    for i, r in enumerate(results):
        n_frames = i + 1
        if r.boxes is None or r.boxes.id is None:
            continue
        box_conf = r.boxes.conf.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy().astype(int)

        for idx in cap_by_confidence(box_conf, max_people):
            tid = int(track_ids[idx])
            id_frame_count[tid] += 1
            id_first_frame.setdefault(tid, i)
            id_last_frame[tid] = i

    n_ids = len(id_frame_count)
    lifespans = sorted(id_frame_count.values())
    # 15 frame = soglia min_signature_frames di default in reid.py: un id
    # piu' corto di cosi' non arriverebbe mai ad avere una firma, quindi
    # non e' recuperabile da reid.py in nessun caso (ne' normale ne' forzato).
    short_lived = sum(1 for v in lifespans if v < 15)

    print(f"Modello: {model_name}  |  tracker: {tracker_config}  |  "
          f"conf-threshold: {conf_threshold}"
          + (f"  |  max-people: {max_people}" if max_people is not None else ""))
    print(f"Frame totali processati: {n_frames}  (~{n_frames / fps:.1f}s a {fps} fps)")
    print(f"Id distinti: {n_ids}")
    if lifespans:
        mediana = lifespans[len(lifespans) // 2]
        print(f"Durata id in frame: min={lifespans[0]}  mediana={mediana}  max={lifespans[-1]}")
        print(f"Id sotto 15 frame (troppo brevi per reid.py, min_signature_frames di default): "
              f"{short_lived}/{n_ids} ({100 * short_lived / n_ids:.0f}%)")
    else:
        print("Nessun id rilevato.")


def main():
    parser = argparse.ArgumentParser(
        description="Conta gli id distinti prodotti da ByteTrack su un modello di "
                     "segmentation (yolo26-seg), per confrontare la stabilita' del "
                     "tracking rispetto al modello di pose gia' in uso in pose_estimation.py. "
                     "Non estrae keypoint: solo diagnostica di tracking.")
    parser.add_argument("--source", required=True, help="Percorso video o indice webcam")
    parser.add_argument("--fps", type=float, default=15.0,
                         help="Frame rate della sorgente (solo per il riepilogo in secondi)")
    parser.add_argument("--model", default="yolo26s-seg.pt",
                         help="Modello YOLO26 di instance segmentation "
                              "(yolo26n/s/m/l/x-seg.pt)")
    parser.add_argument("--device", default="mps", help="mps | cpu | cuda")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                         help="Stessa raccomandazione di pose_estimation.py: tenere a o "
                              "sotto track_low_thresh (0.1 di default in bytetrack.yaml)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Config tracker Ultralytics, es. bytetrack_permissive.yaml")
    parser.add_argument("--max-people", type=int, default=None,
                         help="Come in pose_estimation.py: tiene solo le N detection piu' "
                              "sicure per frame, se il numero di partecipanti e' noto")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source, fps=args.fps, model_name=args.model, device=args.device,
        conf_threshold=args.conf_threshold, tracker_config=args.tracker,
        max_people=args.max_people)


if __name__ == "__main__":
    main()
