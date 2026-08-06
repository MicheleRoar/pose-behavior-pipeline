"""
benchmark_backends.py
========================
Confronta i backend di tracking disponibili (YOLO26-seg+ByteTrack
euristico, SAM 3.1, SAMURAI -- questi ultimi due con o senza reseeding di
persone nuove ai confini dei chunk, vedi segmentation/sam_backend.py) sullo
STESSO video, per capire quale mantiene l'identita' delle persone piu'
stabile nel tempo. Nato da un problema concreto: durante una sessione di
gioco terapeutico i bambini entrano ed escono continuamente dall'
inquadratura, o cambiano vestiti -- l'id di tracking dovrebbe restare lo
stesso.

Nessuna ground truth richiesta: le metriche qui sono di AUTO-consistenza,
non IDF1/HOTA veri (che richiederebbero label frame-per-frame dell'
identita' reale, non disponibili al momento):

- quante identita' "grezze" (raw track_id di ByteTrack o equivalente SAM)
  vengono create in tutta la sessione -- se il numero di persone reali e'
  noto (vedi --max-people) e il conteggio grezzo e' molto piu' alto, e'
  segno che il metodo perde e "reinventa" identita' quando qualcuno esce/
  rientra o cambia aspetto;
- quanto durano in media le tracce (min/mediana/max, in frame) e quante
  sono "brevi" (sotto SHORT_LIVED_THRESHOLD_FRAMES) -- tante tracce brevi
  indicano frammentazione: la stessa persona viene spezzata in piu' id nel
  tempo invece di restarne una sola;
- tempo di elaborazione / fps -- il costo pratico di ciascun metodo, non
  solo la sua qualita'.

Se in futuro emergono annotazioni (anche solo su singoli eventi tipo "il
bambino X esce al frame N e rientra al frame M"), si puo' estendere il
confronto aggiungendo una funzione dedicata a quegli eventi specifici,
senza toccare la struttura qui sotto.

Non disegna overlay ne' apre finestre (piu' veloce, e le metriche sopra
non richiedono guardare i frame): usa direttamente `tracker.run()` (il
protocollo comune `SegmentationBackend`, vedi segmentation/backend.py),
non `iter_segmentation_frames()` (che disegna l'overlay, qui inutile).

Se un metodo non e' eseguibile su questa macchina (device diverso da
"cuda" per sam31/samurai, o libreria non installata) viene SALTATO con un
avviso invece di far fallire l'intero confronto -- utile per rilanciare lo
stesso comando su macchine diverse (Mac senza CUDA: gira solo "yolo";
macchina CUDA con solo SAMURAI installato: le varianti "sam31*" vengono
saltate).

Uso:
    python benchmark_backends.py --source video.mp4 --fps 15 \\
        --methods yolo,sam31,sam31-noreseed,samurai,samurai-noreseed \\
        --max-people 3 --out benchmark_results.csv
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import pandas as pd

from common.device import detect_default_device
from segmentation_demo import build_tracker

METHOD_PRESETS = {
    "yolo": dict(backend="yolo", reseed=True),
    "sam31": dict(backend="sam31", reseed=True),
    "sam31-noreseed": dict(backend="sam31", reseed=False),
    "samurai": dict(backend="samurai", reseed=True),
    "samurai-noreseed": dict(backend="samurai", reseed=False),
}

# Stesso riferimento usato altrove nel progetto (reid.py::min_signature_frames,
# run_segmentation()/track_stability_check.py) -- confrontabile con quelle
# statistiche gia' familiari, non un numero inventato qui apposta.
SHORT_LIVED_THRESHOLD_FRAMES = 15


def run_one_method(method: str, *, source, fps: float, device: str,
                    model_scale: str = "s", conf_threshold: float = 0.1,
                    tracker_config: str = "bytetrack.yaml",
                    max_people: int | None = None,
                    sam_chunk_size: int = 600, sam_overlap: int = 50) -> dict | None:
    """Esegue UN metodo sul video e ritorna un dict di metriche, oppure
    `None` se il metodo va saltato (device incompatibile o libreria
    mancante -- vedi il docstring del modulo). Non solleva mai per uno di
    questi due motivi attesi, solo per un vero bug (es. parametro
    sconosciuto altrove)."""
    preset = METHOD_PRESETS[method]
    backend = preset["backend"]
    reseed = preset["reseed"]

    if backend in ("sam31", "samurai") and device != "cuda":
        print(f"[{method}] saltato: richiede device='cuda' (rilevato {device!r})")
        return None

    try:
        tracker = build_tracker(
            backend, model_name=f"yolo26{model_scale}-seg.pt", device=device,
            conf_threshold=conf_threshold, tracker_config=tracker_config,
            max_people=max_people, sam_chunk_size=sam_chunk_size,
            sam_overlap=sam_overlap, sam_chunk_store_dir=None,
            sam_reseed_new_people=reseed,
        )
    except ImportError as exc:
        print(f"[{method}] saltato: {exc}")
        return None

    raw_id_frame_count: dict[int, int] = defaultdict(int)
    n_frames = 0
    t_start = time.time()
    for frame_result in tracker.run(source=source):
        n_frames = frame_result.frame_index + 1
        for track_id, _bbox, _poly, _conf in frame_result.people:
            raw_id_frame_count[track_id] += 1
    elapsed_s = time.time() - t_start

    n_ids = len(raw_id_frame_count)
    lifespans = sorted(raw_id_frame_count.values())
    short_lived = sum(1 for v in lifespans if v < SHORT_LIVED_THRESHOLD_FRAMES)
    median_frames = lifespans[len(lifespans) // 2] if lifespans else 0

    return {
        "method": method,
        "backend": backend,
        "reseed_new_people": reseed,
        "n_frames": n_frames,
        "n_raw_ids": n_ids,
        "lifespan_min_frames": lifespans[0] if lifespans else 0,
        "lifespan_median_frames": median_frames,
        "lifespan_median_s": round(median_frames / fps, 2) if fps > 0 else 0.0,
        "lifespan_max_frames": lifespans[-1] if lifespans else 0,
        "short_lived_ids_pct": round(100 * short_lived / n_ids, 1) if n_ids else 0.0,
        # elapsed_s/processing_fps: TEMPO DI CALCOLO (wall-clock per elaborare
        # il video), da non confondere con lifespan_median_s (durata media di
        # una traccia nella TIMELINE del video, basata su --fps sorgente).
        "elapsed_s": round(elapsed_s, 1),
        "processing_fps": round(n_frames / elapsed_s, 2) if elapsed_s > 0 else 0.0,
    }


def run_benchmark(methods: list[str], *, source, fps: float, device: str | None = None,
                   **kwargs) -> pd.DataFrame:
    """Esegue tutti i `methods` (nell'ordine dato) sullo stesso `source` e
    ritorna un DataFrame con una riga per metodo NON saltato. Colonna
    mancante di un metodo saltato: semplicemente assente dal risultato,
    non una riga con valori nulli -- il chiamante vede subito quanti/quali
    metodi hanno davvero girato."""
    device = device or detect_default_device()
    rows = []
    for method in methods:
        if method not in METHOD_PRESETS:
            raise ValueError(f"metodo sconosciuto: {method!r} (atteso uno tra {sorted(METHOD_PRESETS)})")
        print(f"--- {method} ---")
        result = run_one_method(method, source=source, fps=fps, device=device, **kwargs)
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Confronta i backend di tracking (YOLO/SAM 3.1/SAMURAI, con/senza "
                     "reseeding di persone nuove) sullo stesso video: quante identita' "
                     "'grezze' crea ciascuno, quanto durano le tracce, quanto e' veloce. "
                     "Nessuna ground truth richiesta -- vedi il docstring del modulo.")
    parser.add_argument("--source", required=True, help="Percorso video")
    parser.add_argument("--fps", type=float, required=True,
                         help="Frame rate della sorgente -- usato per convertire la durata "
                              "mediana delle tracce da frame a secondi (lifespan_median_s)")
    parser.add_argument("--methods", default=",".join(METHOD_PRESETS),
                         help=f"Elenco separato da virgole tra {sorted(METHOD_PRESETS)} "
                              f"(default: tutti)")
    parser.add_argument("--device", default=None,
                         help="Sovrascrive l'auto-rilevamento (cuda/mps/cpu)")
    parser.add_argument("--scale", default="s", choices=["n", "s", "m"],
                         help="Taglia del modello YOLO (usato come tracker per 'yolo', "
                              "e come proposer di prompt per sam31/samurai)")
    parser.add_argument("--conf-threshold", type=float, default=0.1)
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Config ByteTrack (solo per il metodo 'yolo')")
    parser.add_argument("--max-people", type=int, default=None,
                         help="Numero noto di partecipanti alla sessione, se lo conosci -- "
                              "usato come tetto per YOLO e per interpretare n_raw_ids "
                              "(molto maggiore del numero reale = identita' perse/reinventate)")
    parser.add_argument("--sam-chunk-size", type=int, default=600)
    parser.add_argument("--sam-overlap", type=int, default=50)
    parser.add_argument("--out", default="benchmark_results.csv")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    df = run_benchmark(
        methods, source=args.source, fps=args.fps, device=args.device,
        model_scale=args.scale, conf_threshold=args.conf_threshold,
        tracker_config=args.tracker, max_people=args.max_people,
        sam_chunk_size=args.sam_chunk_size, sam_overlap=args.sam_overlap,
    )
    if df.empty:
        print("Nessun metodo eseguito (tutti saltati) -- niente da salvare.")
        return
    df.to_csv(args.out, index=False)
    print(f"\nSalvato {args.out}:\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
