"""
seg_estimation.py
==================
Wrapper sottile su Ultralytics YOLO26 (instance segmentation) per estrarre
sagome multi-persona con tracking, da un file video o da una sorgente live.

Stato attuale (vedi README): sostituisce TEMPORANEAMENTE pose_estimation.py
come base della pipeline principale. Motivazione: su scene difficili
(visione dall'alto, movimento rapido, illuminazione artificiale) il modello
di pose produceva un numero di id spuri troppo alto (50+ in pochi minuti
anche con tracker/soglie gia' tarati) -- l'ipotesi da verificare con
track_stability_check.py / segmentation_demo.py e' che un modello che deve
solo delimitare la sagoma (non regredire 17 keypoint precisi) mantenga una
confidenza di detection piu' stabile su una persona parzialmente visibile o
in movimento rapido, e quindi tracci piu' in continuita'.

Piano (non ancora implementato): se il tracking di base risulta
sufficientemente stabile, ricollegare la pose applicandola SOLO dentro la
sagoma tracciata (ritaglio guidato dalla maschera, non dal semplice box) --
non sostituisce pose_estimation.py, si affianca usando l'id gia' stabile
della segmentation. Le feature comportamentali che dipendono dai keypoint
(features.py, gaze_head.py, hands.py, reid.py, chuv_features.py) restano
nel repository, testate, ma non sono attualmente collegate a questa
pipeline.

I modelli YOLO26-seg sono addestrati su COCO (80 classi): a differenza dei
modelli -pose (una sola classe, "persona"), qui e' necessario filtrare
esplicitamente la classe persona (id 0 in COCO) -- altrimenti ByteTrack
traccerebbe anche sedie, tavoli, borse ecc. presenti nella stanza.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.tracking_common import cap_by_confidence

COCO_PERSON_CLASS_ID = 0


@dataclass
class SegFrameResult:
    frame_index: int
    frame: np.ndarray
    people: list[tuple[int, np.ndarray, np.ndarray, float]] = field(default_factory=list)
    # ogni elemento: (track_id, bbox_xyxy (4,), mask_polygon (N,2), box_conf)
    # mask_polygon e' un array vuoto (0,2) se il modello non ha prodotto una
    # maschera valida per quella detection in quel frame.


class SegTracker:
    """Estrae sagome (box + maschera) multi-persona con tracking da una
    sorgente video usando Ultralytics YOLO26 instance segmentation.

    Stessa interfaccia di `pose_estimation.PoseTracker` (run() restituisce
    un generatore di FrameResult-like), cosi' il resto della pipeline che
    tratta gia' l'id come chiave generica non richiede altre modifiche.

    Parameters
    ----------
    model_name : modello YOLO26-seg (es. "yolo26s-seg.pt"; "yolo26n-seg.pt"
        piu' veloce, "yolo26m/l-seg.pt" piu' accurato -- stesso trade-off
        di pose_estimation.py).
    device : "mps" su Apple Silicon, "cpu" come fallback, "cuda" se
        disponibile una GPU NVIDIA.
    conf_threshold : soglia di confidenza minima, stessa raccomandazione di
        pose_estimation.py (tenere a o sotto track_low_thresh di ByteTrack,
        0.1 di default in bytetrack.yaml).
    tracker : config di tracking Ultralytics ("bytetrack.yaml" di default,
        oppure "configs/bytetrack_permissive.yaml" per scene difficili).
    max_people : come in pose_estimation.py, tiene solo le N detection piu'
        sicure per frame quando il numero di partecipanti alla sessione e'
        noto (vedi pose_estimation.py per il razionale completo).
    """

    def __init__(self, model_name: str = "yolo26s-seg.pt", device: str = "mps",
                 conf_threshold: float = 0.1, tracker: str = "bytetrack.yaml",
                 max_people: int | None = None):
        # Import ritardato: stessa ragione di pose_estimation.py (il resto
        # del pacchetto resta testabile senza ultralytics/torch installati).
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf_threshold
        self.tracker = tracker
        self.max_people = max_people

    def run(self, source, stream: bool = True):
        """Esegue segmentation + tracking sulla sorgente indicata.
        Restituisce un generatore di `SegFrameResult`.
        """
        results = self.model.track(
            source=source,
            device=self.device,
            conf=self.conf_threshold,
            tracker=self.tracker,
            classes=[COCO_PERSON_CLASS_ID],  # solo persone, non le altre 79 classi COCO
            stream=stream,
            verbose=False,
        )

        for i, r in enumerate(results):
            people = []
            if r.boxes is not None and r.boxes.id is not None:
                box_xyxy = r.boxes.xyxy.cpu().numpy()
                box_conf = r.boxes.conf.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                # r.masks puo' essere None se il modello caricato non e'
                # -seg, o se in quel frame non e' stata prodotta nessuna
                # maschera valida -- trattato come "poligono vuoto" per
                # ogni detection, mai un fallback inventato.
                polys = r.masks.xy if r.masks is not None else [None] * len(track_ids)

                for idx in cap_by_confidence(box_conf, self.max_people):
                    poly = polys[idx] if idx < len(polys) else None
                    poly_arr = np.asarray(poly) if poly is not None else np.empty((0, 2))
                    people.append((int(track_ids[idx]), box_xyxy[idx], poly_arr, float(box_conf[idx])))
            yield SegFrameResult(frame_index=i, frame=r.orig_img, people=people)


def mask_centroid(poly: np.ndarray) -> np.ndarray:
    """Centro approssimato della sagoma (media dei vertici del poligono
    maschera). Non e' il vero centroide geometrico dell'area per poligoni
    molto irregolari, ma per una sagoma umana la differenza e' trascurabile
    ed evita di dover rasterizzare una maschera piena solo per questo.
    NaN (2,) se il poligono e' vuoto."""
    if poly.shape[0] == 0:
        return np.full(2, np.nan)
    return poly.mean(axis=0)


def mask_area(poly: np.ndarray) -> float:
    """Area del poligono maschera in pixel^2 (formula shoelace, standard
    per l'area di un poligono semplice dati i suoi vertici in ordine).
    0.0 se il poligono e' vuoto o degenere (meno di 3 punti)."""
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
