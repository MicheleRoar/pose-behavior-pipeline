"""
sam_backend.py
================
Base condivisa da `Sam31Tracker` (sam31_estimation.py) e `SamuraiTracker`
(samurai_estimation.py): entrambe le librerie espongono la STESSA API video
stateful (documentazione ufficiale: facebookresearch/sam3 e
yangchris11/samurai) --

    state = predictor.init_state(frames)
    predictor.add_new_points_or_box(state, frame_idx=..., obj_id=..., box=...)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        ...

-- quindi tutta la logica di chunking/prompting/riconciliazione/persistenza
vive UNA volta sola qui; le due sottoclassi implementano solo
`_build_predictor()` (quale libreria importare/istanziare, import ritardato
perche' ne' sam3 ne' samurai sono installabili in questo ambiente: servono
Python 3.12+/CUDA 12.6+, vedi requirements.txt).

Perche' il chunking (vedi anche segmentation/chunking.py)
------------------------------------------------------------
`init_state()` carica in memoria i pixel di TUTTI i frame passati -- su un
video di diversi minuti non e' praticabile passarlo intero. Si processa a
finestre sovrapposte (`chunk_size` frame, `overlap` in comune tra un chunk
e il successivo).

Strategia di prompting/continuita' degli id
------------------------------------------------
- Primo chunk: nessuna persona nota ancora. Si usa YOLO (lo stesso modello
  gia' usato da `SegTracker`, qui solo come RILEVATORE su un singolo frame,
  non come tracker) per proporre le box iniziali sul primo frame del
  chunk. ATTENZIONE: questo significa che la qualita' del prompt iniziale
  dipende comunque da YOLO -- SAM qui sostituisce il TRACKING/re-id nel
  tempo, non necessariamente la detection iniziale (si potrebbe passare a
  un prompt testuale "person" se il modello SAM 3.1 concept-prompting lo
  supporta a sufficienza; da verificare sulla macchina CUDA, vedi
  Sam31Tracker).
- Chunk successivi: per ogni persona gia' nota (id globale) si ricava un
  box prompt dalla sua maschera nell'ULTIMO frame del chunk precedente
  (che e' anche il PRIMO frame -- "frame di ancoraggio" -- del nuovo
  chunk, essendo nella finestra di overlap), e lo si registra con
  `obj_id=<id globale>`: SAM continua quindi a usare direttamente lo
  stesso id, non serve un abbinamento a posteriori nel caso comune. Si fa
  girare ANCHE YOLO sul frame di ancoraggio per individuare persone NUOVE
  (entrate nel campo durante il chunk precedente) non gia' coperte da un
  prompt esistente (IoU basso con tutte le box gia' seminate): a queste si
  assegna un id globale mai usato.
- Come controllo di coerenza (non come meccanismo di abbinamento primario,
  vedi sopra), sul frame di ancoraggio si confronta comunque la maschera
  che SAM produce con quella seminata (`chunking.polygon_iou`): se l'IoU
  cala sotto `iou_threshold` viene solo loggato un avviso -- puo' voler
  dire che SAM ha perso la persona o "scambiato" identita' nell'overlap,
  utile da vedere nei log ma non gestito automaticamente in questa prima
  versione (vedi chunking.py per il limite noto).

Onesta' su cosa e' verificato qui
--------------------------------------
Questo modulo e' stato scritto e testato SOLO con un predictor finto
iniettato al posto di sam3/samurai (vedi `demo/sam_backend_check.py`): la
logica di chunking/seeding/persistenza e' verificata, la vera inferenza SAM
NO (nessuna GPU CUDA in questo ambiente). Sulla macchina CUDA verificare
soprattutto: la firma esatta di `init_state()` (se richiede una lista di
frame in memoria o un percorso/cartella di frame su disco -- qui si
assume la prima forma, vedi `_init_state()` da adattare se serve) e il
formato esatto delle maschere restituite da `propagate_in_video()`.
"""

from __future__ import annotations

import os
from typing import Iterator

import cv2
import numpy as np

from segmentation.chunk_store import save_chunk
from segmentation.chunking import GlobalIdAllocator, iter_chunk_ranges, polygon_iou
from segmentation.seg_estimation import SegFrameResult

COCO_PERSON_CLASS_ID = 0
NEW_PERSON_IOU_THRESHOLD = 0.2  # sotto questa soglia una detection YOLO sul
# frame di ancoraggio e' considerata una persona NUOVA (non gia' seminata)


def _probe_video(source) -> tuple[int, tuple[int, int]]:
    """Numero di frame totali e (height, width) del video sorgente --
    usato per dimensionare le rasterizzazioni di `polygon_iou` e per
    calcolare i chunk. Stesso approccio "solo metadati, nessuna inferenza"
    di `webui/api.py::probe_video_metadata`, reimplementato qui per non
    creare una dipendenza di `segmentation/` verso `webui/`."""
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise ValueError(f"Impossibile aprire la sorgente video: {source!r}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return total_frames, (height, width)
    finally:
        cap.release()


def _read_frame_range(source, start: int, end: int) -> list[np.ndarray]:
    """Legge in memoria i frame `[start, end)` (BGR). Il posizionamento via
    `CAP_PROP_POS_FRAMES` non e' garantito perfettamente accurato su tutti i
    codec/contenitori (nota gia' presente altrove nel progetto, vedi
    `probe_video_metadata`), ma sufficiente per questo uso: eventuali
    scostamenti di uno-due frame non compromettono la riconciliazione, che
    lavora comunque su una finestra di overlap di decine di frame."""
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(end - start):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames
    finally:
        cap.release()


def _mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    """Maschera binaria (H,W) -> poligono (N,2) del contorno esterno piu'
    grande (una persona puo' produrre piu' componenti connesse per un
    'buco' nella maschera; si tiene solo la principale, stessa scelta
    pragmatica di `mask_area`/`mask_centroid` in seg_estimation.py che
    trattano una maschera come un singolo poligono). Poligono vuoto (0,2)
    se la maschera non contiene pixel positivi."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2))
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(float)


def _polygon_to_box(poly: np.ndarray) -> np.ndarray:
    """Bounding box (x1,y1,x2,y2) del poligono, usata come prompt per
    `add_new_points_or_box()`. `[0,0,0,0]` se il poligono e' vuoto."""
    if poly.shape[0] == 0:
        return np.zeros(4)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    return np.array([x1, y1, x2, y2], dtype=float)


class ChunkedVideoPredictorBackend:
    """Vedi il docstring del modulo. Le sottoclassi devono implementare
    `_build_predictor()`; possono ridefinire `_init_state()` /
    `_add_box_prompt()` / `_propagate()` se la firma reale della libreria
    diverge da quella assunta qui (vedi la nota "Onesta' su cosa e'
    verificato" sopra)."""

    def __init__(self, *, device: str = "cuda", chunk_size: int = 600,
                 overlap: int = 50, iou_threshold: float = 0.3,
                 prompt_model: str = "yolo26s-seg.pt",
                 conf_threshold: float = 0.1,
                 chunk_store_dir: str | None = None,
                 max_people: int | None = None):
        if device != "cuda":
            # SAM 3.1/SAMURAI non dichiarano supporto mps/cpu (vedi ricerca
            # citata nel README) -- meglio fallire subito con un messaggio
            # chiaro che lasciar provare e ottenere un errore oscuro dentro
            # la libreria.
            raise ValueError(
                f"{type(self).__name__} richiede device='cuda' (SAM 3.1/SAMURAI "
                f"non supportano mps/cpu al momento) -- ricevuto device={device!r}"
            )
        self.device = device
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold
        self.prompt_model = prompt_model
        self.conf_threshold = conf_threshold
        self.chunk_store_dir = chunk_store_dir
        self.max_people = max_people
        self._detector = None  # YOLO caricato pigramente in _detect_people()

    # -------------------------------------------------- da implementare
    def _build_predictor(self):
        raise NotImplementedError

    # ------------------------------------------------- override opzionale
    def _init_state(self, predictor, frames: list[np.ndarray]):
        return predictor.init_state(frames)

    def _add_box_prompt(self, predictor, state, *, frame_idx: int, obj_id: int, box: np.ndarray) -> None:
        predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id, box=box)

    def _propagate(self, predictor, state) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        """Deve restituire, per ogni frame locale al chunk, `(frame_idx,
        {obj_id: mask_binaria})`. Adattare qui se `propagate_in_video()`
        restituisce un formato diverso nella libreria reale (es. logits
        invece di maschere binarie -- in quel caso sogliare qui)."""
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            yield frame_idx, dict(zip(obj_ids, masks))

    # --------------------------------------------------------------- run
    def run(self, source, stream: bool = True) -> Iterator[SegFrameResult]:
        total_frames, frame_shape = _probe_video(source)
        predictor = self._build_predictor()
        allocator = GlobalIdAllocator()
        prev_anchor_polys: dict[int, np.ndarray] = {}

        for chunk_index, (start, end) in enumerate(iter_chunk_ranges(total_frames, self.chunk_size, self.overlap)):
            chunk_frames = _read_frame_range(source, start, end)
            if not chunk_frames:
                break
            state = self._init_state(predictor, chunk_frames)

            seed_boxes: dict[int, np.ndarray] = {}
            if chunk_index == 0:
                for box in self._detect_people(chunk_frames[0]):
                    seed_boxes[allocator.next_id()] = box
            else:
                for global_id, poly in prev_anchor_polys.items():
                    seed_boxes[global_id] = _polygon_to_box(poly)
                for box in self._detect_people(chunk_frames[0]):
                    if not _overlaps_any(box, seed_boxes.values(), frame_shape, NEW_PERSON_IOU_THRESHOLD):
                        seed_boxes[allocator.next_id()] = box  # persona nuova, mai vista prima

            if self.max_people is not None and len(seed_boxes) > self.max_people:
                # tetto rigido, stessa logica di cap_by_confidence altrove nel
                # progetto: qui non abbiamo una confidenza per ordinare, si
                # tiene semplicemente l'ordine di scoperta (persone gia' note
                # prima delle nuove, essendo inserite per prime nel dict).
                seed_boxes = dict(list(seed_boxes.items())[: self.max_people])

            for global_id, box in seed_boxes.items():
                self._add_box_prompt(predictor, state, frame_idx=0, obj_id=global_id, box=box)

            chunk_results: list[SegFrameResult] = []
            polys_by_local_frame: dict[int, dict[int, np.ndarray]] = {}
            for local_idx, masks_by_id in self._propagate(predictor, state):
                people = []
                polys_this_frame: dict[int, np.ndarray] = {}
                for obj_id, mask in masks_by_id.items():
                    poly = _mask_to_polygon(mask)
                    polys_this_frame[obj_id] = poly
                    box = _polygon_to_box(poly)
                    # SAM non produce una confidenza di detection comparabile
                    # a quella di YOLO: 1.0 come segnaposto esplicito, MAI
                    # usato per il tetto max_people qui (gia' applicato sopra
                    # sui seed) -- vedi cap_by_confidence in tracking_common.py
                    # per il caso YOLO, dove la confidenza e' invece reale.
                    people.append((obj_id, box, poly, 1.0))
                polys_by_local_frame[local_idx] = polys_this_frame
                chunk_results.append(SegFrameResult(
                    frame_index=start + local_idx, frame=chunk_frames[local_idx], people=people,
                ))

            # controllo di coerenza (solo log, vedi docstring del modulo)
            anchor_polys = polys_by_local_frame.get(0, {})
            for global_id, seeded_poly in prev_anchor_polys.items():
                produced_poly = anchor_polys.get(global_id)
                if produced_poly is None:
                    print(f"[{type(self).__name__}] avviso: id {global_id} non ritrovato "
                          f"al frame di ancoraggio del chunk {chunk_index}")
                    continue
                iou = polygon_iou(seeded_poly, produced_poly, frame_shape)
                if iou < self.iou_threshold:
                    print(f"[{type(self).__name__}] avviso: id {global_id} IoU basso "
                          f"({iou:.2f}) al frame di ancoraggio del chunk {chunk_index} "
                          f"-- possibile perdita/scambio di identita'")

            if self.chunk_store_dir:
                save_chunk(chunk_results, self.chunk_store_dir, chunk_index)

            # frame di ancoraggio per il PROSSIMO chunk: l'ultimo frame di
            # questo chunk che ricade nella finestra di overlap col successivo
            next_anchor_local = len(chunk_frames) - self.overlap
            prev_anchor_polys = polys_by_local_frame.get(next_anchor_local, {})

            # evita di riemettere due volte i frame della finestra di overlap
            # (gia' emessi dal chunk precedente, tranne per il primo chunk)
            skip = 0 if chunk_index == 0 else self.overlap
            for r in chunk_results[skip:]:
                yield r

    # --------------------------------------------------------------- YOLO
    def _detect_people(self, frame: np.ndarray) -> list[np.ndarray]:
        """Box (x1,y1,x2,y2) delle persone rilevate da YOLO su un singolo
        frame -- usato SOLO per proporre prompt iniziali a SAM (non per
        tracciare), vedi il docstring del modulo per il perche'. Import
        ritardato: stesso motivo di `SegTracker`."""
        if self._detector is None:
            from ultralytics import YOLO
            self._detector = YOLO(self.prompt_model)
        result = self._detector.predict(
            source=frame, device=self.device, conf=self.conf_threshold,
            classes=[COCO_PERSON_CLASS_ID], verbose=False,
        )[0]
        if result.boxes is None:
            return []
        return [box for box in result.boxes.xyxy.cpu().numpy()]


def _overlaps_any(box: np.ndarray, existing_boxes, frame_shape: tuple[int, int], threshold: float) -> bool:
    """True se `box` ha IoU >= `threshold` con almeno una delle
    `existing_boxes` -- box, non poligoni, quindi si costruisce un
    poligono rettangolare al volo per riusare `polygon_iou`."""
    box_poly = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
    for other in existing_boxes:
        other_poly = np.array([[other[0], other[1]], [other[2], other[1]],
                                [other[2], other[3]], [other[0], other[3]]])
        if polygon_iou(box_poly, other_poly, frame_shape) >= threshold:
            return True
    return False
