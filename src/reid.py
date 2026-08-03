"""
reid.py
=======
Re-identificazione in tempo reale basata su firma antropometrica, per
recuperare l'identita' di una persona quando ByteTrack le assegna un nuovo
track_id dopo che e' uscita completamente dall'inquadratura (o quando
l'aspetto cambia, es. vestiti diversi, tra un'uscita e un rientro nella
stessa sessione).

Stessa idea di `reid_signature.py` nel repository CHUV (Video-Annotation-
System, vedi quel modulo per il contesto completo), qui riadattata su due
assi:
  - schema COCO-17 (questa pipeline) invece di BODY-25 (psifx/OpenPose);
  - funzionamento IN TEMPO REALE invece che batch: li' si confrontano
    tracce gia' concluse leggendo un CSV completo; qui dobbiamo decidere
    "e' una persona mai vista o una gia' vista?" nell'istante in cui un
    nuovo track_id compare, contro una memoria di persone scomparse di
    recente dall'inquadratura.

Perche' prototiparla qui e non nel repository CHUV: quella pipeline
richiede SAM3 su GPU CUDA, non eseguibile su un Mac M1 -- e questo e' un
limite hardware, non della strategia. La firma antropometrica come segnale
di re-identificazione e' indipendente dal tracker sottostante (ByteTrack
qui, SAM3 la')); ha senso validarla dove si puo' iterare velocemente, su
dati non protetti, prima di riproporla per il repository CHUV.

Design (in breve)
------------------
- Ogni volta che ByteTrack presenta un track_id MAI visto prima, gli viene
  assegnato subito un `person_id` provvisorio (nessun ritardo percepibile
  nell'overlay live).
- In parallelo, si accumula un piccolo buffer di proporzioni corporee per
  quel track_id. Una volta raccolti abbastanza frame validi, si calcola
  una firma mediana e la si confronta con le persone recentemente
  scomparse dall'inquadratura (`lost`, con scadenza dopo `max_lost_seconds`).
- Se c'e' un match sotto soglia, TUTTI i frame SUCCESSIVI a quel punto
  vengono attribuiti al `person_id` precedente invece che al nuovo -- i
  frame gia' emessi (CSV/overlay) con il person_id provvisorio NON vengono
  riscritti. L'evento di merge viene loggato esplicitamente
  (`self.merge_log`) per trasparenza/audit, cosi' un'analisi a posteriori
  puo' ricollegare i due pezzi se necessario -- stessa filosofia
  "proponi/rendi tracciabile, non decidere in silenzio" di reid_signature.py,
  adattata al fatto che qui non c'e' un umano nel loop in tempo reale.

Segnale opzionale: colore maglia/pantaloni
-------------------------------------------
La sola firma antropometrica, in pratica, puo' essere troppo debole quando
i keypoint sono rumorosi (occlusioni parziali, persona ai bordi
dell'inquadratura durante l'uscita/rientro): due proporzioni corporee
leggermente sballate possono far mancare un match vero. Se viene passato il
frame video a `resolve()`, viene calcolato ANCHE un colore medio (tonalita'
+ saturazione, non luminosita' -- piu' robusto a cambi di esposizione) sulla
regione torso (maglia) e sulla regione coscia (pantaloni), campionato dai
pixel dentro il poligono definito dai keypoint di spalle/anche/ginocchia.

Il colore NON sostituisce le proporzioni ne' alza mai la soglia di
rifiuto: se il colore e' molto diverso (es. cambio vestiti tra uscita e
rientro) il match si decide esattamente come prima, solo sulle proporzioni.
Se invece il colore e' simile (il caso piu' comune: stessi vestiti durante
la sessione), la distanza tra le proporzioni viene "scontata" -- rendendo
piu' facile recuperare un match vero anche con proporzioni un po' rumorose,
senza mai peggiorare la robusta invarianza al vestiario che era l'obiettivo
originale del prototipo.

Limiti onesti
-------------
  - La firma ha bisogno di un numero minimo di frame con confidenza
    sufficiente sui giunti chiave; un passaggio molto breve nell'inquadratura
    non produrra' mai una firma affidabile e restera' un person_id a se'.
  - Due persone di corporatura simile possono generare un falso positivo di
    merge -- la soglia di default e' prudente ma va calibrata sui vostri
    dati reali, non e' un valore validato.
  - Il colore aiuta quando i vestiti restano gli stessi, ma per lo stesso
    motivo puo' aumentare il rischio di falso positivo se due persone
    diverse indossano vestiti di colore simile E hanno proporzioni corporee
    vicine -- e' un compromesso esplicito, non un problema nascosto.
  - Il merge, una volta deciso, e' applicato automaticamente (non c'e' modo
    di chiedere conferma a un umano in tempo reale) -- per questo l'evento
    resta sempre nel log, invece di sparire silenziosamente.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from keypoints import KP
from features import torso_length

# ---------------------------------------------------------------------------
# Firma antropometrica (adattata da reid_signature.py, schema COCO-17)
# ---------------------------------------------------------------------------

SIGNATURE_SEGMENTS: dict[str, tuple[str, str]] = {
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "hip_width": ("left_hip", "right_hip"),
    "upper_arm_l": ("left_shoulder", "left_elbow"),
    "upper_arm_r": ("right_shoulder", "right_elbow"),
    "forearm_l": ("left_elbow", "left_wrist"),
    "forearm_r": ("right_elbow", "right_wrist"),
    "thigh_l": ("left_hip", "left_knee"),
    "thigh_r": ("right_hip", "right_knee"),
    "shin_l": ("left_knee", "left_ankle"),
    "shin_r": ("right_knee", "right_ankle"),
}
SIGNATURE_COLS = list(SIGNATURE_SEGMENTS.keys())


def _segment_length(kxy: np.ndarray, a_name: str, b_name: str) -> float:
    a, b = kxy[KP[a_name]], kxy[KP[b_name]]
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    return float(np.linalg.norm(a - b))


def compute_signature_frame(kxy: np.ndarray) -> np.ndarray:
    """Proporzioni corporee normalizzate sul busto per un singolo frame
    (array nell'ordine di `SIGNATURE_COLS`, NaN dove non calcolabile).
    Riusa `features.torso_length` (gia' usata per self-touch/escursione
    verticale) come unita' di scala, cosi' la firma e' invariante alla
    distanza dalla camera.
    """
    torso = torso_length(kxy)
    out = np.full(len(SIGNATURE_COLS), np.nan)
    if torso < 1e-6 or np.isnan(torso):
        return out
    for i, (a_name, b_name) in enumerate(SIGNATURE_SEGMENTS.values()):
        seg = _segment_length(kxy, a_name, b_name)
        out[i] = seg / torso if not np.isnan(seg) else np.nan
    return out


def signature_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """Distanza RMS tra due firme, solo sulle dimensioni valide in
    entrambe. `None` se restano troppe poche dimensioni per fidarsi."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 3:
        return None
    return float(np.linalg.norm(a[valid] - b[valid]) / np.sqrt(valid.sum()))


# ---------------------------------------------------------------------------
# Colore maglia/pantaloni (segnale opzionale, complementare alla firma
# antropometrica -- vedi "Segnale opzionale" nel docstring del modulo)
# ---------------------------------------------------------------------------

COLOR_SEGMENTS: dict[str, tuple[str, str, str, str]] = {
    "shirt": ("left_shoulder", "right_shoulder", "right_hip", "left_hip"),
    "pants": ("left_hip", "right_hip", "right_knee", "left_knee"),
}
COLOR_COLS = ["shirt_h", "shirt_s", "pants_h", "pants_s"]
_HUE_IDX = [0, 2]   # indici circolari (0..1) in COLOR_COLS
_SAT_IDX = [1, 3]   # indici lineari (0..1) in COLOR_COLS


def _region_mean_hs(frame: np.ndarray, kxy: np.ndarray, corner_names: tuple[str, ...]) -> tuple[float, float]:
    """Tonalita'/saturazione medie (OpenCV HSV, poi normalizzate 0-1) dei
    pixel dentro il poligono definito dai keypoint indicati. (nan, nan) se
    uno dei keypoint manca (NaN) o il poligono e' degenere/fuori frame."""
    pts = kxy[[KP[name] for name in corner_names]]
    if np.isnan(pts).any():
        return np.nan, np.nan
    h, w = frame.shape[:2]
    poly = np.round(pts).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if cv2.countNonZero(mask) < 9:  # poligono troppo piccolo per fidarsi
        return np.nan, np.nan
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_h, mean_s = cv2.mean(hsv, mask=mask)[:2]
    return float(mean_h) / 180.0, float(mean_s) / 255.0


def compute_color_signature(frame: np.ndarray, kxy: np.ndarray) -> np.ndarray:
    """Colore medio (tonalita', saturazione) di maglia e pantaloni per un
    singolo frame, nell'ordine di `COLOR_COLS`, NaN dove non campionabile.
    Solo Hue/Saturation (non Value/luminosita'): piu' robusto a cambi di
    esposizione/illuminazione tra un'uscita e un rientro."""
    out = np.full(len(COLOR_COLS), np.nan)
    for i, corners in enumerate(COLOR_SEGMENTS.values()):
        h, s = _region_mean_hs(frame, kxy, corners)
        out[2 * i], out[2 * i + 1] = h, s
    return out


def color_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    """Similarita' 0..1 (1 = colore identico) tra due firme di colore,
    sulle dimensioni valide in entrambe. Distanza di tonalita' circolare
    (la scala Hue "avvolge" a 1.0). `None` se restano troppe poche
    dimensioni valide."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 2:
        return None
    dist = np.empty(len(a))
    diff = np.abs(a - b)
    dist[_HUE_IDX] = np.minimum(diff[_HUE_IDX], 1.0 - diff[_HUE_IDX])
    dist[_SAT_IDX] = diff[_SAT_IDX]
    return float(np.clip(1.0 - dist[valid].mean(), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Stato per persona/traccia
# ---------------------------------------------------------------------------

@dataclass
class _LostPerson:
    signature: np.ndarray
    lost_time: float
    color: np.ndarray | None = None


@dataclass
class _MergeEvent:
    raw_track_id: int
    provisional_person_id: int
    matched_person_id: int
    distance: float
    frame_time: float
    color_used: bool = False


class ReIdentifier:
    """Mantiene la corrispondenza track_id (ByteTrack) -> person_id
    (identita' stabile), ri-associando le persone che rientrano
    nell'inquadratura in base alla firma antropometrica.

    Uso in `live_demo.py` (un solo punto di wiring):

        reidentifier = ReIdentifier()
        ...
        for frame_result in tracker.run(...):
            people = reidentifier.resolve(frame_result.people, now=time.time(),
                                           frame=frame_result.frame)  # frame opzionale
            # 'people' ha la stessa forma di frame_result.people
            # [(id, kxy, kconf), ...], ma id e' ora un person_id stabile
    """

    def __init__(self, max_lost_seconds: float = 30.0,
                 max_signature_dist: float = 0.12,
                 min_signature_frames: int = 15,
                 color_bonus_weight: float = 0.5):
        self.max_lost_seconds = max_lost_seconds
        self.max_signature_dist = max_signature_dist
        self.min_signature_frames = min_signature_frames
        self.color_bonus_weight = color_bonus_weight

        self.raw_to_person: dict[int, int] = {}
        self.buffers: dict[int, deque] = {}
        self.color_buffers: dict[int, deque] = {}
        self.pending_check: set[int] = set()  # raw_track_id ancora da valutare
        self.lost: dict[int, _LostPerson] = {}
        self.merge_log: list[_MergeEvent] = []
        self._next_person_id = 1

    # -- API pubblica ---------------------------------------------------

    def resolve(self, people: list[tuple[int, np.ndarray, np.ndarray]],
                now: float, frame: np.ndarray | None = None) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Traduce la lista (raw_track_id, kxy, kconf) di un frame in una
        lista (person_id, kxy, kconf), ri-associando le identita' quando
        possibile. Va chiamata una volta per frame, con TUTTE le persone
        rilevate in quel frame (anche solo per tenere aggiornato lo stato
        interno di chi e' ancora presente).

        `frame` e' opzionale: se passato (il frame BGR corrente), viene
        usato anche il colore di maglia/pantaloni come segnale
        complementare alla firma antropometrica (vedi docstring del
        modulo). Se omesso, il comportamento e' identico alla versione
        basata solo sulle proporzioni corporee.
        """
        current_raw_ids = set()
        out = []

        for raw_id, kxy, kconf in people:
            current_raw_ids.add(raw_id)

            if raw_id not in self.raw_to_person:
                self._assign_provisional(raw_id)

            person_id = self.raw_to_person[raw_id]
            self.buffers[person_id].append(compute_signature_frame(kxy))
            if frame is not None:
                self.color_buffers[person_id].append(compute_color_signature(frame, kxy))

            if raw_id in self.pending_check and len(self.buffers[person_id]) >= self.min_signature_frames:
                self.pending_check.discard(raw_id)
                match = self._find_lost_match(self.buffers[person_id], self.color_buffers[person_id])
                if match is not None:
                    matched_person_id, dist, color_used = match
                    self.merge_log.append(_MergeEvent(
                        raw_track_id=raw_id, provisional_person_id=person_id,
                        matched_person_id=matched_person_id, distance=dist, frame_time=now,
                        color_used=color_used,
                    ))
                    print(f"[reid] track {raw_id}: provisional person_id {person_id} "
                          f"re-matched to person_id {matched_person_id} "
                          f"(distance={dist:.3f}{', color-assisted' if color_used else ''})")
                    del self.lost[matched_person_id]
                    del self.buffers[person_id]
                    self.color_buffers.pop(person_id, None)
                    self.raw_to_person[raw_id] = matched_person_id
                    person_id = matched_person_id
                    # buffer nuovo per l'identita' ripristinata: quello
                    # provvisorio e' stato appena eliminato, ma serve un
                    # buffer per continuare ad accumulare la firma nel
                    # caso questa persona sparisca di nuovo piu' avanti.
                    self.buffers[person_id] = deque(maxlen=self.min_signature_frames)
                    self.color_buffers[person_id] = deque(maxlen=self.min_signature_frames)

            out.append((person_id, kxy, kconf))

        self._retire_disappeared_tracks(current_raw_ids, now)
        self._expire_old_lost_people(now)
        return out

    # -- interno ----------------------------------------------------------

    def _assign_provisional(self, raw_id: int) -> None:
        person_id = self._next_person_id
        self._next_person_id += 1
        self.raw_to_person[raw_id] = person_id
        self.buffers[person_id] = deque(maxlen=self.min_signature_frames)
        self.color_buffers[person_id] = deque(maxlen=self.min_signature_frames)
        self.pending_check.add(raw_id)

    def _find_lost_match(self, buffer: deque, color_buffer: deque) -> tuple[int, float, bool] | None:
        median_sig = np.nanmedian(np.array(buffer), axis=0)
        median_color = np.nanmedian(np.array(color_buffer), axis=0) if color_buffer else None

        best_person_id, best_dist, best_color_used = None, self.max_signature_dist, False
        for person_id, lost in self.lost.items():
            prop_dist = signature_distance(median_sig, lost.signature)
            if prop_dist is None:
                continue

            color_used = False
            dist = prop_dist
            if median_color is not None and lost.color is not None:
                sim = color_similarity(median_color, lost.color)
                if sim is not None:
                    # il colore puo' solo AVVICINARE (mai allontanare) --
                    # resta clothing-invariant quando i vestiti cambiano
                    # davvero (sim bassa -> dist ~ invariata).
                    dist = prop_dist * (1.0 - self.color_bonus_weight * sim)
                    color_used = True

            if dist < best_dist:
                best_person_id, best_dist, best_color_used = person_id, dist, color_used
        return (best_person_id, best_dist, best_color_used) if best_person_id is not None else None

    def _retire_disappeared_tracks(self, current_raw_ids: set[int], now: float) -> None:
        gone_raw_ids = [rid for rid in self.raw_to_person if rid not in current_raw_ids]
        for raw_id in gone_raw_ids:
            person_id = self.raw_to_person.pop(raw_id)
            self.pending_check.discard(raw_id)
            buffer = self.buffers.pop(person_id, None)
            color_buffer = self.color_buffers.pop(person_id, None)
            if buffer and len(buffer) > 0:
                median_sig = np.nanmedian(np.array(buffer), axis=0)
                if not np.isnan(median_sig).all():
                    median_color = None
                    if color_buffer and len(color_buffer) > 0:
                        cand = np.nanmedian(np.array(color_buffer), axis=0)
                        if not np.isnan(cand).all():
                            median_color = cand
                    self.lost[person_id] = _LostPerson(
                        signature=median_sig, lost_time=now, color=median_color)

    def _expire_old_lost_people(self, now: float) -> None:
        expired = [pid for pid, lost in self.lost.items()
                   if now - lost.lost_time > self.max_lost_seconds]
        for pid in expired:
            del self.lost[pid]
