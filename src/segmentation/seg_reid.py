"""
seg_reid.py
============
Re-identificazione per la pipeline di sola segmentazione
(seg_estimation.py / segmentation_demo.py), analoga a reid.py ma SENZA
keypoint: qui la "firma" e' costruita da posizione (centroide maschera),
colore campionato dentro la maschera e forma della sagoma (aspect ratio
del box, fill ratio maschera/box) -- non da proporzioni corporee, che non
esistono senza keypoint.

Differenza di design deliberata rispetto a reid.py
----------------------------------------------------
Quando il numero di partecipanti alla sessione e' noto E PICCOLO (1-2, il
caso 1v1), qui il tetto e' DAVVERO invalicabile -- non "nella maggior
parte dei casi" come il fallback opzionale `max_people` di reid.py (che
puo' rinunciare e coniare comunque un nuovo person_id se non trova nessun
candidato eleggibile), ma un vincolo assoluto: MAI piu' di `max_people`
identita' in tutta la sessione, senza eccezioni.

Questo e' difendibile proprio perche' qui il numero e' piccolo e certo:
con 1 sola persona attesa, ogni detection e' per definizione quella
persona, senza bisogno di alcun confronto. Con 2 (o piu', fino a una
decina), quando il tetto e' gia' raggiunto un nuovo raw track_id puo'
sempre essere legato al piu' vicino per posizione/colore/forma, anche
senza un segnale forte -- il costo di questa garanzia e' che in rarissimi
casi patologici (es. una detection doppia spuria nello stesso frame, piu'
raw_id "nuovi" di quanti slot liberi restano) due raw_id diversi possono
finire sullo stesso person_id per quel singolo frame, invece che uno dei
due restare "senza identita'" -- scelta esplicita, coerente con la
richiesta di non avere MAI la possibilita' di un id in piu'.

Due livelli di aggancio (importante se `max_people` e' impostato con
margine, es. 20 per un gruppo di ~10 bambini "per sicurezza"): un raw_id
nuovo tenta SEMPRE prima un aggancio "morbido" (con soglia
`soft_match_threshold`) a una persona nota ma non visibile in questo
frame, anche se ci sono ancora slot liberi -- senza questo passo, un
`max_people` largo faceva si' che il tetto non si raggiungesse mai
davvero, e ogni breve sparizione (occlusione, uscita dal bordo
inquadratura) apriva un id nuovo invece di ricollegarsi: uno scambio
temporaneo di id visibile anche con la re-id attiva. Solo se l'aggancio
morbido fallisce (nessun candidato sopra soglia) SI mette in campo la
logica precedente: slot libero -> id nuovo, tetto raggiunto -> aggancio
forzato senza soglia.

A differenza di reid.py, qui posizione/colore/forma non sono "sconti" su
una firma primaria (non esiste una firma antropometrica senza keypoint):
sono combinati con una media pesata (non "il piu' forte vince" come in
reid.py, che aveva senso li' perche' erano segnali secondari su un segnale
primario gia' significativo -- qui sono gli unici segnali disponibili).

Nessun buffer/attesa multi-frame: a differenza di reid.py (che ha bisogno
di ~15 frame per stabilizzare una firma antropometrica), qui posizione e
colore sono gia' utilizzabili dal primo frame in cui un raw track_id
compare -- la decisione viene presa subito, senza finestra scorrevole ne'
retry.

Limiti onesti
-------------
  - Nessuna soglia minima di somiglianza sul percorso "tetto raggiunto":
    la scelta e' sempre "il candidato migliore disponibile", mai "nessun
    candidato e' abbastanza buono" -- e' la garanzia richiesta (mai un id
    in piu'), ma significa che un aggancio puo' avvenire anche con un
    punteggio di somiglianza basso se non c'e' niente di meglio.
  - Nessuna scadenza per la memoria delle identita': a differenza di
    reid.py (`max_lost_seconds`), qui le posizioni/colori delle persone
    note restano validi per tutta la sessione -- corretto per il caso
    d'uso (sessione breve, poche persone fisse), ma renderebbe il
    confronto via via meno affidabile su una sessione molto lunga con
    grandi cambi di scenario.
  - Pesi e soglie di default non sono calibrati su dati reali.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from segmentation.seg_estimation import mask_area, mask_centroid


def _mask_mean_hs(frame: np.ndarray, poly: np.ndarray) -> tuple[float, float]:
    """Tonalita'/saturazione medie (OpenCV HSV, normalizzate 0-1) dei pixel
    dentro il poligono maschera -- usa la sagoma intera, non un'approssimazione
    a quadrilatero come reid.py (li' serviva perche' si partiva solo da
    keypoint; qui la maschera vera e' gia' disponibile). (nan, nan) se il
    poligono e' vuoto/degenere o copre un'area trascurabile."""
    if poly.shape[0] < 3:
        return np.nan, np.nan
    h, w = frame.shape[:2]
    pts = np.round(poly).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    if cv2.countNonZero(mask) < 25:
        return np.nan, np.nan
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_h, mean_s = cv2.mean(hsv, mask=mask)[:2]
    return float(mean_h) / 180.0, float(mean_s) / 255.0


def _shape_descriptor(bbox: np.ndarray, poly: np.ndarray) -> tuple[float, float]:
    """(aspect ratio w/h del box, fill ratio area-maschera/area-box) --
    descrittore di forma grezzo, invariante alla posizione/scala assoluta,
    segnale debole di postura (in piedi/seduto/accovacciato). (nan, nan)
    se il poligono e' vuoto."""
    x1, y1, x2, y2 = bbox
    box_w, box_h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    if poly.shape[0] < 3:
        return np.nan, np.nan
    aspect = box_w / box_h
    fill = mask_area(poly) / (box_w * box_h)
    return aspect, fill


def _color_similarity(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Similarita' 0..1 tra due coppie (hue, sat), hue circolare (come
    color_similarity in reid.py)."""
    dh = abs(a[0] - b[0])
    dh = min(dh, 1.0 - dh)
    ds = abs(a[1] - b[1])
    return float(np.clip(1.0 - (dh + ds) / 2.0, 0.0, 1.0))


def _shape_similarity(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Similarita' 0..1 tra due coppie (aspect, fill)."""
    d_aspect = abs(a[0] - b[0]) / max(a[0], b[0], 1e-6)
    d_fill = abs(a[1] - b[1])
    return float(np.clip(1.0 - (min(d_aspect, 1.0) + d_fill) / 2.0, 0.0, 1.0))


@dataclass
class _PersonState:
    position: np.ndarray                      # ultimo centroide maschera noto
    scale: float                               # sqrt(area) o diagonale box
    color: tuple[float, float] | None
    shape: tuple[float, float] | None
    last_seen: float


@dataclass
class _MergeEvent:
    raw_track_id: int
    matched_person_id: int
    frame_time: float
    score: float
    forced: bool = False  # True solo se legato senza soglia perche' il tetto era gia' raggiunto


class SegReIdentifier:
    """Mantiene la corrispondenza track_id (ByteTrack) -> person_id per la
    pipeline di segmentazione, con un tetto RIGIDO di `max_people`
    identita' -- vedi il docstring del modulo per il perche' e i limiti.

    Uso in segmentation_demo.py (un solo punto di wiring):

        seg_reidentifier = SegReIdentifier(max_people=2)
        ...
        for frame_result in tracker.run(...):
            people = seg_reidentifier.resolve(frame_result.people, now=...,
                                               frame=frame_result.frame)
            # 'people' ha la stessa forma di frame_result.people
            # [(id, bbox, poly, conf), ...], ma id e' ora un person_id
            # stabile, mai piu' di max_people valori distinti in tutta
            # la sessione.
    """

    def __init__(self, max_people: int, position_weight: float = 0.5,
                 color_weight: float = 0.3, shape_weight: float = 0.2,
                 max_position_dist_scales: float = 6.0,
                 max_position_gap_seconds: float = 30.0,
                 soft_match_threshold: float = 0.6):
        if max_people < 1:
            raise ValueError("max_people deve essere almeno 1")
        self.max_people = max_people
        self.position_weight = position_weight
        self.color_weight = color_weight
        self.shape_weight = shape_weight
        self.max_position_dist_scales = max_position_dist_scales
        self.max_position_gap_seconds = max_position_gap_seconds
        # soglia sotto la quale un aggancio NON forzato (tetto non ancora
        # raggiunto) viene rifiutato -- vedi resolve() per il perche' serve
        # anche quando c'e' ancora un slot libero.
        self.soft_match_threshold = soft_match_threshold

        self.raw_to_person: dict[int, int] = {}
        self.persons: dict[int, _PersonState] = {}  # TUTTE le identita' mai create (<= max_people)
        self.merge_log: list[_MergeEvent] = []
        self._next_person_id = 1

    # -- API pubblica -----------------------------------------------------

    def resolve(self, people: list[tuple[int, np.ndarray, np.ndarray, float]],
                now: float, frame: np.ndarray | None = None,
                ) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
        """Traduce la lista (raw_track_id, bbox, poly, conf) di un frame in
        una lista (person_id, bbox, poly, conf), garantendo che person_id
        non superi mai `max_people` valori distinti in tutta la sessione.
        `frame` opzionale: se assente, il colore non e' un segnale
        disponibile (si usano solo posizione e forma)."""
        current_raw_ids = {rid for rid, *_ in people}
        claimed_this_frame: set[int] = {
            self.raw_to_person[rid] for rid in current_raw_ids if rid in self.raw_to_person
        }

        out = []
        for raw_id, bbox, poly, conf in people:
            centroid = mask_centroid(poly)
            area = mask_area(poly)
            scale = float(np.sqrt(area)) if area > 1.0 else float(np.linalg.norm(bbox[2:] - bbox[:2]))
            color = _mask_mean_hs(frame, poly) if frame is not None else (np.nan, np.nan)
            shape = _shape_descriptor(bbox, poly)

            if raw_id not in self.raw_to_person:
                person_id = None

                # Aggancio "morbido" (con soglia), tentato SEMPRE per primo,
                # anche se il tetto max_people non e' ancora raggiunto: un
                # raw_id nuovo di ByteTrack e' spessissimo una persona gia'
                # vista a cui il tracker ha appena riassegnato un id diverso
                # (breve occlusione, uscita/rientro dal bordo inquadratura),
                # non una persona davvero mai vista prima. Senza questo
                # passo, con un max_people impostato largo apposta per
                # tenersi margine (es. 20 per un gruppo di ~10 bambini), il
                # tetto non si raggiunge mai per davvero e ogni sparizione
                # apriva sempre un id NUOVO invece di ricollegarsi -- lo
                # scambio temporaneo di id visibile anche a re-id attiva.
                if self.persons:
                    candidate_pid, candidate_score = self._best_match(
                        centroid, scale, color, shape, now, exclude=claimed_this_frame)
                    if candidate_pid is not None and candidate_score >= self.soft_match_threshold:
                        person_id = candidate_pid
                        self.merge_log.append(_MergeEvent(
                            raw_track_id=raw_id, matched_person_id=person_id,
                            frame_time=now, score=candidate_score, forced=False))

                if person_id is None and len(self.persons) < self.max_people:
                    # nessun aggancio morbido convincente E c'e' ancora uno
                    # slot libero: e' ragionevole trattarla come una persona
                    # davvero mai vista prima.
                    person_id = self._next_person_id
                    self._next_person_id += 1
                elif person_id is None:
                    # tetto raggiunto E nessun aggancio morbido trovato
                    # sopra: NON puo' comunque essere una persona in piu',
                    # si forza il migliore disponibile SENZA soglia (vedi
                    # docstring del modulo) tra gli slot non gia'
                    # rivendicati in questo stesso frame.
                    person_id, score = self._best_match(centroid, scale, color, shape, now,
                                                          exclude=claimed_this_frame)
                    if person_id is None:
                        # caso patologico: piu' raw_id "nuovi" in questo
                        # frame di quanti slot restino liberi (es.
                        # detection doppia spuria). Il vincolo "mai un id
                        # in piu'" ha priorita' assoluta: riusa comunque lo
                        # slot migliore, anche se gia' rivendicato.
                        person_id, score = self._best_match(centroid, scale, color, shape, now,
                                                              exclude=set())
                    self.merge_log.append(_MergeEvent(
                        raw_track_id=raw_id, matched_person_id=person_id,
                        frame_time=now, score=score, forced=True))
                self.raw_to_person[raw_id] = person_id
                claimed_this_frame.add(person_id)

            person_id = self.raw_to_person[raw_id]
            prev = self.persons.get(person_id)
            self.persons[person_id] = _PersonState(
                position=centroid, scale=scale,
                color=color if not np.isnan(color).any() else (prev.color if prev else None),
                shape=shape if not np.isnan(shape).any() else (prev.shape if prev else None),
                last_seen=now,
            )
            out.append((person_id, bbox, poly, conf))

        gone_raw_ids = [rid for rid in self.raw_to_person if rid not in current_raw_ids]
        for rid in gone_raw_ids:
            del self.raw_to_person[rid]

        return out

    # -- interno ------------------------------------------------------

    def _best_match(self, centroid: np.ndarray, scale: float,
                     color: tuple[float, float], shape: tuple[float, float],
                     now: float, exclude: set[int]) -> tuple[int | None, float]:
        best_pid, best_score = None, -1.0
        for pid, state in self.persons.items():
            if pid in exclude:
                continue

            pos_sim = 0.0
            if not np.isnan(centroid).any() and not np.isnan(state.position).any() and state.scale > 1e-6:
                dist_scales = float(np.linalg.norm(centroid - state.position)) / state.scale
                spatial = max(0.0, 1.0 - dist_scales / self.max_position_dist_scales)
                temporal = max(0.0, 1.0 - (now - state.last_seen) / self.max_position_gap_seconds)
                pos_sim = spatial * temporal

            color_sim = 0.0
            if state.color is not None and not np.isnan(color).any():
                color_sim = _color_similarity(color, state.color)

            shape_sim = 0.0
            if state.shape is not None and not np.isnan(shape).any():
                shape_sim = _shape_similarity(shape, state.shape)

            score = (self.position_weight * pos_sim + self.color_weight * color_sim
                     + self.shape_weight * shape_sim)
            if score > best_score:
                best_pid, best_score = pid, score

        return best_pid, best_score
