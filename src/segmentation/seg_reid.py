"""
seg_reid.py
============
Re-identificazione per la pipeline di sola segmentazione
(seg_estimation.py / segmentation_demo.py), analoga a reid.py ma SENZA
keypoint: qui la "firma" e' costruita da posizione (centroide maschera),
colore campionato dentro la maschera e forma della sagoma (aspect ratio
del box, fill ratio maschera/box) -- non da proporzioni corporee, che non
esistono senza keypoint.

Il colore e' un ISTOGRAMMA di tonalita' (vedi `_mask_hue_histogram`), non
una tonalita' media: un capo a righe/bicolore si media in un colore
generico indistinguibile da tanti altri (verificato su una sessione reale:
un bambino con maglia a righe non veniva ricollegato al rientro), mentre
l'istogramma cattura la distribuzione (due picchi per un bicolore) --
confrontato per intersezione, non per distanza di un singolo valore.

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

Segnale opzionale: embedding di aspetto (OSNet)
--------------------------------------------------
Quarto termine della media pesata, stesso schema di posizione/colore/forma
qui sopra -- vedi `pose/appearance_embedding.py` per il modulo e il perche'
di un embedding vero (OSNet) invece di un'altra euristica geometrica/
cromatica. E' tipicamente il piu' affidabile dei quattro (peso di default
piu' alto, vedi `embedding_weight`): a differenza di posizione/forma/colore,
non dipende da dove si trova la persona nel frame ne' da un singolo canale
di colore, solo dal suo aspetto complessivo. Come in reid.py, l'embedding
di una persona nota viene aggiornato ad ogni frame in cui e' visibile con
una media mobile esponenziale (`_update_person_state`, l'idea di StrongSORT
citata in `appearance_embedding.ema_update`) invece che sostituito col solo
frame corrente -- la stima si consolida piu' a lungo la persona resta
visibile, esattamente il comportamento richiesto ("restare in memoria per
associarli facilmente al rientro"). Richiede un `embedder` (istanza di
`OSNetEmbedder`) passato al costruttore; se `None` (default), il
comportamento e' identico a prima di questa aggiunta.

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
from pose.identity_manager import (
    SessionMode, IdentityManagerConfig, resolve_batch, suggested_max_people_policy,
)
from pose.appearance_embedding import OSNetEmbedder, embedding_similarity, ema_update


_HUE_HIST_BINS = 16  # 180deg/16 = 11.25deg/bin -- abbastanza grezzo da reggere il rumore
                      # di illuminazione, abbastanza fine da separare due colori distinti


def _mask_hue_histogram(frame: np.ndarray, poly: np.ndarray,
                         bins: int = _HUE_HIST_BINS) -> np.ndarray | None:
    """Istogramma di tonalita' (OpenCV Hue, 0-179) dei pixel dentro il
    poligono maschera, pesato per saturazione (i pixel quasi desaturati --
    ombre, tessuti neutri -- contano meno: la loro tonalita' e' rumore, non
    segnale) e normalizzato a somma 1.

    Sostituisce la tonalita' MEDIA usata in precedenza: un capo bicolore o a
    righe (es. maglia a righe olivastre/crema) si media in un colore
    intermedio generico, indistinguibile da tanti altri indumenti -- una
    sessione di test reale ha mostrato proprio questo, un bambino con
    maglia a righe che non veniva ricollegato al rientro nell'inquadratura.
    Un istogramma cattura invece la distribuzione (due picchi distinti per
    un bicolore), molto piu' vicino a come un umano riconosce un pattern a
    colpo d'occhio.

    `None` se il poligono e' vuoto/degenere, copre un'area trascurabile, o
    (dopo aver scartato i pixel troppo desaturati) non restano abbastanza
    pixel per fidarsi -- stesso trattamento "nessun segnale disponibile"
    del resto del modulo, mai un istogramma inventato."""
    if poly.shape[0] < 3:
        return None
    h, w = frame.shape[:2]
    pts = np.round(poly).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    if cv2.countNonZero(mask) < 25:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    hue = hsv[ys, xs, 0].astype(np.float64)          # 0..179
    sat = hsv[ys, xs, 1].astype(np.float64) / 255.0  # 0..1, usata come peso

    valid = sat > 0.15  # pixel quasi grigi: la loro tonalita' e' rumore di sensore
    if valid.sum() < 25:
        return None
    hue, sat = hue[valid], sat[valid]

    hist, _ = np.histogram(hue, bins=bins, range=(0, 180), weights=sat)
    total = hist.sum()
    if total <= 0:
        return None
    return hist / total


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


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarita' 0..1 tra due istogrammi di tonalita' normalizzati (somma
    1) via INTERSEZIONE (sum(min(a, b))) -- metrica standard per istogrammi
    di colore: 1.0 = distribuzioni identiche, 0.0 = nessuna sovrapposizione.
    Cattura un pattern bicolore/a righe molto meglio di una singola distanza
    tra tonalita' medie (vedi il docstring di `_mask_hue_histogram`)."""
    return float(np.clip(np.minimum(a, b).sum(), 0.0, 1.0))


def _shape_similarity(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Similarita' 0..1 tra due coppie (aspect, fill)."""
    d_aspect = abs(a[0] - b[0]) / max(a[0], b[0], 1e-6)
    d_fill = abs(a[1] - b[1])
    return float(np.clip(1.0 - (min(d_aspect, 1.0) + d_fill) / 2.0, 0.0, 1.0))


@dataclass
class _PersonState:
    position: np.ndarray                      # ultimo centroide maschera noto
    scale: float                               # sqrt(area) o diagonale box
    color: np.ndarray | None                   # istogramma di tonalita' (vedi _mask_hue_histogram)
    shape: tuple[float, float] | None
    embedding: np.ndarray | None                # OSNet, media EMA (vedi appearance_embedding.ema_update)
    last_seen: float


@dataclass
class _MergeEvent:
    raw_track_id: int
    matched_person_id: int
    frame_time: float
    score: float
    forced: bool = False  # True solo se legato senza soglia perche' il tetto era gia' raggiunto


@dataclass
class _UncertainEvent:
    """Candidato in zona grigia (vedi identity_manager.resolve_batch): sotto
    `soft_match_threshold` ma non cosi' lontano da escludere del tutto un
    legame con una persona nota -- NON agganciato automaticamente (a meno
    che il tetto `max_people` non forzi comunque un aggancio, vedi
    `resolve()`), solo segnalato per revisione."""
    raw_track_id: int
    candidate_person_id: int
    frame_time: float
    score: float


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
                 soft_match_threshold: float = 0.6,
                 session_mode: SessionMode = SessionMode.MULTIPLE,
                 flag_uncertain: bool = True,
                 uncertain_score_margin: float = 0.15,
                 embedder: OSNetEmbedder | None = None,
                 embedding_weight: float = 0.7,
                 embedding_ema_alpha: float = 0.9):
        """`session_mode` (nuovo): SINGLE forza `max_people=1` (vedi
        `identity_manager.suggested_max_people_policy`), sovrascrivendo
        `max_people` se in conflitto -- coerente con `ReIdentifier` (reid.py).

        `flag_uncertain` / `uncertain_score_margin` (nuovi): quando l'aggancio
        morbido di un nuovo raw_id ha un punteggio sotto `soft_match_threshold`
        ma sopra `soft_match_threshold - uncertain_score_margin`, con
        `flag_uncertain=True` (default) NON viene agganciato ma registrato in
        `self.uncertain_log`/`self.last_uncertain` per revisione (a meno che
        il tetto max_people non forzi comunque un aggancio, vedi resolve());
        con `flag_uncertain=False` il comportamento torna binario come prima.

        `embedder` / `embedding_weight` / `embedding_ema_alpha` (nuovi): vedi
        "Segnale opzionale: embedding di aspetto (OSNet)" nel docstring del
        modulo. Il peso e' ADDITIVO rispetto ai tre esistenti (non li
        rescala): con `embedder=None` (default) il termine embedding e'
        sempre 0 per tutti, quindi `_pair_score` si comporta esattamente
        come prima di questa aggiunta (`position_weight`/`color_weight`/
        `shape_weight` di default sommano ancora a 1.0, stesso significato
        di prima per `soft_match_threshold`) -- nessuna regressione per chi
        non passa un `embedder`. Quando l'embedder e' attivo, il punteggio
        massimo possibile sale fino a 1.7 (0.7 in piu'): un match con
        embedding molto simile puo' superare la soglia anche con
        posizione/colore/forma mediocri, coerente col fatto che
        l'embedding, da solo, e' il segnale singolarmente piu' affidabile."""
        max_people = suggested_max_people_policy(session_mode, max_people)
        if max_people < 1:
            raise ValueError("max_people deve essere almeno 1")
        self.max_people = max_people
        self.session_mode = session_mode
        self.position_weight = position_weight
        self.color_weight = color_weight
        self.shape_weight = shape_weight
        self.max_position_dist_scales = max_position_dist_scales
        self.max_position_gap_seconds = max_position_gap_seconds
        # soglia sotto la quale un aggancio NON forzato (tetto non ancora
        # raggiunto) viene rifiutato -- vedi resolve() per il perche' serve
        # anche quando c'e' ancora un slot libero.
        self.soft_match_threshold = soft_match_threshold
        self.flag_uncertain = flag_uncertain
        self.uncertain_score_margin = uncertain_score_margin
        self.embedder = embedder
        self.embedding_weight = embedding_weight
        self.embedding_ema_alpha = embedding_ema_alpha

        self.raw_to_person: dict[int, int] = {}
        self.persons: dict[int, _PersonState] = {}  # TUTTE le identita' mai create (<= max_people)
        self.merge_log: list[_MergeEvent] = []
        self.uncertain_log: list[_UncertainEvent] = []
        # raw_track_id -> (candidate_person_id, score) SOLO per il frame
        # dell'ultima resolve() -- stesso pattern di ReIdentifier.last_uncertain
        # (reid.py), cosi' un chiamante puo' aggiungere una colonna
        # "identity_uncertain" al CSV senza che questo modulo cambi la forma
        # del valore di ritorno.
        self.last_uncertain: dict[int, tuple[int, float]] = {}
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
        self.last_uncertain = {}

        # -- descrittori per ciascuna persona in questo frame (una volta
        # sola, riusati sia per l'eventuale match batch sia per aggiornare
        # lo stato a fine funzione) --
        descriptors: dict[int, tuple[np.ndarray, float, tuple, tuple, tuple]] = {}
        for raw_id, bbox, poly, conf in people:
            centroid = mask_centroid(poly)
            area = mask_area(poly)
            scale = float(np.sqrt(area)) if area > 1.0 else float(np.linalg.norm(bbox[2:] - bbox[:2]))
            color = _mask_hue_histogram(frame, poly) if frame is not None else None
            shape = _shape_descriptor(bbox, poly)
            embedding = (self.embedder.embed(frame, bbox, poly=poly)
                         if frame is not None and self.embedder is not None else None)
            descriptors[raw_id] = (centroid, scale, color, shape, embedding)

        # -- risoluzione batch (ungherese) di TUTTI i raw_id mai visti prima
        # in questo frame in un colpo solo, invece del loop sequenziale
        # precedente (un raw_id alla volta, che si accaparrava lo slot
        # preferito senza considerare gli altri raw_id nuovi nello stesso
        # frame) -- vedi identity_manager.resolve_batch(). --
        new_raw_ids = [rid for rid, *_ in people if rid not in self.raw_to_person]
        if new_raw_ids and self.persons:
            self._resolve_new_batch(new_raw_ids, descriptors, now, claimed_this_frame)

        # -- raw_id ancora senza person_id dopo il match batch (nessuna
        # persona nota nel roster, o nessun candidato sopra soglia): slot
        # libero -> id nuovo; tetto raggiunto -> aggancio forzato (unica
        # eccezione a "mai forzare" del modulo, vedi docstring). `known_count`
        # conta il roster COME SARA' dopo questo frame (persone gia' note +
        # quelle appena coniate qui sotto): `self.persons` non viene
        # aggiornato fino al loop finale di resolve() (serve prima il
        # centroide/colore/forma di OGNI persona del frame, non solo di
        # quelle nuove), quindi senza questo contatore locale piu' raw_id
        # "nuovi" nello stesso frame vedrebbero tutti ancora il roster
        # com'era ALL'INIZIO del frame e supererebbero il tetto insieme
        # (bug: la versione precedente aggiornava self.persons dentro lo
        # stesso loop, qui i due passi sono separati apposta per il batch
        # ungherese sopra). --
        known_count = len(self.persons)
        for raw_id in new_raw_ids:
            if raw_id in self.raw_to_person:
                continue
            centroid, scale, color, shape, embedding = descriptors[raw_id]
            if known_count < self.max_people:
                person_id = self._next_person_id
                self._next_person_id += 1
                known_count += 1
            else:
                person_id, score = self._best_match(centroid, scale, color, shape, embedding, now,
                                                      exclude=claimed_this_frame)
                if person_id is None:
                    # caso patologico: piu' raw_id "nuovi" in questo frame
                    # di quanti slot restino liberi (es. detection doppia
                    # spuria). Il vincolo "mai un id in piu'" ha priorita'
                    # assoluta: riusa comunque lo slot migliore, anche se
                    # gia' rivendicato.
                    person_id, score = self._best_match(centroid, scale, color, shape, embedding,
                                                          now, exclude=set())
                self.merge_log.append(_MergeEvent(
                    raw_track_id=raw_id, matched_person_id=person_id,
                    frame_time=now, score=score, forced=True))
            self.raw_to_person[raw_id] = person_id
            claimed_this_frame.add(person_id)
            # Aggiorna subito lo stato della persona (non solo alla fine,
            # vedi il loop `out` sotto): il fallback "caso patologico" poco
            # sopra (`_best_match`) e i raw_id "nuovi" successivi in QUESTO
            # STESSO frame devono poter vedere le persone appena coniate/
            # agganciate qui, non solo quelle gia' note all'inizio del
            # frame -- altrimenti (bug gia' visto una volta, vedi
            # `known_count` sopra) il fallback non troverebbe candidati e
            # `_best_match` restituirebbe `None`.
            self._update_person_state(person_id, centroid, scale, color, shape, embedding, now)

        out = []
        for raw_id, bbox, poly, conf in people:
            centroid, scale, color, shape, embedding = descriptors[raw_id]
            person_id = self.raw_to_person[raw_id]
            self._update_person_state(person_id, centroid, scale, color, shape, embedding, now)
            out.append((person_id, bbox, poly, conf))

        gone_raw_ids = [rid for rid in self.raw_to_person if rid not in current_raw_ids]
        for rid in gone_raw_ids:
            del self.raw_to_person[rid]

        return out

    # -- interno ------------------------------------------------------

    def _update_person_state(self, person_id: int, centroid: np.ndarray, scale: float,
                              color: np.ndarray | None, shape: tuple[float, float],
                              embedding: np.ndarray | None, now: float) -> None:
        """Aggiorna (o crea) lo stato di `person_id` col descrittore di
        questo frame -- colore assente (`None`, vedi `_mask_hue_histogram`)
        o forma NaN non sovrascrivono un valore precedente valido (stesso
        comportamento di prima: un frame con colore non campionabile non
        deve far "dimenticare" l'ultimo istogramma buono noto). L'embedding
        NON sostituisce quello precedente ma lo affina con una media mobile
        esponenziale (`appearance_embedding.ema_update`, vedi "Segnale
        opzionale: embedding di aspetto" nel docstring del modulo) -- si
        consolida piu' a lungo la persona resta visibile, invece di
        dipendere da un solo frame."""
        prev = self.persons.get(person_id)
        self.persons[person_id] = _PersonState(
            position=centroid, scale=scale,
            color=color if color is not None else (prev.color if prev else None),
            shape=shape if not np.isnan(shape).any() else (prev.shape if prev else None),
            embedding=ema_update(prev.embedding if prev else None, embedding, self.embedding_ema_alpha),
            last_seen=now,
        )

    def _resolve_new_batch(self, new_raw_ids: list[int],
                            descriptors: dict[int, tuple[np.ndarray, float, tuple, tuple, tuple]],
                            now: float, claimed_this_frame: set[int]) -> None:
        """Aggancio "morbido" (con soglia), tentato SEMPRE per primo per
        OGNI raw_id nuovo di questo frame, anche se il tetto max_people non
        e' ancora raggiunto: un raw_id nuovo di ByteTrack e' spessissimo una
        persona gia' vista a cui il tracker ha appena riassegnato un id
        diverso (breve occlusione, uscita/rientro dal bordo inquadratura),
        non una persona davvero mai vista prima (vedi docstring del modulo
        per il perche' serve anche con un tetto largo apposta per tenersi
        margine). Risolto in un colpo solo (ungherese) tra tutti i raw_id
        nuovi di questo frame e tutte le persone note NON gia' rivendicate
        da un raw_id gia' esistente in questo stesso frame."""
        person_ids = [pid for pid in self.persons if pid not in claimed_this_frame]
        if not person_ids:
            return
        cost_matrix = np.full((len(new_raw_ids), len(person_ids)), np.inf)
        for i, raw_id in enumerate(new_raw_ids):
            centroid, scale, color, shape, embedding = descriptors[raw_id]
            for j, person_id in enumerate(person_ids):
                score = self._pair_score(centroid, scale, color, shape, embedding,
                                          now, self.persons[person_id])
                cost_matrix[i, j] = 1.0 - score

        config = IdentityManagerConfig(
            max_people=self.max_people, flag_uncertain=self.flag_uncertain,
            accept_cost=1.0 - self.soft_match_threshold,
            reject_cost=1.0 - self.soft_match_threshold + self.uncertain_score_margin,
        )
        outcomes = resolve_batch(cost_matrix, person_ids, config)

        for outcome in outcomes:
            raw_id = new_raw_ids[outcome.candidate_index]
            if outcome.status == "matched":
                person_id, score = outcome.matched_lost_id, 1.0 - outcome.cost
                self.raw_to_person[raw_id] = person_id
                claimed_this_frame.add(person_id)
                self.merge_log.append(_MergeEvent(
                    raw_track_id=raw_id, matched_person_id=person_id,
                    frame_time=now, score=score, forced=False))
            elif outcome.status == "uncertain":
                self.uncertain_log.append(_UncertainEvent(
                    raw_track_id=raw_id, candidate_person_id=outcome.matched_lost_id,
                    frame_time=now, score=1.0 - outcome.cost))
                self.last_uncertain[raw_id] = (outcome.matched_lost_id, 1.0 - outcome.cost)
            # "new": nessuna azione qui, gestito dal chiamante (resolve())
            # con l'allocazione di uno slot libero o l'aggancio forzato.

    def _pair_score(self, centroid: np.ndarray, scale: float,
                     color: np.ndarray | None, shape: tuple[float, float],
                     embedding: np.ndarray | None,
                     now: float, state: "_PersonState") -> float:
        """Punteggio 0..1 di UNA coppia raw_id-nuovo/persona-nota (media
        pesata posizione/colore/forma/embedding) -- estratto da `_best_match`
        per essere richiamabile anche nella costruzione della matrice di
        costo batch (`_resolve_new_batch`)."""
        pos_sim = 0.0
        if not np.isnan(centroid).any() and not np.isnan(state.position).any() and state.scale > 1e-6:
            dist_scales = float(np.linalg.norm(centroid - state.position)) / state.scale
            spatial = max(0.0, 1.0 - dist_scales / self.max_position_dist_scales)
            temporal = max(0.0, 1.0 - (now - state.last_seen) / self.max_position_gap_seconds)
            pos_sim = spatial * temporal

        color_sim = 0.0
        if state.color is not None and color is not None:
            color_sim = _histogram_similarity(color, state.color)

        shape_sim = 0.0
        if state.shape is not None and not np.isnan(shape).any():
            shape_sim = _shape_similarity(shape, state.shape)

        embedding_sim = 0.0
        if state.embedding is not None and embedding is not None:
            sim = embedding_similarity(embedding, state.embedding)
            if sim is not None:
                embedding_sim = sim

        return (self.position_weight * pos_sim + self.color_weight * color_sim
                + self.shape_weight * shape_sim + self.embedding_weight * embedding_sim)

    def _best_match(self, centroid: np.ndarray, scale: float,
                     color: np.ndarray | None, shape: tuple[float, float],
                     embedding: np.ndarray | None,
                     now: float, exclude: set[int]) -> tuple[int | None, float]:
        """Ricerca sequenziale del miglior candidato -- riusata SOLO per
        l'aggancio forzato (tetto max_people gia' raggiunto, vedi
        resolve()): li' l'accoppiamento batch non serve, ogni raw_id ancora
        senza person_id a quel punto viene comunque legato senza soglia, uno
        alla volta, escludendo gli slot gia' rivendicati in questo frame."""
        best_pid, best_score = None, -1.0
        for pid, state in self.persons.items():
            if pid in exclude:
                continue
            score = self._pair_score(centroid, scale, color, shape, embedding, now, state)
            if score > best_score:
                best_pid, best_score = pid, score
        return best_pid, best_score
