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
- In parallelo, si accumula un piccolo buffer (finestra scorrevole) di
  proporzioni corporee per quel track_id. Una volta raccolti abbastanza
  frame validi, si calcola una firma mediana e la si confronta con le
  persone recentemente scomparse dall'inquadratura (`lost`, con scadenza
  dopo `max_lost_seconds`). Se non c'e' match, il tentativo NON si ferma
  subito: ad ogni frame successivo la finestra si aggiorna (i frame piu'
  vecchi escono, quelli nuovi entrano) e si ritenta -- un rientro con i
  primi frame rumorosi (persona ancora ai bordi dell'inquadratura) non e'
  quindi perso per sempre, solo ritardato finche' i dati non sono
  abbastanza puliti. Due guardrail evitano pero' che il retry diventi
  pericoloso: (1) un track non viene mai confrontato con una persona
  "persa" DOPO che il track stesso era gia' apparso -- i due erano visibili
  insieme (sotto raw_id diversi), quindi non possono essere la stessa
  persona (altrimenti chi e' presente fin dall'inizio della sessione
  rischierebbe di essere agganciato per coincidenza a un'identita' persa
  molto piu' tardi); (2) il retry si arrende comunque dopo
  `_PENDING_RETRY_SECONDS` dall'apparizione del track, per non ricontrollarlo
  per sempre.
- Se c'e' un match sotto soglia, TUTTI i frame SUCCESSIVI a quel punto
  vengono attribuiti al `person_id` precedente invece che al nuovo -- i
  frame gia' emessi (CSV/overlay) con il person_id provvisorio NON vengono
  riscritti. L'evento di merge viene loggato esplicitamente
  (`self.merge_log`) per trasparenza/audit, cosi' un'analisi a posteriori
  puo' ricollegare i due pezzi se necessario -- stessa filosofia
  "proponi/rendi tracciabile, non decidere in silenzio" di reid_signature.py,
  adattata al fatto che qui non c'e' un umano nel loop in tempo reale.

Segnale opzionale: colore maglia/pantaloni/capelli
----------------------------------------------------
La sola firma antropometrica, in pratica, puo' essere troppo debole quando
i keypoint sono rumorosi (occlusioni parziali, persona ai bordi
dell'inquadratura durante l'uscita/rientro): due proporzioni corporee
leggermente sballate possono far mancare un match vero. Se viene passato il
frame video a `resolve()`, viene calcolato ANCHE un colore medio (tonalita'
+ saturazione, non luminosita' -- piu' robusto a cambi di esposizione) su
tre regioni: torso (maglia), coscia (pantaloni) -- campionate dai pixel
dentro il poligono definito dai keypoint di spalle/anche/ginocchia -- e una
zona sopra le orecchie (proxy del colore dei capelli, stimata estendendo
verso l'alto la larghezza orecchio-orecchio della distanza collo-naso).
Il colore capelli e' indipendente dai vestiti: aiuta proprio nel caso in
cui una persona rientra con abiti diversi (es. senza maglietta).

Il colore NON sostituisce le proporzioni ne' alza mai la soglia di
rifiuto: se il colore e' molto diverso (es. cambio vestiti tra uscita e
rientro) il match si decide esattamente come prima, solo sulle proporzioni.
Se invece il colore e' simile (il caso piu' comune: stessi vestiti durante
la sessione), la distanza tra le proporzioni viene "scontata" -- rendendo
piu' facile recuperare un match vero anche con proporzioni un po' rumorose,
senza mai peggiorare la robusta invarianza al vestiario che era l'obiettivo
originale del prototipo.

Segnale opzionale: posizione
------------------------------
Pensato per il caso concreto di un cambio vestiti "in scena" (es. mettere
una giacca/grembiule a un bambino durante un'attivita', tipicamente una
decina di secondi): il bambino non esce mai dall'inquadratura, ma
l'occlusione parziale da parte di chi lo veste puo' far perdere il track e
generarne uno nuovo al distacco. In questo caso il bambino non si e' quasi
spostato, quindi l'ultima posizione nota (centro-bacino) prima della
perdita e la prima posizione del nuovo track sono molto vicine -- stesso
principio anche per un'uscita/rientro dalla stessa porta. Ogni frame si
memorizza posizione e scala (lunghezza busto) della persona corrente; alla
perdita, quella posizione viene congelata insieme alla firma. Un nuovo
track viene confrontato su vicinanza spaziale (in lunghezze di busto,
`MAX_POSITION_DIST_TORSOS`) E vicinanza temporale (`MAX_POSITION_GAP_SECONDS`)
-- entrambe devono valere qualcosa, non basta una sola.

Stesso principio del colore: la posizione puo' solo scontare la distanza,
MAI bloccare un match ne' sostituirsi alla firma. Colore e posizione non si
sommano tra loro -- si prende il PIU' FORTE dei due sconti disponibili,
non la somma, per evitare che due segnali solo mediocri si accumulino a una
fiducia che nessuno dei due giustificherebbe da solo.

Segnale opzionale: numero massimo di persone (sessione chiusa)
------------------------------------------------------------------
Quando si conosce a priori quante persone possono comparire nella sessione
(es. 2 per un 1v1 bambino-caregiver, fino a una decina per una sessione di
gruppo), quel numero diventa un vincolo forte, non solo un altro "sconto":
se sono gia' state confermate `max_people` identita' distinte e un nuovo
track non trova match con le soglie normali, NON puo' comunque trattarsi di
un'undicesima persona -- deve per forza essere una delle persone gia' note
che al momento risulta "persa". In questo caso, e solo in questo caso, si
FORZA il merge con la persona persa piu' vicina per firma (anche sopra
`max_signature_dist`), invece di rinunciare e coniare un nuovo person_id.

Due paletti tengono la forzatura sicura: (1) vale la stessa regola di
causalita' delle altre corrispondenze -- non si puo' forzare un match con
qualcuno che era gia' visibile insieme a questo track sotto un altro id;
(2) si forza SOLO contro persone attualmente "perse" (fuori inquadratura),
mai contro persone attive in quel momento -- due persone visibili
contemporaneamente restano sempre due identita' distinte, il tetto massimo
non le fonde mai. Se il tetto e' raggiunto ma non c'e' nessuno "perso" da
recuperare, si rinuncia comunque a forzare (nessun candidato sensato) e si
conia un nuovo person_id stampando un avviso -- segnale che il conteggio
configurato potrebbe essere sbagliato o che una detection spuria ha
superato il filtro `max_people` di `pose_estimation.py`.

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
  - La posizione e' un'arma a doppio taglio in scene con due persone
    vicine (es. bambino+caregiver): SOLO lo sconto (mai la forzatura) tiene
    a bada il rischio, ma se le proporzioni di due persone sono gia'
    ambigue di per se', la vicinanza spaziale puo' spingere un confronto
    borderline oltre soglia nella direzione sbagliata -- scelta deliberata,
    non un problema nascosto.
  - Il merge, una volta deciso, e' applicato automaticamente (non c'e' modo
    di chiedere conferma a un umano in tempo reale) -- per questo l'evento
    resta sempre nel log, invece di sparire silenziosamente.
  - `max_people` e' una forzatura vera (unica eccezione a "mai forzare" nel
    modulo): se il numero configurato e' sbagliato (es. un adulto in piu'
    entra brevemente in scena, fuori dal conteggio previsto) il merge
    forzato puo' attribuire i suoi frame alla persona persa sbagliata --
    va usato solo quando il numero di partecipanti e' davvero fisso e noto.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from pose.keypoints import KP
from pose.features import torso_length, hip_center

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
    # geometria della testa: indipendente dai vestiti quanto le proporzioni
    # del corpo, stesso trattamento paritario degli altri segmenti (nessun
    # peso speciale) -- aggiunge segnale quando corpo/braccia sono rumorosi.
    "eye_to_eye": ("left_eye", "right_eye"),
    "ear_to_ear": ("left_ear", "right_ear"),
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
COLOR_COLS = ["shirt_h", "shirt_s", "pants_h", "pants_s", "hair_h", "hair_s"]
_HUE_IDX = [0, 2, 4]   # indici circolari (0..1) in COLOR_COLS
_SAT_IDX = [1, 3, 5]   # indici lineari (0..1) in COLOR_COLS


def _region_mean_hs(frame: np.ndarray, pts: np.ndarray) -> tuple[float, float]:
    """Tonalita'/saturazione medie (OpenCV HSV, poi normalizzate 0-1) dei
    pixel dentro il poligono definito dai 4 punti (x, y) indicati. (nan, nan)
    se un punto manca (NaN) o il poligono e' degenere/troppo piccolo."""
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


def hair_corners(kxy: np.ndarray) -> np.ndarray:
    """Quattro angoli approssimati della regione sopra le orecchie (proxy
    del colore dei capelli, indipendente dai vestiti): larghezza
    orecchio-orecchio, estesa verso l'alto della distanza collo-naso
    (proporzionale alla dimensione della testa, invariante alla distanza
    dalla camera). NaN se un keypoint sorgente manca -- stesso trattamento
    delle altre regioni di colore, mai un fallback inventato."""
    neck = (kxy[KP["left_shoulder"]] + kxy[KP["right_shoulder"]]) / 2.0
    nose, l_ear, r_ear = kxy[KP["nose"]], kxy[KP["left_ear"]], kxy[KP["right_ear"]]
    if np.isnan(np.array([neck, nose, l_ear, r_ear])).any():
        return np.full((4, 2), np.nan)
    up = nose - neck
    return np.array([r_ear, l_ear, l_ear + up, r_ear + up])


def compute_color_signature(frame: np.ndarray, kxy: np.ndarray) -> np.ndarray:
    """Colore medio (tonalita', saturazione) di maglia, pantaloni e capelli
    per un singolo frame, nell'ordine di `COLOR_COLS`, NaN dove non
    campionabile. Solo Hue/Saturation (non Value/luminosita'): piu' robusto
    a cambi di esposizione/illuminazione tra un'uscita e un rientro."""
    out = np.full(len(COLOR_COLS), np.nan)
    for i, corner_names in enumerate(COLOR_SEGMENTS.values()):
        pts = kxy[[KP[name] for name in corner_names]]
        out[2 * i], out[2 * i + 1] = _region_mean_hs(frame, pts)
    out[4], out[5] = _region_mean_hs(frame, hair_corners(kxy))
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
# Posizione (segnale opzionale, complementare alla firma antropometrica --
# vedi "Segnale opzionale: posizione" nel docstring del modulo)
# ---------------------------------------------------------------------------

# Raggio spaziale (in lunghezze di busto, quindi invariante alla distanza
# dalla camera) oltre il quale la posizione non conta piu' come "stesso
# punto". 4 lunghezze di busto copre sia un rientro dalla stessa porta
# (piccolo spostamento) sia un bambino fermo mentre gli mettono una giacca
# (spostamento quasi nullo).
MAX_POSITION_DIST_TORSOS = 4.0

# Tempo oltre il quale il segnale posizionale si azzera. Un cambio di
# giacca dura tipicamente una decina di secondi: a 20s il peso e' zero, a
# 10s e' ancora a meta' -- non ai margini della validita'.
MAX_POSITION_GAP_SECONDS = 20.0


def position_similarity(pos_a: np.ndarray, pos_b: np.ndarray, torso_scale: float,
                         gap_seconds: float) -> float | None:
    """Similarita' posizionale 0..1 (1 = stesso punto, appena successo) tra
    due posizioni (centro-bacino), combinando vicinanza spaziale (in
    lunghezze di busto, decadimento lineare fino a `MAX_POSITION_DIST_TORSOS`)
    e vicinanza temporale (decadimento lineare fino a
    `MAX_POSITION_GAP_SECONDS`). `None` se una posizione manca o la scala
    non e' valida."""
    if np.isnan(pos_a).any() or np.isnan(pos_b).any():
        return None
    if torso_scale < 1e-6 or np.isnan(torso_scale):
        return None
    dist_torsos = float(np.linalg.norm(pos_a - pos_b)) / torso_scale
    spatial = max(0.0, 1.0 - dist_torsos / MAX_POSITION_DIST_TORSOS)
    temporal = max(0.0, 1.0 - gap_seconds / MAX_POSITION_GAP_SECONDS)
    return spatial * temporal


# ---------------------------------------------------------------------------
# Stato per persona/traccia
# ---------------------------------------------------------------------------

@dataclass
class _LostPerson:
    signature: np.ndarray
    lost_time: float
    color: np.ndarray | None = None
    position: np.ndarray | None = None    # ultimo centro-bacino noto
    last_torso: float | None = None       # scala per normalizzare la posizione


@dataclass
class _MergeEvent:
    raw_track_id: int
    provisional_person_id: int
    matched_person_id: int
    distance: float
    frame_time: float
    color_used: bool = False
    position_used: bool = False
    forced: bool = False  # match forzato da max_people (vedi docstring del modulo)


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

    # Tempo massimo (dall'apparizione di un track MAI visto prima) durante
    # cui si ritenta il match contro le persone "perse": oltre questa
    # soglia si rinuncia e si accetta il track come identita' nuova per
    # sempre, invece di ricontrollarlo a ogni frame per il resto della
    # sessione (vedi nota nel docstring del modulo su perche' e' necessario
    # un limite, non solo "ritenta finche' non trovi qualcosa").
    _PENDING_RETRY_SECONDS = 2.0

    def __init__(self, max_lost_seconds: float = 180.0,
                 max_signature_dist: float = 0.12,
                 min_signature_frames: int = 15,
                 color_bonus_weight: float = 0.5,
                 position_bonus_weight: float = 0.5,
                 max_people: int | None = None):
        self.max_lost_seconds = max_lost_seconds
        self.max_signature_dist = max_signature_dist
        self.min_signature_frames = min_signature_frames
        self.color_bonus_weight = color_bonus_weight
        self.position_bonus_weight = position_bonus_weight
        self.max_people = max_people

        self.raw_to_person: dict[int, int] = {}
        self.buffers: dict[int, deque] = {}
        self.color_buffers: dict[int, deque] = {}
        self.pending_check: set[int] = set()  # raw_track_id ancora da valutare
        self.pending_since: dict[int, float] = {}  # raw_track_id -> now al primo avvistamento
        self.last_position: dict[int, tuple[np.ndarray, float]] = {}  # person_id -> (hip_xy, torso)
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
                self._assign_provisional(raw_id, now)

            person_id = self.raw_to_person[raw_id]
            self.buffers[person_id].append(compute_signature_frame(kxy))
            if frame is not None:
                self.color_buffers[person_id].append(compute_color_signature(frame, kxy))

            hip, torso = hip_center(kxy), torso_length(kxy)
            if not np.isnan(hip).any() and not np.isnan(torso) and torso > 1e-6:
                self.last_position[person_id] = (hip, torso)

            if raw_id in self.pending_check and len(self.buffers[person_id]) >= self.min_signature_frames:
                match = self._find_lost_match(self.buffers[person_id], self.color_buffers[person_id],
                                               self.pending_since[raw_id], now,
                                               self.last_position.get(person_id))
                if match is None and now - self.pending_since[raw_id] > self._PENDING_RETRY_SECONDS:
                    # rinuncia al match normale: prima di coniare un nuovo
                    # person_id per sempre, se e' impostato un tetto noto di
                    # persone e quel tetto e' gia' raggiunto, si prova un
                    # ultimo tentativo forzato (vedi "Segnale opzionale:
                    # numero massimo di persone" nel docstring del modulo).
                    self.pending_check.discard(raw_id)
                    if self.max_people is not None:
                        match = self._force_match_at_capacity(
                            self.buffers[person_id], self.pending_since[raw_id], person_id)
                if match is not None:
                    self.pending_check.discard(raw_id)
                    matched_person_id, dist, color_used, position_used, forced = match
                    self.merge_log.append(_MergeEvent(
                        raw_track_id=raw_id, provisional_person_id=person_id,
                        matched_person_id=matched_person_id, distance=dist, frame_time=now,
                        color_used=color_used, position_used=position_used, forced=forced,
                    ))
                    assist = ", ".join(s for s, used in (("color", color_used), ("position", position_used)) if used)
                    tag = "FORCED (max_people reached)" if forced else (f'{assist}-assisted' if assist else None)
                    print(f"[reid] track {raw_id}: provisional person_id {person_id} "
                          f"re-matched to person_id {matched_person_id} "
                          f"(distance={dist:.3f}{f', {tag}' if tag else ''})")
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

    def _assign_provisional(self, raw_id: int, now: float) -> None:
        person_id = self._next_person_id
        self._next_person_id += 1
        self.raw_to_person[raw_id] = person_id
        self.buffers[person_id] = deque(maxlen=self.min_signature_frames)
        self.color_buffers[person_id] = deque(maxlen=self.min_signature_frames)
        self.pending_check.add(raw_id)
        self.pending_since[raw_id] = now

    def _find_lost_match(self, buffer: deque, color_buffer: deque,
                          pending_since: float, now: float,
                          current_position: tuple[np.ndarray, float] | None,
                          ) -> tuple[int, float, bool, bool, bool] | None:
        median_sig = np.nanmedian(np.array(buffer), axis=0)
        median_color = np.nanmedian(np.array(color_buffer), axis=0) if color_buffer else None

        best_person_id, best_dist = None, self.max_signature_dist
        best_color_used, best_position_used = False, False
        for person_id, lost in self.lost.items():
            if lost.lost_time > pending_since:
                # causalita': questa persona e' sparita DOPO che il track
                # in valutazione era gia' apparso -- i due erano quindi
                # visibili insieme (o il track esisteva gia') sotto due
                # raw_id diversi, non possono essere la stessa persona.
                # Evita falsi match tra chi e' presente fin dall'inizio e
                # chi se ne va molto piu' tardi nella stessa sessione.
                continue
            prop_dist = signature_distance(median_sig, lost.signature)
            if prop_dist is None:
                continue

            # Due sconti indipendenti (colore, posizione): si prende il
            # PIU' FORTE dei due, non si sommano/moltiplicano insieme --
            # due segnali solo mediocri non devono sommarsi a una fiducia
            # che nessuno dei due giustificherebbe da solo. Nessuno dei due
            # puo' MAI alzare la distanza: nel peggiore dei casi (nessun
            # segnale utilizzabile) il comportamento e' identico alla sola
            # firma antropometrica.
            color_bonus, color_used = 0.0, False
            if median_color is not None and lost.color is not None:
                sim = color_similarity(median_color, lost.color)
                if sim is not None:
                    color_bonus, color_used = self.color_bonus_weight * sim, True

            position_bonus, position_used = 0.0, False
            if current_position is not None and lost.position is not None:
                cur_xy, cur_torso = current_position
                scale = cur_torso if lost.last_torso is None else (cur_torso + lost.last_torso) / 2.0
                sim = position_similarity(cur_xy, lost.position, scale, now - lost.lost_time)
                if sim is not None:
                    position_bonus, position_used = self.position_bonus_weight * sim, True

            if color_bonus >= position_bonus:
                dist, used_color, used_position = prop_dist * (1.0 - color_bonus), color_used, False
            else:
                dist, used_color, used_position = prop_dist * (1.0 - position_bonus), False, position_used

            if dist < best_dist:
                best_person_id, best_dist = person_id, dist
                best_color_used, best_position_used = used_color, used_position
        return ((best_person_id, best_dist, best_color_used, best_position_used, False)
                if best_person_id is not None else None)

    def _force_match_at_capacity(self, buffer: deque, pending_since: float,
                                  pending_person_id: int,
                                  ) -> tuple[int, float, bool, bool, bool] | None:
        """Ultimo tentativo, solo quando `max_people` e' impostato: se,
        SENZA CONTARE la persona provvisoria in valutazione, il numero di
        identita' note (attive + perse) ha gia' raggiunto il tetto, allora
        questa non puo' essere una persona in piu' -- deve essere una di
        quelle attualmente "perse". Sceglie la piu' vicina per firma anche
        sopra `max_signature_dist` (vedi "Segnale opzionale: numero massimo
        di persone" nel docstring del modulo). Non forza mai contro persone
        attualmente attive, solo contro quelle perse, e rispetta la stessa
        regola di causalita' dei match normali. Nessun candidato idoneo ->
        None (si rinuncia, non si forza a vuoto).

        Il conteggio esclude deliberatamente `pending_person_id`: se non lo
        facesse, anche la prima apparizione in assoluto di una persona
        genuinamente nuova (es. la seconda di una sessione 1v1, mai persa
        da nessuno) scatterebbe come "al tetto" appena il suo stesso
        retry scade, rischiando un merge forzato sbagliato con chiunque
        risulti perso in quel momento -- il tetto deve riferirsi a QUANTE
        ALTRE identita' esistono gia', non contare se stessa.
        """
        roster_size = len((set(self.raw_to_person.values()) | set(self.lost.keys()))
                           - {pending_person_id})
        if roster_size < self.max_people:
            return None
        if not self.lost:
            print(f"[reid] warning: max_people={self.max_people} raggiunto ma nessuna "
                  f"persona 'persa' da recuperare -- conio comunque un nuovo person_id "
                  f"(controlla il numero configurato, o una detection spuria ha superato "
                  f"il filtro max_people di pose_estimation.py)")
            return None

        median_sig = np.nanmedian(np.array(buffer), axis=0)
        best_person_id, best_dist = None, None
        for person_id, lost in self.lost.items():
            if lost.lost_time > pending_since:
                continue  # stessa regola di causalita' dei match normali
            prop_dist = signature_distance(median_sig, lost.signature)
            if prop_dist is None:
                continue
            if best_dist is None or prop_dist < best_dist:
                best_person_id, best_dist = person_id, prop_dist

        if best_person_id is None:
            print(f"[reid] warning: max_people={self.max_people} raggiunto ma nessun "
                  f"candidato idoneo (regola di causalita') -- conio comunque un nuovo "
                  f"person_id")
            return None
        return (best_person_id, best_dist, False, False, True)

    def _retire_disappeared_tracks(self, current_raw_ids: set[int], now: float) -> None:
        gone_raw_ids = [rid for rid in self.raw_to_person if rid not in current_raw_ids]
        for raw_id in gone_raw_ids:
            person_id = self.raw_to_person.pop(raw_id)
            self.pending_check.discard(raw_id)
            self.pending_since.pop(raw_id, None)
            buffer = self.buffers.pop(person_id, None)
            color_buffer = self.color_buffers.pop(person_id, None)
            last_pos = self.last_position.pop(person_id, None)
            if buffer and len(buffer) > 0:
                median_sig = np.nanmedian(np.array(buffer), axis=0)
                if not np.isnan(median_sig).all():
                    median_color = None
                    if color_buffer and len(color_buffer) > 0:
                        cand = np.nanmedian(np.array(color_buffer), axis=0)
                        if not np.isnan(cand).all():
                            median_color = cand
                    position, last_torso = (last_pos if last_pos is not None else (None, None))
                    self.lost[person_id] = _LostPerson(
                        signature=median_sig, lost_time=now, color=median_color,
                        position=position, last_torso=last_torso)

    def _expire_old_lost_people(self, now: float) -> None:
        expired = [pid for pid, lost in self.lost.items()
                   if now - lost.lost_time > self.max_lost_seconds]
        for pid in expired:
            del self.lost[pid]
