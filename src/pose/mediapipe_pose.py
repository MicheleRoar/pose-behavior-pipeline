"""
mediapipe_pose.py
===================
Stima della posa (keypoint corporei) con MediaPipe Tasks PoseLandmarker,
applicata DENTRO il ritaglio (bbox) di una singola persona GIA' tracciata
dalla pipeline di segmentazione (`segmentation/seg_estimation.py` +
`segmentation/seg_reid.py`) -- non un rilevatore multi-persona sull'intero
frame.

Perche' "dentro il ritaglio" e non sull'intero frame
------------------------------------------------------
MediaPipe Tasks PoseLandmarker supporta il rilevamento multi-persona
(`num_poses`), ma NON fornisce alcun tracking temporale: nessun id
persistente tra un frame e l'altro, a differenza di YOLO+ByteTrack.
Costruire un tracker equivalente da zero sopra i suoi rilevamenti grezzi
sarebbe un lavoro consistente per un guadagno incerto (esattamente il tipo
di instabilita' di tracking per cui questa pipeline e' passata a
segmentation_demo.py in primo luogo, vedi README).

La scelta fatta qui e' piu' semplice e riusa quello che gia' funziona: la
pipeline di segmentazione traccia gia' l'identita' di ciascuna persona in
modo stabile (silhouette + posizione/colore/forma, vedi seg_reid.py).
Questo modulo si limita ad applicare MediaPipe in modalita' SINGOLA
PERSONA (la piu' matura e affidabile della libreria, nessun problema di
associazione multi-persona da risolvere) dentro il box di ciascuna persona
GIA' tracciata, frame per frame -- l'identita' viene presa in prestito
dalla segmentazione, non ricostruita qui. Era gia' il piano descritto nel
docstring di `seg_estimation.py` ("ricollegare la pose applicata dentro la
sagoma tracciata").

Rimappatura su COCO-17
------------------------
I 33 landmark BlazePose vengono rimappati sui 17 nomi COCO-17 gia' usati in
tutta la pipeline pose (`pose/keypoints.py`, vedi `BLAZEPOSE_TO_COCO`
sotto), cosi' le funzioni di feature engineering esistenti
(`pose/features.py`, `common/viz.draw_skeleton`, ecc.) funzionano
IDENTICHE indipendentemente dal modello che ha prodotto i keypoint. I 16
landmark BlazePose senza equivalente COCO diretto (occhi interni/esterni,
angoli bocca, dita, talloni, punte piedi) vengono scartati: non servono
alle feature esistenti, tutte scritte per lo schema COCO-17.

Limiti onesti
-------------
  - Nessun tracking/feature con finestra scorrevole (energia di movimento,
    repetitivita', ecc.) e' ancora collegato qui -- solo angoli articolari
    calcolabili istantaneamente, frame per frame (vedi wiring in
    segmentation_demo.py). Collegare il resto di pose/features.py
    richiederebbe gestire buffer per-persona anche in segmentation_demo.py,
    non ancora fatto.
  - Un box di segmentazione stretto sulla sagoma puo' tagliare mani alzate
    sopra la testa o piedi vicino al bordo: il padding in `estimate()`
    aiuta ma non elimina il problema.
  - La confidenza per giunto (`visibility` di MediaPipe) e la confidenza
    delle detection YOLO (`pose_estimation.py`) non sono necessariamente
    sulla stessa scala: trattarle come intercambiabili in analisi
    quantitative va validato.

Setup richiesto:

    pip install mediapipe

Il modello Pose Landmarker ("lite", il piu' veloce -- "full"/"heavy" sono
piu' precisi ma piu' lenti, vedi `_MODEL_URL` sotto per l'URL delle altre
varianti) viene scaricato IN AUTOMATICO alla prima esecuzione in una
cache fissa dentro il progetto (`<repo>/models/pose_landmarker_lite.task`,
vedi `_DEFAULT_MODEL_CACHE_PATH`), non serve piu' un `curl` manuale.

Perche' il download automatico (bug reale osservato): il default
precedente era il nome nudo `"pose_landmarker_lite.task"`, risolto da
MediaPipe come path RELATIVO ALLA CWD -- funzionava solo se si lanciava
lo script dalla stessa cartella in cui si era fatto il `curl` a mano, e
falliva con un errore poco chiaro ("unable to find pose_landmarker_lite")
se lanciato da una cwd diversa (es. `cd src && python webui_app.py`
invece che dalla root del progetto). Un path assoluto fisso + download
automatico rende il comportamento indipendente da dove viene lanciato lo
script. Se invece si passa esplicitamente un `model_path` diverso
(es. per usare la variante "full"/"heavy" gia' scaricata altrove), quel
path viene rispettato cosi' com'e', nessun download automatico.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from pose.keypoints import KP

# Stessa variante "lite" gia' documentata prima -- vedi il docstring sopra
# per "full"/"heavy" (sostituire "lite" con "full"/"heavy" nell'URL).
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
# .../pose-behavior-pipeline/src/pose/mediapipe_pose.py -> parents[2] e' la
# root del progetto (src/pose -> src -> root), stessa convenzione di
# Path(__file__).resolve().parents[N] gia' usata in segmentation/sam2_estimation.py.
_DEFAULT_MODEL_CACHE_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"
_DEFAULT_MODEL_BASENAME = "pose_landmarker_lite.task"


def _resolve_model_path(model_path: str) -> str:
    """Se `model_path` esiste gia' (path esplicito dell'utente, anche
    relativo alla cwd corrente -- comportamento invariato per chi lo passa
    apposta), lo usa cosi' com'e'. Altrimenti, SOLO se e' il nome nudo di
    default (non un path custom che l'utente ha sbagliato: in quel caso
    meglio l'errore originale di MediaPipe che indovinare), lo risolve/
    scarica nella cache fissa del progetto -- vedi il docstring del
    modulo per il bug che questo risolve."""
    if os.path.isfile(model_path):
        return model_path
    if os.path.basename(model_path) != _DEFAULT_MODEL_BASENAME:
        return model_path
    if not _DEFAULT_MODEL_CACHE_PATH.exists():
        _download_pose_landmarker_lite(_DEFAULT_MODEL_CACHE_PATH)
    return str(_DEFAULT_MODEL_CACHE_PATH)


def _download_pose_landmarker_lite(dest: Path) -> None:
    """Scarica il modello "lite" (~5-6 MB) in `dest`, creando le cartelle
    mancanti. Nessun retry/hash-check: se il download si interrompe a
    meta', il file parziale resta li' e il prossimo avvio lo tratterebbe
    come 'gia' presente' (bug noto, accettabile per ora -- se capita,
    basta cancellare il file e rilanciare)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mediapipe_pose] scarico il modello Pose Landmarker (una tantum) in {dest} ...")
    urllib.request.urlretrieve(_MODEL_URL, dest)
    print(f"[mediapipe_pose] fatto: {dest}")

# Indice landmark BlazePose (0-32, schema MediaPipe Pose Landmarker) -> nome
# COCO-17 (pose/keypoints.py). I landmark BlazePose senza equivalente COCO
# diretto (occhi interni/esterni, angoli bocca, dita, talloni, punte piedi)
# non compaiono qui: vengono scartati.
BLAZEPOSE_TO_COCO: dict[int, str] = {
    0: "nose",
    2: "left_eye", 5: "right_eye",
    7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}


def _empty_pose() -> tuple[np.ndarray, np.ndarray]:
    """(kxy, kconf) "vuoti": 17 giunti NaN/confidenza zero, stesso schema
    di un frame in cui nessuna posa e' stata rilevata."""
    return np.full((17, 2), np.nan), np.zeros(17)


def blazepose_to_coco(landmarks, frame_offset_xy: tuple[float, float],
                       crop_size_wh: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Converte una lista di 33 landmark BlazePose (in coordinate
    NORMALIZZATE 0-1 rispetto al ritaglio, come restituiti da MediaPipe) in
    (kxy, kconf) COCO-17 in coordinate PIXEL DEL FRAME INTERO.

    `frame_offset_xy`: angolo in alto a sinistra del ritaglio nel frame
    intero (x1, y1). `crop_size_wh`: dimensioni del ritaglio in pixel.
    Isolata dalla classe che chiama MediaPipe per essere testabile senza
    mediapipe/camera (vedi demo/mediapipe_pose_check.py).
    """
    x1, y1 = frame_offset_xy
    crop_w, crop_h = crop_size_wh
    kxy, kconf = _empty_pose()
    for blaze_idx, coco_name in BLAZEPOSE_TO_COCO.items():
        lm = landmarks[blaze_idx]
        coco_idx = KP[coco_name]
        kxy[coco_idx] = [x1 + lm.x * crop_w, y1 + lm.y * crop_h]
        visibility = getattr(lm, "visibility", None)
        kconf[coco_idx] = float(visibility) if visibility is not None else 1.0
    return kxy, kconf


def padded_crop_box(bbox: np.ndarray, frame_shape: tuple[int, int],
                     padding: float = 0.15) -> tuple[int, int, int, int]:
    """Box di ritaglio (x1,y1,x2,y2, interi, clampati ai bordi del frame) a
    partire da un bbox di segmentazione, allargato di `padding` (frazione
    di larghezza/altezza) per non tagliare le estremita' (mani alzate,
    piedi) quando il box e' stretto sulla sagoma. Isolata per essere
    testabile senza mediapipe/camera."""
    h, w = frame_shape
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(w, int(x2 + bw * padding))
    y2 = min(h, int(y2 + bh * padding))
    return x1, y1, x2, y2


class MediaPipeCropPoseEstimator:
    """Wrapper su MediaPipe Tasks PoseLandmarker in modalita' SINGOLA
    persona (`num_poses=1`), applicato a un ritaglio del frame -- vedi il
    docstring del modulo per il perche'. Import di mediapipe ritardato
    (come `pose_estimation.PoseTracker` / `gaze_head.HeadGazeEstimator`),
    cosi' il resto della pipeline resta utilizzabile/testabile anche senza
    mediapipe installato.
    """

    def __init__(self, model_path: str = "pose_landmarker_lite.task",
                 min_pose_detection_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = _resolve_model_path(model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms: int | None = None  # vedi la clamp in estimate()

    def estimate(self, frame_bgr: np.ndarray, bbox: np.ndarray, timestamp_ms: int,
                 padding: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        """Rileva la posa DENTRO `bbox` (x1,y1,x2,y2, es. dal tracker di
        segmentazione). Ritorna (kxy, kconf) in coordinate PIXEL DEL FRAME
        INTERO (non del ritaglio), stesso schema COCO-17 di
        `pose_estimation.py` -- NaN/0 per i giunti senza equivalente
        BlazePose o se nessuna posa e' stata rilevata nel ritaglio.

        ATTENZIONE (vedi anche il docstring di `MediaPipePoseByTrack`):
        `detect_for_video` (modalita' VIDEO di MediaPipe) richiede
        timestamp STRETTAMENTE crescenti per la stessa istanza -- chiamare
        questo metodo con lo stesso timestamp due volte (es. per due
        persone diverse nello stesso frame, sulla STESSA istanza) solleva
        `ValueError: Input timestamp must be monotonically increasing`.
        Un'istanza va quindi usata per UNA sola persona nel tempo (vedi
        `MediaPipePoseByTrack`, che aggancia un'istanza per track_id). Come
        rete di sicurezza aggiuntiva -- non come sostituto di quel design --
        qui il timestamp effettivo viene comunque forzato a essere maggiore
        dell'ultimo usato, cosi' un timestamp duplicato o fuori ordine (es.
        per arrotondamento a fps molto bassi) non manda in crash ma perde
        al piu' un millisecondo di precisione."""
        import cv2

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = padded_crop_box(bbox, (h, w), padding)
        if x2 <= x1 or y2 <= y1:
            return _empty_pose()

        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        crop = frame_bgr[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return _empty_pose()

        landmarks = result.pose_landmarks[0]  # num_poses=1: al massimo una posa
        crop_h, crop_w = crop.shape[:2]
        return blazepose_to_coco(landmarks, (x1, y1), (crop_w, crop_h))


class MediaPipePoseByTrack:
    """Pool di un `MediaPipeCropPoseEstimator` INDIPENDENTE per ciascun
    `track_id`, invece di un'unica istanza condivisa fra tutte le persone
    del frame.

    Perche' non basta un'istanza sola
    ------------------------------------
    `PoseLandmarker.detect_for_video` (modalita' VIDEO) e' pensato per UNO
    stream continuo per istanza: mantiene stato interno tra una chiamata e
    l'altra (filtraggio/smussamento temporale dei landmark) e richiede
    timestamp strettamente crescenti. Il ciclo per-persona di
    `segmentation_demo.py` chiama pero' `.estimate(...)` UNA VOLTA PER
    PERSONA nello stesso frame, tutte con lo stesso timestamp (`now` del
    frame) -- su un'istanza condivisa questo (a) fa scattare
    `ValueError: Input timestamp must be monotonically increasing` alla
    seconda persona del frame, e (b) anche aggirando il crash, mescolerebbe
    lo stato di smussamento temporale di persone diverse come se fossero
    un'unica persona che si teletrasporta da un corpo all'altro.

    La soluzione e' che ogni track_id ottenga la propria istanza/il proprio
    "stream" indipendente -- creata alla prima apparizione del track e
    riusata per tutta la sua vita, cosi' ciascuna vede una sequenza di
    timestamp coerente e uno stato di smussamento che appartiene solo a
    lei."""

    def __init__(self, model_path: str = "pose_landmarker_lite.task",
                 min_pose_detection_confidence: float = 0.5):
        # Risolto/scaricato UNA volta qui (non ad ogni nuovo track_id in
        # estimate()): _resolve_model_path() e' economico da richiamare,
        # ma non ha senso ripetere il controllo/stampare il messaggio di
        # download per ogni persona che entra in scena.
        self._model_path = _resolve_model_path(model_path)
        self._min_conf = min_pose_detection_confidence
        self._estimators: dict[int, MediaPipeCropPoseEstimator] = {}

    def estimate(self, track_id: int, frame_bgr: np.ndarray, bbox: np.ndarray,
                 timestamp_ms: int, padding: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        estimator = self._estimators.get(track_id)
        if estimator is None:
            estimator = MediaPipeCropPoseEstimator(
                model_path=self._model_path, min_pose_detection_confidence=self._min_conf)
            self._estimators[track_id] = estimator
        return estimator.estimate(frame_bgr, bbox, timestamp_ms=timestamp_ms, padding=padding)

    def forget(self, track_id: int) -> None:
        """Rimuove l'istanza di un track uscito di scena (es. scaduto in
        seg_reid) -- evita di accumulare landmarker per id ormai morti in
        sessioni lunghe con molto ricambio. Non obbligatorio (un handful di
        istanze in piu' non e' un problema pratico), ma economico da
        chiamare quando si sa gia' che un track e' morto."""
        self._estimators.pop(track_id, None)
