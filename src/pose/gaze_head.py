"""
gaze_head.py
============
Stima di head-pose e un proxy di "attenzione condivisa" (joint attention) a
partire da MediaPipe Tasks FaceLandmarker, da usare in combinazione con il
tracking multi-persona di YOLO-pose: YOLO individua e segue le persone nel
frame (bambino/caregiver), FaceLandmarker viene applicato sull'intero frame
per ottenere landmark facciali densi + matrice di trasformazione facciale
(da cui si ricava l'orientamento della testa), e le due uscite vengono poi
associate per prossimità spaziale.

Perché la testa/sguardo prima delle mani a livello di dita: la letteratura
su marker comportamentali in ambito neurosviluppo infantile (vedi README)
cita esplicitamente la coordinazione dello sguardo e la frequenza dei
movimenti della testa come indicatori rilevanti (es. "head turn to
disengage attention", joint attention bambino-caregiver).

Nota metodologica importante: quanto implementato qui è un PROXY 2D
semplificato, non un vero gaze-tracking 3D calibrato. Con una singola
camera RGB, senza calibrazione intrinseca né stima di profondità, non è
possibile ricostruire con precisione dove una persona sta guardando nello
spazio 3D. L'euristica usata (`joint_attention_score`) confronta l'head-yaw
stimato con la direzione (bearing) verso la testa dell'altra persona
nell'immagine, assumendo che le due persone siano a distanza comparabile
dalla camera — un'approssimazione ragionevole per una singola stanza di
osservazione, ma da validare empiricamente prima di qualunque uso
interpretativo.

Setup richiesto:

    pip install mediapipe

Il modello Face Landmarker viene scaricato IN AUTOMATICO alla prima
esecuzione in una cache fissa dentro il progetto (`<repo>/models/`), non
serve piu' un `curl` manuale -- vedi `common/mediapipe_models.py` per i
dettagli (stesso bug/fix di `pose/mediapipe_pose.py`/`pose/hands.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.mediapipe_models import resolve_model_path

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


# ---------------------------------------------------------------------------
# Geometria: matrice di rotazione -> angoli di Eulero
# ---------------------------------------------------------------------------

def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Estrae (yaw, pitch, roll) in gradi da una matrice di rotazione 3x3.

    Convenzione: yaw = rotazione attorno all'asse verticale (destra/sinistra,
    positivo verso l'immagine-destra), pitch = su/giù, roll = inclinazione
    laterale. Il segno esatto dipende dalla convenzione d'assi di MediaPipe:
    va validato empiricamente sul proprio setup (es. muovendo la testa a
    destra e verificando che lo yaw stimato aumenti); se risulta invertito,
    basta negare il valore in `HeadGazeEstimator`.
    """
    assert R.shape == (3, 3), f"attesa matrice 3x3, ricevuta {R.shape}"
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
        roll = np.arctan2(-R[1, 2], R[1, 1])

    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Inversa di `rotation_matrix_to_euler`, usata solo per generare casi di
    test sintetici (nessun uso nella pipeline live).
    """
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Proxy di attenzione condivisa (joint attention)
# ---------------------------------------------------------------------------

def bearing_to_target(head_xy: np.ndarray, target_xy: np.ndarray,
                       frame_width: float, fov_deg: float = 60.0) -> float:
    """Angolo orizzontale approssimato (gradi) verso `target_xy` nell'immagine,
    visto da `head_xy`, assumendo una camera pinhole con field-of-view
    orizzontale `fov_deg` e le due persone a distanza comparabile dalla
    camera (nessuna stima di profondità reale). Positivo = target a destra.
    """
    dx = target_xy[0] - head_xy[0]
    return float((dx / (frame_width / 2.0)) * (fov_deg / 2.0))


def joint_attention_score(head_xy: np.ndarray, yaw_deg: float, target_xy: np.ndarray,
                           frame_width: float, fov_deg: float = 60.0,
                           tolerance_deg: float = 20.0) -> float:
    """Punteggio 0-1 di quanto la testa risulti orientata verso `target_xy`
    (es. la testa dell'altra persona tracciata), come proxy semplificato di
    attenzione condivisa. Vedi nota metodologica in testa al modulo.
    """
    expected_yaw = bearing_to_target(head_xy, target_xy, frame_width, fov_deg)
    diff = abs(yaw_deg - expected_yaw)
    return float(max(0.0, 1.0 - diff / tolerance_deg))


# ---------------------------------------------------------------------------
# Bocca (mouth aspect ratio) e occhi (eye aspect ratio / blink)
# ---------------------------------------------------------------------------
#
# Indici dei landmark del volto (schema a 468/478 punti di MediaPipe Face
# Landmarker) usati per bocca e occhi. Sono gli indici comunemente usati in
# letteratura/community per il calcolo di mouth/eye aspect ratio (lo stesso
# principio del classico Eye Aspect Ratio di Soukupová & Čech, qui applicato
# ai landmark MediaPipe invece che a quelli dlib originali).

MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291

# Ordine [angolo_sx, palpebra_sup_1, palpebra_sup_2, angolo_dx, palpebra_inf_2, palpebra_inf_1]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

# Contorno del sopracciglio (5 punti ciascuno), stessi indici usati dalle
# connessioni FACEMESH_LEFT/RIGHT_EYEBROW di MediaPipe.
RIGHT_EYEBROW_IDX = [70, 63, 105, 66, 107]
LEFT_EYEBROW_IDX = [336, 296, 334, 293, 300]


def mouth_aspect_ratio(face_xy: np.ndarray) -> float:
    """Rapporto apertura verticale / larghezza orizzontale della bocca.
    Valori bassi = bocca chiusa, valori più alti = bocca aperta. Utile come
    proxy grezzo di vocalizzazione o di movimenti ripetitivi della bocca
    (es. mouthing), non come riconoscimento del parlato.
    """
    top, bottom = face_xy[MOUTH_TOP], face_xy[MOUTH_BOTTOM]
    left, right = face_xy[MOUTH_LEFT], face_xy[MOUTH_RIGHT]
    horizontal = np.linalg.norm(left - right)
    if horizontal < 1e-6:
        return np.nan
    vertical = np.linalg.norm(top - bottom)
    return float(vertical / horizontal)


def eye_aspect_ratio(face_xy: np.ndarray, idx: list[int]) -> float:
    """Eye Aspect Ratio (EAR) per un occhio: rapporto tra apertura verticale
    media e larghezza orizzontale. Scende bruscamente durante un ammiccamento
    (blink). `idx`: 6 indici [angolo_sx, sup1, sup2, angolo_dx, inf2, inf1].
    """
    p1, p2, p3, p4, p5, p6 = [face_xy[i] for i in idx]
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return np.nan
    vertical = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / 2.0
    return float(vertical / horizontal)


def mean_eye_aspect_ratio(face_xy: np.ndarray) -> float:
    """Media dell'EAR sui due occhi (più robusta di un singolo occhio in
    caso di lieve rotazione della testa)."""
    left = eye_aspect_ratio(face_xy, LEFT_EYE_EAR_IDX)
    right = eye_aspect_ratio(face_xy, RIGHT_EYE_EAR_IDX)
    values = [v for v in (left, right) if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


# ---------------------------------------------------------------------------
# Sopracciglia
# ---------------------------------------------------------------------------

def interocular_distance(face_xy: np.ndarray) -> float:
    """Distanza tra gli angoli esterni dei due occhi: usata come unità di
    scala del volto (invariante rispetto alla distanza dalla camera) per
    normalizzare il sollevamento del sopracciglio."""
    return float(np.linalg.norm(face_xy[RIGHT_EYE_EAR_IDX[0]] - face_xy[LEFT_EYE_EAR_IDX[0]]))


def eyebrow_raise_ratio(face_xy: np.ndarray, eyebrow_idx: list[int], eye_idx: list[int],
                         ref_distance: float) -> float:
    """Punteggio di sollevamento del sopracciglio: distanza verticale tra il
    sopracciglio e la palpebra superiore corrispondente, normalizzata sulla
    distanza interoculare (così è confrontabile indipendentemente dalla
    distanza dalla camera). Valori più alti = sopracciglio sollevato (es.
    sorpresa); valori più bassi/negativi = sopracciglio abbassato o
    aggrottato (es. concentrazione, disappunto). Soglie di interpretazione
    non calibrate: da tarare sul soggetto/contesto specifico, come per
    MAR/EAR.
    """
    if ref_distance < 1e-6 or np.isnan(ref_distance):
        return np.nan
    eyebrow_y = float(np.mean([face_xy[i][1] for i in eyebrow_idx]))
    # palpebra superiore = punti sup1, sup2 dello schema EAR (indici 1, 2)
    eyelid_y = float(np.mean([face_xy[eye_idx[1]][1], face_xy[eye_idx[2]][1]]))
    return (eyelid_y - eyebrow_y) / ref_distance


def mean_eyebrow_raise(face_xy: np.ndarray) -> tuple[float, float]:
    """Sollevamento sopracciglio sinistro e destro (in quest'ordine)."""
    ref = interocular_distance(face_xy)
    left = eyebrow_raise_ratio(face_xy, LEFT_EYEBROW_IDX, LEFT_EYE_EAR_IDX, ref)
    right = eyebrow_raise_ratio(face_xy, RIGHT_EYEBROW_IDX, RIGHT_EYE_EAR_IDX, ref)
    return left, right


# ---------------------------------------------------------------------------
# Associazione volti (FaceLandmarker) <-> persone tracciate (YOLO)
# ---------------------------------------------------------------------------

def match_faces_to_tracks(face_centers: list[np.ndarray], track_ids: list[int],
                           track_head_centers: list[np.ndarray],
                           max_distance: float = 80.0) -> dict[int, int]:
    """Associa ogni volto rilevato da FaceLandmarker al track_id (YOLO) con
    il centro-testa più vicino, entro `max_distance` pixel. Nearest-neighbor
    greedy: sufficiente per 2-3 persone nella scena (bambino + caregiver).

    Returns
    -------
    dict {indice_volto: track_id}
    """
    assignments: dict[int, int] = {}
    used_tracks: set[int] = set()

    order = sorted(
        range(len(face_centers)),
        key=lambda i: min(
            (np.linalg.norm(face_centers[i] - c) for c in track_head_centers),
            default=np.inf,
        ),
    )

    for face_idx in order:
        best_tid, best_dist = None, max_distance
        for tid, center in zip(track_ids, track_head_centers):
            if tid in used_tracks:
                continue
            d = float(np.linalg.norm(face_centers[face_idx] - center))
            if d < best_dist:
                best_tid, best_dist = tid, d
        if best_tid is not None:
            assignments[face_idx] = best_tid
            used_tracks.add(best_tid)

    return assignments


# ---------------------------------------------------------------------------
# Wrapper su MediaPipe Tasks FaceLandmarker (richiede mediapipe + modello)
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    landmarks_xy: np.ndarray       # (478, 2) in coordinate pixel del frame
    yaw: float
    pitch: float
    roll: float
    mouth_ratio: float = float("nan")
    eye_ratio: float = float("nan")
    left_eyebrow_raise: float = float("nan")
    right_eyebrow_raise: float = float("nan")


class HeadGazeEstimator:
    """Wrapper su MediaPipe Tasks FaceLandmarker per head-pose multi-volto.

    Import di mediapipe ritardato (come in `pose_estimation.PoseTracker`) in
    modo che il resto della pipeline resti utilizzabile/testabile anche
    senza mediapipe installato.
    """

    def __init__(self, model_path: str = "face_landmarker.task", num_faces: int = 3):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = resolve_model_path(model_path, download_url=_MODEL_URL)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[FaceResult]:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame_bgr.shape[:2]
        out = []
        for i, face_landmarks in enumerate(result.face_landmarks):
            xy = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])
            if result.facial_transformation_matrixes:
                R = np.array(result.facial_transformation_matrixes[i])[:3, :3]
                yaw, pitch, roll = rotation_matrix_to_euler(R)
            else:
                yaw, pitch, roll = np.nan, np.nan, np.nan
            mouth_ratio = mouth_aspect_ratio(xy)
            eye_ratio = mean_eye_aspect_ratio(xy)
            left_brow, right_brow = mean_eyebrow_raise(xy)
            out.append(FaceResult(landmarks_xy=xy, yaw=yaw, pitch=pitch, roll=roll,
                                   mouth_ratio=mouth_ratio, eye_ratio=eye_ratio,
                                   left_eyebrow_raise=left_brow, right_eyebrow_raise=right_brow))
        return out
