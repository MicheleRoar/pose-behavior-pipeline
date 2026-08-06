"""
hands.py
========
Tracking delle mani a livello di dita (21 landmark per mano) tramite
MediaPipe Tasks HandLandmarker, da combinare col tracking multi-persona di
YOLO-pose: YOLO segue le persone (bambino/caregiver) e fornisce i polsi
come punto di ancoraggio; HandLandmarker gira sull'intero frame e le mani
rilevate vengono associate al polso YOLO più vicino.

Perché a livello di dita e non solo il polso: lo score di "movimento
ripetitivo" già presente in `features.py` usa la velocità del polso come
proxy delle stereotipie manuali — funziona, ma non distingue, ad esempio,
un batter di mani (wrist quasi fermo, dita/mani che si aprono e chiudono)
da un vero hand-flapping (polso oscillante). Con i 21 landmark per mano si
può aggiungere un indice di apertura/chiusura della mano nel tempo,
complementare alla sola cinematica del polso.

Schema dei 21 landmark MediaPipe Hands (indici):
    0 wrist
    1-4   thumb (CMC, MCP, IP, TIP)
    5-8   index (MCP, PIP, DIP, TIP)
    9-12  middle (MCP, PIP, DIP, TIP)
    13-16 ring (MCP, PIP, DIP, TIP)
    17-20 pinky (MCP, PIP, DIP, TIP)

Setup richiesto:

    pip install mediapipe

Il modello Hand Landmarker viene scaricato IN AUTOMATICO alla prima
esecuzione in una cache fissa dentro il progetto (`<repo>/models/`), non
serve piu' un `curl` manuale -- vedi `common/mediapipe_models.py` per i
dettagli (stesso bug/fix di `pose/mediapipe_pose.py`: il default nudo
"hand_landmarker.task" veniva risolto da MediaPipe relativo alla cwd,
rompendosi se lanciato da una cwd diversa da quella del download manuale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.mediapipe_models import resolve_model_path
from pose.geometry import angle_at

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WRIST = 0
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

# Connessioni tra i 21 landmark, per disegnare lo scheletro della mano
# (approssimazione del set ufficiale MediaPipe HAND_CONNECTIONS).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # pollice
    (0, 5), (5, 6), (6, 7), (7, 8),        # indice
    (0, 9), (9, 10), (10, 11), (11, 12),   # medio
    (0, 13), (13, 14), (14, 15), (15, 16),  # anulare
    (0, 17), (17, 18), (18, 19), (19, 20),  # mignolo
    (5, 9), (9, 13), (13, 17),             # nocche (palmo)
]

# Terne (mcp, pip, tip) per stimare quanto ciascun dito è "piegato": l'angolo
# al giunto PIP (o equivalente) si avvicina a 180 gradi quando il dito è
# disteso, e si riduce quando il dito si piega verso il palmo.
FINGER_CURL_TRIPLETS = {
    "thumb": (2, 3, 4),      # MCP, IP, TIP (il pollice non ha PIP)
    "index": (5, 6, 8),      # MCP, PIP, TIP
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


def compute_finger_curls(hand_xy: np.ndarray) -> dict[str, float]:
    """Angolo (gradi) di flessione per ciascun dito: ~180 = disteso, valori
    più bassi = piegato verso il palmo. `hand_xy`: array (21, 2).
    """
    curls = {}
    for finger, (a_idx, b_idx, c_idx) in FINGER_CURL_TRIPLETS.items():
        curls[f"{finger}_curl"] = angle_at(hand_xy[a_idx], hand_xy[b_idx], hand_xy[c_idx])
    return curls


def hand_openness(hand_xy: np.ndarray) -> float:
    """Indice 0 (pugno chiuso) - 1 (mano aperta), basato sulla distanza
    media polso-punta di ciascun dito, normalizzata sulla dimensione della
    mano (distanza polso - nocca del medio, robusta alla scala/distanza
    dalla camera).
    """
    palm_size = np.linalg.norm(hand_xy[WRIST] - hand_xy[9])  # nocca medio
    if palm_size < 1e-6:
        return np.nan
    tip_distances = [np.linalg.norm(hand_xy[WRIST] - hand_xy[idx]) for idx in FINGER_TIPS.values()]
    avg_extension = np.mean(tip_distances) / palm_size
    # normalizzazione empirica: pugno chiuso ~1.0-1.3, mano aperta ~1.8-2.2
    return float(np.clip((avg_extension - 1.0) / 1.0, 0.0, 1.0))


def match_hands_to_wrists(hand_wrist_points: list[np.ndarray],
                           track_wrists: list[tuple[int, str, np.ndarray]],
                           max_distance: float = 60.0) -> dict[int, tuple[int, str]]:
    """Associa ogni mano rilevata (punto polso MediaPipe, indice 0) al
    polso YOLO più vicino tra tutte le persone tracciate.

    Parameters
    ----------
    hand_wrist_points : lista di punti (x, y), uno per mano rilevata
    track_wrists : lista di (track_id, "left"|"right", punto polso YOLO)
    max_distance : soglia oltre la quale l'associazione viene scartata

    Returns
    -------
    dict {indice_mano: (track_id, "left"|"right")}
    """
    assignments: dict[int, tuple[int, str]] = {}
    used: set[tuple[int, str]] = set()

    order = sorted(
        range(len(hand_wrist_points)),
        key=lambda i: min(
            (np.linalg.norm(hand_wrist_points[i] - w) for _, _, w in track_wrists),
            default=np.inf,
        ),
    )

    for hand_idx in order:
        best_key, best_dist = None, max_distance
        for tid, side, w in track_wrists:
            key = (tid, side)
            if key in used:
                continue
            d = float(np.linalg.norm(hand_wrist_points[hand_idx] - w))
            if d < best_dist:
                best_key, best_dist = key, d
        if best_key is not None:
            assignments[hand_idx] = best_key
            used.add(best_key)

    return assignments


# ---------------------------------------------------------------------------
# Wrapper su MediaPipe Tasks HandLandmarker (richiede mediapipe + modello)
# ---------------------------------------------------------------------------

@dataclass
class HandResult:
    landmarks_xy: np.ndarray   # (21, 2) in coordinate pixel del frame
    handedness: str            # "Left" | "Right" (etichetta MediaPipe, vedi nota)


class HandTracker:
    """Wrapper su MediaPipe Tasks HandLandmarker.

    Nota sulla lateralità: l'etichetta Left/Right di MediaPipe è calcolata
    dal punto di vista della persona nell'immagine (non della camera), e
    può risultare invertita a seconda che il feed sia specchiato o meno.
    Per questo l'associazione a una persona/lato specifico in questa
    pipeline si basa sulla vicinanza spaziale al polso YOLO
    (`match_hands_to_wrists`), non sull'etichetta di MediaPipe.
    """

    def __init__(self, model_path: str = "hand_landmarker.task", num_hands: int = 4):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = resolve_model_path(model_path, download_url=_MODEL_URL)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandResult]:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame_bgr.shape[:2]
        out = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            xy = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
            label = handedness[0].category_name if handedness else "Unknown"
            out.append(HandResult(landmarks_xy=xy, handedness=label))
        return out
