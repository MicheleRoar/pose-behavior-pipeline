"""
chuv_features.py
=================
Replica in tempo reale del feature engineering del repository CHUV
(Video-Annotation-System, `src/models/train.py::normalize_keypoints` +
`add_derived_pose_features` + `add_temporal_features`), adattata da
BODY-25 (psifx/SAM3 + MediaPipe, OpenPose-style) a COCO-17 (YOLO-pose) e da
un calcolo offline/batch a un calcolo frame-per-frame in tempo reale.

Perche' qui e non nel repository CHUV: stesso motivo di `reid.py` -- quella
pipeline richiede SAM3 su GPU CUDA, non disponibile su MacBook M1. Il
feature engineering (angoli, distanze, simmetria, centro di massa, derivate
temporali) non dipende da SAM3/psifx: e' pura geometria su keypoint 2D,
quindi e' portabile e testabile qui con YOLO+ByteTrack su dati non protetti
-- l'obiettivo di questo modulo e' vedere GLI STESSI NUMERI che calcolerebbe
il repository originale, prodotti pero' da un tracker molto piu' leggero.

Cosa NON viene replicato
-------------------------
- 5 colonne del set di feature finale del repository CHUV sono coordinate
  grezze di punta-piede/tallone/sfondo (l_big_toe_y, l_heel_y,
  r_small_toe_x, r_small_toe_y, r_heel_x, background_y): derivano dallo
  schema BODY-25 (OpenPose), che li include; COCO-17 (YOLO-pose) NON li
  ha, quindi queste colonne non sono riproducibili qui.
- Il modello addestrato (model_xgboost.joblib) NON viene caricato: le sue
  classi sono codici di annotazione clinica specifici (formato WAKEE) che
  richiedono dati etichettati da un osservatore umano secondo un
  protocollo che qui non esiste, e il file che mappa gli indici numerici
  del modello alle etichette (il LabelEncoder) non e' salvato dal
  repository originale. Questo modulo si ferma al feature engineering.

Differenza deliberata: derivate temporali
-------------------------------------------
Nel repository CHUV le velocita'/accelerazioni sono calcolate con
`df.groupby(annotation_label).diff()` -- un effetto collaterale del fatto
che il training e' offline e raggruppato per classe (la derivata si azzera
ai confini tra classi di annotazione). Qui, in tempo reale, non esiste "la
classe" del frame corrente mentre lo si acquisisce, quindi la derivata e'
calcolata in modo continuo, frame-per-frame, per ciascun person_id/track_id
(vedi `ChuvFeatureTracker`) -- piu' corretto fisicamente, ma non e' un
numero direttamente confrontabile 1:1 con l'output del repository
originale su uno stesso video.

Nota sul centro di massa (com_x, com_y): dopo la normalizzazione rispetto
al bacino (mid_hip diventa sempre l'origine (0,0)), com_x/com_y si
riducono matematicamente a meta' della posizione del collo -- non un vero
centro di massa fisico multi-segmento. E' una caratteristica del calcolo
originale del repository CHUV (`com_x = (mid_hip_x + neck_x) / 2` su
coordinate gia' normalizzate), riprodotta qui fedelmente, non "corretta":
l'obiettivo di questo modulo e' la fedelta' al repository, non il suo
miglioramento.

Nota sulla normalizzazione: nel repository CHUV, una torso_length pari a
zero viene sostituita con la mediana calcolata sull'intero dataset offline
(`normalize_keypoints`). Qui, in tempo reale, non esiste "l'intero
dataset" da cui stimare una mediana in anticipo: se la torso_length di un
frame e' invalida (nan/troppo piccola), le coordinate normalizzate di quel
frame sono NaN -- una scelta deliberatamente onesta piuttosto che un
fallback arbitrario.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from keypoints import KP

# ---------------------------------------------------------------------------
# Keypoint "virtuali" BODY-25 ricostruiti da COCO-17
# ---------------------------------------------------------------------------

def _neck_xy(kxy: np.ndarray) -> np.ndarray:
    return (kxy[KP["left_shoulder"]] + kxy[KP["right_shoulder"]]) / 2.0


def _mid_hip_xy(kxy: np.ndarray) -> np.ndarray:
    return (kxy[KP["left_hip"]] + kxy[KP["right_hip"]]) / 2.0


RAW_POINTS = [
    "nose", "neck", "mid_hip",
    "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist",
    "r_hip", "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

_COCO_NAME = {  # nome CHUV (r_/l_ prefix, stile BODY-25) -> nome COCO-17 (KP)
    "r_shoulder": "right_shoulder", "l_shoulder": "left_shoulder",
    "r_elbow": "right_elbow", "l_elbow": "left_elbow",
    "r_wrist": "right_wrist", "l_wrist": "left_wrist",
    "r_hip": "right_hip", "l_hip": "left_hip",
    "r_knee": "right_knee", "l_knee": "left_knee",
    "r_ankle": "right_ankle", "l_ankle": "left_ankle",
    "r_eye": "right_eye", "l_eye": "left_eye",
    "r_ear": "right_ear", "l_ear": "left_ear",
}


def _raw_point(kxy: np.ndarray, name: str) -> np.ndarray:
    if name == "neck":
        return _neck_xy(kxy)
    if name == "mid_hip":
        return _mid_hip_xy(kxy)
    if name == "nose":
        return kxy[KP["nose"]]
    return kxy[KP[_COCO_NAME[name]]]


# ---------------------------------------------------------------------------
# Stage 1: normalizzazione rispetto al bacino (identica a
# normalize_keypoints del repository CHUV, salvo la nota sul fallback sopra)
# ---------------------------------------------------------------------------

def normalize_keypoints(kxy: np.ndarray) -> dict[str, np.ndarray]:
    """Coordinate normalizzate rispetto al bacino: (x - mid_hip_x) /
    torso_length, (y - mid_hip_y) / torso_length -- stessa formula di
    `train.py::normalize_keypoints` nel repository CHUV. Ritorna un dict
    nome -> array (2,) [x, y]; NaN dove il/i keypoint sorgente mancano o la
    torso_length e' invalida."""
    mid_hip = _mid_hip_xy(kxy)
    neck = _neck_xy(kxy)
    torso = float(np.linalg.norm(neck - mid_hip))
    valid_torso = np.isfinite(torso) and torso >= 1e-3

    out = {}
    for name in RAW_POINTS:
        p = _raw_point(kxy, name)
        out[name] = (p - mid_hip) / torso if valid_torso else np.full(2, np.nan)
    return out


# ---------------------------------------------------------------------------
# Stage 2: feature derivate (angoli, distanze, simmetria, COM, spread) --
# stessa logica di add_derived_pose_features del repository CHUV, sulle
# coordinate gia' normalizzate.
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


CHUV_ANGLE_TRIPLETS: dict[str, tuple[str, str, str]] = {
    "r_elbow_angle": ("r_shoulder", "r_elbow", "r_wrist"),
    "l_elbow_angle": ("l_shoulder", "l_elbow", "l_wrist"),
    "r_shoulder_angle": ("neck", "r_shoulder", "r_elbow"),
    "l_shoulder_angle": ("neck", "l_shoulder", "l_elbow"),
    "r_knee_angle": ("r_hip", "r_knee", "r_ankle"),
    "l_knee_angle": ("l_hip", "l_knee", "l_ankle"),
    "r_hip_angle": ("mid_hip", "r_hip", "r_knee"),
    "l_hip_angle": ("mid_hip", "l_hip", "l_knee"),
    "trunk_angle": ("nose", "neck", "mid_hip"),
}
# nota: r_shoulder_angle/l_shoulder_angle qui usano il vertice "neck" (come
# nel repository CHUV) -- una definizione diversa dai
# left_shoulder_angle/right_shoulder_angle gia' presenti in features.py
# (che usano il vertice "hip"), quindi NON sono duplicati: sono i due
# angoli-spalla calcolati con due convenzioni diverse, entrambi conservati.

DERIVED_COLS = [
    *CHUV_ANGLE_TRIPLETS.keys(),
    "eye_to_eye", "nose_to_neck", "r_wrist_to_hip", "l_wrist_to_hip",
    "r_wrist_to_nose", "l_wrist_to_nose", "nose_to_ankles", "hip_to_ankle",
    "shoulder_y_diff", "hip_y_diff", "elbow_angle_diff", "knee_angle_diff",
    "wrist_to_hip_diff", "shoulder_angle_diff",
    "com_x", "com_y", "body_spread_x", "body_spread_y",
]


def compute_derived_features(norm: dict[str, np.ndarray]) -> dict[str, float]:
    """Angoli, distanze, simmetria, COM e spread -- stessa formula di
    `add_derived_pose_features` nel repository CHUV, applicata alle
    coordinate gia' normalizzate da `normalize_keypoints`."""
    out: dict[str, float] = {}

    for name, (a, b, c) in CHUV_ANGLE_TRIPLETS.items():
        out[name] = _angle_deg(norm[a], norm[b], norm[c])

    out["eye_to_eye"] = _dist(norm["l_eye"], norm["r_eye"])
    out["nose_to_neck"] = _dist(norm["nose"], norm["neck"])
    out["r_wrist_to_hip"] = _dist(norm["r_wrist"], norm["mid_hip"])
    out["l_wrist_to_hip"] = _dist(norm["l_wrist"], norm["mid_hip"])
    out["r_wrist_to_nose"] = _dist(norm["r_wrist"], norm["nose"])
    out["l_wrist_to_nose"] = _dist(norm["l_wrist"], norm["nose"])
    out["nose_to_ankles"] = (_dist(norm["nose"], norm["l_ankle"]) + _dist(norm["nose"], norm["r_ankle"])) / 2.0
    out["hip_to_ankle"] = (_dist(norm["mid_hip"], norm["l_ankle"]) + _dist(norm["mid_hip"], norm["r_ankle"])) / 2.0

    out["shoulder_y_diff"] = float(norm["l_shoulder"][1] - norm["r_shoulder"][1])
    out["hip_y_diff"] = float(norm["l_hip"][1] - norm["r_hip"][1])
    out["elbow_angle_diff"] = out["l_elbow_angle"] - out["r_elbow_angle"]
    out["knee_angle_diff"] = out["l_knee_angle"] - out["r_knee_angle"]
    out["wrist_to_hip_diff"] = out["l_wrist_to_hip"] - out["r_wrist_to_hip"]
    out["shoulder_angle_diff"] = out["l_shoulder_angle"] - out["r_shoulder_angle"]

    out["com_x"] = float((norm["mid_hip"][0] + norm["neck"][0]) / 2.0)
    out["com_y"] = float((norm["mid_hip"][1] + norm["neck"][1]) / 2.0)

    spread_x_pts = np.array([norm["l_wrist"][0], norm["r_wrist"][0], norm["l_ankle"][0], norm["r_ankle"][0]])
    spread_y_pts = np.array([norm["nose"][1], norm["l_ankle"][1], norm["r_ankle"][1]])
    out["body_spread_x"] = float(np.nanmax(spread_x_pts) - np.nanmin(spread_x_pts))
    out["body_spread_y"] = float(np.nanmax(spread_y_pts) - np.nanmin(spread_y_pts))

    return out


# ---------------------------------------------------------------------------
# Stage 3: derivate temporali (velocita'/accelerazione) -- stessa selezione
# di keypoint di add_temporal_features nel repository CHUV, ma calcolate
# frame-per-frame in tempo reale (vedi "Differenza deliberata" nel
# docstring del modulo).
# ---------------------------------------------------------------------------

TEMPORAL_POINTS = ["com", "nose", "l_wrist", "r_wrist", "l_ankle", "r_ankle", "neck", "mid_hip"]
TEMPORAL_COLS = [f"{name}_{axis}_{kind}" for name in TEMPORAL_POINTS
                  for axis in ("x", "y") for kind in ("vel", "acc")]


@dataclass
class ChuvFeatureTracker:
    """Mantiene, per person_id/track_id, l'ultimo frame normalizzato e
    l'ultima velocita' calcolata, per derivare velocita'/accelerazione
    frame-per-frame senza dover tenere in memoria l'intera sessione (a
    differenza del repository CHUV, che opera offline su un CSV completo).
    """
    _prev: dict[int, dict] = field(default_factory=dict)

    def update(self, track_id: int, norm: dict[str, np.ndarray], now: float) -> dict[str, float]:
        points = {
            "com": np.array([(norm["mid_hip"][0] + norm["neck"][0]) / 2.0,
                              (norm["mid_hip"][1] + norm["neck"][1]) / 2.0]),
            **{name: norm[name] for name in TEMPORAL_POINTS if name != "com"},
        }

        prev = self._prev.get(track_id)
        out: dict[str, float] = {}

        if prev is None:
            for col in TEMPORAL_COLS:
                out[col] = np.nan
            self._prev[track_id] = {
                "points": points,
                "vel": {name: np.full(2, np.nan) for name in TEMPORAL_POINTS},
                "t": now,
            }
            return out

        dt = max(now - prev["t"], 1e-3)
        vel = {}
        for name in TEMPORAL_POINTS:
            v = (points[name] - prev["points"][name]) / dt
            vel[name] = v
            out[f"{name}_x_vel"], out[f"{name}_y_vel"] = float(v[0]), float(v[1])
            a = (v - prev["vel"][name]) / dt
            out[f"{name}_x_acc"], out[f"{name}_y_acc"] = float(a[0]), float(a[1])

        self._prev[track_id] = {"points": points, "vel": vel, "t": now}
        return out

    def forget(self, track_id: int) -> None:
        """Da chiamare quando un track_id/person_id esce definitivamente
        dall'inquadratura, per non lasciare stato agganciato a un id che
        non ricomparira' -- e per evitare una velocita' vicina a zero
        (anziche' NaN) se quello stesso id viene riassegnato molto piu'
        tardi dopo un gap enorme (es. da `reid.py` dopo un lungo rientro)."""
        self._prev.pop(track_id, None)


# ---------------------------------------------------------------------------
# Punto di ingresso unico per l'uso in live_demo.py
# ---------------------------------------------------------------------------

def compute_chuv_features(kxy: np.ndarray, track_id: int, now: float,
                           tracker: ChuvFeatureTracker) -> dict[str, float]:
    """Tutte le feature "in stile CHUV" per un singolo frame di una
    persona: coordinate normalizzate, feature derivate (angoli/distanze/
    simmetria/COM/spread) e derivate temporali (velocita'/accelerazione).
    """
    norm = normalize_keypoints(kxy)
    out: dict[str, float] = {}
    for name, xy in norm.items():
        out[f"{name}_x"], out[f"{name}_y"] = float(xy[0]), float(xy[1])
    out.update(compute_derived_features(norm))
    out.update(tracker.update(track_id, norm, now))
    return out
