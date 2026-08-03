"""
viz.py
======
Utility di disegno condivise tra la pipeline live (`live_demo.py`, che legge
da Canon R8 / webcam / file video) e gli script di verifica eseguibili in
ambienti senza fotocamera (vedi `demo/live_render_check.py`), così la stessa
logica di rendering è testata anche dove non è disponibile una sorgente
video reale.
"""

from __future__ import annotations

import cv2
import numpy as np

from keypoints import KP, SKELETON_EDGES

# Palette di colori distinti (BGR, come OpenCV) assegnati ciclicamente per
# track_id, cosi' ogni persona nell'inquadratura ha un colore diverso e
# riconoscibile a colpo d'occhio (scheletro, bordo riquadro metriche,
# etichetta ID) invece del verde fisso usato per tutti prima di questa
# funzione.
TRACK_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (0, 220, 0),      # verde
    (0, 140, 255),    # arancione
    (255, 0, 255),    # magenta
    (255, 220, 0),    # ciano
    (0, 255, 255),    # giallo
    (255, 0, 0),      # blu
    (180, 105, 255),  # rosa
    (0, 128, 128),    # oliva/teal
]


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Colore stabile e distinto per un dato track_id, ciclico sulla
    palette. Usato per differenziare visivamente più persone tracciate
    nella stessa inquadratura (scheletro, riquadro metriche, etichetta)."""
    return TRACK_COLOR_PALETTE[track_id % len(TRACK_COLOR_PALETTE)]


def draw_person_label(frame: np.ndarray, position: np.ndarray, track_id: int,
                       color: tuple[int, int, int], is_target: bool = False) -> np.ndarray:
    """Disegna un'etichetta "ID N" grande e leggibile sopra la testa della
    persona, con sfondo colorato (stesso colore del suo scheletro) — più
    prominente del solo testo piccolo nel riquadro metriche. Se `is_target`
    è True (persona selezionata con --target-track-id), aggiunge un
    indicatore visivo extra.
    """
    text = f"ID {track_id}" + (" ★ TARGET" if is_target else "")
    x, y = int(position[0]), max(int(position[1]) - 20, 20)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x - 6, y - th - 10), (x + tw + 6, y + 6), color, -1)
    cv2.rectangle(frame, (x - 6, y - th - 10), (x + tw + 6, y + 6), (0, 0, 0), 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, conf: np.ndarray | None = None,
                   color: tuple[int, int, int] = (0, 220, 0), conf_threshold: float = 0.3) -> np.ndarray:
    """Disegna keypoint e connessioni scheletriche su un frame (in place)."""
    def ok(idx: int) -> bool:
        if conf is None:
            return True
        return conf[idx] >= conf_threshold

    for a_name, b_name in SKELETON_EDGES:
        a_idx, b_idx = KP[a_name], KP[b_name]
        if not (ok(a_idx) and ok(b_idx)):
            continue
        a, b = kpts[a_idx], kpts[b_idx]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, 2, cv2.LINE_AA)

    for idx in range(kpts.shape[0]):
        if not ok(idx) or np.isnan(kpts[idx]).any():
            continue
        cv2.circle(frame, tuple(kpts[idx].astype(int)), 3, (0, 165, 255), -1, cv2.LINE_AA)

    return frame


def text_block_size(lines: list[str], font_scale: float = 0.5) -> tuple[int, int]:
    """Dimensioni (larghezza, altezza) in pixel del riquadro che
    `draw_text_block` disegnerebbe per queste righe — usato per impilare
    più riquadri (uno per persona) senza sovrapporli, senza duplicare la
    logica di dimensionamento."""
    line_height = int(22 * font_scale / 0.5)
    box_h = line_height * len(lines) + 10
    box_w = max((len(l) for l in lines), default=0) * int(9 * font_scale / 0.5) + 16
    return box_w, box_h


def draw_text_block(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 10),
                     font_scale: float = 0.5, color: tuple[int, int, int] = (255, 255, 255),
                     border_color: tuple[int, int, int] | None = None) -> np.ndarray:
    """Disegna un blocco di testo multi-riga con sfondo semi-trasparente per
    leggibilità (usato per mostrare le metriche live di ciascuna persona).
    Se `border_color` è specificato, disegna anche un bordo di quel colore
    (tipicamente lo stesso colore dello scheletro della persona, vedi
    `get_track_color`) per collegare visivamente il riquadro alla persona
    corrispondente quando ce n'è più di una nell'inquadratura.
    """
    x, y = origin
    line_height = int(22 * font_scale / 0.5)
    box_w, box_h = text_block_size(lines, font_scale)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if border_color is not None:
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), border_color, 2)

    for i, line in enumerate(lines):
        ty = y + 18 + i * line_height
        cv2.putText(frame, line, (x + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return frame


def draw_hand(frame: np.ndarray, hand_xy: np.ndarray, color: tuple[int, int, int] = (255, 200, 0)) -> np.ndarray:
    """Disegna i 21 landmark di una mano e le relative connessioni.
    Import di HAND_CONNECTIONS ritardato per evitare un import pesante di
    `hands.py` (e quindi di mediapipe) quando non serve.
    """
    from hands import HAND_CONNECTIONS

    for a_idx, b_idx in HAND_CONNECTIONS:
        a, b = hand_xy[a_idx], hand_xy[b_idx]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, 2, cv2.LINE_AA)
    for pt in hand_xy:
        if np.isnan(pt).any():
            continue
        cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 100, 255), -1, cv2.LINE_AA)
    return frame


def draw_face_signals(frame: np.ndarray, mouth_pts: np.ndarray | None = None,
                       left_eye_pts: np.ndarray | None = None,
                       right_eye_pts: np.ndarray | None = None,
                       left_eyebrow_pts: np.ndarray | None = None,
                       right_eyebrow_pts: np.ndarray | None = None,
                       mouth_color: tuple[int, int, int] = (0, 220, 255),
                       eye_color: tuple[int, int, int] = (255, 220, 0),
                       eyebrow_color: tuple[int, int, int] = (0, 140, 255)) -> np.ndarray:
    """Disegna un piccolo overlay per bocca, occhi e sopracciglia (punti +
    linee), a partire dai landmark MediaPipe già estratti in `gaze_head.py`
    (indici MOUTH_TOP/BOTTOM/LEFT/RIGHT, *_EYE_EAR_IDX, *_EYEBROW_IDX).

    `mouth_pts`: array (4, 2) nell'ordine [top, bottom, left, right].
    `left_eye_pts`/`right_eye_pts`: array (6, 2) nell'ordine EAR
    [angolo_sx, sup1, sup2, angolo_dx, inf2, inf1].
    `left_eyebrow_pts`/`right_eyebrow_pts`: array (5, 2), contorno del
    sopracciglio (polilinea aperta, non chiusa come gli occhi).
    Senza questo overlay, bocca/occhi/sopracciglia comparivano solo come
    numero nel riquadro di testo, senza alcun segno disegnato sul volto — a
    differenza di testa (freccia) e mani (scheletro).
    """
    if mouth_pts is not None and not np.isnan(mouth_pts).any():
        top, bottom, left, right = mouth_pts
        cv2.line(frame, tuple(top.astype(int)), tuple(bottom.astype(int)), mouth_color, 2, cv2.LINE_AA)
        cv2.line(frame, tuple(left.astype(int)), tuple(right.astype(int)), mouth_color, 1, cv2.LINE_AA)
        for pt in mouth_pts:
            cv2.circle(frame, tuple(pt.astype(int)), 2, mouth_color, -1, cv2.LINE_AA)

    for eye_pts in (left_eye_pts, right_eye_pts):
        if eye_pts is None or np.isnan(eye_pts).any():
            continue
        pts = eye_pts.astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)

    for brow_pts in (left_eyebrow_pts, right_eyebrow_pts):
        if brow_pts is None or np.isnan(brow_pts).any():
            continue
        pts = brow_pts.astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=False, color=eyebrow_color, thickness=2, lineType=cv2.LINE_AA)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1] - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
