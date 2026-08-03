"""
anonymize.py
============
Offuscamento del volto basato sui keypoint della testa (naso, occhi,
orecchie), da applicare ai frame PRIMA di salvare o condividere qualunque
video/estratto che coinvolga minori.

Approccio: dato che i keypoint della testa sono già disponibili dall'output
del modello di pose estimation, non serve un rilevatore di volti separato:
si stima un raggio a partire dalla distanza tra le spalle (proxy della
scala della persona nell'immagine) e si applica un blur Gaussiano forte
sulla regione corrispondente.

Questo NON sostituisce una policy di gestione dati approvata dal comitato
etico; è una misura tecnica di difesa in profondità da applicare comunque.
"""

from __future__ import annotations

import numpy as np
import cv2

from keypoints import KP, HEAD_KEYPOINTS


def _valid_points(frame_kpts: np.ndarray, conf: np.ndarray | None, names: list[str],
                   conf_threshold: float = 0.3) -> np.ndarray:
    idxs = [KP[n] for n in names]
    pts = frame_kpts[idxs]
    if conf is not None:
        mask = conf[idxs] >= conf_threshold
        pts = pts[mask]
    return pts[~np.isnan(pts).any(axis=1)] if len(pts) else pts


def estimate_head_radius(frame_kpts: np.ndarray) -> float:
    """Stima un raggio ragionevole per il blur del volto usando la distanza
    spalla-spalla come riferimento di scala (più robusta della sola
    distanza tra occhi, che può essere molto piccola o assente di profilo).
    """
    l_sh, r_sh = frame_kpts[KP["left_shoulder"]], frame_kpts[KP["right_shoulder"]]
    shoulder_width = np.linalg.norm(l_sh - r_sh)
    if not np.isfinite(shoulder_width) or shoulder_width < 1e-3:
        return 40.0  # fallback in pixel, da adattare alla risoluzione video
    return float(0.6 * shoulder_width)


def blur_face(frame: np.ndarray, frame_kpts: np.ndarray,
              conf: np.ndarray | None = None) -> np.ndarray:
    """Applica un blur Gaussiano forte sulla regione del volto di UNA
    persona, individuata dai suoi keypoint della testa.

    Parameters
    ----------
    frame : immagine BGR (come restituita da OpenCV)
    frame_kpts : array (17, 2) per la persona corrente
    conf : array (17,) opzionale di confidenze per keypoint

    Returns
    -------
    Il frame con la regione del volto sfocata (copia, non modifica in place
    l'array originale se non esplicitamente voluto).
    """
    head_pts = _valid_points(frame_kpts, conf, HEAD_KEYPOINTS)
    if len(head_pts) == 0:
        return frame

    center = head_pts.mean(axis=0)
    radius = estimate_head_radius(frame_kpts)

    x, y = int(center[0]), int(center[1])
    r = max(int(radius), 10)

    h, w = frame.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)
    if x1 <= x0 or y1 <= y0:
        return frame

    out = frame.copy()
    roi = out[y0:y1, x0:x1]
    # kernel dispari, proporzionale alla regione
    k = max(3, (min(roi.shape[:2]) // 2) | 1)
    out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out
