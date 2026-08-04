"""
tracking_common.py
===================
Utility condivisa tra pose_estimation.py, seg_estimation.py e
track_stability_check.py: il cap "tieni solo le N detection piu' sicure di
questo frame" (vedi `max_people` in pose_estimation.py per il razionale
completo) era duplicato in piu' punti -- centralizzato qui cosi' una
correzione futura non deve essere ripetuta ovunque.
"""

from __future__ import annotations

import numpy as np


def cap_by_confidence(box_conf: np.ndarray, max_people: int | None):
    """Indici da tenere in questo frame: tutti (come `range(n)`), se
    `max_people` non e' impostato o il numero di detection e' gia' entro il
    limite; altrimenti solo gli indici delle `max_people` detection con
    confidenza piu' alta. Non scarta mai nulla quando si e' sotto il
    limite -- il filtro scatta solo in eccesso, non abbassa mai il numero
    di persone tenute sotto il tetto configurato."""
    n = len(box_conf)
    if max_people is None or n <= max_people:
        return range(n)
    return np.argsort(-box_conf)[:max_people]
