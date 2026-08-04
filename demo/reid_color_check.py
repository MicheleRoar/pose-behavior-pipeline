"""
reid_color_check.py
====================
Verifica del segnale opzionale di colore maglia/pantaloni aggiunto a
`reid.py` (vedi "Segnale opzionale: colore maglia/pantaloni" nel docstring
del modulo), SENZA fotocamera/YOLO: costruisce frame sintetici (rettangoli
colorati) invece di un video reale.

Tre parti:
  1. Verifica unitaria di `compute_color_signature` (il colore campionato
     dal poligono spalle/anche/ginocchia corrisponde al colore disegnato).
  2. Verifica unitaria di `color_similarity` (identico -> 1.0, tonalita'
     opposta -> bassa).
  3. Verifica di valore: uno scenario di rientro con proporzioni corporee
     DISTORTE (keypoint rumorosi, come puo' capitare ai bordi
     dell'inquadratura) che:
       - FALLISCE con la sola firma antropometrica (frame=None) -- il
         problema concreto segnalato ("non funziona molto bene");
       - RIESCE quando si passa anche il frame con lo stesso colore di
         vestiti (frame=...) -- la distanza si "sconta" abbastanza da
         rientrare sotto soglia;
       - un cambio di vestiti vero (colore diverso) NON impedisce un
         match altrimenti valido sulle sole proporzioni -- il colore aiuta,
         non sostituisce ne' blocca la firma antropometrica.

Esegui con: python reid_color_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from keypoints import KP
from reid import (
    ReIdentifier, COLOR_SEGMENTS, compute_color_signature, color_similarity,
    hair_corners, MAX_POSITION_GAP_SECONDS,
)
from reid_check import make_skeleton, PERSON_A

FPS = 30.0
CONF = np.ones(17)
CANVAS = (400, 700, 3)  # (h, w, 3)


def draw_person_patches(kxy: np.ndarray, shirt_bgr: tuple, pants_bgr: tuple,
                         hair_bgr: tuple | None = None) -> np.ndarray:
    """Frame sintetico (sfondo grigio) con maglia/pantaloni/capelli colorati
    nella stessa regione campionata da `compute_color_signature`."""
    frame = np.full(CANVAS, 128, dtype=np.uint8)
    for region, color in (("shirt", shirt_bgr), ("pants", pants_bgr)):
        corners = COLOR_SEGMENTS[region]
        pts = np.round(kxy[[KP[c] for c in corners]]).astype(np.int32)
        cv2.fillPoly(frame, [pts], color)
    if hair_bgr is not None:
        pts = np.round(hair_corners(kxy)).astype(np.int32)
        cv2.fillPoly(frame, [pts], hair_bgr)
    return frame


def expected_hs(bgr: tuple) -> tuple[float, float]:
    patch = np.uint8([[bgr]])
    h, s, _ = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0]
    return float(h) / 180.0, float(s) / 255.0


RED = (0, 0, 220)      # BGR
BLUE = (200, 30, 0)
GREEN = (0, 180, 0)
BROWN = (19, 69, 139)  # proxy colore capelli


def part1_signature_matches_drawn_color():
    kxy = make_skeleton(**PERSON_A, tx=300, ty=150, jitter=0.0, rng=None)
    frame = draw_person_patches(kxy, RED, BLUE, hair_bgr=BROWN)
    sig = compute_color_signature(frame, kxy)

    exp_shirt = expected_hs(RED)
    exp_pants = expected_hs(BLUE)
    exp_hair = expected_hs(BROWN)
    assert np.allclose(sig[:2], exp_shirt, atol=0.02), f"shirt {sig[:2]} vs atteso {exp_shirt}"
    assert np.allclose(sig[2:4], exp_pants, atol=0.02), f"pants {sig[2:4]} vs atteso {exp_pants}"
    assert np.allclose(sig[4:], exp_hair, atol=0.02), f"hair {sig[4:]} vs atteso {exp_hair}"
    print(f"Parte 1: colore campionato maglia={sig[:2]} (atteso {exp_shirt}), "
          f"pantaloni={sig[2:4]} (atteso {exp_pants}), capelli={sig[4:]} (atteso {exp_hair}) — OK")


def part2_color_similarity_sane():
    a = np.array([*expected_hs(RED), *expected_hs(BLUE), *expected_hs(BROWN)])
    b = np.array([*expected_hs(RED), *expected_hs(BLUE), *expected_hs(BROWN)])
    same = color_similarity(a, b)
    assert same is not None and same > 0.98, f"colore identico dovrebbe dare similarita' ~1, trovato {same}"

    c = np.array([*expected_hs(GREEN), *expected_hs(BLUE), *expected_hs(BROWN)])
    diff = color_similarity(a, c)
    assert diff is not None and diff < same, "maglia rossa vs verde deve avere similarita' minore di rossa vs rossa"
    print(f"Parte 2: similarita' stesso colore={same:.3f}, maglia rossa-vs-verde={diff:.3f} — OK")


def _run_reentry_scenario(*, reentry_scale: float, reentry_shirt: tuple, reentry_pants: tuple,
                           use_color: bool) -> tuple[int | None, int]:
    """Persona presente 20 frame (proporzioni normali, vestiti rossi/blu),
    esce abbastanza a lungo da azzerare anche il segnale posizionale
    (oltre `MAX_POSITION_GAP_SECONDS`, per isolare questo test sul solo
    contributo di proporzioni+colore), poi rientra con un nuovo raw
    track_id per 15 frame con le proporzioni/vestiti indicati. Ritorna
    (person_id_originale, person_id_al_rientro) — se il reid non scatta,
    il secondo sara' un id nuovo (diverso dal primo).
    """
    reid = ReIdentifier(max_lost_seconds=60.0, max_signature_dist=0.12,
                         min_signature_frames=15, color_bonus_weight=0.5)
    rng = np.random.default_rng(1)
    frame_t = 0
    person_id_initial = None

    for _ in range(20):
        now = frame_t / FPS
        kxy = make_skeleton(**PERSON_A, tx=300, ty=150, jitter=0.01, rng=rng)
        frame = draw_person_patches(kxy, RED, BLUE, hair_bgr=BROWN) if use_color else None
        resolved = reid.resolve([(1, kxy, CONF)], now, frame=frame)
        person_id_initial = resolved[0][0]
        frame_t += 1

    absence_frames = int(FPS * (MAX_POSITION_GAP_SECONDS + 5))
    for _ in range(absence_frames):
        # resolve() con lista vuota ad ogni frame, non solo un incremento di
        # frame_t: serve perche' il "lost_time" viene registrato quando il
        # sistema SI ACCORGE dell'assenza (prossima resolve() senza quel
        # raw_id), non quando l'assenza inizia -- senza queste chiamate
        # lost_time coinciderebbe con il rientro, azzerando artificialmente
        # il gap che vogliamo simulare.
        reid.resolve([], frame_t / FPS)
        frame_t += 1  # assente dall'inquadratura (abbastanza a lungo da azzerare il bonus posizionale)

    distorted = dict(PERSON_A)
    distorted["shoulder_w"] *= reentry_scale
    distorted["hip_w"] /= reentry_scale
    distorted["upper_arm"] *= reentry_scale
    distorted["thigh"] /= reentry_scale
    rng2 = np.random.default_rng(2)
    person_id_reentry = None
    for _ in range(15):
        now = frame_t / FPS
        kxy = make_skeleton(**distorted, tx=300, ty=150, jitter=0.01, rng=rng2)
        frame = draw_person_patches(kxy, reentry_shirt, reentry_pants, hair_bgr=BROWN) if use_color else None
        resolved = reid.resolve([(2, kxy, CONF)], now, frame=frame)
        person_id_reentry = resolved[0][0]
        frame_t += 1

    return person_id_initial, person_id_reentry


def part3_color_recovers_noisy_reentry():
    # (a) keypoint rumorosi al rientro, SENZA colore: deve FALLIRE
    initial, reentry = _run_reentry_scenario(
        reentry_scale=1.6, reentry_shirt=RED, reentry_pants=BLUE, use_color=False)
    assert reentry != initial, (
        "atteso un fallimento del reid basato solo su proporzioni rumorose "
        f"(person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Parte 3a (senza colore, proporzioni rumorose): person_id iniziale={initial}, "
          f"al rientro={reentry} -> NON ri-associato (atteso, dimostra il problema)")

    # (b) stessi keypoint rumorosi, CON colore (stessi vestiti): deve RIUSCIRE
    initial, reentry = _run_reentry_scenario(
        reentry_scale=1.6, reentry_shirt=RED, reentry_pants=BLUE, use_color=True)
    assert reentry == initial, (
        f"atteso match grazie al colore (person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Parte 3b (con colore, stessi vestiti): person_id iniziale={initial}, "
          f"al rientro={reentry} -> ri-associato correttamente")

    # (c) vestiti CAMBIATI ma proporzioni corrette (non distorte): deve
    # comunque riuscire solo sulle proporzioni -- il colore diverso non
    # deve MAI bloccare un match altrimenti valido.
    initial, reentry = _run_reentry_scenario(
        reentry_scale=1.0, reentry_shirt=GREEN, reentry_pants=GREEN, use_color=True)
    assert reentry == initial, (
        f"un cambio di vestiti non deve impedire il match sulle proporzioni "
        f"(person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Parte 3c (vestiti cambiati, proporzioni corrette): person_id iniziale={initial}, "
          f"al rientro={reentry} -> ri-associato comunque (colore non ha bloccato il match)")


def main():
    part1_signature_matches_drawn_color()
    part2_color_similarity_sane()
    part3_color_recovers_noisy_reentry()
    print("\nVerifica completata senza errori: il segnale di colore aiuta a "
          "recuperare rientri con proporzioni rumorose quando i vestiti "
          "restano gli stessi, senza compromettere l'invarianza al "
          "vestiario quando i vestiti cambiano davvero.")


if __name__ == "__main__":
    main()
