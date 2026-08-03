"""
reid_check.py
=============
Verifica della logica di `reid.py` (re-identificazione in tempo reale via
firma antropometrica) SENZA fotocamera/YOLO, simulando frame-by-frame una
sessione con: due persone di corporatura diversa presenti insieme, una
delle due che esce dall'inquadratura e rientra dopo un vuoto temporale con
un NUOVO track_id (come farebbe ByteTrack), e una terza persona "estranea"
comparsa nello stesso periodo, per verificare che non venga confusa con
nessuna delle prime due.

Esegui con: python reid_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from keypoints import KP
from reid import ReIdentifier

N_JOINTS = 17  # schema COCO-17


def make_skeleton(shoulder_w: float, hip_w: float, upper_arm: float, forearm: float,
                   thigh: float, shin: float, torso: float = 100.0,
                   tx: float = 0.0, ty: float = 0.0, jitter: float = 0.0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Scheletro COCO-17 sintetico coerente con le proporzioni date (come
    frazione della lunghezza del busto), centrato su (tx, ty). A differenza
    di `demo/synth_data.py` (che scala TUTTI i giunti uniformemente, quindi
    produce sempre la stessa firma normalizzata), qui ogni segmento ha un
    rapporto indipendente — necessario per simulare persone diverse.
    """
    j = lambda v: (v + rng.normal(0, jitter * torso)) if (rng is not None and jitter) else v
    kxy = np.zeros((N_JOINTS, 2))

    def set_pt(name, x, y):
        kxy[KP[name]] = [j(x), j(y)]

    # centro-spalle a (tx, ty), centro-anche a (tx, ty+torso): stessa
    # convenzione di features.torso_length (shoulder_center -> hip_center)
    set_pt("nose", tx, ty - 0.25 * torso)
    set_pt("left_eye", tx - 3, ty - 0.27 * torso)
    set_pt("right_eye", tx + 3, ty - 0.27 * torso)
    set_pt("left_ear", tx - 6, ty - 0.26 * torso)
    set_pt("right_ear", tx + 6, ty - 0.26 * torso)

    set_pt("left_shoulder", tx - shoulder_w * torso / 2, ty)
    set_pt("right_shoulder", tx + shoulder_w * torso / 2, ty)
    set_pt("left_hip", tx - hip_w * torso / 2, ty + torso)
    set_pt("right_hip", tx + hip_w * torso / 2, ty + torso)

    set_pt("left_elbow", tx - shoulder_w * torso / 2, ty + upper_arm * torso)
    set_pt("left_wrist", tx - shoulder_w * torso / 2, ty + (upper_arm + forearm) * torso)
    set_pt("right_elbow", tx + shoulder_w * torso / 2, ty + upper_arm * torso)
    set_pt("right_wrist", tx + shoulder_w * torso / 2, ty + (upper_arm + forearm) * torso)

    set_pt("left_knee", tx - hip_w * torso / 2, ty + torso + thigh * torso)
    set_pt("left_ankle", tx - hip_w * torso / 2, ty + torso + (thigh + shin) * torso)
    set_pt("right_knee", tx + hip_w * torso / 2, ty + torso + thigh * torso)
    set_pt("right_ankle", tx + hip_w * torso / 2, ty + torso + (thigh + shin) * torso)

    return kxy


PERSON_A = dict(shoulder_w=0.50, hip_w=0.40, upper_arm=0.35, forearm=0.30, thigh=0.45, shin=0.42)
PERSON_B = dict(shoulder_w=0.35, hip_w=0.32, upper_arm=0.28, forearm=0.25, thigh=0.38, shin=0.35)
PERSON_C = dict(shoulder_w=0.60, hip_w=0.52, upper_arm=0.42, forearm=0.38, thigh=0.50, shin=0.48)

FPS = 30.0
CONF = np.ones(N_JOINTS)  # confidenza piena su tutti i giunti, non e' oggetto di questo test


def main():
    reid = ReIdentifier(max_lost_seconds=30.0, max_signature_dist=0.12, min_signature_frames=15)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    rng_c = np.random.default_rng(3)

    person_id_a_initial = None
    person_id_b = None
    person_id_a_reentry = None
    person_id_c = None

    frame = 0

    # --- Fase 1: A e B presenti insieme, frame 0-49 (raw track 1=A, 2=B) ---
    for i in range(50):
        now = frame / FPS
        people_raw = [
            (1, make_skeleton(**PERSON_A, tx=0, ty=0, jitter=0.01, rng=rng_a), CONF),
            (2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF),
        ]
        resolved = reid.resolve(people_raw, now)
        by_raw = {1: resolved[0][0], 2: resolved[1][0]}
        frame += 1

    person_id_a_initial = by_raw[1]
    person_id_b = by_raw[2]
    print(f"Fase 1: A -> person_id={person_id_a_initial}, B -> person_id={person_id_b}")
    assert person_id_a_initial != person_id_b

    # --- Fase 2: A esce dall'inquadratura, B resta sola, frame 50-149 (100 frame = ~3.3s) ---
    for i in range(100):
        now = frame / FPS
        people_raw = [(2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF)]
        resolved = reid.resolve(people_raw, now)
        assert resolved[0][0] == person_id_b, "B non deve mai cambiare identita'"
        frame += 1

    print(f"Fase 2: A e' scomparsa (in memoria 'lost'), B continua con person_id={person_id_b}")
    assert person_id_a_initial in {lost_id for lost_id in reid.lost}, "A deve risultare tra le persone 'perse'"

    # --- Fase 3: A rientra con un NUOVO raw track_id (3), simula clothing
    # change/nuova traccia SAM/ByteTrack; C (estranea) appare in parallelo
    # con un track_id diverso (4) per verificare l'assenza di falsi match ---
    for i in range(30):
        now = frame / FPS
        people_raw = [
            (2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF),
            (3, make_skeleton(**PERSON_A, tx=50, ty=0, jitter=0.01, rng=rng_a), CONF),   # A rientra
            (4, make_skeleton(**PERSON_C, tx=350, ty=0, jitter=0.01, rng=rng_c), CONF),  # estranea
        ]
        resolved = reid.resolve(people_raw, now)
        by_raw = {raw_id: pid for (raw_id, kxy, kconf), (pid, _, _) in
                  zip(people_raw, resolved)}
        frame += 1

    person_id_a_reentry = by_raw[3]
    person_id_c = by_raw[4]
    print(f"Fase 3: A rientrata (raw track 3) -> person_id={person_id_a_reentry}, "
          f"C (estranea, raw track 4) -> person_id={person_id_c}")

    assert person_id_a_reentry == person_id_a_initial, (
        f"A deve essere ri-associata alla sua identita' originale "
        f"({person_id_a_initial}), invece ha person_id={person_id_a_reentry}"
    )
    assert person_id_c not in {person_id_a_initial, person_id_b}, (
        "C (persona estranea) non deve essere confusa con A o B"
    )
    assert person_id_b == by_raw[2], "B deve restare la stessa persona per tutta la sessione"

    assert len(reid.merge_log) == 1, f"atteso esattamente 1 evento di merge, trovati {len(reid.merge_log)}"
    event = reid.merge_log[0]
    print(f"Evento di merge registrato: raw_track={event.raw_track_id}, "
          f"provvisorio={event.provisional_person_id} -> "
          f"ripristinato={event.matched_person_id}, distanza={event.distance:.3f}")

    print("\nVerifica completata senza errori: re-identificazione in tempo reale "
          "funziona su un'uscita/rientro simulata, senza confondere una persona estranea.")


if __name__ == "__main__":
    main()
