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
from pose.keypoints import KP
from pose.reid import (
    ReIdentifier, SIGNATURE_COLS, compute_signature_frame,
    MAX_POSITION_DIST_TORSOS,
)

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


def head_segments_computed_correctly():
    """Verifica unitaria dei due segmenti-testa (eye_to_eye/ear_to_ear)
    aggiunti alla firma: `make_skeleton` posiziona gli occhi a +-3 e le
    orecchie a +-6 dal centro, quindi ci si attende 6/torso e 12/torso."""
    kxy = make_skeleton(**PERSON_A, torso=100.0, tx=0, ty=0, jitter=0.0, rng=None)
    sig = compute_signature_frame(kxy)
    eye_val = sig[SIGNATURE_COLS.index("eye_to_eye")]
    ear_val = sig[SIGNATURE_COLS.index("ear_to_ear")]
    assert np.isclose(eye_val, 6.0 / 100.0, atol=1e-6), f"eye_to_eye atteso 0.06, trovato {eye_val}"
    assert np.isclose(ear_val, 12.0 / 100.0, atol=1e-6), f"ear_to_ear atteso 0.12, trovato {ear_val}"
    print(f"Segmenti testa: eye_to_eye={eye_val:.3f}, ear_to_ear={ear_val:.3f} — OK")


def noisy_reentry_recovers_via_retry():
    """Verifica che il match NON venga tentato una volta sola: un rientro
    con i primi frame rumorosi (persona ancora ai bordi dell'inquadratura,
    proporzioni distorte) non deve restare "perso per sempre" -- una volta
    che la finestra scorrevole si ripulisce con frame corretti, il match
    deve comunque scattare, senza bisogno di un nuovo track_id."""
    reid = ReIdentifier(max_lost_seconds=30.0, max_signature_dist=0.12, min_signature_frames=15)
    rng = np.random.default_rng(42)
    frame_t = 0

    for _ in range(50):
        now = frame_t / FPS
        kxy = make_skeleton(**PERSON_A, tx=0, ty=0, jitter=0.01, rng=rng)
        resolved = reid.resolve([(1, kxy, CONF)], now)
        person_id_initial = resolved[0][0]
        frame_t += 1

    for _ in range(100):
        frame_t += 1  # A fuori dall'inquadratura

    distorted = dict(PERSON_A)
    distorted["shoulder_w"] *= 1.45
    distorted["hip_w"] /= 1.45
    distorted["upper_arm"] *= 1.45
    distorted["thigh"] /= 1.45

    matched_at = None
    for i in range(30):
        now = frame_t / FPS
        # primi 10 frame del rientro: proporzioni rumorose (persona ancora
        # ai bordi dell'inquadratura); dal frame 11 in poi: pulite.
        params = distorted if i < 10 else PERSON_A
        kxy = make_skeleton(**params, tx=0, ty=0, jitter=0.01, rng=rng)
        resolved = reid.resolve([(5, kxy, CONF)], now)
        if matched_at is None and resolved[0][0] == person_id_initial:
            matched_at = i
        frame_t += 1

    assert matched_at is not None, "il rientro non e' mai stato ri-associato, nemmeno dopo che i frame si sono ripuliti"
    assert matched_at >= 10, (
        f"il match e' scattato al frame {matched_at}, prima che i dati rumorosi (frame 0-9) "
        "potessero uscire dalla finestra -- suggerisce che la soglia sia troppo permissiva "
        "per essere un test valido, non che il retry funzioni davvero"
    )
    print(f"Rientro rumoroso poi pulito: ri-associato al frame relativo {matched_at} "
          "(il tentativo iniziale a 15 frame, con dati ancora rumorosi, fallisce; "
          "il retry sui frame successivi, piu' puliti, recupera il match) — OK")


def _run_in_place_scenario(*, reentry_tx: float, absence_seconds: float) -> tuple[int, int]:
    """Persona presente 20 frame a tx=300 (proporzioni normali), assente per
    `absence_seconds` (simulate chiamando resolve([], now) ad ogni frame,
    cosi' lost_time riflette il momento vero della scomparsa, non quello
    del rientro), poi rientra con proporzioni distorte (troppo diverse per
    un match sulla sola firma) a `reentry_tx`. Nessun colore passato: se il
    rientro viene comunque recuperato, e' merito solo della posizione."""
    reid = ReIdentifier(max_lost_seconds=60.0, max_signature_dist=0.12, min_signature_frames=15)
    rng = np.random.default_rng(7)
    frame_t = 0
    person_id_initial = None

    for _ in range(20):
        now = frame_t / FPS
        kxy = make_skeleton(**PERSON_A, tx=300, ty=150, jitter=0.01, rng=rng)
        resolved = reid.resolve([(1, kxy, CONF)], now)
        person_id_initial = resolved[0][0]
        frame_t += 1

    for _ in range(int(FPS * absence_seconds)):
        reid.resolve([], frame_t / FPS)
        frame_t += 1

    distorted = dict(PERSON_A)
    distorted["shoulder_w"] *= 1.6
    distorted["hip_w"] /= 1.6
    distorted["upper_arm"] *= 1.6
    distorted["thigh"] /= 1.6

    rng2 = np.random.default_rng(8)
    person_id_reentry = None
    for _ in range(15):
        now = frame_t / FPS
        kxy = make_skeleton(**distorted, tx=reentry_tx, ty=150, jitter=0.01, rng=rng2)
        resolved = reid.resolve([(2, kxy, CONF)], now)
        person_id_reentry = resolved[0][0]
        frame_t += 1

    return person_id_initial, person_id_reentry


def position_signal_recovers_in_place_reentry():
    """Scenario "cambio giacca": il bambino non esce dall'inquadratura ma
    resta occluso ~10s (es. mentre lo vestono), poi ricompare NELLO STESSO
    PUNTO con proporzioni troppo distorte per un match sulla sola firma
    (nessun colore passato). Deve comunque essere ri-associato grazie alla
    sola posizione. Controllo negativo: stesse proporzioni distorte ma
    rientro LONTANO dall'ultima posizione nota -- non deve scattare, a
    riprova che la posizione aiuta solo quando e' davvero vicina, non
    forza un match a prescindere."""
    initial, reentry = _run_in_place_scenario(reentry_tx=300, absence_seconds=10.0)
    assert reentry == initial, (
        f"atteso un match grazie alla sola posizione (persona non spostata, ~10s di assenza) "
        f"(person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Rientro sul posto dopo occlusione (~10s, es. cambio giacca), senza colore: "
          f"person_id iniziale={initial}, al rientro={reentry} -> ri-associato grazie alla posizione")

    far_tx = 300 + (MAX_POSITION_DIST_TORSOS + 2) * 100  # ben oltre il raggio di 4 lunghezze di busto (torso=100)
    initial, reentry = _run_in_place_scenario(reentry_tx=far_tx, absence_seconds=10.0)
    assert reentry != initial, (
        "controllo negativo fallito: un rientro lontano con proporzioni distorte non deve "
        f"essere ri-associato solo perche' il tempo di assenza e' breve "
        f"(person_id_initial={initial}, person_id_reentry={reentry})"
    )
    print(f"Controllo negativo (stesso gap, ma rientro lontano dall'ultima posizione nota): "
          f"person_id iniziale={initial}, al rientro={reentry} -> NON ri-associato (atteso, "
          "la posizione non forza mai un match")


def max_people_forces_capacity_reentry():
    """Sessione 1v1 (max_people=2): A e B presenti insieme (roster al
    tetto), A esce e rientra con proporzioni troppo distorte E lontano
    dall'ultima posizione nota -- nessun segnale normale (firma, posizione,
    colore non passato) porterebbe a un match. Con max_people=2 e B ancora
    attivo (unico candidato "perso" e' A), il rientro deve comunque essere
    forzato su A: non puo' trattarsi di una terza persona per definizione.
    Vedi il controllo negativo in position_signal_recovers_in_place_reentry
    per lo stesso scenario SENZA max_people, dove il match non scatta."""
    reid = ReIdentifier(max_lost_seconds=60.0, max_signature_dist=0.12,
                         min_signature_frames=15, max_people=2)
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(12)
    frame_t = 0

    # A e B presenti insieme: il roster raggiunge il tetto di 2.
    by_raw = {}
    for _ in range(20):
        now = frame_t / FPS
        people = [
            (1, make_skeleton(**PERSON_A, tx=0, ty=0, jitter=0.01, rng=rng_a), CONF),
            (2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF),
        ]
        resolved = reid.resolve(people, now)
        by_raw = {1: resolved[0][0], 2: resolved[1][0]}
        frame_t += 1
    person_id_a_initial, person_id_b = by_raw[1], by_raw[2]

    # A esce dall'inquadratura, B resta sola.
    for _ in range(int(FPS * 5)):
        now = frame_t / FPS
        reid.resolve([(2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF)], now)
        frame_t += 1

    # A rientra: proporzioni troppo distorte per la firma, posizione troppo
    # lontana per il segnale posizionale, nessun colore passato -- nessun
    # segnale "onesto" porterebbe a un match.
    distorted = dict(PERSON_A)
    distorted["shoulder_w"] *= 1.6
    distorted["hip_w"] /= 1.6
    distorted["upper_arm"] *= 1.6
    distorted["thigh"] /= 1.6
    far_tx = (MAX_POSITION_DIST_TORSOS + 2) * 100

    rng_a2 = np.random.default_rng(13)
    person_id_a_reentry = None
    for _ in range(int(FPS * 3)):  # oltre _PENDING_RETRY_SECONDS, cosi' scatta il fallback forzato
        now = frame_t / FPS
        people = [
            (2, make_skeleton(**PERSON_B, tx=200, ty=0, jitter=0.01, rng=rng_b), CONF),
            (3, make_skeleton(**distorted, tx=far_tx, ty=0, jitter=0.01, rng=rng_a2), CONF),
        ]
        resolved = reid.resolve(people, now)
        by_raw = {raw_id: pid for (raw_id, *_rest), (pid, *_rest2) in zip(people, resolved)}
        person_id_a_reentry = by_raw[3]
        frame_t += 1

    assert person_id_a_reentry == person_id_a_initial, (
        f"con max_people=2 e roster al tetto, il rientro doveva essere forzato su A "
        f"(person_id_a_initial={person_id_a_initial}, person_id_a_reentry={person_id_a_reentry})"
    )
    forced_events = [e for e in reid.merge_log if e.forced]
    assert len(forced_events) == 1, f"atteso esattamente 1 evento forzato, trovati {len(forced_events)}"
    print(f"max_people=2, rientro senza segnali normali disponibili: "
          f"person_id iniziale={person_id_a_initial}, al rientro={person_id_a_reentry} "
          "-> ri-associato FORZATO (roster al tetto, unico candidato perso)")


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

    head_segments_computed_correctly()
    noisy_reentry_recovers_via_retry()
    position_signal_recovers_in_place_reentry()
    max_people_forces_capacity_reentry()

    print("\nVerifica completata senza errori: re-identificazione in tempo reale "
          "funziona su un'uscita/rientro simulata, senza confondere una persona estranea.")


if __name__ == "__main__":
    main()
