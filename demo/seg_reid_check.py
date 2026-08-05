"""
seg_reid_check.py
==================
Verifica della logica di `seg_reid.py` (re-identificazione per la pipeline
di sola segmentazione, tetto rigido su max_people) SENZA fotocamera/YOLO,
simulando frame-by-frame poligoni maschera sintetici (quadrati/rettangoli
posizionati a mano, non vere sagome).

Esegui con: python seg_reid_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from segmentation.seg_reid import SegReIdentifier

FPS = 15.0


def make_person(cx: float, cy: float, w: float = 60.0, h: float = 140.0,
                 jitter: float = 0.0, rng: np.random.Generator | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Bbox + poligono rettangolare sintetico centrato su (cx, cy), con
    jitter opzionale sui vertici (per simulare rumore di segmentazione)."""
    j = lambda v: v + (rng.normal(0, jitter) if (rng is not None and jitter) else 0.0)
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    bbox = np.array([x1, y1, x2, y2])
    poly = np.array([[j(x1), j(y1)], [j(x2), j(y1)], [j(x2), j(y2)], [j(x1), j(y2)]])
    return bbox, poly


def hard_cap_never_exceeded_under_heavy_churn():
    """Sessione 1v1 (max_people=2): dopo il warm-up, simula MOLTI raw
    track_id nuovi che compaiono/scompaiono in rapida successione (churn
    pesante, come nel caso reale) vicino alle due posizioni note. Verifica
    che il numero di person_id distinti non superi MAI 2, qualunque sia il
    numero di raw track_id generati."""
    reid = SegReIdentifier(max_people=2)
    rng = np.random.default_rng(1)
    frame_t = 0
    next_raw_id = 1

    # warm-up: A e B, due posizioni ben distinte
    a_bbox, a_poly = make_person(100, 300)
    b_bbox, b_poly = make_person(500, 300)
    resolved = reid.resolve([(1, a_bbox, a_poly, 0.9), (2, b_bbox, b_poly, 0.9)], frame_t / FPS)
    person_ids_seen = {pid for pid, *_ in resolved}
    assert len(person_ids_seen) == 2, f"attese 2 identita' distinte dopo il warm-up, trovate {person_ids_seen}"
    next_raw_id = 3
    frame_t += 1

    # 60 frame di churn pesante: ogni frame, A e B ricompaiono ENTRAMBI con
    # un raw_id nuovo (come se ByteTrack perdesse e ricreasse il track ad
    # ogni frame) vicino alla loro posizione, con un po' di jitter.
    all_person_ids: set[int] = set(person_ids_seen)
    for _ in range(60):
        now = frame_t / FPS
        a_bbox, a_poly = make_person(100 + rng.normal(0, 5), 300 + rng.normal(0, 5), jitter=2.0, rng=rng)
        b_bbox, b_poly = make_person(500 + rng.normal(0, 5), 300 + rng.normal(0, 5), jitter=2.0, rng=rng)
        people = [(next_raw_id, a_bbox, a_poly, 0.9), (next_raw_id + 1, b_bbox, b_poly, 0.9)]
        resolved = reid.resolve(people, now)
        all_person_ids.update(pid for pid, *_ in resolved)
        next_raw_id += 2
        frame_t += 1

    assert len(all_person_ids) <= 2, (
        f"il tetto max_people=2 e' stato superato: {len(all_person_ids)} person_id distinti "
        f"({sorted(all_person_ids)}) nonostante {next_raw_id - 1} raw track_id generati"
    )
    print(f"Churn pesante (raw track_id generati: {next_raw_id - 1}) -> "
          f"person_id distinti: {sorted(all_person_ids)} (tetto rispettato) — OK")


def churned_track_relinks_to_nearest_position():
    """A esce (il suo raw track sparisce), ricompare con un NUOVO raw_id
    vicino alla stessa posizione mentre B mantiene un raw_id stabile senza
    interruzioni. Il nuovo raw_id di A deve legarsi al person_id originale
    di A (posizione vicina), non a quello di B (che nello stesso frame e'
    gia' rivendicato da un raw_id diverso e comunque e' lontano)."""
    reid = SegReIdentifier(max_people=2)
    frame_t = 0

    a_bbox, a_poly = make_person(100, 300)
    b_bbox, b_poly = make_person(500, 300)
    resolved = reid.resolve([(1, a_bbox, a_poly, 0.9), (2, b_bbox, b_poly, 0.9)], frame_t / FPS)
    by_raw = {1: resolved[0][0], 2: resolved[1][0]}
    person_id_a, person_id_b = by_raw[1], by_raw[2]
    frame_t += 1

    # A sparisce per qualche frame, B resta con lo stesso raw_id
    for _ in range(5):
        now = frame_t / FPS
        b_bbox, b_poly = make_person(500, 300)
        reid.resolve([(2, b_bbox, b_poly, 0.9)], now)
        frame_t += 1

    # A rientra con un NUOVO raw_id (3), vicino alla vecchia posizione; B
    # continua con lo stesso raw_id 2, nello stesso frame
    now = frame_t / FPS
    a_bbox, a_poly = make_person(110, 305)  # leggermente spostata, come un rientro reale
    b_bbox, b_poly = make_person(500, 300)
    resolved = reid.resolve([(2, b_bbox, b_poly, 0.9), (3, a_bbox, a_poly, 0.9)], now)
    by_raw = {2: resolved[0][0], 3: resolved[1][0]}

    assert by_raw[3] == person_id_a, (
        f"il nuovo raw_id 3 (vicino alla posizione di A) doveva legarsi a person_id={person_id_a}, "
        f"invece ha person_id={by_raw[3]}"
    )
    assert by_raw[2] == person_id_b, "B non deve mai cambiare identita'"
    print(f"Rientro con nuovo raw_id vicino alla posizione nota: "
          f"person_id iniziale A={person_id_a}, dopo il rientro={by_raw[3]} -> ri-associato — OK")


def soft_match_below_cap_relinks_instead_of_minting_new():
    """Il bug segnalato su footage reale: con `max_people` impostato largo
    apposta per tenersi margine (es. 5 per un gruppo con al massimo 2-3
    bambini attesi), una persona che sparisce brevemente (occlusione, uscita
    dal bordo inquadratura) e rientra con un nuovo raw_id DEVE ricollegarsi
    alla sua identita' originale anche se il tetto max_people e' ancora
    ben lontano dall'essere raggiunto -- prima della fix, sotto al tetto non
    veniva tentato alcun confronto e ogni sparizione apriva sempre un id
    nuovo (scambio temporaneo visibile anche a re-id attiva)."""
    reid = SegReIdentifier(max_people=5)  # tetto largo, headcount reale = 2
    frame_t = 0

    a_bbox, a_poly = make_person(100, 300)
    b_bbox, b_poly = make_person(500, 300)
    resolved = reid.resolve([(1, a_bbox, a_poly, 0.9), (2, b_bbox, b_poly, 0.9)], frame_t / FPS)
    by_raw = {1: resolved[0][0], 2: resolved[1][0]}
    person_id_a, person_id_b = by_raw[1], by_raw[2]
    assert len(reid.persons) == 2, "dopo il warm-up devono esserci solo 2 identita', non 5"
    frame_t += 1

    # A sparisce per qualche frame (B resta con lo stesso raw_id)
    for _ in range(5):
        now = frame_t / FPS
        b_bbox, b_poly = make_person(500, 300)
        reid.resolve([(2, b_bbox, b_poly, 0.9)], now)
        frame_t += 1

    # A rientra con un NUOVO raw_id (3), vicino alla vecchia posizione --
    # il tetto (5) e' ancora ben lontano dall'essere raggiunto (solo 2
    # identita' esistono finora).
    now = frame_t / FPS
    a_bbox, a_poly = make_person(110, 305)
    resolved = reid.resolve([(3, a_bbox, a_poly, 0.9)], now)
    person_id_a_reentry = resolved[0][0]

    assert person_id_a_reentry == person_id_a, (
        f"con tetto=5 ancora lontano (2 identita' esistenti), il rientro di A doveva ricollegarsi "
        f"a person_id={person_id_a} tramite aggancio morbido, invece ha aperto person_id="
        f"{person_id_a_reentry} (bug: sotto al tetto non veniva tentato alcun confronto)"
    )
    assert len(reid.persons) == 2, (
        f"nessuna identita' in piu' doveva essere aperta (aggancio morbido riuscito): "
        f"trovate {len(reid.persons)} identita' invece di 2"
    )
    print(f"Tetto largo (max_people=5, headcount reale=2): il rientro di A sotto al tetto "
          f"si ricollega a person_id={person_id_a} invece di aprirne uno nuovo — OK")


def single_person_session_always_id_one():
    """max_people=1: qualunque numero di cambi di raw track_id, il
    person_id deve restare sempre lo stesso -- nessun confronto necessario,
    per definizione c'e' una sola persona possibile."""
    reid = SegReIdentifier(max_people=1)
    frame_t = 0
    seen_person_ids = set()
    for raw_id in range(1, 21):  # 20 raw track_id diversi in sequenza
        now = frame_t / FPS
        bbox, poly = make_person(200 + raw_id * 3, 300)  # si sposta leggermente ogni volta
        resolved = reid.resolve([(raw_id, bbox, poly, 0.9)], now)
        seen_person_ids.add(resolved[0][0])
        frame_t += 1

    assert seen_person_ids == {1}, f"con max_people=1 ci si aspetta un solo person_id, trovati {seen_person_ids}"
    print(f"Sessione a 1 persona, 20 raw track_id diversi -> person_id sempre {seen_person_ids} — OK")


def pathological_same_frame_overflow_does_not_crash():
    """Caso patologico: max_people=2, ma in un frame compaiono 3 raw_id
    "nuovi" insieme (es. detection doppia spuria). Il tetto va comunque
    rispettato (mai un terzo person_id) anche se questo significa che due
    raw_id diversi condividono lo stesso person_id per quel frame."""
    reid = SegReIdentifier(max_people=2)
    now = 0.0
    people = [
        (1, *make_person(100, 300), 0.9),
        (2, *make_person(500, 300), 0.9),
        (3, *make_person(300, 300), 0.9),  # terzo raw_id spurio, stesso frame
    ]
    resolved = reid.resolve(people, now)
    person_ids = {pid for pid, *_ in resolved}
    assert len(person_ids) <= 2, f"overflow nello stesso frame non gestito correttamente: {person_ids}"
    print(f"Overflow nello stesso frame (3 raw_id, max_people=2) -> "
          f"person_id distinti: {sorted(person_ids)} (nessun crash, tetto rispettato) — OK")


def main():
    hard_cap_never_exceeded_under_heavy_churn()
    churned_track_relinks_to_nearest_position()
    soft_match_below_cap_relinks_instead_of_minting_new()
    single_person_session_always_id_one()
    pathological_same_frame_overflow_does_not_crash()
    print("\nVerifica completata senza errori: seg_reid.py rispetta sempre il tetto "
          "max_people, anche sotto churn pesante o overflow nello stesso frame.")


if __name__ == "__main__":
    main()
