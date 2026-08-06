"""
chunking_check.py
===================
Verifica `segmentation/chunking.py` (suddivisione in chunk sovrapposti,
IoU su poligoni rasterizzati, riconciliazione id greedy, allocatore id
globali) con dati sintetici -- nessuna dipendenza da SAM/SAM2/GPU, solo
numpy/cv2 gia' richiesti dal resto del progetto.

Uso:
    python demo/chunking_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.chunking import (  # noqa: E402
    GlobalIdAllocator, iter_chunk_ranges, polygon_iou, reconcile_ids,
)


def _square(x: int, y: int, size: int = 40) -> np.ndarray:
    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=float)


def part1_chunk_ranges_cover_full_video_with_overlap():
    ranges = list(iter_chunk_ranges(total_frames=1000, chunk_size=300, overlap=50))
    assert ranges[0] == (0, 300), ranges[0]
    assert ranges[1] == (250, 550), ranges[1]
    assert ranges[2] == (500, 800), ranges[2]
    assert ranges[3] == (750, 1000), ranges[3]  # ultimo chunk piu' corto
    assert ranges[-1][1] == 1000, "deve coprire fino all'ultimo frame"
    # ogni chunk (tranne il primo) inizia esattamente 'overlap' frame prima
    # della fine del precedente
    for (prev_start, prev_end), (start, _end) in zip(ranges, ranges[1:]):
        assert start == prev_end - 50, (prev_end, start)
    print("PASS part1_chunk_ranges_cover_full_video_with_overlap")


def part2_chunk_ranges_exact_division_no_dangling_tiny_chunk():
    # 900 frame esatti / chunk 300 overlap 50 -> deve fermarsi in modo pulito
    ranges = list(iter_chunk_ranges(total_frames=900, chunk_size=300, overlap=50))
    assert ranges[-1][1] == 900
    assert all(end - start > 0 for start, end in ranges)
    print("PASS part2_chunk_ranges_exact_division_no_dangling_tiny_chunk")


def part3_chunk_size_must_exceed_overlap():
    try:
        list(iter_chunk_ranges(total_frames=100, chunk_size=50, overlap=50))
        raise AssertionError("atteso ValueError con chunk_size <= overlap")
    except ValueError:
        pass
    print("PASS part3_chunk_size_must_exceed_overlap")


def part4_polygon_iou_known_values():
    shape = (200, 200)
    a = _square(10, 10, 40)
    identical = _square(10, 10, 40)
    disjoint = _square(150, 150, 40)
    half_overlap = _square(30, 10, 40)  # sovrapposizione parziale su x

    assert abs(polygon_iou(a, identical, shape) - 1.0) < 1e-6
    assert polygon_iou(a, disjoint, shape) == 0.0
    iou_partial = polygon_iou(a, half_overlap, shape)
    assert 0.0 < iou_partial < 1.0, iou_partial
    # poligono degenere (< 3 punti) -> 0.0, non un crash
    assert polygon_iou(a, np.empty((0, 2)), shape) == 0.0
    print(f"PASS part4_polygon_iou_known_values (iou parziale={iou_partial:.3f})")


def part5_reconcile_ids_matches_by_geometry():
    shape = (200, 200)
    # chunk precedente: id globali 10 e 20, in due posizioni distinte
    prev = {10: _square(10, 10), 20: _square(100, 100)}
    # nuovo chunk: SAM ha assegnato id locali 0 e 1, nelle stesse posizioni
    # (persone ferme nel frame di ancoraggio) ma enumerate in ordine diverso
    new = {0: _square(100, 100), 1: _square(10, 10)}

    mapping = reconcile_ids(prev, new, shape, iou_threshold=0.3)
    assert mapping == {0: 20, 1: 10}, mapping
    print("PASS part5_reconcile_ids_matches_by_geometry")


def part6_reconcile_ids_unmatched_local_id_gets_no_mapping():
    shape = (200, 200)
    prev = {10: _square(10, 10)}
    # persona 1 e' la stessa di prima (10,10); persona 2 e' NUOVA (entrata
    # nel campo durante questo chunk, nessuna maschera corrispondente prima)
    new = {0: _square(10, 10), 1: _square(150, 150)}

    mapping = reconcile_ids(prev, new, shape, iou_threshold=0.3)
    assert mapping == {0: 10}, mapping
    assert 1 not in mapping, "l'id locale 1 (persona nuova) non deve avere corrispondenza"
    print("PASS part6_reconcile_ids_unmatched_local_id_gets_no_mapping")


def part7_reconcile_ids_never_double_assigns():
    # due nuovi id sono entrambi geometricamente vicini allo stesso vecchio
    # id (caso ambiguo/raro): solo uno dei due puo' ereditarlo, l'altro resta
    # senza mapping invece di duplicare l'identita'.
    shape = (200, 200)
    prev = {10: _square(10, 10)}
    new = {0: _square(12, 12), 1: _square(15, 15)}

    mapping = reconcile_ids(prev, new, shape, iou_threshold=0.3)
    assert len(mapping) == 1, mapping
    assert list(mapping.values()) == [10]
    print("PASS part7_reconcile_ids_never_double_assigns")


def part8_global_id_allocator_never_repeats():
    allocator = GlobalIdAllocator()
    ids = [allocator.next_id() for _ in range(5)]
    assert ids == [1, 2, 3, 4, 5], ids
    print("PASS part8_global_id_allocator_never_repeats")


def main():
    part1_chunk_ranges_cover_full_video_with_overlap()
    part2_chunk_ranges_exact_division_no_dangling_tiny_chunk()
    part3_chunk_size_must_exceed_overlap()
    part4_polygon_iou_known_values()
    part5_reconcile_ids_matches_by_geometry()
    part6_reconcile_ids_unmatched_local_id_gets_no_mapping()
    part7_reconcile_ids_never_double_assigns()
    part8_global_id_allocator_never_repeats()
    print("\nTutti i test di chunking.py sono passati.")


if __name__ == "__main__":
    main()
