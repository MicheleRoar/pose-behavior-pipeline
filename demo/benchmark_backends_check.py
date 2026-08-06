"""
benchmark_backends_check.py
==============================
Verifica `benchmark_backends.py`: la parte di aggregazione (durata delle
tracce, numero di id grezzi, percentuale "brevi", fps di elaborazione) con
un tracker FINTO iniettato al posto di `build_tracker()` (nessuna
dipendenza da YOLO/SAM/SAM2 reali, stessa filosofia degli altri
demo/*_check.py), e la logica di skip per i metodi sam31/sam2 quando il
device rilevato non e' "cuda".

Uso:
    python demo/benchmark_backends_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import benchmark_backends as bb  # noqa: E402
from segmentation.seg_estimation import SegFrameResult  # noqa: E402


class _FakeTracker:
    """Tracker finto: restituisce una sequenza di `SegFrameResult` fissata
    a mano, per verificare che `run_one_method()` aggreghi le metriche
    giuste a partire da un input noto (non un video/modello vero)."""

    def __init__(self, results):
        self._results = results

    def run(self, source, stream: bool = True):
        yield from self._results


def _fake_results():
    # id 1 presente in tutti e 3 i frame, id 2 solo nel frame di mezzo
    # (traccia piu' "corta") -- quello che conta e' il confronto RELATIVO
    # tra le due durate, non il valore assoluto (entrambe sotto
    # SHORT_LIVED_THRESHOLD_FRAMES=15, volutamente: un video di test da 3
    # frame non deve pretendere di avere tracce "lunghe" in senso assoluto).
    box = np.array([0.0, 0.0, 10.0, 10.0])
    poly = np.array([[0, 0], [10, 0], [10, 10]])
    return [
        SegFrameResult(frame_index=0, frame=np.zeros((2, 2, 3), dtype=np.uint8),
                        people=[(1, box, poly, 0.9)]),
        SegFrameResult(frame_index=1, frame=np.zeros((2, 2, 3), dtype=np.uint8),
                        people=[(1, box, poly, 0.9), (2, box, poly, 0.8)]),
        SegFrameResult(frame_index=2, frame=np.zeros((2, 2, 3), dtype=np.uint8),
                        people=[(1, box, poly, 0.9)]),
    ]


def part1_run_one_method_aggregates_lifespans_correctly():
    original_build_tracker = bb.build_tracker
    bb.build_tracker = lambda *a, **k: _FakeTracker(_fake_results())
    try:
        result = bb.run_one_method("yolo", source="unused.mp4", fps=15.0, device="cpu")
    finally:
        bb.build_tracker = original_build_tracker

    assert result is not None
    assert result["method"] == "yolo"
    assert result["n_frames"] == 3
    assert result["n_raw_ids"] == 2
    assert result["lifespan_min_frames"] == 1, "id 2 e' presente in un solo frame"
    assert result["lifespan_max_frames"] == 3, "id 1 e' presente in tutti e 3 i frame"
    assert result["short_lived_ids_pct"] == 100.0, "entrambi gli id sono sotto la soglia (15 frame)"
    assert result["lifespan_median_s"] == round(result["lifespan_median_frames"] / 15.0, 2), \
        "la conversione in secondi deve usare l'fps passato"
    print("PASS part1_run_one_method_aggregates_lifespans_correctly")


def part2_sam_methods_skipped_without_cuda():
    result = bb.run_one_method("sam31", source="unused.mp4", fps=15.0, device="mps")
    assert result is None, "sam31 su device!='cuda' deve essere saltato (None), senza costruire nulla"
    print("PASS part2_sam_methods_skipped_without_cuda")


def part3_run_benchmark_skips_gracefully_and_keeps_valid_methods():
    original_build_tracker = bb.build_tracker
    bb.build_tracker = lambda *a, **k: _FakeTracker(_fake_results())
    try:
        df = bb.run_benchmark(["yolo", "sam31"], source="unused.mp4", fps=15.0, device="mps")
    finally:
        bb.build_tracker = original_build_tracker

    assert len(df) == 1, "sam31 va saltato (device mps), deve restare solo 'yolo' (finto)"
    assert df.iloc[0]["method"] == "yolo"
    print("PASS part3_run_benchmark_skips_gracefully_and_keeps_valid_methods")


def part4_unknown_method_raises():
    try:
        bb.run_benchmark(["metodo-inventato"], source="unused.mp4", fps=15.0, device="cpu")
        raise AssertionError("atteso ValueError per un metodo sconosciuto")
    except ValueError:
        pass
    print("PASS part4_unknown_method_raises")


def main():
    part1_run_one_method_aggregates_lifespans_correctly()
    part2_sam_methods_skipped_without_cuda()
    part3_run_benchmark_skips_gracefully_and_keeps_valid_methods()
    part4_unknown_method_raises()
    print("\nTutti i test di benchmark_backends.py sono passati.")


if __name__ == "__main__":
    main()
