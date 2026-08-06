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
import pandas as pd

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


def part5_sweep_produces_cartesian_product_for_sam_only():
    # sam_chunk_size/overlap/redetect_every riportati nel CSV: yolo li
    # ignora (un solo run, None nelle colonne), sam31 esegue il prodotto
    # cartesiano di tutte le combinazioni date.
    original_build_tracker = bb.build_tracker
    bb.build_tracker = lambda *a, **k: _FakeTracker(_fake_results())
    try:
        df = bb.run_benchmark(
            ["yolo", "sam31"], source="unused.mp4", fps=15.0, device="cuda",
            sam_chunk_sizes=[300, 600], sam_overlaps=[30, 50], sam_redetect_everys=[100, None],
        )
    finally:
        bb.build_tracker = original_build_tracker

    yolo_rows = df[df["method"] == "yolo"]
    sam_rows = df[df["method"] == "sam31"]
    assert len(yolo_rows) == 1, \
        "yolo ignora i tre parametri sam_* -- un solo run, non ripetuto per ogni combinazione"
    # pandas converte None -> NaN in una colonna numerica mista con gli
    # interi delle righe sam31 -- pd.isna(), non "is None", per verificarlo.
    assert pd.isna(yolo_rows.iloc[0]["sam_chunk_size"]), \
        "yolo non usa sam_chunk_size -- NaN/None nel CSV, non un valore fuorviante"
    assert yolo_rows.iloc[0]["run_label"] == "yolo", \
        "con un solo run (yolo ignora la sweep), run_label == method, senza suffisso"
    assert len(sam_rows) == 8, "2 chunk_size x 2 overlap x 2 redetect_every = 8 combinazioni per sam31"
    assert set(sam_rows["sam_chunk_size"]) == {300, 600}
    assert set(sam_rows["sam_overlap"]) == {30, 50}
    # sam_redetect_every: pandas trasforma la colonna intera (incluso il
    # None di yolo) in float64 -- None diventa NaN, non confrontabile con
    # "==" ne' presente in un set nel modo consueto (nan != nan).
    redetect_values = sam_rows["sam_redetect_every"].tolist()
    redetect_non_null = {v for v in redetect_values if pd.notna(v)}
    assert redetect_non_null == {100}, f"atteso 100 tra i valori non-null, trovato {redetect_non_null}"
    assert any(pd.isna(v) for v in redetect_values), \
        "atteso anche il caso redetect_every=None (disattivato) nella sweep"
    assert sam_rows["run_label"].str.contains(r"sam31\[cs=").all(), \
        "con piu' di una combinazione, run_label deve distinguerle"
    print("PASS part5_sweep_produces_cartesian_product_for_sam_only")


def part6_sweep_skips_invalid_chunk_size_overlap_combo():
    original_build_tracker = bb.build_tracker
    bb.build_tracker = lambda *a, **k: _FakeTracker(_fake_results())
    try:
        df = bb.run_benchmark(
            ["sam31"], source="unused.mp4", fps=15.0, device="cuda",
            sam_chunk_sizes=[50, 300], sam_overlaps=[100], sam_redetect_everys=[None],
        )
    finally:
        bb.build_tracker = original_build_tracker
    # chunk_size=50 <= overlap=100 -- combinazione non valida, saltata; resta solo chunk_size=300
    assert len(df) == 1, "la combinazione non valida (50<=100) va saltata, non deve crashare"
    assert df.iloc[0]["sam_chunk_size"] == 300
    print("PASS part6_sweep_skips_invalid_chunk_size_overlap_combo")


def part7_parse_int_list_handles_commas_and_none():
    assert bb._parse_int_list("600") == [600]
    assert bb._parse_int_list("300,600") == [300, 600]
    assert bb._parse_int_list("300, 600 ") == [300, 600], "spazi attorno alla virgola tollerati"
    assert bb._parse_int_list("", allow_none=True) == [None], \
        "stringa vuota con allow_none -- default 'disattivato', un solo run"
    assert bb._parse_int_list("100,", allow_none=True) == [100, None], \
        "elemento vuoto tra le virgole -- include anche il caso 'disattivato' nella sweep"
    print("PASS part7_parse_int_list_handles_commas_and_none")


def main():
    part1_run_one_method_aggregates_lifespans_correctly()
    part2_sam_methods_skipped_without_cuda()
    part3_run_benchmark_skips_gracefully_and_keeps_valid_methods()
    part4_unknown_method_raises()
    part5_sweep_produces_cartesian_product_for_sam_only()
    part6_sweep_skips_invalid_chunk_size_overlap_combo()
    part7_parse_int_list_handles_commas_and_none()
    print("\nTutti i test di benchmark_backends.py sono passati.")


if __name__ == "__main__":
    main()
