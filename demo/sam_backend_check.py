"""
sam_backend_check.py
=======================
Verifica `segmentation/sam_backend.py::ChunkedVideoPredictorBackend` con un
predictor SAM FINTO (nessuna dipendenza da sam3/samurai/GPU CUDA) --
stessa filosofia di `demo/device_check.py` (iniezione di un doppio finto al
posto della libreria pesante). Genera un piccolo video sintetico su disco
(cv2.VideoWriter) cosi' `run()` puo' leggere frame reali via
cv2.VideoCapture esattamente come farebbe con un video vero.

Scenario testato: persona A presente fin dal primo frame (rilevata al
bootstrap del chunk 0), persona B che "entra" a meta' video (rilevata solo
al chunk 2) -- verifica che: gli id restano continui tra un chunk e il
successivo per le persone gia' note, una persona nuova rilevata a meta'
video riceve un id mai usato prima, non ci sono frame duplicati o mancanti
nell'output finale, e la persistenza su disco scrive un file per chunk.

Uso:
    python demo/sam_backend_check.py
"""

import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.chunk_store import load_chunk  # noqa: E402
from segmentation.sam_backend import ChunkedVideoPredictorBackend  # noqa: E402

FRAME_SHAPE = (64, 64)  # (height, width)
TOTAL_FRAMES = 40
CHUNK_SIZE = 15
OVERLAP = 5

BOX_A = np.array([5.0, 5.0, 15.0, 15.0])
BOX_B = np.array([30.0, 30.0, 45.0, 45.0])


def _make_synthetic_video(path: str) -> None:
    height, width = FRAME_SHAPE
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    for _ in range(TOTAL_FRAMES):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def _box_to_mask(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box.astype(int)
    mask[y1:y2, x1:x2] = True
    return mask


class _FakePredictor:
    """Predictor SAM finto: 'ricorda' solo la box con cui e' stato seminato
    per ogni obj_id (posizione statica, non simula un vero movimento -- non
    e' il suo scopo, vedi il docstring del modulo) e la restituisce come
    maschera identica per ogni frame del chunk."""

    def __init__(self, frame_shape: tuple[int, int]):
        self.frame_shape = frame_shape

    def init_state(self, frames):
        return SimpleNamespace(num_frames=len(frames), seeds={})

    def add_new_points_or_box(self, state, *, frame_idx, obj_id, box):
        state.seeds[obj_id] = box

    def propagate_in_video(self, state):
        for local_idx in range(state.num_frames):
            obj_ids = list(state.seeds.keys())
            masks = [_box_to_mask(state.seeds[oid], self.frame_shape) for oid in obj_ids]
            yield local_idx, obj_ids, masks


class _FakeBackend(ChunkedVideoPredictorBackend):
    """Sottoclasse di test: `_build_predictor()` restituisce il predictor
    finto, `_detect_people()` segue uno 'copione' fisso invece di chiamare
    YOLO (non installato in questo ambiente) -- un elemento dello schedule
    per ogni chiamata attesa (una per chunk, vedi il docstring del
    modulo)."""

    def __init__(self, *, detect_schedule: list[list[np.ndarray]], **kwargs):
        super().__init__(**kwargs)
        self._detect_schedule = detect_schedule
        self._call_count = 0

    def _build_predictor(self):
        return _FakePredictor(FRAME_SHAPE)

    def _detect_people(self, frame):
        boxes = self._detect_schedule[self._call_count] if self._call_count < len(self._detect_schedule) else []
        self._call_count += 1
        return boxes


def _run_backend(chunk_store_dir: str | None):
    return _FakeBackend(
        device="cuda", chunk_size=CHUNK_SIZE, overlap=OVERLAP,
        detect_schedule=[[BOX_A], [], [BOX_B], []],
        chunk_store_dir=chunk_store_dir,
    )


def part1_ids_stay_continuous_across_chunks():
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _run_backend(chunk_store_dir=None)

        results = list(backend.run(video_path))
        frame_indices = [r.frame_index for r in results]
        assert frame_indices == list(range(TOTAL_FRAMES)), \
            f"attesi tutti i frame 0..{TOTAL_FRAMES - 1} senza salti/duplicati, ottenuto {frame_indices}"

        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}
        # persona A: presente fin dal frame 0, sempre lo stesso id in TUTTI i frame
        a_ids = {next(iter(ids)) for f, ids in ids_per_frame.items() if f < 20 and len(ids) == 1}
        assert len(a_ids) == 1, f"id di A deve restare lo stesso in tutti i frame < 20, trovati {a_ids}"
        print("PASS part1_ids_stay_continuous_across_chunks")
    finally:
        shutil.rmtree(tmp_dir)


def part2_new_person_mid_video_gets_fresh_id_with_expected_lag():
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _run_backend(chunk_store_dir=None)

        results = list(backend.run(video_path))
        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}

        # B viene rilevata durante il chunk 2 (che parte al frame originale
        # 20), ma quel frame di ancoraggio e' gia' stato emesso dal chunk 1
        # (che non conosceva B) -- quindi B compare per la prima volta solo
        # dal primo frame REALMENTE emesso dal chunk 2, cioe' 20 + overlap.
        first_seen = min(f for f, ids in ids_per_frame.items() if len(ids) == 2)
        assert first_seen == 20 + OVERLAP, f"atteso {20 + OVERLAP}, ottenuto {first_seen}"
        assert all(len(ids_per_frame[f]) == 1 for f in range(0, first_seen)), \
            "prima di questo, deve esserci solo A"
        assert all(len(ids_per_frame[f]) == 2 for f in range(first_seen, TOTAL_FRAMES)), \
            "da qui in poi, sia A che B devono essere presenti"

        all_ids = set()
        for ids in ids_per_frame.values():
            all_ids |= ids
        assert len(all_ids) == 2, f"attesi esattamente 2 id globali distinti in tutta la sessione, trovati {all_ids}"
        print(f"PASS part2_new_person_mid_video_gets_fresh_id_with_expected_lag (B vista da frame {first_seen})")
    finally:
        shutil.rmtree(tmp_dir)


def part3_chunk_persistence_writes_one_file_per_chunk():
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        store_dir = os.path.join(tmp_dir, "chunks")
        _make_synthetic_video(video_path)
        backend = _run_backend(chunk_store_dir=store_dir)

        list(backend.run(video_path))  # consuma il generatore

        files = sorted(os.listdir(store_dir))
        assert len(files) == 4, f"attesi 4 chunk (40 frame / 15 chunk_size / 5 overlap), trovati {files}"
        # il primo chunk contiene le righe di A su 15 frame
        records = load_chunk(os.path.join(store_dir, files[0]))
        assert len(records) == CHUNK_SIZE, f"il primo chunk deve avere una riga per frame (solo A), trovate {len(records)}"
        print("PASS part3_chunk_persistence_writes_one_file_per_chunk")
    finally:
        shutil.rmtree(tmp_dir)


def part4_non_cuda_device_rejected_immediately():
    try:
        _FakeBackend(device="mps", detect_schedule=[])
        raise AssertionError("atteso ValueError per device diverso da 'cuda'")
    except ValueError:
        pass
    print("PASS part4_non_cuda_device_rejected_immediately")


def main():
    part1_ids_stay_continuous_across_chunks()
    part2_new_person_mid_video_gets_fresh_id_with_expected_lag()
    part3_chunk_persistence_writes_one_file_per_chunk()
    part4_non_cuda_device_rejected_immediately()
    print("\nTutti i test di sam_backend.py (con predictor finto) sono passati.")


if __name__ == "__main__":
    main()
