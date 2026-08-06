"""
sam_backend_check.py
=======================
Verifica `segmentation/sam_backend.py::ChunkedVideoPredictorBackend` con un
predictor SAM FINTO (nessuna dipendenza da sam3/sam2/GPU CUDA) --
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
from segmentation.sam_backend import ChunkedVideoPredictorBackend, _to_boolean_mask  # noqa: E402

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
    maschera identica per ogni frame del chunk.

    `init_state()` riceve un PERCORSO a una cartella (non una lista di
    frame in memoria) -- stessa interfaccia richiesta dal vero SAM2/SAM
    (vedi `ChunkedVideoPredictorBackend._init_state()`): qui si limita a
    contare i file JPEG scritti da `_init_state()`, cosi' il test verifica
    davvero che quella scrittura sia avvenuta con il numero giusto di
    frame, non solo che il codice non sollevi un'eccezione."""

    def __init__(self, frame_shape: tuple[int, int]):
        self.frame_shape = frame_shape

    def init_state(self, video_dir: str):
        num_frames = len([f for f in os.listdir(video_dir) if f.endswith(".jpg")])
        return SimpleNamespace(num_frames=num_frames, seeds={})

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


def part4b_no_detection_in_bootstrap_chunk_yields_empty_frames_no_crash():
    # Errore reale osservato su una macchina CUDA (col fork SAMURAI, stesso
    # sam2_video_predictor di SAM2 vanilla):
    # "RuntimeError: No points are provided; please add points first",
    # sollevato da propagate_in_video() quando nessun prompt e' mai stato
    # registrato -- capita se YOLO non trova nessuno nel frame di
    # ancoraggio del primo chunk (es. video che si apre su una stanza
    # vuota). Qui si verifica che il nostro codice lo gestisca PRIMA di
    # arrivare a chiamare propagate_in_video, restituendo frame vuoti
    # invece di propagare l'eccezione.
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        # tutti i chunk restituiscono lista vuota da _detect_people(): mai
        # nessuna persona rilevata in tutto il video
        backend = _FakeBackend(
            device="cuda", chunk_size=CHUNK_SIZE, overlap=OVERLAP,
            detect_schedule=[[], [], [], []],
        )

        results = list(backend.run(video_path))  # non deve sollevare nulla

        frame_indices = [r.frame_index for r in results]
        assert frame_indices == list(range(TOTAL_FRAMES)), \
            f"anche senza nessuna persona rilevata, la copertura frame deve restare completa, ottenuto {frame_indices}"
        assert all(r.people == [] for r in results), "nessuna persona rilevata -> tutti i frame devono avere people=[]"
        print("PASS part4b_no_detection_in_bootstrap_chunk_yields_empty_frames_no_crash")
    finally:
        shutil.rmtree(tmp_dir)


def part5_chunks_are_written_as_jpeg_and_cleaned_up_after_each_chunk():
    # Verifica il fix del 2026-08 (confermato su una macchina CUDA reale col
    # fork SAMURAI: "Only MP4 video and JPEG folder are supported", il predictor
    # non accetta una lista di frame in memoria) -- _init_state() deve
    # scrivere ogni chunk come cartella JPEG temporanea E ripulirla subito
    # dopo, non lasciarne in giro una per chunk su un video con molti chunk.
    import tempfile as tempfile_mod

    tmp_dir = tempfile.mkdtemp()
    created_dirs: list[str] = []
    original_mkdtemp = tempfile_mod.mkdtemp

    def _spy_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        created_dirs.append(path)
        return path

    tempfile_mod.mkdtemp = _spy_mkdtemp
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _run_backend(chunk_store_dir=None)

        list(backend.run(video_path))  # consuma il generatore, un _init_state() per chunk

        assert len(created_dirs) == 4, f"attesa una cartella temporanea per chunk (4), trovate {len(created_dirs)}"
        for d in created_dirs:
            assert not os.path.exists(d), f"{d} avrebbe dovuto essere ripulita dopo il suo chunk, esiste ancora"
        print(f"PASS part5_chunks_are_written_as_jpeg_and_cleaned_up_after_each_chunk "
              f"({len(created_dirs)} cartelle create e ripulite, una per chunk)")
    finally:
        tempfile_mod.mkdtemp = original_mkdtemp
        shutil.rmtree(tmp_dir)


class _FakeTorchTensor:
    """Duck-type minimo di un torch.Tensor: solo i tre metodi usati da
    `_to_boolean_mask()` (detach/cpu/numpy) -- verifica quel percorso senza
    dipendere da torch (non installato in questo ambiente)."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def part6b_reseed_new_people_false_never_adds_the_late_entrant():
    # Confronto "SAM puro" vs "SAM + reseeding" (vedi discussione sulla
    # baseline falsata): con reseed_new_people=False, YOLO viene chiamato
    # SOLO al bootstrap del chunk 0 -- la persona B (che nello scenario di
    # part2 entra a meta' video) non deve MAI comparire, perche' nessuno la
    # scopre piu' dopo il chunk 0.
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        backend = _FakeBackend(
            device="cuda", chunk_size=CHUNK_SIZE, overlap=OVERLAP,
            detect_schedule=[[BOX_A], [], [BOX_B], []],  # B verrebbe "vista" al chunk 2 se reseed fosse attivo
            reseed_new_people=False,
        )

        results = list(backend.run(video_path))
        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}

        all_ids = set()
        for ids in ids_per_frame.values():
            all_ids |= ids
        assert len(all_ids) == 1, f"con reseed disattivato B non deve mai comparire, id trovati: {all_ids}"
        assert all(len(ids) == 1 for ids in ids_per_frame.values()), \
            "ogni frame deve avere solo A, mai B, per tutta la sessione"
        print("PASS part6b_reseed_new_people_false_never_adds_the_late_entrant")
    finally:
        shutil.rmtree(tmp_dir)


def part6_to_boolean_mask_handles_bool_logits_3d_and_tensor_like_input():
    # Fix del 2026-08: sintomo reale osservato su una macchina CUDA
    # ("il video parte ma non appaiono le maschere") -- causato da
    # propagate_in_video() che restituisce logit (non gia' booleani),
    # spesso in tensori con una dimensione canale in piu' -- vedi il
    # docstring di _to_boolean_mask() per il perche' di ogni caso qui sotto.
    already_bool = np.array([[True, False], [False, True]])
    assert np.array_equal(_to_boolean_mask(already_bool), already_bool)

    logits_2d = np.array([[-2.0, 3.0], [0.5, -0.1]])
    result = _to_boolean_mask(logits_2d)
    assert result.dtype == bool
    assert result.tolist() == [[False, True], [True, False]], "soglia a 0.0: >0 = foreground"

    logits_3d = logits_2d.reshape(1, 2, 2)  # forma (1,H,W), tipica di SAM2
    result_3d = _to_boolean_mask(logits_3d)
    assert result_3d.shape == (2, 2), "il canale extra va rimosso, non lasciato nella forma"
    assert result_3d.tolist() == [[False, True], [True, False]]

    tensor_like = _FakeTorchTensor(logits_2d)
    result_tensor = _to_boolean_mask(tensor_like)
    assert result_tensor.tolist() == [[False, True], [True, False]], \
        "un oggetto con .detach()/.cpu()/.numpy() (duck-typing torch.Tensor) va convertito allo stesso modo"

    print("PASS part6_to_boolean_mask_handles_bool_logits_3d_and_tensor_like_input")


def main():
    part1_ids_stay_continuous_across_chunks()
    part2_new_person_mid_video_gets_fresh_id_with_expected_lag()
    part3_chunk_persistence_writes_one_file_per_chunk()
    part4_non_cuda_device_rejected_immediately()
    part4b_no_detection_in_bootstrap_chunk_yields_empty_frames_no_crash()
    part5_chunks_are_written_as_jpeg_and_cleaned_up_after_each_chunk()
    part6b_reseed_new_people_false_never_adds_the_late_entrant()
    part6_to_boolean_mask_handles_bool_logits_3d_and_tensor_like_input()
    print("\nTutti i test di sam_backend.py (con predictor finto) sono passati.")


if __name__ == "__main__":
    main()
