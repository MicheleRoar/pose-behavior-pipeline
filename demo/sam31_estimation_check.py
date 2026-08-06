"""
sam31_estimation_check.py
============================
Verifica `segmentation/sam31_estimation.py::Sam31Tracker` con un predictor
SAM 3 FINTO che parla l'API `handle_request()` REALE (confermata dal
README ufficiale facebookresearch/sam3 -- vedi il docstring del modulo
sotto test), non quella stile-SAM2 usata da `sam_backend_check.py` per
`Sam2Tracker`/`ChunkedVideoPredictorBackend` di base.

Due scenari:
- Modalita' box (default, `text_prompt=None`): stessa logica YOLO di
  sempre, ma instradata attraverso `handle_request()` invece delle
  chiamate stile-SAM2 -- verifica che la traduzione non rompa nulla.
- Modalita' prompt testuale (`text_prompt="person"`): SAM 3 "scopre" da
  solo le istanze (id LOCALI al chunk, diversi ogni volta) -- verifica che
  `chunking.reconcile_ids()` le riconcili correttamente con gli id
  GLOBALI del chunk precedente per geometria, e che una persona con box
  molto diverso (nessuna sovrapposizione) riceva un id nuovo.

Uso:
    python demo/sam31_estimation_check.py
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.sam31_estimation import Sam31Tracker  # noqa: E402

FRAME_SHAPE = (64, 64)  # (height, width)
TOTAL_FRAMES = 20
CHUNK_SIZE = 15
OVERLAP = 5

BOX_A = np.array([5.0, 5.0, 15.0, 15.0])
BOX_B = np.array([40.0, 40.0, 55.0, 55.0])  # lontana da A, nessuna sovrapposizione


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


class _FakeSam3Predictor:
    """Simula `handle_request()`/`handle_stream_request()` di SAM 3 (vedi
    il docstring di sam31_estimation.py): `start_session` apre una sessione
    (conta i JPEG scritti da `_init_state()`, stessa verifica di
    `sam_backend_check.py`), `add_prompt` con `text=` "scopre" istanze da
    uno script fisso (`discover_schedule`, un elemento per ogni chiamata
    attesa) e risponde con la forma REALE confermata su macchina CUDA
    (out_obj_ids/out_boxes_xywh/out_binary_masks, non boxes/object_ids come
    ipotizzato inizialmente), `add_prompt` con `box=`/`obj_id=` registra un
    seed noto. `handle_request()` RIFIUTA "propagate_in_video" (come il
    dispatcher SAM 3 reale, confermato su macchina CUDA) -- la
    propagazione passa da `handle_stream_request()`, un GENERATORE che
    restituisce le maschere statiche per gli obj_id correnti, una risposta
    per frame, nella finestra richiesta."""

    def __init__(self, frame_shape: tuple[int, int], discover_schedule: list[dict[int, np.ndarray]]):
        self.frame_shape = frame_shape
        self._discover_schedule = discover_schedule
        self._discover_call_count = 0
        self._sessions: dict[int, dict] = {}
        self._next_session = 0

    def handle_request(self, *, request: dict) -> dict:
        req_type = request["type"]

        if req_type == "start_session":
            sid = self._next_session
            self._next_session += 1
            num_frames = len([f for f in os.listdir(request["resource_path"]) if f.endswith(".jpg")])
            self._sessions[sid] = {"num_frames": num_frames, "seeds": {}}
            return {"session_id": sid}

        if req_type == "add_prompt":
            session = self._sessions[request["session_id"]]
            if "text" in request:
                discovered = self._discover_schedule[self._discover_call_count]
                self._discover_call_count += 1
                session["seeds"].update(discovered)
                obj_ids = list(discovered.keys())
                # Forma REALE confermata su macchina CUDA (vedi il docstring
                # del modulo sotto test): out_boxes_xywh e' (x,y,w,h), non
                # (x1,y1,x2,y2) -- qui si converte da BOX_A/BOX_B (gia'
                # x1y1x2y2, usate anche per _box_to_mask) SOLO per il
                # payload di rete, cosi' il test verifica anche la
                # conversione fatta da _xywh_to_xyxy_pixels().
                boxes_xywh = [
                    np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]], dtype=float)
                    for b in discovered.values()
                ]
                return {
                    "frame_index": request["frame_index"],
                    "outputs": {
                        "out_obj_ids": obj_ids,
                        "out_boxes_xywh": boxes_xywh,
                        "out_binary_masks": [_box_to_mask(discovered[oid], self.frame_shape) for oid in obj_ids],
                    },
                }
            session["seeds"][request["obj_id"]] = request["box"]
            return {"outputs": {}}

        # CONFERMATO su una run reale (vedi il docstring del modulo sotto
        # test): handle_request() NON riconosce "propagate_in_video", solo
        # handle_stream_request() lo fa -- qui si replica lo stesso rifiuto
        # del dispatcher SAM 3 reale, cosi' il test fallisce rumorosamente
        # se il codice sotto test tornasse per errore a chiamare
        # handle_request() per la propagazione.
        raise RuntimeError(f"invalid request type: {req_type}")

    def handle_stream_request(self, *, request: dict):
        """Simula la propagazione VIDEO reale: un GENERATORE, una risposta
        per frame (non un unico dict con una lista "outputs"), vedi il
        docstring del modulo sotto test per la conferma su macchina CUDA."""
        req_type = request["type"]
        if req_type != "propagate_in_video":
            raise RuntimeError(f"invalid stream request type: {req_type}")
        session = self._sessions[request["session_id"]]
        start = request["start_frame_index"]
        # Chiave OMESSA (non None esplicito) quando non impostata -- vedi
        # sam31_estimation.py::_propagate() -- .get() invece di [] per
        # rispecchiare esattamente cosa manda il codice reale.
        max_n = request.get("max_frame_num_to_track")
        end = session["num_frames"] if max_n is None else min(session["num_frames"], start + max_n)
        for local_idx in range(start, end):
            obj_ids = list(session["seeds"].keys())
            masks = [_box_to_mask(session["seeds"][oid], self.frame_shape) for oid in obj_ids]
            yield {
                "frame_index": local_idx,
                "outputs": {"out_obj_ids": obj_ids, "out_binary_masks": masks},
            }


def _make_tracker(fake_predictor, **kwargs) -> Sam31Tracker:
    tracker = Sam31Tracker(device="cuda", chunk_size=CHUNK_SIZE, overlap=OVERLAP, **kwargs)
    tracker._build_predictor = lambda: fake_predictor  # bypass import di sam3 (non installato qui)
    return tracker


def part1_box_mode_via_handle_request_matches_original_behaviour():
    # text_prompt=None (default): stessa logica YOLO/box di sempre, solo
    # instradata attraverso handle_request() -- verifica che la traduzione
    # dell'API non abbia rotto nulla del comportamento originale.
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        fake_predictor = _FakeSam3Predictor(FRAME_SHAPE, discover_schedule=[])
        tracker = _make_tracker(fake_predictor)
        tracker._detect_people = lambda frame: [BOX_A]  # sempre la stessa persona, per semplicita'

        results = list(tracker.run(video_path))
        frame_indices = [r.frame_index for r in results]
        assert frame_indices == list(range(TOTAL_FRAMES)), \
            f"attesi tutti i frame senza salti/duplicati, ottenuto {frame_indices}"
        assert all(len(r.people) == 1 for r in results), "una sola persona (A) in ogni frame"
        all_ids = {p[0] for r in results for p in r.people}
        assert len(all_ids) == 1, f"lo stesso id in tutta la sessione (box-mode = noi scegliamo obj_id), trovati {all_ids}"
        print("PASS part1_box_mode_via_handle_request_matches_original_behaviour")
    finally:
        shutil.rmtree(tmp_dir)


def part2_text_prompt_reconciles_continuing_person_and_assigns_new_id_to_stranger():
    # chunk 0: SAM 3 "scopre" una persona con il suo id locale 0 (box A).
    # chunk 1: SAM 3 riscopre la SCENA da zero (nuovo prompt testuale) e
    # assegna id locali DIVERSI (7 e 8, arbitrari: dimostra che non ci si
    # affida a una coincidenza di numerazione) -- 7 e' nella STESSA
    # posizione di A (deve riconciliare allo stesso id globale), 8 e' in
    # una posizione mai vista (box B, deve ricevere un id globale nuovo).
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        _make_synthetic_video(video_path)
        fake_predictor = _FakeSam3Predictor(FRAME_SHAPE, discover_schedule=[
            {0: BOX_A},
            {7: BOX_A, 8: BOX_B},
        ])
        tracker = _make_tracker(fake_predictor, text_prompt="person")

        results = list(tracker.run(video_path))
        ids_per_frame = {r.frame_index: {p[0] for p in r.people} for r in results}

        # Il chunk 1 scopre B al SUO frame 0 (globale 10), ma quel frame e'
        # nella finestra di overlap gia' emessa dal chunk 0 -- come per il
        # reseeding via YOLO (vedi part2 in sam_backend_check.py), B compare
        # nell'output solo dal primo frame REALMENTE emesso dal chunk 1,
        # cioe' 10 + OVERLAP.
        first_seen = 10 + OVERLAP
        assert all(len(ids_per_frame[f]) == 1 for f in range(0, first_seen)), \
            f"prima di {first_seen}, deve esserci solo la persona A"
        assert all(len(ids_per_frame[f]) == 2 for f in range(first_seen, TOTAL_FRAMES)), \
            f"da {first_seen} in poi, devono esserci sia A (continua) che B (nuova)"

        a_id_before = next(iter(ids_per_frame[0]))
        a_id_after = ids_per_frame[first_seen] & ids_per_frame[0]  # id in comune tra prima e dopo
        assert a_id_after == {a_id_before}, (
            f"l'id globale di A deve restare lo stesso nonostante SAM abbia riassegnato "
            f"un id locale diverso (0 -> 7) -- atteso {{{a_id_before}}}, trovato {a_id_after}"
        )
        new_id = ids_per_frame[first_seen] - {a_id_before}
        assert len(new_id) == 1, f"atteso esattamente un id nuovo per B, trovato {new_id}"
        assert next(iter(new_id)) != a_id_before, "B deve avere un id GLOBALE diverso da A"

        all_ids = set()
        for ids in ids_per_frame.values():
            all_ids |= ids
        assert len(all_ids) == 2, f"attesi esattamente 2 id globali in tutta la sessione, trovati {all_ids}"
        print("PASS part2_text_prompt_reconciles_continuing_person_and_assigns_new_id_to_stranger")
    finally:
        shutil.rmtree(tmp_dir)


def main():
    part1_box_mode_via_handle_request_matches_original_behaviour()
    part2_text_prompt_reconciles_continuing_person_and_assigns_new_id_to_stranger()
    print("\nTutti i test di sam31_estimation.py (con predictor SAM 3 finto) sono passati.")


if __name__ == "__main__":
    main()
