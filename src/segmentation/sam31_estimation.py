"""
sam31_estimation.py
=====================
`Sam31Tracker`: backend di segmentazione/tracking basato su SAM 3.1
(facebookresearch/sam3, checkpoint `facebook/sam3.1` -- ultima versione
disponibile al 2026-08, rilasciata 2026-03-27 come aggiornamento "Object
Multiplex" di SAM 3, vedi https://github.com/facebookresearch/sam3). Tutta
la logica di chunking/persistenza vive in
`segmentation/sam_backend.py::ChunkedVideoPredictorBackend` -- qui la parte
specifica: come costruire il predictor SAM 3.1 e, novita', come seminare un
chunk usando il prompt TESTUALE di SAM 3 invece del box-seeding via YOLO
(vedi `text_prompt` sotto).

API reale (confermata dal README ufficiale facebookresearch/sam3, sezione
"Basic Usage" -- NON quella stile-SAM2 assunta inizialmente da questo
modulo/da `ChunkedVideoPredictorBackend`, completamente diversa):

    from sam3.model_builder import build_sam3_video_predictor
    video_predictor = build_sam3_video_predictor()
    response = video_predictor.handle_request(request=dict(
        type="start_session", resource_path=video_path,  # cartella JPEG o file MP4
    ))
    response = video_predictor.handle_request(request=dict(
        type="add_prompt", session_id=response["session_id"],
        frame_index=0, text="<PROMPT>",
    ))
    output = response["outputs"]

Un'unica funzione (`handle_request`, un dizionario "richiesta" con un campo
`type`) sostituisce sia `init_state`/`add_new_points_or_box` che
(presumibilmente) `propagate_in_video` di SAM2 -- qui tradotta nei tre
metodi overridabili di `ChunkedVideoPredictorBackend` (`_init_state`,
`_add_box_prompt`, `_propagate`) per riusare la logica di chunking/
riconciliazione/persistenza senza duplicarla.

Onesta' sul livello di certezza (nessuna GPU CUDA in questo ambiente,
nessun test su una macchina reale per questo modulo specifico -- solo per
`ChunkedVideoPredictorBackend`/`Sam2Tracker`, che parlano un'API diversa):
- `start_session` e `add_prompt` con `text=`/`frame_index=` -- CONFERMATI
  dal README ufficiale (snippet copiato sopra).
- `add_prompt` con `box=`/`obj_id=` invece di `text=` (usato dal
  box-seeding via YOLO, mantenuto per compatibilita' con l'uso originale
  di questa classe) -- NON mostrato nel README (che mostra solo l'esempio
  testuale), estrapolato per analogia con SAM2. Se il predictor reale si
  aspetta chiavi diverse, e' il primo punto da correggere.
- Il `type` per continuare la propagazione sui frame successivi al prompt
  (`"propagate_in_video"` sotto) e i nomi esatti dei campi in
  `response["outputs"]` per il caso VIDEO (masks/object_ids per frame) --
  NON mostrati nel README (che per il video si ferma al primo output del
  prompt), scelti per coerenza col resto della libreria SAM2/SAM3. Da
  verificare su una macchina CUDA reale confrontando con
  `examples/sam3_video_predictor_example.ipynb` prima di fare affidamento
  su questo modulo.

Modalita' prompt testuale (`text_prompt`, es. "person")
-----------------------------------------------------------
Nato da un confronto diretto con la pipeline di produzione CHUV
(Video-Annotation-System, che usa `psifx video tracking sam3 inference
--text_prompt "person"`, confermato funzionante in produzione): SAM 3 puo'
scoprire da SOLO tutte le istanze di un concetto testuale in un frame,
senza bisogno di YOLO come proposer. Quando `text_prompt` e' impostato,
`_seed_new_chunk()` chiama SAM 3 con quel prompt sul frame di ancoraggio
invece di YOLO: SAM 3 assegna PERO' i propri id LOCALI alle istanze
scoperte (non possiamo chiedergli di riusare un nostro id globale, a
differenza del box-prompt) -- vengono quindi riconciliati per geometria
con gli id globali del chunk precedente usando `chunking.reconcile_ids()`
(la stessa utility di IoU-matching gia' scritta per il controllo di
coerenza in `ChunkedVideoPredictorBackend.run()`, qui riusata come
meccanismo di ABBINAMENTO PRIMARIO invece che solo di log). Se
`text_prompt` e' `None` (default), il comportamento resta quello
originale a box via YOLO, invariato -- `Sam31Tracker()` senza argomenti si
comporta esattamente come prima di questa modifica.

`redetect_every` (vedi sam_backend.py) resta compatibile con
`text_prompt`: la ri-detection periodica dentro il chunk usa ancora
YOLO/box (non un nuovo prompt testuale, per non dover riconciliare due
volte a meta' chunk), ma i box-prompt passano `obj_id` scelto da NOI
(un id globale) -- essendo gia' un id globale non necessita di traduzione,
`_propagate()` lo lascia passare invariato (vedi `_local_to_global` sotto).
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from segmentation.chunking import GlobalIdAllocator, reconcile_ids
from segmentation.sam_backend import ChunkedVideoPredictorBackend, _box_to_polygon, _to_boolean_mask

DEFAULT_CHECKPOINT = "facebook/sam3.1"


class Sam31Tracker(ChunkedVideoPredictorBackend):
    """Vedi il docstring del modulo. `text_prompt=None` (default): box-
    seeding via YOLO, comportamento originale invariato. `text_prompt=
    "person"` (o altro concetto aperto): SAM 3 scopre le istanze da solo,
    YOLO non viene mai chiamato per il seeding di un chunk (resta
    disponibile solo per `redetect_every`, vedi il docstring del modulo)."""

    def __init__(self, *, checkpoint: str = DEFAULT_CHECKPOINT,
                 text_prompt: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.text_prompt = text_prompt
        # {id_locale_SAM: id_globale} per il chunk CORRENTE, popolata da
        # _seed_new_chunk() solo in modalita' text_prompt -- vedi
        # _propagate(). None/vuoto in modalita' box (gli id locali SONO
        # gia' gli id globali, essendo stati scelti da noi).
        self._local_to_global: dict[int, int] = {}

    def _build_predictor(self):
        # Import ritardato: solleva ImportError con un messaggio chiaro se
        # il pacchetto sam3 non e' installato, invece di rompere l'import
        # dell'intero modulo `segmentation` per chi non usa questo backend.
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as exc:
            raise ImportError(
                "Sam31Tracker richiede il pacchetto 'sam3' (non installato). "
                "Vedi https://github.com/facebookresearch/sam3 -- "
                "'git clone ... && pip install -e .', checkpoint gated su "
                "Hugging Face (facebook/sam3.1, richiede accesso approvato)."
            ) from exc
        return build_sam3_video_predictor()

    # ------------------------------------------------- API handle_request
    def _init_state(self, predictor, frames: list[np.ndarray]):
        """Come `ChunkedVideoPredictorBackend._init_state()` (scrive il
        chunk come cartella JPEG temporanea, stesso vincolo "MP4 o cartella
        JPEG" atteso ereditato da SAM2), ma qui la 'state' restituita e' un
        `session_id` (stringa) da `start_session`, non un oggetto di
        libreria -- vedi il docstring del modulo per la firma reale
        confermata dal README."""
        self._current_chunk_tmpdir = tempfile.mkdtemp(prefix="sam31_tracker_")
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(self._current_chunk_tmpdir, f"{i:06d}.jpg"), frame)
        response = predictor.handle_request(request=dict(
            type="start_session", resource_path=self._current_chunk_tmpdir,
        ))
        return response["session_id"]

    def _add_box_prompt(self, predictor, state, *, frame_idx: int, obj_id: int, box: np.ndarray) -> None:
        # NON confermato dal README (che mostra solo l'esempio testuale) --
        # vedi la nota "Onesta'" nel docstring del modulo.
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=state, frame_index=frame_idx,
            box=box, obj_id=obj_id,
        ))

    def _add_text_prompt(self, predictor, state, *, frame_idx: int, text: str) -> dict[int, np.ndarray]:
        """Chiama SAM 3 col prompt testuale sul frame indicato: REGISTRA il
        prompt E ritorna le istanze scoperte in un solo colpo (a
        differenza del box-prompt, qui non c'e' modo di separare 'scopri'
        da 'traccia' -- e' lo stesso identico snippet "Basic Usage" del
        README). Ritorna `{id_locale_SAM: box}`: questi id sono LOCALI a
        questa chiamata/chunk, non persistenti -- la riconciliazione con
        gli id globali avviene in `_seed_new_chunk()`, non qui."""
        response = predictor.handle_request(request=dict(
            type="add_prompt", session_id=state, frame_index=frame_idx, text=text,
        ))
        outputs = response["outputs"]
        # Forma esatta NON confermata per il caso multi-istanza da video (il
        # README mostra masks/boxes/scores solo per l'API immagine, vedi il
        # docstring del modulo) -- si assume qui la stessa convenzione, con
        # un id per istanza preso da 'object_ids' se presente, altrimenti
        # l'indice di scoperta come id locale.
        boxes = outputs["boxes"]
        object_ids = outputs.get("object_ids", range(len(boxes)))
        return {int(oid): np.asarray(box, dtype=float) for oid, box in zip(object_ids, boxes)}

    def _propagate(self, predictor, state, *, start_frame_idx: int = 0,
                    max_frame_num_to_track: int | None = None):
        """Vedi la firma di base in `ChunkedVideoPredictorBackend`. In piu':
        traduce gli id LOCALI scoperti dal prompt testuale (vedi
        `_add_text_prompt`) negli id GLOBALI corrispondenti, usando la
        mappa costruita da `_seed_new_chunk()`. Un id assente dalla mappa
        (caso box-mode, o box-prompt di `redetect_every`) passa invariato:
        e' gia' l'id globale, essendo stato scelto da noi al momento del
        seeding (vedi `_add_box_prompt`)."""
        response = predictor.handle_request(request=dict(
            type="propagate_in_video", session_id=state,
            start_frame_idx=start_frame_idx, max_frame_num_to_track=max_frame_num_to_track,
        ))
        local_to_global = self._local_to_global
        for frame_out in response["outputs"]:
            frame_idx = frame_out["frame_index"]
            obj_ids = frame_out.get("object_ids", frame_out.get("obj_ids"))
            masks = frame_out["masks"]
            remapped: dict[int, np.ndarray] = {}
            for oid, mask in zip(obj_ids, masks):
                oid = int(oid)
                global_id = local_to_global.get(oid, oid)
                remapped[global_id] = _to_boolean_mask(mask)
            yield frame_idx, remapped

    # ------------------------------------------------------------ seeding
    def _seed_new_chunk(self, predictor, state, *, chunk_frames: list[np.ndarray],
                         chunk_index: int, frame_shape: tuple[int, int],
                         prev_anchor_polys: dict[int, np.ndarray],
                         allocator: GlobalIdAllocator) -> dict[int, np.ndarray]:
        if not self.text_prompt:
            self._local_to_global = {}
            return super()._seed_new_chunk(
                predictor, state, chunk_frames=chunk_frames, chunk_index=chunk_index,
                frame_shape=frame_shape, prev_anchor_polys=prev_anchor_polys, allocator=allocator,
            )

        discovered = self._add_text_prompt(predictor, state, frame_idx=0, text=self.text_prompt)

        local_to_global: dict[int, int] = {}
        seed_boxes: dict[int, np.ndarray] = {}
        if chunk_index == 0 or not prev_anchor_polys:
            for local_id, box in discovered.items():
                global_id = allocator.next_id()
                local_to_global[local_id] = global_id
                seed_boxes[global_id] = box
        else:
            discovered_polys = {local_id: _box_to_polygon(box) for local_id, box in discovered.items()}
            mapping = reconcile_ids(prev_anchor_polys, discovered_polys, frame_shape,
                                     iou_threshold=self.iou_threshold)
            for local_id, box in discovered.items():
                global_id = mapping.get(local_id)
                if global_id is None:
                    global_id = allocator.next_id()  # persona nuova, mai vista prima
                local_to_global[local_id] = global_id
                seed_boxes[global_id] = box

        if self.max_people is not None and len(seed_boxes) > self.max_people:
            keep = set(list(seed_boxes.keys())[: self.max_people])
            seed_boxes = {gid: box for gid, box in seed_boxes.items() if gid in keep}
            local_to_global = {lid: gid for lid, gid in local_to_global.items() if gid in keep}

        self._local_to_global = local_to_global
        return seed_boxes
