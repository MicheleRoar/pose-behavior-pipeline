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

Onesta' sul livello di certezza (aggiornato dopo il primo test su macchina
CUDA reale di Michele, 2026-08-06 -- vedi sotto per cosa e' cambiato):
- `start_session` e `add_prompt` con `text=`/`frame_index=` -- CONFERMATI
  dal README ufficiale (snippet copiato sopra) E dalla run reale.
- Forma REALE di `response` per `add_prompt` con `text=` (CONFERMATA dalla
  run reale, diversa da quanto assunto inizialmente -- quella assunta era
  la forma dell'API IMMAGINE di SAM 3, `boxes`/`object_ids`/`scores`, non
  quella VIDEO):
      {
          "frame_index": ...,
          "outputs": {
              "out_obj_ids": [...],
              "out_boxes_xywh": [...],   # (x, y, width, height) per box
              "out_binary_masks": [...],
          },
      }
  `_add_text_prompt()`/`_propagate()` leggono queste chiavi con un
  fallback alle vecchie (`boxes`/`object_ids`/`masks`) per sicurezza, e
  sollevano un `RuntimeError` esplicito (chiavi trovate incluse nel
  messaggio) se non trovano nessuna delle due varianti, invece di un
  `KeyError` muto -- vedi cosa e' successo la prima volta.
- Formato box `out_boxes_xywh`: (x, y, larghezza, altezza), NON
  (x1,y1,x2,y2) come il resto della pipeline (`_polygon_to_box`,
  `_detect_people` via YOLO). Non e' inoltre chiaro dal solo valore se le
  coordinate sono gia' in pixel o normalizzate [0,1] (convenzione comune
  per API di detection, non documentata esplicitamente nel README) --
  vedi `_xywh_to_xyxy_pixels()` sotto per l'euristica usata (e il suo
  limite).
- `add_prompt` con `box=`/`obj_id=` invece di `text=` (usato dal
  box-seeding via YOLO, mantenuto per compatibilita' con l'uso originale
  di questa classe) -- ANCORA NON confermato su una run reale (nella
  prima prova YOLO non ha trovato nessuno sul frame di ancoraggio, quindi
  questo percorso non e' mai stato eseguito): resta un'estrapolazione per
  analogia con SAM2. Primo punto da verificare se emerge un problema qui.
- `propagate_in_video` NON passa da `handle_request()` -- CONFERMATO su
  una run reale (Michele, macchina CUDA, 2026-08-06): il dispatcher SAM 3
  ha sollevato `RuntimeError("invalid request type: propagate_in_video")`.
  La propagazione multi-frame usa un metodo STREAMING separato,
  `handle_stream_request()`, che ritorna un generatore (una risposta per
  frame), vedi `_propagate()` sotto. `handle_request()` resta corretto
  solo per `start_session`/`add_prompt` (risposta singola, confermato).
- Nomi esatti dei campi della richiesta di streaming (`start_frame_index`
  invece di `start_frame_idx`, `propagation_direction="forward"`) e forma
  esatta di ogni risposta del generatore -- NON confermati con la stessa
  certezza del cambio di metodo: dedotti per coerenza con la chiave
  `frame_index` gia' confermata in `add_prompt`. Il codice prova prima le
  chiavi reali gia' confermate per `add_prompt` (`out_obj_ids`/
  `out_binary_masks`, per analogia riusate anche qui), poi quelle vecchie
  come fallback, e solleva un errore esplicito con le chiavi trovate se
  nessuna corrisponde -- invece di indovinare di nuovo in silenzio.

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


def _xywh_to_xyxy_pixels(box, frame_shape: tuple[int, int]) -> np.ndarray:
    """Converte un box SAM 3.1 `out_boxes_xywh` (x, y, larghezza, altezza)
    nel formato (x1,y1,x2,y2) in pixel usato dal resto della pipeline
    (`_polygon_to_box`, `_detect_people` via YOLO). Non e' documentato se
    le coordinate arrivano gia' in pixel o normalizzate [0,1] -- si
    assume normalizzato solo se tutti e 4 i valori sono <= 1.5 (margine
    sopra 1.0): una persona reale occupa quasi sempre piu' di 1.5 pixel,
    quindi il caso ambiguo (pixel scambiati per normalizzati) e'
    improbabile. Se nell'overlay le box appaiono minuscole/ammassate in un
    angolo, e' il primo punto da controllare a mano."""
    x, y, w, h = (float(v) for v in box[:4])
    height, width = frame_shape
    if max(x, y, w, h) <= 1.5:
        x, y, w, h = x * width, y * height, w * width, h * height
    return np.array([x, y, x + w, y + h], dtype=float)


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

    def _add_text_prompt(self, predictor, state, *, frame_idx: int, text: str,
                          frame_shape: tuple[int, int]) -> dict[int, np.ndarray]:
        """Chiama SAM 3 col prompt testuale sul frame indicato: REGISTRA il
        prompt E ritorna le istanze scoperte in un solo colpo (a
        differenza del box-prompt, qui non c'e' modo di separare 'scopri'
        da 'traccia'). Ritorna `{id_locale_SAM: box_x1y1x2y2_pixel}`:
        questi id sono LOCALI a questa chiamata/chunk, non persistenti --
        la riconciliazione con gli id globali avviene in
        `_seed_new_chunk()`, non qui.

        Forma di `response` CONFERMATA su una run reale (Michele, macchina
        CUDA, 2026-08-06) -- vedi il docstring del modulo per i dettagli e
        cosa e' cambiato rispetto all'ipotesi iniziale (che era la forma
        dell'API IMMAGINE, non quella VIDEO)."""
        response = predictor.handle_request(request=dict(
            type="add_prompt", session_id=state, frame_index=frame_idx, text=text,
        ))
        outputs = response.get("outputs", response)
        boxes_xywh = outputs.get("out_boxes_xywh", outputs.get("boxes"))
        if boxes_xywh is None:
            raise RuntimeError(
                f"SAM 3.1 non ha restituito bounding box per il prompt testuale {text!r}. "
                f"Chiavi in response: {list(response.keys())}; "
                f"chiavi in outputs: {list(outputs.keys())}."
            )
        object_ids = outputs.get("out_obj_ids", outputs.get("object_ids", range(len(boxes_xywh))))
        return {
            int(oid): _xywh_to_xyxy_pixels(box, frame_shape)
            for oid, box in zip(object_ids, boxes_xywh)
        }

    def _propagate(self, predictor, state, *, start_frame_idx: int = 0,
                    max_frame_num_to_track: int | None = None):
        """Vedi la firma di base in `ChunkedVideoPredictorBackend`. In piu':
        traduce gli id LOCALI scoperti dal prompt testuale (vedi
        `_add_text_prompt`) negli id GLOBALI corrispondenti, usando la
        mappa costruita da `_seed_new_chunk()`. Un id assente dalla mappa
        (caso box-mode, o box-prompt di `redetect_every`) passa invariato:
        e' gia' l'id globale, essendo stato scelto da noi al momento del
        seeding (vedi `_add_box_prompt`).

        CONFERMATO su una run reale (Michele, macchina CUDA, 2026-08-06):
        `handle_request()` NON riconosce `type="propagate_in_video"` --
        il dispatcher SAM 3 solleva `RuntimeError("invalid request type")`.
        La propagazione multi-frame passa da un metodo STREAMING separato,
        `handle_stream_request()`, che ritorna un GENERATORE (una risposta
        per frame) invece di un unico dict con una lista "outputs" --
        `handle_request()` resta corretto solo per le richieste a risposta
        singola (`start_session`, `add_prompt`, gia' confermate). Nomi
        esatti dei campi della richiesta di streaming (`start_frame_index`
        invece di `start_frame_idx`, `propagation_direction`) -- dedotti
        per coerenza con la chiave `frame_index` gia' confermata in
        `add_prompt`, poi corroborati indipendentemente (stessi nomi)
        confrontando col predictor ufficiale SAM 3: se il prossimo errore
        e' di nuovo un "invalid request"/parametro sconosciuto, e' il
        primo punto da rivedere. `max_frame_num_to_track` viene OMESSO
        dalla richiesta se `None` invece di passarlo esplicitamente
        (difensivo: se l'API valida i tipi rigidamente, un `None`
        esplicito potrebbe rompersi dove una chiave assente andrebbe al
        default)."""
        request = {
            "type": "propagate_in_video", "session_id": state,
            "propagation_direction": "forward", "start_frame_index": start_frame_idx,
        }
        if max_frame_num_to_track is not None:
            request["max_frame_num_to_track"] = max_frame_num_to_track
        local_to_global = self._local_to_global
        for response in predictor.handle_stream_request(request=request):
            outputs = response.get("outputs", response)
            frame_idx = response.get("frame_index", outputs.get("frame_index"))
            obj_ids = outputs.get("out_obj_ids", outputs.get("object_ids", outputs.get("obj_ids")))
            masks = outputs.get("out_binary_masks", outputs.get("masks"))
            if frame_idx is None or obj_ids is None or masks is None:
                raise RuntimeError(
                    f"SAM 3.1 handle_stream_request: risposta frame inattesa. "
                    f"Chiavi in response: {list(response.keys())}; "
                    f"chiavi in outputs: {list(outputs.keys()) if outputs is not response else '(non annidato)'}."
                )
            remapped: dict[int, np.ndarray] = {}
            for oid, mask in zip(obj_ids, masks):
                oid = int(oid)
                global_id = local_to_global.get(oid, oid)
                remapped[global_id] = _to_boolean_mask(mask)
            yield int(frame_idx), remapped

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

        discovered = self._add_text_prompt(
            predictor, state, frame_idx=0, text=self.text_prompt, frame_shape=frame_shape,
        )

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
