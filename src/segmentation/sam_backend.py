"""
sam_backend.py
================
Base condivisa da `Sam31Tracker` (sam31_estimation.py) e `Sam2Tracker`
(sam2_estimation.py): entrambe le librerie espongono la STESSA API video
stateful (documentazione ufficiale: facebookresearch/sam3 e
facebookresearch/sam2) --

    state = predictor.init_state(frames)
    predictor.add_new_points_or_box(state, frame_idx=..., obj_id=..., box=...)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        ...

-- quindi tutta la logica di chunking/prompting/riconciliazione/persistenza
vive UNA volta sola qui; le due sottoclassi implementano solo
`_build_predictor()` (quale libreria importare/istanziare, import ritardato
perche' ne' sam3 ne' sam2 sono installabili in questo ambiente: servono
Python 3.12+/CUDA 12.6+, vedi requirements.txt).

Perche' il chunking (vedi anche segmentation/chunking.py)
------------------------------------------------------------
`init_state()` carica in memoria i pixel di TUTTI i frame passati -- su un
video di diversi minuti non e' praticabile passarlo intero. Si processa a
finestre sovrapposte (`chunk_size` frame, `overlap` in comune tra un chunk
e il successivo).

Strategia di prompting/continuita' degli id
------------------------------------------------
- Primo chunk: nessuna persona nota ancora. Si usa YOLO (lo stesso modello
  gia' usato da `SegTracker`, qui solo come RILEVATORE su un singolo frame,
  non come tracker) per proporre le box iniziali sul primo frame del
  chunk. ATTENZIONE: questo significa che la qualita' del prompt iniziale
  dipende comunque da YOLO -- SAM qui sostituisce il TRACKING/re-id nel
  tempo, non necessariamente la detection iniziale (si potrebbe passare a
  un prompt testuale "person" se il modello SAM 3.1 concept-prompting lo
  supporta a sufficienza; da verificare sulla macchina CUDA, vedi
  Sam31Tracker).
- Chunk successivi: per ogni persona gia' nota (id globale) si ricava un
  box prompt dalla sua maschera nell'ULTIMO frame del chunk precedente
  (che e' anche il PRIMO frame -- "frame di ancoraggio" -- del nuovo
  chunk, essendo nella finestra di overlap), e lo si registra con
  `obj_id=<id globale>`: SAM continua quindi a usare direttamente lo
  stesso id, non serve un abbinamento a posteriori nel caso comune. Si fa
  girare ANCHE YOLO sul frame di ancoraggio per individuare persone NUOVE
  (entrate nel campo durante il chunk precedente) non gia' coperte da un
  prompt esistente (IoU basso con tutte le box gia' seminate): a queste si
  assegna un id globale mai usato.
- Come controllo di coerenza (non come meccanismo di abbinamento primario,
  vedi sopra), sul frame di ancoraggio si confronta comunque la maschera
  che SAM produce con quella seminata (`chunking.polygon_iou`): se l'IoU
  cala sotto `iou_threshold` viene solo loggato un avviso -- puo' voler
  dire che SAM ha perso la persona o "scambiato" identita' nell'overlap,
  utile da vedere nei log ma non gestito automaticamente in questa prima
  versione (vedi chunking.py per il limite noto).

Ri-detection periodica dentro il chunk (`redetect_every`)
-------------------------------------------------------------
Problema osservato (Michele, confronto diretto YOLO+ByteTrack vs
Sam2Tracker sullo stesso video): YOLO+ByteTrack rileva su OGNI frame,
mentre nello schema sopra YOLO gira una sola volta per chunk (il frame di
ancoraggio) -- con `chunk_size` di default 600, una detection ogni ~40s a
15fps. Chi non e' esattamente nel frame di ancoraggio resta invisibile per
TUTTO il resto del chunk, anche se e' visibile nel 99% degli altri frame;
in piu' SAM propaga per inerzia (memoria interna) e se perde una persona a
meta' chunk (occlusione, movimento rapido) non ha modo di ririlevarla
prima del prossimo confine di chunk. Risultato: molte meno maschere
rispetto a YOLO+ByteTrack, non per una differenza di qualita' del modello
ma per la frequenza di ri-detection.

`redetect_every` (default `None` = comportamento invariato, un'unica
finestra grande quanto il chunk) spezza ogni chunk in sotto-finestre da
`redetect_every` frame: si propaga una sotto-finestra alla volta
(`propagate_in_video(state, start_frame_idx=..., max_frame_num_to_track=...)`,
assunto disponibile perche' documentato nelle notebook ufficiali SAM2/SAM3
per aggiungere oggetti a meta' video -- NON ancora verificato su una
macchina CUDA reale, vedi la nota "Onesta'" sotto), poi si richiama YOLO
sul primo frame della sotto-finestra SUCCESSIVA per proporre eventuali
persone nuove non ancora seminate (stesso confronto IoU di
`reseed_new_people`, e infatti rispetta lo stesso flag: con
`reseed_new_people=False` nessuna ri-detection avviene, ne' al confine di
chunk ne' dentro il chunk -- resta la condizione "SAM puro"). Le persone
gia' note continuano a propagare automaticamente (stesso `state`, stessa
memoria SAM) -- non serve riseminarle ad ogni sotto-finestra, la
ri-detection propone SOLO le eventuali new entries.

Onesta' su cosa e' verificato qui
--------------------------------------
Questo modulo e' stato scritto e testato inizialmente SOLO con un predictor
finto iniettato al posto di sam3/sam2 (vedi `demo/sam_backend_check.py`,
nessuna GPU CUDA in questo ambiente). `_init_state()` sotto e' stato pero'
CORRETTO a partire da un test reale su una macchina CUDA (col fork
SAMURAI, che vendorizza lo stesso `sam2_video_predictor` di SAM2 vanilla --
stesso comportamento atteso con `Sam2Tracker`): il predictor NON accetta
una lista di frame in memoria come si era assunto all'inizio -- si e'
scontrato con "Only MP4 video and JPEG folder are supported". Si scrive
quindi ogni chunk come sequenza JPEG in una cartella temporanea (vedi
sotto) prima di chiamare `init_state()`. Non ancora confermato se SAM 3.1
ha lo stesso vincolo (eredita lo stesso codice video di SAM2, quindi
probabile) -- se si scoprisse che accetta anche liste di frame,
`_init_state()` resta comunque overridabile per sottoclasse se in futuro
conviene differenziare.

Perche' Sam2Tracker e non SamuraiTracker
------------------------------------------
`SamuraiTracker` (rimosso) usava lo stesso predictor con la modalita'
motion-aware di SAMURAI attiva. Verificato su una macchina CUDA reale:
quel codice (`sam2_base.py::_forward_sam_heads`) assume un solo oggetto
per sessione (scritto/validato sui benchmark di visual object tracking
single-target LaSOT/GOT-10k/TrackingNet) -- seminando piu' persone nella
stessa sessione (il caso normale qui) va in crash con `RuntimeError:
Boolean value of Tensor with more than one value is ambiguous`. SAM2
vanilla (`Sam2Tracker`, sam2_estimation.py) usa lo stesso predictor SENZA
quella patch: supporta il multi-oggetto batchato nativamente, a costo di
perdere il motion-modeling attraverso le occlusioni. Vedi il docstring di
`segmentation/sam2_estimation.py` per il dettaglio completo.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Iterator

import cv2
import numpy as np

from segmentation.chunk_store import save_chunk
from segmentation.chunking import GlobalIdAllocator, iter_chunk_ranges, polygon_iou
from segmentation.seg_estimation import SegFrameResult

COCO_PERSON_CLASS_ID = 0
NEW_PERSON_IOU_THRESHOLD = 0.2  # sotto questa soglia una detection YOLO sul
# frame di ancoraggio e' considerata una persona NUOVA (non gia' seminata)


def _probe_video(source) -> tuple[int, tuple[int, int]]:
    """Numero di frame totali e (height, width) del video sorgente --
    usato per dimensionare le rasterizzazioni di `polygon_iou` e per
    calcolare i chunk. Stesso approccio "solo metadati, nessuna inferenza"
    di `webui/api.py::probe_video_metadata`, reimplementato qui per non
    creare una dipendenza di `segmentation/` verso `webui/`."""
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise ValueError(f"Impossibile aprire la sorgente video: {source!r}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return total_frames, (height, width)
    finally:
        cap.release()


def _read_frame_range(source, start: int, end: int) -> list[np.ndarray]:
    """Legge in memoria i frame `[start, end)` (BGR). Il posizionamento via
    `CAP_PROP_POS_FRAMES` non e' garantito perfettamente accurato su tutti i
    codec/contenitori (nota gia' presente altrove nel progetto, vedi
    `probe_video_metadata`), ma sufficiente per questo uso: eventuali
    scostamenti di uno-due frame non compromettono la riconciliazione, che
    lavora comunque su una finestra di overlap di decine di frame."""
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(end - start):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames
    finally:
        cap.release()


def _to_boolean_mask(mask) -> np.ndarray:
    """Converte l'oggetto maschera restituito da `propagate_in_video()` in
    un array numpy booleano (H,W), qualunque sia il formato di partenza --
    verificato SOLO col predictor finto dei test fin qui (nessuna GPU CUDA
    in questo ambiente): SAM2 restituisce tipicamente un tensore
    PyTorch di LOGIT (valori reali, non gia' 0/1), su GPU, spesso con una
    dimensione canale in piu' (es. forma (1,H,W) invece di (H,W)). Senza
    questa conversione `_mask_to_polygon()` riceveva dati nel formato
    sbagliato e produceva contorni vuoti o insensati SENZA sollevare
    un'eccezione -- da qui il sintomo osservato ("il video parte ma non
    appaiono le maschere"), non un crash.

    - Se l'oggetto ha un metodo `.detach()` (duck-typing per torch.Tensor,
      niente `import torch` qui: non deve essere richiesto per usare il
      solo backend YOLO), lo si porta su CPU e si converte in numpy.
    - Se resta un array a 3 dimensioni (canale extra tipo (1,H,W)), si
      tiene solo il primo canale.
    - Se e' gia' booleano, si restituisce cosi' com'e' (caso del predictor
      finto nei test, e di un ipotetico predictor che restituisce gia'
      maschere pronte).
    - Altrimenti si ASSUME siano logit e si soglia a 0.0 (foreground se
      > 0), la convenzione usata da SAM2 per i suoi mask_logits. Se sulla
      macchina CUDA le maschere risultassero palesemente sbagliate (troppo
      piccole/grandi/vuote) nonostante questo fix, e' il primo punto da
      controllare: potrebbero essere gia' probabilita' in [0,1], nel qual
      caso la soglia giusta sarebbe 0.5, non 0.0."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.dtype == bool:
        return mask
    return mask > 0.0


def _mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    """Maschera binaria (H,W) -> poligono (N,2) del contorno esterno piu'
    grande (una persona puo' produrre piu' componenti connesse per un
    'buco' nella maschera; si tiene solo la principale, stessa scelta
    pragmatica di `mask_area`/`mask_centroid` in seg_estimation.py che
    trattano una maschera come un singolo poligono). Poligono vuoto (0,2)
    se la maschera non contiene pixel positivi."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2))
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(float)


def _polygon_to_box(poly: np.ndarray) -> np.ndarray:
    """Bounding box (x1,y1,x2,y2) del poligono, usata come prompt per
    `add_new_points_or_box()`. `[0,0,0,0]` se il poligono e' vuoto."""
    if poly.shape[0] == 0:
        return np.zeros(4)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    return np.array([x1, y1, x2, y2], dtype=float)


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    """Inverso di `_polygon_to_box`: poligono rettangolare (4,2) da una box
    (x1,y1,x2,y2) -- serve per riusare `chunking.polygon_iou`/`reconcile_ids`
    (che lavorano su poligoni) quando l'unica informazione disponibile e'
    una box (es. detection YOLO, o istanze scoperte da un prompt testuale
    SAM 3, vedi `Sam31Tracker._seed_new_chunk()`)."""
    return np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])


class ChunkedVideoPredictorBackend:
    """Vedi il docstring del modulo. Le sottoclassi devono implementare
    `_build_predictor()`; possono ridefinire `_init_state()` /
    `_add_box_prompt()` / `_propagate()` se la firma reale della libreria
    diverge da quella assunta qui (vedi la nota "Onesta' su cosa e'
    verificato" sopra)."""

    def __init__(self, *, device: str = "cuda", chunk_size: int = 600,
                 overlap: int = 50, iou_threshold: float = 0.3,
                 prompt_model: str = "yolo26s-seg.pt",
                 conf_threshold: float = 0.1,
                 chunk_store_dir: str | None = None,
                 max_people: int | None = None,
                 reseed_new_people: bool = True,
                 redetect_every: int | None = None):
        if device != "cuda":
            # SAM 3.1/SAM2 non dichiarano supporto mps/cpu (vedi ricerca
            # citata nel README) -- meglio fallire subito con un messaggio
            # chiaro che lasciar provare e ottenere un errore oscuro dentro
            # la libreria.
            raise ValueError(
                f"{type(self).__name__} richiede device='cuda' (SAM 3.1/SAM2 "
                f"non supportano mps/cpu al momento) -- ricevuto device={device!r}"
            )
        if overlap < 1:
            # La riconciliazione ID tra chunk consecutivi (reconcile_ids,
            # vedi run() sotto) confronta le maschere sullo STESSO frame
            # prodotto da entrambi i chunk (il "frame di ancoraggio", vedi
            # il docstring del modulo/di chunking.py) -- con overlap=0 i
            # chunk sono adiacenti ma non condividono NESSUN frame, quindi
            # `prev_anchor_polys` risulterebbe sempre vuoto e ogni nuovo
            # chunk assegnerebbe id globali TUTTI nuovi, perdendo in
            # silenzio la continuita' d'identita' ad ogni confine -- un
            # fallimento peggiore di un errore esplicito qui. La pipeline
            # CHUV di produzione non ha un parametro di overlap (vedi
            # Video-Annotation-System, chunk_size=400 senza overlap), ma
            # riconcilia diversamente (IoU diretto tra l'ultimo frame
            # tracciato del chunk N e il primo del chunk N+1, non un
            # confronto sullo stesso frame) -- un design diverso da questo,
            # non replicabile solo mettendo overlap=0 qui.
            raise ValueError(
                f"overlap deve essere >= 1 (ricevuto {overlap}) -- la riconciliazione "
                "degli id tra chunk richiede almeno un frame in comune, vedi il "
                "docstring di ChunkedVideoPredictorBackend.__init__"
            )
        self.device = device
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold
        self.prompt_model = prompt_model
        self.conf_threshold = conf_threshold
        self.chunk_store_dir = chunk_store_dir
        self.max_people = max_people
        # Se False: YOLO viene chiamato SOLO al bootstrap del chunk 0 (deve
        # pur esserci un primo prompt da qualche parte), mai per scoprire
        # persone NUOVE ai confini dei chunk successivi -- SAM/SAM2 resta
        # libero di gestire (o non gestire) da solo l'ingresso di qualcuno
        # a meta' video. Serve per ottenere una condizione "SAM puro" da
        # confrontare con quella di default (con reseeding), invece di
        # avere un solo metodo ibrido spacciato per "capacita' di SAM" --
        # vedi la discussione sulla baseline falsata e benchmark_backends.py.
        self.reseed_new_people = reseed_new_people
        # Vedi "Ri-detection periodica" nel docstring del modulo. None =
        # una sola finestra per chunk (comportamento originale, invariato).
        self.redetect_every = redetect_every
        self._detector = None  # YOLO caricato pigramente in _detect_people()
        self._current_chunk_tmpdir: str | None = None  # vedi _init_state()/run()

    # -------------------------------------------------- da implementare
    def _build_predictor(self):
        raise NotImplementedError

    # ------------------------------------------------- override opzionale
    def _init_state(self, predictor, frames: list[np.ndarray]):
        """SAM2 (e presumibilmente SAM 3.1, stesso codice video di
        origine) NON accetta una lista di frame in memoria: vuole un
        percorso a un video MP4 o a una cartella di JPEG in sequenza
        (verificato su una macchina CUDA reale, vedi il docstring del
        modulo). Si scrive quindi il chunk corrente come `000000.jpg`,
        `000001.jpg`, ... in una cartella temporanea -- che resta viva
        finche' il chunk non e' stato completamente propagato: la
        cancella `run()` subito dopo (non qui, perche' questo metodo
        ritorna prima che `propagate_in_video()` abbia letto i file)."""
        self._current_chunk_tmpdir = tempfile.mkdtemp(prefix="chunked_video_predictor_")
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(self._current_chunk_tmpdir, f"{i:06d}.jpg"), frame)
        return predictor.init_state(self._current_chunk_tmpdir)

    def _cleanup_chunk_tmpdir(self) -> None:
        if self._current_chunk_tmpdir is not None:
            shutil.rmtree(self._current_chunk_tmpdir, ignore_errors=True)
            self._current_chunk_tmpdir = None

    def _add_box_prompt(self, predictor, state, *, frame_idx: int, obj_id: int, box: np.ndarray) -> None:
        predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id, box=box)

    def _seed_new_chunk(self, predictor, state, *, chunk_frames: list[np.ndarray],
                         chunk_index: int, frame_shape: tuple[int, int],
                         prev_anchor_polys: dict[int, np.ndarray],
                         allocator: GlobalIdAllocator) -> dict[int, np.ndarray]:
        """Decide chi seguire in questo chunk, REGISTRA i prompt presso il
        predictor (side effect, non lasciato al chiamante: vedi
        `Sam31Tracker._seed_new_chunk()` per il perche' -- in modalita'
        prompt testuale una singola chiamata scopre E registra le persone
        insieme, non si possono separare i due passi come nel caso a box),
        e ritorna `{id_globale: box}` per chi e' attualmente noto.

        Default (usato da `Sam2Tracker` e da `Sam31Tracker` in modalita' a
        box): YOLO propone le box, vedi il docstring del modulo."""
        seed_boxes: dict[int, np.ndarray] = {}
        if chunk_index == 0:
            for box in self._detect_people(chunk_frames[0]):
                seed_boxes[allocator.next_id()] = box
        else:
            for global_id, poly in prev_anchor_polys.items():
                seed_boxes[global_id] = _polygon_to_box(poly)
            if self.reseed_new_people:
                for box in self._detect_people(chunk_frames[0]):
                    if not _overlaps_any(box, seed_boxes.values(), frame_shape, NEW_PERSON_IOU_THRESHOLD):
                        seed_boxes[allocator.next_id()] = box  # persona nuova, mai vista prima
            # se reseed_new_people e' False: nessuna chiamata a YOLO qui,
            # SAM/SAM2 continua SOLO le tracce gia' note (o non ne trova
            # piu' nessuna se tutti sono usciti dal campo) -- e' la
            # condizione "SAM puro" per il confronto in benchmark_backends.py.

        if self.max_people is not None and len(seed_boxes) > self.max_people:
            # tetto rigido, stessa logica di cap_by_confidence altrove nel
            # progetto: qui non abbiamo una confidenza per ordinare, si
            # tiene semplicemente l'ordine di scoperta (persone gia' note
            # prima delle nuove, essendo inserite per prime nel dict).
            seed_boxes = dict(list(seed_boxes.items())[: self.max_people])

        for global_id, box in seed_boxes.items():
            self._add_box_prompt(predictor, state, frame_idx=0, obj_id=global_id, box=box)
        return seed_boxes

    def _propagate(self, predictor, state, *, start_frame_idx: int = 0,
                    max_frame_num_to_track: int | None = None
                    ) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        """Deve restituire, per ogni frame locale al chunk, `(frame_idx,
        {obj_id: mask_booleana})`. La conversione a maschera booleana (da
        tensore torch/logit a numpy bool, vedi `_to_boolean_mask()`)
        avviene QUI, non nel chiamante, cosi' il resto di `run()` puo'
        assumere sempre lo stesso formato indipendentemente da cosa
        restituisce la libreria reale.

        `start_frame_idx`/`max_frame_num_to_track`: propagano solo una
        SOTTO-finestra del chunk (usato da `redetect_every`, vedi il
        docstring del modulo) invece dell'intero chunk in un colpo solo --
        firma presente nelle notebook ufficiali SAM2/SAM3 per riprendere la
        propagazione dopo aver aggiunto nuovi oggetti a meta' video, ma MAI
        chiamata su una macchina CUDA reale in questo progetto finora: se
        il predittore reale la rifiuta o si comporta diversamente, e' il
        primo punto da correggere (insieme a `_add_box_prompt()` per il
        frame_idx non-zero usato dalla ri-detection)."""
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            state, start_frame_idx=start_frame_idx, max_frame_num_to_track=max_frame_num_to_track,
        ):
            yield frame_idx, {obj_id: _to_boolean_mask(mask) for obj_id, mask in zip(obj_ids, masks)}

    # --------------------------------------------------------------- run
    def run(self, source, stream: bool = True) -> Iterator[SegFrameResult]:
        total_frames, frame_shape = _probe_video(source)
        predictor = self._build_predictor()
        allocator = GlobalIdAllocator()
        prev_anchor_polys: dict[int, np.ndarray] = {}

        for chunk_index, (start, end) in enumerate(iter_chunk_ranges(total_frames, self.chunk_size, self.overlap)):
            chunk_frames = _read_frame_range(source, start, end)
            if not chunk_frames:
                break
            state = self._init_state(predictor, chunk_frames)

            # Decide chi seguire E registra i prompt presso il predictor
            # (side effect di _seed_new_chunk, vedi il suo docstring) --
            # default YOLO/box, sovrascritto da Sam31Tracker per il prompt
            # testuale di SAM 3 quando `text_prompt` e' impostato.
            seed_boxes = self._seed_new_chunk(
                predictor, state, chunk_frames=chunk_frames, chunk_index=chunk_index,
                frame_shape=frame_shape, prev_anchor_polys=prev_anchor_polys, allocator=allocator,
            )

            chunk_results: list[SegFrameResult] = []
            polys_by_local_frame: dict[int, dict[int, np.ndarray]] = {}
            # Ultima box nota per ogni id (aggiornata dopo ogni sotto-finestra
            # propagata) -- usata dalla ri-detection periodica per decidere se
            # una detection YOLO e' una persona GIA' nota (IoU alta con la sua
            # ultima posizione) o NUOVA. Deliberatamente NON i seed_boxes
            # originali (che diventerebbero stantii dopo che qualcuno si
            # muove): vedi "Ri-detection periodica" nel docstring del modulo.
            known_boxes: dict[int, np.ndarray] = dict(seed_boxes)

            try:
                # Una sola finestra grande quanto il chunk se redetect_every
                # non e' impostato (comportamento originale, invariato).
                window_size = self.redetect_every or len(chunk_frames)
                local_idx = 0
                while local_idx < len(chunk_frames):
                    window_end = min(local_idx + window_size, len(chunk_frames))

                    if not known_boxes:
                        # Nessuna persona da seguire in questa finestra: ne' una
                        # traccia in corso ne' una nuova rilevata da YOLO -- puo'
                        # succedere con un video che si apre su una stanza vuota,
                        # o se YOLO manca la detection su quel frame specifico
                        # (illuminazione, posa, soglia di confidenza).
                        # `propagate_in_video()` di SAM/SAM2 SOLLEVA un errore
                        # ("No points are provided; please add points first") se
                        # chiamata senza nessun prompt registrato -- qui si
                        # emettono frame vuoti per QUESTA finestra invece di far
                        # esplodere l'intera pipeline, e si riprova comunque a
                        # ririlevare all'inizio della finestra successiva (se
                        # redetect_every e' impostato). Se questo compare per
                        # OGNI finestra di OGNI chunk, il problema e' quasi
                        # certamente a monte: verificare che `_detect_people()`
                        # trovi davvero qualcuno sul frame.
                        print(f"[{type(self).__name__}] avviso: nessuna persona da seguire "
                              f"nella finestra locale [{local_idx},{window_end}) del chunk "
                              f"{chunk_index} (frame originali [{start + local_idx},"
                              f"{start + window_end})) -- frame vuoti per questa finestra.")
                        chunk_results.extend(
                            SegFrameResult(frame_index=start + i, frame=chunk_frames[i], people=[])
                            for i in range(local_idx, window_end)
                        )
                    else:
                        for local_out_idx, masks_by_id in self._propagate(
                            predictor, state, start_frame_idx=local_idx,
                            max_frame_num_to_track=window_end - local_idx,
                        ):
                            people = []
                            polys_this_frame: dict[int, np.ndarray] = {}
                            for obj_id, mask in masks_by_id.items():
                                poly = _mask_to_polygon(mask)
                                polys_this_frame[obj_id] = poly
                                box = _polygon_to_box(poly)
                                # SAM non produce una confidenza di detection comparabile
                                # a quella di YOLO: 1.0 come segnaposto esplicito, MAI
                                # usato per il tetto max_people qui (gia' applicato sopra
                                # sui seed) -- vedi cap_by_confidence in tracking_common.py
                                # per il caso YOLO, dove la confidenza e' invece reale.
                                people.append((obj_id, box, poly, 1.0))
                            polys_by_local_frame[local_out_idx] = polys_this_frame
                            chunk_results.append(SegFrameResult(
                                frame_index=start + local_out_idx, frame=chunk_frames[local_out_idx],
                                people=people,
                            ))
                        # posizione piu' recente nota per ogni id ancora
                        # tracciato (un poligono vuoto = SAM l'ha perso: esce
                        # da known_boxes, ririlevabile come "nuovo" in seguito)
                        last_polys = polys_by_local_frame.get(window_end - 1, {})
                        known_boxes = {
                            obj_id: _polygon_to_box(poly)
                            for obj_id, poly in last_polys.items() if poly.shape[0] > 0
                        }

                    local_idx = window_end
                    if local_idx >= len(chunk_frames):
                        break

                    if self.redetect_every and self.reseed_new_people:
                        # Ri-detection periodica: propone SOLO persone nuove (IoU
                        # bassa con tutte le posizioni note) -- chi e' gia'
                        # tracciato continua a propagare da solo, non va riseminato.
                        for box in self._detect_people(chunk_frames[local_idx]):
                            if not _overlaps_any(box, known_boxes.values(), frame_shape, NEW_PERSON_IOU_THRESHOLD):
                                new_id = allocator.next_id()
                                self._add_box_prompt(predictor, state, frame_idx=local_idx, obj_id=new_id, box=box)
                                known_boxes[new_id] = box
            finally:
                # la cartella temporanea coi JPEG del chunk (vedi _init_state())
                # non serve piu' una volta che propagate_in_video() e' stata
                # consumata fino in fondo (o non e' mai stata chiamata, vedi
                # sopra) -- ripulita anche se l'inferenza solleva un'eccezione
                # a meta' chunk, per non lasciare cartelle temporanee orfane
                # su un video lungo con molti chunk.
                self._cleanup_chunk_tmpdir()

            # controllo di coerenza (solo log, vedi docstring del modulo)
            anchor_polys = polys_by_local_frame.get(0, {})
            for global_id, seeded_poly in prev_anchor_polys.items():
                produced_poly = anchor_polys.get(global_id)
                if produced_poly is None:
                    print(f"[{type(self).__name__}] avviso: id {global_id} non ritrovato "
                          f"al frame di ancoraggio del chunk {chunk_index}")
                    continue
                iou = polygon_iou(seeded_poly, produced_poly, frame_shape)
                if iou < self.iou_threshold:
                    print(f"[{type(self).__name__}] avviso: id {global_id} IoU basso "
                          f"({iou:.2f}) al frame di ancoraggio del chunk {chunk_index} "
                          f"-- possibile perdita/scambio di identita'")

            if self.chunk_store_dir:
                save_chunk(chunk_results, self.chunk_store_dir, chunk_index)

            # frame di ancoraggio per il PROSSIMO chunk: l'ultimo frame di
            # questo chunk che ricade nella finestra di overlap col successivo
            next_anchor_local = len(chunk_frames) - self.overlap
            prev_anchor_polys = polys_by_local_frame.get(next_anchor_local, {})

            # evita di riemettere due volte i frame della finestra di overlap
            # (gia' emessi dal chunk precedente, tranne per il primo chunk)
            skip = 0 if chunk_index == 0 else self.overlap
            for r in chunk_results[skip:]:
                yield r

    # --------------------------------------------------------------- YOLO
    def _detect_people(self, frame: np.ndarray) -> list[np.ndarray]:
        """Box (x1,y1,x2,y2) delle persone rilevate da YOLO su un singolo
        frame -- usato SOLO per proporre prompt iniziali a SAM (non per
        tracciare), vedi il docstring del modulo per il perche'. Import
        ritardato: stesso motivo di `SegTracker`.

        Logga sempre cosa trova (o non trova): utile per distinguere "il
        frame e' davvero senza persone" da "YOLO ha un problema" -- vedi
        la sessione di debug con SAMURAI in cui l'avviso 'nessuna persona
        da seguire' usciva anche con persone chiaramente visibili nel
        frame (poi risolta: il vero problema era il crash multi-oggetto di
        SAMURAI, non YOLO, vedi il docstring del modulo)."""
        if self._detector is None:
            from ultralytics import YOLO
            self._detector = YOLO(self.prompt_model)
            print(f"[{type(self).__name__}] proposer YOLO caricato: modello={self.prompt_model!r} "
                  f"device={self.device!r} conf_threshold={self.conf_threshold}")
        result = self._detector.predict(
            source=frame, device=self.device, conf=self.conf_threshold,
            classes=[COCO_PERSON_CLASS_ID], verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            # "result.boxes is None" e "len(...) == 0 ma non None" sono
            # entrambi "zero detection", ma distinguerli nel log serve a
            # chi debugga: se qui compare la riga sotto anche su un frame
            # con persone ben visibili, il sospetto si sposta da "il frame
            # e' vuoto" a "il modello/soglia/device di questo detector ha
            # un problema" (es. confrontare con --backend yolo sullo
            # stesso video: se LI' YOLO trova le persone, il problema e'
            # specifico di questa chiamata .predict(), non del modello).
            print(f"[{type(self).__name__}] YOLO: 0 persone rilevate su questo frame "
                  f"(conf_threshold={self.conf_threshold})")
            return []
        confs = [round(c, 2) for c in result.boxes.conf.cpu().numpy().tolist()]
        print(f"[{type(self).__name__}] YOLO: {len(confs)} persona/e rilevate su questo frame "
              f"(confidenze: {confs})")
        return [box for box in result.boxes.xyxy.cpu().numpy()]


def _overlaps_any(box: np.ndarray, existing_boxes, frame_shape: tuple[int, int], threshold: float) -> bool:
    """True se `box` ha IoU >= `threshold` con almeno una delle
    `existing_boxes` -- box, non poligoni, quindi si costruisce un
    poligono rettangolare al volo per riusare `polygon_iou`."""
    box_poly = _box_to_polygon(box)
    for other in existing_boxes:
        other_poly = _box_to_polygon(other)
        if polygon_iou(box_poly, other_poly, frame_shape) >= threshold:
            return True
    return False
