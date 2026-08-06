"""
webui/api.py
=============
Ponte tra il frontend web (webui/index.html + app.js, dentro una finestra
pywebview) e la pipeline esistente. Non duplica NESSUNA logica di
elaborazione: riusa `VideoPlayer` (gui/video_player.py) e
`iter_pipeline_frames` (gui/pipeline_runner.py) esattamente come fa
gui/app.py (la GUI Tkinter, che resta nel repo invariata) -- questo modulo
si occupa solo di: dialoghi file nativi, un thread di riproduzione in
background, codifica dei frame in JPEG base64, e calcolo delle metriche
mostrate nella status bar.

Perche' un thread invece di `root.after()` come in app.py
-------------------------------------------------------------
Tkinter impone un unico thread per gli aggiornamenti UI (vedi il docstring
di app.py); pywebview no: la finestra e' un processo/webview separato che
riceve aggiornamenti tramite `window.evaluate_js(...)`, quindi si puo' far
girare la riproduzione su un thread Python dedicato che chiama
`step_forward()` e spinge ogni frame al DOM, invece di far "pollare" il
lato JS. La cadenza (attesa "delay dopo la fine dell'elaborazione
precedente", non un timer a frequenza fissa) replica deliberatamente la
stessa logica non-garantita di `app.py._tick()` -- vedi li' per il perche'.

Onesta' delle metriche
------------------------
`processing_fps` e `avg_latency_ms` sono calcolati da una media mobile di
tempi reali attorno a `step_forward()` (vedi `_LatencyTracker`), non
inventati. `people_count` viene da `RunnerFrame.people_count` (vedi
pipeline_runner.py: perche' NON da `len(rows)`, che in modalita' "pose" puo'
restare a zero finche' la finestra scorrevole delle feature non si
riempie). Il device mostrato e' l'etichetta configurata (es. "mps"), non
una telemetria di utilizzo GPU reale -- non affidabilmente leggibile da
puro Python/PyTorch su Apple Silicon, vedi discussione nel README.

Import di `webview` ritardato (dentro i soli metodi che ne hanno bisogno),
cosi' `build_player_kwargs` / `encode_frame_jpeg_b64` / `build_status` /
`_LatencyTracker` restano testabili senza pywebview installato -- vedi
`demo/webui_api_check.py`.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import traceback
from collections import deque

import cv2
import pandas as pd

from gui.pipeline_runner import iter_pipeline_frames, RunnerFrame
from gui.video_player import VideoPlayer
from common.device import detect_default_device  # solo la funzione: non importa torch
                                                    # finche' non viene CHIAMATA (vedi sotto)

MODE_KEYS = {"segmentation", "pose", "both"}


def probe_video_metadata(path: str) -> dict:
    """Legge SOLO i metadati del file (frame count e fps dichiarati dal
    container) via `cv2.VideoCapture`, SENZA processare/decodificare i frame
    uno per uno -- non e' in contraddizione con "i tracker sono sequenziali,
    niente salto arbitrario" (vedi gui/video_player.py): qui non c'e'
    nessuna inferenza, solo la lettura di un header, cosi' com'e' gia' cosa
    fanno i player video normali per mostrare la durata prima ancora di
    aver iniziato a riprodurre. Usato per il timecode "corrente / totale" e
    per le tacche della timeline sull'intera durata nota -- il prefisso
    ELABORATO resta comunque l'unico punto in cui si puo' saltare
    istantaneamente (vedi `VideoPlayer.seek`), la durata totale qui e' solo
    informativa.

    Isolata da `Api` per essere testabile senza pywebview (basta un file
    video reale o, nei test, viene aggirata passando un percorso che non
    apre -- vedi `demo/webui_api_check.py`). Ritorna valori `None` se il
    file non si apre o il container non dichiara questi metadati (capita
    con alcuni codec/contenitori): il chiamante deve trattarli come "durata
    sconosciuta", non come zero.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return {"frame_count": None, "duration_s": None, "container_fps": None}
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        container_fps = cap.get(cv2.CAP_PROP_FPS) or None
        duration_s = (frame_count / container_fps) if frame_count and container_fps else None
        return {"frame_count": frame_count, "duration_s": duration_s, "container_fps": container_fps}
    finally:
        cap.release()


def build_player_kwargs(params: dict) -> dict:
    """Funzione pura: converte il dict di parametri inviato da JS negli
    argomenti attesi da `iter_pipeline_frames(...)`. Isolata da `Api` per
    essere testabile senza una finestra vera -- vedi
    `demo/webui_api_check.py`. Rispecchia ESATTAMENTE la logica di
    `gui/app.py::App._build_player()` (stessi default, stesso gating
    condizionale di mani/viso/reid/mediapipe-pose in base alla modalita'),
    cosi' il comportamento della GUI web non diverge da quella Tkinter.

    Richiede in `params`: "mode" ("segmentation"|"pose"|"both"), "source"
    (percorso video), "fps" (fps sorgente, numero o stringa numerica).
    Opzionali (default coerenti con app.py): "device" (se assente/None,
    QUESTA funzione lo lascia None -- e' `Api.build_player()`, non questa
    funzione pura, a risolverlo con `detect_default_device()`, cosi'
    `build_player_kwargs` resta testabile senza richiedere torch
    installato, vedi `demo/webui_api_check.py`), "scale" ("n"|"s"|"m",
    default "s"), "max_people" (int, stringa, o None/""), "with_hands",
    "with_eyes", "with_mouth", "with_eyebrows", "with_head_movement",
    "with_mediapipe_pose" (bool), "reid" (bool, abilita re-id/seg-reid se
    un max_people e' impostato), "seg_backend" ("yolo"|"sam31"|"sam2",
    default "yolo" -- gli ultimi due solo in modalita' Segmentation/Both,
    vedi segmentation/sam_backend.py: la GATING per device=cuda avviene
    lato JS (vedi `Api.detect_device()`) E qui in `Api.build_player()`
    come rete di sicurezza, non in questa funzione pura che non conosce
    ancora il device risolto), "sam_chunk_size", "sam_overlap" (int,
    solo con seg_backend != "yolo"), "sam_redetect_every" (int o None/"",
    solo con seg_backend != "yolo" -- ri-detection periodica dentro il
    chunk, vedi sam_backend.py), "sam_text_prompt" (stringa o None/"",
    solo con seg_backend == "sam31" -- prompt testuale SAM 3, vedi
    sam31_estimation.py).
    """
    mode = params.get("mode")
    if mode not in MODE_KEYS:
        raise ValueError(f"mode sconosciuto: {mode!r} (atteso 'pose'|'segmentation'|'both')")
    if not params.get("source"):
        raise ValueError("source mancante (nessun video caricato)")
    fps = float(params["fps"])

    max_people_raw = params.get("max_people")
    if max_people_raw in (None, ""):
        max_people = None
    else:
        max_people = int(max_people_raw)

    scale = str(params.get("scale", "s"))[0]

    pose_capable = mode in ("pose", "both")
    with_hands = pose_capable and bool(params.get("with_hands"))
    with_eyes = pose_capable and bool(params.get("with_eyes"))
    with_mouth = pose_capable and bool(params.get("with_mouth"))
    with_eyebrows = pose_capable and bool(params.get("with_eyebrows"))
    with_head_movement = pose_capable and bool(params.get("with_head_movement"))

    # re-id/seg-reid richiedono un numero di persone noto (tetto rigido):
    # senza max_people la spunta viene semplicemente ignorata, non fa
    # fallire l'avvio -- stesso comportamento di app.py.
    reid_requested = bool(params.get("reid")) and max_people is not None
    with_reid = reid_requested and mode in ("pose", "both")
    with_seg_reid = reid_requested and mode in ("segmentation", "both")

    # Pose per-maschera MediaPipe ha senso solo dove c'e' una maschera, cioe'
    # solo in modalita' Segmentation -- vedi pipeline_runner.py.
    with_mediapipe_pose = mode == "segmentation" and bool(params.get("with_mediapipe_pose"))

    # Backend di segmentazione (YOLO/SAM 3.1/SAM2): rilevante solo in
    # Segmentation/Both, "yolo" altrove (ignorato da iter_pipeline_frames se
    # mode="pose"). Il controllo "serve device=cuda" NON avviene qui (vedi
    # docstring sopra): questa funzione resta pura, il controllo vive in
    # Api.build_player().
    seg_backend = str(params.get("seg_backend") or "yolo")
    sam_chunk_size = int(params.get("sam_chunk_size") or 600)
    sam_overlap = int(params.get("sam_overlap") or 50)
    sam_chunk_store_dir = params.get("sam_chunk_store_dir") or None
    sam_redetect_every_raw = params.get("sam_redetect_every")
    sam_redetect_every = int(sam_redetect_every_raw) if sam_redetect_every_raw else None
    sam_text_prompt = params.get("sam_text_prompt") or None

    return dict(
        mode=mode, source=params["source"], fps=fps,
        device=params.get("device") or None,  # None = "auto-rileva a valle", vedi sopra
        pose_model=f"yolo26{scale}-pose.pt",
        with_hands=with_hands,
        with_eyes=with_eyes, with_mouth=with_mouth,
        with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
        with_reid=with_reid,
        seg_model=f"yolo26{scale}-seg.pt", with_seg_reid=with_seg_reid,
        with_mediapipe_pose=with_mediapipe_pose,
        max_people=max_people,
        seg_backend=seg_backend, sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
        sam_chunk_store_dir=sam_chunk_store_dir, sam_redetect_every=sam_redetect_every,
        sam_text_prompt=sam_text_prompt,
    )


def encode_frame_jpeg_b64(frame_bgr, max_width: int = 1600, quality: int = 80) -> str:
    """ndarray BGR -> data-URL base64 JPEG pronto per un `<img src="...">`.
    Isolata per essere testabile senza pywebview/camera (vedi
    `demo/webui_api_check.py`, che le passa un array sintetico). Ridimensiona
    solo verso il basso (mai verso l'alto) fino a `max_width`, stesso non-
    obiettivo di `MAX_DISPLAY_WIDTH` in gui/app.py: influenza solo cosa viene
    mostrato/trasferito, mai la risoluzione sorgente usata per l'inferenza.
    """
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        ratio = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * ratio)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Codifica JPEG fallita")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class _LatencyTracker:
    """Media mobile del tempo di elaborazione per frame, per le metriche
    'FPS elaborazione' / 'Latenza media' della status bar -- numeri reali
    calcolati dal tempo di orologio attorno a ogni `step_forward()`, non
    decorativi. Isolata da `Api` per essere testabile."""

    def __init__(self, window: int = 30):
        self._durations: deque[float] = deque(maxlen=window)

    def record(self, duration_s: float) -> None:
        self._durations.append(duration_s)

    @property
    def avg_latency_ms(self) -> float:
        if not self._durations:
            return 0.0
        return 1000.0 * sum(self._durations) / len(self._durations)

    @property
    def processing_fps(self) -> float:
        avg = self.avg_latency_ms
        return 1000.0 / avg if avg > 0 else 0.0


def build_status(*, runner_frame: RunnerFrame, cached_frame_count: int,
                  latency: _LatencyTracker, device: str, mode: str,
                  is_finished: bool, max_people: int | None = None,
                  total_frame_count: int | None = None,
                  total_duration_s: float | None = None) -> dict:
    """Dict di stato spedito a JS insieme a ogni frame -- ogni campo e' dato
    reale (vedi la nota 'Onesta' delle metriche' nel docstring del modulo):
    indice frame, fps/latenza di elaborazione da `_LatencyTracker`, numero
    di tracce attive da `RunnerFrame.people_count` (non `len(rows)`, vedi
    pipeline_runner.py), timecode da `RunnerFrame.now`, ed etichetta del
    device configurato invece di una finta telemetria GPU. `total_frame_count`
    / `total_duration_s` vengono da `probe_video_metadata` (metadati letti
    UNA VOLTA dal file, non ricalcolati qui) e possono essere None se il
    container non li dichiara -- il frontend deve trattarli come "totale
    sconosciuto", non zero."""
    return {
        "frame_index": cached_frame_count - 1,
        "total_frame_count": total_frame_count,
        "timecode_s": round(runner_frame.now, 2),
        "total_duration_s": total_duration_s,
        "people_count": runner_frame.people_count,
        "max_people": max_people,
        "processing_fps": round(latency.processing_fps, 1),
        "avg_latency_ms": round(latency.avg_latency_ms, 1),
        "device": device,
        "mode": mode,
        "is_finished": is_finished,
        "rows_this_frame": len(runner_frame.rows),
    }


class Api:
    """Bridge esposto a JS come `window.pywebview.api.<metodo>(...)`
    (chiamato come promise). Riusa `VideoPlayer`/`iter_pipeline_frames`
    invariati -- vedi il docstring del modulo."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # protegge player/_playing tra il
        # thread di riproduzione in background e le chiamate innescate da JS
        # (pywebview puo' eseguirle su thread diversi dal main)
        self.window = None  # impostato da set_window() dopo webview.create_window()
        self.video_path: str | None = None
        self.player: VideoPlayer | None = None
        self._device: str | None = None  # risolto in build_player() -- vedi li'
        self._mode = "segmentation"
        self._max_people: int | None = None
        self._playback_fps = 15.0
        self._latency = _LatencyTracker()
        self._playing = False
        self._play_thread: threading.Thread | None = None
        # metadati letti UNA VOLTA da pick_video_file() (vedi
        # probe_video_metadata): solo informativi, mai usati per decidere
        # cosa si puo' o non si puo' saltare (quello resta governato dalla
        # cache reale in VideoPlayer).
        self._total_frame_count: int | None = None
        self._total_duration_s: float | None = None

    def detect_device(self) -> dict:
        """Espone `detect_default_device()` a JS, chiamato una volta al
        caricamento della pagina (vedi app.js): usato SOLO per abilitare o
        disabilitare le opzioni SAM 3.1/SAM2 nel selettore backend
        (richiedono device='cuda', vedi segmentation/sam_backend.py) --
        `Api.build_player()` fa comunque il controllo definitivo lato
        server, questo e' solo per non far apparire nella UI un'opzione
        che fallirebbe subito."""
        return {"device": detect_default_device()}

    def set_window(self, window) -> None:
        """Chiamato dal launcher (webui_app.py) subito dopo
        `webview.create_window(..., js_api=api)`: serve un riferimento alla
        finestra per i dialoghi file e per `evaluate_js`."""
        self.window = window

    # ------------------------------------------------------------ dialoghi
    def pick_video_file(self) -> dict | None:
        """Apre il dialogo nativo e, se un file viene scelto, ne legge anche
        SUBITO i metadati (durata/numero di frame totali dichiarati dal
        container, vedi `probe_video_metadata`) -- cosi' JS puo' mostrare
        "corrente / totale" nel timecode e disegnare le tacche della
        timeline sull'intera durata fin da subito, senza aspettare che
        l'elaborazione arrivi in fondo al video."""
        import webview
        if self.window is None:
            return None
        result = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video files (*.mp4;*.mov;*.avi;*.mkv)", "All files (*.*)"),
        )
        if not result:
            return None
        self.video_path = result[0]
        meta = probe_video_metadata(self.video_path)
        self._total_frame_count = meta["frame_count"]
        self._total_duration_s = meta["duration_s"]
        return {"path": self.video_path, **meta}

    def pick_save_csv_path(self) -> str | None:
        import webview
        if self.window is None:
            return None
        result = self.window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename="session.csv",
            file_types=("CSV files (*.csv)",),
        )
        if not result:
            return None
        # a seconda della piattaforma/versione pywebview SAVE_DIALOG puo'
        # restituire una stringa o una tupla di un elemento
        return result if isinstance(result, str) else result[0]

    # -------------------------------------------------------- ciclo di vita
    def build_player(self, params: dict) -> dict:
        """Costruisce (o ricostruisce) il `VideoPlayer` da un dict di
        parametri mandato da JS -- vedi `build_player_kwargs`. Da chiamare
        ogni volta che l'utente cambia modello/feature/max-people, esattamente
        come "Restart" in app.py: un tracker gia' avviato non si puo'
        riconfigurare a meta' strada."""
        if self.video_path is None:
            return {"ok": False, "error": "No video loaded."}
        params = dict(params or {})
        params.setdefault("source", self.video_path)
        try:
            kwargs = build_player_kwargs(params)
        except (KeyError, ValueError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}
        if kwargs["device"] is None:
            # build_player_kwargs() lascia "device" a None quando JS non ne
            # specifica uno esplicito -- lo risolviamo QUI (non li', vedi il
            # suo docstring) cosi' quella resta una funzione pura testabile
            # senza torch installato. cuda se c'e' una GPU NVIDIA, altrimenti
            # mps su Apple Silicon, altrimenti cpu -- prima era fisso a
            # "mps", il che rompeva silenziosamente su una macchina CUDA.
            kwargs["device"] = detect_default_device()
        if kwargs["seg_backend"] != "yolo" and kwargs["device"] != "cuda":
            # rete di sicurezza server-side: il frontend gia' disabilita la
            # scelta SAM 3.1/SAM2 quando detect_device() non e' "cuda"
            # (vedi app.js), ma qui rifiutiamo comunque esplicitamente
            # invece di lasciare che Sam31Tracker/Sam2Tracker sollevino
            # un ValueError meno chiaro dentro il thread di riproduzione.
            return {"ok": False, "error": (
                f"Il backend '{kwargs['seg_backend']}' richiede una GPU CUDA "
                f"(device rilevato: '{kwargs['device']}')."
            )}
        with self._lock:
            self._playing = False
            self._device = kwargs["device"]
            self._mode = kwargs["mode"]
            self._max_people = kwargs["max_people"]
            self._latency = _LatencyTracker()
            self.player = VideoPlayer(generator_factory=lambda: iter_pipeline_frames(**kwargs))
        return {"ok": True}

    def play(self, fps: float | None = None) -> dict:
        if self.player is None:
            return {"ok": False, "error": "No player built yet."}
        with self._lock:
            if fps is not None:
                self._playback_fps = float(fps)
            if self._playing:
                return {"ok": True}
            self._playing = True
        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self._play_thread.start()
        return {"ok": True}

    def pause(self) -> dict:
        with self._lock:
            self._playing = False
        return {"ok": True}

    def step_forward(self) -> dict:
        with self._lock:
            self._playing = False
        return self._advance(back=False)

    def step_back(self) -> dict:
        with self._lock:
            self._playing = False
        return self._advance(back=True)

    def seek(self, index: int) -> dict:
        """Salto istantaneo dentro il prefisso gia' elaborato (vedi
        `VideoPlayer.seek`) -- usato dallo scrubber della timeline. Fuori
        dalla cache non fa nulla (nessuna elaborazione di recupero
        automatica da qui: il frontend, se l'utente clicca oltre il
        prefisso cache, deve invece richiamare step_forward()/play()
        ripetutamente, cosi' l'utente vede il recupero avanzare invece di
        restare bloccato in attesa di un salto che non puo' essere
        istantaneo)."""
        if self.player is None:
            return {"ok": False, "error": "No player built yet."}
        with self._lock:
            self._playing = False
        frame = self.player.seek(int(index))
        if frame is None:
            return {"ok": False, "error": "Index not yet processed."}
        return self._frame_payload(frame)

    def export_csv(self, path: str) -> dict:
        if self.player is None or self.player.cached_frame_count == 0:
            return {"ok": False, "error": "Nothing processed yet."}
        df = pd.DataFrame(self.player.all_rows())
        df.to_csv(path, index=False)
        return {"ok": True, "rows": len(df)}

    # ------------------------------------------------------------ interni
    def _advance(self, *, back: bool) -> dict:
        if self.player is None:
            return {"ok": False, "error": "No player built yet."}
        t0 = time.time()
        try:
            frame = self.player.step_back() if back else self.player.step_forward()
        except Exception as exc:
            # Un'eccezione qui e' quasi sempre un bug reale nel backend di
            # segmentazione/pose (es. un modello mancante, un formato dati
            # inatteso) -- la trasformiamo in un {"ok": False, "error": ...}
            # come le altre chiamate di questo modulo, invece di lasciarla
            # propagare come rifiuto di promise JS non gestito. Il
            # traceback completo resta comunque sul terminale per il debug.
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not back:
            self._latency.record(time.time() - t0)
        if frame is None:
            return {"ok": True, "frame": None,
                     "status": {"is_finished": True, "mode": self._mode}}
        return self._frame_payload(frame)

    def _frame_payload(self, frame: RunnerFrame) -> dict:
        status = build_status(
            runner_frame=frame, cached_frame_count=self.player.cached_frame_count,
            latency=self._latency, device=self._device, mode=self._mode,
            is_finished=self.player.is_exhausted, max_people=self._max_people,
            total_frame_count=self._total_frame_count,
            total_duration_s=self._total_duration_s,
        )
        return {"ok": True, "frame": encode_frame_jpeg_b64(frame.frame), "status": status}

    def _play_loop(self) -> None:
        """Gira su un thread in background (NON quello delle chiamate JS):
        replica la cadenza di `app.py::App._tick()` (prossimo step
        `delay_ms` dopo la fine dell'elaborazione precedente, non un timer a
        frequenza fissa -- stessa non-garanzia, vedi li'), ma spinge ogni
        frame al DOM da solo via `evaluate_js`, invece di restituirlo a chi
        chiama, perche' qui non c'e' nessuno che stia facendo polling."""
        while True:
            with self._lock:
                if not self._playing or self.player is None:
                    return
                fps = self._playback_fps
            t0 = time.time()
            try:
                frame = self.player.step_forward()
            except Exception as exc:
                # PRIMA questa eccezione uccideva il thread daemon in
                # silenzio: nessun errore in GUI, solo un traceback nel
                # terminale (facile da non notare mentre si guarda la
                # finestra, vedi la sessione di debug con SAMURAI che ha
                # scoperto questo, poi rimosso -- vedi sam2_estimation.py).
                # Ora si ferma la riproduzione e si manda
                # l'errore a JS -- onPipelineFrame() in app.js gia' sa
                # mostrare un {"ok": false, "error": ...} nella status pill,
                # non serve cambiare nulla lato frontend. Il traceback
                # completo resta comunque stampato sul terminale.
                traceback.print_exc()
                with self._lock:
                    self._playing = False
                self._push_error(exc)
                return
            self._latency.record(time.time() - t0)
            if frame is None:
                with self._lock:
                    self._playing = False
                self._push_status_only(finished=True)
                return
            self._push_frame(frame)
            delay_s = max(0.001, 1.0 / max(fps, 1e-3))
            time.sleep(delay_s)

    def _push_frame(self, frame: RunnerFrame) -> None:
        if self.window is None:
            return
        self._evaluate_js_safe(self._frame_payload(frame))

    def _push_status_only(self, *, finished: bool) -> None:
        if self.window is None or self.player is None:
            return
        self._evaluate_js_safe({"ok": True, "frame": None,
                                 "status": {"is_finished": finished, "mode": self._mode}})

    def _push_error(self, exc: Exception) -> None:
        if self.window is None:
            return
        self._evaluate_js_safe({"ok": False, "frame": None,
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "status": {"is_finished": True, "mode": self._mode}})

    def _evaluate_js_safe(self, payload: dict) -> None:
        # `window.onPipelineFrame` e' definito in webui/app.js: riceve lo
        # stesso payload {"ok", "frame", "status"} restituito dalle chiamate
        # dirette (step_forward/step_back), cosi' il rendering lato JS ha un
        # solo punto d'ingresso indipendentemente dalla fonte del frame.
        js = f"window.onPipelineFrame && window.onPipelineFrame({json.dumps(payload)})"
        try:
            self.window.evaluate_js(js)
        except Exception:
            # la finestra puo' essere stata chiusa mentre il thread di
            # riproduzione era ancora attivo: non fatale, il prossimo giro
            # del loop esce perche' _playing e' gia' False (chiuso -> pausa
            # innescata da JS) o perche' il player e' esaurito.
            pass
