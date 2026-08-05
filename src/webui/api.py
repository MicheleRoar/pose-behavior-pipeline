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
from collections import deque

import cv2
import pandas as pd

from gui.pipeline_runner import iter_pipeline_frames, RunnerFrame
from gui.video_player import VideoPlayer

MODE_KEYS = {"segmentation", "pose", "both"}


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
    Opzionali (default coerenti con app.py): "device" ("mps"), "scale"
    ("n"|"s"|"m", default "s"), "max_people" (int, stringa, o None/""),
    "with_hands", "with_eyes", "with_mouth", "with_eyebrows",
    "with_head_movement", "with_mediapipe_pose" (bool), "reid" (bool,
    abilita re-id/seg-reid se un max_people e' impostato).
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

    return dict(
        mode=mode, source=params["source"], fps=fps,
        device=params.get("device", "mps"),
        pose_model=f"yolo26{scale}-pose.pt",
        with_hands=with_hands,
        with_eyes=with_eyes, with_mouth=with_mouth,
        with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
        with_reid=with_reid,
        seg_model=f"yolo26{scale}-seg.pt", with_seg_reid=with_seg_reid,
        with_mediapipe_pose=with_mediapipe_pose,
        max_people=max_people,
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
                  is_finished: bool) -> dict:
    """Dict di stato spedito a JS insieme a ogni frame -- ogni campo e' dato
    reale (vedi la nota 'Onesta' delle metriche' nel docstring del modulo):
    indice frame, fps/latenza di elaborazione da `_LatencyTracker`, numero
    di tracce attive da `RunnerFrame.people_count` (non `len(rows)`, vedi
    pipeline_runner.py), timecode da `RunnerFrame.now`, ed etichetta del
    device configurato invece di una finta telemetria GPU."""
    return {
        "frame_index": cached_frame_count - 1,
        "timecode_s": round(runner_frame.now, 2),
        "people_count": runner_frame.people_count,
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
        self._device = "mps"
        self._mode = "segmentation"
        self._playback_fps = 15.0
        self._latency = _LatencyTracker()
        self._playing = False
        self._play_thread: threading.Thread | None = None

    def set_window(self, window) -> None:
        """Chiamato dal launcher (webui_app.py) subito dopo
        `webview.create_window(..., js_api=api)`: serve un riferimento alla
        finestra per i dialoghi file e per `evaluate_js`."""
        self.window = window

    # ------------------------------------------------------------ dialoghi
    def pick_video_file(self) -> str | None:
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
        return self.video_path

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
        with self._lock:
            self._playing = False
            self._device = kwargs["device"]
            self._mode = kwargs["mode"]
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
        frame = self.player.step_back() if back else self.player.step_forward()
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
            is_finished=self.player.is_exhausted,
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
            frame = self.player.step_forward()
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
