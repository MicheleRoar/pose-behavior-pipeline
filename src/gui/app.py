"""
app.py
=======
Interfaccia grafica locale (Tkinter) per la pipeline: carica un video,
sceglie modello/feature dal pannello di controllo, mostra l'overlay in
diretta nella stessa finestra, con controlli Play/Pausa/Avanti/Indietro.

Va lanciata con `python gui_app.py` da dentro `src/` (non `python
gui/app.py` direttamente) -- vedi il docstring di `gui_app.py` per il
perche'.

Scelte di design (vedi anche pipeline_runner.py e video_player.py):
- Frame video incorporato DIRETTAMENTE nella finestra Tkinter (via
  PIL/ImageTk in un tk.Label), non una finestra cv2.imshow separata: su
  macOS mescolare l'event loop di Tkinter con quello di OpenCV HighGUI in
  due finestre native diverse rischia conflitti a livello di thread della
  GUI.
- "Play" avanza un frame alla volta tramite `root.after(...)`, MAI in un
  thread separato: Tkinter non e' thread-safe, ogni aggiornamento della UI
  deve avvenire nel thread principale.
- SAM3 e' visibile nel menu a tendina dell'architettura ma non
  selezionabile per ora (gira sulle GPU dedicate del gruppo di ricerca, non
  su questo Mac): se scelto, mostra un avviso e torna automaticamente su
  YOLO -- vedi `_on_arch_change`.
- Mani/Viso sono selezionabili solo per le modalita' "Pose estimation" e
  "Entrambi": in "Segmentazione" non esiste ancora un aggancio persona <->
  mani/viso (serve prima l'integrazione descritta nel README, non ancora
  fatta) -- vedi `_on_mode_change`.

Non eseguibile nell'ambiente sandbox usato per sviluppare il resto della
pipeline (richiede un display grafico + un video reale). La logica di
cache/seek che sta dietro Play/Avanti/Indietro e' invece verificata
separatamente, senza GUI, in `demo/video_player_check.py`.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import pandas as pd
from PIL import Image, ImageTk

from gui.pipeline_runner import iter_pipeline_frames, RunnerFrame
from gui.video_player import VideoPlayer

ARCH_YOLO = "YOLO"
ARCH_SAM3 = "SAM3 (non disponibile qui — gira su GPU dedicate, vedi README)"

MODE_SEGMENTATION = "Segmentazione"
MODE_POSE = "Pose estimation"
MODE_BOTH = "Entrambi"
MODE_LABEL_TO_KEY = {
    MODE_SEGMENTATION: "segmentation",
    MODE_POSE: "pose",
    MODE_BOTH: "both",
}

SCALE_LABELS = ["n (piu' veloce)", "s (bilanciato)", "m (piu' stabile)"]

MAX_DISPLAY_WIDTH = 960  # ridimensionamento del frame MOSTRATO, non della sorgente


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pose / Segmentation behaviour pipeline")

        self.video_path: str | None = None
        self.player: VideoPlayer | None = None
        self.playing = False
        self._photo = None  # riferimento tenuto vivo: Tkinter non trattiene da solo le PhotoImage

        self._build_widgets()
        self._on_mode_change()  # stato iniziale coerente dei checkbox mani/viso

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        control = ttk.Frame(self.root, padding=10)
        control.grid(row=0, column=0, sticky="ns")
        video_frame = ttk.Frame(self.root, padding=10)
        video_frame.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # -- video --------------------------------------------------------
        self.video_label = ttk.Label(video_frame, text="Carica un video per iniziare",
                                      anchor="center", background="#222", foreground="#ccc")
        self.video_label.pack(fill="both", expand=True)

        # -- sorgente -------------------------------------------------------
        ttk.Button(control, text="Carica video...", command=self._on_load_video).pack(fill="x")
        self.source_label = ttk.Label(control, text="(nessun video caricato)", wraplength=220)
        self.source_label.pack(fill="x", pady=(2, 10))

        # -- fps --------------------------------------------------------------
        fps_row = ttk.Frame(control)
        fps_row.pack(fill="x", pady=(0, 10))
        ttk.Label(fps_row, text="FPS sorgente:").pack(side="left")
        self.fps_var = tk.StringVar(value="15")
        ttk.Entry(fps_row, textvariable=self.fps_var, width=6).pack(side="left", padx=(6, 0))

        # -- architettura del modello --------------------------------------------
        ttk.Label(control, text="Architettura modello:").pack(fill="x")
        self.arch_var = tk.StringVar(value=ARCH_YOLO)
        arch_combo = ttk.Combobox(control, textvariable=self.arch_var, state="readonly",
                                   values=[ARCH_YOLO, ARCH_SAM3])
        arch_combo.pack(fill="x", pady=(0, 10))
        arch_combo.bind("<<ComboboxSelected>>", self._on_arch_change)

        # -- modalita' pipeline ---------------------------------------------------
        ttk.Label(control, text="Modalita':").pack(fill="x")
        self.mode_var = tk.StringVar(value=MODE_SEGMENTATION)
        mode_combo = ttk.Combobox(control, textvariable=self.mode_var, state="readonly",
                                   values=[MODE_SEGMENTATION, MODE_POSE, MODE_BOTH])
        mode_combo.pack(fill="x", pady=(0, 10))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        # -- dimensione modello (n/s/m) ---------------------------------------------
        ttk.Label(control, text="Dimensione modello:").pack(fill="x")
        self.scale_var = tk.StringVar(value=SCALE_LABELS[1])
        ttk.Combobox(control, textvariable=self.scale_var, state="readonly",
                     values=SCALE_LABELS).pack(fill="x", pady=(0, 10))

        # -- numero massimo di persone ------------------------------------------------
        max_row = ttk.Frame(control)
        max_row.pack(fill="x", pady=(0, 10))
        ttk.Label(max_row, text="Numero persone (max):").pack(side="left")
        self.max_people_var = tk.StringVar(value="2")
        ttk.Entry(max_row, textvariable=self.max_people_var, width=4).pack(side="left", padx=(6, 0))

        # -- re-identificazione --------------------------------------------------------
        self.reid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control, text="Re-identificazione (tetto rigido su max persone)",
                         variable=self.reid_var).pack(fill="x", pady=(0, 10))

        # -- feature mani/viso (abilitate solo per Pose/Entrambi) ---------------------
        features = ttk.LabelFrame(control, text="Feature (solo Pose estimation / Entrambi)")
        features.pack(fill="x", pady=(0, 10))
        self.hands_var = tk.BooleanVar(value=False)
        self.hands_check = ttk.Checkbutton(features, text="Mani", variable=self.hands_var)
        self.hands_check.pack(anchor="w")
        self.face_var = tk.BooleanVar(value=False)
        self.face_check = ttk.Checkbutton(
            features, text="Viso (occhi, bocca, sopracciglia, movimento testa)",
            variable=self.face_var)
        self.face_check.pack(anchor="w")

        # -- trasporto --------------------------------------------------------------
        transport = ttk.Frame(control)
        transport.pack(fill="x", pady=(10, 0))
        ttk.Button(transport, text="<< Indietro", command=self._on_back).pack(side="left", expand=True, fill="x")
        self.play_button = ttk.Button(transport, text="Play", command=self._on_play_pause)
        self.play_button.pack(side="left", expand=True, fill="x")
        ttk.Button(transport, text="Avanti >>", command=self._on_forward_one).pack(side="left", expand=True, fill="x")

        ttk.Button(control, text="Riavvia (applica nuovi parametri)",
                   command=self._on_restart).pack(fill="x", pady=(10, 0))
        ttk.Button(control, text="Salva CSV...", command=self._on_save_csv).pack(fill="x", pady=(4, 0))

        self.status_var = tk.StringVar(value="Pronto.")
        ttk.Label(control, textvariable=self.status_var, wraplength=220,
                  foreground="#555").pack(fill="x", pady=(10, 0))

    # --------------------------------------------------------------- eventi
    def _on_arch_change(self, _event=None) -> None:
        if self.arch_var.get() == ARCH_SAM3:
            messagebox.showinfo(
                "SAM3 non disponibile qui",
                "SAM3 girera' sui computer con GPU dedicata del gruppo di ricerca, "
                "non su questo Mac. Per ora resta su YOLO."
            )
            self.arch_var.set(ARCH_YOLO)

    def _on_mode_change(self, _event=None) -> None:
        pose_capable = self.mode_var.get() in (MODE_POSE, MODE_BOTH)
        state = "normal" if pose_capable else "disabled"
        self.hands_check.configure(state=state)
        self.face_check.configure(state=state)
        if not pose_capable:
            self.hands_var.set(False)
            self.face_var.set(False)

    def _on_load_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Scegli un video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("Tutti i file", "*.*")],
        )
        if not path:
            return
        self.video_path = path
        self.source_label.configure(text=path.rsplit("/", 1)[-1])
        self._teardown_player()
        self.status_var.set("Video caricato. Premi Play per iniziare.")

    def _teardown_player(self) -> None:
        self.playing = False
        self.player = None
        self.play_button.configure(text="Play")

    def _on_restart(self) -> None:
        self._teardown_player()
        self.status_var.set("Parametri aggiornati: la prossima Play ripartira' da zero.")

    def _build_player(self) -> VideoPlayer | None:
        if self.video_path is None:
            messagebox.showwarning("Nessun video", "Carica prima un video.")
            return None
        try:
            fps = float(self.fps_var.get())
        except ValueError:
            messagebox.showerror("FPS non valido", "Inserisci un numero, es. 15.")
            return None

        max_people_raw = self.max_people_var.get().strip()
        try:
            max_people = int(max_people_raw) if max_people_raw else None
        except ValueError:
            messagebox.showerror("Numero persone non valido", "Inserisci un intero, es. 2, o lascia vuoto.")
            return None

        mode_key = MODE_LABEL_TO_KEY[self.mode_var.get()]
        scale = self.scale_var.get()[0]  # "n"/"s"/"m": primo carattere della label

        pose_capable = mode_key in ("pose", "both")
        with_hands = pose_capable and self.hands_var.get()
        with_face = pose_capable and self.face_var.get()
        # la re-id (pose.reid / segmentation.seg_reid) richiede un numero di
        # persone noto: senza --max-people il checkbox e' semplicemente
        # ignorato invece di far fallire l'avvio.
        with_reid = self.reid_var.get() and mode_key in ("pose", "both") and max_people is not None
        with_seg_reid = self.reid_var.get() and mode_key in ("segmentation", "both") and max_people is not None

        kwargs = dict(
            mode=mode_key, source=self.video_path, fps=fps, device="mps",
            pose_model=f"yolo26{scale}-pose.pt",
            with_hands=with_hands, with_face=with_face, with_reid=with_reid,
            seg_model=f"yolo26{scale}-seg.pt", with_seg_reid=with_seg_reid,
            max_people=max_people,
        )
        return VideoPlayer(generator_factory=lambda: iter_pipeline_frames(**kwargs))

    # -- trasporto ------------------------------------------------------------
    def _on_play_pause(self) -> None:
        if self.playing:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        if self.player is None:
            self.player = self._build_player()
            if self.player is None:
                return
        self.playing = True
        self.play_button.configure(text="Pausa")
        self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        frame = self.player.step_forward()
        if frame is None:
            self.playing = False
            self.play_button.configure(text="Play")
            self.status_var.set(f"Video terminato. Frame elaborati: {self.player.cached_frame_count}.")
            return
        self._display_frame(frame)
        try:
            fps = float(self.fps_var.get())
        except ValueError:
            fps = 15.0
        # Nota: questo NON garantisce una riproduzione a fps costanti --
        # scandisce semplicemente "delay_ms dopo la fine dell'elaborazione
        # del frame precedente" (stessa non-garanzia gia' presente in
        # live_demo.py/pipeline.py su file, vedi i loro docstring). Con
        # modelli/feature piu' pesanti del previsto il player rallenta
        # visibilmente invece di saltare frame silenziosamente.
        delay_ms = max(1, int(1000 / max(fps, 1e-3)))
        self.root.after(delay_ms, self._tick)

    def _on_back(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.player is None:
            return
        frame = self.player.step_back()
        if frame is not None:
            self._display_frame(frame)
        else:
            self.status_var.set("Gia' al primo frame.")

    def _on_forward_one(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.player is None:
            self.player = self._build_player()
            if self.player is None:
                return
        frame = self.player.step_forward()
        if frame is not None:
            self._display_frame(frame)
        else:
            self.status_var.set(f"Video terminato. Frame elaborati: {self.player.cached_frame_count}.")

    def _on_save_csv(self) -> None:
        if self.player is None or self.player.cached_frame_count == 0:
            messagebox.showwarning("Niente da salvare", "Elabora almeno un frame prima di salvare.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        df = pd.DataFrame(self.player.all_rows())
        df.to_csv(path, index=False)
        self.status_var.set(f"Salvate {len(df)} righe in {path.rsplit('/', 1)[-1]}.")

    # -- rendering --------------------------------------------------------------
    def _display_frame(self, runner_frame: RunnerFrame) -> None:
        frame_rgb = cv2.cvtColor(runner_frame.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        if image.width > MAX_DISPLAY_WIDTH:
            ratio = MAX_DISPLAY_WIDTH / image.width
            image = image.resize((MAX_DISPLAY_WIDTH, int(image.height * ratio)))
        self._photo = ImageTk.PhotoImage(image=image)  # riferimento tenuto in self, vedi __init__
        self.video_label.configure(image=self._photo, text="")
        self.status_var.set(
            f"Frame in cache: {self.player.cached_frame_count}  |  "
            f"t={runner_frame.now:.2f}s  |  righe dati in questo frame: {len(runner_frame.rows)}"
        )


def run_gui() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
