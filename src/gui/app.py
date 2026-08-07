"""
app.py
=======
Local GUI (Tkinter) for the pipeline: load a video, pick model/features from
the control panel, watch the overlay live in the same window, with
Play/Pause/Back/Forward controls.

Launch with `python gui_app.py` from inside `src/` (not `python gui/app.py`
directly) -- see gui_app.py's docstring for why.

Design choices (see also pipeline_runner.py and video_player.py):
- The video frame is embedded DIRECTLY in the Tkinter window (via PIL/
  ImageTk in a tk.Label), not a separate cv2.imshow window: on macOS,
  mixing Tkinter's event loop with OpenCV HighGUI's in two separate native
  windows risks GUI-thread conflicts.
- "Play" advances one frame at a time via `root.after(...)`, NEVER in a
  separate thread: Tkinter is not thread-safe, every UI update has to
  happen on the main thread.
- SAM3 is listed under model architecture but not selectable yet (it will
  run on the research group's dedicated GPU machines, not this Mac): if
  picked, it shows a note and reverts to YOLO -- see `_on_arch_change`.
- Hands/Face are only selectable for "Pose estimation" and "Both": under
  "Segmentation" there is no person <-> hands/face matching yet (needs the
  integration described in the README, not done) -- see `_on_mode_change`.

Not runnable in the sandbox used to develop the rest of the pipeline
(needs a real display and a real video). The cache/seek logic behind
Play/Forward/Back is verified separately, without a GUI, in
`tests/video_player_check.py`.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import pandas as pd
from PIL import Image, ImageTk

from gui.pipeline_runner import iter_pipeline_frames, RunnerFrame
from gui.video_player import VideoPlayer
from common.device import detect_default_device  # only the function: doesn't import torch
                                                    # until it's CALLED (see below)

ARCH_YOLO = "YOLO"
ARCH_SAM3 = "SAM3 (not available here — runs on dedicated GPUs, see README)"

MODE_SEGMENTATION = "Segmentation"
MODE_POSE = "Pose estimation"
MODE_BOTH = "Both"
MODE_LABEL_TO_KEY = {
    MODE_SEGMENTATION: "segmentation",
    MODE_POSE: "pose",
    MODE_BOTH: "both",
}

SCALE_LABELS = ["n (fastest)", "s (balanced)", "m (most stable)"]

MAX_DISPLAY_WIDTH = 1600  # resizing of the DISPLAYED frame only, not the source


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pose / Segmentation behaviour pipeline")

        self.video_path: str | None = None
        self.player: VideoPlayer | None = None
        self.playing = False
        self._photo = None  # kept alive here: Tkinter doesn't retain PhotoImage on its own

        self._build_widgets()
        self._on_mode_change()  # consistent initial state for the hands/face checkboxes

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        control = ttk.Frame(self.root, padding=10)
        control.grid(row=0, column=0, sticky="ns")
        video_frame = ttk.Frame(self.root, padding=10)
        video_frame.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # -- video --------------------------------------------------------
        self.video_label = ttk.Label(video_frame, text="Load a video to get started",
                                      anchor="center", background="#222", foreground="#ccc")
        self.video_label.pack(fill="both", expand=True)

        # -- source ---------------------------------------------------------
        ttk.Button(control, text="Load video...", command=self._on_load_video).pack(fill="x")
        self.source_label = ttk.Label(control, text="(no video loaded)", wraplength=220)
        self.source_label.pack(fill="x", pady=(2, 10))

        # -- fps --------------------------------------------------------------
        fps_row = ttk.Frame(control)
        fps_row.pack(fill="x", pady=(0, 10))
        ttk.Label(fps_row, text="Source FPS:").pack(side="left")
        self.fps_var = tk.StringVar(value="15")
        ttk.Entry(fps_row, textvariable=self.fps_var, width=6).pack(side="left", padx=(6, 0))

        # -- model architecture --------------------------------------------------
        ttk.Label(control, text="Model architecture:").pack(fill="x")
        self.arch_var = tk.StringVar(value=ARCH_YOLO)
        arch_combo = ttk.Combobox(control, textvariable=self.arch_var, state="readonly",
                                   values=[ARCH_YOLO, ARCH_SAM3])
        arch_combo.pack(fill="x", pady=(0, 10))
        arch_combo.bind("<<ComboboxSelected>>", self._on_arch_change)

        # -- pipeline mode ---------------------------------------------------------
        ttk.Label(control, text="Mode:").pack(fill="x")
        self.mode_var = tk.StringVar(value=MODE_SEGMENTATION)
        mode_combo = ttk.Combobox(control, textvariable=self.mode_var, state="readonly",
                                   values=[MODE_SEGMENTATION, MODE_POSE, MODE_BOTH])
        mode_combo.pack(fill="x", pady=(0, 10))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        # -- model size (n/s/m) ---------------------------------------------------
        ttk.Label(control, text="Model size:").pack(fill="x")
        self.scale_var = tk.StringVar(value=SCALE_LABELS[1])
        ttk.Combobox(control, textvariable=self.scale_var, state="readonly",
                     values=SCALE_LABELS).pack(fill="x", pady=(0, 10))

        # -- max number of people ------------------------------------------------
        max_row = ttk.Frame(control)
        max_row.pack(fill="x", pady=(0, 10))
        ttk.Label(max_row, text="Max number of people (optional):").pack(side="left")
        # No default -- the user must set it explicitly, not a silent
        # cap of "2" applied even if they don't touch it
        # (see the same fix in webui/index.html).
        self.max_people_var = tk.StringVar(value="")
        ttk.Entry(max_row, textvariable=self.max_people_var, width=4).pack(side="left", padx=(6, 0))

        # -- re-identification --------------------------------------------------------
        self.reid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control, text="Re-identification (hard cap on max people)",
                         variable=self.reid_var).pack(fill="x", pady=(0, 10))

        # -- hands/face features (enabled only for Pose/Both) ---------------------
        features = ttk.LabelFrame(control, text="Features (Pose estimation / Both only)")
        features.pack(fill="x", pady=(0, 10))
        self.hands_var = tk.BooleanVar(value=False)
        self.hands_check = ttk.Checkbutton(features, text="Hands", variable=self.hands_var)
        self.hands_check.pack(anchor="w")

        # Face is decomposed into four independent checkboxes (not one
        # "Face" toggle): each drives its own iter_live_frames() flag
        # (with_eyes/with_mouth/with_eyebrows/with_head_movement), all
        # sharing a single underlying FaceLandmarker call per frame -- see
        # live_demo.py's docstring on with_face_any for why that's cheap.
        ttk.Label(features, text="Face:").pack(anchor="w", pady=(4, 0))
        self.eyes_var = tk.BooleanVar(value=False)
        self.eyes_check = ttk.Checkbutton(features, text="  Eyes (blink rate)", variable=self.eyes_var)
        self.eyes_check.pack(anchor="w")
        self.mouth_var = tk.BooleanVar(value=False)
        self.mouth_check = ttk.Checkbutton(features, text="  Mouth (opening + repetitiveness)",
                                            variable=self.mouth_var)
        self.mouth_check.pack(anchor="w")
        self.eyebrows_var = tk.BooleanVar(value=False)
        self.eyebrows_check = ttk.Checkbutton(features, text="  Eyebrows (raise)", variable=self.eyebrows_var)
        self.eyebrows_check.pack(anchor="w")
        self.head_movement_var = tk.BooleanVar(value=False)
        self.head_movement_check = ttk.Checkbutton(
            features, text="  Head movement (yaw/pitch, shake/nod, shared attention)",
            variable=self.head_movement_var)
        self.head_movement_check.pack(anchor="w")

        # -- MediaPipe pose inside each tracked mask (Segmentation only) ------------
        seg_extras = ttk.LabelFrame(control, text="Segmentation extras")
        seg_extras.pack(fill="x", pady=(0, 10))
        self.mediapipe_pose_var = tk.BooleanVar(value=False)
        self.mediapipe_pose_check = ttk.Checkbutton(
            seg_extras, text="MediaPipe pose (inside each tracked mask)",
            variable=self.mediapipe_pose_var)
        self.mediapipe_pose_check.pack(anchor="w")

        # -- transport --------------------------------------------------------------
        transport = ttk.Frame(control)
        transport.pack(fill="x", pady=(10, 0))
        ttk.Button(transport, text="<< Back", command=self._on_back).pack(side="left", expand=True, fill="x")
        self.play_button = ttk.Button(transport, text="Play", command=self._on_play_pause)
        self.play_button.pack(side="left", expand=True, fill="x")
        ttk.Button(transport, text="Forward >>", command=self._on_forward_one).pack(side="left", expand=True, fill="x")

        ttk.Button(control, text="Restart (apply new parameters)",
                   command=self._on_restart).pack(fill="x", pady=(10, 0))
        ttk.Button(control, text="Save CSV...", command=self._on_save_csv).pack(fill="x", pady=(4, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(control, textvariable=self.status_var, wraplength=220,
                  foreground="#555").pack(fill="x", pady=(10, 0))

    # --------------------------------------------------------------- events
    def _on_arch_change(self, _event=None) -> None:
        if self.arch_var.get() == ARCH_SAM3:
            messagebox.showinfo(
                "SAM3 not available here",
                "SAM3 will run on the research group's dedicated GPU machines, "
                "not on this Mac. Staying on YOLO for now."
            )
            self.arch_var.set(ARCH_YOLO)

    def _on_mode_change(self, _event=None) -> None:
        pose_capable = self.mode_var.get() in (MODE_POSE, MODE_BOTH)
        state = "normal" if pose_capable else "disabled"
        self.hands_check.configure(state=state)
        for check in (self.eyes_check, self.mouth_check, self.eyebrows_check, self.head_movement_check):
            check.configure(state=state)
        if not pose_capable:
            self.hands_var.set(False)
            self.eyes_var.set(False)
            self.mouth_var.set(False)
            self.eyebrows_var.set(False)
            self.head_movement_var.set(False)

        # MediaPipe pose-per-mask only makes sense where there IS a mask,
        # i.e. Segmentation mode -- see gui/pipeline_runner.py / README for
        # why it's not wired into Pose estimation or Both (v1).
        seg_only = self.mode_var.get() == MODE_SEGMENTATION
        self.mediapipe_pose_check.configure(state="normal" if seg_only else "disabled")
        if not seg_only:
            self.mediapipe_pose_var.set(False)

    def _on_load_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path = path
        self.source_label.configure(text=path.rsplit("/", 1)[-1])
        self._teardown_player()
        self.status_var.set("Video loaded. Press Play to start.")

    def _teardown_player(self) -> None:
        self.playing = False
        self.player = None
        self.play_button.configure(text="Play")

    def _on_restart(self) -> None:
        self._teardown_player()
        self.status_var.set("Parameters updated: the next Play will start over from scratch.")

    def _build_player(self) -> VideoPlayer | None:
        if self.video_path is None:
            messagebox.showwarning("No video", "Load a video first.")
            return None
        try:
            fps = float(self.fps_var.get())
        except ValueError:
            messagebox.showerror("Invalid FPS", "Enter a number, e.g. 15.")
            return None

        max_people_raw = self.max_people_var.get().strip()
        try:
            max_people = int(max_people_raw) if max_people_raw else None
        except ValueError:
            messagebox.showerror("Invalid number of people", "Enter an integer, e.g. 2, or leave it empty.")
            return None

        mode_key = MODE_LABEL_TO_KEY[self.mode_var.get()]
        scale = self.scale_var.get()[0]  # "n"/"s"/"m": first character of the label

        pose_capable = mode_key in ("pose", "both")
        with_hands = pose_capable and self.hands_var.get()
        with_eyes = pose_capable and self.eyes_var.get()
        with_mouth = pose_capable and self.mouth_var.get()
        with_eyebrows = pose_capable and self.eyebrows_var.get()
        with_head_movement = pose_capable and self.head_movement_var.get()
        # re-id (pose.reid / segmentation.seg_reid) requires a known number
        # of people: without --max-people the checkbox is simply ignored
        # instead of making startup fail.
        with_reid = self.reid_var.get() and mode_key in ("pose", "both") and max_people is not None
        with_seg_reid = self.reid_var.get() and mode_key in ("segmentation", "both") and max_people is not None
        # MediaPipe pose-per-mask only makes sense where there's a mask to
        # apply it inside of -- see _on_mode_change (checkbox disabled
        # outside Segmentation) and pipeline_runner.py.
        with_mediapipe_pose = mode_key == "segmentation" and self.mediapipe_pose_var.get()

        # Detected every time the player is built (not at the top of the
        # module): this way the GUI stays launchable even without torch
        # installed, and detection only requires a torch call when it's
        # truly needed (Play/Restart) -- before, "mps" was hardcoded, which
        # silently broke on a machine with a CUDA GPU (see common/device.py).
        device = detect_default_device()
        kwargs = dict(
            mode=mode_key, source=self.video_path, fps=fps, device=device,
            pose_model=f"yolo26{scale}-pose.pt",
            with_hands=with_hands,
            with_eyes=with_eyes, with_mouth=with_mouth,
            with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
            with_reid=with_reid,
            seg_model=f"yolo26{scale}-seg.pt", with_seg_reid=with_seg_reid,
            with_mediapipe_pose=with_mediapipe_pose,
            max_people=max_people,
        )
        return VideoPlayer(generator_factory=lambda: iter_pipeline_frames(**kwargs))

    # -- transport ------------------------------------------------------------
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
        self.play_button.configure(text="Pause")
        self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        frame = self.player.step_forward()
        if frame is None:
            self.playing = False
            self.play_button.configure(text="Play")
            self.status_var.set(f"Video finished. Frames processed: {self.player.cached_frame_count}.")
            return
        self._display_frame(frame)
        try:
            fps = float(self.fps_var.get())
        except ValueError:
            fps = 15.0
        # Note: this does NOT guarantee constant-fps playback -- it simply
        # schedules "delay_ms after the previous frame finished processing"
        # (same non-guarantee already present in live_demo.py/pipeline.py
        # on file sources, see their docstrings). With a heavier-than-
        # expected model/feature combo, the player visibly slows down
        # instead of silently dropping frames.
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
            self.status_var.set("Already at the first frame.")

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
            self.status_var.set(f"Video finished. Frames processed: {self.player.cached_frame_count}.")

    def _on_save_csv(self) -> None:
        if self.player is None or self.player.cached_frame_count == 0:
            messagebox.showwarning("Nothing to save", "Process at least one frame before saving.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        df = pd.DataFrame(self.player.all_rows())
        df.to_csv(path, index=False)
        self.status_var.set(f"Saved {len(df)} rows to {path.rsplit('/', 1)[-1]}.")

    # -- rendering --------------------------------------------------------------
    def _display_frame(self, runner_frame: RunnerFrame) -> None:
        frame_rgb = cv2.cvtColor(runner_frame.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        if image.width > MAX_DISPLAY_WIDTH:
            ratio = MAX_DISPLAY_WIDTH / image.width
            image = image.resize((MAX_DISPLAY_WIDTH, int(image.height * ratio)))
        self._photo = ImageTk.PhotoImage(image=image)  # reference kept on self, see __init__
        self.video_label.configure(image=self._photo, text="")
        self.status_var.set(
            f"Frames cached: {self.player.cached_frame_count}  |  "
            f"t={runner_frame.now:.2f}s  |  data rows in this frame: {len(runner_frame.rows)}"
        )


def _maximize(root: tk.Tk) -> None:
    """Fills the screen on launch. `state('zoomed')` works on Windows and
    some Linux window managers; on macOS Tkinter doesn't support it, so we
    fall back to an explicit geometry matching the screen size -- keeps the
    title bar/traffic lights (unlike `-fullscreen`, which hides them and
    can make it awkward to reach other apps)."""
    try:
        root.state("zoomed")
    except tk.TclError:
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+0+0")


def run_gui() -> None:
    root = tk.Tk()
    _maximize(root)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
