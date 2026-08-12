"""
webui/api.py
=============
Bridge between the web frontend (webui/index.html + app.js, inside a
pywebview window) and the existing pipeline. It does NOT duplicate ANY
processing logic: it reuses `VideoPlayer` (gui/video_player.py) and
`iter_pipeline_frames` (gui/pipeline_runner.py) exactly like gui/app.py
does (the Tkinter GUI, which remains unchanged in the repo) -- this
module only handles: native file dialogs, a background playback thread,
JPEG base64 frame encoding, and computing the metrics shown in the
status bar.

Why a thread instead of `root.after()` like in app.py
-------------------------------------------------------------
Tkinter requires a single thread for UI updates (see app.py's
docstring); pywebview doesn't: the window is a separate process/webview
that receives updates via `window.evaluate_js(...)`, so playback can run
on a dedicated Python thread that calls `step_forward()` and pushes each
frame to the DOM, instead of having the JS side poll. The cadence
(waiting a "delay after the end of the previous processing", not a
fixed-frequency timer) deliberately replicates the same non-guaranteed
logic of `app.py._tick()` -- see there for why.

Honesty of the metrics
------------------------
`processing_fps` and `avg_latency_ms` are computed from a moving average
of real timings around `step_forward()` (see `_LatencyTracker`), not
made up. `people_count` comes from `RunnerFrame.people_count` (see
pipeline_runner.py: why NOT from `len(rows)`, which in "pose" mode can
stay at zero until the feature sliding window fills up). The device
shown is the configured label (e.g. "mps"), not real GPU usage
telemetry -- not reliably readable from pure Python/PyTorch on Apple
Silicon, see the discussion in the README.

The `webview` import is delayed (only inside the methods that need it),
so `build_player_kwargs` / `encode_frame_jpeg_b64` / `build_status` /
`_LatencyTracker` remain testable without pywebview installed -- see
`tests/webui_api_check.py`.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import pandas as pd

from gui.pipeline_runner import iter_pipeline_frames, RunnerFrame
from gui.video_player import VideoPlayer
from common.device import detect_default_device  # only the function: doesn't import torch
                                                    # until it's CALLED (see below)
from webui.local_media_server import LocalMediaServer
from pose.identity_manager import IdentityMode, SessionMode, wants_reid_engine
from pose.appearance_embedding import torchreid_available  # only the lightweight check: doesn't import torch

COMPARE_HTML = Path(__file__).resolve().parent / "compare.html"  # see Api.open_compare_window()

MODE_KEYS = {"segmentation", "pose", "both"}
POSE_BACKEND_KEYS = {"yolo", "mediapipe"}
IDENTITY_MODE_KEYS = {"frame_by_frame", "tracking_only", "tracking_reid"}
SESSION_MODE_KEYS = {"single", "multiple"}
# SAMURAI (yangchris11/samurai) is NOT a valid backend here: its Kalman
# filter assumes a single tracked object per session and crashes with
# multiple people (see requirements.txt) -- it stays listed in the UI
# only as a VISIBLY disabled option with the reason, never as a choice
# that would reach this far.


def probe_video_metadata(path: str) -> dict:
    """Reads ONLY the file's metadata (frame count and fps declared by
    the container) via `cv2.VideoCapture`, WITHOUT processing/decoding
    frames one by one -- this doesn't contradict "trackers are
    sequential, no arbitrary seeking" (see gui/video_player.py): there's
    no inference here, just reading a header, exactly what normal video
    players already do to show the duration before even starting
    playback. Used for the "current / total" timecode and for the
    timeline ticks over the entire known duration -- the PROCESSED
    prefix remains the only point where you can jump instantly (see
    `VideoPlayer.seek`), the total duration here is purely informational.

    Isolated from `Api` to be testable without pywebview (a real video
    file is enough, or in tests it's bypassed by passing a path that
    doesn't open -- see `tests/webui_api_check.py`). Returns `None`
    values if the file doesn't open or the container doesn't declare
    these metadata (happens with some codecs/containers): the caller
    must treat them as "unknown duration", not as zero.
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
    """Pure function: converts the parameter dict sent from JS into the
    arguments expected by `iter_pipeline_frames(...)`. Isolated from
    `Api` to be testable without a real window -- see
    `tests/webui_api_check.py`. Mirrors EXACTLY the logic of
    `gui/app.py::App._build_player()` (same defaults, same conditional
    gating of hands/face/reid/mediapipe-pose based on mode), so the web
    GUI's behavior doesn't diverge from the Tkinter one.

    Requires in `params`: "mode" ("segmentation"|"pose"|"both"), "source"
    (video path), "fps" (source fps, number or numeric string).
    Optional (defaults consistent with app.py): "device" (if
    absent/None, THIS function leaves it None -- it's `Api.build_player()`,
    not this pure function, that resolves it with
    `detect_default_device()`, so `build_player_kwargs` stays testable
    without requiring torch installed, see `tests/webui_api_check.py`),
    "scale" ("n"|"s"|"m", default "s"), "max_people" (int, string, or
    None/""), "with_hands", "with_eyes", "with_mouth", "with_eyebrows",
    "with_head_movement", "with_mediapipe_pose" (bool), "reid" (bool,
    enables re-id/seg-reid if a max_people is set), "seg_backend"
    ("yolo"|"sam31"|"sam2", default "yolo" -- the latter two only in
    Segmentation/Both mode, see segmentation/sam_backend.py: GATING for
    device=cuda happens on the JS side (see `Api.detect_device()`) AND
    here in `Api.build_player()` as a safety net, not in this pure
    function which doesn't yet know the resolved device), "sam_chunk_size",
    "sam_overlap" (int, only with seg_backend != "yolo"),
    "sam_redetect_every" (int or None/"", only with seg_backend != "yolo"
    -- periodic re-detection within the chunk, see sam_backend.py),
    "sam_text_prompt" (string or None/"", only with seg_backend == "sam31"
    -- SAM 3 text prompt, see sam31_estimation.py).
    """
    mode = params.get("mode")
    if mode not in MODE_KEYS:
        raise ValueError(f"unknown mode: {mode!r} (expected 'pose'|'segmentation'|'both')")
    if not params.get("source"):
        raise ValueError("missing source (no video loaded)")
    fps = float(params["fps"])

    max_people_raw = params.get("max_people")
    if max_people_raw in (None, ""):
        max_people = None
    else:
        max_people = int(max_people_raw)

    scale = str(params.get("scale", "s"))[0]

    # -- Pose: model INDEPENDENT of segmentation (see pipeline_runner.py)
    # -- "yolo" (default, full-frame + ByteTrack) or "mediapipe" (driven
    # by box/mask, see there for the exact wiring of each Task/backend
    # combination).
    pose_backend = str(params.get("pose_backend") or "yolo")
    if pose_backend not in POSE_BACKEND_KEYS:
        raise ValueError(f"unknown pose_backend: {pose_backend!r} (expected 'yolo'|'mediapipe')")

    # -- Identity & Re-identification (see identity_manager.py) --
    identity_mode_raw = str(params.get("identity_mode") or "tracking_reid")
    if identity_mode_raw not in IDENTITY_MODE_KEYS:
        raise ValueError(f"unknown identity_mode: {identity_mode_raw!r}")
    identity_mode = IdentityMode(identity_mode_raw)

    session_mode_raw = str(params.get("session_mode") or "multiple")
    if session_mode_raw not in SESSION_MODE_KEYS:
        raise ValueError(f"unknown session_mode: {session_mode_raw!r}")
    session_mode = SessionMode(session_mode_raw)

    flag_uncertain = bool(params.get("flag_uncertain", True))
    # Applies only to the keypoint-based path (ReIdentifier): see
    # pipeline_runner.iter_pipeline_frames for why it's ignored in
    # Segmentation-only mode (SegReIdentifier has no expiry by design).
    reid_max_lost_seconds = float(params.get("lost_identity_memory_s") or 180.0)

    pose_capable = mode in ("pose", "both")
    with_hands = pose_capable and pose_backend == "yolo" and bool(params.get("with_hands"))
    with_eyes = pose_capable and pose_backend == "yolo" and bool(params.get("with_eyes"))
    with_mouth = pose_capable and pose_backend == "yolo" and bool(params.get("with_mouth"))
    with_eyebrows = pose_capable and pose_backend == "yolo" and bool(params.get("with_eyebrows"))
    with_head_movement = pose_capable and pose_backend == "yolo" and bool(params.get("with_head_movement"))
    # Hands/face (additional MediaPipe FaceLandmarker/HandLandmarker on
    # top of the skeleton) remain wired ONLY on the pose_backend="yolo"
    # path (see live_demo.py): with pose_backend="mediapipe" the
    # skeleton already comes from MediaPipe Tasks PoseLandmarker, but
    # finger/blink-level hands/face aren't yet wired on that path (see
    # pipeline_runner._iter_pose_mediapipe for the honest limitation).

    # -- Identity & Re-identification: "tracking_reid" is the only mode
    # that instantiates ReIdentifier/SegReIdentifier (see
    # identity_manager.wants_reid_engine); "frame_by_frame"/"tracking_only"
    # use the underlying tracker's native id with no recovery after a
    # loss -- see their docstring in identity_manager.py for the honest
    # limitation on "frame_by_frame" (today it behaves like
    # "tracking_only": a true mode with NO continuity at all, not even
    # the tracker's native one, would require replacing
    # ByteTrack/SAM, out of scope here). NOTE, a real asymmetry between
    # the two engines (cause of a bug already seen: "Re-ID active" shown
    # in the UI but no re-association actually happening, see the
    # debugging session with the unstable-id screenshots on masks):
    #   - ReIdentifier (pose, keypoint) works WITHOUT max_people: it
    #     still does signature/color/position matching, it just doesn't
    #     have the last resort "force the match because the cap is
    #     reached". Doesn't make sense to disable it entirely just
    #     because the user hasn't filled in a field they don't need.
    #   - SegReIdentifier (segmentation, mask) instead REQUIRES
    #     max_people in the constructor (raises ValueError otherwise,
    #     see seg_reid.py): without a cap it can't guarantee "never more
    #     than N identities", so here the field remains mandatory
    #     (session_mode "single" forces it to 1 anyway, see
    #     identity_manager.suggested_max_people_policy).
    # Treating them the same way (as before this comment) SILENTLY
    # disabled pose-reid too whenever only segmentation required the
    # cap -- now they're two separate conditions.
    reid_mode_selected = wants_reid_engine(identity_mode)
    seg_reid_ready = reid_mode_selected and (max_people is not None or session_mode == SessionMode.SINGLE)

    # `with_reid` is used both by `_iter_pose` (pose_backend="yolo") and
    # `_iter_pose_mediapipe` (pose_backend="mediapipe") -- see
    # pipeline_runner.iter_pipeline_frames, which forwards the SAME
    # value to both "pose" branches; for mode="both" with
    # pose_backend="mediapipe" this value is simply ignored (that branch
    # only uses `with_seg_reid`, see there).
    with_reid = reid_mode_selected and mode in ("pose", "both")
    with_seg_reid = seg_reid_ready and mode in ("segmentation", "both")
    # NB: if `mode` includes segmentation and `seg_reid_ready` is False
    # (missing max_people, session_mode "multiple"), the sidebar must
    # flag it BEFORE reaching here -- see app.js::applyIdentityGating,
    # which mirrors this same condition client-side for the status
    # pill, so no round trip to the backend is needed just to know if a
    # field is filled in.

    # Pose inside the mask/box, MediaPipe: active when pose_backend is
    # "mediapipe" AND the mode calls for it (see pipeline_runner.py for
    # the exact wiring of each combination).
    with_mediapipe_pose = mode == "segmentation" and pose_backend == "mediapipe"

    # Segmentation backend (YOLO/SAM 3.1/SAM2): relevant only in
    # Segmentation/Both, "yolo" elsewhere (ignored by
    # iter_pipeline_frames if mode="pose"). The "requires device=cuda"
    # check does NOT happen here (see docstring above): this function
    # stays pure, the check lives in Api.build_player().
    seg_backend = str(params.get("seg_backend") or "yolo")
    sam_chunk_size = int(params.get("sam_chunk_size") or 600)
    sam_overlap = int(params.get("sam_overlap") or 50)
    sam_chunk_store_dir = params.get("sam_chunk_store_dir") or None
    sam_redetect_every_raw = params.get("sam_redetect_every")
    sam_redetect_every = int(sam_redetect_every_raw) if sam_redetect_every_raw else None
    sam_text_prompt = params.get("sam_text_prompt") or None

    # -- Advanced settings (new, collapsed section in the UI): previously
    # always left at `iter_pipeline_frames`'s default because not
    # exposed by any control -- now an explicit user value overrides
    # that default, unchanged if the field stays empty.
    conf_threshold = float(params.get("conf_threshold") or 0.1)
    tracker_config = str(params.get("tracker_config") or "bytetrack.yaml")

    # OSNet appearance embedding (new, optional -- see
    # pose/appearance_embedding.py and pipeline_runner.iter_pipeline_frames).
    # Relevant only if a re-id engine is actually active: if neither
    # `with_reid` nor `with_seg_reid` ends up True, the flag changes
    # nothing (no embedder is ever built, see
    # pipeline_runner._build_embedder), but we pass it through as-is
    # anyway -- it's not this pure function's job to recompute that
    # condition twice.
    use_appearance_embedding = bool(params.get("use_appearance_embedding"))

    return dict(
        mode=mode, source=params["source"], fps=fps,
        device=params.get("device") or None,  # None = "auto-detect downstream", see above
        pose_backend=pose_backend, pose_model=f"yolo26{scale}-pose.pt",
        with_hands=with_hands,
        with_eyes=with_eyes, with_mouth=with_mouth,
        with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
        with_reid=with_reid,
        seg_model=f"yolo26{scale}-seg.pt", with_seg_reid=with_seg_reid,
        with_mediapipe_pose=with_mediapipe_pose,
        max_people=max_people,
        session_mode=session_mode, flag_uncertain=flag_uncertain,
        reid_max_lost_seconds=reid_max_lost_seconds,
        seg_backend=seg_backend, sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
        sam_chunk_store_dir=sam_chunk_store_dir, sam_redetect_every=sam_redetect_every,
        sam_text_prompt=sam_text_prompt,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
        use_appearance_embedding=use_appearance_embedding,
    )


def encode_frame_jpeg_b64(frame_bgr, max_width: int = 1600, quality: int = 80) -> str:
    """BGR ndarray -> base64 JPEG data-URL ready for an `<img src="...">`.
    Isolated to be testable without pywebview/a camera (see
    `tests/webui_api_check.py`, which passes it a synthetic array).
    Resizes only downward (never upward) up to `max_width`, the same
    non-goal as `MAX_DISPLAY_WIDTH` in gui/app.py: it only affects
    what's shown/transferred, never the source resolution used for
    inference.
    """
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        ratio = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * ratio)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class _LatencyTracker:
    """Moving average of per-frame processing time, for the status
    bar's 'Processing FPS' / 'Average latency' metrics -- real numbers
    computed from wall-clock time around every `step_forward()`, not
    decorative. Isolated from `Api` to be testable."""

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
    """Status dict sent to JS alongside every frame -- every field is
    real data (see the 'Honesty of the metrics' note in the module
    docstring): frame index, processing fps/latency from
    `_LatencyTracker`, number of active tracks from
    `RunnerFrame.people_count` (not `len(rows)`, see pipeline_runner.py),
    timecode from `RunnerFrame.now`, and the configured device label
    instead of fake GPU telemetry. `total_frame_count` / `total_duration_s`
    come from `probe_video_metadata` (metadata read ONCE from the file,
    not recomputed here) and can be None if the container doesn't
    declare them -- the frontend must treat them as "unknown total", not
    zero."""
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
    """Bridge exposed to JS as `window.pywebview.api.<method>(...)`
    (called as a promise). Reuses `VideoPlayer`/`iter_pipeline_frames`
    unchanged -- see the module docstring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # protects player/_playing between
        # the background playback thread and JS-triggered calls
        # (pywebview may run them on threads other than the main one)
        self.window = None  # set by set_window() after webview.create_window()
        self.video_path: str | None = None
        self.player: VideoPlayer | None = None
        self._device: str | None = None  # resolved in build_player() -- see there
        self._mode = "segmentation"
        self._max_people: int | None = None
        self._playback_fps = 15.0
        self._source_fps: float | None = None  # set in build_player() -- the fps the
        # RUN was configured with, used by export_video() so the saved file plays at
        # the original speed (NOT self._playback_fps, which is only the live-preview
        # speed and can be changed independently via the speed selector, see play())
        self._latency = _LatencyTracker()
        self._playing = False
        self._play_thread: threading.Thread | None = None
        # metadata read ONCE by pick_video_file() (see
        # probe_video_metadata): informational only, never used to
        # decide what can or can't be skipped to (that remains governed
        # by the real cache in VideoPlayer).
        self._total_frame_count: int | None = None
        self._total_duration_s: float | None = None

    def detect_device(self) -> dict:
        """Exposes `detect_default_device()` to JS, called once on page
        load (see app.js): used ONLY to enable or disable the SAM
        3.1/SAM2 options in the backend selector (they require
        device='cuda', see segmentation/sam_backend.py) --
        `Api.build_player()` still does the definitive check server-side,
        this is only to avoid showing an option in the UI that would
        immediately fail.

        `torchreid_available` (new): same scheme, but for the
        "Appearance embedding (OSNet)" toggle in Advanced settings (see
        pose/appearance_embedding.py) -- 'torch'+'torchreid' are an
        optional heavy dependency, NOT guaranteed installed. A
        lightweight check (just an import attempt, no model loaded),
        not the definitive check: if the user forces the toggle anyway
        (impossible from the UI, but the pure function
        `build_player_kwargs` doesn't prevent it), `pipeline_runner._build_embedder`
        still raises a clear `ImportError` inside the playback thread."""
        return {"device": detect_default_device(), "torchreid_available": torchreid_available()}

    def set_window(self, window) -> None:
        """Called by the launcher (webui_app.py) right after
        `webview.create_window(..., js_api=api)`: needs a window
        reference for file dialogs and for `evaluate_js`."""
        self.window = window

    # ------------------------------------------------------------ dialogs
    def pick_video_file(self) -> dict | None:
        """Opens the native dialog and, if a file is chosen, IMMEDIATELY
        reads its metadata too (duration/total frame count declared by
        the container, see `probe_video_metadata`) -- so JS can show
        "current / total" in the timecode and draw the timeline ticks
        over the entire duration right away, without waiting for
        processing to reach the end of the video."""
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
        # depending on the platform/pywebview version SAVE_DIALOG can
        # return a string or a one-element tuple
        return result if isinstance(result, str) else result[0]

    def pick_save_video_path(self) -> str | None:
        """Same idea as `pick_save_csv_path()`, for `export_video()`
        (Michele, 2026-08: wanted the annotated video saved at the end
        of a run to compare different runs/parameter choices side by
        side, not just re-watch them live in the player one at a
        time)."""
        import webview
        if self.window is None:
            return None
        result = self.window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename="annotated_session.mp4",
            file_types=("MP4 video (*.mp4)",),
        )
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def open_compare_window(self) -> dict:
        """Opens the "compare up to 4 runs" second window (Michele,
        2026-08: load several already-exported annotated videos --
        `export_video()` above -- side by side and play them in sync to
        compare backend/parameter combinations). A plain HTML5 <video>
        player (compare.html/compare.js), NOT another VideoPlayer/
        pipeline instance: these are already-finished MP4 files, no
        inference involved, so there's nothing here to reuse from the
        main window's playback machinery.

        Uses its OWN `CompareApi` instance (below) as the JS bridge --
        deliberately NOT `self` (this same `Api` instance): each
        pywebview window's `js_api` object needs its own `.window`
        reference for its native file dialogs (see `set_window()`), and
        reusing `self` here would overwrite THIS window's `self.window`
        with the new one the moment `set_window()` fires on it, breaking
        every dialog in the main window afterwards.

        pywebview supports creating additional windows after `start()`
        has already been called (this is called from a JS-triggered
        callback, i.e. always after start()) -- see pywebview's
        multi-window docs."""
        import webview
        compare_api = CompareApi()
        window = webview.create_window(
            "Behaviour Vision Lab — Compare runs",
            url=str(COMPARE_HTML),
            js_api=compare_api,
            width=1280,
            height=860,
            min_size=(900, 600),
        )
        compare_api.set_window(window)
        return {"ok": True}

    # -------------------------------------------------------- lifecycle
    def build_player(self, params: dict) -> dict:
        """Builds (or rebuilds) the `VideoPlayer` from a parameter dict
        sent by JS -- see `build_player_kwargs`. Meant to be called
        every time the user changes model/feature/max-people, exactly
        like "Restart" in app.py: an already-started tracker can't be
        reconfigured midway."""
        if self.video_path is None:
            return {"ok": False, "error": "No video loaded."}
        params = dict(params or {})
        params.setdefault("source", self.video_path)
        try:
            kwargs = build_player_kwargs(params)
        except (KeyError, ValueError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}
        if kwargs["device"] is None:
            # build_player_kwargs() leaves "device" as None when JS
            # doesn't specify one explicitly -- we resolve it HERE (not
            # there, see its docstring) so that function stays a pure,
            # testable function without torch installed. cuda if there's
            # an NVIDIA GPU, otherwise mps on Apple Silicon, otherwise
            # cpu -- it used to be fixed to "mps", which silently broke
            # on a CUDA machine.
            kwargs["device"] = detect_default_device()
        if kwargs["seg_backend"] != "yolo" and kwargs["device"] != "cuda":
            # server-side safety net: the frontend already disables the
            # SAM 3.1/SAM2 choice when detect_device() isn't "cuda" (see
            # app.js), but here we still explicitly reject instead of
            # letting Sam31Tracker/Sam2Tracker raise a less clear
            # ValueError inside the playback thread.
            return {"ok": False, "error": (
                f"Backend '{kwargs['seg_backend']}' requires a CUDA GPU "
                f"(detected device: '{kwargs['device']}')."
            )}
        with self._lock:
            self._playing = False
            self._device = kwargs["device"]
            self._mode = kwargs["mode"]
            self._max_people = kwargs["max_people"]
            self._source_fps = kwargs["fps"]
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
        """Instant jump within the already-processed prefix (see
        `VideoPlayer.seek`) -- used by the timeline scrubber. Does
        nothing outside the cache (no automatic catch-up processing
        from here: the frontend, if the user clicks past the cache
        prefix, must instead repeatedly call step_forward()/play(), so
        the user sees the catch-up progress instead of being stuck
        waiting for a jump that can't be instant).

        The `self.player.seek()` call itself is now INSIDE `self._lock`
        (bug fix, Michele 2026-08: dragging/clicking the timeline
        several times fast crashed the whole app) -- see `_advance()`'s
        docstring for why."""
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
        with self._lock:  # snapshot the cache while nothing else can append to it, see _advance()
            if self.player.cached_frame_count == 0:
                return {"ok": False, "error": "Nothing processed yet."}
            rows = self.player.all_rows()
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        return {"ok": True, "rows": len(df)}

    def export_video(self, path: str) -> dict:
        """Writes every already-processed frame (overlay already drawn,
        see `RunnerFrame`/`gui/pipeline_runner.py`) to an MP4 file, in
        order, at the fps the run was actually configured with
        (`self._source_fps`, see `build_player()`). Lets Michele save
        each run's annotated output to compare parameter choices
        side by side later, not just live in the player -- can be
        called any time after at least one frame has been processed
        (doesn't require the run to have reached the end of the
        video)."""
        if self.player is None or self.player.cached_frame_count == 0:
            return {"ok": False, "error": "Nothing processed yet."}
        with self._lock:  # snapshot the cache, see export_csv()/_advance()
            if self.player.cached_frame_count == 0:
                return {"ok": False, "error": "Nothing processed yet."}
            frames = self.player.all_frames()
        height, width = frames[0].shape[:2]
        fps = self._source_fps or 15.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not writer.isOpened():
            return {"ok": False, "error": f"Could not open '{path}' for writing "
                                           f"(unsupported codec/container for this platform?)."}
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return {"ok": True, "frames": len(frames)}

    # ------------------------------------------------------------ internal
    def _advance(self, *, back: bool) -> dict:
        """BUG FIX (Michele, 2026-08): dragging/clicking the video
        timeline repeatedly in quick succession used to crash the app.
        Root cause: `VideoPlayer` wraps a single, plain (non-reentrant)
        Python generator (see gui/video_player.py) -- `self._lock` was
        only ever held around the `self._playing` flag, NOT around the
        actual `self.player.step_forward()`/`step_back()`/`seek()`
        calls, so two overlapping calls (e.g. two rapid timeline clicks
        each starting their own catch-up loop, or a click landing while
        the background `_play_loop` thread was mid-frame) could call
        `next(self._generator)` from two threads AT THE SAME TIME --
        `ValueError: generator already executing`, or worse, silent
        `self._cache`/`self._cursor` corruption. Now the player call
        itself is inside `self._lock` here, in `seek()`, and in
        `_play_loop()`, fully serializing every access to the shared
        player -- a second caller simply waits its turn instead of
        racing."""
        if self.player is None:
            return {"ok": False, "error": "No player built yet."}
        t0 = time.time()
        try:
            with self._lock:
                frame = self.player.step_back() if back else self.player.step_forward()
        except Exception as exc:
            # An exception here is almost always a real bug in the
            # segmentation/pose backend (e.g. a missing model, an
            # unexpected data format) -- we turn it into a
            # {"ok": False, "error": ...} like this module's other
            # calls, instead of letting it propagate as an unhandled JS
            # promise rejection. The full traceback still goes to the
            # terminal for debugging.
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not back:
            self._latency.record(time.time() - t0)
        if frame is None:
            return {"ok": True, "frame": None,
                     "status": {"is_finished": True, "mode": self._mode}}
        return self._frame_payload(frame)

    def _frame_payload(self, frame: RunnerFrame) -> dict:
        # Self-correct the known "total" if the container's declared
        # metadata (probe_video_metadata, read ONCE when the file was
        # picked) turns out to have UNDERESTIMATED the real length.
        # BUG (Michele, 2026-08): a real ~1:17 video was reported as
        # 0:59 total -- some containers/codecs (variable frame rate,
        # re-muxed/edited files) declare a frame_count/fps in their
        # header that doesn't match what's actually decodable, so
        # processing (which reads the real stream until it truly ends,
        # NOT bounded by this metadata) kept going right past the
        # displayed "end", leaving the timeline stuck at 100% and the
        # "current / total" timecode showing current > total. The
        # moment we've actually decoded past what the metadata claimed,
        # that's proof the estimate was wrong -- bump it up to match
        # reality. Only ever grows, never shrinks below what's
        # confirmed real, and does nothing if metadata was already
        # accurate (or unknown, i.e. None).
        if self.player is not None:
            cached = self.player.cached_frame_count
            if self._total_frame_count is not None and cached > self._total_frame_count:
                self._total_frame_count = cached
            if self._total_duration_s is not None and frame.now > self._total_duration_s:
                self._total_duration_s = frame.now
        status = build_status(
            runner_frame=frame, cached_frame_count=self.player.cached_frame_count,
            latency=self._latency, device=self._device, mode=self._mode,
            is_finished=self.player.is_exhausted, max_people=self._max_people,
            total_frame_count=self._total_frame_count,
            total_duration_s=self._total_duration_s,
        )
        return {"ok": True, "frame": encode_frame_jpeg_b64(frame.frame), "status": status}

    def _play_loop(self) -> None:
        """Runs on a background thread (NOT the one handling JS calls):
        replicates the cadence of `app.py::App._tick()` (next step
        `delay_ms` after the end of the previous processing, not a
        fixed-frequency timer -- same non-guarantee, see there), but
        pushes each frame to the DOM on its own via `evaluate_js`,
        instead of returning it to a caller, because there's no one
        polling here."""
        while True:
            with self._lock:
                if not self._playing or self.player is None:
                    return
                fps = self._playback_fps
            t0 = time.time()
            try:
                with self._lock:
                    frame = self.player.step_forward()
            except Exception as exc:
                # BEFORE, this exception silently killed the daemon
                # thread: no error in the GUI, just a traceback in the
                # terminal (easy to miss while watching the window, see
                # the SAMURAI debugging session that uncovered this,
                # later removed -- see sam2_estimation.py). Now playback
                # stops and the error is sent to JS -- app.js's
                # onPipelineFrame() already knows how to show a
                # {"ok": false, "error": ...} in the status pill, no
                # frontend change needed. The full traceback is still
                # printed to the terminal.
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
        # `window.onPipelineFrame` is defined in webui/app.js: it
        # receives the same {"ok", "frame", "status"} payload returned
        # by direct calls (step_forward/step_back), so JS-side rendering
        # has a single entry point regardless of the frame's source.
        js = f"window.onPipelineFrame && window.onPipelineFrame({json.dumps(payload)})"
        try:
            self.window.evaluate_js(js)
        except Exception:
            # the window may have been closed while the playback thread
            # was still active: not fatal, the next loop iteration exits
            # because _playing is already False (closed -> pause
            # triggered by JS) or because the player is exhausted.
            pass


class CompareApi:
    """`js_api` bridge for the "compare up to 4 runs" second window (see
    `Api.open_compare_window()` above and compare.html/compare.js).
    Deliberately tiny and separate from `Api`: the compare window plays
    already-exported MP4 files directly via HTML5 `<video>`, no
    inference/VideoPlayer/playback thread involved -- the only thing
    Python needs to do for it is the native "pick a video file" dialog
    and serving the picked file over a local HTTP server (see
    `local_media_server.py` for why a plain `file://` src doesn't work
    on every platform -- BUG, Michele 2026-08, Linux/CUDA machine:
    "Not allowed to load local resource"), everything else (loading,
    sync playback, scrubbing) happens entirely in compare.js against the
    browser's own video decoder."""

    def __init__(self) -> None:
        self.window = None  # set by Api.open_compare_window() right after create_window()
        self._media_server: LocalMediaServer | None = None

    def set_window(self, window) -> None:
        self.window = window

    def pick_video_path(self) -> dict | None:
        """Generic "pick any video file" dialog -- unlike
        `Api.pick_video_file()`, does NOT probe/return container
        metadata (frame count/fps/duration): the compare window doesn't
        need it, playback is handled natively by the <video> element.

        Returns `{"path": <real filesystem path>, "url": <http://127.0.0.1:.../...
        the <video> element should actually use as its src>}` -- the
        server is created lazily (once per CompareApi/window, reused
        across multiple pick_video_path() calls for the 4 slots) so
        opening the compare window with no intention of loading a video
        never spins up a server for nothing."""
        import webview
        if self.window is None:
            return None
        result = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video files (*.mp4;*.mov;*.avi;*.mkv)", "All files (*.*)"),
        )
        if not result:
            return None
        path = result[0] if not isinstance(result, str) else result
        if self._media_server is None:
            self._media_server = LocalMediaServer()
        return {"path": path, "url": self._media_server.url_for(path)}
