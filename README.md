# Pose-based behavioural feature pipeline

Real-time and batch pipeline for extracting quantitative behavioural
markers from interaction video, using multi-person pose estimation
(YOLO26-pose + ByteTrack) and time-series feature engineering on keypoints.
Runs on Apple Silicon (MPS, no CUDA) as a fast-iterating companion to a
CUDA/SAM3-based production pipeline for video-based child neurodevelopment
research — new modules and re-identification strategies get prototyped and
validated here on non-protected data before being ported over.

Motivated by two strands of literature: automated General Movements
Assessment for early cerebral-palsy detection, and 2D pose-based autism
screening from RGB video (Kojovic et al. 2021, Univ. of Geneva, 80.9%
classification accuracy). Features here are exploratory and technical, not
validated diagnostic markers — any clinical use requires ethics-committee
approval and validation on annotated data.

**Current status:** on real footage (overhead camera, fast motion,
artificial lighting) the pose model produced too many spurious ids (50+ in
a few minutes even with tracker/threshold tuning). The active pipeline is
temporarily `segmentation_demo.py` (YOLO26-seg + ByteTrack, silhouettes
only, no keypoints) to test whether a model that only has to outline a
person — instead of regressing 17 precise keypoints — tracks more
continuously. If confirmed, the plan is to reattach pose *inside* the
tracked silhouette rather than replace it permanently — see
`segmentation/seg_estimation.py`. The pose-based pipeline (`pipeline.py`,
`live_demo.py`, and everything under `pose/`, which everything keypoint-
based depends on) stays in the repository, tested, on hold.

## Structure

```
pose-behavior-pipeline/
├── src/
│   ├── gui_app.py                 # Tkinter GUI launcher: cd src && python gui_app.py
│   ├── webui_app.py               # Web GUI launcher (pywebview): cd src && python webui_app.py
│   ├── segmentation_demo.py       # ACTIVE main CLI: overlay + CSV, no keypoints
│   ├── track_stability_check.py   # ACTIVE diagnostic: id count/lifespan, no overlay/CSV
│   ├── pipeline.py                # batch CLI (recorded video, pose-based, on hold)
│   ├── live_demo.py               # real-time CLI (Canon R8 / webcam, pose-based, on hold)
│   ├── gui/                       # local Tkinter GUI (video file + parameter picker + live overlay)
│   │   ├── app.py                 # control panel + embedded video (Tkinter/PIL)
│   │   ├── video_player.py        # frame cache + seek (Back/Forward without re-inference)
│   │   └── pipeline_runner.py     # dispatches to iter_live_frames()/iter_segmentation_frames()
│   ├── webui/                     # "Behaviour Vision Lab" web GUI (pywebview + HTML/CSS/JS)
│   │   ├── api.py                 # pywebview bridge: reuses VideoPlayer/iter_pipeline_frames as-is
│   │   ├── index.html             # layout: header, sidebar cards, video panel, status bar
│   │   ├── style.css              # dark theme matching the mock
│   │   └── app.js                 # wires the DOM to window.pywebview.api.*
│   ├── segmentation/              # ACTIVE library: silhouettes only, no keypoints
│   │   ├── seg_estimation.py      # YOLO26-seg + ByteTrack wrapper
│   │   └── seg_reid.py            # hard-capped id linking (position/color/shape)
│   ├── pose/                      # ON HOLD library: everything keypoint-based
│   │   ├── keypoints.py           # COCO-17 index names, skeleton edges
│   │   ├── geometry.py            # shared angle/vector math
│   │   ├── pose_estimation.py     # YOLO-pose + ByteTrack wrapper
│   │   ├── features.py            # angles, velocity, symmetry, repetitiveness, synchrony
│   │   ├── gaze_head.py           # head pose, mouth/eye/eyebrow signals (MediaPipe)
│   │   ├── hands.py               # finger-level tracking (MediaPipe HandLandmarker)
│   │   ├── mediapipe_pose.py      # single-person pose (MediaPipe), applied per tracked mask
│   │   ├── reid.py                # re-id after exit/re-entry (body-shape signature + color)
│   │   ├── chuv_features.py       # reference-pipeline feature set, replicated in real time
│   │   └── anonymize.py           # face blurring
│   ├── common/                    # shared by both pipelines
│   │   ├── tracking_common.py     # shared max_people per-frame cap logic
│   │   ├── device.py              # auto-detect cuda/mps/cpu, default for --device everywhere
│   │   └── viz.py                 # overlay drawing
│   └── configs/
│       └── bytetrack_permissive.yaml  # tuned ByteTrack config for hard scenes
└── demo/                          # camera-free tests for every module above
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## GUI (recommended for exploratory use)

```bash
cd src && python gui_app.py
```

A local Tkinter window (starts maximized), fully in English: load a video
file, pick the pipeline mode (Segmentation / Pose estimation / Both), model
size, FPS, max number of people, re-identification on/off, and — only when
the mode is Pose estimation or Both — Hands and four independent Face
checkboxes (Eyes, Mouth, Eyebrows, Head movement — pick any subset, they
share a single underlying MediaPipe FaceLandmarker call per frame so
enabling more of them is nearly free). Then Play to watch the overlay live
in the same window, with Back/Forward to step through already-processed
frames instantly (from a cache, no re-inference) or resume live processing
past the cached point. SAM3 is listed under model architecture but not
selectable yet — it will run on the group's GPU machines, not this Mac;
picking it shows a note and reverts to YOLO. "Both" runs both pipelines
independently on the same source and draws the pose skeleton (+ hands/face
if enabled) directly on top of the segmentation overlay (same frame, not
side by side) — the two pipelines still don't share an identity, so the
"ID N" label stays the segmentation's; the pose skeleton has no label of
its own to avoid two conflicting numberings on the same person, see
`gui/pipeline_runner.py`. In Segmentation mode only, an extra "MediaPipe
pose (inside each tracked mask)" checkbox draws a skeleton inside each
already-tracked silhouette — see `pose/mediapipe_pose.py` below for what
this does and doesn't do. Internally the GUI calls the exact same
`iter_live_frames()` / `iter_segmentation_frames()` generators as the CLIs
below, not a reimplementation — CLI and GUI can't drift apart.

## Web GUI ("Behaviour Vision Lab")

```bash
pip install pywebview   # not in the default install, only this GUI needs it
cd src && python webui_app.py
```

A second, visually-polished GUI (dark theme, pill toggle switches, a
pipeline-flow diagram, live metrics) matching a provided mockup — an
alternative presentation layer over the *exact same* pipeline, not a
reimplementation: `webui/api.py` calls `VideoPlayer` and
`iter_pipeline_frames()` (the same `gui/video_player.py` /
`gui/pipeline_runner.py` used by the Tkinter GUI) unchanged. Built with
[pywebview](https://pywebview.flowrl.com/) (a native window wrapping a
local `webui/index.html` + `style.css` + `app.js`, with a Python `Api`
class exposed to JS as `window.pywebview.api.<method>(...)`) rather than a
separate HTTP server, to avoid an extra dependency and port/lifecycle
management.

Same controls as the Tkinter GUI (video source, mode, model size, max
people, re-identification, hands/face sub-checkboxes gated to Pose
estimation/Both, MediaPipe pose-per-mask gated to Segmentation), plus:

- A status bar with a pipeline-flow diagram built from the ACTUAL
  configured steps (e.g. `YOLO26s Segment → ByteTrack → Re-ID → MediaPipe
  Pose`), not a fixed label — see `app.js::updatePipelineFlow()`.
- Live metrics (current frame, processing FPS, average latency, active
  tracks) computed from real timing/data, not decorative: FPS/latency come
  from a rolling average of actual `step_forward()` wall-clock time
  (`webui/api.py::_LatencyTracker`), active tracks from
  `RunnerFrame.people_count` (reliable even in Pose mode before the sliding
  feature window fills — see `gui/pipeline_runner.py`).
- A timeline scrubber restricted to the already-processed prefix
  (`gui/video_player.py::VideoPlayer.seek()`, instant, no re-inference);
  clicking beyond it triggers sequential catch-up processing instead of an
  impossible instant jump — trackers are sequential/stateful, same
  constraint as Back/Forward in the Tkinter GUI, see that module's
  docstring.

Deliberate differences from the mock, not oversights: Face stays as four
independent checkboxes rather than one combined "Face" toggle (collapsing
them back would silently revert an earlier explicit request); there's no
functional volume control (this pipeline has no audio); the "GPU" badge
shows the configured device string (e.g. `mps`) rather than live
utilization telemetry, which isn't reliably queryable from pure
Python/PyTorch on Apple Silicon.

Pick whichever GUI fits: `gui_app.py` (Tkinter) has no extra dependency
beyond Pillow and a lighter startup; `webui_app.py` needs `pywebview` but
matches the mock closely. Both call the same generators underneath, so
behaviour never diverges between them — only the presentation layer
differs.

## Usage (CLI)

**Active pipeline (segmentation, no keypoints):**

```bash
cd src && python segmentation_demo.py --source video.mp4 --fps 15 \
    --model yolo26s-seg.pt --tracker configs/bytetrack_permissive.yaml \
    --conf-threshold 0.1 --max-people 2 --out session_seg.csv
```

Add `--no-window` to skip the live overlay (faster, log + CSV only). Same
flags as the diagnostic-only `track_stability_check.py`, plus the overlay
and the CSV (frame, track_id, bbox, mask centroid, mask area, box
confidence). `--max-people N`: known session headcount (2 for a 1v1
child-caregiver session, up to ~10 for a group) — keeps only the N most
confident detections per frame. `--tracker configs/bytetrack_permissive.yaml`:
longer track memory and more tolerant thresholds for scenes with heavy ID
churn (overhead camera, fast motion, artificial lighting) — see that file
for details. Keep `--conf-threshold` at or below 0.1: ByteTrack's own
low-confidence recovery stage expects to see weak detections, a higher
value strips them out first.

For a small, certain headcount (1-2 people), `--with-seg-reid` goes
further: it links every raw ByteTrack id to a fixed pool of `--max-people`
identities (position/color/shape of the silhouette, see `seg_reid.py`),
guaranteeing — not just discouraging — that no more than `--max-people`
ids are ever created in the whole session. A new raw id always tries a
threshold-based soft match to a recently-missing identity first (covers
the common case: brief occlusion, edge-of-frame exit/re-entry), even while
under the cap — without this, setting `--max-people` generously above the
real headcount (e.g. 20 as a safety margin for ~10 kids) meant the cap was
never actually reached, so soft-matching never kicked in and every
disappearance opened a new id instead of relinking (visible as a brief id
swap that "corrects itself" a moment later). Only once the cap is truly
reached does it fall back to `reid.py`'s optional `max_people` behavior
with no escape valve: an unmatched track is always bound to the closest
known identity regardless of signal strength. Read `seg_reid.py`'s
docstring for the trade-offs both of these accept.

Optionally, `--with-mediapipe-pose` applies MediaPipe Pose Landmarker in
SINGLE-person mode inside the crop of each already-tracked mask (not a
multi-person detector on the whole frame — MediaPipe has no built-in
frame-to-frame tracking, so identity is borrowed entirely from the
segmentation/`--with-seg-reid` tracking above; see `pose/mediapipe_pose.py`
for why). Draws the skeleton on top of the mask and adds joint-angle
columns (`pose_*`) to the CSV — no sliding-window features (movement
energy, repetitiveness, gaze, hands) yet, only per-frame angles. Requires
`pip install mediapipe` and a one-time model download:

```bash
curl -L -o pose_landmarker_lite.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
```

**Pose-based pipeline (on hold — keypoints, all behavioural features):**

```bash
cd src && python pipeline.py --source video.mp4 --fps 30 --out features.csv
cd src && python live_demo.py --source 0 --fps 30 --out live_session.csv
```

`--device` is optional everywhere in this project (CLIs and both GUIs):
left unset, it auto-detects the best available backend -- `cuda` if an
NVIDIA GPU is available, else `mps` on Apple Silicon, else `cpu` (see
`common/device.py`). Pass `--device cuda`/`--device mps`/`--device cpu`
explicitly to override.

Optional flags stack freely: `--with-eyes` (blink rate), `--with-mouth`
(opening + repetitiveness), `--with-eyebrows` (raise), `--with-head-movement`
(yaw/pitch, shake/nod, shared-attention proxy between two people) — all four
independent, share one MediaPipe FaceLandmarker call per frame so enabling
more of them barely costs anything extra — `--with-hands` (finger tracking),
`--with-reid` (re-identification across exit/re-entry), `--with-chuv-features`
(reference feature set), `--target-track-id N` (restrict face/hand signals to
one person), `--blur-faces`, plus the same
`--tracker`/`--conf-threshold`/`--max-people` as above (with `--with-reid`,
once `--max-people` identities are confirmed, an unmatched re-entry is forced
onto the closest missing identity instead of minting a new one — the one
deliberate exception to reid.py's "discount, never force" rule, see
`reid.py` for the safety guardrails). See `--help` on each script for the
full flag list and defaults.

## Modules, briefly

- **`pose/features.py`** — joint angles, movement energy, left/right
  symmetry, an FFT-based repetitive-motion score, child-caregiver
  proximity and motor synchrony.
- **`pose/gaze_head.py`** — head pose (yaw/pitch/roll), a 2D shared-attention
  proxy, mouth aspect ratio, blink rate, eyebrow raise. Single uncalibrated
  camera, so treat as a rough proxy, not 3D gaze tracking. In `live_demo.py`
  these are exposed as four independent flags (`--with-eyes`/`--with-mouth`/
  `--with-eyebrows`/`--with-head-movement`), not one on/off switch, even
  though they all come from a single FaceLandmarker call per frame.
- **`pose/hands.py`** — 21 landmarks/hand, finger flexion, open/closed
  index, fingertip repetitiveness. Matched to the nearest YOLO wrist.
- **`pose/mediapipe_pose.py`** — MediaPipe Pose Landmarker in single-person
  mode, applied inside a bbox crop rather than the whole frame, with the 33
  BlazePose landmarks remapped onto the same COCO-17 names used everywhere
  else (`pose/keypoints.py`) so existing feature code doesn't care which
  model produced the keypoints. Built specifically to reuse the
  segmentation pipeline's already-stable tracking (see its docstring for
  why this design — no multi-person tracking of its own) rather than
  building a second one from scratch. `MediaPipePoseByTrack` pools one
  landmarker instance per `track_id` instead of sharing a single one across
  everyone in the frame — MediaPipe's VIDEO running mode keeps per-instance
  temporal state and requires strictly increasing timestamps, so a shared
  instance called once per person per frame (same frame timestamp for each)
  crashes with `ValueError: Input timestamp must be monotonically
  increasing` as soon as a second person appears in the same frame.
- **`pose/reid.py`** — restores a person's ID after they leave and re-enter
  frame (or get briefly occluded in place, e.g. someone putting a jacket
  on them), using a clothing-invariant body-proportion signature
  (shoulders/hips/limbs + head geometry), optionally boosted by
  shirt/pants/hair color and by being in roughly the same spot shortly
  after disappearing. Retries the match every frame with a rolling window
  instead of once, and only compares against people who went missing
  before the current track appeared (avoids false matches between someone
  present the whole time and someone else who leaves later). All auxiliary
  signals only ever discount the distance, never force a match — the
  strongest single signal wins, they don't stack. Verified on synthetic
  scenarios in `demo/reid_check.py` / `demo/reid_color_check.py`; default
  thresholds aren't validated on real footage and need on-camera
  calibration.
- **`pose/chuv_features.py`** — real-time reimplementation of the reference
  pipeline's feature formulas (angles, distances, symmetry, COM, temporal
  derivatives), for testing that strategy without CUDA or clinical video.
  Feature engineering only — no trained classifier.

## Known limitations

- ByteTrack alone loses identity on full exit/re-entry or major appearance
  change; `pose/reid.py` (pose pipeline) / `segmentation/seg_reid.py`
  (segmentation pipeline) mitigate but don't eliminate this.
- All thresholds (activity, self-touch, blink, reid distance) are
  reasonable starting points, not clinically validated — calibrate on your
  own footage.
- COCO-17 (YOLO-pose) lacks toe/heel keypoints present in BODY-25, so a few
  reference-pipeline features can't be replicated here.
- No trained classifier included — only feature extraction.

## Ethics & privacy

Video of minors in a clinical context requires: face blurring as early as
possible in the pipeline (`pose/anonymize.py` — not currently wired into
the active segmentation pipeline, see status above), compliance with the
Swiss LPD and, where applicable, GDPR, and separation of raw video from
derived features with distinct retention policies.
