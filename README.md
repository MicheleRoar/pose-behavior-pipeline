# Pose-based behavioural feature pipeline

Real-time and batch pipeline for extracting quantitative behavioural
markers from interaction video, using multi-person pose/segmentation
tracking and time-series feature engineering. Runs on Apple Silicon
(MPS, no CUDA) as a testing/development platform alongside a
CUDA/SAM3-based production pipeline for child neurodevelopment research.

Exploratory and technical, not validated diagnostic markers. Any
clinical use requires ethics-committee approval and validation on
annotated data.

**Current status:** active pipeline is segmentation-based
(`segmentation_demo.py`, YOLO26-seg + ByteTrack, no keypoints) — the
pose model alone produced too many spurious ids on real footage. Plan
is to reattach pose *inside* the tracked silhouette (see
`segmentation/seg_estimation.py`). The pose-based pipeline
(`pipeline.py`, `live_demo.py`, `pose/`) stays in the repo, on hold.

## Structure

```
pose-behavior-pipeline/
├── src/
│   ├── webui_app.py               # GUI launcher: cd src && python webui_app.py
│   ├── segmentation_demo.py       # ACTIVE main CLI: overlay + CSV, no keypoints
│   ├── track_stability_check.py   # ACTIVE diagnostic: id count/lifespan
│   ├── pipeline.py                # batch CLI (pose-based, on hold)
│   ├── live_demo.py               # real-time CLI (pose-based, on hold)
│   ├── gui/                       # shared player/dispatch logic behind the GUI
│   ├── webui/                     # "Behaviour Vision Lab" GUI (pywebview + HTML/CSS/JS)
│   ├── segmentation/              # ACTIVE library: silhouettes only, no keypoints
│   │   ├── seg_estimation.py      # YOLO26-seg + ByteTrack wrapper (default backend)
│   │   ├── seg_reid.py            # hard-capped id linking (position/color/shape)
│   │   ├── sam_backend.py         # shared SAM/SAM2 chunking logic
│   │   ├── sam31_estimation.py    # SAM 3.1 (needs CUDA, gated HF checkpoint)
│   │   └── sam2_estimation.py     # SAM2 vanilla (needs CUDA)
│   ├── pose/                      # ON HOLD library: everything keypoint-based
│   │   ├── keypoints.py           # COCO-17 index names, skeleton edges
│   │   ├── geometry.py            # shared angle/vector math
│   │   ├── pose_estimation.py     # YOLO-pose + ByteTrack wrapper
│   │   ├── features.py            # angles, velocity, symmetry, repetitiveness, synchrony
│   │   ├── gaze_head.py           # head pose, mouth/eye/eyebrow signals (MediaPipe)
│   │   ├── hands.py               # finger-level tracking (MediaPipe HandLandmarker)
│   │   ├── mediapipe_pose.py      # single-person pose (MediaPipe), applied per tracked mask
│   │   ├── identity_manager.py    # shared Hungarian batch re-id decision layer
│   │   ├── reid.py                # re-id after exit/re-entry (body-shape signature + color)
│   │   ├── appearance_embedding.py # OSNet embedding signal + EMA gallery
│   │   ├── chuv_features.py       # reference-pipeline feature set, replicated in real time
│   │   └── anonymize.py           # face blurring
│   ├── common/                    # shared: device detection, model-path resolution, drawing
│   └── configs/
│       ├── bytetrack.yaml             # Ultralytics' unmodified default, kept for reference
│       └── bytetrack_permissive.yaml  # tuned variant for hard scenes
├── models/                        # auto-downloaded weights — gitignored, not code
└── tests/                          # camera-free tests for every module above
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pywebview   # for the GUI
```

## GUI

```bash
cd src && python webui_app.py
```

Load a video, pick pipeline mode (Segmentation / Pose estimation /
Both), model size, max people, re-identification settings. Same
generators as the CLIs underneath (`gui/pipeline_runner.py`), so
behaviour never diverges — see that module and `webui/api.py` for the
full design.

## SAM 3.1 / SAM2 (CUDA only)

`--backend sam31`/`sam2` swap in
[SAM 3.1](https://github.com/facebookresearch/sam3) or
[SAM2](https://github.com/facebookresearch/sam2). SAM 3.1 checkpoints
are gated on Hugging Face (`facebook/sam3.1` — request access, then
`hf auth login`); SAM2's are public. Both need `device=cuda`, Python
3.12+, PyTorch 2.7+, CUDA 12.6+. Processed in overlapping chunks
(`--sam-chunk-size`/`--sam-overlap`, default 600/50) — see
`segmentation/sam_backend.py`'s docstring for the chunking design and
why SAMURAI was dropped for vanilla SAM2.

Not yet run against the real `sam3`/`sam2` API in this environment (no
CUDA here) — verified against a fake predictor only
(`demo/sam_backend_check.py`). Check `_init_state()` /
`_add_box_prompt()` / `_propagate()` against the real API before
relying on it.

## Usage (CLI)

**Active pipeline (segmentation, no keypoints):**

```bash
cd src && python segmentation_demo.py --source video.mp4 --fps 15 \
    --model yolo26s-seg.pt --tracker configs/bytetrack_permissive.yaml \
    --conf-threshold 0.1 --max-people 2 --out session_seg.csv
```

Model weights auto-download into `models/` on first use, regardless of
launch cwd. `--max-people N`: known headcount, caps detections per
frame. `--with-seg-reid` (needs `--max-people`) links ids to a fixed
identity pool instead of just capping — see `seg_reid.py`.
`--with-mediapipe-pose` adds a skeleton inside each tracked mask — see
`pose/mediapipe_pose.py`.

**Pose-based pipeline (on hold):**

```bash
cd src && python pipeline.py --source video.mp4 --fps 30 --out features.csv
cd src && python live_demo.py --source 0 --fps 30 --out live_session.csv
```

`--device` auto-detects cuda/mps/cpu. Flags stack freely:
`--with-eyes/mouth/eyebrows/head-movement`, `--with-hands`, `--with-reid`,
`--with-chuv-features`, `--target-track-id N`, `--blur-faces`, plus
`--tracker`/`--conf-threshold`/`--max-people`. See `--help` per script,
and `reid.py`'s docstring for re-identification details.

## Modules

- **`pose/features.py`** — joint angles, movement energy, symmetry,
  repetitive-motion score, proximity/synchrony.
- **`pose/gaze_head.py`** — head pose, shared-attention proxy, mouth
  ratio, blink rate, eyebrow raise (rough proxy, single camera).
- **`pose/hands.py`** — finger flexion, fingertip repetitiveness.
- **`pose/mediapipe_pose.py`** — single-person MediaPipe pose remapped
  onto COCO-17, applied inside a bbox crop.
- **`pose/reid.py`** — restores identity after exit/re-entry, via a
  body-proportion signature optionally boosted by color/position/
  embedding (never forced, strongest signal wins).
- **`pose/chuv_features.py`** — real-time reimplementation of the
  reference pipeline's feature formulas. No trained classifier.

## Known limitations

- ByteTrack alone loses identity on full exit/re-entry or major
  appearance change; `reid.py`/`seg_reid.py` mitigate, don't eliminate.
- Thresholds (activity, self-touch, blink, reid distance) are starting
  points, not clinically validated.
- COCO-17 lacks toe/heel keypoints present in BODY-25.
- No trained classifier — feature extraction only.

## Ethics & privacy

Video of minors in a clinical context requires: face blurring as early
as possible (`pose/anonymize.py` — not yet wired into the active
pipeline), compliance with Swiss LPD and, where applicable, GDPR, and
separation of raw video from derived features with distinct retention
policies.
