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
`seg_estimation.py`. The pose-based pipeline (`pipeline.py`, `live_demo.py`,
and everything that depends on keypoints: `features.py`, `gaze_head.py`,
`hands.py`, `reid.py`, `chuv_features.py`) stays in the repository, tested,
on hold.

## Structure

```
pose-behavior-pipeline/
├── src/
│   ├── seg_estimation.py  # YOLO26-seg + ByteTrack wrapper (ACTIVE, silhouettes only)
│   ├── segmentation_demo.py      # main CLI right now: overlay + CSV, no keypoints
│   ├── track_stability_check.py  # diagnostic: id count/lifespan, no overlay/CSV
│   ├── tracking_common.py # shared max_people cap logic
│   ├── pose_estimation.py # YOLO-pose + ByteTrack wrapper (ON HOLD, see status above)
│   ├── features.py        # angles, velocity, symmetry, repetitiveness, synchrony
│   ├── gaze_head.py       # head pose, mouth/eye/eyebrow signals (MediaPipe FaceLandmarker)
│   ├── hands.py           # finger-level tracking (MediaPipe HandLandmarker)
│   ├── reid.py            # re-identification after exit/re-entry (body-shape signature + color)
│   ├── chuv_features.py   # reference-pipeline feature set, replicated in real time
│   ├── anonymize.py       # face blurring
│   ├── viz.py              # overlay drawing
│   ├── pipeline.py        # batch CLI (recorded video, pose-based, on hold)
│   └── live_demo.py       # real-time CLI (Canon R8 / webcam, pose-based, on hold)
└── demo/                  # camera-free tests for every module above
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Active pipeline (segmentation, no keypoints):**

```bash
cd src && python segmentation_demo.py --source video.mp4 --fps 15 \
    --model yolo26s-seg.pt --tracker bytetrack_permissive.yaml \
    --conf-threshold 0.1 --max-people 2 --out session_seg.csv
```

Add `--no-window` to skip the live overlay (faster, log + CSV only). Same
flags as the diagnostic-only `track_stability_check.py`, plus the overlay
and the CSV (frame, track_id, bbox, mask centroid, mask area, box
confidence). `--max-people N`: known session headcount (2 for a 1v1
child-caregiver session, up to ~10 for a group) — keeps only the N most
confident detections per frame. `--tracker bytetrack_permissive.yaml`:
longer track memory and more tolerant thresholds for scenes with heavy ID
churn (overhead camera, fast motion, artificial lighting) — see that file
for details. Keep `--conf-threshold` at or below 0.1: ByteTrack's own
low-confidence recovery stage expects to see weak detections, a higher
value strips them out first.

**Pose-based pipeline (on hold — keypoints, all behavioural features):**

```bash
cd src && python pipeline.py --source video.mp4 --fps 30 --out features.csv
cd src && python live_demo.py --source 0 --fps 30 --device mps --out live_session.csv
```

Optional flags stack freely: `--with-gaze` (head pose, mouth, eyes, eyebrows),
`--with-hands` (finger tracking), `--with-reid` (re-identification across
exit/re-entry), `--with-chuv-features` (reference feature set),
`--target-track-id N` (restrict face/hand signals to one person),
`--blur-faces`, plus the same `--tracker`/`--conf-threshold`/`--max-people`
as above (with `--with-reid`, once `--max-people` identities are confirmed,
an unmatched re-entry is forced onto the closest missing identity instead
of minting a new one — the one deliberate exception to reid.py's "discount,
never force" rule, see `reid.py` for the safety guardrails). See `--help`
on each script for the full flag list and defaults.

## Modules, briefly

- **`features.py`** — joint angles, movement energy, left/right symmetry,
  an FFT-based repetitive-motion score, child-caregiver proximity and
  motor synchrony.
- **`gaze_head.py`** — head pose (yaw/pitch/roll), a 2D shared-attention
  proxy, mouth aspect ratio, blink rate, eyebrow raise. Single uncalibrated
  camera, so treat as a rough proxy, not 3D gaze tracking.
- **`hands.py`** — 21 landmarks/hand, finger flexion, open/closed index,
  fingertip repetitiveness. Matched to the nearest YOLO wrist.
- **`reid.py`** — restores a person's ID after they leave and re-enter
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
- **`chuv_features.py`** — real-time reimplementation of the reference
  pipeline's feature formulas (angles, distances, symmetry, COM, temporal
  derivatives), for testing that strategy without CUDA or clinical video.
  Feature engineering only — no trained classifier.

## Known limitations

- ByteTrack alone loses identity on full exit/re-entry or major appearance
  change; `reid.py` mitigates but doesn't eliminate this.
- All thresholds (activity, self-touch, blink, reid distance) are
  reasonable starting points, not clinically validated — calibrate on your
  own footage.
- COCO-17 (YOLO-pose) lacks toe/heel keypoints present in BODY-25, so a few
  reference-pipeline features can't be replicated here.
- No trained classifier included — only feature extraction.

## Ethics & privacy

Video of minors in a clinical context requires: face blurring as early as
possible in the pipeline (`anonymize.py`), compliance with the Swiss LPD
and, where applicable, GDPR, and separation of raw video from derived
features with distinct retention policies.
