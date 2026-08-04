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

## Structure

```
pose-behavior-pipeline/
├── src/
│   ├── pose_estimation.py # YOLO-pose + ByteTrack wrapper
│   ├── features.py        # angles, velocity, symmetry, repetitiveness, synchrony
│   ├── gaze_head.py       # head pose, mouth/eye/eyebrow signals (MediaPipe FaceLandmarker)
│   ├── hands.py           # finger-level tracking (MediaPipe HandLandmarker)
│   ├── reid.py            # re-identification after exit/re-entry (body-shape signature + color)
│   ├── chuv_features.py   # reference-pipeline feature set, replicated in real time
│   ├── anonymize.py       # face blurring
│   ├── viz.py              # overlay drawing
│   ├── pipeline.py        # batch CLI (recorded video)
│   └── live_demo.py       # real-time CLI (Canon R8 / webcam)
└── demo/                  # camera-free tests for every module above
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Batch, on a recorded video:

```bash
cd src && python pipeline.py --source video.mp4 --fps 30 --out features.csv
```

Live, from a webcam or capture card:

```bash
cd src && python live_demo.py --source 0 --fps 30 --device mps --out live_session.csv
```

Optional flags stack freely: `--with-gaze` (head pose, mouth, eyes, eyebrows),
`--with-hands` (finger tracking), `--with-reid` (re-identification across
exit/re-entry), `--with-chuv-features` (reference feature set),
`--target-track-id N` (restrict face/hand signals to one person),
`--blur-faces`. See `--help` on each script for the full flag list and
defaults.

## Modules, briefly

- **`features.py`** — joint angles, movement energy, left/right symmetry,
  an FFT-based repetitive-motion score, child-caregiver proximity and
  motor synchrony.
- **`gaze_head.py`** — head pose (yaw/pitch/roll), a 2D shared-attention
  proxy, mouth aspect ratio, blink rate, eyebrow raise. Single uncalibrated
  camera, so treat as a rough proxy, not 3D gaze tracking.
- **`hands.py`** — 21 landmarks/hand, finger flexion, open/closed index,
  fingertip repetitiveness. Matched to the nearest YOLO wrist.
- **`reid.py`** — restores a person's ID after they fully leave and
  re-enter frame, using a clothing-invariant body-proportion signature
  (shoulders/hips/limbs + head geometry), optionally boosted by
  shirt/pants/hair color signals. Retries the match every frame with a
  rolling window instead of once, and only compares against people who
  went missing before the current track appeared (avoids false matches
  between someone present the whole time and someone else who leaves
  later). Verified on synthetic scenarios in `demo/reid_check.py` /
  `demo/reid_color_check.py`; default thresholds aren't validated on real
  footage and need on-camera calibration.
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
