# psifx SAM3 identity-persistence pipeline

Post-processing fix for [psifx](https://github.com/psifx/psifx)'s SAM3
cross-chunk identity tracking, built for CHUV (child neurodevelopment
research video). Vanilla psifx chunks a video for SAM3 tracking and
re-links object ids across chunk boundaries using a single-frame
greedy match -- fragile: a person occluded, off-screen, or lost
mid-propagation reappears under a brand-new id. This repo runs the
real `psifx` package unmodified (fidelity to what CHUV actually runs
in production) and repairs the fragmentation as a whole-video
post-process.

**Approach: SAM3 + OSNet + a learned heuristic, then pose.**

1. **SAM3** (real psifx's `Sam3TrackingTool`) produces the raw,
   fragmented per-chunk MaskDir.
2. **OSNet** appearance embeddings + a hue-histogram color signal give
   each mask fragment a signature.
3. **Heuristic merge** re-links fragments that are the same person
   split by a chunk boundary or a mid-chunk tracking loss (global
   Hungarian assignment), and separately resolves same-time
   overlapping tracks (two ids alive at once -- same body split into
   simultaneous fragments vs. a genuine second person) via fixed
   thresholds or a small classifier trained on labeled examples.
4. **Pose** (after segmentation): MediaPipe PoseLandmarker per tracked
   identity, then confidence gating + a One Euro Filter to stabilize
   the raw keypoints, with an optional overlay video to QA the result
   visually -- see the "Pose" usage section below.
5. **Depth** (after pose, Azure Kinect sessions only): reads each
   camera's own embedded depth stream from the ORIGINAL `.mkv` (never
   the transcoded clip -- it doesn't carry depth) and, using the
   device's factory depth<->color calibration, attaches a real 3D
   `(X, Y, Z)` point to every already-stabilized 2D keypoint, then
   computes per-person distance-traveled/velocity/movement-variability
   from that 3D skeleton -- see the "Depth" usage section below.
6. **Table-based cross-camera registration** (optional, only for
   sessions filmed by two Kinect cameras at once -- today just
   `10_individual_12`): registers `camera_b`'s 3D frame onto
   `camera_a`'s using the shared table as a common reference, so the
   two cameras' independently-computed 3D skeletons become directly
   comparable/mergeable -- see the "Table-based cross-camera
   registration" usage section below.

## Structure

```
pose-behavior-pipeline/
├── src/
│   ├── segmentation/                # step 1-3: segmentation + identity merge
│   │   ├── run_pipeline.py          # MAIN ENTRY POINT -- see Usage below
│   │   ├── run_sam3_baseline.py     # step 1: real-psifx SAM3 baseline tracking
│   │   ├── merging/                 # step 2-3: the merge_fragments algorithm
│   │   │   ├── merge_fragments.py   #   orchestrator (called by run_pipeline)
│   │   │   ├── mask_io.py           #   MaskDir I/O
│   │   │   ├── mask_utils.py        #   polygon/mask geometry
│   │   │   ├── signatures.py        #   OSNet + color appearance signatures
│   │   │   ├── reappearance_merge.py#   pass-1/2: track ends -> track starts
│   │   │   └── overlap_resolution.py#   zeroth pass: same-time overlaps
│   │   ├── classifier/              # builds the overlap classifier (optional)
│   │   │   ├── extract_overlap_candidates.py  # dumps unlabeled feature CSV
│   │   │   └── train_overlap_classifier.py    # fits weights from labeled CSV
│   │   └── tools/                   # manual inspection/QA, not in the main path
│   │       ├── subvideo.py          #   cuts a time window from video+MaskDir
│   │       ├── run_osnet_window.py  #   runs merge_fragments on that window
│   │       ├── overlay_subvideo.py  #   renders a labeled overlay for QA
│   │       └── check_overlap_iou.py #   raw IoU/distance between two ids
│   └── pose/                        # step 4: body keypoints (runs after segmentation)
│       ├── appearance_embedding.py  # OSNetEmbedder + EMA gallery update (used by segmentation/merging)
│       ├── keypoints.py             # COCO-17 schema constants (ported from *_legacy, see below)
│       ├── model_cache.py           # <repo>/models/ .task caching (ported from *_legacy, see below)
│       ├── extract_keypoints.py     # step 4a: MediaPipe PoseLandmarker per person per frame
│       ├── stabilization.py         # step 4b: confidence gating + One Euro Filter
│       ├── viz.py                   # skeleton/label drawing helpers (ported from *_legacy, see below)
│       ├── video_writer.py          # VP9-first annotated-video writer (ported from *_legacy, see below)
│       ├── visualize.py             # step 4c (optional): renders the skeleton overlay video, --overlay
│       ├── run_pose.py              # MAIN ENTRY POINT for pose -- see Usage below
│       ├── depth.py                 # step 5a: Azure Kinect depth -> 3D (X,Y,Z) per keypoint
│       ├── movement_metrics.py      # step 5b: distance traveled / velocity / variability from the 3D skeleton
│       ├── run_depth.py             # MAIN ENTRY POINT for depth -- see Usage below
│       └── table_calibration.py     # step 6 (optional): camera_a<->camera_b registration via the shared table
└── requirements.txt
```

**Provenance of the pose step.** This repo has a `pose-behavior-pipeline_legacy`
sibling: an older, more mature pipeline (YOLO-pose+ByteTrack as its main
backend, with a documented plan to "reconnect" a crop-based MediaPipe
PoseLandmarker once segmentation stability was verified -- exactly the
situation this repo is now in). `pose/keypoints.py` and `pose/model_cache.py`
are ported close to verbatim from it, and `extract_keypoints.py`'s schema
and conventions (COCO-17 output, `<repo>/models/` caching, pinned model
URL, `min_pose_detection_confidence`, a timestamp-monotonicity clamp,
0.15 bbox padding) were matched to its `pose/mediapipe_pose.py` rather
than reinvented independently, so results and code stay consistent with
it. `pose/viz.py` (skeleton/label drawing) and `pose/video_writer.py`
(VP9/WebM-first output, fixing a real "video written but silently won't
play in a browser" bug found in the legacy pipeline) are likewise ported
from `common/viz.py` and `common/video_writer.py`. `pose/anonymize.py`
(face blurring from head keypoints), `pose/chuv_features.py` (CHUV
feature engineering: joint angles, distances, symmetry, center of mass),
`pose/hands.py`, and `pose/gaze_head.py` were **not** ported -- flagged
as candidates for a follow-up, not done here (see Known limitations,
especially the anonymization gap).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchreid gdown tensorboard  # OSNet -- see requirements.txt's notes
pip install mediapipe  # pose step only -- see pose/extract_keypoints.py
```

**`psifx` itself** (not a PyPI package):

```bash
git clone https://github.com/psifx/psifx
cd psifx && pip install .
```

SAM3 checkpoint access is gated (Meta requires ethical-approval access
via Hugging Face). To avoid psifx's automatic Hugging Face auth flow,
clone the SAM3 checkpoint locally and point `SAM3_PATH` in
`psifx/utils/constants.py` at it -- see psifx's own docs
(https://psifx.github.io/psifx/) for the exact steps and CUDA/PyTorch
requirements (needs a CUDA GPU; not runnable on a Mac).

The PoseLandmarker model (`.task` file, pose step) is a **public**
download, not gated like the SAM3 checkpoint -- `pose/model_cache.py`
fetches it itself into `<repo>/models/` (not `~/.cache/...`: a fixed
location inside the project, independent of the cwd the script is
launched from -- see that module's docstring for the real bug this
fixes) on first use. The URL is **pinned to a specific version**
(`float16/1/...`, not `.../latest/...`) for reproducibility across runs.
The default variant is `lite` (fastest; matches the legacy pipeline's
own choice -- pass `--model-variant full` or `heavy` for more accuracy
at the cost of speed). If the auto-download can't reach the internet
from wherever this runs, grab it by hand instead and pass `--model-path`:

```bash
curl -o pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

**`ffmpeg`** (system binary, `run_pipeline.py` shells out to it for
the transcode step): install via your OS package manager.

**`pyk4a`** (depth step only, Azure Kinect sessions -- see the "Depth"
usage section below):

```bash
pip install pyk4a
```

This alone is **not enough** -- `pyk4a` is a thin wrapper around the
real Azure Kinect Sensor SDK (`libk4a`), which has to be installed
system-wide separately. Azure Kinect DK is an officially discontinued
Microsoft product, so there's no current first-party install guide for
a modern Ubuntu -- expect to need a community guide (search "install
libk4a Ubuntu 22.04/24.04" or similar) and some trial and error;
`pip install pyk4a` failing to import (`ImportError` mentioning
`libk4a` or `.so`) means the SDK isn't found, not that `pyk4a` itself
is broken. **Not yet verified on a real machine** -- run
`pip install pyk4a && python -c "import pyk4a"` first and treat a
clean import as the actual go/no-go signal for the depth step, before
relying on anything below.

## Usage

```bash
cd src
python -m segmentation.run_pipeline --video ~/Bureau/The\ Sense/Sessions/9_group_1_3/camera_a.mkv \
    --ss 00:22:34 --to 00:27:40 --device cuda
```

Runs, in one resumable pass (each step skipped if its output already
exists, unless `--overwrite`):

1. ffmpeg transcode/trim (always runs -- normalizes source `.mkv`
   codecs) -> `processed/<name>.mp4`
2. `run_sam3_baseline` -> `masks/<name>/` (raw MaskDir)
3. `merging.merge_fragments` -> `merged/<name>/` (merged MaskDir +
   `merge_report.json`)
4. psifx's `TrackingTool.visualize` -> `merged/<name>/overlay.mp4`

All rooted next to the source video, never a separate output root --
a whole-video run and any number of `--ss`/`--to` ranged runs on the
same source video coexist as sibling folders. Omit `--ss`/`--to` to
run the whole video.

`--no-osnet` disables the OSNet signal (color only). `--overlap-classifier
<path>` swaps in a weights JSON from `train_overlap_classifier.py`
instead of the two fixed overlap thresholds. See `run_pipeline.py
--help` for every parameter.

### Pose (after segmentation)

Once `segmentation.run_pipeline` has produced `merged/<name>/` for a
video/range, extract and stabilize body keypoints for each tracked
identity -- **same `--video`/`--ss`/`--to` you gave `run_pipeline`**,
so the two steps agree on `<name>` and this step finds the right
`merged/<name>/`:

```bash
cd src
python -m pose.run_pose --video ~/Bureau/The\ Sense/Sessions/9_group_1_3/camera_a.mkv \
    --ss 00:22:34 --to 00:27:40
```

Runs, in one resumable pass (same skip-if-output-exists policy as
`run_pipeline`):

1. `extract_keypoints` -> `pose/<name>/keypoints_raw.csv` -- MediaPipe
   PoseLandmarker (Tasks API), one crop per known `global_person_id`
   per frame. Identity comes from the merged MaskDir, never from
   MediaPipe's own multi-person detection (which isn't
   identity-persistent across frames) -- see the module docstring for
   why. Output is the **COCO-17** schema (`pose/keypoints.py`), not
   BlazePose's raw 33 landmarks: the 16 with no COCO equivalent
   (inner/outer eyes, mouth corners, fingers, heels, foot tips) are
   discarded, matching `pose-behavior-pipeline_legacy`'s convention so
   the two codebases' keypoint names/indices stay interchangeable.
2. `stabilization` -> `pose/<name>/keypoints_smoothed.csv` -- a
   keypoint below `--conf-threshold` (or simply missing that frame --
   occluded, nothing detected) is gated out; gaps of at most
   `--max-gap-frames` are linearly interpolated, longer gaps stay
   `NaN` (no invented values). Then a One Euro Filter smooths each
   keypoint's x/y over time.

`--model-variant {lite,full,heavy}` picks which PoseLandmarker model to
auto-download (see Setup); `--model-path` points at an
already-downloaded `.task` file instead. `--device {cpu,gpu}` is
MediaPipe's own delegate, unrelated to `segmentation.run_pipeline`'s
`--device` (SAM3's CUDA device) -- `cpu` is the default and the safe
choice (BlazePose is light enough to run on CPU; the GPU delegate is
OpenGL/EGL-based and can be finicky headless on a server).

**Tune `--min-cutoff`/`--beta` before trusting the defaults on a full
session.** `beta=0.0` (the flag's own default) is `OneEuroFilter`'s
conservative default, not a recommendation -- see the "IMPORTANT" note
in `pose/stabilization.py`'s docstring: with no speed-adaptive cutoff,
a fast-moving keypoint (a gesturing hand) can end up filtered *worse*
than the raw signal. Compare `keypoints_raw.csv` against
`keypoints_smoothed.csv` on one short clip first (plot a fast
keypoint's `x` over time, raw vs. smoothed) before running it on a
full session. `python -m pose.stabilization` runs a synthetic
self-test and saves a before/after plot as a starting sanity check --
not a substitute for tuning on real keypoints.

**Visual QA: `--overlay`.** Add `--overlay` to also render an
annotated video with the COCO-17 skeleton drawn on top, per identity, in
a distinct color -- the fastest way to actually *see* whether the
stabilization is helping (rather than just eyeballing numbers in a CSV):

```bash
python -m pose.run_pose --video ~/Bureau/The\ Sense/Sessions/9_group_1_3/camera_a.mkv \
    --overlay
```

Writes `pose/<name>/keypoints_overlay.webm` (falls back to `.mp4`/H.264
or `.mp4`/MPEG-4 if this machine's OpenCV/FFmpeg build has no VP9
encoder -- see `pose/video_writer.py`; the real written path/codec is
always printed). By default renders `keypoints_smoothed.csv`
(`--overlay-source raw` renders the ungated MediaPipe output instead):
any point/edge that came from `stabilization.py`'s short-gap
interpolation (no real detection that frame) is drawn in **muted gray**
instead of the person's own color, so it's immediately visible on
playback which parts of the skeleton are filled-in vs. actually
observed. Can also be run standalone on an existing CSV:
`python -m pose.visualize --video <processed_clip> --keypoints-csv
<keypoints_smoothed.csv> --out <path>`.

### Depth (after pose, Azure Kinect sessions only)

Once `pose.run_pose` has produced `pose/<name>/keypoints_smoothed.csv`
for a video/range, this step turns those 2D keypoints into a real 3D
skeleton and computes movement metrics from it -- **same
`--video`/`--ss`/`--to` you gave `run_pipeline`/`run_pose`**, and
**`--video` must be the ORIGINAL `.mkv`**, not `processed/<name>.mp4`
(the transcode drops the depth stream entirely -- see `pose/depth.py`'s
module docstring):

```bash
cd src
python -m pose.run_depth --video ~/Bureau/The\ Sense/Sessions/9_group_1_3/camera_a.mkv \
    --ss 00:22:34 --to 00:27:40
```

Runs, in one resumable pass (same skip-if-output-exists policy as
`run_pipeline`/`run_pose`):

1. `depth` -> `pose/<name>/keypoints_3d.csv` -- opens the original
   `.mkv`'s depth stream via `pyk4a` (see Setup above), seeks to `--ss`,
   then walks forward one depth capture per processed-clip frame
   (frame-accurate, since the transcode's `-ss`/`-to` come after `-i`
   and the source's own fps is preserved -- see `pose/depth.py`). For
   each `(frame, person, keypoint)` in `keypoints_smoothed.csv`, looks
   up that pixel in `transformed_depth_point_cloud` (the depth frame,
   already registered onto the color camera's own resolution/coordinate
   space using the device's factory calibration -- no manual stereo math
   needed). Every `Z` is either a genuine sensor reading or `NaN` --
   never guessed -- because the depth sensor's field of view is
   **narrower** than the color camera's (confirmed on this project's
   own recordings: Azure Kinect NFOV-2x2-binned depth mode, 320x288 vs.
   the color stream's 1280x720): a keypoint near an edge of frame, or an
   extended limb, can fall entirely outside what the depth sensor could
   see. `NaN` also covers ordinary depth-sensor dropouts (out of range,
   low-angle reflection, or the whole frame's depth capture missing) --
   the run prints how many keypoints hit each case.
2. `movement_metrics` -> `pose/<name>/movement_summary.csv` (one row
   per person: `total_distance_mm`, `mean_velocity_mm_s`,
   `velocity_std_mm_s` -- the movement-variability figure --
   `frames_with_root`, `n_outliers_rejected`, `outlier_fraction`,
   `total_frames`, `coverage_fraction`) and `pose/<name>/movement_per_frame.csv`
   (the velocity time series, with an `is_outlier` column). "Root"
   position is the `left_hip`/`right_hip` midpoint (COCO-17 has no
   dedicated pelvis keypoint). **Always read `coverage_fraction` and
   `outlier_fraction` alongside the distance/velocity numbers** -- a
   gap (missing pose, missing depth, or both) contributes nothing to
   `total_distance_mm` rather than assuming straight-line motion across
   it, which *under*-counts true distance whenever coverage is low;
   velocity at a frame divides by the actual elapsed time since the
   last valid frame, not a fixed `1/fps`. A single-frame jump implying
   a velocity above `--max-velocity-mm-s` (default 4000 mm/s, a fast
   jog) is treated as an unreliable depth reading, not real movement --
   rejected the same way a gap is (NaN, doesn't poison the next
   frame's displacement either), see `movement_metrics.py`'s module
   docstring "Rejecting single-frame velocity spikes" for why this was
   added and how the default was chosen.

`--overwrite` forces both steps to re-run. See `run_depth.py --help`
for the full parameter list.

**Run on a real session (2026-09-22, `10_individual_12`, a "seated"
session).** Confirmed on real data: `Z` isn't temporally smoothed the
way `x`/`y` are (see Known limitations), and a small fraction of frames
(well under 1% per person) had single-frame velocity spikes of several
m/s to tens of m/s -- physically impossible, clearly sensor noise, not
movement. Left unfiltered, that handful of frames distorted
`total_distance_mm` by 12-17% and (more importantly) `velocity_std_mm_s`
-- the movement-variability figure Matthew's brief specifically asked
for -- by 37-65%; the outlier rejection above fixes this. Separately,
and more encouragingly: a real, brief standing-and-walking episode in
that same session (confirmed against the actual footage, ~1:30-2:30)
shows up cleanly even after outlier rejection -- median velocity 1.9-2.9x
higher in that window than in the rest of the session, sustained across
many consecutive frames (not a one-frame spike) -- good evidence the
pipeline is picking up genuine movement, not just noise. The
group-vs-individual-session comparison this step was originally meant
to validate with (`9_group_1_3` vs. an individual session) still hasn't
been run, and is confounded anyway (a group session has inherently more
activity regardless of any one person's mobility) -- the within-session
before/after result above is arguably the cleaner validation of the two.

### Table-based cross-camera registration (optional, two-camera sessions)

For a session filmed by two Kinect cameras at once (today: only
`10_individual_12`), each camera's `pose.run_depth` output is a 3D
skeleton in THAT camera's own coordinate frame -- not directly
comparable to the other camera's. This step registers `camera_b`'s
frame onto `camera_a`'s using the shared LEGO table as a common
physical reference (Matthew's own suggestion) -- a rigid 3D point-set
fit (Kabsch/Umeyama) between a few table points read from each
camera's own depth, not a classic checkerboard stereo calibration (see
`pose/table_calibration.py`'s module docstring for why this is
simpler). Five steps, once per session:

```bash
cd src
# 1. Grab one clean color+depth capture per camera (pick a --t with a
#    clear, unobstructed view of the table)
python -m pose.table_calibration extract --video .../camera_a.mkv --out-prefix /tmp/cal/cam_a --t 30
python -m pose.table_calibration extract --video .../camera_b.mkv --out-prefix /tmp/cal/cam_b --t 30

# 2. Click the SAME physical table corners, in the SAME order, on each
#    camera's saved PNG (opens an interactive window)
python -m pose.table_calibration pick --color-image /tmp/cal/cam_a_color.png --out /tmp/cal/cam_a_points.json --n-points 4 --label "camera A"
python -m pose.table_calibration pick --color-image /tmp/cal/cam_b_color.png --out /tmp/cal/cam_b_points.json --n-points 4 --label "camera B"

# 3. Resolve those clicks to real 3D points via each camera's own depth
python -m pose.table_calibration lookup3d --pointcloud /tmp/cal/cam_a_pointcloud.npy --points /tmp/cal/cam_a_points.json --out /tmp/cal/cam_a_points3d.json
python -m pose.table_calibration lookup3d --pointcloud /tmp/cal/cam_b_pointcloud.npy --points /tmp/cal/cam_b_points.json --out /tmp/cal/cam_b_points3d.json

# 4. Fit the rigid transform (camera_b -> camera_a)
python -m pose.table_calibration register --source /tmp/cal/cam_b_points3d.json --target /tmp/cal/cam_a_points3d.json --out /tmp/cal/b_to_a_transform.json

# 5. Apply it to camera_b's keypoints_3d.csv -- now directly comparable to camera_a's own
python -m pose.table_calibration apply --keypoints-3d-csv .../camera_b/pose/.../keypoints_3d.csv --transform /tmp/cal/b_to_a_transform.json --out .../keypoints_3d_in_camera_a_frame.csv
```

`register` prints the fit's residual RMSE in mm -- a real, rigid table
surface should fit to a few mm; anything much larger (tens of mm)
means the clicked points don't actually correspond (wrong order, wrong
physical point, or a click landed on a spot with no valid depth --
`lookup3d` warns about that separately, re-pick those before
registering). **Only the rigid-transform math is tested so far**
(synthetic point sets with a known transform, exact recovery with no
noise, ~1.5mm residual with 1.5mm of injected noise, NaN rows correctly
preserved through the CSV apply step) -- the `extract`/`pick` steps
need a real two-camera session to try, not yet done. Not required for
the core movement-metrics validation above; only for comparing or
fusing the two camera views of the same session.

### Building/updating the overlap classifier

```bash
cd src
python -m segmentation.classifier.extract_overlap_candidates --masks-dir .../masks/<name> --out candidates.csv
# label candidates.csv by hand: 1 = same body/simultaneous fragments, 0 = genuinely different people
# (use tools/overlay_subvideo.py / tools/check_overlap_iou.py to inspect each candidate's frame range)
python -m segmentation.classifier.train_overlap_classifier --csv candidates.csv --out classifier.json
```

## Known limitations

- Needs a CUDA GPU (SAM3 + the OSNet embedder can run on CPU, but not
  in practical time for a full session).
- The overlap classifier is only as good as its labeled examples --
  `extract_overlap_candidates.py`'s CSVs need to keep growing across
  sessions as new failure modes show up.
- No face-blurring/anonymization step exists in this pipeline (removed
  with the old pipeline in the 2026-08 cleanup, and was never wired
  into an active path even before that). Needs to be reintroduced
  before any video of minors leaves a controlled environment.
  `pose-behavior-pipeline_legacy/src/pose/anonymize.py` already solves
  this (Gaussian blur over a head region estimated from `nose`/`eyes`/
  `ears` + shoulder-width scale) and now only depends on the just-ported
  `pose/keypoints.py` -- a natural next port, not done here since it
  wasn't explicitly asked for yet.
- **Pose step run end-to-end on a real session (2026-09-16, `9_group_1_3`,
  whole video, 6 identities, ~28.6k frames @ 15fps):** completed without
  errors, ~1.9M raw keypoint rows, 158 person/frame crops skipped as
  too-small (out of ~170k), densification found ~165k frames with no
  detection at all across all (person, keypoint) sequences, ~42.6k of
  those filled by short-gap interpolation. Numbers look sane but haven't
  been visually verified yet -- run with `--overlay` and actually watch
  a few minutes before trusting `keypoints_smoothed.csv` on further
  sessions, and definitely before tuning `--min-cutoff`/`--beta` away
  from the defaults.
- **`mediapipe` installed as `1.0.1`, not the `0.10.32` this code was
  verified against.** MediaPipe apparently crossed 1.0 at some point
  after this repo's Tasks API research -- it ran successfully end-to-end
  on the real session above, so the API surface used here
  (`PoseLandmarker`, `PoseLandmarkerOptions`, `detect_for_video`, etc.)
  is evidently still compatible, but this hasn't been diffed against
  MediaPipe's own 1.0 changelog. `pip install mediapipe` also pulled in
  `numpy 2.5.3` and `opencv-contrib-python 5.0.0.93` -- both **newer**
  than this repo's own `requirements.txt` pins for the segmentation half
  (`numpy<2`, and implicitly `opencv-python<5` since opencv-python 5.x
  forces numpy>=2). Worked fine for the pose step in isolation, but if
  `mediapipe`/`pandas` and the segmentation dependencies (`opencv-python`,
  `torch`, `torchreid`) end up in the **same** venv, whichever installs
  last currently wins the `numpy`/`cv2` version -- not yet verified that
  segmentation still behaves correctly under `numpy 2.x`/`opencv 5.x`.
  Safer for now: keep pose's `mediapipe`/`pandas` and segmentation's
  `opencv-python`/`torch`/`torchreid` in separate venvs until this is
  checked, or pin `mediapipe` explicitly (`pip install
  "mediapipe==0.10.32"`) if reproducing exactly what was verified here
  matters more than the newer release.
- Pose extraction crops per identity from the merged MaskDir and calls
  PoseLandmarker once per person per frame (`num_poses=1`) rather than
  once per frame for the whole scene -- trades a bit of extra compute
  for never having to re-match MediaPipe's own (non-identity-persistent)
  multi-person output back to the identities segmentation already
  solved for.
- One Euro Filter defaults (`--min-cutoff`/`--beta`) need tuning per
  keypoint/session, not just accepted as-is -- see the Pose usage
  section above.
- **Depth step (`pose/depth.py`, `pose/movement_metrics.py`,
  `pose/run_depth.py`) -- `pyk4a`+`libk4a` confirmed working on the
  real project machine (2026-09-22, despite Azure Kinect DK being
  discontinued -- installed via the Ubuntu 18.04 `.deb` packages, no
  depth engine needed for playback-only use), and the pipeline has now
  run end-to-end on one real session (`10_individual_12`, "seated").**
  Still only synthetic-data-tested for `table_calibration.py` (the
  rigid-transform math, not the interactive `extract`/`pick` steps,
  which need a real two-camera session -- see that file's usage
  section). The originally-planned "moves a lot vs. seated" cross-session
  comparison (`9_group_1_3` vs. an individual session) hasn't been run
  yet, and turned out to be confounded anyway (see the Depth usage
  section) -- a within-session before/after test (a real
  standing-and-walking episode inside `10_individual_12` itself) served
  as the cleaner validation instead.
- **`Z` (depth) is not temporally smoothed by `stabilization.py`.**
  That module's One Euro Filter only ever ran over `x`/`y` -- `Z` comes
  straight from the depth sensor's own per-pixel reading with no
  filtering. Confirmed on the real `10_individual_12` run that this
  produces occasional single-frame velocity spikes of several to tens
  of m/s (physically impossible) -- `movement_metrics.py`'s
  `max_velocity_mm_s` outlier rejection (added 2026-09-22, see its
  module docstring) catches and drops these specifically, and cut
  `velocity_std_mm_s` (the movement-variability figure) by 37-65% on
  that session with under 1% of frames rejected per person. This fixes
  the worst, most obviously-wrong artifacts but is a coarser tool than
  real temporal smoothing -- it doesn't reduce ordinary (non-outlier)
  frame-to-frame `Z` jitter the way a proper filter would. Whether
  that residual jitter matters enough to extend `stabilization.py` to
  3D is still worth checking once more real sessions have been run.

## Ethics & privacy

Video of minors in a clinical context requires face blurring as early
as possible (see Known limitations -- not currently implemented),
compliance with Swiss LPD and, where applicable, GDPR, and separation
of raw video from derived features with distinct retention policies.
