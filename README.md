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
│       └── run_depth.py             # MAIN ENTRY POINT for depth -- see Usage below
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
   `frames_with_root`, `total_frames`, `coverage_fraction`) and
   `pose/<name>/movement_per_frame.csv` (the velocity time series).
   "Root" position is the `left_hip`/`right_hip` midpoint (COCO-17 has
   no dedicated pelvis keypoint). **Always read `coverage_fraction`
   alongside the distance/velocity numbers** -- a gap (missing pose,
   missing depth, or both) contributes nothing to `total_distance_mm`
   rather than assuming straight-line motion across it, which
   *under*-counts true distance whenever coverage is low; velocity at a
   frame divides by the actual elapsed time since the last valid frame,
   not a fixed `1/fps`. Same "no made-up signal" principle as
   `stabilization.py`'s gap handling.

`--overwrite` forces both steps to re-run. See `run_depth.py --help`
for the full parameter list.

**Not yet run on a real session.** The two "test on a video where
people move a lot vs. one where people are seated" comparison cases
this step exists for haven't been run yet -- do that before trusting
these numbers for anything beyond a sanity check, and expect to tune
things (e.g. `stabilization.py`'s filter is not currently applied to
`Z`, only `x`/`y` -- see Known limitations).

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
  `pose/run_depth.py`) is new and only tested against synthetic data
  so far** (a fake `pyk4a` module + hand-built captures/point clouds --
  never real Kinect device data, per this project's privacy policy of
  not staging real clinical video/derived data off the recording
  machine). Specifically verified: depth-lookup NaN handling (sensor
  frame drop, out-of-depth-FOV pixel, in-FOV-but-no-return all counted
  and NaN'd correctly, not conflated with each other), the movement
  metrics correctly separate a synthetic "moves a lot" case from a
  "seated" one and handle gaps as documented (`coverage_fraction`,
  distance summed only across valid-to-valid spans), and
  `run_depth.py`'s path resolution/resumable-step skipping. **Not yet
  verified**: that `pyk4a` + the real `libk4a` SDK actually installs on
  this project's Ubuntu machine (Azure Kinect DK is discontinued -- see
  Setup); and the whole pipeline on one real "moves a lot" and one real
  "seated" session, the actual test this step exists for.
- **`Z` (depth) is not smoothed by `stabilization.py`.** That module's
  One Euro Filter only ever ran over `x`/`y` -- `Z` comes straight from
  the depth sensor's own per-pixel reading with no temporal filtering
  at all, so `movement_metrics.py`'s distance/velocity numbers will
  likely be noisier along the depth axis than in the image plane.
  Whether that matters enough to extend `stabilization.py` to 3D (or
  filter `keypoints_3d.csv` separately) is worth checking once real
  numbers exist -- not done here since it wasn't clear yet whether it's
  needed.

## Ethics & privacy

Video of minors in a clinical context requires face blurring as early
as possible (see Known limitations -- not currently implemented),
compliance with Swiss LPD and, where applicable, GDPR, and separation
of raw video from derived features with distinct retention policies.
