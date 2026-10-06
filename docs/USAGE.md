# Usage

```bash
cd src
python -m segmentation.run_pipeline --video <path/to/session>/camera_a.mkv \
    --ss 00:22:34 --to 00:27:40 --device cuda --crop-margin 25
```

Runs, in one resumable pass (each step skipped if its output already
exists, unless `--overwrite`):

1. ffmpeg transcode/trim (always runs -- normalizes source `.mkv`
   codecs) -> `processed/<name>.mp4`
2. `run_sam3_baseline` -> `masks/<name>/` (raw MaskDir)
3. `merging.merge_fragments` -> `merged/<name>/` (merged MaskDir +
   `merge_report.json`)
4. psifx's `TrackingTool.visualize` -> `merged/<name>/overlay.mp4`
5. `tools.crop_outputs` -> `merged/<name>_cropped<margin>/` (crops
   camera_a's border artifact off the merged MaskDir + source video;
   purely additive, never touches `merged/<name>/`; `--no-crop` to skip)
6. overlay on the cropped output -> `merged/<name>_cropped<margin>/overlay.mp4`

All rooted next to the source video, never a separate output root --
a whole-video run and any number of `--ss`/`--to` ranged runs on the
same source video coexist as sibling folders. Omit `--ss`/`--to` to
run the whole video.

`--no-osnet` disables the OSNet signal (color only). `--overlap-classifier
<path>` swaps in a weights JSON from `train_overlap_classifier.py`
instead of the two fixed overlap thresholds. `--crop-margin <px>`
(default 25) controls step 5's crop. See `run_pipeline.py --help` for
every parameter.

## Pose (after segmentation)

Once `segmentation.run_pipeline` has produced `merged/<name>/` for a
video/range, extract and stabilize body keypoints for each tracked
identity -- **same `--video`/`--ss`/`--to` you gave `run_pipeline`**,
so the two steps agree on `<name>` and this step finds the right
`merged/<name>/`:

```bash
cd src
python -m pose.run_pose --video <path/to/session>/camera_a.mkv \
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
auto-download (see [SETUP.md](SETUP.md)); `--model-path` points at an
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
python -m pose.run_pose --video <path/to/session>/camera_a.mkv \
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

## Depth (after pose, Azure Kinect sessions only)

Once `pose.run_pose` has produced `pose/<name>/keypoints_smoothed.csv`
for a video/range, this step turns those 2D keypoints into a real 3D
skeleton and computes movement metrics from it -- **same
`--video`/`--ss`/`--to` you gave `run_pipeline`/`run_pose`**, and
**`--video` must be the ORIGINAL `.mkv`**, not `processed/<name>.mp4`
(the transcode drops the depth stream entirely -- see `pose/depth.py`'s
module docstring):

```bash
cd src
python -m pose.run_depth --video <path/to/session>/camera_a.mkv \
    --ss 00:22:34 --to 00:27:40
```

Runs, in one resumable pass (same skip-if-output-exists policy as
`run_pipeline`/`run_pose`):

1. `depth` -> `pose/<name>/keypoints_3d.csv` -- opens the original
   `.mkv`'s depth stream via `pyk4a` (see [SETUP.md](SETUP.md)), seeks to `--ss`,
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
way `x`/`y` are (see [LIMITATIONS.md](LIMITATIONS.md)), and a small fraction of frames
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

## Table-based cross-camera registration (optional, two-camera sessions)

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

## Building/updating the overlap classifier

```bash
cd src
python -m segmentation.classifier.extract_overlap_candidates --masks-dir .../masks/<name> --out candidates.csv
# label candidates.csv by hand: 1 = same body/simultaneous fragments, 0 = genuinely different people
# (use tools/overlay_subvideo.py / tools/check_overlap_iou.py to inspect each candidate's frame range)
python -m segmentation.classifier.train_overlap_classifier --csv candidates.csv --out classifier.json
```
