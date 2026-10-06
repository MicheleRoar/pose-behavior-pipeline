# Known limitations

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
  keypoint/session, not just accepted as-is -- see
  [USAGE.md](USAGE.md#pose-after-segmentation).
- **Depth step (`pose/depth.py`, `pose/movement_metrics.py`,
  `pose/run_depth.py`) -- `pyk4a`+`libk4a` confirmed working on the
  real project machine (2026-09-22, despite Azure Kinect DK being
  discontinued -- installed via the Ubuntu 18.04 `.deb` packages, no
  depth engine needed for playback-only use), and the pipeline has now
  run end-to-end on one real session (`10_individual_12`, "seated").**
  Still only synthetic-data-tested for `table_calibration.py` (the
  rigid-transform math, not the interactive `extract`/`pick` steps,
  which need a real two-camera session -- see
  [USAGE.md](USAGE.md#table-based-cross-camera-registration-optional-two-camera-sessions)).
  The originally-planned "moves a lot vs. seated" cross-session
  comparison (`9_group_1_3` vs. an individual session) hasn't been run
  yet, and turned out to be confounded anyway (see USAGE.md's Depth
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
