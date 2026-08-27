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

**Approach: SAM3 + OSNet + a learned heuristic.**

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

## Structure

```
pose-behavior-pipeline/
├── src/
│   ├── segmentation/                # the pipeline
│   │   ├── run_pipeline.py          # MAIN ENTRY POINT -- see Usage below
│   │   ├── run_sam3_baseline.py     # step 1: real-psifx SAM3 baseline tracking
│   │   ├── merging/                 # step 2: the merge_fragments algorithm
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
│   └── pose/
│       └── appearance_embedding.py  # OSNetEmbedder + EMA gallery update
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchreid gdown tensorboard  # OSNet -- see requirements.txt's notes
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

**`ffmpeg`** (system binary, `run_pipeline.py` shells out to it for
the transcode step): install via your OS package manager.

## Usage

```bash
cd src
python -m segmentation.run_pipeline --video ~/Bureau/9_group_1_3/camera_a.mkv \
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

## Ethics & privacy

Video of minors in a clinical context requires face blurring as early
as possible (see Known limitations -- not currently implemented),
compliance with Swiss LPD and, where applicable, GDPR, and separation
of raw video from derived features with distinct retention policies.
