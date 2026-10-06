# Architecture

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
│   │       ├── check_overlap_iou.py #   raw IoU/distance between two ids
│   │       └── crop_outputs.py      #   crops merged MaskDir + source video (border-artifact removal)
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

## Provenance of the pose step

This repo has a `pose-behavior-pipeline_legacy` sibling: an older, more
mature pipeline (YOLO-pose+ByteTrack as its main backend, with a
documented plan to "reconnect" a crop-based MediaPipe PoseLandmarker
once segmentation stability was verified -- exactly the situation this
repo is now in). `pose/keypoints.py` and `pose/model_cache.py` are
ported close to verbatim from it, and `extract_keypoints.py`'s schema
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
as candidates for a follow-up, not done here (see
[LIMITATIONS.md](LIMITATIONS.md), especially the anonymization gap).
