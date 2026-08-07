"""
track_stability_check.py
=========================
Diagnostic comparison: how many distinct track_ids does ByteTrack produce
using a SEGMENTATION model (yolo26-seg) instead of the POSE model
(yolo26-pose) currently used in pose_estimation.py, on the same source and
with the same tracker/threshold configuration.

Motivation: on real video (top-down view, fast movement, artificial
lighting) ByteTrack + YOLO-pose produces a huge number of distinct ids
(50+ within a few minutes even with conf-threshold/tracker/max-people
already tuned). Hypothesis to verify: a segmentation model, which only
has to delineate the silhouette (not estimate 17 precise keypoints),
might keep a more stable detection confidence on a partially visible or
fast-moving person, and therefore more continuous tracking -- regardless
of whether usable keypoints can then be extracted from it or not.

This script does NOT extract keypoints (Ultralytics' -seg models don't
have any, only mask + box): it only counts distinct ids and their
duration, to decide whether it's worth investing in the "segmentation +
pose on the silhouette" path (which has been discussed but is NOT yet
implemented -- see the pose_estimation.py module for the current
single-model YOLO-pose approach).

Suggested comparison: run this script AND live_demo.py/pipeline.py on the
SAME video with the SAME --tracker/--conf-threshold/--max-people
configuration, and compare the number of distinct ids reported here with
the one observed in the normal pipeline's CSV/log.

Usage:
    python track_stability_check.py --source video.mp4 --fps 15 \
        --model yolo26s-seg.pt --tracker configs/bytetrack_permissive.yaml \
        --conf-threshold 0.1 --max-people 2
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from common.device import detect_default_device


def run(source, fps: float, model_name: str, device: str, conf_threshold: float,
        tracker_config: str, max_people: int | None) -> None:
    # Delayed import, same reason as pose_estimation.py: the rest of the
    # package stays testable without ultralytics/torch installed.
    from ultralytics import YOLO

    from common.tracking_common import cap_by_confidence

    model = YOLO(model_name)
    id_frame_count: dict[int, int] = defaultdict(int)
    id_first_frame: dict[int, int] = {}
    id_last_frame: dict[int, int] = {}

    results = model.track(
        source=source,
        device=device,
        conf=conf_threshold,
        tracker=tracker_config,
        stream=True,
        verbose=False,
    )

    n_frames = 0
    for i, r in enumerate(results):
        n_frames = i + 1
        if r.boxes is None or r.boxes.id is None:
            continue
        box_conf = r.boxes.conf.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy().astype(int)

        for idx in cap_by_confidence(box_conf, max_people):
            tid = int(track_ids[idx])
            id_frame_count[tid] += 1
            id_first_frame.setdefault(tid, i)
            id_last_frame[tid] = i

    n_ids = len(id_frame_count)
    lifespans = sorted(id_frame_count.values())
    # 15 frames = the default min_signature_frames threshold in reid.py: an
    # id shorter than this would never build up a signature, so it can
    # never be recovered by reid.py (neither normally nor forced).
    short_lived = sum(1 for v in lifespans if v < 15)

    print(f"Model: {model_name}  |  tracker: {tracker_config}  |  "
          f"conf-threshold: {conf_threshold}"
          + (f"  |  max-people: {max_people}" if max_people is not None else ""))
    print(f"Total frames processed: {n_frames}  (~{n_frames / fps:.1f}s at {fps} fps)")
    print(f"Distinct ids: {n_ids}")
    if lifespans:
        median = lifespans[len(lifespans) // 2]
        print(f"Id duration in frames: min={lifespans[0]}  median={median}  max={lifespans[-1]}")
        print(f"Ids under 15 frames (too short for reid.py's default min_signature_frames): "
              f"{short_lived}/{n_ids} ({100 * short_lived / n_ids:.0f}%)")
    else:
        print("No ids detected.")


def main():
    parser = argparse.ArgumentParser(
        description="Counts the distinct ids produced by ByteTrack on a segmentation "
                     "model (yolo26-seg), to compare tracking stability against the pose "
                     "model already in use in pose_estimation.py. "
                     "Does not extract keypoints: tracking diagnostics only.")
    parser.add_argument("--source", required=True, help="Video path or webcam index")
    parser.add_argument("--fps", type=float, default=15.0,
                         help="Source frame rate (only used for the summary in seconds)")
    parser.add_argument("--model", default="yolo26s-seg.pt",
                         help="YOLO26 instance segmentation model "
                              "(yolo26n/s/m/l/x-seg.pt)")
    parser.add_argument("--device", default=None,
                         help="mps | cpu | cuda (default: auto-detected, see common/device.py)")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                         help="Same recommendation as pose_estimation.py: keep at or "
                              "below track_low_thresh (0.1 by default in bytetrack.yaml)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Ultralytics tracker config, e.g. configs/bytetrack_permissive.yaml")
    parser.add_argument("--max-people", type=int, default=None,
                         help="As in pose_estimation.py: keeps only the N most confident "
                              "detections per frame, if the number of participants is known")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = args.device or detect_default_device()
    run(source, fps=args.fps, model_name=args.model, device=device,
        conf_threshold=args.conf_threshold, tracker_config=args.tracker,
        max_people=args.max_people)


if __name__ == "__main__":
    main()
