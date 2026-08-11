"""
seg_estimation.py
==================
Thin wrapper over Ultralytics YOLO26 (instance segmentation) to extract
multi-person silhouettes with tracking, from a video file or a live
source.

Current status (see README): TEMPORARILY replaces pose_estimation.py as
the basis of the main pipeline. Motivation: on difficult scenes
(top-down view, fast movement, artificial lighting) the pose model
produced too many spurious ids (50+ within a few minutes even with
tracker/thresholds already tuned) -- the hypothesis to verify with
track_stability_check.py / segmentation_demo.py is that a model that
only has to delineate the silhouette (not regress 17 precise keypoints)
keeps a more stable detection confidence on a partially visible or
fast-moving person, and therefore tracks more continuously.

Plan (not yet implemented): if base tracking turns out to be
sufficiently stable, reconnect pose estimation by applying it ONLY
inside the tracked silhouette (mask-guided crop, not a simple box) --
this does not replace pose_estimation.py, it works alongside it using
the already-stable id from segmentation. The behavioral features that
depend on keypoints (features.py, gaze_head.py, hands.py, reid.py,
chuv_features.py) remain in the repository, tested, but are not
currently wired into this pipeline.

YOLO26-seg models are trained on COCO (80 classes): unlike -pose models
(a single class, "person"), here the person class (id 0 in COCO) must be
explicitly filtered -- otherwise ByteTrack would also track chairs,
tables, bags etc. present in the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.tracking_common import cap_by_confidence
from common.yolo_models import resolve_tracker_config, resolve_yolo_weights

COCO_PERSON_CLASS_ID = 0


@dataclass
class SegFrameResult:
    frame_index: int
    frame: np.ndarray
    people: list[tuple[int, np.ndarray, np.ndarray, float]] = field(default_factory=list)
    # each element: (track_id, bbox_xyxy (4,), mask_polygon (N,2), box_conf)
    # mask_polygon is an empty (0,2) array if the model did not produce a
    # valid mask for that detection in that frame.


class SegTracker:
    """Extracts silhouettes (box + mask) for multiple people with tracking
    from a video source using Ultralytics YOLO26 instance segmentation.

    Same interface as `pose_estimation.PoseTracker` (run() returns a
    generator of FrameResult-like objects), so the rest of the pipeline
    that already treats the id as a generic key requires no other
    changes.

    Parameters
    ----------
    model_name : YOLO26-seg model (e.g. "yolo26s-seg.pt"; "yolo26n-seg.pt"
        faster, "yolo26m/l-seg.pt" more accurate -- same trade-off as
        pose_estimation.py).
    device : "mps" on Apple Silicon, "cpu" as fallback, "cuda" if an
        NVIDIA GPU is available.
    conf_threshold : minimum confidence threshold, same recommendation
        as pose_estimation.py (keep at or below ByteTrack's
        track_low_thresh, 0.1 by default in bytetrack.yaml).
    tracker : Ultralytics tracking config ("bytetrack.yaml" by default,
        or "configs/bytetrack_permissive.yaml" for difficult scenes).
    max_people : as in pose_estimation.py, keeps only the N most
        confident detections per frame when the number of session
        participants is known (see pose_estimation.py for the full
        rationale).
    """

    def __init__(self, model_name: str = "yolo26s-seg.pt", device: str = "mps",
                 conf_threshold: float = 0.1, tracker: str = "bytetrack.yaml",
                 max_people: int | None = None):
        # Delayed import: same reason as pose_estimation.py (the rest of
        # the package remains testable without ultralytics/torch installed).
        from ultralytics import YOLO

        self.model = YOLO(resolve_yolo_weights(model_name))
        self.device = device
        self.conf_threshold = conf_threshold
        self.tracker = resolve_tracker_config(tracker)
        self.max_people = max_people

    def run(self, source, stream: bool = True):
        """Runs segmentation + tracking on the given source.
        Returns a generator of `SegFrameResult`.
        """
        results = self.model.track(
            source=source,
            device=self.device,
            conf=self.conf_threshold,
            tracker=self.tracker,
            classes=[COCO_PERSON_CLASS_ID],  # people only, not the other 79 COCO classes
            stream=stream,
            verbose=False,
        )

        for i, r in enumerate(results):
            people = []
            if r.boxes is not None and r.boxes.id is not None:
                box_xyxy = r.boxes.xyxy.cpu().numpy()
                box_conf = r.boxes.conf.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                # r.masks can be None if the loaded model isn't -seg, or if
                # no valid mask was produced for that frame -- treated as
                # an "empty polygon" for every detection, never a made-up
                # fallback.
                polys = r.masks.xy if r.masks is not None else [None] * len(track_ids)

                for idx in cap_by_confidence(box_conf, self.max_people):
                    poly = polys[idx] if idx < len(polys) else None
                    poly_arr = np.asarray(poly) if poly is not None else np.empty((0, 2))
                    people.append((int(track_ids[idx]), box_xyxy[idx], poly_arr, float(box_conf[idx])))
            yield SegFrameResult(frame_index=i, frame=r.orig_img, people=people)


def mask_centroid(poly: np.ndarray) -> np.ndarray:
    """Approximate center of the silhouette (mean of the mask polygon's
    vertices). Not the true geometric centroid of the area for very
    irregular polygons, but for a human silhouette the difference is
    negligible and it avoids having to rasterize a full mask just for
    this. NaN (2,) if the polygon is empty."""
    if poly.shape[0] == 0:
        return np.full(2, np.nan)
    return poly.mean(axis=0)


def mask_area(poly: np.ndarray) -> float:
    """Area of the mask polygon in pixels^2 (shoelace formula, standard
    for the area of a simple polygon given its vertices in order).
    0.0 if the polygon is empty or degenerate (fewer than 3 points)."""
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
