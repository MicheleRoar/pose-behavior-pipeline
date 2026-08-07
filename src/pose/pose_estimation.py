"""
pose_estimation.py
===================
Thin wrapper over Ultralytics YOLO-pose to extract multi-person COCO-17
keypoints with tracking, from a video file or a live source (e.g. webcam
/ Canon R8 via EOS Webcam Utility, or an HDMI capture card).

Current status: the main pipeline TEMPORARILY uses
segmentation/seg_estimation.py (silhouettes only, no keypoints) instead
of this module -- see that file's docstring for why, and the plan to
reconnect pose estimation (applied inside the tracked silhouette, not on
the whole box) once the stability of the base tracking has been
verified. This module remains functional and tested, it has not been
removed.

Designed to run on Apple Silicon (M1/M2/...) using the MPS backend
(`device="mps"`); on other machines it automatically falls back to
CPU/CUDA if available.

This module requires `ultralytics` and `opencv-python`, not installed in
the sandbox environment used to develop/test `features.py` (see the
README for local installation instructions on the Mac).

Usage example:

    from pose.pose_estimation import PoseTracker

    tracker = PoseTracker(model_name="yolo26n-pose.pt", device="mps")
    for result in tracker.run(source=0):   # 0 = first available webcam
        for track_id, kpts, conf in result.people:
            ...  # kpts: array (17, 2), conf: array (17,)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.tracking_common import cap_by_confidence


@dataclass
class FrameResult:
    frame_index: int
    frame: np.ndarray
    people: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    # each element: (track_id, keypoints (17,2), confidences (17,))


class PoseTracker:
    """Extracts multi-person keypoints with tracking from a video source
    using Ultralytics YOLO-pose.

    Parameters
    ----------
    model_name : model name/path (e.g. "yolo26n-pose.pt" for the nano
        model, faster; "yolo26s-pose.pt"/"yolo26m-pose.pt" for higher
        accuracy at the cost of speed -- for offline batch processing,
        with no real-time constraint, it's worth using the largest model
        the device can handle).
    device : "mps" on Apple Silicon, "cpu" as fallback, "cuda" if an
        NVIDIA GPU is available.
    conf_threshold : minimum confidence threshold to consider a detection
        valid. Defaults to 0.1 (not 0.4): ByteTrack has a low-confidence
        recovery phase (track_low_thresh, 0.1 by default in
        bytetrack.yaml) specifically designed to handle weak detections
        without losing identity -- too high a conf_threshold discards
        them before ByteTrack can use them, causing spurious IDs on
        difficult scenes (top-down view, fast movement, artificial
        lighting).
    tracker : Ultralytics tracking config ("bytetrack.yaml" by default,
        or "configs/bytetrack_permissive.yaml" for scenes with frequent
        confidence drops not caused by real occlusion -- see that file
        for parameter details).
    max_people : if set, caps the number of people per frame to this
        value, keeping only the highest-confidence detections (useful
        when the number of session participants is known in advance,
        e.g. 2 for a 1v1 child-caregiver session, or a dozen for a group
        session: suppresses spurious detections from noise/reflections/
        double-detections above that number before they become a track).
        Does not solve the problem of a real person losing and regaining
        an ID after a true occlusion -- for that see reid.py and
        configs/bytetrack_permissive.yaml. None (default) = no limit.
    """

    def __init__(self, model_name: str = "yolo26n-pose.pt", device: str = "mps",
                 conf_threshold: float = 0.1, tracker: str = "bytetrack.yaml",
                 max_people: int | None = None):
        # Delayed import: so the rest of the package (features.py,
        # anonymize.py) remains usable/testable even without
        # ultralytics/torch installed (useful for lightweight unit tests).
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf_threshold
        self.tracker = tracker
        self.max_people = max_people

    def run(self, source, stream: bool = True):
        """Runs pose estimation + tracking on the given source.

        `source` can be:
        - an integer (webcam index, e.g. 0)
        - the path to a video file
        - a stream string (e.g. RTSP)

        Returns a generator of `FrameResult`.
        """
        results = self.model.track(
            source=source,
            device=self.device,
            conf=self.conf_threshold,
            tracker=self.tracker,
            stream=stream,
            verbose=False,
        )

        for i, r in enumerate(results):
            people = []
            if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
                kpts_xy = r.keypoints.xy.cpu().numpy()       # (n_people, 17, 2)
                kpts_conf = r.keypoints.conf.cpu().numpy()   # (n_people, 17)
                box_conf = r.boxes.conf.cpu().numpy()        # (n_people,) detection confidence
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                for idx in cap_by_confidence(box_conf, self.max_people):
                    people.append((int(track_ids[idx]), kpts_xy[idx], kpts_conf[idx]))
            yield FrameResult(frame_index=i, frame=r.orig_img, people=people)


def keypoints_dict_to_array(kpts_xy: np.ndarray) -> np.ndarray:
    """Utility to guarantee the shape (17, 2) expected by features.py,
    even if the model returns a different number of keypoints (e.g.
    custom models): here the standard Ultralytics COCO-17 schema is
    assumed.
    """
    assert kpts_xy.shape[-2:] == (17, 2), (
        f"Expected 17 keypoints (COCO schema), got shape {kpts_xy.shape}. "
        "If using a custom model, update keypoints.py accordingly."
    )
    return kpts_xy
