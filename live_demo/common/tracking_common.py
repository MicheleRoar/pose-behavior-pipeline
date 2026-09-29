"""
tracking_common.py
===================
Shared utility between pose_estimation.py, seg_estimation.py and
track_stability_check.py: the "keep only the N most confident detections
in this frame" cap (see `max_people` in pose_estimation.py for the full
rationale) was duplicated in several places -- centralized here so a
future fix doesn't have to be repeated everywhere.
"""

from __future__ import annotations

import numpy as np


def cap_by_confidence(box_conf: np.ndarray, max_people: int | None):
    """Indices to keep in this frame: all of them (like `range(n)`) if
    `max_people` is not set or the number of detections is already within
    the limit; otherwise only the indices of the `max_people` highest-
    confidence detections. Never discards anything when under the limit --
    the filter only kicks in on excess, it never lowers the number of
    people kept below the configured cap."""
    n = len(box_conf)
    if max_people is None or n <= max_people:
        return range(n)
    return np.argsort(-box_conf)[:max_people]
