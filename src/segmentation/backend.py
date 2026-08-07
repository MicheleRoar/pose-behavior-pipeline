"""
backend.py
===========
Minimal protocol every segmentation/tracking engine must satisfy to be
pluggable into `segmentation_demo.py` / `pipeline_runner.py` without
touching them: a single method, `run(source, stream=True)`, that returns
an iterator of `SegFrameResult` (frame_index, frame, people = list of
(track_id, bbox_xyxy, mask_polygon, box_conf)).

`SegTracker` (YOLO26-seg + ByteTrack, seg_estimation.py) already satisfies
this protocol by construction, with no changes needed: it's defined HERE
after the fact only to make explicit the contract that new backends
(`Sam31Tracker`, `Sam2Tracker`, see sam_backend.py) must also satisfy too.
It's a `typing.Protocol` (static duck typing): no need to inherit from any
base class, just having the same method with the same signature is enough
-- `isinstance(x, SegmentationBackend)` still works thanks to
`@runtime_checkable`, used in the tests to verify conformance without
having to actually instantiate YOLO/SAM.

Why this decoupling matters here: `iter_segmentation_frames()` (in
segmentation_demo.py) and everything above it (Tkinter and web GUI,
VideoPlayer, CSV export) treat the backend as a black box that produces
the same kind of result per frame -- choosing YOLO, SAM 3.1, or SAM2 then
becomes just a choice of WHICH class to instantiate inside
`iter_segmentation_frames()`, zero other downstream changes.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from segmentation.seg_estimation import SegFrameResult


@runtime_checkable
class SegmentationBackend(Protocol):
    """Minimal contract: an object built with its own specific parameters
    (model, device, thresholds, ...) and a `run()` method that, given a
    video source, returns an iterator of `SegFrameResult` in increasing
    frame order, with no gaps."""

    def run(self, source, stream: bool = True) -> Iterator[SegFrameResult]:
        ...
