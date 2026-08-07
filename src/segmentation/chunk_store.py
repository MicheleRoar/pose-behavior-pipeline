"""
chunk_store.py
================
Incremental on-disk persistence of a chunk's results (masks + ids +
confidences). `sam_backend.py` calls `save_chunk()` as soon as a chunk is
completed, BEFORE moving on to the next one -- so on a long video (or in
case of a crash/interruption partway through) the work already done isn't
lost. Automatic resumption from an already-saved chunk isn't implemented
yet (write/read only for now): it remains a natural extension of
`sam_backend.py` if it's ever needed in practice.

Format: one `.npz` file per chunk, parallel arrays `frame_index`,
`track_id`, `box`, `polygon`, `conf` -- one per (person, frame). `box` and
`polygon` are object arrays (`dtype=object`, requires `allow_pickle=True`
on load) because each polygon has a different number of vertices:
packing them into a flat CSV/parquet-style table would still need an
"object" column for the same reason. The frame (image) is NOT saved: only
derived data, to avoid blowing up disk space -- the image can always be
re-read from the source video by frame index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from segmentation.seg_estimation import SegFrameResult


def chunk_filename(chunk_index: int) -> str:
    return f"chunk_{chunk_index:04d}.npz"


def save_chunk(results: list[SegFrameResult], out_dir: str, chunk_index: int) -> str:
    """Writes an already-completed chunk to disk (list of `SegFrameResult`,
    one per frame processed in this chunk). Returns the path of the
    written file. A chunk with no person detected in any frame still
    produces a valid file (empty arrays), so `load_chunk()` doesn't have
    to distinguish "missing chunk" from "empty chunk"."""
    os.makedirs(out_dir, exist_ok=True)
    frame_indices: list[int] = []
    track_ids: list[int] = []
    boxes: list[np.ndarray] = []
    polygons: list[np.ndarray] = []
    confs: list[float] = []
    for r in results:
        for track_id, bbox, poly, conf in r.people:
            frame_indices.append(r.frame_index)
            track_ids.append(track_id)
            boxes.append(bbox)
            polygons.append(poly)
            confs.append(conf)

    path = os.path.join(out_dir, chunk_filename(chunk_index))
    np.savez_compressed(
        path,
        frame_index=np.asarray(frame_indices, dtype=np.int64),
        track_id=np.asarray(track_ids, dtype=np.int64),
        box=np.asarray(boxes, dtype=object) if boxes else np.empty((0,), dtype=object),
        polygon=np.asarray(polygons, dtype=object) if polygons else np.empty((0,), dtype=object),
        conf=np.asarray(confs, dtype=np.float32),
    )
    return path


@dataclass
class ChunkRecord:
    """A (person, frame) row read back from a saved chunk."""
    frame_index: int
    track_id: int
    bbox: np.ndarray
    polygon: np.ndarray
    conf: float


def load_chunk(path: str) -> list[ChunkRecord]:
    """Reads back a chunk saved by `save_chunk()`. Returns a flat list of
    `ChunkRecord` (one element per person/frame, not grouped by frame) --
    it's up to the caller to re-aggregate it by `frame_index` if
    reconstructing `SegFrameResult`s is needed."""
    data = np.load(path, allow_pickle=True)
    n = len(data["frame_index"])
    return [
        ChunkRecord(
            frame_index=int(data["frame_index"][i]),
            track_id=int(data["track_id"][i]),
            bbox=data["box"][i],
            polygon=data["polygon"][i],
            conf=float(data["conf"][i]),
        )
        for i in range(n)
    ]
