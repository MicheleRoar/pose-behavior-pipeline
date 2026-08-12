"""
psifx_eval/overlap_tracking.py
=================================
Thin glue between real psifx/SAM3 and the pure algorithmic logic in
`overlap_strategy.py` (see that module's docstring for the actual
strategy and why it should recover identities vanilla psifx's single-
frame chunk-boundary comparison loses). This file is deliberately
almost empty of decisions: every actual choice (windowing math,
matching, what to skip when writing) lives in `overlap_strategy.py`,
already fully unit-tested without psifx/GPU (see
`tests/overlap_strategy_check.py`). This file only wires it to real
SAM3 inference (`Sam3TrackingTool._segment_chunk`, inherited unchanged
-- SAME model calls as vanilla psifx, see run_baseline_vs_oracle.py's
docstring for why fidelity to psifx's own segmentation matters even
while replacing its stitching) and real video I/O (`psifx.io.video`).

Not runnable/testable in this project's sandbox (needs the real `psifx`
package, a CUDA GPU, gated SAM3 checkpoint access) -- verify on
Michele's real machine, same as `run_baseline_vs_oracle.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from psifx.io.video import VideoReader, VideoWriter
from psifx.video.tracking.sam3.tool import Sam3TrackingTool

from psifx_eval.overlap_strategy import (
    chunk_with_overlap,
    local_frames_to_write,
    map_chunk_ids_via_overlap,
    map_first_chunk_ids,
    stash_overlap_tail,
)


class OverlappingChunkSam3TrackingTool(Sam3TrackingTool):
    """Same model/segmentation as vanilla `Sam3TrackingTool` (inherits
    `__init__`, `_segment_chunk`, OOM handling helpers unchanged) --
    only the chunk iteration and cross-chunk linking are replaced, via
    `infer_with_overlap()` alongside (not instead of) the inherited
    `infer()`, so the SAME instance/loaded model can produce both a
    vanilla-psifx baseline run and an overlap-strategy run for a fair
    side-by-side comparison (see `run_overlap_experiment.py`)."""

    def infer_with_overlap(
        self,
        video_path: str | Path,
        mask_dir: str | Path,
        text_prompt: str = "person",
        chunk_size: int = 300,
        overlap: int = 75,
        iou_threshold: float = 0.3,
    ) -> None:
        """Same contract as `Sam3TrackingTool.infer()` (writes the same
        `<global_id>.mp4` MaskDir format, so `mask_io.load_mask_dir()`
        and `id_metrics.compute_metrics()` work unchanged on the
        output) -- ADDS `overlap`: number of frames shared between
        consecutive chunks (Michele/Loic brief suggests 50-100). Set
        `overlap=0` to fall back to vanilla single-frame-boundary
        linking (still not byte-identical to `infer()`, since it goes
        through this class's own code path, but algorithmically
        equivalent -- use `infer()` itself for the real psifx baseline,
        this is for testing the degenerate case only)."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

        mask_dir = Path(mask_dir)
        if mask_dir.exists() and any(mask_dir.iterdir()):
            if self.overwrite:
                print(f"Mask directory {mask_dir} is non-empty")
            else:
                raise FileExistsError(f"Mask directory {mask_dir} is non-empty.")
        mask_dir.mkdir(parents=True, exist_ok=True)

        writers: Dict[int, VideoWriter] = {}
        written_frames: Dict[int, int] = {}
        next_global_id = 0
        prev_overlap_window: Optional[dict] = None
        frame_size = (0, 0)
        is_first_chunk = True
        processed_frame_count = 0

        try:
            # Single streaming pass over the source video (VideoReader stays
            # open for the whole loop) -- `chunk_with_overlap` only ever
            # holds `chunk_size` PIL frames in memory at once (it carries
            # the overlap tail forward instead of re-reading), same memory
            # profile as vanilla psifx's own `_iter_video_chunks`, NOT the
            # whole video at once.
            with VideoReader(path=video_path) as video_reader:
                frame_rate = video_reader.frame_rate
                pil_frames = (Image.fromarray(frame) for frame in video_reader)

                for start_frame, chunk in chunk_with_overlap(
                    pil_frames, chunk_size=chunk_size, overlap=overlap
                ):
                    if frame_size == (0, 0):
                        frame_size = chunk[0].size

                    chunk_outputs = self._segment_chunk(chunk, text_prompt)

                    if is_first_chunk or overlap == 0 or prev_overlap_window is None:
                        id_mapping, next_global_id = map_first_chunk_ids(
                            chunk_outputs, next_global_id, self.max_num_objects,
                        )
                    else:
                        id_mapping, next_global_id = map_chunk_ids_via_overlap(
                            chunk_outputs=chunk_outputs,
                            overlap=overlap,
                            prev_overlap_window=prev_overlap_window,
                            iou_threshold=iou_threshold,
                            next_global_id=next_global_id,
                            max_num_objects=self.max_num_objects,
                        )

                    skip_local_frames = 0 if is_first_chunk else overlap
                    write_range = local_frames_to_write(len(chunk), skip_local_frames)
                    self._write_chunk_masks(
                        chunk_outputs=chunk_outputs,
                        id_mapping=id_mapping,
                        writers=writers,
                        written_frames=written_frames,
                        mask_dir=mask_dir,
                        frame_rate=frame_rate,
                        frame_size=frame_size,
                        start_frame=start_frame,
                        local_frames=write_range,
                    )

                    if overlap > 0:
                        prev_overlap_window = stash_overlap_tail(
                            chunk_outputs, id_mapping, chunk_length=len(chunk), overlap=overlap,
                        )

                    processed_frame_count += len(write_range)
                    is_first_chunk = False

                    del chunk_outputs
                    if self.device == "cuda":
                        self._clear_cuda_memory()
        finally:
            for writer in writers.values():
                writer.close()

        if processed_frame_count == 0:
            raise ValueError(f"No frames found in input video: {video_path}")
        if not writers:
            print("No masks to write.")

    def _write_chunk_masks(
        self, chunk_outputs, id_mapping, writers, written_frames, mask_dir,
        frame_rate, frame_size, start_frame, local_frames,
    ):
        """Same as `Sam3TrackingTool._write_chunk_masks`, plus one
        change: only writes the frames in `local_frames` (the NEW ones
        this chunk contributes -- see
        `overlap_strategy.local_frames_to_write`), so a chunk's carried-
        over overlap tail, already written by the previous chunk, is
        never duplicated on disk."""
        width, height = frame_size
        empty_mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        for local_frame_idx in local_frames:
            if local_frame_idx not in chunk_outputs:
                continue
            global_frame_idx = start_frame + local_frame_idx
            frame_out = chunk_outputs[local_frame_idx]

            masks_by_global_id = {}
            for local_obj_id, local_mask in zip(frame_out["object_ids"], frame_out["masks"]):
                global_obj_id = id_mapping.get(local_obj_id)
                if global_obj_id is not None:
                    masks_by_global_id[global_obj_id] = local_mask

            for global_obj_id in sorted(masks_by_global_id.keys()):
                if global_obj_id in writers:
                    continue
                writers[global_obj_id] = VideoWriter(
                    path=mask_dir / f"{global_obj_id}.mp4",
                    input_dict={"-r": frame_rate},
                    output_dict={"-c:v": "libx264", "-crf": "0", "-pix_fmt": "yuv420p"},
                    overwrite=self.overwrite,
                )
                written_frames[global_obj_id] = 0
                for _ in range(global_frame_idx):
                    writers[global_obj_id].write(image=empty_mask_rgb)
                    written_frames[global_obj_id] += 1

            for global_obj_id, writer in sorted(writers.items()):
                mask = masks_by_global_id.get(global_obj_id)
                if mask is None:
                    mask_rgb = empty_mask_rgb
                else:
                    mask_uint8 = mask.astype(np.uint8) * 255
                    mask_rgb = np.repeat(mask_uint8[..., np.newaxis], 3, axis=-1)
                writer.write(image=mask_rgb)
                written_frames[global_obj_id] += 1
