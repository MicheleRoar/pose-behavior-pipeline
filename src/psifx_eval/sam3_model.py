"""
psifx_eval/sam3_model.py
===========================
Minimal, vendored wrapper around the SAM3 video segmentation model
itself -- exactly what psifx's `Sam3TrackingTool.__init__` /
`._segment_chunk` call (transformers' `Sam3VideoModel`/
`Sam3VideoProcessor`), and NOTHING else from psifx. Deliberately does
NOT install or import the `psifx` package (per Michele, 2026-08: "non
voglio installare psifx, voglio solo emulare gli stessi pacchetti che
usa" -- only the segmentation model matters right now, everything else
psifx bundles is out of scope for now).

Why this is still a faithful reproduction of what matters: the thing
under investigation is psifx's cross-chunk ID *stitching*, which sits
entirely OUTSIDE the model call -- the model call itself
(`init_video_session` -> `add_text_prompt` -> `propagate_in_video_iterator`
-> `postprocess_outputs`) is just the transformers API, identical
whether psifx's own code invokes it or this module does, same weights,
same checkpoint (`facebook/sam3` / `facebook/sam3.1`), same outputs.
Only psifx's surrounding orchestration (its own video I/O via
`skvideo`/ffmpeg-python, `langchain`/`whisperx`/`pyannote.audio` for
unrelated audio/text modules, etc. -- see psifx's requirements.txt) is
skipped, none of which touches segmentation behaviour.

Verified line-by-line against the real source
(psifx/video/tracking/sam3/tool.py, `Sam3TrackingTool.__init__` and
`._segment_chunk`, github.com/psifx/psifx) at the time this was
written -- if psifx's model-loading/inference code changes upstream,
re-diff against that file before trusting a new comparison run.

Only dependency beyond this project's existing stack: `transformers`
recent enough to expose `Sam3VideoModel`/`Sam3VideoProcessor` (psifx
itself pins `transformers==5.3.0`; anything >= that which still
exposes these two classes should work -- `Sam3SegmentationModel.__init__`
will raise immediately and clearly if the import/checkpoint doesn't
resolve). No `psifx`, no `skvideo`/`absolutely-not-scikit-video`, no
new venv.

Chunking, cross-chunk stitching, and video I/O are deliberately NOT
here -- out of scope for now, per Michele's direction. This module's
only contract is: frames in memory -> per-frame object ids + masks.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor

# Same default psifx itself uses (psifx/utils/constants.py::SAM3_PATH) --
# pass a different model_path (e.g. "facebook/sam3.1") to compare
# checkpoints without touching this default.
DEFAULT_SAM3_PATH = os.environ.get("SAM3_PATH", "facebook/sam3")


class Sam3SegmentationModel:
    """Loads the real SAM3 video model/processor and runs text-prompted
    segmentation+tracking on one chunk of frames at a time -- a faithful,
    minimal port of `Sam3TrackingTool.__init__` + `._segment_chunk` (see
    module docstring). Everything psifx wraps around this (chunk
    iteration over a whole video, cross-chunk id stitching, MaskDir
    writing) is intentionally the CALLER's responsibility, not this
    class's -- keeps this module tiny and independently testable."""

    def __init__(
        self,
        device: str = "cpu",
        model_path: str = DEFAULT_SAM3_PATH,
        api_token: Optional[str] = None,
        max_num_objects: Optional[int] = None,
        verbose: bool = True,
    ):
        self.device = device
        # Same dtype policy as psifx: full bf16 compute on GPU, fp32 on CPU.
        self.compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model_path = model_path
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.max_num_objects = max_num_objects
        self.verbose = verbose

        # Same env var psifx sets before loading -- some SAM3 checkpoints
        # ship non-tensor objects in their state dict that a newer
        # torch's default weights_only=True load would otherwise reject.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        if self.verbose:
            print(f"Loading SAM3 model from '{self.model_path}' on device '{self.device}'")

        try:
            self.model = Sam3VideoModel.from_pretrained(
                self.model_path, token=self.api_token,
            ).to(self.device, dtype=self.compute_dtype)
            self.processor = Sam3VideoProcessor.from_pretrained(
                self.model_path, token=self.api_token,
            )
            if self.max_num_objects is not None:
                self._configure_max_num_objects(self.max_num_objects)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load SAM3 model. Check HF token/model access "
                "(huggingface-cli login / HF_TOKEN), or pass a different model_path."
            ) from exc

    def segment_chunk(
        self,
        frames: List[Union[np.ndarray, "Image.Image"]],
        text_prompt: str = "person",
    ) -> Dict[int, Dict[str, list]]:
        """Runs SAM3 on one chunk of frames already in memory (accepts
        either RGB numpy arrays or PIL Images -- converts as needed).
        Returns `{local_frame_idx: {"object_ids": [int, ...], "masks":
        [bool (H, W) array, ...]}}`, the exact same shape psifx's own
        `_segment_chunk` produces, so any stitching/metrics code written
        against that contract works unchanged regardless of which of the
        two actually produced it."""
        pil_frames = [f if isinstance(f, Image.Image) else Image.fromarray(f) for f in frames]
        chunk_outputs: Dict[int, Dict[str, list]] = {
            idx: {"object_ids": [], "masks": []} for idx in range(len(pil_frames))
        }

        session = self.processor.init_video_session(
            video=pil_frames,
            inference_device=self.device,
            processing_device=self.device,
            video_storage_device="cpu" if self.device == "cuda" else self.device,
            dtype=self.compute_dtype,
        )
        try:
            self.processor.add_text_prompt(session, text_prompt)
            for out in self.model.propagate_in_video_iterator(session):
                processed = self.processor.postprocess_outputs(session, out)
                object_ids = self._to_int_list(processed["object_ids"])
                masks = self._to_bool_mask_list(processed["masks"])
                chunk_outputs[out.frame_idx] = {"object_ids": object_ids, "masks": masks}
        finally:
            del session

        return chunk_outputs

    def _configure_max_num_objects(self, max_num_objects: int) -> None:
        """Same best-effort, multi-attribute-path config as psifx's own
        `_configure_model_max_num_objects` -- the exact attribute the
        installed transformers version reads for this isn't guaranteed
        stable across releases, so try every path psifx itself tries."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "max_num_objects"):
            self.model.config.max_num_objects = max_num_objects
        tracker_config = getattr(getattr(self.model, "config", None), "tracker_config", None)
        if tracker_config is not None and hasattr(tracker_config, "max_num_objects"):
            tracker_config.max_num_objects = max_num_objects
        if hasattr(self.model, "max_num_objects"):
            self.model.max_num_objects = max_num_objects
        if self.verbose:
            print(f"Limiting SAM3 detections to at most {max_num_objects} object tracks.")

    @staticmethod
    def _to_int_list(ids) -> List[int]:
        if isinstance(ids, torch.Tensor):
            return [int(v) for v in ids.detach().cpu().tolist()]
        return [int(v) for v in np.asarray(ids).tolist()]

    @staticmethod
    def _to_bool_mask_list(masks) -> List[np.ndarray]:
        if isinstance(masks, torch.Tensor):
            return [m.detach().cpu().numpy().astype(bool) for m in masks]
        return [np.asarray(m).astype(bool) for m in masks]
