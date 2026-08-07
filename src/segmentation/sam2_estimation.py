"""
sam2_estimation.py
=====================
`Sam2Tracker`: segmentation/tracking backend based on "vanilla" SAM2
(facebookresearch/sam2). Same structure as `Sam31Tracker`: all the shared
logic (chunking, id seeding, reconciliation, persistence) lives in
`segmentation/sam_backend.py::ChunkedVideoPredictorBackend`, here it's
just how to build the predictor.

Why vanilla SAM2 and not SAMURAI
------------------------------------
This module replaces `samurai_estimation.py` (removed). SAMURAI adds a
Kalman filter on top of SAM2 ("motion-aware mask selection", inside
`sam2_base.py::_forward_sam_heads`) written and validated ONLY for
single-target visual object tracking: the benchmarks it was tested on
(LaSOT, GOT-10k, TrackingNet) track ONE object per video, never more than
one together.

Verified on a real CUDA machine: seeding several people in the same
session (the normal case here -- several children/therapist in the scene
together), SAM2 groups them into a single batch for inference, but
SAMURAI's Kalman code assumes a batch of size 1 -- it crashes with

    RuntimeError: Boolean value of Tensor with more than one value is
    ambiguous

inside `_forward_sam_heads()` (line `ious[0][best_iou_inds]`: with more
than one tracked object, `best_iou_inds` has one entry per person instead
of a scalar, and the resulting indexing is no longer a single value).
Making it work for N people would require a separate SAM session for each
one (N times the encoder cost, the heaviest part of the whole pipeline) --
not worth it compared to vanilla SAM2, which supports batched multi-object
NATIVELY (no patch, no crash), even without the motion-modeling through
occlusions that was SAMURAI's real added value for single-target tracking.

Requirements: `sam2` package (facebookresearch/sam2, `git clone` + `pip
install -e .`), PUBLIC checkpoints (no gated access). The default below
points to the checkpoint INSIDE the `samurai/` checkout (which vendors
sam2 internally) instead of a separate facebookresearch/sam2 clone: it's
fine to reuse the SAME .pt file (SAMURAI doesn't retrain SAM2's weights,
it only applies the Kalman filter at inference time) -- just point
`config` to the "standard" SAM2.1 config instead of samurai's own
(`configs/samurai/...`), vendored in the same checkout. If in the future
you prefer a clean, separate clone of facebookresearch/sam2, override
`checkpoint=`/`config=` explicitly on the constructor.

Delayed import, same reason as `Sam31Tracker`.
"""

from __future__ import annotations

from pathlib import Path

from segmentation.sam_backend import ChunkedVideoPredictorBackend

# Same "sibling folder" convention already used by samurai_estimation.py
# (see README -- "clone it outside pose-behavior-pipeline"): here though it
# points INSIDE the already-present samurai/ checkout
# (samurai/checkpoints/*.pt), not a separate facebookresearch/sam2 clone --
# the checkpoint is identical, no need to download it twice. Overridable by
# passing an explicit `checkpoint=` to the constructor if you prefer your
# own separate sam2/ clone.
DEFAULT_CHECKPOINT = str(
    Path(__file__).resolve().parents[3] / "samurai" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
)

# "Standard" SAM2.1 config -- NOT samurai's own (configs/samurai/...):
# this is the difference that disables the single-object Kalman filter and
# enables native multi-object batching, see the module docstring.
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


class Sam2Tracker(ChunkedVideoPredictorBackend):
    """See the module docstring and `ChunkedVideoPredictorBackend`.
    `checkpoint` and `config` follow sam2's convention (.pt file name +
    associated .yaml config)."""

    def __init__(self, *, checkpoint: str = DEFAULT_CHECKPOINT,
                 config: str = DEFAULT_CONFIG, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.config = config

    def _build_predictor(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "Sam2Tracker requires the 'sam2' package (not installed). "
                "See https://github.com/facebookresearch/sam2 -- "
                "public checkpoints (no gated access required)."
            ) from exc
        return build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)
