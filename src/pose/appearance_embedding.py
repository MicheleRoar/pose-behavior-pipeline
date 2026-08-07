"""
appearance_embedding.py
=========================
Appearance signal based on a real deep re-identification embedding
(OSNet, via `torchreid`), instead of the color/shape/position heuristics
already present in `reid.py`/`seg_reid.py`. Born from the explicit
request to add OSNet and "the idea" of StrongSORT to the pipeline, with
absolute priority: ids must not change, and people must stay in memory
to be easily re-associated on return.

Why "OSNet + the idea of StrongSORT" and not "replace everything with
StrongSORT" (scoping choice, see also the project memory)
------------------------------------------------------------------------
StrongSORT (Du et al. 2022) is DeepSORT plus three main additions: (1) a
real appearance embedding (typically OSNet) instead of SORT's IoU-only
features, (2) a per-track "feature bank" updated with an exponential
moving average (EMA) instead of keeping only the last embedding seen,
(3) an "NSA" Kalman filter (process noise scaled by detection confidence)
plus camera motion compensation (ECC) -- these last two designed for a
MOVING camera (e.g. drone/vehicle tracking), not the case here (fixed
camera, clinical context).

Rewriting the whole tracker from scratch (NSA Kalman + ECC + matching
cascade) would throw away the already-built and tested
`identity_manager.py` (batch Hungarian matching, "uncertain" policy
instead of silent merging, `session_mode`, causality, the `max_people`
cap for closed sessions) -- all logic tailored to the clinical context
that a generic tracker doesn't know about. The two StrongSORT ideas that
actually matter for the request ("don't change ids" + "stay in memory
for re-entry") are exactly (1) and (2): a stronger appearance embedding
than the current heuristics, and a memory that CONSOLIDATES over time
instead of relying on a single frame. This module provides (1)
(`OSNetEmbedder`); (2) is applied below (`ema_update`) and used by
`reid.py`/`seg_reid.py` to update `self.embedding` on every frame the
person is visible, not just at the moment of loss.

Heavy, optional dependency
--------------------------------
`torchreid` (and therefore `torch`) are NOT listed in requirements.txt as
a normal dependency -- same treatment as SAM 3.1/SAM2, see there. The
import is therefore delayed inside `OSNetEmbedder.__init__`: without
torchreid installed, the rest of the pipeline (including the rest of the
heuristic Re-ID) keeps working normally, simply without this extra
signal. The first use of a known `model_name` (e.g. 'osnet_x1_0')
without `model_path` makes torchreid download the pretrained weights
from its model zoo -- so an internet connection is required on first
run, not just installing the package.

Crop format
-----------------
`torchreid.utils.FeatureExtractor` accepts numpy arrays (H, W, C) and
converts them internally with `torchvision.transforms.ToPILImage()`,
which for a 3-channel array produces an image in 'RGB' mode -- so the
crop must be prepared in RGB, not OpenCV's native BGR (see
`_crop_person`, `[:, :, ::-1]`). If the mask polygon is also passed (not
just the bbox), the background inside the bbox but outside the
silhouette is zeroed out before passing the crop to the model: the
embedding focuses on the person, not the context (background, other
people partially inside the same bbox) -- an improvement over a raw
bbox-crop, discussed in the initial architectural consultation.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL_NAME = "osnet_x1_0"

# Below these dimensions (pixels) a crop is too small/too squashed to
# trust the resulting embedding -- same "no made-up signal" principle as
# compute_color_signature/_mask_hue_histogram: better None than a noisy
# embedding passed off as reliable.
MIN_CROP_W = 24
MIN_CROP_H = 48

# Minimum fraction of mask pixels inside the bbox for it to be worth
# zeroing out the background (below this threshold the polygon is
# probably too degenerate/noisy for reliable masking -- the raw
# bbox-crop is used anyway, the frame isn't discarded).
_MIN_MASK_FILL = 0.15


def _resolve_feature_extractor():
    """Resolves the `FeatureExtractor` class, handling TWO different
    layouts of the 'torchreid' package that `pip install torchreid` can
    end up installing:

    - the original project (github.com/KaiyangZhou/deep-person-reid,
      installable with `pip install git+...` or `pip install -e .` from
      a clone) exposes `torchreid/utils/` as a real subpackage --
      `from torchreid.utils import FeatureExtractor` works.
    - the third-party PyPI "torchreid-pip" distribution (the one that
      plain `pip install torchreid` installs by default, verified: it's
      an unofficial repackaging) hides everything under
      `torchreid.reid.*` and only rebinds `utils` as a top-level package
      ATTRIBUTE inside its own `torchreid/__init__.py` (`from
      torchreid.reid import ..., utils`) -- not a real submodule. With
      this layout, `from torchreid.utils import FeatureExtractor` fails
      with `ModuleNotFoundError: No module named 'torchreid.utils'` even
      if the package is correctly installed (the real cause of a bug
      reported by the user: "torchreid" importable on its own, but this
      specific import isn't).

    Importing `torchreid` and then accessing it by ATTRIBUTE
    (`torchreid.utils.FeatureExtractor`) works in both cases, so that's
    what we use here instead of a direct `from torchreid.utils import
    ...`."""
    import torchreid
    return torchreid.utils.FeatureExtractor


class OSNetEmbedder:
    """Minimal wrapper over `torchreid.utils.FeatureExtractor` for a
    single OSNet model. Usage in reid.py/seg_reid.py (a single wiring
    point):

        embedder = OSNetEmbedder(device="cpu")  # or "cuda"/"mps" if available
        ...
        vec = embedder.embed(frame_bgr, bbox_xyxy, poly=poly)  # None if the crop is poor
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model_path: str | None = None,
                 device: str = "cpu"):
        try:
            FeatureExtractor = _resolve_feature_extractor()
        except ImportError as exc:
            raise ImportError(
                "OSNet appearance embedding requires 'torch' and 'torchreid', "
                "not installed by default (heavy, optional dependency -- "
                "see requirements.txt). Install with: "
                "pip install torch torchreid"
            ) from exc
        except AttributeError as exc:
            raise ImportError(
                "'torchreid' appears to be installed but does not expose "
                "'torchreid.utils.FeatureExtractor' -- likely that "
                "'pip install torchreid' installed the third-party PyPI "
                "distribution 'torchreid-pip' (a repackaging with a "
                "different layout, not the original deep-person-reid "
                "project) or an incomplete installation. Check with: "
                "python -c \"import torchreid; print(torchreid.utils.FeatureExtractor)\""
            ) from exc

        kwargs = dict(model_name=model_name, device=device, verbose=False)
        if model_path:
            kwargs["model_path"] = model_path
        # pretrained=True when model_path is absent: torchreid downloads
        # the weights from its model zoo on first use of a known
        # model_name (requires internet the first time, then cached
        # locally).
        self._extractor = FeatureExtractor(**kwargs)
        self.model_name = model_name
        self.device = device

    def embed(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
               poly: np.ndarray | None = None) -> np.ndarray | None:
        """L2-normalized embedding (1D np.ndarray) of the person in
        `bbox_xyxy` (and optionally masked by `poly`) inside
        `frame_bgr`, or `None` if the crop is too small/degenerate to
        trust."""
        crop_rgb = _crop_person(frame_bgr, bbox_xyxy, poly)
        if crop_rgb is None:
            return None
        features = self._extractor([crop_rgb])  # torch tensor (1, D), internal no_grad
        vec = features[0].detach().cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-9 or not np.isfinite(norm):
            return None
        return vec / norm


def _crop_person(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
                  poly: np.ndarray | None) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1, x2 = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
    y1, y2 = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))
    if x2 - x1 < MIN_CROP_W or y2 - y1 < MIN_CROP_H:
        return None

    crop = frame_bgr[y1:y2, x1:x2].copy()
    if poly is not None and poly.shape[0] >= 3:
        import cv2
        shifted = poly.copy().astype(np.float64)
        shifted[:, 0] -= x1
        shifted[:, 1] -= y1
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 255)
        fill = cv2.countNonZero(mask) / float(mask.size)
        if fill >= _MIN_MASK_FILL:
            crop[mask == 0] = 0  # zero out the background, the embedding focuses on the person

    return crop[:, :, ::-1]  # BGR (OpenCV) -> RGB (expected by torchreid)


def embedding_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """0..1 similarity between two L2-normalized embeddings, via cosine
    similarity rescaled from [-1, 1] to [0, 1] (same 0..1 convention as
    the module's other signals, not cosine's native [-1, 1] convention).
    `None` if either embedding is missing."""
    if a is None or b is None:
        return None
    cos = float(np.dot(a, b))
    return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))


def torchreid_available() -> bool:
    """True if `torchreid.utils.FeatureExtractor` is actually reachable
    in this environment (same resolution as `_resolve_feature_extractor()`
    used by `OSNetEmbedder`, not just a bare `import torchreid` -- an
    `import torchreid` can succeed while `torchreid.utils.FeatureExtractor`
    doesn't, see `_resolve_feature_extractor`'s docstring for the
    concrete reason, verified on a real case). Doesn't instantiate any
    model (doesn't download weights, doesn't touch the GPU) -- used only
    for GUI-side gating (checkbox disabled with a reason, same pattern as
    `cudaAvailable`/the SAM 3.1/SAM2 options in app.js), NEVER to
    silently decide to skip the embedding: if the user explicitly
    requests it but the dependency is missing, `OSNetEmbedder.__init__`
    still raises its `ImportError` with installation instructions, it
    doesn't fail silently."""
    try:
        _resolve_feature_extractor()
    except (ImportError, AttributeError):
        return False
    except Exception as exc:
        # torchreid is a package that's been stalled since 2021: the
        # import can sometimes fail with something else (not
        # ImportError) on an environment with more recent numpy/torch --
        # e.g. `np.float`/`np.int` removed in numpy>=1.24, referenced by
        # some versions of torchreid/yacs, which raise AttributeError
        # during import, not ImportError. If we didn't catch it here,
        # the exception would propagate all the way up to
        # `Api.detect_device()` (no try/except there, see webui/api.py)
        # and would make cuda/mps/cpu detection fail ALONG with it
        # (app.js handles the whole call with a single try/catch) -- a
        # confusing symptom ("SAM 3.1/SAM2 and the embedding are both
        # disabled, even though torch is installed") for a non-obvious
        # cause. Better to flag ONLY the embedding as unavailable here,
        # printing the real reason to the terminal (not visible in the
        # UI, but useful for diagnosis) instead of a silent error or a
        # crash that also disables device detection.
        print(f"[appearance_embedding] torchreid installed but the import fails "
              f"({type(exc).__name__}: {exc}) -- OSNet embedding not available. "
              f"Likely a version incompatibility (torchreid is a package "
              f"no longer actively maintained, often conflicting with recent "
              f"numpy/torch).")
        return False
    return True


def ema_update(prev: np.ndarray | None, new: np.ndarray | None,
                alpha: float = 0.9) -> np.ndarray | None:
    """Exponential moving average on an embedding "in memory" -- the
    StrongSORT idea (2) cited in the module docstring: instead of
    recomputing the embedding from a single frame (noisy: motion blur,
    pose, partial occlusion), it's refined over time, so the signature
    stored for a person becomes progressively more stable the longer it
    stays visible -- exactly the requested behavior ("staying in memory
    to easily re-associate them on return"). High `alpha` (default 0.9,
    typical in the StrongSORT/DeepSORT literature) gives a lot of weight
    to history, little to the current frame: a single anomalous frame
    doesn't make the stored embedding "jump". Re-normalized to unit norm
    after averaging (the average of two unit vectors isn't generally
    unitary)."""
    if new is None:
        return prev
    if prev is None:
        return new
    updated = alpha * prev + (1.0 - alpha) * new
    norm = np.linalg.norm(updated)
    if norm < 1e-9:
        return prev
    return updated / norm
