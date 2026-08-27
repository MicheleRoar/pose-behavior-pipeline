"""
pose/appearance_embedding.py
=============================
OSNet deep re-identification embedding (via `torchreid`), used by
`segmentation/merging/signatures.py` as one of the two appearance
signals (alongside the hue-histogram color signal) that
`merge_fragments.py` uses to decide whether two mask fragments belong
to the same person.

`torch`/`torchreid` are a heavy, optional dependency (not in
requirements.txt, same treatment as any other GPU-only package): the
import is delayed inside `OSNetEmbedder.__init__`, so the rest of the
pipeline still works without them installed, just without the OSNet
signal. First use of a `model_name` without `model_path` downloads
pretrained weights from torchreid's model zoo (needs internet once).

Crop format: `torchreid.utils.FeatureExtractor` expects RGB, not
OpenCV's native BGR, so `_crop_person` flips channels before handing
the crop to the model. When a mask polygon is available (not just the
bbox), pixels outside the silhouette are zeroed out first so the
embedding focuses on the person rather than the background.
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
    """Resolves `FeatureExtractor`, handling two different layouts of
    the 'torchreid' package: the original project
    (github.com/KaiyangZhou/deep-person-reid) exposes `torchreid.utils`
    as a real subpackage, while the third-party PyPI "torchreid-pip"
    distribution (what plain `pip install torchreid` gets by default)
    hides everything under `torchreid.reid.*` and only rebinds `utils`
    as an attribute on the top-level package -- so
    `from torchreid.utils import FeatureExtractor` raises
    `ModuleNotFoundError` on that layout even though the package is
    installed correctly. Importing `torchreid` and accessing
    `torchreid.utils.FeatureExtractor` by attribute works on both."""
    import torchreid
    return torchreid.utils.FeatureExtractor


class OSNetEmbedder:
    """Minimal wrapper over `torchreid.utils.FeatureExtractor` for a
    single OSNet model.

        embedder = OSNetEmbedder(device="cpu")  # or "cuda"
        vec = embedder.embed(frame_bgr, bbox_xyxy, poly=poly)  # None if the crop is too poor to trust
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
    (same check as `_resolve_feature_extractor`, not just a bare
    `import torchreid`, which can succeed even when that attribute
    isn't reachable -- see that function's docstring). Doesn't
    instantiate a model or touch the GPU. For availability checks only:
    if OSNet is explicitly requested but unavailable,
    `OSNetEmbedder.__init__` still raises with install instructions
    rather than silently skipping the signal."""
    try:
        _resolve_feature_extractor()
    except (ImportError, AttributeError):
        return False
    except Exception as exc:
        # torchreid is unmaintained and can fail on import with something
        # other than ImportError on newer numpy/torch (e.g. np.float/
        # np.int removed in numpy>=1.24, still referenced by some
        # torchreid/yacs versions -> AttributeError instead). Caught
        # here and reported, rather than left to propagate and look like
        # an unrelated crash.
        print(f"[appearance_embedding] torchreid installed but the import fails "
              f"({type(exc).__name__}: {exc}) -- OSNet embedding not available. "
              f"Likely a version incompatibility (torchreid is a package "
              f"no longer actively maintained, often conflicting with recent "
              f"numpy/torch).")
        return False
    return True


def ema_update(prev: np.ndarray | None, new: np.ndarray | None,
                alpha: float = 0.9) -> np.ndarray | None:
    """Exponential moving average of an embedding over time (StrongSORT-
    style "feature bank" update): blends in a new frame's embedding
    without letting one noisy frame (motion blur, occlusion) swing the
    stored signature. High `alpha` (default 0.9) weights history over
    the current frame. Re-normalized to unit norm after averaging."""
    if new is None:
        return prev
    if prev is None:
        return new
    updated = alpha * prev + (1.0 - alpha) * new
    norm = np.linalg.norm(updated)
    if norm < 1e-9:
        return prev
    return updated / norm
