"""
segmentation/merging/mask_utils.py
===================================
Small, self-contained mask/polygon geometry and color helpers used by
the rest of `merging/` -- bounding boxes, hue histograms, centroids,
pixel-to-pixel distance. numpy/cv2 only, no other project dependency.
"""

from __future__ import annotations

import cv2
import numpy as np

_HUE_HIST_BINS = 16  # 180deg/16 = 11.25deg/bin -- coarse enough to withstand
                      # lighting noise, fine enough to separate two distinct colors


def _polygon_to_box(poly: np.ndarray) -> np.ndarray:
    """Bounding box (x1,y1,x2,y2) of the polygon. `[0,0,0,0]` if empty."""
    if poly.shape[0] == 0:
        return np.zeros(4)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    return np.array([x1, y1, x2, y2], dtype=float)


def _mask_hue_histogram(frame: np.ndarray, poly: np.ndarray,
                         bins: int = _HUE_HIST_BINS) -> np.ndarray | None:
    """Hue histogram (OpenCV Hue, 0-179) of the pixels inside the mask
    polygon, weighted by saturation and normalized to sum 1. Captures a
    two-tone/striped garment as two peaks, unlike a single average hue.
    `None` if the polygon is empty/degenerate or too few trustworthy
    (non-desaturated) pixels remain."""
    if poly.shape[0] < 3:
        return None
    h, w = frame.shape[:2]
    pts = np.round(poly).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    if cv2.countNonZero(mask) < 25:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    hue = hsv[ys, xs, 0].astype(np.float64)          # 0..179
    sat = hsv[ys, xs, 1].astype(np.float64) / 255.0  # 0..1, used as weight

    valid = sat > 0.15  # nearly gray pixels: their hue is sensor noise
    if valid.sum() < 25:
        return None
    hue, sat = hue[valid], sat[valid]

    hist, _ = np.histogram(hue, bins=bins, range=(0, 180), weights=sat)
    total = hist.sum()
    if total <= 0:
        return None
    return hist / total


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """0..1 similarity between two normalized hue histograms via
    intersection (sum(min(a, b))): 1.0 identical, 0.0 no overlap."""
    return float(np.clip(np.minimum(a, b).sum(), 0.0, 1.0))


def _mask_to_polygon_single_component(mask_frame: np.ndarray) -> np.ndarray:
    """Polygon of the mask frame's connected component, but ONLY if
    there's EXACTLY ONE region (empty `(0,2)` polygon otherwise, e.g. a
    stray second blob makes the frame too ambiguous to trust)."""
    mask_u8 = mask_frame.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return np.empty((0, 2))
    return contours[0].reshape(-1, 2).astype(float)


def _touches_border(poly: np.ndarray, height: int, width: int, margin: int = 1) -> bool:
    """True if `poly`'s bounding box comes within `margin` px of the
    frame edge -- used to reject a single-component frame whose one
    region is actually a stray artifact stuck to the border rather than
    the tracked person."""
    if poly.shape[0] == 0:
        return False
    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    return bool(
        x_min <= margin or y_min <= margin
        or x_max >= width - 1 - margin or y_max >= height - 1 - margin
    )


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Pixel centroid `(x, y)` of a boolean mask (caller ensures `mask.any()`)."""
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Euclidean distance between two masks' centroids."""
    ax, ay = _mask_centroid(mask_a)
    bx, by = _mask_centroid(mask_b)
    return float(np.hypot(ax - bx, ay - by))


def _mask_bbox_diagonal(mask: np.ndarray) -> float:
    """Diagonal (px) of a mask's tight bounding box -- a cheap proxy for
    "how big is this fragment on screen right now", used to normalize a
    raw pixel distance into a scale-portable ratio."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0
    h = float(ys.max() - ys.min())
    w = float(xs.max() - xs.min())
    return float(np.hypot(h, w))


def _mask_min_pixel_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Minimum real pixel-to-pixel distance between two masks'
    silhouettes (0.0 if they touch/overlap) -- via `cv2.distanceTransform`
    on the complement of `mask_a`, read off at `mask_b`'s pixels.
    O(H*W), and unlike a bounding-box gap, doesn't false-positive when
    two boxes touch but the actual silhouettes are far apart."""
    inv_a = (~mask_a).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv_a, cv2.DIST_L2, 5)
    return float(dist_map[mask_b].min())
