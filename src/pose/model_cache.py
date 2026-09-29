"""
pose/model_cache.py
====================
Ported (near-verbatim) from `pose-behavior-pipeline_legacy/src/common/mediapipe_models.py`,
so `extract_keypoints.py` reuses the same fix instead of reinventing it.

Why this exists (real bug, already hit once in the legacy pipeline)
---------------------------------------------------------------------
A bare model filename like `"pose_landmarker_lite.task"` is resolved by
MediaPipe as a path RELATIVE TO THE CWD -- this only works when launching
from the exact folder a manual `curl` happened to be done in, and fails
with an unclear error ("unable to find pose_landmarker_lite.task") when
launching from anywhere else (e.g. `cd src && python -m pose.run_pose`).
The legacy pipeline hit this for real (Michele had to manually symlink the
file as a workaround) before fixing it with the pattern ported here:
resolve the bare default name into a FIXED cache inside the project
(`<repo>/models/`, independent of cwd) and download it there once if
missing. An explicit path the caller already resolved (pre-existing file,
or any name other than the bare default) is left untouched.

Kept as its own small module (rather than inlined in extract_keypoints.py)
so the same logic could be reused later by other MediaPipe-based steps
(hands/gaze, if ever ported) without duplicating it -- same reasoning the
legacy repo gives for factoring it out of pose/mediapipe_pose.py.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# .../pose-behavior-pipeline/src/pose/model_cache.py -> parents[2] is the
# project root (src/pose -> src -> root), same depth/convention as the
# legacy module (src/common -> src -> root).
MODELS_CACHE_DIR = Path(__file__).resolve().parents[2] / "models"


def resolve_model_path(model_path: str, *, download_url: str) -> str:
    """If `model_path` already exists as a file (an explicit path the
    caller passed on purpose, even relative to the current cwd -- left
    unchanged), it's used as-is. Otherwise, ONLY if its name is exactly
    the bare default name (the last path segment of `download_url`, not a
    custom path that's simply wrong -- in that case MediaPipe's own error
    is more informative than guessing), it's resolved into the project's
    fixed cache (`MODELS_CACHE_DIR`), downloading it there if not already
    present."""
    if os.path.isfile(model_path):
        return model_path
    default_basename = download_url.rsplit("/", 1)[-1]
    if os.path.basename(model_path) != default_basename:
        return model_path
    cache_path = MODELS_CACHE_DIR / default_basename
    if not cache_path.exists():
        _download(download_url, cache_path)
    return str(cache_path)


def _download(url: str, dest: Path) -> None:
    """Downloads `url` to `dest`, creating missing folders. No retry/hash
    check: if the download is interrupted halfway, the partial file is
    left there and would be treated as 'already present' on the next run
    (known limitation, inherited as-is from the legacy module -- if it
    happens, delete the file and rerun)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model_cache] downloading {dest.name} (one-time) to {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[model_cache] done: {dest}")
