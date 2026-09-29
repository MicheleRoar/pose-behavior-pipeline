"""
mediapipe_models.py
=====================
Automatic resolution/download of MediaPipe Tasks models (`.task`) used by
`pose/mediapipe_pose.py` (Pose Landmarker), `pose/hands.py` (Hand
Landmarker), and `pose/gaze_head.py` (Face Landmarker) -- shared helper so
the same logic isn't tripled across the three modules.

Why this exists (real bug, observed by Michele on a Linux machine)
---------------------------------------------------------------------
The original defaults for these three modules were bare names like
`"hand_landmarker.task"`, resolved by MediaPipe as a path RELATIVE TO THE
CWD -- this only worked when launching the script from the same folder
where the manual `curl` had been done, and failed with an unclear error
like "unable to find hand_landmarker" when launching from a different cwd
(e.g. `cd src && python webui_app.py` instead of the folder where the file
had been manually downloaded). The fix for `pose_landmarker_lite.task`
(mediapipe_pose.py) was the first one; here the same logic is reused for
`hand_landmarker.task`/`face_landmarker.task`, which had exactly the same
problem (confirmed: Michele had to manually create a symlink as a
workaround before this fix).

`resolve_model_path()` resolves the bare default name into a FIXED cache
inside the project (`<repo>/models/`, independent of where the script is
launched from) and downloads the file there if missing -- no more manual
`curl`/symlink needed. An explicit path passed by the user (already
existing, or a name different from the bare default) is left unchanged,
never overwritten -- this allows pointing to variants already downloaded
elsewhere (e.g. "full"/"heavy" instead of "lite").
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# .../pose-behavior-pipeline/src/common/mediapipe_models.py -> parents[2]
# is the project root (src/common -> src -> root), same convention as
# Path(__file__).resolve().parents[N] already used in
# segmentation/sam2_estimation.py.
MODELS_CACHE_DIR = Path(__file__).resolve().parents[2] / "models"


def resolve_model_path(model_path: str, *, download_url: str) -> str:
    """If `model_path` already exists as a file (an explicit path from the
    user, even relative to the current cwd -- unchanged behavior for
    whoever passes it on purpose), it's used as-is. Otherwise, ONLY if its
    name is exactly the bare default name (the last piece of
    `download_url`, not a custom path the user got wrong -- in that case
    MediaPipe's original error is better than guessing), it's resolved
    into the project's fixed cache (`MODELS_CACHE_DIR`), downloading it
    there if not present."""
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
    check: if the download is interrupted halfway, the partial file stays
    there and the next startup would treat it as 'already present' (known
    bug, acceptable for now -- if it happens, just delete the file and
    relaunch)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mediapipe_models] downloading {dest.name} (one-time) to {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[mediapipe_models] done: {dest}")
