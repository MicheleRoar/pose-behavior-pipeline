"""
yolo_models.py
================
Resolves bare Ultralytics YOLO weight names (e.g. "yolo26n-pose.pt")
into a fixed cache inside the project (`<repo>/models/`), independent
of the current working directory -- same fix, same motivation, as
`common/mediapipe_models.py::resolve_model_path()` (see its docstring
for the original bug this class of fix addresses: a bare default name
resolved relative to the cwd works only when launching from the one
folder where the file happened to land, and fails elsewhere), applied
here to the Ultralytics side instead of MediaPipe Tasks. Kept as a
separate small module rather than folding into `mediapipe_models.py`:
`tests/mediapipe_models_check.py` monkeypatches
`mediapipe_models.MODELS_CACHE_DIR` directly, so that module's public
surface is left untouched here.

Unlike `mediapipe_models.py`, this module does NOT download anything
itself: Ultralytics' own `YOLO(path)` constructor already auto-downloads
the official pretrained weights when given a path whose basename
matches a known release asset and the file doesn't exist yet, creating
any missing parent directories (`ultralytics.utils.downloads`) --
this is the documented/expected Ultralytics behavior, NOT verified end
to end with a real download in this sandbox (no network / ultralytics
not exercised here, same honesty caveat as elsewhere in the project for
things only checked by reading library source, not by running it).
`resolve_yolo_weights()` only computes WHERE that download should land,
so it ends up in `models/` instead of wherever the script happened to
be launched from.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../pose-behavior-pipeline/src/common/yolo_models.py -> parents[2] is
# the project root (src/common -> src -> root) -- same convention as
# MODELS_CACHE_DIR in mediapipe_models.py.
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# .../src/common/yolo_models.py -> parents[1] is `src/` -- the tracker
# YAMLs live at `src/configs/`, one level up from models/ (project
# root), NOT two -- see resolve_tracker_config() below.
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def resolve_yolo_weights(model_name: str) -> str:
    """If `model_name` already exists as a file (an explicit path,
    including one already sitting in `models/` or anywhere else), it's
    returned unchanged. Otherwise, if it's a BARE name (no directory
    component -- a plain weight filename like "yolo26n-pose.pt", not a
    path the caller got wrong), it's redirected to `MODELS_DIR /
    model_name`: Ultralytics then downloads it there on first use. A
    name that already has a directory component (e.g.
    "some/other/dir/custom.pt") is left untouched -- same "don't guess
    on an explicit path" rule as `mediapipe_models.resolve_model_path`.
    """
    if os.path.isfile(model_name):
        return model_name
    if os.path.dirname(model_name):
        return model_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR / model_name)


def resolve_tracker_config(tracker: str) -> str:
    """Same fix, same motivation as `resolve_yolo_weights()` above,
    applied to Ultralytics tracker YAML configs instead of model
    weights.

    BUG (Michele, 2026-08, real run on his Linux/CUDA machine):
    "bytetrack_permissive.yaml does not exist". The GUI passes the
    literal string "configs/bytetrack_permissive.yaml" straight
    through to `model.track(tracker=...)` (see webui/index.html's
    `<option value="configs/bytetrack_permissive.yaml">`) -- Ultralytics
    resolves a relative path like that against the CURRENT WORKING
    DIRECTORY at launch time, not against this project's own
    `src/configs/` folder. It only worked when the app happened to be
    launched with `src/` as the cwd (as webui_app.py's docstring says
    it must be); it breaks the moment it's launched any other way (a
    desktop shortcut, a different terminal cwd, an IDE run
    configuration, a systemd unit, ...).

    A BARE name Ultralytics already knows on its own ("bytetrack.yaml",
    "botsort.yaml" -- no directory component: these resolve fine by
    themselves, Ultralytics ships them inside its own package) is
    returned unchanged, same as an explicit path that already resolves
    from the current cwd. Only a "configs/<file>.yaml"-shaped value
    that does NOT already resolve gets redirected to this project's
    real `src/configs/` folder, independent of the cwd."""
    if os.path.isfile(tracker):
        return tracker
    if not os.path.dirname(tracker):
        # bare name, e.g. "bytetrack.yaml" -- Ultralytics resolves
        # these against its OWN bundled configs, leave untouched.
        return tracker
    candidate = CONFIGS_DIR / os.path.basename(tracker)
    if candidate.is_file():
        return str(candidate)
    return tracker  # unresolvable either way -- let Ultralytics raise its own clear error
