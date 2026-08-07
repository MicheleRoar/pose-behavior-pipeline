"""
gui_app.py
===========
Launches the local (Tkinter) graphical interface for the pose/segmentation
pipeline. Must be run from inside `src/`:

    cd src && python gui_app.py

Lives at the `src/` level (not inside `gui/`) for the same reason as
`live_demo.py` / `segmentation_demo.py` / `pipeline.py`: when Python runs a
script directly, it automatically adds ITS OWN folder (`src/`) to the top
of `sys.path`, which makes the `from pose...`, `from segmentation...`,
`from common...` imports used throughout the pipeline resolvable without
manually touching sys.path. Running `python gui/app.py` directly instead
would make sys.path[0] `src/gui/` instead of `src/`, and those imports
would fail.

Requires a real video source and a graphical display: not runnable in the
sandbox environment used to develop the rest of the pipeline. The
cache/seek logic behind Play/Forward/Back (in `gui/video_player.py`) is
instead verified separately, with no GUI or video, in
`demo/video_player_check.py`.
"""

from __future__ import annotations

from gui.app import run_gui

if __name__ == "__main__":
    run_gui()
