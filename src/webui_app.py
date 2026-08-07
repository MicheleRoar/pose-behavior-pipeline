"""
webui_app.py
=============
Launches the new "Behaviour Vision Lab" graphical interface (a pywebview
window that loads webui/index.html, with webui/api.py as the bridge to the
pipeline). Must be run from inside `src/`:

    cd src && python webui_app.py

Lives at the `src/` level (not inside `webui/`) for the same reason as
`gui_app.py` / `live_demo.py` / `segmentation_demo.py`: when Python runs a
script directly, it automatically adds ITS OWN folder (`src/`) to the top
of `sys.path`, which makes the `from gui...`, `from pose...`, `from
segmentation...` imports used by webui/api.py resolvable without manually
touching sys.path. Running `python webui/api.py` (or a hypothetical
`webui/app.py`) directly instead would make sys.path[0] `src/webui/`
instead of `src/`, and those imports would fail.

This is an ALTERNATIVE interface to the Tkinter one (`gui_app.py`), not a
replacement: same pipeline underneath (VideoPlayer/iter_pipeline_frames,
unchanged), different presentation. See the README for which one to pick:
- `gui_app.py` (Tkinter): no extra dependency beyond Pillow, lighter
  startup, native but limited look (no rounded switches, gradients, dark
  cards like in the "Behaviour Vision Lab" mockup).
- `webui_app.py` (this file, pywebview + HTML/CSS/JS): requires
  `pip install pywebview`, faithfully reproduces the provided mockup.

Requires a real video source and a graphical display: not runnable in the
sandbox environment used to develop the rest of the pipeline. The bridge's
pure logic (parameters, frame encoding, metrics) is instead verified
separately, with no pywebview or window, in `tests/webui_api_check.py`; the
cache/seek logic behind Play/Back/Forward/timeline is verified in
`tests/video_player_check.py`.
"""

from __future__ import annotations

from pathlib import Path

from webui.api import Api

INDEX_HTML = Path(__file__).resolve().parent / "webui" / "index.html"


def run_webui() -> None:
    import webview  # delayed import: only whoever launches this GUI needs pywebview

    api = Api()
    window = webview.create_window(
        "Behaviour Vision Lab",
        url=str(INDEX_HTML),
        js_api=api,
        width=1440,
        height=900,
        min_size=(1024, 680),
    )
    api.set_window(window)
    webview.start()


if __name__ == "__main__":
    run_webui()
