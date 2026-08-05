"""
webui_app.py
=============
Lancia la nuova interfaccia grafica "Behaviour Vision Lab" (finestra
pywebview che carica webui/index.html, con webui/api.py come bridge verso
la pipeline). Va eseguito da dentro `src/`:

    cd src && python webui_app.py

Sta a livello di `src/` (non dentro `webui/`) per lo stesso motivo di
`gui_app.py` / `live_demo.py` / `segmentation_demo.py`: quando Python esegue
uno script direttamente, aggiunge automaticamente la SUA cartella (`src/`)
in cima a `sys.path`, il che rende risolvibili gli import `from gui...`,
`from pose...`, `from segmentation...` usati da webui/api.py senza bisogno
di manipolare sys.path a mano. Lanciando invece `python webui/api.py` (o un
ipotetico `webui/app.py`) direttamente, sys.path[0] sarebbe `src/webui/`
invece di `src/`, e quegli import fallirebbero.

Questa e' un'interfaccia ALTERNATIVA a quella Tkinter (`gui_app.py`), non
una sostituzione: stessa pipeline sotto (VideoPlayer/iter_pipeline_frames,
invariati), presentazione diversa. Vedi README per quale scegliere:
- `gui_app.py` (Tkinter): nessuna dipendenza aggiuntiva oltre Pillow, avvio
  piu' leggero, look nativo ma limitato (niente switch arrotondati, gradienti,
  card scure come nel mock "Behaviour Vision Lab").
- `webui_app.py` (questo file, pywebview + HTML/CSS/JS): richiede
  `pip install pywebview`, riproduce fedelmente il mock fornito.

Richiede una sorgente video reale e un display grafico: non eseguibile
nell'ambiente sandbox usato per sviluppare il resto della pipeline. La
logica pura del bridge (parametri, codifica frame, metriche) e' invece
verificata separatamente, senza pywebview ne' finestra, in
`demo/webui_api_check.py`; la logica di cache/seek dietro Play/Indietro/
Avanti/timeline in `demo/video_player_check.py`.
"""

from __future__ import annotations

from pathlib import Path

from webui.api import Api

INDEX_HTML = Path(__file__).resolve().parent / "webui" / "index.html"


def run_webui() -> None:
    import webview  # import ritardato: solo chi lancia questa GUI ha bisogno di pywebview

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
