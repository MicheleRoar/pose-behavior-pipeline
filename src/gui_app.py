"""
gui_app.py
===========
Lancia l'interfaccia grafica locale (Tkinter) per la pipeline pose/
segmentazione. Va eseguito da dentro `src/`:

    cd src && python gui_app.py

Sta a livello di `src/` (non dentro `gui/`) per lo stesso motivo di
`live_demo.py` / `segmentation_demo.py` / `pipeline.py`: quando Python
esegue uno script direttamente, aggiunge automaticamente la SUA cartella
(`src/`) in cima a `sys.path`, il che rende risolvibili gli import
`from pose...`, `from segmentation...`, `from common...` usati da tutta la
pipeline senza bisogno di manipolare sys.path a mano. Lanciando invece
`python gui/app.py` direttamente, sys.path[0] sarebbe `src/gui/` invece di
`src/`, e quegli import fallirebbero.

Richiede una sorgente video reale e un display grafico: non eseguibile
nell'ambiente sandbox usato per sviluppare il resto della pipeline. La
logica di cache/seek dietro Play/Avanti/Indietro (in `gui/video_player.py`)
e' invece verificata separatamente, senza GUI ne' video, in
`demo/video_player_check.py`.
"""

from __future__ import annotations

from gui.app import run_gui

if __name__ == "__main__":
    run_gui()
