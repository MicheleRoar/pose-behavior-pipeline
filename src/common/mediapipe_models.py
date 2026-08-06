"""
mediapipe_models.py
=====================
Risoluzione/download automatico dei modelli MediaPipe Tasks (`.task`)
usati da `pose/mediapipe_pose.py` (Pose Landmarker), `pose/hands.py` (Hand
Landmarker) e `pose/gaze_head.py` (Face Landmarker) -- helper condiviso
per non triplicare la stessa logica nei tre moduli.

Perche' esiste (bug reale, osservato da Michele su macchina Linux)
---------------------------------------------------------------------
I default originali di questi tre moduli erano nomi nudi tipo
`"hand_landmarker.task"`, risolti da MediaPipe come path RELATIVO ALLA
CWD -- funzionava solo lanciando lo script dalla stessa cartella in cui
si era fatto il `curl` manuale, falliva con un errore poco chiaro tipo
"unable to find hand_landmarker" lanciando da una cwd diversa (es.
`cd src && python webui_app.py` invece che dalla cartella in cui il file
era stato scaricato a mano). Il fix per `pose_landmarker_lite.task`
(mediapipe_pose.py) e' stato il primo; qui la stessa logica viene
riusata per `hand_landmarker.task`/`face_landmarker.task`, che avevano
esattamente lo stesso problema (confermato: Michele ha dovuto creare un
symlink a mano come workaround prima di questo fix).

`resolve_model_path()` risolve il nome nudo di default in una cache
FISSA dentro il progetto (`<repo>/models/`, indipendente da dove viene
lanciato lo script) e scarica il file li' se assente -- nessun `curl`/
symlink manuale piu' necessario. Un path esplicito passato dall'utente
(gia' esistente, o un nome diverso dal default nudo) resta invariato,
nessuna sovrascrittura -- permette di puntare a varianti gia' scaricate
altrove (es. "full"/"heavy" invece di "lite").
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# .../pose-behavior-pipeline/src/common/mediapipe_models.py -> parents[2]
# e' la root del progetto (src/common -> src -> root), stessa convenzione
# di Path(__file__).resolve().parents[N] gia' usata in
# segmentation/sam2_estimation.py.
MODELS_CACHE_DIR = Path(__file__).resolve().parents[2] / "models"


def resolve_model_path(model_path: str, *, download_url: str) -> str:
    """Se `model_path` esiste gia' come file (path esplicito dell'utente,
    anche relativo alla cwd corrente -- comportamento invariato per chi lo
    passa apposta), lo usa cosi' com'e'. Altrimenti, SOLO se il suo nome e'
    esattamente il nome nudo di default (l'ultimo pezzo di `download_url`,
    non un path custom che l'utente ha sbagliato -- in quel caso e'
    meglio l'errore originale di MediaPipe che indovinare), lo risolve
    nella cache fissa del progetto (`MODELS_CACHE_DIR`), scaricandolo li'
    se non presente."""
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
    """Scarica `url` in `dest`, creando le cartelle mancanti. Nessun
    retry/hash-check: se il download si interrompe a meta', il file
    parziale resta li' e il prossimo avvio lo tratterebbe come 'gia'
    presente' (bug noto, accettabile per ora -- se capita, basta
    cancellare il file e rilanciare)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mediapipe_models] scarico {dest.name} (una tantum) in {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[mediapipe_models] fatto: {dest}")
