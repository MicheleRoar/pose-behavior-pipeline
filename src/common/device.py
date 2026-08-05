"""
device.py
==========
Autodetect del device di calcolo migliore disponibile (CUDA > MPS > CPU),
cosi' non serve impostare a mano `--device`/il device della GUI su ogni
macchina diversa. Prima di questo modulo, `device="mps"` era fisso in
entrambe le GUI (gui/app.py, webui/api.py + webui/app.js) e come default
argparse in tutte le CLI: su un PC con GPU NVIDIA (CUDA, niente Metal)
questo faceva fallire silenziosamente la pipeline invece di usare la GPU
disponibile.

Le CLI mantengono comunque `--device` come override esplicito -- questo
modulo fornisce solo il DEFAULT quando l'utente non specifica nulla.
"""

from __future__ import annotations


def detect_default_device() -> str:
    """"cuda" se una GPU NVIDIA e' disponibile via PyTorch, altrimenti
    "mps" se il device Metal di Apple Silicon e' disponibile, altrimenti
    "cpu". Import di torch dentro la funzione (non in cima al modulo), per
    coerenza con lo stile "import pesante ritardato" gia' usato altrove nel
    progetto per le dipendenze pesanti -- anche se qui torch e' comunque
    una dipendenza indiretta obbligatoria (arriva con ultralytics), quindi
    l'import non fallisce mai in pratica."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"
