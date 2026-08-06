"""
samurai_estimation.py
========================
`SamuraiTracker`: backend di segmentazione/tracking basato su SAMURAI
(yangchris11/samurai, SAM2 + memoria motion-aware). Stessa struttura di
`Sam31Tracker` (sam31_estimation.py): tutta la logica condivisa vive in
`segmentation/sam_backend.py::ChunkedVideoPredictorBackend`, qui solo come
costruire il predictor.

A differenza di SAM 3.1, i checkpoint SAM2/SAMURAI sono PUBBLICI (nessun
accesso gated da richiedere su Hugging Face) -- utile come alternativa piu'
rapida da provare se l'approvazione dell'accesso a SAM 3.1 richiedesse
tempo. Requisiti: vedi https://github.com/yangchris11/samurai (basato su
facebookresearch/sam2, richiede comunque una GPU CUDA).

Import ritardato, stesso motivo di `Sam31Tracker`.
"""

from __future__ import annotations

from pathlib import Path

from segmentation.sam_backend import ChunkedVideoPredictorBackend

# Percorso di default, assumendo che 'samurai/' sia clonato come cartella
# SORELLA di questo repository (vedi README -- "clonalo fuori da
# pose-behavior-pipeline"), con i checkpoint scaricati nella sua
# sottocartella checkpoints/. Verificato su una macchina CUDA reale (non
# solo dedotto): un percorso assoluto tipo "/home/utente/..." si sarebbe
# rotto su qualunque altra macchina/utente, quindi si calcola relativamente
# alla posizione di QUESTO file invece che alla working directory da cui
# viene lanciata la GUI (che cambiava a seconda di come si avviava
# webui_app.py). Sovrascrivibile passando `checkpoint=` esplicito al
# costruttore se la tua struttura di cartelle e' diversa.
DEFAULT_CHECKPOINT = str(
    Path(__file__).resolve().parents[3] / "samurai" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
)

# NON solo "sam2.1_hiera_b+.yaml": Hydra (usato internamente da SAM2/SAMURAI
# per caricare le config) cerca questo file DENTRO il package sam2 con
# questo percorso relativo -- verificato su una macchina CUDA reale, l'
# errore osservato passando solo il nome del file era "Cannot find primary
# config 'sam2.1_hiera_b+.yaml'". Esiste anche una config SAM2.1 "standard"
# con lo stesso nome file in un percorso diverso: questa e' quella
# SPECIFICA di SAMURAI (con la memoria motion-aware), da non confondere.
DEFAULT_CONFIG = "configs/samurai/sam2.1_hiera_b+.yaml"


class SamuraiTracker(ChunkedVideoPredictorBackend):
    """Vedi il docstring del modulo e di `ChunkedVideoPredictorBackend`.
    `checkpoint` e `config` seguono la convenzione di samurai/sam2 (nome
    file .pt + config .yaml associato)."""

    def __init__(self, *, checkpoint: str = DEFAULT_CHECKPOINT,
                 config: str = DEFAULT_CONFIG, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.config = config

    def _build_predictor(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "SamuraiTracker richiede il pacchetto 'sam2' con le patch SAMURAI "
                "(non installato). Vedi https://github.com/yangchris11/samurai -- "
                "checkpoint pubblici (nessun accesso gated richiesto)."
            ) from exc
        return build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)
