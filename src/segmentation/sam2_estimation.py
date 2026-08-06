"""
sam2_estimation.py
=====================
`Sam2Tracker`: backend di segmentazione/tracking basato su SAM2 "vanilla"
(facebookresearch/sam2). Stessa struttura di `Sam31Tracker`: tutta la
logica condivisa (chunking, seeding degli id, riconciliazione, persistenza)
vive in `segmentation/sam_backend.py::ChunkedVideoPredictorBackend`, qui
solo come costruire il predictor.

Perche' SAM2 vanilla e non SAMURAI
------------------------------------
Questo modulo sostituisce `samurai_estimation.py` (rimosso). SAMURAI
aggiunge sopra SAM2 un filtro di Kalman ("motion-aware mask selection",
dentro `sam2_base.py::_forward_sam_heads`) scritto e validato SOLO per il
visual object tracking single-target: i benchmark su cui e' stato provato
(LaSOT, GOT-10k, TrackingNet) tracciano UN oggetto per video, mai piu' di
uno insieme.

Verificato su una macchina CUDA reale: seminando piu' persone nella stessa
sessione (il caso normale qui -- piu' bambini/terapista in campo insieme),
SAM2 le raggruppa in un unico batch per l'inferenza, ma il codice Kalman di
SAMURAI assume un batch di dimensione 1 -- va in crash con

    RuntimeError: Boolean value of Tensor with more than one value is
    ambiguous

dentro `_forward_sam_heads()` (riga `ious[0][best_iou_inds]`: con piu' di
un oggetto tracciato, `best_iou_inds` ha una entry per persona invece che
uno scalare, e l'indicizzazione risultante non e' piu' un singolo valore).
Farlo funzionare per N persone richiederebbe una sessione SAM separata per
ciascuna (N volte il costo dell'encoder, la parte piu' pesante di tutta la
pipeline) -- non ne vale la pena rispetto a SAM2 vanilla, che supporta il
multi-oggetto batchato NATIVAMENTE (nessuna patch, nessun crash), pur senza
il motion-modeling attraverso le occlusioni che era il vero valore aggiunto
di SAMURAI per il single-target tracking.

Requisiti: pacchetto `sam2` (facebookresearch/sam2, `git clone` + `pip
install -e .`), checkpoint PUBBLICI (nessun accesso gated). Se hai gia'
clonato `samurai/` per il tentativo precedente, va bene riusare lo STESSO
file .pt (SAMURAI non riaddestra i pesi di SAM2, applica solo il filtro di
Kalman a inferenza) passando `checkpoint=".../samurai/checkpoints/
sam2.1_hiera_base_plus.pt"` esplicito al costruttore -- basta puntare `config`
alla config SAM2.1 "standard" invece di quella specifica di samurai
(`configs/samurai/...`), vendorizzata nello stesso checkout. In alternativa,
un clone pulito di facebookresearch/sam2 e' la dipendenza piu' diretta per
chi non ha bisogno di nient'altro di SAMURAI.

Import ritardato, stesso motivo di `Sam31Tracker`.
"""

from __future__ import annotations

from pathlib import Path

from segmentation.sam_backend import ChunkedVideoPredictorBackend

# Stessa convenzione "cartella sorella" gia' usata per samurai/ (vedi
# README -- "clonalo fuori da pose-behavior-pipeline"): un clone pulito di
# facebookresearch/sam2 ha la stessa struttura (sam2/checkpoints/*.pt).
# Sovrascrivibile passando `checkpoint=` esplicito al costruttore, es. per
# riusare il checkpoint gia' scaricato dentro un checkout samurai/.
DEFAULT_CHECKPOINT = str(
    Path(__file__).resolve().parents[3] / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
)

# Config SAM2.1 "standard" -- NON quella di samurai (configs/samurai/...):
# e' questa la differenza che disattiva il filtro di Kalman single-object e
# abilita il batching multi-oggetto nativo, vedi il docstring del modulo.
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


class Sam2Tracker(ChunkedVideoPredictorBackend):
    """Vedi il docstring del modulo e di `ChunkedVideoPredictorBackend`.
    `checkpoint` e `config` seguono la convenzione di sam2 (nome file .pt +
    config .yaml associato)."""

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
                "Sam2Tracker richiede il pacchetto 'sam2' (non installato). "
                "Vedi https://github.com/facebookresearch/sam2 -- "
                "checkpoint pubblici (nessun accesso gated richiesto)."
            ) from exc
        return build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)
