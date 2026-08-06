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

from segmentation.sam_backend import ChunkedVideoPredictorBackend

DEFAULT_CHECKPOINT = "sam2.1_hiera_base_plus.pt"


class SamuraiTracker(ChunkedVideoPredictorBackend):
    """Vedi il docstring del modulo e di `ChunkedVideoPredictorBackend`.
    `checkpoint` e `config` seguono la convenzione di samurai/sam2 (nome
    file .pt + config .yaml associato)."""

    def __init__(self, *, checkpoint: str = DEFAULT_CHECKPOINT,
                 config: str = "sam2.1_hiera_b+.yaml", **kwargs):
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
