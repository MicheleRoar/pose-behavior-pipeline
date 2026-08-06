"""
sam31_estimation.py
=====================
`Sam31Tracker`: backend di segmentazione/tracking basato su SAM 3.1
(facebookresearch/sam3, Object Multiplex). Tutta la logica (chunking,
seeding degli id, riconciliazione, persistenza) vive in
`segmentation/sam_backend.py::ChunkedVideoPredictorBackend` -- qui solo la
parte specifica: come costruire il predictor SAM 3.1.

Requisiti (verificati SOLO da documentazione, non da esecuzione reale in
questo ambiente -- nessuna GPU CUDA qui): Python 3.12+, PyTorch 2.7+, CUDA
12.6+, pacchetto `sam3` (`pip install -e .` da
https://github.com/facebookresearch/sam3), checkpoint gated su Hugging Face
(`facebook/sam3.1`, richiede accesso approvato + `hf auth login` -- vedi
README). Import ritardato: il resto del pacchetto resta importabile/
testabile senza `sam3` installato (stessa scelta di `ultralytics`/`torch`
altrove nel progetto).

Interfaccia usata (dalla documentazione ufficiale): `build_sam3_video_predictor()`
o equivalente, poi lo stesso pattern `init_state`/`add_new_points_or_box`/
`propagate_in_video` gia' assunto dalla classe base -- ADATTARE i nomi
esatti (potrebbero differire leggermente da versione a versione del
repository) verificando `examples/sam3_video_predictor_example.ipynb` nel
repository ufficiale sulla macchina CUDA prima del primo uso reale.
"""

from __future__ import annotations

from segmentation.sam_backend import ChunkedVideoPredictorBackend

DEFAULT_CHECKPOINT = "facebook/sam3.1"


class Sam31Tracker(ChunkedVideoPredictorBackend):
    """Vedi il docstring del modulo e di `ChunkedVideoPredictorBackend` per
    il disegno completo. `checkpoint` e' l'id del repository Hugging Face
    (gated, richiede accesso approvato) o un percorso locale gia'
    scaricato."""

    def __init__(self, *, checkpoint: str = DEFAULT_CHECKPOINT, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint

    def _build_predictor(self):
        # Import ritardato: solleva ImportError con un messaggio chiaro se
        # il pacchetto sam3 non e' installato, invece di rompere l'import
        # dell'intero modulo `segmentation` per chi non usa questo backend.
        try:
            from sam3.video_predictor import build_sam3_video_predictor
        except ImportError as exc:
            raise ImportError(
                "Sam31Tracker richiede il pacchetto 'sam3' (non installato). "
                "Vedi https://github.com/facebookresearch/sam3 -- "
                "'git clone ... && pip install -e .', checkpoint gated su "
                "Hugging Face (facebook/sam3.1, richiede accesso approvato)."
            ) from exc
        return build_sam3_video_predictor(checkpoint=self.checkpoint, device=self.device)
