"""
backend.py
===========
Protocollo minimo che ogni motore di segmentazione/tracking deve rispettare
per poter essere collegato a `segmentation_demo.py` / `pipeline_runner.py`
senza toccarli: un solo metodo, `run(source, stream=True)`, che restituisce
un iteratore di `SegFrameResult` (frame_index, frame, people = lista di
(track_id, bbox_xyxy, mask_polygon, box_conf)).

`SegTracker` (YOLO26-seg + ByteTrack, seg_estimation.py) rispetta gia'
questo protocollo per costruzione, senza bisogno di modifiche: e' definito
QUI a posteriori solo per rendere esplicito il contratto che anche i nuovi
backend (`Sam31Tracker`, `Sam2Tracker`, vedi sam_backend.py) devono
rispettare. E' un `typing.Protocol` (duck typing statico): non serve
ereditare da nessuna classe base, basta avere lo stesso metodo con la stessa
firma -- `isinstance(x, SegmentationBackend)` funziona comunque grazie a
`@runtime_checkable`, usato nei test per verificare la conformita' senza
dover istanziare davvero YOLO/SAM.

Perche' questo disaccoppiamento conta qui: `iter_segmentation_frames()` (in
segmentation_demo.py) e tutto cio' che le sta sopra (GUI Tkinter e web,
VideoPlayer, export CSV) trattano il backend come una scatola nera che
produce lo stesso tipo di risultato per frame -- selezionare YOLO, SAM 3.1 o
SAM2 diventa quindi solo una scelta di QUALE classe istanziare dentro
`iter_segmentation_frames()`, zero altre modifiche a valle.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from segmentation.seg_estimation import SegFrameResult


@runtime_checkable
class SegmentationBackend(Protocol):
    """Contratto minimo: un oggetto costruito con i suoi parametri specifici
    (modello, device, soglie, ...) e un metodo `run()` che, data una
    sorgente video, restituisce un iteratore di `SegFrameResult` in ordine
    di frame crescente, senza salti."""

    def run(self, source, stream: bool = True) -> Iterator[SegFrameResult]:
        ...
