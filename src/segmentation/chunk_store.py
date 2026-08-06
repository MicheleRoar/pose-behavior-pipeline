"""
chunk_store.py
================
Persistenza incrementale su disco dei risultati di un chunk (maschere + id
+ confidenze). `sam_backend.py` chiama `save_chunk()` non appena un chunk e'
completato, PRIMA di passare al successivo -- cosi' su un video lungo (o in
caso di crash/interruzione a meta' elaborazione) il lavoro gia' fatto non va
perso. La ripresa automatica da un chunk gia' salvato non e' ancora
implementata (solo scrittura/lettura per ora): resta un'estensione naturale
di `sam_backend.py` se servisse in pratica.

Formato: un file `.npz` per chunk, array paralleli `frame_index`,
`track_id`, `box`, `polygon`, `conf` -- uno per (persona, frame). `box` e
`polygon` sono array di oggetto (`dtype=object`, richiede
`allow_pickle=True` al caricamento) perche' ogni poligono ha un numero di
vertici diverso: incapsularli in una tabella piatta tipo CSV/parquet
richiederebbe comunque una colonna "oggetto" per lo stesso motivo. NON si
salva il frame (l'immagine): solo dati derivati, per non esplodere lo
spazio disco -- l'immagine si puo' sempre rileggere dal video sorgente per
indice frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from segmentation.seg_estimation import SegFrameResult


def chunk_filename(chunk_index: int) -> str:
    return f"chunk_{chunk_index:04d}.npz"


def save_chunk(results: list[SegFrameResult], out_dir: str, chunk_index: int) -> str:
    """Scrive su disco un chunk gia' completato (lista di `SegFrameResult`,
    uno per frame elaborato in questo chunk). Ritorna il percorso del file
    scritto. Un chunk senza nessuna persona rilevata in nessun frame produce
    comunque un file valido (array vuoti), cosi' `load_chunk()` non deve
    distinguere "chunk mancante" da "chunk vuoto"."""
    os.makedirs(out_dir, exist_ok=True)
    frame_indices: list[int] = []
    track_ids: list[int] = []
    boxes: list[np.ndarray] = []
    polygons: list[np.ndarray] = []
    confs: list[float] = []
    for r in results:
        for track_id, bbox, poly, conf in r.people:
            frame_indices.append(r.frame_index)
            track_ids.append(track_id)
            boxes.append(bbox)
            polygons.append(poly)
            confs.append(conf)

    path = os.path.join(out_dir, chunk_filename(chunk_index))
    np.savez_compressed(
        path,
        frame_index=np.asarray(frame_indices, dtype=np.int64),
        track_id=np.asarray(track_ids, dtype=np.int64),
        box=np.asarray(boxes, dtype=object) if boxes else np.empty((0,), dtype=object),
        polygon=np.asarray(polygons, dtype=object) if polygons else np.empty((0,), dtype=object),
        conf=np.asarray(confs, dtype=np.float32),
    )
    return path


@dataclass
class ChunkRecord:
    """Una riga (persona, frame) riletta da un chunk salvato."""
    frame_index: int
    track_id: int
    bbox: np.ndarray
    polygon: np.ndarray
    conf: float


def load_chunk(path: str) -> list[ChunkRecord]:
    """Rilegge un chunk salvato da `save_chunk()`. Ritorna una lista piatta
    di `ChunkRecord` (un elemento per persona/frame, non raggruppata per
    frame) -- e' compito del chiamante riaggregarla per `frame_index` se
    serve ricostruire dei `SegFrameResult`."""
    data = np.load(path, allow_pickle=True)
    n = len(data["frame_index"])
    return [
        ChunkRecord(
            frame_index=int(data["frame_index"][i]),
            track_id=int(data["track_id"][i]),
            bbox=data["box"][i],
            polygon=data["polygon"][i],
            conf=float(data["conf"][i]),
        )
        for i in range(n)
    ]
