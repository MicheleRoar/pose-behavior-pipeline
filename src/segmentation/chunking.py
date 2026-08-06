"""
chunking.py
============
Logica di suddivisione in finestre (chunk) sovrapposte e di riconciliazione
degli ID tra un chunk e il successivo -- indipendente da SAM/SAM2, usata
da `sam_backend.py`. Nessuna dipendenza pesante (solo numpy/cv2, gia'
richiesti dal resto del progetto): testabile con maschere sintetiche, senza
avere ne' una GPU ne' i pesi SAM installati (vedi demo/chunking_check.py).

Perche' il chunking e' necessario (non solo un'ottimizzazione)
------------------------------------------------------------------
L'API video di SAM 3.1 e di SAM2 e' stateful: `init_state(video)`
carica in memoria i pixel di TUTTI i frame passati, prima ancora di
cominciare a propagare le maschere. Su un video di svariati minuti questo
non e' semplicemente lento, e' un problema di memoria (VRAM/RAM) -- quindi
non si passa mai il video intero a `init_state()`, si passa una finestra di
`chunk_size` frame alla volta. Il prezzo da pagare e' che ogni chunk parte
"senza memoria" del chunk precedente: gli ID che SAM assegna dentro un
chunk sono locali a quel chunk, non c'e' garanzia che l'id 1 del chunk 2 sia
la stessa persona dell'id 1 del chunk 1. Da qui la sovrapposizione
(`overlap` frame in comune tra un chunk e il successivo) e la
riconciliazione geometrica sotto: si confrontano le maschere prodotte dai
due chunk sugli STESSI frame (quelli in comune) e si ricostruisce un id
globale stabile.

Limite noto di questa prima versione: la riconciliazione usa solo IoU
geometrico sul frame di ancoraggio (l'ultimo frame di overlap). Funziona
bene se le persone non si sono scambiate di posizione dentro la finestra di
overlap; in scene affollate con occlusioni proprio a cavallo del confine
chunk puo' sbagliare -- estensione naturale, se servisse: aggiungere un
punteggio di somiglianza d'aspetto (colore/texture, come gia' fa
`segmentation/seg_reid.py` per ByteTrack) accanto all'IoU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np


def iter_chunk_ranges(total_frames: int, chunk_size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Genera coppie (start, end) [end escluso] che coprono `[0, total_frames)`
    con una sovrapposizione di `overlap` frame tra un chunk e il successivo.
    L'ultimo chunk puo' essere piu' corto di `chunk_size` (non si estende
    oltre `total_frames`). Solleva `ValueError` se `chunk_size <= overlap`
    (altrimenti non si avanzerebbe mai, loop infinito)."""
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) deve essere maggiore di overlap ({overlap})")
    if total_frames <= 0:
        return
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        yield start, end
        if end >= total_frames:
            break
        start = end - overlap


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray, frame_shape: tuple[int, int]) -> float:
    """IoU (intersection over union) tra due poligoni maschera, calcolato
    rasterizzandoli su una griglia della dimensione del frame (`frame_shape`
    = (height, width)) e confrontando le maschere binarie risultanti --
    corretto anche per poligoni non convessi, a differenza di un IoU
    calcolato sui soli bounding box. Ritorna 0.0 se uno dei due poligoni ha
    meno di 3 punti (degenere/assente)."""
    if poly_a.shape[0] < 3 or poly_b.shape[0] < 3:
        return 0.0
    mask_a = np.zeros(frame_shape, dtype=np.uint8)
    mask_b = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillPoly(mask_a, [poly_a.astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [poly_b.astype(np.int32)], 1)
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union > 0 else 0.0


def reconcile_ids(
    prev_polys_at_anchor: dict[int, np.ndarray],
    new_polys_at_anchor: dict[int, np.ndarray],
    frame_shape: tuple[int, int],
    iou_threshold: float = 0.3,
) -> dict[int, int]:
    """Confronta le maschere del chunk precedente (`prev_polys_at_anchor`,
    chiavi = id GLOBALI gia' assegnati) e quelle del chunk nuovo
    (`new_polys_at_anchor`, chiavi = id LOCALI assegnati da SAM dentro
    questo chunk) sullo stesso frame di ancoraggio (un frame che entrambi i
    chunk hanno prodotto, dentro la finestra di overlap).

    Ritorna un dict `{id_locale: id_globale}` solo per gli id locali che
    hanno trovato una corrispondenza sopra `iou_threshold`. Un id locale
    assente dal dict e' una persona "nuova" per il chiamante (entrata nel
    campo durante il chunk, o corrispondenza troppo incerta per essere
    sicuri) -- il chiamante gli assegnera' un id globale mai usato prima
    (vedi `GlobalIdAllocator`).

    Abbinamento greedy per IoU decrescente: ogni id (vecchio o nuovo) viene
    usato al massimo una volta, cosi' due persone vicine non vengono
    entrambe abbinate allo stesso id."""
    candidates: list[tuple[float, int, int]] = []
    for old_id, old_poly in prev_polys_at_anchor.items():
        for new_id, new_poly in new_polys_at_anchor.items():
            iou = polygon_iou(old_poly, new_poly, frame_shape)
            if iou > 0.0:
                candidates.append((iou, old_id, new_id))
    candidates.sort(key=lambda c: c[0], reverse=True)

    mapping: dict[int, int] = {}
    used_old: set[int] = set()
    used_new: set[int] = set()
    for iou, old_id, new_id in candidates:
        if iou < iou_threshold:
            break  # ordinato per iou decrescente: tutto il resto e' sotto soglia
        if old_id in used_old or new_id in used_new:
            continue
        mapping[new_id] = old_id
        used_old.add(old_id)
        used_new.add(new_id)
    return mapping


@dataclass
class GlobalIdAllocator:
    """Distributore di id globali progressivi, condiviso da tutti i chunk di
    una stessa sessione. `next_id()` restituisce sempre un intero mai
    restituito prima -- usato per gli id locali che `reconcile_ids()` non e'
    riuscita ad abbinare a nessun id gia' noto."""
    _next: int = field(default=1)

    def next_id(self) -> int:
        value = self._next
        self._next += 1
        return value
