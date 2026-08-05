"""
pipeline_runner.py
====================
Punto di ingresso unico usato dalla GUI (app.py / video_player.py) per
ottenere frame con overlay gia' pronti da mostrare, qualunque sia la
pipeline scelta (pose, segmentazione, o entrambe). Non duplica NESSUNA
logica per-frame: chiama direttamente `iter_live_frames()` (pose, da
live_demo.py) e/o `iter_segmentation_frames()` (segmentazione, da
segmentation_demo.py) -- le stesse funzioni gia' usate dai rispettivi CLI,
cosi' GUI e CLI restano garantiti identici nel comportamento (vedi i
docstring di quelle due funzioni per il perche' di questa scelta).

Modalita' "both" (v1): le due pipeline girano in parallelo, ciascuna con la
propria istanza di tracker/reid -- NESSUNA identita' condivisa tra pose e
segmentazione per ora (limitazione nota, da affrontare in futuro). I due
overlay vengono mostrati AFFIANCATI (side-by-side: pose a sinistra,
segmentazione a destra) invece che sovrapposti sullo stesso frame: fondere
pixel-per-pixel due tracker indipendenti che decodificano la sorgente
separatamente richiederebbe assumere che le due decodifiche restino
byte-identiche frame per frame, un'assunzione fragile che preferiamo non
fare silenziosamente.

Nota sul timestamp `now` restituito: la pipeline pose usa `time.time()`
(orologio di sistema, coerente con live_demo.py pensato anche per sorgenti
live), quella di segmentazione usa `frame_index / fps` (timeline del
video, coerente con segmentation_demo.py pensato per video registrati).
Sono due basi diverse, preesistenti a questo modulo: qui non le
uniformiamo silenziosamente, ci limitiamo a etichettare ogni riga con la
pipeline di provenienza (colonna "pipeline" in modalita' "both") cosi' chi
analizza il CSV sa quale colonna guardare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from live_demo import iter_live_frames
from segmentation_demo import iter_segmentation_frames
from segmentation.seg_reid import SegReIdentifier


@dataclass
class RunnerFrame:
    """Un frame gia' pronto per la GUI: overlay disegnato, righe dati per il
    CSV, timestamp, e quale modalita' lo ha prodotto."""
    frame: np.ndarray
    rows: list[dict]
    now: float
    mode: str  # "pose" | "segmentation" | "both"


def iter_pipeline_frames(
    *, mode: str, source, fps: float, device: str = "mps",
    # -- pose (usata se mode e' "pose" o "both") --
    pose_model: str = "yolo26n-pose.pt",
    with_hands: bool = False, hand_model: str = "hand_landmarker.task",
    with_face: bool = False, face_model: str = "face_landmarker.task",
    with_reid: bool = False, blur_faces: bool = False,
    window_seconds: float = 3.0,
    # -- segmentazione (usata se mode e' "segmentation" o "both") --
    seg_model: str = "yolo26s-seg.pt", with_seg_reid: bool = False,
    # -- condivisi --
    max_people: int | None = None, conf_threshold: float = 0.1,
    tracker_config: str = "bytetrack.yaml",
) -> Iterator[RunnerFrame]:
    """Dispatcher: sceglie la (o le) pipeline in base a `mode` e restituisce
    un iteratore di `RunnerFrame` uniforme, qualunque sia la combinazione di
    feature selezionata nella GUI (mani/viso solo disponibili in "pose" o
    "both", vedi app.py per l'abilitazione condizionale dei checkbox)."""
    if mode == "pose":
        yield from _iter_pose(
            source=source, fps=fps, device=device, pose_model=pose_model,
            with_hands=with_hands, hand_model=hand_model,
            with_face=with_face, face_model=face_model,
            with_reid=with_reid, max_people=max_people, blur_faces=blur_faces,
            window_seconds=window_seconds, conf_threshold=conf_threshold,
            tracker_config=tracker_config,
        )
    elif mode == "segmentation":
        yield from _iter_segmentation(
            source=source, fps=fps, device=device, seg_model=seg_model,
            max_people=max_people, with_seg_reid=with_seg_reid,
            conf_threshold=conf_threshold, tracker_config=tracker_config,
        )
    elif mode == "both":
        yield from _iter_both(
            source=source, fps=fps, device=device, pose_model=pose_model,
            with_hands=with_hands, hand_model=hand_model,
            with_face=with_face, face_model=face_model,
            with_reid=with_reid, blur_faces=blur_faces,
            window_seconds=window_seconds, seg_model=seg_model,
            with_seg_reid=with_seg_reid, max_people=max_people,
            conf_threshold=conf_threshold, tracker_config=tracker_config,
        )
    else:
        raise ValueError(f"mode sconosciuto: {mode!r} (atteso 'pose'|'segmentation'|'both')")


def _iter_pose(*, source, fps, device, pose_model, with_hands, hand_model,
               with_face, face_model, with_reid, max_people, blur_faces,
               window_seconds, conf_threshold, tracker_config) -> Iterator[RunnerFrame]:
    for frame, rows, now, _smoothed_fps in iter_live_frames(
        source=source, fps=fps, model_name=pose_model, device=device,
        window_seconds=window_seconds, blur_faces=blur_faces,
        with_gaze=with_face, face_model=face_model,
        with_hands=with_hands, hand_model=hand_model,
        with_reid=with_reid, conf_threshold=conf_threshold,
        tracker_config=tracker_config, max_people=max_people,
    ):
        yield RunnerFrame(frame=frame, rows=rows, now=now, mode="pose")


def _iter_segmentation(*, source, fps, device, seg_model, max_people,
                        with_seg_reid, conf_threshold, tracker_config) -> Iterator[RunnerFrame]:
    if with_seg_reid and max_people is None:
        raise ValueError("with_seg_reid richiede max_people (il tetto rigido ha senso "
                          "solo con un numero di persone noto)")
    seg_reidentifier = SegReIdentifier(max_people=max_people) if with_seg_reid else None
    for vis, rows, now, _frame_index, _raw_ids in iter_segmentation_frames(
        source=source, fps=fps, model_name=seg_model, device=device,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
        max_people=max_people, seg_reidentifier=seg_reidentifier,
    ):
        yield RunnerFrame(frame=vis, rows=rows, now=now, mode="segmentation")


def _iter_both(*, source, fps, device, pose_model, with_hands, hand_model,
               with_face, face_model, with_reid, blur_faces, window_seconds,
               seg_model, with_seg_reid, max_people, conf_threshold,
               tracker_config) -> Iterator[RunnerFrame]:
    pose_iter = _iter_pose(
        source=source, fps=fps, device=device, pose_model=pose_model,
        with_hands=with_hands, hand_model=hand_model,
        with_face=with_face, face_model=face_model,
        with_reid=with_reid, max_people=max_people, blur_faces=blur_faces,
        window_seconds=window_seconds, conf_threshold=conf_threshold,
        tracker_config=tracker_config,
    )
    seg_iter = _iter_segmentation(
        source=source, fps=fps, device=device, seg_model=seg_model,
        max_people=max_people, with_seg_reid=with_seg_reid,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
    )
    # zip() (non zip_longest): se le due sorgenti indipendenti finissero con
    # un numero di frame diverso per qualche motivo, ci si ferma alla piu'
    # corta piuttosto che restituire un frame "both" con meta' mancante.
    for pose_result, seg_result in zip(pose_iter, seg_iter):
        combined = _hstack_same_height(pose_result.frame, seg_result.frame)
        rows = (
            [{**r, "pipeline": "pose"} for r in pose_result.rows]
            + [{**r, "pipeline": "segmentation"} for r in seg_result.rows]
        )
        yield RunnerFrame(frame=combined, rows=rows, now=pose_result.now, mode="both")


def _hstack_same_height(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Affianca due frame BGR ridimensionando il secondo alla stessa altezza
    del primo (per sicurezza, nel caso i due tracker restituissero frame di
    dimensioni leggermente diverse), con un separatore verticale sottile."""
    h = left.shape[0]
    if right.shape[0] != h:
        scale = h / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), h))
    separator = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.hstack([left, separator, right])
