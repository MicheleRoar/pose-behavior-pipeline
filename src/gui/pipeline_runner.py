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

Modalita' "both": le due pipeline girano in parallelo, ciascuna con la
propria istanza di tracker/reid -- NESSUNA identita' condivisa tra pose e
segmentazione per ora (limitazione nota, da affrontare in futuro: le
etichette "ID N" disegnate dalla segmentazione e i track_id della pose non
si corrispondono). Lo scheletro pose viene disegnato SOPRA il frame gia'
annotato dalla segmentazione (non piu' affiancati in due pannelli), per
richiesta esplicita e perche' qui non c'e' alcun vincolo di privacy/
copyright da preservare nel farlo (video pubblici, non clinici). Questo
richiede che le due sorgenti indipendenti (due `cv2.VideoCapture` separati
sullo stesso file) restino sincronizzate frame-per-frame -- assunzione
ragionevole per un file registrato decodificato in sequenza da entrambe
senza salti, ma non garantita se una delle due pipeline scartasse
internamente dei frame (nessuna lo fa oggi).

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

import numpy as np

from live_demo import iter_live_frames, head_center
from segmentation_demo import iter_segmentation_frames
from segmentation.seg_reid import SegReIdentifier
from pose.mediapipe_pose import MediaPipeCropPoseEstimator
from common.viz import draw_skeleton, draw_face_signals, draw_hand, get_track_color
import cv2


@dataclass
class RunnerFrame:
    """Un frame gia' pronto per la GUI: overlay disegnato, righe dati per il
    CSV, timestamp, e quale modalita' lo ha prodotto."""
    frame: np.ndarray
    rows: list[dict]
    now: float
    mode: str  # "pose" | "segmentation" | "both"
    people_count: int = 0  # tracce attive in questo frame -- NON deducibile
    # in modo affidabile da len(rows): in modalita' "pose" le righe vengono
    # aggiunte solo quando la finestra scorrevole delle feature si riempie
    # (vedi live_demo.py), quindi len(rows) sarebbe 0 per i primi secondi pur
    # con persone gia' tracciate. Qui usiamo invece il conteggio delle tracce
    # effettivamente attive nel frame corrente.


def iter_pipeline_frames(
    *, mode: str, source, fps: float, device: str = "mps",
    # -- pose (usata se mode e' "pose" o "both") --
    pose_model: str = "yolo26n-pose.pt",
    with_hands: bool = False, hand_model: str = "hand_landmarker.task",
    with_eyes: bool = False, with_mouth: bool = False,
    with_eyebrows: bool = False, with_head_movement: bool = False,
    face_model: str = "face_landmarker.task",
    with_reid: bool = False, blur_faces: bool = False,
    window_seconds: float = 3.0,
    # -- segmentazione (usata se mode e' "segmentation" o "both") --
    seg_model: str = "yolo26s-seg.pt", with_seg_reid: bool = False,
    # -- pose dentro la maschera, MediaPipe (usata solo se mode e'
    # "segmentation" -- vedi pose/mediapipe_pose.py e _iter_segmentation) --
    with_mediapipe_pose: bool = False,
    pose_landmarker_model: str = "pose_landmarker_lite.task",
    # -- condivisi --
    max_people: int | None = None, conf_threshold: float = 0.1,
    tracker_config: str = "bytetrack.yaml",
) -> Iterator[RunnerFrame]:
    """Dispatcher: sceglie la (o le) pipeline in base a `mode` e restituisce
    un iteratore di `RunnerFrame` uniforme, qualunque sia la combinazione di
    feature selezionata nella GUI (mani/occhi/bocca/sopracciglia/movimento
    testa solo disponibili in "pose" o "both"; pose MediaPipe per maschera
    solo in "segmentation" -- vedi app.py per l'abilitazione condizionale
    dei checkbox)."""
    if mode == "pose":
        yield from _iter_pose(
            source=source, fps=fps, device=device, pose_model=pose_model,
            with_hands=with_hands, hand_model=hand_model,
            with_eyes=with_eyes, with_mouth=with_mouth,
            with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
            face_model=face_model,
            with_reid=with_reid, max_people=max_people, blur_faces=blur_faces,
            window_seconds=window_seconds, conf_threshold=conf_threshold,
            tracker_config=tracker_config,
        )
    elif mode == "segmentation":
        yield from _iter_segmentation(
            source=source, fps=fps, device=device, seg_model=seg_model,
            max_people=max_people, with_seg_reid=with_seg_reid,
            with_mediapipe_pose=with_mediapipe_pose,
            pose_landmarker_model=pose_landmarker_model,
            conf_threshold=conf_threshold, tracker_config=tracker_config,
        )
    elif mode == "both":
        yield from _iter_both(
            source=source, fps=fps, device=device, pose_model=pose_model,
            with_hands=with_hands, hand_model=hand_model,
            with_eyes=with_eyes, with_mouth=with_mouth,
            with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
            face_model=face_model,
            with_reid=with_reid, blur_faces=blur_faces,
            window_seconds=window_seconds, seg_model=seg_model,
            with_seg_reid=with_seg_reid, max_people=max_people,
            conf_threshold=conf_threshold, tracker_config=tracker_config,
        )
    else:
        raise ValueError(f"mode sconosciuto: {mode!r} (atteso 'pose'|'segmentation'|'both')")


def _iter_pose(*, source, fps, device, pose_model, with_hands, hand_model,
               with_eyes, with_mouth, with_eyebrows, with_head_movement, face_model,
               with_reid, max_people, blur_faces,
               window_seconds, conf_threshold, tracker_config) -> Iterator[RunnerFrame]:
    for frame, rows, now, _smoothed_fps, people, _gaze, _hands in iter_live_frames(
        source=source, fps=fps, model_name=pose_model, device=device,
        window_seconds=window_seconds, blur_faces=blur_faces,
        with_eyes=with_eyes, with_mouth=with_mouth,
        with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
        face_model=face_model,
        with_hands=with_hands, hand_model=hand_model,
        with_reid=with_reid, conf_threshold=conf_threshold,
        tracker_config=tracker_config, max_people=max_people,
    ):
        # `people` (5o elemento della tupla, prima scartato) e' la lista di
        # tracce pose attive in questo frame -- affidabile anche quando
        # `rows` e' ancora vuoto (finestra scorrevole non ancora piena).
        yield RunnerFrame(frame=frame, rows=rows, now=now, mode="pose",
                           people_count=len(people))


def _iter_segmentation(*, source, fps, device, seg_model, max_people,
                        with_seg_reid, with_mediapipe_pose=False,
                        pose_landmarker_model="pose_landmarker_lite.task",
                        conf_threshold, tracker_config) -> Iterator[RunnerFrame]:
    if with_seg_reid and max_people is None:
        raise ValueError("with_seg_reid richiede max_people (il tetto rigido ha senso "
                          "solo con un numero di persone noto)")
    seg_reidentifier = SegReIdentifier(max_people=max_people) if with_seg_reid else None
    mediapipe_pose_estimator = (
        MediaPipeCropPoseEstimator(model_path=pose_landmarker_model) if with_mediapipe_pose else None
    )
    for vis, rows, now, _frame_index, _raw_ids in iter_segmentation_frames(
        source=source, fps=fps, model_name=seg_model, device=device,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
        max_people=max_people, seg_reidentifier=seg_reidentifier,
        mediapipe_pose_estimator=mediapipe_pose_estimator,
    ):
        # In segmentazione non c'e' finestra scorrevole: una riga per persona
        # tracciata per frame, quindi len(rows) e' gia' il conteggio esatto.
        yield RunnerFrame(frame=vis, rows=rows, now=now, mode="segmentation",
                           people_count=len(rows))


def _iter_both(*, source, fps, device, pose_model, with_hands, hand_model,
               with_eyes, with_mouth, with_eyebrows, with_head_movement, face_model,
               with_reid, blur_faces, window_seconds,
               seg_model, with_seg_reid, max_people, conf_threshold,
               tracker_config) -> Iterator[RunnerFrame]:
    """Fa girare le due pipeline in parallelo e disegna lo scheletro pose
    (+ mani/viso se attivi) DIRETTAMENTE sul frame gia' annotato dalla
    segmentazione, invece di affiancare due pannelli -- vedi il docstring
    del modulo. Non chiama `_iter_pose()`/`_iter_segmentation()` (che
    restituirebbero solo il frame gia' composito, senza i dati grezzi per
    ridisegnare): usa direttamente `iter_live_frames()` per avere
    `people`/`gaze_by_track`/`hands_by_track`, e le stesse funzioni di
    disegno di `common/viz.py` gia' usate da `iter_live_frames()` -- niente
    logica di disegno duplicata, solo un secondo richiamo delle stesse
    funzioni su un frame diverso.

    Etichette/pannello metriche della pose NON vengono ridisegnati qui
    (solo scheletro/mani/viso): l'etichetta "ID N" gia' presente sul frame
    di segmentazione resta l'unico riferimento di identita' visibile,
    evitando due numerazioni diverse sovrapposte per la stessa persona.
    """
    if with_seg_reid and max_people is None:
        raise ValueError("with_seg_reid richiede max_people (il tetto rigido ha senso "
                          "solo con un numero di persone noto)")
    seg_reidentifier = SegReIdentifier(max_people=max_people) if with_seg_reid else None

    pose_gen = iter_live_frames(
        source=source, fps=fps, model_name=pose_model, device=device,
        window_seconds=window_seconds, blur_faces=blur_faces,
        with_eyes=with_eyes, with_mouth=with_mouth,
        with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
        face_model=face_model,
        with_hands=with_hands, hand_model=hand_model,
        with_reid=with_reid, conf_threshold=conf_threshold,
        tracker_config=tracker_config, max_people=max_people,
    )
    seg_gen = iter_segmentation_frames(
        source=source, fps=fps, model_name=seg_model, device=device,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
        max_people=max_people, seg_reidentifier=seg_reidentifier,
    )

    # zip() (non zip_longest): se le due sorgenti indipendenti finissero con
    # un numero di frame diverso per qualche motivo, ci si ferma alla piu'
    # corta piuttosto che restituire un frame "both" con meta' mancante.
    for (pose_frame, pose_rows, pose_now, _pose_fps, people, gaze_by_track, hands_by_track), \
            (seg_vis, seg_rows, _seg_now, _frame_index, _raw_ids) in zip(pose_gen, seg_gen):
        canvas = seg_vis
        if canvas.shape[:2] != pose_frame.shape[:2]:
            # non dovrebbe succedere per due tracker sulla stessa sorgente,
            # ma se capitasse disegnare coordinate pose su un canvas di
            # dimensioni diverse le sposterebbe fuori posto -- meglio
            # saltare l'overlay pose per questo frame che disegnare punti
            # sbagliati silenziosamente.
            yield RunnerFrame(
                frame=canvas,
                rows=[{**r, "pipeline": "segmentation"} for r in seg_rows],
                now=pose_now, mode="both", people_count=len(seg_rows),
            )
            continue

        for track_id, kxy, kconf in people:
            color = get_track_color(track_id)
            draw_skeleton(canvas, kxy, kconf, color=color)
            gaze = gaze_by_track.get(track_id)
            if gaze:
                # draw_face_signals ignora silenziosamente le parti assenti
                # (None): con le sotto-feature del viso indipendenti, "gaze"
                # puo' contenere solo un sottoinsieme di bocca/occhi/
                # sopracciglia.
                draw_face_signals(canvas, gaze.get("mouth_pts"),
                                   gaze.get("left_eye_pts"), gaze.get("right_eye_pts"),
                                   gaze.get("left_eyebrow_pts"), gaze.get("right_eyebrow_pts"))
            if gaze and not np.isnan(gaze.get("yaw", np.nan)):
                hc = head_center(kxy)
                ang = np.radians(gaze["yaw"])
                tip = (int(hc[0] + 60 * np.sin(ang)), int(hc[1] - 60 * np.cos(ang)))
                cv2.arrowedLine(canvas, tuple(hc.astype(int)), tip, (255, 0, 255), 2, tipLength=0.3)
            for info in hands_by_track.get(track_id, {}).values():
                draw_hand(canvas, info["landmarks"])

        rows = (
            [{**r, "pipeline": "pose"} for r in pose_rows]
            + [{**r, "pipeline": "segmentation"} for r in seg_rows]
        )
        # Usiamo il conteggio della segmentazione come riferimento canonico di
        # "tracce attive": e' la pipeline la cui etichetta "ID N" resta
        # visibile in questa modalita' (vedi docstring del modulo).
        yield RunnerFrame(frame=canvas, rows=rows, now=pose_now, mode="both",
                           people_count=len(seg_rows))
