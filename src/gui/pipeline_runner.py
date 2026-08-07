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
from segmentation.seg_estimation import SegTracker
from segmentation.seg_reid import SegReIdentifier
from pose.mediapipe_pose import MediaPipePoseByTrack
from pose.reid import ReIdentifier
from pose.features import compute_joint_angles
from pose.identity_manager import SessionMode
from pose.appearance_embedding import OSNetEmbedder
from common.viz import draw_skeleton, draw_face_signals, draw_hand, draw_person_label, get_track_color
import cv2

# Modelli/backend riconosciuti per la scelta INDIPENDENTE di segmentazione e
# pose (sezione "Segmentation"/"Pose" della GUI, vedi webui/app.js): non
# piu' una manciata di pipeline predefinite ("yolo"/"sam31"/"sam2" come UNICA
# scelta che decideva sia sagoma che scheletro), ma due assi separati che si
# possono combinare liberamente -- vedi `iter_pipeline_frames()` sotto per
# quali combinazioni sono davvero cablate e perche'.
POSE_BACKEND_KEYS = {"yolo", "mediapipe"}

# Modello YOLO usato SOLO come proponente di box+tracking quando si sceglie
# "Pose: MediaPipe" SENZA segmentazione attiva (task "Pose estimation" da
# sola, vedi `_iter_pose_mediapipe`): MediaPipe Tasks PoseLandmarker non ha
# un tracker multi-persona proprio (vedi pose/mediapipe_pose.py), quindi
# serve comunque un detector+tracker a monte per sapere DOVE applicarlo e
# con quale identita' -- esattamente come gia' avviene per "Segmentation +
# MediaPipe pose" (segmentation_demo.py), solo che qui non e' scelto
# dall'utente ne' mostrato come un vero e proprio modello di segmentazione:
# e' un dettaglio implementativo interno, non un output (nessuna maschera
# viene disegnata/salvata da questo percorso). "n" (nano): il piu' veloce,
# qui basta un box ragionevole, non una segmentazione accurata.
_MEDIAPIPE_BOX_PROPOSER_MODEL = "yolo26n-seg.pt"


def _build_embedder(use_appearance_embedding: bool, embedding_device: str) -> OSNetEmbedder | None:
    """Costruita UNA volta per generatore (non ad ogni frame) -- se
    `use_appearance_embedding` e' False, `None` senza toccare torch/
    torchreid affatto (nessun costo, nessun import). Se True e la
    dipendenza manca, `OSNetEmbedder()` solleva `ImportError` subito, qui
    non intercettato: risale al chiamante (webui/api.py gia' lo trasforma
    in un {"ok": False, "error": ...} per la status pill, vedi
    `Api._push_error`)."""
    return OSNetEmbedder(device=embedding_device) if use_appearance_embedding else None


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
    pose_backend: str = "yolo",  # "yolo" (default, full-frame + ByteTrack) | "mediapipe"
    pose_model: str = "yolo26n-pose.pt",
    with_hands: bool = False, hand_model: str = "hand_landmarker.task",
    with_eyes: bool = False, with_mouth: bool = False,
    with_eyebrows: bool = False, with_head_movement: bool = False,
    face_model: str = "face_landmarker.task",
    with_reid: bool = False, blur_faces: bool = False,
    window_seconds: float = 3.0,
    # -- segmentazione (usata se mode e' "segmentation" o "both") --
    seg_model: str = "yolo26s-seg.pt", with_seg_reid: bool = False,
    seg_backend: str = "yolo", sam_chunk_size: int = 600, sam_overlap: int = 50,
    sam_chunk_store_dir: str | None = None,
    sam_redetect_every: int | None = None, sam_text_prompt: str | None = None,
    # -- pose dentro la maschera/box, MediaPipe (usata se `pose_backend` e'
    # "mediapipe" -- vedi pose/mediapipe_pose.py e _iter_segmentation /
    # _iter_pose_mediapipe) --
    with_mediapipe_pose: bool = False,
    pose_landmarker_model: str = "pose_landmarker_lite.task",
    # -- Identity & Re-identification (condiviso, vedi identity_manager.py).
    # `reid_max_lost_seconds` si applica SOLO al percorso basato su keypoint
    # (ReIdentifier, pose_backend="yolo"|"mediapipe" con mode "pose"/"both"):
    # SegReIdentifier (segmentazione) non ha scadenza per design, vedi il suo
    # docstring -- qui viene semplicemente ignorato in quel caso. --
    session_mode: SessionMode = SessionMode.MULTIPLE, flag_uncertain: bool = True,
    reid_max_lost_seconds: float = 180.0,
    # `use_appearance_embedding` (nuovo, opzionale): aggiunge un vero
    # embedding di aspetto (OSNet) come segnale supplementare per
    # ReIdentifier/SegReIdentifier -- vedi pose/appearance_embedding.py per
    # il "perche'" (richiesta esplicita: id stabili, persone "in memoria"
    # per il rientro, ispirato a OSNet+l'idea EMA di StrongSORT). Richiede
    # 'torch'+'torchreid' (dipendenza pesante, opzionale, NON in
    # requirements.txt di default): se mancante, la costruzione
    # dell'embedder solleva un ImportError chiaro, propagato fino alla GUI
    # (stesso trattamento di SAM 3.1/SAM2 quando manca CUDA/il pacchetto).
    use_appearance_embedding: bool = False, embedding_device: str = "cpu",
    # -- condivisi --
    max_people: int | None = None, conf_threshold: float = 0.1,
    tracker_config: str = "bytetrack.yaml",
) -> Iterator[RunnerFrame]:
    """Dispatcher: sceglie la (o le) pipeline in base a `mode` (Task:
    Segmentation/Pose/Both) e a `pose_backend` (Pose: YOLO26 Pose/MediaPipe,
    indipendente dal modello di segmentazione scelto -- vedi il docstring
    del modulo per il "perche'" architetturale) e restituisce un iteratore
    di `RunnerFrame` uniforme.

    Combinazioni cablate (vedi anche webui/app.js per l'auto-selezione
    dell'input mostrata in UI per ciascuna):
      - "pose" + pose_backend="yolo" (default): `_iter_pose` (YOLO26 Pose
        full-frame + ByteTrack, INVARIATO).
      - "pose" + pose_backend="mediapipe": `_iter_pose_mediapipe` (un
        detector/tracker YOLO interno propone i box, MediaPipe stima la
        posa dentro ciascuno -- nessuna maschera prodotta/mostrata, vedi
        li' per i limiti onesti).
      - "segmentation": `_iter_segmentation`, INVARIATA (`seg_backend`
        sceglie YOLO26 Segment/SAM 3.1/SAM2; `with_mediapipe_pose` --
        equivalente a pose_backend="mediapipe" qui -- applica MediaPipe
        dentro ciascuna maschera tracciata).
      - "both" + pose_backend="mediapipe": delega a `_iter_segmentation(...,
        with_mediapipe_pose=True)` con `mode` sovrascritto a "both" nel
        `RunnerFrame` restituito -- e' esattamente lo stesso percorso di
        "Segmentation + MediaPipe pose", perche' MediaPipe qui e' comunque
        guidato dalla maschera/box gia' tracciata dalla segmentazione: non
        ha senso duplicare la logica per un nome di modalita' diverso.
      - "both" + pose_backend="yolo" (default): `_iter_both`, INVARIATA
        (le due pipeline girano IN PARALLELO con tracker/reid indipendenti
        -- limite noto, NESSUNA identita' condivisa tra pose e
        segmentazione, vedi il suo docstring)."""
    if mode == "pose":
        if pose_backend == "mediapipe":
            yield from _iter_pose_mediapipe(
                source=source, fps=fps, device=device,
                box_proposer_model=_MEDIAPIPE_BOX_PROPOSER_MODEL,
                pose_landmarker_model=pose_landmarker_model,
                with_reid=with_reid, max_people=max_people,
                session_mode=session_mode, flag_uncertain=flag_uncertain,
                reid_max_lost_seconds=reid_max_lost_seconds,
                conf_threshold=conf_threshold, tracker_config=tracker_config,
                use_appearance_embedding=use_appearance_embedding,
                embedding_device=embedding_device,
            )
        else:
            yield from _iter_pose(
                source=source, fps=fps, device=device, pose_model=pose_model,
                with_hands=with_hands, hand_model=hand_model,
                with_eyes=with_eyes, with_mouth=with_mouth,
                with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
                face_model=face_model,
                with_reid=with_reid, max_people=max_people, blur_faces=blur_faces,
                window_seconds=window_seconds, conf_threshold=conf_threshold,
                tracker_config=tracker_config, session_mode=session_mode,
                flag_uncertain=flag_uncertain, reid_max_lost_seconds=reid_max_lost_seconds,
                use_appearance_embedding=use_appearance_embedding,
                embedding_device=embedding_device,
            )
    elif mode == "segmentation":
        yield from _iter_segmentation(
            source=source, fps=fps, device=device, seg_model=seg_model,
            max_people=max_people, with_seg_reid=with_seg_reid,
            with_mediapipe_pose=with_mediapipe_pose or pose_backend == "mediapipe",
            pose_landmarker_model=pose_landmarker_model,
            conf_threshold=conf_threshold, tracker_config=tracker_config,
            seg_backend=seg_backend, sam_chunk_size=sam_chunk_size,
            sam_overlap=sam_overlap, sam_chunk_store_dir=sam_chunk_store_dir,
            sam_redetect_every=sam_redetect_every, sam_text_prompt=sam_text_prompt,
            session_mode=session_mode, flag_uncertain=flag_uncertain,
            use_appearance_embedding=use_appearance_embedding,
            embedding_device=embedding_device,
        )
    elif mode == "both":
        if pose_backend == "mediapipe":
            # Stesso percorso di "Segmentation + MediaPipe pose"
            # (`iter_segmentation_frames`, INVARIATA): qui MediaPipe e'
            # comunque guidato dal box/maschera gia' tracciata dalla
            # segmentazione, non ha senso duplicare la logica solo perche'
            # la modalita' si chiama "both" -- l'unica differenza e' il
            # `mode` etichettato nel `RunnerFrame` restituito.
            if with_seg_reid and max_people is None:
                raise ValueError("with_seg_reid richiede max_people (il tetto rigido ha senso "
                                  "solo con un numero di persone noto)")
            embedder = _build_embedder(use_appearance_embedding, embedding_device)
            seg_reidentifier = (
                SegReIdentifier(max_people=max_people, session_mode=session_mode,
                                 flag_uncertain=flag_uncertain, embedder=embedder)
                if with_seg_reid else None
            )
            mediapipe_pose_estimator = MediaPipePoseByTrack(model_path=pose_landmarker_model)
            for vis, rows, now, frame_index, raw_ids in iter_segmentation_frames(
                source=source, fps=fps, model_name=seg_model, device=device,
                conf_threshold=conf_threshold, tracker_config=tracker_config,
                max_people=max_people, seg_reidentifier=seg_reidentifier,
                mediapipe_pose_estimator=mediapipe_pose_estimator,
                backend=seg_backend, sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
                sam_chunk_store_dir=sam_chunk_store_dir, sam_redetect_every=sam_redetect_every,
                sam_text_prompt=sam_text_prompt,
            ):
                yield RunnerFrame(frame=vis, rows=rows, now=now, mode="both", people_count=len(rows))
        else:
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
                seg_backend=seg_backend, sam_chunk_size=sam_chunk_size,
                sam_overlap=sam_overlap, sam_chunk_store_dir=sam_chunk_store_dir,
                sam_redetect_every=sam_redetect_every, sam_text_prompt=sam_text_prompt,
                session_mode=session_mode, flag_uncertain=flag_uncertain,
                use_appearance_embedding=use_appearance_embedding,
                embedding_device=embedding_device,
            )
    else:
        raise ValueError(f"mode sconosciuto: {mode!r} (atteso 'pose'|'segmentation'|'both')")


def _iter_pose(*, source, fps, device, pose_model, with_hands, hand_model,
               with_eyes, with_mouth, with_eyebrows, with_head_movement, face_model,
               with_reid, max_people, blur_faces,
               window_seconds, conf_threshold, tracker_config,
               session_mode: SessionMode = SessionMode.MULTIPLE,
               flag_uncertain: bool = True,
               reid_max_lost_seconds: float = 180.0,
               use_appearance_embedding: bool = False,
               embedding_device: str = "cpu") -> Iterator[RunnerFrame]:
    for frame, rows, now, _smoothed_fps, people, _gaze, _hands in iter_live_frames(
        source=source, fps=fps, model_name=pose_model, device=device,
        window_seconds=window_seconds, blur_faces=blur_faces,
        with_eyes=with_eyes, with_mouth=with_mouth,
        with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
        face_model=face_model,
        with_hands=with_hands, hand_model=hand_model,
        with_reid=with_reid, conf_threshold=conf_threshold,
        tracker_config=tracker_config, max_people=max_people,
        session_mode=session_mode, flag_uncertain=flag_uncertain,
        reid_max_lost_seconds=reid_max_lost_seconds,
        use_appearance_embedding=use_appearance_embedding,
        embedding_device=embedding_device,
    ):
        # `people` (5o elemento della tupla, prima scartato) e' la lista di
        # tracce pose attive in questo frame -- affidabile anche quando
        # `rows` e' ancora vuoto (finestra scorrevole non ancora piena).
        yield RunnerFrame(frame=frame, rows=rows, now=now, mode="pose",
                           people_count=len(people))


def _iter_pose_mediapipe(*, source, fps, device, box_proposer_model, pose_landmarker_model,
                          with_reid, max_people, conf_threshold, tracker_config,
                          session_mode: SessionMode = SessionMode.MULTIPLE,
                          flag_uncertain: bool = True,
                          reid_max_lost_seconds: float = 180.0,
                          use_appearance_embedding: bool = False,
                          embedding_device: str = "cpu") -> Iterator[RunnerFrame]:
    """Task "Pose estimation" con `pose_backend="mediapipe"`, SENZA
    segmentazione attiva: MediaPipe Tasks PoseLandmarker non ha un tracker
    multi-persona proprio (vedi pose/mediapipe_pose.py), quindi qui un
    tracker YOLO leggero (`box_proposer_model`, solo box+track, la maschera
    viene ignorata) fa da proponente di regione -- stesso schema di
    "Segmentation + MediaPipe pose" (segmentation_demo.py), solo che qui il
    box-proposer e' un dettaglio interno: nessuna maschera viene
    disegnata/salvata, e la UI non lo presenta come una scelta di
    segmentazione (vedi `iter_pipeline_frames` per il "perche'").

    Limiti onesti (ereditati da pose/mediapipe_pose.py): solo angoli
    articolari istantanei nel CSV (`pose_*`), nessuna feature a finestra
    scorrevole (energia di movimento, ripetitivita', gaze, mani) -- quelle
    restano disponibili solo con `pose_backend="yolo"` (vedi `_iter_pose`).
    """
    tracker = SegTracker(model_name=box_proposer_model, device=device,
                          conf_threshold=conf_threshold, tracker=tracker_config,
                          max_people=max_people)
    mediapipe_pose_estimator = MediaPipePoseByTrack(model_path=pose_landmarker_model)
    embedder = _build_embedder(use_appearance_embedding, embedding_device) if with_reid else None
    reidentifier = ReIdentifier(
        max_people=max_people, session_mode=session_mode, flag_uncertain=flag_uncertain,
        max_lost_seconds=reid_max_lost_seconds, embedder=embedder,
    ) if with_reid else None

    for frame_result in tracker.run(source=source):
        now = frame_result.frame_index / fps
        vis = frame_result.frame.copy()
        people_kxy: list[tuple[int, np.ndarray, np.ndarray]] = []
        for track_id, bbox, _poly, _conf in frame_result.people:
            kxy, kconf = mediapipe_pose_estimator.estimate(
                track_id, frame_result.frame, bbox, timestamp_ms=int(now * 1000))
            people_kxy.append((track_id, kxy, kconf))

        if reidentifier is not None:
            people_kxy = reidentifier.resolve(people_kxy, now, frame=frame_result.frame)

        rows_this_frame = []
        for track_id, kxy, kconf in people_kxy:
            color = get_track_color(track_id)
            draw_skeleton(vis, kxy, kconf, color=color)
            draw_person_label(vis, head_center(kxy), track_id, color)
            angles = compute_joint_angles(kxy)
            rows_this_frame.append({
                "frame": frame_result.frame_index, "time_s": now, "track_id": track_id,
                **{f"pose_{k}": v for k, v in angles.items()},
            })

        yield RunnerFrame(frame=vis, rows=rows_this_frame, now=now, mode="pose",
                           people_count=len(people_kxy))


def _iter_segmentation(*, source, fps, device, seg_model, max_people,
                        with_seg_reid, with_mediapipe_pose=False,
                        pose_landmarker_model="pose_landmarker_lite.task",
                        conf_threshold, tracker_config,
                        seg_backend="yolo", sam_chunk_size=600, sam_overlap=50,
                        sam_chunk_store_dir=None, sam_redetect_every=None,
                        sam_text_prompt=None,
                        session_mode: SessionMode = SessionMode.MULTIPLE,
                        flag_uncertain: bool = True,
                        use_appearance_embedding: bool = False,
                        embedding_device: str = "cpu") -> Iterator[RunnerFrame]:
    if with_seg_reid and max_people is None:
        raise ValueError("with_seg_reid richiede max_people (il tetto rigido ha senso "
                          "solo con un numero di persone noto)")
    embedder = _build_embedder(use_appearance_embedding, embedding_device) if with_seg_reid else None
    seg_reidentifier = (
        SegReIdentifier(max_people=max_people, session_mode=session_mode,
                         flag_uncertain=flag_uncertain, embedder=embedder)
        if with_seg_reid else None
    )
    mediapipe_pose_estimator = (
        MediaPipePoseByTrack(model_path=pose_landmarker_model) if with_mediapipe_pose else None
    )
    for vis, rows, now, _frame_index, _raw_ids in iter_segmentation_frames(
        source=source, fps=fps, model_name=seg_model, device=device,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
        max_people=max_people, seg_reidentifier=seg_reidentifier,
        mediapipe_pose_estimator=mediapipe_pose_estimator,
        backend=seg_backend, sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
        sam_chunk_store_dir=sam_chunk_store_dir, sam_redetect_every=sam_redetect_every,
        sam_text_prompt=sam_text_prompt,
    ):
        # In segmentazione non c'e' finestra scorrevole: una riga per persona
        # tracciata per frame, quindi len(rows) e' gia' il conteggio esatto.
        yield RunnerFrame(frame=vis, rows=rows, now=now, mode="segmentation",
                           people_count=len(rows))


def _iter_both(*, source, fps, device, pose_model, with_hands, hand_model,
               with_eyes, with_mouth, with_eyebrows, with_head_movement, face_model,
               with_reid, blur_faces, window_seconds,
               seg_model, with_seg_reid, max_people, conf_threshold,
               tracker_config, seg_backend="yolo", sam_chunk_size=600,
               sam_overlap=50, sam_chunk_store_dir=None,
               sam_redetect_every=None, sam_text_prompt=None,
               session_mode: SessionMode = SessionMode.MULTIPLE,
               flag_uncertain: bool = True,
               reid_max_lost_seconds: float = 180.0,
               use_appearance_embedding: bool = False,
               embedding_device: str = "cpu") -> Iterator[RunnerFrame]:
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
    # Due embedder distinti (non condiviso tra le due pipeline indipendenti,
    # vedi il docstring del modulo su "nessuna identita' condivisa tra pose
    # e segmentazione" -- stesso limite, qui semplicemente esteso
    # all'embedding): ciascuno carica il proprio modello OSNet in memoria,
    # costo raddoppiato in modalita' "both" con l'embedding attivo.
    seg_reidentifier = (
        SegReIdentifier(max_people=max_people, session_mode=session_mode,
                         flag_uncertain=flag_uncertain,
                         embedder=_build_embedder(use_appearance_embedding, embedding_device))
        if with_seg_reid else None
    )

    pose_gen = iter_live_frames(
        source=source, fps=fps, model_name=pose_model, device=device,
        window_seconds=window_seconds, blur_faces=blur_faces,
        with_eyes=with_eyes, with_mouth=with_mouth,
        with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
        face_model=face_model,
        with_hands=with_hands, hand_model=hand_model,
        with_reid=with_reid, conf_threshold=conf_threshold,
        tracker_config=tracker_config, max_people=max_people,
        session_mode=session_mode, flag_uncertain=flag_uncertain,
        reid_max_lost_seconds=reid_max_lost_seconds,
        use_appearance_embedding=use_appearance_embedding,
        embedding_device=embedding_device,
    )
    seg_gen = iter_segmentation_frames(
        source=source, fps=fps, model_name=seg_model, device=device,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
        max_people=max_people, seg_reidentifier=seg_reidentifier,
        backend=seg_backend, sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
        sam_chunk_store_dir=sam_chunk_store_dir, sam_redetect_every=sam_redetect_every,
        sam_text_prompt=sam_text_prompt,
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
