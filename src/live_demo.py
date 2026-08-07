"""
live_demo.py
============
Riconoscimento in tempo reale dei movimenti da sorgente live (pensato per la
Canon EOS R8 collegata al MacBook Pro M1), con overlay dello scheletro e
delle metriche comportamentali calcolate su una finestra scorrevole.

Sorgenti supportate:
- Canon EOS R8 via USB + EOS Webcam Utility: compare come webcam standard,
  quindi `--source 0` (o l'indice corretto se ci sono altre webcam/virtual
  camera attive: su Mac puoi verificare l'ordine in
  Impostazioni di Sistema > Privacy e Sicurezza > Fotocamera, oppure con
  `python -c "import cv2;[print(i, cv2.VideoCapture(i).isOpened()) for i in range(4)]"`).
- Canon EOS R8 con uscita HDMI pulita + capture card (es. Elgato Cam Link):
  la capture card compare a sua volta come webcam standard, stesso discorso.
- Un file video, passando il percorso invece di un intero.

Esempio (solo scheletro):

    python live_demo.py --source 0 --fps 30 --model yolo26n-pose.pt --device mps \\
        --window-seconds 3 --blur-faces --out live_session.csv

Esempio con segnali del viso (occhi/bocca/sopracciglia/movimento testa,
selezionabili indipendentemente) e mani a livello di dita (richiede
mediapipe e i rispettivi modelli, vedi README):

    python live_demo.py --source 0 --fps 30 --device mps \\
        --with-eyes --with-mouth --with-eyebrows --with-head-movement \\
        --with-hands --out live_session.csv

Con più persone nell'inquadratura (es. bambino + caregiver), YOLO+ByteTrack
assegna un track_id distinto a ciascuna e li tratta separatamente per
default (scheletro/postura per tutti). Per limitare i segnali di
volto/mani a una sola persona fissa (es. non calcolare il blink del
caregiver), fai una prima prova senza `--target-track-id` per vedere quali
ID vengono stampati a console ("Nuova persona rilevata: ID N"), poi
rilancia specificando quello scelto:

    python live_demo.py --source 0 --fps 30 --device mps \\
        --with-eyes --with-mouth --with-hands --target-track-id 1 --out live_session.csv

Per interrompere la sessione: premi 'q' con la finestra video attiva (il
tasto viene intercettato solo se la finestra ha il focus — clicca sul
video, non sul terminale), oppure chiudi la finestra dal pulsante, oppure
Ctrl+C dal terminale. In tutti e tre i casi le feature accumulate vengono
comunque salvate in `--out` prima di uscire.

Nota: questo script richiede `ultralytics` + `opencv-python` installati
(vedi requirements.txt) e una sorgente video reale; non è eseguibile
nell'ambiente sandbox usato per sviluppare il resto della pipeline. La
logica di rendering (scheletro + overlay metriche) è invece verificata
separatamente in `demo/live_render_check.py` con dati sintetici.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd

from pose.features import (
    compute_joint_angles, repetitive_motion_score,
    vertical_excursion, activity_ratio, self_touch_score, torso_length,
)
from pose.pose_estimation import PoseTracker
from pose.anonymize import blur_face
from common.device import detect_default_device
from common.viz import (
    draw_skeleton, draw_text_block, draw_fps, draw_hand, draw_face_signals,
    get_track_color, draw_person_label, text_block_size,
)
from pose.keypoints import KP
from pose.reid import ReIdentifier
from pose.identity_manager import SessionMode
from pose.appearance_embedding import OSNetEmbedder
from pose.chuv_features import ChuvFeatureTracker, compute_chuv_features


def head_center(kxy: np.ndarray) -> np.ndarray:
    """Centro approssimato della testa da naso/occhi/orecchie (media dei
    punti disponibili, ignorando eventuali NaN)."""
    idxs = [KP["nose"], KP["left_eye"], KP["right_eye"], KP["left_ear"], KP["right_ear"]]
    pts = kxy[idxs]
    valid = pts[~np.isnan(pts).any(axis=1)]
    return valid.mean(axis=0) if len(valid) else kxy[KP["nose"]]


def format_metrics(track_id: int, angles: dict, energy: float, rep_score: dict,
                    gaze: dict | None = None, hands_info: dict | None = None,
                    posture: dict | None = None) -> list[str]:
    lines = [f"ID {track_id}"]
    lines.append(f"movement energy: {energy:6.1f}")
    if not np.isnan(rep_score.get("peak_power_ratio", np.nan)):
        lines.append(f"wrist repetitiveness: {rep_score['peak_power_ratio']:.2f} @ {rep_score['peak_freq_hz']:.1f}Hz")
    for name in ("left_elbow_angle", "right_elbow_angle"):
        if name in angles and not np.isnan(angles[name]):
            lines.append(f"{name}: {angles[name]:5.0f} deg")
    if posture:
        if not np.isnan(posture.get("vertical_excursion", np.nan)):
            lines.append(f"vertical excursion: {posture['vertical_excursion']:.2f}")
        if not np.isnan(posture.get("activity_ratio", np.nan)):
            lines.append(f"time in motion: {posture['activity_ratio']*100:4.0f}%")
        if not np.isnan(posture.get("self_touch_ratio", np.nan)):
            lines.append(f"self-touch: {posture['self_touch_ratio']*100:4.0f}%")
    if gaze:
        if not np.isnan(gaze.get("yaw", np.nan)):
            lines.append(f"head yaw: {gaze['yaw']:5.0f} deg")
        if gaze.get("attention_target") is not None:
            lines.append(f"looking at ID {gaze['attention_target']}: {gaze['attention_score']:.2f}")
        if not np.isnan(gaze.get("mouth_ratio", np.nan)):
            lines.append(f"mouth opening: {gaze['mouth_ratio']:.2f}")
        if not np.isnan(gaze.get("mouth_rep_power_ratio", np.nan)):
            lines.append(f"mouth repetitiveness: {gaze['mouth_rep_power_ratio']:.2f} @ {gaze['mouth_rep_freq_hz']:.1f}Hz")
        if not np.isnan(gaze.get("yaw_rep_power_ratio", np.nan)):
            lines.append(f"head shake: {gaze['yaw_rep_power_ratio']:.2f} @ {gaze['yaw_rep_freq_hz']:.1f}Hz")
        if not np.isnan(gaze.get("pitch_rep_power_ratio", np.nan)):
            lines.append(f"head nod: {gaze['pitch_rep_power_ratio']:.2f} @ {gaze['pitch_rep_freq_hz']:.1f}Hz")
        if not np.isnan(gaze.get("blink_rate_per_min", np.nan)):
            lines.append(f"blink/min: {gaze['blink_rate_per_min']:.0f}")
        left_b, right_b = gaze.get("left_eyebrow_raise", np.nan), gaze.get("right_eyebrow_raise", np.nan)
        if not (np.isnan(left_b) and np.isnan(right_b)):
            lines.append(f"eyebrows: L={left_b:.2f} R={right_b:.2f}")
    if hands_info:
        for side, info in hands_info.items():
            lines.append(f"{side} hand: openness={info['openness']:.2f}")
            if not np.isnan(info.get("rep_power_ratio", np.nan)):
                lines.append(f"  finger repetitiveness: {info['rep_power_ratio']:.2f} @ {info['rep_freq_hz']:.1f}Hz")
    return lines


def iter_live_frames(source, fps: float, model_name: str, device: str,
                      window_seconds: float, blur_faces: bool,
                      with_eyes: bool = False, with_mouth: bool = False,
                      with_eyebrows: bool = False, with_head_movement: bool = False,
                      face_model: str = "face_landmarker.task",
                      with_hands: bool = False, hand_model: str = "hand_landmarker.task",
                      activity_threshold: float = 40.0, self_touch_threshold: float = 0.5,
                      blink_ear_threshold: float = 0.2,
                      target_track_id: int | None = None,
                      with_reid: bool = False, reid_max_lost_seconds: float = 180.0,
                      reid_max_signature_dist: float = 0.12,
                      reid_min_signature_frames: int = 15,
                      reid_color_weight: float = 0.5, reid_position_weight: float = 0.5,
                      with_chuv_features: bool = False, conf_threshold: float = 0.1,
                      tracker_config: str = "bytetrack.yaml",
                      max_people: int | None = None,
                      session_mode: SessionMode = SessionMode.MULTIPLE,
                      flag_uncertain: bool = True,
                      use_appearance_embedding: bool = False,
                      embedding_device: str = "cpu"):
    """Generatore che contiene TUTTA la logica per-frame della pipeline pose
    (tracking, reid, gaze, mani, feature engineering, disegno overlay) in un
    unico punto, condiviso da `run_live()` (CLI, sotto) e da `pipeline_runner.py`
    (GUI): evita di duplicare ~300 righe di logica stateful che altrimenti
    rischierebbero di divergere tra le due interfacce nel tempo.

    Disegna SEMPRE l'overlay sul frame restituito (skeleton, etichette,
    pannelli metriche, fps) -- la GUI lo vuole sempre a schermo; il costo di
    disegnarlo comunque quando `run_live()` e' in modalita' `--no-window` e'
    trascurabile rispetto al costo dell'inferenza.

    Yield per ogni frame processato: `(frame, rows, now, smoothed_fps, people,
    gaze_by_track, hands_by_track)`. I primi quattro sono per `run_live()`
    (CLI) e per l'uso base della GUI; gli ultimi tre sono gia' pronti per
    essere ridisegnati SU UN FRAME DIVERSO con le stesse funzioni di
    `common/viz.py` (`draw_skeleton`, ecc.) -- serve a `pipeline_runner.py`
    per sovrapporre lo scheletro pose al frame gia' annotato dalla
    segmentazione in modalita' "both", senza duplicare la logica di
    tracking/reid/gaze/mani qui sopra.
    """
    tracker = PoseTracker(model_name=model_name, device=device,
                           conf_threshold=conf_threshold, tracker=tracker_config,
                           max_people=max_people)
    window_len = max(8, int(window_seconds * fps))
    seen_track_ids: set[int] = set()

    # --- feature engineering "in stile CHUV" (opzionale): stessi angoli/
    # distanze/simmetria/COM/derivate temporali di train.py nel repository
    # Video-Annotation-System, ricalcolati qui su COCO-17/YOLO invece di
    # BODY-25/SAM3 (vedi chuv_features.py per il dettaglio e i limiti).
    chuv_tracker = ChuvFeatureTracker() if with_chuv_features else None

    # --- re-identificazione (opzionale): ri-associa una persona che rientra
    # nell'inquadratura con un nuovo track_id (ByteTrack) alla sua identita'
    # precedente, in base alla firma antropometrica + colore maglia/pantaloni/
    # capelli + posizione (vedi reid.py). Punto di wiring unico: se attiva,
    # sostituisce subito frame_result.people con la versione a person_id
    # stabile, cosi' tutto il resto della funzione (che tratta gia' l'id
    # come una chiave generica) non richiede altre modifiche.
    # `use_appearance_embedding` (opzionale, richiede 'torch'+'torchreid',
    # vedi pose/appearance_embedding.py): costruito UNA volta qui, non ad
    # ogni frame -- se la dipendenza manca, `OSNetEmbedder()` solleva
    # `ImportError` subito, con un messaggio chiaro (nessun fallback
    # silenzioso, stesso trattamento di Sam2Tracker/Sam31Tracker).
    embedder = (
        OSNetEmbedder(device=embedding_device)
        if (with_reid and use_appearance_embedding) else None
    )
    reidentifier = ReIdentifier(
        max_lost_seconds=reid_max_lost_seconds,
        max_signature_dist=reid_max_signature_dist,
        min_signature_frames=reid_min_signature_frames,
        color_bonus_weight=reid_color_weight,
        position_bonus_weight=reid_position_weight,
        max_people=max_people,
        session_mode=session_mode, flag_uncertain=flag_uncertain,
        embedder=embedder,
    ) if with_reid else None

    kpt_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_len))
    conf_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_len))
    fingertip_buffers: dict[tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=window_len))
    ear_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_len))
    mar_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_len))
    yaw_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_len))
    pitch_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_len))
    self_touch_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=window_len))

    # --- viso: le quattro sotto-feature (occhi/blink, bocca, sopracciglia,
    # movimento testa) condividono UNA SOLA chiamata a FaceLandmarker per
    # frame (restituisce sempre tutti i landmark del volto insieme, non ha
    # senso richiamarlo piu' volte con sottoinsiemi diversi) -- ma ciascuna
    # viene poi calcolata/salvata/disegnata solo se il proprio flag e'
    # attivo, cosi' si possono selezionare indipendentemente in GUI/CLI. ---
    with_face_any = with_eyes or with_mouth or with_eyebrows or with_head_movement
    head_gaze = None
    if with_face_any:
        # Import ritardato: gaze_head richiede mediapipe + il modello
        # face_landmarker.task, non necessari per il resto della pipeline.
        from pose.gaze_head import (
            HeadGazeEstimator, match_faces_to_tracks, joint_attention_score,
            MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT,
            LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
            LEFT_EYEBROW_IDX, RIGHT_EYEBROW_IDX,
        )
        head_gaze = HeadGazeEstimator(model_path=face_model, num_faces=4)

    hand_tracker = None
    if with_hands:
        # Import ritardato: hands richiede mediapipe + il modello
        # hand_landmarker.task, non necessari per il resto della pipeline.
        from pose.hands import HandTracker, match_hands_to_wrists, compute_finger_curls, hand_openness
        hand_tracker = HandTracker(model_path=hand_model, num_hands=4)

    prev_t = time.time()
    smoothed_fps = fps

    for frame_result in tracker.run(source=source, stream=True):
        frame = frame_result.frame
        now = time.time()
        dt = max(now - prev_t, 1e-6)
        smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / dt)
        prev_t = now

        if reidentifier is not None:
            frame_result.people = reidentifier.resolve(frame_result.people, now, frame=frame)

        # --- nuove persone: stampa a console il track_id la prima volta che
        # compare, cosi' si puo' identificare rapidamente chi e' chi (es. per
        # scegliere --target-track-id) senza dover rileggere il CSV a posteriori ---
        for tid, _, _ in frame_result.people:
            if tid not in seen_track_ids:
                seen_track_ids.add(tid)
                print(f"New person detected: ID {tid}"
                      + (" (target)" if target_track_id == tid else ""))

        # --- head-pose / gaze (una volta per frame, poi associato ai track) ---
        gaze_by_track: dict[int, dict] = {}
        head_centers: dict[int, np.ndarray] = {
            tid: head_center(kxy) for tid, kxy, _ in frame_result.people
        }
        if with_face_any and frame_result.people:
            timestamp_ms = int(now * 1000)
            face_results = head_gaze.process(frame, timestamp_ms)
            face_centers = [fr.landmarks_xy.mean(axis=0) for fr in face_results]
            track_ids = list(head_centers.keys())
            assignment = match_faces_to_tracks(
                face_centers, track_ids, [head_centers[t] for t in track_ids]
            )
            for face_idx, tid in assignment.items():
                # --- filtro target: se e' impostato --target-track-id, calcola
                # i segnali derivati dal volto (blink, bocca, sopracciglia,
                # scuotimento/annuimento) solo per quella persona (es. il
                # bambino), non per chiunque compaia nell'inquadratura (es. il
                # caregiver). Lo scheletro/postura restano comunque tracciati
                # per tutti (vedi loop principale piu' sotto). ---
                if target_track_id is not None and tid != target_track_id:
                    continue

                fr = face_results[face_idx]
                # attention_target/score sempre presenti (servono da placeholder
                # anche se with_head_movement e' spento): il resto delle chiavi
                # viene aggiunto SOLO dalla sotto-feature corrispondente attiva,
                # cosi' ciascuna e' selezionabile indipendentemente -- il resto
                # della funzione (format_metrics, CSV, disegno) legge sempre con
                # gaze.get(key, ...) quindi una chiave assente equivale a "non
                # disponibile", non a un errore.
                entry: dict = {"attention_target": None, "attention_score": 0.0}

                if with_head_movement:
                    entry.update({
                        "yaw": fr.yaw, "pitch": fr.pitch, "roll": fr.roll,
                        "yaw_rep_power_ratio": np.nan, "yaw_rep_freq_hz": np.nan,
                        "pitch_rep_power_ratio": np.nan, "pitch_rep_freq_hz": np.nan,
                    })
                    # --- scuotimento (yaw) e annuimento (pitch) della testa:
                    # stessa logica FFT usata per polso/dita, applicata pero'
                    # al segnale grezzo di yaw/pitch (non alla sua velocita')
                    # perche' e' gia' un angolo con segno intorno a uno zero
                    # naturale -- evita l'artefatto di raddoppio della
                    # frequenza documentato per lo score su polso/dita (che usa
                    # la velocita' perche' la posizione del polso deriva col
                    # corpo).
                    if not np.isnan(fr.yaw):
                        yaw_buffers[tid].append(fr.yaw)
                    if not np.isnan(fr.pitch):
                        pitch_buffers[tid].append(fr.pitch)
                    if len(yaw_buffers[tid]) >= window_len:
                        yaw_rep = repetitive_motion_score(np.array(yaw_buffers[tid]), fps)
                        entry["yaw_rep_power_ratio"] = yaw_rep["peak_power_ratio"]
                        entry["yaw_rep_freq_hz"] = yaw_rep["peak_freq_hz"]
                    if len(pitch_buffers[tid]) >= window_len:
                        pitch_rep = repetitive_motion_score(np.array(pitch_buffers[tid]), fps)
                        entry["pitch_rep_power_ratio"] = pitch_rep["peak_power_ratio"]
                        entry["pitch_rep_freq_hz"] = pitch_rep["peak_freq_hz"]

                if with_mouth:
                    entry.update({
                        "mouth_ratio": fr.mouth_ratio,
                        "mouth_rep_power_ratio": np.nan, "mouth_rep_freq_hz": np.nan,
                        "mouth_pts": fr.landmarks_xy[[MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]],
                    })
                    # --- repetitivita' della bocca (proxy di mouthing/
                    # vocalizzazione ripetuta): la lingua non e' tracciabile
                    # con FaceLandmarker (che modella solo la superficie del
                    # volto, non strutture intraorali), quindi usiamo
                    # l'oscillazione periodica del MAR come sostituto
                    # comportamentalmente informativo, con la stessa logica
                    # FFT gia' usata per polso/dita.
                    if not np.isnan(fr.mouth_ratio):
                        mar_buffers[tid].append(fr.mouth_ratio)
                    if len(mar_buffers[tid]) >= window_len:
                        mouth_rep = repetitive_motion_score(np.array(mar_buffers[tid]), fps)
                        entry["mouth_rep_power_ratio"] = mouth_rep["peak_power_ratio"]
                        entry["mouth_rep_freq_hz"] = mouth_rep["peak_freq_hz"]

                if with_eyes:
                    entry.update({
                        "blink_rate_per_min": np.nan,
                        "left_eye_pts": fr.landmarks_xy[LEFT_EYE_EAR_IDX],
                        "right_eye_pts": fr.landmarks_xy[RIGHT_EYE_EAR_IDX],
                    })
                    if not np.isnan(fr.eye_ratio):
                        ear_buffers[tid].append(fr.eye_ratio)
                    if len(ear_buffers[tid]) >= window_len:
                        ear_seq = np.array(ear_buffers[tid])
                        closed = ear_seq < blink_ear_threshold
                        n_blinks = int(np.sum(np.diff(closed.astype(int)) == 1))
                        entry["blink_rate_per_min"] = n_blinks / window_seconds * 60.0

                if with_eyebrows:
                    entry.update({
                        "left_eyebrow_raise": fr.left_eyebrow_raise,
                        "right_eyebrow_raise": fr.right_eyebrow_raise,
                        "left_eyebrow_pts": fr.landmarks_xy[LEFT_EYEBROW_IDX],
                        "right_eyebrow_pts": fr.landmarks_xy[RIGHT_EYEBROW_IDX],
                    })

                gaze_by_track[tid] = entry

            # euristica di attenzione condivisa: richiede lo yaw, quindi solo
            # se with_head_movement e' attivo, e solo se ci sono esattamente
            # due persone con head-pose disponibile in questo frame
            if with_head_movement:
                tracked_with_gaze = [t for t in track_ids if t in gaze_by_track]
                if len(tracked_with_gaze) == 2:
                    a, b = tracked_with_gaze
                    score_a_to_b = joint_attention_score(
                        head_centers[a], gaze_by_track[a]["yaw"], head_centers[b], frame.shape[1])
                    score_b_to_a = joint_attention_score(
                        head_centers[b], gaze_by_track[b]["yaw"], head_centers[a], frame.shape[1])
                    gaze_by_track[a]["attention_target"] = b
                    gaze_by_track[a]["attention_score"] = score_a_to_b
                    gaze_by_track[b]["attention_target"] = a
                    gaze_by_track[b]["attention_score"] = score_b_to_a

        # --- mani/dita (una volta per frame, poi associate ai polsi YOLO) ---
        hands_by_track: dict[int, dict] = defaultdict(dict)
        if with_hands and frame_result.people:
            timestamp_ms = int(now * 1000)
            hand_results = hand_tracker.process(frame, timestamp_ms)
            hand_wrist_points = [hr.landmarks_xy[0] for hr in hand_results]
            track_wrists = []
            for tid, kxy, _ in frame_result.people:
                track_wrists.append((tid, "left", kxy[KP["left_wrist"]]))
                track_wrists.append((tid, "right", kxy[KP["right_wrist"]]))
            assignment = match_hands_to_wrists(hand_wrist_points, track_wrists)

            for hand_idx, (tid, side) in assignment.items():
                if target_track_id is not None and tid != target_track_id:
                    continue
                hand_xy = hand_results[hand_idx].landmarks_xy
                openness = hand_openness(hand_xy)
                curls = compute_finger_curls(hand_xy)

                key = (tid, side)
                fingertip_buffers[key].append(hand_xy[8])  # punta indice

                rep_power_ratio, rep_freq_hz = np.nan, np.nan
                if len(fingertip_buffers[key]) >= window_len:
                    tip_seq = np.stack(fingertip_buffers[key])
                    tip_speed = np.linalg.norm(np.diff(tip_seq, axis=0), axis=1) * fps
                    rep = repetitive_motion_score(tip_speed, fps)
                    rep_power_ratio, rep_freq_hz = rep["peak_power_ratio"], rep["peak_freq_hz"]

                hands_by_track[tid][side] = {
                    "landmarks": hand_xy, "openness": openness,
                    "rep_power_ratio": rep_power_ratio, "rep_freq_hz": rep_freq_hz,
                    **curls,
                }

        rows_this_frame = []
        panel_y = 10  # riquadri metriche impilati in un angolo fisso (non seguono piu' la testa)
        for track_id, kxy, kconf in frame_result.people:
            if blur_faces:
                frame = blur_face(frame, kxy, kconf)

            kpt_buffers[track_id].append(kxy)
            conf_buffers[track_id].append(kconf)

            angles = compute_joint_angles(kxy)
            gaze = gaze_by_track.get(track_id)
            hands_info = hands_by_track.get(track_id)

            # --- self-touch: calcolato ogni frame (non serve la finestra) ---
            scale = torso_length(kxy)
            touch_left = self_touch_score(kxy[KP["left_wrist"]], head_centers[track_id], scale)
            touch_right = self_touch_score(kxy[KP["right_wrist"]], head_centers[track_id], scale)
            touch_now = np.nanmax([touch_left, touch_right]) if not (np.isnan(touch_left) and np.isnan(touch_right)) else np.nan
            if not np.isnan(touch_now):
                self_touch_buffers[track_id].append(touch_now)

            energy = np.nan
            rep_score = {"peak_freq_hz": np.nan, "peak_power_ratio": np.nan}
            posture = {"vertical_excursion": np.nan, "activity_ratio": np.nan,
                       "self_touch_ratio": np.nan, "self_touch_now": touch_now}
            if len(kpt_buffers[track_id]) >= window_len:
                seq = np.stack(kpt_buffers[track_id])
                diffs = np.diff(seq, axis=0) * fps
                speed = np.linalg.norm(diffs, axis=2)
                energy_series = (speed ** 2).sum(axis=1)
                energy = float(energy_series.mean())
                wrist_speed = speed[:, [KP["left_wrist"], KP["right_wrist"]]].mean(axis=1)
                rep_score = repetitive_motion_score(wrist_speed, fps)

                posture["vertical_excursion"] = vertical_excursion(seq)
                posture["activity_ratio"] = activity_ratio(energy_series, activity_threshold)
                if len(self_touch_buffers[track_id]) >= window_len:
                    touch_arr = np.array(self_touch_buffers[track_id])
                    posture["self_touch_ratio"] = float(np.mean(touch_arr > self_touch_threshold))

                row = {
                    "t": now, "track_id": track_id, "movement_energy": energy,
                    "peak_freq_hz": rep_score["peak_freq_hz"],
                    "peak_power_ratio": rep_score["peak_power_ratio"],
                    "vertical_excursion": posture["vertical_excursion"],
                    "activity_ratio": posture["activity_ratio"],
                    "self_touch_ratio": posture["self_touch_ratio"],
                    "self_touch_now": touch_now,
                    **angles,
                }
                if gaze:
                    # .get(key, NaN/None) ovunque: quali chiavi esistono
                    # dipende da quali sotto-feature del viso sono attive
                    # (vedi sopra) -- una colonna CSV resta comunque sempre
                    # presente, solo NaN se la sua sotto-feature e' spenta.
                    row.update({
                        "head_yaw": gaze.get("yaw", np.nan), "head_pitch": gaze.get("pitch", np.nan),
                        "head_roll": gaze.get("roll", np.nan),
                        "attention_target": gaze.get("attention_target"),
                        "attention_score": gaze.get("attention_score", 0.0),
                        "mouth_ratio": gaze.get("mouth_ratio", np.nan),
                        "blink_rate_per_min": gaze.get("blink_rate_per_min", np.nan),
                        "mouth_rep_power_ratio": gaze.get("mouth_rep_power_ratio", np.nan),
                        "mouth_rep_freq_hz": gaze.get("mouth_rep_freq_hz", np.nan),
                        "yaw_rep_power_ratio": gaze.get("yaw_rep_power_ratio", np.nan),
                        "yaw_rep_freq_hz": gaze.get("yaw_rep_freq_hz", np.nan),
                        "pitch_rep_power_ratio": gaze.get("pitch_rep_power_ratio", np.nan),
                        "pitch_rep_freq_hz": gaze.get("pitch_rep_freq_hz", np.nan),
                        "left_eyebrow_raise": gaze.get("left_eyebrow_raise", np.nan),
                        "right_eyebrow_raise": gaze.get("right_eyebrow_raise", np.nan),
                    })
                if hands_info:
                    for side, info in hands_info.items():
                        row.update({
                            f"{side}_hand_openness": info["openness"],
                            f"{side}_hand_rep_power_ratio": info["rep_power_ratio"],
                            f"{side}_hand_rep_freq_hz": info["rep_freq_hz"],
                            **{f"{side}_{k}": v for k, v in info.items()
                               if k.endswith("_curl")},
                        })
                if chuv_tracker is not None:
                    chuv_feats = compute_chuv_features(kxy, track_id, now, chuv_tracker)
                    row.update({f"chuv_{k}": v for k, v in chuv_feats.items()})
                rows_this_frame.append(row)

            track_color = get_track_color(track_id)
            is_target = target_track_id == track_id
            draw_skeleton(frame, kxy, kconf, color=track_color)
            draw_person_label(frame, head_centers[track_id], track_id, track_color, is_target=is_target)
            if gaze and not np.isnan(gaze.get("yaw", np.nan)):
                hc = head_centers[track_id]
                ang = np.radians(gaze["yaw"])
                tip = (int(hc[0] + 60 * np.sin(ang)), int(hc[1] - 60 * np.cos(ang)))
                cv2.arrowedLine(frame, tuple(hc.astype(int)), tip, (255, 0, 255), 2, tipLength=0.3)
            if gaze:
                # draw_face_signals ignora silenziosamente le parti assenti
                # (None): con le sotto-feature del viso indipendenti, "gaze"
                # puo' contenere solo un sottoinsieme di bocca/occhi/
                # sopracciglia -- non serve piu' controllare quale chiave
                # specifica sia presente prima di chiamarla.
                draw_face_signals(frame, gaze.get("mouth_pts"),
                                   gaze.get("left_eye_pts"), gaze.get("right_eye_pts"),
                                   gaze.get("left_eyebrow_pts"), gaze.get("right_eyebrow_pts"))
            if hands_info:
                for info in hands_info.values():
                    draw_hand(frame, info["landmarks"])
            # riquadro metriche impilato in un angolo fisso (alto a sinistra),
            # non piu' sopra la testa della persona: cosi' non si sposta/non
            # copre la scena mentre la persona si muove. Il colore del bordo
            # (uguale allo scheletro) e la riga "ID N" lo collegano comunque
            # alla persona giusta anche se non e' piu' vicino a lei sullo schermo.
            metrics_lines = format_metrics(track_id, angles, energy, rep_score, gaze, hands_info, posture)
            draw_text_block(frame, metrics_lines, origin=(10, panel_y), border_color=track_color)
            _, panel_h = text_block_size(metrics_lines)
            panel_y += panel_h + 8

        draw_fps(frame, smoothed_fps)
        yield frame, rows_this_frame, now, smoothed_fps, frame_result.people, gaze_by_track, hands_by_track


def run_live(source, fps: float, model_name: str, device: str,
             window_seconds: float, blur_faces: bool, out_csv: str,
             show_window: bool = True,
             with_eyes: bool = False, with_mouth: bool = False,
             with_eyebrows: bool = False, with_head_movement: bool = False,
             face_model: str = "face_landmarker.task",
             with_hands: bool = False, hand_model: str = "hand_landmarker.task",
             activity_threshold: float = 40.0, self_touch_threshold: float = 0.5,
             blink_ear_threshold: float = 0.2,
             target_track_id: int | None = None,
             with_reid: bool = False, reid_max_lost_seconds: float = 180.0,
             reid_max_signature_dist: float = 0.12,
             reid_min_signature_frames: int = 15,
             reid_color_weight: float = 0.5, reid_position_weight: float = 0.5,
             with_chuv_features: bool = False, conf_threshold: float = 0.1,
             tracker_config: str = "bytetrack.yaml",
             max_people: int | None = None) -> pd.DataFrame:
    """CLI: consuma `iter_live_frames()` (unica fonte della logica per-frame,
    condivisa con la GUI), gestisce la finestra cv2 (se show_window) e
    accumula/salva il CSV finale. Comportamento identico a prima del
    refactor -- vedi `iter_live_frames()` per la logica vera e propria."""
    log_rows = []
    window_name = "Live pose behaviour (q per uscire, o chiudi la finestra)"

    try:
        for frame, rows, now, smoothed_fps, _people, _gaze_by_track, _hands_by_track in iter_live_frames(
            source=source, fps=fps, model_name=model_name, device=device,
            window_seconds=window_seconds, blur_faces=blur_faces,
            with_eyes=with_eyes, with_mouth=with_mouth,
            with_eyebrows=with_eyebrows, with_head_movement=with_head_movement,
            face_model=face_model,
            with_hands=with_hands, hand_model=hand_model,
            activity_threshold=activity_threshold,
            self_touch_threshold=self_touch_threshold,
            blink_ear_threshold=blink_ear_threshold,
            target_track_id=target_track_id,
            with_reid=with_reid, reid_max_lost_seconds=reid_max_lost_seconds,
            reid_max_signature_dist=reid_max_signature_dist,
            reid_min_signature_frames=reid_min_signature_frames,
            reid_color_weight=reid_color_weight, reid_position_weight=reid_position_weight,
            with_chuv_features=with_chuv_features, conf_threshold=conf_threshold,
            tracker_config=tracker_config, max_people=max_people,
        ):
            log_rows.extend(rows)

            if show_window:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                # 'q' funziona solo se la finestra video ha il focus (clicca
                # sulla finestra, non sul terminale, prima di premere il tasto).
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break  # finestra chiusa dal pulsante
    except KeyboardInterrupt:
        print("\nInterrupted from keyboard (Ctrl+C): saving the session anyway...")
    finally:
        if show_window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # su macOS serve dopo destroyAllWindows perché la finestra si chiuda davvero

    df = pd.DataFrame(log_rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} rows to {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Real-time movement recognition (Canon R8 / webcam / video)")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path")
    parser.add_argument("--fps", type=float, default=30.0, help="Expected frame rate of the source")
    parser.add_argument("--model", default="yolo26n-pose.pt",
                         help="Ultralytics YOLO-pose model. On a recorded video (not truly live) "
                              "there's no real-time constraint, so for hard footage (overhead "
                              "camera, fast motion) consider a bigger model, e.g. yolo26s-pose.pt "
                              "or yolo26m-pose.pt, for more stable keypoints and fewer "
                              "tracking-related ID switches.")
    parser.add_argument("--device", default=None,
                         help="mps | cpu | cuda (default: auto-rilevato -- cuda se una GPU "
                              "NVIDIA e' disponibile, altrimenti mps su Apple Silicon, "
                              "altrimenti cpu, vedi common/device.py)")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                         help="Minimum detection confidence passed to YOLO/ByteTrack. Keep this "
                              "at or below ByteTrack's track_low_thresh (0.1 by default): a higher "
                              "value strips out low-confidence detections before ByteTrack's own "
                              "low-confidence recovery stage ever sees them, causing unnecessary "
                              "new IDs on confidence dips (e.g. overhead camera, fast motion).")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Ultralytics tracker config. Use configs/bytetrack_permissive.yaml for "
                              "scenes with frequent brief confidence dips without real occlusion "
                              "(overhead camera, fast motion, artificial lighting) — longer "
                              "track_buffer and more tolerant thresholds, at the cost of a "
                              "slightly higher risk of ID switches on close interaction.")
    parser.add_argument("--window-seconds", type=float, default=3.0, help="Rolling window for the metrics")
    parser.add_argument("--blur-faces", action="store_true", help="Blur faces in real time (privacy)")
    parser.add_argument("--out", default="live_session.csv", help="Output CSV")
    parser.add_argument("--no-window", action="store_true", help="Run without a video window (logging only)")
    parser.add_argument("--with-eyes", action="store_true",
                         help="Enable eye tracking / blink rate (requires mediapipe + face_landmarker.task)")
    parser.add_argument("--with-mouth", action="store_true",
                         help="Enable mouth opening + repetitiveness (requires mediapipe + face_landmarker.task)")
    parser.add_argument("--with-eyebrows", action="store_true",
                         help="Enable eyebrow raise (requires mediapipe + face_landmarker.task)")
    parser.add_argument("--with-head-movement", action="store_true",
                         help="Enable head yaw/pitch, shake/nod repetitiveness, and the shared-"
                              "attention proxy between two people (requires mediapipe + "
                              "face_landmarker.task)")
    parser.add_argument("--face-model", default="face_landmarker.task",
                         help="Path to the MediaPipe FaceLandmarker model (used by any of "
                              "--with-eyes/--with-mouth/--with-eyebrows/--with-head-movement)")
    parser.add_argument("--with-hands", action="store_true",
                         help="Enable finger-level hand tracking (requires mediapipe + hand_landmarker.task)")
    parser.add_argument("--hand-model", default="hand_landmarker.task",
                         help="Path to the MediaPipe HandLandmarker model")
    parser.add_argument("--activity-threshold", type=float, default=40.0,
                         help="Movement-energy threshold above which a frame counts as 'active' (needs calibration)")
    parser.add_argument("--self-touch-threshold", type=float, default=0.5,
                         help="Threshold (0-1) above which a wrist near the head counts as self-touch")
    parser.add_argument("--blink-ear-threshold", type=float, default=0.2,
                         help="Eye Aspect Ratio threshold below which the eye is considered closed (requires --with-eyes)")
    parser.add_argument("--target-track-id", type=int, default=None,
                         help="If set, compute face/hand signals (blink, mouth, "
                              "eyebrows, head shake/nod, fingers) only for this "
                              "track_id, ignoring other people in frame (e.g. the "
                              "caregiver). Skeleton/posture keep being tracked for everyone. "
                              "Each person's ID is printed to the console the first time they "
                              "appear — run once without this flag to find it out, then "
                              "relaunch with --target-track-id N.")
    parser.add_argument("--with-reid", action="store_true",
                         help="Enable re-identification via anthropometric signature (see "
                              "reid.py): if a person leaves the frame and re-enters with a "
                              "new track_id (e.g. after a change of clothes or an absence), try to "
                              "restore their previous person_id instead of treating them as "
                              "new. When a merge happens, it's printed to the console and logged.")
    parser.add_argument("--reid-max-lost-seconds", type=float, default=180.0,
                         help="How many seconds a disappeared person stays in memory as a "
                              "re-entry candidate before being forgotten (default 3 minutes, "
                              "e.g. a bathroom break; requires --with-reid).")
    parser.add_argument("--reid-max-signature-dist", type=float, default=0.12,
                         help="Distance threshold between signatures below which a re-entry is "
                              "considered a match; a conservative value, not validated on real "
                              "data (requires --with-reid).")
    parser.add_argument("--reid-min-signature-frames", type=int, default=15,
                         help="Minimum number of valid frames to compute a reliable signature "
                              "(requires --with-reid).")
    parser.add_argument("--reid-color-weight", type=float, default=0.5,
                         help="How much the shirt/pants/hair color can 'discount' the distance "
                              "between body proportions when clothing looks the same "
                              "(0=color ignored, as before; 1=maximum effect). Can never "
                              "block a match that's already valid on proportions alone if "
                              "clothing looks different (requires --with-reid).")
    parser.add_argument("--reid-position-weight", type=float, default=0.5,
                         help="How much being in roughly the same spot, shortly after "
                              "disappearing, can 'discount' the distance between body "
                              "proportions (0=position ignored; 1=maximum effect). Useful when "
                              "someone is briefly occluded in place (e.g. a jacket put on over "
                              "existing clothes) rather than fully leaving and re-entering the "
                              "frame. Can never block a match that's valid on proportions alone "
                              "(requires --with-reid).")
    parser.add_argument("--max-people", type=int, default=None,
                         help="Known headcount for the session (e.g. 2 for a 1v1 child-caregiver "
                              "session, up to ~10 for a group session). Caps detections per frame "
                              "to the N most confident (suppresses spurious extra tracks from "
                              "noise/reflections), and, with --with-reid, once that many identities "
                              "have been confirmed, forces a re-entering track that finds no normal "
                              "match to reclaim the closest currently-missing identity instead of "
                              "minting a new one (it can only be one of the known N people) — the "
                              "one deliberate exception to reid.py's 'discount, never force' rule, "
                              "see reid.py docstring for the safety guardrails.")
    parser.add_argument("--with-chuv-features", action="store_true",
                         help="Add to the CSV the same features (angles, distances, symmetry, "
                              "center of mass, velocity/acceleration) computed by the "
                              "reference pipeline (Video-Annotation-System), recomputed here in "
                              "real time on COCO-17/YOLO instead of BODY-25/SAM3 (see "
                              "chuv_features.py for details and what's NOT replicated). "
                              "Columns prefixed with 'chuv_'.")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = args.device or detect_default_device()
    run_live(source, fps=args.fps, model_name=args.model, device=device,
              conf_threshold=args.conf_threshold, tracker_config=args.tracker,
              max_people=args.max_people,
              window_seconds=args.window_seconds, blur_faces=args.blur_faces,
              out_csv=args.out, show_window=not args.no_window,
              with_eyes=args.with_eyes, with_mouth=args.with_mouth,
              with_eyebrows=args.with_eyebrows, with_head_movement=args.with_head_movement,
              face_model=args.face_model,
              with_hands=args.with_hands, hand_model=args.hand_model,
              activity_threshold=args.activity_threshold,
              self_touch_threshold=args.self_touch_threshold,
              blink_ear_threshold=args.blink_ear_threshold,
              target_track_id=args.target_track_id,
              with_reid=args.with_reid,
              reid_max_lost_seconds=args.reid_max_lost_seconds,
              reid_max_signature_dist=args.reid_max_signature_dist,
              reid_min_signature_frames=args.reid_min_signature_frames,
              reid_color_weight=args.reid_color_weight,
              reid_position_weight=args.reid_position_weight,
              with_chuv_features=args.with_chuv_features)


if __name__ == "__main__":
    main()
