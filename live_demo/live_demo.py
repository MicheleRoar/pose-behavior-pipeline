"""
live_demo.py
============
Real-time movement recognition from a live source (designed for the
Canon EOS R8 connected to the MacBook Pro M1), with skeleton overlay and
behavioral metrics computed over a sliding window.

Supported sources:
- Canon EOS R8 via USB + EOS Webcam Utility: appears as a standard
  webcam, so `--source 0` (or the correct index if other
  webcams/virtual cameras are active: on Mac you can check the order in
  System Settings > Privacy & Security > Camera, or with
  `python -c "import cv2;[print(i, cv2.VideoCapture(i).isOpened()) for i in range(4)]"`).
- Canon EOS R8 with clean HDMI output + capture card (e.g. Elgato Cam
  Link): the capture card also appears as a standard webcam, same deal.
- A video file, by passing the path instead of an integer.

Example (skeleton only):

    python live_demo.py --source 0 --fps 30 --model yolo26n-pose.pt --device mps \\
        --window-seconds 3 --blur-faces --out live_session.csv

Example with face signals (eyes/mouth/eyebrows/head movement,
independently selectable) and finger-level hands (requires mediapipe
and the respective models, see README):

    python live_demo.py --source 0 --fps 30 --device mps \\
        --with-eyes --with-mouth --with-eyebrows --with-head-movement \\
        --with-hands --out live_session.csv

With multiple people in frame (e.g. child + caregiver), YOLO+ByteTrack
assigns a distinct track_id to each and treats them separately by
default (skeleton/posture for everyone). To restrict face/hand signals
to a single fixed person (e.g. don't compute the caregiver's blink
rate), do a first run without `--target-track-id` to see which IDs get
printed to the console ("New person detected: ID N"), then relaunch
specifying the chosen one:

    python live_demo.py --source 0 --fps 30 --device mps \\
        --with-eyes --with-mouth --with-hands --target-track-id 1 --out live_session.csv

To stop the session: press 'q' with the video window focused (the key
is only intercepted if the window has focus — click on the video, not
the terminal), or close the window with the button, or Ctrl+C from the
terminal. In all three cases the accumulated features are still saved
to `--out` before exiting.

Note: this script requires `ultralytics` + `opencv-python` installed
(see requirements.txt) and a real video source; it is not runnable in
the sandbox environment used to develop the rest of the pipeline. The
rendering logic (skeleton + metrics overlay) is instead verified
separately in `tests/live_render_check.py` with synthetic data.
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
    """Approximate head center from nose/eyes/ears (average of the
    available points, ignoring any NaNs)."""
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
    """Generator that holds ALL the per-frame logic of the pose pipeline
    (tracking, reid, gaze, hands, feature engineering, overlay drawing)
    in a single place, shared by `run_live()` (CLI, below) and by
    `pipeline_runner.py` (GUI): avoids duplicating ~300 lines of
    stateful logic that would otherwise risk diverging between the two
    interfaces over time.

    ALWAYS draws the overlay on the returned frame (skeleton, labels,
    metrics panels, fps) -- the GUI always wants it on screen; the cost
    of drawing it anyway when `run_live()` is in `--no-window` mode is
    negligible compared to the cost of inference.

    Yields for every processed frame: `(frame, rows, now, smoothed_fps,
    people, gaze_by_track, hands_by_track)`. The first four are for
    `run_live()` (CLI) and the GUI's basic use; the last three are
    already ready to be redrawn ON A DIFFERENT FRAME with the same
    functions from `common/viz.py` (`draw_skeleton`, etc.) -- needed by
    `pipeline_runner.py` to overlay the pose skeleton on the frame
    already annotated by segmentation in "both" mode, without
    duplicating the tracking/reid/gaze/hands logic above.
    """
    tracker = PoseTracker(model_name=model_name, device=device,
                           conf_threshold=conf_threshold, tracker=tracker_config,
                           max_people=max_people)
    window_len = max(8, int(window_seconds * fps))
    seen_track_ids: set[int] = set()

    # --- "CHUV-style" feature engineering (optional): the same angles/
    # distances/symmetry/COM/time derivatives as train.py in the
    # Video-Annotation-System repository, recomputed here on
    # COCO-17/YOLO instead of BODY-25/SAM3 (see chuv_features.py for
    # the detail and the limitations).
    chuv_tracker = ChuvFeatureTracker() if with_chuv_features else None

    # --- re-identification (optional): re-associates a person who
    # re-enters the frame with a new track_id (ByteTrack) to their
    # previous identity, based on anthropometric signature + shirt/
    # pants/hair color + position (see reid.py). Single wiring point:
    # if active, it immediately replaces frame_result.people with the
    # stable-person_id version, so the rest of the function (which
    # already treats the id as a generic key) requires no further
    # changes. `use_appearance_embedding` (optional, requires
    # 'torch'+'torchreid', see pose/appearance_embedding.py): built
    # ONCE here, not on every frame -- if the dependency is missing,
    # `OSNetEmbedder()` raises `ImportError` immediately, with a clear
    # message (no silent fallback, same treatment as
    # Sam2Tracker/Sam31Tracker).
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

    # --- face: the four sub-features (eyes/blink, mouth, eyebrows, head
    # movement) share A SINGLE call to FaceLandmarker per frame (it
    # always returns all face landmarks together, no point calling it
    # multiple times with different subsets) -- but each is then
    # computed/saved/drawn only if its own flag is active, so they can
    # be independently selected in the GUI/CLI. ---
    with_face_any = with_eyes or with_mouth or with_eyebrows or with_head_movement
    head_gaze = None
    if with_face_any:
        # Delayed import: gaze_head requires mediapipe + the
        # face_landmarker.task model, not needed for the rest of the
        # pipeline.
        from pose.gaze_head import (
            HeadGazeEstimator, match_faces_to_tracks, joint_attention_score,
            MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT,
            LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
            LEFT_EYEBROW_IDX, RIGHT_EYEBROW_IDX,
        )
        head_gaze = HeadGazeEstimator(model_path=face_model, num_faces=4)

    hand_tracker = None
    if with_hands:
        # Delayed import: hands requires mediapipe + the
        # hand_landmarker.task model, not needed for the rest of the
        # pipeline.
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

        # --- new people: prints the track_id to the console the first
        # time it appears, so you can quickly identify who is who (e.g.
        # to choose --target-track-id) without having to re-read the
        # CSV afterward ---
        for tid, _, _ in frame_result.people:
            if tid not in seen_track_ids:
                seen_track_ids.add(tid)
                print(f"New person detected: ID {tid}"
                      + (" (target)" if target_track_id == tid else ""))

        # --- head-pose / gaze (once per frame, then matched to tracks) ---
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
                # --- target filter: if --target-track-id is set,
                # compute the face-derived signals (blink, mouth,
                # eyebrows, shake/nod) only for that person (e.g. the
                # child), not for whoever appears in frame (e.g. the
                # caregiver). Skeleton/posture keep being tracked for
                # everyone regardless (see the main loop below). ---
                if target_track_id is not None and tid != target_track_id:
                    continue

                fr = face_results[face_idx]
                # attention_target/score are always present (needed as
                # a placeholder even if with_head_movement is off): the
                # rest of the keys are added ONLY by the corresponding
                # active sub-feature, so each is independently
                # selectable -- the rest of the function
                # (format_metrics, CSV, drawing) always reads with
                # gaze.get(key, ...) so a missing key just means "not
                # available", not an error.
                entry: dict = {"attention_target": None, "attention_score": 0.0}

                if with_head_movement:
                    entry.update({
                        "yaw": fr.yaw, "pitch": fr.pitch, "roll": fr.roll,
                        "yaw_rep_power_ratio": np.nan, "yaw_rep_freq_hz": np.nan,
                        "pitch_rep_power_ratio": np.nan, "pitch_rep_freq_hz": np.nan,
                    })
                    # --- head shake (yaw) and nod (pitch): the same FFT
                    # logic used for wrist/fingers, applied however to
                    # the raw yaw/pitch signal (not to its velocity)
                    # because it's already a signed angle around a
                    # natural zero -- avoids the frequency-doubling
                    # artifact documented for the wrist/finger score
                    # (which uses velocity because the wrist's position
                    # drifts along with the body).
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
                    # --- mouth repetitiveness (proxy for mouthing/
                    # repeated vocalization): the tongue isn't trackable
                    # with FaceLandmarker (which only models the
                    # surface of the face, not intraoral structures), so
                    # we use the periodic oscillation of the MAR as a
                    # behaviorally-informative substitute, with the
                    # same FFT logic already used for wrist/fingers.
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

            # shared attention heuristic: requires yaw, so only if
            # with_head_movement is active, and only if there are
            # exactly two people with head-pose available in this frame
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

        # --- hands/fingers (once per frame, then matched to YOLO wrists) ---
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
                fingertip_buffers[key].append(hand_xy[8])  # index fingertip

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
        panel_y = 10  # metrics panels stacked in a fixed corner (no longer follow the head)
        for track_id, kxy, kconf in frame_result.people:
            if blur_faces:
                frame = blur_face(frame, kxy, kconf)

            kpt_buffers[track_id].append(kxy)
            conf_buffers[track_id].append(kconf)

            angles = compute_joint_angles(kxy)
            gaze = gaze_by_track.get(track_id)
            hands_info = hands_by_track.get(track_id)

            # --- self-touch: computed every frame (doesn't need the window) ---
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
                    # .get(key, NaN/None) everywhere: which keys exist
                    # depends on which face sub-features are active (see
                    # above) -- a CSV column is always present regardless,
                    # just NaN if its sub-feature is off.
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
                # draw_face_signals silently ignores absent parts
                # (None): with the face sub-features independent,
                # "gaze" may contain only a subset of mouth/eyes/
                # eyebrows -- no need to check which specific key is
                # present before calling it anymore.
                draw_face_signals(frame, gaze.get("mouth_pts"),
                                   gaze.get("left_eye_pts"), gaze.get("right_eye_pts"),
                                   gaze.get("left_eyebrow_pts"), gaze.get("right_eyebrow_pts"))
            if hands_info:
                for info in hands_info.values():
                    draw_hand(frame, info["landmarks"])
            # metrics panel stacked in a fixed corner (top left), no
            # longer above the person's head: so it doesn't move/cover
            # the scene as the person moves. The border color (same as
            # the skeleton) and the "ID N" line still tie it to the
            # right person even if it's no longer close to them on
            # screen.
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
    """CLI: consumes `iter_live_frames()` (the single source of the
    per-frame logic, shared with the GUI), manages the cv2 window (if
    show_window), and accumulates/saves the final CSV. Behavior
    identical to before the refactor -- see `iter_live_frames()` for
    the actual logic."""
    log_rows = []
    window_name = "Live pose behaviour (press q to quit, or close the window)"

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
                # 'q' only works if the video window has focus (click on
                # the window, not the terminal, before pressing the key).
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed via the button
    except KeyboardInterrupt:
        print("\nInterrupted from keyboard (Ctrl+C): saving the session anyway...")
    finally:
        if show_window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # needed on macOS after destroyAllWindows for the window to actually close

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
                         help="mps | cpu | cuda (default: auto-detected -- cuda if an NVIDIA "
                              "GPU is available, otherwise mps on Apple Silicon, "
                              "otherwise cpu, see common/device.py)")
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
