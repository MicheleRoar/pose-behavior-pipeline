"""
pipeline_runner.py
====================
Single entry point used by the GUI (app.py / video_player.py) to obtain
frames with the overlay already drawn, ready to display, whichever
pipeline is chosen (pose, segmentation, or both). It does NOT duplicate
ANY per-frame logic: it calls directly `iter_live_frames()` (pose, from
live_demo.py) and/or `iter_segmentation_frames()` (segmentation, from
segmentation_demo.py) -- the same functions already used by the
respective CLIs, so GUI and CLI are guaranteed to stay identical in
behavior (see the docstrings of those two functions for the reasoning
behind this choice).

"both" mode: the two pipelines run in parallel, each with its own
tracker/reid instance -- NO identity is shared between pose and
segmentation for now (known limitation, to be addressed in the future:
the "ID N" labels drawn by segmentation and the track_id of pose don't
correspond). The pose skeleton is drawn ON TOP OF the frame already
annotated by segmentation (no longer side by side in two panels), per
explicit request and because there's no privacy/copyright constraint to
preserve here in doing so (public videos, not clinical). This requires
that the two independent sources (two separate `cv2.VideoCapture` on the
same file) stay in sync frame-by-frame -- a reasonable assumption for a
recorded file decoded sequentially by both without skips, but not
guaranteed if one of the two pipelines were to internally drop frames
(neither does today).

Note on the returned `now` timestamp: the pose pipeline uses
`time.time()` (system clock, consistent with live_demo.py which is also
meant for live sources), the segmentation one uses `frame_index / fps`
(video timeline, consistent with segmentation_demo.py which is meant for
recorded video). These are two different, pre-existing bases predating
this module: here we don't silently unify them, we just label every row
with its pipeline of origin (the "pipeline" column in "both" mode) so
whoever analyzes the CSV knows which column to look at.
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

# Recognized models/backends for the INDEPENDENT choice of segmentation
# and pose (the "Segmentation"/"Pose" section of the GUI, see
# webui/app.js): no longer a handful of predefined pipelines
# ("yolo"/"sam31"/"sam2" as the SOLE choice deciding both shape and
# skeleton), but two separate axes that can be freely combined -- see
# `iter_pipeline_frames()` below for which combinations are actually
# wired up and why.
POSE_BACKEND_KEYS = {"yolo", "mediapipe"}

# YOLO model used ONLY as a box+tracking proposer when "Pose: MediaPipe"
# is chosen WITHOUT active segmentation (the "Pose estimation" task
# alone, see `_iter_pose_mediapipe`): MediaPipe Tasks PoseLandmarker has
# no multi-person tracker of its own (see pose/mediapipe_pose.py), so a
# detector+tracker upstream is still needed to know WHERE to apply it and
# with which identity -- exactly as already happens for "Segmentation +
# MediaPipe pose" (segmentation_demo.py), except here it's not chosen by
# the user nor shown as an actual segmentation model: it's an internal
# implementation detail, not an output (no mask is drawn/saved by this
# path). "n" (nano): the fastest, a reasonable box is enough here, not an
# accurate segmentation.
_MEDIAPIPE_BOX_PROPOSER_MODEL = "yolo26n-seg.pt"


def _build_embedder(use_appearance_embedding: bool, embedding_device: str) -> OSNetEmbedder | None:
    """Built ONCE per generator (not on every frame) -- if
    `use_appearance_embedding` is False, `None` without touching torch/
    torchreid at all (no cost, no import). If True and the dependency is
    missing, `OSNetEmbedder()` raises `ImportError` immediately, not
    caught here: it propagates up to the caller (webui/api.py already
    turns it into a {"ok": False, "error": ...} for the status pill, see
    `Api._push_error`)."""
    return OSNetEmbedder(device=embedding_device) if use_appearance_embedding else None


@dataclass
class RunnerFrame:
    """A frame already ready for the GUI: overlay drawn, data rows for
    the CSV, timestamp, and which mode produced it."""
    frame: np.ndarray
    rows: list[dict]
    now: float
    mode: str  # "pose" | "segmentation" | "both"
    people_count: int = 0  # active tracks in this frame -- NOT reliably
    # derivable from len(rows): in "pose" mode rows are added only once
    # the feature sliding window fills up (see live_demo.py), so
    # len(rows) would be 0 for the first few seconds even with people
    # already tracked. Here we instead use the count of tracks actually
    # active in the current frame.


def iter_pipeline_frames(
    *, mode: str, source, fps: float, device: str = "mps",
    # -- pose (used if mode is "pose" or "both") --
    pose_backend: str = "yolo",  # "yolo" (default, full-frame + ByteTrack) | "mediapipe"
    pose_model: str = "yolo26n-pose.pt",
    with_hands: bool = False, hand_model: str = "hand_landmarker.task",
    with_eyes: bool = False, with_mouth: bool = False,
    with_eyebrows: bool = False, with_head_movement: bool = False,
    face_model: str = "face_landmarker.task",
    with_reid: bool = False, blur_faces: bool = False,
    window_seconds: float = 3.0,
    # -- segmentation (used if mode is "segmentation" or "both") --
    seg_model: str = "yolo26s-seg.pt", with_seg_reid: bool = False,
    seg_backend: str = "yolo", sam_chunk_size: int = 600, sam_overlap: int = 50,
    sam_chunk_store_dir: str | None = None,
    sam_redetect_every: int | None = None, sam_text_prompt: str | None = None,
    sam_appearance_fallback: bool = True,
    # -- pose inside the mask/box, MediaPipe (used if `pose_backend` is
    # "mediapipe" -- see pose/mediapipe_pose.py and _iter_segmentation /
    # _iter_pose_mediapipe) --
    with_mediapipe_pose: bool = False,
    pose_landmarker_model: str = "pose_landmarker_lite.task",
    # -- Identity & Re-identification (shared, see identity_manager.py).
    # `reid_max_lost_seconds` applies ONLY to the keypoint-based path
    # (ReIdentifier, pose_backend="yolo"|"mediapipe" with mode
    # "pose"/"both"): SegReIdentifier (segmentation) has no expiry by
    # design, see its docstring -- here it's simply ignored in that
    # case. --
    session_mode: SessionMode = SessionMode.MULTIPLE, flag_uncertain: bool = True,
    reid_max_lost_seconds: float = 180.0,
    # `use_appearance_embedding` (new, optional): adds a real appearance
    # embedding (OSNet) as a supplementary signal for
    # ReIdentifier/SegReIdentifier -- see pose/appearance_embedding.py
    # for the "why" (explicit request: stable ids, people "remembered"
    # for re-entry, inspired by OSNet+the EMA idea from StrongSORT).
    # Requires 'torch'+'torchreid' (heavy, optional dependency, NOT in
    # requirements.txt by default): if missing, building the embedder
    # raises a clear ImportError, propagated up to the GUI (same
    # treatment as SAM 3.1/SAM2 when CUDA/the package is missing).
    use_appearance_embedding: bool = False, embedding_device: str = "cpu",
    # -- shared --
    max_people: int | None = None, conf_threshold: float = 0.1,
    tracker_config: str = "bytetrack.yaml",
) -> Iterator[RunnerFrame]:
    """Dispatcher: chooses the pipeline(s) based on `mode` (Task:
    Segmentation/Pose/Both) and `pose_backend` (Pose: YOLO26 Pose/
    MediaPipe, independent of the chosen segmentation model -- see the
    module docstring for the architectural "why") and returns a uniform
    `RunnerFrame` iterator.

    Wired-up combinations (see also webui/app.js for the input
    auto-selection shown in the UI for each):
      - "pose" + pose_backend="yolo" (default): `_iter_pose` (YOLO26
        Pose full-frame + ByteTrack, UNCHANGED).
      - "pose" + pose_backend="mediapipe": `_iter_pose_mediapipe` (an
        internal YOLO detector/tracker proposes the boxes, MediaPipe
        estimates the pose inside each -- no mask produced/shown, see
        there for the honest limitations).
      - "segmentation": `_iter_segmentation`, UNCHANGED (`seg_backend`
        chooses YOLO26 Segment/SAM 3.1/SAM2; `with_mediapipe_pose` --
        equivalent to pose_backend="mediapipe" here -- applies MediaPipe
        inside each tracked mask).
      - "both" + pose_backend="mediapipe": delegates to
        `_iter_segmentation(..., with_mediapipe_pose=True)` with `mode`
        overridden to "both" in the returned `RunnerFrame` -- it's
        exactly the same path as "Segmentation + MediaPipe pose",
        because MediaPipe here is driven by the mask/box already tracked
        by segmentation regardless: no point duplicating the logic for a
        differently-named mode.
      - "both" + pose_backend="yolo" (default): `_iter_both`, UNCHANGED
        (the two pipelines run IN PARALLEL with independent
        tracker/reid -- known limitation, NO identity is shared between
        pose and segmentation, see its docstring)."""
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
            sam_appearance_fallback=sam_appearance_fallback,
            session_mode=session_mode, flag_uncertain=flag_uncertain,
            use_appearance_embedding=use_appearance_embedding,
            embedding_device=embedding_device,
        )
    elif mode == "both":
        if pose_backend == "mediapipe":
            # Same path as "Segmentation + MediaPipe pose"
            # (`iter_segmentation_frames`, UNCHANGED): here MediaPipe is
            # driven by the box/mask already tracked by segmentation
            # regardless, no point duplicating the logic just because
            # the mode is called "both" -- the only difference is the
            # `mode` labeled in the returned `RunnerFrame`.
            if with_seg_reid and max_people is None:
                raise ValueError("with_seg_reid requires max_people (the hard cap only "
                                  "makes sense with a known number of people)")
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
                sam_text_prompt=sam_text_prompt, sam_appearance_fallback=sam_appearance_fallback,
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
                sam_appearance_fallback=sam_appearance_fallback,
                session_mode=session_mode, flag_uncertain=flag_uncertain,
                use_appearance_embedding=use_appearance_embedding,
                embedding_device=embedding_device,
            )
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'pose'|'segmentation'|'both')")


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
        # `people` (5th element of the tuple, previously discarded) is
        # the list of active pose tracks in this frame -- reliable even
        # when `rows` is still empty (sliding window not yet full).
        yield RunnerFrame(frame=frame, rows=rows, now=now, mode="pose",
                           people_count=len(people))


def _iter_pose_mediapipe(*, source, fps, device, box_proposer_model, pose_landmarker_model,
                          with_reid, max_people, conf_threshold, tracker_config,
                          session_mode: SessionMode = SessionMode.MULTIPLE,
                          flag_uncertain: bool = True,
                          reid_max_lost_seconds: float = 180.0,
                          use_appearance_embedding: bool = False,
                          embedding_device: str = "cpu") -> Iterator[RunnerFrame]:
    """"Pose estimation" task with `pose_backend="mediapipe"`, WITHOUT
    active segmentation: MediaPipe Tasks PoseLandmarker has no
    multi-person tracker of its own (see pose/mediapipe_pose.py), so a
    lightweight YOLO tracker (`box_proposer_model`, box+track only, the
    mask is ignored) acts as a region proposer here -- same scheme as
    "Segmentation + MediaPipe pose" (segmentation_demo.py), except here
    the box-proposer is an internal detail: no mask is drawn/saved, and
    the UI doesn't present it as a segmentation choice (see
    `iter_pipeline_frames` for the "why").

    Honest limitations (inherited from pose/mediapipe_pose.py): only
    instantaneous joint angles in the CSV (`pose_*`), no sliding-window
    features (movement energy, repetitiveness, gaze, hands) -- those
    remain available only with `pose_backend="yolo"` (see `_iter_pose`).
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
                        sam_text_prompt=None, sam_appearance_fallback=True,
                        session_mode: SessionMode = SessionMode.MULTIPLE,
                        flag_uncertain: bool = True,
                        use_appearance_embedding: bool = False,
                        embedding_device: str = "cpu") -> Iterator[RunnerFrame]:
    if with_seg_reid and max_people is None:
        raise ValueError("with_seg_reid requires max_people (the hard cap only makes "
                          "sense with a known number of people)")
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
        sam_text_prompt=sam_text_prompt, sam_appearance_fallback=sam_appearance_fallback,
    ):
        # In segmentation there's no sliding window: one row per tracked
        # person per frame, so len(rows) is already the exact count.
        yield RunnerFrame(frame=vis, rows=rows, now=now, mode="segmentation",
                           people_count=len(rows))


def _iter_both(*, source, fps, device, pose_model, with_hands, hand_model,
               with_eyes, with_mouth, with_eyebrows, with_head_movement, face_model,
               with_reid, blur_faces, window_seconds,
               seg_model, with_seg_reid, max_people, conf_threshold,
               tracker_config, seg_backend="yolo", sam_chunk_size=600,
               sam_overlap=50, sam_chunk_store_dir=None,
               sam_redetect_every=None, sam_text_prompt=None,
               sam_appearance_fallback=True,
               session_mode: SessionMode = SessionMode.MULTIPLE,
               flag_uncertain: bool = True,
               reid_max_lost_seconds: float = 180.0,
               use_appearance_embedding: bool = False,
               embedding_device: str = "cpu") -> Iterator[RunnerFrame]:
    """Runs the two pipelines in parallel and draws the pose skeleton
    (+ hands/face if active) DIRECTLY on the frame already annotated by
    segmentation, instead of placing two panels side by side -- see the
    module docstring. It does not call `_iter_pose()`/`_iter_segmentation()`
    (which would return only the already-composited frame, without the
    raw data needed to redraw): it uses `iter_live_frames()` directly to
    get `people`/`gaze_by_track`/`hands_by_track`, and the same drawing
    functions from `common/viz.py` already used by `iter_live_frames()`
    -- no duplicated drawing logic, just a second call of the same
    functions on a different frame.

    Pose labels/metrics panel are NOT redrawn here (only
    skeleton/hands/face): the "ID N" label already present on the
    segmentation frame remains the only visible identity reference,
    avoiding two different overlapping numberings for the same person.
    """
    if with_seg_reid and max_people is None:
        raise ValueError("with_seg_reid requires max_people (the hard cap only makes "
                          "sense with a known number of people)")
    # Two distinct embedders (not shared between the two independent
    # pipelines, see the module docstring on "no identity shared between
    # pose and segmentation" -- same limitation, here simply extended to
    # the embedding): each loads its own OSNet model into memory, so the
    # cost is doubled in "both" mode with the embedding active.
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
        sam_text_prompt=sam_text_prompt, sam_appearance_fallback=sam_appearance_fallback,
    )

    # zip() (not zip_longest): if the two independent sources were to
    # end with a different number of frames for some reason, we stop at
    # the shorter one rather than returning a "both" frame with half
    # missing.
    for (pose_frame, pose_rows, pose_now, _pose_fps, people, gaze_by_track, hands_by_track), \
            (seg_vis, seg_rows, _seg_now, _frame_index, _raw_ids) in zip(pose_gen, seg_gen):
        canvas = seg_vis
        if canvas.shape[:2] != pose_frame.shape[:2]:
            # shouldn't happen for two trackers on the same source, but
            # if it did, drawing pose coordinates on a canvas of
            # different dimensions would put them out of place -- better
            # to skip the pose overlay for this frame than to silently
            # draw wrong points.
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
                # draw_face_signals silently ignores absent parts (None):
                # with the face sub-features independent, "gaze" may
                # contain only a subset of mouth/eyes/eyebrows.
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
        # We use the segmentation count as the canonical reference for
        # "active tracks": it's the pipeline whose "ID N" label stays
        # visible in this mode (see the module docstring).
        yield RunnerFrame(frame=canvas, rows=rows, now=pose_now, mode="both",
                           people_count=len(seg_rows))
