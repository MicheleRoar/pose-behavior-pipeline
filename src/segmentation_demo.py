"""
segmentation_demo.py
=====================
CURRENT main pipeline (temporary, see seg_estimation.py and README):
silhouette tracking via YOLO26-seg + ByteTrack, live overlay with mask
outline + ID label, CSV with one row per (frame, person). No
sliding-window behavioral features for now (movement energy,
repetitiveness, gaze, hands -- see the pose pipeline, on hold).

Optional, with `--with-mediapipe-pose`: applies MediaPipe Pose Landmarker
in SINGLE-person mode INSIDE the crop of each already-tracked silhouette
(not a multi-person detector on the whole frame -- see
`pose/mediapipe_pose.py` for why this choice was made), draws the
skeleton over the mask and adds the instantaneous (not sliding-window)
joint angles to the CSV.

Usage, on an already-recorded video:

    python segmentation_demo.py --source video.mp4 --fps 15 \\
        --model yolo26s-seg.pt --tracker configs/bytetrack_permissive.yaml \\
        --conf-threshold 0.1 --max-people 2 --out session_seg.csv

With --no-window, processes without opening a window (faster, useful for
a batch run with no need to watch the overlay live).
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from segmentation.seg_estimation import SegTracker, mask_area, mask_centroid
from segmentation.seg_reid import SegReIdentifier
from common.viz import draw_fps, draw_person_label, draw_skeleton, get_track_color
from common.device import detect_default_device
from pose.mediapipe_pose import MediaPipePoseByTrack
from pose.features import compute_joint_angles

BACKEND_KEYS = {"yolo", "sam31", "sam2"}


def build_tracker(backend: str, *, model_name: str, device: str, conf_threshold: float,
                    tracker_config: str, max_people: int | None,
                    sam_chunk_size: int, sam_overlap: int, sam_chunk_store_dir: str | None,
                    sam_reseed_new_people: bool = True,
                    sam_redetect_every: int | None = None,
                    sam_text_prompt: str | None = None):
    """Instantiates the right tracker based on `backend` -- the single
    place where the YOLO/SAM 3.1/SAM2 choice translates into a concrete
    class. All three follow the same `SegmentationBackend` protocol (see
    segmentation/backend.py), so the rest of this function (below)
    doesn't need to know which one was chosen. No longer "private" (no
    underscore): also reused by `benchmark_backends.py` to build the same
    tracker without going through `iter_segmentation_frames()` (which
    draws an overlay, useless here).

    `sam_reseed_new_people` (sam31/sam2 only, ignored with "yolo"):
    False gives the "pure SAM" condition for comparing methods (see
    benchmark_backends.py) -- YOLO proposes boxes ONLY on the video's
    first frame, never to discover new people at subsequent chunk
    boundaries. Default True (behavior already in use so far, unchanged).

    `sam_redetect_every` (sam31/sam2 only): reruns YOLO every N frames
    INSIDE the chunk, not just at its boundary -- solves the "sam2
    produces far fewer masks than yolo+bytetrack on the same video"
    problem (YOLO+ByteTrack detects on every frame, whereas SAM otherwise
    does so once every `sam_chunk_size` frames), see sam_backend.py.
    `None` (default) = original behavior, a single detection per chunk.

    `sam_text_prompt` (sam31 ONLY, ignored elsewhere -- SAM2 has no text
    prompt): if set (e.g. "person"), SAM 3.1 discovers people ON ITS OWN
    in the chunk instead of relying on YOLO as a proposer -- same
    technique as the CHUV production pipeline (`psifx video tracking sam3
    inference --text_prompt`), see sam31_estimation.py for details and
    the certainty limits about the real API."""
    if backend == "yolo":
        return SegTracker(model_name=model_name, device=device,
                           conf_threshold=conf_threshold, tracker=tracker_config,
                           max_people=max_people)
    if backend == "sam31":
        from segmentation.sam31_estimation import Sam31Tracker
        return Sam31Tracker(device=device, conf_threshold=conf_threshold,
                             chunk_size=sam_chunk_size, overlap=sam_overlap,
                             chunk_store_dir=sam_chunk_store_dir, max_people=max_people,
                             reseed_new_people=sam_reseed_new_people,
                             redetect_every=sam_redetect_every, text_prompt=sam_text_prompt)
    if backend == "sam2":
        from segmentation.sam2_estimation import Sam2Tracker
        return Sam2Tracker(device=device, conf_threshold=conf_threshold,
                            chunk_size=sam_chunk_size, overlap=sam_overlap,
                            chunk_store_dir=sam_chunk_store_dir, max_people=max_people,
                            reseed_new_people=sam_reseed_new_people,
                            redetect_every=sam_redetect_every)
    raise ValueError(f"unknown backend: {backend!r} (expected 'yolo'|'sam31'|'sam2')")


def iter_segmentation_frames(source, fps: float, model_name: str = "yolo26s-seg.pt",
                              device: str = "mps", conf_threshold: float = 0.1,
                              tracker_config: str = "bytetrack.yaml",
                              max_people: int | None = None,
                              seg_reidentifier: SegReIdentifier | None = None,
                              mediapipe_pose_estimator: MediaPipePoseByTrack | None = None,
                              backend: str = "yolo",
                              sam_chunk_size: int = 600, sam_overlap: int = 50,
                              sam_chunk_store_dir: str | None = None,
                              sam_reseed_new_people: bool = True,
                              sam_redetect_every: int | None = None,
                              sam_text_prompt: str | None = None):
    """Generator holding ALL the per-frame logic of the segmentation
    pipeline (tracking, optional re-id, optional per-mask pose, overlay
    drawing), shared by `run_segmentation()` (CLI, below) and by
    `pipeline_runner.py` (GUI) -- same choice as `iter_live_frames()` in
    live_demo.py, see its docstring for why. `seg_reidentifier` and
    `mediapipe_pose_estimator` must be built by the caller (instances
    persistent for the whole session, not recreatable here frame by
    frame); pass `None` to disable them. `mediapipe_pose_estimator` is a
    `MediaPipePoseByTrack` (an independent MediaPipe instance PER PERSON,
    not shared) -- see its docstring for why a single instance shared
    across the frame's people would crash MediaPipe ("Input timestamp
    must be monotonically increasing").

    `backend` selects the tracking/segmentation engine: "yolo" (default,
    YOLO26-seg + ByteTrack, unchanged), "sam31" or "sam2" (see
    segmentation/sam_backend.py -- require device="cuda" and the
    respective libraries installed, not available on mps/cpu).
    `sam_chunk_size` / `sam_overlap` / `sam_chunk_store_dir` are used only
    with these last two (ignored with "yolo").

    ALWAYS draws the overlay (mask, outline, ID label, + pose skeleton if
    `mediapipe_pose_estimator` is active) on the returned frame.

    Yields for every processed frame: `(vis, rows, now, frame_index, raw_ids)`
    where `rows` is the list of dicts (one row per person in this frame,
    same schema as the final CSV -- plus the `pose_*` joint angles if
    `mediapipe_pose_estimator` is active) and `raw_ids` are ByteTrack's
    raw track_ids BEFORE any re-id (useful only for the churn statistics
    printed by `run_segmentation()`).
    """
    tracker = build_tracker(
        backend, model_name=model_name, device=device, conf_threshold=conf_threshold,
        tracker_config=tracker_config, max_people=max_people,
        sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
        sam_chunk_store_dir=sam_chunk_store_dir, sam_reseed_new_people=sam_reseed_new_people,
        sam_redetect_every=sam_redetect_every, sam_text_prompt=sam_text_prompt,
    )

    for frame_result in tracker.run(source=source):
        now = frame_result.frame_index / fps
        vis = frame_result.frame.copy()
        raw_ids = [p[0] for p in frame_result.people]

        people = frame_result.people
        if seg_reidentifier is not None:
            people = seg_reidentifier.resolve(people, now=now, frame=frame_result.frame)

        rows_this_frame = []
        for track_id, bbox, poly, conf in people:
            centroid = mask_centroid(poly)
            area = mask_area(poly)
            row = {
                "frame": frame_result.frame_index, "time_s": now, "track_id": track_id,
                "bbox_x1": float(bbox[0]), "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]), "bbox_y2": float(bbox[3]),
                "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
                "mask_area_px": area, "box_conf": conf,
            }

            color = get_track_color(track_id)
            if poly.shape[0] >= 3:
                pts = poly.astype(np.int32).reshape(-1, 1, 2)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            else:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_pos = centroid if not np.isnan(centroid).any() else bbox[:2]
            draw_person_label(vis, label_pos, track_id, color)

            # -- pose INSIDE the tracked mask (optional): identity
            # "borrowed" from seg_reid/ByteTrack, see
            # pose/mediapipe_pose.py for why this design.
            if mediapipe_pose_estimator is not None:
                kxy, kconf = mediapipe_pose_estimator.estimate(
                    track_id, frame_result.frame, bbox, timestamp_ms=int(now * 1000))
                draw_skeleton(vis, kxy, kconf, color=color)
                angles = compute_joint_angles(kxy)
                row.update({f"pose_{k}": v for k, v in angles.items()})

            rows_this_frame.append(row)

        yield vis, rows_this_frame, now, frame_result.frame_index, raw_ids


def run_segmentation(source, fps: float, model_name: str = "yolo26s-seg.pt",
                      device: str = "mps", conf_threshold: float = 0.1,
                      tracker_config: str = "bytetrack.yaml",
                      max_people: int | None = None,
                      with_seg_reid: bool = False,
                      with_mediapipe_pose: bool = False,
                      pose_landmarker_model: str = "pose_landmarker_lite.task",
                      out_csv: str = "segmentation_session.csv",
                      show_window: bool = True,
                      backend: str = "yolo",
                      sam_chunk_size: int = 600, sam_overlap: int = 50,
                      sam_chunk_store_dir: str | None = None,
                      sam_reseed_new_people: bool = True,
                      sam_redetect_every: int | None = None,
                      sam_text_prompt: str | None = None) -> pd.DataFrame:
    """CLI: consumes `iter_segmentation_frames()` (single source of the
    per-frame logic, shared with the GUI), manages the cv2 window (if
    show_window) and prints the final churn/re-id statistics."""
    # -- re-identification (optional, requires --max-people): immediately
    # replaces frame_result.people with the stable-person_id version,
    # with a hard cap on max_people -- see seg_reid.py for why and the
    # limits (single wiring point, as in live_demo.py for reid.py).
    if with_seg_reid and max_people is None:
        raise ValueError("--with-seg-reid requires --max-people (the hard cap "
                          "only makes sense with a known number of people)")
    seg_reidentifier = SegReIdentifier(max_people=max_people) if with_seg_reid else None
    mediapipe_pose_estimator = (
        MediaPipePoseByTrack(model_path=pose_landmarker_model) if with_mediapipe_pose else None
    )

    rows: list[dict] = []
    raw_id_frame_count: dict[int, int] = defaultdict(int)   # raw ids assigned by ByteTrack
    final_id_frame_count: dict[int, int] = defaultdict(int)  # final ids (= raw if seg_reid off)
    win_name = "segmentation_demo"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    t_start = time.time()
    n_frames = 0
    for vis, rows_this_frame, now, frame_index, raw_ids in iter_segmentation_frames(
        source=source, fps=fps, model_name=model_name, device=device,
        conf_threshold=conf_threshold, tracker_config=tracker_config,
        max_people=max_people, seg_reidentifier=seg_reidentifier,
        mediapipe_pose_estimator=mediapipe_pose_estimator,
        backend=backend, sam_chunk_size=sam_chunk_size, sam_overlap=sam_overlap,
        sam_chunk_store_dir=sam_chunk_store_dir, sam_reseed_new_people=sam_reseed_new_people,
        sam_redetect_every=sam_redetect_every, sam_text_prompt=sam_text_prompt,
    ):
        n_frames = frame_index + 1
        for raw_id in raw_ids:
            raw_id_frame_count[raw_id] += 1
        for row in rows_this_frame:
            final_id_frame_count[row["track_id"]] += 1
        rows.extend(rows_this_frame)

        if show_window:
            elapsed = time.time() - t_start
            draw_fps(vis, n_frames / elapsed if elapsed > 0 else 0.0)
            cv2.imshow(win_name, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show_window:
        cv2.destroyAllWindows()

    n_raw_ids = len(raw_id_frame_count)
    lifespans = sorted(raw_id_frame_count.values())
    # 15 frames = reid.py's default min_signature_frames: same reference
    # used in track_stability_check.py, for comparability.
    short_lived = sum(1 for v in lifespans if v < 15)
    print(f"Frames processed: {n_frames}  |  Raw ids assigned by ByteTrack: {n_raw_ids}")
    if lifespans:
        median = lifespans[len(lifespans) // 2]
        print(f"Raw id duration in frames: min={lifespans[0]}  median={median}  max={lifespans[-1]}")
        print(f"Raw ids under 15 frames: {short_lived}/{n_raw_ids} ({100 * short_lived / n_raw_ids:.0f}%)")
    if seg_reidentifier is not None:
        n_final_ids = len(final_id_frame_count)
        print(f"seg_reid: {len(seg_reidentifier.merge_log)} raw track_ids re-associated -> "
              f"{n_final_ids} final ids (max_people={max_people} cap respected by construction)")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Segmentation-based tracking pipeline (YOLO26-seg + ByteTrack), "
                     "with live overlay and CSV. No keypoints for now -- see seg_estimation.py.")
    parser.add_argument("--source", required=True, help="Video path or webcam index")
    parser.add_argument("--fps", type=float, required=True, help="Source frame rate")
    parser.add_argument("--model", default="yolo26s-seg.pt",
                         help="YOLO26 instance segmentation model (yolo26n/s/m/l/x-seg.pt)")
    parser.add_argument("--device", default=None,
                         help="mps | cpu | cuda (default: auto-detected -- cuda if an NVIDIA "
                              "GPU is available, otherwise mps on Apple Silicon, "
                              "otherwise cpu, see common/device.py)")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                         help="Keep at or below ByteTrack's track_low_thresh (0.1 by default)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Ultralytics tracker config, e.g. configs/bytetrack_permissive.yaml")
    parser.add_argument("--max-people", type=int, default=None,
                         help="Known number of session participants (2 for 1v1, up to about "
                              "ten for a group): keeps only the N most confident detections "
                              "per frame; with --with-seg-reid it also becomes a hard cap "
                              "on the total number of identities in the session")
    parser.add_argument("--with-seg-reid", action="store_true",
                         help="Re-associates ByteTrack's raw track_ids to a fixed number of "
                              "stable person_ids (silhouette position/color/shape, see "
                              "seg_reid.py), guaranteeing that NEVER more than "
                              "--max-people ids are created in the whole session. Requires --max-people.")
    parser.add_argument("--with-mediapipe-pose", action="store_true",
                         help="Applies MediaPipe Pose Landmarker (single-person mode) "
                              "inside the crop of each tracked silhouette: draws the "
                              "skeleton and adds the joint angles (pose_*) to the CSV. "
                              "Requires mediapipe + the --pose-landmarker-model model, see "
                              "pose/mediapipe_pose.py.")
    parser.add_argument("--pose-landmarker-model", default="pose_landmarker_lite.task",
                         help="MediaPipe Pose Landmarker model (used only with "
                              "--with-mediapipe-pose)")
    parser.add_argument("--out", default="segmentation_session.csv", help="Output CSV")
    parser.add_argument("--no-window", action="store_true",
                         help="Run without a video window (log + CSV only, faster)")
    parser.add_argument("--backend", default="yolo", choices=sorted(BACKEND_KEYS),
                         help="Segmentation/tracking engine: 'yolo' (default, YOLO26-seg + "
                              "ByteTrack) | 'sam31' | 'sam2' (see segmentation/sam_backend.py "
                              "-- require device=cuda and the respective libraries installed)")
    parser.add_argument("--sam-chunk-size", type=int, default=600,
                         help="Only with --backend sam31/sam2: number of frames per chunk")
    parser.add_argument("--sam-overlap", type=int, default=50,
                         help="Only with --backend sam31/sam2: frames shared between one "
                              "chunk and the next, used for id reconciliation")
    parser.add_argument("--sam-chunk-store-dir", default=None,
                         help="Only with --backend sam31/sam2: folder to incrementally save "
                              "the results of each chunk (optional)")
    parser.add_argument("--sam-no-reseed-new-people", action="store_true",
                         help="Only with --backend sam31/sam2: disables discovery of "
                              "NEW people at chunk boundaries (YOLO proposes boxes ONLY "
                              "on the video's first frame). Gives the 'pure SAM' condition "
                              "for comparing against the default version (with reseeding) -- "
                              "see benchmark_backends.py and segmentation/sam_backend.py.")
    parser.add_argument("--sam-redetect-every", type=int, default=None,
                         help="Only with --backend sam31/sam2: reruns YOLO every N frames "
                              "INSIDE the chunk (not just at its boundary) to discover new "
                              "people more often -- useful if sam31/sam2 produce far fewer "
                              "masks than yolo+bytetrack on the same video (YOLO+ByteTrack "
                              "detects on every frame). Default: no intermediate re-detection.")
    parser.add_argument("--sam-text-prompt", default=None,
                         help="Only with --backend sam31 (SAM2 has no text prompt): open-ended "
                              "concept (e.g. 'person') with which SAM 3.1 discovers people ON "
                              "ITS OWN in the chunk, without YOLO as a proposer -- same "
                              "technique as the CHUV pipeline (psifx --text_prompt). See "
                              "sam31_estimation.py for the certainty limits about the real API "
                              "(not yet verified on a CUDA machine in this project).")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = args.device or detect_default_device()
    run_segmentation(source, fps=args.fps, model_name=args.model, device=device,
                      conf_threshold=args.conf_threshold, tracker_config=args.tracker,
                      max_people=args.max_people, with_seg_reid=args.with_seg_reid,
                      with_mediapipe_pose=args.with_mediapipe_pose,
                      pose_landmarker_model=args.pose_landmarker_model,
                      out_csv=args.out, show_window=not args.no_window,
                      backend=args.backend, sam_chunk_size=args.sam_chunk_size,
                      sam_overlap=args.sam_overlap, sam_chunk_store_dir=args.sam_chunk_store_dir,
                      sam_reseed_new_people=not args.sam_no_reseed_new_people,
                      sam_redetect_every=args.sam_redetect_every, sam_text_prompt=args.sam_text_prompt)


if __name__ == "__main__":
    main()
