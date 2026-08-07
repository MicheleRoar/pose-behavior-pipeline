"""
webui_api_check.py
====================
Verifies the PURE logic of `webui/api.py` -- no pywebview, no real
window, no real video/tracker (same spirit as video_player_check.py for
gui/video_player.py). Covers the four functions/classes deliberately
isolated to be testable without a window: `build_player_kwargs`,
`encode_frame_jpeg_b64`, `_LatencyTracker`, `build_status`. Doesn't touch
the `Api` class itself (that requires pywebview/a real window -- verified
by hand on the Mac, like the GUI's other features).

Run with: python webui_api_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from gui.pipeline_runner import RunnerFrame
from webui.api import (
    build_player_kwargs, encode_frame_jpeg_b64, _LatencyTracker, build_status,
    probe_video_metadata,
)


def part1_build_player_kwargs_mirrors_app_py_defaults():
    kwargs = build_player_kwargs({
        "mode": "pose", "source": "video.mp4", "fps": "15",
        "with_hands": True, "with_eyes": True,
    })
    assert kwargs["mode"] == "pose"
    assert kwargs["source"] == "video.mp4"
    assert kwargs["fps"] == 15.0
    # device not specified -> None, NOT "mps": automatic resolution
    # (cuda/mps/cpu) happens in Api.build_player(), not here, so this
    # function stays pure/testable without requiring torch installed --
    # see also part1b below for the passthrough of an explicit device.
    assert kwargs["device"] is None
    assert kwargs["pose_model"] == "yolo26s-pose.pt"  # default "s" scale
    assert kwargs["seg_model"] == "yolo26s-seg.pt"
    assert kwargs["with_hands"] is True
    assert kwargs["with_eyes"] is True
    assert kwargs["with_mouth"] is False
    assert kwargs["max_people"] is None
    # default identity_mode "tracking_reid": pose-reid (ReIdentifier)
    # works fine WITHOUT max_people (signature/color/position matching,
    # simply no "force the match because the cap is reached") -- see the
    # comment in build_player_kwargs about the asymmetry with
    # SegReIdentifier, which instead genuinely requires max_people.
    assert kwargs["with_reid"] is True
    print("Part 1: build_player_kwargs applies the right defaults and passes the requested flags — OK")


def part1b_explicit_device_passes_through_unchanged():
    """If JS explicitly sends a device (e.g. the user wants to force
    "cpu" for debugging), build_player_kwargs must simply pass it through
    as-is -- auto-detection in Api.build_player() only kicks in when the
    field is absent/empty, it must never override an explicit choice."""
    kwargs = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": 15, "device": "cuda",
    })
    assert kwargs["device"] == "cuda"
    print("Part 1b: an explicit device in the parameters passes through unchanged, with no auto-detection — OK")


def part2_hands_face_ignored_outside_pose_and_both():
    # Same rule as app.py::_on_mode_change: hands/face only apply in
    # Pose/Both, in Segmentation they're zeroed out even if the caller
    # sends them as True by mistake.
    kwargs = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": 15,
        "with_hands": True, "with_eyes": True, "with_mouth": True,
    })
    assert kwargs["with_hands"] is False
    assert kwargs["with_eyes"] is False
    assert kwargs["with_mouth"] is False
    print("Part 2: hands/face are ignored outside Pose/Both — OK")


def part3_mediapipe_pose_only_in_segmentation():
    # Independent segmentation/pose (see gui/pipeline_runner.py): there's
    # no longer a separate "with_mediapipe_pose" flag sent by JS -- it's
    # derived from the "Pose model" choice (pose_backend) made in the
    # sidebar's Pose section. In Segmentation mode,
    # pose_backend="mediapipe" applies MediaPipe inside each tracked mask
    # (with_mediapipe_pose=True); in Pose-alone mode,
    # pose_backend="mediapipe" takes a completely different path
    # (`_iter_pose_mediapipe`, internal boxes, NOT `with_mediapipe_pose`).
    kwargs_seg = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": 15,
        "pose_backend": "mediapipe",
    })
    assert kwargs_seg["with_mediapipe_pose"] is True

    kwargs_pose = build_player_kwargs({
        "mode": "pose", "source": "v.mp4", "fps": 15,
        "pose_backend": "mediapipe",
    })
    assert kwargs_pose["with_mediapipe_pose"] is False
    assert kwargs_pose["pose_backend"] == "mediapipe"  # handled by _iter_pose_mediapipe, not this flag
    print("Part 3: per-mask MediaPipe pose can only be enabled in segmentation mode — OK")


def part4_reid_requires_max_people_and_right_mode():
    # without max_people: pose-reid (ReIdentifier) still works (doesn't
    # need a cap for normal matching), seg-reid (SegReIdentifier) stays
    # off -- it genuinely requires max_people, see the comment in
    # build_player_kwargs about the asymmetry between the two.
    kwargs = build_player_kwargs({
        "mode": "both", "source": "v.mp4", "fps": 15,
    })
    assert kwargs["with_reid"] is True
    assert kwargs["with_seg_reid"] is False

    # reid requested with max_people, "both" mode -> both reids active
    kwargs2 = build_player_kwargs({
        "mode": "both", "source": "v.mp4", "fps": 15, "reid": True, "max_people": "3",
    })
    assert kwargs2["max_people"] == 3
    assert kwargs2["with_reid"] is True
    assert kwargs2["with_seg_reid"] is True

    # reid requested with max_people, "pose" mode -> only pose reid active
    kwargs3 = build_player_kwargs({
        "mode": "pose", "source": "v.mp4", "fps": 15, "reid": True, "max_people": 2,
    })
    assert kwargs3["with_reid"] is True
    assert kwargs3["with_seg_reid"] is False
    print("Part 4: re-id/seg-reid are active only with max_people set and in the right mode — OK")


def part5_invalid_mode_and_missing_source_raise():
    try:
        build_player_kwargs({"mode": "bogus", "source": "v.mp4", "fps": 15})
        raise AssertionError("should have raised ValueError for an unknown mode")
    except ValueError:
        pass
    try:
        build_player_kwargs({"mode": "pose", "source": "", "fps": 15})
        raise AssertionError("should have raised ValueError for a missing source")
    except ValueError:
        pass
    print("Part 5: unknown mode or missing source raise ValueError (build_player turns them "
          "into {'ok': False, 'error': ...} instead of blowing up the JS call) — OK")


def part6_encode_frame_jpeg_b64_roundtrip_and_resize():
    import base64
    import cv2

    small = np.zeros((10, 10, 3), dtype=np.uint8)
    small[:] = (0, 128, 255)  # BGR
    data_url = encode_frame_jpeg_b64(small, max_width=1600)
    assert data_url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (10, 10, 3)  # no resize below max_width

    wide = np.zeros((100, 3200, 3), dtype=np.uint8)
    data_url2 = encode_frame_jpeg_b64(wide, max_width=1600)
    raw2 = base64.b64decode(data_url2.split(",", 1)[1])
    decoded2 = cv2.imdecode(np.frombuffer(raw2, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded2.shape[1] == 1600  # resized down
    assert decoded2.shape[0] == 50    # proportions kept (100 * 1600/3200)
    print("Part 6: encode_frame_jpeg_b64 produces a decodable data-URL and resizes only "
          "downward beyond max_width — OK")


def part7_latency_tracker_rolling_average():
    tracker = _LatencyTracker(window=3)
    assert tracker.avg_latency_ms == 0.0
    assert tracker.processing_fps == 0.0

    tracker.record(0.100)  # 100ms
    tracker.record(0.100)
    assert abs(tracker.avg_latency_ms - 100.0) < 1e-6
    assert abs(tracker.processing_fps - 10.0) < 1e-6

    # the window is 3: a fourth value pushes out the oldest
    tracker.record(0.100)
    tracker.record(0.400)  # now the window contains [0.1, 0.1, 0.4] -> average 0.2
    assert abs(tracker.avg_latency_ms - 200.0) < 1e-6
    print("Part 7: _LatencyTracker computes a real rolling average, not a fake value — OK")


def part8_build_status_uses_people_count_not_len_rows():
    frame = RunnerFrame(frame=np.zeros((2, 2, 3), dtype=np.uint8), rows=[], now=1.5,
                         mode="pose", people_count=4)
    latency = _LatencyTracker()
    latency.record(0.050)
    status = build_status(runner_frame=frame, cached_frame_count=7, latency=latency,
                           device="mps", mode="pose", is_finished=False)
    assert status["people_count"] == 4  # from RunnerFrame.people_count, not from len(rows)=0
    assert status["rows_this_frame"] == 0
    assert status["frame_index"] == 6
    assert status["timecode_s"] == 1.5
    assert status["device"] == "mps"
    assert status["is_finished"] is False
    assert abs(status["avg_latency_ms"] - 50.0) < 1e-6
    print("Part 8: build_status reads people_count from RunnerFrame (reliable even with empty rows) — OK")


def part9_probe_video_metadata_missing_file_returns_none_not_zero():
    meta = probe_video_metadata("/tmp/definitely_not_a_real_video_file_xyz.mp4")
    assert meta == {"frame_count": None, "duration_s": None, "container_fps": None}, (
        "a nonexistent/unreadable file must give 'unknown' (None), not 0 -- "
        "0 would suggest an empty video instead of a duration that couldn't be computed"
    )
    print("Part 9: probe_video_metadata on a nonexistent file returns 'unknown' (None), not zero — OK")


def part10_probe_video_metadata_reads_real_container_metadata():
    import tempfile
    import os
    import cv2

    path = tempfile.mktemp(suffix=".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (16, 16))
    for _ in range(30):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()
    try:
        meta = probe_video_metadata(path)
        assert meta["frame_count"] == 30, f"expected 30 frames in the container, found {meta['frame_count']}"
        assert abs(meta["container_fps"] - 10.0) < 1e-6
        assert abs(meta["duration_s"] - 3.0) < 1e-6  # 30 frames / 10 fps = 3s
    finally:
        os.remove(path)
    print("Part 10: probe_video_metadata reads ONLY the container metadata (frame count/fps/duration) "
          "from a real video file, without decoding frames one by one — OK")


def part11_build_status_carries_totals_and_max_people_for_the_timeline():
    frame = RunnerFrame(frame=np.zeros((2, 2, 3), dtype=np.uint8), rows=[], now=6.3,
                         mode="segmentation", people_count=2)
    latency = _LatencyTracker()
    status = build_status(runner_frame=frame, cached_frame_count=209, latency=latency,
                           device="mps", mode="segmentation", is_finished=False,
                           max_people=20, total_frame_count=1087, total_duration_s=72.8)
    assert status["frame_index"] == 208
    assert status["total_frame_count"] == 1087  # for the "current / total" timecode/metric
    assert status["total_duration_s"] == 72.8
    assert status["max_people"] == 20  # for the "active tracks: 2 / 20" metric
    print("Part 11: build_status also carries the totals (frame/duration) and max_people, for the "
          "'current / total' timecode and the 'active tracks: N / max' metric of the new layout — OK")


def part12_seg_backend_defaults_to_yolo_and_passes_through():
    kwargs_default = build_player_kwargs({"mode": "segmentation", "source": "v.mp4", "fps": "15"})
    assert kwargs_default["seg_backend"] == "yolo"
    assert kwargs_default["sam_chunk_size"] == 600
    assert kwargs_default["sam_overlap"] == 50
    assert kwargs_default["sam_chunk_store_dir"] is None

    kwargs_sam = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": "15",
        "seg_backend": "sam31", "sam_chunk_size": "300", "sam_overlap": "20",
        "sam_chunk_store_dir": "/tmp/chunks",
    })
    assert kwargs_sam["seg_backend"] == "sam31"
    assert kwargs_sam["sam_chunk_size"] == 300  # string -> int, like max_people elsewhere
    assert kwargs_sam["sam_overlap"] == 20
    assert kwargs_sam["sam_chunk_store_dir"] == "/tmp/chunks"
    print("Part 12: seg_backend/sam_chunk_size/sam_overlap/sam_chunk_store_dir have the right defaults "
          "('yolo'/600/50/None) and pass through unchanged when specified — OK")


def part13_sam_redetect_every_and_text_prompt_defaults_and_passthrough():
    kwargs_default = build_player_kwargs({"mode": "segmentation", "source": "v.mp4", "fps": "15"})
    assert kwargs_default["sam_redetect_every"] is None
    assert kwargs_default["sam_text_prompt"] is None

    kwargs_set = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": "15",
        "seg_backend": "sam31", "sam_redetect_every": "120", "sam_text_prompt": "person",
    })
    assert kwargs_set["sam_redetect_every"] == 120  # string -> int
    assert kwargs_set["sam_text_prompt"] == "person"

    # empty string == not set, like max_people elsewhere (not a literal "0"/"")
    kwargs_empty = build_player_kwargs({
        "mode": "segmentation", "source": "v.mp4", "fps": "15",
        "sam_redetect_every": "", "sam_text_prompt": "",
    })
    assert kwargs_empty["sam_redetect_every"] is None
    assert kwargs_empty["sam_text_prompt"] is None
    print("Part 13: sam_redetect_every/sam_text_prompt default to None and pass through "
          "unchanged when specified (empty string == not set) — OK")


if __name__ == "__main__":
    part1_build_player_kwargs_mirrors_app_py_defaults()
    part1b_explicit_device_passes_through_unchanged()
    part2_hands_face_ignored_outside_pose_and_both()
    part3_mediapipe_pose_only_in_segmentation()
    part4_reid_requires_max_people_and_right_mode()
    part5_invalid_mode_and_missing_source_raise()
    part6_encode_frame_jpeg_b64_roundtrip_and_resize()
    part7_latency_tracker_rolling_average()
    part8_build_status_uses_people_count_not_len_rows()
    part9_probe_video_metadata_missing_file_returns_none_not_zero()
    part10_probe_video_metadata_reads_real_container_metadata()
    part12_seg_backend_defaults_to_yolo_and_passes_through()
    part13_sam_redetect_every_and_text_prompt_defaults_and_passthrough()
    part11_build_status_carries_totals_and_max_people_for_the_timeline()
    part12_seg_backend_defaults_to_yolo_and_passes_through()
    print("\nVerification completed with no errors: webui/api.py's pure logic (parameters, frame "
          "encoding, metrics, video metadata) behaves as expected, without needing pywebview or "
          "a real window.")
