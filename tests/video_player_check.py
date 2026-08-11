"""
video_player_check.py
=======================
Verifies the cache/seek logic in `gui/video_player.py` WITHOUT a GUI, real
video, or tracker (no Tkinter, no ultralytics): injects a synthetic
generator that produces numbered `RunnerFrame`s, so we can check exactly
which frame is returned by each Forward/Back and how many times the
generator is actually advanced (to prove that "Back" NEVER triggers new
inference).

Run with: python video_player_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from gui.pipeline_runner import RunnerFrame
from gui.video_player import VideoPlayer


def make_generator_factory(n_frames: int, counter: dict):
    """Factory of synthetic generators: each frame is a small array with a
    single pixel encoding its own index, so tests can check "which frame
    is this" without a real video. `counter["calls"]` counts how many
    times the factory was invoked (how many times VideoPlayer "started
    over from scratch") and `counter["frames_produced"]` counts how many
    frames were actually generated (indirect proof that the cache avoids
    regenerating them)."""
    def factory():
        counter["calls"] = counter.get("calls", 0) + 1
        def gen():
            for i in range(n_frames):
                counter["frames_produced"] = counter.get("frames_produced", 0) + 1
                frame = np.full((2, 2, 3), i, dtype=np.uint8)
                yield RunnerFrame(frame=frame, rows=[{"frame_idx": i}], now=float(i), mode="test")
        return gen()
    return factory


def frame_index_of(runner_frame: RunnerFrame) -> int:
    return int(runner_frame.frame[0, 0, 0])


def part1_forward_advances_and_produces_new_frames():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))

    seen = [frame_index_of(player.step_forward()) for _ in range(5)]
    assert seen == [0, 1, 2, 3, 4], f"expected [0..4] in order, found {seen}"
    assert counter["frames_produced"] == 5, "expected exactly 5 frames generated (one per step_forward)"
    # exhaustion is only discovered by attempting to go PAST the last frame
    # (the generator itself doesn't know it's at the last element until
    # asked for the next one) -- before this call is_exhausted must
    # therefore still be False.
    assert not player.is_exhausted, "must not report exhausted before attempting past the last frame"
    assert player.step_forward() is None, "past the end must return None, not raise or restart"
    assert player.is_exhausted, "after attempting past the last frame the player must report exhausted"
    print("Part 1: 5x step_forward() returns frames 0..4 in the right order, "
          "exactly 5 frames generated, end-of-video correctly detected — OK")


def part2_back_never_reprocesses():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))

    for _ in range(3):
        player.step_forward()  # cursor at 0,1,2 -> frames 0,1,2 in cache
    assert counter["frames_produced"] == 3

    back1 = frame_index_of(player.step_back())
    back2 = frame_index_of(player.step_back())
    assert (back1, back2) == (1, 0), f"expected (1,0) going back from 2, found {(back1, back2)}"
    assert counter["frames_produced"] == 3, (
        "step_back() must NEVER generate new frames (must only read from the cache): "
        f"expected still 3 frames produced, found {counter['frames_produced']}"
    )
    assert player.step_back() is None, "cannot go back further from frame 0"
    print("Part 2: step_back() reads from the cache (0 new frames generated) and stops "
          "correctly at the first frame — OK")


def part3_forward_after_back_resumes_from_cache_then_live():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))

    for _ in range(4):
        player.step_forward()  # cache: 0,1,2,3 -- cursor at 3
    player.step_back()          # cursor at 2 (reads from cache)
    player.step_back()          # cursor at 1 (reads from cache)
    assert counter["frames_produced"] == 4

    # from here on, "forward" must read from the cache (2, then 3) BEFORE
    # going back to generating new frames (4) -- must not regenerate 2 and 3.
    resumed = [frame_index_of(player.step_forward()) for _ in range(3)]
    assert resumed == [2, 3, 4], f"expected [2,3,4] resuming from cursor=1, found {resumed}"
    assert counter["frames_produced"] == 5, (
        "only frame 4 (never seen before) should have triggered new inference: "
        f"expected 5 frames produced in total, found {counter['frames_produced']}"
    )
    print("Part 3: after going back, forward reads from the cache first (2,3) "
          "then resumes live processing only for the truly new frame (4) — OK")


def part4_reset_starts_a_fresh_generator():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))
    player.step_forward()
    player.step_forward()
    assert counter["calls"] == 1

    player.reset()
    assert player.current is None, "after reset() there must be no frame 'in view'"
    assert player.cached_frame_count == 0, "after reset() the cache must be empty"

    first_after_reset = frame_index_of(player.step_forward())
    assert first_after_reset == 0, "after reset() must restart from frame 0 of a NEW generator"
    assert counter["calls"] == 2, "reset() must invoke the generator_factory again (new session)"
    print("Part 4: reset() empties the cache and starts over from a brand-new generator — OK")


def part5_all_rows_matches_cached_frames_in_order():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(4, counter))
    for _ in range(4):
        player.step_forward()

    rows = player.all_rows()
    assert [r["frame_idx"] for r in rows] == [0, 1, 2, 3], (
        f"expected one row per frame in order 0..3, found {[r['frame_idx'] for r in rows]}"
    )
    print("Part 5: all_rows() concatenates the rows of every cached frame, in the right order — OK")


def part5b_all_frames_matches_cached_frames_in_order():
    # Mirrors part5, for all_frames() (added 2026-08 for
    # webui/api.py::Api.export_video -- saving the annotated video at
    # the end of a run, Michele).
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(4, counter))
    for _ in range(4):
        player.step_forward()

    frames = player.all_frames()  # raw ndarrays (RunnerFrame.frame), not RunnerFrame objects
    indices = [int(f[0, 0, 0]) for f in frames]
    assert indices == [0, 1, 2, 3], f"expected one frame per cached entry in order 0..3, found {indices}"
    print("Part 5b: all_frames() returns every cached (already-annotated) frame, in the right order — OK")


def part6_seek_jumps_within_cache_without_reprocessing():
    counter: dict = {}
    player = VideoPlayer(generator_factory=make_generator_factory(5, counter))
    for _ in range(4):
        player.step_forward()  # cache: 0,1,2,3 -- cursor at 3
    assert counter["frames_produced"] == 4

    jumped = frame_index_of(player.seek(1))
    assert jumped == 1, f"expected to jump to frame 1, found {jumped}"
    assert player.cursor == 1
    assert counter["frames_produced"] == 4, "seek() within the cache must not generate new frames"

    back_to_edge = frame_index_of(player.seek(3))
    assert back_to_edge == 3
    assert counter["frames_produced"] == 4

    # outside the cache (frame 4 not yet processed): seek() must not move
    # the cursor nor generate anything -- it's the caller's job to use
    # step_forward() for catch-up processing.
    assert player.seek(4) is None
    assert player.cursor == 3, "a seek() past the cache must not move the cursor"
    assert counter["frames_produced"] == 4
    print("Part 6: seek() jumps instantly within the already-processed cache, without ever "
          "generating new frames, and rejects a jump past the already-processed prefix — OK")


def main():
    part1_forward_advances_and_produces_new_frames()
    part2_back_never_reprocesses()
    part3_forward_after_back_resumes_from_cache_then_live()
    part4_reset_starts_a_fresh_generator()
    part5_all_rows_matches_cached_frames_in_order()
    part5b_all_frames_matches_cached_frames_in_order()
    part6_seek_jumps_within_cache_without_reprocessing()
    print("\nVerification completed with no errors: VideoPlayer advances/generates new frames only "
          "when truly needed, never regenerates an already-seen frame when going back, and "
          "reset() starts over cleanly from a new generator.")


if __name__ == "__main__":
    main()
