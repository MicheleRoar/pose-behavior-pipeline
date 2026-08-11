"""
video_player.py
=================
State of the "player" used by app.py: stepping forward/back through
already-processed frames, resuming live processing when past the last
cached frame.

Why a cache instead of true "seeking": trackers (ByteTrack, and
optionally reid.py/seg_reid.py) are inherently sequential and stateful
-- you can't jump to an arbitrary frame in the middle of a video without
reprocessing everything from scratch, otherwise the history that re-id
and sliding feature windows rely on would be lost. So:
- "Back" re-reads from the already-computed cache: instant, no new
  inference;
- "Forward" past the last cached frame resumes live processing from the
  `iter_pipeline_frames()` generator (see pipeline_runner.py), appending
  each new frame to the cache as it's produced.

`VideoPlayer` knows nothing about Tkinter/OpenCV/drawing: it receives a
`generator_factory` (typically `lambda: iter_pipeline_frames(**kwargs)`)
and returns already-ready `RunnerFrame` objects (overlay already drawn).
This makes it testable without a real video, see
`tests/video_player_check.py`, which injects a synthetic generator.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

from gui.pipeline_runner import RunnerFrame


class VideoPlayer:
    def __init__(self, generator_factory: Callable[[], Iterator[RunnerFrame]]):
        """`generator_factory` is called to create a NEW generator every
        time we need to start over (first call to step_forward()/
        step_back(), or after reset()) -- not an already-instantiated
        generator, because an exhausted generator can't be "rewound"."""
        self._generator_factory = generator_factory
        self._generator: Optional[Iterator[RunnerFrame]] = None
        self._cache: list[RunnerFrame] = []
        self._cursor: int = -1  # index in the cache of the frame currently shown
        self._exhausted: bool = False  # the generator has run out of frames (end of video)

    @property
    def current(self) -> Optional[RunnerFrame]:
        """The frame currently "in view" (the last one returned by
        step_forward()/step_back()), or None if no frame has been shown
        yet."""
        if 0 <= self._cursor < len(self._cache):
            return self._cache[self._cursor]
        return None

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def cached_frame_count(self) -> int:
        return len(self._cache)

    @property
    def at_live_edge(self) -> bool:
        """True if the cursor is on the last already-processed frame: the
        next step_forward() will require new inference, not just the
        cache."""
        return self._cursor == len(self._cache) - 1

    @property
    def is_exhausted(self) -> bool:
        """True if the video has ended (the generator stopped producing
        frames) AND we've reached the end of the cache."""
        return self._exhausted and self.at_live_edge

    def step_forward(self) -> Optional[RunnerFrame]:
        """Advances by one frame: from the cache if already available
        (instant), otherwise processes the next frame live. Returns None
        if the video is already fully exhausted."""
        if self._cursor + 1 < len(self._cache):
            self._cursor += 1
            return self._cache[self._cursor]

        if self._exhausted:
            return None

        if self._generator is None:
            self._generator = self._generator_factory()

        try:
            frame = next(self._generator)
        except StopIteration:
            self._exhausted = True
            return None

        self._cache.append(frame)
        self._cursor += 1
        return frame

    def step_back(self) -> Optional[RunnerFrame]:
        """Goes back one frame, always from the cache. Returns None if
        already at the first frame (or if nothing has been shown yet)."""
        if self._cursor <= 0:
            return None
        self._cursor -= 1
        return self._cache[self._cursor]

    def seek(self, index: int) -> Optional[RunnerFrame]:
        """Jumps INSTANTLY to `index` WITHIN the already-processed cache
        (no new inference) -- used by the web GUI's timeline scrubber,
        which for the same reason explained in the module docstring can
        only offer free seeking over the already-processed prefix.
        Returns None (without touching the cursor) if `index` is outside
        the cache: unlike step_forward(), this method NEVER extends the
        cache -- the caller must use step_forward() in sequence to reach
        a point not yet processed (catch-up processing), not this
        method."""
        if 0 <= index < len(self._cache):
            self._cursor = index
            return self._cache[self._cursor]
        return None

    def all_rows(self) -> list[dict]:
        """All data rows (one per person per frame) accumulated so far in
        the cache, in frame order -- used for the final CSV."""
        rows: list[dict] = []
        for f in self._cache:
            rows.extend(f.rows)
        return rows

    def all_frames(self):
        """All annotated frames (overlay already drawn, see
        `RunnerFrame`) accumulated so far in the cache, in frame order
        -- mirrors `all_rows()`, used to export the analyzed video (see
        `webui/api.py::Api.export_video`) so different runs/parameter
        choices can be compared side by side outside the app, not just
        live in the player."""
        return [f.frame for f in self._cache]

    def reset(self) -> None:
        """Starts over: new inference session, cache cleared. Necessary
        whenever the parameters change (model, features, --max-people,
        etc.), because a tracker already started can't be reconfigured
        halfway through -- see the module docstring."""
        self._generator = None
        self._cache = []
        self._cursor = -1
        self._exhausted = False
