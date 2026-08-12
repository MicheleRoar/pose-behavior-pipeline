"""
psifx_eval/id_metrics.py
==========================
Quantifies psifx's cross-chunk identity persistence by comparing a
CHUNKED run (real psifx, `Sam3TrackingTool.infer()` with its normal
`chunk_size` -- i.e. exactly what CHUV runs in production) against a
CONTINUOUS "oracle" run of the SAME video (same tool, `chunk_size` set
large enough that the whole video is one chunk -- see
`run_baseline_vs_oracle.py`). The oracle isn't ground truth in the
human-annotation sense; it's SAM3 tracking the same footage with the
chunk-boundary re-linking step never triggered at all, which isolates
exactly the failure mode Michele was asked to fix: does breaking the
video into chunks and re-linking IDs at the seams introduce identity
errors BEYOND whatever SAM3 itself would do in one continuous pass.

Approach (per-frame IoU correspondence, not a single global comparison):
for every frame, the oracle's identities and the baseline's identities
are matched by mask IoU (Hungarian, i.e. the best possible one-to-one
assignment for THAT single frame -- a different problem from psifx's
own chunk-boundary matching, which only looks at one frame on each
side and is greedy, not optimal). Walking that per-frame correspondence
across the whole video reveals, per real (oracle) person, every DISTINCT
baseline id that ever stood in for them (fragmentation), and per
baseline id, every DISTINCT real person it was ever matched to (an
incorrect merge / identity swap -- the more serious error, since it
silently conflates two different people). Each such event is classified
`cross_chunk` or not by checking whether the old and new evidence
straddle a real chunk boundary (known exactly from `chunk_size`, since
this is OUR run, not a black box) -- an event entirely inside one chunk
is a native SAM3 tracking failure (would happen in the oracle's own
single-session tracking too), not something chunk-stitching caused.

`boundary_checks` is the most direct measurement of psifx's own
stitching decisions specifically: for every real chunk boundary, and
every oracle person visible on both sides of it, was the SAME baseline
id used before and after? This is literally asking "did
`_map_chunk_object_ids`'s single-frame greedy IoU match get it right,
this time, for this person, at this seam" -- the number CHUV's fix
needs to move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Same formula as real psifx's `Sam3TrackingTool._compute_mask_iou`
    (psifx/video/tracking/sam3/tool.py) -- reimplemented here (not
    imported) so this module works even when the `psifx` package isn't
    installed, e.g. re-analyzing a MaskDir someone already produced
    elsewhere."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def match_frame_by_iou(
    oracle_frame: dict[int, np.ndarray],
    baseline_frame: dict[int, np.ndarray],
    iou_threshold: float = 0.1,
) -> dict[int, int | None]:
    """Optimal (Hungarian) one-to-one matching between oracle and
    baseline identities present at a SINGLE frame, by mask IoU. This is
    NOT a reproduction of psifx's own chunk-boundary matching (which is
    greedy and only ever looks at one frame per side, see the module
    docstring) -- here we already know both masks belong to the same
    frame of the same video, so the only question is "which oracle
    person does this baseline mask correspond to," a much simpler,
    well-posed problem where using the optimal assignment is correct
    and cheap (per-frame person counts are small).

    Returns `{oracle_id: baseline_id_or_None}` -- `None` means no
    baseline identity overlapped this oracle identity above
    `iou_threshold` at this frame (i.e. psifx lost/never found them
    here)."""
    oracle_ids = list(oracle_frame.keys())
    baseline_ids = list(baseline_frame.keys())
    result: dict[int, int | None] = {oid: None for oid in oracle_ids}
    if not oracle_ids or not baseline_ids:
        return result

    cost = np.ones((len(oracle_ids), len(baseline_ids)), dtype=float)
    for i, oid in enumerate(oracle_ids):
        for j, bid in enumerate(baseline_ids):
            cost[i, j] = 1.0 - mask_iou(oracle_frame[oid], baseline_frame[bid])

    row_idx, col_idx = linear_sum_assignment(cost)
    for i, j in zip(row_idx, col_idx):
        iou = 1.0 - cost[i, j]
        if iou >= iou_threshold:
            result[oracle_ids[i]] = baseline_ids[j]
    return result


def build_correspondence(
    oracle: dict[int, np.ndarray],
    baseline: dict[int, np.ndarray],
    iou_threshold: float = 0.1,
) -> list[dict[int, int | None]]:
    """Per-frame `oracle_id -> baseline_id` mapping across the whole
    video. `oracle`/`baseline` are `{id: (T, H, W) bool}`, as returned
    by `mask_io.load_mask_dir` -- both MUST be the same source video at
    the same frame count (raises `ValueError` otherwise: comparing runs
    of different videos, or a truncated run against a full one, would
    silently produce meaningless metrics)."""
    oracle_lengths = {arr.shape[0] for arr in oracle.values()}
    baseline_lengths = {arr.shape[0] for arr in baseline.values()}
    if len(oracle_lengths) != 1:
        raise ValueError(f"oracle masks have inconsistent frame counts: {oracle_lengths}")
    if len(baseline_lengths) != 1:
        raise ValueError(f"baseline masks have inconsistent frame counts: {baseline_lengths}")
    total_frames = oracle_lengths.pop()
    baseline_total = baseline_lengths.pop()
    if total_frames != baseline_total:
        raise ValueError(
            f"oracle ({total_frames} frames) and baseline ({baseline_total} frames) "
            f"must be runs of the SAME video -- can't compare mismatched lengths."
        )

    correspondence: list[dict[int, int | None]] = []
    for t in range(total_frames):
        oracle_frame = {oid: arr[t] for oid, arr in oracle.items() if arr[t].any()}
        baseline_frame = {bid: arr[t] for bid, arr in baseline.items() if arr[t].any()}
        correspondence.append(match_frame_by_iou(oracle_frame, baseline_frame, iou_threshold))
    return correspondence


@dataclass
class SwitchEvent:
    """One 'a brand-new id took over' event. For fragmentation events,
    `entity_id` is the oracle (real person) id and `from_id`/`to_id` are
    baseline ids; for swap events it's the reverse (`entity_id` is the
    baseline id, `from_id`/`to_id` are oracle/real-person ids)."""
    entity_id: int
    from_id: int | None
    to_id: int
    last_seen_frame: int  # last frame the OLD id was matched
    new_seen_frame: int   # first frame the NEW id was matched
    cross_chunk: bool     # True if last_seen_frame and new_seen_frame are in different chunks


@dataclass
class BoundaryCheck:
    """Did the SAME baseline id represent this oracle person immediately
    before and immediately after one real chunk boundary. This is the
    most direct measurement of psifx's own stitching decision quality."""
    oracle_id: int
    boundary_frame: int
    before_baseline_id: int | None
    after_baseline_id: int | None

    @property
    def correct(self) -> bool:
        return (
            self.before_baseline_id is not None
            and self.before_baseline_id == self.after_baseline_id
        )


@dataclass
class IdPersistenceReport:
    total_frames: int
    chunk_size: int
    fragmentation_events: list[SwitchEvent] = field(default_factory=list)
    swap_events: list[SwitchEvent] = field(default_factory=list)
    boundary_checks: list[BoundaryCheck] = field(default_factory=list)

    @property
    def fragmentation_count(self) -> int:
        return len(self.fragmentation_events)

    @property
    def swap_count(self) -> int:
        return len(self.swap_events)

    @property
    def cross_chunk_fragmentation_count(self) -> int:
        return sum(1 for e in self.fragmentation_events if e.cross_chunk)

    @property
    def intra_chunk_fragmentation_count(self) -> int:
        return sum(1 for e in self.fragmentation_events if not e.cross_chunk)

    @property
    def cross_chunk_swap_count(self) -> int:
        return sum(1 for e in self.swap_events if e.cross_chunk)

    @property
    def intra_chunk_swap_count(self) -> int:
        return sum(1 for e in self.swap_events if not e.cross_chunk)

    @property
    def correct_boundary_matches(self) -> int:
        return sum(1 for b in self.boundary_checks if b.correct)

    @property
    def total_boundaries_checked(self) -> int:
        return len(self.boundary_checks)

    @property
    def boundary_accuracy(self) -> float | None:
        if not self.boundary_checks:
            return None
        return self.correct_boundary_matches / len(self.boundary_checks)

    def summary(self) -> str:
        acc = self.boundary_accuracy
        acc_str = f"{acc:.1%}" if acc is not None else "n/a (no boundaries with a visible person on both sides)"
        return (
            f"IdPersistenceReport ({self.total_frames} frames, chunk_size={self.chunk_size})\n"
            f"  Fragmentation events (1 real person -> multiple ids): {self.fragmentation_count} "
            f"({self.cross_chunk_fragmentation_count} cross-chunk, {self.intra_chunk_fragmentation_count} intra-chunk)\n"
            f"  Incorrect assignments / swaps (1 id -> multiple real people): {self.swap_count} "
            f"({self.cross_chunk_swap_count} cross-chunk, {self.intra_chunk_swap_count} intra-chunk)\n"
            f"  Chunk boundary accuracy: {self.correct_boundary_matches}/{self.total_boundaries_checked} correct ({acc_str})"
        )

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "chunk_size": self.chunk_size,
            "fragmentation_count": self.fragmentation_count,
            "cross_chunk_fragmentation_count": self.cross_chunk_fragmentation_count,
            "intra_chunk_fragmentation_count": self.intra_chunk_fragmentation_count,
            "swap_count": self.swap_count,
            "cross_chunk_swap_count": self.cross_chunk_swap_count,
            "intra_chunk_swap_count": self.intra_chunk_swap_count,
            "correct_boundary_matches": self.correct_boundary_matches,
            "total_boundaries_checked": self.total_boundaries_checked,
            "boundary_accuracy": self.boundary_accuracy,
            "fragmentation_events": [vars(e) for e in self.fragmentation_events],
            "swap_events": [vars(e) for e in self.swap_events],
            "boundary_checks": [
                {**vars(b), "correct": b.correct} for b in self.boundary_checks
            ],
        }


def _chunk_index(frame_idx: int, chunk_size: int) -> int:
    return frame_idx // chunk_size


def _build_entity_tracks(
    correspondence: list[dict[int, int | None]], *, key_is_oracle: bool
) -> dict[int, list[tuple[int, int]]]:
    """`key_is_oracle=True` -> `{oracle_id: [(frame, baseline_id), ...]}`
    (for fragmentation); `key_is_oracle=False` -> `{baseline_id:
    [(frame, oracle_id), ...]}` (for swaps). Frames where the match was
    `None` (lost) are dropped -- a temporary gap isn't itself an event,
    only what happens on either side of it matters, which the switch
    detection below already handles correctly using whatever the
    nearest surrounding matched frames were."""
    tracks: dict[int, list[tuple[int, int]]] = {}
    for frame_idx, frame_map in enumerate(correspondence):
        for oid, bid in frame_map.items():
            if bid is None:
                continue
            key, other = (oid, bid) if key_is_oracle else (bid, oid)
            tracks.setdefault(key, []).append((frame_idx, other))
    return tracks


def _find_switch_events(
    tracks: dict[int, list[tuple[int, int]]], chunk_size: int
) -> list[SwitchEvent]:
    events: list[SwitchEvent] = []
    for entity_id, seq in tracks.items():
        seen: set[int] = set()
        current_other_id: int | None = None
        last_frame_of_current: int | None = None
        for frame_idx, other_id in seq:
            if other_id != current_other_id:
                if current_other_id is not None and other_id not in seen:
                    events.append(SwitchEvent(
                        entity_id=entity_id,
                        from_id=current_other_id,
                        to_id=other_id,
                        last_seen_frame=last_frame_of_current,
                        new_seen_frame=frame_idx,
                        cross_chunk=(
                            _chunk_index(last_frame_of_current, chunk_size)
                            != _chunk_index(frame_idx, chunk_size)
                        ),
                    ))
                current_other_id = other_id
            seen.add(other_id)
            last_frame_of_current = frame_idx
    return events


def _iter_chunk_boundaries(total_frames: int, chunk_size: int):
    n_chunks = (total_frames + chunk_size - 1) // chunk_size
    for k in range(1, n_chunks):
        yield k * chunk_size


def _compute_boundary_checks(
    tracks_by_oracle: dict[int, list[tuple[int, int]]],
    total_frames: int,
    chunk_size: int,
) -> list[BoundaryCheck]:
    checks: list[BoundaryCheck] = []
    for boundary in _iter_chunk_boundaries(total_frames, chunk_size):
        for oracle_id, seq in tracks_by_oracle.items():
            before: tuple[int, int] | None = None
            after: tuple[int, int] | None = None
            for frame_idx, baseline_id in seq:
                if frame_idx < boundary:
                    before = (frame_idx, baseline_id)
                elif after is None:
                    after = (frame_idx, baseline_id)
                    break
            if before is None or after is None:
                continue  # this person isn't visible on both sides of this boundary
            # Only a meaningful check of THIS boundary if the nearest evidence on
            # each side is within one chunk of it -- otherwise "before" could be
            # from many chunks earlier (a long absence), which tests something
            # else entirely (re-acquisition after a long gap, not this seam).
            if boundary - before[0] > chunk_size or after[0] - boundary > chunk_size:
                continue
            checks.append(BoundaryCheck(
                oracle_id=oracle_id,
                boundary_frame=boundary,
                before_baseline_id=before[1],
                after_baseline_id=after[1],
            ))
    return checks


def compute_metrics(
    oracle: dict[int, np.ndarray],
    baseline: dict[int, np.ndarray],
    chunk_size: int,
    iou_threshold: float = 0.1,
) -> IdPersistenceReport:
    """The main entry point: `oracle`/`baseline` are `{id: (T, H, W)
    bool}` dicts (see `mask_io.load_mask_dir`), `chunk_size` MUST be the
    real chunk size the baseline run used (needed to classify every
    event as cross-chunk vs intra-chunk, see the module docstring)."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    correspondence = build_correspondence(oracle, baseline, iou_threshold)
    total_frames = len(correspondence)

    tracks_by_oracle = _build_entity_tracks(correspondence, key_is_oracle=True)
    tracks_by_baseline = _build_entity_tracks(correspondence, key_is_oracle=False)

    return IdPersistenceReport(
        total_frames=total_frames,
        chunk_size=chunk_size,
        fragmentation_events=_find_switch_events(tracks_by_oracle, chunk_size),
        swap_events=_find_switch_events(tracks_by_baseline, chunk_size),
        boundary_checks=_compute_boundary_checks(tracks_by_oracle, total_frames, chunk_size),
    )
