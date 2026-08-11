"""
identity_gallery.py
=====================
Appearance-based fallback for SAM chunk-boundary reconciliation
(`segmentation/chunking.py`, `sam_backend.py`, `sam31_estimation.py`):
when a person can't be matched geometrically to anyone already known --
either `chunking.reconcile_ids_windowed()` found no match above
`iou_threshold`, or it's a brand-new YOLO/text-prompt detection in
box mode that doesn't overlap any currently-tracked box -- this is the
last chance to avoid minting a new global id for someone who's actually
been seen before (occluded across a whole chunk, walked off and back
into frame, ...), by comparing an OSNet embedding of the crop against a
short-term memory of recently "lost" identities.

Born from the same request that produced `pose/appearance_embedding.py`
("ids must not change, people must stay in memory to be re-associated
on return") -- deliberately REUSED here (`OSNetEmbedder`, `ema_update`,
`embedding_similarity`), not reimplemented, but kept in its OWN id
space: segmentation global ids (`chunking.GlobalIdAllocator`) are a
different numbering from the pose pipeline's person ids, on purpose
(see project memory: decoupled seg/pose GUI) -- this class only avoids
avoidable id churn INSIDE the segmentation backend's own chunk
reconciliation; it doesn't try to unify the two id spaces.

Why a short-term "lost" gallery, not "match against everyone always"
------------------------------------------------------------------------
Matching a new detection against every embedding ever seen (including
people CURRENTLY tracked elsewhere in the frame) risks merging two
different people who simply look similar (same OSNet failure mode as
in `reid.py`/`identity_manager.py`, which is why those already only
match against `self.lost`, not the active set). Here the same
discipline applies: `match_or_none()` only ever compares against ids
explicitly marked `mark_lost()` -- an id that's still being matched
geometrically every chunk is never a candidate, no matter how similar
its embedding might look to a new detection.

Optional, like OSNet everywhere else in the project: if torch/torchreid
aren't installed, construction catches the `ImportError` and the
gallery becomes a no-op (`enabled=False`, every `match_or_none()` call
returns `None`) -- identical behavior to before this feature existed,
no crash, no new hard dependency for anyone not using SAM backends'
appearance fallback (see requirements.txt for the same treatment of
torch/torchreid/SAM 3.1/SAM2).
"""

from __future__ import annotations

import numpy as np

from pose.appearance_embedding import OSNetEmbedder, ema_update, embedding_similarity

# Cosine-based similarity in [0, 1] (see appearance_embedding.embedding_similarity).
# Deliberately fairly high: a WRONG appearance-based reuse (silently
# merging two different people under one id) is a worse failure than
# falling back to a new id, which is recoverable -- same "uncertain
# means don't merge" philosophy as identity_manager.py.
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# After this many chunks without being re-matched, a lost identity is
# forgotten -- bounds memory on long videos and avoids matching someone
# against a person who left the scene for good many minutes ago.
DEFAULT_MAX_LOST_AGE_CHUNKS = 10


class SegmentationIdentityGallery:
    """Tracks one EMA-updated appearance embedding per segmentation
    global id, and offers `match_or_none()` to try re-identifying an
    otherwise-unmatched detection against RECENTLY LOST ids before the
    caller mints a brand-new one (`GlobalIdAllocator.next_id()`).

    Typical per-chunk usage (see `sam_backend.py::run()`):
        gallery.observe(global_id, anchor_frame, box, poly=poly)  # for
            # every id still active at the new anchor frame
        gallery.mark_lost(global_id, chunk_index)  # for every id that
            # WAS known but didn't survive into this chunk
        gallery.forget_stale(chunk_index)
        ...
        matched = gallery.match_or_none(frame, box)
        if matched is not None:
            gallery.revive(matched)
            global_id = matched
        else:
            global_id = allocator.next_id()
    """

    def __init__(self, *, device: str = "cpu",
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 max_lost_age_chunks: int = DEFAULT_MAX_LOST_AGE_CHUNKS,
                 enabled: bool = True, embedder: OSNetEmbedder | None = None):
        self.similarity_threshold = similarity_threshold
        self.max_lost_age_chunks = max_lost_age_chunks
        self._embeddings: dict[int, np.ndarray] = {}  # global_id -> EMA embedding (active OR lost)
        self._lost_since_chunk: dict[int, int] = {}  # global_id -> chunk_index it was lost at
        if embedder is not None:
            # Dependency injection for tests (see
            # tests/identity_gallery_check.py): a fake embedder with a
            # deterministic embed() lets the bookkeeping logic
            # (observe/mark_lost/forget_stale/match_or_none/revive) be
            # verified without torch/torchreid installed.
            self._embedder = embedder
            self.enabled = enabled
            return
        self.enabled = False
        self._embedder = None
        if enabled:
            try:
                self._embedder = OSNetEmbedder(device=device)
                self.enabled = True
            except ImportError as exc:
                print(f"[SegmentationIdentityGallery] appearance fallback disabled: {exc}")

    def observe(self, global_id: int, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
                poly: np.ndarray | None = None) -> None:
        """Updates (EMA) the stored embedding for a person who's
        currently confidently tracked -- call this whenever a global id
        is seen at a chunk anchor, so the gallery reflects their most
        recent look by the time they might get lost. Also clears any
        stale `mark_lost()` bookkeeping for this id (being observed
        again means it's active, not lost)."""
        if not self.enabled:
            return
        vec = self._embedder.embed(frame_bgr, bbox_xyxy, poly=poly)
        if vec is not None:
            self._embeddings[global_id] = ema_update(self._embeddings.get(global_id), vec)
        self._lost_since_chunk.pop(global_id, None)

    def mark_lost(self, global_id: int, chunk_index: int) -> None:
        """Moves an id from 'active' to 'lost' bookkeeping (its
        embedding, if any, is kept so it can still be matched against
        later) -- call this for any global id that WAS known but didn't
        survive reconciliation into the new chunk. A no-op if this id
        was never `observe()`-d (no embedding to match against anyway)."""
        if global_id in self._embeddings:
            self._lost_since_chunk[global_id] = chunk_index

    def forget_stale(self, chunk_index: int) -> None:
        """Drops lost ids older than `max_lost_age_chunks` (bounded
        memory on a long video) -- call once per chunk."""
        stale = [gid for gid, since in self._lost_since_chunk.items()
                 if chunk_index - since > self.max_lost_age_chunks]
        for gid in stale:
            self._lost_since_chunk.pop(gid, None)
            self._embeddings.pop(gid, None)

    def match_or_none(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
                       poly: np.ndarray | None = None) -> int | None:
        """Best LOST global id whose embedding is similar enough to the
        crop at `bbox_xyxy`, or `None` (no confident match -- the caller
        should mint a new id). Never matches against a still-ACTIVE id
        (see the module docstring for why) -- only ids currently in the
        `mark_lost()` bookkeeping are candidates."""
        if not self.enabled or not self._lost_since_chunk:
            return None
        vec = self._embedder.embed(frame_bgr, bbox_xyxy, poly=poly)
        if vec is None:
            return None
        best_id: int | None = None
        best_score = self.similarity_threshold
        for gid in self._lost_since_chunk:
            score = embedding_similarity(self._embeddings.get(gid), vec)
            if score is not None and score >= best_score:
                best_id, best_score = gid, score
        return best_id

    def revive(self, global_id: int) -> None:
        """Call after `match_or_none()` returns a hit and the caller
        decides to reuse that id: moves it back to 'active' bookkeeping.
        The embedding itself is left untouched (still gets refined by
        future `observe()` calls)."""
        self._lost_since_chunk.pop(global_id, None)
