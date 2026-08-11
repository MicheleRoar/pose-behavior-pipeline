"""
identity_gallery_check.py
============================
Verifies `segmentation/identity_gallery.py::SegmentationIdentityGallery`
with a FAKE embedder (no torch/torchreid dependency, not installed in
this environment -- same philosophy as the other *_check.py tests that
inject a fake double in place of a heavy/optional library) -- and
separately, that the class degrades gracefully (becomes a no-op,
doesn't raise) when neither a real nor a fake embedder is available.

Usage:
    python tests/identity_gallery_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.identity_gallery import SegmentationIdentityGallery  # noqa: E402

FRAME = np.zeros((64, 64, 3), dtype=np.uint8)
BOX = np.array([0.0, 0.0, 40.0, 40.0])


class _FakeEmbedder:
    """Deterministic stand-in for OSNetEmbedder: returns a fixed
    unit-norm 2D vector per 'identity label', so similarity is exactly
    controllable in the tests instead of depending on real pixels. The
    'label' is smuggled through the crop's mean pixel value (set by the
    test via distinctly-colored fake frames) purely so a single
    embed(frame, box) call signature (matching the real API) is enough
    -- see `_labelled_frame()`."""

    def embed(self, frame_bgr, bbox_xyxy, poly=None):
        label = int(frame_bgr[0, 0, 0])  # smuggled label, see _labelled_frame()
        if label == 0:
            return None  # simulates "crop too small/degenerate"
        angle = label * 0.3  # distinct labels -> distinct directions
        vec = np.array([np.cos(angle), np.sin(angle)])
        return vec / np.linalg.norm(vec)


def _labelled_frame(label: int) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = label
    return frame


def part1_disabled_gallery_is_a_total_noop():
    gallery = SegmentationIdentityGallery(enabled=False)
    assert not gallery.enabled
    gallery.observe(1, FRAME, BOX)  # must not raise
    gallery.mark_lost(1, chunk_index=0)
    gallery.forget_stale(chunk_index=1)
    assert gallery.match_or_none(FRAME, BOX) is None
    print("PASS part1_disabled_gallery_is_a_total_noop")


def part2_missing_torchreid_disables_without_raising():
    # No `embedder=` injected and torchreid isn't installed in this
    # environment -- construction must catch the ImportError internally
    # (see OSNetEmbedder/_resolve_feature_extractor) and simply disable
    # itself, exactly like torchreid_available() gating elsewhere in the
    # project (appearance_embedding.py).
    gallery = SegmentationIdentityGallery(enabled=True)
    assert gallery.enabled is False, "expected graceful degradation without torch/torchreid installed"
    assert gallery.match_or_none(FRAME, BOX) is None
    print("PASS part2_missing_torchreid_disables_without_raising")


def part3_lost_identity_is_matched_by_appearance_and_revived():
    gallery = SegmentationIdentityGallery(embedder=_FakeEmbedder(), similarity_threshold=0.9)
    assert gallery.enabled

    gallery.observe(42, _labelled_frame(10), BOX)  # active
    gallery.mark_lost(42, chunk_index=3)

    # a near-identical crop (same label -> same embedding) must match
    matched = gallery.match_or_none(_labelled_frame(10), BOX)
    assert matched == 42, f"expected id 42 to be re-identified, got {matched}"

    gallery.revive(42)
    # revived: no longer a candidate for a FUTURE match_or_none() unless
    # marked lost again
    assert gallery.match_or_none(_labelled_frame(10), BOX) is None
    print("PASS part3_lost_identity_is_matched_by_appearance_and_revived")


def part4_dissimilar_crop_does_not_match():
    gallery = SegmentationIdentityGallery(embedder=_FakeEmbedder(), similarity_threshold=0.9)
    gallery.observe(1, _labelled_frame(10), BOX)
    gallery.mark_lost(1, chunk_index=0)

    # a very different label -> very different embedding direction
    matched = gallery.match_or_none(_labelled_frame(90), BOX)
    assert matched is None, f"expected no match for a dissimilar crop, got {matched}"
    print("PASS part4_dissimilar_crop_does_not_match")


def part5_never_matches_a_still_active_id():
    # match_or_none() must only ever consider ids explicitly mark_lost()
    # -- an id that's currently being tracked (never lost) must never be
    # "stolen" by a coincidentally-similar new detection.
    gallery = SegmentationIdentityGallery(embedder=_FakeEmbedder(), similarity_threshold=0.9)
    gallery.observe(7, _labelled_frame(10), BOX)  # active, never marked lost

    matched = gallery.match_or_none(_labelled_frame(10), BOX)
    assert matched is None, f"an active (non-lost) id must never be matched, got {matched}"
    print("PASS part5_never_matches_a_still_active_id")


def part6_forget_stale_drops_old_lost_identities():
    gallery = SegmentationIdentityGallery(embedder=_FakeEmbedder(), similarity_threshold=0.9,
                                           max_lost_age_chunks=2)
    gallery.observe(1, _labelled_frame(10), BOX)
    gallery.mark_lost(1, chunk_index=0)

    gallery.forget_stale(chunk_index=2)  # age 2, within max_lost_age_chunks -- kept
    assert gallery.match_or_none(_labelled_frame(10), BOX) == 1

    gallery.mark_lost(1, chunk_index=0)  # re-mark (match_or_none doesn't revive by itself)
    gallery.forget_stale(chunk_index=3)  # age 3 > max_lost_age_chunks=2 -- forgotten
    assert gallery.match_or_none(_labelled_frame(10), BOX) is None, \
        "expected the identity to be forgotten after exceeding max_lost_age_chunks"
    print("PASS part6_forget_stale_drops_old_lost_identities")


def main():
    part1_disabled_gallery_is_a_total_noop()
    part2_missing_torchreid_disables_without_raising()
    part3_lost_identity_is_matched_by_appearance_and_revived()
    part4_dissimilar_crop_does_not_match()
    part5_never_matches_a_still_active_id()
    part6_forget_stale_drops_old_lost_identities()
    print("\nAll identity_gallery.py tests (with a fake embedder) passed.")


if __name__ == "__main__":
    main()
