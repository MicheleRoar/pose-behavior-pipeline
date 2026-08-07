"""
appearance_embedding_check.py
===============================
Verifies the pure functions of `pose/appearance_embedding.py`
(`embedding_similarity`, `ema_update`, `torchreid_available`) WITHOUT
requiring 'torch'/'torchreid' installed -- `OSNetEmbedder` itself (which
needs them) is NOT tested here, by design: its behavior is verified only
structurally (code review, crop format expected by
`torchreid.utils.FeatureExtractor`) and indirectly in the integration
tests of `reid_check.py`/`seg_reid_check.py`, which use a stub
(`_FakeEmbedder`) with the same interface instead of the real OSNet.

Run with: python appearance_embedding_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pose.appearance_embedding import embedding_similarity, ema_update, torchreid_available


def embedding_similarity_matches_cosine_convention():
    identical = np.array([1.0, 0.0, 0.0])
    assert np.isclose(embedding_similarity(identical, identical), 1.0), "same vector -> similarity 1.0"

    orthogonal_a, orthogonal_b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.isclose(embedding_similarity(orthogonal_a, orthogonal_b), 0.5), (
        "orthogonal vectors (cosine 0) -> similarity 0.5 in the module's 0..1 convention"
    )

    opposite_a, opposite_b = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    assert np.isclose(embedding_similarity(opposite_a, opposite_b), 0.0), (
        "opposite vectors (cosine -1) -> similarity 0.0"
    )

    assert embedding_similarity(None, identical) is None, "a missing embedding -> None, never a made-up value"
    assert embedding_similarity(identical, None) is None
    print("embedding_similarity: cosine [-1,1] -> [0,1] convention respected, None propagated correctly — OK")


def ema_update_converges_and_stays_normalized():
    rng = np.random.default_rng(5)
    true_direction = np.array([1.0, 0.0, 0.0])
    ema = None
    for _ in range(50):
        # noisy observations around the true direction, then renormalized
        # (as a real L2-normalized OSNet embedding would be, per frame)
        noisy = true_direction + rng.normal(0, 0.3, size=3)
        noisy = noisy / np.linalg.norm(noisy)
        ema = ema_update(ema, noisy, alpha=0.9)
        assert np.isclose(np.linalg.norm(ema), 1.0, atol=1e-6), "the EMA must stay a unit vector"

    sim = embedding_similarity(ema, true_direction)
    assert sim > 0.95, (
        f"after 50 noisy updates, the EMA must have stabilized close to the true "
        f"direction (similarity={sim:.3f}) -- this is exactly the required behavior "
        f"('staying in memory and refining over time', the StrongSORT idea cited "
        f"in the module docstring)"
    )
    print(f"ema_update: after 50 noisy observations, similarity with the true direction={sim:.3f} "
          "(converges, stays unitary) — OK")


def ema_update_handles_missing_values():
    v = np.array([1.0, 0.0])
    assert ema_update(None, v) is v, "no prior history -> the observation becomes the memory"
    assert ema_update(v, None) is v, "no new observation -> the prior memory stays unchanged"
    assert ema_update(None, None) is None
    print("ema_update: edge cases (no history / no observation / neither) — OK")


def torchreid_available_is_a_safe_boolean_check():
    # We don't assert True or False (depends on the machine): only that the
    # function never raises an error and returns a real boolean -- it's
    # meant for gating the checkbox in webui/app.js, it must be safe to
    # call even without 'torch' installed.
    result = torchreid_available()
    assert isinstance(result, bool)
    print(f"torchreid_available(): {result} (no error raised, regardless of the outcome) — OK")


def main():
    embedding_similarity_matches_cosine_convention()
    ema_update_converges_and_stays_normalized()
    ema_update_handles_missing_values()
    torchreid_available_is_a_safe_boolean_check()
    print("\nVerification completed with no errors: appearance_embedding.py's pure functions "
          "behave as expected, without requiring 'torch'/'torchreid' installed.")


if __name__ == "__main__":
    main()
